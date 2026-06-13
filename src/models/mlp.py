"""MLP wrapper (3-layer feed-forward). Lifted from the old ``src/mlp.py``;
the internal Standardizer is removed because the pipeline z-scores upstream.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import optuna

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel, register
from ..torch_utils import default_device


class SmallMLP(nn.Module):
    """input -> h1 -> h2 -> 1 with ReLU + dropout between linear layers."""

    def __init__(self, n_features: int, hidden1: int = 128, hidden2: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.drop(h)
        h = F.relu(self.fc2(h))
        h = self.drop(h)
        return self.fc3(h).squeeze(-1)


def _to_tensor(arr, dtype, device):
    return torch.as_tensor(np.asarray(arr), dtype=dtype, device=device)


@register("MLP")
class MLPModel(BaseModel):
    backend = "torch"
    supports_intercept = False
    supports_non_negative = False

    @classmethod
    def hpo_space(cls, trial: optuna.Trial, fixed: dict[str, Any]) -> dict[str, Any]:
        hp: dict[str, Any] = dict(fixed) if fixed else {}
        if not isinstance(hp.get("hidden1"), int):
            hp["hidden1"] = trial.suggest_categorical("hidden1", [64, 128, 256, 512])
        if not isinstance(hp.get("hidden2"), int):
            hp["hidden2"] = trial.suggest_categorical("hidden2", [32, 64, 128, 256])
        if not isinstance(hp.get("dropout"), (int, float)) or isinstance(hp.get("dropout"), bool):
            hp["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)
        if not isinstance(hp.get("lr"), (int, float)) or isinstance(hp.get("lr"), bool):
            hp["lr"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        if not isinstance(hp.get("weight_decay"), (int, float)) or isinstance(hp.get("weight_decay"), bool):
            hp["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        hp.setdefault("batch_size", 256)
        hp.setdefault("max_epochs", 300)
        hp.setdefault("patience", 40)
        return hp

    def fit(
        self,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        *, hp: dict[str, Any],
    ) -> "MLPModel":
        self.feature_names_ = list(X_train.columns)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = default_device(self.device)
        self._device = device

        Xtr = _to_tensor(X_train.to_numpy(dtype=np.float64), torch.float32, device)
        ytr = _to_tensor(y_train.to_numpy(dtype=np.float64), torch.float32, device)
        use_val = X_val is not None and y_val is not None and len(X_val) > 0
        if use_val:
            Xva = _to_tensor(X_val.to_numpy(dtype=np.float64), torch.float32, device)
            yva = _to_tensor(y_val.to_numpy(dtype=np.float64), torch.float32, device)

        model = SmallMLP(
            n_features=Xtr.shape[1],
            hidden1=int(hp["hidden1"]),
            hidden2=int(hp["hidden2"]),
            dropout=float(hp["dropout"]),
        ).to(device)

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(hp["lr"]),
            weight_decay=float(hp["weight_decay"]),
        )
        max_epochs = int(hp["max_epochs"])
        patience = int(hp["patience"])
        batch_size = int(hp["batch_size"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, max_epochs))

        history = {"train_loss": [], "val_loss": [], "best_epoch": 0}
        best_val = float("inf")
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        wait = 0
        n_train = Xtr.size(0)

        for epoch in range(max_epochs):
            model.train()
            perm = torch.randperm(n_train, device=device)
            epoch_losses = []
            for start in range(0, n_train, batch_size):
                idx = perm[start:start + batch_size]
                pred = model(Xtr[idx])
                loss = F.mse_loss(pred, ytr[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_losses.append(float(loss.detach()))
            sched.step()
            train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            history["train_loss"].append(train_loss)

            if use_val:
                model.eval()
                with torch.no_grad():
                    val_loss = float(F.mse_loss(model(Xva), yva))
                history["val_loss"].append(val_loss)
                improved = val_loss < best_val - 1e-7
                if improved:
                    best_val = val_loss
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    history["best_epoch"] = epoch
                    wait = 0
                else:
                    wait += 1
                if wait >= patience:
                    break
            else:
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                history["best_epoch"] = epoch

        model.load_state_dict(best_state)
        self.model_ = model
        self.history_ = history
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.model_.eval()
        Xt = _to_tensor(X.to_numpy(dtype=np.float64), torch.float32, self._device)
        preds = []
        bs = 1024
        with torch.no_grad():
            for start in range(0, Xt.size(0), bs):
                preds.append(self.model_(Xt[start:start + bs]).cpu().numpy())
        return np.concatenate(preds) if preds else np.zeros((0,))

    def importance(self) -> pd.Series:
        W1 = self.model_.fc1.weight.detach().cpu().numpy()  # (h1, n_features)
        return pd.Series(np.abs(W1).sum(axis=0), index=self.feature_names_)

    def convergence_history(self) -> dict[str, list[float]] | None:
        return self.history_
