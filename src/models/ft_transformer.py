"""FT-Transformer wrapper. Wraps the pre-existing ``train_ft_transformer``
and ``extract_attention`` from ``src/ft_transformer_model.py``. Pre-standardized
input passes through the internal Standardizer as a no-op."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import optuna

from .base import BaseModel, register
from ..ft_transformer_model import (
    FTTransformer, Standardizer, default_device,
    train_ft_transformer, extract_attention,
)


@register("FT-Transformer")
class FTTransformerModel(BaseModel):
    backend = "torch"
    supports_intercept = False
    supports_non_negative = False

    @classmethod
    def hpo_space(cls, trial: optuna.Trial, fixed: dict[str, Any]) -> dict[str, Any]:
        hp: dict[str, Any] = dict(fixed) if fixed else {}
        if not isinstance(hp.get("d_token"), int):
            hp["d_token"] = trial.suggest_categorical("d_token", [16, 32, 64])
        if not isinstance(hp.get("n_blocks"), int):
            hp["n_blocks"] = trial.suggest_int("n_blocks", 2, 4)
        if not isinstance(hp.get("n_heads"), int):
            hp["n_heads"] = trial.suggest_categorical("n_heads", [2, 4, 8])
        if not isinstance(hp.get("d_ffn"), int):
            hp["d_ffn"] = trial.suggest_categorical("d_ffn", [32, 64, 128, 256])
        if not isinstance(hp.get("dropout"), (int, float)) or isinstance(hp.get("dropout"), bool):
            hp["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)
        if not isinstance(hp.get("attn_dropout"), (int, float)) or isinstance(hp.get("attn_dropout"), bool):
            hp["attn_dropout"] = trial.suggest_float("attn_dropout", 0.0, 0.3)
        if not isinstance(hp.get("lr"), (int, float)) or isinstance(hp.get("lr"), bool):
            hp["lr"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        if not isinstance(hp.get("weight_decay"), (int, float)) or isinstance(hp.get("weight_decay"), bool):
            hp["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        if not isinstance(hp.get("input_noise_std"), (int, float)) or isinstance(hp.get("input_noise_std"), bool):
            hp["input_noise_std"] = trial.suggest_float("input_noise_std", 0.0, 0.05)
        # d_token must be divisible by n_heads
        if hp["d_token"] % hp["n_heads"] != 0:
            # Round n_heads down to a divisor; falls through Optuna's
            # parameter-importance plot but keeps the search valid.
            for h in (8, 4, 2, 1):
                if hp["d_token"] % h == 0:
                    hp["n_heads"] = h
                    break
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
    ) -> "FTTransformerModel":
        import torch

        self.feature_names_ = list(X_train.columns)
        device = default_device(self.device)
        self._device = device
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        n_features = X_train.shape[1]
        model = FTTransformer(
            n_features=n_features,
            d_token=int(hp["d_token"]),
            n_blocks=int(hp["n_blocks"]),
            n_heads=int(hp["n_heads"]),
            d_ffn=int(hp["d_ffn"]),
            dropout=float(hp["dropout"]),
            attn_dropout=float(hp["attn_dropout"]),
        )

        use_val = X_val is not None and y_val is not None and len(X_val) > 0
        if not use_val:
            # train_ft_transformer requires a val set for early stopping;
            # supply a copy of train (no early-stop signal but training runs).
            X_val = X_train
            y_val = y_train

        Xtr = X_train.to_numpy(dtype=np.float64, copy=False)
        ytr = y_train.to_numpy(dtype=np.float64, copy=False)
        Xva = X_val.to_numpy(dtype=np.float64, copy=False)
        yva = y_val.to_numpy(dtype=np.float64, copy=False)

        model, history, std = train_ft_transformer(
            model, Xtr, ytr, Xva, yva,
            lr=float(hp["lr"]),
            weight_decay=float(hp["weight_decay"]),
            max_epochs=int(hp["max_epochs"]),
            patience=int(hp["patience"]),
            batch_size=int(hp["batch_size"]),
            device=device,
            input_noise_std=float(hp["input_noise_std"]),
            seed=self.seed,
            verbose=self.verbose,
        )
        self.model_ = model
        self.history_ = history
        # std is an identity transform here (input was pre-standardized),
        # but kept so extract_attention / predict signatures match.
        self.std_ = std
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        import torch
        self.model_.eval()
        Xz = self.std_.transform_X(X.to_numpy(dtype=np.float64, copy=False))
        Xt = torch.as_tensor(Xz, dtype=torch.float32, device=self._device)
        preds = []
        bs = 512
        with torch.no_grad():
            for start in range(0, Xt.size(0), bs):
                preds.append(self.model_(Xt[start:start + bs]).cpu().numpy())
        yz = np.concatenate(preds) if preds else np.zeros((0,))
        return self.std_.inverse_y(yz)

    def importance(self) -> pd.Series:
        # CLS-token attention to each feature (averaged across batches/heads/layers).
        # The 'attention' tensor returned is in the model's z-scored input space,
        # but importance ranking is invariant to scale.
        # Use a small sample of training data; not stored, so use a zeros vector
        # of cls attention from a 1-batch forward if necessary. The pipeline
        # provides the data via extract_interactions instead, but for raw
        # importance we approximate with the row-norm of the tokenizer weights.
        W = self.model_.tokenizer.weight.detach().cpu().numpy()  # (F, d_token)
        scores = np.linalg.norm(W, axis=1)
        return pd.Series(scores, index=self.feature_names_)

    def interaction_matrix(
        self,
        X: pd.DataFrame,
        top_features: list[str],
    ) -> pd.DataFrame | None:
        if not top_features:
            return None
        # Use up to 1024 rows for attention extraction.
        n = min(1024, len(X))
        if n == 0:
            return None
        sample = X.iloc[:n].to_numpy(dtype=np.float64, copy=False)
        cls_attn, ff_attn = extract_attention(self.model_, sample, self.std_,
                                              device=self._device)
        top_idx = np.array(
            [self.feature_names_.index(f) for f in top_features], dtype=int,
        )
        sub = ff_attn[np.ix_(top_idx, top_idx)]
        np.fill_diagonal(sub, 0.0)
        return pd.DataFrame(sub, index=top_features, columns=top_features)

    def convergence_history(self) -> dict[str, list[float]] | None:
        return self.history_
