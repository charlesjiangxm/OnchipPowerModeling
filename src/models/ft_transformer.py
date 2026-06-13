"""FT-Transformer for the c906 power-modeling pipeline.

Self-contained PyTorch implementation of the architecture from
Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data"
(NeurIPS 2021). Single-target regression; numerical features only
(c906 has no categorical signals).

This module bundles three things:

1. The model and its building blocks (``NumericalFeatureTokenizer``,
   ``TransformerBlock``, ``FTTransformer``).
2. The training loop (``train_ft_transformer``) plus inference helpers
   (``predict``, ``extract_attention``).
3. The pipeline-facing wrapper ``FTTransformerModel`` that conforms to
   the ``BaseModel`` interface.

Generic helpers (device picking, standardizer, tensor conversion) live in
``src/torch_utils.py``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import optuna

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel, register
from ..torch_utils import (
    Standardizer,
    default_device,
    to_tensor,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class NumericalFeatureTokenizer(nn.Module):
    """Tokenize numerical features per the FT-Transformer paper, sec 3.3.

    For each feature j and each example i:  T_ij = b_j + x_ij * W_j,
    where W_j, b_j are vectors of length d_token.
    """

    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    def forward(self, x):
        # x: (B, F) -> (B, F, d_token)
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class TransformerBlock(nn.Module):
    """PreNorm Transformer block (MHSA + FFN), batch_first.

    If `is_first=True`, the first LayerNorm before MHSA is omitted, per
    the FT-Transformer paper section 3.3 ("we found it to be necessary
    to remove the first normalization from the first Transformer layer").
    """

    def __init__(self, d_token, n_heads, d_ffn, dropout, attn_dropout,
                 is_first=False):
        super().__init__()
        self.is_first = is_first
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_token),
        )
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        # MHSA branch (PreNorm; first block skips norm1).
        x_for_attn = x if self.is_first else self.norm1(x)
        attn_out, attn_w = self.attn(
            x_for_attn, x_for_attn, x_for_attn,
            need_weights=return_attention,
            average_attn_weights=False,  # keep per-head for richer extraction
        )
        x = x + self.attn_dropout(attn_out)

        # FFN branch.
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.ffn_dropout(ffn_out)
        return x, attn_w


class FTTransformer(nn.Module):
    """Feature Tokenizer + Transformer for single-target regression.

    Forward returns ``y`` of shape ``(B,)``. If ``return_attention=True``,
    also returns a list of per-block attention tensors of shape
    ``(B, n_heads, 1+F, 1+F)`` (in MultiheadAttention's per-head layout
    with ``average_attn_weights=False``).
    """

    def __init__(self, n_features, d_token=32, n_blocks=3, n_heads=4,
                 d_ffn=64, dropout=0.1, attn_dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token
        self.tokenizer = NumericalFeatureTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.xavier_uniform_(self.cls_token)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_token, n_heads, d_ffn, dropout, attn_dropout,
                             is_first=(i == 0))
            for i in range(n_blocks)
        ])
        self.final_norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)

    def forward(self, x, return_attention=False):
        # x: (B, F)
        tokens = self.tokenizer(x)                          # (B, F, d)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)  # (B, 1, d)
        tokens = torch.cat([cls, tokens], dim=1)            # (B, 1+F, d)

        attn_list = [] if return_attention else None
        for block in self.blocks:
            tokens, attn_w = block(tokens, return_attention=return_attention)
            if return_attention:
                attn_list.append(attn_w)

        cls_out = self.final_norm(tokens[:, 0, :])          # (B, d)
        y = self.head(F.relu(cls_out)).squeeze(-1)           # (B,)
        if return_attention:
            return y, attn_list
        return y


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_ft_transformer(
    model,
    X_train, y_train,
    X_val, y_val,
    *,
    lr=1e-3,
    weight_decay=1e-4,
    max_epochs=300,
    patience=40,
    batch_size=256,
    device=None,
    input_noise_std=0.02,
    grad_clip=1.0,
    seed=42,
    use_amp=False,
    verbose=True,
):
    """Fit `model` with AdamW + cosine LR + early stopping. Returns
    (best_model, history_dict, standardizer).

    Parameters
    ----------
    use_amp : bool
        Enable mixed-precision autocast on CUDA (no-op on MPS / CPU).
        Combines `torch.amp.autocast` with `GradScaler` for an ~2x
        forward/backward speed-up on Ampere+ GPUs.
    """
    device = device or default_device("auto")
    torch.manual_seed(seed)
    np.random.seed(seed)

    std = Standardizer.fit(X_train, y_train)
    Xtr_z = std.transform_X(X_train)
    ytr_z = std.transform_y(y_train)
    Xva_z = std.transform_X(X_val)
    yva_z = std.transform_y(y_val)

    Xtr_t = to_tensor(Xtr_z, torch.float32, device)
    ytr_t = to_tensor(ytr_z, torch.float32, device)
    Xva_t = to_tensor(Xva_z, torch.float32, device)
    yva_t = to_tensor(yva_z, torch.float32, device)

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, max_epochs))

    amp_enabled = bool(use_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if use_amp and not amp_enabled and verbose:
        print(f"    [train] --ft_amp requested but device={device.type}; "
              f"autocast is CUDA-only -- running in float32.")

    history = {"train_loss": [], "val_loss": [], "best_epoch": 0}
    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    wait = 0
    n_train = Xtr_t.size(0)

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_losses = []
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            xb = Xtr_t[idx]
            if input_noise_std > 0:
                xb = xb + torch.randn_like(xb) * input_noise_std
            yb = ytr_t[idx]
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                pred = model(xb)
                loss = F.mse_loss(pred, yb)
            if amp_enabled:
                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
            epoch_losses.append(float(loss.detach()))
        sched.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(Xva_t)
            val_loss = float(F.mse_loss(pred_val, yva_t))
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        improved = val_loss < best_val - 1e-7
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch
            wait = 0
        else:
            wait += 1

        if verbose and (epoch < 3 or epoch % 25 == 0 or improved):
            print(f"    epoch {epoch:3d}  train={train_loss:.5f}  "
                  f"val={val_loss:.5f}{' *' if improved else ''}")

        if wait >= patience:
            if verbose:
                print(f"    early stop at epoch {epoch} "
                      f"(best epoch {history['best_epoch']}, val={best_val:.5f})")
            break

    model.load_state_dict(best_state)
    return model, history, std


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def predict(model, X, std, device=None, batch_size=512):
    """Standardize X, forward in batches, inverse-transform y."""
    device = device or default_device("auto")
    model.eval()
    Xz = std.transform_X(X)
    Xz_t = to_tensor(Xz, torch.float32, device)
    preds = []
    with torch.no_grad():
        for start in range(0, Xz_t.size(0), batch_size):
            preds.append(model(Xz_t[start:start + batch_size]).cpu().numpy())
    yz = np.concatenate(preds, axis=0) if preds else np.zeros((0,))
    return std.inverse_y(yz)


def extract_attention(model, X, std, device=None, batch_size=64):
    """Return (cls_attn, ff_attn) averaged across heads, layers, and examples.

    cls_attn : (F,)   -- CLS-token's attention to each feature.
    ff_attn  : (F, F) -- feature-feature attention block.
    """
    device = device or default_device("auto")
    model.eval()
    Xz = std.transform_X(X)
    Xz_t = to_tensor(Xz, torch.float32, device)
    n = Xz_t.size(0)
    F_dim = model.n_features

    sum_cls = np.zeros(F_dim, dtype=np.float64)
    sum_ff = np.zeros((F_dim, F_dim), dtype=np.float64)
    n_seen = 0

    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb = Xz_t[start:start + batch_size]
            _, attn_list = model(xb, return_attention=True)
            B = xb.size(0)

            # Average over (heads, layers).
            stacked = torch.stack(attn_list, dim=0)        # (L, B, H, 1+F, 1+F)
            mean_lh = stacked.mean(dim=(0, 2))             # (B, 1+F, 1+F)
            cls_b = mean_lh[:, 0, 1:].sum(dim=0)            # (F,)
            ff_b = mean_lh[:, 1:, 1:].sum(dim=0)            # (F, F)
            sum_cls += cls_b.cpu().numpy()
            sum_ff += ff_b.cpu().numpy()
            n_seen += B

    if n_seen == 0:
        return np.zeros(F_dim), np.zeros((F_dim, F_dim))
    return sum_cls / n_seen, sum_ff / n_seen


# ---------------------------------------------------------------------------
# Pipeline wrapper
# ---------------------------------------------------------------------------

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
