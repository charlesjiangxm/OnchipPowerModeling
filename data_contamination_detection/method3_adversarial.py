"""Method 3: Adversarial Validation.

Trains a LightGBM binary classifier to distinguish train rows from test rows.
AUC ≈ 0.5 means the two are distributionally indistinguishable. Combined with
abnormally small nearest-neighbor distances (Method 1), this is the classic
signature of near-duplication. High AUC indicates distribution drift.
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass
class Method3Result:
    auc: float
    n_estimators_used: int
    top_features: list = field(default_factory=list)
    fpr: np.ndarray = None
    tpr: np.ndarray = None
    thresholds: np.ndarray = None
    val_pred_proba: np.ndarray = None
    val_y: np.ndarray = None


def run(X_train: np.ndarray, X_test: np.ndarray, feature_names: list = None) -> Method3Result:
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    d = X_train.shape[1]
    logger.info(f"Method 3: train {n_train}, test {n_test}, features {d}")

    X = np.vstack([X_train, X_test])
    y = np.concatenate([np.zeros(n_train, dtype=np.int8), np.ones(n_test, dtype=np.int8)])
    del X_train, X_test

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    logger.info(f"  split: train {len(y_tr)}, val {len(y_val)} (pos_rate={y_val.mean():.3f})")

    model = LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    logger.info("Training LightGBM...")
    t0 = time.time()
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_names=["val"],
        callbacks=[early_stopping(20), log_evaluation(0)],
    )
    logger.info(f"  training: {time.time()-t0:.1f}s, best_iter={model.best_iteration_}")

    val_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_proba)
    fpr, tpr, thresholds = roc_curve(y_val, val_proba)

    importances = model.feature_importances_
    names = feature_names if feature_names else [f"f{i}" for i in range(d)]
    sorted_idx = np.argsort(importances)[::-1]
    top_features = [
        {"name": names[i], "importance": int(importances[i])}
        for i in sorted_idx[:20]
        if importances[i] > 0
    ]

    logger.info(f"Method 3 results: AUC={auc:.4f}, best_iter={model.best_iteration_}")
    if top_features:
        logger.info(f"  top feature: {top_features[0]['name']} (imp={top_features[0]['importance']})")

    return Method3Result(
        auc=float(auc),
        n_estimators_used=model.best_iteration_ or 200,
        top_features=top_features,
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        val_pred_proba=val_proba,
        val_y=y_val,
    )
