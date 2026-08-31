"""Method 1: Sample-Level Neighbors (ANN + Baseline).

Uses faiss IndexFlatL2 for exact nearest-neighbor search. Compares test→train
distances against the train→train baseline to detect anomalous proximity.

Key insight from spec: absolute distance values are meaningless without a
control group. Only when the test→train distribution shifts noticeably left
relative to train→train do we have something anomalous.
"""

import logging
import time
from dataclasses import dataclass, field

import faiss
import numpy as np
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)

FAISS_BATCH_SIZE = 8192


@dataclass
class Method1Result:
    d_te: np.ndarray
    d_tr: np.ndarray
    I_te: np.ndarray
    contamination_rate: float
    epsilon: float
    distance_ratio: float
    ks_statistic: float
    ks_pvalue: float
    d_te_median: float
    d_tr_median: float
    d_te_mean: float
    d_tr_mean: float
    n_suspicious: int
    top_matches: list = field(default_factory=list)


def run(X_train: np.ndarray, X_test: np.ndarray, source_ids_train: np.ndarray = None,
        boundaries_train: list = None) -> Method1Result:
    n_train, d = X_train.shape
    n_test = X_test.shape[0]
    logger.info(f"Method 1: train {n_train}x{d}, test {n_test}x{d}")

    faiss.omp_set_num_threads(faiss.omp_get_max_threads())
    X_train_c = np.ascontiguousarray(X_train, dtype=np.float32)
    X_test_c = np.ascontiguousarray(X_test, dtype=np.float32)

    index = faiss.IndexFlatL2(d)
    logger.info("Building faiss index...")
    t0 = time.time()
    index.add(X_train_c)
    logger.info(f"  index.add: {time.time()-t0:.1f}s")

    # test→train 1-NN (batched)
    logger.info("Searching test→train 1-NN...")
    t0 = time.time()
    d_te_list = []
    I_te_list = []
    for start in range(0, n_test, FAISS_BATCH_SIZE):
        end = min(start + FAISS_BATCH_SIZE, n_test)
        D, I = index.search(X_test_c[start:end], 1)
        d_te_list.append(D[:, 0])
        I_te_list.append(I[:, 0])
    d_te = np.concatenate(d_te_list)
    I_te = np.concatenate(I_te_list)
    logger.info(f"  search test→train: {time.time()-t0:.1f}s")

    # train→train 2-NN (1st is self, take 2nd)
    logger.info("Searching train→train 2-NN...")
    t0 = time.time()
    d_tr_list = []
    for start in range(0, n_train, FAISS_BATCH_SIZE):
        end = min(start + FAISS_BATCH_SIZE, n_train)
        D, _ = index.search(X_train_c[start:end], 2)
        d_tr_list.append(D[:, 1])
    d_tr = np.concatenate(d_tr_list)
    logger.info(f"  search train→train: {time.time()-t0:.1f}s")

    # Metrics
    epsilon = float(np.percentile(d_tr, 1))
    if epsilon == 0.0:
        nonzero = d_tr[d_tr > 0]
        epsilon = float(np.percentile(nonzero, 1)) if len(nonzero) > 0 else 1e-10
        logger.info(f"  1st pct was 0 (duplicates in train), using epsilon={epsilon:.10f}")
    contamination_rate = float(np.mean(d_te <= epsilon))
    d_te_median = float(np.median(d_te))
    d_tr_median = float(np.median(d_tr))
    distance_ratio = d_te_median / d_tr_median if d_tr_median > 0 else float("inf")
    ks_stat, ks_p = ks_2samp(d_te, d_tr)
    n_suspicious = int(np.sum(d_te <= epsilon))

    # Top-20 closest test samples
    sorted_idx = np.argsort(d_te)[:20]
    top_matches = []
    for idx in sorted_idx:
        entry = {
            "test_pos": int(idx),
            "nn_train_pos": int(I_te[idx]),
            "distance": float(d_te[idx]),
        }
        if source_ids_train is not None and boundaries_train is not None:
            src = source_ids_train[I_te[idx]]
            entry["nn_benchmark"] = boundaries_train[src].name
        top_matches.append(entry)

    logger.info(f"Method 1 results:")
    logger.info(f"  contamination_rate={contamination_rate:.6f} (epsilon={epsilon:.6f})")
    logger.info(f"  distance_ratio={distance_ratio:.4f} (te_med={d_te_median:.4f}, tr_med={d_tr_median:.4f})")
    logger.info(f"  ks_stat={ks_stat:.6f}, ks_p={ks_p:.2e}")
    logger.info(f"  n_suspicious={n_suspicious}")

    return Method1Result(
        d_te=d_te, d_tr=d_tr, I_te=I_te,
        contamination_rate=contamination_rate,
        epsilon=epsilon,
        distance_ratio=distance_ratio,
        ks_statistic=float(ks_stat),
        ks_pvalue=float(ks_p),
        d_te_median=d_te_median,
        d_tr_median=d_tr_median,
        d_te_mean=float(np.mean(d_te)),
        d_tr_mean=float(np.mean(d_tr)),
        n_suspicious=n_suspicious,
        top_matches=top_matches,
    )
