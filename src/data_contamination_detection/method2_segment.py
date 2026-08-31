"""Method 2: Segment-Level (Shingling + MinHash LSH).

Tokenizes rows via KMeans cluster IDs, converts sequences to shingled bigram
sets, and uses MinHash LSH to find near-duplicate segments between test and
train. Optionally runs stumpy matrix profile as a 1D cross-check.
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from datasketch import MinHash, MinHashLSH
from sklearn.cluster import MiniBatchKMeans

logger = logging.getLogger(__name__)

KMEANS_N_CLUSTERS = 256
SHINGLE_LENGTH = 20
MINHASH_NUM_PERM = 128
MINHASH_THRESHOLD = 0.5
STUMPY_M = 50


@dataclass
class Method2Result:
    n_train_windows: int
    n_test_windows: int
    n_matches: int
    match_rate: float
    jaccard_values: list = field(default_factory=list)
    matches_detail: list = field(default_factory=list)
    stumpy_profile: np.ndarray = None
    stumpy_stats: dict = None


def _window_to_bigrams(tokens, start, k):
    window = tokens[start:start + k]
    bigrams = set()
    for i in range(len(window) - 1):
        bigrams.add((int(window[i]), int(window[i + 1])))
    return bigrams


def _bigrams_to_minhash(bigrams, num_perm):
    m = MinHash(num_perm=num_perm)
    for bg in bigrams:
        m.update(str(bg).encode("utf-8"))
    return m


def run_kmeans_minhash(X_train, X_test, source_ids_train=None, boundaries_train=None,
                       source_ids_test=None, boundaries_test=None):
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    k = SHINGLE_LENGTH
    logger.info(f"Method 2 (KMeans+MinHash): train {n_train}, test {n_test}")

    # Step 1: KMeans tokenization
    logger.info(f"Fitting MiniBatchKMeans (n_clusters={KMEANS_N_CLUSTERS})...")
    t0 = time.time()
    km = MiniBatchKMeans(
        n_clusters=KMEANS_N_CLUSTERS,
        random_state=42,
        batch_size=10000,
        n_init=3,
    )
    km.fit(X_train)
    train_tokens = km.predict(X_train).astype(np.int32)
    test_tokens = km.predict(X_test).astype(np.int32)
    logger.info(f"  KMeans: {time.time()-t0:.1f}s, {len(np.unique(train_tokens))} unique train clusters")

    # Step 2: Shingling + MinHash LSH
    n_train_windows = max(0, n_train - k + 1)
    n_test_windows = max(0, n_test - k + 1)
    logger.info(f"Building LSH index: {n_train_windows} train windows, {n_test_windows} test windows")

    t0 = time.time()
    lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=MINHASH_NUM_PERM)
    train_minhashes = {}
    for i in range(n_train_windows):
        bigrams = _window_to_bigrams(train_tokens, i, k)
        if len(bigrams) == 0:
            continue
        mh = _bigrams_to_minhash(bigrams, MINHASH_NUM_PERM)
        lsh.insert(f"train_{i}", mh)
        train_minhashes[i] = mh
    logger.info(f"  LSH build: {time.time()-t0:.1f}s")

    # Step 3: Query test windows
    logger.info("Querying test windows...")
    t0 = time.time()
    matches = []
    jaccard_values = []
    MAX_CANDIDATES_PER_QUERY = 500
    for i in range(n_test_windows):
        bigrams = _window_to_bigrams(test_tokens, i, k)
        if len(bigrams) == 0:
            continue
        mh = _bigrams_to_minhash(bigrams, MINHASH_NUM_PERM)
        result = lsh.query(mh)
        if len(result) > MAX_CANDIDATES_PER_QUERY:
            result = result[:MAX_CANDIDATES_PER_QUERY]
        for key in result:
            train_idx = int(key.split("_")[1])
            train_mh = train_minhashes[train_idx]
            jaccard = mh.jaccard(train_mh)
            if jaccard >= MINHASH_THRESHOLD:
                matches.append({
                    "test_pos": i,
                    "train_pos": train_idx,
                    "jaccard": float(jaccard),
                })
                jaccard_values.append(float(jaccard))
    logger.info(f"  query: {time.time()-t0:.1f}s, {len(matches)} matches")

    match_rate = len(matches) / n_test_windows if n_test_windows > 0 else 0.0

    # Sort matches by jaccard descending, keep top 100 for detail
    matches.sort(key=lambda x: -x["jaccard"])
    matches_detail = matches[:100]

    return Method2Result(
        n_train_windows=n_train_windows,
        n_test_windows=n_test_windows,
        n_matches=len(matches),
        match_rate=match_rate,
        jaccard_values=jaccard_values,
        matches_detail=matches_detail,
    )


def run_stumpy(X_train, X_test):
    import stumpy

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    m = STUMPY_M
    logger.info(f"Method 2 (stumpy): m={m}, reducing to 1D via L2 norm")

    train_1d = np.linalg.norm(X_train, axis=1).astype(np.float64)
    test_1d = np.linalg.norm(X_test, axis=1).astype(np.float64)
    n_test_windows = max(0, n_test - m + 1)

    logger.info(f"Computing matrix profile AB-join ({n_test_windows} test subsequences)...")
    t0 = time.time()
    mp = stumpy.stump(test_1d, m, train_1d, ignore_trivial=False)
    elapsed = time.time() - t0
    logger.info(f"  stumpy: {elapsed:.1f}s")

    profile = mp[:, 0].astype(np.float32)
    finite = profile[np.isfinite(profile)]

    stats = {
        "m": m,
        "n_test_subsequences": n_test_windows,
        "median": float(np.median(finite)) if len(finite) > 0 else None,
        "mean": float(np.mean(finite)) if len(finite) > 0 else None,
        "p5": float(np.percentile(finite, 5)) if len(finite) > 0 else None,
        "p25": float(np.percentile(finite, 25)) if len(finite) > 0 else None,
        "min": float(np.min(finite)) if len(finite) > 0 else None,
    }
    logger.info(f"  profile: median={stats['median']}, min={stats['min']}")

    return profile, stats


def run(X_train, X_test, source_ids_train=None, boundaries_train=None,
        source_ids_test=None, boundaries_test=None, enable_stumpy=False):
    result = run_kmeans_minhash(
        X_train, X_test, source_ids_train, boundaries_train,
        source_ids_test, boundaries_test,
    )
    if enable_stumpy:
        result.stumpy_profile, result.stumpy_stats = run_stumpy(X_train, X_test)
    return result
