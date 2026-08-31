problem: 

I am training a table-based model with a training set and a test set. The samples in both datasets form continuous sequences (sample order cannot be shuffled). I want to quantitatively estimate whether sequence fragments in the test set have sample-level proximity in the training set to check for near-duplicate sequences or data leakage. What are the most common algorithms used for this in engineering?

answer:

try methods 1 to 3 -- 

1. Sample-Level Neighbors: ANN + a Baseline Comparison

This is the quantitative core of what you're asking for.

python
import faiss
d = X_tr.shape[1]
index = faiss.IndexFlatL2(d)          # exact search is fine under ~1M samples; use HNSW / IVF-PQ beyond that
index.add(X_tr)

d_te, _ = index.search(X_te, 1)       # 1-NN distance, test → train
d_tr, _ = index.search(X_tr, 2)       # baseline: 2-NN train → train (the 1st is the point itself)
d_tr = d_tr[:, 1]

The key methodological point: absolute distance values are meaningless without a control group. The training set already has its own natural neighbor density. Only when the test→train distribution shifts noticeably left relative to train→train do you have something anomalous. The three metrics usually reported:

Contamination rate P(d_test→train < ε), with ε set to the 1st percentile of d_train→train
Distance ratio median(d_test) / median(d_train) — should be ≈ 1 or slightly above when healthy
The KS test statistic between the two distributions

One pitfall specific to tabular data: Euclidean distance doesn't necessarily reflect the similarity your model actually perceives. Apply a quantile transform to numeric columns first, and use Gower distance for mixed types. A more faithful approach is to use the leaf-node encoding (leaf embedding) from your trained GBDT as the representation and compute distances there — that measures neighbors as the model sees them.

2. Segment-Level: Shingling + MinHash LSH

Sample-level neighbors can't answer "is this contiguous segment a duplicate?" The approach: turn each row into a token (bin the numeric values and hash, or use cluster IDs from KMeans / Product Quantization), which turns the whole sequence into a symbol string. Then:

Approximate matching (most common): take the token set of every length-k sliding window, build an index with datasketch's MinHashLSH, and query test segments for training segments with Jaccard similarity above a threshold. Scales to billions.
Exact longest common substring: build a suffix array / suffix automaton over the training string, query the longest match length at each test position, and report the length distribution. This is the ExactSubstr deduplication from Lee et al. 2021, and it pinpoints exactly which segment was copied.
Pure time-series case: stumpy.stump(T_test, m, T_train, ignore_trivial=False) performs a matrix profile AB-join, directly giving you the z-normalized distance from every length-m subsequence in the test set to its nearest subsequence in the training set. This tool was practically designed for your problem.

3. Cross-Check: Adversarial Validation

Train a LightGBM binary classifier to distinguish train rows from test rows. AUC ≈ 0.5 means the two are indistinguishable in distribution — on its own this doesn't prove leakage, but "AUC ≈ 0.5" plus "abnormally small nearest-neighbor distances" occurring together is the classic signature of near-duplication. Conversely, a high AUC indicates distribution drift, which is a different problem.