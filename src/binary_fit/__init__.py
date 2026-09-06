"""binary_fit: staged per-cycle power modeling from single-bit signal features.

Three standalone stages, one CLI (``python src/binary_fit/run.py``):

* ``--build_db``       materialize the single-bit feature dataset to disk once.
* ``--feature_select`` LR-MCP proxy selection -> a ranked ``proxies.csv``.
* ``--fit``            fit a model (``--model tree|nn|ridge|both``) per proxy count ``-q``.

The three Stage-2 regressors (XGBoost gradient-boosted trees, a two-layer sklearn
MLP, and L2-penalized linear ridge) share the same binary features, MCP proxy
set, evaluation and reporting -- they differ only in the estimator. ``--model
both`` fits all of them.
"""

__version__ = "0.1.0"
