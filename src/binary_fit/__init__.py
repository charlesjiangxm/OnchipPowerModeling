"""binary_fit: staged per-cycle power modeling from single-bit signal features.

Three standalone stages, one CLI (``python src/binary_fit/run.py``):

* ``--build_db``       materialize the single-bit feature dataset to disk once.
* ``--feature_select`` LR-MCP proxy selection -> a ranked ``proxies.csv``.
* ``--fit``            fit a model (``--model tree|nn|both``) per proxy count ``-q``.

The two Stage-2 regressors (XGBoost gradient-boosted trees and a two-layer
sklearn MLP) share the same binary features, MCP proxy set, evaluation and
reporting -- they differ only in the estimator.
"""

__version__ = "0.1.0"
