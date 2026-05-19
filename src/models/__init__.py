"""Model wrappers. Each subclass of BaseModel exposes a uniform
hpo_space / fit / predict / importance / interaction / convergence interface
so the pipeline orchestrator can drive them identically."""

from .base import BaseModel, MODEL_REGISTRY
from . import rulefit, ft_transformer, gbdt, mlp, elasticnet  # noqa: F401  (register)

__all__ = ["BaseModel", "MODEL_REGISTRY"]
