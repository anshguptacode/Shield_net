"""Model wrappers and the registry that maps a config key to one of them.

Import from here rather than from the submodules::

    from shieldnet.models import build, available, describe_all

Nothing in this package imports scikit-learn, XGBoost, LightGBM or TensorFlow at module
level. The heavy import happens inside ``fit``, so ``shieldnet --help`` stays instant and
the Streamlit app never initialises TensorFlow to serve an XGBoost model.
"""

from __future__ import annotations

from .base import MissingDependency, ModelInfo, ShieldModel, as_array
from .registry import (ALIASES, CATALOGUE, REGISTRY, UnknownModel, available, build,
                       describe_all, is_deep, resolve, search_space_for)

__all__ = [
    "ShieldModel", "ModelInfo", "MissingDependency", "as_array",
    "REGISTRY", "CATALOGUE", "ALIASES", "UnknownModel",
    "build", "available", "describe_all", "is_deep", "resolve", "search_space_for",
]
