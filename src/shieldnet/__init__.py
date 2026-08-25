"""ShieldNet - explainable multi-class network intrusion detection on CICIDS2017.

This top-level module deliberately imports **nothing heavy**. ``import shieldnet``
works with only numpy and pandas present, so ``shieldnet doctor`` can tell you what is
missing instead of dying on an ImportError while trying to tell you.

Reach for submodules directly:

    from shieldnet.config import Config
    from shieldnet.data.load import load_raw
    from shieldnet.train import prepare_data, train
    from shieldnet.inference import Detector
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__", "schema", "config", "logging_utils"]


def __getattr__(name: str):
    """Lazily expose the dependency-light submodules as attributes."""
    if name in {"schema", "config", "logging_utils", "persist"}:
        import importlib
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
