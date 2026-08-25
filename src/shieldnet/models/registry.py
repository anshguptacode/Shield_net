"""The model registry: one name -> one wrapper class.

Everything downstream refers to models by string key - the config file, the CLI's
``--models`` flag, the Optuna study name, the artifact manifest, the app's model picker.
Keeping that mapping in one place means a new architecture is a single entry here rather
than an edit in five files.

:func:`available` probes which models can actually run in the current environment. That
probe is what lets ``train.py`` skip LightGBM on a machine where it failed to build,
report it honestly in the results table, and carry on - instead of dying forty minutes
into a run.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Tuple

from ..logging_utils import get_logger
from .base import MissingDependency, ModelInfo, ShieldModel

log = get_logger(__name__)

__all__ = ["REGISTRY", "CATALOGUE", "build", "available", "describe_all",
           "resolve", "is_deep", "search_space_for", "UnknownModel"]


class UnknownModel(KeyError):
    """Raised for a model key that is not in the registry."""

    def __init__(self, name: str):
        super().__init__(
            f"unknown model {name!r}. Available: {', '.join(sorted(REGISTRY))}"
        )
        self.name = name


#: Registry key -> ``(module, class name)``, imported lazily.
#:
#: Lazy by design: importing this module must not pull in TensorFlow. A bare
#: ``shieldnet --help`` that takes eight seconds because it initialised CUDA is a bad
#: tool, and the Streamlit app should not load TF at all when the deployed model is
#: XGBoost.
REGISTRY: Dict[str, Tuple[str, str]] = {
    "logistic_regression": (".classical", "LogisticRegressionModel"),
    "naive_bayes":         (".classical", "NaiveBayesModel"),
    "decision_tree":       (".classical", "DecisionTreeModel"),
    "random_forest":       (".classical", "RandomForestModel"),
    "extra_trees":         (".classical", "ExtraTreesModel"),
    "xgboost":             (".classical", "XGBoostModel"),
    "lightgbm":            (".classical", "LightGBMModel"),
    "mlp":                 (".classical", "MLPModel"),
    "cnn1d":               (".deep", "CNN1DModel"),
    "bilstm":              (".deep", "BiLSTMModel"),
    "cnn_bilstm":          (".deep", "CNNBiLSTMModel"),
}

#: Aliases, so a reasonable guess in a config file works.
ALIASES: Dict[str, str] = {
    "logreg": "logistic_regression",
    "lr": "logistic_regression",
    "nb": "naive_bayes",
    "gnb": "naive_bayes",
    "dt": "decision_tree",
    "tree": "decision_tree",
    "rf": "random_forest",
    "et": "extra_trees",
    "xgb": "xgboost",
    "lgbm": "lightgbm",
    "lgb": "lightgbm",
    "cnn": "cnn1d",
    "lstm": "bilstm",
    "cnn_lstm": "cnn_bilstm",
    "cnnbilstm": "cnn_bilstm",
    "neural_network": "mlp",
}

#: Static metadata, safe to read without importing any ML library.
CATALOGUE: Dict[str, ModelInfo] = {
    "logistic_regression": ModelInfo(
        "logistic_regression", "Logistic Regression", "linear", "scikit-learn", True,
        notes="linear baseline; establishes how much is linearly separable"),
    "naive_bayes": ModelInfo(
        "naive_bayes", "Gaussian Naive Bayes", "probabilistic", "scikit-learn", True,
        notes="deliberate weak baseline; its independence assumption is violated here"),
    "decision_tree": ModelInfo(
        "decision_tree", "Decision Tree", "tree", "scikit-learn", False,
        notes="single readable tree; splits can be compared against SHAP"),
    "random_forest": ModelInfo(
        "random_forest", "Random Forest", "ensemble", "scikit-learn", False,
        notes="strong, robust, no scaling needed"),
    "extra_trees": ModelInfo(
        "extra_trees", "Extremely Randomised Trees", "ensemble", "scikit-learn", False,
        notes="more randomisation than RF; often better on noisy features"),
    "xgboost": ModelInfo(
        "xgboost", "XGBoost", "boosting", "xgboost", False,
        notes="primary candidate; early stopping on the validation split"),
    "lightgbm": ModelInfo(
        "lightgbm", "LightGBM", "boosting", "lightgbm", False,
        notes="leaf-wise growth; fastest of the boosters on this data"),
    "mlp": ModelInfo(
        "mlp", "Multi-Layer Perceptron", "neural", "scikit-learn", True,
        notes="dense-network control for the deep models"),
    "cnn1d": ModelInfo(
        "cnn1d", "1-D Convolutional Network", "deep", "tensorflow", True, True,
        notes="weight-shared local feature extractor"),
    "bilstm": ModelInfo(
        "bilstm", "Bidirectional LSTM", "deep", "tensorflow", True, True,
        notes="reads the feature vector in both directions"),
    "cnn_bilstm": ModelInfo(
        "cnn_bilstm", "CNN + Bidirectional LSTM", "deep", "tensorflow", True, True,
        notes="headline deep model; matches the published architecture"),
}


def resolve(name: str) -> str:
    """Canonical registry key for *name*, accepting aliases and loose spelling."""
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    key = ALIASES.get(key, key)
    if key not in REGISTRY:
        raise UnknownModel(name)
    return key


def _load_class(name: str) -> type:
    """Import the wrapper class for *name*. Does not touch its ML dependency."""
    key = resolve(name)
    module_name, class_name = REGISTRY[key]
    module = importlib.import_module(module_name, package=__package__)
    return getattr(module, class_name)


def build(
    name: str,
    *,
    n_classes: int,
    n_features: Optional[int] = None,
    seed: int = 42,
    n_jobs: int = -1,
    params: Optional[Dict[str, Any]] = None,
) -> ShieldModel:
    """Instantiate a model wrapper by key.

    Construction never imports the heavy dependency - that happens on ``fit`` - so this
    is cheap and safe to call for introspection, and :meth:`ModelBundle.restore` can
    rebuild a wrapper before deciding whether it needs TensorFlow.
    """
    cls = _load_class(name)
    return cls(n_classes=n_classes, n_features=n_features, seed=seed, n_jobs=n_jobs,
               params=dict(params or {}))


def is_deep(name: str) -> bool:
    """Whether *name* is a Keras model, without importing TensorFlow."""
    return CATALOGUE[resolve(name)].is_deep


def search_space_for(name: str, trial: Any) -> Dict[str, Any]:
    """Optuna search space for *name*, sampled from *trial*."""
    return _load_class(name).search_space(trial)


def _probe(package: str) -> bool:
    """Is an installable dependency importable? Cached per process by importlib."""
    if not package:
        return True
    module = {"scikit-learn": "sklearn", "tensorflow": "tensorflow"}.get(package, package)
    try:
        importlib.import_module(module)
        return True
    except Exception:                    # noqa: BLE001 - a broken TF install raises much
        return False                     # more exotic things than ImportError


def available(
    names: Optional[List[str]] = None, *, warn: bool = True
) -> Tuple[List[str], Dict[str, str]]:
    """Split *names* into what can run here and what cannot.

    Returns ``(runnable, {name: reason})``. TensorFlow is probed once even when several
    deep models are requested, because importing it is slow.

    Pass ``warn=False`` when the answer is only being displayed - a listing command that
    also emits a WARNING about everything it just listed is noise.
    """
    keys = [resolve(n) for n in (names if names is not None else REGISTRY)]
    runnable: List[str] = []
    skipped: Dict[str, str] = {}
    cache: Dict[str, bool] = {}

    for key in keys:
        package = CATALOGUE[key].package
        if package not in cache:
            cache[package] = _probe(package)
        if cache[package]:
            runnable.append(key)
        else:
            skipped[key] = f"{package} is not installed"

    if skipped and warn:
        log.warning("%d model(s) cannot run in this environment and will be skipped: "
                    "%s", len(skipped),
                    ", ".join(f"{k} ({v})" for k, v in skipped.items()))
    return runnable, skipped


def describe_all(only_available: bool = False) -> str:
    """A table of every model, for ``shieldnet models``."""
    runnable, skipped = available(warn=False)
    rows = [f"{'key':<20} {'family':<14} {'requires':<14} {'status':<11} notes",
            "-" * 104]
    for key, info in CATALOGUE.items():
        status = "ready" if key in runnable else "unavailable"
        if only_available and status != "ready":
            continue
        rows.append(f"{key:<20} {info.family:<14} "
                    f"{(info.package or 'numpy only'):<14} {status:<11} {info.notes}")
    if skipped and not only_available:
        rows.append("")
        rows.append("Install the missing packages with:  pip install -r requirements.txt")
    return "\n".join(rows)
