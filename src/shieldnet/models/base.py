"""The common interface every model in this project implements.

Why a wrapper at all, instead of using estimators directly
----------------------------------------------------------
XGBoost, LightGBM, scikit-learn and Keras disagree about almost everything: whether
class weights are a constructor argument or a ``fit`` keyword, whether early stopping
takes a validation tuple or a callback, whether ``predict`` returns labels or
probabilities, and what shape the input should be. Threading those differences through
``train.py``, ``tune.py``, ``evaluate.py`` and ``explain.py`` four times over is how
those files become unmaintainable. One thin wrapper each, and the rest of the pipeline
speaks a single language: ``fit(X, y, X_val, y_val, class_weight)`` and
``predict_proba(X) -> (n_rows, n_classes)``.

Two defences live here because both are silent failures
------------------------------------------------------
**Dense label remapping.** XGBoost requires training labels to be exactly
``0..num_class-1``. Hand it labels ``[0, 1, 2, 4]`` - which happens the moment a class
is absent from a fold, and with a 21-row class and 3-fold CV that happens often - and it
raises ``label must be in [0, num_class)``, or worse, silently trains a 5-class model
whose class 3 means nothing. So labels are compressed to a dense range for fitting, and
the mapping is remembered.

**Probability expansion.** Having compressed, ``predict_proba`` would return one column
per *present* class. Downstream code indexes probability columns by the codec's class
index, so a 12-column matrix where 13 were expected either raises far away from the
cause or, if the shapes happen to line up, mislabels every prediction. ``_expand_proba``
puts the columns back where they belong and fills the absent classes with zero.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["ShieldModel", "ModelInfo", "MissingDependency", "as_array"]


class MissingDependency(ImportError):
    """Raised when a model's library is not installed, with install instructions."""

    def __init__(self, model: str, package: str, extra: str = ""):
        self.model, self.package = model, package
        super().__init__(
            f"the {model!r} model needs {package}, which is not installed.\n"
            f"    pip install {package}\n"
            f"{extra}"
            f"Or drop {model!r} from train.models in your config - the pipeline runs "
            "whatever is available and reports the rest as skipped."
        )


@dataclass
class ModelInfo:
    """Static description of a model, for the CLI listing and the app sidebar."""

    key: str
    label: str
    family: str                     # linear | tree | ensemble | boosting | deep
    package: str                    # pip name of the dependency, "" for numpy-only
    needs_scaling: bool
    is_deep: bool = False
    notes: str = ""

    def render(self) -> str:
        dep = self.package or "numpy only"
        # 14 wide, not 9: "probabilistic" is 13 characters and would push the rest of
        # the row out of alignment.
        return f"{self.key:<20} {self.family:<14} {dep:<14} {self.notes}"


def as_array(X: Any) -> np.ndarray:
    """Coerce a frame or array to a contiguous float32 matrix.

    float32 rather than float64: it halves the memory a tree ensemble copies internally,
    XGBoost and LightGBM convert to float32 anyway, and Keras wants float32. The
    precision loss is meaningless next to the noise in flow measurements.
    """
    if hasattr(X, "to_numpy"):
        X = X.to_numpy()
    arr = np.ascontiguousarray(X, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


class ShieldModel(abc.ABC):
    """Base class for every classifier in the project."""

    #: Registry key. Set by subclasses.
    key: str = ""
    #: Human label for reports.
    label: str = ""
    family: str = "other"
    package: str = ""
    #: Whether this model cares about feature scale. Trees do not; everything else does.
    needs_scaling: bool = True
    #: Keras-backed models set this; :class:`~shieldnet.persist.ModelBundle` checks it.
    is_deep: bool = False
    #: Whether SHAP's fast TreeExplainer applies.
    is_tree: bool = False

    def __init__(
        self,
        *,
        n_classes: int,
        n_features: Optional[int] = None,
        seed: int = 42,
        n_jobs: int = -1,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")
        self.n_classes = int(n_classes)
        self.n_features = int(n_features) if n_features else None
        self.seed = int(seed)
        self.n_jobs = int(n_jobs)
        self.params: Dict[str, Any] = dict(params or {})
        self.model: Any = None
        #: Sorted class indices actually seen during fit.
        self.classes_present_: Optional[np.ndarray] = None
        self.fit_history_: Dict[str, Any] = {}

    # -- label compression ---------------------------------------------------

    def _densify(self, y: np.ndarray) -> np.ndarray:
        """Record which classes are present and compress labels to ``0..m-1``."""
        y = np.asarray(y, dtype=np.int64)
        present = np.unique(y)
        if present.size < 2:
            raise ValueError(
                f"training labels contain only class {present.tolist()}; a classifier "
                "needs at least two. This usually means a CV fold was built without "
                "stratification."
            )
        self.classes_present_ = present
        if present.size < self.n_classes:
            log.debug("%s: %d of %d classes present in this fit; labels compressed",
                      self.key, present.size, self.n_classes)
        lookup = np.full(int(present.max()) + 1, -1, dtype=np.int64)
        lookup[present] = np.arange(present.size)
        return lookup[y]

    @property
    def n_classes_fitted(self) -> int:
        """How many classes the underlying estimator actually predicts."""
        if self.classes_present_ is None:
            return self.n_classes
        return int(self.classes_present_.size)

    def _expand_proba(self, proba: np.ndarray) -> np.ndarray:
        """Scatter dense probability columns back to the full class order."""
        proba = np.asarray(proba, dtype=np.float64)
        if proba.ndim == 1:                       # binary estimators return one column
            proba = np.column_stack([1.0 - proba, proba])
        if self.classes_present_ is None or proba.shape[1] == self.n_classes:
            return proba
        if proba.shape[1] != self.classes_present_.size:
            raise ValueError(
                f"{self.key}: estimator returned {proba.shape[1]} probability columns "
                f"but {self.classes_present_.size} classes were present at fit time"
            )
        full = np.zeros((len(proba), self.n_classes), dtype=np.float64)
        full[:, self.classes_present_] = proba
        return full

    def _sample_weight(
        self, y: np.ndarray, class_weight: Optional[Dict[int, float]]
    ) -> Optional[np.ndarray]:
        """Turn a class-weight mapping into per-row weights.

        Per-row weights are the lowest common denominator: every library here accepts
        ``sample_weight`` in ``fit``, whereas ``class_weight`` is spelled differently or
        missing in half of them.
        """
        if not class_weight:
            return None
        y = np.asarray(y, dtype=np.int64)
        table = np.ones(self.n_classes, dtype=np.float64)
        for idx, weight in class_weight.items():
            if 0 <= int(idx) < self.n_classes:
                table[int(idx)] = float(weight)
        return table[y]

    # -- the interface -------------------------------------------------------

    @abc.abstractmethod
    def fit(
        self,
        X: Any,
        y: np.ndarray,
        *,
        X_val: Any = None,
        y_val: Optional[np.ndarray] = None,
        class_weight: Optional[Dict[int, float]] = None,
    ) -> "ShieldModel":
        """Train in place and return self.

        ``X_val``/``y_val`` are advisory: models that support early stopping use them,
        the rest ignore them. Passing them is always safe.
        """

    @abc.abstractmethod
    def predict_proba(self, X: Any) -> np.ndarray:
        """Class probabilities, shape ``(n_rows, n_classes)``, rows summing to 1."""

    def predict(self, X: Any) -> np.ndarray:
        """Hard predictions as class indices in the codec's numbering."""
        return np.asarray(self.predict_proba(X)).argmax(axis=1)

    # -- optional capabilities ----------------------------------------------

    @staticmethod
    def search_space(trial: Any) -> Dict[str, Any]:
        """Optuna search space. Empty means "nothing to tune"."""
        return {}

    def feature_importance(self) -> Optional[np.ndarray]:
        """Native importance, when the model has one. ``None`` otherwise.

        Note this is *not* the project's explanation of record - it is unsigned,
        class-agnostic and, for tree ensembles, biased towards high-cardinality
        features. SHAP is the explanation of record; this is a cheap cross-check.
        """
        for attr in ("feature_importances_", "coef_"):
            value = getattr(self.model, attr, None)
            if value is None:
                continue
            arr = np.abs(np.asarray(value, dtype=np.float64))
            return arr.mean(axis=0) if arr.ndim > 1 else arr
        return None

    @property
    def fitted(self) -> bool:
        return self.model is not None

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError(
                f"{self.key} has not been fitted. Call fit() first, or restore a "
                "trained bundle with ModelBundle.restore()."
            )

    def describe(self) -> str:
        bits = [f"{self.key} ({self.family})", f"{self.n_classes} classes"]
        if self.n_features:
            bits.append(f"{self.n_features} features")
        if self.params:
            shown = ", ".join(f"{k}={v}" for k, v in list(self.params.items())[:6])
            bits.append(shown)
        return " | ".join(bits)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "fitted" if self.fitted else "unfitted"
        return f"<{type(self).__name__} {self.key} {state}>"
