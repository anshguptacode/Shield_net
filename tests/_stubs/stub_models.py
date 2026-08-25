"""Genuinely working numpy models, used to exercise the pipeline without ML libraries.

Why these exist
---------------
Every model in :mod:`shieldnet.models` needs scikit-learn, XGBoost, LightGBM or
TensorFlow. That makes the orchestration code - ``tune.py``, ``train.py``, ``explain.py``,
``inference.py`` - untestable in any environment where those are not installed, which
includes this project's own development sandbox and any CI runner that has not paid the
two-minute cost of installing TensorFlow.

The temptation is to write mocks that return canned arrays. That tests nothing: a mock
cannot catch a shape mismatch, cannot exercise the dense-label remapping, and will
happily "train" on data that would make a real estimator raise. So these are not mocks.
:class:`SoftmaxModel` is a real multinomial logistic regression fitted by gradient
descent, and :class:`CentroidModel` is a real nearest-centroid classifier. They honour
``sample_weight``, they respond to their hyper-parameters in the direction you would
expect, and they implement the full :class:`~shieldnet.models.base.ShieldModel`
interface - including ``feature_importance`` - so the orchestration code around them is
under genuine test.

They are deliberately *not* registered in the production catalogue. Tests register them
by putting this module on ``sys.path`` and adding an entry to ``REGISTRY``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from shieldnet.models.base import ShieldModel, as_array

__all__ = ["SoftmaxModel", "CentroidModel", "register", "unregister"]


class SoftmaxModel(ShieldModel):
    """Multinomial logistic regression by full-batch gradient descent, in numpy.

    Real learning, so the tuner has a real signal to find: ``learning_rate`` too high
    diverges, too low underfits, and ``l2`` genuinely trades bias for variance.
    """

    key, label, family = "stub_softmax", "Softmax Regression (numpy)", "linear"
    package, needs_scaling = "", True

    def fit(self, X, y, *, X_val=None, y_val=None, class_weight=None):
        y_dense = self._densify(y)
        Xa = np.asarray(as_array(X), dtype=np.float64)
        n, f = Xa.shape
        k = self.n_classes_fitted

        lr = float(self.params.get("learning_rate", 0.5))
        epochs = int(self.params.get("epochs", 200))
        l2 = float(self.params.get("l2", 1e-4))

        Xb = np.hstack([Xa, np.ones((n, 1))])          # bias column
        W = np.zeros((f + 1, k))
        Y = np.zeros((n, k))
        Y[np.arange(n), y_dense] = 1.0

        sw = self._sample_weight(y, class_weight)
        sw = np.ones(n) if sw is None else np.asarray(sw, dtype=np.float64)
        sw = sw / sw.mean()                            # keep the gradient scale stable
        sw = sw[:, None]

        losses = []
        for _ in range(epochs):
            logits = Xb @ W
            logits -= logits.max(axis=1, keepdims=True)     # stabilise the exponential
            expo = np.exp(logits)
            P = expo / expo.sum(axis=1, keepdims=True)
            grad = Xb.T @ ((P - Y) * sw) / n + l2 * W
            W -= lr * grad
            if not np.isfinite(W).all():
                # Diverged. Report it as a bad trial rather than returning NaN
                # probabilities that would poison every metric downstream.
                raise FloatingPointError(
                    f"softmax diverged at learning_rate={lr}; the tuner should record "
                    "this as a failed trial")
            losses.append(float(-(np.log(np.clip(P[np.arange(n), y_dense], 1e-15, 1))
                                  * sw.ravel()).mean()))

        self.model = W
        self.fit_history_ = {"epochs": epochs, "final_loss": losses[-1],
                             "first_loss": losses[0]}
        return self

    def predict_proba(self, X):
        self._require_fitted()
        Xa = np.asarray(as_array(X), dtype=np.float64)
        Xb = np.hstack([Xa, np.ones((Xa.shape[0], 1))])
        logits = Xb @ self.model
        logits -= logits.max(axis=1, keepdims=True)
        expo = np.exp(logits)
        return self._expand_proba(expo / expo.sum(axis=1, keepdims=True))

    def feature_importance(self):
        if self.model is None:
            return None
        return np.abs(self.model[:-1]).mean(axis=1)     # drop the bias row

    @staticmethod
    def search_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 2.0, log=True),
            "epochs": trial.suggest_int("epochs", 40, 400, step=40),
            "l2": trial.suggest_float("l2", 1e-7, 1e-1, log=True),
        }


class CentroidModel(ShieldModel):
    """Nearest class centroid with a softmax over negative distances.

    Cheap enough to stand in for a slow model in timing tests, and it has the useful
    property of predicting every class it was trained on - which makes it a good probe
    for code paths that assume some class is never predicted.
    """

    key, label, family = "stub_centroid", "Nearest Centroid (numpy)", "other"
    package, needs_scaling = "", True

    def fit(self, X, y, *, X_val=None, y_val=None, class_weight=None):
        y_dense = self._densify(y)
        Xa = np.asarray(as_array(X), dtype=np.float64)
        k = self.n_classes_fitted
        centroids = np.zeros((k, Xa.shape[1]))
        for c in range(k):
            rows = Xa[y_dense == c]
            centroids[c] = rows.mean(axis=0) if len(rows) else 0.0
        self.model = centroids
        self.temperature = float(self.params.get("temperature", 1.0))
        return self

    def predict_proba(self, X):
        self._require_fitted()
        Xa = np.asarray(as_array(X), dtype=np.float64)
        d2 = ((Xa[:, None, :] - self.model[None, :, :]) ** 2).sum(axis=2)
        logits = -d2 / max(self.temperature, 1e-6)
        logits -= logits.max(axis=1, keepdims=True)
        expo = np.exp(logits)
        return self._expand_proba(expo / expo.sum(axis=1, keepdims=True))

    @staticmethod
    def search_space(trial):
        return {"temperature": trial.suggest_float("temperature", 0.1, 50.0, log=True)}


# ---------------------------------------------------------------------------
# registry plumbing for tests
# ---------------------------------------------------------------------------

_ENTRIES = {
    "stub_softmax": (__name__, "SoftmaxModel"),
    "stub_centroid": (__name__, "CentroidModel"),
}


def register() -> None:
    """Add the stubs to the live registry.

    ``importlib.import_module`` ignores the ``package`` argument for an absolute module
    name, so an absolute name here resolves against ``sys.path`` and no change to
    :mod:`shieldnet.models.registry` is needed.
    """
    from shieldnet.models import base, registry

    for key, entry in _ENTRIES.items():
        registry.REGISTRY[key] = entry
        cls = SoftmaxModel if "softmax" in key else CentroidModel
        registry.CATALOGUE[key] = base.ModelInfo(
            key, cls.label, cls.family, "", cls.needs_scaling, False,
            notes="numpy test double; not part of the study")


def unregister() -> None:
    from shieldnet.models import registry

    for key in _ENTRIES:
        registry.REGISTRY.pop(key, None)
        registry.CATALOGUE.pop(key, None)
