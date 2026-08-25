"""Classical baselines and the gradient-boosting workhorses.

Model choices, and what each is here to prove
---------------------------------------------
``logistic_regression``
    The linear baseline. Its job is to establish how much of CICIDS2017 is linearly
    separable - a surprising amount, because port numbers and packet-length statistics
    separate several attack families almost trivially. Any deep model that cannot clear
    this comfortably is not earning its complexity.
``naive_bayes``
    Deliberately the weakest model in the study. Its independence assumption is badly
    violated here (packet-length mean, max and std are all functions of each other), so
    it quantifies how much the correlated-feature structure actually matters.
``decision_tree``
    A single interpretable tree. Useful in the report because its splits can be read
    directly and compared against what SHAP says about the ensembles.
``random_forest`` / ``extra_trees``
    Bagged ensembles: strong, robust, no scaling needed, and cheap to explain with
    TreeExplainer.
``xgboost`` / ``lightgbm``
    The models that actually win on tabular flow data, and the project's primary
    candidates. Both get early stopping against the validation split.
``mlp``
    A dense network via sklearn, to separate "deep learning helps" from "the specific
    1-D convolutional architecture helps" - without this control, the CNN's win cannot
    be attributed.

Every class here degrades gracefully: if its library is missing, constructing it raises
:class:`~shieldnet.models.base.MissingDependency` with the pip command, and ``train.py``
records it as skipped instead of aborting a long run.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..logging_utils import get_logger
from .base import MissingDependency, ShieldModel, as_array

log = get_logger(__name__)

__all__ = ["LogisticRegressionModel", "NaiveBayesModel", "DecisionTreeModel",
           "RandomForestModel", "ExtraTreesModel", "XGBoostModel", "LightGBMModel",
           "MLPModel"]


def _sklearn(model_key: str, module: str, cls_name: str):
    """Import one sklearn estimator, or raise with instructions."""
    try:
        mod = __import__(f"sklearn.{module}", fromlist=[cls_name])
    except ImportError as exc:
        raise MissingDependency(model_key, "scikit-learn") from exc
    return getattr(mod, cls_name)


# ---------------------------------------------------------------------------
# linear and probabilistic baselines
# ---------------------------------------------------------------------------

class LogisticRegressionModel(ShieldModel):
    """Multinomial logistic regression with L2 regularisation."""

    key, label, family = "logistic_regression", "Logistic Regression", "linear"
    package, needs_scaling = "scikit-learn", True

    def fit(self, X, y, *, X_val=None, y_val=None, class_weight=None):
        LogisticRegression = _sklearn(self.key, "linear_model", "LogisticRegression")
        y_dense = self._densify(y)
        params: Dict[str, Any] = {
            # lbfgs handles the multinomial case natively and is the only solver here
            # that scales acceptably to 300k rows without a sparse matrix.
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "n_jobs": self.n_jobs,
            "random_state": self.seed,
            **self.params,
        }
        self.model = LogisticRegression(**params)
        # Passed as sample weights rather than class_weight="balanced" so the *capped*
        # weights from preprocess.class_weights are what actually get used.
        self.model.fit(as_array(X), y_dense,
                       sample_weight=self._sample_weight(y, class_weight))
        n_iter = np.atleast_1d(getattr(self.model, "n_iter_", [0]))
        self.fit_history_["n_iter"] = int(n_iter.max())
        if self.fit_history_["n_iter"] >= params["max_iter"]:
            log.warning("logistic regression hit max_iter=%d without converging; the "
                        "coefficients are not at the optimum. Raise max_iter or check "
                        "that the features are scaled.", params["max_iter"])
        return self

    def predict_proba(self, X):
        self._require_fitted()
        return self._expand_proba(self.model.predict_proba(as_array(X)))

    @staticmethod
    def search_space(trial):
        return {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "max_iter": trial.suggest_categorical("max_iter", [1000, 2000, 3000]),
        }


class NaiveBayesModel(ShieldModel):
    """Gaussian naive Bayes - the intentional weak baseline."""

    key, label, family = "naive_bayes", "Gaussian Naive Bayes", "probabilistic"
    package, needs_scaling = "scikit-learn", True

    def fit(self, X, y, *, X_val=None, y_val=None, class_weight=None):
        GaussianNB = _sklearn(self.key, "naive_bayes", "GaussianNB")
        y_dense = self._densify(y)
        # var_smoothing matters more than usual here: several CICIDS2017 features are
        # constant *within* a class (a flag that is always 0 for BENIGN), giving zero
        # variance and an infinite likelihood without smoothing.
        params = {"var_smoothing": 1e-8, **self.params}
        self.model = GaussianNB(**params)
        self.model.fit(as_array(X), y_dense,
                       sample_weight=self._sample_weight(y, class_weight))
        return self

    def predict_proba(self, X):
        self._require_fitted()
        return self._expand_proba(self.model.predict_proba(as_array(X)))

    @staticmethod
    def search_space(trial):
        return {"var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-3,
                                                     log=True)}


# ---------------------------------------------------------------------------
# trees
# ---------------------------------------------------------------------------

class DecisionTreeModel(ShieldModel):
    """A single tree, kept shallow enough to actually read."""

    key, label, family = "decision_tree", "Decision Tree", "tree"
    package, needs_scaling, is_tree = "scikit-learn", False, True

    def fit(self, X, y, *, X_val=None, y_val=None, class_weight=None):
        DecisionTreeClassifier = _sklearn(self.key, "tree", "DecisionTreeClassifier")
        y_dense = self._densify(y)
        params = {
            "max_depth": 18,
            "min_samples_leaf": 2,
            "criterion": "gini",
            "random_state": self.seed,
            **self.params,
        }
        self.model = DecisionTreeClassifier(**params)
        self.model.fit(as_array(X), y_dense,
                       sample_weight=self._sample_weight(y, class_weight))
        self.fit_history_["depth"] = int(self.model.get_depth())
        self.fit_history_["leaves"] = int(self.model.get_n_leaves())
        return self

    def predict_proba(self, X):
        self._require_fitted()
        return self._expand_proba(self.model.predict_proba(as_array(X)))

    @staticmethod
    def search_space(trial):
        return {
            "max_depth": trial.suggest_int("max_depth", 4, 40),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        }


class _ForestBase(ShieldModel):
    """Shared plumbing for the two bagged ensembles."""

    _module, _cls = "ensemble", ""

    def fit(self, X, y, *, X_val=None, y_val=None, class_weight=None):
        Estimator = _sklearn(self.key, self._module, self._cls)
        y_dense = self._densify(y)
        params = {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "n_jobs": self.n_jobs,
            "random_state": self.seed,
            **self.params,
        }
        self.model = Estimator(**params)
        self.model.fit(as_array(X), y_dense,
                       sample_weight=self._sample_weight(y, class_weight))
        return self

    def predict_proba(self, X):
        self._require_fitted()
        return self._expand_proba(self.model.predict_proba(as_array(X)))

    @staticmethod
    def search_space(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [None, 12, 20, 30]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical("max_features",
                                                      ["sqrt", "log2", 0.5]),
        }


class RandomForestModel(_ForestBase):
    key, label, family = "random_forest", "Random Forest", "ensemble"
    package, needs_scaling, is_tree = "scikit-learn", False, True
    _cls = "RandomForestClassifier"


class ExtraTreesModel(_ForestBase):
    key, label, family = "extra_trees", "Extremely Randomised Trees", "ensemble"
    package, needs_scaling, is_tree = "scikit-learn", False, True
    _cls = "ExtraTreesClassifier"


# ---------------------------------------------------------------------------
# gradient boosting
# ---------------------------------------------------------------------------

class XGBoostModel(ShieldModel):
    """XGBoost, the project's primary candidate.

    Early stopping is set up defensively because the API moved: in xgboost >= 2.0
    ``early_stopping_rounds`` belongs in the constructor and passing it to ``fit`` is a
    ``TypeError``, while in 1.x it belongs in ``fit``. Both are attempted.
    """

    key, label, family = "xgboost", "XGBoost", "boosting"
    package, needs_scaling, is_tree = "xgboost", False, True

    def fit(self, X, y, *, X_val=None, y_val=None, class_weight=None):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise MissingDependency(
                self.key, "xgboost",
                extra="On Apple silicon you may also need: brew install libomp\n",
            ) from exc

        y_dense = self._densify(y)
        n_dense = self.n_classes_fitted
        params: Dict[str, Any] = {
            "n_estimators": 600,
            "max_depth": 8,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
            "objective": "multi:softprob",
            "num_class": n_dense,
            "tree_method": "hist",
            "eval_metric": "mlogloss",
            "n_jobs": self.n_jobs,
            "random_state": self.seed,
            "verbosity": 0,
            **self.params,
        }
        patience = params.pop("early_stopping_rounds", 40)

        has_val = X_val is not None and y_val is not None and len(np.asarray(y_val)) > 0
        eval_set = None
        if has_val:
            # The validation labels must be compressed with the *training* mapping, and
            # any class absent from training cannot be scored - drop those rows rather
            # than emitting a label outside [0, num_class).
            y_val_arr = np.asarray(y_val, dtype=np.int64)
            keep = np.isin(y_val_arr, self.classes_present_)
            if not keep.all():
                log.info("%s: %d validation row(s) belong to classes absent from this "
                         "training fold and are excluded from early stopping",
                         self.key, int((~keep).sum()))
            if keep.any():
                lookup = np.full(int(self.classes_present_.max()) + 1, -1, np.int64)
                lookup[self.classes_present_] = np.arange(n_dense)
                eval_set = [(as_array(X_val)[keep], lookup[y_val_arr[keep]])]

        if eval_set:
            params["early_stopping_rounds"] = patience

        self.model = XGBClassifier(**params)
        weight = self._sample_weight(y, class_weight)
        try:
            self.model.fit(as_array(X), y_dense, sample_weight=weight,
                           eval_set=eval_set, verbose=False)
        except TypeError:
            # xgboost 1.x: early stopping is a fit argument, not a constructor one.
            params.pop("early_stopping_rounds", None)
            self.model = XGBClassifier(**params)
            kwargs: Dict[str, Any] = {"sample_weight": weight, "verbose": False}
            if eval_set:
                kwargs.update(eval_set=eval_set, early_stopping_rounds=patience)
            self.model.fit(as_array(X), y_dense, **kwargs)

        best = getattr(self.model, "best_iteration", None)
        if best is not None:
            self.fit_history_["best_iteration"] = int(best)
            log.info("%s: early stopping chose %d of %d boosting rounds", self.key,
                     int(best) + 1, params["n_estimators"])
        return self

    def predict_proba(self, X):
        self._require_fitted()
        return self._expand_proba(self.model.predict_proba(as_array(X)))

    @staticmethod
    def search_space(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 14),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        }


class LightGBMModel(ShieldModel):
    """LightGBM. Leaf-wise growth, so ``num_leaves`` is the capacity knob, not depth.

    ``min_child_samples`` is lowered from LightGBM's default of 20 because several
    classes here have fewer than 20 training rows after splitting - at the default,
    LightGBM cannot create a pure leaf for them and quietly never predicts them.
    """

    key, label, family = "lightgbm", "LightGBM", "boosting"
    package, needs_scaling, is_tree = "lightgbm", False, True

    def fit(self, X, y, *, X_val=None, y_val=None, class_weight=None):
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise MissingDependency(self.key, "lightgbm") from exc

        y_dense = self._densify(y)
        params: Dict[str, Any] = {
            "n_estimators": 800,
            "num_leaves": 63,
            "learning_rate": 0.08,
            "min_child_samples": 5,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "objective": "multiclass",
            "num_class": self.n_classes_fitted,
            "n_jobs": self.n_jobs,
            "random_state": self.seed,
            "verbosity": -1,
            **self.params,
        }
        patience = params.pop("early_stopping_rounds", 50)
        self.model = lgb.LGBMClassifier(**params)

        fit_kwargs: Dict[str, Any] = {
            "sample_weight": self._sample_weight(y, class_weight)
        }
        if X_val is not None and y_val is not None and len(np.asarray(y_val)):
            y_val_arr = np.asarray(y_val, dtype=np.int64)
            keep = np.isin(y_val_arr, self.classes_present_)
            if keep.any():
                lookup = np.full(int(self.classes_present_.max()) + 1, -1, np.int64)
                lookup[self.classes_present_] = np.arange(self.n_classes_fitted)
                fit_kwargs["eval_set"] = [(as_array(X_val)[keep],
                                           lookup[y_val_arr[keep]])]
                # Callbacks, not the removed early_stopping_rounds fit argument -
                # LightGBM 4.x deleted that keyword outright.
                fit_kwargs["callbacks"] = [
                    lgb.early_stopping(patience, verbose=False),
                    lgb.log_evaluation(0),
                ]

        try:
            self.model.fit(as_array(X), y_dense, **fit_kwargs)
        except TypeError:
            fit_kwargs.pop("callbacks", None)
            self.model.fit(as_array(X), y_dense, **fit_kwargs)

        best = getattr(self.model, "best_iteration_", None)
        if best:
            self.fit_history_["best_iteration"] = int(best)
        return self

    def predict_proba(self, X):
        self._require_fitted()
        return self._expand_proba(self.model.predict_proba(as_array(X)))

    @staticmethod
    def search_space(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 2, 60),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        }


# ---------------------------------------------------------------------------
# dense network (the control for the deep models)
# ---------------------------------------------------------------------------

class MLPModel(ShieldModel):
    """Dense feed-forward network via sklearn, with its own internal early stopping."""

    key, label, family = "mlp", "Multi-Layer Perceptron", "neural"
    package, needs_scaling = "scikit-learn", True

    def fit(self, X, y, *, X_val=None, y_val=None, class_weight=None):
        MLPClassifier = _sklearn(self.key, "neural_network", "MLPClassifier")
        y_dense = self._densify(y)
        params = {
            "hidden_layer_sizes": (128, 64),
            "activation": "relu",
            "alpha": 1e-4,
            "batch_size": 512,
            "learning_rate_init": 1e-3,
            "max_iter": 120,
            "early_stopping": True,
            "n_iter_no_change": 10,
            "validation_fraction": 0.1,
            "random_state": self.seed,
            **self.params,
        }
        self.model = MLPClassifier(**params)
        if class_weight:
            # MLPClassifier accepts neither class_weight nor sample_weight, so the only
            # way to weight classes is to resample. Say so rather than appearing to
            # honour a setting that is being ignored.
            log.info("%s ignores class weights (sklearn's MLPClassifier supports "
                     "neither class_weight nor sample_weight); rely on SMOTE instead",
                     self.key)
        self.model.fit(as_array(X), y_dense)
        self.fit_history_["n_iter"] = int(getattr(self.model, "n_iter_", 0))
        self.fit_history_["loss"] = float(getattr(self.model, "loss_", float("nan")))
        return self

    def predict_proba(self, X):
        self._require_fitted()
        return self._expand_proba(self.model.predict_proba(as_array(X)))

    @staticmethod
    def search_space(trial):
        depth = trial.suggest_int("n_layers", 1, 3)
        width = trial.suggest_categorical("width", [64, 128, 256])
        return {
            "hidden_layer_sizes": tuple(max(16, width // (2 ** i))
                                        for i in range(depth)),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2,
                                                      log=True),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
        }
