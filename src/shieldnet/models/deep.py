"""Keras models: 1-D CNN, BiLSTM, and the hybrid CNN-BiLSTM.

An honest note about applying convolution to tabular flow features
-----------------------------------------------------------------
A 1-D convolution assumes adjacent positions are related - that is what makes it work on
audio, text and time series. The 25 selected CICIDS2017 features are **not** a sequence:
column 7 sitting next to column 8 is an artefact of the order CICFlowMeter writes its
output, not a statement that the two measurements are neighbours. Papers that stack
Conv1D on tabular flow features rarely acknowledge this.

So why is it here, and why does it work? A Conv1D over the feature axis is really a
weight-shared, locally-connected layer: each filter learns one small pattern and looks
for it at every offset. That is a strong regulariser - far fewer parameters than a dense
layer - and on 300k rows with 13 badly imbalanced classes, regularisation is worth more
than a perfectly matched inductive bias. The pipeline also feeds features **in selection
rank order**, so neighbouring columns have comparable importance and the locality
assumption is at least not actively wrong.

The BiLSTM is included on the same terms: it reads the feature vector in both directions
and its gates learn which features to carry forward, which is a legitimate mechanism even
without genuine temporal ordering. What none of these can claim is to model *flow
sequences over time* - that would need the flows grouped and ordered per host, which
CICIDS2017's row-per-flow layout does not support without re-deriving sessions from the
timestamps we deliberately dropped as leakage. The report should say that rather than
implying the LSTM is doing temporal modelling.

Keras 2 vs Keras 3
------------------
``import keras`` gives Keras 3 when TensorFlow >= 2.16 is installed and Keras 2
otherwise; ``tensorflow.keras`` is the older spelling and is a shim in new versions.
:func:`_keras` tries both and reports which it found, because the two differ in ways that
matter here: Keras 3 requires the ``.keras`` extension on ``save()``, and it renamed the
``lr`` argument to ``learning_rate`` (already the spelling used below).
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..logging_utils import get_logger
from .base import MissingDependency, ShieldModel, as_array

log = get_logger(__name__)

__all__ = ["KerasModel", "CNN1DModel", "BiLSTMModel", "CNNBiLSTMModel"]


def _keras():
    """Return the Keras module, preferring Keras 3, or raise with instructions."""
    # Quiet the C++ logs before TF is imported; after import it is too late.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import keras                       # Keras 3, or Keras 2 standalone
        return keras
    except ImportError:
        pass
    try:
        from tensorflow import keras       # TF-bundled Keras 2
        return keras
    except ImportError as exc:
        raise MissingDependency(
            "deep models", "tensorflow",
            extra=("For a GPU on Colab nothing extra is needed - the runtime already "
                   "has it. Locally, CPU-only TensorFlow trains these models in a few "
                   "minutes on the working chunk.\n"),
        ) from exc


class KerasModel(ShieldModel):
    """Shared plumbing for the Keras architectures.

    Subclasses implement :meth:`_architecture` only. Everything else - reshaping,
    early stopping, class weights, native save/load, and keeping the object picklable -
    lives here.
    """

    is_deep = True
    needs_scaling = True
    package = "tensorflow"
    family = "deep"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.epochs = int(self.params.pop("epochs", 40))
        self.batch_size = int(self.params.pop("batch_size", 512))
        self.patience = int(self.params.pop("early_stopping_patience", 6))
        self.learning_rate = float(self.params.pop("learning_rate", 1e-3))
        #: Plain lists only - a Keras ``History`` object is not reliably picklable.
        self.history_: Dict[str, List[float]] = {}

    # -- shape handling ------------------------------------------------------

    def _reshape(self, X: Any) -> np.ndarray:
        """``(n, features)`` -> ``(n, features, 1)``, the layout Conv1D and LSTM want."""
        arr = as_array(X)
        if arr.ndim == 3:
            return arr
        return arr.reshape(arr.shape[0], arr.shape[1], 1)

    # -- architecture --------------------------------------------------------

    def _architecture(self, keras, n_features: int, n_out: int):  # pragma: no cover
        raise NotImplementedError

    def _compile(self, model, keras) -> None:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            # sparse_* keeps the targets as integers: one-hot encoding 300k x 13 floats
            # wastes 15 MB and buys nothing.
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def build_graph(self, n_features: int) -> Any:
        """Construct and compile the network without training it."""
        keras = _keras()
        self.n_features = int(n_features)
        model = self._architecture(keras, int(n_features), self.n_classes_fitted)
        self._compile(model, keras)
        return model

    # -- training ------------------------------------------------------------

    def fit(self, X, y, *, X_val=None, y_val=None, class_weight=None):
        keras = _keras()
        y_dense = self._densify(y)
        X_arr = self._reshape(X)
        n_features = X_arr.shape[1]

        self.model = self.build_graph(n_features)

        validation: Optional[Tuple[np.ndarray, np.ndarray]] = None
        if X_val is not None and y_val is not None and len(np.asarray(y_val)):
            y_val_arr = np.asarray(y_val, dtype=np.int64)
            keep = np.isin(y_val_arr, self.classes_present_)
            if keep.any():
                lookup = np.full(int(self.classes_present_.max()) + 1, -1, np.int64)
                lookup[self.classes_present_] = np.arange(self.n_classes_fitted)
                validation = (self._reshape(X_val)[keep], lookup[y_val_arr[keep]])
            if not keep.all():
                log.info("%s: %d validation row(s) from classes absent in training were "
                         "excluded", self.key, int((~keep).sum()))

        callbacks = []
        monitor = "val_loss" if validation is not None else "loss"
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor=monitor, patience=self.patience, restore_best_weights=True,
            verbose=0,
        ))
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, factor=0.5, patience=max(2, self.patience // 2),
            min_lr=1e-6, verbose=0,
        ))
        if validation is None:
            log.warning("%s is training without a validation split, so early stopping "
                        "watches the training loss and cannot detect overfitting",
                        self.key)

        weights = None
        if class_weight:
            # Keras wants the dense label space, so remap the keys.
            dense_map = {int(orig): i for i, orig in enumerate(self.classes_present_)}
            weights = {dense_map[k]: float(v) for k, v in class_weight.items()
                       if int(k) in dense_map}

        log.info("%s: training up to %d epoch(s), batch %d, %s rows%s", self.key,
                 self.epochs, self.batch_size, f"{len(X_arr):,}",
                 f", validating on {len(validation[0]):,}" if validation else "")

        history = self.model.fit(
            X_arr, y_dense,
            validation_data=validation,
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=weights,
            callbacks=callbacks,
            verbose=0,
            shuffle=True,
        )
        self.history_ = {k: [float(v) for v in vals]
                         for k, vals in history.history.items()}
        epochs_run = len(self.history_.get("loss", []))
        self.fit_history_ = {
            "epochs_run": epochs_run,
            "epochs_allowed": self.epochs,
            "stopped_early": epochs_run < self.epochs,
            "final_loss": self.history_.get("loss", [float("nan")])[-1],
            "best_val_loss": (min(self.history_["val_loss"])
                              if "val_loss" in self.history_ else None),
            "parameters": int(self.model.count_params()),
        }
        log.info("%s: stopped after %d epoch(s); best val loss %s", self.key,
                 epochs_run,
                 f"{self.fit_history_['best_val_loss']:.4f}"
                 if self.fit_history_["best_val_loss"] is not None else "n/a")
        return self

    def predict_proba(self, X):
        self._require_fitted()
        proba = self.model.predict(self._reshape(X), verbose=0,
                                  batch_size=max(self.batch_size, 1024))
        return self._expand_proba(np.asarray(proba, dtype=np.float64))

    # -- persistence ---------------------------------------------------------

    def save_native(self, path: os.PathLike | str) -> Path:
        """Write the Keras graph and weights in the native format."""
        self._require_fitted()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix != ".keras":
            # Keras 3 refuses any other extension; be explicit rather than let it raise.
            target = target.with_suffix(".keras")
        self.model.save(target)
        return target

    def load_native(self, path: os.PathLike | str) -> "KerasModel":
        """Reattach a previously saved graph."""
        keras = _keras()
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"no saved Keras model at {source}")
        self.model = keras.models.load_model(source, compile=False)
        # Recompile so the loaded model can be evaluated or fine-tuned. compile=False on
        # load avoids version-skew errors in the optimizer state, which is the usual
        # cause of "could not deserialise" on a model trained in Colab and loaded
        # locally.
        self._compile(self.model, keras)
        return self

    def without_native(self) -> "KerasModel":
        """A picklable copy with the Keras graph removed.

        :meth:`shieldnet.persist.ModelBundle.save` calls this so joblib never touches a
        Keras object. The copy is shallow apart from blanking ``model``, so the
        hyper-parameters and class mapping needed to rebuild are all preserved.
        """
        clone = copy.copy(self)
        clone.model = None
        return clone

    def feature_importance(self) -> Optional[np.ndarray]:
        """Neural nets have no native importance; SHAP handles these models instead."""
        return None


# ---------------------------------------------------------------------------
# architectures
# ---------------------------------------------------------------------------

class CNN1DModel(KerasModel):
    """Two convolution blocks, global pooling, dense head."""

    key, label = "cnn1d", "1-D Convolutional Network"

    def _architecture(self, keras, n_features: int, n_out: int):
        L = keras.layers
        # kernel_size 3 with padding="same" keeps the length stable so a 25-feature
        # input survives two blocks; kernel 5 on 25 features with valid padding would
        # shrink to nothing after pooling.
        return keras.Sequential([
            L.Input(shape=(n_features, 1)),
            L.Conv1D(64, 3, padding="same", activation="relu"),
            L.BatchNormalization(),
            L.MaxPooling1D(2, padding="same"),
            L.Conv1D(128, 3, padding="same", activation="relu"),
            L.BatchNormalization(),
            # Global pooling rather than Flatten: Flatten ties the parameter count to the
            # feature count, so a 25-feature and a 40-feature run would not be
            # architecturally comparable in the ablation study.
            L.GlobalMaxPooling1D(),
            L.Dropout(0.3),
            L.Dense(128, activation="relu"),
            L.Dropout(0.3),
            L.Dense(n_out, activation="softmax"),
        ], name="shieldnet_cnn1d")


class BiLSTMModel(KerasModel):
    """Bidirectional LSTM over the feature axis."""

    key, label = "bilstm", "Bidirectional LSTM"

    def _architecture(self, keras, n_features: int, n_out: int):
        L = keras.layers
        return keras.Sequential([
            L.Input(shape=(n_features, 1)),
            L.Bidirectional(L.LSTM(64, return_sequences=True)),
            L.Dropout(0.3),
            L.Bidirectional(L.LSTM(32)),
            L.Dropout(0.3),
            L.Dense(64, activation="relu"),
            L.Dense(n_out, activation="softmax"),
        ], name="shieldnet_bilstm")


class CNNBiLSTMModel(KerasModel):
    """Convolutional front end, recurrent back end - the report's headline deep model.

    The convolutions compress 25 features into a shorter sequence of learned filter
    responses, and the BiLSTM then reads that compressed sequence in both directions.
    This is the arrangement most of the CICIDS2017 deep-learning literature uses, which
    is the main reason it is the model reproduced here: it makes the results comparable
    to published numbers.
    """

    key, label = "cnn_bilstm", "CNN + Bidirectional LSTM"

    def _architecture(self, keras, n_features: int, n_out: int):
        L = keras.layers
        return keras.Sequential([
            L.Input(shape=(n_features, 1)),
            L.Conv1D(64, 3, padding="same", activation="relu"),
            L.BatchNormalization(),
            L.MaxPooling1D(2, padding="same"),
            L.Conv1D(128, 3, padding="same", activation="relu"),
            L.BatchNormalization(),
            L.Bidirectional(L.LSTM(64, return_sequences=False)),
            L.Dropout(0.4),
            L.Dense(128, activation="relu"),
            L.Dropout(0.3),
            L.Dense(n_out, activation="softmax"),
        ], name="shieldnet_cnn_bilstm")

    @staticmethod
    def search_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
        }
