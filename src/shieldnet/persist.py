"""Artifact bundle: everything the web app needs to reproduce a prediction.

A trained model on its own is useless for inference. To turn a raw CICFlowMeter row
into a prediction you also need the exact feature list and its order, the fitted
scaler, the per-feature training medians used to fill absent columns, and the label
encoder's class order. Ship them together or the app will silently mispredict - a
scaler fitted on 77 features applied to a 25-feature frame does not raise, it just
produces nonsense.

Keras models are the one thing that cannot go in the pickle: a compiled Keras model is
not reliably picklable across versions. They are written alongside in the native
``.keras`` format and re-attached on load.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["ARTIFACT_VERSION", "DEPENDENCIES", "ModelBundle", "dependency_report",
           "dump", "load", "environment_report", "file_digest", "BundleError"]

ARTIFACT_VERSION = "1.0.0"

BUNDLE_FILE = "bundle.joblib"
MANIFEST_FILE = "manifest.json"
DEEP_MODEL_FILE = "deep_model.keras"


class BundleError(RuntimeError):
    """Raised when an artifact bundle is missing, stale or internally inconsistent."""


# ---------------------------------------------------------------------------
# joblib with a pickle fallback
# ---------------------------------------------------------------------------

def dump(obj: Any, path: os.PathLike | str) -> Path:
    """Serialise *obj*. Uses joblib when available (far faster on big numpy arrays)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        import joblib
        joblib.dump(obj, target, compress=3)
    except ImportError:
        import pickle
        with open(target, "wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return target


def load(path: os.PathLike | str) -> Any:
    """Inverse of :func:`dump`, tolerating a file written by either backend."""
    source = Path(path)
    if not source.exists():
        raise BundleError(f"artifact not found: {source}")
    try:
        import joblib
        return joblib.load(source)
    except ImportError:
        import pickle
        with open(source, "rb") as fh:
            return pickle.load(fh)


def file_digest(path: os.PathLike | str, length: int = 12) -> str:
    """Short sha256 of a file, for the manifest."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:length]


def environment_report() -> Dict[str, str]:
    """Versions of everything that can break a *reload*, captured at save time.

    Deliberately narrower than :func:`dependency_report`. This goes into the manifest,
    and what belongs there is the set of libraries whose version could make
    ``joblib.load`` produce a different object than the one that was saved. Streamlit
    cannot; scikit-learn can.
    """
    report: Dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "shieldnet_artifact_version": ARTIFACT_VERSION,
    }
    for module in ("numpy", "pandas", "sklearn", "xgboost", "lightgbm", "shap",
                   "tensorflow", "keras", "imblearn", "optuna", "joblib"):
        try:
            mod = __import__(module)
            report[module] = getattr(mod, "__version__", "unknown")
        except Exception:
            report[module] = "absent"
    return report


#: ``(import name, pip name, what its absence costs you)``, in the order doctor prints.
#:
#: Every optional dependency in ``pyproject.toml`` appears here, and that is the point:
#: `doctor`'s job is to answer "will the next command work", and a library it does not
#: probe is a command that fails later with a traceback instead of now with a sentence.
#: `pyarrow` earns its place by failing *quietly* - without it the parquet cache silently
#: becomes CSV and every re-read costs minutes that look like slowness rather than a
#: missing package.
DEPENDENCIES: List[tuple] = [
    ("numpy",         "numpy",            "required - nothing runs"),
    ("pandas",        "pandas",           "required - nothing runs"),
    ("sklearn",       "scikit-learn",     "7 of 11 models, RFE, the metrics"),
    ("scipy",         "scipy",            "chi2 feature ranking"),
    ("imblearn",      "imbalanced-learn", "SMOTE via imblearn (numpy fallback exists)"),
    ("xgboost",       "xgboost",          "the xgboost model"),
    ("lightgbm",      "lightgbm",         "the lightgbm model"),
    ("tensorflow",    "tensorflow",       "cnn1d, bilstm, cnn_bilstm"),
    ("shap",          "shap",             "TreeSHAP/KernelSHAP (occlusion fallback exists)"),
    ("optuna",        "optuna",           "hyper-parameter tuning; --no-tune without it"),
    ("joblib",        "joblib",           "fast bundles (pickle fallback exists)"),
    ("yaml",          "pyyaml",           "--config; built-in defaults only without it"),
    ("pyarrow",       "pyarrow",          "the parquet cache - falls back to slow CSV"),
    ("matplotlib",    "matplotlib",       "the report figures"),
    ("seaborn",       "seaborn",          "the heatmaps in the report figures"),
    ("streamlit",     "streamlit",        "shieldnet serve"),
    ("plotly",        "plotly",           "the app's per-class confidence chart"),
    ("kaggle",        "kaggle",           "shieldnet download"),
]


def dependency_report() -> List[Dict[str, str]]:
    """Every dependency, its version or ``"absent"``, and what its absence costs.

    Used by ``shieldnet doctor``. Importing is the only honest test - a package can be
    on disk, on ``sys.path``, and still raise on import because a binary wheel does not
    match the interpreter, which is exactly the TensorFlow failure worth catching before
    a forty-minute run rather than during one.
    """
    rows: List[Dict[str, str]] = []
    for module, pip_name, unlocks in DEPENDENCIES:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "installed")
        except Exception as exc:                                # noqa: BLE001
            version = "absent" if isinstance(exc, ImportError) else "BROKEN"
        rows.append({"module": module, "pip": pip_name,
                     "version": version, "unlocks": unlocks})
    return rows


# ---------------------------------------------------------------------------
# The bundle
# ---------------------------------------------------------------------------

@dataclass
class ModelBundle:
    """Self-contained inference artifact.

    Attributes
    ----------
    model_name:
        Registry key, e.g. ``"xgboost"``. Used to rebuild the wrapper on load.
    feature_names:
        The selected features **in training order**. Inference must reindex to this
        exact order; column order matters to every tree model and to the scaler.
    label_classes:
        Class names indexed by the integer the model predicts, i.e.
        ``label_classes[3]`` is the name of class 3.
    scaler:
        Fitted transformer over ``feature_names``, or ``None`` if the chosen model
        needs no scaling.
    medians:
        Training-split median per *canonical* feature name, pre-scaling. Covers all 77
        features, not just the selected ones, so the manual-entry form can fill the
        gaps. Also the fallback for a column absent from an uploaded CSV.
    """

    model_name: str
    feature_names: List[str]
    label_classes: List[str]
    scaler: Any = None
    medians: Dict[str, float] = field(default_factory=dict)
    model: Any = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- integrity -----------------------------------------------------------

    def validate(self) -> "ModelBundle":
        """Fail loudly on the inconsistencies that cause silent mispredictions."""
        if not self.feature_names:
            raise BundleError("bundle has no feature_names")
        if len(set(self.feature_names)) != len(self.feature_names):
            dupes = sorted({f for f in self.feature_names
                            if self.feature_names.count(f) > 1})
            raise BundleError(f"feature_names contains duplicates: {dupes}")
        if not self.label_classes:
            raise BundleError("bundle has no label_classes")
        if len(set(self.label_classes)) != len(self.label_classes):
            raise BundleError("label_classes contains duplicates")

        n = len(self.feature_names)
        expected = getattr(self.scaler, "n_features_in_", None)
        if expected is not None and int(expected) != n:
            raise BundleError(
                f"scaler was fitted on {int(expected)} features but the bundle lists "
                f"{n}. The scaler must be fitted on the *selected* feature subset, in "
                "the same order, or every prediction will be wrong without erroring."
            )
        missing_medians = [f for f in self.feature_names if f not in self.medians]
        if missing_medians:
            raise BundleError(
                f"medians missing for {len(missing_medians)} selected feature(s), "
                f"e.g. {missing_medians[:4]}. Inference cannot fill an absent column."
            )
        return self

    @property
    def n_classes(self) -> int:
        return len(self.label_classes)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def is_deep(self) -> bool:
        return bool(getattr(self.model, "is_deep", False)) or bool(
            self.metadata.get("is_deep", False)
        )

    # -- persistence ---------------------------------------------------------

    def save(self, directory: os.PathLike | str) -> Path:
        """Write the bundle plus a human-readable manifest into *directory*."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self.validate()

        model = self.model
        is_deep = bool(getattr(model, "is_deep", False))
        self.metadata["is_deep"] = is_deep
        self.metadata.setdefault("environment", environment_report())

        try:
            if is_deep:
                # Hand the native model to the wrapper to save, then blank it so
                # joblib never tries to pickle a Keras graph.
                model.save_native(out / DEEP_MODEL_FILE)
                self.model = model.without_native()
                log.info("saved deep model natively -> %s", DEEP_MODEL_FILE)
            bundle_path = dump(self, out / BUNDLE_FILE)
        finally:
            self.model = model  # always restore, even if the dump raised

        manifest = {
            "artifact_version": ARTIFACT_VERSION,
            "model_name": self.model_name,
            "is_deep": is_deep,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "feature_names": self.feature_names,
            "label_classes": self.label_classes,
            "metrics": _jsonable(self.metrics),
            "metadata": _jsonable(self.metadata),
            "bundle_sha256_12": file_digest(bundle_path),
        }
        (out / MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8"
        )
        log.info("bundle saved -> %s (%s, %d features, %d classes)",
                 bundle_path, self.model_name, self.n_features, self.n_classes)
        return bundle_path

    @classmethod
    def restore(cls, directory: os.PathLike | str) -> "ModelBundle":
        """Load a bundle written by :meth:`save`, re-attaching a deep model."""
        src = Path(directory)
        bundle_path = src / BUNDLE_FILE if src.is_dir() else src
        if not bundle_path.exists():
            raise BundleError(
                f"no trained model found at {bundle_path}. Run `shieldnet train` "
                "first, or point --artifacts at a directory containing "
                f"{BUNDLE_FILE}."
            )
        obj = load(bundle_path)
        if not isinstance(obj, cls):
            raise BundleError(
                f"{bundle_path} does not contain a ModelBundle (got "
                f"{type(obj).__name__}). It may have been written by a different "
                "version of ShieldNet."
            )
        if obj.metadata.get("is_deep"):
            native = bundle_path.parent / DEEP_MODEL_FILE
            if not native.exists():
                raise BundleError(
                    f"bundle says the model is a deep network but {DEEP_MODEL_FILE} is "
                    "missing next to it. Copy the whole artifacts directory, not just "
                    "the .joblib."
                )
            from .models.registry import build  # local import: keeps deps lazy
            wrapper = build(obj.model_name, n_classes=obj.n_classes,
                            n_features=obj.n_features)
            wrapper.load_native(native)
            obj.model = wrapper
        return obj.validate()


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of numpy scalars/arrays so json.dumps cannot fail."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item") and getattr(value, "shape", ()) == ():
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
