"""Typed configuration for the whole pipeline, with optional YAML overrides.

Every tunable lives here so that a run is reproducible from one file. Load order is
built-in defaults -> YAML file -> explicit keyword overrides, deep-merged, so a YAML
file only needs to mention what it changes.

    from shieldnet.config import Config
    cfg = Config.load("config/default.yaml")
    cfg = Config.load(None, seed=7)            # defaults with one override
"""

from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

__all__ = ["Paths", "DataConfig", "FeatureConfig", "BalanceConfig", "TuneConfig",
           "TrainConfig", "Config", "seed_everything", "PROJECT_ROOT"]


def _find_project_root() -> Path:
    """Walk up from this file until a directory containing ``pyproject.toml`` is found.

    Falls back to three levels up (``src/shieldnet/config.py`` -> project root), which
    is correct for an editable install and for running straight from a clone.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[2]


PROJECT_ROOT = Path(os.environ.get("SHIELDNET_ROOT", _find_project_root()))


@dataclass
class Paths:
    """Filesystem layout. Relative paths resolve against :data:`PROJECT_ROOT`."""

    root: str = "."
    raw: str = "data/raw"
    interim: str = "data/interim"
    processed: str = "data/processed"
    samples: str = "data/samples"
    artifacts: str = "artifacts"
    figures: str = "reports/figures"
    reports: str = "reports"

    def resolve(self, key: str) -> Path:
        """Absolute path for one attribute, creating nothing."""
        value = Path(getattr(self, key))
        if value.is_absolute():
            return value
        base = PROJECT_ROOT if self.root == "." else Path(self.root)
        return (base / value).resolve()

    def ensure(self, *keys: str) -> None:
        """``mkdir -p`` the named directories (all of them when *keys* is empty)."""
        for key in keys or tuple(f.name for f in fields(self) if f.name != "root"):
            self.resolve(key).mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Dataset acquisition, cleaning and working-chunk construction."""

    #: Kaggle dataset slug used by ``shieldnet download``.
    kaggle_dataset: str = "chethuhn/network-intrusion-dataset"
    #: Fallback slug if the primary one is unavailable to your account.
    kaggle_dataset_alt: str = "cicdataset/cicids2017"

    #: Per-class row caps for the stratified working chunk. Classes absent from this
    #: mapping are kept in full - that is the whole point of the design, so every
    #: rare-attack row survives sampling.
    caps: Dict[str, int] = field(default_factory=lambda: {
        "BENIGN": 150_000,
        "DoS Hulk": 40_000,
        "PortScan": 40_000,
        "DDoS": 35_000,
    })
    #: Merge Heartbleed + Infiltration + SQL Injection into one "Rare Attacks" class.
    merge_rare: bool = True
    #: Drop exact-duplicate flow rows during consolidation.
    drop_duplicates: bool = True
    #: Drop features whose variance is zero on the training split.
    drop_constant: bool = True
    #: How to fill NaN/inf. ``median`` is fast and robust; ``knn`` is slower but keeps
    #: local structure and is what the report compares against.
    imputer: str = "median"
    knn_neighbours: int = 5
    #: ``standard`` | ``minmax`` | ``robust`` | ``none``. Irrelevant for trees, essential
    #: for logistic regression, the MLP and the 1-D convolutional models.
    scaler: str = "standard"
    #: Winsorise each feature at this quantile (and its mirror) using *training* values
    #: only. CICIDS2017 contains Flow Bytes/s values around 1e9 next to a median near
    #: 1e3; without this, min-max scaling squashes almost every row to ~0 and the
    #: distance-based and gradient-based models never recover. ``None`` disables it.
    clip_quantile: Optional[float] = 0.9995
    #: Encoding used to read the raw CSVs. See :mod:`shieldnet.schema` for why this
    #: must not be utf-8.
    encoding: str = "latin-1"
    #: Rows per chunk when streaming the raw files. At 250,000 a chunk is about 170 MB,
    #: and the two-pass streaming load holds no more than a couple of those at a time.
    #: This does *not* bound the non-streaming path: ``load_raw`` keeps every chunk until
    #: the concat, so its peak is a little over twice the finished frame - roughly 2 GB
    #: for the full dataset - at any chunk size.
    read_chunk_rows: int = 250_000
    #: Held-out fractions, both measured against the whole frame rather than against what
    #: the previous split left, so these two are 20/10/70 and not 20/8/72. ``split_frame``
    #: allocates per class, so the fractions hold within every class as well as overall.
    test_size: float = 0.20
    val_size: float = 0.10
    #: Any class with fewer than this many rows is reported as unstratifiable.
    min_class_rows: int = 10


@dataclass
class FeatureConfig:
    """Feature selection."""

    #: Number of features to keep in the final subset.
    n_features: int = 25
    #: Rankers to run. Their ranks are combined by mean reciprocal rank.
    methods: List[str] = field(default_factory=lambda: ["mutual_info", "chi2", "rfe"])
    #: Rows subsampled for the expensive rankers (MI and RFE are O(n) and O(n*d)).
    ranking_sample_rows: int = 60_000
    #: Estimator used inside RFE.
    rfe_estimator: str = "random_forest"
    rfe_step: float = 0.1
    #: Drop one of any feature pair correlated above this before ranking.
    correlation_threshold: float = 0.95
    #: Re-run selection on this many bootstrap resamples and report how often each
    #: feature survives. 0 disables. A feature chosen in 5/5 resamples is a finding; one
    #: chosen in 2/5 is noise, and saying which is which is most of the value.
    stability_runs: int = 0
    #: Feature-count grid for the ablation study.
    ablation_sizes: List[int] = field(default_factory=lambda: [10, 15, 20, 25, 30, 40])


@dataclass
class BalanceConfig:
    """Class-imbalance handling. Applied to the training split only, never to
    validation or test - resampling those would invent a score that cannot be
    reproduced on real traffic."""

    #: ``smote`` | ``smote_tomek`` | ``class_weight`` | ``none``
    strategy: str = "smote"
    #: Cap on how far a minority class is oversampled, as a fraction of the majority
    #: class. 0.25 means "bring every class up to at most 25% of BENIGN", which stops
    #: the balancer from trying to make the rarest class the size of the largest one.
    max_ratio: float = 0.25
    #: SMOTE's k. Reduced automatically when the smallest class has fewer than k+1
    #: rows, which is otherwise a hard crash.
    k_neighbours: int = 5
    #: Hard ceiling on how many times a class may be multiplied, regardless of
    #: *max_ratio*. Bringing the 47 Rare Attacks rows in the train split up to 25% of a
    #: 105,000-row majority means inventing 26,203 rows from 47 - a 559x expansion that
    #: interpolates noise and teaches the model the shape of the SMOTE line segments
    #: rather than the attack. 20x is aggressive but still bounded.
    max_expansion: float = 20.0
    #: Also pass class weights to estimators that support it.
    use_class_weight: bool = True


@dataclass
class TuneConfig:
    """Optuna hyper-parameter search."""

    enabled: bool = True
    n_trials: int = 40
    timeout_seconds: Optional[int] = 1800
    cv_folds: int = 3
    #: Optuna's objective. Must be a key of :data:`shieldnet.tune.METRICS`.
    #:
    #: Deliberately *not* the same as ``TrainConfig.selection_metric``. Tuning needs a
    #: smooth, low-variance signal: log loss is a proper scoring rule that moves for
    #: every row, so a trial that improves the model slightly gets credit for it. Macro
    #: F1 over 13 classes is a step function of the arg-max, and with roughly a dozen
    #: rows of the rarest class inside a CV fold, a single row flipping moves it by
    #: about 0.02 - noise that Optuna's sampler would happily chase for 40 trials.
    metric: str = "log_loss"
    pruner: str = "median"
    study_name: str = "shieldnet"
    #: Written next to the artifacts so a study can be resumed after a disconnect,
    #: which matters on Colab.
    storage: Optional[str] = "sqlite:///artifacts/optuna.db"


@dataclass
class TrainConfig:
    """Which models to train and how the winner is picked."""

    models: List[str] = field(default_factory=lambda: [
        "logistic_regression", "random_forest", "xgboost", "lightgbm", "cnn_bilstm",
    ])
    #: Model shipped when the leaderboard is overridden with ``--select primary``.
    #: With the default ``--select auto`` this is only the tie-break, so setting it is a
    #: statement of intent rather than a decision.
    primary: str = "xgboost"
    #: Metric used to choose the deployed model, evaluated on the **validation** split.
    #:
    #: Macro F1, not accuracy and not log loss. Accuracy is unusable here - predicting
    #: BENIGN for every row scores 0.803 on full CICIDS2017 while detecting nothing.
    #: Log loss is the tuning objective because it is smooth, but it rewards calibration
    #: on the majority class, so a model that is beautifully calibrated about benign
    #: traffic and blind to Bot can win on log loss. Macro F1 weights all 13 classes
    #: equally, which is the actual deployment requirement.
    selection_metric: str = "macro_f1"
    #: Deep-learning knobs.
    epochs: int = 40
    batch_size: int = 512
    early_stopping_patience: int = 6
    learning_rate: float = 1e-3
    #: Rows sampled for the SHAP background distribution.
    shap_background_rows: int = 200
    #: Rows explained for the global SHAP summary.
    shap_explain_rows: int = 2_000


@dataclass
class Config:
    """Top-level config object threaded through every stage."""

    seed: int = 42
    n_jobs: int = -1
    verbose: bool = True
    paths: Paths = field(default_factory=Paths)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    balance: BalanceConfig = field(default_factory=BalanceConfig)
    tune: TuneConfig = field(default_factory=TuneConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # -- construction --------------------------------------------------------

    @classmethod
    def load(cls, path: Optional[os.PathLike | str] = None, **overrides: Any) -> "Config":
        """Build a config from defaults, an optional YAML/JSON file, and kwargs."""
        data: Dict[str, Any] = {}
        if path is not None:
            data = _read_mapping(Path(path))
        if overrides:
            data = _deep_merge(data, overrides)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Config":
        """Recursively build from a (possibly partial) nested mapping."""
        return _build(cls, data)

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: os.PathLike | str) -> Path:
        """Write the fully-resolved config beside the artifacts for provenance."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), "utf-8")
        return target

    # -- convenience ---------------------------------------------------------

    def resolved_paths(self) -> Dict[str, Path]:
        return {f.name: self.paths.resolve(f.name)
                for f in fields(self.paths) if f.name != "root"}

    def describe(self) -> str:
        """Single-paragraph summary used in log headers."""
        return (
            f"seed={self.seed} chunk_target~{sum(self.data.caps.values()):,}+minority "
            f"features={self.features.n_features} balance={self.balance.strategy} "
            f"tuning={'on' if self.tune.enabled else 'off'}"
            f"({self.tune.n_trials} trials) models={','.join(self.train.models)}"
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_mapping(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    text = path.read_text("utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - yaml is in requirements
            raise ImportError(
                "PyYAML is needed to read a .yaml config. Either `pip install pyyaml` "
                "or supply the same settings as JSON."
            ) from exc
        loaded = yaml.safe_load(text) or {}
    else:
        loaded = json.loads(text or "{}")
    if not isinstance(loaded, dict):
        raise TypeError(f"{path} must contain a mapping at the top level")
    return loaded


def _deep_merge(base: Mapping[str, Any], other: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge; *other* wins. Lists are replaced, not concatenated."""
    out: Dict[str, Any] = dict(copy.deepcopy(dict(base)))
    for key, value in other.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _build(target_cls: type, data: Mapping[str, Any]) -> Any:
    """Instantiate a (possibly nested) dataclass from a partial mapping.

    Unknown keys raise, because a silently-ignored typo in a config file is a very
    expensive way to lose an afternoon of training.
    """
    known = {f.name: f for f in fields(target_cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise KeyError(
            f"unknown config key(s) for {target_cls.__name__}: {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}"
        )
    kwargs: Dict[str, Any] = {}
    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        if is_dataclass(f.type) and isinstance(value, Mapping):
            kwargs[name] = _build(f.type, value)
        elif isinstance(f.type, str) and f.type in _NESTED and isinstance(value, Mapping):
            # `from __future__ import annotations` turns annotations into strings, so
            # is_dataclass(f.type) is False here and we resolve by name instead.
            kwargs[name] = _build(_NESTED[f.type], value)
        else:
            kwargs[name] = value
    return target_cls(**kwargs)


_NESTED: Dict[str, type] = {
    "Paths": Paths,
    "DataConfig": DataConfig,
    "FeatureConfig": FeatureConfig,
    "BalanceConfig": BalanceConfig,
    "TuneConfig": TuneConfig,
    "TrainConfig": TrainConfig,
}


def seed_everything(seed: int = 42) -> int:
    """Seed every RNG we might touch and return the seed.

    Also sets ``PYTHONHASHSEED`` and TensorFlow's op-level determinism when TF is
    importable. Full bit-for-bit reproducibility on GPU is still not guaranteed -
    cuDNN picks non-deterministic kernels for some convolutions - so the deep-learning
    numbers move by a few thousandths between runs even with this in place.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:  # pragma: no cover - exercised only when TF is installed
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass
    try:  # pragma: no cover
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass
    return seed
