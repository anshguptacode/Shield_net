"""End-to-end training orchestration: raw CSVs in, deployable bundle out.

Every stage in this file exists somewhere else as a well-tested function. What this
module owns is the *order* they run in, and the order is where leakage lives. The
sequence below is not arbitrary and each boundary is load-bearing:

1. **Load and cap.** Stream the eight day-wise CSVs, tallying labels in one pass and
   Bernoulli-sampling in a second, so peak memory is a function of the chunk size and
   not of the 2.83 M-row dataset. Classes absent from ``data.caps`` are kept whole -
   that is deliberate, because the rare attacks are the entire point.
2. **Clean.** Replace infinities, drop exact-duplicate flows and empty rows. No
   statistic is fitted here, so it is safe to do before the split.
3. **Split.** Test carved out first, then validation from the remainder, both
   stratified. Everything after this line is fitted on train only.
4. **Fit preprocessing on train.** Clip at training quantiles, impute with training
   medians, scale with training statistics.
5. **Select features on train.** Selecting on the full dataset is the classic quiet
   leak: it moves no single number much, it just makes all of them permanently
   optimistic.
6. **Refit preprocessing on the selected columns only.** See
   :func:`build_feature_space` for why this is exact rather than an approximation.
7. **Balance train only.** Resampling validation or test would be measuring a
   distribution that does not exist in production.
8. **Tune on unbalanced train.** SMOTE inside cross-validation leaks: a synthetic row
   interpolated between two neighbours that end up in different folds puts information
   from the held-out fold into the training fold. :mod:`shieldnet.tune` therefore
   re-weights classes instead of resampling them, and is handed the unbalanced split.
9. **Fit, evaluate, explain, persist.**

The two splits do different jobs and mixing them up is the other easy mistake. The
**validation** split picks the winning model and its hyper-parameters. The **test**
split is touched exactly once per model, to produce the number that goes in the report.
Choosing the model by its test score would make that number a best-of-N maximum rather
than an estimate.

Nothing here raises when an optional dependency is missing. A model whose library is
absent is recorded as skipped with the pip command that would fix it, and the run
continues with whatever is installed - a training run that dies at model four of five
after twenty minutes of data preparation is worse than useless.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field, fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import schema as sch
from .balance import BalanceReport, resample
from .config import Config, seed_everything
from .data.chunk import ChunkReport, build_working_chunk, build_working_chunk_streaming
from .data.load import LoadReport, discover_raw_files, load_raw
from .data.synthetic import synthesise
from .evaluate import EvaluationReport, compare, evaluate
from .explain import Explainer, GlobalExplanation, LocalExplanation
from .features import SelectionReport, select_features
from .logging_utils import get_logger, human_count, human_duration, stage
from .models import available, build, is_deep, resolve
from .narrate import narrate_evaluation
from .persist import ModelBundle, dump, environment_report, load as load_pickle
from .preprocess import (CleanReport, LabelCodec, Preprocessor, SplitData, class_weights,
                         clean_frame, split_frame)
from .tune import METRICS, TuneReport, objective_value, tune_model

log = get_logger(__name__)

__all__ = ["DataBundle", "FeatureSpace", "TrainedModel", "TrainingRun",
           "prepare_data", "build_feature_space", "train_one", "train",
           "plot_confusion", "PREPARED_PREFIX"]

PREPARED_PREFIX = "prepared"


# ---------------------------------------------------------------------------
# stage 1-3: data
# ---------------------------------------------------------------------------

@dataclass
class DataBundle:
    """The split data plus every report produced on the way to it."""

    split: SplitData
    clean: CleanReport
    source: str
    chunk: Optional[ChunkReport] = None
    load: Optional[LoadReport] = None
    seconds: float = 0.0
    cache_key: str = ""

    @property
    def synthetic(self) -> bool:
        """True when the generator produced this data instead of CICIDS2017.

        Worth a first-class property rather than a string test at each call site: every
        number a synthetic run produces describes the generator, and the one way that
        becomes a real problem is a figure from a smoke test quietly appearing in the
        report. Anything that writes an artifact or a metric consults this.
        """
        return self.source.startswith("synthetic")

    @property
    def codec(self) -> LabelCodec:
        return self.split.codec

    @property
    def class_names(self) -> List[str]:
        return list(self.split.codec.classes)

    @property
    def feature_names(self) -> List[str]:
        return [str(c) for c in self.split.X_train.columns]

    def render(self) -> str:
        lines = [f"Source: {self.source}"]
        if self.load is not None:
            lines.append(self.load.render())
        if self.chunk is not None:
            lines.append(self.chunk.render())
        lines.append(self.clean.render())
        lines.append(self.split.render())
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "source": self.source,
            "seconds": round(self.seconds, 2),
            "cache_key": self.cache_key,
            "clean": self.clean.to_dict(),
            "sizes": self.split.sizes(),
            "class_names": self.class_names,
            "n_features": len(self.feature_names),
            "warnings": list(self.split.warnings),
        }
        if self.load is not None:
            out["load"] = self.load.to_dict()
        if self.chunk is not None:
            out["chunk"] = self.chunk.to_dict()
        return out


def prepare_data(
    cfg: Config,
    *,
    source: Optional[os.PathLike | str] = None,
    synthetic_rows: int = 0,
    streaming: Optional[bool] = None,
    cache: bool = True,
    quiet: bool = False,
) -> DataBundle:
    """Load, cap, clean and split. The expensive half of a run.

    Parameters
    ----------
    source:
        Directory of raw CICFlowMeter CSVs. Defaults to ``paths.raw``.
    synthetic_rows:
        Force the synthetic generator instead of reading files. Used by the smoke test
        and by anyone who wants to see the pipeline run before the ~1 GB download
        finishes.
    streaming:
        Two-pass streaming load. Defaults to on whenever real files are present,
        because the alternative needs about 1.7 GB of resident frame for the full
        dataset and free Colab instances do not survive it.
    cache:
        Reuse (and write) a prepared split under ``paths.processed``, keyed by a digest
        of every config field that can change the result. Colab runtimes disconnect;
        re-running should not mean re-reading 2.8 M rows.
    """
    started = time.perf_counter()
    key = _cache_key(cfg, source=source, synthetic_rows=synthetic_rows)
    cache_path = cfg.paths.resolve("processed") / f"{PREPARED_PREFIX}_{key}.joblib"

    if cache and cache_path.exists():
        try:
            cached = load_pickle(cache_path)
        except Exception as exc:                      # a stale or half-written cache
            log.warning("ignoring unreadable prepared-data cache %s (%s)",
                        cache_path.name, exc)
        else:
            if isinstance(cached, DataBundle) and cached.cache_key == key:
                log.info("reusing prepared split from %s (%s rows train)",
                         cache_path.name, human_count(len(cached.split.y_train)))
                return cached
            log.warning("prepared-data cache %s does not match the current config; "
                        "rebuilding", cache_path.name)

    with stage(log, "data preparation", quiet=quiet) as st:
        load_report: Optional[LoadReport] = None
        chunk_report: Optional[ChunkReport] = None

        if synthetic_rows > 0:
            frame = synthesise(n_rows=synthetic_rows, seed=cfg.seed, inject_defects=True)
            origin = f"synthetic ({synthetic_rows:,} rows, seed {cfg.seed})"
            log.warning("training on SYNTHETIC data. Numbers from this run describe the "
                        "generator, not CICIDS2017, and must not go in the report.")
            frame, chunk_report = build_working_chunk(
                frame, caps=cfg.data.caps, seed=cfg.seed,
                min_class_rows=cfg.data.min_class_rows)
        else:
            raw_dir = Path(source) if source is not None else cfg.paths.resolve("raw")
            files = discover_raw_files(raw_dir)
            if not files:
                raise FileNotFoundError(
                    f"no CICIDS2017 CSVs found in {raw_dir}. Either run "
                    "`shieldnet download` (needs a Kaggle token at ~/.kaggle/kaggle.json), "
                    "put the eight *.pcap_ISCX.csv files there yourself, or pass "
                    "`--synthetic 40000` to exercise the pipeline without the data."
                )
            origin = f"{len(files)} file(s) in {raw_dir}"
            use_stream = len(files) > 1 if streaming is None else bool(streaming)
            if use_stream:
                frame, chunk_report, load_report = build_working_chunk_streaming(
                    files, caps=cfg.data.caps, seed=cfg.seed,
                    encoding=cfg.data.encoding, chunk_rows=cfg.data.read_chunk_rows,
                    merge_rare=cfg.data.merge_rare,
                    min_class_rows=cfg.data.min_class_rows)
            else:
                frame, load_report = load_raw(
                    raw_dir, encoding=cfg.data.encoding,
                    merge_rare=cfg.data.merge_rare,
                    chunk_rows=cfg.data.read_chunk_rows, files=files)
                frame, chunk_report = build_working_chunk(
                    frame, caps=cfg.data.caps, seed=cfg.seed,
                    min_class_rows=cfg.data.min_class_rows)

        frame, clean = clean_frame(
            frame, drop_duplicates=cfg.data.drop_duplicates,
            label_column=sch.LABEL_COLUMN)
        if frame.empty:
            raise ValueError(
                "cleaning removed every row. That normally means the CSVs were read "
                "with the wrong encoding and the label column is full of NaN - check "
                f"data.encoding (currently {cfg.data.encoding!r})."
            )

        split = split_frame(
            frame, test_size=cfg.data.test_size, val_size=cfg.data.val_size,
            seed=cfg.seed, label_column=sch.LABEL_COLUMN)

        st["summary"] = (f"{human_count(len(frame))} clean rows, "
                         f"{len(split.codec.classes)} classes, "
                         f"{len(split.X_train.columns)} features")

    bundle = DataBundle(split=split, clean=clean, source=origin, chunk=chunk_report,
                        load=load_report, seconds=time.perf_counter() - started,
                        cache_key=key)
    for warning in split.warnings:
        log.warning("split: %s", warning)

    if cache:
        try:
            dump(bundle, cache_path)
            log.info("prepared split cached -> %s", cache_path.name)
        except Exception as exc:                      # never fail a run over a cache
            log.warning("could not cache the prepared split: %s", exc)
    return bundle


def _cache_key(cfg: Config, *, source: Any, synthetic_rows: int) -> str:
    """Digest of every setting that changes the prepared split, and nothing else.

    Deliberately excludes the model list, the tuner and the balancer: changing those
    should reuse the cache, and including them would make it useless.
    """
    payload = {
        "schema": sch.SCHEMA_VERSION,
        "seed": cfg.seed,
        "source": str(source) if source is not None else "<default>",
        "synthetic_rows": int(synthetic_rows),
        "caps": dict(sorted(cfg.data.caps.items())),
        "merge_rare": cfg.data.merge_rare,
        "drop_duplicates": cfg.data.drop_duplicates,
        "encoding": cfg.data.encoding,
        "test_size": cfg.data.test_size,
        "val_size": cfg.data.val_size,
        "min_class_rows": cfg.data.min_class_rows,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:10]


# ---------------------------------------------------------------------------
# stage 4-7: features
# ---------------------------------------------------------------------------

@dataclass
class FeatureSpace:
    """Fitted preprocessing, the selected subset, and the arrays models train on."""

    preprocessor: Preprocessor
    selection: SelectionReport
    selected: List[str]
    class_names: List[str]
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_fit: np.ndarray
    y_fit: np.ndarray
    balance: BalanceReport
    class_weight: Dict[int, float] = field(default_factory=dict)
    raw_medians: Dict[str, float] = field(default_factory=dict)
    dropped_constant: List[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def n_features(self) -> int:
        return len(self.selected)

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def render(self) -> str:
        lines = [self.selection.render(top=min(30, len(self.selection.ranking))),
                 self.balance.render()]
        if self.class_weight:
            worst = sorted(self.class_weight.items(), key=lambda kv: -kv[1])[:4]
            lines.append("Residual class weights (post-balance): " + ", ".join(
                f"{self.class_names[i]}={w:.1f}" for i, w in worst))
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_features": self.n_features,
            "selected": list(self.selected),
            "dropped_constant": list(self.dropped_constant),
            "preprocessor": self.preprocessor.to_dict(),
            "selection": self.selection.to_dict(),
            "balance": self.balance.to_dict(),
            "class_weight": {self.class_names[i]: round(float(w), 4)
                             for i, w in sorted(self.class_weight.items())},
            "rows": {"train": int(len(self.y_train)), "fit": int(len(self.y_fit)),
                     "val": int(len(self.y_val)), "test": int(len(self.y_test))},
            "seconds": round(self.seconds, 2),
        }


def build_feature_space(cfg: Config, data: DataBundle, *,
                        quiet: bool = False) -> FeatureSpace:
    """Fit preprocessing, select features, refit on the subset, balance the train split.

    The refit in the middle looks redundant and is not. The first preprocessor has to
    see all 77 columns, because feature selection needs comparable, scaled values to
    rank. But the artifact that ships to the app must be fitted on the 25 *selected*
    columns, in selection order: a scaler carrying 77 columns of statistics applied to a
    25-column frame does not raise, it silently subtracts the wrong mean from every
    feature.

    Refitting costs nothing in accuracy because every transform here is per-column -
    clipping quantiles, medians, and standardisation statistics are all computed one
    column at a time. Restricting the input to a subset therefore yields byte-identical
    statistics for the columns that remain; the second fit is a projection, not a new
    estimate. ``tests/check_train.py`` asserts exactly that, comparing the full
    preprocessor's per-column centres, spreads and medians against the refitted one and
    demanding agreement to within 1e-9.
    """
    started = time.perf_counter()
    X_train_raw = data.split.X_train
    y_train = np.asarray(data.split.y_train)

    with stage(log, "preprocessing (all features)", quiet=quiet) as st:
        full = Preprocessor.fit(
            X_train_raw,
            drop_constant=cfg.data.drop_constant,
            clip_quantile=cfg.data.clip_quantile,
            scaler=cfg.data.scaler,
            imputer=cfg.data.imputer,
            knn_neighbours=cfg.data.knn_neighbours,
        )
        scaled_train = full.frame(full.transform(X_train_raw))
        st["summary"] = (f"{len(full.feature_names_in_)} -> "
                         f"{full.n_features_out} features "
                         f"({len(full.dropped_constant)} constant)")

    selection = select_features(
        scaled_train, y_train,
        n_features=cfg.features.n_features,
        methods=cfg.features.methods,
        correlation_threshold=cfg.features.correlation_threshold,
        ranking_sample_rows=cfg.features.ranking_sample_rows,
        stability_runs=cfg.features.stability_runs,
        seed=cfg.seed,
        n_jobs=cfg.n_jobs,
        rfe_estimator=cfg.features.rfe_estimator,
        rfe_step=cfg.features.rfe_step,
    )
    selected = list(selection.selected)
    if not selected:
        raise ValueError("feature selection returned nothing; check features.n_features")

    with stage(log, "preprocessing (selected features)", quiet=quiet) as st:
        # drop_constant=False on purpose. A selected feature cannot be constant on the
        # training split: the rankers scored it on a stratified subsample *of* that
        # split, and every ranker here scores a constant column zero. Leaving the flag
        # on would risk the preprocessor silently emitting 24 columns for a 25-name
        # bundle, which validate() would catch but only after training.
        final = Preprocessor.fit(
            X_train_raw[selected],
            drop_constant=False,
            clip_quantile=cfg.data.clip_quantile,
            scaler=cfg.data.scaler,
            imputer=cfg.data.imputer,
            knn_neighbours=cfg.data.knn_neighbours,
        )
        if final.feature_names_out_ != selected:
            raise RuntimeError(
                "the refitted preprocessor does not emit the selected features in "
                f"order: {_first_difference(final.feature_names_out_, selected)}"
            )
        X_tr = final.transform(X_train_raw[selected])
        X_va = final.transform(data.split.X_val[selected])
        X_te = final.transform(data.split.X_test[selected])
        st["summary"] = f"{X_tr.shape[1]} features, {len(X_tr):,} train rows"

    y_val = np.asarray(data.split.y_val)
    y_test = np.asarray(data.split.y_test)

    X_fit, y_fit, balance = resample(
        X_tr, y_train,
        strategy=cfg.balance.strategy,
        max_ratio=cfg.balance.max_ratio,
        k_neighbours=cfg.balance.k_neighbours,
        max_expansion=cfg.balance.max_expansion,
        seed=cfg.seed,
        class_names=data.class_names,
    )

    # Class weights are computed on the *post-balance* distribution, not the original.
    # SMOTE is capped by max_ratio and max_expansion, so it closes part of the gap and
    # leaves the rest; weighting the original distribution on top of that would correct
    # the same imbalance twice and push the model into over-predicting rare attacks.
    weights: Dict[int, float] = {}
    if cfg.balance.use_class_weight:
        weights = class_weights(y_fit, n_classes=len(data.class_names),
                                scheme="balanced", cap=50.0)

    space = FeatureSpace(
        preprocessor=final, selection=selection, selected=selected,
        class_names=data.class_names,
        X_train=X_tr, y_train=y_train, X_val=X_va, y_val=y_val,
        X_test=X_te, y_test=y_test, X_fit=X_fit, y_fit=y_fit,
        balance=balance, class_weight=weights,
        raw_medians=dict(full.raw_medians),
        dropped_constant=list(full.dropped_constant),
        seconds=time.perf_counter() - started,
    )
    log.info("feature space ready: %d features, fit on %s rows (%s synthetic), "
             "val %s, test %s", space.n_features, human_count(len(y_fit)),
             human_count(balance.synthetic_rows), human_count(len(y_val)),
             human_count(len(y_test)))
    return space


def _first_difference(got: Sequence[str], want: Sequence[str]) -> str:
    for i, (a, b) in enumerate(zip(got, want)):
        if a != b:
            return f"position {i}: got {a!r}, expected {b!r}"
    if len(got) != len(want):
        return f"length {len(got)} != {len(want)}"
    return "no difference"


# ---------------------------------------------------------------------------
# stage 8-9: models
# ---------------------------------------------------------------------------

@dataclass
class TrainedModel:
    """One candidate: the fitted estimator and both of its evaluations."""

    name: str
    model: Any = None
    val: Optional[EvaluationReport] = None
    test: Optional[EvaluationReport] = None
    tune: Optional[TuneReport] = None
    params: Dict[str, Any] = field(default_factory=dict)
    fit_seconds: float = 0.0
    predict_seconds: float = 0.0
    rows_fit: int = 0
    failed: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed and self.val is not None

    def score(self, metric: str) -> float:
        """Validation score, sign-normalised so that larger is always better."""
        if self.val is None:
            return float("-inf")
        value = objective_value(self.val, metric)
        return -value if METRICS[metric] else value

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "name": self.name,
            "params": _jsonable(self.params),
            "fit_seconds": round(self.fit_seconds, 2),
            "predict_seconds": round(self.predict_seconds, 3),
            "rows_fit": self.rows_fit,
            "failed": self.failed,
            "notes": list(self.notes),
        }
        if self.val is not None:
            out["val"] = self.val.to_dict(include_confusion=False)
        if self.test is not None:
            out["test"] = self.test.to_dict(include_confusion=True)
        if self.tune is not None:
            out["tune"] = self.tune.to_dict()
        return out


def _score_on_test(record: TrainedModel, space: FeatureSpace, *,
                   quiet: bool = False) -> None:
    """Measure the selected model on the held-out test split. Called once per run.

    Separate from ``train_one`` so that the call is somewhere a reader can find it and
    count it. A failure here is logged rather than raised: the model is already chosen
    and already saveable, and losing a finished run because the final measurement threw
    would be the wrong trade.
    """
    if record.model is None or record.test is not None:
        return
    try:
        t0 = time.perf_counter()
        test_proba = record.model.predict_proba(space.X_test)
        secs = time.perf_counter() - t0
        record.test = evaluate(
            space.y_test, test_proba, class_names=space.class_names, model=record.name,
            split="test", fit_seconds=record.fit_seconds, predict_seconds=secs)
        if not quiet:
            log.info("test macro F1 %.4f, accuracy %.4f  (%d rows, scored once)",
                     record.test.macro_f1, record.test.accuracy, len(space.y_test))
    except Exception as exc:                                    # noqa: BLE001
        log.warning("test evaluation failed for %s: %s. The model is still selected and "
                    "saved; the artifact will simply carry no test metrics.",
                    record.name, exc)
        record.notes.append(f"test evaluation failed: {exc}")


def train_one(
    name: str,
    space: FeatureSpace,
    cfg: Config,
    *,
    tune: Optional[bool] = None,
    params: Optional[Dict[str, Any]] = None,
    evaluate_test: bool = True,
    quiet: bool = False,
) -> TrainedModel:
    """Tune (optionally), fit, and evaluate one model. Never raises for one model.

    A failure here is captured on the returned object rather than propagated: with five
    models queued behind twenty minutes of data preparation, one library that segfaults
    on this machine should cost one row of the leaderboard, not the run.
    """
    key = resolve(name)
    record = TrainedModel(name=key)
    do_tune = cfg.tune.enabled if tune is None else bool(tune)

    try:
        with stage(log, f"model: {key}", quiet=quiet) as st:
            chosen: Dict[str, Any] = dict(params or {})

            if do_tune and not params:
                report = tune_model(
                    key, space.X_train, space.y_train,
                    class_names=space.class_names,
                    metric=cfg.tune.metric,
                    n_trials=cfg.tune.n_trials,
                    cv_folds=cfg.tune.cv_folds,
                    timeout=cfg.tune.timeout_seconds,
                    balance_weights=cfg.balance.use_class_weight,
                    seed=cfg.seed,
                    n_jobs=cfg.n_jobs,
                    study_name=f"{cfg.tune.study_name}_{key}",
                    storage=_resolve_storage(cfg),
                    quiet=quiet,
                )
                record.tune = report
                chosen = dict(report.best_params)
                if report.stopped_early:
                    record.notes.append(f"tuning stopped early: {report.stopped_early}")
                log.info("%s: tuned %s = %.5f (baseline %.5f) over %d trial(s)",
                         key, report.metric, report.best_value,
                         report.baseline_value, len(report.trials))
            elif do_tune and params:
                record.notes.append("tuning skipped: explicit params were supplied")

            # Deep-model knobs come from TrainConfig, not from the search space, so a
            # tuned run still honours the configured epoch budget unless the tuner
            # actually proposed a value. Classical models must never see these keys -
            # scikit-learn rejects unknown constructor arguments.
            if is_deep(key):
                chosen.setdefault("epochs", cfg.train.epochs)
                chosen.setdefault("batch_size", cfg.train.batch_size)
                chosen.setdefault("early_stopping_patience",
                                  cfg.train.early_stopping_patience)
                chosen.setdefault("learning_rate", cfg.train.learning_rate)

            model = build(key, n_classes=space.n_classes, n_features=space.n_features,
                          seed=cfg.seed, n_jobs=cfg.n_jobs, params=chosen)
            record.params = dict(chosen)

            t0 = time.perf_counter()
            model.fit(space.X_fit, space.y_fit,
                      X_val=space.X_val, y_val=space.y_val,
                      class_weight=space.class_weight or None)
            record.fit_seconds = time.perf_counter() - t0
            record.rows_fit = int(len(space.y_fit))
            record.model = model

            t0 = time.perf_counter()
            val_proba = model.predict_proba(space.X_val)
            record.predict_seconds = time.perf_counter() - t0
            record.val = evaluate(
                space.y_val, val_proba, class_names=space.class_names, model=key,
                split="validation", fit_seconds=record.fit_seconds,
                predict_seconds=record.predict_seconds)

            if evaluate_test:
                t0 = time.perf_counter()
                test_proba = model.predict_proba(space.X_test)
                secs = time.perf_counter() - t0
                record.test = evaluate(
                    space.y_test, test_proba, class_names=space.class_names, model=key,
                    split="test", fit_seconds=record.fit_seconds,
                    predict_seconds=secs)

            if model.fit_history_:
                record.notes.append("; ".join(
                    f"{k}={v}" for k, v in model.fit_history_.items()
                    if not isinstance(v, (list, dict, np.ndarray))))

            st["summary"] = (f"val macro F1 {record.val.macro_f1:.4f}, "
                             f"log loss {record.val.log_loss:.4f}, "
                             f"fit {human_duration(record.fit_seconds)}")
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        record.failed = f"{type(exc).__name__}: {exc}"
        log.error("%s failed and was skipped: %s", key, record.failed)
        log.debug("traceback for %s", key, exc_info=True)
    return record


def _resolve_storage(cfg: Config) -> Optional[str]:
    """Make the Optuna sqlite URL absolute, through the same paths every output uses.

    ``sqlite:///artifacts/optuna.db`` is interpreted by SQLAlchemy against the *current
    working directory*, so the same config resumes a different study depending on where
    the command was run from - and on Colab, where the notebook cds around, that quietly
    wastes the forty minutes it was meant to protect.

    Resolving it here rather than by ``Path(...).resolve()`` matters for two reasons the
    docs promise. ``SHIELDNET_ROOT`` and ``--root`` relocate ``artifacts/`` wholesale, and
    the study has to move with it or a resumed run reads a study belonging to a different
    dataset. And ``paths.artifacts`` is overridable, so hard-coding the ``artifacts/`` in
    the URL would put the study somewhere the notebook's cleanup step does not look.

    So the leading segment is resolved through :class:`~shieldnet.config.Paths` when it
    names one of its keys, and anything else is placed in the artifacts directory:

    * ``sqlite:///artifacts/optuna.db`` -> ``<resolved artifacts>/optuna.db``
    * ``sqlite:///optuna.db``           -> ``<resolved artifacts>/optuna.db``
    * ``sqlite:////var/lib/study.db``   -> unchanged, absolute is absolute
    * ``postgresql://...``              -> unchanged, not ours to rewrite
    """
    url = cfg.tune.storage
    if not url or not url.startswith("sqlite:///"):
        return url
    tail = url[len("sqlite:///"):]
    if not tail:
        return None

    path = Path(tail)
    if not path.is_absolute():
        parts = path.parts
        keys = {f.name for f in dataclass_fields(cfg.paths)} - {"root"}
        if len(parts) > 1 and parts[0] in keys:
            path = cfg.paths.resolve(parts[0]).joinpath(*parts[1:])
        else:
            path = cfg.paths.resolve("artifacts") / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.as_posix()}"


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

@dataclass
class TrainingRun:
    """Everything a run produced, in one object that can be pickled or serialised."""

    config: Config
    data: DataBundle
    space: FeatureSpace
    models: List[TrainedModel]
    selection_metric: str
    best: Optional[TrainedModel] = None
    explanation: Optional[GlobalExplanation] = None
    samples: List[LocalExplanation] = field(default_factory=list)
    skipped: Dict[str, str] = field(default_factory=dict)
    bundle_path: Optional[Path] = None
    figures: Dict[str, str] = field(default_factory=dict)
    started_at: str = ""
    seconds: float = 0.0

    @property
    def succeeded(self) -> List[TrainedModel]:
        return [m for m in self.models if m.ok]

    def leaderboard(self) -> pd.DataFrame:
        """Validation-ordered table. Only the selected row carries test columns.

        The asymmetry is the point: candidates are compared on validation, and the test
        columns are blank for everything except the winner because nothing else was
        measured on test. A table with a test score in every row invites the reader to
        sort by it.
        """
        rows = []
        for m in sorted(self.succeeded, key=lambda r: -r.score(self.selection_metric)):
            row = {
                "model": m.name,
                f"val_{self.selection_metric}": getattr(m.val, self.selection_metric),
                "val_macro_f1": m.val.macro_f1,
                "val_log_loss": m.val.log_loss,
                "val_accuracy": m.val.accuracy,
            }
            if m.test is not None:
                row["test_macro_f1"] = m.test.macro_f1
                row["test_accuracy"] = m.test.accuracy
                row["test_false_alarm"] = (m.test.binary.false_alarm_rate
                                           if m.test.binary else float("nan"))
                row["missed_classes"] = len(m.test.classes_never_predicted)
            row["fit_s"] = round(m.fit_seconds, 1)
            row["tuned"] = m.tune is not None
            rows.append(row)
        return pd.DataFrame(rows)

    def render(self) -> str:
        parts = [
            "=" * 78,
            f"ShieldNet training run - {self.started_at}",
            "=" * 78,
            self.data.render(),
            "",
            self.space.render(),
            "",
        ]
        if self.succeeded:
            parts.append(f"Leaderboard (ranked by validation {self.selection_metric})")
            parts.append(self.leaderboard().to_string(index=False))
            if len(self.succeeded) > 1:
                # Blank test cells look like a bug unless you say why they are blank.
                parts.append("")
                parts.append("The test columns are empty for every model except the "
                             "selected one. That is deliberate: candidates are ranked on "
                             "validation, and the test split is scored once, afterwards, "
                             "for the winner alone.")
            parts.append("")
            # Compare on validation for every model - the one split all of them were
            # measured on. Mixing the winner's test report into this table would rank
            # like against unlike and quietly favour whichever split happened to be
            # easier.
            parts.append(compare([m.val for m in self.succeeded], sort_by="macro_f1"))
        for name, reason in self.skipped.items():
            parts.append(f"skipped {name}: {reason}")
        for m in self.models:
            if m.failed:
                parts.append(f"FAILED {m.name}: {m.failed}")
        if self.best is not None:
            parts += ["", "-" * 78,
                      f"Selected model: {self.best.name}", "-" * 78]
            report = self.best.test or self.best.val
            parts.append(report.render(confusion=True, sweep=True))
            parts.append("")
            parts.append(narrate_evaluation(
                report, baseline_accuracy=_majority_baseline(self.space.y_test)))
        if self.explanation is not None:
            parts += ["", self.explanation.render(top=20)]
        if self.bundle_path is not None:
            parts += ["", f"Artifact: {self.bundle_path}"]
        parts.append(f"\nTotal wall clock: {human_duration(self.seconds)}")
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "seconds": round(self.seconds, 2),
            "selection_metric": self.selection_metric,
            "best": self.best.name if self.best else None,
            "config": self.config.to_dict(),
            "environment": environment_report(),
            "data": self.data.to_dict(),
            "features": self.space.to_dict(),
            "models": [m.to_dict() for m in self.models],
            "skipped": dict(self.skipped),
            "explanation": (self.explanation.to_dict(top=30)
                            if self.explanation is not None else None),
            "samples": [s.to_dict() for s in self.samples],
            "figures": dict(self.figures),
            "bundle": str(self.bundle_path) if self.bundle_path else None,
        }

    def bundle(self) -> ModelBundle:
        """Assemble the deployable artifact for the selected model."""
        if self.best is None or self.best.model is None:
            raise RuntimeError("no model was selected, so there is nothing to bundle")
        report = self.best.test or self.best.val
        return ModelBundle(
            model_name=self.best.name,
            feature_names=list(self.space.selected),
            label_classes=list(self.space.class_names),
            scaler=self.space.preprocessor,
            medians=dict(self.space.raw_medians),
            model=self.best.model,
            metrics={
                "split": report.split,
                "accuracy": report.accuracy,
                "balanced_accuracy": report.balanced_accuracy,
                "macro_f1": report.macro_f1,
                "weighted_f1": report.weighted_f1,
                "log_loss": report.log_loss,
                "mcc": report.mcc,
                "macro_roc_auc": report.macro_roc_auc,
                "calibration_error": report.calibration_error,
                "false_alarm_rate": (report.binary.false_alarm_rate
                                     if report.binary else None),
                "attack_recall": report.binary.recall if report.binary else None,
                "per_class_recall": {c.name: c.recall for c in report.per_class},
                "classes_never_predicted": list(report.classes_never_predicted),
                "selection_metric": self.selection_metric,
                "val_selection_score": getattr(self.best.val, self.selection_metric),
            },
            metadata={
                "started_at": self.started_at,
                "seed": self.config.seed,
                "source": self.data.source,
                # One boolean, restated from `source`, because the string is for people
                # and this is for code. Every metric in this bundle either describes
                # CICIDS2017 or describes a random-number generator, and there is no way
                # to tell them apart once a figure has been copied into a report. So the
                # flag travels with the artifact and `shieldnet info` shouts about it.
                "synthetic": self.data.synthetic,
                "rows_fit": self.best.rows_fit,
                "rows_train": int(len(self.space.y_train)),
                "rows_test": int(len(self.space.y_test)),
                "params": _jsonable(self.best.params),
                "tuned": self.best.tune is not None,
                "balance": self.space.balance.strategy,
                # Named for what it is: rows SMOTE invented while balancing the training
                # split. The old name, "synthetic_rows", sat two lines under `source` and
                # read like "this data was synthetic" - a reader could see 8,070 against a
                # real CICIDS2017 run and conclude the whole thing was fabricated.
                "smote_rows": self.space.balance.synthetic_rows,
                "class_weight": {self.space.class_names[i]: round(float(w), 4)
                                 for i, w in sorted(self.space.class_weight.items())},
                "scaler": self.config.data.scaler,
                "clip_quantile": self.config.data.clip_quantile,
                "feature_selection": {
                    "methods": list(self.space.selection.methods_run),
                    "skipped": dict(self.space.selection.methods_skipped),
                    "mi_estimator": self.space.selection.mi_estimator,
                    "pruned_correlated": len(self.space.selection.dropped_correlated),
                },
                "explanation_method": (self.explanation.method
                                       if self.explanation else "not computed"),
                "top_features": ([n for n, _ in self.explanation.top_features(10)]
                                 if self.explanation else []),
                "candidates": [m.name for m in self.succeeded],
                "is_deep": bool(getattr(self.best.model, "is_deep", False)),
            },
        )


def _majority_baseline(y: np.ndarray) -> float:
    """Accuracy of always predicting the most common class. The number to beat."""
    y = np.asarray(y).ravel()
    if y.size == 0:
        return 0.0
    return float(np.bincount(y).max() / y.size)


def train(
    cfg: Optional[Config] = None,
    *,
    data: Optional[DataBundle] = None,
    space: Optional[FeatureSpace] = None,
    source: Optional[os.PathLike | str] = None,
    synthetic_rows: int = 0,
    models: Optional[Sequence[str]] = None,
    tune: Optional[bool] = None,
    explain: bool = True,
    save: bool = True,
    cache: bool = True,
    select: str = "auto",
    quiet: bool = False,
) -> TrainingRun:
    """Run the whole pipeline and return everything it produced.

    Parameters
    ----------
    models:
        Registry keys to try. Defaults to ``cfg.train.models``. Keys whose library is
        not installed are dropped with an explanatory note rather than attempted.
    select:
        ``"auto"`` ships whichever candidate wins on the validation
        ``selection_metric``; ``"primary"`` ships ``cfg.train.primary`` regardless, for
        when you need a specific model in the app.
    """
    cfg = cfg or Config.load()
    seed_everything(cfg.seed)
    cfg.paths.ensure("processed", "artifacts", "reports", "figures", "interim")
    started = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    log.info("ShieldNet training run at %s (seed %d)", stamp, cfg.seed)

    if space is not None and data is None:
        raise ValueError("pass the DataBundle alongside a prebuilt FeatureSpace; the "
                         "run summary reports on both")
    data = data or prepare_data(cfg, source=source, synthetic_rows=synthetic_rows,
                                cache=cache, quiet=quiet)
    space = space or build_feature_space(cfg, data, quiet=quiet)

    metric = cfg.train.selection_metric
    if metric not in METRICS:
        raise ValueError(f"train.selection_metric={metric!r} is not one of "
                         f"{sorted(METRICS)}")

    wanted = [resolve(n) for n in (models if models is not None else cfg.train.models)]
    seen: set = set()
    wanted = [k for k in wanted if not (k in seen or seen.add(k))]
    usable, reasons = available(wanted, warn=True)
    skipped = {k: reasons.get(k, "unavailable") for k in wanted if k not in usable}
    if not usable:
        raise RuntimeError(
            "none of the requested models can run in this environment:\n  "
            + "\n  ".join(f"{k}: {v}" for k, v in skipped.items())
            + "\nInstall one of them, or add 'logistic_regression' (scikit-learn only)."
        )
    log.info("training %d model(s): %s", len(usable), ", ".join(usable))

    # evaluate_test=False for every candidate, deliberately. Selection reads validation
    # only, so scoring test here would change no decision the program makes - but it
    # would put a test column next to each candidate on the leaderboard, and the reader
    # is then one glance away from picking the model with the best test score. That is
    # selection on test, done by a human instead of by a line of code, and it is not
    # detectable afterwards. The winner is scored on test below, once, after the choice
    # is already fixed.
    records = [train_one(key, space, cfg, tune=tune, evaluate_test=False, quiet=quiet)
               for key in usable]

    run = TrainingRun(config=cfg, data=data, space=space, models=records,
                      selection_metric=metric, skipped=skipped, started_at=stamp)

    ranked = sorted(run.succeeded, key=lambda m: -m.score(metric))
    if not ranked:
        run.seconds = time.perf_counter() - started
        raise RuntimeError(
            "every model failed to train. First error: "
            + (records[0].failed if records else "no models were attempted")
        )

    if select == "primary":
        primary = resolve(cfg.train.primary)
        chosen = next((m for m in ranked if m.name == primary), None)
        if chosen is None:
            log.warning("train.primary=%s did not train successfully; falling back to "
                        "the best available model", primary)
            chosen = ranked[0]
    elif select == "auto":
        chosen = ranked[0]
    else:
        raise ValueError(f"select must be 'auto' or 'primary', got {select!r}")
    run.best = chosen

    log.info("selected %s (validation %s = %.5f)", chosen.name, metric,
             getattr(chosen.val, metric))
    if len(ranked) > 1 and ranked[0] is chosen:
        gap = chosen.score(metric) - ranked[1].score(metric)
        log.info("  margin over %s: %.5f", ranked[1].name, gap)
        if gap < 0.005:
            log.info("  that margin is inside the noise for this validation size; "
                     "prefer the cheaper model if inference latency matters")

    # The test split is touched here and nowhere else in the run, for the winner and
    # nobody else. Everything above this line decided which model ships; this line only
    # measures how well it does on data no fitting, tuning or selection step has seen.
    # That is what makes the number reportable.
    _score_on_test(chosen, space, quiet=quiet)

    if explain:
        run.explanation, run.samples = _explain_best(cfg, space, chosen, quiet=quiet)

    # Set the elapsed time before saving: run_summary.txt reports it, and a summary that
    # claims the run took no time is the kind of small wrongness that makes a reader
    # distrust the rest of the file.
    run.seconds = time.perf_counter() - started
    if save:
        _save_run(cfg, run)

    run.seconds = time.perf_counter() - started
    log.info("run complete in %s", human_duration(run.seconds))
    return run


# ---------------------------------------------------------------------------
# explanation and persistence
# ---------------------------------------------------------------------------

def _explain_best(
    cfg: Config, space: FeatureSpace, best: TrainedModel, *, quiet: bool = False
) -> Tuple[Optional[GlobalExplanation], List[LocalExplanation]]:
    """Global importance plus one representative local explanation per class."""
    try:
        with stage(log, "explanation", quiet=quiet) as st:
            explainer = Explainer(
                best.model, space.selected, space.class_names, seed=cfg.seed,
                max_background=cfg.train.shap_background_rows,
                preprocessor=space.preprocessor)
            explainer.set_background(space.X_train, space.y_train)
            global_exp = explainer.global_explanation(
                space.X_test, space.y_test,
                max_rows=cfg.train.shap_explain_rows, quiet=quiet)

            check = explainer.verify_additivity(space.X_test[:32])
            if check.get("checked") and not check.get("passed", True):
                log.warning("SHAP additivity check failed (worst error %.3g). The "
                            "attribution array may be transposed; treat the per-class "
                            "ranking with suspicion.", check.get("worst_error", 0.0))
                global_exp.notes.append(
                    f"additivity check failed: worst error "
                    f"{check.get('worst_error'):.3g} against a tolerance of "
                    f"{check.get('tolerance')}")
            elif check.get("checked"):
                global_exp.notes.append(
                    f"additivity verified on {check.get('rows')} rows "
                    f"(worst error {check.get('worst_error'):.2e})")

            samples = _sample_explanations(explainer, space, best)
            agreement = global_exp.agreement_with(space.selection.selected)
            log.info("explanation agrees with feature selection on %d of the top 10",
                     agreement.get("top10_overlap", 0))
            global_exp.notes.append(
                f"top-10 overlap with the selection ranking: "
                f"{agreement.get('top10_overlap', 0)}/10")
            st["summary"] = (f"{global_exp.method}, {global_exp.rows_explained:,} rows, "
                             f"{len(samples)} worked example(s)")
        return global_exp, samples
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        log.error("explanation failed; the model is still usable: %s: %s",
                  type(exc).__name__, exc)
        log.debug("explanation traceback", exc_info=True)
        return None, []


def _sample_explanations(
    explainer: Explainer, space: FeatureSpace, best: TrainedModel
) -> List[LocalExplanation]:
    """One worked example per class, for the report and the app's demo tab.

    Prefers a correctly-classified row so the narration describes a real detection.
    Falls back to the highest-scoring row of the class when the model never gets it
    right, because "here is what the model does with a Bot flow" is still the most
    informative thing to show for the class it is failing on.
    """
    if best.test is None or best.model is None:
        return []
    proba = best.model.predict_proba(space.X_test)
    predicted = proba.argmax(axis=1)
    out: List[LocalExplanation] = []
    raw_test = space.preprocessor.scaler.inverse_transform(space.X_test)
    for index in range(len(space.class_names)):
        rows = np.nonzero(space.y_test == index)[0]
        if rows.size == 0:
            continue
        correct = rows[predicted[rows] == index]
        pool = correct if correct.size else rows
        pick = int(pool[int(np.argmax(proba[pool, index]))])
        try:
            out.append(explainer.explain_row(space.X_test[pick],
                                             raw_values=raw_test[pick]))
        except Exception as exc:                     # one bad row must not kill the set
            log.debug("could not explain row %d (%s): %s", pick,
                      space.class_names[index], exc)
    return out


def _promote(staging: Path, artifacts: Path) -> Path:
    """Move a verified bundle out of *staging* into *artifacts* and return its path.

    Only the files the bundle actually wrote are moved, so anything else already living in
    ``artifacts/`` - ``optuna.db`` above all, which a resumed study needs - is left alone.
    ``os.replace`` overwrites in one step rather than unlink-then-write, so a reader that
    opens ``bundle.joblib`` during a re-train sees either the old file or the new one and
    never a half-written one.
    """
    from .persist import BUNDLE_FILE

    moved: List[str] = []
    for source in sorted(staging.iterdir()):
        if source.is_file():
            os.replace(source, artifacts / source.name)
            moved.append(source.name)
    shutil.rmtree(staging, ignore_errors=True)
    log.debug("promoted %s -> %s", ", ".join(moved), artifacts)
    return artifacts / BUNDLE_FILE


def _save_run(cfg: Config, run: TrainingRun) -> None:
    """Write the bundle, the machine-readable summary, the tables and the figures."""
    artifacts = cfg.paths.resolve("artifacts")
    reports = cfg.paths.resolve("reports")
    figures = cfg.paths.resolve("figures")
    for directory in (artifacts, reports, figures):
        directory.mkdir(parents=True, exist_ok=True)

    with stage(log, "saving artifacts") as st:
        bundle = run.bundle()

        # Write into a staging directory, verify there, and only then move the files into
        # place. Round-tripping *after* writing to artifacts/ would make the check a log
        # line rather than a gate: the unshippable bundle would already be sitting where
        # `shieldnet serve` loads it, and it would have overwritten the last good one on
        # the way in. Staging is a subdirectory of artifacts/ so the promotion is a rename
        # within one filesystem, which is atomic per file.
        staging = artifacts / ".staging"
        if staging.exists():
            shutil.rmtree(staging)
        bundle.save(staging)

        # A bundle that cannot be restored is worse than no bundle, because the failure
        # surfaces in the app in front of an audience rather than here where it can be
        # fixed.
        restored = ModelBundle.restore(staging)
        check = restored.model.predict_proba(run.space.X_test[:64])
        original = run.best.model.predict_proba(run.space.X_test[:64])
        drift = float(np.abs(check - original).max())
        if drift > 1e-5:
            shutil.rmtree(staging, ignore_errors=True)
            raise RuntimeError(
                f"the saved bundle predicts differently from the in-memory model "
                f"(max probability difference {drift:.3g}), so it was not installed. "
                f"{artifacts} still holds whatever it held before this run."
            )
        log.info("bundle round-trip verified (max probability drift %.2e)", drift)
        run.bundle_path = _promote(staging, artifacts)

        run.leaderboard().to_csv(reports / "leaderboard.csv", index=False)
        run.space.selection.ranking.to_csv(reports / "feature_ranking.csv", index=False)

        for record in run.succeeded:
            report = record.test or record.val
            (reports / f"evaluation_{record.name}.json").write_text(
                report.to_json(), encoding="utf-8")
            if record.tune is not None:
                (reports / f"tuning_{record.name}.json").write_text(
                    record.tune.to_json(), encoding="utf-8")

        if run.explanation is not None:
            run.explanation.frame().to_csv(reports / "global_importance.csv",
                                           index=False)
            if run.explanation.per_class is not None:
                run.explanation.per_class_frame().to_csv(
                    reports / "per_class_importance.csv")
            for label, path in (
                ("importance", run.explanation.plot(
                    figures / "importance.png", top=20,
                    title=f"{run.best.name}: global feature importance")),
                ("per_class_importance", run.explanation.plot_per_class(
                    figures / "per_class_importance.png", top=15)),
            ):
                if path is not None:
                    run.figures[label] = str(path)
            if run.samples:
                (reports / "worked_examples.txt").write_text(
                    _worked_examples_text(run), encoding="utf-8")

        best_report = run.best.test or run.best.val
        confusion = plot_confusion(best_report, figures / "confusion.png",
                                   normalise=True)
        if confusion is not None:
            run.figures["confusion"] = str(confusion)

        # Written last, deliberately: the summary records the figure paths and the
        # bundle digest, so anything that adds to `run` has to happen before it.
        (reports / "run_summary.json").write_text(
            json.dumps(run.to_dict(), indent=2, default=str), encoding="utf-8")
        (reports / "run_summary.txt").write_text(run.render(), encoding="utf-8")

        st["summary"] = (f"bundle + {len(list(reports.glob('*.json')))} report file(s) "
                         f"+ {len(run.figures)} figure(s)")


def _worked_examples_text(run: TrainingRun) -> str:
    from .narrate import narrate_prediction
    blocks = []
    for sample in run.samples:
        blocks.append("=" * 78)
        blocks.append(f"Worked example: {sample.predicted_class}")
        blocks.append("=" * 78)
        blocks.append(narrate_prediction(sample, top=5))
        blocks.append("")
        blocks.append(sample.render(top=10))
        blocks.append("")
    return "\n".join(blocks)


def plot_confusion(
    report: EvaluationReport,
    path: os.PathLike | str,
    *,
    normalise: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
) -> Optional[Path]:
    """Confusion-matrix heatmap, row-normalised by default.

    Row normalisation is the default because the raw counts are unreadable here: the
    BENIGN row holds six figures while the Rare Attacks row holds single digits, so on
    a shared colour scale every attack row is the same shade of white and the plot shows
    nothing. Normalising by row turns each cell into "what fraction of true class *i*
    was called class *j*", which is per-class recall on the diagonal and the actual
    confusions off it.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                          # pragma: no cover
        log.warning("matplotlib unavailable, skipping the confusion figure: %s", exc)
        return None

    cm = np.asarray(report.confusion, dtype=np.float64)
    names = list(report.class_names)
    shown = cm
    if normalise:
        totals = cm.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            shown = np.where(totals > 0, cm / np.maximum(totals, 1e-12), 0.0)

    n = len(names)
    size = figsize or (max(7.0, 0.62 * n + 3.4), max(6.0, 0.55 * n + 2.8))
    fig, ax = plt.subplots(figsize=size, dpi=150)
    image = ax.imshow(shown, cmap="Blues", vmin=0.0,
                      vmax=1.0 if normalise else float(shown.max() or 1.0))

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([f"{name}  ({int(cm[i].sum()):,})"
                        for i, name in enumerate(names)], fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("actual (support)")
    ax.set_title(f"{report.model} - confusion on {report.split} "
                 f"({'row-normalised' if normalise else 'counts'})", fontsize=11)

    limit = shown.max() if not normalise else 1.0
    for i in range(n):
        for j in range(n):
            value = shown[i, j]
            if normalise and value < 0.005:
                continue
            if not normalise and cm[i, j] == 0:
                continue
            text = f"{value:.0%}" if normalise else human_count(cm[i, j])
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    color="white" if value > 0.55 * limit else "#20303f")

    fig.colorbar(image, ax=ax, fraction=0.042, pad=0.03,
                 label="fraction of the true class" if normalise else "flows")
    fig.tight_layout()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return target


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
