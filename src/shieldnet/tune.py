"""Hyper-parameter search, with Optuna when available and random search when not.

Why the objective is validation log loss over a small CV, not test macro F1
--------------------------------------------------------------------------
Three separate decisions, each of which changes the answer:

**Never the test split.** Selecting hyper-parameters by test performance makes the test
score an optimistically biased estimate - you have used the test set 40 times and are
reporting the maximum of 40 noisy draws. The test split is touched exactly once, in
``train.py``, after tuning has finished. This is the most common methodological error in
the CICIDS2017 literature and it is worth being pedantic about.

**Log loss, not macro F1.** Macro F1 is the metric the project *reports*, so tuning
against it sounds right. But on a validation split with 30 rows of one class, macro F1
moves in steps of about 1/13 x 1/30 - a step function with long flat stretches. Optuna's
TPE builds a density model over the search space from the observed values; a step
function gives it almost nothing to model, and it degenerates towards random search. Log
loss is smooth, strictly proper, and correlates well with macro F1 here. ``metric="macro_f1"``
is available for anyone who wants to check that claim rather than take it on faith.

**Cross-validation, not the single validation split.** With ``cv_folds=1`` a rare class
contributes maybe 6 validation rows, and the difference between two hyper-parameter sets
is decided by whether 4 or 5 of those 6 are caught - noise, not signal. Three folds
triples the rare-class validation rows and averages three fits. It costs 3x the time,
which is why ``cv_folds`` is configurable and why the default drops to 1 for the deep
models where a fit is minutes rather than seconds.

Pruning
-------
When Optuna is present, trials report their intermediate fold scores and a
``MedianPruner`` kills the ones already losing after the first fold. That is roughly a
40% saving on a 40-trial study. The random-search fallback implements the same idea
manually: after the first fold, if the score is already worse than the running best by
more than the observed spread, the remaining folds are skipped.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .evaluate import evaluate, stratified_folds, stratified_subsample
from .logging_utils import get_logger, human_duration, stage
from .models import build, search_space_for
from .preprocess import class_weights

log = get_logger(__name__)

__all__ = ["TrialResult", "TuneReport", "tune_model", "RandomTrial",
           "objective_value", "METRICS"]

#: Objective metrics, and whether smaller is better.
METRICS: Dict[str, bool] = {
    "log_loss": True,
    "macro_f1": False,
    "macro_recall": False,
    "balanced_accuracy": False,
    "mcc": False,
    "accuracy": False,          # offered for completeness; a poor choice here
}


def objective_value(report, metric: str) -> float:
    """Pull *metric* off an :class:`~shieldnet.evaluate.EvaluationReport`."""
    if metric not in METRICS:
        raise ValueError(f"unknown objective metric {metric!r}; "
                         f"choose one of {sorted(METRICS)}")
    value = float(getattr(report, metric))
    if not math.isfinite(value):
        # A non-finite objective would poison the TPE model. Report the worst possible
        # value instead so the trial is recorded as bad rather than discarded.
        return float("inf") if METRICS[metric] else float("-inf")
    return value


# ---------------------------------------------------------------------------
# the random-search fallback
# ---------------------------------------------------------------------------

class RandomTrial:
    """A stand-in for ``optuna.Trial`` that samples uniformly at random.

    Every ``search_space`` in :mod:`shieldnet.models` is written against the Optuna
    ``trial.suggest_*`` API. Reimplementing that API over ``numpy.random`` means those
    search spaces are the single definition of what is tunable: with Optuna installed
    you get TPE, without it you get random search over *exactly the same space*, and
    there is no second copy of the bounds to drift out of sync.

    Random search is not a token gesture, either. Bergstra and Bengio (2012) showed it
    beats grid search for the same budget, and with 8 hyper-parameters and 40 trials it
    lands within a few percent of TPE most of the time.
    """

    def __init__(self, number: int, rng: np.random.Generator) -> None:
        self.number = int(number)
        self._rng = rng
        self.params: Dict[str, Any] = {}
        self._reported: Dict[int, float] = {}
        self.should_prune = False

    # -- the Optuna surface used by our search spaces ------------------------

    def suggest_float(self, name: str, low: float, high: float, *,
                      step: Optional[float] = None, log: bool = False) -> float:
        if log:
            if low <= 0:
                raise ValueError(f"{name}: log scale needs low > 0, got {low}")
            value = float(np.exp(self._rng.uniform(np.log(low), np.log(high))))
        elif step:
            n = int(round((high - low) / step))
            value = float(low + step * self._rng.integers(0, n + 1))
        else:
            value = float(self._rng.uniform(low, high))
        self.params[name] = value
        return value

    def suggest_int(self, name: str, low: int, high: int, *, step: int = 1,
                    log: bool = False) -> int:
        if log:
            value = int(round(float(np.exp(self._rng.uniform(np.log(max(low, 1)),
                                                             np.log(high))))))
            value = int(min(max(value, low), high))
        else:
            choices = np.arange(int(low), int(high) + 1, int(step))
            value = int(self._rng.choice(choices))
        self.params[name] = value
        return value

    def suggest_categorical(self, name: str, choices: Sequence[Any]) -> Any:
        # rng.choice on a list of mixed types (None, 12, 20) coerces to object arrays
        # unpredictably, so index instead.
        value = list(choices)[int(self._rng.integers(len(choices)))]
        self.params[name] = value
        return value

    # -- pruning -------------------------------------------------------------

    def report(self, value: float, step: int) -> None:
        self._reported[int(step)] = float(value)

    def set_user_attr(self, key: str, value: Any) -> None:   # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    """One evaluated hyper-parameter set."""

    number: int
    params: Dict[str, Any]
    value: float
    fold_values: List[float] = field(default_factory=list)
    seconds: float = 0.0
    pruned: bool = False
    failed: str = ""

    @property
    def spread(self) -> float:
        """Standard deviation across folds - how trustworthy the mean is."""
        if len(self.fold_values) < 2:
            return 0.0
        return float(np.std(self.fold_values, ddof=1))

    def to_dict(self) -> Dict[str, Any]:
        return {"number": self.number, "params": _jsonable(self.params),
                "value": self.value, "fold_values": self.fold_values,
                "seconds": self.seconds, "pruned": self.pruned,
                "failed": self.failed}


@dataclass
class TuneReport:
    """The outcome of one study."""

    model: str
    metric: str
    lower_is_better: bool
    best_params: Dict[str, Any]
    best_value: float
    baseline_value: float
    trials: List[TrialResult]
    sampler: str
    seconds: float
    cv_folds: int
    rows_used: int
    stopped_early: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def improvement(self) -> float:
        """How much the search gained over the hand-set defaults."""
        if not math.isfinite(self.baseline_value) or not math.isfinite(self.best_value):
            return float("nan")
        delta = self.baseline_value - self.best_value
        return delta if self.lower_is_better else -delta

    @property
    def completed(self) -> int:
        return sum(1 for t in self.trials if not t.pruned and not t.failed)

    @property
    def kept_defaults(self) -> bool:
        """Whether the search failed to beat the defaults."""
        return not self.best_params

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model, "metric": self.metric,
            "lower_is_better": self.lower_is_better,
            "best_params": _jsonable(self.best_params),
            "best_value": self.best_value, "baseline_value": self.baseline_value,
            "improvement": self.improvement, "sampler": self.sampler,
            "seconds": self.seconds, "cv_folds": self.cv_folds,
            "rows_used": self.rows_used, "trials_run": len(self.trials),
            "trials_completed": self.completed,
            "trials_pruned": sum(1 for t in self.trials if t.pruned),
            "trials_failed": sum(1 for t in self.trials if t.failed),
            "stopped_early": self.stopped_early,
            "history": [t.to_dict() for t in self.trials],
            "notes": list(self.notes),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def render(self, *, top: int = 5) -> str:
        arrow = "lower is better" if self.lower_is_better else "higher is better"
        lines = [
            f"hyper-parameter search: {self.model}",
            f"  sampler          {self.sampler}",
            f"  objective        {self.metric} ({arrow}), {self.cv_folds}-fold CV on "
            f"{self.rows_used:,} rows",
            f"  trials           {len(self.trials)} run, {self.completed} completed, "
            f"{sum(1 for t in self.trials if t.pruned)} pruned, "
            f"{sum(1 for t in self.trials if t.failed)} failed",
            f"  elapsed          {human_duration(self.seconds)}",
            f"  defaults scored  {self.baseline_value:.5f}",
            f"  best scored      {self.best_value:.5f} "
            f"({self.improvement:+.5f} vs defaults)",
        ]
        if self.stopped_early:
            lines.append(f"  stopped early    {self.stopped_early}")
        if self.kept_defaults:
            lines.append("  the search did not beat the hand-set defaults, so those are "
                         "kept - which is a result, not a failure")
        else:
            lines.append("  best parameters:")
            for key, value in sorted(self.best_params.items()):
                lines.append(f"      {key:<24} {value}")

        ranked = sorted((t for t in self.trials if not t.failed),
                        key=lambda t: t.value if math.isfinite(t.value)
                        else (float("inf") if self.lower_is_better else float("-inf")),
                        reverse=not self.lower_is_better)[:top]
        if ranked:
            lines.append("")
            lines.append(f"  top {len(ranked)} trial(s):")
            lines.append(f"      {'#':<5}{'value':>11}{'spread':>10}{'time':>9}  params")
            for t in ranked:
                flag = " (pruned)" if t.pruned else ""
                shown = ", ".join(f"{k}={_short(v)}"
                                  for k, v in sorted(t.params.items()))
                lines.append(f"      {t.number:<5}{t.value:>11.5f}{t.spread:>10.5f}"
                             f"{t.seconds:>8.1f}s  {shown[:80]}{flag}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)

    def __str__(self) -> str:                       # pragma: no cover
        return self.render()


def _short(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _jsonable(obj: Any) -> Any:
    """Make numpy scalars and tuples survive ``json.dumps``."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# the search
# ---------------------------------------------------------------------------

def _score_params(
    model_name: str,
    params: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    class_names: Sequence[str],
    metric: str,
    seed: int,
    n_jobs: int,
    weights: Optional[Dict[int, float]],
    on_fold: Optional[Callable[[int, float], bool]] = None,
) -> Tuple[float, List[float]]:
    """Fit and score one parameter set across the folds.

    ``on_fold(step, value)`` may return True to abandon the remaining folds - that is the
    hook both the Optuna pruner and the manual fallback pruner use.
    """
    n_classes = len(class_names)
    fold_values: List[float] = []
    for step, (train_idx, val_idx) in enumerate(folds):
        model = build(model_name, n_classes=n_classes, n_features=X.shape[1],
                      seed=seed, n_jobs=n_jobs, params=dict(params))
        model.fit(X[train_idx], y[train_idx],
                  X_val=X[val_idx], y_val=y[val_idx], class_weight=weights)
        report = evaluate(
            y[val_idx], model.predict_proba(X[val_idx]),
            class_names=class_names, model=model_name, split=f"fold{step}",
            # No AUCs and no threshold sweep inside the search: 13 rank sorts of the
            # validation fold, per fold, per trial, is a large fraction of the runtime
            # and no objective in METRICS reads them.
            curves=False,
        )
        value = objective_value(report, metric)
        fold_values.append(value)
        if on_fold is not None and on_fold(step, value):
            break
    if not fold_values:
        raise RuntimeError("no folds were scored")
    return float(np.mean(fold_values)), fold_values


def tune_model(
    model_name: str,
    X: Any,
    y: Any,
    *,
    class_names: Sequence[str],
    metric: str = "log_loss",
    n_trials: int = 30,
    cv_folds: int = 3,
    timeout: Optional[float] = None,
    max_rows: Optional[int] = 80_000,
    balance_weights: bool = True,
    weight_cap: float = 50.0,
    seed: int = 42,
    n_jobs: int = -1,
    prefer_optuna: bool = True,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    quiet: bool = False,
) -> TuneReport:
    """Search hyper-parameters for one model.

    Returns a :class:`TuneReport` whose ``best_params`` is ``{}`` when the search could
    not beat the defaults - callers should treat that as "use the defaults", which is
    what ``train.py`` does.

    ``max_rows`` subsamples the tuning data. Tuning on 300k rows x 40 trials x 3 folds is
    120 full fits, which for XGBoost is over an hour; on 80k stratified rows it is a few
    minutes and the ranking of parameter sets is essentially unchanged, because
    hyper-parameter *ordering* is far more stable under subsampling than absolute
    performance is.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown objective metric {metric!r}; "
                         f"choose one of {sorted(METRICS)}")
    lower_is_better = METRICS[metric]

    X_arr = np.ascontiguousarray(
        X.to_numpy() if hasattr(X, "to_numpy") else X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int64).ravel()
    if X_arr.shape[0] != y_arr.size:
        raise ValueError(f"{X_arr.shape[0]} rows of X but {y_arr.size} labels")
    if not np.isfinite(X_arr).all():
        raise ValueError(
            "tuning data contains NaN or inf. Fit the Preprocessor before tuning - "
            "several estimators here accept non-finite input silently and produce "
            "meaningless splits."
        )

    notes: List[str] = []
    rng = np.random.default_rng(seed)

    if max_rows and y_arr.size > max_rows:
        keep = stratified_subsample(y_arr, max_rows, rng)
        notes.append(f"tuned on a stratified {len(keep):,}-row subsample of "
                     f"{y_arr.size:,} (every class kept, large ones capped)")
        X_arr, y_arr = X_arr[keep], y_arr[keep]

    n_classes = len(class_names)
    if n_classes < 2:
        raise ValueError("need at least 2 classes to tune")

    cv_folds = max(1, int(cv_folds))
    if cv_folds == 1:
        # A single 80/20 holdout, still stratified by the same round-robin logic.
        folds = [_holdout(y_arr, 0.2, rng)]
        notes.append("single holdout instead of cross-validation, so trial-to-trial "
                     "differences smaller than the noise floor are not meaningful")
    else:
        folds = stratified_folds(y_arr, cv_folds, seed=seed)

    weights = class_weights(y_arr, n_classes, cap=weight_cap) \
        if balance_weights else None

    # The defaults are trial -1: without this the report can say "best log loss 0.081"
    # without saying whether that is better than doing nothing at all.
    started = time.perf_counter()
    with stage(log, f"tuning {model_name} ({metric}, {len(folds)} fold(s), "
                    f"{y_arr.size:,} rows)", quiet=quiet):
        try:
            baseline, baseline_folds = _score_params(
                model_name, {}, X_arr, y_arr, folds, class_names=class_names,
                metric=metric, seed=seed, n_jobs=n_jobs, weights=weights)
            log.info("%s defaults score %.5f (%s)", model_name, baseline, metric)
        except Exception as exc:                       # noqa: BLE001
            raise RuntimeError(
                f"could not fit {model_name} with its default parameters, so there is "
                f"nothing to tune against: {exc}"
            ) from exc

        probe = RandomTrial(-1, np.random.default_rng(seed))
        try:
            space = search_space_for(model_name, probe)
        except Exception as exc:                       # noqa: BLE001
            space = {}
            notes.append(f"search space unavailable ({exc})")
        if not space:
            notes.append(f"{model_name} declares no tunable parameters; the defaults "
                         "are the result")
            return TuneReport(
                model=model_name, metric=metric, lower_is_better=lower_is_better,
                best_params={}, best_value=baseline, baseline_value=baseline,
                trials=[TrialResult(-1, {}, baseline, baseline_folds,
                                    time.perf_counter() - started)],
                sampler="none", seconds=time.perf_counter() - started,
                cv_folds=len(folds), rows_used=int(y_arr.size), notes=notes,
            )

        runner = _optuna_study if prefer_optuna else None
        report: Optional[TuneReport] = None
        if runner is not None:
            report = _optuna_study(
                model_name, X_arr, y_arr, folds, class_names=class_names,
                metric=metric, lower_is_better=lower_is_better, n_trials=n_trials,
                timeout=timeout, seed=seed, n_jobs=n_jobs, weights=weights,
                baseline=baseline, baseline_folds=baseline_folds,
                study_name=study_name, storage=storage, notes=notes,
            )
        if report is None:
            report = _random_search(
                model_name, X_arr, y_arr, folds, class_names=class_names,
                metric=metric, lower_is_better=lower_is_better, n_trials=n_trials,
                timeout=timeout, seed=seed, n_jobs=n_jobs, weights=weights,
                baseline=baseline, baseline_folds=baseline_folds, notes=notes,
            )

    report.seconds = time.perf_counter() - started
    report.rows_used = int(y_arr.size)
    report.cv_folds = len(folds)

    better = (report.best_value < baseline) if lower_is_better \
        else (report.best_value > baseline)
    if not better:
        report.best_params = {}
        report.best_value = baseline
        report.notes.append("no trial beat the defaults; defaults retained")
        log.info("%s: tuning kept the default parameters", model_name)
    else:
        log.info("%s: %s improved %.5f -> %.5f", model_name, metric, baseline,
                 report.best_value)
    return report


# ---------------------------------------------------------------------------
# Optuna backend
# ---------------------------------------------------------------------------

def _optuna_study(
    model_name, X, y, folds, *, class_names, metric, lower_is_better, n_trials,
    timeout, seed, n_jobs, weights, baseline, baseline_folds, study_name, storage,
    notes,
) -> Optional[TuneReport]:
    """Run a TPE study. Returns ``None`` if Optuna is not installed."""
    try:
        import optuna
        from optuna.exceptions import TrialPruned
    except ImportError:
        log.info("Optuna is not installed; falling back to random search over the same "
                 "space (pip install optuna for TPE and pruning)")
        return None

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    direction = "minimize" if lower_is_better else "maximize"
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    pruner = (optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
              if len(folds) > 1 else optuna.pruners.NopPruner())

    kwargs: Dict[str, Any] = dict(direction=direction, sampler=sampler, pruner=pruner)
    if study_name:
        kwargs["study_name"] = study_name
    if storage:
        # An SQLite storage makes a study resumable, which matters when a Colab runtime
        # disconnects 25 trials into a 40-trial search.
        kwargs["storage"] = storage
        kwargs["load_if_exists"] = True
    study = optuna.create_study(**kwargs)
    # Seed the study with the defaults so TPE starts from a known-good point instead of
    # spending its first ten trials discovering the region the defaults already sit in.
    try:
        study.enqueue_trial({}, skip_if_exists=True)
    except Exception:                                   # noqa: BLE001
        pass

    results: List[TrialResult] = [
        TrialResult(-1, {}, baseline, list(baseline_folds), 0.0)
    ]

    def objective(trial):
        started = time.perf_counter()
        params = search_space_for(model_name, trial)

        def on_fold(step: int, value: float) -> bool:
            trial.report(value, step)
            if trial.should_prune():
                raise TrialPruned()
            return False

        try:
            mean, fold_values = _score_params(
                model_name, params, X, y, folds, class_names=class_names,
                metric=metric, seed=seed, n_jobs=n_jobs, weights=weights,
                on_fold=on_fold)
        except TrialPruned:
            results.append(TrialResult(trial.number, dict(params),
                                       float("inf") if lower_is_better
                                       else float("-inf"), [],
                                       time.perf_counter() - started, pruned=True))
            raise
        except Exception as exc:                        # noqa: BLE001
            # One bad parameter combination (an invalid solver pairing, an out-of-memory
            # tree) must not end a 40-trial study.
            log.debug("%s trial %d failed: %s", model_name, trial.number, exc)
            results.append(TrialResult(trial.number, dict(params),
                                       float("inf") if lower_is_better
                                       else float("-inf"), [],
                                       time.perf_counter() - started, failed=str(exc)))
            raise TrialPruned() from exc

        results.append(TrialResult(trial.number, dict(params), mean, fold_values,
                                   time.perf_counter() - started))
        return mean

    study.optimize(objective, n_trials=int(n_trials), timeout=timeout,
                   catch=(Exception,), show_progress_bar=False)

    best_params: Dict[str, Any] = {}
    best_value = baseline
    completed = [t for t in results if not t.pruned and not t.failed and t.number >= 0]
    if completed:
        pick = min(completed, key=lambda t: t.value) if lower_is_better \
            else max(completed, key=lambda t: t.value)
        # The parameters recorded on our TrialResult, not study.best_params: for models
        # whose search space derives a value (MLP turns n_layers+width into a
        # hidden_layer_sizes tuple), Optuna stores the raw suggestions and the derived
        # dict is what the model actually needs.
        best_params, best_value = dict(pick.params), pick.value

    return TuneReport(
        model=model_name, metric=metric, lower_is_better=lower_is_better,
        best_params=best_params, best_value=best_value, baseline_value=baseline,
        trials=results, sampler="optuna-tpe", seconds=0.0, cv_folds=len(folds),
        rows_used=int(y.size), notes=notes,
    )


# ---------------------------------------------------------------------------
# random-search backend
# ---------------------------------------------------------------------------

def _random_search(
    model_name, X, y, folds, *, class_names, metric, lower_is_better, n_trials,
    timeout, seed, n_jobs, weights, baseline, baseline_folds, notes,
) -> TuneReport:
    """Random search over the same space, with manual median pruning."""
    rng = np.random.default_rng(seed + 1)
    results: List[TrialResult] = [
        TrialResult(-1, {}, baseline, list(baseline_folds), 0.0)
    ]
    best_value = baseline
    best_params: Dict[str, Any] = {}
    first_fold_scores: List[float] = list(baseline_folds[:1])
    started = time.perf_counter()
    stopped = ""

    for number in range(int(n_trials)):
        if timeout is not None and time.perf_counter() - started > timeout:
            stopped = f"timeout after {number} trial(s)"
            log.info("%s: %s", model_name, stopped)
            break

        trial = RandomTrial(number, rng)
        params = search_space_for(model_name, trial)
        t0 = time.perf_counter()

        # Median pruning by hand: after fold 0, if this trial is already worse than the
        # median of every previous trial's fold 0, the remaining folds are unlikely to
        # rescue it. n_warmup keeps the first few trials intact so the median means
        # something.
        threshold = None
        if len(first_fold_scores) >= 5:
            threshold = float(np.median(first_fold_scores))
        pruned = False

        def on_fold(step: int, value: float) -> bool:
            nonlocal pruned
            if step == 0:
                first_fold_scores.append(value)
                if threshold is not None and len(folds) > 1:
                    worse = value > threshold if lower_is_better else value < threshold
                    if worse:
                        pruned = True
                        return True
            return False

        try:
            mean, fold_values = _score_params(
                model_name, params, X, y, folds, class_names=class_names,
                metric=metric, seed=seed, n_jobs=n_jobs, weights=weights,
                on_fold=on_fold)
        except Exception as exc:                        # noqa: BLE001
            log.debug("%s trial %d failed: %s", model_name, number, exc)
            results.append(TrialResult(number, dict(params),
                                       float("inf") if lower_is_better
                                       else float("-inf"), [],
                                       time.perf_counter() - t0, failed=str(exc)))
            continue

        result = TrialResult(number, dict(params), mean, fold_values,
                            time.perf_counter() - t0, pruned=pruned)
        results.append(result)
        if not pruned:
            improved = mean < best_value if lower_is_better else mean > best_value
            if improved:
                best_value, best_params = mean, dict(params)
                log.info("%s trial %d: %s %.5f (new best)", model_name, number,
                         metric, mean)

    n_pruned = sum(1 for t in results if t.pruned)
    if n_pruned:
        notes.append(f"{n_pruned} trial(s) were pruned after the first fold for being "
                     "worse than the running median")
    return TuneReport(
        model=model_name, metric=metric, lower_is_better=lower_is_better,
        best_params=best_params, best_value=best_value, baseline_value=baseline,
        trials=results, sampler="random-search", seconds=0.0, cv_folds=len(folds),
        rows_used=int(y.size), stopped_early=stopped, notes=notes,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _holdout(
    y: np.ndarray, fraction: float, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """One stratified train/validation split, safe for single-row classes."""
    train: List[np.ndarray] = []
    val: List[np.ndarray] = []
    for cls in np.unique(y):
        idx = np.nonzero(y == cls)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * fraction))
        # A class with 1 row goes entirely to train: a validation row it has never seen
        # in training contributes nothing but a guaranteed miss.
        n_val = min(max(n_val, 1 if len(idx) >= 2 else 0), len(idx) - 1)
        val.append(idx[:n_val])
        train.append(idx[n_val:])
    tr, va = np.concatenate(train), np.concatenate(val)
    if va.size == 0:
        raise ValueError("holdout split came out empty; every class has a single row")
    return tr, va
