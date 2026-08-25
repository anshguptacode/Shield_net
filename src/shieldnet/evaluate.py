"""Metrics, written in numpy so they are testable without scikit-learn.

Why accuracy is close to meaningless on this dataset
----------------------------------------------------
The working chunk is roughly half BENIGN. A classifier that answers "BENIGN" to
everything therefore scores about 50%, and on the *full* CICIDS2017 it scores 80.3%.
Published intrusion-detection results quoting 99.8% accuracy on this data are quoting a
number that a nearly-useless model gets most of the way to. Worse, the 21 SQL-injection
flows are 21/2,830,743 of the data - 0.00074%: a model can miss every single one and lose
seven ten-thousandths of one percent of its accuracy.

So the metrics that decide anything here are:

``macro_f1``
    Unweighted mean of the per-class F1 scores, so the 36 Infiltration flows count as
    much as the 2,273,097 BENIGN ones. This is the project's model-selection metric.
``recall`` per class
    The detection rate. In an IDS a missed attack costs more than a false alarm, and
    per-class recall is the only number that shows a class being silently ignored.
``false_alarm_rate``
    On the attack-vs-benign collapse: what fraction of benign traffic gets flagged. An
    analyst drowning in false positives switches the system off, so this is the metric
    that decides whether a deployment survives contact with a real SOC.
``log_loss``
    Punishes confident mistakes. Used as the tuning objective because it is smooth -
    macro F1 moves in discrete jumps as a rare class flips, which gives Optuna almost
    nothing to optimise over.

Averaging convention
--------------------
Macro and weighted averages run over classes with **support > 0** in ``y_true``. A test
split that happens to contain no Heartbleed flows would otherwise contribute an F1 of
zero for that class and silently drag macro F1 down by 1/13, making two runs on
different splits incomparable. :attr:`EvaluationReport.classes_evaluated` records how
many classes were actually averaged over, so the omission is visible rather than
implied.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from . import schema as sch
from .logging_utils import get_logger

log = get_logger(__name__)

__all__ = [
    "ClassMetrics", "BinaryMetrics", "ThresholdSweep", "EvaluationReport",
    "confusion_matrix", "log_loss", "roc_auc_binary", "roc_auc_ovr",
    "average_precision_binary", "average_precision_ovr", "matthews_corrcoef",
    "cohen_kappa", "top_k_accuracy", "expected_calibration_error",
    "threshold_sweep", "evaluate", "compare", "stratified_folds",
    "stratified_subsample",
]

#: Guard for ``log(0)``. 1e-15 is what scikit-learn uses, so log-loss values here are
#: directly comparable with numbers produced by ``sklearn.metrics.log_loss``.
EPS = 1e-15


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------

def _as_labels(y: Any) -> np.ndarray:
    return np.asarray(y, dtype=np.int64).ravel()


def _as_proba(proba: Any, n_classes: int) -> np.ndarray:
    """Validate and normalise a probability matrix."""
    arr = np.asarray(proba, dtype=np.float64)
    if arr.ndim == 1:
        raise ValueError(
            "expected a (n_rows, n_classes) probability matrix, got a 1-D array. If "
            "these are hard predictions, use one-hot columns or pass them as y_pred."
        )
    if arr.shape[1] != n_classes:
        raise ValueError(
            f"probability matrix has {arr.shape[1]} columns but {n_classes} classes "
            "were declared. A model that saw fewer classes at fit time must expand its "
            "columns back to the full class order (ShieldModel._expand_proba does this)."
        )
    if not np.isfinite(arr).all():
        raise ValueError("probability matrix contains NaN or inf")
    if (arr < -1e-9).any():
        raise ValueError("probability matrix contains negative values")
    arr = np.clip(arr, 0.0, None)
    totals = arr.sum(axis=1, keepdims=True)
    bad = (totals <= 0).ravel()
    if bad.any():
        # An all-zero row means the model assigned no probability to anything, which
        # happens with a degenerate tree. Uniform is the honest stand-in.
        arr[bad] = 1.0 / n_classes
        totals[bad] = 1.0
        log.warning("%d prediction row(s) summed to zero and were replaced with a "
                    "uniform distribution", int(bad.sum()))
    return arr / totals


def confusion_matrix(y_true: Any, y_pred: Any, n_classes: int) -> np.ndarray:
    """Rows are true classes, columns are predicted classes.

    Built with ``np.bincount`` on the flattened index rather than a Python loop: on a
    60k-row test split the loop version costs seconds and this costs microseconds.
    """
    t, p = _as_labels(y_true), _as_labels(y_pred)
    if t.shape != p.shape:
        raise ValueError(f"y_true has {t.size} rows, y_pred has {p.size}")
    if t.size == 0:
        return np.zeros((n_classes, n_classes), dtype=np.int64)
    for name, arr in (("y_true", t), ("y_pred", p)):
        if arr.size and (arr.min() < 0 or arr.max() >= n_classes):
            raise ValueError(
                f"{name} contains class index {int(arr.min())}..{int(arr.max())} "
                f"outside [0, {n_classes})"
            )
    flat = np.bincount(t * n_classes + p, minlength=n_classes * n_classes)
    return flat.reshape(n_classes, n_classes).astype(np.int64)


def log_loss(y_true: Any, proba: Any, n_classes: Optional[int] = None) -> float:
    """Multi-class cross-entropy, averaged over rows."""
    t = _as_labels(y_true)
    arr = np.asarray(proba, dtype=np.float64)
    n_classes = n_classes or arr.shape[1]
    arr = _as_proba(arr, n_classes)
    if t.size == 0:
        return float("nan")
    picked = arr[np.arange(t.size), t]
    return float(-np.log(np.clip(picked, EPS, 1.0)).mean())


def roc_auc_binary(y_true: Any, score: Any) -> float:
    """One-vs-rest ROC AUC via the Mann-Whitney U statistic.

    Rank-based rather than trapezoidal: it is exact, it needs no threshold grid, and
    ``scipy.stats.rankdata``-style average ranks give ties the 0.5 credit they deserve.
    A trapezoid over unique thresholds gets the same answer but is easy to get subtly
    wrong when many rows share a score - which is the norm for tree ensembles, where
    every row in a leaf has an identical probability.
    """
    y = _as_labels(y_true).astype(np.float64)
    s = np.asarray(score, dtype=np.float64).ravel()
    n_pos = float(y.sum())
    n_neg = float(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")            # undefined with one class present
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    # Average ranks within tied groups.
    ranks = np.empty(s.size, dtype=np.float64)
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0     # 1-based average rank
        i = j + 1
    rank_of = np.empty(s.size, dtype=np.float64)
    rank_of[order] = ranks
    u = rank_of[y == 1].sum() - n_pos * (n_pos + 1.0) / 2.0
    return float(u / (n_pos * n_neg))


def average_precision_binary(y_true: Any, score: Any) -> float:
    """Area under the precision-recall curve, computed as sklearn's average precision.

    ``sum_n (R_n - R_{n-1}) * P_n`` - a step-wise sum, deliberately not a trapezoid.
    Interpolating precision between thresholds overstates the area, which is why
    ``sklearn.metrics.average_precision_score`` does not do it either.
    """
    y = _as_labels(y_true)
    s = np.asarray(score, dtype=np.float64).ravel()
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == y.size:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    # Only evaluate at the end of each tie group: a threshold cannot separate rows that
    # share a score, so intermediate points are not achievable operating points.
    last = np.r_[np.nonzero(np.diff(s_sorted))[0], s_sorted.size - 1]
    tp, fp = tp[last], fp[last]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    d_recall = np.diff(np.r_[0.0, recall])
    return float((d_recall * precision).sum())


def _ovr(metric, y_true: Any, proba: Any, n_classes: int) -> np.ndarray:
    """Apply a binary metric one-vs-rest, ``nan`` where a class is absent."""
    t = _as_labels(y_true)
    arr = _as_proba(proba, n_classes)
    out = np.full(n_classes, np.nan)
    for k in range(n_classes):
        out[k] = metric((t == k).astype(np.int64), arr[:, k])
    return out


def roc_auc_ovr(y_true: Any, proba: Any, n_classes: int) -> np.ndarray:
    """Per-class one-vs-rest ROC AUC."""
    return _ovr(roc_auc_binary, y_true, proba, n_classes)


def average_precision_ovr(y_true: Any, proba: Any, n_classes: int) -> np.ndarray:
    """Per-class one-vs-rest average precision.

    Reported alongside ROC AUC because ROC AUC is optimistic under extreme imbalance:
    with 21 positives against 300k negatives, the false-positive rate barely moves no
    matter how many false alarms are raised, so ROC AUC stays near 1.0 while precision
    is catastrophic. Average precision reacts to exactly that failure.
    """
    return _ovr(average_precision_binary, y_true, proba, n_classes)


def matthews_corrcoef(cm: np.ndarray) -> float:
    """Multi-class Matthews correlation coefficient from a confusion matrix.

    Included because it is the one scalar here that cannot be gamed by ignoring a rare
    class: it uses all four cells of every class's contingency table, so predicting the
    majority everywhere gives 0.0, not 0.5.
    """
    c = np.asarray(cm, dtype=np.float64)
    total = c.sum()
    if total == 0:
        return float("nan")
    correct = np.trace(c)
    pred = c.sum(axis=0)
    true = c.sum(axis=1)
    numerator = correct * total - float(pred @ true)
    denominator = math.sqrt(max(total ** 2 - float(pred @ pred), 0.0)) * \
        math.sqrt(max(total ** 2 - float(true @ true), 0.0))
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def cohen_kappa(cm: np.ndarray) -> float:
    """Agreement corrected for chance."""
    c = np.asarray(cm, dtype=np.float64)
    total = c.sum()
    if total == 0:
        return float("nan")
    observed = np.trace(c) / total
    expected = float(c.sum(axis=0) @ c.sum(axis=1)) / (total ** 2)
    if abs(1.0 - expected) < 1e-12:
        return 0.0
    return float((observed - expected) / (1.0 - expected))


def top_k_accuracy(y_true: Any, proba: Any, k: int = 3) -> float:
    """Fraction of rows whose true class is among the *k* highest probabilities.

    Useful in the write-up for the confusable pairs: if top-1 is 0.86 but top-2 is 0.99,
    the model is not confused about whether traffic is malicious, only about which of
    two similar attack variants it is - a much milder failure than the top-1 number
    suggests, and worth saying explicitly.
    """
    t = _as_labels(y_true)
    arr = np.asarray(proba, dtype=np.float64)
    if t.size == 0:
        return float("nan")
    k = int(min(max(k, 1), arr.shape[1]))
    # argpartition is O(n) against argsort's O(n log n); only the top-k membership
    # matters, not their order.
    top = np.argpartition(-arr, k - 1, axis=1)[:, :k]
    return float((top == t[:, None]).any(axis=1).mean())


def expected_calibration_error(y_true: Any, proba: Any, bins: int = 10) -> float:
    """Weighted gap between confidence and accuracy over equal-width bins.

    Answers "when the model says 90%, is it right 90% of the time?". It matters for the
    app: the confidence bar is shown to a human who will trust it, and boosted trees are
    systematically overconfident, so a large ECE is a caveat the report should carry.
    """
    t = _as_labels(y_true)
    arr = np.asarray(proba, dtype=np.float64)
    if t.size == 0:
        return float("nan")
    confidence = arr.max(axis=1)
    correct = (arr.argmax(axis=1) == t).astype(np.float64)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    # -1 so the smallest confidence lands in bin 0 rather than bin -1.
    which = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, bins - 1)
    error = 0.0
    for b in range(bins):
        mask = which == b
        if not mask.any():
            continue
        error += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return float(error)


# ---------------------------------------------------------------------------
# per-class and binary views
# ---------------------------------------------------------------------------

@dataclass
class ClassMetrics:
    """One row of the per-class report."""

    index: int
    name: str
    support: int                  # rows of this class in y_true
    predicted: int                # rows predicted as this class
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    roc_auc: float
    average_precision: float
    confused_with: str = ""       # class that absorbed most of this one's misses
    confused_count: int = 0

    @property
    def missed(self) -> int:
        return int(self.fn)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index, "name": self.name, "support": self.support,
            "predicted": self.predicted, "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
            "roc_auc": self.roc_auc, "average_precision": self.average_precision,
            "confused_with": self.confused_with,
            "confused_count": self.confused_count,
        }


@dataclass
class BinaryMetrics:
    """The attack-vs-benign collapse, which is how an IDS is judged operationally.

    A model can be excellent at telling DoS Hulk from DoS GoldenEye and still be useless
    if it lets attacks through as benign. Collapsing all 12 attack classes into one
    positive class isolates that question, and gives numbers comparable with the binary
    intrusion-detection literature, which is most of it.
    """

    tp: int
    fp: int
    fn: int
    tn: int
    accuracy: float
    precision: float
    recall: float                 # detection rate
    f1: float
    false_alarm_rate: float       # FP / (FP + TN)
    miss_rate: float              # FN / (FN + TP)
    roc_auc: float
    average_precision: float
    benign_class: str = sch.BENIGN_LABEL

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    def render(self) -> str:
        return "\n".join([
            f"attack vs {self.benign_class.lower()} (all attack classes merged)",
            f"  detection rate (recall) {self.recall:8.4f}"
            f"     missed attacks {self.fn:>9,}",
            f"  precision               {self.precision:8.4f}"
            f"     false alarms   {self.fp:>9,}",
            f"  F1                      {self.f1:8.4f}"
            f"     ROC AUC        {self.roc_auc:9.4f}",
            f"  false alarm rate        {self.false_alarm_rate:8.4f}"
            f"     PR AUC         {self.average_precision:9.4f}",
        ])


@dataclass
class ThresholdSweep:
    """Operating points of the binary detector, and three ways to pick one.

    The argmax rule shown everywhere else in this project implies a threshold of 0.5 on
    the attack probability, which is an arbitrary choice inherited from the loss
    function, not an operational one. A SOC with capacity for 100 alerts a day wants the
    threshold that maximises detection *subject to* a false-alarm budget, and that is
    almost never 0.5. This makes the trade-off explicit and gives the app a slider with
    real meaning behind it.
    """

    thresholds: np.ndarray
    tpr: np.ndarray
    fpr: np.ndarray
    precision: np.ndarray
    f1: np.ndarray
    best_f1_threshold: float
    best_f1: float
    youden_threshold: float       # maximises TPR - FPR, the balanced choice
    budget_threshold: Optional[float] = None   # best TPR within the FPR budget
    budget_fpr: Optional[float] = None
    budget_tpr: Optional[float] = None

    def at(self, threshold: float) -> Dict[str, float]:
        """Interpolate the curve at an arbitrary threshold."""
        if self.thresholds.size == 0:
            return {}
        # thresholds run high -> low, so search the reversed array.
        idx = int(np.searchsorted(-self.thresholds, -float(threshold), side="left"))
        idx = min(max(idx, 0), self.thresholds.size - 1)
        return {
            "threshold": float(self.thresholds[idx]),
            "tpr": float(self.tpr[idx]),
            "fpr": float(self.fpr[idx]),
            "precision": float(self.precision[idx]),
            "f1": float(self.f1[idx]),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_f1_threshold": self.best_f1_threshold,
            "best_f1": self.best_f1,
            "youden_threshold": self.youden_threshold,
            "budget_threshold": self.budget_threshold,
            "budget_fpr": self.budget_fpr,
            "budget_tpr": self.budget_tpr,
            "points": int(self.thresholds.size),
        }

    def render(self, points: int = 9) -> str:
        rows = ["threshold      TPR      FPR   precision       F1"]
        if self.thresholds.size == 0:
            return "\n".join(rows + ["  (no operating points)"])
        picks = np.unique(np.linspace(0, self.thresholds.size - 1, points).astype(int))
        for i in picks:
            rows.append(f"{self.thresholds[i]:9.4f} {self.tpr[i]:8.4f} "
                        f"{self.fpr[i]:8.4f} {self.precision[i]:11.4f} "
                        f"{self.f1[i]:8.4f}")
        rows.append(f"  best F1 at {self.best_f1_threshold:.4f} (F1 {self.best_f1:.4f}); "
                    f"Youden J at {self.youden_threshold:.4f}")
        if self.budget_threshold is not None:
            rows.append(f"  within the false-alarm budget: threshold "
                        f"{self.budget_threshold:.4f} -> detection "
                        f"{self.budget_tpr:.4f} at FPR {self.budget_fpr:.4f}")
        return "\n".join(rows)


def threshold_sweep(
    y_attack: Any,
    p_attack: Any,
    *,
    fpr_budget: Optional[float] = 0.01,
    max_points: int = 512,
) -> ThresholdSweep:
    """Sweep the decision threshold of the binary attack detector."""
    y = _as_labels(y_attack)
    p = np.asarray(p_attack, dtype=np.float64).ravel()
    n_pos, n_neg = int(y.sum()), int(y.size - y.sum())
    empty = np.zeros(0)
    if n_pos == 0 or n_neg == 0:
        return ThresholdSweep(empty, empty, empty, empty, empty,
                              float("nan"), float("nan"), float("nan"))

    order = np.argsort(-p, kind="mergesort")
    y_sorted, p_sorted = y[order], p[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    last = np.r_[np.nonzero(np.diff(p_sorted))[0], p_sorted.size - 1]
    thresholds, tp, fp = p_sorted[last], tp[last], fp[last]

    if thresholds.size > max_points:
        # Keep the extremes and thin the middle: a 300k-row sweep has 300k operating
        # points, which is far more resolution than any decision needs and makes the
        # manifest enormous.
        keep = np.unique(np.r_[
            np.linspace(0, thresholds.size - 1, max_points - 2).astype(int), 0,
            thresholds.size - 1])
        thresholds, tp, fp = thresholds[keep], tp[keep], fp[keep]

    fn = n_pos - tp
    tpr = tp / n_pos
    fpr = fp / n_neg
    precision = tp / np.maximum(tp + fp, 1)
    f1 = 2.0 * tp / np.maximum(2.0 * tp + fp + fn, 1)

    best = int(np.argmax(f1))
    youden = int(np.argmax(tpr - fpr))
    sweep = ThresholdSweep(
        thresholds=thresholds, tpr=tpr, fpr=fpr, precision=precision, f1=f1,
        best_f1_threshold=float(thresholds[best]), best_f1=float(f1[best]),
        youden_threshold=float(thresholds[youden]),
    )
    if fpr_budget is not None:
        within = np.nonzero(fpr <= float(fpr_budget))[0]
        if within.size:
            pick = int(within[np.argmax(tpr[within])])
            sweep.budget_threshold = float(thresholds[pick])
            sweep.budget_fpr = float(fpr[pick])
            sweep.budget_tpr = float(tpr[pick])
    return sweep


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

@dataclass
class EvaluationReport:
    """Everything measured about one model on one split."""

    model: str
    split: str
    class_names: List[str]
    n_rows: int
    accuracy: float
    balanced_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    log_loss: float
    mcc: float
    kappa: float
    top2_accuracy: float
    top3_accuracy: float
    calibration_error: float
    macro_roc_auc: float
    macro_average_precision: float
    classes_evaluated: int
    classes_absent: List[str]
    classes_never_predicted: List[str]
    per_class: List[ClassMetrics]
    confusion: np.ndarray
    binary: Optional[BinaryMetrics] = None
    sweep: Optional[ThresholdSweep] = None
    fit_seconds: Optional[float] = None
    predict_seconds: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    # -- convenience ---------------------------------------------------------

    @property
    def selection_metric(self) -> float:
        """The single number ``train.py`` ranks models by."""
        return self.macro_f1

    def by_name(self, name: str) -> Optional[ClassMetrics]:
        target = sch.canonical_label(name)
        for row in self.per_class:
            if row.name == target:
                return row
        return None

    def worst_classes(self, n: int = 3) -> List[ClassMetrics]:
        """Lowest-recall classes that are actually present - the report's failure list."""
        present = [c for c in self.per_class if c.support > 0]
        return sorted(present, key=lambda c: (c.recall, c.support))[:n]

    # -- serialisation -------------------------------------------------------

    def to_dict(self, *, include_confusion: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "split": self.split,
            "n_rows": self.n_rows,
            "class_names": list(self.class_names),
            "overall": {
                "accuracy": self.accuracy,
                "balanced_accuracy": self.balanced_accuracy,
                "macro_precision": self.macro_precision,
                "macro_recall": self.macro_recall,
                "macro_f1": self.macro_f1,
                "weighted_precision": self.weighted_precision,
                "weighted_recall": self.weighted_recall,
                "weighted_f1": self.weighted_f1,
                "log_loss": self.log_loss,
                "mcc": self.mcc,
                "cohen_kappa": self.kappa,
                "top2_accuracy": self.top2_accuracy,
                "top3_accuracy": self.top3_accuracy,
                "calibration_error": self.calibration_error,
                "macro_roc_auc": self.macro_roc_auc,
                "macro_average_precision": self.macro_average_precision,
            },
            "coverage": {
                "classes_evaluated": self.classes_evaluated,
                "classes_absent": list(self.classes_absent),
                "classes_never_predicted": list(self.classes_never_predicted),
            },
            "per_class": [c.to_dict() for c in self.per_class],
            "timing": {"fit_seconds": self.fit_seconds,
                       "predict_seconds": self.predict_seconds},
            "notes": list(self.notes),
        }
        if self.binary is not None:
            payload["binary"] = self.binary.to_dict()
        if self.sweep is not None:
            payload["threshold_sweep"] = self.sweep.to_dict()
        if include_confusion:
            payload["confusion"] = self.confusion.tolist()
        return payload

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=float)

    def confusion_frame(self):
        """The confusion matrix as a labelled DataFrame, for the app and notebooks."""
        import pandas as pd
        return pd.DataFrame(self.confusion,
                            index=[f"true:{c}" for c in self.class_names],
                            columns=[f"pred:{c}" for c in self.class_names])

    def per_class_frame(self):
        import pandas as pd
        return pd.DataFrame([c.to_dict() for c in self.per_class])

    # -- text output ---------------------------------------------------------

    def render(self, *, confusion: bool = False, sweep: bool = False) -> str:
        head = f"{self.model} on {self.split} ({self.n_rows:,} rows)"
        lines = [head, "=" * len(head), ""]
        lines.append(
            f"  macro F1            {self.macro_f1:8.4f}"
            f"      accuracy             {self.accuracy:8.4f}")
        lines.append(
            f"  macro recall        {self.macro_recall:8.4f}"
            f"      balanced accuracy    {self.balanced_accuracy:8.4f}")
        lines.append(
            f"  macro precision     {self.macro_precision:8.4f}"
            f"      weighted F1          {self.weighted_f1:8.4f}")
        lines.append(
            f"  MCC                 {self.mcc:8.4f}"
            f"      Cohen kappa          {self.kappa:8.4f}")
        lines.append(
            f"  log loss            {self.log_loss:8.4f}"
            f"      calibration error    {self.calibration_error:8.4f}")
        lines.append(
            f"  macro ROC AUC       {self.macro_roc_auc:8.4f}"
            f"      macro PR AUC         {self.macro_average_precision:8.4f}")
        lines.append(
            f"  top-2 accuracy      {self.top2_accuracy:8.4f}"
            f"      top-3 accuracy       {self.top3_accuracy:8.4f}")
        if self.fit_seconds is not None:
            lines.append(f"  fit {self.fit_seconds:.1f}s"
                         f"   predict {self.predict_seconds or 0.0:.2f}s")
        lines.append("")
        lines.append(f"{'class':<28}{'support':>9}{'prec':>8}{'recall':>8}"
                     f"{'F1':>8}{'PR AUC':>9}  most confused with")
        lines.append("-" * 100)
        for row in self.per_class:
            if row.support == 0:
                lines.append(f"{row.name:<28}{0:>9}{'':>8}{'':>8}{'':>8}{'':>9}"
                             "  absent from this split")
                continue
            auc = "     n/a" if math.isnan(row.average_precision) \
                else f"{row.average_precision:9.4f}"
            confused = (f"  {row.confused_with} ({row.confused_count:,})"
                        if row.confused_count else "  -")
            lines.append(f"{row.name:<28}{row.support:>9,}{row.precision:>8.4f}"
                         f"{row.recall:>8.4f}{row.f1:>8.4f}{auc}{confused}")
        lines.append("-" * 100)
        lines.append(f"averaged over {self.classes_evaluated} class(es) present in "
                     f"{self.split}")
        if self.classes_absent:
            lines.append(f"absent from this split: {', '.join(self.classes_absent)}")
        if self.classes_never_predicted:
            lines.append(f"never predicted at all: "
                         f"{', '.join(self.classes_never_predicted)}")
        if self.binary is not None:
            lines.append("")
            lines.append(self.binary.render())
        if sweep and self.sweep is not None:
            lines.append("")
            lines.append(self.sweep.render())
        if confusion:
            lines.append("")
            lines.append(self.render_confusion())
        for note in self.notes:
            lines.append(f"note: {note}")
        return "\n".join(lines)

    def render_confusion(self, *, width: int = 7) -> str:
        """Compact confusion matrix with abbreviated headers."""
        short = [_abbreviate(c, width - 1) for c in self.class_names]
        header = " " * 24 + "".join(f"{s:>{width}}" for s in short)
        lines = [header, " " * 24 + "-" * (width * len(short))]
        for i, name in enumerate(self.class_names):
            cells = "".join(_compact(int(v), width) for v in self.confusion[i])
            lines.append(f"{_abbreviate(name, 22):<24}{cells}")
        lines.append("")
        lines.append("rows = true class, columns = predicted; "
                     f"column keys: {', '.join(f'{s}={c}' for s, c in zip(short, self.class_names))}")
        return "\n".join(lines)

    def __str__(self) -> str:                       # pragma: no cover
        return self.render()


def _abbreviate(name: str, width: int) -> str:
    if len(name) <= width:
        return name
    # Drop vowels from the tail before truncating: "Web Attack - Brute Force" becomes
    # something still recognisable rather than "Web Att".
    trimmed = name.replace("Web Attack - ", "WA:").replace("Attack", "Atk")
    if len(trimmed) <= width:
        return trimmed
    return trimmed[:width]


def _compact(value: int, width: int) -> str:
    """Render a count in at most ``width`` characters."""
    if value == 0:
        return f"{'.':>{width}}"
    if value < 100_000:
        return f"{value:>{width},}" if len(f"{value:,}") <= width else f"{value:>{width}}"
    return f"{value / 1000:>{width - 1}.0f}k"


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------

def evaluate(
    y_true: Any,
    proba: Any,
    *,
    class_names: Sequence[str],
    model: str = "model",
    split: str = "test",
    fit_seconds: Optional[float] = None,
    predict_seconds: Optional[float] = None,
    fpr_budget: Optional[float] = 0.01,
    benign_label: str = sch.BENIGN_LABEL,
    curves: bool = True,
) -> EvaluationReport:
    """Score one model's probability output against the truth.

    Probabilities rather than hard labels, always: log loss, ROC AUC, average precision,
    calibration and the threshold sweep all need them, and deriving labels from
    probabilities is free while the reverse is impossible.

    ``curves=False`` skips the AUC and sweep computation. Worth it inside a tuning loop,
    where the objective is log loss and the per-class ROC AUCs are 13 rank sorts of
    300k rows that nobody looks at.
    """
    names = [str(c) for c in class_names]
    n_classes = len(names)
    if n_classes < 2:
        raise ValueError(f"need at least 2 classes to evaluate, got {n_classes}")

    t = _as_labels(y_true)
    arr = _as_proba(proba, n_classes)
    if arr.shape[0] != t.size:
        raise ValueError(f"{t.size} true labels but {arr.shape[0]} prediction rows")
    y_pred = arr.argmax(axis=1)

    cm = confusion_matrix(t, y_pred, n_classes)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm).astype(np.int64)
    fp = predicted - tp
    fn = support - tp

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / np.maximum(predicted, 1), 0.0)
        recall = np.where(support > 0, tp / np.maximum(support, 1), 0.0)
        f1 = np.where(precision + recall > 0,
                      2 * precision * recall / np.maximum(precision + recall, EPS), 0.0)

    if curves:
        auc = roc_auc_ovr(t, arr, n_classes)
        ap = average_precision_ovr(t, arr, n_classes)
    else:
        auc = np.full(n_classes, np.nan)
        ap = np.full(n_classes, np.nan)

    present = support > 0
    n_present = int(present.sum())
    if n_present == 0:
        raise ValueError("y_true is empty - nothing to evaluate")

    weights = support[present].astype(np.float64)
    weights = weights / weights.sum()

    per_class: List[ClassMetrics] = []
    for k in range(n_classes):
        # The off-diagonal cell that absorbed most of this class's misses. This single
        # column is what turns "recall is 0.61" into "61% detected, the rest went to DoS
        # GoldenEye" - an actionable statement about which pair is confusable.
        confused_with, confused_count = "", 0
        if support[k] > 0 and fn[k] > 0:
            row = cm[k].copy()
            row[k] = 0
            j = int(row.argmax())
            if row[j] > 0:
                confused_with, confused_count = names[j], int(row[j])
        per_class.append(ClassMetrics(
            index=k, name=names[k], support=int(support[k]),
            predicted=int(predicted[k]), tp=int(tp[k]), fp=int(fp[k]), fn=int(fn[k]),
            precision=float(precision[k]), recall=float(recall[k]), f1=float(f1[k]),
            roc_auc=float(auc[k]), average_precision=float(ap[k]),
            confused_with=confused_with, confused_count=confused_count,
        ))

    notes: List[str] = []
    absent = [names[k] for k in range(n_classes) if support[k] == 0]
    never = [names[k] for k in range(n_classes)
             if predicted[k] == 0 and support[k] > 0]
    if never:
        notes.append(
            f"{len(never)} class(es) present in the data were never predicted "
            f"({', '.join(never)}); their recall is 0 and macro F1 already reflects it")
    if absent:
        notes.append(
            f"{len(absent)} class(es) are absent from {split} and were excluded from "
            "the macro averages, so this macro F1 is over "
            f"{n_present} of {n_classes} classes")

    report = EvaluationReport(
        model=model,
        split=split,
        class_names=names,
        n_rows=int(t.size),
        accuracy=float((y_pred == t).mean()),
        balanced_accuracy=float(recall[present].mean()),
        macro_precision=float(precision[present].mean()),
        macro_recall=float(recall[present].mean()),
        macro_f1=float(f1[present].mean()),
        weighted_precision=float(precision[present] @ weights),
        weighted_recall=float(recall[present] @ weights),
        weighted_f1=float(f1[present] @ weights),
        log_loss=log_loss(t, arr, n_classes),
        mcc=matthews_corrcoef(cm),
        kappa=cohen_kappa(cm),
        top2_accuracy=top_k_accuracy(t, arr, 2),
        top3_accuracy=top_k_accuracy(t, arr, 3),
        calibration_error=expected_calibration_error(t, arr),
        macro_roc_auc=_nanmean(auc[present]),
        macro_average_precision=_nanmean(ap[present]),
        classes_evaluated=n_present,
        classes_absent=absent,
        classes_never_predicted=never,
        per_class=per_class,
        confusion=cm,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        notes=notes,
    )

    benign = _benign_index(names, benign_label)
    if benign is None:
        report.notes.append(
            f"no {benign_label!r} class found among {names}, so the binary "
            "attack-vs-benign collapse was skipped")
    else:
        y_attack = (t != benign).astype(np.int64)
        p_attack = 1.0 - arr[:, benign]
        pred_attack = (y_pred != benign).astype(np.int64)
        b_tp = int(((y_attack == 1) & (pred_attack == 1)).sum())
        b_fp = int(((y_attack == 0) & (pred_attack == 1)).sum())
        b_fn = int(((y_attack == 1) & (pred_attack == 0)).sum())
        b_tn = int(((y_attack == 0) & (pred_attack == 0)).sum())
        prec = b_tp / max(b_tp + b_fp, 1)
        rec = b_tp / max(b_tp + b_fn, 1)
        report.binary = BinaryMetrics(
            tp=b_tp, fp=b_fp, fn=b_fn, tn=b_tn,
            accuracy=(b_tp + b_tn) / max(t.size, 1),
            precision=prec, recall=rec,
            f1=(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0,
            false_alarm_rate=b_fp / max(b_fp + b_tn, 1),
            miss_rate=b_fn / max(b_fn + b_tp, 1),
            roc_auc=roc_auc_binary(y_attack, p_attack) if curves else float("nan"),
            average_precision=(average_precision_binary(y_attack, p_attack)
                               if curves else float("nan")),
            benign_class=names[benign],
        )
        if curves:
            report.sweep = threshold_sweep(y_attack, p_attack, fpr_budget=fpr_budget)
    return report


def _nanmean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else float("nan")


def _benign_index(names: Sequence[str], benign_label: str) -> Optional[int]:
    target = str(benign_label).strip().upper()
    for i, name in enumerate(names):
        if str(name).strip().upper() == target:
            return i
    return None


# ---------------------------------------------------------------------------
# cross-model comparison
# ---------------------------------------------------------------------------

def compare(
    reports: Sequence[EvaluationReport],
    *,
    sort_by: str = "macro_f1",
    ascending: bool = False,
) -> str:
    """Leaderboard table over several models on the same split."""
    if not reports:
        return "(no models were evaluated)"
    splits = {r.split for r in reports}
    if len(splits) > 1:
        log.warning("comparing reports from different splits (%s); the numbers are not "
                    "on the same footing", ", ".join(sorted(splits)))

    def key(r: EvaluationReport) -> float:
        value = getattr(r, sort_by, float("nan"))
        return -1e18 if (isinstance(value, float) and math.isnan(value)) else value

    ordered = sorted(reports, key=key, reverse=not ascending)
    head = (f"{'#':<3}{'model':<22}{'macro F1':>10}{'accuracy':>10}{'macro rec':>11}"
            f"{'log loss':>10}{'MCC':>8}{'detect':>8}{'false al':>10}{'fit s':>8}")
    lines = [head, "-" * len(head)]
    for rank, r in enumerate(ordered, 1):
        detect = f"{r.binary.recall:8.4f}" if r.binary else "     n/a"
        alarm = f"{r.binary.false_alarm_rate:10.5f}" if r.binary else "       n/a"
        fit = f"{r.fit_seconds:8.1f}" if r.fit_seconds is not None else "     n/a"
        lines.append(f"{rank:<3}{r.model:<22}{r.macro_f1:>10.4f}{r.accuracy:>10.4f}"
                     f"{r.macro_recall:>11.4f}{r.log_loss:>10.4f}{r.mcc:>8.4f}"
                     f"{detect}{alarm}{fit}")
    lines.append("-" * len(head))
    lines.append(f"ranked by {sort_by}; 'detect' is the attack detection rate and "
                 "'false al' the fraction of benign traffic flagged")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# cross-validation folds
# ---------------------------------------------------------------------------

def stratified_folds(
    y: Any,
    n_splits: int = 3,
    *,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Stratified K-fold indices, tolerant of classes smaller than ``n_splits``.

    ``sklearn.model_selection.StratifiedKFold`` warns and degrades when a class has
    fewer members than there are folds - with 21 SQL-injection flows and 5 folds that is
    fine, but a 3-row class is not. This assigns each class's shuffled rows round-robin
    across folds, so a 2-row class appears in exactly 2 validation folds and is simply
    absent from the third. Absent is safe: ``ShieldModel._densify`` compresses the label
    space per fit and ``evaluate`` excludes absent classes from the macro average.
    """
    labels = _as_labels(y)
    n_splits = int(n_splits)
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    if labels.size < n_splits:
        raise ValueError(f"cannot make {n_splits} folds from {labels.size} rows")

    rng = np.random.default_rng(seed)
    assignment = np.empty(labels.size, dtype=np.int64)
    for cls in np.unique(labels):
        idx = np.nonzero(labels == cls)[0]
        rng.shuffle(idx)
        # Rotating the starting fold per class stops every small class from piling into
        # fold 0, which would leave fold 0 with all the rare classes and the rest with
        # none.
        offset = int(rng.integers(n_splits))
        assignment[idx] = (np.arange(idx.size) + offset) % n_splits

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for f in range(n_splits):
        val = np.nonzero(assignment == f)[0]
        train = np.nonzero(assignment != f)[0]
        if val.size == 0 or train.size == 0:
            raise ValueError(f"fold {f} came out empty; use fewer splits")
        if np.unique(labels[train]).size < 2:
            raise ValueError(
                f"fold {f} leaves fewer than 2 classes in the training half; the data "
                "is too small or too skewed for this many folds")
        folds.append((train, val))

    sizes = ", ".join(f"{len(v):,}" for _, v in folds)
    log.debug("built %d stratified fold(s), validation sizes: %s", n_splits, sizes)
    return folds


def stratified_subsample(
    y: Any,
    max_rows: int,
    rng: Optional[np.random.Generator] = None,
    *,
    floor: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """Draw at most *max_rows* row indices, capping large classes and keeping small ones.

    Used wherever a cost ceiling has to be imposed on an operation that must still see
    every class: hyper-parameter search in :mod:`shieldnet.tune` and explanation in
    :mod:`shieldnet.explain`. It lives here rather than in either of them because two
    copies of this arithmetic would silently drift apart, and the failure mode is subtle -
    a rare class quietly reduced to a handful of rows produces metrics that look fine and
    mean nothing.

    Each class is guaranteed ``min(count, floor)`` rows before anything is distributed
    proportionally. Without that guarantee a purely proportional draw takes a 21-row class
    down to 5 rows, whose validation slice is 1 row - and a hyper-parameter search decided
    by a single row is decided by noise.

    The floor first shrinks to ``max_rows // n_classes``, so it can never overshoot the
    budget: a 100-row budget over 13 classes guarantees 7 rows each rather than 200 each.
    The returned index count is therefore always ``<= max_rows``, and lands exactly on it
    whenever the data is large enough - the rounding remainder is handed to the largest
    classes rather than dropped.

    The result is shuffled, so callers may slice it without reintroducing class order.
    """
    labels = np.asarray(y).ravel()
    if labels.size == 0:
        return np.empty(0, dtype=np.int64)
    max_rows = int(max_rows)
    if max_rows <= 0:
        raise ValueError(f"max_rows must be positive, got {max_rows}")
    if rng is None:
        rng = np.random.default_rng(seed)
    if labels.size <= max_rows:
        keep = np.arange(labels.size)
        rng.shuffle(keep)
        return keep

    classes, counts = np.unique(labels, return_counts=True)
    counts = counts.astype(np.int64)
    # Never promise more per class than an equal share of the budget.
    effective_floor = int(min(int(floor), max(1, max_rows // max(classes.size, 1))))
    floors = np.minimum(counts, effective_floor)

    if floors.sum() >= max_rows:
        targets = floors
        log.debug("an equal share of the %s-row budget is %d row(s) per class, which "
                  "accounts for the whole budget; every class gets its share and nothing "
                  "is distributed proportionally", f"{max_rows:,}", effective_floor)
    else:
        headroom = counts - floors                     # rows available above the floor
        budget = max_rows - int(floors.sum())
        total_headroom = int(headroom.sum())
        if total_headroom <= budget:
            targets = counts                           # everything fits
        else:
            extra = np.floor(headroom * (budget / total_headroom)).astype(np.int64)
            # Hand out the rounding remainder to the largest classes, largest first, so
            # the total lands exactly on the budget instead of a few rows under it.
            shortfall = budget - int(extra.sum())
            if shortfall > 0:
                room = headroom - extra
                for idx in np.argsort(-headroom):
                    if shortfall == 0:
                        break
                    give = int(min(shortfall, room[idx]))
                    extra[idx] += give
                    shortfall -= give
            targets = floors + extra

    picked: List[np.ndarray] = []
    for cls, target in zip(classes, targets):
        idx = np.nonzero(labels == cls)[0]
        picked.append(idx if target >= idx.size
                      else rng.choice(idx, size=int(target), replace=False))
    keep = np.concatenate(picked)
    rng.shuffle(keep)
    return keep
