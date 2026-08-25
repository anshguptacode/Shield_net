"""Class-imbalance handling for the training split only.

The imbalance after chunking is 150,000 BENIGN against 68 Rare Attacks rows - 2,206:1 in
a 304,616-row frame. Left alone, a model that predicts BENIGN for everything scores about
49% accuracy and exactly 1/13 = 0.077 macro recall, which is why this project reports
macro metrics and why this module exists.

Three rules that are easy to get wrong
--------------------------------------
**Resample the training split and nothing else.** Oversampling validation or test data
invents rows and then scores the model on its own inventions. The number that comes out
is not reproducible on real traffic. Every function here takes only ``X_train``.

**Bound the expansion, not just the ratio.** "Bring every class up to 25% of the
majority" sounds reasonable until you put the real counts in. The 70% train split holds
105,000 BENIGN rows and 47 Rare Attacks rows, so 25% of the majority is 26,250 - which
means synthesising 26,203 rows from 47 real ones, a 559x expansion. Those rows all lie on
the line segments between 47 points, so the model learns a 47-point skeleton very
confidently and generalises worse than if you had left the class small. ``max_expansion``
caps the multiplier: at 20x the class stops at 940 rows instead.

**Reduce k when the class is tiny.** SMOTE needs k+1 members to interpolate. A class
with 4 rows and the default k=5 is an immediate ``ValueError`` deep inside imblearn -
a genuinely common way for a CICIDS2017 pipeline to die twenty minutes into a run.
``k`` is reduced per class here, and a class with one row is duplicated with a note
rather than crashing.

SMOTE is implemented in numpy as a fallback so the stage works without imblearn. The
implementation is the original Chawla et al. (2002) algorithm: pick a minority row, pick
one of its k in-class nearest neighbours, and place a new point uniformly at random on
the segment between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .logging_utils import get_logger, human_count, stage

log = get_logger(__name__)

__all__ = ["BalanceReport", "resample", "plan_targets", "smote_numpy",
           "STRATEGIES"]

#: Strategy -> one-line description, used by the CLI's help text and the app sidebar.
STRATEGIES: Dict[str, str] = {
    "none": "train on the raw distribution; rely on macro metrics to expose the bias",
    "class_weight": "no resampling; estimators get inverse-frequency weights instead",
    "smote": "synthetic interpolation of minority classes (Chawla et al., 2002)",
    "smote_tomek": "SMOTE, then remove borderline pairs that straddle the boundary",
    "random_oversample": "duplicate minority rows; a baseline SMOTE should beat",
}


@dataclass
class BalanceReport:
    """Before/after counts plus every adjustment that was forced on us."""

    strategy: str = "none"
    rows_in: int = 0
    rows_out: int = 0
    synthetic_rows: int = 0
    removed_rows: int = 0
    before: Dict[str, int] = field(default_factory=dict)
    after: Dict[str, int] = field(default_factory=dict)
    k_reductions: Dict[str, int] = field(default_factory=dict)
    expansion_capped: List[str] = field(default_factory=list)
    duplicated_classes: List[str] = field(default_factory=list)
    implementation: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def imbalance_before(self) -> float:
        return _ratio(self.before)

    @property
    def imbalance_after(self) -> float:
        return _ratio(self.after)

    def table(self) -> pd.DataFrame:
        classes = list(self.before) or list(self.after)
        rows = [{
            "class": c,
            "before": self.before.get(c, 0),
            "after": self.after.get(c, 0),
            "change": self.after.get(c, 0) - self.before.get(c, 0),
            "factor": (self.after.get(c, 0) / self.before[c]
                       if self.before.get(c) else float("nan")),
        } for c in classes]
        out = pd.DataFrame(rows).sort_values("before", ascending=False)
        out["factor"] = out["factor"].map(lambda v: f"{v:.2f}x" if v == v else "-")
        return out.reset_index(drop=True)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "implementation": self.implementation,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "synthetic_rows": self.synthetic_rows,
            "removed_rows": self.removed_rows,
            "imbalance_before": round(self.imbalance_before, 1),
            "imbalance_after": round(self.imbalance_after, 1),
            "before": self.before,
            "after": self.after,
            "k_reductions": self.k_reductions,
            "expansion_capped": self.expansion_capped,
            "duplicated_classes": self.duplicated_classes,
            "notes": self.notes,
        }

    def render(self) -> str:
        lines = [f"strategy: {self.strategy}"
                 + (f"  ({self.implementation})" if self.implementation else ""),
                 f"rows {human_count(self.rows_in)} -> {human_count(self.rows_out)}"
                 f"   synthetic {human_count(self.synthetic_rows)}"
                 f"   removed {human_count(self.removed_rows)}",
                 f"majority:minority ratio {self.imbalance_before:,.0f}:1 -> "
                 f"{self.imbalance_after:,.0f}:1",
                 ""]
        lines.append(self.table().to_string(index=False))
        if self.k_reductions:
            lines.append("")
            lines.append("k reduced for small classes: " + ", ".join(
                f"{c} (k={k})" for c, k in self.k_reductions.items()))
        if self.expansion_capped:
            lines.append(f"expansion capped for: {', '.join(self.expansion_capped)}")
        if self.duplicated_classes:
            lines.append("duplicated rather than interpolated (too few rows): "
                         + ", ".join(self.duplicated_classes))
        for note in self.notes:
            lines.append(f"note: {note}")
        return "\n".join(lines)


def _ratio(counts: Dict[str, int]) -> float:
    values = [v for v in counts.values() if v > 0]
    return max(values) / min(values) if values else 1.0


# ---------------------------------------------------------------------------
# target planning
# ---------------------------------------------------------------------------

def plan_targets(
    counts: np.ndarray,
    *,
    max_ratio: float = 0.25,
    max_expansion: float = 20.0,
    minimum: int = 0,
) -> Tuple[np.ndarray, List[int]]:
    """Decide the post-resampling size of every class.

    ``target = min(max_ratio * majority, count * max_expansion)``, never below the
    current count - this module only ever adds rows for a class, so a class that is
    already large is untouched.

    Returns ``(targets, indices_where_expansion_capped)``.
    """
    counts = np.asarray(counts, dtype=np.int64)
    if not 0 < max_ratio <= 1.0:
        raise ValueError(f"max_ratio must be in (0, 1], got {max_ratio}")
    if max_expansion < 1.0:
        raise ValueError(f"max_expansion must be >= 1, got {max_expansion}")

    majority = counts.max() if counts.size else 0
    wanted = np.ceil(max_ratio * majority).astype(np.int64)
    ceiling = np.floor(counts * max_expansion).astype(np.int64)

    targets = np.minimum(np.maximum(wanted, minimum), ceiling)
    targets = np.maximum(targets, counts)         # never shrink
    capped = [int(i) for i in range(len(counts))
              if counts[i] < wanted and targets[i] < wanted]
    return targets, capped


# ---------------------------------------------------------------------------
# numpy SMOTE
# ---------------------------------------------------------------------------

def _knn_within(block: np.ndarray, k: int, *, chunk: int = 512) -> np.ndarray:
    """Indices of each row's *k* nearest neighbours within *block*, excluding itself.

    Brute-force squared euclidean distance, computed in row chunks so peak memory is
    ``chunk x len(block)`` rather than ``len(block)^2``. Only ever called on classes
    small enough to need oversampling, so the quadratic term stays bounded.
    """
    n = len(block)
    k = max(1, min(k, n - 1))
    sq = (block ** 2).sum(axis=1)
    out = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        # |a-b|^2 = |a|^2 - 2ab + |b|^2; the constant sq term does not affect ordering
        # but keeping it makes the distances real, which helps when debugging.
        d = sq[start:stop, None] - 2.0 * block[start:stop] @ block.T + sq[None, :]
        rows = np.arange(start, stop)
        d[rows - start, rows] = np.inf         # never pick yourself
        out[start:stop] = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
    return out


def smote_numpy(
    block: np.ndarray,
    n_new: int,
    *,
    k: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate *n_new* synthetic rows from a single class's rows.

    The original algorithm: choose a seed row, choose one of its k in-class nearest
    neighbours, and return ``seed + u * (neighbour - seed)`` with ``u ~ U(0, 1)``.

    With one row there is nothing to interpolate towards, so the row is duplicated
    exactly. Duplicates are honest - the alternative, jittering with invented noise,
    fabricates a covariance structure that was never measured.
    """
    rng = rng or np.random.default_rng(0)
    if n_new <= 0:
        return np.empty((0, block.shape[1]), dtype=np.float64)
    if len(block) == 1:
        return np.repeat(block, n_new, axis=0)

    neighbours = _knn_within(block, k)
    seeds = rng.integers(0, len(block), size=n_new)
    picks = neighbours[seeds, rng.integers(0, neighbours.shape[1], size=n_new)]
    gaps = rng.random((n_new, 1))
    return block[seeds] + gaps * (block[picks] - block[seeds])


def _tomek_mask(X: np.ndarray, y: np.ndarray, *, chunk: int = 256) -> np.ndarray:
    """Boolean mask that removes Tomek links: mutual nearest neighbours of different class.

    O(n^2) with no spatial index, so the caller guards on size. Both members of a link
    are dropped, which is what imblearn's ``TomekLinks(sampling_strategy='all')`` does;
    the pair sits exactly on a class boundary, and removing both widens the margin
    rather than biasing it towards one class.
    """
    n = len(X)
    sq = (X ** 2).sum(axis=1)
    nearest = np.empty(n, dtype=np.int64)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d = sq[start:stop, None] - 2.0 * X[start:stop] @ X.T + sq[None, :]
        rows = np.arange(start, stop)
        d[rows - start, rows] = np.inf
        nearest[start:stop] = d.argmin(axis=1)
    mutual = nearest[nearest] == np.arange(n)
    link = mutual & (y[nearest] != y)
    keep = ~link
    return keep


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------

def resample(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    *,
    strategy: str = "smote",
    max_ratio: float = 0.25,
    k_neighbours: int = 5,
    max_expansion: float = 20.0,
    seed: int = 42,
    class_names: Optional[List[str]] = None,
    prefer_imblearn: bool = True,
    tomek_row_limit: int = 40_000,
) -> Tuple[np.ndarray, np.ndarray, BalanceReport]:
    """Rebalance the training split. Returns ``(X_resampled, y_resampled, report)``.

    ``X`` may be a frame or an array; the return is always a float64 array, because
    synthetic rows have no index and pretending otherwise causes silent misalignment
    later.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from "
                         f"{sorted(STRATEGIES)}")

    columns = list(X.columns) if isinstance(X, pd.DataFrame) else None
    values = (X.to_numpy(dtype=np.float64, copy=True) if isinstance(X, pd.DataFrame)
              else np.array(X, dtype=np.float64, copy=True))
    y = np.asarray(y, dtype=np.int64)
    if len(values) != len(y):
        raise ValueError(f"X has {len(values)} rows but y has {len(y)}")
    if values.size and not np.isfinite(values).all():
        raise ValueError(
            "resampling requires finite values - interpolating towards NaN silently "
            "poisons whole synthetic rows. Run the preprocessor first."
        )

    n_classes = int(y.max()) + 1 if y.size else 0
    counts = np.bincount(y, minlength=n_classes)
    names = class_names or [str(i) for i in range(n_classes)]

    report = BalanceReport(strategy=strategy, rows_in=len(values))
    report.before = {names[i]: int(c) for i, c in enumerate(counts) if c}

    if strategy in {"none", "class_weight"}:
        report.rows_out = len(values)
        report.after = dict(report.before)
        report.implementation = "no resampling"
        if strategy == "class_weight":
            report.notes.append("imbalance is handled by estimator class weights; see "
                                "preprocess.class_weights")
        log.info("balancing strategy %r leaves the %s training rows untouched "
                 "(ratio %.0f:1)", strategy, human_count(len(values)),
                 report.imbalance_before)
        return values, y, report

    with stage(log, f"balancing ({strategy})") as st:
        targets, capped_idx = plan_targets(counts, max_ratio=max_ratio,
                                           max_expansion=max_expansion)
        report.expansion_capped = [names[i] for i in capped_idx]

        used_imblearn = False
        if prefer_imblearn:
            result = _try_imblearn(values, y, strategy=strategy, targets=targets,
                                   counts=counts, k=k_neighbours, seed=seed,
                                   report=report, names=names)
            if result is not None:
                values, y = result
                used_imblearn = True

        if not used_imblearn:
            values, y = _resample_numpy(values, y, strategy=strategy, targets=targets,
                                        counts=counts, k=k_neighbours, seed=seed,
                                        report=report, names=names,
                                        tomek_row_limit=tomek_row_limit)

        after = np.bincount(y, minlength=n_classes)
        report.after = {names[i]: int(c) for i, c in enumerate(after) if c}
        report.rows_out = len(values)
        st["summary"] = (f"{human_count(report.rows_in)} -> "
                         f"{human_count(report.rows_out)} rows, ratio "
                         f"{report.imbalance_before:,.0f}:1 -> "
                         f"{report.imbalance_after:,.0f}:1")

    if columns is not None:
        # Caller may want names back; hand over an array plus the columns it matches.
        report.notes.append(f"feature order preserved: {len(columns)} columns")
    return values, y, report


def _try_imblearn(
    X: np.ndarray, y: np.ndarray, *, strategy: str, targets: np.ndarray,
    counts: np.ndarray, k: int, seed: int, report: BalanceReport, names: List[str],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Use imbalanced-learn when it is installed. Returns None to fall back."""
    try:
        if strategy in {"smote", "smote_tomek"}:
            from imblearn.over_sampling import SMOTE
        elif strategy == "random_oversample":
            from imblearn.over_sampling import RandomOverSampler
        else:                                     # pragma: no cover
            return None
    except ImportError:
        log.info("imbalanced-learn not installed; using the built-in numpy "
                 "implementation (same algorithm, no extra dependency)")
        return None

    sampling = {int(i): int(t) for i, t in enumerate(targets)
                if counts[i] > 0 and t > counts[i]}
    if not sampling:
        report.implementation = "imblearn (nothing to do)"
        return X, y

    if strategy == "random_oversample":
        sampler = RandomOverSampler(sampling_strategy=sampling, random_state=seed)
        report.implementation = "imblearn.RandomOverSampler"
    else:
        # k must be < the smallest class being oversampled, or SMOTE raises.
        smallest = min(int(counts[i]) for i in sampling)
        k_eff = max(1, min(k, smallest - 1))
        if k_eff != k:
            report.k_reductions = {names[i]: k_eff for i in sampling
                                   if counts[i] <= k}
            log.info("reduced SMOTE k from %d to %d: the smallest oversampled class "
                     "has %d rows", k, k_eff, smallest)
        if smallest < 2:
            log.info("a class being oversampled has 1 row, which SMOTE cannot "
                     "interpolate; falling back to the numpy path which duplicates it")
            return None
        sampler = SMOTE(sampling_strategy=sampling, k_neighbors=k_eff,
                        random_state=seed)
        report.implementation = "imblearn.SMOTE"

    try:
        Xr, yr = sampler.fit_resample(X, y)
    except Exception as exc:                      # noqa: BLE001 - imblearn raises broadly
        log.warning("imblearn failed (%s); falling back to the numpy implementation",
                    exc)
        return None

    report.synthetic_rows = len(Xr) - len(X)

    if strategy == "smote_tomek":
        try:
            from imblearn.under_sampling import TomekLinks
            before = len(Xr)
            Xr, yr = TomekLinks(sampling_strategy="all").fit_resample(Xr, yr)
            report.removed_rows = before - len(Xr)
            report.implementation = "imblearn.SMOTE + TomekLinks"
        except Exception as exc:                  # noqa: BLE001
            log.warning("Tomek-link cleaning skipped (%s)", exc)
            report.notes.append(f"Tomek cleaning skipped: {exc}")

    return np.asarray(Xr, dtype=np.float64), np.asarray(yr, dtype=np.int64)


def _resample_numpy(
    X: np.ndarray, y: np.ndarray, *, strategy: str, targets: np.ndarray,
    counts: np.ndarray, k: int, seed: int, report: BalanceReport, names: List[str],
    tomek_row_limit: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Built-in resampler: SMOTE / duplication, optionally followed by Tomek cleaning."""
    rng = np.random.default_rng(seed)
    new_blocks: List[np.ndarray] = []
    new_labels: List[np.ndarray] = []

    for cls in range(len(counts)):
        have, want = int(counts[cls]), int(targets[cls])
        if have == 0 or want <= have:
            continue
        block = X[y == cls]
        n_new = want - have

        if strategy == "random_oversample":
            picks = rng.integers(0, have, size=n_new)
            synth = block[picks]
        else:
            k_eff = max(1, min(k, have - 1))
            if k_eff != k:
                report.k_reductions[names[cls]] = k_eff
            if have == 1:
                report.duplicated_classes.append(names[cls])
            synth = smote_numpy(block, n_new, k=k_eff, rng=rng)

        new_blocks.append(synth)
        new_labels.append(np.full(n_new, cls, dtype=np.int64))
        log.debug("class %-24s %6d -> %6d (+%d, k=%s)", names[cls], have, want,
                  n_new, report.k_reductions.get(names[cls], k))

    report.implementation = ("built-in numpy SMOTE" if strategy != "random_oversample"
                             else "built-in numpy duplication")
    if report.k_reductions:
        log.info("reduced k for %d small class(es): %s", len(report.k_reductions),
                 report.k_reductions)
    if report.duplicated_classes:
        log.warning("class(es) with a single training row were duplicated, not "
                    "interpolated: %s - their metrics are meaningless and are reported "
                    "as such", report.duplicated_classes)

    if new_blocks:
        X = np.vstack([X] + new_blocks)
        y = np.concatenate([y] + new_labels)
        report.synthetic_rows = int(sum(len(b) for b in new_blocks))
        # Shuffle so the synthetic rows are not all at the end, which matters for any
        # estimator that takes a validation slice off the tail (Keras does by default).
        order = rng.permutation(len(X))
        X, y = X[order], y[order]

    if strategy == "smote_tomek":
        if len(X) > tomek_row_limit:
            msg = (f"Tomek cleaning skipped: {human_count(len(X))} rows exceeds the "
                   f"{human_count(tomek_row_limit)}-row limit for the O(n^2) built-in. "
                   "Install imbalanced-learn for the indexed version.")
            log.warning(msg)
            report.notes.append(msg)
        else:
            keep = _tomek_mask(X, y)
            report.removed_rows = int((~keep).sum())
            X, y = X[keep], y[keep]
            report.implementation += " + built-in Tomek cleaning"
            log.info("Tomek cleaning removed %s boundary row(s)",
                     human_count(report.removed_rows))

    return X, y
