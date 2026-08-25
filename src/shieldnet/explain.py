"""SHAP explanations, with a genuinely working fallback when SHAP is not installed.

What is being explained, and why that choice
--------------------------------------------
Three different questions get called "explainability", and conflating them is how
explanation sections end up saying nothing:

*Which features does the model rely on overall?* - :meth:`Explainer.global_explanation`.
Mean absolute SHAP value per feature, summed across classes. This is what the report's
feature-importance figure shows, and it is the number to compare against the selection
ranking from :mod:`shieldnet.features`: strong agreement means the filter methods picked
features the model genuinely uses, and disagreement is worth a paragraph.

*Which features distinguish this particular attack class?* -
:meth:`GlobalExplanation.for_class`. Mean absolute SHAP for one output class only. This
is where the interesting findings live, because it is the difference between "flow
duration matters" and "flow duration is what separates slowloris from Slowhttptest".

*Why was this specific flow flagged?* - :meth:`Explainer.explain_row`. Signed
per-feature contributions for one prediction. This is what the app shows an analyst, and
it is the only one of the three that is an *explanation* rather than a summary.

Why not stop at ``feature_importances_``
---------------------------------------
A tree ensemble's built-in importance counts how often a feature was split on, weighted
by impurity gain. It is unsigned, so it cannot say whether a high value pushed towards or
away from a class; it is class-agnostic, so it cannot say *which* attack a feature
detects; and it is biased towards high-cardinality features, which on CICIDS2017 means
the continuous timing columns are systematically flattered over the binary flag columns.
SHAP values are signed, per-class, per-row, and additive - they sum to the difference
between this prediction and the average prediction, which is a property you can actually
check, and :meth:`Explainer.verify_additivity` does check it.

The fallback is not a stub
--------------------------
Without SHAP, :func:`permutation_importance` measures the real drop in macro F1 when a
feature's column is shuffled, and :meth:`Explainer.explain_row` falls back to occlusion -
replacing each feature with its background median and measuring how far the predicted
probability moves. Both are legitimate, model-agnostic attribution methods that predate
SHAP; permutation importance is Breiman's (2001) original proposal. They are slower and
noisier than SHAP and they do not decompose additively, and the report says so, but the
pipeline never produces an empty explanation section because a pip install failed.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import schema as sch
from .evaluate import evaluate, stratified_subsample
from .logging_utils import get_logger, human_duration, stage

log = get_logger(__name__)

#: One-shot guard so the "shap is not importable" notice is logged once per process
#: rather than once per explained row. Whether the package exists is a property of the
#: environment, not of the row being explained.
_SHAP_IMPORT_LOGGED = False

__all__ = [
    "FeatureContribution", "LocalExplanation", "GlobalExplanation", "Explainer",
    "permutation_importance", "shap_available", "normalise_shap_values",
]


def shap_available() -> bool:
    """Whether the ``shap`` package can be imported."""
    try:
        import shap                                     # noqa: F401
        return True
    except Exception:                                   # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

@dataclass
class FeatureContribution:
    """One feature's signed contribution to one prediction."""

    name: str
    value: float                  # the feature's value for this row (scaled units)
    contribution: float           # signed push towards the predicted class
    raw_value: Optional[float] = None      # the value before scaling, when known

    @property
    def direction(self) -> str:
        if self.contribution > 0:
            return "increases"
        if self.contribution < 0:
            return "decreases"
        return "does not affect"

    @property
    def magnitude(self) -> float:
        return abs(self.contribution)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "value": self.value,
                "contribution": self.contribution, "raw_value": self.raw_value,
                "direction": self.direction}


@dataclass
class LocalExplanation:
    """Why one flow received the prediction it did."""

    model: str
    predicted_index: int
    predicted_class: str
    confidence: float
    contributions: List[FeatureContribution]
    base_value: float
    method: str
    class_names: List[str] = field(default_factory=list)
    probabilities: Optional[np.ndarray] = None
    runner_up: str = ""
    runner_up_confidence: float = 0.0
    additivity_error: Optional[float] = None

    def top(self, n: int = 8) -> List[FeatureContribution]:
        """The *n* features with the largest absolute contribution."""
        return sorted(self.contributions, key=lambda c: -c.magnitude)[:n]

    def supporting(self, n: int = 5) -> List[FeatureContribution]:
        """Features that pushed towards the predicted class."""
        pos = [c for c in self.contributions if c.contribution > 0]
        return sorted(pos, key=lambda c: -c.contribution)[:n]

    def opposing(self, n: int = 5) -> List[FeatureContribution]:
        """Features that pushed away from it - the case against the prediction."""
        neg = [c for c in self.contributions if c.contribution < 0]
        return sorted(neg, key=lambda c: c.contribution)[:n]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model, "predicted_class": self.predicted_class,
            "predicted_index": self.predicted_index, "confidence": self.confidence,
            "base_value": self.base_value, "method": self.method,
            "runner_up": self.runner_up,
            "runner_up_confidence": self.runner_up_confidence,
            "additivity_error": self.additivity_error,
            "contributions": [c.to_dict() for c in self.top(20)],
        }

    def frame(self):
        import pandas as pd
        return pd.DataFrame([c.to_dict() for c in
                             sorted(self.contributions, key=lambda c: -c.magnitude)])

    def render(self, *, top: int = 8) -> str:
        lines = [
            f"predicted {self.predicted_class} with {self.confidence:.1%} confidence",
        ]
        if self.runner_up:
            lines.append(f"  next most likely: {self.runner_up} "
                         f"({self.runner_up_confidence:.1%})")
        lines.append(f"  attribution method: {self.method}")
        if self.additivity_error is not None:
            lines.append(f"  additivity check: contributions sum to within "
                         f"{self.additivity_error:.2e} of the prediction")
        lines.append("")
        lines.append(f"  {'feature':<32}{'value':>14}{'contribution':>14}  effect")
        lines.append("  " + "-" * 76)
        for c in self.top(top):
            shown = c.raw_value if c.raw_value is not None else c.value
            lines.append(f"  {c.name[:32]:<32}{shown:>14.4g}{c.contribution:>14.5f}"
                         f"  {c.direction} {self.predicted_class}")
        return "\n".join(lines)

    def __str__(self) -> str:                           # pragma: no cover
        return self.render()


@dataclass
class GlobalExplanation:
    """What the model relies on across many rows."""

    model: str
    feature_names: List[str]
    class_names: List[str]
    importance: np.ndarray                    # (n_features,)
    per_class: Optional[np.ndarray]           # (n_classes, n_features) or None
    method: str
    rows_explained: int
    background_rows: int
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    # -- queries -------------------------------------------------------------

    def top_features(self, n: int = 15) -> List[Tuple[str, float]]:
        order = np.argsort(-self.importance)[:n]
        return [(self.feature_names[i], float(self.importance[i])) for i in order]

    def for_class(self, name_or_index: Any, n: int = 10) -> List[Tuple[str, float]]:
        """Top features for one output class."""
        if self.per_class is None:
            raise ValueError(
                f"{self.method} produced no per-class breakdown, only a single "
                "importance vector. Install shap for per-class attributions.")
        idx = self._class_index(name_or_index)
        row = self.per_class[idx]
        order = np.argsort(-row)[:n]
        return [(self.feature_names[i], float(row[i])) for i in order]

    def _class_index(self, name_or_index: Any) -> int:
        if isinstance(name_or_index, (int, np.integer)):
            return int(name_or_index)
        target = sch.canonical_label(str(name_or_index))
        for i, name in enumerate(self.class_names):
            if name == target or name == str(name_or_index):
                return i
        raise KeyError(f"unknown class {name_or_index!r}; have {self.class_names}")

    def signature_features(self, n: int = 3) -> Dict[str, List[str]]:
        """The features that are distinctively important to each class.

        Not simply the top-*n* per class: several classes share the same globally strong
        features, so a plain top-*n* prints "Flow Duration" thirteen times and says
        nothing. This divides each class's importance by the mean across classes first,
        so what comes out is what makes that class *different*.
        """
        if self.per_class is None:
            return {}
        mean = self.per_class.mean(axis=0)
        mean = np.where(mean > 0, mean, 1.0)
        out: Dict[str, List[str]] = {}
        for i, cls in enumerate(self.class_names):
            if self.per_class[i].sum() <= 0:
                continue
            distinctive = self.per_class[i] / mean
            order = np.argsort(-distinctive)[:n]
            out[cls] = [self.feature_names[j] for j in order]
        return out

    def agreement_with(self, ranking: Sequence[str]) -> Dict[str, Any]:
        """Compare this importance ordering against the feature-selection ranking.

        The overlap in the top *k* is the honest way to report whether filter-based
        selection chose features the trained model actually uses. Rank correlation over
        all features would be dominated by the long tail of features nobody cares about.
        """
        mine = [n for n, _ in self.top_features(len(self.feature_names))]
        theirs = [str(r) for r in ranking]
        out: Dict[str, Any] = {}
        for k in (5, 10, 15):
            k = min(k, len(mine), len(theirs))
            if k == 0:
                continue
            shared = set(mine[:k]) & set(theirs[:k])
            out[f"top{k}_overlap"] = len(shared)
            out[f"top{k}_shared"] = sorted(shared)
        return out

    # -- output --------------------------------------------------------------

    def frame(self):
        import pandas as pd
        data = {"feature": self.feature_names, "importance": self.importance}
        df = pd.DataFrame(data).sort_values("importance", ascending=False)
        df["share"] = df["importance"] / max(self.importance.sum(), 1e-12)
        df["cumulative_share"] = df["share"].cumsum()
        return df.reset_index(drop=True)

    def per_class_frame(self):
        import pandas as pd
        if self.per_class is None:
            raise ValueError(f"{self.method} produced no per-class breakdown")
        return pd.DataFrame(self.per_class, index=self.class_names,
                            columns=self.feature_names)

    def to_dict(self, *, top: int = 25) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model, "method": self.method,
            "rows_explained": self.rows_explained,
            "background_rows": self.background_rows,
            "seconds": self.seconds,
            "importance": {n: float(v) for n, v in self.top_features(top)},
            "notes": list(self.notes),
        }
        if self.per_class is not None:
            payload["signature_features"] = self.signature_features()
            payload["per_class_top"] = {
                cls: dict(self.for_class(i, 8))
                for i, cls in enumerate(self.class_names)
                if self.per_class[i].sum() > 0
            }
        return payload

    def render(self, *, top: int = 15, per_class: bool = True) -> str:
        total = max(float(self.importance.sum()), 1e-12)
        lines = [
            f"global feature importance for {self.model} ({self.method})",
            f"  {self.rows_explained:,} row(s) explained against a "
            f"{self.background_rows:,}-row background, "
            f"{human_duration(self.seconds)}",
            "",
            f"  {'#':<4}{'feature':<34}{'importance':>13}{'share':>9}{'cumul':>9}",
            "  " + "-" * 68,
        ]
        running = 0.0
        for rank, (name, value) in enumerate(self.top_features(top), 1):
            running += value / total
            lines.append(f"  {rank:<4}{name[:34]:<34}{value:>13.5f}"
                         f"{value / total:>9.1%}{running:>9.1%}")
        if per_class and self.per_class is not None:
            lines.append("")
            lines.append("  what distinguishes each class (importance relative to the "
                         "cross-class mean):")
            for cls, feats in self.signature_features(3).items():
                lines.append(f"    {cls:<28} {', '.join(feats)}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)

    def __str__(self) -> str:                           # pragma: no cover
        return self.render()

    # -- figures -------------------------------------------------------------

    def plot(self, path: os.PathLike | str, *, top: int = 20,
             title: Optional[str] = None) -> Optional[Path]:
        """Horizontal bar chart of the top features. Returns ``None`` without matplotlib."""
        try:
            import matplotlib
            matplotlib.use("Agg")                       # no display in CI or Colab
            import matplotlib.pyplot as plt
        except ImportError:
            log.info("matplotlib is not installed; skipping the importance figure")
            return None

        pairs = self.top_features(top)[::-1]            # smallest at the bottom
        names = [p[0] for p in pairs]
        values = [p[1] for p in pairs]
        height = max(3.0, 0.32 * len(pairs) + 1.2)
        fig, ax = plt.subplots(figsize=(9.5, height), dpi=150)
        ax.barh(range(len(pairs)), values, color="#2b6cb0", height=0.72)
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels([n[:38] for n in names], fontsize=8)
        ax.set_xlabel(f"mean |contribution| ({self.method})", fontsize=9)
        ax.set_title(title or f"{self.model}: feature importance", fontsize=11)
        ax.grid(axis="x", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, bbox_inches="tight")
        plt.close(fig)
        log.info("wrote %s", target)
        return target

    def plot_per_class(self, path: os.PathLike | str, *, top: int = 12
                       ) -> Optional[Path]:
        """Heatmap of per-class importance over the top features."""
        if self.per_class is None:
            return None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        order = np.argsort(-self.importance)[:top]
        block = self.per_class[:, order]
        # Row-normalise: BENIGN has far larger absolute SHAP values than a 30-row class
        # simply because it dominates the background, and without normalising the
        # heatmap shows only that fact.
        rows = block / np.maximum(block.max(axis=1, keepdims=True), 1e-12)
        keep = self.per_class.sum(axis=1) > 0
        rows, labels = rows[keep], [c for c, k in zip(self.class_names, keep) if k]

        fig, ax = plt.subplots(figsize=(1.0 + 0.62 * len(order), 1.2 + 0.42 * len(rows)),
                               dpi=150)
        im = ax.imshow(rows, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([self.feature_names[i][:22] for i in order],
                           rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(f"{self.model}: per-class importance (row-normalised)", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8, label="relative importance")
        fig.tight_layout()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, bbox_inches="tight")
        plt.close(fig)
        log.info("wrote %s", target)
        return target


# ---------------------------------------------------------------------------
# SHAP shape normalisation
# ---------------------------------------------------------------------------

def normalise_shap_values(
    values: Any, *, n_rows: int, n_features: int, n_classes: int
) -> np.ndarray:
    """Coerce whatever SHAP returned into ``(n_rows, n_features, n_classes)``.

    This function exists because SHAP's multi-class output shape has changed three times
    and depends on the explainer:

    * ``TreeExplainer`` in shap < 0.42 returns a **list** of ``n_classes`` arrays, each
      ``(n_rows, n_features)``.
    * shap >= 0.45 returns a single ``(n_rows, n_features, n_classes)`` array.
    * ``Explanation`` objects wrap the array in ``.values``.
    * ``GradientExplainer`` on a Keras model whose input is ``(n, f, 1)`` returns
      ``(n_rows, n_features, 1, n_classes)`` - the singleton channel axis survives.
    * Single-row calls sometimes drop the row axis entirely.

    Getting this wrong does not raise; it transposes features against classes and
    produces an importance ranking that is confidently, silently wrong. Hence one place
    that handles every case and asserts the result.
    """
    if hasattr(values, "values") and not isinstance(values, np.ndarray):
        values = values.values                          # shap.Explanation

    if isinstance(values, (list, tuple)):
        stacked = np.stack([np.asarray(v, dtype=np.float64) for v in values], axis=-1)
    else:
        stacked = np.asarray(values, dtype=np.float64)

    # Drop singleton channel axes left over from a (n, f, 1) Keras input.
    while stacked.ndim > 3:
        squeezable = [ax for ax in range(1, stacked.ndim - 1) if stacked.shape[ax] == 1]
        if not squeezable:
            break
        stacked = np.squeeze(stacked, axis=squeezable[0])

    if stacked.ndim == 2:
        if stacked.shape == (n_features, n_classes):
            stacked = stacked[None, :, :]               # single row, row axis dropped
        elif stacked.shape == (n_rows, n_features):
            stacked = stacked[:, :, None]               # binary or single-output
        else:
            raise ValueError(
                f"cannot interpret SHAP values of shape {stacked.shape} for "
                f"{n_rows} rows, {n_features} features, {n_classes} classes")
    elif stacked.ndim == 1:
        if stacked.size != n_features:
            raise ValueError(f"1-D SHAP values of size {stacked.size} do not match "
                             f"{n_features} features")
        stacked = stacked.reshape(1, n_features, 1)

    if stacked.ndim != 3:
        raise ValueError(f"unexpected SHAP value shape {stacked.shape}")

    # Classes-first layouts do occur; transpose rather than mis-reading them.
    if stacked.shape[1] != n_features:
        if stacked.shape[2] == n_features and stacked.shape[1] in (n_classes, 1):
            stacked = np.transpose(stacked, (0, 2, 1))
        elif stacked.shape[0] == n_classes and stacked.shape[1] == n_rows:
            stacked = np.transpose(stacked, (1, 2, 0))
    if stacked.shape[1] != n_features:
        raise ValueError(
            f"SHAP values have {stacked.shape[1]} feature slots but the model has "
            f"{n_features} features (raw shape after normalisation {stacked.shape})")
    if not np.isfinite(stacked).all():
        n_bad = int((~np.isfinite(stacked)).sum())
        log.warning("%d non-finite SHAP value(s) were replaced with zero", n_bad)
        stacked = np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)
    return stacked


# ---------------------------------------------------------------------------
# permutation importance - the model-agnostic fallback
# ---------------------------------------------------------------------------

def permutation_importance(
    predict_proba: Callable[[np.ndarray], np.ndarray],
    X: Any,
    y: Any,
    *,
    class_names: Sequence[str],
    metric: str = "macro_f1",
    n_repeats: int = 3,
    max_rows: int = 4000,
    seed: int = 42,
    per_class: bool = True,
    feature_names: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Breiman's permutation importance: how much does shuffling a column cost?

    Returns ``(importance, per_class_importance, info)``. Importance is the *drop* in the
    metric, so larger means the model depends on that feature more; a negative value
    means shuffling the column helped, which is noise and is clipped to zero in the
    reported figure but preserved in ``info["negative_features"]`` because a strongly
    negative value is a hint that the feature is actively harmful.

    Known limitation, stated because it changes how the numbers should be read: with
    correlated features - and CICIDS2017 is full of them, since packet-length mean, max
    and std are functions of each other - permuting one feature leaves its information
    available through its correlates, so importance is *understated* for every member of
    a correlated group. :func:`shieldnet.features.prune_correlated` mitigates this
    upstream by dropping features above 0.95 correlation before the model ever sees them.
    """
    Xa = np.ascontiguousarray(
        X.to_numpy() if hasattr(X, "to_numpy") else X, dtype=np.float32)
    ya = np.asarray(y, dtype=np.int64).ravel()
    if Xa.shape[0] != ya.size:
        raise ValueError(f"{Xa.shape[0]} rows of X but {ya.size} labels")
    names = list(feature_names) if feature_names is not None \
        else [f"f{i}" for i in range(Xa.shape[1])]
    rng = np.random.default_rng(seed)

    if ya.size > max_rows:
        keep = stratified_subsample(ya, max_rows, rng, floor=40)
        Xa, ya = Xa[keep], ya[keep]

    def score(matrix: np.ndarray):
        return evaluate(ya, predict_proba(matrix), class_names=class_names,
                        curves=False, split="permutation")

    base_report = score(Xa)
    baseline = float(getattr(base_report, metric))
    base_recall = np.array([c.recall for c in base_report.per_class])

    n_features = Xa.shape[1]
    drops = np.zeros((n_repeats, n_features))
    class_drops = np.zeros((n_repeats, len(class_names), n_features)) \
        if per_class else None

    started = time.perf_counter()
    for rep in range(int(n_repeats)):
        for j in range(n_features):
            column = Xa[:, j].copy()
            # Shuffle in place and restore, rather than copying the whole matrix per
            # feature: on 4000x25 float32 the copy is trivial, but this same loop runs
            # on the full test split in the report pipeline.
            Xa[:, j] = rng.permutation(column)
            report = score(Xa)
            Xa[:, j] = column
            drops[rep, j] = baseline - float(getattr(report, metric))
            if class_drops is not None:
                shuffled = np.array([c.recall for c in report.per_class])
                class_drops[rep, :, j] = base_recall - shuffled
        log.debug("permutation importance: repeat %d/%d done", rep + 1, n_repeats)

    mean = drops.mean(axis=0)
    info: Dict[str, Any] = {
        "metric": metric,
        "baseline": baseline,
        "n_repeats": int(n_repeats),
        "rows_used": int(ya.size),
        "seconds": time.perf_counter() - started,
        "std": drops.std(axis=0, ddof=1).tolist() if n_repeats > 1 else None,
        "negative_features": [names[j] for j in np.nonzero(mean < -1e-6)[0]],
    }
    per_class_mean = None
    if class_drops is not None:
        per_class_mean = np.clip(class_drops.mean(axis=0), 0.0, None)
    return np.clip(mean, 0.0, None), per_class_mean, info


def _stratified_head(y: np.ndarray, max_rows: int, rng: np.random.Generator
                     ) -> np.ndarray:
    """Deprecated alias kept only so older saved notebooks keep working.

    The budget arithmetic now lives in :func:`shieldnet.evaluate.stratified_subsample`,
    because three near-identical copies of it had already drifted into three different
    off-by-a-floor behaviours.
    """
    return stratified_subsample(y, max_rows, rng, floor=40)


# ---------------------------------------------------------------------------
# the explainer
# ---------------------------------------------------------------------------

class Explainer:
    """Wraps SHAP for one fitted model, falling back to permutation and occlusion.

    Usage::

        ex = Explainer(model, feature_names, class_names)
        ex.set_background(X_train)
        g = ex.global_explanation(X_test, y_test)
        local = ex.explain_row(X_test[0])
    """

    def __init__(
        self,
        model: Any,
        feature_names: Sequence[str],
        class_names: Sequence[str],
        *,
        seed: int = 42,
        max_background: int = 200,
        prefer_shap: bool = True,
        preprocessor: Any = None,
    ) -> None:
        self.model = model
        self.feature_names = [str(f) for f in feature_names]
        self.class_names = [str(c) for c in class_names]
        self.seed = int(seed)
        self.max_background = int(max_background)
        self.prefer_shap = bool(prefer_shap)
        #: Optional :class:`~shieldnet.preprocess.Preprocessor`, used only to report
        #: feature values in original units in a local explanation. Explanations are
        #: computed in scaled space either way - that is the space the model sees.
        self.preprocessor = preprocessor

        self.background_: Optional[np.ndarray] = None
        self.background_median_: Optional[np.ndarray] = None
        self._explainer: Any = None
        self._explainer_kind: str = ""
        self.notes: List[str] = []

    # -- background ----------------------------------------------------------

    def set_background(self, X: Any, y: Any = None) -> "Explainer":
        """Choose the reference distribution SHAP compares against.

        The background is what "average" means in "this feature pushed the prediction
        above average", so it is not an implementation detail. Two choices matter:

        *Size.* KernelExplainer is O(background x features x rows); 200 rows is the
        practical ceiling and already dominates the runtime.

        *Composition.* A uniform random sample of CICIDS2017 is about 50% BENIGN, so
        "average" means "roughly benign traffic" and every attack explanation reads as a
        deviation from normal - which is exactly the framing an analyst wants. Passing
        *y* switches to a class-balanced background, which instead answers "what makes
        this attack different from other attacks". The default is the uniform sample.
        """
        Xa = np.ascontiguousarray(
            X.to_numpy() if hasattr(X, "to_numpy") else X, dtype=np.float32)
        if Xa.shape[1] != len(self.feature_names):
            raise ValueError(
                f"background has {Xa.shape[1]} columns but {len(self.feature_names)} "
                "feature names were given")
        rng = np.random.default_rng(self.seed)
        if y is not None:
            ya = np.asarray(y, dtype=np.int64).ravel()
            per = max(1, self.max_background // max(len(np.unique(ya)), 1))
            picked = []
            for cls in np.unique(ya):
                idx = np.nonzero(ya == cls)[0]
                picked.append(idx if idx.size <= per
                              else rng.choice(idx, per, replace=False))
            keep = np.concatenate(picked)
            self.notes.append("class-balanced background, so contributions are relative "
                              "to an average across classes rather than to typical "
                              "traffic")
        elif Xa.shape[0] > self.max_background:
            keep = rng.choice(Xa.shape[0], self.max_background, replace=False)
        else:
            keep = np.arange(Xa.shape[0])
        self.background_ = Xa[keep]
        self.background_median_ = np.median(self.background_, axis=0)
        self._explainer = None                          # force a rebuild
        log.debug("background set to %d row(s)", len(self.background_))
        return self

    def _require_background(self) -> np.ndarray:
        if self.background_ is None:
            raise RuntimeError(
                "no background set. Call set_background(X_train) first - SHAP values "
                "are meaningless without a reference distribution to compare against.")
        return self.background_

    # -- SHAP construction ---------------------------------------------------

    def _build_shap(self) -> Optional[Any]:
        """Pick the fastest applicable SHAP explainer, or return ``None``."""
        if self._explainer is not None:
            return self._explainer
        if not self.prefer_shap:
            return None
        try:
            import shap
        except Exception as exc:                        # noqa: BLE001
            # Turn the flag off rather than just returning None. A failed import is not
            # cached in sys.modules, so without this every explained row re-walks
            # sys.path looking for a package that is not there - and, more visibly, logs
            # the same notice again. Explaining 1,500 rows one at a time printed this
            # fifteen times before the flag was added.
            self.prefer_shap = False
            global _SHAP_IMPORT_LOGGED
            if not _SHAP_IMPORT_LOGGED:
                _SHAP_IMPORT_LOGGED = True
                log.info("shap is not importable (%s); using permutation importance and "
                         "occlusion instead (pip install shap for exact attributions)",
                         type(exc).__name__)
            return None

        background = self._require_background()
        inner = getattr(self.model, "model", self.model)

        # Tree models first: TreeExplainer is exact and thousands of times faster than
        # KernelExplainer, which is the difference between 20 seconds and 6 hours.
        if getattr(self.model, "is_tree", False):
            try:
                self._explainer = shap.TreeExplainer(inner)
                self._explainer_kind = "shap.TreeExplainer"
                return self._explainer
            except Exception as exc:                    # noqa: BLE001
                log.debug("TreeExplainer refused this model: %s", exc)

        if getattr(self.model, "is_deep", False):
            for name in ("GradientExplainer", "DeepExplainer"):
                try:
                    reshape = getattr(self.model, "_reshape", None)
                    data = reshape(background) if reshape else background
                    self._explainer = getattr(shap, name)(inner, data)
                    self._explainer_kind = f"shap.{name}"
                    return self._explainer
                except Exception as exc:                # noqa: BLE001
                    log.debug("%s refused this model: %s", name, exc)

        if getattr(self.model, "family", "") == "linear":
            try:
                self._explainer = shap.LinearExplainer(inner, background)
                self._explainer_kind = "shap.LinearExplainer"
                return self._explainer
            except Exception as exc:                    # noqa: BLE001
                log.debug("LinearExplainer refused this model: %s", exc)

        # KernelExplainer works on anything but is brutally slow, so it is last and the
        # cost is logged rather than discovered.
        try:
            summary = background
            if hasattr(shap, "kmeans") and len(background) > 50:
                # k-means summarisation of the background is the standard KernelExplainer
                # speed-up; 50 weighted centroids stand in for 200 rows.
                try:
                    summary = shap.kmeans(background, 50)
                except Exception:                       # noqa: BLE001
                    summary = background
            self._explainer = shap.KernelExplainer(self.model.predict_proba, summary)
            self._explainer_kind = "shap.KernelExplainer"
            self.notes.append("KernelExplainer is model-agnostic but approximate and "
                              "slow; explanations were computed on a subsample")
            log.warning("falling back to shap.KernelExplainer for %s - this is slow, so "
                        "explanation rows are capped",
                        getattr(self.model, "key", "model"))
            return self._explainer
        except Exception as exc:                        # noqa: BLE001
            log.info("no SHAP explainer applies to this model (%s); using permutation "
                     "importance instead", exc)
            return None

    def _shap_matrix(self, X: np.ndarray) -> Optional[np.ndarray]:
        """``(n_rows, n_features, n_classes)`` SHAP values, or ``None``."""
        explainer = self._build_shap()
        if explainer is None:
            return None
        reshape = getattr(self.model, "_reshape", None)
        data = reshape(X) if (reshape and getattr(self.model, "is_deep", False)) else X
        try:
            try:
                raw = explainer.shap_values(data, check_additivity=False)
            except TypeError:
                # Not every explainer accepts check_additivity.
                raw = explainer.shap_values(data)
        except Exception as exc:                        # noqa: BLE001
            log.warning("SHAP evaluation failed (%s); falling back", exc)
            self._explainer = None
            self.prefer_shap = False
            self.notes.append(f"SHAP failed at evaluation time ({type(exc).__name__}); "
                              "the fallback method was used instead")
            return None
        return normalise_shap_values(
            raw, n_rows=X.shape[0], n_features=X.shape[1],
            n_classes=len(self.class_names))

    # -- global --------------------------------------------------------------

    def global_explanation(
        self,
        X: Any,
        y: Any = None,
        *,
        max_rows: int = 2000,
        permutation_metric: str = "macro_f1",
        n_repeats: int = 3,
        quiet: bool = False,
    ) -> GlobalExplanation:
        """Importance across many rows, by SHAP when possible."""
        Xa = np.ascontiguousarray(
            X.to_numpy() if hasattr(X, "to_numpy") else X, dtype=np.float32)
        if Xa.shape[1] != len(self.feature_names):
            raise ValueError(f"X has {Xa.shape[1]} columns but "
                             f"{len(self.feature_names)} feature names were given")
        if self.background_ is None:
            self.set_background(Xa)

        rng = np.random.default_rng(self.seed)
        ya = np.asarray(y, dtype=np.int64).ravel() if y is not None else None
        if Xa.shape[0] > max_rows:
            keep = (stratified_subsample(ya, max_rows, rng, floor=40) if ya is not None
                    else rng.choice(Xa.shape[0], max_rows, replace=False))
            Xa = Xa[keep]
            if ya is not None:
                ya = ya[keep]

        started = time.perf_counter()
        with stage(log, f"explaining {getattr(self.model, 'key', 'model')} on "
                        f"{len(Xa):,} row(s)", quiet=quiet):
            values = self._shap_matrix(Xa)
            if values is not None:
                # Mean |SHAP| over rows, then summed over classes. Absolute first, then
                # mean: signed values cancel across rows and would report near-zero
                # importance for a feature that strongly separates two classes in
                # opposite directions.
                per_class = np.abs(values).mean(axis=0).T          # (classes, features)
                importance = per_class.sum(axis=0)
                method = self._explainer_kind
                notes = list(self.notes)
            else:
                if ya is None:
                    raise ValueError(
                        "SHAP is unavailable and permutation importance needs labels. "
                        "Pass y, or install shap.")
                importance, per_class, info = permutation_importance(
                    self.model.predict_proba, Xa, ya, class_names=self.class_names,
                    metric=permutation_metric, n_repeats=n_repeats, seed=self.seed,
                    feature_names=self.feature_names)
                method = f"permutation ({permutation_metric})"
                notes = list(self.notes) + [
                    f"permutation importance over {info['n_repeats']} repeat(s); values "
                    f"are the drop in {permutation_metric} from a baseline of "
                    f"{info['baseline']:.4f}",
                    "correlated features share credit under permutation, so these "
                    "values understate importance within correlated groups",
                ]
                if info["negative_features"]:
                    notes.append("shuffling these features *improved* the score, which "
                                 "means they contribute nothing: "
                                 + ", ".join(info["negative_features"][:6]))

        return GlobalExplanation(
            model=getattr(self.model, "key", "model"),
            feature_names=list(self.feature_names),
            class_names=list(self.class_names),
            importance=np.asarray(importance, dtype=np.float64),
            per_class=None if per_class is None
            else np.asarray(per_class, dtype=np.float64),
            method=method,
            rows_explained=int(len(Xa)),
            background_rows=int(len(self.background_)),
            seconds=time.perf_counter() - started,
            notes=notes,
        )

    # -- local ---------------------------------------------------------------

    def explain_row(
        self,
        x: Any,
        *,
        class_index: Optional[int] = None,
        raw_values: Optional[Sequence[float]] = None,
    ) -> LocalExplanation:
        """Explain a single prediction.

        ``class_index`` explains a class other than the predicted one, which is how the
        app answers "why *not* BENIGN?" - often the more useful question.
        """
        row = np.asarray(x.to_numpy() if hasattr(x, "to_numpy") else x,
                         dtype=np.float32).reshape(1, -1)
        if row.shape[1] != len(self.feature_names):
            raise ValueError(f"row has {row.shape[1]} values but "
                             f"{len(self.feature_names)} features were expected")
        if self.background_ is None:
            raise RuntimeError("call set_background() before explaining a row")

        proba = np.asarray(self.model.predict_proba(row), dtype=np.float64).ravel()
        target = int(np.argmax(proba)) if class_index is None else int(class_index)
        order = np.argsort(-proba)
        runner = int(order[1]) if len(order) > 1 else target

        values = self._shap_matrix(row)
        additivity: Optional[float] = None
        if values is not None:
            col = min(target, values.shape[2] - 1)
            contributions = values[0, :, col]
            base = float(_base_value(self._explainer, target, proba, values, col))
            method = self._explainer_kind
            # SHAP's defining property: base + sum(contributions) = f(x). Checking it is
            # the cheapest possible guard against having mis-transposed the array.
            additivity = abs(base + contributions.sum() - proba[target])
        else:
            contributions, base = self._occlusion(row, target)
            method = "occlusion in log-odds (feature replaced by its background median)"

        raw = list(raw_values) if raw_values is not None else None
        items = [
            FeatureContribution(
                name=self.feature_names[i],
                value=float(row[0, i]),
                contribution=float(contributions[i]),
                raw_value=float(raw[i]) if raw is not None and i < len(raw) else None,
            )
            for i in range(len(self.feature_names))
        ]
        return LocalExplanation(
            model=getattr(self.model, "key", "model"),
            predicted_index=target,
            predicted_class=self.class_names[target]
            if target < len(self.class_names) else str(target),
            confidence=float(proba[target]),
            contributions=items,
            base_value=float(base),
            method=method,
            class_names=list(self.class_names),
            probabilities=proba,
            runner_up=self.class_names[runner] if runner < len(self.class_names) else "",
            runner_up_confidence=float(proba[runner]),
            additivity_error=additivity,
        )

    def _occlusion(self, row: np.ndarray, target: int) -> Tuple[np.ndarray, float]:
        """Ablation attribution: how far does the prediction move without this feature?

        For each feature, substitute the background median and re-predict. The signed
        change is that feature's contribution. This is one batched forward pass of
        ``n_features + 1`` rows, so it is fast enough for the app's interactive path even
        on a deep model.

        Measured in **log-odds, not probability**, and that detail is the difference
        between a useful panel and an empty one. A tuned gradient-boosted model on
        CICIDS2017 routinely predicts a DoS flow at 0.99999; knocking out any single
        feature moves that to 0.9999, a probability delta of 9e-5 that rounds to 0.00000
        in every column of the display. The same change in log-odds is 11.5 -> 9.2, which
        ranks and renders perfectly well. Since the logit transform is monotone, the sign
        of every contribution and the ordering between features are identical to what
        probability space would give - only the crushed dynamic range is repaired. This is
        the same reasoning behind SHAP's ``link="logit"`` option.

        It is not SHAP: it measures each feature in isolation against one reference point,
        rather than averaging over all subsets, so interaction effects are attributed to
        whichever feature is asked about first and the parts do not sum exactly to the
        whole. It is honest about direction and roughly right about magnitude, which is
        what a human reading the app needs.
        """
        median = self.background_median_
        assert median is not None
        n_features = row.shape[1]
        batch = np.repeat(row, n_features + 1, axis=0)
        for j in range(n_features):
            batch[j + 1, j] = median[j]
        proba = np.asarray(self.model.predict_proba(batch), dtype=np.float64)
        target = min(target, proba.shape[1] - 1)
        odds = _log_odds(proba[:, target])
        contributions = odds[0] - odds[1:]
        # The baseline an occlusion explanation implies is the all-median row.
        neutral = self.model.predict_proba(median.reshape(1, -1).astype(row.dtype))
        base = float(_log_odds(
            np.asarray(neutral, dtype=np.float64).ravel()[[target]])[0])
        return contributions, base

    # -- checks --------------------------------------------------------------

    def verify_additivity(self, X: Any, *, tolerance: float = 1e-3) -> Dict[str, Any]:
        """Check that ``base + sum(SHAP) == predict_proba`` on a few rows.

        A failure here almost always means the explainer was handed the wrong data
        layout - the most common cause being a deep model whose ``(n, f, 1)`` reshape was
        skipped. Worth running once per artifact rather than trusting the plots.
        """
        Xa = np.ascontiguousarray(
            X.to_numpy() if hasattr(X, "to_numpy") else X, dtype=np.float32)[:20]
        values = self._shap_matrix(Xa)
        if values is None:
            return {"checked": False,
                    "reason": "no SHAP explainer; the fallback is not additive by design"}
        proba = np.asarray(self.model.predict_proba(Xa), dtype=np.float64)
        errors = []
        for i in range(len(Xa)):
            k = int(proba[i].argmax())
            col = min(k, values.shape[2] - 1)
            base = _base_value(self._explainer, k, proba[i], values, col)
            errors.append(abs(base + values[i, :, col].sum() - proba[i, k]))
        worst = float(max(errors)) if errors else 0.0
        ok = worst <= tolerance
        if not ok:
            log.warning("SHAP additivity is off by up to %.3g, above the %.3g tolerance. "
                        "For a model whose output is a margin rather than a probability "
                        "this is expected; otherwise the explainer input layout is "
                        "suspect.", worst, tolerance)
        return {"checked": True, "rows": len(Xa), "worst_error": worst,
                "tolerance": tolerance, "passed": bool(ok),
                "explainer": self._explainer_kind}


def _log_odds(p: np.ndarray) -> np.ndarray:
    """``log(p / (1 - p))``, clipped so a saturated probability cannot become infinite.

    Tree ensembles and softmax layers both return exact 1.0 and exact 0.0 in float64 once
    a prediction is confident enough, and ``log(1/0)`` would put ``inf`` into the
    explanation. Clipping at 1e-12 caps the scale at +-27.6, which is far wider than any
    real contribution and keeps every value finite and comparable.
    """
    clipped = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return np.log(clipped / (1.0 - clipped))


def _base_value(explainer: Any, target: int, proba: np.ndarray,
                values: np.ndarray, col: int) -> float:
    """Best available expected value for the target class."""
    expected = getattr(explainer, "expected_value", None)
    if expected is None:
        return float(proba[target] - values[..., col].sum(axis=-1).mean())
    arr = np.atleast_1d(np.asarray(expected, dtype=np.float64))
    return float(arr[target] if target < arr.size else arr[0])
