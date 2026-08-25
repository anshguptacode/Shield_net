"""Feature selection: correlation pruning, four rankers, and rank fusion.

CICIDS2017 has 77 usable features and many of them measure nearly the same thing -
``Total Length of Fwd Packets`` and ``Fwd Packet Length Mean`` differ only by the packet
count, and there are several such families. Feeding all 77 to a tree ensemble works but
makes the SHAP story unreadable, because importance gets split arbitrarily between
near-duplicates. So the pipeline reduces to a compact subset first.

Four rankers, deliberately chosen to disagree
---------------------------------------------
``anova_f``
    One-way ANOVA F. Linear, per-feature, instant. Blind to interactions.
``chi2``
    Dependence between a non-negative feature and the class. Sensitive to a feature
    being *concentrated* in one class even when its mean barely moves.
``mutual_info``
    Captures non-linear and non-monotonic dependence, which the first two cannot see.
``rfe``
    The only multivariate ranker here: it removes features in the presence of the
    others, so it is the one that notices redundancy.

Their ranks are fused by **mean reciprocal rank** rather than by averaging scores,
because the scores are on incomparable scales (an F statistic in the thousands next to a
mutual information in nats). MRR also has the property we want: a feature ranked 1st by
one method and 40th by three others still places well, so a signal only one ranker can
see is not thrown away.

Nothing here imports sklearn at module level. ``anova_f`` and ``chi2`` are written out in
numpy, ``mutual_info`` falls back to a quantile-binned estimator when sklearn is absent,
and ``rfe`` reports itself as skipped rather than crashing the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .evaluate import stratified_subsample
from .logging_utils import get_logger, human_count, stage

log = get_logger(__name__)

__all__ = ["SelectionReport", "select_features", "rank_features",
           "prune_correlated", "anova_f_scores", "chi2_scores",
           "mutual_info_scores", "rfe_ranking", "RANKERS"]


# ---------------------------------------------------------------------------
# individual rankers
# ---------------------------------------------------------------------------

def anova_f_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """One-way ANOVA F statistic per feature, vectorised over all features at once.

    F = (between-group variance) / (within-group variance). Written out rather than
    imported so the whole selection stage runs on numpy alone; the arithmetic is the
    textbook decomposition and matches ``sklearn.feature_selection.f_classif`` to
    floating-point tolerance.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    n, d = X.shape
    classes = np.unique(y)
    k = len(classes)
    if k < 2:
        return np.zeros(d)

    grand_mean = X.mean(axis=0)
    ss_between = np.zeros(d)
    ss_within = np.zeros(d)
    for cls in classes:
        rows = X[y == cls]
        n_c = len(rows)
        mean_c = rows.mean(axis=0)
        ss_between += n_c * (mean_c - grand_mean) ** 2
        ss_within += ((rows - mean_c) ** 2).sum(axis=0)

    df_between, df_within = k - 1, n - k
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (ss_between / df_between) / (ss_within / df_within)
    # A constant feature gives 0/0. It carries no information, so score it 0.
    return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)


def chi2_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Chi-squared statistic between each non-negative feature and the class.

    The inputs are min-max rescaled to [0, 1] by the caller: chi2 treats the feature
    value as a frequency-like quantity, so a negative value is meaningless and sklearn
    raises on one. Anything still slightly negative from floating-point error is
    clamped rather than raising, because failing a whole run over -1e-17 is absurd.
    """
    X = np.clip(np.asarray(X, dtype=np.float64), 0.0, None)
    y = np.asarray(y, dtype=int)
    classes = np.unique(y)
    if len(classes) < 2:
        return np.zeros(X.shape[1])

    # One-hot the target, then observed[c, f] = total of feature f within class c.
    onehot = np.zeros((len(y), len(classes)), dtype=np.float64)
    for i, cls in enumerate(classes):
        onehot[y == cls, i] = 1.0

    observed = onehot.T @ X                      # (k, d)
    feature_total = X.sum(axis=0)                # (d,)
    class_prob = onehot.mean(axis=0)             # (k,)
    expected = np.outer(class_prob, feature_total)

    with np.errstate(divide="ignore", invalid="ignore"):
        terms = (observed - expected) ** 2 / expected
    return np.nan_to_num(terms.sum(axis=0), nan=0.0, posinf=0.0, neginf=0.0)


def mutual_info_scores(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 42,
    bins: int = 32,
) -> Tuple[np.ndarray, str]:
    """Mutual information I(feature; class) in nats. Returns ``(scores, method_used)``.

    Prefers sklearn's ``mutual_info_classif``, which uses the Ross (2014) k-nearest-
    neighbour estimator and handles continuous features properly. Without sklearn it
    falls back to quantile binning: each feature is discretised into up to *bins*
    equal-frequency buckets and the discrete MI is computed exactly. The fallback
    slightly underestimates MI for smooth features but preserves the ranking, which is
    all this stage needs.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    try:
        from sklearn.feature_selection import mutual_info_classif
        scores = mutual_info_classif(X, y, random_state=seed)
        return np.asarray(scores, dtype=np.float64), "sklearn-knn"
    except ImportError:
        log.info("scikit-learn unavailable; using the quantile-binned mutual "
                 "information estimator (rankings agree, absolute values are lower)")

    n = len(y)
    class_ids, class_counts = np.unique(y, return_counts=True)
    p_y = class_counts / n
    h_y = -(p_y * np.log(p_y)).sum()

    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        col = X[:, j]
        edges = np.unique(np.quantile(col, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:                       # constant or near-constant
            continue
        codes = np.clip(np.searchsorted(edges, col, side="right") - 1, 0, len(edges) - 2)
        joint = np.zeros((len(edges) - 1, len(class_ids)), dtype=np.float64)
        for i, cls in enumerate(class_ids):
            np.add.at(joint[:, i], codes[y == cls], 1.0)
        joint /= n
        p_x = joint.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = joint / (p_x * p_y[None, :])
            terms = joint * np.log(ratio)
        out[j] = float(np.nansum(np.where(joint > 0, terms, 0.0)))
    # MI cannot exceed H(Y); numerical noise occasionally nudges past it.
    return np.clip(out, 0.0, h_y), f"binned-{bins}"


def rfe_ranking(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_features: int,
    estimator: str = "random_forest",
    step: float = 0.1,
    seed: int = 42,
    n_jobs: int = -1,
) -> Optional[np.ndarray]:
    """Recursive feature elimination. Returns a rank array (1 = best) or ``None``.

    ``None`` means the dependency is missing; the caller records RFE as skipped and
    fuses the remaining rankers rather than aborting a long run over an optional stage.
    RFE is by far the most expensive ranker - it refits the estimator once per
    elimination round - which is why the caller subsamples the rows first.
    """
    try:
        from sklearn.feature_selection import RFE
    except ImportError:
        log.warning("RFE skipped: scikit-learn is not installed")
        return None

    model = _rfe_estimator(estimator, seed=seed, n_jobs=n_jobs)
    if model is None:
        return None

    log.info("running RFE (%s, step=%g) down to %d features on %s rows - the slow one",
             estimator, step, n_features, human_count(len(X)))
    selector = RFE(model, n_features_to_select=n_features, step=step)
    selector.fit(X, y)
    return np.asarray(selector.ranking_, dtype=float)


def _rfe_estimator(name: str, *, seed: int, n_jobs: int):
    """Build a cheap-but-honest estimator for RFE, or None if unavailable."""
    if name in {"random_forest", "rf"}:
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError:
            return None
        # Shallow and few trees on purpose: RFE refits this many times and we only
        # need a stable importance ordering, not a good classifier.
        return RandomForestClassifier(
            n_estimators=60, max_depth=14, min_samples_leaf=2, n_jobs=n_jobs,
            random_state=seed, class_weight="balanced_subsample",
        )
    if name in {"logistic_regression", "logreg"}:
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            return None
        return LogisticRegression(max_iter=400, random_state=seed, n_jobs=n_jobs)
    if name in {"extra_trees", "et"}:
        try:
            from sklearn.ensemble import ExtraTreesClassifier
        except ImportError:
            return None
        return ExtraTreesClassifier(n_estimators=80, n_jobs=n_jobs, random_state=seed)
    log.warning("unknown rfe_estimator %r; RFE skipped", name)
    return None


#: Ranker name -> whether a higher raw score means a better feature.
RANKERS: Dict[str, bool] = {
    "anova_f": True,
    "chi2": True,
    "mutual_info": True,
    "rfe": False,          # RFE returns a rank, where 1 is best
}


# ---------------------------------------------------------------------------
# correlation pruning
# ---------------------------------------------------------------------------

def prune_correlated(
    X: pd.DataFrame,
    *,
    threshold: float = 0.95,
    priority: Optional[Mapping[str, float]] = None,
) -> Tuple[List[str], Dict[str, str]]:
    """Drop one of each near-duplicate feature pair before ranking.

    Which one to drop is not arbitrary. Given ``priority`` (here, the univariate ANOVA
    score) the *weaker* member of the pair goes, so the survivor is the more informative
    spelling of the same measurement. Without that, you get whichever happened to come
    first in column order.

    Returns ``(kept_names, {dropped: "correlated 0.99 with <kept>"})``.
    """
    names = [str(c) for c in X.columns]
    if len(names) < 2:
        return names, {}

    values = X.to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(values, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)          # constant column -> undefined corr
    np.fill_diagonal(corr, 0.0)

    rank = priority or {n: 0.0 for n in names}
    # Consider the strongest features first so they become the survivors.
    order = sorted(range(len(names)), key=lambda i: -rank.get(names[i], 0.0))

    kept: List[int] = []
    dropped: Dict[str, str] = {}
    for i in order:
        clash = next((j for j in kept if abs(corr[i, j]) >= threshold), None)
        if clash is None:
            kept.append(i)
        else:
            dropped[names[i]] = (f"|r| = {abs(corr[i, clash]):.3f} with "
                                 f"{names[clash]}")

    kept_names = [names[i] for i in sorted(kept)]   # restore original column order
    if dropped:
        log.info("correlation pruning at |r| >= %.2f removed %d of %d features",
                 threshold, len(dropped), len(names))
        for name, why in list(dropped.items())[:6]:
            log.debug("  drop %s (%s)", name, why)
    return kept_names, dropped


# ---------------------------------------------------------------------------
# fusion
# ---------------------------------------------------------------------------

@dataclass
class SelectionReport:
    """Full audit trail for the selection stage."""

    selected: List[str] = field(default_factory=list)
    ranking: pd.DataFrame = field(default_factory=pd.DataFrame)
    dropped_correlated: Dict[str, str] = field(default_factory=dict)
    methods_run: List[str] = field(default_factory=list)
    methods_skipped: Dict[str, str] = field(default_factory=dict)
    mi_estimator: str = ""
    rows_used: int = 0
    stability: Optional[pd.DataFrame] = None

    @property
    def n_selected(self) -> int:
        return len(self.selected)

    def to_dict(self) -> dict:
        out = {
            "selected": list(self.selected),
            "n_selected": self.n_selected,
            "methods_run": list(self.methods_run),
            "methods_skipped": dict(self.methods_skipped),
            "mi_estimator": self.mi_estimator,
            "rows_used": self.rows_used,
            "dropped_correlated": dict(list(self.dropped_correlated.items())[:20]),
        }
        if self.stability is not None:
            out["stability"] = self.stability.to_dict(orient="records")
        return out

    def render(self, top: int = 25) -> str:
        cols = ["feature", "mrr"] + [f"rank_{m}" for m in self.methods_run]
        cols = [c for c in cols if c in self.ranking.columns]
        if self.stability is not None and "frequency" in self.stability.columns:
            merged = self.ranking.merge(self.stability, on="feature", how="left")
            table = merged[cols + ["frequency"]]
        else:
            table = self.ranking[cols]
        lines = [f"rankers run: {', '.join(self.methods_run)}"
                 + (f"   (MI: {self.mi_estimator})" if self.mi_estimator else "")]
        if self.methods_skipped:
            for m, why in self.methods_skipped.items():
                lines.append(f"skipped {m}: {why}")
        lines.append(f"correlation pruning removed {len(self.dropped_correlated)} "
                     f"feature(s)")
        lines.append("")
        lines.append(table.head(top).to_string(index=False))
        if len(self.ranking) > top:
            lines.append(f"... {len(self.ranking) - top} more")
        lines.append("")
        lines.append(f"selected {self.n_selected}: {', '.join(self.selected)}")
        return "\n".join(lines)


def rank_features(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    methods: Sequence[str] = ("mutual_info", "chi2", "rfe"),
    n_features: int = 25,
    seed: int = 42,
    n_jobs: int = -1,
    rfe_estimator: str = "random_forest",
    rfe_step: float = 0.1,
) -> Tuple[pd.DataFrame, List[str], Dict[str, str], str]:
    """Score every feature with each requested ranker and fuse by mean reciprocal rank.

    Returns ``(ranking_frame, methods_run, methods_skipped, mi_estimator)``.
    """
    names = [str(c) for c in X.columns]
    values = X.to_numpy(dtype=np.float64)
    frame = pd.DataFrame({"feature": names})

    requested = list(dict.fromkeys(methods))     # de-dupe, keep order
    unknown = [m for m in requested if m not in RANKERS and m != "anova_f"]
    if unknown:
        raise ValueError(f"unknown ranker(s) {unknown}; choose from {sorted(RANKERS)}")

    # ANOVA is free and is needed as the correlation-pruning tie-breaker, so it always
    # runs even when it was not requested. It is only *reported* if requested.
    anova = anova_f_scores(values, y)

    run: List[str] = []
    skipped: Dict[str, str] = {}
    mi_estimator = ""

    for method in requested:
        if method == "anova_f":
            scores, higher_better = anova, True
        elif method == "chi2":
            # chi2 needs non-negative input; rescale per feature to [0, 1].
            lo = values.min(axis=0)
            span = values.max(axis=0) - lo
            span = np.where(np.abs(span) < 1e-12, 1.0, span)
            scores, higher_better = chi2_scores((values - lo) / span, y), True
        elif method == "mutual_info":
            scores, mi_estimator = mutual_info_scores(values, y, seed=seed)
            higher_better = True
        elif method == "rfe":
            ranks = rfe_ranking(values, y, n_features=n_features,
                                estimator=rfe_estimator, step=rfe_step, seed=seed,
                                n_jobs=n_jobs)
            if ranks is None:
                skipped[method] = "scikit-learn not installed"
                continue
            scores, higher_better = ranks, False
        else:                                    # pragma: no cover - guarded above
            continue

        frame[f"score_{method}"] = scores
        # ascending rank: 1 is best under either sign convention
        frame[f"rank_{method}"] = (
            pd.Series(scores).rank(ascending=not higher_better, method="min")
        )
        run.append(method)

    if not run:
        raise RuntimeError(
            "every requested ranker was skipped. Set features.methods to include at "
            "least one of anova_f / chi2 / mutual_info, which need only numpy."
        )

    rank_cols = [f"rank_{m}" for m in run]
    frame["mrr"] = (1.0 / frame[rank_cols]).mean(axis=1)
    # Deterministic ordering: break MRR ties by ANOVA score, then by name, so two runs
    # on the same data select the same features rather than a coin-flip subset.
    frame["_anova"] = anova
    frame = (frame.sort_values(["mrr", "_anova", "feature"],
                               ascending=[False, False, True])
                  .drop(columns="_anova")
                  .reset_index(drop=True))
    frame.insert(0, "position", np.arange(1, len(frame) + 1))
    return frame, run, skipped, mi_estimator


def select_features(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_features: int = 25,
    methods: Sequence[str] = ("mutual_info", "chi2", "rfe"),
    correlation_threshold: float = 0.95,
    ranking_sample_rows: int = 60_000,
    stability_runs: int = 0,
    seed: int = 42,
    n_jobs: int = -1,
    rfe_estimator: str = "random_forest",
    rfe_step: float = 0.1,
) -> SelectionReport:
    """Prune correlated features, rank the survivors, keep the top *n_features*.

    Fitted on the **training split only**. Selecting features using the test set is a
    leak that is easy to miss because it does not change any single number by much - it
    just makes every number slightly and unfixably optimistic.

    Subsampling for ranking is stratified, so the rare classes are not sampled out of
    existence before the rankers ever see them.
    """
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
    if n_features < 1:
        raise ValueError("n_features must be >= 1")

    with stage(log, "feature selection") as st:
        X_s, y_s = _stratified_subsample(X, y, ranking_sample_rows, seed)

        # ANOVA first, purely to decide which member of a correlated pair survives.
        priority = dict(zip((str(c) for c in X_s.columns),
                            anova_f_scores(X_s.to_numpy(dtype=np.float64), y_s)))
        kept, dropped = prune_correlated(X_s, threshold=correlation_threshold,
                                         priority=priority)

        ranking, run, skipped, mi_est = rank_features(
            X_s[kept], y_s, methods=methods, n_features=min(n_features, len(kept)),
            seed=seed, n_jobs=n_jobs, rfe_estimator=rfe_estimator, rfe_step=rfe_step,
        )

        if n_features > len(kept):
            log.warning("asked for %d features but only %d survived correlation "
                        "pruning; keeping all of them", n_features, len(kept))
        selected = ranking["feature"].head(min(n_features, len(kept))).tolist()
        ranking["selected"] = ranking["feature"].isin(selected)

        report = SelectionReport(
            selected=selected, ranking=ranking, dropped_correlated=dropped,
            methods_run=run, methods_skipped=skipped, mi_estimator=mi_est,
            rows_used=len(X_s),
        )

        if stability_runs > 0:
            report.stability = _stability(
                X_s[kept], y_s, n_features=min(n_features, len(kept)),
                methods=[m for m in methods if m != "rfe"] or ["anova_f"],
                runs=stability_runs, seed=seed, n_jobs=n_jobs,
            )
            merged = report.stability.set_index("feature")["frequency"]
            fragile = [f for f in selected if merged.get(f, 1.0) < 0.6]
            if fragile:
                log.warning("%d selected feature(s) survived fewer than 60%% of "
                            "bootstrap resamples - treat their SHAP importance as "
                            "provisional: %s", len(fragile), fragile[:6])

        st["summary"] = (f"{len(selected)} of {len(X.columns)} features "
                         f"({len(dropped)} pruned as correlated)")
    return report


def _stratified_subsample(
    X: pd.DataFrame, y: np.ndarray, max_rows: int, seed: int
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Subsample to *max_rows* keeping every class, for the expensive rankers.

    A thin frame-shaped wrapper over :func:`shieldnet.evaluate.stratified_subsample`.
    The floor is 30 rather than 200: a class sampled down to 3 rows contributes noise to
    mutual information and can make chi2 pick a feature that separates 3 specific flows,
    but ranking does not need the larger floor that a cross-validated hyper-parameter
    search does.
    """
    if max_rows <= 0 or len(X) <= max_rows:
        return X, np.asarray(y)

    y = np.asarray(y)
    # Sorted, so the subsample keeps the original row order - it makes no difference to
    # any ranker, but it keeps saved rankings diffable between runs.
    idx = np.sort(stratified_subsample(y, max_rows, floor=30, seed=seed))
    log.info("ranking on a stratified subsample of %s rows (from %s)",
             human_count(len(idx)), human_count(len(X)))
    return X.iloc[idx].reset_index(drop=True), y[idx]


def _stability(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_features: int,
    methods: Sequence[str],
    runs: int,
    seed: int,
    n_jobs: int,
) -> pd.DataFrame:
    """How often each feature is selected across bootstrap resamples.

    RFE is excluded from the loop: refitting a forest *runs* times to answer a
    diagnostic question is not a good trade, and the cheap rankers already reveal the
    instability we are looking for.
    """
    log.info("stability check: re-selecting on %d bootstrap resample(s)", runs)
    counter: Dict[str, int] = {str(c): 0 for c in X.columns}
    rng = np.random.default_rng(seed)
    for r in range(runs):
        rows = rng.choice(len(X), size=len(X), replace=True)
        Xb, yb = X.iloc[rows].reset_index(drop=True), np.asarray(y)[rows]
        if len(np.unique(yb)) < 2:               # pathological resample
            continue
        ranked, _, _, _ = rank_features(Xb, yb, methods=methods,
                                        n_features=n_features,
                                        seed=seed + r + 1, n_jobs=n_jobs)
        for name in ranked["feature"].head(n_features):
            counter[str(name)] += 1
    out = pd.DataFrame({"feature": list(counter),
                        "times_selected": list(counter.values())})
    out["frequency"] = out["times_selected"] / max(runs, 1)
    return out.sort_values("frequency", ascending=False).reset_index(drop=True)
