"""Cleaning, splitting, imputation, scaling and label encoding.

Why the transformers are hand-written instead of imported from sklearn
---------------------------------------------------------------------
``StandardScaler`` and friends would be one import each. But the fitted scaler has to
travel *with* the model into the Streamlit app and be unpickled there, and a pickled
sklearn estimator is only guaranteed to load under the same sklearn minor version that
wrote it. On a student project that gets zipped, emailed, and opened on a lab machine six
weeks later, that is the single most likely way for the demo to die.

So the scaler, the clipper and the median imputer here are plain dataclasses holding
numpy arrays, with the arithmetic written out. They pickle as data, not as code, and the
app needs only numpy and pandas to run inference. ``KNNImputer`` is the one exception -
it has real algorithmic content, so when ``imputer="knn"`` is requested we defer to
sklearn and record that the bundle now carries that dependency.

The order of operations matters and is not arbitrary:

1. **inf -> NaN.** ``Flow Bytes/s`` is bytes divided by duration, and duration is 0 for
   single-packet flows, so the raw files contain literal ``Infinity``. Left alone it
   poisons the mean, the quantiles and every gradient.
2. **Deduplicate.** CICIDS2017 has a large number of byte-identical flow rows. Left in,
   the same row can land in both train and test, which inflates the score.
3. **Split.** Everything after this point is fitted on the training split only.
4. **Drop constants** (measured on train), **clip**, **impute**, **scale**.

Doing (4) before (3) is the classic leak: the test set's own median and variance end up
baked into the transform, and the reported accuracy is a fiction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import schema as sch
from .logging_utils import get_logger, human_count

log = get_logger(__name__)

__all__ = [
    "CleanReport", "clean_frame",
    "LabelCodec", "Clipper", "Scaler", "MedianImputer", "Preprocessor",
    "SplitData", "split_frame", "class_weights",
]


# ---------------------------------------------------------------------------
# cleaning
# ---------------------------------------------------------------------------

@dataclass
class CleanReport:
    """What cleaning changed. Folded into the run manifest for provenance."""

    rows_in: int = 0
    rows_out: int = 0
    inf_replaced: int = 0
    duplicate_rows: int = 0
    all_nan_rows: int = 0
    negative_duration_rows: int = 0
    nan_cells: int = 0
    nan_by_feature: Dict[str, int] = field(default_factory=dict)

    @property
    def rows_dropped(self) -> int:
        return self.rows_in - self.rows_out

    def to_dict(self) -> dict:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_dropped": self.rows_dropped,
            "inf_replaced": self.inf_replaced,
            "duplicate_rows": self.duplicate_rows,
            "all_nan_rows": self.all_nan_rows,
            "negative_duration_rows": self.negative_duration_rows,
            "nan_cells": self.nan_cells,
            "nan_by_feature": dict(sorted(self.nan_by_feature.items(),
                                          key=lambda kv: -kv[1])[:15]),
        }

    def render(self) -> str:
        lines = [
            f"rows in              {human_count(self.rows_in)}",
            f"rows out             {human_count(self.rows_out)}",
            f"  duplicates removed {human_count(self.duplicate_rows)}",
            f"  all-NaN rows       {human_count(self.all_nan_rows)}",
            f"inf cells -> NaN     {human_count(self.inf_replaced)}",
            f"NaN cells to impute  {human_count(self.nan_cells)}",
        ]
        if self.negative_duration_rows:
            lines.append(f"negative durations   {human_count(self.negative_duration_rows)}"
                         "  (kept; the capture tool emits these for malformed flows)")
        if self.nan_by_feature:
            lines.append("missing values by feature:")
            for name, n in sorted(self.nan_by_feature.items(), key=lambda kv: -kv[1])[:8]:
                lines.append(f"    - {name}: {human_count(n)}")
        return "\n".join(lines)


def clean_frame(
    df: pd.DataFrame,
    *,
    drop_duplicates: bool = True,
    label_column: str = sch.LABEL_COLUMN,
) -> Tuple[pd.DataFrame, CleanReport]:
    """Replace infinities, drop duplicate and empty rows. No fitting happens here.

    Deliberately *not* done: filling NaN, dropping constant columns or scaling. Those
    all require statistics, and statistics must come from the training split alone.
    """
    report = CleanReport(rows_in=len(df))
    df = df.copy()

    feature_cols = [c for c in df.columns if c != label_column]
    if not feature_cols:
        raise ValueError("frame has no feature columns to clean")

    block = df[feature_cols]

    # 1. inf -> NaN. Count first so the report is honest about how much was synthetic.
    values = block.to_numpy(dtype="float64", copy=True)
    inf_mask = np.isinf(values)
    report.inf_replaced = int(inf_mask.sum())
    if report.inf_replaced:
        values[inf_mask] = np.nan
        df[feature_cols] = pd.DataFrame(values, index=df.index, columns=feature_cols)
        log.info("replaced %s infinite value(s) with NaN (division-by-zero-duration "
                 "artefacts in %s)", human_count(report.inf_replaced),
                 ", ".join(sch.DIVISION_ARTEFACT_FEATURES))
    del values, inf_mask

    # A negative Flow Duration is physically impossible but CICFlowMeter emits a few.
    # They are informative rather than corrupt, so they are counted, not removed.
    if "Flow Duration" in df.columns:
        report.negative_duration_rows = int((df["Flow Duration"] < 0).sum())

    # 2. Rows where every feature is NaN carry no information.
    all_nan = df[feature_cols].isna().all(axis=1)
    report.all_nan_rows = int(all_nan.sum())
    if report.all_nan_rows:
        df = df.loc[~all_nan]

    # 3. Exact duplicates. Compared on features *and* label: two identical feature
    # vectors with different labels are a genuine ambiguity in the data and dropping
    # one of them silently picks a winner, so those are kept and reported later.
    if drop_duplicates:
        before = len(df)
        subset = feature_cols + ([label_column] if label_column in df.columns else [])
        df = df.drop_duplicates(subset=subset, keep="first")
        report.duplicate_rows = before - len(df)
        if report.duplicate_rows:
            log.info("dropped %s duplicate flow row(s) (%.1f%%) - these leak between "
                     "train and test if kept", human_count(report.duplicate_rows),
                     100 * report.duplicate_rows / max(before, 1))

    na_counts = df[feature_cols].isna().sum()
    report.nan_by_feature = {k: int(v) for k, v in na_counts.items() if v}
    report.nan_cells = int(na_counts.sum())
    report.rows_out = len(df)

    df = df.reset_index(drop=True)
    df.attrs["cleaned"] = True
    return df, report


# ---------------------------------------------------------------------------
# label encoding
# ---------------------------------------------------------------------------

@dataclass
class LabelCodec:
    """Class-name <-> integer index, ordered meaningfully rather than alphabetically.

    ``sklearn.LabelEncoder`` sorts alphabetically, which puts "BENIGN" between "Bot" and
    "DDoS" and makes every confusion matrix in the report harder to read than it needs
    to be. Here the order follows :data:`shieldnet.schema.CLASS_SCHEME_13` - benign
    first, then attacks in descending prevalence - so row 0 of a confusion matrix is
    always the benign class and the heavy classes cluster at the top left.

    The order is part of the artifact. If it drifts between training and inference,
    every prediction is silently mislabelled, which is why :class:`ModelBundle`
    cross-checks it.
    """

    classes: List[str] = field(default_factory=list)

    @classmethod
    def fit(cls, y: Sequence[str] | pd.Series) -> "LabelCodec":
        series = pd.Series(y)
        if isinstance(series.dtype, pd.CategoricalDtype):
            observed = set(series.astype(str).unique())
        else:
            observed = set(series.dropna().astype(str).unique())
        if not observed:
            raise ValueError("no labels to fit a codec on")

        ordered = [c for c in sch.CLASS_SCHEME_13 if c in observed]
        # Anything unexpected (a new mirror, a typo we failed to normalise) goes last
        # rather than being dropped - losing rows quietly is worse than an odd order.
        extra = sorted(observed - set(ordered))
        if extra:
            log.warning("label(s) outside the 13-class scheme kept at the end of the "
                        "class order: %s", extra)
        return cls(classes=ordered + extra)

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def mapping(self) -> Dict[str, int]:
        return {c: i for i, c in enumerate(self.classes)}

    def transform(self, y: Sequence[str] | pd.Series) -> np.ndarray:
        lookup = self.mapping
        series = pd.Series(y).astype(str)
        codes = series.map(lookup)
        if codes.isna().any():
            unseen = sorted(series[codes.isna()].unique())
            raise ValueError(
                f"label(s) not seen during fitting: {unseen[:5]}. Known classes: "
                f"{self.classes}. This usually means the training chunk and the "
                "evaluation data were built with different `data.merge_rare` settings - "
                "one collapsed Heartbleed, Infiltration and SQL Injection into "
                f"'{sch.RARE_CLASS_NAME}' and the other did not. Check it in the config "
                "file you passed, or run `shieldnet config` to print the resolved value."
            )
        return codes.to_numpy(dtype=np.int64)

    def inverse_transform(self, codes: Sequence[int] | np.ndarray) -> np.ndarray:
        arr = np.asarray(codes, dtype=int)
        if arr.size and (arr.min() < 0 or arr.max() >= self.n_classes):
            raise ValueError(f"code out of range for {self.n_classes} classes")
        table = np.asarray(self.classes, dtype=object)
        return table[arr]

    def family_of(self, name: str) -> str:
        return sch.family_of(name)

    def to_dict(self) -> dict:
        return {"classes": list(self.classes)}


# ---------------------------------------------------------------------------
# numeric transformers (pickle as plain data, no sklearn at load time)
# ---------------------------------------------------------------------------

@dataclass
class Clipper:
    """Two-sided winsoriser fitted on training quantiles.

    Necessary because a handful of CICIDS2017 flows report rates around 1e9 while the
    median is near 1e3. Min-max scaling with that outlier present maps 99.99% of rows
    into the first thousandth of the [0, 1] range, and chi2 feature scores, the MLP and
    the convolutional models all degrade badly. Clipping to the 0.9995 quantile keeps the
    outlier *ranked* highest without letting its magnitude dominate.
    """

    lower: Optional[np.ndarray] = None
    upper: Optional[np.ndarray] = None
    quantile: Optional[float] = None

    @classmethod
    def fit(cls, X: np.ndarray, quantile: Optional[float] = 0.9995) -> "Clipper":
        if quantile is None:
            return cls(None, None, None)
        if not 0.5 < quantile < 1.0:
            raise ValueError(f"clip_quantile must be in (0.5, 1.0), got {quantile}")
        upper = np.nanquantile(X, quantile, axis=0)
        lower = np.nanquantile(X, 1.0 - quantile, axis=0)
        # An all-NaN column yields NaN bounds; np.clip with NaN bounds silently returns
        # NaN, wiping the column. Neutralise those before they do damage.
        upper = np.where(np.isnan(upper), np.inf, upper)
        lower = np.where(np.isnan(lower), -np.inf, lower)
        degenerate = lower > upper
        if degenerate.any():                      # possible with heavy ties
            lower[degenerate] = -np.inf
        return cls(lower=lower, upper=upper, quantile=quantile)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.lower is None or self.upper is None:
            return X
        return np.clip(X, self.lower, self.upper)

    @property
    def enabled(self) -> bool:
        return self.lower is not None


@dataclass
class MedianImputer:
    """Column-wise median fill. Medians are also kept for use as inference defaults."""

    medians: Optional[np.ndarray] = None

    @classmethod
    def fit(cls, X: np.ndarray) -> "MedianImputer":
        with np.errstate(all="ignore"):
            med = np.nanmedian(X, axis=0)
        # A feature that is NaN for every training row has no median. Zero is the only
        # defensible fill, and the column is almost certainly constant-dropped anyway.
        n_empty = int(np.isnan(med).sum())
        if n_empty:
            log.warning("%d feature(s) were NaN for every training row; imputing 0",
                        n_empty)
            med = np.where(np.isnan(med), 0.0, med)
        return cls(medians=med.astype(np.float64))

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.medians is None:
            raise RuntimeError("MedianImputer.transform called before fit")
        out = np.array(X, dtype=np.float64, copy=True)
        mask = np.isnan(out)
        if mask.any():
            out[mask] = np.take(self.medians, np.where(mask)[1])
        return out


@dataclass
class Scaler:
    """standard | minmax | robust | none, as arithmetic over stored numpy arrays.

    Every branch stores a ``centre`` and a ``spread`` and applies ``(X - centre) /
    spread``, so transform is one expression regardless of kind. Zero spread (a constant
    column that survived) is replaced by 1.0: dividing by it would produce inf, and
    mapping a constant column to 0 is the correct, harmless answer.
    """

    kind: str = "standard"
    centre: Optional[np.ndarray] = None
    spread: Optional[np.ndarray] = None
    n_features_in_: int = 0

    _KINDS = ("standard", "minmax", "robust", "none")

    @classmethod
    def fit(cls, X: np.ndarray, kind: str = "standard") -> "Scaler":
        if kind not in cls._KINDS:
            raise ValueError(f"unknown scaler {kind!r}; choose from {cls._KINDS}")
        X = np.asarray(X, dtype=np.float64)
        if kind == "none":
            n = X.shape[1]
            return cls(kind, np.zeros(n), np.ones(n), n)
        if kind == "standard":
            centre = X.mean(axis=0)
            spread = X.std(axis=0)
        elif kind == "minmax":
            centre = X.min(axis=0)
            spread = X.max(axis=0) - centre
        else:  # robust
            centre = np.median(X, axis=0)
            q75, q25 = np.percentile(X, [75, 25], axis=0)
            spread = q75 - q25
        spread = np.where(~np.isfinite(spread) | (np.abs(spread) < 1e-12), 1.0, spread)
        return cls(kind, centre.astype(np.float64), spread.astype(np.float64),
                   X.shape[1])

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.centre is None or self.spread is None:
            raise RuntimeError("Scaler.transform called before fit")
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"scaler was fitted on {self.n_features_in_} features but received "
                f"{X.shape[1]}. The feature list and the scaler must come from the "
                "same artifact bundle."
            )
        return (X - self.centre) / self.spread

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) * self.spread + self.centre


# ---------------------------------------------------------------------------
# the composed preprocessor
# ---------------------------------------------------------------------------

@dataclass
class Preprocessor:
    """Fitted on the training split, applied everywhere else.

    Holds, in application order: the input feature list, which columns were dropped as
    constant, the clipper, the imputer, and the scaler. Also the medians in *raw* units,
    which the Streamlit app uses to pre-fill the manual-entry form and to substitute for
    features a user's CSV happens to be missing.
    """

    feature_names_in_: List[str] = field(default_factory=list)
    dropped_constant: List[str] = field(default_factory=list)
    clipper: Clipper = field(default_factory=Clipper)
    imputer: MedianImputer = field(default_factory=MedianImputer)
    scaler: Scaler = field(default_factory=Scaler)
    knn_imputer: object = None          # sklearn KNNImputer when imputer="knn"
    raw_medians: Dict[str, float] = field(default_factory=dict)
    schema_version: str = sch.SCHEMA_VERSION

    # -- fitting -------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        X: pd.DataFrame,
        *,
        drop_constant: bool = True,
        clip_quantile: Optional[float] = 0.9995,
        scaler: str = "standard",
        imputer: str = "median",
        knn_neighbours: int = 5,
    ) -> "Preprocessor":
        if X.empty:
            raise ValueError("cannot fit a preprocessor on an empty frame")

        names_in = [str(c) for c in X.columns]
        values = X.to_numpy(dtype=np.float64, copy=True)

        # Raw medians before any transformation, for form defaults and gap filling.
        with np.errstate(all="ignore"):
            raw_med = np.nanmedian(values, axis=0)
        raw_med = np.where(np.isnan(raw_med), 0.0, raw_med)
        raw_medians = {n: float(v) for n, v in zip(names_in, raw_med)}

        # Constant columns, measured on train. nunique ignores NaN, so a column that is
        # 0-or-NaN counts as one distinct value and is correctly dropped.
        dropped: List[str] = []
        if drop_constant:
            nunique = X.nunique(dropna=True)
            dropped = [str(c) for c in X.columns if nunique[c] <= 1]
            expected = set(sch.KNOWN_CONSTANT_FEATURES)
            surprises = [c for c in dropped if c not in expected]
            missing = [c for c in expected if c in names_in and c not in dropped]
            log.info("dropping %d zero-variance feature(s) measured on the training "
                     "split", len(dropped))
            if surprises:
                log.info("  beyond the %d documented all-zero columns: %s",
                         len(expected), surprises)
            if missing:
                log.info("  documented-constant column(s) that DO vary in this sample "
                         "and are kept: %s", missing)

        keep_mask = np.array([n not in set(dropped) for n in names_in], dtype=bool)
        kept = values[:, keep_mask]
        if kept.shape[1] == 0:
            raise ValueError(
                "every feature was dropped as constant. That means the input frame has "
                "one row, or one distinct row - check the working chunk."
            )

        clipper = Clipper.fit(kept, clip_quantile)
        kept = clipper.transform(kept)

        median_imp = MedianImputer.fit(kept)
        knn_obj = None
        if imputer == "knn":
            knn_obj = _fit_knn_imputer(kept, knn_neighbours)
            filled = knn_obj.transform(kept)
        elif imputer == "median":
            filled = median_imp.transform(kept)
        else:
            raise ValueError(f"unknown imputer {imputer!r}; use 'median' or 'knn'")

        scaler_obj = Scaler.fit(filled, scaler)

        obj = cls(
            feature_names_in_=names_in,
            dropped_constant=dropped,
            clipper=clipper,
            imputer=median_imp,
            scaler=scaler_obj,
            knn_imputer=knn_obj,
            raw_medians=raw_medians,
        )
        log.info("preprocessor fitted: %d -> %d features, scaler=%s, imputer=%s",
                 len(names_in), len(obj.feature_names_out_), scaler, imputer)
        return obj

    # -- application ---------------------------------------------------------

    @property
    def feature_names_out_(self) -> List[str]:
        dropped = set(self.dropped_constant)
        return [n for n in self.feature_names_in_ if n not in dropped]

    @property
    def n_features_out(self) -> int:
        return len(self.feature_names_out_)

    @property
    def n_features_in_(self) -> int:
        """How many columns this preprocessor expects. Named for duck-typing.

        :meth:`shieldnet.persist.ModelBundle.validate` probes ``n_features_in_`` on
        whatever sits in the bundle's ``scaler`` slot to catch the one mismatch that
        never raises on its own - a transformer carrying statistics for 77 columns
        applied to a 25-column frame subtracts the wrong mean from every feature and
        returns a perfectly well-shaped array of nonsense. Exposing the attribute here,
        with the same name scikit-learn uses, means the whole composed preprocessor gets
        that check rather than only a bare ``Scaler``.
        """
        return len(self.feature_names_in_)

    def align(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """Reorder an arbitrary frame onto the fitted input columns.

        This is the function that makes the app forgiving. A user's export can have the
        columns in a different order, carry extras, or omit some; a model fed columns in
        the wrong order produces confident nonsense rather than an error, so alignment
        happens explicitly and reports what it had to invent.

        Returns ``(aligned, missing, extra)``.
        """
        present = {str(c) for c in X.columns}
        missing = [n for n in self.feature_names_in_ if n not in present]
        extra = sorted(present - set(self.feature_names_in_) - {sch.LABEL_COLUMN})

        aligned = X.reindex(columns=self.feature_names_in_)
        for name in missing:
            aligned[name] = self.raw_medians.get(name, 0.0)
        return aligned[self.feature_names_in_], missing, extra

    def transform(self, X: pd.DataFrame | np.ndarray, *, align: bool = False) -> np.ndarray:
        """Clip, impute and scale. Returns a dense float64 array, never NaN.

        Set *align* when the caller cannot guarantee column order - anything that came
        from a user upload rather than from the training pipeline.
        """
        if isinstance(X, pd.DataFrame):
            if align:
                X, missing, extra = self.align(X)
                if missing:
                    log.warning("%d feature(s) absent from the input and filled with "
                                "the training median: %s%s", len(missing),
                                ", ".join(missing[:5]),
                                " ..." if len(missing) > 5 else "")
                if extra:
                    log.info("ignoring %d column(s) the model was not trained on: %s",
                             len(extra), ", ".join(extra[:5]))
            elif [str(c) for c in X.columns] != self.feature_names_in_:
                raise ValueError(
                    "column order does not match the fitted preprocessor. Pass "
                    "align=True to reorder, or supply columns in "
                    f"feature_names_in_ order (first mismatch near "
                    f"{_first_mismatch(list(map(str, X.columns)), self.feature_names_in_)!r})."
                )
            keep = [n for n in self.feature_names_out_]
            values = X[keep].to_numpy(dtype=np.float64, copy=True)
        else:
            values = np.asarray(X, dtype=np.float64)
            if values.ndim == 1:
                values = values.reshape(1, -1)
            if values.shape[1] == len(self.feature_names_in_):
                dropped = set(self.dropped_constant)
                mask = np.array([n not in dropped for n in self.feature_names_in_])
                values = values[:, mask]
            elif values.shape[1] != self.n_features_out:
                raise ValueError(
                    f"expected {len(self.feature_names_in_)} raw or "
                    f"{self.n_features_out} retained features, got {values.shape[1]}"
                )

        # inf can arrive from a user upload even though training data was cleaned.
        values[np.isinf(values)] = np.nan
        values = self.clipper.transform(values)
        if self.knn_imputer is not None:
            values = np.asarray(self.knn_imputer.transform(values), dtype=np.float64)
            # KNNImputer leaves a column NaN if it was all-NaN at fit time.
            if np.isnan(values).any():
                values = self.imputer.transform(values)
        else:
            values = self.imputer.transform(values)
        out = self.scaler.transform(values)
        if not np.isfinite(out).all():
            n_bad = int((~np.isfinite(out)).sum())
            log.warning("%d non-finite cell(s) survived scaling and were zeroed; check "
                        "the input for absurd magnitudes", n_bad)
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    def frame(self, X: np.ndarray) -> pd.DataFrame:
        """Wrap a transformed array back into a named frame (SHAP wants names)."""
        return pd.DataFrame(np.asarray(X), columns=self.feature_names_out_)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "n_features_in": len(self.feature_names_in_),
            "n_features_out": self.n_features_out,
            "dropped_constant": list(self.dropped_constant),
            "clip_quantile": self.clipper.quantile,
            "scaler": self.scaler.kind,
            "imputer": "knn" if self.knn_imputer is not None else "median",
        }


def _first_mismatch(got: List[str], want: List[str]) -> str:
    for a, b in zip(got, want):
        if a != b:
            return f"got {a!r}, expected {b!r}"
    return f"length {len(got)} vs {len(want)}"


def _fit_knn_imputer(X: np.ndarray, n_neighbours: int):
    """Fit sklearn's KNNImputer, with a clear message when it is unavailable."""
    try:
        from sklearn.impute import KNNImputer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "imputer='knn' needs scikit-learn. Either `pip install scikit-learn` or "
            "set data.imputer='median' in the config (median is the default and is "
            "what the shipped artifacts use)."
        ) from exc
    log.info("fitting KNNImputer(k=%d) on %s rows - this is O(n^2) in the worst case "
             "and is much slower than the median path", n_neighbours, human_count(len(X)))
    imp = KNNImputer(n_neighbors=n_neighbours)
    imp.fit(X)
    return imp


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------

@dataclass
class SplitData:
    """Train / validation / test, as frames plus encoded targets."""

    X_train: pd.DataFrame
    y_train: np.ndarray
    X_val: pd.DataFrame
    y_val: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    codec: LabelCodec
    seed: int = 42
    warnings: List[str] = field(default_factory=list)

    @property
    def feature_names(self) -> List[str]:
        return [str(c) for c in self.X_train.columns]

    def sizes(self) -> Dict[str, int]:
        return {"train": len(self.X_train), "val": len(self.X_val),
                "test": len(self.X_test)}

    def distribution(self) -> pd.DataFrame:
        """Per-class counts in each split - the table that proves stratification held."""
        rows = []
        for cls_idx, cls in enumerate(self.codec.classes):
            rows.append({
                "class": cls,
                "train": int((self.y_train == cls_idx).sum()),
                "val": int((self.y_val == cls_idx).sum()),
                "test": int((self.y_test == cls_idx).sum()),
            })
        out = pd.DataFrame(rows)
        out["total"] = out[["train", "val", "test"]].sum(axis=1)
        return out.sort_values("total", ascending=False).reset_index(drop=True)

    def render(self) -> str:
        lines = [self.distribution().to_string(index=False), ""]
        sizes = self.sizes()
        total = sum(sizes.values())
        lines.append("  ".join(f"{k}={human_count(v)} ({v / total:.0%})"
                               for k, v in sizes.items()))
        for w in self.warnings:
            lines.append(f"WARNING - {w}")
        return "\n".join(lines)


def split_frame(
    df: pd.DataFrame,
    *,
    test_size: float = 0.20,
    val_size: float = 0.10,
    seed: int = 42,
    label_column: str = sch.LABEL_COLUMN,
    codec: Optional[LabelCodec] = None,
) -> SplitData:
    """Stratified three-way split that survives classes with a handful of rows.

    ``train_test_split(stratify=y)`` raises ``ValueError: The least populated class in y
    has only 1 member`` - and after the 15 -> 13 merge there are still classes in the
    tens. Rather than dropping those classes or abandoning stratification, this splits
    *within* each class and allocates by hand:

    * 1 row  -> train (the model at least sees the class exist)
    * 2 rows -> train + test
    * 3 rows -> one each
    * more   -> proportional, with at least one row in val and test

    So every class is represented in every split whenever that is arithmetically
    possible, and the report says so explicitly when it is not.
    """
    if label_column not in df.columns:
        raise KeyError(f"{label_column!r} not in frame")
    if not 0 < test_size < 1 or not 0 <= val_size < 1:
        raise ValueError("test_size must be in (0,1) and val_size in [0,1)")
    if test_size + val_size >= 0.9:
        raise ValueError(
            f"test_size + val_size = {test_size + val_size:.2f} leaves almost nothing "
            "to train on"
        )

    labels = df[label_column].astype(str)
    codec = codec or LabelCodec.fit(labels)
    y_all = codec.transform(labels)
    features = [c for c in df.columns if c != label_column]

    rng = np.random.default_rng(seed)
    idx_train: List[np.ndarray] = []
    idx_val: List[np.ndarray] = []
    idx_test: List[np.ndarray] = []
    warnings: List[str] = []

    for cls_idx, cls in enumerate(codec.classes):
        pool = np.flatnonzero(y_all == cls_idx)
        n = len(pool)
        if n == 0:
            warnings.append(f"class {cls!r} has no rows and appears in no split")
            continue
        pool = rng.permutation(pool)

        if n == 1:
            n_test = n_val = 0
            warnings.append(f"class {cls!r} has 1 row: train only, so it can never be "
                            "scored")
        elif n == 2:
            n_test, n_val = 1, 0
            warnings.append(f"class {cls!r} has 2 rows: no validation row, so early "
                            "stopping and threshold tuning ignore it")
        elif n == 3:
            n_test = n_val = 1
        else:
            n_test = max(1, int(round(n * test_size)))
            n_val = max(1, int(round(n * val_size))) if val_size > 0 else 0
            # Never starve training: keep at least half the class for fitting.
            while n - n_test - n_val < max(1, n // 2) and (n_test + n_val) > 2:
                if n_test >= n_val and n_test > 1:
                    n_test -= 1
                elif n_val > 1:
                    n_val -= 1
                else:
                    break

        idx_test.append(pool[:n_test])
        idx_val.append(pool[n_test:n_test + n_val])
        idx_train.append(pool[n_test + n_val:])

    def _gather(parts: List[np.ndarray]) -> np.ndarray:
        joined = np.concatenate(parts) if parts else np.array([], dtype=int)
        joined.sort()
        return joined

    tr, va, te = _gather(idx_train), _gather(idx_val), _gather(idx_test)
    assert len(tr) + len(va) + len(te) == len(df), "split lost or duplicated rows"
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))

    split = SplitData(
        X_train=df.iloc[tr][features].reset_index(drop=True), y_train=y_all[tr],
        X_val=df.iloc[va][features].reset_index(drop=True), y_val=y_all[va],
        X_test=df.iloc[te][features].reset_index(drop=True), y_test=y_all[te],
        codec=codec, seed=seed, warnings=warnings,
    )
    log.info("split %s rows -> train %s / val %s / test %s across %d classes",
             human_count(len(df)), human_count(len(tr)), human_count(len(va)),
             human_count(len(te)), codec.n_classes)
    for w in warnings:
        log.warning(w)
    return split


# ---------------------------------------------------------------------------
# class weights
# ---------------------------------------------------------------------------

def class_weights(
    y: np.ndarray,
    n_classes: Optional[int] = None,
    *,
    scheme: str = "balanced",
    cap: Optional[float] = 50.0,
) -> Dict[int, float]:
    """Inverse-frequency weights, capped.

    ``balanced`` is sklearn's formula, ``n / (k * count)``. On the real train split -
    213,230 rows, 13 classes, 47 of them Rare Attacks - that hands Rare Attacks a weight
    of 349 while BENIGN gets 0.156, a spread of about 2,200x. A single misclassified rare
    row then dominates the gradient: training becomes unstable and the model starts
    predicting that class everywhere. Capping at 50 keeps the correction useful without
    letting it take over. ``sqrt`` is the gentler alternative.
    """
    y = np.asarray(y, dtype=int)
    k = int(n_classes) if n_classes is not None else int(y.max()) + 1
    counts = np.bincount(y, minlength=k).astype(np.float64)
    present = counts > 0

    weights = np.ones(k, dtype=np.float64)
    if scheme == "balanced":
        weights[present] = len(y) / (present.sum() * counts[present])
    elif scheme == "sqrt":
        weights[present] = np.sqrt(counts[present].max() / counts[present])
    elif scheme == "none":
        pass
    else:
        raise ValueError(f"unknown weighting scheme {scheme!r}")

    if cap is not None:
        clipped = int((weights > cap).sum())
        if clipped:
            log.info("capped %d class weight(s) at %.0f (largest was %.0f) to keep "
                     "the gradient stable", clipped, cap, weights.max())
        weights = np.minimum(weights, cap)
    return {i: float(w) for i, w in enumerate(weights)}
