"""Building the stratified working chunk.

The design decision, and why
----------------------------
Retraining on 2.83M rows for every hyper-parameter trial is not feasible on student
hardware: an Optuna study of 40 trials with 3-fold CV means 120 fits, and SHAP on top of
that. The naive fixes are both bad:

* **Uniform random sampling** shrinks the rare classes proportionally. Take 10% and
  Heartbleed's 11 rows become 1. The rare-attack signal - the interesting part - is
  destroyed.
* **Full balancing by undersampling** throws away the overwhelming majority of BENIGN
  rows, and a model tuned on a balanced sample is badly calibrated on real traffic where
  80% of flows are benign. Log loss punishes exactly that.

So we cap the head and keep the tail whole: the four large classes are stratified down
to a fixed budget, every class below the cap is kept in full, and the three ultra-rare
labels are merged. The result is roughly 300k rows - about 11% of the dataset by row
count, but 100% of the minority-class information.

The imbalance is *reduced, not removed*: on the default caps the chunk holds 150,000
BENIGN rows against the 68 in the merged Rare Attacks class, so BENIGN still dominates at
about 2,200:1. That is down from 2.27M:68 - roughly 33,000:1 - in the full dataset, which
is a real improvement and nowhere near enough on its own. It is why SMOTE and class
weights still exist downstream, and why every metric is reported macro-averaged as well as
overall.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .. import schema as sch
from ..logging_utils import get_logger, human_count
from .load import LoadReport, iter_raw_chunks

log = get_logger(__name__)

__all__ = ["ChunkReport", "build_working_chunk", "build_working_chunk_streaming"]


@dataclass
class ChunkReport:
    """Per-class accounting for the sampling step."""

    rows_in: int = 0
    rows_out: int = 0
    per_class: pd.DataFrame = field(default_factory=pd.DataFrame)
    unstratifiable: List[str] = field(default_factory=list)
    dropped_classes: List[str] = field(default_factory=list)
    seed: int = 42

    @property
    def retained_fraction(self) -> float:
        return self.rows_out / self.rows_in if self.rows_in else 0.0

    def to_dict(self) -> dict:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "retained_fraction": round(self.retained_fraction, 4),
            "seed": self.seed,
            "unstratifiable": self.unstratifiable,
            "dropped_classes": self.dropped_classes,
            "per_class": self.per_class.to_dict(orient="records"),
        }

    def render(self) -> str:
        lines = [self.per_class.to_string(index=False)]
        lines.append("")
        lines.append(f"rows in   {human_count(self.rows_in)}")
        lines.append(f"rows out  {human_count(self.rows_out)}  "
                     f"({self.retained_fraction:.1%} of input)")
        minority = self.per_class[self.per_class["strategy"] == "kept in full"]
        if not minority.empty:
            lines.append(f"minority rows preserved in full: "
                         f"{human_count(int(minority['kept'].sum()))} across "
                         f"{len(minority)} class(es)")
        if self.unstratifiable:
            lines.append("")
            lines.append("WARNING - too few rows for a stable stratified split: "
                         + ", ".join(self.unstratifiable))
        if self.dropped_classes:
            lines.append("WARNING - classes dropped entirely (zero rows): "
                         + ", ".join(self.dropped_classes))
        return "\n".join(lines)


def _plan(counts: Mapping[str, int], caps: Mapping[str, int]) -> pd.DataFrame:
    """Decide, per class, how many rows to keep."""
    rows = []
    for cls, available in sorted(counts.items(), key=lambda kv: -kv[1]):
        cap = caps.get(cls)
        if cap is None or available <= cap:
            keep, strategy = available, "kept in full"
        else:
            keep, strategy = cap, f"capped at {cap:,}"
        rows.append({
            "class": cls,
            "available": int(available),
            "cap": int(cap) if cap is not None else None,
            "kept": int(keep),
            "strategy": strategy,
            "share": keep / available if available else 0.0,
        })
    plan = pd.DataFrame(rows)
    if not plan.empty:
        plan["share"] = plan["share"].map(lambda v: f"{v:.1%}")
    return plan


def build_working_chunk(
    df: pd.DataFrame,
    *,
    caps: Optional[Mapping[str, int]] = None,
    seed: int = 42,
    min_class_rows: int = 10,
    label_column: str = sch.LABEL_COLUMN,
) -> tuple[pd.DataFrame, ChunkReport]:
    """Cap the large classes, keep everything else whole.

    Sampling is *within class* and seeded, so the chunk is byte-reproducible from
    ``(df, caps, seed)``.
    """
    if label_column not in df.columns:
        raise KeyError(f"{label_column!r} not in frame; columns: {list(df.columns)[:6]}")

    caps = dict(caps or {})
    counts = df[label_column].value_counts()
    counts = counts[counts > 0]                      # categoricals keep empty levels
    counts = {str(k): int(v) for k, v in counts.items()}

    report = ChunkReport(rows_in=len(df), seed=seed)
    report.per_class = _plan(counts, caps)
    report.unstratifiable = [c for c, n in counts.items() if n < min_class_rows]
    report.dropped_classes = [c for c in caps if c not in counts]

    rng = np.random.default_rng(seed)
    keep_index: List[np.ndarray] = []
    # Explicit zip rather than itertuples: "class" is a Python keyword, so itertuples
    # silently renames the field to `_1` and the loop breaks if a column is reordered.
    for cls, keep in zip(report.per_class["class"], report.per_class["kept"]):
        cls_idx = df.index[df[label_column] == cls].to_numpy()
        if keep >= len(cls_idx):
            keep_index.append(cls_idx)
        else:
            keep_index.append(rng.choice(cls_idx, size=int(keep), replace=False))

    selected = np.concatenate(keep_index) if keep_index else np.array([], dtype=int)
    selected.sort()                                  # preserve original row order
    chunk = df.loc[selected].reset_index(drop=True)

    # Drop now-unused category levels so downstream value_counts / encoders do not
    # emit phantom zero-count classes.
    if isinstance(chunk[label_column].dtype, pd.CategoricalDtype):
        chunk[label_column] = chunk[label_column].cat.remove_unused_categories()

    report.rows_out = len(chunk)
    chunk.attrs.update(df.attrs)
    chunk.attrs["chunk_seed"] = seed

    log.info("working chunk: %s rows (%.1f%% of %s) across %d classes",
             human_count(len(chunk)), 100 * report.retained_fraction,
             human_count(len(df)), chunk[label_column].nunique())
    if report.unstratifiable:
        log.warning("class(es) with fewer than %d rows: %s - metrics for these will be "
                    "unstable and SMOTE may not be able to oversample them",
                    min_class_rows, report.unstratifiable)
    return chunk, report


def build_working_chunk_streaming(
    files: Sequence[Path],
    *,
    caps: Optional[Mapping[str, int]] = None,
    seed: int = 42,
    encoding: str = "latin-1",
    chunk_rows: int = 250_000,
    merge_rare: bool = True,
    min_class_rows: int = 10,
    oversample_factor: float = 1.06,
) -> tuple[pd.DataFrame, ChunkReport, LoadReport]:
    """Build the chunk in two streaming passes, never holding the full dataset.

    Pass 1 tallies labels. Pass 2 keeps each row of a capped class with probability
    ``cap / count``, which makes peak memory a function of ``chunk_rows`` rather than of
    dataset size - the difference between "runs on a free Colab instance" and "dies".

    Bernoulli sampling only hits the cap in expectation, so pass 2 aims slightly high
    (*oversample_factor*) and the result is trimmed to the exact cap at the end. The
    trim is itself seeded, so the output is still reproducible.

    Use this when RAM is tight. :func:`build_working_chunk` on an already-loaded frame
    is simpler and gives exact per-class counts in one pass.
    """
    caps = dict(caps or {})

    # ---- pass 1: count ---------------------------------------------------
    log.info("pass 1/2 - counting labels across %d file(s)", len(files))
    counts: Dict[str, int] = {}
    rows_in = 0
    for piece in iter_raw_chunks(files, encoding=encoding, chunk_rows=chunk_rows):
        labels = piece[sch.LABEL_COLUMN]
        if merge_rare:
            labels = labels.map(sch.collapse_rare)
        rows_in += len(piece)
        for cls, n in labels.value_counts().items():
            counts[str(cls)] = counts.get(str(cls), 0) + int(n)
    log.info("pass 1 complete: %s rows, %d classes", human_count(rows_in), len(counts))

    report = ChunkReport(rows_in=rows_in, seed=seed)
    report.per_class = _plan(counts, caps)
    report.unstratifiable = [c for c, n in counts.items() if n < min_class_rows]

    keep_prob = {
        cls: min(1.0, oversample_factor * caps[cls] / counts[cls])
        for cls in caps if cls in counts and counts[cls] > caps[cls]
    }

    # ---- pass 2: sample --------------------------------------------------
    log.info("pass 2/2 - sampling (%d class(es) capped)", len(keep_prob))
    rng = np.random.default_rng(seed)
    load_report = LoadReport()
    kept: List[pd.DataFrame] = []
    for piece in iter_raw_chunks(files, encoding=encoding, chunk_rows=chunk_rows,
                                report=load_report):
        if merge_rare:
            piece[sch.LABEL_COLUMN] = piece[sch.LABEL_COLUMN].map(sch.collapse_rare)
        probs = piece[sch.LABEL_COLUMN].map(keep_prob).fillna(1.0).to_numpy(dtype=float)
        mask = rng.random(len(piece)) < probs
        if mask.any():
            kept.append(piece.loc[mask])

    chunk = (pd.concat(kept, ignore_index=True) if kept
             else pd.DataFrame(columns=sch.CANONICAL_FEATURES + [sch.LABEL_COLUMN]))
    chunk = chunk.reindex(columns=sch.CANONICAL_FEATURES + [sch.LABEL_COLUMN])

    # ---- exact trim ------------------------------------------------------
    trim_index: List[np.ndarray] = []
    for cls, planned in zip(report.per_class["class"], report.per_class["kept"]):
        cls_idx = chunk.index[chunk[sch.LABEL_COLUMN] == cls].to_numpy()
        target = min(int(planned), len(cls_idx))
        if target < len(cls_idx):
            cls_idx = rng.choice(cls_idx, size=target, replace=False)
        trim_index.append(cls_idx)
        # Record what we actually got, which can undershoot for a capped class.
        report.per_class.loc[report.per_class["class"] == cls, "kept"] = target

    selected = np.concatenate(trim_index) if trim_index else np.array([], dtype=int)
    selected.sort()
    chunk = chunk.loc[selected].reset_index(drop=True)
    chunk[sch.LABEL_COLUMN] = chunk[sch.LABEL_COLUMN].astype("category")

    report.rows_out = len(chunk)
    load_report.rows_kept = len(chunk)
    load_report.class_counts = {str(k): int(v) for k, v
                                in chunk[sch.LABEL_COLUMN].value_counts().items()}

    log.info("streaming chunk: %s rows (%.1f%% of %s)", human_count(len(chunk)),
             100 * report.retained_fraction, human_count(rows_in))
    return chunk, report, load_report
