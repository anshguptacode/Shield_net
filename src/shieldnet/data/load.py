"""Reading the raw CICIDS2017 CSVs into one canonical frame.

Two variants of CICIDS2017 circulate, and code written against one breaks on the other:

``MachineLearningCVE/`` (the usual Kaggle mirror)
    79 columns - 77 features, the duplicated ``Fwd Header Length.1``, and ``Label``.

``TrafficLabelling/`` / ``GeneratedLabelledFlows``
    85+ columns - the same features plus ``Flow ID``, ``Source IP``, ``Source Port``,
    ``Destination IP``, ``Protocol``, ``Timestamp`` and sometimes ``External IP``.

The identifier columns in the second variant are **label leakage, not features**. The
attacks in CICIDS2017 were launched from a small fixed set of source addresses, so
``Source IP`` alone predicts the attack class almost perfectly - and a model that learns
it has learned the lab's IP plan, not intrusion detection. Reported accuracies above
99.9% on this dataset are very often this bug. :func:`load_raw` drops those columns and
logs that it did.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd

from .. import schema as sch
from ..logging_utils import get_logger, human_count

log = get_logger(__name__)

__all__ = ["discover_raw_files", "read_flow_csv", "load_raw", "iter_raw_chunks",
           "audit_class_counts", "LoadReport", "IDENTIFIER_COLUMNS"]


#: Columns dropped on sight: direct identifiers, timestamps and known leakage.
#:
#: Canonical (whitespace-stripped) spellings; matching is case-insensitive.
IDENTIFIER_COLUMNS: Dict[str, str] = {
    "flow id": "flow identifier - unique per row, pure leakage",
    "source ip": "attacker/victim address - encodes the label directly",
    "src ip": "attacker/victim address - encodes the label directly",
    "destination ip": "victim address - encodes the label directly",
    "dst ip": "victim address - encodes the label directly",
    "external ip": "capture-harness artefact",
    "source port": "ephemeral client port - high-cardinality noise",
    "src port": "ephemeral client port - high-cardinality noise",
    "timestamp": "capture time - separates attack windows, so it leaks the label",
    "protocol": "not present in the MachineLearningCVE variant; dropped for parity",
    "simillarhttp": "empty column in some redistributions (sic, misspelled upstream)",
    "inbound": "direction flag absent from the standard 77-feature schema",
    "fwd header length.1": "exact duplicate of Fwd Header Length",
}


class LoadReport:
    """What happened during a load - printed by the CLI and folded into the manifest."""

    def __init__(self) -> None:
        self.files: List[str] = []
        self.rows_read: int = 0
        self.rows_kept: int = 0
        self.dropped_columns: Dict[str, str] = {}
        self.missing_expected: List[str] = []
        self.unexpected_labels: Dict[str, int] = {}
        self.inf_cells: int = 0
        self.nan_cells: int = 0
        self.non_numeric_coerced: Dict[str, int] = {}
        self.class_counts: Dict[str, int] = {}

    def to_dict(self) -> dict:
        return {
            "files": self.files,
            "rows_read": self.rows_read,
            "rows_kept": self.rows_kept,
            "dropped_columns": self.dropped_columns,
            "missing_expected_files": self.missing_expected,
            "unexpected_labels": self.unexpected_labels,
            "inf_cells": self.inf_cells,
            "nan_cells": self.nan_cells,
            "non_numeric_coerced": self.non_numeric_coerced,
            "class_counts": self.class_counts,
        }

    def render(self) -> str:
        lines = [
            f"files read           {len(self.files)}",
            f"rows read            {human_count(self.rows_read)}",
            f"rows kept            {human_count(self.rows_kept)}",
            f"inf cells            {human_count(self.inf_cells)}",
            f"NaN cells            {human_count(self.nan_cells)}",
        ]
        if self.dropped_columns:
            lines.append(f"columns dropped      {len(self.dropped_columns)}")
            for name, why in self.dropped_columns.items():
                lines.append(f"    - {name}: {why}")
        if self.missing_expected:
            lines.append("expected files not found:")
            lines += [f"    - {n}" for n in self.missing_expected]
        if self.non_numeric_coerced:
            lines.append("non-numeric values coerced to NaN:")
            for col, n in sorted(self.non_numeric_coerced.items(),
                                 key=lambda kv: -kv[1])[:8]:
                lines.append(f"    - {col}: {human_count(n)}")
        if self.unexpected_labels:
            lines.append("labels not in the published class list:")
            for lab, n in self.unexpected_labels.items():
                lines.append(f"    - {lab!r}: {human_count(n)}")
        if self.class_counts:
            lines.append("class counts:")
            width = max(len(c) for c in self.class_counts)
            for cls, n in sorted(self.class_counts.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {cls:<{width}}  {human_count(n):>12}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def discover_raw_files(directory: Path | str) -> List[Path]:
    """Find the CICIDS2017 CSVs under *directory*, searching recursively.

    Globs rather than requiring exact names, because Kaggle mirrors nest the files
    differently and the upstream names contain quirks (``Wednesday-workingHours`` with
    a lower-case w, ``Infilteration`` misspelled). Files are returned in a stable
    sorted order so a run is reproducible.
    """
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(
            f"raw data directory does not exist: {root}\n"
            "Run `shieldnet download` to fetch CICIDS2017 from Kaggle, or "
            "`shieldnet prepare --synthetic 40000` for a smoke test - that route "
            "generates flows in memory and never looks at this directory."
        )
    found = sorted(
        p for p in root.rglob("*.csv")
        if p.is_file() and not p.name.startswith(".")
    )
    if not found:
        raise FileNotFoundError(
            f"no .csv files under {root}. Expected the eight day-wise CICIDS2017 "
            "files, e.g. 'Monday-WorkingHours.pcap_ISCX.csv'."
        )
    return found


# ---------------------------------------------------------------------------
# single-file reading
# ---------------------------------------------------------------------------

def _canonicalise(
    df: pd.DataFrame,
    report: Optional[LoadReport] = None,
    *,
    require_label: bool = True,
) -> pd.DataFrame:
    """Rename to canonical names, drop junk columns, coerce numerics, fix labels."""
    df = df.rename(columns={c: sch.normalise_column(c) for c in df.columns})

    # Drop identifier / leakage / duplicate / pandas-index columns.
    to_drop: Dict[str, str] = {}
    for col in list(df.columns):
        key = col.lower()
        if key in IDENTIFIER_COLUMNS:
            to_drop[col] = IDENTIFIER_COLUMNS[key]
        elif col.startswith("Unnamed:") or key == "index":
            to_drop[col] = "unnamed index column written by a previous to_csv()"
    if to_drop:
        df = df.drop(columns=list(to_drop))
        if report is not None:
            report.dropped_columns.update(to_drop)

    # Labels first, so a bad label column cannot be coerced to NaN as a "feature".
    if sch.LABEL_COLUMN in df.columns:
        df[sch.LABEL_COLUMN] = (
            df[sch.LABEL_COLUMN].astype("string").map(sch.normalise_label)
        )
    elif require_label:
        raise sch.SchemaError(
            f"no {sch.LABEL_COLUMN!r} column found. Columns seen: "
            f"{list(df.columns)[:8]}{' ...' if len(df.columns) > 8 else ''}"
        )

    feature_cols = [c for c in df.columns if c != sch.LABEL_COLUMN]
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            before = df[col].notna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            lost = int(before - df[col].notna().sum())
            if lost and report is not None:
                report.non_numeric_coerced[col] = (
                    report.non_numeric_coerced.get(col, 0) + lost
                )
    return df


def read_flow_csv(
    path: Path | str,
    *,
    encoding: str = "latin-1",
    require_label: bool = False,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Read one flow CSV and return it with canonical columns.

    This is the function the Streamlit app uses for user uploads, so it must not
    require a label column and must not assume all 77 features are present.
    """
    source = Path(path)
    try:
        df = pd.read_csv(source, encoding=encoding, low_memory=False,
                         skipinitialspace=False, nrows=nrows)
    except UnicodeDecodeError:
        # Only reachable if a caller overrode encoding with something strict.
        log.warning("%s is not valid %s; retrying as latin-1", source.name, encoding)
        df = pd.read_csv(source, encoding="latin-1", low_memory=False, nrows=nrows)
    return _canonicalise(df, require_label=require_label)


def iter_raw_chunks(
    files: Sequence[Path],
    *,
    encoding: str = "latin-1",
    chunk_rows: int = 250_000,
    report: Optional[LoadReport] = None,
) -> Iterator[pd.DataFrame]:
    """Stream canonical chunks from the raw files.

    Peak memory is a function of ``chunk_rows`` rather than of dataset size. Measured on
    the canonical 78-column layout a row costs about 679 bytes as parsed, so a
    250,000-row chunk is roughly 170 MB and two are briefly alive at once - a few hundred
    megabytes, against the 872 MB the finished float32 frame would occupy and the 2 GB
    :func:`load_raw` peaks at. That is what makes a full 2.83M-row pass possible on a
    12 GB Colab instance with a model in memory beside it.
    """
    for path in files:
        if report is not None:
            report.files.append(path.name)
        reader = pd.read_csv(path, encoding=encoding, low_memory=False,
                             chunksize=chunk_rows)
        for raw_chunk in reader:
            if report is not None:
                report.rows_read += len(raw_chunk)
            yield _canonicalise(raw_chunk, report, require_label=True)


# ---------------------------------------------------------------------------
# full load
# ---------------------------------------------------------------------------

def load_raw(
    directory: Path | str,
    *,
    encoding: str = "latin-1",
    merge_rare: bool = True,
    downcast: bool = True,
    chunk_rows: int = 250_000,
    files: Optional[Sequence[Path]] = None,
) -> tuple[pd.DataFrame, LoadReport]:
    """Load and consolidate every raw CSV into one canonical labelled frame.

    Parameters
    ----------
    merge_rare:
        Apply the 15 -> 13 class collapse (see :data:`shieldnet.schema.RARE_SOURCE_LABELS`).
    downcast:
        Store features as ``float32``, per chunk as they are read. Halves memory - 2.83M
        x 77 goes from about 1.7 GB to 872 MB - at a precision cost that is irrelevant
        for these models. It also halves the peak, because the cast happens before the
        chunks are accumulated rather than after they are joined.

    Notes
    -----
    Consolidation is the memory peak of a full run and ``chunk_rows`` does not bound it:
    every chunk is held until the ``concat``, so the list and the joined frame coexist and
    the peak is a little over twice the finished frame - roughly 2 GB for all 2.83M rows,
    even at 100k-row chunks. What ``chunk_rows`` bounds is the pandas parser's own
    buffer, which is why it still matters on a small machine, and
    :func:`shieldnet.data.chunk.build_working_chunk_streaming` exists for when 2 GB is
    too much: it discards rows as it goes and never holds the full dataset at all. That
    is the default whenever more than one raw file is present.

    Returns
    -------
    ``(frame, report)``. The frame's ``Label`` column is a pandas ``category``.
    """
    root = Path(directory)
    paths = list(files) if files is not None else discover_raw_files(root)
    report = LoadReport()

    found_names = {p.name for p in paths}
    report.missing_expected = [n for n in sch.EXPECTED_RAW_FILES if n not in found_names]
    if report.missing_expected:
        log.warning("%d of the 8 expected raw files were not found by name (%s). "
                    "Continuing with the %d .csv file(s) that are present.",
                    len(report.missing_expected),
                    ", ".join(report.missing_expected[:3]) +
                    (" ..." if len(report.missing_expected) > 3 else ""),
                    len(paths))

    # Count the inf/NaN cells and cast to float32 chunk by chunk, before anything is
    # accumulated. Doing both after the concat - which is where they used to live - meant
    # the list of chunks and the consolidated frame were alive at the same time in
    # float64, so the peak was twice the full-precision frame: about 3.8 GB for all
    # 2.83M rows, whatever `chunk_rows` was set to. Casting first makes the same peak
    # twice the float32 frame instead, a little over 2 GB, and the counting pass no
    # longer has to materialise a 1.7 GB float64 view of the whole thing.
    frames: List[pd.DataFrame] = []
    for chunk in iter_raw_chunks(paths, encoding=encoding, chunk_rows=chunk_rows,
                                 report=report):
        present = [c for c in sch.CANONICAL_FEATURES if c in chunk.columns]
        # float64 for the count: the cast below turns nothing into inf at CICIDS2017's
        # magnitudes (the largest values are around 1e9, float32 tops out near 3.4e38),
        # but counting first means the report describes the file rather than the cast.
        values = chunk[present].to_numpy(dtype="float64", copy=False)
        report.inf_cells += int(np.isinf(values).sum())
        report.nan_cells += int(np.isnan(values).sum())
        del values
        if downcast:
            # inf survives the float32 cast; preprocess converts it to NaN and imputes.
            chunk[present] = chunk[present].astype("float32")
        frames.append(chunk)
    df = pd.concat(frames, ignore_index=True)
    del frames

    # Reindex onto the canonical feature order, materialising any absent feature as
    # NaN so downstream stages can rely on a fixed 77-column layout.
    absent = [c for c in sch.CANONICAL_FEATURES if c not in df.columns]
    if absent:
        log.warning("%d canonical feature(s) absent from the raw files and filled "
                    "with NaN: %s", len(absent), absent[:6])
    df = df.reindex(columns=sch.CANONICAL_FEATURES + [sch.LABEL_COLUMN])
    # The per-chunk pass above could only count the columns the files actually had, so
    # add the cells this reindex just invented. Without this the report would understate
    # `nan_cells` by one column's worth of rows for every absent feature - and an absent
    # feature is precisely the situation where someone is reading the report closely.
    if absent:
        report.nan_cells += len(df) * len(absent)
        if downcast:
            # Only the invented columns need casting; the rest were cast per chunk, and
            # re-casting all 77 would copy the whole feature block for nothing.
            df[absent] = df[absent].astype("float32")

    labels = df[sch.LABEL_COLUMN]
    unexpected = labels[~labels.isin(sch.RAW_LABELS)]
    if len(unexpected):
        report.unexpected_labels = {
            str(k): int(v) for k, v in unexpected.value_counts().items()
        }
        log.warning("%s row(s) carry a label outside the published 15: %s",
                    human_count(len(unexpected)), list(report.unexpected_labels)[:5])

    # Rows with a missing label are unusable for supervised training.
    before = len(df)
    df = df[df[sch.LABEL_COLUMN].notna() & (df[sch.LABEL_COLUMN] != "")]
    if len(df) < before:
        log.warning("dropped %s row(s) with an empty label", human_count(before - len(df)))

    if merge_rare:
        df[sch.LABEL_COLUMN] = df[sch.LABEL_COLUMN].map(sch.collapse_rare)

    df[sch.LABEL_COLUMN] = df[sch.LABEL_COLUMN].astype("category")
    report.rows_kept = len(df)
    report.class_counts = {str(k): int(v) for k, v
                           in df[sch.LABEL_COLUMN].value_counts().items()}

    log.info("loaded %s rows x %d features from %d file(s)",
             human_count(len(df)), len(features), len(paths))
    return df.reset_index(drop=True), report


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def audit_class_counts(
    counts: Dict[str, int],
    *,
    tolerance: float = 0.02,
    merged_rare: bool = True,
) -> pd.DataFrame:
    """Compare observed class counts against the published CICIDS2017 figures.

    Redistributed copies differ by a few rows, and deduplication legitimately removes
    a lot, so this reports a signed difference for you to eyeball rather than
    asserting. A class that is off by more than *tolerance* is flagged ``CHECK``.
    """
    reference = dict(sch.REFERENCE_CLASS_COUNTS)
    if merged_rare:
        reference[sch.RARE_CLASS_NAME] = sum(
            reference.pop(k) for k in sch.RARE_SOURCE_LABELS
        )

    rows = []
    for cls in sorted(set(reference) | set(counts), key=lambda c: -reference.get(c, 0)):
        expected = reference.get(cls)
        observed = counts.get(cls, 0)
        if expected:
            delta = (observed - expected) / expected
            status = "ok" if abs(delta) <= tolerance else "CHECK"
            pct = f"{delta:+.1%}"
        else:
            delta, status, pct = float("nan"), "unlisted", "-"
        rows.append({"class": cls, "published": expected, "observed": observed,
                     "difference": pct, "status": status})
    return pd.DataFrame(rows)
