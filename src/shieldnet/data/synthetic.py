"""Synthetic CICIDS2017-shaped flow generator.

Why this exists
---------------
The real dataset is a 2.8-million-row, ~1 GB download behind a Kaggle login. That is a
terrible feedback loop for checking that a pipeline runs. This module manufactures a
small dataset with the *same schema, the same class names and the same defects*, so the
entire pipeline - clean, chunk, select, balance, train, explain, serve - can be
exercised end to end in seconds.

It is a test fixture and a demo aid, not a research contribution. Numbers produced from
synthetic data must never be reported as results; :func:`synthesise` stamps
``synthetic=True`` into the frame's ``attrs`` and the training pipeline propagates that
into the artifact manifest so a synthetic model cannot be mistaken for a real one.

Realism that matters for testing
--------------------------------
* Per-class feature signatures loosely modelled on the real attacks (a PortScan flow is
  one or two packets in microseconds; slowloris is minutes with almost no data), so
  feature selection and SHAP produce sensible rather than arbitrary output.
* Features are *derived* from a few drivers rather than drawn independently, which
  reproduces the heavy multicollinearity of the real data - the thing the correlation
  filter exists to handle.
* The real defects are injected on purpose: ``inf`` from division by a zero duration,
  scattered ``NaN``, exact duplicate rows, and the eight all-zero columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from .. import schema as sch
from ..logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["synthesise", "write_raw_like", "CLASS_PROFILES"]


# Per-class drivers. Fields:
#   ports        candidate destination ports (None -> random ephemeral)
#   dur          (log10 mean, log10 sigma) of flow duration in microseconds
#   fwd          (log mean, log sigma) of forward packet count
#   bwd_ratio    backward packets as a fraction of forward
#   fwd_len      mean forward payload bytes per packet
#   bwd_len      mean backward payload bytes per packet
#   flags        (syn, fin, rst, psh, ack, urg) expected counts
#   win          (fwd initial window, bwd initial window)
#   idle         fraction of the flow spent idle
CLASS_PROFILES: Dict[str, dict] = {
    "BENIGN": dict(ports=[80, 443, 53, 22, 445, 8080], dur=(5.6, 1.1), fwd=(1.5, 1.0),
                   bwd_ratio=0.9, fwd_len=180, bwd_len=420,
                   flags=(1, 1, 0, 2, 3, 0), win=(8192, 8192), idle=0.25),
    "DoS Hulk": dict(ports=[80], dur=(6.0, 0.6), fwd=(2.6, 0.5),
                     bwd_ratio=0.75, fwd_len=340, bwd_len=1100,
                     flags=(1, 0, 0, 4, 5, 0), win=(8192, 235), idle=0.05),
    "PortScan": dict(ports=None, dur=(1.4, 0.9), fwd=(0.15, 0.25),
                     bwd_ratio=0.35, fwd_len=0, bwd_len=0,
                     flags=(1, 0, 1, 0, 0, 0), win=(1024, -1), idle=0.0),
    "DDoS": dict(ports=[80], dur=(5.1, 0.7), fwd=(2.2, 0.6),
                 bwd_ratio=0.55, fwd_len=900, bwd_len=140,
                 flags=(1, 1, 0, 3, 4, 0), win=(8192, 229), idle=0.02),
    "DoS GoldenEye": dict(ports=[80], dur=(6.6, 0.5), fwd=(1.9, 0.4),
                          bwd_ratio=0.85, fwd_len=260, bwd_len=780,
                          flags=(1, 1, 0, 2, 4, 0), win=(8192, 251), idle=0.45),
    "FTP-Patator": dict(ports=[21], dur=(6.1, 0.5), fwd=(1.7, 0.4),
                        bwd_ratio=1.15, fwd_len=28, bwd_len=52,
                        flags=(1, 1, 1, 3, 5, 0), win=(8192, 227), idle=0.35),
    "SSH-Patator": dict(ports=[22], dur=(6.4, 0.5), fwd=(2.0, 0.4),
                        bwd_ratio=1.05, fwd_len=88, bwd_len=132,
                        flags=(1, 1, 0, 4, 6, 0), win=(8192, 231), idle=0.30),
    "DoS slowloris": dict(ports=[80], dur=(7.2, 0.4), fwd=(1.2, 0.5),
                          bwd_ratio=0.5, fwd_len=42, bwd_len=96,
                          flags=(1, 0, 0, 1, 2, 0), win=(8192, 237), idle=0.80),
    "DoS Slowhttptest": dict(ports=[80], dur=(7.0, 0.45), fwd=(1.4, 0.5),
                             bwd_ratio=0.45, fwd_len=36, bwd_len=88,
                             flags=(1, 0, 0, 1, 2, 0), win=(8192, 239), idle=0.72),
    "Bot": dict(ports=[8080], dur=(5.4, 0.8), fwd=(1.1, 0.5),
                bwd_ratio=1.0, fwd_len=64, bwd_len=310,
                flags=(1, 1, 0, 2, 3, 0), win=(8192, 254), idle=0.55),
    "Web Attack - Brute Force": dict(ports=[80], dur=(6.2, 0.6), fwd=(1.6, 0.5),
                                     bwd_ratio=0.95, fwd_len=430, bwd_len=620,
                                     flags=(1, 1, 0, 3, 4, 0), win=(8192, 243),
                                     idle=0.28),
    "Web Attack - XSS": dict(ports=[80], dur=(6.0, 0.6), fwd=(1.4, 0.5),
                             bwd_ratio=0.9, fwd_len=560, bwd_len=540,
                             flags=(1, 1, 0, 3, 4, 1), win=(8192, 245), idle=0.30),
    sch.RARE_CLASS_NAME: dict(ports=[444, 445, 8888], dur=(6.3, 1.2), fwd=(1.3, 1.0),
                              bwd_ratio=2.4, fwd_len=120, bwd_len=2600,
                              flags=(1, 1, 1, 2, 3, 1), win=(8192, 248), idle=0.40),
}


def _class_allocation(n_rows: int, rng: np.random.Generator,
                      proportions: Optional[Mapping[str, float]]) -> Dict[str, int]:
    """Decide how many rows each class gets.

    Defaults to the real dataset's proportions so the imbalance - and therefore the
    need for SMOTE and macro-averaged metrics - is reproduced. Every class is
    guaranteed at least 30 rows, otherwise a stratified three-way split is impossible
    and the pipeline would fail for a reason that has nothing to do with the code.
    """
    if proportions is None:
        counts = {c: sch.REFERENCE_CLASS_COUNTS.get(c, 0) for c in CLASS_PROFILES}
        counts[sch.RARE_CLASS_NAME] = sum(
            sch.REFERENCE_CLASS_COUNTS[k] for k in sch.RARE_SOURCE_LABELS
        )
        total = sum(counts.values())
        proportions = {c: v / total for c, v in counts.items()}

    floor = 30
    alloc = {c: max(floor, int(round(n_rows * proportions.get(c, 0.0))))
             for c in CLASS_PROFILES}
    # Rescale the classes that are above the floor so the total lands on n_rows.
    slack = n_rows - sum(v for v in alloc.values() if v == floor)
    adjustable = {c: v for c, v in alloc.items() if v > floor}
    if adjustable and slack > 0:
        scale = slack / sum(adjustable.values())
        for c in adjustable:
            alloc[c] = max(floor, int(round(alloc[c] * scale)))
    return alloc


def synthesise(
    n_rows: int = 20_000,
    *,
    seed: int = 42,
    proportions: Optional[Mapping[str, float]] = None,
    inject_defects: bool = True,
    labelled: bool = True,
) -> pd.DataFrame:
    """Build a synthetic flow dataset with canonical column names.

    Parameters
    ----------
    n_rows:
        Approximate total rows; per-class floors mean the result can be slightly
        larger.
    inject_defects:
        Add the ``inf`` / ``NaN`` / duplicate-row defects of the real dataset. Leave on
        for testing the cleaning stage; turn off when you want a clean demo CSV.
    labelled:
        Include the ``Label`` column.

    Returns
    -------
    DataFrame with the 77 canonical features (plus ``Label``), and
    ``df.attrs["synthetic"] = True``.
    """
    rng = np.random.default_rng(seed)
    alloc = _class_allocation(n_rows, rng, proportions)
    frames = [_one_class(name, count, rng) for name, count in alloc.items() if count]
    df = pd.concat(frames, ignore_index=True)

    # Shuffle so the class order is not an implicit feature.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if inject_defects:
        df = _inject_defects(df, rng)
    if not labelled:
        df = df.drop(columns=[sch.LABEL_COLUMN])
    else:
        # Match load_raw's dtype so the two entry points are interchangeable
        # downstream - train.py accepts a frame from either.
        df[sch.LABEL_COLUMN] = df[sch.LABEL_COLUMN].astype("category")

    df.attrs["synthetic"] = True
    df.attrs["seed"] = seed
    log.info("synthesised %s rows x %s features across %d classes",
             f"{len(df):,}", df.shape[1] - (1 if labelled else 0), len(alloc))
    return df


def _one_class(name: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate *n* rows for one class from its profile."""
    p = CLASS_PROFILES[name]

    # --- drivers ---------------------------------------------------------
    if p["ports"] is None:
        port = rng.integers(1024, 65535, n)
    else:
        port = rng.choice(p["ports"], n)

    dur_mu, dur_sigma = p["dur"]
    duration = np.clip(10 ** rng.normal(dur_mu, dur_sigma, n), 0, 1.2e8)

    fwd_mu, fwd_sigma = p["fwd"]
    fwd_pkts = np.clip(np.round(rng.lognormal(fwd_mu, fwd_sigma, n)), 1, 40_000)
    bwd_pkts = np.clip(np.round(fwd_pkts * p["bwd_ratio"] *
                                rng.uniform(0.6, 1.4, n)), 0, 40_000)

    # Payload sizes. A mean of 0 means a header-only flow (PortScan), which must stay
    # exactly 0 rather than picking up noise.
    def _len(mean: float) -> np.ndarray:
        if mean <= 0:
            return np.zeros(n)
        return np.clip(rng.gamma(4.0, mean / 4.0, n), 0, 18_000)

    fwd_len_mean = _len(p["fwd_len"])
    bwd_len_mean = _len(p["bwd_len"])

    fwd_bytes = fwd_pkts * fwd_len_mean
    bwd_bytes = bwd_pkts * bwd_len_mean
    total_pkts = fwd_pkts + bwd_pkts
    total_bytes = fwd_bytes + bwd_bytes
    seconds = duration / 1e6

    # --- derived, with the same relationships the real features have -----
    with np.errstate(divide="ignore", invalid="ignore"):
        flow_bytes_s = np.where(seconds > 0, total_bytes / seconds, np.inf)
        flow_pkts_s = np.where(seconds > 0, total_pkts / seconds, np.inf)
        fwd_pkts_s = np.where(seconds > 0, fwd_pkts / seconds, np.inf)
        bwd_pkts_s = np.where(seconds > 0, bwd_pkts / seconds, np.inf)
        avg_pkt_size = np.where(total_pkts > 0, total_bytes / total_pkts, 0.0)
        down_up = np.where(fwd_bytes > 0, np.round(bwd_bytes / fwd_bytes), 0.0)

    iat_mean = duration / np.maximum(total_pkts - 1, 1)
    iat_std = iat_mean * rng.uniform(0.2, 1.8, n)
    iat_max = iat_mean + iat_std * rng.uniform(1.0, 4.0, n)
    iat_min = np.clip(iat_mean - iat_std, 0, None) * rng.uniform(0.0, 0.6, n)

    fwd_share = rng.uniform(0.35, 0.75, n)
    fwd_iat_total = duration * fwd_share
    bwd_iat_total = duration * (1 - fwd_share)

    pkt_lens = np.stack([fwd_len_mean, bwd_len_mean])
    pkt_len_mean = np.where(total_pkts > 0,
                            (fwd_bytes + bwd_bytes) / np.maximum(total_pkts, 1), 0.0)
    pkt_len_std = pkt_lens.std(axis=0) * rng.uniform(0.6, 1.5, n)
    pkt_len_min = np.minimum(fwd_len_mean, bwd_len_mean) * rng.uniform(0.0, 0.4, n)
    pkt_len_max = np.maximum(fwd_len_mean, bwd_len_mean) * rng.uniform(1.1, 2.4, n)

    syn, fin, rst, psh, ack, urg = p["flags"]
    def _flag(expected: int) -> np.ndarray:
        if expected == 0:
            return np.zeros(n)
        return rng.poisson(expected, n).astype(float)

    idle_total = duration * p["idle"] * rng.uniform(0.7, 1.3, n)
    active_total = np.clip(duration - idle_total, 0, None)
    n_bursts = np.maximum(np.round(total_pkts / 8), 1)

    win_f, win_b = p["win"]

    values: Dict[str, np.ndarray] = {
        "Destination Port": port.astype(float),
        "Flow Duration": duration,
        "Total Fwd Packets": fwd_pkts,
        "Total Backward Packets": bwd_pkts,
        "Total Length of Fwd Packets": fwd_bytes,
        "Total Length of Bwd Packets": bwd_bytes,
        "Fwd Packet Length Max": fwd_len_mean * rng.uniform(1.0, 2.2, n),
        "Fwd Packet Length Min": fwd_len_mean * rng.uniform(0.0, 0.5, n),
        "Fwd Packet Length Mean": fwd_len_mean,
        "Fwd Packet Length Std": fwd_len_mean * rng.uniform(0.05, 0.7, n),
        "Bwd Packet Length Max": bwd_len_mean * rng.uniform(1.0, 2.2, n),
        "Bwd Packet Length Min": bwd_len_mean * rng.uniform(0.0, 0.5, n),
        "Bwd Packet Length Mean": bwd_len_mean,
        "Bwd Packet Length Std": bwd_len_mean * rng.uniform(0.05, 0.7, n),
        "Flow Bytes/s": flow_bytes_s,
        "Flow Packets/s": flow_pkts_s,
        "Flow IAT Mean": iat_mean,
        "Flow IAT Std": iat_std,
        "Flow IAT Max": iat_max,
        "Flow IAT Min": iat_min,
        "Fwd IAT Total": fwd_iat_total,
        "Fwd IAT Mean": fwd_iat_total / np.maximum(fwd_pkts - 1, 1),
        "Fwd IAT Std": iat_std * rng.uniform(0.5, 1.4, n),
        "Fwd IAT Max": fwd_iat_total * rng.uniform(0.3, 0.9, n),
        "Fwd IAT Min": iat_min * rng.uniform(0.5, 1.5, n),
        "Bwd IAT Total": bwd_iat_total,
        "Bwd IAT Mean": bwd_iat_total / np.maximum(bwd_pkts - 1, 1),
        "Bwd IAT Std": iat_std * rng.uniform(0.5, 1.4, n),
        "Bwd IAT Max": bwd_iat_total * rng.uniform(0.3, 0.9, n),
        "Bwd IAT Min": iat_min * rng.uniform(0.5, 1.5, n),
        "Fwd PSH Flags": _flag(psh) * 0.5,
        "Bwd PSH Flags": np.zeros(n),          # all-zero in the real dataset
        "Fwd URG Flags": _flag(urg),
        "Bwd URG Flags": np.zeros(n),          # all-zero in the real dataset
        "Fwd Header Length": fwd_pkts * 32,
        "Bwd Header Length": bwd_pkts * 32,
        "Fwd Packets/s": fwd_pkts_s,
        "Bwd Packets/s": bwd_pkts_s,
        "Min Packet Length": pkt_len_min,
        "Max Packet Length": pkt_len_max,
        "Packet Length Mean": pkt_len_mean,
        "Packet Length Std": pkt_len_std,
        "Packet Length Variance": pkt_len_std ** 2,
        "FIN Flag Count": _flag(fin),
        "SYN Flag Count": _flag(syn),
        "RST Flag Count": _flag(rst),
        "PSH Flag Count": _flag(psh),
        "ACK Flag Count": _flag(ack),
        "URG Flag Count": _flag(urg),
        # Set on about 1 flow in 1000. Not in KNOWN_CONSTANT_FEATURES because it is not
        # constant upstream either - it is rare, which is a different thing, and zeroing
        # it here used to make the generator produce nine all-zero columns against the
        # schema's documented eight. The preprocessor logs that mismatch as a surprise on
        # every synthetic run, which is noise about the fixture rather than the data.
        "CWE Flag Count": (rng.random(n) < 0.001).astype(float),
        "ECE Flag Count": _flag(1) * 0.05,
        "Down/Up Ratio": down_up,
        "Average Packet Size": avg_pkt_size,
        "Avg Fwd Segment Size": fwd_len_mean,
        "Avg Bwd Segment Size": bwd_len_mean,
        # These six plus Bwd PSH Flags and Bwd URG Flags above are the eight columns
        # schema.KNOWN_CONSTANT_FEATURES documents as never populated by CICFlowMeter.
        "Fwd Avg Bytes/Bulk": np.zeros(n),
        "Fwd Avg Packets/Bulk": np.zeros(n),
        "Fwd Avg Bulk Rate": np.zeros(n),
        "Bwd Avg Bytes/Bulk": np.zeros(n),
        "Bwd Avg Packets/Bulk": np.zeros(n),
        "Bwd Avg Bulk Rate": np.zeros(n),
        "Subflow Fwd Packets": fwd_pkts,
        "Subflow Fwd Bytes": fwd_bytes,
        "Subflow Bwd Packets": bwd_pkts,
        "Subflow Bwd Bytes": bwd_bytes,
        "Init_Win_bytes_forward": np.full(n, float(win_f)),
        "Init_Win_bytes_backward": np.full(n, float(win_b)),
        "act_data_pkt_fwd": np.clip(fwd_pkts - rng.integers(0, 3, n), 0, None),
        "min_seg_size_forward": rng.choice([20.0, 24.0, 32.0, 40.0], n),
        "Active Mean": active_total / n_bursts,
        "Active Std": (active_total / n_bursts) * rng.uniform(0.1, 1.2, n),
        "Active Max": (active_total / n_bursts) * rng.uniform(1.0, 3.0, n),
        "Active Min": (active_total / n_bursts) * rng.uniform(0.0, 0.8, n),
        "Idle Mean": idle_total / n_bursts,
        "Idle Std": (idle_total / n_bursts) * rng.uniform(0.1, 1.2, n),
        "Idle Max": (idle_total / n_bursts) * rng.uniform(1.0, 3.0, n),
        "Idle Min": (idle_total / n_bursts) * rng.uniform(0.0, 0.8, n),
    }

    missing = [c for c in sch.CANONICAL_FEATURES if c not in values]
    if missing:  # pragma: no cover - guards against edits to CANONICAL_FEATURES
        raise AssertionError(f"synthetic generator is missing features: {missing}")

    frame = pd.DataFrame({c: values[c] for c in sch.CANONICAL_FEATURES})
    frame[sch.LABEL_COLUMN] = name
    return frame


def _inject_defects(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Reproduce the real dataset's data-quality problems."""
    n = len(df)

    # 1. Zero-duration flows -> inf throughput. Already produced naturally by the
    #    generator for very short flows, but force a handful so the count is never 0.
    n_inf = max(1, int(n * 0.004))
    idx = rng.choice(n, n_inf, replace=False)
    df.loc[idx, "Flow Bytes/s"] = np.inf
    df.loc[idx[: n_inf // 2], "Flow Packets/s"] = np.inf

    # 2. Scattered NaN in the same two columns the real files have them in.
    n_nan = max(1, int(n * 0.003))
    for col in sch.DIVISION_ARTEFACT_FEATURES:
        nan_idx = rng.choice(n, n_nan, replace=False)
        df.loc[nan_idx, col] = np.nan

    # 3. Exact duplicate rows - CICIDS2017 has hundreds of thousands.
    n_dupes = max(1, int(n * 0.02))
    dupes = df.iloc[rng.choice(n, n_dupes, replace=False)].copy()
    df = pd.concat([df, dupes], ignore_index=True)

    log.debug("injected %d inf, %d NaN per column, %d duplicate rows",
              n_inf, n_nan, n_dupes)
    return df


# ---------------------------------------------------------------------------
# Raw-format writer - exercises the loader, not just the cleaner
# ---------------------------------------------------------------------------

def write_raw_like(
    directory: Path | str,
    *,
    n_rows: int = 20_000,
    seed: int = 42,
    n_files: int = 8,
) -> list[Path]:
    """Write synthetic data as *raw-format* CSVs, defects and all.

    The output deliberately reproduces the header and encoding defects described in
    :mod:`shieldnet.schema`: leading spaces on most headers, a duplicated
    ``Fwd Header Length.1`` column, the ultra-rare classes as their own labels rather
    than pre-merged, and web-attack labels encoded with cp1252 byte ``0x96`` rather
    than an ASCII hyphen.

    This is the fixture that proves :func:`shieldnet.data.load.load_raw` actually
    handles a real download, rather than only handling already-clean input.
    """
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = synthesise(n_rows, seed=seed, inject_defects=True)
    # Undo the category dtype: we are about to rewrite label values into spellings that
    # are not existing categories, which a categorical column rejects.
    df[sch.LABEL_COLUMN] = df[sch.LABEL_COLUMN].astype(str)

    # Undo the rare-class merge so the raw files carry the three original labels.
    rare_mask = df[sch.LABEL_COLUMN] == sch.RARE_CLASS_NAME
    if rare_mask.any():
        rng = np.random.default_rng(seed)
        df.loc[rare_mask, sch.LABEL_COLUMN] = rng.choice(
            sch.RARE_SOURCE_LABELS, int(rare_mask.sum())
        )

    # Re-encode the web-attack dash as the cp1252 byte the real files use.
    raw_label = df[sch.LABEL_COLUMN].replace({
        "Web Attack - Brute Force": "Web Attack \x96 Brute Force",
        "Web Attack - XSS": "Web Attack \x96 XSS",
        "Web Attack - Sql Injection": "Web Attack \x96 Sql Injection",
    })

    # Rebuild the raw header: a leading space on everything except the handful of
    # columns the real files leave unpadded.
    unpadded = {
        "Total Length of Fwd Packets", "Bwd Packet Length Max", "Flow Bytes/s",
        "Fwd IAT Total", "Bwd IAT Total", "Fwd PSH Flags", "Fwd Packets/s",
        "FIN Flag Count", "Subflow Fwd Packets", "Init_Win_bytes_forward",
        "Active Mean", "Idle Mean", "Bwd Avg Bulk Rate",
    }
    raw = pd.DataFrame()
    for col in sch.CANONICAL_FEATURES:
        raw[col if col in unpadded else f" {col}"] = df[col]
        if col == "Fwd Header Length":
            raw[" Fwd Header Length.1"] = df[col]   # the duplicated column
    raw[f" {sch.LABEL_COLUMN}"] = raw_label

    written: list[Path] = []
    parts = np.array_split(np.arange(len(raw)), n_files)
    names = sch.EXPECTED_RAW_FILES[:n_files]
    for name, part in zip(names, parts):
        path = out_dir / name
        # latin-1 turns U+0096 back into byte 0x96, exactly as upstream ships it.
        raw.iloc[part].to_csv(path, index=False, encoding="latin-1")
        written.append(path)

    log.info("wrote %d raw-format CSVs (%s rows total) -> %s",
             len(written), f"{len(raw):,}", out_dir)
    return written
