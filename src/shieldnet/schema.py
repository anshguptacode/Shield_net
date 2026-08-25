"""Canonical CICIDS2017 schema, header/label normalisation and the 13-class scheme.

Everything in ShieldNet speaks *canonical* column names. The raw Kaggle
``MachineLearningCVE`` CSVs do not: they carry a pile of well-documented defects that
break naive pipelines. This module is the single place that knows about them.

The defects we normalise away
-----------------------------
1. **Leading spaces on most headers.** The raw files ship ``" Flow Duration"``,
   ``" Label"``, ``" Destination Port"`` and so on - but *not* consistently
   (``"Flow Bytes/s"`` and ``"Fwd IAT Total"`` have none). Selecting
   ``df["Flow Duration"]`` on raw data raises ``KeyError``.
2. **A duplicated column.** ``Fwd Header Length`` appears twice; pandas silently
   renames the second one ``Fwd Header Length.1``. It is an exact copy and must be
   dropped or it inflates every correlation and feature-importance ranking.
3. **cp1252 label encoding.** The web-attack labels contain byte ``0x96`` (an en dash
   in Windows-1252), so the files are *not* valid UTF-8. Reading them with the default
   encoding either raises ``UnicodeDecodeError`` or yields mojibake such as
   ``"Web Attack \\ufffd Brute Force"``, which then fails a string comparison against
   any hand-typed label. See :func:`normalise_label`.
4. **Inconsistent label casing.** ``DoS slowloris`` and ``DoS Slowhttptest`` disagree
   about capitalisation; ``Sql Injection`` is not ``SQL Injection``.
5. **Eight all-zero feature columns**, listed in :data:`KNOWN_CONSTANT_FEATURES`.
   They carry no information and make several scikit-learn selectors emit
   divide-by-zero warnings. We still detect constants dynamically rather than trusting
   this list, because it varies between redistributed copies of the dataset.

Canonical form
--------------
A canonical name is the raw name with surrounding whitespace stripped and internal
runs of whitespace collapsed to one space. Original capitalisation and punctuation are
kept so that names still match the ones used in the CICIDS2017 literature.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List, Mapping, Sequence

__all__ = [
    "SCHEMA_VERSION",
    "LABEL_COLUMN",
    "CANONICAL_FEATURES",
    "DUPLICATE_COLUMNS",
    "KNOWN_CONSTANT_FEATURES",
    "RAW_LABELS",
    "RARE_SOURCE_LABELS",
    "RARE_CLASS_NAME",
    "CLASS_SCHEME_13",
    "ATTACK_FAMILY",
    "FEATURE_GLOSSARY",
    "MANUAL_ENTRY_FEATURES",
    "EXPECTED_RAW_FILES",
    "normalise_column",
    "normalise_columns",
    "normalise_label",
    "collapse_rare",
    "canonical_label",
    "family_of",
    "LABEL_COLUMN",
    "BENIGN_LABEL",
    "describe_feature",
    "validate_feature_frame",
    "SchemaError",
]

SCHEMA_VERSION = "1.0.0"

#: Canonical name of the target column (raw files use ``" Label"``).
LABEL_COLUMN = "Label"

#: The negative class. Named rather than assumed to be index 0, because the whole
#: attack-vs-benign collapse in :mod:`shieldnet.evaluate` hinges on finding it, and a
#: silent fallback to column 0 would invert every false-alarm figure in the report.
BENIGN_LABEL = "BENIGN"


class SchemaError(ValueError):
    """Raised when a frame does not satisfy the schema ShieldNet expects."""


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

#: The 77 unique flow features, in the order the raw files present them.
#:
#: The raw header row has 79 entries: these 77, plus the duplicated
#: ``Fwd Header Length.1`` (see :data:`DUPLICATE_COLUMNS`), plus ``Label``.
CANONICAL_FEATURES: List[str] = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]

#: Columns that are exact duplicates of another column and are dropped on load.
DUPLICATE_COLUMNS: Dict[str, str] = {"Fwd Header Length.1": "Fwd Header Length"}

#: Features that are identically zero in the published CICIDS2017 release.
#:
#: Kept for documentation and for cross-checking the dynamic constant detector in
#: :mod:`shieldnet.preprocess`; never used as a hard-coded drop list, because
#: redistributed copies of the dataset differ.
KNOWN_CONSTANT_FEATURES: List[str] = [
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
]

#: Features that legitimately contain ``inf`` in the raw data, because CICFlowMeter
#: divides by a flow duration that can be zero for single-packet flows.
DIVISION_ARTEFACT_FEATURES: List[str] = ["Flow Bytes/s", "Flow Packets/s"]


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

#: The 15 canonical labels present in the raw dataset, after normalisation.
RAW_LABELS: List[str] = [
    "BENIGN",
    "DoS Hulk",
    "PortScan",
    "DDoS",
    "DoS GoldenEye",
    "FTP-Patator",
    "SSH-Patator",
    "DoS slowloris",
    "DoS Slowhttptest",
    "Bot",
    "Web Attack - Brute Force",
    "Web Attack - XSS",
    "Infiltration",
    "Web Attack - Sql Injection",
    "Heartbleed",
]

#: Ultra-rare labels merged into one class: Heartbleed 11, Infiltration 36, Web Attack -
#: Sql Injection 21 - 68 rows between them, listed below in that order.
#:
#: A class with 11 examples cannot be stratified into train/validation/test *and*
#: evaluated with any stability, and it breaks SMOTE's k-nearest-neighbour step.
RARE_SOURCE_LABELS: List[str] = [
    "Heartbleed",
    "Infiltration",
    "Web Attack - Sql Injection",
]

RARE_CLASS_NAME = "Rare Attacks"

#: The 13-class target scheme, in descending order of support.
CLASS_SCHEME_13: List[str] = [
    "BENIGN",
    "DoS Hulk",
    "PortScan",
    "DDoS",
    "DoS GoldenEye",
    "FTP-Patator",
    "SSH-Patator",
    "DoS slowloris",
    "DoS Slowhttptest",
    "Bot",
    "Web Attack - Brute Force",
    "Web Attack - XSS",
    RARE_CLASS_NAME,
]

#: Coarse family for each class, used for grouped reporting in the web app.
ATTACK_FAMILY: Dict[str, str] = {
    "BENIGN": "Benign",
    "DoS Hulk": "Denial of Service",
    "DoS GoldenEye": "Denial of Service",
    "DoS slowloris": "Denial of Service",
    "DoS Slowhttptest": "Denial of Service",
    "DDoS": "Distributed Denial of Service",
    "PortScan": "Reconnaissance",
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    "Bot": "Botnet",
    "Web Attack - Brute Force": "Web Attack",
    "Web Attack - XSS": "Web Attack",
    RARE_CLASS_NAME: "Rare / Low-Support",
}

#: Published row counts for the full dataset, used only to sanity-check a load.
#:
#: Exact counts differ by a few rows between redistributed copies, so
#: :func:`shieldnet.data.load.audit_class_counts` compares with a tolerance rather
#: than asserting equality.
REFERENCE_CLASS_COUNTS: Dict[str, int] = {
    "BENIGN": 2273097,
    "DoS Hulk": 231073,
    "PortScan": 158930,
    "DDoS": 128027,
    "DoS GoldenEye": 10293,
    "FTP-Patator": 7938,
    "SSH-Patator": 5897,
    "DoS slowloris": 5796,
    "DoS Slowhttptest": 5499,
    "Bot": 1966,
    "Web Attack - Brute Force": 1507,
    "Web Attack - XSS": 652,
    "Infiltration": 36,
    "Web Attack - Sql Injection": 21,
    "Heartbleed": 11,
}

#: The eight raw CSVs, with the dataset's own spelling quirks preserved.
#:
#: Note ``Wednesday-workingHours`` (lower-case *w*) and ``Infilteration``
#: (misspelled) - both are how the files are actually named upstream. The loader
#: globs for ``*.csv`` rather than requiring these names, but reports which of them
#: it could not find.
EXPECTED_RAW_FILES: List[str] = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_WS_RUN = re.compile(r"\s+")

# Every dash-like character that shows up in the label column across the various
# redistributed copies of CICIDS2017, written as explicit escapes so that no invisible
# control character is left to chance.
#
# The critical two are U+0096 and U+0097. Raw byte 0x96 is an en dash in Windows-1252,
# but reading the file as latin-1 - which is what you must do to avoid a
# UnicodeDecodeError - maps it to the C1 control character U+0096 instead. That is not
# a dash as far as Unicode is concerned, so NFKC normalisation leaves it alone and it
# never matches a hand-typed "-". Reading as cp1252 yields U+2013 instead, so both
# spellings have to be accepted. U+FFFD covers a lossy UTF-8 decode with
# errors="replace".
_DASHES = (
    "\u0096"  # cp1252 en dash read as latin-1
    "\u0097"  # cp1252 em dash read as latin-1
    "\u2010"  # hyphen
    "\u2011"  # non-breaking hyphen
    "\u2012"  # figure dash
    "\u2013"  # en dash
    "\u2014"  # em dash
    "\u2015"  # horizontal bar
    "\u2212"  # minus sign
    "\ufffd"  # replacement char from a lossy decode
)
_DASH_RUN = re.compile("[" + _DASHES + "]+")


def normalise_column(name: str) -> str:
    """Return the canonical form of one raw column name.

    Strips surrounding whitespace (including the non-breaking space some exports
    use) and collapses internal whitespace runs to a single space.

    >>> normalise_column(" Flow Duration")
    'Flow Duration'
    >>> normalise_column("Fwd  IAT   Total")
    'Fwd IAT Total'
    """
    if not isinstance(name, str):
        name = str(name)
    cleaned = name.replace("\xa0", " ").strip()
    return _WS_RUN.sub(" ", cleaned)


def normalise_columns(names: Iterable[str]) -> List[str]:
    """Canonicalise an iterable of column names, preserving order."""
    return [normalise_column(n) for n in names]


def normalise_label(value: str) -> str:
    """Return the canonical label for one raw label cell.

    Handles the cp1252 dash, mojibake from a lossy decode, stray whitespace and the
    dataset's inconsistent capitalisation. Unknown labels are returned in cleaned
    form rather than raising, so an unexpected value surfaces in a value-count table
    instead of aborting a two-million-row load.

    >>> normalise_label("Web Attack \\x96 Brute Force")
    'Web Attack - Brute Force'
    >>> normalise_label("web attack - xss")
    'Web Attack - XSS'
    >>> normalise_label(" DoS  slowloris ")
    'DoS slowloris'
    """
    if not isinstance(value, str):
        value = str(value)

    # Decompose any composed characters, then map every dash variant to ASCII '-'.
    text = unicodedata.normalize("NFKC", value)
    text = text.replace("\xa0", " ")
    text = _DASH_RUN.sub("-", text)
    text = _WS_RUN.sub(" ", text).strip()
    # Normalise spacing around the dash: "Web Attack-XSS" -> "Web Attack - XSS".
    text = re.sub(r"\s*-\s*", " - ", text)
    text = _WS_RUN.sub(" ", text).strip()

    # Patator labels are genuinely hyphenated with no spaces; undo the split above.
    text = re.sub(r"\b(FTP|SSH)\s+-\s+(Patator)\b", r"\1-\2", text, flags=re.I)

    return _LABEL_LOOKUP.get(_fold(text), text)


def _fold(text: str) -> str:
    """Case- and space-insensitive key for label matching."""
    return re.sub(r"[\s_]+", "", text).lower()


# Built after normalise_label so every canonical label maps to itself, plus the
# spelling variants seen in the wild.
_LABEL_LOOKUP: Dict[str, str] = {}
for _canon in RAW_LABELS:
    _LABEL_LOOKUP[_fold(_canon)] = _canon
for _variant, _target in {
    "web attack - sql injection": "Web Attack - Sql Injection",
    "web attack - sqlinjection": "Web Attack - Sql Injection",
    "sql injection": "Web Attack - Sql Injection",
    "web attack - brute force": "Web Attack - Brute Force",
    "brute force": "Web Attack - Brute Force",
    "web attack - xss": "Web Attack - XSS",
    "xss": "Web Attack - XSS",
    "dos slowloris": "DoS slowloris",
    "dos slowhttptest": "DoS Slowhttptest",
    "dos hulk": "DoS Hulk",
    "dos goldeneye": "DoS GoldenEye",
    "ddos": "DDoS",
    "portscan": "PortScan",
    "port scan": "PortScan",
    "benign": "BENIGN",
    "normal": "BENIGN",
    "bot": "Bot",
    "botnet": "Bot",
    "botnet ares": "Bot",
    "heartbleed": "Heartbleed",
    "infiltration": "Infiltration",
    "infilteration": "Infiltration",
    "ftp-patator": "FTP-Patator",
    "ssh-patator": "SSH-Patator",
}.items():
    _LABEL_LOOKUP[_fold(_variant)] = _target


def collapse_rare(label: str) -> str:
    """Map the three ultra-rare labels onto :data:`RARE_CLASS_NAME`.

    Any other label passes through untouched, so this is safe to apply to a whole
    column after :func:`normalise_label`.
    """
    return RARE_CLASS_NAME if label in RARE_SOURCE_LABELS else label


def canonical_label(value: str) -> str:
    """Normalise *and* collapse in one step - the 13-class target for a raw cell."""
    return collapse_rare(normalise_label(value))


def family_of(label: str) -> str:
    """Coarse attack family for a 13-class label."""
    return ATTACK_FAMILY.get(label, "Unknown")


# ---------------------------------------------------------------------------
# Human-readable feature glossary (used by the SHAP narrator and the app form)
# ---------------------------------------------------------------------------

#: Plain-English gloss for each feature, phrased to slot into a sentence after
#: "this flow's ...". Keys are canonical names.
FEATURE_GLOSSARY: Dict[str, str] = {
    "Destination Port": "destination port number",
    "Flow Duration": "total duration of the conversation in microseconds",
    "Total Fwd Packets": "number of packets sent from client to server",
    "Total Backward Packets": "number of packets sent from server back to client",
    "Total Length of Fwd Packets": "total bytes sent from client to server",
    "Total Length of Bwd Packets": "total bytes sent back from server to client",
    "Fwd Packet Length Max": "largest packet sent by the client",
    "Fwd Packet Length Min": "smallest packet sent by the client",
    "Fwd Packet Length Mean": "average client packet size",
    "Fwd Packet Length Std": "variability of client packet sizes",
    "Bwd Packet Length Max": "largest packet returned by the server",
    "Bwd Packet Length Min": "smallest packet returned by the server",
    "Bwd Packet Length Mean": "average server packet size",
    "Bwd Packet Length Std": "variability of server packet sizes",
    "Flow Bytes/s": "throughput in bytes per second",
    "Flow Packets/s": "packet rate per second",
    "Flow IAT Mean": "average gap between consecutive packets",
    "Flow IAT Std": "variability of the gaps between packets",
    "Flow IAT Max": "longest gap between two packets",
    "Flow IAT Min": "shortest gap between two packets",
    "Fwd IAT Total": "total time spent between client packets",
    "Fwd IAT Mean": "average gap between client packets",
    "Fwd IAT Std": "variability of gaps between client packets",
    "Fwd IAT Max": "longest gap between client packets",
    "Fwd IAT Min": "shortest gap between client packets",
    "Bwd IAT Total": "total time spent between server packets",
    "Bwd IAT Mean": "average gap between server packets",
    "Bwd IAT Std": "variability of gaps between server packets",
    "Bwd IAT Max": "longest gap between server packets",
    "Bwd IAT Min": "shortest gap between server packets",
    "Fwd PSH Flags": "count of PSH flags set by the client",
    "Bwd PSH Flags": "count of PSH flags set by the server",
    "Fwd URG Flags": "count of URG (urgent) flags set by the client",
    "Bwd URG Flags": "count of URG flags set by the server",
    "Fwd Header Length": "total bytes of client-side packet headers",
    "Bwd Header Length": "total bytes of server-side packet headers",
    "Fwd Packets/s": "client packet rate per second",
    "Bwd Packets/s": "server packet rate per second",
    "Min Packet Length": "smallest packet anywhere in the flow",
    "Max Packet Length": "largest packet anywhere in the flow",
    "Packet Length Mean": "average packet size across the flow",
    "Packet Length Std": "variability of packet sizes across the flow",
    "Packet Length Variance": "variance of packet sizes across the flow",
    "FIN Flag Count": "number of FIN (connection-finish) flags",
    "SYN Flag Count": "number of SYN (connection-open) flags",
    "RST Flag Count": "number of RST (connection-reset) flags",
    "PSH Flag Count": "number of PSH (push-data) flags",
    "ACK Flag Count": "number of ACK (acknowledgement) flags",
    "URG Flag Count": "number of URG (urgent) flags",
    "CWE Flag Count": "number of CWE congestion flags",
    "ECE Flag Count": "number of ECE congestion-echo flags",
    "Down/Up Ratio": "ratio of download volume to upload volume",
    "Average Packet Size": "mean size of every packet in the flow",
    "Avg Fwd Segment Size": "average TCP segment size sent by the client",
    "Avg Bwd Segment Size": "average TCP segment size sent by the server",
    "Fwd Avg Bytes/Bulk": "average bytes per client bulk transfer",
    "Fwd Avg Packets/Bulk": "average packets per client bulk transfer",
    "Fwd Avg Bulk Rate": "client bulk transfer rate",
    "Bwd Avg Bytes/Bulk": "average bytes per server bulk transfer",
    "Bwd Avg Packets/Bulk": "average packets per server bulk transfer",
    "Bwd Avg Bulk Rate": "server bulk transfer rate",
    "Subflow Fwd Packets": "average packets per client subflow",
    "Subflow Fwd Bytes": "average bytes per client subflow",
    "Subflow Bwd Packets": "average packets per server subflow",
    "Subflow Bwd Bytes": "average bytes per server subflow",
    "Init_Win_bytes_forward": "initial TCP receive window advertised by the client",
    "Init_Win_bytes_backward": "initial TCP receive window advertised by the server",
    "act_data_pkt_fwd": "number of client packets carrying actual payload",
    "min_seg_size_forward": "smallest client segment size observed",
    "Active Mean": "average length of the flow's active bursts",
    "Active Std": "variability in the length of active bursts",
    "Active Max": "longest active burst",
    "Active Min": "shortest active burst",
    "Idle Mean": "average length of the flow's idle periods",
    "Idle Std": "variability in the length of idle periods",
    "Idle Max": "longest idle period",
    "Idle Min": "shortest idle period",
}

#: Features offered on the app's manual single-flow entry form.
#:
#: Chosen to be the ones a human can reason about without a packet capture in front
#: of them; every other feature is imputed from the training-set median.
#: Each entry is ``(canonical name, minimum, maximum, demo default)``.
MANUAL_ENTRY_FEATURES: List[tuple] = [
    ("Destination Port", 0, 65535, 80),
    ("Flow Duration", 0, 120_000_000, 1_500_000),
    ("Total Fwd Packets", 0, 20_000, 6),
    ("Total Backward Packets", 0, 20_000, 4),
    ("Total Length of Fwd Packets", 0, 20_000_000, 720),
    ("Total Length of Bwd Packets", 0, 20_000_000, 1240),
    ("Flow Packets/s", 0, 3_000_000, 12.0),
    ("Flow IAT Mean", 0, 120_000_000, 180_000.0),
    ("Packet Length Mean", 0, 20_000, 190.0),
    ("Average Packet Size", 0, 20_000, 210.0),
    ("SYN Flag Count", 0, 100, 1),
    ("ACK Flag Count", 0, 100, 1),
]


def describe_feature(name: str) -> str:
    """Plain-English gloss for *name*, falling back to the name itself."""
    return FEATURE_GLOSSARY.get(name, name)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_feature_frame(
    columns: Sequence[str],
    required: Sequence[str],
    *,
    context: str = "input",
) -> Mapping[str, List[str]]:
    """Compare *columns* against *required* and describe the difference.

    Returns a mapping with ``missing`` and ``extra`` keys. Callers decide whether a
    difference is fatal - the Streamlit app, for instance, tolerates extra columns
    (a stray index or a ``Label`` column) but not missing ones.
    """
    have = {normalise_column(c) for c in columns}
    want = [normalise_column(c) for c in required]
    missing = [c for c in want if c not in have]
    extra = sorted(have - set(want))
    if missing and len(missing) == len(want):
        raise SchemaError(
            f"None of the {len(want)} required feature columns were found in {context}. "
            "This does not look like CICFlowMeter output - check that you uploaded a "
            "flow-feature CSV and not a packet capture or a results file."
        )
    return {"missing": missing, "extra": extra}
