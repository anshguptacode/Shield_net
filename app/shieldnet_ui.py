"""Everything the Streamlit app does, minus Streamlit.

Why this file exists
--------------------
A Streamlit script cannot be tested. It is a module whose every interesting line is
guarded by a widget call, so importing it runs the UI and asserting on it means driving
a browser. The usual consequence is an app that is the least-tested part of a project
while being the only part anyone looks at.

So all of the logic lives here as ordinary functions that take a
:class:`~shieldnet.inference.Detector` and return DataFrames, dicts and strings, and
``streamlit_app.py`` is a thin layer of widgets over the top. ``tests/check_app.py``
exercises this module against a real trained artifact, which means the numbers on the
screen are checked even though the screen is not.

Nothing here imports streamlit, plotly or matplotlib at module scope. That is deliberate:
the module must be importable in a bare environment so the test can run.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from shieldnet import schema as sch
from shieldnet.inference import Detector, PredictionBatch, PreparedInput, read_flows
from shieldnet.narrate import PROFILES, profile_for, severity_label

__all__ = [
    "APP_TITLE", "APP_TAGLINE", "SEVERITY_COLOURS", "PLOT_COLOURS",
    "find_artifacts", "load_detector", "model_card", "provenance_lines",
    "class_reference_table", "feature_table", "selection_caption",
    "manual_form_groups",
    "sample_files", "sample_rows_by_class", "SHAPE_PRESETS",
    "score_frame", "apply_thresholds", "batch_overview", "reconciliation_note",
    "class_distribution", "triage_table", "confidence_histogram",
    "verdict_table", "csv_bytes", "explanation_table", "probability_table",
    "evaluation_overview", "per_class_table", "confusion_frame", "sweep_table",
    "sweep_recommendations", "binary_table", "format_metric", "humanise",
]

APP_TITLE = "ShieldNet"
APP_TAGLINE = "Explainable multi-class intrusion detection on CICIDS2017 flow features"

#: Severity 0-5 -> a colour. Deliberately not a red/green pair: severity is ordinal, so
#: it wants an ordered ramp, and an analyst scanning a queue needs to see 4-vs-5 at a
#: glance rather than "bad" vs "fine".
SEVERITY_COLOURS: Dict[int, str] = {
    0: "#2e7d5b",   # benign
    1: "#2e7d5b",
    2: "#8a8f2f",
    3: "#c47b1a",
    4: "#c04f1a",
    5: "#a8202a",
}

#: Categorical palette for the class charts. Twelve distinguishable hues plus a
#: deliberately muted green for BENIGN, which is 80.3% of CICIDS2017 and the majority of
#: any real capture, and would otherwise dominate the eye as well as the axis.
PLOT_COLOURS: List[str] = [
    "#5b8c76", "#a8202a", "#c04f1a", "#c47b1a", "#b8a032", "#6b8f3a",
    "#2f7f7f", "#3a6ea8", "#5a4fa8", "#8a3fa0", "#a03f6e", "#7a5c3a", "#4f5a66",
]


# ---------------------------------------------------------------------------
# finding and loading the artifact
# ---------------------------------------------------------------------------

def find_artifacts(explicit: Optional[os.PathLike | str] = None) -> Path:
    """Locate the artifacts directory, in the order a user would expect.

    ``shieldnet serve`` exports ``SHIELDNET_ARTIFACTS`` before launching streamlit, so
    the app picks up whatever ``--artifacts`` the command was given. Someone running
    ``streamlit run app/streamlit_app.py`` by hand gets the project's own directory.
    Returned whether or not it exists - :func:`load_detector` produces the better error.
    """
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("SHIELDNET_ARTIFACTS")
    if env:
        return Path(env).expanduser()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent / "artifacts"
    return Path("artifacts")


def load_detector(artifacts: Optional[os.PathLike | str] = None, *,
                  attack_threshold: float = 0.5,
                  min_confidence: float = 0.5) -> Detector:
    """Load the shipped bundle. Wrap in ``st.cache_resource`` at the call site."""
    return Detector.load(find_artifacts(artifacts),
                         attack_threshold=attack_threshold,
                         min_confidence=min_confidence)


# ---------------------------------------------------------------------------
# describing the model
# ---------------------------------------------------------------------------

def model_card(det: Detector) -> Dict[str, Any]:
    """Headline facts about the loaded artifact, ready for ``st.metric``.

    Every value is either a number the training run measured or ``None``. Nothing is
    recomputed here, because a metric that the app calculates differently from the
    training report is worse than no metric: the two disagree and neither is trusted.
    """
    m = det.metrics or {}
    meta = det.bundle.metadata or {}
    return {
        "model": det.model_name,
        "n_features": det.n_features,
        "n_classes": det.n_classes,
        "split": m.get("split", "test"),
        "macro_f1": m.get("macro_f1"),
        "accuracy": m.get("accuracy"),
        "balanced_accuracy": m.get("balanced_accuracy"),
        "attack_recall": m.get("attack_recall"),
        "false_alarm_rate": m.get("false_alarm_rate"),
        "macro_roc_auc": m.get("macro_roc_auc"),
        "calibration_error": m.get("calibration_error"),
        "never_predicted": list(m.get("classes_never_predicted") or []),
        "selection_metric": m.get("selection_metric", "macro_f1"),
        "synthetic": bool(meta.get("synthetic")),
        "source": meta.get("source", "unrecorded"),
        "trained_at": meta.get("started_at", ""),
        "seed": meta.get("seed"),
        "tuned": bool(meta.get("tuned")),
        "balance": meta.get("balance", ""),
        "smote_rows": meta.get("smote_rows"),
        "rows_train": meta.get("rows_train"),
        "rows_test": meta.get("rows_test"),
        "explanation_method": meta.get("explanation_method", "not computed"),
        "candidates": list(meta.get("candidates") or []),
    }


def provenance_lines(det: Detector) -> List[str]:
    """Sentences that answer "where did this model come from?".

    Shown in the sidebar rather than buried, because the single most damaging thing this
    app could do is present a number from a 12,000-row synthetic smoke test as a
    CICIDS2017 result. The first line makes that impossible to miss.
    """
    card = model_card(det)
    out: List[str] = []
    if card["synthetic"]:
        out.append(
            "**Synthetic data.** This artifact was trained on generated traffic, not on "
            "CICIDS2017. Every score below describes the generator. Retrain with "
            "`shieldnet train` on the real dataset before quoting anything."
        )
    out.append(f"Trained on {card['source']}.")
    if card["rows_train"]:
        out.append(f"{int(card['rows_train']):,} training rows, "
                   f"{int(card['rows_test'] or 0):,} held-out test rows.")
    if card["trained_at"]:
        out.append(f"Run started {card['trained_at']}.")
    out.append(
        f"Selected on {card['selection_metric']} across "
        f"{len(card['candidates']) or 1} candidate model(s)"
        + (", hyper-parameters tuned by Optuna." if card["tuned"]
           else ", default hyper-parameters (tuning off).")
    )
    out.append(f"Class balance: {card['balance'] or 'none'}"
               + (f", {int(card['smote_rows']):,} rows synthesised by SMOTE on the "
                  "training split only." if card.get("smote_rows") else "."))
    out.append(f"Attributions: {card['explanation_method']}.")
    if card["never_predicted"]:
        out.append(
            "**Never predicted on the test split:** "
            + ", ".join(card["never_predicted"])
            + ". The model has learnt nothing usable about these classes; treat their "
              "absence from any result below as a property of the model, not of the "
              "traffic."
        )
    return out


def class_reference_table(det: Detector) -> pd.DataFrame:
    """Every class the model can output, with what it is and how well it is detected.

    Recall comes from the training run's held-out test split - it is the honest answer
    to "if this class shows up, will the model catch it?", and it belongs next to the
    class name so that a confident PortScan verdict and a confident Bot verdict are not
    read as equally trustworthy when their recalls differ by forty points.

    Ordered by class index, deliberately, even though severity order would read better on
    its own. This is a lookup table, and the two things a reader looks up beside it - the
    per-class metrics and the confusion matrix axes - are both in class order. Three
    tables in three different orders costs more than one table in the second-best order,
    and the ``severity`` column is there to sort by on screen.
    """
    recalls = (det.metrics or {}).get("per_class_recall") or {}
    rows = []
    for i, name in enumerate(det.class_names):
        profile = profile_for(name)
        rows.append({
            "class": name,
            "family": sch.family_of(name),
            "severity": profile.severity if name != sch.BENIGN_LABEL else 0,
            "severity_label": severity_label(profile.severity)
                              if name != sch.BENIGN_LABEL else "benign",
            "test_recall": (float(recalls[name]) if name in recalls else np.nan),
            "what_it_is": profile.summary,
            "on_the_wire": profile.on_the_wire,
            "what_to_do": profile.action,
            "confusable_with": ", ".join(profile.confusable_with) or "-",
            "index": i,
        })
    return pd.DataFrame(rows)


def selection_caption(det: Detector) -> str:
    """One sentence naming the rankers that actually produced this feature order.

    Read from the artifact, never hardcoded. The default config runs three filters, but
    ``config/fast.yaml`` runs two, RFE removes itself when scikit-learn is missing, and a
    ranker that raises is recorded as skipped rather than killing the run - so "the three
    filter methods agreed on this" is a sentence that is only sometimes true, and it is
    false in exactly the situations where a reader most needs to know what was used.
    """
    selection = (det.bundle.metadata or {}).get("feature_selection") or {}
    ran = [str(m) for m in (selection.get("methods") or [])]
    skipped = dict(selection.get("skipped") or {})
    if not ran:
        head = ("This is the order the training run's feature ranking produced, so "
                "reading down the list is reading the model's own account of what "
                "matters.")
    else:
        names = ", ".join(ran[:-1]) + f" and {ran[-1]}" if len(ran) > 1 else ran[0]
        agreed = "agreed on" if len(ran) > 1 else "produced"
        head = (f"This is the order {'all ' if len(ran) > 2 else ''}{len(ran)} filter "
                f"method{'s' if len(ran) > 1 else ''} ({names}) {agreed}, so reading "
                "down the list is reading the model's own account of what matters.")
    if skipped:
        head += (" Skipped: "
                 + "; ".join(f"{name} ({why})" for name, why in sorted(skipped.items()))
                 + ".")
    return head


def feature_table(det: Detector) -> pd.DataFrame:
    """The selected features in selection order, glossed.

    Selection order is the interesting order: it is the ranking the filters that ran
    agreed on - :func:`selection_caption` says which ones those were - so reading down the
    list is reading the model's own account of what matters. Sorting alphabetically would
    throw that away.
    """
    top = set((det.bundle.metadata or {}).get("top_features") or [])
    return pd.DataFrame([
        {
            "rank": i + 1,
            "feature": name,
            "means": sch.describe_feature(name),
            "training_median": float((det.bundle.medians or {}).get(name, float("nan"))),
            "top_by_shap": name in top,
        }
        for i, name in enumerate(det.feature_names)
    ])


# ---------------------------------------------------------------------------
# manual single-flow entry
# ---------------------------------------------------------------------------

#: Manual-entry fields grouped for the form, so the layout says something about the
#: features rather than being twelve boxes in schema order.
_FIELD_GROUPS: List[Tuple[str, Tuple[str, ...]]] = [
    ("Endpoint and duration", ("Destination Port", "Flow Duration")),
    ("Packet counts", ("Total Fwd Packets", "Total Backward Packets")),
    ("Bytes moved", ("Total Length of Fwd Packets", "Total Length of Bwd Packets")),
    ("Rates and timing", ("Flow Packets/s", "Flow IAT Mean")),
    ("Packet sizes", ("Packet Length Mean", "Average Packet Size")),
    ("TCP flags", ("SYN Flag Count", "ACK Flag Count")),
]


def manual_form_groups(det: Detector) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """``[(group title, [field spec, ...])]`` for the single-flow form.

    Each spec carries ``used_by_model``. Showing a field the selected model ignores is
    not a bug - the user typed a value and deserves to see it accepted - but the form
    has to say so, or someone will spend ten minutes changing SYN Flag Count and
    reasonably conclude the model is broken when the verdict does not move.
    """
    specs = {f["name"]: f for f in det.manual_fields()}
    groups: List[Tuple[str, List[Dict[str, Any]]]] = []
    placed = set()
    for title, names in _FIELD_GROUPS:
        block = [specs[n] for n in names if n in specs]
        placed.update(n for n in names if n in specs)
        if block:
            groups.append((title, block))
    leftover = [s for n, s in specs.items() if n not in placed]
    if leftover:
        groups.append(("Other", leftover))
    return groups


#: Hand-built flow shapes, offered as a starting point for the form.
#:
#: These are NOT captured flows and must never be presented as such. Each one sets the
#: dozen human-reasonable features to values characteristic of the named attack and
#: leaves the other ~65 at their training median, so the model sees a chimera: the timing
#: of a slowloris attack with the median attack-free value for every window and flag
#: feature. The model may well disagree with the label in the key, and when it does that
#: is information - it says the shape needs the features the form cannot offer.
SHAPE_PRESETS: Dict[str, Dict[str, float]] = {
    "Typical benign web request": {
        "Destination Port": 443, "Flow Duration": 1_800_000,
        "Total Fwd Packets": 9, "Total Backward Packets": 11,
        "Total Length of Fwd Packets": 1_120, "Total Length of Bwd Packets": 8_400,
        "Flow Packets/s": 11.1, "Flow IAT Mean": 94_000,
        "Packet Length Mean": 476, "Average Packet Size": 500,
        "SYN Flag Count": 1, "ACK Flag Count": 1,
    },
    "Port-scan shape (many tiny one-way flows)": {
        "Destination Port": 3389, "Flow Duration": 62,
        "Total Fwd Packets": 1, "Total Backward Packets": 0,
        "Total Length of Fwd Packets": 0, "Total Length of Bwd Packets": 0,
        "Flow Packets/s": 16_129.0, "Flow IAT Mean": 0,
        "Packet Length Mean": 0, "Average Packet Size": 0,
        "SYN Flag Count": 1, "ACK Flag Count": 0,
    },
    "Flood shape (high rate, high volume)": {
        "Destination Port": 80, "Flow Duration": 1_400_000,
        "Total Fwd Packets": 1_600, "Total Backward Packets": 4,
        "Total Length of Fwd Packets": 1_900_000, "Total Length of Bwd Packets": 260,
        "Flow Packets/s": 1_145.0, "Flow IAT Mean": 870,
        "Packet Length Mean": 1_186, "Average Packet Size": 1_190,
        "SYN Flag Count": 0, "ACK Flag Count": 1,
    },
    "Slow-read shape (long duration, trickle of packets)": {
        "Destination Port": 80, "Flow Duration": 96_000_000,
        "Total Fwd Packets": 14, "Total Backward Packets": 6,
        "Total Length of Fwd Packets": 1_960, "Total Length of Bwd Packets": 320,
        "Flow Packets/s": 0.21, "Flow IAT Mean": 5_050_000,
        "Packet Length Mean": 114, "Average Packet Size": 120,
        "SYN Flag Count": 1, "ACK Flag Count": 1,
    },
    "Credential-guessing shape (short repeated sessions)": {
        "Destination Port": 21, "Flow Duration": 3_100_000,
        "Total Fwd Packets": 12, "Total Backward Packets": 14,
        "Total Length of Fwd Packets": 148, "Total Length of Bwd Packets": 640,
        "Flow Packets/s": 8.4, "Flow IAT Mean": 124_000,
        "Packet Length Mean": 30, "Average Packet Size": 33,
        "SYN Flag Count": 1, "ACK Flag Count": 1,
    },
}


def sample_files(det: Optional[Detector] = None,
                 extra: Optional[os.PathLike | str] = None) -> List[Path]:
    """CSVs under ``data/samples`` that the user can score without finding a file."""
    roots: List[Path] = []
    if extra:
        roots.append(Path(extra).expanduser())
    base = find_artifacts().parent
    roots.append(base / "data" / "samples")
    found: List[Path] = []
    for root in roots:
        if root.is_dir():
            found.extend(sorted(p for p in root.glob("*.csv") if p.is_file()))
    seen, out = set(), []
    for p in found:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            out.append(p)
    return out


def sample_rows_by_class(path: os.PathLike | str, det: Detector,
                         *, per_class: int = 1) -> Dict[str, Dict[str, Any]]:
    """One labelled row per class from a sample CSV, keyed by its true label.

    Strictly better than :data:`SHAPE_PRESETS` when a sample file exists, because these
    are real rows with all 77 features populated: the model sees exactly what it was
    trained to see, so a correct verdict means something. The presets exist for the case
    where no sample has been generated yet.
    """
    frame = read_flows(path)
    if sch.LABEL_COLUMN not in [sch.normalise_column(str(c)) for c in frame.columns]:
        return {}
    frame.columns = [sch.normalise_column(str(c)) for c in frame.columns]
    labels = frame[sch.LABEL_COLUMN].map(lambda v: sch.canonical_label(str(v)))
    out: Dict[str, Dict[str, Any]] = {}
    for name in det.class_names:
        hit = frame.loc[labels == name]
        if len(hit) == 0:
            continue
        row = hit.iloc[0].drop(labels=[sch.LABEL_COLUMN]).to_dict()
        out[name] = {k: v for k, v in row.items()}
        if per_class > 1:                      # room for a "next example" button later
            out[name]["__alternatives__"] = len(hit)
    return out


# ---------------------------------------------------------------------------
# batch scoring
# ---------------------------------------------------------------------------

def score_frame(det: Detector, frame: pd.DataFrame, *,
                threshold: float = 0.5, min_confidence: float = 0.5,
                chunk_rows: int = 100_000,
                source: str = "the uploaded file") -> Tuple[PredictionBatch, PreparedInput]:
    """Prepare and score an upload, returning both halves.

    The :class:`PreparedInput` is returned alongside the batch rather than discarded
    because it is where the repair story lives - which columns were missing, which cells
    were junk - and the app has to show that next to the verdicts. A dashboard that
    reports 800 confident predictions without mentioning that 17 features were absent is
    not reporting, it is reassuring.
    """
    apply_thresholds(det, None, threshold=threshold, min_confidence=min_confidence)
    prep = det.prepare(frame)
    batch = det.predict(None, prepared=prep, chunk_rows=chunk_rows, source=source)
    apply_thresholds(det, batch, threshold=threshold, min_confidence=min_confidence)
    return batch, prep


def apply_thresholds(det: Detector, batch: Optional[PredictionBatch], *,
                     threshold: float, min_confidence: float) -> None:
    """Push the sidebar's sliders onto the detector and an existing batch.

    Re-scoring is unnecessary: both knobs are read-offs from probabilities that are
    already computed, so moving a slider must not cost another pass over two million
    rows. Setting them in both places keeps ``batch.summary()`` and any later
    ``det.predict_one`` consistent, which is the bug this function exists to prevent.
    """
    det.attack_threshold = float(threshold)
    det.min_confidence = float(min_confidence)
    if batch is not None:
        batch.attack_threshold = float(threshold)
        batch.min_confidence = float(min_confidence)


def batch_overview(batch: PredictionBatch) -> Dict[str, Any]:
    """The numbers for the dashboard's metric row."""
    s = batch.summary()
    s["attack_share_pct"] = 100.0 * s["attack_share"]
    s["low_confidence_share_pct"] = 100.0 * s["low_confidence_rows"] / max(len(batch), 1)
    s["critical_flows"] = int(sum(
        int(count) for name, count, severity in batch.triage() if severity >= 5))
    s["throughput"] = s.get("rows_per_second")
    return s


def reconciliation_note(batch: PredictionBatch) -> str:
    """Explain the gap between the threshold count and the label count, or say there is none.

    ``P(attack) >= threshold`` and ``argmax != BENIGN`` are different questions and give
    different answers - 25 rows apart on a 2,801-row file in testing. Two different attack
    totals on one screen with no explanation is how a user decides the whole dashboard is
    broken, so the screen always carries the sentence that reconciles them.
    """
    by_threshold = int(batch.is_attack.sum())
    by_label = int((batch.predicted != batch.benign_index).sum())
    gap = by_threshold - by_label
    if gap == 0:
        return (f"Both ways of counting agree here: {by_threshold:,} flow(s). The headline "
                f"figure uses P(attack) >= {batch.attack_threshold:.0%}; the class table "
                "counts the winning label.")
    if gap > 0:
        return (
            f"The headline counts {by_threshold:,} attack(s) but the class table shows "
            f"{by_label:,}. The {gap:,} extra flow(s) reach "
            f"{batch.attack_threshold:.0%} combined attack probability without any single "
            "attack class leading, so they are flagged but not labelled. That pattern is "
            "what a genuinely novel attack looks like to a 13-class model, which is why "
            "the headline uses the larger number."
        )
    return (
        f"The class table shows {by_label:,} attack label(s) but only {by_threshold:,} "
        f"reach the {batch.attack_threshold:.0%} threshold. The other {abs(gap):,} won "
        "their class narrowly while most of the probability stayed on BENIGN. Lower the "
        "threshold to include them in the headline, or leave it to suppress them."
    )


def class_distribution(batch: PredictionBatch) -> pd.DataFrame:
    """Predicted-class counts with severity and colour, present classes only.

    Absent classes are dropped rather than plotted at zero: on a normal capture nine of
    the thirteen bars would be zero-height and the four that matter would be squeezed
    into a third of the axis.
    """
    counts = batch.counts()
    rows = []
    for i, (name, count) in enumerate(counts.items()):
        if not count:
            continue
        severity = 0 if name == sch.BENIGN_LABEL else profile_for(name).severity
        mask = batch.predicted == batch.class_names.index(name)
        rows.append({
            "class": name,
            "flows": int(count),
            "share": float(count) / max(len(batch), 1),
            "severity": severity,
            "severity_label": "benign" if name == sch.BENIGN_LABEL
                              else severity_label(severity),
            "mean_confidence": float(batch.confidence[mask].mean()),
            "colour": SEVERITY_COLOURS.get(severity, "#4f5a66"),
        })
    frame = pd.DataFrame(rows)
    if len(frame):
        frame = frame.sort_values(["severity", "flows"], ascending=[False, False],
                                  ignore_index=True)
    return frame


def triage_table(batch: PredictionBatch) -> pd.DataFrame:
    """The work queue: attack classes present, most urgent first, with the action."""
    rows = []
    for name, count, severity in batch.triage():
        profile = profile_for(name)
        mask = batch.predicted == batch.class_names.index(name)
        rows.append({
            "priority": len(rows) + 1,
            "class": name,
            "flows": int(count),
            "severity": severity,
            "severity_label": severity_label(severity),
            "mean_confidence": float(batch.confidence[mask].mean()) if mask.any() else 0.0,
            "needs_review": int((batch.confidence[mask] < batch.min_confidence).sum()),
            "what_to_do": profile.action,
        })
    return pd.DataFrame(rows)


def confidence_histogram(batch: PredictionBatch, *, bins: int = 20) -> pd.DataFrame:
    """Confidence distribution split by benign/attack, for the calibration panel.

    Split rather than pooled because the shapes differ and the difference is the point:
    a model that is confident about benign traffic and hesitant about attacks looks fine
    on a pooled histogram dominated by the benign majority - 80.3% of CICIDS2017 as a
    whole, and the bulk of any day-file someone uploads here.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2.0
    attack = batch.is_attack
    out = []
    for label, mask in (("flagged as attack", attack), ("left as benign", ~attack)):
        counts, _ = np.histogram(batch.confidence[mask], bins=edges)
        for centre, count in zip(centres, counts):
            out.append({"confidence": float(centre), "group": label,
                        "flows": int(count)})
    return pd.DataFrame(out)


def verdict_table(batch: PredictionBatch, *, probabilities: bool = False,
                  top_k: int = 3, only_attacks: bool = False,
                  only_review: bool = False,
                  limit: Optional[int] = None) -> pd.DataFrame:
    """Filtered verdicts for the on-screen table.

    Filtering happens here and never in the download: what the analyst is looking at is a
    view, but the CSV is a record, and a record that quietly omitted the benign rows would
    make ``len(csv) == len(input)`` false - the one invariant this whole pipeline promises.
    """
    frame = batch.frame(probabilities=probabilities, top_k=top_k)
    if only_attacks:
        frame = frame.loc[frame["is_attack"]]
    if only_review:
        frame = frame.loc[frame["status"] == "review"]
    frame = frame.sort_values(["severity", "attack_probability"],
                             ascending=[False, False])
    if limit is not None:
        frame = frame.head(int(limit))
    return frame.reset_index(drop=True)


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """UTF-8 CSV bytes for ``st.download_button``, index dropped.

    ``index=False`` matters more than it looks: with the index written, a round trip
    through this app adds an ``Unnamed: 0`` column, and the second pass would see a
    feature-shaped column that is really a row number.
    """
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# one row, explained
# ---------------------------------------------------------------------------

def explanation_table(explanation: Any, *, top: int = 10) -> pd.DataFrame:
    """Signed per-feature contributions, largest magnitude first.

    ``value`` is the feature as captured whenever the raw value could be recovered, and
    the standardised value when it could not - which is why ``scale`` exists. One column
    is right for the screen (an analyst wants a number, not two), but silently swapping
    96,000,000 for 1.83 under the same heading would let someone try to match a z-score
    against their capture and conclude the tool is wrong.
    """
    if explanation is None:
        return pd.DataFrame(columns=["feature", "value", "scale", "contribution",
                                     "effect", "means"])
    rows = []
    for c in explanation.top(top):
        raw = c.raw_value is not None
        rows.append({
            "feature": c.name,
            "value": float(c.raw_value if raw else c.value),
            "scale": "as captured" if raw else "standardised",
            "contribution": float(c.contribution),
            "effect": f"{c.direction} {explanation.predicted_class}",
            "means": sch.describe_feature(c.name),
            "colour": "#a8202a" if c.contribution > 0 else "#3a6ea8",
        })
    return pd.DataFrame(rows)


def probability_table(prediction: Any, *, top: Optional[int] = None) -> pd.DataFrame:
    """Every class's probability for one flow, highest first."""
    pairs = prediction.top_classes(top or len(getattr(prediction, "_names", []) or [13]))
    rows = [{"class": name, "probability": float(p),
             "severity": 0 if name == sch.BENIGN_LABEL else profile_for(name).severity}
            for name, p in pairs]
    frame = pd.DataFrame(rows)
    frame["colour"] = [SEVERITY_COLOURS.get(int(s), "#4f5a66") for s in frame["severity"]]
    return frame


# ---------------------------------------------------------------------------
# evaluation mode
# ---------------------------------------------------------------------------

def evaluation_overview(report: Any) -> Dict[str, Any]:
    """Headline metrics from an :class:`~shieldnet.evaluate.EvaluationReport`."""
    return {
        "model": report.model,
        "split": report.split,
        "rows": report.n_rows,
        "macro_f1": report.macro_f1,
        "accuracy": report.accuracy,
        "balanced_accuracy": report.balanced_accuracy,
        "macro_recall": report.macro_recall,
        "macro_precision": report.macro_precision,
        "weighted_f1": report.weighted_f1,
        "log_loss": report.log_loss,
        "mcc": report.mcc,
        "kappa": report.kappa,
        "top3_accuracy": report.top3_accuracy,
        "calibration_error": report.calibration_error,
        "macro_roc_auc": report.macro_roc_auc,
        "classes_absent": list(report.classes_absent),
        "classes_never_predicted": list(report.classes_never_predicted),
        "worst": [(c.name, c.recall, c.support) for c in report.worst_classes(3)],
        "notes": list(report.notes),
    }


def per_class_table(report: Any) -> pd.DataFrame:
    """Per-class metrics with the class it is most often confused with."""
    frame = report.per_class_frame()
    if "name" in frame.columns:
        frame = frame.rename(columns={"name": "class"})
    return frame


def confusion_frame(report: Any, *, normalise: bool = True) -> pd.DataFrame:
    """Confusion matrix as a labelled frame, row-normalised by default.

    Row normalisation is not cosmetic. Raw counts put six figures in the BENIGN row and
    single digits in Rare Attacks, so on one colour scale every attack row renders the
    same shade and the plot shows nothing at all. Normalised, the diagonal *is* per-class
    recall and the off-diagonal is where each class leaks.
    """
    cm = np.asarray(report.confusion, dtype=np.float64)
    if normalise:
        totals = cm.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            cm = np.where(totals > 0, cm / np.maximum(totals, 1e-12), 0.0)
    return pd.DataFrame(cm, index=list(report.class_names),
                        columns=list(report.class_names))


def sweep_table(report: Any) -> pd.DataFrame:
    """Attack-threshold operating points, empty frame when the report has none.

    Read from the arrays rather than from ``sweep.to_dict()``: that method returns the
    three recommended thresholds and a point count, which is the right payload for a JSON
    summary and useless for drawing a curve.

    Rows come back in the report's own order, thresholds running high to low, which is not
    re-sorted here. A line chart draws the same line either way, and
    :meth:`ThresholdSweep.at` binary-searches the descending array - so a tidy-looking
    ``sort_values("threshold")`` in this function would be free to add and would quietly
    break threshold lookup for anyone who passed the frame back.
    """
    sweep = getattr(report, "sweep", None)
    if sweep is None or np.asarray(sweep.thresholds).size == 0:
        return pd.DataFrame(columns=["threshold", "detection_rate",
                                     "false_alarm_rate", "precision", "f1"])
    frame = pd.DataFrame({
        "threshold": np.asarray(sweep.thresholds, dtype=float),
        # tpr/fpr are the literature's names; on screen they are the detection rate and
        # the false-alarm rate, which is what the reader is deciding between.
        "detection_rate": np.asarray(sweep.tpr, dtype=float),
        "false_alarm_rate": np.asarray(sweep.fpr, dtype=float),
        "precision": np.asarray(sweep.precision, dtype=float),
        "f1": np.asarray(sweep.f1, dtype=float),
    })
    return frame


def sweep_recommendations(report: Any) -> List[Tuple[str, Optional[float], str]]:
    """``(name, threshold, why)`` for the three ways of choosing an operating point."""
    sweep = getattr(report, "sweep", None)
    if sweep is None:
        return []
    out: List[Tuple[str, Optional[float], str]] = [
        ("best F1", sweep.best_f1_threshold,
         f"maximises the attack-vs-benign F1 ({sweep.best_f1:.4f}); the default choice "
         "when detection and false alarms cost about the same"),
        ("Youden J", sweep.youden_threshold,
         "maximises detection minus false alarms, ignoring how common attacks are - the "
         "balanced choice, and the one to prefer if your traffic mix differs from the "
         "dataset's"),
    ]
    if sweep.budget_threshold is not None:
        out.append((
            "within the false-alarm budget", sweep.budget_threshold,
            f"the most detection ({sweep.budget_tpr:.2%}) achievable while keeping the "
            f"false-alarm rate at {sweep.budget_fpr:.2%} - the choice a team with a fixed "
            "alert capacity actually has to make"))
    return out


def binary_table(report: Any) -> pd.DataFrame:
    """The attack-vs-benign collapse as a two-column frame.

    This is the operational question - does anything get through, and how often does the
    sensor cry wolf - and it is the one comparable with the binary IDS literature, which
    is most of it.
    """
    binary = getattr(report, "binary", None)
    if binary is None:
        return pd.DataFrame(columns=["measure", "value"])
    rows = [
        ("detection rate (recall)", binary.recall),
        ("false alarm rate", binary.false_alarm_rate),
        ("miss rate", binary.miss_rate),
        ("precision", binary.precision),
        ("F1", binary.f1),
        ("ROC AUC", binary.roc_auc),
        ("average precision", binary.average_precision),
        ("true positives", binary.tp),
        ("false positives", binary.fp),
        ("false negatives", binary.fn),
        ("true negatives", binary.tn),
    ]
    return pd.DataFrame(rows, columns=["measure", "value"])


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

def format_metric(value: Any, kind: str = "ratio") -> str:
    """Format a metric for display, or ``"n/a"`` when it was never measured.

    ``None`` renders as ``n/a`` rather than ``0.00`` on purpose: a false alarm rate of
    zero is a remarkable claim and an unmeasured one is not a claim at all.
    """
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if kind == "pct":
        return f"{float(value):.2%}"
    if kind == "int":
        return f"{int(value):,}"
    if kind == "loss":
        return f"{float(value):.4f}"
    return f"{float(value):.4f}"


def humanise(n: Any) -> str:
    """Thousands-separated integer, tolerant of numpy scalars and ``None``."""
    if n is None:
        return "n/a"
    return f"{int(n):,}"
