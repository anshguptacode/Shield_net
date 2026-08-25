"""ShieldNet - Streamlit front end.

    shieldnet serve                       # preferred: exports the artifact path for you
    streamlit run app/streamlit_app.py    # equivalent, if the artifact is in ./artifacts

This file is deliberately thin. Every number it renders is computed in
``app/shieldnet_ui.py``, which has no Streamlit dependency and is exercised by
``tests/check_app.py`` against a real trained bundle. What is left here is widgets,
layout and the session-state bookkeeping that keeps a two-million-row capture from being
re-scored every time a slider moves.

Two rules the layout obeys throughout:

* The provenance of the model is in the sidebar, above the metrics, not below them. A
  score from a synthetic smoke run must be impossible to read as a CICIDS2017 result.
* Whatever repairs the upload needed are shown next to the verdicts, never instead of
  them and never hidden behind an expander that nobody opens.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

# `streamlit run` puts this file's directory on sys.path, so `import shieldnet_ui` works.
# The src/ entry is for the case where the package was never pip-installed - a clone plus
# `streamlit run` should just work, because that is what most people will actually do.
_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE, _HERE.parent / "src"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

import numpy as np
import pandas as pd
import streamlit as st

import shieldnet_ui as ui
from shieldnet import schema as sch
from shieldnet.inference import read_flows
from shieldnet.narrate import profile_for
from shieldnet.persist import BundleError

st.set_page_config(page_title=ui.APP_TITLE, page_icon="🛡", layout="wide",
                   initial_sidebar_state="expanded")


# ---------------------------------------------------------------------------
# charts: plotly when available, Streamlit's built-ins when not
# ---------------------------------------------------------------------------

def _plotly():
    """Return plotly.express, or None. The app must not require it."""
    try:
        import plotly.express as px
        return px
    except Exception:                                       # noqa: BLE001
        return None


def bar_chart(frame: pd.DataFrame, *, x: str, y: str, colour: Optional[str] = None,
              title: str = "", horizontal: bool = False) -> None:
    """One bar chart, drawn with whatever is installed.

    The fallback is not a downgrade worth apologising for: ``st.bar_chart`` renders the
    same numbers. Plotly is used when present because per-bar colour carries the severity
    ordering, and severity is the only reason to look at this chart twice.
    """
    if frame.empty:
        st.caption("nothing to plot")
        return
    px = _plotly()
    if px is None:
        st.bar_chart(frame.set_index(x)[y], height=340)
        return
    kwargs: Dict[str, Any] = dict(title=title or None)
    if colour and colour in frame.columns:
        kwargs["color"] = colour
        kwargs["color_discrete_map"] = {c: c for c in frame[colour].unique()}
    fig = (px.bar(frame, y=x, x=y, orientation="h", **kwargs) if horizontal
           else px.bar(frame, x=x, y=y, **kwargs))
    fig.update_layout(showlegend=False, height=340,
                      margin=dict(l=8, r=8, t=36 if title else 8, b=8))
    st.plotly_chart(fig, use_container_width=True)


def grouped_bars(frame: pd.DataFrame, *, x: str, y: str, group: str,
                 title: str = "") -> None:
    px = _plotly()
    if px is None:
        wide = frame.pivot_table(index=x, columns=group, values=y, aggfunc="sum")
        st.bar_chart(wide, height=340)
        return
    fig = px.bar(frame, x=x, y=y, color=group, barmode="group", title=title or None)
    fig.update_layout(height=340, margin=dict(l=8, r=8, t=36 if title else 8, b=8),
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)


def line_chart(frame: pd.DataFrame, *, x: str, ys: list, title: str = "") -> None:
    if frame.empty:
        st.caption("nothing to plot")
        return
    px = _plotly()
    if px is None:
        st.line_chart(frame.set_index(x)[ys], height=340)
        return
    long = frame.melt(id_vars=[x], value_vars=ys, var_name="measure", value_name="value")
    fig = px.line(long, x=x, y="value", color="measure", title=title or None)
    fig.update_layout(height=340, margin=dict(l=8, r=8, t=36 if title else 8, b=8),
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)


def heatmap(frame: pd.DataFrame, *, title: str = "") -> None:
    px = _plotly()
    if px is None:
        # Background gradient needs matplotlib; a plain frame is still readable and is
        # better than an exception on a machine that has neither plotly nor matplotlib.
        try:
            st.dataframe(frame.style.background_gradient(cmap="Blues", vmin=0, vmax=1)
                         .format("{:.2f}"), use_container_width=True)
        except Exception:                                   # noqa: BLE001
            st.dataframe(frame.round(3), use_container_width=True)
        return
    fig = px.imshow(frame, color_continuous_scale="Blues", zmin=0, zmax=1,
                    aspect="auto", title=title or None,
                    labels=dict(x="predicted", y="actual", color="fraction"))
    fig.update_layout(height=520, margin=dict(l=8, r=8, t=40, b=8))
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="loading the trained model...")
def get_detector(artifacts: str):
    """Cached across reruns. The Detector is read-only apart from its two thresholds."""
    return ui.load_detector(artifacts or None)


def artifact_choice() -> str:
    default = str(ui.find_artifacts())
    with st.sidebar:
        st.markdown(f"### {ui.APP_TITLE}")
        st.caption(ui.APP_TAGLINE)
        with st.expander("artifact directory", expanded=False):
            chosen = st.text_input("path", value=default,
                                   help="where bundle.joblib and manifest.json live")
            if st.button("reload model", use_container_width=True):
                get_detector.clear()
                st.session_state.pop("batch", None)
    return chosen or default


ARTIFACTS = artifact_choice()

try:
    det = get_detector(ARTIFACTS)
except (BundleError, FileNotFoundError, OSError) as exc:
    st.error(f"No usable model in `{ARTIFACTS}`.")
    st.code(str(exc), language="text")
    st.markdown(
        "Train one first - about two minutes on generated data, which is enough to see "
        "the whole app working:\n"
        "```bash\n"
        "shieldnet train --synthetic 20000 --no-tune\n"
        "```\n"
        "Then, for real numbers:\n"
        "```bash\n"
        "shieldnet download && shieldnet train --config config/default.yaml\n"
        "```"
    )
    st.stop()

card = ui.model_card(det)


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.divider()
    if card["synthetic"]:
        st.error("Trained on SYNTHETIC data - these scores describe the generator, "
                 "not CICIDS2017.", icon="⚠")
    st.markdown(f"**{card['model']}**")
    a, b = st.columns(2)
    a.metric("macro F1", ui.format_metric(card["macro_f1"]),
             help=f"on the held-out {card['split']} split, all "
                  f"{card['n_classes']} classes weighted equally")
    b.metric("accuracy", ui.format_metric(card["accuracy"]),
             help="shown for completeness only. Predicting BENIGN for every row scores "
                  "0.803 on CICIDS2017 while detecting nothing, so accuracy is not the "
                  "number to judge this on.")
    c, d = st.columns(2)
    c.metric("attack recall", ui.format_metric(card["attack_recall"], "pct"),
             help="of all attack flows, the share caught - attack vs benign, ignoring "
                  "which attack")
    d.metric("false alarms", ui.format_metric(card["false_alarm_rate"], "pct"),
             help="of all benign flows, the share wrongly flagged. This is what decides "
                  "whether a sensor is deployable.")

    st.divider()
    st.markdown("#### Decision thresholds")
    threshold = st.slider(
        "flag as attack when P(attack) >=", 0.05, 0.95, 0.50, 0.05,
        help="P(attack) is 1 - P(BENIGN), so it accumulates evidence spread across "
             "several attack classes. Lower catches more and cries wolf more.")
    min_conf = st.slider(
        "mark for review when confidence <", 0.05, 0.95, 0.50, 0.05,
        help="A low-confidence BENIGN is not a clean flow. These rows are still "
             "predicted; they are just labelled as worth a human look.")
    st.caption("Both are read off probabilities that are already computed, so moving "
               "them re-labels the results instantly without re-scoring.")

    st.divider()
    st.markdown("#### Provenance")
    for line in ui.provenance_lines(det):
        st.markdown(f"- {line}")

ui.apply_thresholds(det, st.session_state.get("batch"),
                    threshold=threshold, min_confidence=min_conf)


# ---------------------------------------------------------------------------
# header
# ---------------------------------------------------------------------------

st.title(f"🛡 {ui.APP_TITLE}")
st.caption(ui.APP_TAGLINE + f" · {card['n_features']} features · "
           f"{card['n_classes']} classes · {card['model']}")

tab_batch, tab_single, tab_eval, tab_model = st.tabs(
    ["Score a capture", "Single flow", "Evaluate labelled data", "Model & classes"])


# ---------------------------------------------------------------------------
# tab 1: score a capture
# ---------------------------------------------------------------------------

def input_health(prep) -> None:
    """Show what had to be repaired before anything was scored."""
    warnings, notes = prep.warnings(), prep.notes()
    head = (f"{prep.rows:,} row(s) read · matrix {prep.X.shape[0]:,} x {prep.X.shape[1]}"
            f" · {prep.repaired_cells:,} cell(s) repaired")
    if not warnings:
        st.success(head + " · every expected feature present", icon="✓")
    else:
        with st.container(border=True):
            st.markdown(f"**Input health** — {head}")
            for w in warnings:
                st.warning(w, icon="!")
    for n in notes:
        st.caption(n)
    st.caption("No row is ever dropped: row *i* of the results is the verdict on row *i* "
               "of your file. Unreadable cells are repaired from the training median and "
               "reported above, not excised.")


with tab_batch:
    st.subheader("Score a capture")
    left, right = st.columns([3, 2])
    with left:
        upload = st.file_uploader(
            "CICFlowMeter CSV (or .csv.gz / .parquet)",
            type=["csv", "gz", "parquet", "pq"],
            help="79-column CICFlowMeter output. A Label column is optional here and "
                 "required in the Evaluate tab.")
    with right:
        samples = ui.sample_files()
        names = ["-"] + [p.name for p in samples]
        picked = st.selectbox(
            "or score a bundled sample", names, index=1 if len(names) > 1 else 0,
            help="written by `shieldnet prepare --sample`; a stratified slice of the "
                 "dataset with its labels kept")
        max_rows = st.number_input("rows to read (0 = all)", 0, 5_000_000, 0, 1_000)

    source_label, payload = None, None
    if upload is not None:
        source_label, payload = upload.name, upload
    elif picked and picked != "-":
        hit = [p for p in samples if p.name == picked]
        if hit:
            source_label, payload = hit[0].name, hit[0]

    if st.button("Score", type="primary", disabled=payload is None):
        try:
            with st.spinner(f"reading {source_label}..."):
                frame = read_flows(payload, nrows=int(max_rows) or None)
            with st.spinner(f"scoring {len(frame):,} flow(s)..."):
                batch, prep = ui.score_frame(
                    det, frame, threshold=threshold, min_confidence=min_conf,
                    source=source_label or "the uploaded file")
            st.session_state["batch"] = batch
            st.session_state["prep"] = prep
            st.session_state["source"] = source_label
            # A per-row explanation belongs to the capture it came from. Session state
            # outlives the upload, so without this a scored second file shows the first
            # file's row 17 - same shape, same column names, wrong flow, and nothing on
            # screen to say so.
            st.session_state.pop("row_pred", None)
        except (ValueError, KeyError) as exc:
            st.error(str(exc))
            st.session_state.pop("batch", None)
            st.session_state.pop("row_pred", None)

    batch = st.session_state.get("batch")
    prep = st.session_state.get("prep")

    if batch is None:
        st.info("Upload a capture or pick a sample, then press **Score**.")
    else:
        ui.apply_thresholds(det, batch, threshold=threshold, min_confidence=min_conf)
        overview = ui.batch_overview(batch)
        st.divider()
        input_health(prep)

        m = st.columns(6)
        m[0].metric("flows scored", ui.humanise(overview["rows"]))
        m[1].metric("flagged as attack", ui.humanise(overview["attack_flows"]),
                    delta=f"{overview['attack_share_pct']:.1f}% of the file",
                    delta_color="off")
        m[2].metric("attack classes seen",
                    ui.humanise(max(overview["distinct_classes"] - 1, 0)))
        m[3].metric("critical (severity 5)", ui.humanise(overview["critical_flows"]))
        m[4].metric("needs review", ui.humanise(overview["low_confidence_rows"]),
                    delta=f"{overview['low_confidence_share_pct']:.1f}%",
                    delta_color="off")
        m[5].metric("mean confidence", ui.format_metric(overview["mean_confidence"], "pct"))

        st.caption(ui.reconciliation_note(batch))
        st.info(batch.narrative())

        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.markdown("**What was found**")
            dist = ui.class_distribution(batch)
            bar_chart(dist, x="class", y="flows", colour="colour", horizontal=True)
            st.caption("Ordered by severity then volume - the order to work them in, not "
                       "the model's class order. Classes with no flows are omitted.")
        with chart_right:
            st.markdown("**How sure the model was**")
            hist = ui.confidence_histogram(batch)
            grouped_bars(hist, x="confidence", y="flows", group="group")
            st.caption("Split by verdict because the shapes differ, and the difference "
                       "is the point: confident about benign traffic and hesitant about "
                       "attacks looks fine on a pooled histogram.")

        st.markdown("#### Triage queue")
        triage = ui.triage_table(batch)
        if triage.empty:
            st.success("No attack classes predicted in this capture.", icon="✓")
        else:
            st.dataframe(triage, use_container_width=True, hide_index=True)

        st.markdown("#### Verdicts")
        f1, f2, f3 = st.columns([1, 1, 2])
        only_attacks = f1.checkbox("attacks only", value=False)
        only_review = f2.checkbox("needs review only", value=False)
        shown = f3.number_input("rows to display", 10, 5_000, 200, 10)
        view = ui.verdict_table(batch, probabilities=False, top_k=3,
                                only_attacks=only_attacks, only_review=only_review,
                                limit=int(shown))
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.caption(f"showing {len(view):,} of {len(batch):,} row(s), sorted by severity")

        d1, d2 = st.columns(2)
        # Strip the uploaded file's own extension, or the browser saves
        # `shieldnet_verdicts_capture.csv.csv`.
        stem = Path(str(st.session_state.get("source") or "capture")).stem
        d1.download_button(
            "Download all verdicts (CSV)",
            data=ui.csv_bytes(batch.frame(probabilities=False, top_k=3)),
            file_name=f"shieldnet_verdicts_{stem}.csv",
            mime="text/csv", use_container_width=True)
        d2.download_button(
            "Download with all class probabilities (CSV)",
            data=ui.csv_bytes(batch.frame(probabilities=True, top_k=3)),
            file_name=f"shieldnet_probabilities_{stem}.csv",
            mime="text/csv", use_container_width=True,
            help="one column per class. The download is always the whole file - the "
                 "filters above change what you are looking at, not what you get.")

        st.divider()
        st.markdown("#### Why was this flow flagged?")
        pick = st.number_input("row number", 0, max(len(batch) - 1, 0), 0, 1)
        if st.button("Explain this row"):
            with st.spinner("attributing..."):
                pred = det.inspect(prep, batch, int(pick), narrate=True)
            st.session_state["row_pred"] = pred
        pred = st.session_state.get("row_pred")
        if pred is not None and pred.row <= len(batch) - 1:
            sev_colour = ui.SEVERITY_COLOURS.get(pred.severity, "#4f5a66")
            st.markdown(
                f"<div style='padding:0.7rem 1rem;border-left:5px solid {sev_colour};"
                f"background:#f6f7f9'><b>Row {pred.row}: {pred.predicted_class}</b> "
                f"at {pred.confidence:.1%} confidence "
                f"(severity {pred.severity}, {pred.status})</div>",
                unsafe_allow_html=True)
            if pred.true_class:
                (st.success if pred.correct else st.error)(
                    f"ground truth: {pred.true_class} - "
                    + ("correct" if pred.correct else "wrong"))
            e1, e2 = st.columns([2, 3])
            with e1:
                st.markdown("**Class probabilities**")
                probs = ui.probability_table(pred, top=8)
                bar_chart(probs, x="class", y="probability", colour="colour",
                          horizontal=True)
            with e2:
                st.markdown("**Feature contributions**")
                table = ui.explanation_table(pred.explanation, top=10)
                st.dataframe(table.drop(columns=["colour"], errors="ignore"),
                             use_container_width=True, hide_index=True)
            if pred.narrative:
                st.info(pred.narrative)
            if pred.explanation is not None:
                st.caption(f"attribution method: {pred.explanation.method}"
                           + (f" · contributions sum to within "
                              f"{pred.explanation.additivity_error:.1e} of the prediction"
                              if pred.explanation.additivity_error is not None else ""))
            profile = profile_for(pred.predicted_class)
            with st.expander(f"about {pred.predicted_class}"):
                st.markdown(f"**What it is.** {profile.summary}")
                st.markdown(f"**How it shows up in flow features.** {profile.on_the_wire}")
                st.markdown(f"**What to do.** {profile.action}")
                if profile.confusable_with:
                    st.markdown("**Genuinely hard to separate from:** "
                                + ", ".join(profile.confusable_with))


# ---------------------------------------------------------------------------
# tab 2: single flow
# ---------------------------------------------------------------------------

with tab_single:
    st.subheader("Score one flow by hand")
    st.caption(
        f"The form offers the {len(det.manual_fields())} features a human can reason "
        f"about without a packet capture. The other "
        f"{max(len(det.template_row()) - len(det.manual_fields()), 0)} are filled from "
        "the training median, so this is a what-if against an otherwise typical flow "
        "rather than a real capture - useful for probing the decision boundary, not for "
        "claiming a detection.")

    real_rows: Dict[str, Dict[str, Any]] = {}
    samples = ui.sample_files()
    if samples:
        try:
            real_rows = ui.sample_rows_by_class(samples[0], det)
        except (ValueError, KeyError, OSError):
            real_rows = {}

    preset_names = ["blank (training medians)"] + list(ui.SHAPE_PRESETS)
    preset_names += [f"real {name} flow from {samples[0].name}" for name in real_rows]
    preset = st.selectbox("start from", preset_names, index=0)

    seed_values: Dict[str, Any] = {}
    if preset in ui.SHAPE_PRESETS:
        seed_values = dict(ui.SHAPE_PRESETS[preset])
        st.warning(
            "This is a hand-built *shape*, not a captured flow. Twelve features are set "
            "to values characteristic of that attack and the other ~65 stay at their "
            "training median, so the model sees a chimera. If it disagrees with the name, "
            "that is information: the shape needs features this form cannot offer.",
            icon="!")
    elif preset.startswith("real "):
        label = preset[len("real "):].split(" flow from ")[0]
        seed_values = dict(real_rows.get(label, {}))
        st.success(f"A real {label} row with all {len(seed_values)} features populated - "
                   "the model sees exactly what it was trained on.", icon="✓")

    with st.form("single_flow"):
        entered: Dict[str, float] = {}
        for title, group in ui.manual_form_groups(det):
            st.markdown(f"**{title}**")
            cols = st.columns(len(group))
            for col, spec in zip(cols, group):
                default = float(seed_values.get(spec["name"], spec["default"]))
                entered[spec["name"]] = col.number_input(
                    spec["name"], min_value=float(spec["min"]),
                    max_value=float(spec["max"]), value=default,
                    help=spec["help"] + ("" if spec["used_by_model"]
                                         else "  (not among this model's selected "
                                              "features - changing it will not move the "
                                              "verdict)"))
        submitted = st.form_submit_button("Classify this flow", type="primary")

    if submitted:
        # Start from every feature the seed supplied - a real sample row carries all 77 -
        # then overlay whatever the form changed. Sending only the twelve form fields
        # would silently throw away the 65 real values the preset just loaded.
        values = {**seed_values, **entered}
        with st.spinner("scoring..."):
            pred = det.predict_one(values, explain=True, narrate=True)
        sev_colour = ui.SEVERITY_COLOURS.get(pred.severity, "#4f5a66")
        st.markdown(
            f"<div style='padding:0.9rem 1.1rem;border-left:6px solid {sev_colour};"
            f"background:#f6f7f9'><h3 style='margin:0'>{pred.predicted_class}</h3>"
            f"<span>{pred.confidence:.1%} confidence · severity {pred.severity} · "
            f"P(attack) {pred.attack_probability:.1%} · "
            f"{'flagged' if pred.is_attack else 'not flagged'} at the current threshold"
            f"</span></div>", unsafe_allow_html=True)
        if pred.status == "review":
            st.warning(f"Below the {min_conf:.0%} confidence line - the model is not sure. "
                       f"Its second choice is {pred.runner_up} at "
                       f"{pred.runner_up_confidence:.1%}.", icon="!")
        s1, s2 = st.columns([2, 3])
        with s1:
            st.markdown("**Class probabilities**")
            bar_chart(ui.probability_table(pred, top=8), x="class", y="probability",
                      colour="colour", horizontal=True)
        with s2:
            st.markdown("**Feature contributions**")
            st.dataframe(ui.explanation_table(pred.explanation, top=10)
                         .drop(columns=["colour"], errors="ignore"),
                         use_container_width=True, hide_index=True)
        if pred.narrative:
            st.info(pred.narrative)
        profile = profile_for(pred.predicted_class)
        with st.expander(f"about {pred.predicted_class}", expanded=True):
            st.markdown(f"**What it is.** {profile.summary}")
            st.markdown(f"**How it shows up in flow features.** {profile.on_the_wire}")
            st.markdown(f"**What to do.** {profile.action}")


# ---------------------------------------------------------------------------
# tab 3: evaluate labelled data
# ---------------------------------------------------------------------------

with tab_eval:
    st.subheader("Evaluate against ground truth")
    st.caption("Needs a `Label` column. Raw CICIDS2017 spellings are fine - the en-dash "
               "in `Web Attack \\x96 XSS` and the rare classes are canonicalised the same "
               "way training did, so a file straight from the dataset lines up.")
    e_left, e_right = st.columns([3, 2])
    with e_left:
        eval_upload = st.file_uploader("labelled CSV", type=["csv", "gz", "parquet", "pq"],
                                       key="eval_upload")
    with e_right:
        eval_samples = ui.sample_files()
        eval_names = ["-"] + [p.name for p in eval_samples]
        eval_pick = st.selectbox("or a bundled sample", eval_names,
                                 index=1 if len(eval_names) > 1 else 0, key="eval_pick")
        budget = st.slider("false-alarm budget", 0.001, 0.10, 0.01, 0.001,
                           format="%.3f",
                           help="the sweep reports the most detection achievable while "
                                "keeping the false-alarm rate under this")

    eval_payload = eval_upload
    if eval_payload is None and eval_pick != "-":
        hit = [p for p in eval_samples if p.name == eval_pick]
        eval_payload = hit[0] if hit else None

    if st.button("Evaluate", type="primary", disabled=eval_payload is None,
                 key="eval_go"):
        try:
            with st.spinner("scoring and scoring the scores..."):
                frame = read_flows(eval_payload)
                report = det.evaluate(frame, fpr_budget=float(budget))
            st.session_state["report"] = report
        except (ValueError, KeyError, RuntimeError) as exc:
            st.error(str(exc))
            st.session_state.pop("report", None)

    report = st.session_state.get("report")
    if report is None:
        st.info("Pick a labelled file and press **Evaluate**.")
    else:
        o = ui.evaluation_overview(report)
        k = st.columns(6)
        k[0].metric("macro F1", ui.format_metric(o["macro_f1"]),
                    help="all classes weighted equally - the number this project is "
                         "judged on")
        k[1].metric("balanced accuracy", ui.format_metric(o["balanced_accuracy"]))
        k[2].metric("accuracy", ui.format_metric(o["accuracy"]),
                    help="inflated by the benign majority; here for comparability only")
        k[3].metric("macro recall", ui.format_metric(o["macro_recall"]))
        k[4].metric("MCC", ui.format_metric(o["mcc"]))
        k[5].metric("rows", ui.humanise(o["rows"]))

        if o["classes_absent"]:
            st.info("Not present in this file, so not measured: "
                    + ", ".join(o["classes_absent"]))
        if o["classes_never_predicted"]:
            st.warning("Never predicted on this file: "
                       + ", ".join(o["classes_never_predicted"])
                       + ". For these classes the model contributes nothing, and their "
                         "zero recall is dragging macro F1 down - which is the honest "
                         "thing for it to do.", icon="!")
        for note in o["notes"]:
            st.caption(note)
        if o["worst"]:
            st.markdown("**Weakest classes.** " + " · ".join(
                f"{name}: recall {recall:.1%} on {support:,} row(s)"
                for name, recall, support in o["worst"]))

        st.markdown("#### Per class")
        st.dataframe(ui.per_class_table(report), use_container_width=True,
                     hide_index=True)

        st.markdown("#### Confusion")
        norm = st.checkbox("row-normalise", value=True,
                           help="On: each row shows what fraction of that true class went "
                                "where, so the diagonal is per-class recall. Off: raw "
                                "counts, where the benign row's six figures flatten every "
                                "attack row to the same colour.")
        heatmap(ui.confusion_frame(report, normalise=norm))

        b_left, b_right = st.columns([2, 3])
        with b_left:
            st.markdown("#### Attack vs benign")
            st.caption("The operational question, and the one comparable with the binary "
                       "IDS literature.")
            st.dataframe(ui.binary_table(report), use_container_width=True,
                         hide_index=True)
        with b_right:
            st.markdown("#### Where to set the threshold")
            sweep = ui.sweep_table(report)
            if sweep.empty:
                st.caption("no sweep available for this file")
            else:
                line_chart(sweep, x="threshold",
                           ys=["detection_rate", "false_alarm_rate", "precision", "f1"])
                for name, value, why in ui.sweep_recommendations(report):
                    st.markdown(f"- **{name}: {value:.3f}** — {why}")

        st.download_button("Download the full report (JSON)",
                           data=report.to_json().encode("utf-8"),
                           file_name="shieldnet_evaluation.json",
                           mime="application/json")


# ---------------------------------------------------------------------------
# tab 4: model & classes
# ---------------------------------------------------------------------------

with tab_model:
    st.subheader("What this model is")
    g = st.columns(4)
    g[0].metric("features used", ui.humanise(card["n_features"]),
                help=f"selected from {len(sch.CANONICAL_FEATURES)} flow features")
    g[1].metric("classes", ui.humanise(card["n_classes"]))
    g[2].metric("macro ROC AUC", ui.format_metric(card["macro_roc_auc"]))
    g[3].metric("calibration error", ui.format_metric(card["calibration_error"]),
                help="mean gap between stated confidence and observed accuracy. Small "
                     "means the percentages on this page can be taken at face value.")

    st.markdown("#### Classes, and how well each is detected")
    st.caption("Recall is from the training run's held-out test split. It belongs next to "
               "the class name because a confident PortScan verdict and a confident Bot "
               "verdict are not equally trustworthy when their recalls differ by forty "
               "points.")
    classes = ui.class_reference_table(det)
    st.dataframe(classes, use_container_width=True, hide_index=True,
                 column_config={"test_recall": st.column_config.ProgressColumn(
                     "test recall", min_value=0.0, max_value=1.0, format="%.3f")}
                 if hasattr(st, "column_config") else None)

    st.markdown("#### Features, in selection order")
    st.caption(ui.selection_caption(det))
    st.dataframe(ui.feature_table(det), use_container_width=True, hide_index=True)

    with st.expander("everything the artifact records about itself"):
        st.json({"metrics": det.metrics, "metadata": det.bundle.metadata})

    st.caption(f"artifact: `{ARTIFACTS}` · {det.describe()}")
