"""Verify the Streamlit deliverable: the logic module, and the script itself.

Two halves, and the second is the unusual one.

``app/shieldnet_ui.py`` holds every number the app renders and imports no Streamlit, so it
is checked the ordinary way - call the functions against a real trained bundle and assert
on the frames that come back.

``app/streamlit_app.py`` is then *executed*, eight times, against the recording double in
``tests/_stubs/stub_streamlit.py``. Widgets return scripted answers instead of reading a
browser, so a whole user journey - pick a sample, score it, filter it, explain row 17,
classify a hand-entered flow, evaluate against labels - runs as a test. This catches the
class of bug that unit-testing the logic module cannot: a wrong keyword argument, a metric
read from a key nobody writes, a variable used before the branch that defines it. Those
live in the script, and in a normal project nothing ever runs the script.

The last pass points the app at an empty directory, because the first thing a new user
sees is that error and it is the one message that must be actionable.
"""
import io
import os
import runpy
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Derived from this file, never hardcoded: an absolute path here would make a copy of the
# project silently import the original, so the copy would pass while being untested.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests/_stubs"))
sys.path.insert(0, str(ROOT / "app"))
import stub_models; stub_models.register()
import stub_streamlit

from shieldnet import schema as sch
from shieldnet.config import Config
from shieldnet.inference import Detector
from shieldnet.logging_utils import configure_logging
from shieldnet.train import prepare_data, train

configure_logging("WARNING", force=True)

APP = ROOT / "app" / "streamlit_app.py"
WORK = Path("/tmp/shieldnet_app")
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)

# ---------------------------------------------------------------- 0. an artifact
print("=" * 78 + "\n=== 0. a real artifact and a bundled sample ===\n" + "=" * 78)

cfg = Config.load(
    seed=42, n_jobs=1,
    paths={"root": str(WORK)},
    data={"test_size": 0.2, "val_size": 0.1, "min_class_rows": 10},
    features={"n_features": 16, "methods": ["mutual_info", "anova_f"],
              "ranking_sample_rows": 5000, "stability_runs": 0},
    balance={"strategy": "smote", "max_ratio": 0.25, "max_expansion": 20.0},
    tune={"enabled": False},
    train={"models": ["stub_softmax"], "primary": "stub_softmax",
           "selection_metric": "macro_f1", "shap_background_rows": 80,
           "shap_explain_rows": 300},
)
data = prepare_data(cfg, synthetic_rows=13_000, cache=False, quiet=True)
run = train(cfg, data=data, tune=False, explain=False, save=True, quiet=True)
ARTIFACTS = cfg.paths.resolve("artifacts")
print(f"  trained {run.best.name}: test macro F1 {run.best.test.macro_f1:.4f}")

# The app finds samples at `find_artifacts().parent / "data" / "samples"`, so this lands
# exactly where a real `shieldnet prepare --sample` would put it.
labels = pd.Series([data.class_names[i] for i in data.split.y_test],
                   name=sch.LABEL_COLUMN)
upload = data.split.X_test.copy().assign(**{sch.LABEL_COLUMN: labels.values})
SAMPLES = WORK / "data" / "samples"
SAMPLES.mkdir(parents=True, exist_ok=True)
SAMPLE_CSV = SAMPLES / "cicids2017_sample.csv"
upload.to_csv(SAMPLE_CSV, index=False)
print(f"  wrote {SAMPLE_CSV.name}: {upload.shape[0]:,} rows x {upload.shape[1]} columns")

# Everything below - including the app script - resolves the artifact through this.
os.environ["SHIELDNET_ARTIFACTS"] = str(ARTIFACTS)

import shieldnet_ui as ui                                   # noqa: E402

# ---------------------------------------------------------------- 1. find_artifacts
print("\n" + "=" * 78 + "\n=== 1. finding the artifact ===\n" + "=" * 78)
assert ui.find_artifacts() == ARTIFACTS, ui.find_artifacts()
assert ui.find_artifacts("/tmp/elsewhere") == Path("/tmp/elsewhere"), "explicit must win"
print(f"  env -> {ui.find_artifacts()}")
print(f"  explicit argument overrides the env: {ui.find_artifacts('/tmp/elsewhere')}")

det = ui.load_detector()
assert det.attack_threshold == 0.5 and det.min_confidence == 0.5
print("  " + det.describe())

# ---------------------------------------------------------------- 2. provenance
print("\n" + "=" * 78 + "\n=== 2. model card and provenance ===\n" + "=" * 78)
card = ui.model_card(det)
for key in ("model", "n_features", "n_classes", "macro_f1", "accuracy", "attack_recall",
            "false_alarm_rate", "synthetic", "source", "trained_at", "selection_metric"):
    assert key in card, key
# Every headline number must have come from the training run. A None here means the app
# would print "n/a" where the sidebar promises a metric.
for key in ("macro_f1", "accuracy", "attack_recall", "false_alarm_rate",
            "macro_roc_auc", "calibration_error"):
    assert card[key] is not None, f"{key} was never measured into the bundle"
assert card["n_features"] == 16 and card["n_classes"] == len(data.class_names)
assert card["synthetic"] is True, "a generated run must be flagged in the bundle"
assert card["trained_at"], "the run stamp did not survive into the artifact"
print(f"  {card['model']}: macro F1 {card['macro_f1']:.4f}, "
      f"attack recall {card['attack_recall']:.2%}, "
      f"false alarms {card['false_alarm_rate']:.2%}, synthetic={card['synthetic']}")

lines = ui.provenance_lines(det)
print(f"  {len(lines)} provenance line(s); the first is the one that matters:")
print("    " + lines[0][:150])
# The single most damaging thing this app could do is let a generated number be read as a
# CICIDS2017 result, so the warning is first, not merely present.
assert "Synthetic" in lines[0] and "not on CICIDS2017" in lines[0]
assert any("training rows" in line for line in lines)
assert any("Class balance" in line for line in lines)
assert any("Attributions" in line for line in lines)
assert ui.format_metric(None) == "n/a" and ui.humanise(None) == "n/a"
assert ui.format_metric(float("nan")) == "n/a", "an unmeasured metric must not read as 0"
assert ui.format_metric(0.123456, "pct") == "12.35%"
assert ui.humanise(np.int64(1234567)) == "1,234,567"

# ---------------------------------------------------------------- 3. reference tables
print("\n" + "=" * 78 + "\n=== 3. class and feature reference ===\n" + "=" * 78)
classes = ui.class_reference_table(det)
print(f"  class table: {classes.shape[0]} rows x {list(classes.columns)}")
assert len(classes) == det.n_classes
assert set(classes["class"]) == set(det.class_names)
assert "test_recall" in classes.columns
# Recall belongs beside the name: two equally confident verdicts from classes whose recall
# differs by forty points are not equally trustworthy, and the table is where that is said.
measured = classes["test_recall"].notna().sum()
assert measured >= det.n_classes - 2, f"only {measured} classes carry a recall"
assert classes["severity"].max() >= 4, "no severe class in the reference table"
# Class order, not severity order - it has to line up with the per-class metrics table and
# the confusion matrix axes, which the reader is looking at beside it.
assert list(classes["index"]) == list(range(det.n_classes)), "not in class order"
assert list(classes["class"]) == list(det.class_names)
assert classes.loc[classes["class"] == sch.BENIGN_LABEL, "severity"].iloc[0] == 0
assert (classes.loc[classes["class"] != sch.BENIGN_LABEL, "severity"] >= 1).all()

features = ui.feature_table(det)
print(f"  feature table: {features.shape[0]} rows x {list(features.columns)}")
assert len(features) == det.n_features
assert list(features["feature"]) == list(det.feature_names), "selection order was lost"

# The caption above that table used to say "the three filter methods agreed on" as a fixed
# string. This run uses two, config/fast.yaml uses two, and RFE removes itself when
# scikit-learn is missing - so the sentence has to come from the artifact. Asserting the
# count is what makes that non-negotiable.
caption = ui.selection_caption(det)
print(f"  caption: {caption}")
ran = (det.bundle.metadata or {})["feature_selection"]["methods"]
assert len(ran) == 2, ran            # mutual_info + anova_f, per the config above
assert "2 filter methods" in caption, caption
assert "three" not in caption, "the caption is still hardcoded to three methods"
for method in ran:
    assert method in caption, method

# The four shapes that sentence has to take. The RFE-skipped one is not hypothetical: RFE
# removes itself when scikit-learn is missing, which is the environment these checks run in.
_real_meta = det.bundle.metadata
for methods, skipped, wanted in (
        (["mutual_info", "chi2", "rfe"], {}, "all 3 filter methods"),
        (["mutual_info", "anova_f"], {"rfe": "scikit-learn is not installed"},
         "2 filter methods"),
        (["mutual_info"], {}, "1 filter method ("),
        ([], {"chi2": "negative values"}, "the training run's feature ranking"),
):
    det.bundle.metadata = {"feature_selection": {"methods": methods, "skipped": skipped}}
    line = ui.selection_caption(det)
    assert wanted in line, (methods, line)
    assert all(m in line for m in methods), (methods, line)
    for name, why in skipped.items():
        assert f"{name} ({why})" in line, line
    if not skipped:
        assert "Skipped:" not in line, line
    print(f"    {len(methods)} ran, {len(skipped)} skipped -> {line[17:75]}...")
det.bundle.metadata = _real_meta

groups = ui.manual_form_groups(det)
flat = [spec["name"] for _title, block in groups for spec in block]
print(f"  form: {len(groups)} group(s), {len(flat)} field(s) - "
      + ", ".join(t for t, _ in groups))
assert len(flat) == len(set(flat)), "a field is offered twice"
assert set(flat) == {f["name"] for f in det.manual_fields()}, "a field was dropped"
assert all("used_by_model" in spec and spec["help"] for _t, b in groups for spec in b)

# A preset that seeds a feature the form does not show is a value the user can never see
# or change, and a typo in a preset key would be silently ignored by `predict_one`.
manual_names = set(flat)
coverage = []
for name, preset in ui.SHAPE_PRESETS.items():
    unknown = set(preset) - set(sch.CANONICAL_FEATURES)
    assert not unknown, f"{name} seeds a non-existent feature: {unknown}"
    offscreen = set(preset) - manual_names
    assert not offscreen, f"{name} seeds a feature the form cannot show: {offscreen}"
    # Equality, not containment: a preset that left one field at the training median would
    # mix a characteristic value with an average one and nothing on screen would say which
    # is which.
    assert set(preset) == manual_names, f"{name} leaves {manual_names - set(preset)} unset"
    coverage.append(len(preset))
print(f"  {len(ui.SHAPE_PRESETS)} shape preset(s), each seeding "
      f"{min(coverage)}-{max(coverage)} of the {len(manual_names)} form field(s)")

# ---------------------------------------------------------------- 4. samples
print("\n" + "=" * 78 + "\n=== 4. bundled samples ===\n" + "=" * 78)
found = ui.sample_files()
print(f"  {len(found)} sample(s): " + ", ".join(p.name for p in found))
assert [p.name for p in found] == [SAMPLE_CSV.name]

real_rows = ui.sample_rows_by_class(SAMPLE_CSV, det)
print(f"  real rows recovered for {len(real_rows)} of {det.n_classes} class(es)")
assert real_rows, "no labelled rows were recovered from the sample"
assert set(real_rows) <= set(det.class_names)
first = next(iter(real_rows.values()))
# The point of preferring these over the presets is that nothing is left at a median.
assert len(first) >= 70, f"a real row should carry every feature, got {len(first)}"
assert sch.LABEL_COLUMN not in first, "the label leaked into the feature dict"
ATTACK_LABEL = next((n for n in real_rows if n != sch.BENIGN_LABEL), None)
assert ATTACK_LABEL, "the sample carries no attack rows"
print(f"  will hand-classify a real {ATTACK_LABEL} row")

# ---------------------------------------------------------------- 5. batch surfaces
print("\n" + "=" * 78 + "\n=== 5. scoring a capture ===\n" + "=" * 78)
batch, prep = ui.score_frame(det, upload, threshold=0.5, min_confidence=0.5,
                            source=SAMPLE_CSV.name)
assert len(batch) == len(upload), "a row was dropped between input and verdict"
print(f"  {len(batch):,} verdict(s) for {len(upload):,} row(s); "
      f"{prep.repaired_cells:,} cell(s) repaired; "
      f"{len(prep.warnings())} warning(s), {len(prep.notes())} note(s)")

over = ui.batch_overview(batch)
for key in ("rows", "attack_flows", "attack_share_pct", "distinct_classes",
            "critical_flows", "low_confidence_rows", "low_confidence_share_pct",
            "mean_confidence"):
    assert key in over, key
print(f"  {over['attack_flows']:,} flagged ({over['attack_share_pct']:.1f}%), "
      f"{over['distinct_classes']} class(es) present, "
      f"{over['critical_flows']:,} critical, "
      f"{over['low_confidence_rows']:,} need review")
assert 0 <= over["attack_share_pct"] <= 100
assert over["critical_flows"] <= over["labelled_attack_flows"]

note = ui.reconciliation_note(batch)
print("  reconciliation: " + note[:130])
# Whatever the gap, the sentence has to name both counts, because both are on the screen.
assert f"{int(batch.is_attack.sum()):,}" in note
assert f"{int((batch.predicted != batch.benign_index).sum()):,}" in note

dist = ui.class_distribution(batch)
assert int(dist["flows"].sum()) == len(batch), "the chart loses rows"
assert (dist["flows"] > 0).all(), "an absent class is being plotted at zero"
assert list(dist["severity"]) == sorted(dist["severity"], reverse=True)
print(f"  distribution: {len(dist)} bar(s), top is "
      f"{dist.iloc[0]['class']} at {int(dist.iloc[0]['flows']):,}")

hist = ui.confidence_histogram(batch)
assert int(hist["flows"].sum()) == len(batch), "the histogram loses rows"
assert set(hist["group"]) == {"flagged as attack", "left as benign"}

triage = ui.triage_table(batch)
print(f"  triage queue: {len(triage)} attack class(es)")
if len(triage):
    assert list(triage["priority"]) == list(range(1, len(triage) + 1))
    assert list(triage["severity"]) == sorted(triage["severity"], reverse=True)
    assert triage["what_to_do"].map(bool).all(), "a queue entry has no action"

# ---------------------------------------------------------------- 6. view vs record
print("\n" + "=" * 78 + "\n=== 6. the filtered view and the whole record ===\n" + "=" * 78)
full = ui.verdict_table(batch)
attacks_only = ui.verdict_table(batch, only_attacks=True)
review_only = ui.verdict_table(batch, only_review=True)
capped = ui.verdict_table(batch, limit=25)
print(f"  {len(full):,} rows unfiltered, {len(attacks_only):,} attacks, "
      f"{len(review_only):,} needing review, {len(capped)} when capped at 25")
assert len(full) == len(batch)
assert len(capped) == min(25, len(batch))
assert attacks_only["is_attack"].all() if len(attacks_only) else True
assert (review_only["status"] == "review").all() if len(review_only) else True
assert list(full["severity"]) == sorted(full["severity"], reverse=True)

# The invariant the whole pipeline promises. A download that inherited the screen's filter
# would break `len(csv) == len(input)` while looking perfectly reasonable.
record = pd.read_csv(io.BytesIO(ui.csv_bytes(batch.frame(probabilities=False, top_k=3))))
assert len(record) == len(upload), f"the export holds {len(record)} of {len(upload)} rows"
assert "Unnamed: 0" not in record.columns, "the export carries a pandas index column"
wide = pd.read_csv(io.BytesIO(ui.csv_bytes(batch.frame(probabilities=True, top_k=3))))
assert len(wide) == len(upload)
assert wide.shape[1] > record.shape[1] + det.n_classes - 1
print(f"  export: {len(record):,} rows x {record.shape[1]} cols; "
      f"with probabilities {wide.shape[1]} cols")

# ---------------------------------------------------------------- 7. thresholds
print("\n" + "=" * 78 + "\n=== 7. moving a slider must not re-score ===\n" + "=" * 78)
# The per-row probability chart needs the full matrix, so the app's scorer must retain it.
assert batch.proba is not None, "score_frame dropped the probability matrix"
before = batch.proba
ui.apply_thresholds(det, batch, threshold=0.05, min_confidence=0.5)
loose = int(batch.is_attack.sum())
ui.apply_thresholds(det, batch, threshold=0.95, min_confidence=0.5)
strict = int(batch.is_attack.sum())
print(f"  threshold 0.05 flags {loose:,}; 0.95 flags {strict:,}")
assert loose >= strict, "a stricter threshold flagged more"
assert batch.proba is before, "the probabilities were recomputed"
assert det.attack_threshold == 0.95 and batch.attack_threshold == 0.95
ui.apply_thresholds(det, batch, threshold=0.5, min_confidence=0.5)

# ---------------------------------------------------------------- 8. one row
print("\n" + "=" * 78 + "\n=== 8. explaining one row ===\n" + "=" * 78)
target = int(np.argmax(batch.attack_probability))
pred = det.inspect(prep, batch, target, narrate=True)
assert pred.row == target
assert pred.predicted_class == batch.class_names[batch.predicted[target]]
assert abs(pred.confidence - float(batch.confidence[target])) < 1e-9
print(f"  row {target}: {pred.predicted_class} at {pred.confidence:.1%}, "
      f"severity {pred.severity}, method "
      f"{pred.explanation.method if pred.explanation else 'none'}")

probs = ui.probability_table(pred, top=8)
assert len(probs) == 8 and probs["probability"].is_monotonic_decreasing
assert probs.iloc[0]["class"] == pred.predicted_class
table = ui.explanation_table(pred.explanation, top=10)
print(f"  contributions: {len(table)} row(s) x {list(table.columns)}")
assert len(table) == 10
assert table["contribution"].abs().is_monotonic_decreasing
# `Detector.raw_values` is public for this: an analyst matching this row against a capture
# needs Flow Duration = 96,000,000, not the z-score 1.83. When the raw value is available
# the table must say so, because the two live in the same column.
assert "scale" in table.columns
print(f"  value scale: {sorted(set(table['scale']))}")
assert set(table["scale"]) <= {"as captured", "standardised"}
assert (table["scale"] == "as captured").all(), \
    "raw values were recoverable for this row but the table fell back to z-scores"
raw = det.raw_values(prep, target)
assert raw is not None and len(raw) == prep.X.shape[1]
# The number in the table is the number in the upload, not a rescaled cousin of it.
by_name = dict(zip(det.feature_names, raw))
for _, r in table.iterrows():
    assert abs(by_name[r["feature"]] - r["value"]) < 1e-6, r["feature"]
assert pred.narrative, "no narration for the highest-probability attack in the file"
print("  narrative: " + pred.narrative.split(". ")[0][:120])

# ---------------------------------------------------------------- 9. evaluation
print("\n" + "=" * 78 + "\n=== 9. evaluation surfaces ===\n" + "=" * 78)
report = det.evaluate(upload, fpr_budget=0.01, quiet=True)
o = ui.evaluation_overview(report)
print(f"  macro F1 {o['macro_f1']:.4f}, balanced accuracy {o['balanced_accuracy']:.4f}, "
      f"accuracy {o['accuracy']:.4f}, {o['rows']:,} rows")
assert o["rows"] == len(upload)
assert 0.0 <= o["macro_f1"] <= 1.0
assert len(o["worst"]) <= 3 and all(len(w) == 3 for w in o["worst"])

per_class = ui.per_class_table(report)
assert "class" in per_class.columns, per_class.columns
assert len(per_class) == det.n_classes
cm = ui.confusion_frame(report, normalise=True)
sums = cm.sum(axis=1).to_numpy()
present = ~np.isin(cm.index, o["classes_absent"])
assert np.allclose(sums[present], 1.0), f"normalised rows do not sum to 1: {sums}"
assert np.allclose(ui.confusion_frame(report, normalise=False).to_numpy().sum(),
                   len(upload)), "the raw confusion matrix does not account for every row"
print(f"  per-class table {per_class.shape}, confusion {cm.shape}, "
      f"{len(o['classes_absent'])} class(es) absent")

binary = ui.binary_table(report)
assert len(binary) == 11 and set(binary.columns) == {"measure", "value"}
sweep = ui.sweep_table(report)
print(f"  sweep: {len(sweep)} operating point(s)")
assert len(sweep) > 1, "the sweep produced no curve"
assert list(sweep.columns) == ["threshold", "detection_rate", "false_alarm_rate",
                               "precision", "f1"]
# Thresholds run high to low, as ThresholdSweep documents and as its `at()` lookup
# requires. Relaxing the bar cannot catch less or cry wolf less: that is the property that
# makes the curve worth drawing, and reading it off `to_dict()` - which carries no arrays -
# would have silently produced an empty chart.
assert sweep["threshold"].is_monotonic_decreasing, "the sweep order was resorted"
assert sweep["detection_rate"].is_monotonic_increasing
assert sweep["false_alarm_rate"].is_monotonic_increasing
recs = ui.sweep_recommendations(report)
for name, value, why in recs:
    print(f"    {name}: {value:.3f} - {why[:70]}")
    assert 0.0 <= value <= 1.0 and len(why) > 30
assert len(recs) >= 2

# ---------------------------------------------------------------- 10. run the app
print("\n" + "=" * 78 + "\n=== 10. running app/streamlit_app.py ===\n" + "=" * 78)


def run_app(script, *, expect_stop=False, session=None):
    """Execute the app script once against a fresh recorder."""
    st = stub_streamlit.install(script, session)
    try:
        runpy.run_path(str(APP), run_name="__main__")
    except stub_streamlit.StopScript:
        if not expect_stop:
            raise AssertionError("the app called st.stop() unexpectedly: "
                                 + "; ".join(st.errors()))
    else:
        if expect_stop:
            raise AssertionError("the app should have stopped and did not")
    finally:
        stub_streamlit.uninstall()
    # Any Streamlit attribute the stub has not modelled is served as a no-op, which would
    # make a green run meaningless. This is the assertion that keeps the double honest.
    assert not st.unknown, f"unmodelled streamlit API: {sorted(st.unknown)}"
    return st


# --- pass A: first render, nothing uploaded -----------------------------------
st = run_app({})
print("  A (empty state): " + st.summary())
assert st.errors(), "the synthetic-data warning is missing from a synthetic artifact"
assert "SYNTHETIC" in st.errors()[0].upper(), st.errors()[0]
tabs = [c for c in st.calls if c.kind == "tabs"]
assert len(tabs) == 1 and len(tabs[0].label) == 4, tabs
assert st.said("Upload a capture or pick a sample"), "no empty-state guidance"
assert st.said("Pick a labelled file"), "the evaluate tab has no empty state"
metrics = st.metrics()
assert "macro F1" in metrics and "false alarms" in metrics
assert metrics["accuracy"] == ui.format_metric(card["accuracy"])
# The model and class tabs render without any upload, so the tables are already there.
assert len(st.frames()) >= 2, f"only {len(st.frames())} table(s) on a first render"
assert st.said("Trained on"), "no provenance in the sidebar"
assert not st.said("flows scored"), "a verdict metric appeared with nothing scored"

# --- pass A2: Score pressed with nothing chosen -------------------------------
# The button is disabled in that state, and the stub honours `disabled`, so this asserts
# the guard rather than the stub: without it the app would reach read_flows(None).
st = run_app({"Score": True, "or score a bundled sample": "-"})
print("  A2 (nothing chosen): " + st.summary())
assert not st.said("flows scored"), "the app scored with no file selected"
scored = [c for c in st.calls if c.kind == "button" and c.label == "Score"]
assert scored and scored[0].value is False, "a disabled Score button fired"
assert scored[0].kwargs.get("disabled") is True, "Score was not disabled"
assert st.said("Upload a capture or pick a sample")

# --- pass B: score the bundled sample ----------------------------------------
st = run_app({"Score": True, "rows to display": 40})
print("  B (scored): " + st.summary())
assert st.said("flows scored"), "the sample was not scored"
m = st.metrics()
for label in ("flows scored", "flagged as attack", "attack classes seen",
              "critical (severity 5)", "needs review", "mean confidence"):
    assert label in m, label
assert m["flows scored"] == ui.humanise(len(upload)), m["flows scored"]
print(f"    {m['flows scored']} scored, {m['flagged as attack']} flagged, "
      f"{m['needs review']} to review")
assert st.said("Both ways of counting agree") or st.said("The headline counts") \
    or st.said("The class table shows"), "the reconciliation sentence is missing"
charts = st.charts()
print(f"    {len(charts)} chart(s): " + ", ".join(sorted({c.kind for c in charts})))
assert charts, "no charts were drawn"
# plotly is absent here, so this run exercises the st.bar_chart fallback - which is the
# path any machine without plotly will take, and therefore the one worth having run.
assert any(c.kind in ("chart:bar", "chart:plotly") for c in charts)
downloads = st.downloads()
assert len(downloads) == 2, list(downloads)
names = [str(c.kwargs.get("file_name")) for c in st.calls if c.kind == "download_button"]
print("    filenames: " + ", ".join(names))
assert all(n.count(".csv") == 1 for n in names), f"doubled extension: {names}"
assert all(SAMPLE_CSV.stem in n for n in names), "the export does not name its source"
for label, payload in downloads.items():
    got = pd.read_csv(io.BytesIO(payload))
    print(f"    {label!r} -> {len(got):,} rows x {got.shape[1]} cols")
    assert len(got) == len(upload), f"{label} exported {len(got)} of {len(upload)} rows"
shown = [c for c in st.calls if c.kind == "dataframe"]
assert any(len(c.payload) <= 40 for c in shown if hasattr(c.payload, "__len__")), \
    "the row cap did not reach the table"

# --- pass C: explain a row, and a preset shape -------------------------------
preset = "Port-scan shape (many tiny one-way flows)"
st = run_app({"Score": True, "Explain this row": True, "row number": 17,
              "start from": preset, "Classify this flow": True})
print("  C (explained + preset): " + st.summary())
assert st.said("Row 17"), "the row inspector did not run"
assert st.said("attribution method"), "no attribution provenance on screen"
assert any("hand-built" in w for w in st.warnings()), \
    "the preset was not labelled as a synthetic shape"
assert st.said("What it is."), "the attack profile did not render"
assert st.said("How it shows up in flow features.")
assert st.said("What to do.")
expanders = [str(c.label) for c in st.calls if c.kind == "expander"]
print(f"    expanders: " + ", ".join(expanders))
assert any(e.startswith("about ") for e in expanders)

# --- pass D: a real row, and a full evaluation -------------------------------
real_option = f"real {ATTACK_LABEL} flow from {SAMPLE_CSV.name}"
st = run_app({"Score": True, "start from": real_option, "Classify this flow": True,
              "Evaluate": True, "row-normalise": True})
print("  D (real row + evaluation): " + st.summary())
picks = st.widget_values()
assert picks.get("start from") == real_option, picks.get("start from")
assert any("real" in s and "populated" in s for s in st.texts()), \
    "a real sample row was not distinguished from a hand-built shape"
m = st.metrics()
for label in ("macro F1", "balanced accuracy", "macro recall", "MCC", "rows"):
    assert label in m, label
assert m["rows"] == ui.humanise(len(upload)), m["rows"]
assert st.said("best F1"), "the threshold recommendations are missing"
assert st.said("Youden J")
assert any(c.kind in ("chart:line", "chart:plotly") for c in st.charts()), \
    "the sweep curve was not drawn"
json_downloads = [c for c in st.calls if c.kind == "download_button"
                  and "JSON" in str(c.label)]
assert json_downloads, "the evaluation report cannot be exported"
assert len(json_downloads[0].payload) > 2_000, "the exported report looks truncated"
print(f"    report export: {len(json_downloads[0].payload):,} bytes")

# --- pass E: two reruns of one session ---------------------------------------
# Streamlit reruns the script and keeps session_state, which is exactly how a stale
# explanation gets on screen: score file A, explain row 17, score file B, and row 17's
# card is still there describing a flow that is no longer in the table.
shared = stub_streamlit.SessionState()
st = run_app({"Score": True, "Explain this row": True, "row number": 17},
             session=shared)
print("  E (first visit): " + st.summary())
assert st.said("Row 17"), "the explanation did not render on the first rerun"
assert "row_pred" in shared and shared["batch"] is not None
st = run_app({"Score": True}, session=shared)
print("  F (re-scored, same session): " + st.summary())
assert "row_pred" not in shared, "a stale explanation survived a re-score"
assert not st.said("Row 17"), "row 17's card is still on screen after a new capture"
assert st.said("flows scored"), "the second score did not run"

# --- pass G: no artifact -----------------------------------------------------
empty = WORK / "no-artifact-here"
empty.mkdir(exist_ok=True)
os.environ["SHIELDNET_ARTIFACTS"] = str(empty)
try:
    st = run_app({}, expect_stop=True)
finally:
    os.environ["SHIELDNET_ARTIFACTS"] = str(ARTIFACTS)
print("  G (no artifact): " + st.summary())
assert st.errors() and "No usable model" in st.errors()[0]
# The first screen a new user sees. It has to contain the command that fixes it.
advice = " ".join(st.texts())
assert "shieldnet train" in advice, "the error does not say how to get a model"
assert "shieldnet download" in advice, "the error does not mention the real dataset"
assert not st.said("flows scored"), "the app kept rendering after st.stop()"
print("    error text carries both `shieldnet train` and `shieldnet download`")

print("\n" + "=" * 78 + "\nALL APP CHECKS PASSED\n" + "=" * 78)
