"""Verify the inference path reproduces the training path on messy real-world input.

The claim under test is the one that matters for the app: a bundle loaded from disk and
handed a user's CSV - wrong column order, missing columns, ``"Infinity"`` strings,
duplicate headers, raw en-dash label spellings - must produce the same verdicts the
training run measured, row for row.
"""
import io
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Derived from this file, never hardcoded: an absolute path here would make a copy of the
# project silently import the original, so the copy would pass while being untested.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests/_stubs"))
import stub_models; stub_models.register()

from shieldnet import schema as sch
from shieldnet.config import Config
from shieldnet.inference import (DEFAULT_MIN_CONFIDENCE, Detector, PredictionBatch,
                                 PreparedInput, read_flows)
from shieldnet.logging_utils import configure_logging
from shieldnet.persist import BundleError
from shieldnet.train import prepare_data, train

configure_logging("INFO", force=True)

WORK = Path("/tmp/shieldnet_infer")
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)

cfg = Config.load(
    seed=42, n_jobs=1,
    paths={"root": str(WORK)},
    data={"test_size": 0.2, "val_size": 0.1, "min_class_rows": 10},
    features={"n_features": 18, "methods": ["mutual_info", "anova_f"],
              "ranking_sample_rows": 6000, "stability_runs": 0},
    balance={"strategy": "smote", "max_ratio": 0.25, "max_expansion": 20.0},
    tune={"enabled": False},
    train={"models": ["stub_softmax"], "primary": "stub_softmax",
           "selection_metric": "macro_f1", "shap_background_rows": 100,
           "shap_explain_rows": 400},
)

print("=" * 78 + "\n=== 0. a real artifact to score against ===\n" + "=" * 78)
data = prepare_data(cfg, synthetic_rows=14_000, cache=False, quiet=True)
run = train(cfg, data=data, tune=False, explain=False, save=True, quiet=True)
artifacts = cfg.paths.resolve("artifacts")
print(f"  trained {run.best.name}: test macro F1 {run.best.test.macro_f1:.4f}, "
      f"accuracy {run.best.test.accuracy:.4f}")

# The labelled frame a user would upload: raw feature columns plus a Label column.
class_names = data.class_names
raw_test = data.split.X_test.copy()
labels = pd.Series([class_names[i] for i in data.split.y_test], name=sch.LABEL_COLUMN)
upload = raw_test.assign(**{sch.LABEL_COLUMN: labels})
print(f"  upload frame: {upload.shape[0]:,} rows x {upload.shape[1]} columns "
      f"(including {sch.LABEL_COLUMN})")

# ------------------------------------------------------------------ 1. load
print("\n" + "=" * 78 + "\n=== 1. Detector.load ===\n" + "=" * 78)
det = Detector.load(artifacts)
print("  " + det.describe())
assert det.model_name == run.best.name
assert det.feature_names == run.space.selected if hasattr(run, "space") else True
assert det.n_features == cfg.features.n_features, det.n_features
assert det.n_classes == len(class_names)
assert det.class_names == class_names
# The bundle ships medians for every canonical feature, not just the selected ones -
# the manual-entry form needs a default for all of them.
template = det.template_row()
print(f"  template row covers {len(template)} feature(s); "
      f"{len(det.feature_names)} of them are used by the model")
# The form must be able to default any canonical feature, not only the selected ones -
# the user does not know which 18 the selector kept.
assert len(template) >= 70, len(template)
assert all(f in template for f in det.feature_names), "a selected feature has no default"
assert all(isinstance(v, float) for v in template.values())
fields = det.manual_fields()
used = [f["name"] for f in fields if f["used_by_model"]]
print(f"  manual form offers {len(fields)} field(s), {len(used)} of which this model uses")
assert len(fields) == len(sch.MANUAL_ENTRY_FEATURES)
assert all(f["help"] for f in fields)

missing_dir = WORK / "nowhere"
try:
    Detector.load(missing_dir)
except BundleError as exc:
    print(f"  BundleError for a missing artifact: {str(exc).splitlines()[0][:88]}")
    assert "shieldnet train" in str(exc), "the error does not say how to fix it"
else:
    raise AssertionError("loading a nonexistent artifact should have raised")

# ------------------------------------------------------------------ 2. clean input
print("\n" + "=" * 78 + "\n=== 2. a clean upload reproduces the training-time matrix ===\n"
      + "=" * 78)
prep = det.prepare(upload)
print(prep.render())
assert isinstance(prep, PreparedInput)
assert prep.rows == len(upload)
assert prep.X.shape == (len(upload), det.n_features)
assert not prep.missing, prep.missing
assert prep.has_labels and prep.y_true is not None
assert not prep.unknown_labels, prep.unknown_labels
assert np.array_equal(prep.y_true, data.split.y_test), "labels did not round-trip"
# This is the whole point of the two-pass preprocessor: the shipped scaler must give
# bit-comparable values to the one used at fit time.
drift = float(np.abs(prep.X - run.space.X_test).max())
print(f"  max |prepared - training-time X_test| = {drift:.3e}")
assert drift < 1e-9, drift

batch = det.predict(None, prepared=prep)
print("\n" + batch.render())
assert isinstance(batch, PredictionBatch)
assert len(batch) == len(upload), "inference dropped or invented a row"
assert batch.proba is not None and batch.proba.shape == (len(upload), det.n_classes)
assert np.allclose(batch.proba.sum(axis=1), 1.0, atol=1e-9)
# Verdicts must match what the training run measured.
train_pred = run.best.model.predict_proba(run.space.X_test).argmax(1)
agree = float((batch.predicted == train_pred).mean())
print(f"  agreement with the training-time predictions: {agree:.4%}")
assert agree == 1.0, agree
assert np.all(batch.confidence >= batch.runner_up_confidence), "top-2 order is wrong"
assert np.all(batch.confidence <= 1.0 + 1e-12) and np.all(batch.confidence > 0)

# ------------------------------------------------------------------ 3. evaluation
print("\n" + "=" * 78 + "\n=== 3. Detector.evaluate matches the training report ===\n"
      + "=" * 78)
report = det.evaluate(upload, quiet=True)
print(f"  uploaded macro F1 {report.macro_f1:.6f}  vs  training test "
      f"{run.best.test.macro_f1:.6f}")
assert abs(report.macro_f1 - run.best.test.macro_f1) < 1e-12
assert abs(report.accuracy - run.best.test.accuracy) < 1e-12
assert report.split == "uploaded"
assert report.model == det.model_name
print("  the shipped artifact and the training run agree to machine precision")

# ------------------------------------------------------------------ 4. messy input
print("\n" + "=" * 78 + "\n=== 4. the upload a user actually produces ===\n" + "=" * 78)
messy = upload.sample(frac=1.0, axis=1, random_state=3).copy()   # columns shuffled

# (a) Headers as CICFlowMeter writes them: a leading space on every one.
messy.columns = [" " + str(c) for c in messy.columns]
# (b) One selected feature simply absent.
gone = det.feature_names[0]
messy = messy.drop(columns=[" " + gone])
# (c) A duplicated column name, which CICIDS2017 genuinely ships.
messy[" Fwd Header Length.1"] = messy.get(" Fwd Header Length", 0)
messy.columns = [c if c != " Fwd Header Length.1" else " Fwd Header Length"
                 for c in messy.columns]
# (d) Division artefacts as the strings pandas leaves in an object column. The victim has
#     to be a feature the model actually selected, or the repair is correctly ignored.
victim = det.feature_names[1]
messy[" " + victim] = messy[" " + victim].astype(object)
messy.iloc[0, messy.columns.get_loc(" " + victim)] = "Infinity"
messy.iloc[1, messy.columns.get_loc(" " + victim)] = "NaN"
messy.iloc[2, messy.columns.get_loc(" " + victim)] = "not a number at all"
# (e) Raw label spellings, including one that merges into Rare Attacks and one the
#     model has never heard of.
lab = " " + sch.LABEL_COLUMN
messy[lab] = messy[lab].replace({"Web Attack - XSS": "Web Attack \x96 XSS",
                                 "Rare Attacks": "Heartbleed"})
messy.iloc[5, messy.columns.get_loc(lab)] = "Some Attack From 2031"
# (f) Columns the model was never trained on.
messy[" analyst_note"] = "seen in yesterday's capture"
messy[" capture_id"] = 7

print(f"  {messy.shape[1]} columns, shuffled, space-prefixed, one feature missing, "
      f"1 duplicated, 3 junk cells, 2 extras")
messy_prep = det.prepare(messy)
print("\n" + messy_prep.render())
assert messy_prep.rows == len(upload), "a messy upload lost rows"
assert messy_prep.X.shape[1] == det.n_features
assert np.isfinite(messy_prep.X).all(), "non-finite values reached the model"
assert messy_prep.missing == [gone], messy_prep.missing
assert "Fwd Header Length" in messy_prep.duplicates_merged, messy_prep.duplicates_merged
# Two of the three junk cells count as coerced, not three: to_numeric parses "Infinity"
# as a float, so that one shows up under non_finite instead. The literal string "NaN" and
# the free prose both become missing, which is what "coerced" means here.
assert messy_prep.coerced == {victim: 2}, messy_prep.coerced
assert messy_prep.non_finite.get(victim, 0) >= 1, messy_prep.non_finite
# A text column the model never sees must not be reported as a data problem.
assert "analyst_note" not in messy_prep.coerced, messy_prep.coerced
assert "analyst_note" in messy_prep.extra and "capture_id" in messy_prep.extra
assert messy_prep.unknown_labels == {"Some Attack From 2031": 1}, messy_prep.unknown_labels
warns, notes = messy_prep.warnings(), messy_prep.notes()
print(f"  {len(warns)} warning(s) and {len(notes)} note(s)")
# Five things went wrong (a missing feature, junk cells, infinities, a duplicated
# header, an unknown label) and one thing merely happened (columns the model does not
# use). Unused columns are the normal state of every real CICFlowMeter export, so they
# must not dilute the list of actual problems.
assert len(warns) == 5, warns
assert len(notes) == 1 and "not used by this model" in notes[0], notes
assert not any("not used by this model" in w for w in warns), \
    "an unremarkable observation is being reported as a warning"
# The en-dash spelling and Heartbleed must have resolved to the trained classes.
resolved = set(messy_prep.label_names or [])
assert "Web Attack - XSS" in resolved, "the en-dash label did not canonicalise"
assert sch.RARE_CLASS_NAME in resolved, "Heartbleed did not merge into Rare Attacks"

messy_batch = det.predict(None, prepared=messy_prep, quiet=True)
overlap = float((messy_batch.predicted == batch.predicted).mean())
print(f"\n  agreement with the clean upload: {overlap:.2%} "
      f"(one feature was imputed, so this is expected to be high, not perfect)")
assert overlap > 0.75, overlap
print("  a broken upload produces verdicts instead of a traceback")

# ------------------------------------------------------------------ 5. chunking
print("\n" + "=" * 78 + "\n=== 5. chunking cannot change a verdict ===\n" + "=" * 78)
chunked = det.predict(None, prepared=prep, chunk_rows=97, quiet=True)
assert np.array_equal(chunked.predicted, batch.predicted)
# Not bit-identical, and it should not be asserted to be: X @ W picks a different blocking
# strategy for a 97-row slice than for a 2,801-row one, so the last bits of the dot product
# differ. What must not move is the verdict.
drift = float(np.abs(chunked.proba - batch.proba).max())
print(f"  {len(chunked):,} rows in chunks of 97: identical predictions, "
      f"max probability drift {drift:.3e}")
assert drift < 1e-12, drift
lean = det.predict(None, prepared=prep, keep_proba=False, quiet=True)
assert lean.proba is None
assert np.array_equal(lean.predicted, batch.predicted)
assert np.abs(lean.attack_probability - batch.attack_probability).max() == 0.0
print("  keep_proba=False keeps the verdicts and drops only the matrix")

# ------------------------------------------------------------------ 6. thresholds
print("\n" + "=" * 78 + "\n=== 6. the two decisions the app exposes ===\n" + "=" * 78)
print(f"  {'attack_threshold':>16}  {'flagged as attack':>18}  {'share':>7}")
counts = []
for t in (0.05, 0.25, 0.50, 0.75, 0.95):
    batch.attack_threshold = t
    n = int(batch.is_attack.sum())
    counts.append(n)
    print(f"  {t:>16.2f}  {n:>18,}  {n / len(batch):>6.2%}")
assert counts == sorted(counts, reverse=True), "raising the threshold detected more"
batch.attack_threshold = 0.5
# argmax != BENIGN and P(attack) >= 0.5 are different questions; the code must not
# quietly conflate them.
argmax_attack = batch.labels != sch.BENIGN_LABEL
differ = int((argmax_attack != batch.is_attack).sum())
print(f"\n  rows where argmax!=BENIGN disagrees with P(attack)>=0.5: {differ:,}")
assert batch.benign_index == class_names.index(sch.BENIGN_LABEL)

for m in (0.0, DEFAULT_MIN_CONFIDENCE, 0.999):
    batch.min_confidence = m
    print(f"  min_confidence {m:>5.3f} -> {int(batch.flagged.sum()):>6,} row(s) for review")
batch.min_confidence = DEFAULT_MIN_CONFIDENCE
assert set(np.unique(batch.status)) <= {"ok", "review"}

# ------------------------------------------------------------------ 7. per-row
print("\n" + "=" * 78 + "\n=== 7. one row, explained and narrated ===\n" + "=" * 78)
det.explainer(run.space.X_train[:200])          # a real background, not the median row
attack_rows = np.nonzero(batch.labels != sch.BENIGN_LABEL)[0]
i = int(attack_rows[int(np.argmax(batch.confidence[attack_rows]))])
pred = batch.prediction(i, explainer=det.explainer(), X=prep.X,
                        raw_values=det._raw_values(prep, i), narrate=True)
print(f"  row {pred.row}: {pred.predicted_class} at {pred.confidence:.1%} "
      f"(runner-up {pred.runner_up} at {pred.runner_up_confidence:.1%})")
print(f"  severity {pred.severity}, status {pred.status}, true class {pred.true_class!r}, "
      f"correct={pred.correct}")
print(f"  top 3: {pred.top_classes(3)}")
print("\n  " + pred.narrative.replace(". ", ".\n  "))
assert pred.explanation is not None
assert pred.narrative and pred.predicted_class in pred.narrative
assert len(pred.explanation.contributions) == det.n_features
assert pred.correct is True or pred.true_class != pred.predicted_class
assert pred.severity >= 1, "an attack was graded severity 0"
assert abs(sum(p for _, p in pred.top_classes(det.n_classes)) - 1.0) < 1e-9
# A benign row must not be handed an attack severity.
benign_rows = np.nonzero(batch.labels == sch.BENIGN_LABEL)[0]
b = batch.prediction(int(benign_rows[0]))
assert b.severity == 0 and not b.is_attack, (b.severity, b.is_attack)
print(f"\n  benign row {b.row}: severity {b.severity}, is_attack {b.is_attack}")

# ------------------------------------------------------------------ 8. manual entry
print("\n" + "=" * 78 + "\n=== 8. the manual single-flow form ===\n" + "=" * 78)
form = {f["name"]: f["demo"] for f in det.manual_fields()}
print(f"  submitting {len(form)} field(s) out of {len(det.raw_feature_names)}")
manual = det.predict_one(form)
print(f"  -> {manual.predicted_class} at {manual.confidence:.1%}, severity "
      f"{manual.severity}, status {manual.status}")
print("\n  " + manual.narrative.replace(". ", ".\n  ")[:900])
assert manual.row == 0
assert manual.predicted_class in class_names
assert manual.narrative and manual.explanation is not None
assert 0.0 < manual.confidence <= 1.0
# A form is a what-if against a typical flow, not a damaged upload: unsupplied features
# come from the median template, so nothing is reported as missing.
form_prep = det.prepare({**det.template_row(), **form}, quiet=True)
assert not form_prep.missing, form_prep.missing
# A form with a single field must still work: everything else is median-filled.
sparse = det.predict_one({"Destination Port": 80}, explain=False, narrate=False)
print(f"  one-field submission -> {sparse.predicted_class} at {sparse.confidence:.1%}")
assert sparse.explanation is None and not sparse.narrative
assert sparse.predicted_class in class_names

# ------------------------------------------------------------------ 9. output
print("\n" + "=" * 78 + "\n=== 9. the CSV an analyst downloads ===\n" + "=" * 78)
out = batch.frame(top_k=3)
print(f"  {out.shape[0]:,} x {out.shape[1]} columns: {list(out.columns)}")
assert len(out) == len(upload)
assert list(out["row"]) == list(range(len(upload))), "row numbering broke alignment"
for column in ("prediction", "confidence", "attack_probability", "is_attack",
               "severity", "runner_up", "status", "true_class", "correct"):
    assert column in out.columns, column
assert out["correct"].mean() == (batch.predicted == batch.y_true).mean()
assert (out["alt1_probability"] <= out["confidence"] + 1e-12).all()
assert (out["alt2_probability"] <= out["alt1_probability"] + 1e-12).all()
print(out.head(6).to_string(index=False))

wide = batch.frame(probabilities=True)
assert wide.shape[1] == 11 + det.n_classes, wide.shape
assert all(f"p({c})" in wide.columns for c in class_names)
print(f"\n  with per-class probabilities: {wide.shape[1]} columns")

print("\n  -- summary and narrative --")
summary = batch.summary()
print("  " + json.dumps({k: v for k, v in summary.items() if k != "counts"},
                        indent=2, default=str).replace("\n", "\n  "))
assert summary["rows"] == len(upload)
assert summary["attack_flows"] + summary["benign_flows"] == len(upload)
assert sum(summary["counts"].values()) == len(upload)
assert 0 < summary["mean_confidence"] <= 1
# The threshold count and the label count must be reported separately and must reconcile:
# a dashboard that shows 656 attacks above a class table summing to 631 looks broken.
assert (summary["attack_flows"] - summary["labelled_attack_flows"]
        == summary["threshold_only_attacks"]), summary
assert summary["labelled_attack_flows"] == int((batch.labels != sch.BENIGN_LABEL).sum())
assert summary["attack_threshold"] == 0.5
print(f"  P(attack)>=0.5: {summary['attack_flows']:,}   attack labels: "
      f"{summary['labelled_attack_flows']:,}   difference: "
      f"{summary['threshold_only_attacks']:+,}")
if summary["threshold_only_attacks"]:
    assert "split across several attack classes" in batch.narrative() \
        or "fall below the" in batch.narrative(), "the two totals were never reconciled"
triage = batch.triage()
print(f"\n  triage queue: {triage[:5]}")
assert all(name != sch.BENIGN_LABEL for name, _, _ in triage)
assert all(triage[i][2] >= triage[i + 1][2] for i in range(len(triage) - 1))
print("\n  " + batch.narrative().replace(". ", ".\n  "))
assert str(len(upload)) not in "" and batch.narrative()

# ------------------------------------------------------------------ 10. read_flows
print("\n" + "=" * 78 + "\n=== 10. read_flows on a file written the way they arrive ===\n"
      + "=" * 78)
csv_path = WORK / "capture.csv"
export = upload.head(500).copy()
export.columns = [" " + str(c) for c in export.columns]
export.to_csv(csv_path, index=False, encoding="latin-1")
loaded = read_flows(csv_path)
print(f"  {csv_path.name}: {loaded.shape[0]:,} x {loaded.shape[1]}")
assert loaded.shape == (500, upload.shape[1])
sub = det.predict(loaded, source=csv_path.name, quiet=True)
assert len(sub) == 500
assert np.array_equal(sub.predicted, batch.predicted[:500]), \
    "a round trip through CSV changed the verdicts"
print("  a CSV round trip leaves every verdict unchanged")
assert csv_path.name in sub.narrative()

# An index column written by a careless to_csv must not become a feature.
with_index = WORK / "with_index.csv"
upload.head(50).to_csv(with_index, encoding="latin-1")          # index=True
reloaded = read_flows(with_index)
assert not any(str(c).startswith("Unnamed:") for c in reloaded.columns)
print(f"  an unnamed index column was dropped: {reloaded.shape[1]} columns kept")

# A file that is not flow features at all must be refused. Scoring it would answer a
# question the user never asked: with no features present, every row is the median row.
bogus = WORK / "bogus.csv"
pd.DataFrame({"timestamp": [1, 2], "src_ip": ["10.0.0.1", "10.0.0.2"]}).to_csv(
    bogus, index=False)
try:
    det.prepare(read_flows(bogus))
except ValueError as exc:
    print(f"  non-flow CSV refused: {str(exc)[:150]}")
    assert "CICFlowMeter" in str(exc), "the refusal does not say what was expected"
else:
    raise AssertionError("a file with no recognisable features was scored anyway")

# One recognisable feature is enough to proceed, because now the verdict is at least
# partly about the user's data.
known = det.feature_names[0]
one_col = pd.DataFrame({known: [80.0, 443.0, 22.0], "note": ["a", "b", "c"]})
thin = det.prepare(one_col, quiet=True)
print(f"  a single recognisable feature ({known}) is accepted: {thin.X.shape}, "
      f"{len(thin.missing)} filled from medians")
assert thin.X.shape == (3, det.n_features)
assert len(thin.missing) == det.n_features - 1, thin.missing

empty = WORK / "empty.csv"
empty.write_text("a,b,c\n")
try:
    det.predict(read_flows(empty))
except ValueError as exc:
    print(f"  header-only CSV: ValueError: {str(exc)[:80]}")
else:
    raise AssertionError("an empty table should have raised")

print("\nALL INFERENCE CHECKS PASSED")
stub_models.unregister()
