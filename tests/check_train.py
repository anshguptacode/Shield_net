"""Run the whole training pipeline end to end against the numpy test doubles.

This is the test that matters most: every module has its own checks, but nothing else
proves they fit together. It runs synthetic CICIDS2017-shaped data through load, clean,
split, preprocess, select, balance, tune, fit, evaluate, explain and persist, then
reloads the artifact and demands identical predictions.
"""
import json, shutil, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

# Derived from this file, never hardcoded: an absolute path here would make a copy of the
# project silently import the original, so the copy would pass while being untested.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests/_stubs"))
import stub_models; stub_models.register()

from shieldnet.config import Config
from shieldnet.logging_utils import configure_logging
from shieldnet.persist import ModelBundle
from shieldnet.train import (DataBundle, FeatureSpace, TrainingRun, build_feature_space,
                             plot_confusion, prepare_data, train, train_one)

configure_logging("INFO", force=True)

WORK = Path("/tmp/shieldnet_run")
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)

cfg = Config.load(
    seed=42, n_jobs=1, verbose=True,
    paths={"root": str(WORK)},
    data={"test_size": 0.2, "val_size": 0.1, "min_class_rows": 10},
    features={"n_features": 20, "methods": ["mutual_info", "chi2", "anova_f"],
              "ranking_sample_rows": 8000, "stability_runs": 3},
    balance={"strategy": "smote", "max_ratio": 0.25, "max_expansion": 20.0},
    tune={"enabled": True, "n_trials": 5, "cv_folds": 3, "timeout_seconds": 120,
          "metric": "log_loss"},
    train={"models": ["stub_softmax", "stub_centroid"], "primary": "stub_centroid",
           "selection_metric": "macro_f1", "shap_background_rows": 120,
           "shap_explain_rows": 1500},
)
print(cfg.describe())
assert cfg.tune.metric == "log_loss", "the neg_log_loss default was never valid"
assert cfg.train.selection_metric == "macro_f1"

# ------------------------------------------------------------------ 1. data
print("\n" + "=" * 78 + "\n=== 1. prepare_data ===\n" + "=" * 78)
t0 = time.perf_counter()
data = prepare_data(cfg, synthetic_rows=26_000, cache=True)
print(f"\nprepared in {time.perf_counter() - t0:.1f}s")
print(data.render())

assert isinstance(data, DataBundle)
n_total = len(data.split.y_train) + len(data.split.y_val) + len(data.split.y_test)
print(f"\n  total {n_total:,} rows across the three splits")
assert n_total == data.clean.rows_out, (n_total, data.clean.rows_out)
assert len(data.class_names) >= 10, data.class_names
assert data.split.X_train.shape[1] == len(data.feature_names)
# No flow may appear in two splits. split_frame resets each split's index, so index
# identity proves nothing; hash the row contents instead. clean_frame has already
# dropped exact duplicates, which makes a row hash a valid identity - and makes this a
# check on the de-duplication too.
def _row_hashes(frame):
    return set(pd.util.hash_pandas_object(frame, index=False).tolist())
h_tr, h_va, h_te = (_row_hashes(f) for f in
                    (data.split.X_train, data.split.X_val, data.split.X_test))
print(f"  distinct flows: train {len(h_tr):,} val {len(h_va):,} test {len(h_te):,}")
assert not (h_tr & h_va) and not (h_tr & h_te) and not (h_va & h_te), \
    "the same flow content appears in more than one split"
assert len(h_tr) == len(data.split.X_train), "duplicate rows survived cleaning"
print("  splits share no flow, and no split contains a duplicated flow")

# Both fractions are of the WHOLE frame, not of what the previous split left. Documented
# as 0.20/0.10/0.70 in three places, and every headline count in the README and in
# balance.py's docstring is derived from the 70% - so if this ever becomes "10% of the
# remainder", a lot of prose silently becomes wrong. Assert the contract, not the counts.
want_test, want_val = cfg.data.test_size, cfg.data.val_size
got = {k: v / n_total for k, v in data.split.sizes().items()}
print(f"  fractions of the whole: train {got['train']:.4f} val {got['val']:.4f} "
      f"test {got['test']:.4f}")
for name, want in (("test", want_test), ("val", want_val),
                   ("train", 1 - want_test - want_val)):
    assert abs(got[name] - want) < 0.005, (
        f"{name} is {got[name]:.4f} of the frame but {name}_size implies {want:.4f}. "
        "Both held-out fractions are documented as fractions of the whole.")

# The cache must be a real cache: second call cannot re-read anything.
t0 = time.perf_counter()
again = prepare_data(cfg, synthetic_rows=26_000, cache=True)
cached_secs = time.perf_counter() - t0
print(f"  cache hit in {cached_secs:.2f}s, key {again.cache_key}")
assert again.cache_key == data.cache_key
assert np.array_equal(again.split.y_test, data.split.y_test), "cache returned a different split"
assert cached_secs < 5.0, "the cache did not short-circuit the load"
# A config change that affects the split must invalidate it.
other = Config.load(paths={"root": str(WORK)}, data={"test_size": 0.3})
from shieldnet.train import _cache_key
assert _cache_key(other, source=None, synthetic_rows=26_000) != data.cache_key
print("  cache key responds to test_size but is keyed independently of the model list")

# ------------------------------------------------------------------ 2. feature space
print("\n" + "=" * 78 + "\n=== 2. build_feature_space ===\n" + "=" * 78)
space = build_feature_space(cfg, data)
print(space.render())

assert isinstance(space, FeatureSpace)
assert space.n_features == 20, space.n_features
assert space.preprocessor.feature_names_out_ == space.selected
assert space.preprocessor.n_features_in_ == 20, space.preprocessor.n_features_in_
assert space.X_train.shape == (len(space.y_train), 20)
assert space.X_val.shape[1] == 20 and space.X_test.shape[1] == 20
assert np.isfinite(space.X_train).all() and np.isfinite(space.X_test).all()
# medians must cover every raw feature, not just the selected ones: the manual-entry
# form needs a default for all 77.
assert len(space.raw_medians) >= 70, len(space.raw_medians)
assert all(f in space.raw_medians for f in space.selected)

print("\n  -- the refit claim: per-column statistics must be identical --")
from shieldnet.preprocess import Preprocessor
full = Preprocessor.fit(data.split.X_train, drop_constant=cfg.data.drop_constant,
                        clip_quantile=cfg.data.clip_quantile, scaler=cfg.data.scaler,
                        imputer=cfg.data.imputer)
cols = [full.feature_names_out_.index(f) for f in space.selected]
for attr in ("centre", "spread"):
    a = getattr(full.scaler, attr)[cols]
    b = getattr(space.preprocessor.scaler, attr)
    worst = float(np.abs(a - b).max())
    print(f"    scaler.{attr:<7} max |full[selected] - refit| = {worst:.3e}")
    assert worst < 1e-9, (attr, worst)
worst_med = float(np.abs(full.imputer.medians[cols] - space.preprocessor.imputer.medians).max())
print(f"    imputer medians max difference           = {worst_med:.3e}")
assert worst_med < 1e-9
print("  refitting on the subset is a projection, not a new estimate - as documented")

print("\n  -- balancing touched train only --")
print(space.balance.table().to_string(index=False))
assert len(space.y_val) == len(data.split.y_val)
assert len(space.y_test) == len(data.split.y_test)
assert len(space.y_fit) >= len(space.y_train)
before = np.bincount(space.y_train, minlength=space.n_classes)
after = np.bincount(space.y_fit, minlength=space.n_classes)
ratio_before = before.min() / before.max()
ratio_after = after.min() / after.max()
print(f"  min/max class ratio: {ratio_before:.5f} -> {ratio_after:.5f}")
assert ratio_after > ratio_before, "balancing made the imbalance no better"
# max_expansion must actually bind, not be quietly ignored
for i, (b, a) in enumerate(zip(before, after)):
    assert a <= b * cfg.balance.max_expansion + 1, \
        f"{space.class_names[i]} expanded {a / max(b, 1):.1f}x, above the cap"
print(f"  no class expanded beyond {cfg.balance.max_expansion:.0f}x")
# class weights come from the post-balance distribution
if space.class_weight:
    heaviest = max(space.class_weight, key=lambda k: space.class_weight[k])
    print(f"  heaviest residual weight: {space.class_names[heaviest]} "
          f"= {space.class_weight[heaviest]:.2f}")
    assert max(space.class_weight.values()) <= 50.0

# ------------------------------------------------------------------ 3. one model
print("\n" + "=" * 78 + "\n=== 3. train_one ===\n" + "=" * 78)
one = train_one("stub_softmax", space, cfg, tune=False,
                params={"learning_rate": 0.4, "epochs": 120})
assert one.ok, one.failed
print(one.val.render())
assert one.tune is None, "tuning ran despite tune=False"
assert one.params["epochs"] == 120
assert one.val.split == "validation" and one.test.split == "test"
assert one.rows_fit == len(space.y_fit)
print(f"\n  val macro F1 {one.val.macro_f1:.4f} | test macro F1 {one.test.macro_f1:.4f}")
print(f"  score('macro_f1') = {one.score('macro_f1'):+.4f}   "
      f"score('log_loss') = {one.score('log_loss'):+.4f}")
# log_loss is lower-is-better, so its score must be sign-flipped to rank consistently
assert one.score("macro_f1") == one.val.macro_f1
assert one.score("log_loss") == -one.val.log_loss, "direction not normalised"

print("\n  -- a model that cannot possibly fit --")
# The learning rate has to be absurd, not merely bad. SoftmaxModel only raises when the
# weights leave the floating-point range, and the growth per epoch is lr * l2: at
# lr=1e9, l2=1e-4 the weights multiply by 1e5 each step, so from a first step of order
# 1e9 they reach only ~1e254 after 50 epochs - large enough to score macro F1 0.0000 and
# log loss 34, small enough to stay finite. A model that trains to garbage without
# raising is a different failure mode, and one this assertion is not about.
broken = train_one("stub_softmax", space, cfg, tune=False,
                   params={"learning_rate": 1e200, "epochs": 50})
print(f"  captured: {broken.failed[:110]}")
assert not broken.ok and broken.failed, "a diverging fit was reported as success"
assert "FloatingPointError" in broken.failed
assert broken.val is None
print("  a failing model costs one leaderboard row, not the run")

# ------------------------------------------------------------------ 4. full run
print("\n" + "=" * 78 + "\n=== 4. train (full run, tuning on) ===\n" + "=" * 78)
run = train(cfg, data=data, space=space, tune=True, explain=True, save=True)

print("\n" + run.render())

assert isinstance(run, TrainingRun)
assert len(run.succeeded) == 2, [m.name for m in run.models]
assert run.best is not None and run.best.name in ("stub_softmax", "stub_centroid")
assert all(m.tune is not None for m in run.succeeded), "tuning did not run"
lb = run.leaderboard()
print("\n  leaderboard:\n" + lb.to_string(index=False))
assert list(lb["model"])[0] == run.best.name, "the leaderboard head is not the winner"
scores = [m.score("macro_f1") for m in
          sorted(run.succeeded, key=lambda m: -m.score("macro_f1"))]
assert scores == sorted(scores, reverse=True)
assert run.best.score("macro_f1") == max(m.score("macro_f1") for m in run.succeeded)
assert not run.skipped, run.skipped

print("\n  -- tuning searched the space it claims to --")
for m in run.succeeded:
    values = [t.value for t in m.tune.trials if not t.failed]
    print(f"    {m.name:<15} {len(m.tune.trials)} trials, "
          f"best {m.tune.metric}={m.tune.best_value:.4f} "
          f"(baseline {m.tune.baseline_value:.4f}), sampler={m.tune.sampler}")
    assert m.tune.metric == "log_loss"
    assert len(set(round(v, 6) for v in values)) > 1, \
        f"{m.name}: every trial scored the same, the search space is not connected"
    assert m.tune.best_value <= max(values) + 1e-9

print("\n  -- explanation --")
assert run.explanation is not None
print(f"    method: {run.explanation.method}")
print(f"    rows:   {run.explanation.rows_explained:,}")
print(f"    notes:  {run.explanation.notes}")
print("    top 8:  " + ", ".join(f"{n}" for n, _ in run.explanation.top_features(8)))
assert run.explanation.rows_explained <= cfg.train.shap_explain_rows
assert len(run.explanation.importance) == 20
assert set(run.explanation.feature_names) == set(space.selected)
assert any("overlap with the selection ranking" in n for n in run.explanation.notes)
assert len(run.samples) >= 8, len(run.samples)
print(f"    worked examples: {len(run.samples)} of {space.n_classes} classes")
seen = {s.predicted_class for s in run.samples}
print(f"    classes covered: {len(seen)}")

# ------------------------------------------------------------------ 5. artifact
print("\n" + "=" * 78 + "\n=== 5. the artifact ===\n" + "=" * 78)
artifacts = cfg.paths.resolve("artifacts")
reports = cfg.paths.resolve("reports")
figures = cfg.paths.resolve("figures")
for directory in (artifacts, reports, figures):
    for f in sorted(directory.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(WORK)}  {f.stat().st_size:,} bytes")

assert run.bundle_path is not None and run.bundle_path.exists()
manifest = json.loads((artifacts / "manifest.json").read_text())
print(f"\n  manifest: {manifest['model_name']}, {manifest['n_features']} features, "
      f"{manifest['n_classes']} classes, sha {manifest['bundle_sha256_12']}")
assert manifest["feature_names"] == space.selected
assert manifest["label_classes"] == space.class_names
assert manifest["metrics"]["macro_f1"] > 0
assert "per_class_recall" in manifest["metrics"]
assert manifest["metadata"]["seed"] == 42
assert manifest["metadata"]["top_features"], "no top features recorded"

restored = ModelBundle.restore(artifacts)
assert restored.feature_names == space.selected
assert restored.n_features == 20 and restored.n_classes == space.n_classes
a = restored.model.predict_proba(space.X_test)
b = run.best.model.predict_proba(space.X_test)
print(f"  reload drift over {len(space.X_test):,} test rows: {np.abs(a - b).max():.2e}")
assert np.abs(a - b).max() < 1e-12
assert not (artifacts / ".staging").exists(), \
    "the staging directory survived a successful save"

print("\n  -- a bundle that fails its round-trip is not installed --")
# The round-trip check is only worth having if it is a gate. Written straight into
# artifacts/ and checked afterwards, a failure would raise while the unshippable bundle
# sat exactly where the app looks for one - and having already overwritten the last good
# one. Force the failure by making the reload return a model that predicts the complement,
# then assert that what is on disk is still the bundle from the run above, byte for byte.
import shieldnet.train as train_mod                                       # noqa: E402
good_bytes = (artifacts / "bundle.joblib").read_bytes()
good_sha = manifest["bundle_sha256_12"]


class _Complement:
    def __init__(self, inner):
        self.inner = inner

    def predict_proba(self, X):
        return 1.0 - self.inner.predict_proba(X)


real_restore = ModelBundle.restore
ModelBundle.restore = classmethod(
    lambda cls, directory: type("Liar", (), {"model": _Complement(
        real_restore(directory).model)})())
try:
    train(cfg, data=data, space=space, tune=False, explain=False, save=True)
except RuntimeError as exc:
    print(f"    RuntimeError: {str(exc)[:150]}")
    assert "not installed" in str(exc), str(exc)
else:
    raise AssertionError("a bundle that predicts differently from the model was installed")
finally:
    ModelBundle.restore = real_restore
assert not (artifacts / ".staging").exists(), "the rejected bundle was left in .staging"
assert (artifacts / "bundle.joblib").read_bytes() == good_bytes, \
    "a rejected bundle replaced the one that had already been verified"
assert json.loads((artifacts / "manifest.json").read_text())["bundle_sha256_12"] == good_sha
print("    the run failed and artifacts/ still holds the bundle from before it")
assert train_mod.ModelBundle.restore is real_restore

print("\n  -- the shipped preprocessor really is the inference path --")
raw_test = data.split.X_test
shuffled = raw_test.sample(frac=1.0, axis=1, random_state=0)   # columns out of order
dropped = shuffled.drop(columns=[space.selected[0], shuffled.columns[-1]])
aligned = restored.scaler.transform(dropped, align=True)
print(f"    a {dropped.shape[1]}-column upload, columns shuffled, one selected feature "
      f"missing -> {aligned.shape}")
assert aligned.shape == (len(raw_test), 20)
assert np.isfinite(aligned).all()
# With the one missing column filled from the training median, predictions must be
# close to - but not identical to - the full-information ones.
p_partial = restored.model.predict_proba(aligned)
agree = float((p_partial.argmax(1) == b.argmax(1)).mean())
print(f"    agreement with the full-column prediction: {agree:.1%}")
assert agree > 0.60, agree
full_cols = restored.scaler.transform(raw_test[restored.feature_names], align=True)
assert np.abs(full_cols - space.X_test).max() < 1e-9, \
    "the bundled preprocessor disagrees with the one used at training time"
print("    with every column present it reproduces the training-time matrix exactly")

print("\n  -- validate() catches the mismatch that never raises on its own --")
from shieldnet.persist import BundleError
bad = ModelBundle(model_name="x", feature_names=space.selected[:10],
                  label_classes=space.class_names, scaler=space.preprocessor,
                  medians=space.raw_medians)
try:
    bad.validate()
except BundleError as exc:
    print(f"    BundleError: {str(exc)[:110]}")
else:
    raise AssertionError("a 20-column preprocessor passed a 10-feature bundle")

# ------------------------------------------------------------------ 6. reports
print("\n" + "=" * 78 + "\n=== 6. reports and figures ===\n" + "=" * 78)
summary = json.loads((reports / "run_summary.json").read_text())
print(f"  run_summary.json keys: {sorted(summary)}")
assert summary["best"] == run.best.name
assert summary["selection_metric"] == "macro_f1"
assert summary["figures"], "figures were recorded after the summary was written"
assert summary["data"]["sizes"]["train"] == len(space.y_train)
assert len(summary["models"]) == 2
assert summary["features"]["n_features"] == 20
assert summary["explanation"]["method"] == run.explanation.method
assert len(summary["samples"]) == len(run.samples)

text = (reports / "run_summary.txt").read_text()
assert "Total wall clock: " in text and "Total wall clock: 0" not in text
assert "Leaderboard" in text and "Selected model" in text
print(f"  run_summary.txt: {len(text):,} chars, ends with "
      f"{text.strip().splitlines()[-1]!r}")

ranking = pd.read_csv(reports / "feature_ranking.csv")
print(f"  feature_ranking.csv: {ranking.shape}, {int(ranking['selected'].sum())} selected")
assert int(ranking["selected"].sum()) == 20
imp = pd.read_csv(reports / "global_importance.csv")
assert abs(imp["share"].sum() - 1.0) < 1e-9
print(f"  global_importance.csv: {imp.shape}, share sums to 1")
for name in ("importance.png", "per_class_importance.png", "confusion.png"):
    size = (figures / name).stat().st_size
    print(f"  {name}: {size:,} bytes")
    assert size > 10_000, name
examples = (reports / "worked_examples.txt").read_text()
print(f"  worked_examples.txt: {len(examples):,} chars, "
      f"{examples.count('Worked example:')} examples")
assert examples.count("Worked example:") == len(run.samples)

print("\n  -- unnormalised confusion figure --")
p = plot_confusion(run.best.test, figures / "confusion_counts.png", normalise=False)
print(f"    {p.name}: {p.stat().st_size:,} bytes")
assert p.stat().st_size > 10_000

# ------------------------------------------------------------------ 7. select=primary
print("\n" + "=" * 78 + "\n=== 7. select='primary' overrides the leaderboard ===\n" + "=" * 78)
forced = train(cfg, data=data, space=space, models=["stub_softmax", "stub_centroid"],
               tune=False, explain=False, save=False, select="primary")
print(f"  auto would ship {run.best.name}; primary shipped {forced.best.name}")
assert forced.best.name == "stub_centroid"
assert forced.explanation is None and forced.bundle_path is None

print("\n  -- an unavailable model is skipped, not fatal --")
partial = train(cfg, data=data, space=space, models=["stub_softmax", "xgboost", "cnn1d"],
                tune=False, explain=False, save=False)
print(f"    trained: {[m.name for m in partial.succeeded]}")
print(f"    skipped: {partial.skipped}")
assert [m.name for m in partial.succeeded] == ["stub_softmax"]
assert set(partial.skipped) == {"xgboost", "cnn1d"}
assert all("not installed" in r for r in partial.skipped.values())

print("\n  -- nothing runnable at all is fatal, with the fix in the message --")
try:
    train(cfg, data=data, space=space, models=["xgboost", "lightgbm"], save=False)
except RuntimeError as exc:
    print(f"    RuntimeError: {str(exc).splitlines()[0]}")
    assert "logistic_regression" in str(exc)
else:
    raise AssertionError("a run with no usable models should have raised")

print("\n  -- a bad selection metric is caught before any model is fitted --")
bad_cfg = Config.load(paths={"root": str(WORK)},
                      train={"selection_metric": "f1_macro"})
try:
    train(bad_cfg, data=data, space=space, save=False)
except ValueError as exc:
    print(f"    ValueError: {str(exc)[:96]}")
else:
    raise AssertionError("an unknown selection metric was accepted")

print("\nALL TRAIN CHECKS PASSED")
stub_models.unregister()
