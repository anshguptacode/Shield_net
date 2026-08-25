"""Verify shieldnet.explain against the numpy test doubles."""
import shutil, sys, numpy as np
from pathlib import Path

# Derived from this file, never hardcoded: an absolute path here would make a copy of the
# project silently import the original, so the copy would pass while being untested.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests/_stubs"))

import stub_models
stub_models.register()

from shieldnet.explain import (Explainer, permutation_importance, shap_available,
                               normalise_shap_values, GlobalExplanation)
from shieldnet.models import build

print("shap available:", shap_available())

# ------------------------------------------------------------------ synthetic data
# 25 features, 6 classes. Only features 0..5 carry signal; 6..9 are pure noise;
# 10..24 are copies of 0..5 plus noise (correlated decoys). If explain.py works, the
# importance ranking must put the signal features above the noise features.
rng = np.random.default_rng(7)
NAMES = [f"feat_{i:02d}" for i in range(25)]
CLASSES = ["BENIGN", "DoS Hulk", "PortScan", "Bot", "Web Attack - XSS", "Rare Attacks"]
counts = [2400, 900, 700, 260, 120, 40]

blocks_X, blocks_y = [], []
for c, n in enumerate(counts):
    base = np.zeros((n, 25), dtype=np.float32)
    centre = np.zeros(6); centre[c % 6] = 2.6; centre[(c + 2) % 6] = -1.7
    base[:, :6] = centre + rng.normal(0, 1.0, (n, 6))
    base[:, 6:10] = rng.normal(0, 1.0, (n, 4))              # noise
    base[:, 10:16] = base[:, :6] + rng.normal(0, 0.6, (n, 6))   # correlated
    base[:, 16:25] = rng.normal(0, 1.0, (n, 9))             # more noise
    blocks_X.append(base); blocks_y.append(np.full(n, c, dtype=np.int64))

X = np.vstack(blocks_X); y = np.concatenate(blocks_y)
perm = rng.permutation(len(y)); X, y = X[perm], y[perm]
cut = int(0.7 * len(y))
Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]
print(f"train {Xtr.shape} test {Xte.shape} classes {np.bincount(y)}")

model = build("stub_softmax", n_classes=6, params={"learning_rate": 0.6, "epochs": 300})
model.fit(Xtr, ytr)
from shieldnet.evaluate import evaluate
rep = evaluate(yte, model.predict_proba(Xte), class_names=CLASSES, model="stub_softmax")
print(f"baseline: acc={rep.accuracy:.4f} macro_f1={rep.macro_f1:.4f}")

# ------------------------------------------------------------------ 1. permutation
print("\n=== 1. permutation_importance ===")
imp, per_class, info = permutation_importance(
    model.predict_proba, Xte, yte, class_names=CLASSES, n_repeats=2,
    feature_names=NAMES, seed=1)
assert imp.shape == (25,), imp.shape
assert per_class.shape == (6, 25), per_class.shape
order = np.argsort(-imp)
print("top 8:", [(NAMES[i], round(float(imp[i]), 4)) for i in order[:8]])
print("baseline", round(info["baseline"], 4), "rows", info["rows_used"],
      "secs", round(info["seconds"], 2))
print("negative (useless) features:", info["negative_features"][:8])
signal = set(range(6)) | set(range(10, 16))
noise = set(range(6, 10)) | set(range(16, 25))
top6 = set(order[:6].tolist())
assert top6 <= signal, f"noise leaked into the top 6: {[NAMES[i] for i in top6 - signal]}"
mean_signal = imp[sorted(signal)].mean(); mean_noise = imp[sorted(noise)].mean()
print(f"mean importance: signal={mean_signal:.5f} noise={mean_noise:.5f}")
assert mean_signal > mean_noise * 3, (mean_signal, mean_noise)
assert (imp >= 0).all(), "clipping failed"

# ------------------------------------------------------------------ 2. global (fallback)
print("\n=== 2. global_explanation via fallback ===")
ex = Explainer(model, NAMES, CLASSES, seed=3)
ex.set_background(Xtr)
assert ex.background_.shape == (200, 25), ex.background_.shape
g = ex.global_explanation(Xte, yte, max_rows=1200, n_repeats=2)
print(g.render(top=10))
assert g.method.startswith("permutation"), g.method
assert g.rows_explained == 1200, g.rows_explained   # exact, not "about"
assert g.background_rows == 200
top_names = [n for n, _ in g.top_features(6)]
assert all(n in [NAMES[i] for i in signal] for n in top_names), top_names

print("\n-- signature_features --")
for cls, feats in g.signature_features(3).items():
    print(f"  {cls:<20} {feats}")
sig = g.signature_features(3)
# Each class was centred on feature (c % 6): that feature must be distinctive to it.
hits = sum(1 for c, cls in enumerate(CLASSES)
           if cls in sig and (f"feat_{c%6:02d}" in sig[cls]
                              or f"feat_{10+c%6:02d}" in sig[cls]))
print(f"classes whose planted feature is in its top-3 distinctive set: {hits}/6")
assert hits >= 4, hits

print("\n-- for_class --")
print("  DoS Hulk:", g.for_class("DoS Hulk", 4))
print("  index 2 :", g.for_class(2, 4))

print("\n-- frames --")
df = g.frame(); print(df.head(5).to_string(index=False))
assert abs(df["share"].sum() - 1.0) < 1e-9, df["share"].sum()
assert df["cumulative_share"].iloc[-1] > 0.999
pcf = g.per_class_frame(); assert pcf.shape == (6, 25), pcf.shape

print("\n-- agreement_with --")
# Pretend feature selection ranked the true signal features first.
ranking = [NAMES[i] for i in [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]] + \
          [NAMES[i] for i in sorted(noise)]
print(" ", {k: v for k, v in g.agreement_with(ranking).items() if "overlap" in k})
ag = g.agreement_with(ranking)
# 4, not 5: the model ranks feat_03 sixth while the pretend ranking put it fourth, so
# one of the five is genuinely absent. The point of the metric is to surface exactly that
# kind of disagreement, so demanding a perfect 5 would be testing the wrong thing.
assert ag["top5_overlap"] == 4, ag
assert ag["top10_overlap"] >= 8, ag
assert set(ag["top5_shared"]) <= set(ag["top10_shared"]), "nesting broken"

print("\n-- to_dict / json --")
import json
d = g.to_dict(top=8)
blob = json.dumps(d)
print("  keys:", sorted(d.keys()))
print("  json bytes:", len(blob))
assert "signature_features" in d and "per_class_top" in d

# ------------------------------------------------------------------ 3. local
print("\n=== 3. explain_row via occlusion ===")
# Pick a confidently-predicted non-benign row so the explanation is interesting.
proba_te = model.predict_proba(Xte)
cands = np.nonzero((proba_te.argmax(1) == yte) & (yte == 1))[0]
i = int(cands[int(np.argmax(proba_te[cands, 1]))])
loc = ex.explain_row(Xte[i], raw_values=Xte[i] * 3.0 + 100.0)
print(loc.render(top=8))
assert loc.predicted_class == "DoS Hulk", loc.predicted_class
assert loc.method.startswith("occlusion in log-odds")
# The prediction is saturated (100.0%), which is the normal case for a tuned model on
# this data. In probability space every contribution would round to 0.00000 and the app
# would render an empty explanation, so the log-odds scale must produce a real spread.
top_mag = loc.top(1)[0].magnitude
print(f"  saturated-prediction spread: top |contribution| = {top_mag:.3f}")
assert loc.confidence > 0.999, loc.confidence
assert top_mag > 1.0, f"occlusion collapsed on a confident prediction: {top_mag}"
assert loc.top(3)[2].magnitude > 0.05, "only one feature got any credit"
assert np.isfinite([c.contribution for c in loc.contributions]).all()
assert np.isfinite(loc.base_value)
assert len(loc.contributions) == 25
assert loc.runner_up and loc.runner_up != loc.predicted_class
assert loc.additivity_error is None          # not additive by design
sup = loc.supporting(3); opp = loc.opposing(3)
print("\n  supporting:", [(c.name, round(c.contribution, 4)) for c in sup])
print("  opposing:  ", [(c.name, round(c.contribution, 4)) for c in opp])
assert all(c.contribution > 0 for c in sup) and all(c.contribution < 0 for c in opp)
assert sup[0].direction == "increases" and opp[0].direction == "decreases"
# raw_value must be carried through, not silently dropped
assert loc.contributions[0].raw_value is not None
assert abs(loc.contributions[0].raw_value - (float(Xte[i][0]) * 3.0 + 100.0)) < 1e-4
lf = loc.frame(); assert len(lf) == 25
assert lf["contribution"].abs().is_monotonic_decreasing
print("\n  local to_dict keys:", sorted(loc.to_dict().keys()))
json.dumps(loc.to_dict())

print("\n-- why not BENIGN? (class_index=0) --")
alt = ex.explain_row(Xte[i], class_index=0)
print(f"  explaining BENIGN instead: confidence {alt.confidence:.2e}, "
      f"top push {alt.top(1)[0].name} {alt.top(1)[0].contribution:+.4f}")
assert alt.predicted_class == "BENIGN" and alt.predicted_index == 0

# ------------------------------------------------------------------ 4. figures
print("\n=== 4. figures ===")
# Its own directory, like every other check, so `make clean` has one thing to remove
# rather than two loose files it will miss.
WORK = Path("/tmp/shieldnet_explain")
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
p1 = g.plot(str(WORK / "fig_importance.png"), top=15)
p2 = g.plot_per_class(str(WORK / "fig_perclass.png"), top=10)
import os
for p in (p1, p2):
    assert p is not None and os.path.getsize(p) > 8000, p
    print(f"  {p} -> {os.path.getsize(p):,} bytes")

# ------------------------------------------------------------------ 5. shape normaliser
print("\n=== 5. normalise_shap_values ===")
N, F, K = 4, 25, 6
cases = {
    "list of (n,f) per class": [np.full((N, F), c) for c in range(K)],
    "(n,f,k) modern":          np.arange(N * F * K, dtype=float).reshape(N, F, K),
    "(n,f,1,k) keras channel": np.zeros((N, F, 1, K)),
    "(f,k) single row":        np.zeros((F, K)),
    "(n,f) binary":            np.zeros((N, F)),
    "(k,n,f) classes first":   np.zeros((K, N, F)),
    "(n,k,f) transposed":      np.zeros((N, K, F)),
}
for label, raw in cases.items():
    out = normalise_shap_values(raw, n_rows=N, n_features=F, n_classes=K)
    print(f"  {label:<26} -> {out.shape}")
    assert out.shape[1] == F, (label, out.shape)
    assert out.ndim == 3
# list case must keep class identity in the last axis
out = normalise_shap_values(cases["list of (n,f) per class"],
                            n_rows=N, n_features=F, n_classes=K)
assert out.shape == (N, F, K)
assert (out[0, 0, :] == np.arange(K)).all(), out[0, 0, :]
# 1-D single-output
out = normalise_shap_values(np.zeros(F), n_rows=1, n_features=F, n_classes=1)
assert out.shape == (1, F, 1)
# an Explanation-like wrapper
class FakeExplanation:
    def __init__(self, v): self.values = v
out = normalise_shap_values(FakeExplanation(np.zeros((N, F, K))),
                            n_rows=N, n_features=F, n_classes=K)
assert out.shape == (N, F, K)
# non-finite values must be scrubbed, not propagated
bad = np.zeros((N, F, K)); bad[0, 0, 0] = np.nan; bad[1, 2, 3] = np.inf
out = normalise_shap_values(bad, n_rows=N, n_features=F, n_classes=K)
assert np.isfinite(out).all()
print("  non-finite scrubbing: ok")
# genuinely un-interpretable input must raise
for raw in (np.zeros((3, 7)), np.zeros((2, 2, 2, 2, 2))):
    try:
        normalise_shap_values(raw, n_rows=N, n_features=F, n_classes=K)
    except ValueError as e:
        print(f"  rejected {raw.shape}: {str(e)[:60]}")
    else:
        raise AssertionError(f"{raw.shape} should have raised")

# ------------------------------------------------------------------ 6. guard rails
print("\n=== 6. guard rails ===")
def expect(fn, exc, label):
    try:
        fn()
    except exc as e:
        print(f"  {label}: {type(e).__name__}: {str(e)[:78]}")
    else:
        raise AssertionError(f"{label} did not raise")

expect(lambda: Explainer(model, NAMES, CLASSES).explain_row(Xte[0]),
       RuntimeError, "no background")
expect(lambda: Explainer(model, NAMES[:5], CLASSES).set_background(Xtr),
       ValueError, "wrong column count")
expect(lambda: ex.explain_row(Xte[0][:5]), ValueError, "short row")
expect(lambda: ex.global_explanation(Xte[:, :5], yte), ValueError, "narrow X")
expect(lambda: ex.global_explanation(Xte), ValueError, "no labels, no shap")
expect(lambda: g.for_class("Nonexistent"), KeyError, "unknown class")
expect(lambda: permutation_importance(model.predict_proba, Xte, yte[:10],
                                     class_names=CLASSES), ValueError, "y length")
no_pc = GlobalExplanation("m", NAMES, CLASSES, imp, None, "x", 1, 1)
expect(lambda: no_pc.for_class(0), ValueError, "per_class missing")
expect(lambda: no_pc.per_class_frame(), ValueError, "per_class_frame missing")
assert no_pc.signature_features() == {}

# ------------------------------------------------------------------ 7. additivity path
print("\n=== 7. verify_additivity without shap ===")
v = ex.verify_additivity(Xte)
print(" ", v)
assert v["checked"] is False and "not additive" in v["reason"]

# ------------------------------------------------------------------ 8. balanced bg
print("\n=== 8. class-balanced background ===")
ex2 = Explainer(model, NAMES, CLASSES, seed=5, max_background=120)
ex2.set_background(Xtr, ytr)
counts_bg = ex2.background_.shape[0]
print(f"  rows={counts_bg}  notes={ex2.notes}")
assert counts_bg <= 120 and ex2.notes
# every class must be represented, otherwise "average" excludes a class entirely
d2 = ((ex2.background_[:, None, :] - np.stack([Xtr[ytr == c].mean(0)
                                               for c in range(6)])[None]) ** 2).sum(2)
print("  nearest-centroid spread over background:", np.bincount(d2.argmin(1),
                                                                minlength=6))

# ------------------------------------------------------------------ 9. tree/deep routing
print("\n=== 9. explainer selection is guarded when shap is absent ===")
class FakeTree(stub_models.CentroidModel):
    is_tree = True
ft = FakeTree(n_classes=6); ft.fit(Xtr, ytr)
ex3 = Explainer(ft, NAMES, CLASSES); ex3.set_background(Xtr)
assert ex3._build_shap() is None          # shap missing -> None, not a crash
loc3 = ex3.explain_row(Xte[0])
print(f"  tree-flagged model fell back to: {loc3.method[:50]}")
g3 = ex3.global_explanation(Xte[:600], yte[:600], n_repeats=1)
print(f"  and globally to: {g3.method}")

# prefer_shap=False must skip the import attempt entirely
ex4 = Explainer(model, NAMES, CLASSES, prefer_shap=False); ex4.set_background(Xtr)
assert ex4._build_shap() is None
print("  prefer_shap=False honoured")

print("\nALL EXPLAIN CHECKS PASSED")
stub_models.unregister()
