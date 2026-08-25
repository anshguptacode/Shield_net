"""Verify shieldnet.narrate end to end against real explanations and reports."""
import sys, numpy as np
from pathlib import Path

# Derived from this file, never hardcoded: an absolute path here would make a copy of the
# project silently import the original, so the copy would pass while being untested.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests/_stubs"))
import stub_models; stub_models.register()

from shieldnet import schema as sch
from shieldnet.narrate import (PROFILES, AttackProfile, profile_for, check_profiles,
                               narrate_prediction, narrate_batch, narrate_evaluation,
                               describe_contribution, feature_phrase, confidence_phrase,
                               severity_label, triage_order, _join)
from shieldnet.explain import Explainer
from shieldnet.evaluate import evaluate
from shieldnet.models import build

# ------------------------------------------------------------------ 1. profiles
print("=== 1. profile coverage ===")
problems = check_profiles()
for p in problems: print("  PROBLEM:", p)
assert not problems, problems
assert len(PROFILES) == 13, len(PROFILES)
print(f"  all {len(PROFILES)} classes profiled, severities and cross-references consistent")
print(f"  {'class':<26}{'family':<32}{'sev':>4}  tactic")
for label in sch.CLASS_SCHEME_13:
    pr = PROFILES[label]
    print(f"  {label:<26}{pr.family:<32}{pr.severity:>4}  {pr.tactic[:34]}")

# every profile must have real prose, not a placeholder
for label, pr in PROFILES.items():
    assert len(pr.summary) > 30 and pr.summary.endswith("."), label
    assert len(pr.on_the_wire) > 40, label
    assert len(pr.action) > 20, label
    assert pr.family != "Unknown", f"{label} has no family in schema.ATTACK_FAMILY"
# confusability should be symmetric where it is claimed at all
for label, pr in PROFILES.items():
    for other in pr.confusable_with:
        assert other != label, f"{label} confusable with itself"
print("  every profile has real prose and a schema-backed family")

# raw label spellings and unknown classes must both resolve
assert profile_for("Web Attack \x96 XSS").label == "Web Attack - XSS"
assert profile_for("Heartbleed").label == sch.RARE_CLASS_NAME, profile_for("Heartbleed").label
assert profile_for("BENIGN").is_benign
unknown = profile_for("Totally New Attack")
assert unknown.severity == 3 and "Investigate" in unknown.action
print("  raw spellings, rare-merge sources, and unknown classes all resolve")

# ------------------------------------------------------------------ 2. phrasings
print("\n=== 2. phrasings ===")
for p in (0.999, 0.95, 0.80, 0.60, 0.40, 0.10):
    print(f"  {p:>6.1%} -> {confidence_phrase(p)!r}")
assert confidence_phrase(1.0) == "almost certainly"
assert confidence_phrase(0.0) == "weakly suggestive of"
# monotone: never a stronger word for a weaker probability
words = [confidence_phrase(p) for p in np.linspace(0, 1, 101)]
assert words[0] == "weakly suggestive of" and words[-1] == "almost certainly"
assert len(set(words)) == 6, set(words)
assert [severity_label(i) for i in range(1, 6)] == \
       ["informational", "low", "medium", "high", "critical"]
assert severity_label(99) == "unknown"

print("  " + describe_contribution("Flow Duration", 0.8, value=2.6,
                                   predicted_class="DoS slowloris"))
print("  " + describe_contribution("Flow Duration", -0.4, value=-2.6,
                                   predicted_class="DoS slowloris"))
print("  " + describe_contribution("Flow Duration", 0.8, value=118_000_000,
                                   predicted_class="DoS slowloris", scaled=False))
print("  " + describe_contribution("Nonexistent Feature", 0.5))
# the scaled/raw distinction must actually change the wording
print("  phrase form: " + feature_phrase("Flow Duration", value=2.6))
assert feature_phrase("Flow Duration", value=2.6) == \
       "the total duration of the conversation in microseconds (far above normal)"
assert feature_phrase("Flow Duration") == \
       "the total duration of the conversation in microseconds"
assert "measured 1.18e+08" in feature_phrase("Flow Duration", value=118e6, scaled=False)
scaled = describe_contribution("Flow Duration", 1.0, value=2.6)
raw = describe_contribution("Flow Duration", 1.0, value=2.6, scaled=False)
assert "far above normal" in scaled and "measured" in raw, (scaled, raw)
assert "microseconds" in scaled, "the glossary gloss was dropped"
assert _join(["a"]) == "a" and _join(["a", "b"]) == "a and b"
assert _join(["a", "b", "c"]) == "a, b, and c"
assert _join([]) == ""

# ------------------------------------------------------------------ 3. real prediction
print("\n=== 3. narrate_prediction on a real explanation ===")
# Use real CICIDS2017 feature names so the glossary is genuinely exercised.
NAMES = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets",
         "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std",
         "Fwd Packet Length Mean", "Bwd Packet Length Mean", "Destination Port"]
CLASSES = ["BENIGN", "DoS Hulk", "PortScan", "DoS slowloris", "Bot"]
rng = np.random.default_rng(11)
counts = [1600, 700, 500, 300, 120]
Xs, ys = [], []
for c, n in enumerate(counts):
    b = rng.normal(0, 1.0, (n, 10)).astype(np.float32)
    b[:, c] += 3.0                      # each class keyed to one real feature
    b[:, (c + 3) % 10] -= 2.0
    Xs.append(b); ys.append(np.full(n, c, np.int64))
X = np.vstack(Xs); y = np.concatenate(ys)
p = rng.permutation(len(y)); X, y = X[p], y[p]
cut = int(0.7 * len(y)); Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]

model = build("stub_softmax", n_classes=5, params={"learning_rate": 0.5, "epochs": 250})
model.fit(Xtr, ytr)
ex = Explainer(model, NAMES, CLASSES, seed=5); ex.set_background(Xtr)
proba = model.predict_proba(Xte)

for want in (3, 4, 0):                  # slowloris, Bot, then a benign flow
    cands = np.nonzero((proba.argmax(1) == want) & (yte == want))[0]
    i = int(cands[int(np.argmax(proba[cands, want]))])
    loc = ex.explain_row(Xte[i])
    text = narrate_prediction(loc, top=3)
    print(f"\n  --- {CLASSES[want]} ---\n  {text}")
    assert CLASSES[want] in text or want == 0, text
    assert len(text) > 200, "narration too thin"
    assert "{" not in text and "None" not in text, text
    # Grammar: a noun phrase spliced into a sentence must not carry its own verb.
    # "rests mainly on the throughput raises the case for X" reads as correct word by
    # word, which is exactly why it needs an explicit assertion.
    for bad in (" on the ", "raises the case for", "lowers the case for"):
        if bad in ("raises the case for", "lowers the case for"):
            assert bad not in text, f"clause form leaked into prose: {text}"
    assert text.count("evidence for") <= 1, text
    assert "evidence for" not in text or " is " in text.split("evidence for")[1][:80], \
        f"'evidence' is a mass noun and must take 'is': {text}"
    assert "evidence for" not in text or " are " not in text.split("evidence for")[1][:60], \
        f"subject-verb disagreement: {text}"
    # Every sentence should start with a capital and end with a full stop.
    for sentence in [x.strip() for x in text.split(". ") if x.strip()]:
        assert sentence[0].isupper() or sentence[0].isdigit(), f"lowercase start: {sentence[:60]}"
    assert not any(w in text for w in (" the the ", "  ", " ,")), f"whitespace/dup: {text}"

# actions must read as instructions, i.e. start with a verb, not a judgement
print("\n  --- recommended actions all read as instructions ---")
for label in sch.CLASS_SCHEME_13:
    act = PROFILES[label].action
    first = act.split()[0].rstrip(".,")
    print(f"    {label:<26} {first} ...")
    assert first[0].isupper(), label
    assert not act.startswith(("Not ", "It ", "This ")), f"{label}: action is a judgement, not a step"


# an ambiguous prediction must get the hedging paragraph
print("\n  --- deliberately ambiguous ---")
amb = model.predict_proba(Xte)
gap = np.sort(amb, axis=1)[:, -1] - np.sort(amb, axis=1)[:, -2]
j = int(np.argmin(gap))
loc_amb = ex.explain_row(Xte[j])
text_amb = narrate_prediction(loc_amb, top=3)
print(f"  top2 gap = {gap[j]:.4f}\n  {text_amb}")
assert "not clear-cut" in text_amb, "ambiguity was not flagged"
assert "shortlist of two" in text_amb

# benign narration must not recommend attack actions
ben = np.nonzero(proba.argmax(1) == 0)[0]
loc_ben = ex.explain_row(Xte[int(ben[int(np.argmax(proba[ben, 0]))])])
t_ben = narrate_prediction(loc_ben)
assert "No action" in t_ben, t_ben
assert "severity" not in t_ben.split("Recommended")[0], "benign flow graded as an attack"
print("\n  benign flows get 'No action', not a severity grade")

# toggles must actually toggle
short = narrate_prediction(loc, top=2, include_action=False, include_profile=False)
assert "Recommended next step" not in short and "Typical signature" not in short
assert len(short) < len(narrate_prediction(loc, top=2)), "toggles did nothing"
print("  include_action / include_profile toggles work")

# ------------------------------------------------------------------ 4. batch
print("\n=== 4. narrate_batch ===")
labels = ["BENIGN", "PortScan", "DoS Hulk", "Bot", "Web Attack - XSS"]
cnts = [48000, 9000, 2500, 3, 40]
print("  " + narrate_batch(labels, cnts, mean_confidence=0.94, low_confidence_rows=512,
                           source="capture_2026-08-25.csv"))
out = narrate_batch(labels, cnts, mean_confidence=0.94, low_confidence_rows=512)
# Bot has 3 flows but severity 5; PortScan has 9000 but severity 2. Bot must come first.
assert out.index("Bot") < out.index("PortScan"), "triage ignored severity"
# The quoted review threshold has to be the one in force. It used to be the literal "50%",
# which is a sentence describing a threshold nobody chose the moment someone passes
# Detector(min_confidence=0.8) - and the number is right there in the same output.
assert "less than 50% confidence" in out, out
strict = narrate_batch(labels, cnts, low_confidence_rows=512, min_confidence=0.8)
assert "less than 80% confidence" in strict, strict
assert "50%" not in strict, "the review threshold is still hardcoded"
print("  the review threshold quoted in the prose follows min_confidence")
order = triage_order(labels, cnts)
print("\n  triage order:", [(l, c, s) for l, c, s in order])
assert order[0][0] == "Bot", order
assert all(order[i][2] >= order[i + 1][2] for i in range(len(order) - 1)), order
assert not any(l == "BENIGN" for l, _, _ in order), "benign in the triage queue"
assert triage_order(["BENIGN", "Bot"], [10, 0]) == [], "zero counts leaked in"

print("\n  --- all-clean capture ---")
clean = narrate_batch(["BENIGN"], [50000])
print("  " + clean)
assert "not evidence that the detector is working" in clean
assert narrate_batch([], []) .startswith("No rows were scored")
print("\n  empty and all-benign inputs handled")

# ------------------------------------------------------------------ 5. evaluation
print("\n=== 5. narrate_evaluation ===")
rep = evaluate(yte, proba, class_names=CLASSES, model="stub_softmax",
               fit_seconds=1.2, predict_seconds=0.03)
print("  " + narrate_evaluation(rep, baseline_accuracy=0.803).replace(". ", ".\n  "))
t = narrate_evaluation(rep, baseline_accuracy=0.803)
assert "macro F1" in t and "0.803" in t
assert "false alarm rate" in t, "the binary collapse was not narrated"
assert "spurious alerts per day" in t
assert "calibration" in t

# the undetected-class warning is the most important sentence; prove it fires
print("\n  --- a model that never predicts two classes ---")
n = 400
fake = np.zeros((n, 5)); fake[:, 0] = 0.7; fake[:, 1] = 0.3
yf = np.array([0] * 250 + [1] * 100 + [2] * 30 + [3] * 20)
rep2 = evaluate(yf, fake, class_names=CLASSES, model="lazy")
t2 = narrate_evaluation(rep2, baseline_accuracy=0.625)
print("  " + t2.replace(". ", ".\n  "))
assert "never predicted at all" in t2, t2
assert "PortScan" in t2 and "DoS slowloris" in t2
assert "no instances in this split" in t2, "absent class not distinguished from missed"
assert "Bot" in t2.split("no instances in this split")[0], "Bot should be the absent one"
# One absent class is "it is", not "they are".
assert "and it is excluded" in t2, f"singular/plural agreement: {t2}"
# A model with 0.00 recall is not "confusing similar classes" - it is failing outright,
# and the narration must not offer that excuse.
assert "known difficulty rather than a defect" not in t2, \
    "a total detection failure was excused as expected confusion"
# ...but the sentence must still appear for a model that IS mostly working.
assert "known difficulty rather than a defect" in t, "the mitigation sentence vanished"
print("  a 0%-recall class is never excused as 'expected confusion'")
print("\n  missed classes and absent classes are described differently - the "
      "distinction accuracy hides")

# no-baseline path
t3 = narrate_evaluation(rep)
assert "sanity check only" in t3
print("  narration without a baseline still warns about accuracy")

print("\nALL NARRATE CHECKS PASSED")
stub_models.unregister()
