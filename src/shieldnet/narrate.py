"""Turns model output into sentences an analyst can act on.

Why this module exists
----------------------
A per-class probability vector and a list of signed SHAP values are the *evidence*, not
the *finding*. The gap between them is where an intrusion detection system either earns
trust or gets ignored: "class 4, p=0.87, Bwd Packet Length Std=+0.31" tells a security
analyst nothing they can act on, whereas "very likely a slow-rate DoS attack; the flow
held a connection open far longer than normal while sending almost no data" tells them
what happened and what to do next.

So this module carries the domain knowledge needed to translate. It does three things:

* :func:`narrate_prediction` writes a paragraph about one flow, naming the attack,
  grading its severity, and citing the features that drove the decision in plain English.
* :func:`narrate_batch` summarises a whole file - what was found, what to look at first.
* :func:`narrate_evaluation` reads a metrics report aloud, including the parts that are
  easy to misread. Accuracy of 0.98 on this data can coexist with a class the model never
  detects at all, and a narration that omitted that would be actively misleading.

Where the domain knowledge lives, and why not in ``schema.py``
--------------------------------------------------------------
:mod:`shieldnet.schema` owns the *data contract*: exact column spellings, label variants,
the 13-class scheme. That module has to stay narrowly factual because every loader and
validator depends on it. Severity ratings and recommended actions are editorial judgement
about presentation, so they live here instead. :data:`PROFILES` is keyed by the canonical
labels from ``schema.CLASS_SCHEME_13``, and :func:`check_profiles` asserts the two stay in
step, so the split cannot silently rot.

Severity is a triage aid, not a risk assessment. It ranks how urgently a *confirmed*
detection of each class deserves attention, which is not the same as how dangerous the
technique is in the abstract; a port scan is near-harmless in itself but is graded 2
rather than 1 because it usually precedes something worse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import schema as sch
from .logging_utils import get_logger

log = get_logger(__name__)

__all__ = [
    "AttackProfile", "PROFILES", "profile_for", "check_profiles",
    "narrate_prediction", "narrate_batch", "narrate_evaluation",
    "describe_contribution", "feature_phrase", "confidence_phrase", "severity_label",
    "triage_order",
]


# ---------------------------------------------------------------------------
# attack profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AttackProfile:
    """What one class is, how it shows up in flow features, and what to do about it."""

    label: str
    summary: str                  # one sentence: what the attack is
    on_the_wire: str              # how it manifests in CICFlowMeter features
    severity: int                 # 1 (informational) to 5 (act immediately)
    action: str                   # what an analyst should do next
    tactic: str = ""              # rough MITRE ATT&CK tactic, for grouping
    confusable_with: Tuple[str, ...] = ()   # classes it is genuinely hard to separate

    @property
    def family(self) -> str:
        return sch.family_of(self.label)

    @property
    def is_benign(self) -> bool:
        return self.label == sch.BENIGN_LABEL


def _p(label: str, severity: int, summary: str, on_the_wire: str, action: str,
       tactic: str = "", confusable: Tuple[str, ...] = ()) -> AttackProfile:
    return AttackProfile(label, summary, on_the_wire, severity, action, tactic,
                         confusable)


#: Profile for every class in :data:`shieldnet.schema.CLASS_SCHEME_13`.
PROFILES: Dict[str, AttackProfile] = {
    sch.BENIGN_LABEL: _p(
        sch.BENIGN_LABEL, 1,
        "Normal traffic with no attack signature.",
        "Balanced forward and backward packet counts, gaps between packets consistent "
        "with human or ordinary application timing, and a duration proportionate to the "
        "volume of data moved.",
        "No action. Sample periodically to confirm the model is not suppressing real "
        "attacks as normal traffic.",
        tactic="-",
    ),
    "DoS Hulk": _p(
        "DoS Hulk", 4,
        "HTTP flood that overwhelms a web server with a high volume of obfuscated, "
        "hard-to-cache requests.",
        "Very high forward packet rate against a single destination port, short flow "
        "duration repeated many times, and tiny gaps between packets.",
        "Rate-limit or block the source, and check whether the target web server stayed "
        "responsive during the window.",
        tactic="Impact / Endpoint Denial of Service",
        confusable=("DDoS", "DoS GoldenEye"),
    ),
    "PortScan": _p(
        "PortScan", 2,
        "Reconnaissance sweep probing many ports to find services that are listening.",
        "Very short flows carrying almost no payload, often a single forward packet with "
        "no reply, repeated across many destination ports.",
        "Identify the source and look for follow-up traffic to any port that answered. A "
        "scan is not damaging in itself, but it is usually the first stage of an "
        "intrusion.",
        tactic="Reconnaissance / Active Scanning",
        confusable=("BENIGN",),
    ),
    "DDoS": _p(
        "DDoS", 5,
        "Distributed denial of service - many sources flooding one target at once.",
        "Sustained high packet and byte rates, large total forward volume, and flow "
        "statistics that look near-identical across many concurrent flows.",
        "Escalate immediately. Engage upstream filtering or a scrubbing service; "
        "blocking single sources will not be enough.",
        tactic="Impact / Network Denial of Service",
        confusable=("DoS Hulk",),
    ),
    "DoS GoldenEye": _p(
        "DoS GoldenEye", 4,
        "HTTP keep-alive flood that exhausts a server's connection pool.",
        "Moderate packet rate but unusually long flow durations with repeated small "
        "forward packets, holding sockets open rather than maximising throughput.",
        "Block the source and raise the server's connection limits or timeouts.",
        tactic="Impact / Endpoint Denial of Service",
        confusable=("DoS Hulk", "DoS Slowhttptest"),
    ),
    "FTP-Patator": _p(
        "FTP-Patator", 3,
        "Automated password-guessing attack against an FTP service.",
        "Many short, near-identical flows to port 21 in rapid succession, each with a "
        "small and highly regular payload size.",
        "Check the FTP server's authentication log for a successful login from the same "
        "source. Disable plaintext FTP if it is not required.",
        tactic="Credential Access / Brute Force",
        confusable=("SSH-Patator",),
    ),
    "SSH-Patator": _p(
        "SSH-Patator", 4,
        "Automated password-guessing attack against an SSH service.",
        "Repeated short flows to port 22 with consistent packet sizes, reflecting the "
        "fixed handshake of each failed authentication attempt.",
        "Check for a successful authentication from the source, then move SSH to key-only "
        "authentication and add rate limiting.",
        tactic="Credential Access / Brute Force",
        confusable=("FTP-Patator",),
    ),
    "DoS slowloris": _p(
        "DoS slowloris", 4,
        "Slow-rate attack that ties up server threads by sending partial HTTP requests "
        "very slowly.",
        "Long flow duration with a very low packet rate and large, irregular gaps "
        "between forward packets - the opposite profile to a flood.",
        "Block the source and put a reverse proxy in front of the server to buffer "
        "incomplete requests.",
        tactic="Impact / Endpoint Denial of Service",
        confusable=("DoS Slowhttptest",),
    ),
    "DoS Slowhttptest": _p(
        "DoS Slowhttptest", 4,
        "Slow-rate attack using drawn-out HTTP headers or bodies to exhaust server "
        "resources.",
        "Very long duration, minimal data transferred, and long idle periods within an "
        "otherwise active flow.",
        "Block the source and enforce request header and body timeouts.",
        tactic="Impact / Endpoint Denial of Service",
        confusable=("DoS slowloris", "DoS GoldenEye"),
    ),
    "Bot": _p(
        "Bot", 5,
        "Traffic from a compromised host communicating with a command-and-control "
        "server.",
        "Small, regular, periodic flows to an external destination - the beaconing "
        "interval often shows up as a strikingly low variance in inter-arrival time.",
        "Treat the internal host as compromised: isolate it, and hunt for the same "
        "beaconing pattern from other hosts.",
        tactic="Command and Control",
        confusable=("BENIGN",),
    ),
    "Web Attack - Brute Force": _p(
        "Web Attack - Brute Force", 3,
        "Repeated credential guessing against a web login form.",
        "Many similar POST-sized flows to port 80 or 443, with regular forward payload "
        "lengths and short durations.",
        "Check the application's login log for a success, and add rate limiting or "
        "CAPTCHA to the form.",
        tactic="Credential Access / Brute Force",
        confusable=("Web Attack - XSS", "BENIGN"),
    ),
    "Web Attack - XSS": _p(
        "Web Attack - XSS", 3,
        "Cross-site scripting attempt injecting script into a web application's input.",
        "Web-port flows whose forward payloads are larger and more variable than normal "
        "browsing, because the injected script inflates the request.",
        "Inspect the web server logs for the injected payload and confirm whether the "
        "application reflected it. Review output encoding.",
        tactic="Initial Access / Exploit Public-Facing Application",
        confusable=("Web Attack - Brute Force", "BENIGN"),
    ),
    sch.RARE_CLASS_NAME: _p(
        sch.RARE_CLASS_NAME, 5,
        "One of three attacks too rare in this dataset to learn separately - Heartbleed, "
        "Infiltration, or SQL Injection - merged into a single class.",
        "No single signature: Heartbleed shows an abnormal backward-to-forward byte "
        "ratio, Infiltration resembles ordinary internal traffic, and SQL Injection looks "
        "like a web request with an oversized query string.",
        "Investigate manually. This class is a flag for 'rare and serious', not an "
        "identification, so confirm which of the three it is before responding.",
        tactic="Multiple",
        confusable=("BENIGN",),
    ),
}


def profile_for(label: str) -> AttackProfile:
    """Profile for *label*, tolerating raw spellings and unknown classes."""
    canonical = sch.canonical_label(label)
    if canonical in PROFILES:
        return PROFILES[canonical]
    if label in PROFILES:
        return PROFILES[label]
    log.debug("no attack profile for %r; using a neutral placeholder", label)
    return AttackProfile(
        label=str(label),
        summary=f"Traffic classified as {label}.",
        on_the_wire="No profile is recorded for this class.",
        severity=3,
        action="Investigate manually.",
    )


def check_profiles() -> List[str]:
    """Report any mismatch between :data:`PROFILES` and the 13-class scheme.

    Called by ``scripts/doctor.py`` and by the test suite. A missing profile is not fatal
    at runtime - :func:`profile_for` degrades gracefully - but it means the app would show
    an empty explanation for a real class, which is worth catching at build time rather
    than in a demo.
    """
    problems = []
    for label in sch.CLASS_SCHEME_13:
        if label not in PROFILES:
            problems.append(f"no AttackProfile for class {label!r}")
    for label in PROFILES:
        if label not in sch.CLASS_SCHEME_13:
            problems.append(f"PROFILES has {label!r}, which is not in CLASS_SCHEME_13")
    for label, prof in PROFILES.items():
        if not 1 <= prof.severity <= 5:
            problems.append(f"{label}: severity {prof.severity} is outside 1-5")
        for other in prof.confusable_with:
            if other not in PROFILES:
                problems.append(f"{label}: confusable_with names unknown class {other!r}")
    return problems


# ---------------------------------------------------------------------------
# small phrasings
# ---------------------------------------------------------------------------

def confidence_phrase(p: float) -> str:
    """Words for a probability, so the app is not all bare decimals.

    The thresholds are deliberately conservative at the top: a model that is wrong 1 time
    in 20 should not have its output described as "certain", because an analyst who reads
    "certain" and finds a false positive stops believing the next one.
    """
    if p >= 0.99:
        return "almost certainly"
    if p >= 0.90:
        return "very likely"
    if p >= 0.70:
        return "likely"
    if p >= 0.50:
        return "probably"
    if p >= 0.30:
        return "possibly"
    return "weakly suggestive of"


def severity_label(severity: int) -> str:
    return {1: "informational", 2: "low", 3: "medium", 4: "high",
            5: "critical"}.get(int(severity), "unknown")


def feature_phrase(
    name: str, *, value: Optional[float] = None, scaled: bool = True
) -> str:
    """Noun phrase for one feature and its level: "the average client packet size (low)".

    Kept separate from :func:`describe_contribution` because the two are grammatically
    incompatible. A standalone line in the app wants a full clause with a verb ("X raises
    the case for Bot"), while a list inside a sentence needs a bare noun phrase - splicing
    the clause form into "the prediction rests on ..." produces "rests on the throughput
    raises the case for Bot", which is the kind of error that survives review because
    every individual piece looks correct.
    """
    gloss = sch.describe_feature(name)
    if value is None:
        return f"the {gloss}"
    if not scaled:
        return f"the {gloss} (measured {value:,.4g})"
    return f"the {gloss} ({_level(value)})"


def _level(value: float) -> str:
    """Describe a standardised value in words rather than standard deviations."""
    if value >= 2.0:
        return "far above normal"
    if value >= 0.75:
        return "above normal"
    if value <= -2.0:
        return "far below normal"
    if value <= -0.75:
        return "below normal"
    return "close to normal"


def describe_contribution(
    name: str,
    contribution: float,
    *,
    value: Optional[float] = None,
    predicted_class: str = "",
    scaled: bool = True,
) -> str:
    """One full clause explaining what a single feature did, in plain English.

    ``scaled`` tells the truth about units. Features reach the model standardised, so a
    value of 2.4 means "2.4 standard deviations above the training mean", not "2.4
    microseconds". Printing it as though it were a raw measurement is the single easiest
    way to make an explanation panel lie, so when only a scaled value is available the
    phrasing says "far above normal" rather than quoting a meaningless number.
    """
    direction = "raises" if contribution > 0 else "lowers"
    target = f" the case for {predicted_class}" if predicted_class else " the score"
    return f"{feature_phrase(name, value=value, scaled=scaled)} {direction}{target}"


# ---------------------------------------------------------------------------
# one prediction
# ---------------------------------------------------------------------------

def narrate_prediction(
    explanation: Any,
    *,
    top: int = 4,
    include_action: bool = True,
    include_profile: bool = True,
    raw_values: Optional[Sequence[float]] = None,
) -> str:
    """A paragraph about one flow, from a :class:`~shieldnet.explain.LocalExplanation`.

    Deliberately prose, not bullets: the point is that a reader who is not the author of
    the model can follow the reasoning from evidence to conclusion in one pass.
    """
    label = explanation.predicted_class
    prof = profile_for(label)
    conf = float(explanation.confidence)
    parts: List[str] = []

    if prof.is_benign:
        parts.append(f"This flow is {confidence_phrase(conf)} normal traffic "
                     f"({conf:.1%} confidence).")
    else:
        parts.append(
            f"This flow is {confidence_phrase(conf)} {label} - "
            f"{prof.family.lower()} activity rated {severity_label(prof.severity)} "
            f"severity - at {conf:.1%} confidence.")

    # A close runner-up is the single most useful caveat available, so it is stated
    # before the evidence rather than buried after it.
    runner = getattr(explanation, "runner_up", "")
    runner_p = float(getattr(explanation, "runner_up_confidence", 0.0) or 0.0)
    if runner and runner_p > 0.15 and runner != label:
        margin = conf - runner_p
        if margin < 0.25:
            parts.append(
                f"The decision is not clear-cut: {runner} scored {runner_p:.1%}, only "
                f"{margin:.1%} behind, so treat this as a shortlist of two rather than a "
                f"firm identification.")
            if runner in prof.confusable_with:
                parts.append(f"{label} and {runner} are known to be genuinely hard to "
                             f"separate from flow statistics alone.")

    if include_profile and not prof.is_benign:
        parts.append(prof.summary)

    drivers = explanation.top(top)
    if drivers and drivers[0].magnitude > 0:
        # Grouped into evidence for and evidence against, rather than one flat list.
        # An analyst reading a flat list has to check each sign individually; splitting
        # them makes the case and the counter-case legible at a glance, and it is the
        # counter-case that tells them how much to trust the label.
        for_it, against_it = [], []
        for i, c in enumerate(drivers):
            if c.magnitude <= 0:
                continue
            raw = c.raw_value if c.raw_value is not None else (
                raw_values[i] if raw_values is not None and i < len(raw_values) else None)
            phrase = feature_phrase(
                c.name, value=raw if raw is not None else c.value, scaled=raw is None)
            (for_it if c.contribution > 0 else against_it).append(phrase)

        subject = "normal traffic" if prof.is_benign else label
        if for_it:
            # "evidence" is a mass noun: it takes "is" however many items follow it.
            parts.append(f"The strongest evidence for {subject} is {_join(for_it)}.")
        if against_it:
            verb = "points" if len(against_it) == 1 else "point"
            parts.append(f"Against it, {_join(against_it)} {verb} elsewhere.")
    else:
        parts.append("No individual feature stands out as driving this prediction, "
                     "which usually means the decision came from a combination of many "
                     "weak signals rather than one clear indicator.")

    if include_profile and not prof.is_benign:
        parts.append(f"Typical signature: {prof.on_the_wire}")
    if include_action:
        parts.append(f"Recommended next step: {prof.action}")

    return " ".join(parts)


def _join(items: Sequence[str]) -> str:
    """Oxford-comma join, because these clauses end up in report prose."""
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


# ---------------------------------------------------------------------------
# a batch of predictions
# ---------------------------------------------------------------------------

def triage_order(labels: Iterable[str], counts: Iterable[int]) -> List[Tuple[str, int, int]]:
    """Sort detections by severity first, then volume - the order to work them in.

    Sorting by count alone buries a single Bot detection under ten thousand port-scan
    flows, and the Bot is the one that means a host is already compromised.
    """
    rows = []
    for label, count in zip(labels, counts):
        if int(count) <= 0:
            continue
        prof = profile_for(label)
        if prof.is_benign:
            continue
        rows.append((str(label), int(count), int(prof.severity)))
    rows.sort(key=lambda r: (-r[2], -r[1]))
    return rows


def narrate_batch(
    labels: Sequence[str],
    counts: Sequence[int],
    *,
    total: Optional[int] = None,
    mean_confidence: Optional[float] = None,
    low_confidence_rows: int = 0,
    min_confidence: float = 0.50,
    source: str = "the uploaded file",
) -> str:
    """Summarise a scored file: what was found, and what to look at first.

    *min_confidence* is only there to be quoted: the caller has already counted the rows
    below it. It is a parameter rather than the literal 50% because
    ``Detector(min_confidence=...)`` is settable per deployment, and a sentence that says
    "less than 50% confidence" under a detector configured at 0.8 describes a threshold
    nobody chose.
    """
    total = int(total if total is not None else sum(int(c) for c in counts))
    if total == 0:
        return f"No rows were scored from {source}."

    by_label = {str(l): int(c) for l, c in zip(labels, counts) if int(c) > 0}
    benign = by_label.get(sch.BENIGN_LABEL, 0)
    attacks = total - benign
    parts = [f"Scored {total:,} flow(s) from {source}."]

    if attacks == 0:
        parts.append("Every flow was classified as normal traffic. Note that this is the "
                     "expected result for a clean capture, and is not evidence that the "
                     "detector is working - score a known-malicious sample to confirm "
                     "that.")
    else:
        parts.append(f"{attacks:,} ({attacks / total:.1%}) were flagged as malicious "
                     f"across {len(by_label) - (1 if benign else 0)} attack class(es), "
                     f"and {benign:,} ({benign / total:.1%}) as normal.")
        ordered = triage_order(list(by_label), [by_label[k] for k in by_label])
        if ordered:
            worst = ordered[0]
            parts.append(
                f"Work them in this order: " + _join([
                    f"{label} ({count:,} flow(s), {severity_label(sev)})"
                    for label, count, sev in ordered[:5]]) + ".")
            parts.append(f"{worst[0]} is the priority - {profile_for(worst[0]).action}")

    if mean_confidence is not None:
        parts.append(f"Mean confidence across all rows was {mean_confidence:.1%}.")
    if low_confidence_rows > 0:
        parts.append(
            f"{low_confidence_rows:,} row(s) ({low_confidence_rows / total:.1%}) were "
            f"predicted with less than {min_confidence:.0%} confidence and deserve a "
            f"human look regardless of the label assigned - a low-confidence BENIGN is "
            f"not the same thing as a clean flow.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# a metrics report
# ---------------------------------------------------------------------------

def narrate_evaluation(report: Any, *, baseline_accuracy: Optional[float] = None) -> str:
    """Read an :class:`~shieldnet.evaluate.EvaluationReport` aloud, caveats included.

    This is the narration that most needs to exist, because the headline number on this
    dataset is the most misleading one. Accuracy is dominated by BENIGN; a model that
    detects nothing at all still scores around 80% on full CICIDS2017. So the narration
    leads with macro F1, states the majority-class baseline for comparison, and always
    names the classes that went undetected - the failure that accuracy hides.
    """
    parts: List[str] = []
    name = getattr(report, "model", "the model")
    split = getattr(report, "split", "test")
    n = int(getattr(report, "n_rows", 0))

    parts.append(
        f"On {n:,} held-out {split} flow(s), {name} reached a macro F1 of "
        f"{report.macro_f1:.4f} and an accuracy of {report.accuracy:.4f}.")

    if baseline_accuracy is not None:
        gap = report.accuracy - baseline_accuracy
        parts.append(
            f"Accuracy is the weaker of the two figures here: always predicting the "
            f"majority class would score {baseline_accuracy:.4f}, so the accuracy "
            f"headline represents an improvement of only {gap:.4f} over a model that "
            f"detects nothing. Macro F1 weights all {report.classes_evaluated} classes "
            f"equally and is the number to judge this model by.")
    else:
        parts.append(
            f"Because the classes are severely imbalanced, macro F1 - which weights all "
            f"{report.classes_evaluated} evaluated classes equally - is the meaningful "
            f"figure, and accuracy should be read as a sanity check only.")

    parts.append(
        f"Balanced accuracy, the mean per-class recall, was "
        f"{report.balanced_accuracy:.4f}, and Matthews correlation "
        f"{report.mcc:.4f}.")

    absent = list(getattr(report, "classes_absent", []) or [])
    never = list(getattr(report, "classes_never_predicted", []) or [])
    if never:
        parts.append(
            f"{len(never)} class(es) were never predicted at all - "
            f"{_join(never)} - which means every instance of them was missed. Any macro "
            f"average that includes them is being dragged down by a detection rate of "
            f"zero, and no amount of accuracy compensates for an attack the system "
            f"cannot see.")
    if absent:
        plural = len(absent) > 1
        parts.append(
            f"{_join(absent)} had no instances in this split, so no per-class score "
            f"could be computed and {'they are' if plural else 'it is'} excluded from "
            f"the macro averages rather than counted as zero.")

    worst = report.worst_classes(3)
    if worst:
        described = []
        for cm in worst:
            bit = f"{cm.name} (recall {cm.recall:.2f} on {cm.support:,} flow(s)"
            if cm.confused_with:
                bit += f", most often mistaken for {cm.confused_with}"
            described.append(bit + ")")
        parts.append("The weakest classes were " + _join(described) + ".")
        for cm in worst:
            # Only offer the "known-confusable" mitigation when the class is actually
            # being detected some of the time. At a recall of zero the model is not
            # confusing two similar classes, it is failing to predict the class at all,
            # and describing that as "expected rather than a defect" would excuse the
            # single worst outcome the system can produce.
            if cm.confused_with and cm.recall >= 0.2:
                prof = profile_for(cm.name)
                if cm.confused_with in prof.confusable_with:
                    parts.append(
                        f"The {cm.name}/{cm.confused_with} confusion is a known "
                        f"difficulty rather than a defect: the two produce very similar "
                        f"flow statistics.")
                    break

    binary = getattr(report, "binary", None)
    if binary is not None:
        parts.append(
            f"Collapsed to the attack-versus-normal decision that a deployed sensor "
            f"actually makes, the model detected {binary.recall:.2%} of all attacks with "
            f"a false alarm rate of {binary.false_alarm_rate:.2%}. On a link carrying a "
            f"million flows a day, that false alarm rate implies roughly "
            f"{binary.false_alarm_rate * 1_000_000:,.0f} spurious alerts per day, which "
            f"is the figure that decides whether an analyst team can live with this "
            f"model.")

    cal = getattr(report, "calibration_error", None)
    if cal is not None and np.isfinite(cal):
        if cal < 0.05:
            verdict, reading = "well calibrated", "trusted as a probability"
        elif cal < 0.15:
            verdict, reading = ("somewhat overconfident",
                                "read as a ranking rather than a true probability")
        else:
            verdict, reading = ("poorly calibrated",
                                "used only to order alerts, not as a probability")
        parts.append(
            f"Expected calibration error was {cal:.4f}, meaning the model is {verdict}: "
            f"its stated confidence should be {reading}.")
    return " ".join(parts)
