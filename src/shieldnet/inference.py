"""Scoring a trained bundle against traffic that did not come from the pipeline.

Training code may assume its input is clean, because :func:`shieldnet.preprocess.clean_frame`
ran first. Inference code may assume nothing. A CSV dropped into the app can have its
columns in any order, can be missing some, can carry extras, can spell the headers with
the leading spaces CICFlowMeter actually emits, can store ``Flow Bytes/s`` as the string
``"Infinity"``, and can contain the duplicated ``Fwd Header Length.1`` column that the
original release shipped with. Every one of those has to produce a prediction rather than
a traceback.

The rule that shapes this whole module: **inference never drops a row.** Training drops
duplicates and unusable rows because it is estimating a distribution. Here, row *i* of
the output must be the verdict on row *i* of the upload - an analyst matching alerts back
to their capture cannot work with a result set that silently lost 500 flows. So bad cells
are repaired (inf to NaN to training median) rather than excised, and the repairs are
reported instead of hidden.

    from shieldnet.inference import Detector
    det = Detector.load("artifacts")
    batch = det.predict(frame)
    print(batch.narrative())
    batch.frame().to_csv("verdicts.csv", index=False)
"""

from __future__ import annotations

import io
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import schema as sch
from .explain import Explainer, LocalExplanation
from .logging_utils import get_logger, human_duration
from .narrate import narrate_batch, narrate_prediction, profile_for, triage_order
from .persist import BundleError, ModelBundle

log = get_logger(__name__)

__all__ = [
    "Detector", "PreparedInput", "PredictionBatch", "Prediction",
    "read_flows", "DEFAULT_MIN_CONFIDENCE",
]

#: Below this top-class probability a verdict is marked ``review`` rather than trusted.
#:
#: Not a tuned number and not presented as one. It is the point at which the model's own
#: stated confidence is low enough that an analyst should look at the flow before acting,
#: and it exists so that "the model was only 31% sure" is visible in the output instead of
#: being flattened into a bare class name. Callers override it per deployment.
DEFAULT_MIN_CONFIDENCE = 0.50

#: Rows scored per chunk in :meth:`Detector.predict`. Keeps peak memory bounded when
#: someone uploads a full day of traffic: the probability matrix alone is
#: ``rows * n_classes * 8`` bytes, so a chunk costs 200,000 * 13 * 8 = 20.8 MB where
#: scoring all 2,830,743 CICIDS2017 flows in one array would cost 294 MB - before the
#: feature matrix, which is wider.
DEFAULT_CHUNK_ROWS = 200_000


# ---------------------------------------------------------------------------
# reading whatever the user actually has
# ---------------------------------------------------------------------------

def read_flows(
    source: os.PathLike | str | io.IOBase,
    *,
    encoding: str = "latin-1",
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Read a flow-feature table from a path, file object or Streamlit upload.

    Accepts CSV (optionally gzipped) and Parquet. ``latin-1`` is the default for the
    same reason it is in :mod:`shieldnet.schema`: the CICIDS2017 headers contain a 0x96
    en-dash byte that is not valid UTF-8, so a UTF-8 read of the genuine article raises
    ``UnicodeDecodeError`` on the *header row* - the least informative place possible.
    """
    name = getattr(source, "name", str(source))
    suffix = Path(str(name)).suffix.lower()

    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(source)
        if nrows is not None:
            frame = frame.head(nrows)
        return frame

    read_kwargs: Dict[str, Any] = {
        "low_memory": False,
        "skipinitialspace": True,
        "nrows": nrows,
    }
    try:
        frame = pd.read_csv(source, encoding=encoding, **read_kwargs)
    except UnicodeDecodeError:
        # A caller passed something genuinely UTF-8 that latin-1 could not read. Rare
        # (latin-1 decodes every byte sequence) but possible via a wrapped text stream.
        if hasattr(source, "seek"):
            source.seek(0)
        frame = pd.read_csv(source, encoding="utf-8", **read_kwargs)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"{name} contains no data. Export the flow table from CICFlowMeter again, "
            "or check that the file finished uploading."
        ) from exc

    # A DataFrame index written by `to_csv()` without index=False comes back as an
    # unnamed column. It is not a feature and it is not an error worth reporting.
    junk = [c for c in frame.columns if str(c).startswith("Unnamed:")]
    if junk:
        frame = frame.drop(columns=junk)
        log.debug("dropped %d unnamed index column(s)", len(junk))
    return frame


# ---------------------------------------------------------------------------
# prepared input
# ---------------------------------------------------------------------------

@dataclass
class PreparedInput:
    """A user upload turned into a model-ready matrix, with the repairs itemised."""

    frame: pd.DataFrame                     #: canonical column names, numeric dtypes
    X: np.ndarray                           #: scaled, finite, ``(rows, n_features)``
    rows: int
    n_expected: int = 0                     #: features the model wanted, for the message
    #: Every column the upload actually carried, canonicalised, including the label and
    #: the ones the model ignores. ``frame`` holds only the model's own features, so this
    #: is what error messages must quote back at the user - they are looking for a column
    #: they typed, not for one the model selected.
    columns: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    extra: List[str] = field(default_factory=list)
    coerced: Dict[str, int] = field(default_factory=dict)
    non_finite: Dict[str, int] = field(default_factory=dict)
    duplicates_merged: List[str] = field(default_factory=list)
    y_true: Optional[np.ndarray] = None     #: encoded labels when the file carried them
    label_names: Optional[List[str]] = None  #: raw label strings, canonicalised
    unknown_labels: Dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def has_labels(self) -> bool:
        return self.y_true is not None

    @property
    def repaired_cells(self) -> int:
        return sum(self.coerced.values()) + sum(self.non_finite.values())

    def warnings(self) -> List[str]:
        """Things that changed the numbers, worst first. Empty means a clean upload.

        The bar for appearing here is that a reader who ignored the line could
        misread the results. Merely *interesting* observations go to :meth:`notes`,
        because a warning that fires on every correct run is a warning nobody reads.
        """
        out: List[str] = []
        if self.missing:
            shown = ", ".join(self.missing[:6])
            more = f" (+{len(self.missing) - 6} more)" if len(self.missing) > 6 else ""
            out.append(
                f"{len(self.missing)} of the {self.n_expected or len(self.missing)} "
                f"features the model expects were absent and filled with the training "
                f"median: {shown}{more}. Predictions are still produced, but treat them "
                "as weaker evidence - the model is reasoning about columns it cannot see."
            )
        if self.coerced:
            worst = sorted(self.coerced.items(), key=lambda kv: -kv[1])[:4]
            out.append(
                "non-numeric text was found in " + ", ".join(
                    f"{name} ({count:,} cell(s))" for name, count in worst
                ) + " and treated as missing."
            )
        if self.non_finite:
            worst = sorted(self.non_finite.items(), key=lambda kv: -kv[1])[:4]
            out.append(
                "infinite values in " + ", ".join(
                    f"{name} ({count:,})" for name, count in worst
                ) + " were replaced by the training median. This is normal for "
                "Flow Bytes/s and Flow Packets/s on zero-duration flows."
            )
        if self.duplicates_merged:
            out.append(
                f"duplicate column(s) {', '.join(self.duplicates_merged)} appeared more "
                "than once after header normalisation; the first occurrence was kept."
            )
        if self.unknown_labels:
            out.append(
                "the Label column contains value(s) the model was not trained on: "
                + ", ".join(f"{k} ({v:,})" for k, v in
                            sorted(self.unknown_labels.items(), key=lambda kv: -kv[1])[:5])
                + ". Those rows are excluded from accuracy figures but still scored."
            )
        return out

    def notes(self) -> List[str]:
        """True but unalarming observations about the upload.

        Unused columns used to be reported as a warning, which meant every correct run
        of a real CICFlowMeter export opened with ``63 column(s) were not used by this
        model.`` Feature selection keeps a subset by design, so that state is not a
        problem being flagged - it is the design being described, and putting it at
        warning level taught the reader to skim past the lines that do matter.
        """
        out: List[str] = []
        if self.extra:
            shown = ", ".join(self.extra[:4])
            more = f", +{len(self.extra) - 4} more" if len(self.extra) > 4 else ""
            out.append(
                f"{len(self.extra)} of the file's column(s) are not used by this model "
                f"({shown}{more}); feature selection kept {self.X.shape[1]}. They are "
                "left untouched in the output so you can read them beside the verdict."
            )
        return out

    def render(self) -> str:
        lines = [f"{self.rows:,} row(s) prepared in {human_duration(self.seconds)}",
                 f"matrix {self.X.shape[0]:,} x {self.X.shape[1]}"]
        if self.has_labels:
            lines.append(f"ground-truth labels present ({len(set(self.label_names or []))} class(es))")
        for w in self.warnings():
            lines.append("  ! " + w)
        if not self.warnings():
            lines.append("  every expected feature present, no repairs needed")
        for n in self.notes():
            lines.append("  - " + n)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"rows": self.rows, "missing": self.missing, "extra": len(self.extra),
                "coerced": self.coerced, "non_finite": self.non_finite,
                "duplicates_merged": self.duplicates_merged,
                "has_labels": self.has_labels, "seconds": round(self.seconds, 3)}


# ---------------------------------------------------------------------------
# predictions
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    """One flow's verdict, with everything the UI needs to justify it."""

    row: int
    predicted_index: int
    predicted_class: str
    confidence: float
    runner_up: str
    runner_up_confidence: float
    attack_probability: float
    is_attack: bool
    severity: int
    status: str                             #: ``"ok"`` | ``"review"``
    probabilities: Optional[np.ndarray] = None
    explanation: Optional[LocalExplanation] = None
    narrative: str = ""
    true_class: str = ""

    @property
    def correct(self) -> Optional[bool]:
        """``None`` when the input had no label, otherwise whether we got it right."""
        if not self.true_class:
            return None
        return self.true_class == self.predicted_class

    def top_classes(self, n: int = 3) -> List[Tuple[str, float]]:
        if self.probabilities is None:
            return [(self.predicted_class, self.confidence)]
        order = np.argsort(self.probabilities)[::-1][:n]
        return [(self._name(int(i)), float(self.probabilities[int(i)])) for i in order]

    _names: List[str] = field(default_factory=list, repr=False)

    def _name(self, i: int) -> str:
        return self._names[i] if 0 <= i < len(self._names) else f"class {i}"

    def to_dict(self) -> Dict[str, Any]:
        out = {"row": self.row, "predicted_class": self.predicted_class,
               "confidence": round(self.confidence, 6),
               "runner_up": self.runner_up,
               "runner_up_confidence": round(self.runner_up_confidence, 6),
               "attack_probability": round(self.attack_probability, 6),
               "is_attack": self.is_attack, "severity": self.severity,
               "status": self.status}
        if self.true_class:
            out["true_class"] = self.true_class
            out["correct"] = self.correct
        if self.explanation is not None:
            out["explanation"] = self.explanation.to_dict()
        if self.narrative:
            out["narrative"] = self.narrative
        return out


@dataclass
class PredictionBatch:
    """Vectorised verdicts for a whole upload.

    Deliberately array-of-struct-free: a two-million-row capture would need two million
    :class:`Prediction` objects, which is roughly a gigabyte of Python overhead for data
    that fits in 200 MB of numpy. Individual rows are materialised on demand through
    :meth:`prediction`.
    """

    model_name: str
    class_names: List[str]
    predicted: np.ndarray                   #: int codes, ``(rows,)``
    confidence: np.ndarray                  #: float, ``(rows,)``
    attack_probability: np.ndarray          #: ``1 - P(BENIGN)``
    runner_up: np.ndarray                   #: int codes
    runner_up_confidence: np.ndarray
    proba: Optional[np.ndarray] = None      #: ``(rows, n_classes)`` when retained
    y_true: Optional[np.ndarray] = None
    source: str = "the uploaded file"
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    attack_threshold: float = 0.5
    seconds: float = 0.0
    prepared: Optional[PreparedInput] = field(default=None, repr=False)

    # -- basics --------------------------------------------------------------

    def __len__(self) -> int:
        return int(self.predicted.shape[0])

    @property
    def labels(self) -> np.ndarray:
        """Predicted class *names*, ``(rows,)`` of dtype object."""
        names = np.asarray(self.class_names, dtype=object)
        return names[self.predicted]

    @property
    def benign_index(self) -> int:
        try:
            return self.class_names.index(sch.BENIGN_LABEL)
        except ValueError:
            return -1

    @property
    def is_attack(self) -> np.ndarray:
        """The binary decision a sensor actually makes.

        Note this is ``P(attack) >= threshold``, *not* ``argmax != BENIGN``. They differ:
        a flow can be 45% benign and split the remaining 55% across four attack classes,
        so the arg-max is BENIGN while the probability of *some* attack is the majority.
        At the default threshold of 0.5 the two agree often but not always, and the
        threshold is the knob that trades detection against false alarms.
        """
        return self.attack_probability >= self.attack_threshold

    @property
    def flagged(self) -> np.ndarray:
        """Rows whose top probability is below :attr:`min_confidence`."""
        return self.confidence < self.min_confidence

    @property
    def status(self) -> np.ndarray:
        return np.where(self.flagged, "review", "ok").astype(object)

    # -- per-row -------------------------------------------------------------

    def prediction(self, row: int, *, explainer: Optional[Explainer] = None,
                   X: Optional[np.ndarray] = None,
                   raw_values: Optional[Sequence[float]] = None,
                   narrate: bool = False) -> Prediction:
        """Materialise one row as a :class:`Prediction`, optionally explained."""
        i = int(row)
        idx = int(self.predicted[i])
        label = self.class_names[idx]
        explanation = None
        if explainer is not None and X is not None:
            explanation = explainer.explain_row(X[i], raw_values=raw_values)
        pred = Prediction(
            row=i, predicted_index=idx, predicted_class=label,
            confidence=float(self.confidence[i]),
            runner_up=self.class_names[int(self.runner_up[i])],
            runner_up_confidence=float(self.runner_up_confidence[i]),
            attack_probability=float(self.attack_probability[i]),
            is_attack=bool(self.attack_probability[i] >= self.attack_threshold),
            severity=0 if label == sch.BENIGN_LABEL else profile_for(label).severity,
            status="review" if self.confidence[i] < self.min_confidence else "ok",
            probabilities=None if self.proba is None else self.proba[i],
            explanation=explanation,
            true_class="" if self.y_true is None else self.class_names[int(self.y_true[i])],
            _names=self.class_names,
        )
        if narrate and explanation is not None:
            pred.narrative = narrate_prediction(explanation, raw_values=raw_values)
        return pred

    # -- aggregates ----------------------------------------------------------

    def counts(self) -> "pd.Series":
        """Predicted-class counts, ordered by the model's class order."""
        raw = np.bincount(self.predicted, minlength=len(self.class_names))
        return pd.Series(raw, index=self.class_names, name="flows")

    def summary(self) -> Dict[str, Any]:
        counts = self.counts()
        attacks = int(self.is_attack.sum())
        labelled_attacks = int((self.predicted != self.benign_index).sum())
        return {
            "model": self.model_name,
            "rows": len(self),
            # Two honest counts, because there are two questions. attack_flows answers
            # "how many flows have at least a `attack_threshold` chance of being some
            # attack"; labelled_attack_flows answers "how many were labelled as a
            # specific attack class". They differ whenever probability is spread thinly
            # across several attack classes, and the class table below counts labels, so
            # showing only the first number makes the table look wrong.
            "attack_flows": attacks,
            "benign_flows": len(self) - attacks,
            "attack_share": attacks / max(len(self), 1),
            "labelled_attack_flows": labelled_attacks,
            "threshold_only_attacks": attacks - labelled_attacks,
            "attack_threshold": self.attack_threshold,
            "distinct_classes": int((counts > 0).sum()),
            "mean_confidence": float(self.confidence.mean()) if len(self) else 0.0,
            "median_confidence": float(np.median(self.confidence)) if len(self) else 0.0,
            "low_confidence_rows": int(self.flagged.sum()),
            "counts": {k: int(v) for k, v in counts.items() if v},
            "seconds": round(self.seconds, 3),
            "rows_per_second": (len(self) / self.seconds) if self.seconds > 0 else None,
        }

    def triage(self) -> List[Tuple[str, int, int]]:
        """``(class, count, severity)`` for detected attacks, most urgent first."""
        counts = self.counts()
        return triage_order(list(counts.index), [int(v) for v in counts.values])

    def narrative(self) -> str:
        counts = self.counts()
        present = [(k, int(v)) for k, v in counts.items() if v]
        text = narrate_batch(
            [k for k, _ in present], [v for _, v in present],
            total=len(self),
            mean_confidence=float(self.confidence.mean()) if len(self) else None,
            low_confidence_rows=int(self.flagged.sum()),
            min_confidence=self.min_confidence,
            source=self.source,
        )
        # narrate_batch counts labels. The dashboard headline counts probability. Say so,
        # rather than letting a reader find two different attack totals and distrust both.
        gap = int(self.is_attack.sum()) - int((self.predicted != self.benign_index).sum())
        if gap > 0:
            text += (
                f" A further {gap:,} flow(s) carry a benign label but at least a "
                f"{self.attack_threshold:.0%} combined chance of being some attack - "
                "their probability is split across several attack classes without any "
                "one leading, which is the pattern to expect from a genuinely novel "
                "attack, so they are counted as attacks in the headline figure."
            )
        elif gap < 0:
            text += (
                f" {abs(gap):,} flow(s) carry an attack label yet fall below the "
                f"{self.attack_threshold:.0%} threshold, so the headline figure excludes "
                "them; raise the threshold to suppress them from the queue as well."
            )
        return text

    # -- output --------------------------------------------------------------

    def frame(self, *, probabilities: bool = False, top_k: int = 0) -> pd.DataFrame:
        """Tabular verdicts, one row per input row, ready for CSV download."""
        data: Dict[str, Any] = {
            "row": np.arange(len(self)),
            "prediction": self.labels,
            "confidence": np.round(self.confidence, 6),
            "attack_probability": np.round(self.attack_probability, 6),
            "is_attack": self.is_attack,
            "severity": [0 if l == sch.BENIGN_LABEL else profile_for(l).severity
                         for l in self.labels],
            "runner_up": np.asarray(self.class_names, dtype=object)[self.runner_up],
            "runner_up_confidence": np.round(self.runner_up_confidence, 6),
            "status": self.status,
        }
        if self.y_true is not None:
            names = np.asarray(self.class_names, dtype=object)
            data["true_class"] = names[self.y_true]
            data["correct"] = self.predicted == self.y_true
        out = pd.DataFrame(data)

        if self.proba is not None and top_k > 0:
            # Alternatives are what an analyst reads when the top class looks wrong, so
            # rank 2..k are worth columns of their own even though they are derivable.
            order = np.argsort(self.proba, axis=1)[:, ::-1]
            names = np.asarray(self.class_names, dtype=object)
            for k in range(1, min(top_k, len(self.class_names))):
                col = order[:, k]
                out[f"alt{k}_class"] = names[col]
                out[f"alt{k}_probability"] = np.round(
                    self.proba[np.arange(len(self)), col], 6)
        if self.proba is not None and probabilities:
            for j, name in enumerate(self.class_names):
                out[f"p({name})"] = np.round(self.proba[:, j], 6)
        return out

    def to_dict(self) -> Dict[str, Any]:
        out = self.summary()
        out["triage"] = [{"class": c, "flows": n, "severity": s} for c, n, s in self.triage()]
        if self.prepared is not None:
            out["input"] = self.prepared.to_dict()
        return out

    def render(self, *, top: int = 15) -> str:
        s = self.summary()
        lines = [
            f"{self.model_name} scored {s['rows']:,} flow(s) in "
            f"{human_duration(self.seconds)}"
            + (f" ({s['rows_per_second']:,.0f} rows/s)" if s["rows_per_second"] else ""),
            f"  P(attack) >= {self.attack_threshold:.2f}: {s['attack_flows']:,} "
            f"({s['attack_share']:.2%})   below: {s['benign_flows']:,}   "
            f"low confidence {s['low_confidence_rows']:,}",
        ]
        if s["threshold_only_attacks"]:
            lines.append(
                f"  the class table counts labels, so it shows "
                f"{s['labelled_attack_flows']:,} attack(s): "
                f"{s['threshold_only_attacks']:+,} row(s) differ between the two"
            )
        lines.append("")
        counts = self.counts()
        counts = counts[counts > 0].sort_values(ascending=False).head(top)
        width = max((len(str(i)) for i in counts.index), default=10)
        lines.append(f"  {'class':<{width}}  {'flows':>9}  {'share':>7}  sev  mean conf")
        lines.append("  " + "-" * (width + 34))
        for name, n in counts.items():
            mask = self.labels == name
            sev = 0 if name == sch.BENIGN_LABEL else profile_for(str(name)).severity
            lines.append(f"  {str(name):<{width}}  {int(n):>9,}  {n / len(self):>6.2%}  "
                         f"{sev:>3}  {self.confidence[mask].mean():>9.3f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# the detector
# ---------------------------------------------------------------------------

class Detector:
    """A loaded bundle plus the input-repair logic that makes it usable.

    Construct with :meth:`load`. The instance is read-only and safe to cache for the
    lifetime of a Streamlit session; the only mutable state is the lazily-built
    :class:`~shieldnet.explain.Explainer` and its background sample.
    """

    def __init__(self, bundle: ModelBundle, *, artifacts: Optional[Path] = None,
                 min_confidence: float = DEFAULT_MIN_CONFIDENCE,
                 attack_threshold: float = 0.5) -> None:
        self.bundle = bundle.validate()
        self.artifacts = Path(artifacts) if artifacts else None
        self.min_confidence = float(min_confidence)
        self.attack_threshold = float(attack_threshold)
        self._explainer: Optional[Explainer] = None
        self._background: Optional[np.ndarray] = None
        if self.bundle.model is None:
            raise BundleError(
                "the bundle carries no fitted model. If this is a deep model, make sure "
                "deep_model.keras sits next to bundle.joblib.")

    # -- construction --------------------------------------------------------

    @classmethod
    def load(cls, artifacts: os.PathLike | str = "artifacts", **kwargs: Any) -> "Detector":
        """Restore a bundle from disk. Raises :class:`BundleError` with the fix."""
        path = Path(artifacts)
        bundle = ModelBundle.restore(path)
        det = cls(bundle, artifacts=path if path.is_dir() else path.parent, **kwargs)
        log.info("loaded %s (%d features, %d classes) from %s",
                 bundle.model_name, bundle.n_features, bundle.n_classes, path)
        return det

    # -- introspection -------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self.bundle.model_name

    @property
    def feature_names(self) -> List[str]:
        return list(self.bundle.feature_names)

    @property
    def class_names(self) -> List[str]:
        return list(self.bundle.label_classes)

    @property
    def n_features(self) -> int:
        return self.bundle.n_features

    @property
    def n_classes(self) -> int:
        return self.bundle.n_classes

    @property
    def metrics(self) -> Dict[str, Any]:
        return dict(self.bundle.metrics)

    @property
    def raw_feature_names(self) -> List[str]:
        """Every feature this bundle can accept as *input*, in canonical order.

        Not the same as :attr:`feature_names`. The shipped preprocessor was refitted on
        the selected subset, so its ``feature_names_in_`` lists only those columns - but
        ``align=True`` will happily take a full 77-column CICFlowMeter export, use the
        selected ones and ignore the rest, and the bundle carries a median for every
        canonical feature so the manual-entry form can default any of them. The honest
        answer to "what may I send?" is therefore the canonical set the bundle has
        medians for, which is what this returns.
        """
        medians = self.bundle.medians or {}
        known = [name for name in sch.CANONICAL_FEATURES if name in medians]
        # Anything the bundle knows a median for but the schema does not list (a future
        # CICFlowMeter column) still belongs in the answer.
        extra = sorted(set(medians) - set(sch.CANONICAL_FEATURES) - {sch.LABEL_COLUMN})
        return (known + extra) or list(self.feature_names)

    def describe(self) -> str:
        meta = self.bundle.metadata or {}
        m = self.bundle.metrics or {}
        bits = [f"{self.model_name}: {self.n_features} features, {self.n_classes} classes"]
        if m.get("macro_f1") is not None:
            bits.append(f"test macro F1 {float(m['macro_f1']):.4f}")
        if m.get("accuracy") is not None:
            bits.append(f"accuracy {float(m['accuracy']):.4f}")
        if m.get("false_alarm_rate") is not None:
            bits.append(f"false alarm rate {float(m['false_alarm_rate']):.2%}")
        # `started_at` is the key the trainer actually writes. This read used to ask for
        # "trained_at", so it was never once true and the line never once printed - a
        # `.get` against a key nobody writes fails silently and looks like working code.
        stamp = meta.get("started_at") or meta.get("trained_at")
        if stamp:
            bits.append(f"trained {stamp}")
        # Carried into the one-line summary because that string is what the app puts in
        # its sidebar, and a synthetic artifact must never be mistaken for a CICIDS2017
        # result no matter which surface the reader is looking at.
        if meta.get("synthetic"):
            bits.append("SYNTHETIC DATA - numbers describe the generator")
        return "  |  ".join(bits)

    def template_row(self) -> Dict[str, float]:
        """Median value for every feature: the starting point for manual entry."""
        medians = dict(self.bundle.medians)
        return {name: float(medians.get(name, 0.0)) for name in self.raw_feature_names}

    def manual_fields(self) -> List[Dict[str, Any]]:
        """Spec for the app's single-flow form: the human-reasonable features only."""
        medians = self.bundle.medians
        selected = set(self.feature_names)
        out = []
        for name, low, high, default in sch.MANUAL_ENTRY_FEATURES:
            out.append({
                "name": name,
                "min": low,
                "max": high,
                "default": float(medians.get(name, default)),
                "demo": float(default),
                "help": sch.describe_feature(name),
                "used_by_model": name in selected,
            })
        return out

    # -- input preparation ---------------------------------------------------

    def prepare(self, data: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]],
                *, label_column: Optional[str] = sch.LABEL_COLUMN,
                quiet: bool = False) -> PreparedInput:
        """Canonicalise, repair and scale an arbitrary input table.

        Accepts a DataFrame, a single mapping (the manual-entry form), or a sequence of
        mappings. Never drops a row.
        """
        started = time.perf_counter()
        frame = _as_frame(data)
        if frame.empty:
            raise ValueError("no rows to score - the input table is empty.")

        frame, duplicates = _canonicalise_columns(frame)
        uploaded_columns = [str(c) for c in frame.columns]

        y_true, label_names, unknown = None, None, {}
        if label_column and label_column in frame.columns:
            y_true, label_names, unknown = self._encode_labels(frame[label_column])
            frame = frame.drop(columns=[label_column])

        scaler = self.bundle.scaler
        expected = list(getattr(scaler, "feature_names_in_", self.feature_names)) \
            if scaler is not None else list(self.feature_names)

        # Coerce only the columns the model consumes. Running to_numeric over the whole
        # upload would spend real time on columns that cannot affect a single prediction,
        # and - worse - would report them: a free-text "analyst_note" column produced a
        # warning about 2,801 non-numeric cells, which reads like a data disaster and is
        # in fact a note field doing its job. Unused columns are left exactly as uploaded
        # so the app can still display them beside the verdict.
        keep = [c for c in frame.columns if c in set(expected)]
        extra = sorted(set(map(str, frame.columns)) - set(expected))
        numeric, coerced = _coerce_numeric(frame[keep])
        non_finite = {c: int(np.isinf(numeric[c].to_numpy()).sum())
                      for c in numeric.columns}
        non_finite = {k: v for k, v in non_finite.items() if v}
        missing = [c for c in expected if c not in numeric.columns]

        # Repairing an upload is right up to the point where there is nothing left to
        # repair. With zero recognisable features every row would be scored as the
        # training median row, so every verdict would be identical and none of them would
        # be about the user's data - a confident answer to a question never asked. That is
        # the one case where refusing beats repairing.
        if numeric.shape[1] == 0:
            raise ValueError(
                f"none of the {len(expected)} feature(s) {self.model_name} needs are "
                f"present in this file. It has {len(uploaded_columns)} column(s) "
                f"({', '.join(uploaded_columns[:5])}{' ...' if len(uploaded_columns) > 5 else ''}), "
                f"but the model expects CICFlowMeter columns such as "
                f"{', '.join(list(expected)[:4])}. Export the capture with CICFlowMeter, "
                "or check that the header row was not lost."
            )

        if scaler is None:
            # A model that needs no scaling still needs alignment, so do it by hand.
            aligned = numeric.reindex(columns=self.feature_names)
            for name in missing:
                aligned[name] = self.bundle.medians.get(name, 0.0)
            values = aligned.to_numpy(dtype=np.float64)
            values[~np.isfinite(values)] = np.nan
            fill = np.array([self.bundle.medians.get(n, 0.0) for n in self.feature_names])
            idx = np.nonzero(np.isnan(values))
            values[idx] = fill[idx[1]]
            X = values
        else:
            X = scaler.transform(numeric, align=True)

        prepared = PreparedInput(
            frame=numeric, X=np.asarray(X, dtype=np.float64), rows=len(numeric),
            n_expected=len(expected), columns=uploaded_columns,
            missing=sorted(missing), extra=extra, coerced=coerced,
            non_finite=non_finite, duplicates_merged=duplicates,
            y_true=y_true, label_names=label_names, unknown_labels=unknown,
            seconds=time.perf_counter() - started,
        )
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"prepared {X.shape[1]} feature column(s) but {self.model_name} expects "
                f"{self.n_features}. This means the bundle's preprocessor and its "
                "feature list disagree, which validate() should have caught - the "
                "artifact is corrupt; retrain."
            )
        if not quiet:
            for warning in prepared.warnings():
                log.warning("%s", warning)
            for note in prepared.notes():
                log.info("%s", note)
        return prepared

    def _encode_labels(self, column: "pd.Series") -> Tuple[
            Optional[np.ndarray], List[str], Dict[str, int]]:
        """Map a Label column onto the model's class order.

        Applies the same canonicalisation and rare-class merge as training, so a file
        labelled with the raw ``Web Attack \x96 XSS`` spelling or with ``Heartbleed``
        lines up with the 13-class scheme the model was fitted on.
        """
        canonical = [sch.canonical_label(str(v)) for v in column.tolist()]
        lookup = {name: i for i, name in enumerate(self.class_names)}
        codes = np.full(len(canonical), -1, dtype=np.int64)
        unknown: Dict[str, int] = {}
        for i, name in enumerate(canonical):
            j = lookup.get(name)
            if j is None:
                unknown[name] = unknown.get(name, 0) + 1
            else:
                codes[i] = j
        if (codes >= 0).sum() == 0:
            log.warning("a Label column was present but none of its values match the "
                        "model's classes; ignoring it")
            return None, canonical, unknown
        return codes, canonical, unknown

    # -- scoring -------------------------------------------------------------

    def predict(self, data: Any, *, keep_proba: bool = True,
                chunk_rows: int = DEFAULT_CHUNK_ROWS,
                source: str = "the uploaded file",
                prepared: Optional[PreparedInput] = None,
                quiet: bool = False) -> PredictionBatch:
        """Score every row. Returns arrays, not objects - see :class:`PredictionBatch`."""
        prep = prepared if prepared is not None else self.prepare(data, quiet=quiet)
        X = prep.X
        n = X.shape[0]
        started = time.perf_counter()

        proba_chunks: List[np.ndarray] = []
        top1 = np.empty(n, dtype=np.int64)
        top1_p = np.empty(n, dtype=np.float64)
        top2 = np.empty(n, dtype=np.int64)
        top2_p = np.empty(n, dtype=np.float64)
        attack_p = np.empty(n, dtype=np.float64)
        benign = self.class_names.index(sch.BENIGN_LABEL) \
            if sch.BENIGN_LABEL in self.class_names else -1

        for start in range(0, n, max(chunk_rows, 1)):
            stop = min(start + chunk_rows, n)
            block = np.asarray(self.bundle.model.predict_proba(X[start:stop]),
                               dtype=np.float64)
            if block.shape != (stop - start, self.n_classes):
                raise RuntimeError(
                    f"the model returned probabilities shaped {block.shape}, expected "
                    f"{(stop - start, self.n_classes)}. A wrapper is misreporting its "
                    "class count."
                )
            # Two arg-sorts would be wasteful; partition for the top two only.
            order = np.argpartition(-block, min(1, self.n_classes - 1), axis=1)
            first = order[:, 0].copy()
            second = order[:, 1].copy() if self.n_classes > 1 else first
            rows = np.arange(stop - start)
            swap = block[rows, second] > block[rows, first]
            first[swap], second[swap] = second[swap].copy(), first[swap].copy()

            top1[start:stop] = first
            top1_p[start:stop] = block[rows, first]
            top2[start:stop] = second
            top2_p[start:stop] = block[rows, second]
            attack_p[start:stop] = (1.0 - block[:, benign]) if benign >= 0 else 1.0
            if keep_proba:
                proba_chunks.append(block)
            if n > chunk_rows and not quiet:
                log.info("scored %s/%s row(s)", f"{stop:,}", f"{n:,}")

        elapsed = time.perf_counter() - started
        batch = PredictionBatch(
            model_name=self.model_name, class_names=self.class_names,
            predicted=top1, confidence=top1_p, attack_probability=attack_p,
            runner_up=top2, runner_up_confidence=top2_p,
            proba=np.vstack(proba_chunks) if keep_proba and proba_chunks else None,
            y_true=prep.y_true, source=source,
            min_confidence=self.min_confidence,
            attack_threshold=self.attack_threshold,
            seconds=elapsed, prepared=prep,
        )
        if not quiet:
            log.info("%s scored %s row(s) in %s - %s attack(s), %s low confidence",
                     self.model_name, f"{n:,}", human_duration(elapsed),
                     f"{int(batch.is_attack.sum()):,}", f"{int(batch.flagged.sum()):,}")
        return batch

    def predict_one(self, values: Mapping[str, Any], *, explain: bool = True,
                    narrate: bool = True, top: int = 4) -> Prediction:
        """Score a single flow from the manual-entry form, explained and narrated.

        Unsupplied features are taken from :meth:`template_row` - the training medians.
        Filling them here rather than letting ``prepare`` treat them as absent is the
        same arithmetic, but it says something different: a form with four fields typed
        in is not a damaged upload, it is a deliberate what-if against an otherwise
        typical flow, and it should not raise the "the model is reasoning about columns
        it cannot see" warning that a real truncated capture deserves.
        """
        row = self.template_row()
        supplied = {}
        for key, value in dict(values).items():
            name = sch.normalise_column(str(key))
            supplied[name] = value
        row.update(supplied)
        prep = self.prepare(row, quiet=True)
        batch = self.predict(None, prepared=prep, quiet=True)
        untouched = [n for n in self.feature_names if n not in supplied]
        if untouched:
            log.debug("%d of %d model feature(s) left at their training median",
                      len(untouched), self.n_features)
        explainer = self.explainer() if explain else None
        raw = self._raw_values(prep)
        pred = batch.prediction(0, explainer=explainer, X=prep.X, raw_values=raw,
                                narrate=narrate and explain)
        if narrate and pred.explanation is not None and not pred.narrative:
            pred.narrative = narrate_prediction(pred.explanation, top=top, raw_values=raw)
        return pred

    def explain(self, prepared: PreparedInput, row: int,
                *, batch: Optional[PredictionBatch] = None) -> LocalExplanation:
        """SHAP (or occlusion) attribution for one already-prepared row."""
        return self.explainer().explain_row(
            prepared.X[row], raw_values=self._raw_values(prepared, row))

    def raw_values(self, prepared: PreparedInput,
                   row: Optional[int] = None) -> Optional[List[float]]:
        """Pre-scaling feature values, for readable narration. ``None`` if unrecoverable.

        Public because the app needs them: an explanation that says "Flow Duration = 1.83"
        is arithmetic about z-scores, and the analyst is trying to match the row against a
        capture where it reads 96,000,000. Callers outside this module should use this
        rather than the underscored implementation.
        """
        return self._raw_values(prepared, row)

    def inspect(self, prepared: PreparedInput, batch: PredictionBatch, row: int,
                *, narrate: bool = True, background_rows: int = 200) -> Prediction:
        """One row of an already-scored batch, explained and narrated.

        Exists so that the app does not have to assemble four pieces (an explainer with a
        background, the scaled matrix, the unscaled values, the batch) in the right order
        every time a user clicks a row - getting that wrong produces attributions measured
        against the wrong reference point, which look plausible and are wrong.

        The background is taken from the upload itself. That is a better reference than
        the shipped medians: "why is this flow an attack" is a question about how it
        differs from the *other traffic in this capture*, and with a real distribution
        behind it the contributions stop being measured against a single average row.
        """
        n = min(int(background_rows), prepared.X.shape[0])
        explainer = self.explainer(background=prepared.X[:n] if n > 0 else None)
        return batch.prediction(int(row), explainer=explainer, X=prepared.X,
                                raw_values=self._raw_values(prepared, int(row)),
                                narrate=narrate)

    def _raw_values(self, prepared: PreparedInput,
                    row: Optional[int] = None) -> Optional[List[float]]:
        """Pre-scaling values for the selected features, for readable narration.

        The scaled matrix is what the model sees, but "Flow Duration measured 1.18e+08"
        is what a human can check against their capture, and the two differ by the
        training mean and standard deviation. Recovering them by inverse-transforming is
        exact but goes through the clipper, so a winsorised outlier comes back as the
        clip bound rather than its original value - that is the honest number anyway,
        since the clip bound is what the model actually saw.
        """
        scaler = getattr(self.bundle.scaler, "scaler", None)
        if scaler is None or not hasattr(scaler, "inverse_transform"):
            return None
        block = prepared.X if row is None else prepared.X[int(row)][None, :]
        try:
            back = scaler.inverse_transform(block)
        except Exception as exc:                            # noqa: BLE001
            log.debug("inverse_transform failed (%s); narrating in scaled units", exc)
            return None
        return [float(v) for v in np.asarray(back)[0]]

    # -- explanation ---------------------------------------------------------

    def explainer(self, background: Optional[np.ndarray] = None) -> Explainer:
        """Lazily built explainer with a background sample.

        The background is where SHAP's "compared to what?" comes from. A bundle does not
        carry training rows - it would double the artifact size and leak traffic - so the
        first call without an explicit background falls back to the per-feature training
        medians repeated into a single row. That is a legitimate reference point (it is
        what ``align`` already uses for absent columns) but it collapses the background
        distribution to its centre, which makes attributions less informative than a real
        sample. Callers with data in hand should pass it.
        """
        if self._explainer is None:
            self._explainer = Explainer(
                self.bundle.model, self.feature_names, self.class_names,
                preprocessor=self.bundle.scaler,
            )
        if background is not None:
            self._explainer.set_background(background)
            self._background = np.asarray(background)
        elif self._explainer.background_ is None:
            medians = np.array([[self.bundle.medians.get(n, 0.0)
                                 for n in self.feature_names]], dtype=np.float64)
            scaler = self.bundle.scaler
            centre = (scaler.transform(pd.DataFrame(medians, columns=self.feature_names))
                      if scaler is not None else medians)
            self._explainer.set_background(centre)
            self._explainer.notes.append(
                "background is the training median row only - no sample of real traffic "
                "was shipped with the model, so contributions are measured against an "
                "average flow rather than a distribution")
            log.info("explainer background defaulted to the training median row; pass "
                     "real rows for sharper attributions")
        return self._explainer

    # -- evaluation ----------------------------------------------------------

    def evaluate(self, data: Any, *, label_column: str = sch.LABEL_COLUMN,
                 fpr_budget: Optional[float] = 0.01, quiet: bool = False) -> Any:
        """Full :class:`~shieldnet.evaluate.EvaluationReport` for a labelled file.

        Rows whose label is not one of the model's classes are excluded from the metrics
        (they have no correct answer available) but were still scored, and the count is
        reported so the exclusion is never silent.
        """
        from .evaluate import evaluate as _evaluate

        prep = self.prepare(data, label_column=label_column, quiet=quiet)
        if prep.y_true is None:
            raise ValueError(
                f"evaluation needs a '{label_column}' column whose values match the "
                f"model's classes. The file carried {len(prep.columns)} column(s), "
                f"starting {prep.columns[:6]}."
            )
        batch = self.predict(None, prepared=prep, quiet=quiet)
        keep = prep.y_true >= 0
        dropped = int((~keep).sum())
        if dropped:
            log.warning("%s row(s) had labels outside the model's class set and are "
                        "excluded from the metrics", f"{dropped:,}")
        if batch.proba is None:
            raise RuntimeError("evaluation needs probabilities; call with keep_proba=True")
        report = _evaluate(
            prep.y_true[keep], batch.proba[keep], class_names=self.class_names,
            model=self.model_name, split="uploaded", fpr_budget=fpr_budget,
            predict_seconds=batch.seconds,
        )
        return report


# ---------------------------------------------------------------------------
# input plumbing
# ---------------------------------------------------------------------------

def _as_frame(data: Any) -> pd.DataFrame:
    """DataFrame / mapping / sequence-of-mappings / 2-D array -> DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data.reset_index(drop=True)
    if isinstance(data, pd.Series):
        return data.to_frame().T.reset_index(drop=True)
    if isinstance(data, Mapping):
        return pd.DataFrame([dict(data)])
    if isinstance(data, (list, tuple)) and data and isinstance(data[0], Mapping):
        return pd.DataFrame([dict(d) for d in data])
    raise TypeError(
        f"cannot score a {type(data).__name__}. Pass a DataFrame, a dict of "
        "feature -> value, or a list of such dicts."
    )


def _canonicalise_columns(frame: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Normalise headers and resolve collisions, keeping the first of each.

    CICIDS2017 genuinely ships two columns that normalise to ``Fwd Header Length``. Left
    alone, ``frame[name]`` then returns a 2-column DataFrame instead of a Series and the
    downstream ``to_numpy`` produces a 3-D array - an error message far from its cause.
    """
    renamed = sch.normalise_columns([str(c) for c in frame.columns])
    out = frame.copy()
    out.columns = renamed
    duplicates = sorted({n for n in renamed if renamed.count(n) > 1})
    if duplicates:
        out = out.loc[:, ~out.columns.duplicated(keep="first")]
        log.info("kept the first of %d duplicated column name(s): %s",
                 len(duplicates), ", ".join(duplicates))
    return out.reset_index(drop=True), duplicates


def _coerce_numeric(frame: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Force every column to float, counting the cells that were not numbers.

    ``pd.read_csv`` types a column as object the moment one cell holds ``"Infinity"`` or
    a stray thousands separator, and ``DataFrame.to_numpy(dtype=float)`` then raises. The
    strings pandas recognises (``inf``, ``Infinity``, ``NaN``) survive as floats; genuine
    junk becomes NaN and is imputed, with the count kept so the UI can say so.
    """
    data: Dict[str, Any] = {}
    coerced: Dict[str, int] = {}
    for name in frame.columns:
        column = frame[name]
        if pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column):
            data[name] = column.astype(np.float64)
            continue
        converted = pd.to_numeric(column, errors="coerce")
        bad = int(converted.isna().sum() - column.isna().sum())
        if bad > 0:
            coerced[str(name)] = bad
        data[name] = converted.astype(np.float64)
    return pd.DataFrame(data, index=frame.index), coerced
