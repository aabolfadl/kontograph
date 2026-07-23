"""Probability calibration.

§5 of the protocol: *"A score that feeds a cost-based threshold must be a
probability, not a ranking."*

This is not optional polish. The cost model compares an expected loss against an
investigation cost, and that arithmetic is meaningless if the score is a
ranking. A model outputting 0.9 for something that happens 3 % of the time will
have its threshold placed in the wrong location, and the resulting operating
point will be confidently wrong.

Both calibrators are fitted **on the validation split only**. Fitting on train
calibrates against the model's own memorised training predictions, which are far
sharper than its behaviour on unseen data — the calibrator then makes things
worse, and does so invisibly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np

from kg_core.logging import get_logger

log = get_logger(__name__)

Method = Literal["isotonic", "platt", "none"]


class Calibrator(Protocol):
    """Maps raw model scores to calibrated probabilities."""

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> Calibrator: ...
    def transform(self, scores: np.ndarray) -> np.ndarray: ...
    def to_dict(self) -> dict[str, Any]: ...


@dataclass(slots=True)
class IsotonicCalibrator:
    """Monotone piecewise-constant fit. Flexible, and hungry for positives.

    Isotonic regression makes no functional-form assumption, which is its
    strength and its risk: at a 0.1 % positive rate the validation split may
    contain only tens of positives, and isotonic will happily fit their noise.
    :func:`choose_calibrator` picks Platt instead when positives are scarce.
    """

    thresholds: np.ndarray | None = None
    values: np.ndarray | None = None

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> IsotonicCalibrator:
        from sklearn.isotonic import IsotonicRegression

        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(np.asarray(scores, dtype=np.float64), np.asarray(y_true, dtype=np.float64))
        self.thresholds = np.asarray(model.X_thresholds_, dtype=np.float64)
        self.values = np.asarray(model.y_thresholds_, dtype=np.float64)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if self.thresholds is None or self.values is None:
            raise RuntimeError("calibrator is not fitted")
        # np.interp with clipping at the ends reproduces sklearn's
        # out_of_bounds="clip", without needing sklearn at serving time.
        return np.clip(
            np.interp(np.asarray(scores, dtype=np.float64), self.thresholds, self.values),
            0.0,
            1.0,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "isotonic",
            "thresholds": self.thresholds.tolist() if self.thresholds is not None else [],
            "values": self.values.tolist() if self.values is not None else [],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> IsotonicCalibrator:
        return cls(
            thresholds=np.asarray(payload["thresholds"], dtype=np.float64),
            values=np.asarray(payload["values"], dtype=np.float64),
        )


@dataclass(slots=True)
class PlattCalibrator:
    """Logistic fit on the score. Two parameters, so it survives scarce positives."""

    a: float = 1.0
    b: float = 0.0

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> PlattCalibrator:
        from sklearn.linear_model import LogisticRegression

        scores = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        y_true = np.asarray(y_true, dtype=np.int8)
        if len(np.unique(y_true)) < 2:
            # Degenerate fold: leave the identity rather than fitting nonsense.
            log.warning("calibration fold has one class; leaving scores unchanged")
            self.a, self.b = 1.0, 0.0
            return self
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        model.fit(scores, y_true)
        self.a = float(model.coef_[0][0])
        self.b = float(model.intercept_[0])
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        z = self.a * np.asarray(scores, dtype=np.float64) + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

    def to_dict(self) -> dict[str, Any]:
        return {"method": "platt", "a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PlattCalibrator:
        return cls(a=float(payload["a"]), b=float(payload["b"]))


@dataclass(slots=True)
class IdentityCalibrator:
    """No calibration. Recorded explicitly so 'uncalibrated' is a stated choice."""

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> IdentityCalibrator:
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(scores, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {"method": "none"}


#: Below this many positives in the calibration fold, isotonic overfits and
#: Platt is the safer choice. Not tuned on results — set from the shape of the
#: problem, which is what keeps it out of the engineering invariants' prohibition.
MIN_POSITIVES_FOR_ISOTONIC = 50


def build(method: Method) -> Calibrator:
    if method == "isotonic":
        return IsotonicCalibrator()
    if method == "platt":
        return PlattCalibrator()
    return IdentityCalibrator()


def from_dict(payload: dict[str, Any]) -> Calibrator:
    method = payload.get("method", "none")
    if method == "isotonic":
        return IsotonicCalibrator.from_dict(payload)
    if method == "platt":
        return PlattCalibrator.from_dict(payload)
    return IdentityCalibrator()


@dataclass(slots=True)
class CalibrationResult:
    """A fitted calibrator plus the evidence for choosing it."""

    calibrator: Calibrator
    method: Method
    before: dict[str, float]
    after: dict[str, float]
    candidates: dict[str, dict[str, float]]
    reliability: list[dict[str, float]]

    @property
    def improved(self) -> bool:
        return self.after["brier"] <= self.before["brier"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "before": {k: round(v, 6) for k, v in self.before.items()},
            "after": {k: round(v, 6) for k, v in self.after.items()},
            "candidates": {
                name: {k: round(v, 6) for k, v in scores.items()}
                for name, scores in self.candidates.items()
            },
            "improved": self.improved,
            "reliability": self.reliability,
        }


def choose_calibrator(
    scores: np.ndarray, y_true: np.ndarray, *, prefer: Method | None = None
) -> CalibrationResult:
    """Fit both calibrators on the validation fold and keep the better one.

    Selection is on **Brier score**, which is a proper scoring rule: it rewards
    being both calibrated and sharp, so it cannot be gamed by predicting the base
    rate for everything. Selecting on ECE alone would prefer exactly that
    degenerate model, because a constant predictor is perfectly calibrated and
    completely useless.
    """
    from kg_eval.metrics import brier_score, expected_calibration_error, reliability_curve

    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int8)
    positives = int(y_true.sum())

    def measure(values: np.ndarray) -> dict[str, float]:
        return {
            "brier": brier_score(y_true, values),
            "ece": expected_calibration_error(y_true, values),
        }

    before = measure(scores)
    candidates: dict[str, dict[str, float]] = {"none": before}
    fitted: dict[str, Calibrator] = {"none": IdentityCalibrator()}

    methods: list[Method] = ["platt"]
    if positives >= MIN_POSITIVES_FOR_ISOTONIC:
        methods.append("isotonic")
    else:
        log.info(
            "skipping isotonic calibration: too few positives to fit without overfitting",
            positives=positives,
            minimum=MIN_POSITIVES_FOR_ISOTONIC,
        )

    for method in methods:
        calibrator = build(method).fit(scores, y_true)
        candidates[method] = measure(calibrator.transform(scores))
        fitted[method] = calibrator

    # Selected on Brier, a proper scoring rule: it rewards being calibrated AND
    # sharp, so a constant predictor at the base rate cannot win by being
    # perfectly calibrated and useless.
    best_name = min(candidates, key=lambda name: candidates[name]["brier"])
    chosen: Method = prefer if prefer is not None else best_name  # type: ignore[assignment]
    calibrator = fitted[str(chosen)]
    calibrated = calibrator.transform(scores)

    return CalibrationResult(
        calibrator=calibrator,
        method=chosen,
        before=before,
        after=measure(calibrated),
        candidates=candidates,
        reliability=reliability_curve(y_true, calibrated),
    )
