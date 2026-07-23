"""Bootstrap confidence intervals.

§10 of the protocol: *"Every reported point estimate carries a 95 % confidence
interval. A point estimate without an interval is an opinion."*

**Resampling is stratified by day, not by row.** Payment streams are not
independent draws: an account's transactions are correlated, a mule campaign is
a cluster, and payroll day is a spike. Row-level resampling breaks those
clusters apart and produces intervals that are far too narrow — the classic way
a bootstrap gives false confidence. Resampling whole days keeps the within-day
dependence intact.

**Comparisons use a paired bootstrap.** To decide whether model A beats model B,
resample once and evaluate *both* models on the same resample. The two are then
correlated in exactly the way they are in reality, and the interval is on the
*difference*, which is the quantity the claim is about. Comparing two
independently-computed intervals and checking whether they overlap is a weaker
and commonly wrong test.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from kg_core.timeutil import MILLIS_PER_DAY

#: Protocol §10.
DEFAULT_RESAMPLES = 2_000
DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True, slots=True)
class Interval:
    """A point estimate with its confidence interval."""

    point: float
    low: float
    high: float
    confidence: float = DEFAULT_CONFIDENCE
    resamples: int = DEFAULT_RESAMPLES

    @property
    def width(self) -> float:
        return self.high - self.low

    def excludes(self, value: float) -> bool:
        """Whether *value* lies outside the interval.

        ``interval.excludes(0.0)`` on a difference is the test the ablation
        commitment in protocol §6 turns on.
        """
        return value < self.low or value > self.high

    def format(self, digits: int = 4) -> str:
        return f"{self.point:.{digits}f} [{self.low:.{digits}f}, {self.high:.{digits}f}]"

    def as_dict(self) -> dict[str, Any]:
        return {
            "point": round(self.point, 6),
            "ci_low": round(self.low, 6),
            "ci_high": round(self.high, 6),
            "confidence": self.confidence,
            "resamples": self.resamples,
        }


def _day_index(timestamps: np.ndarray) -> np.ndarray:
    return (np.asarray(timestamps, dtype=np.int64) // MILLIS_PER_DAY).astype(np.int64)


def _day_blocks(timestamps: np.ndarray) -> list[np.ndarray]:
    """Row indices grouped by UTC day."""
    days = _day_index(timestamps)
    order = np.argsort(days, kind="stable")
    sorted_days = days[order]
    boundaries = np.flatnonzero(np.diff(sorted_days)) + 1
    return np.split(order, boundaries)


def bootstrap_metric(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    scores: np.ndarray,
    timestamps: np.ndarray,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 20251009,
) -> Interval:
    """Percentile bootstrap for one metric, resampling whole days."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    point = float(metric(y_true, scores))

    blocks = _day_blocks(timestamps)
    if len(blocks) < 2:
        # A single day cannot be resampled by day. Say so with a degenerate
        # interval rather than silently falling back to row resampling, which
        # would produce a confident-looking and wrong number.
        return Interval(point, point, point, confidence, 0)

    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    block_count = len(blocks)

    for index in range(resamples):
        chosen = rng.integers(0, block_count, size=block_count)
        rows = np.concatenate([blocks[c] for c in chosen])
        estimates[index] = metric(y_true[rows], scores[rows])

    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return Interval(point, float(low), float(high), confidence, resamples)


def bootstrap_difference(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    timestamps: np.ndarray,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 20251009,
) -> Interval:
    """Paired bootstrap on ``metric(A) - metric(B)``.

    The quantity the ablation ladder's commitment is about: rung 4 beats rung 1
    only if this interval excludes zero. An interval that straddles zero means
    the two models are not distinguishable on this data, and the protocol
    requires saying so rather than reporting the higher number as a win.
    """
    y_true = np.asarray(y_true)
    scores_a, scores_b = np.asarray(scores_a), np.asarray(scores_b)
    point = float(metric(y_true, scores_a)) - float(metric(y_true, scores_b))

    blocks = _day_blocks(timestamps)
    if len(blocks) < 2:
        return Interval(point, point, point, confidence, 0)

    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    block_count = len(blocks)

    for index in range(resamples):
        chosen = rng.integers(0, block_count, size=block_count)
        rows = np.concatenate([blocks[c] for c in chosen])
        # Same resample for both models: that pairing is the whole point.
        estimates[index] = metric(y_true[rows], scores_a[rows]) - metric(
            y_true[rows], scores_b[rows]
        )

    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return Interval(point, float(low), float(high), confidence, resamples)


@dataclass(slots=True)
class Comparison:
    """Whether one model is distinguishable from another, and by how much."""

    metric: str
    model_a: str
    model_b: str
    difference: Interval

    @property
    def distinguishable(self) -> bool:
        return self.difference.excludes(0.0)

    @property
    def verdict(self) -> str:
        if not self.distinguishable:
            return "not distinguishable"
        return "A wins" if self.difference.point > 0 else "B wins"

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "a": self.model_a,
            "b": self.model_b,
            "difference": self.difference.as_dict(),
            "distinguishable": self.distinguishable,
            "verdict": self.verdict,
        }

    def sentence(self) -> str:
        """A line fit to paste into the README.

        Phrased so a null result reads as a finding rather than an omission —
        which is what protocol §6 commits to.
        """
        delta = self.difference
        if not self.distinguishable:
            return (
                f"{self.model_a} and {self.model_b} are not distinguishable on {self.metric}: "
                f"the difference is {delta.format()}, whose 95% interval includes zero."
            )
        direction = "beats" if delta.point > 0 else "loses to"
        return (
            f"{self.model_a} {direction} {self.model_b} on {self.metric} by "
            f"{delta.format()} (95% CI excludes zero)."
        )
