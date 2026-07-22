"""The metrics from §3 of the evaluation protocol.

Design principle P6: *"ROC-AUC on a 0.1%-positive problem is decoration."* It is
computed here anyway — refusing to compute it would look like hiding it — but it
is tagged ``headline=False`` and the reporting layer never puts it above the
fold.

The metric that matters is **precision@k**, where ``k`` is the compliance team's
actual daily alert capacity. A model with wonderful PR-AUC that puts its true
positives at rank 4000 is useless to a team that reviews 200 alerts a day, and
precision@k is the only metric that notices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kg_core.timeutil import MILLIS_PER_DAY


@dataclass(frozen=True, slots=True)
class Metric:
    """One measured number, with enough context to report it responsibly."""

    name: str
    value: float
    #: Whether this belongs above the fold in the README.
    headline: bool = True
    description: str = ""

    def __str__(self) -> str:
        return f"{self.name}={self.value:.4f}"


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision).

    The headline threshold-free metric. Computed directly rather than through
    sklearn so the tie-handling is explicit: rows sharing a score are ranked
    together, which matters at a 0.1% positive rate where a model that outputs
    the same value for thousands of rows would otherwise be scored by the
    arbitrary order they arrived in.
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(y_true.sum())
    if positives == 0 or y_true.size == 0:
        return 0.0

    order = np.argsort(-scores, kind="stable")
    sorted_true = y_true[order]
    sorted_scores = scores[order]

    true_positives = np.cumsum(sorted_true)
    ranks = np.arange(1, y_true.size + 1)
    precision = true_positives / ranks
    recall = true_positives / positives

    # Collapse tied scores onto their last position, so every row in a tie group
    # gets the same precision/recall - the honest treatment of a model that
    # cannot distinguish them.
    is_last_of_tie = np.ones(y_true.size, dtype=bool)
    is_last_of_tie[:-1] = sorted_scores[:-1] != sorted_scores[1:]

    kept_precision = precision[is_last_of_tie]
    kept_recall = recall[is_last_of_tie]
    recall_deltas = np.diff(np.concatenate([[0.0], kept_recall]))
    return float(np.sum(kept_precision * recall_deltas))


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC. Computed, reported, and never a headline.

    At a 0.1% positive rate this is dominated by the negative class: a model can
    score 0.95 here while its top 200 alerts are almost all false positives.
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    positives, negatives = int(y_true.sum()), int((1 - y_true).sum())
    if positives == 0 or negatives == 0:
        return 0.5
    order = np.argsort(scores, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within tie groups, the standard Mann-Whitney correction.
    sorted_scores = np.asarray(scores)[order]
    start = 0
    for index in range(1, len(sorted_scores) + 1):
        if index == len(sorted_scores) or sorted_scores[index] != sorted_scores[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    positive_rank_sum = ranks[y_true == 1].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Precision among the top *k* scored rows.

    ``k`` is a capacity, not a tuning knob: it is how many alerts the compliance
    team can actually review. Protocol §3 fixes the headline at k = 200/day.
    """
    if k <= 0 or y_true.size == 0:
        return 0.0
    k = min(k, y_true.size)
    top = np.argsort(-np.asarray(scores), kind="stable")[:k]
    return float(np.asarray(y_true)[top].mean())


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Share of all true positives that fall inside the top *k*.

    What the bank loses by having a finite team, rather than a property of the
    model alone.
    """
    positives = int(np.asarray(y_true).sum())
    if positives == 0 or k <= 0:
        return 0.0
    k = min(k, y_true.size)
    top = np.argsort(-np.asarray(scores), kind="stable")[:k]
    return float(np.asarray(y_true)[top].sum() / positives)


def daily_capacity_k(timestamps: np.ndarray, per_day: int) -> int:
    """Convert a daily alert budget into a top-k for this evaluation window.

    A test period of 18 days at 200 alerts/day is a budget of 3600, not 200.
    Getting this wrong by using the daily number directly makes precision@k look
    far better than the team's real experience.
    """
    stamps = np.asarray(timestamps, dtype=np.int64)
    if stamps.size == 0:
        return per_day
    span_days = max(1.0, (int(stamps.max()) - int(stamps.min())) / MILLIS_PER_DAY)
    return max(1, round(per_day * span_days))


def alert_rate(scores: np.ndarray, threshold: float) -> float:
    """Share of transactions that would be alerted at *threshold*."""
    return float((np.asarray(scores) >= threshold).mean()) if scores.size else 0.0


# --------------------------------------------------------------- calibration


def brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Mean squared error of the probabilities. Lower is better."""
    return float(np.mean((np.asarray(probabilities) - np.asarray(y_true)) ** 2))


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> float:
    """ECE over equal-mass bins.

    Equal-mass rather than equal-width: at a 0.1% positive rate almost every
    prediction sits near zero, so equal-width bins put 99% of the data in the
    first bin and measure nothing.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    if probabilities.size == 0:
        return 0.0

    order = np.argsort(probabilities, kind="stable")
    chunks = np.array_split(order, min(bins, max(1, probabilities.size)))
    total = 0.0
    for chunk in chunks:
        if chunk.size == 0:
            continue
        weight = chunk.size / probabilities.size
        total += weight * abs(probabilities[chunk].mean() - y_true[chunk].mean())
    return float(total)


def reliability_curve(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> list[dict[str, float]]:
    """Points for a reliability diagram: predicted vs observed, per bin."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    order = np.argsort(probabilities, kind="stable")
    points: list[dict[str, float]] = []
    for chunk in np.array_split(order, min(bins, max(1, probabilities.size))):
        if chunk.size == 0:
            continue
        points.append(
            {
                "predicted": float(probabilities[chunk].mean()),
                "observed": float(y_true[chunk].mean()),
                "count": int(chunk.size),
            }
        )
    return points


# ------------------------------------------------------------------ bundling


@dataclass(slots=True)
class MetricBundle:
    """Every metric for one model on one fold."""

    metrics: dict[str, Metric] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def add(self, metric: Metric) -> None:
        self.metrics[metric.name] = metric

    def value(self, name: str) -> float:
        return self.metrics[name].value

    def headline(self) -> dict[str, float]:
        return {n: m.value for n, m in self.metrics.items() if m.headline}

    def as_dict(self) -> dict[str, Any]:
        return {
            "metrics": {n: round(m.value, 6) for n, m in self.metrics.items()},
            "headline": sorted(n for n, m in self.metrics.items() if m.headline),
            **self.extras,
        }


def evaluate(
    y_true: np.ndarray,
    scores: np.ndarray,
    timestamps: np.ndarray,
    *,
    capacity_per_day: int = 200,
    threshold: float | None = None,
    cost_model: Any = None,
    amounts_cents: np.ndarray | None = None,
) -> MetricBundle:
    """Compute the full protocol §3 metric set for one set of predictions."""
    bundle = MetricBundle()
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)

    k = daily_capacity_k(timestamps, capacity_per_day)

    bundle.add(
        Metric(
            "pr_auc",
            average_precision(y_true, scores),
            description="area under the precision-recall curve",
        )
    )
    bundle.add(
        Metric(
            f"precision_at_{capacity_per_day}_per_day",
            precision_at_k(y_true, scores, k),
            description="precision within the analyst capacity budget",
        )
    )
    bundle.add(
        Metric(
            f"recall_at_{capacity_per_day}_per_day",
            recall_at_k(y_true, scores, k),
            description="share of true positives caught inside that budget",
        )
    )
    bundle.add(
        Metric(
            "brier",
            brier_score(y_true, scores),
            description="calibration: mean squared error of the probabilities",
        )
    )
    bundle.add(
        Metric(
            "ece",
            expected_calibration_error(y_true, scores),
            description="expected calibration error, 15 equal-mass bins",
        )
    )
    # Computed but deliberately not a headline. See §3 of the protocol.
    bundle.add(
        Metric(
            "roc_auc",
            roc_auc(y_true, scores),
            headline=False,
            description="decoration at this class balance; reported for completeness",
        )
    )

    bundle.extras["positives"] = int(y_true.sum())
    bundle.extras["rows"] = int(y_true.size)
    bundle.extras["positive_rate"] = round(float(y_true.mean()), 6) if y_true.size else 0.0
    bundle.extras["k"] = k

    if threshold is not None:
        bundle.add(Metric("alert_rate", alert_rate(scores, threshold), headline=False))
        bundle.extras["threshold"] = round(float(threshold), 6)

    if cost_model is not None and threshold is not None:
        outcome = cost_model.evaluate(y_true, scores, threshold, amounts_cents)
        bundle.add(
            Metric(
                "expected_cost_per_10k",
                outcome.cost_per_10k,
                description="expected cost per 10,000 transactions",
            )
        )
        bundle.extras["cost"] = outcome.as_dict()

    return bundle
