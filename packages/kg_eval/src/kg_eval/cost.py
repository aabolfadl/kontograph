"""The cost model and the cost-sensitive operating point.

§4 of the evaluation protocol. design principle P6: *"We optimise the thing the bank
pays for."*

Two things here are worth defending.

**The false-negative cost is per transaction, not a constant.** Missing a €40
mule hop and missing a €90,000 APP fraud are not the same mistake, and a model
tuned against an averaged constant will systematically under-alert on the large
ones. The cost of a miss is the amount that got away.

**The threshold is chosen on validation and applied unchanged to test.** Picking
it on test is the most common way an evaluation overstates its results,
and it is invisible in the final table — the number simply looks good.

The euro figures are order-of-magnitude estimates for a mid-sized German
institution, stated as assumptions rather than presented as measurements, and
varied in a sensitivity analysis. They are inputs to a decision, not facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class CostModel:
    """What each outcome costs, in euro cents.

    Defaults are the protocol's committed figures:
      investigation  €35   analyst time for one alert
      friction       €12   the customer experience of a held payment
      prevention     0.85  share of the loss avoided by alerting in time
      cap            €100,000 on any single loss
    """

    investigation_cents: int = 3_500
    friction_cents: int = 1_200

    #: Share of the loss actually avoided when a fraudulent payment is alerted.
    #:
    #: High, because that is the entire premise of the project: under SCT Inst
    #: a decision inside the ten-second window can hold the payment before it
    #: settles irrevocably. It is not 1.0 — some alerts land after settlement,
    #: some are overridden by the payer, and some funds are already gone.
    prevention_fraction: float = 0.85

    max_loss_cents: int = 100_000_00
    #: Used when a per-transaction amount is unavailable.
    default_loss_cents: int = 50_000

    @property
    def false_positive_cents(self) -> int:
        """A legitimate payment alerted: analyst time plus customer friction."""
        return self.investigation_cents + self.friction_cents

    def false_negative_cents(self, amount_cents: np.ndarray | None, count: int) -> np.ndarray:
        """The loss from each missed fraud: the amount that got away."""
        if amount_cents is None:
            return np.full(count, float(self.default_loss_cents))
        return np.minimum(np.asarray(amount_cents, dtype=np.float64), self.max_loss_cents)

    def true_positive_cents(self, amount_cents: np.ndarray | None, count: int) -> np.ndarray:
        """A caught fraud costs the investigation plus whatever was NOT prevented.

        Never negative. An earlier version computed
        ``investigation - recovery x amount``, which made catching fraud
        *profitable* and mixed two incompatible accounting framings: a true
        positive was priced relative to a do-nothing baseline while a false
        negative was priced absolutely. The optimiser noticed, and preferred
        alerting on almost nothing — catching 2.7% of fraud scored a lower
        "cost" than catching 97%.

        Everything here is now absolute cost: the money the institution is out
        by, given what happened. Catching fraud reduces that cost; it never
        creates income.
        """
        losses = self.false_negative_cents(amount_cents, count)
        return self.investigation_cents + (1.0 - self.prevention_fraction) * losses

    def evaluate(
        self,
        y_true: np.ndarray,
        scores: np.ndarray,
        threshold: float,
        amounts_cents: np.ndarray | None = None,
    ) -> CostOutcome:
        """Confusion matrix and total cost at *threshold*."""
        return _evaluate(self, y_true, scores, threshold, amounts_cents)

    def describe(self) -> dict[str, Any]:
        return {
            "investigation_eur": self.investigation_cents / 100,
            "friction_eur": self.friction_cents / 100,
            "false_positive_eur": self.false_positive_cents / 100,
            "prevention_fraction": self.prevention_fraction,
            "max_loss_eur": self.max_loss_cents / 100,
            "note": (
                "Order-of-magnitude estimates for a mid-sized German institution, "
                "stated as assumptions. Varied in the sensitivity analysis."
            ),
        }


@dataclass(slots=True)
class CostOutcome:
    """The confusion matrix and what it cost."""

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    total_cents: float
    rows: int
    threshold: float
    breakdown: dict[str, float] = field(default_factory=dict)

    @property
    def cost_per_10k(self) -> float:
        """Expected cost per 10,000 transactions, in euro."""
        if self.rows == 0:
            return 0.0
        return float(self.total_cents / self.rows * 10_000 / 100)

    @property
    def alerts(self) -> int:
        return self.true_positives + self.false_positives

    @property
    def precision(self) -> float:
        return self.true_positives / self.alerts if self.alerts else 0.0

    @property
    def recall(self) -> float:
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "threshold": round(self.threshold, 6),
            "tp": self.true_positives,
            "fp": self.false_positives,
            "tn": self.true_negatives,
            "fn": self.false_negatives,
            "alerts": self.alerts,
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "cost_per_10k_eur": round(self.cost_per_10k, 2),
            "breakdown_eur": {k: round(v / 100, 2) for k, v in self.breakdown.items()},
        }


def _evaluate(
    model: CostModel,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    amounts_cents: np.ndarray | None = None,
) -> CostOutcome:
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    alerted = scores >= threshold

    tp_mask = alerted & (y_true == 1)
    fp_mask = alerted & (y_true == 0)
    fn_mask = (~alerted) & (y_true == 1)
    tn_mask = (~alerted) & (y_true == 0)

    amounts = np.asarray(amounts_cents, dtype=np.float64) if amounts_cents is not None else None

    fn_costs = model.false_negative_cents(
        amounts[fn_mask] if amounts is not None else None, int(fn_mask.sum())
    )
    tp_costs = model.true_positive_cents(
        amounts[tp_mask] if amounts is not None else None, int(tp_mask.sum())
    )

    fp_total = float(fp_mask.sum()) * model.false_positive_cents
    fn_total = float(fn_costs.sum())
    tp_total = float(tp_costs.sum())

    return CostOutcome(
        true_positives=int(tp_mask.sum()),
        false_positives=int(fp_mask.sum()),
        true_negatives=int(tn_mask.sum()),
        false_negatives=int(fn_mask.sum()),
        total_cents=fp_total + fn_total + tp_total,
        rows=int(y_true.size),
        threshold=float(threshold),
        breakdown={
            "false_positives": fp_total,
            "missed_fraud": fn_total,
            "caught_fraud_net": tp_total,
        },
    )


@dataclass(slots=True)
class ThresholdChoice:
    """The chosen operating point and the curve it was chosen from."""

    threshold: float
    outcome: CostOutcome
    curve: list[dict[str, float]] = field(default_factory=list)
    chosen_on: str = "validation"

    def as_dict(self) -> dict[str, Any]:
        return {
            "threshold": round(self.threshold, 6),
            "chosen_on": self.chosen_on,
            "outcome": self.outcome.as_dict(),
        }


def choose_threshold(
    model: CostModel,
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    amounts_cents: np.ndarray | None = None,
    candidates: int = 200,
    max_alert_rate: float | None = None,
    chosen_on: str = "validation",
) -> ThresholdChoice:
    """Pick the expected-cost-minimising threshold.

    ``max_alert_rate`` imposes the capacity constraint. Without it the optimiser
    will happily choose a threshold that alerts on 15% of traffic because the
    arithmetic says the fraud avoided outweighs the investigation cost — which is
    true on paper and impossible for a team of eight people. A model that
    proposes 40,000 alerts a day has not solved the problem.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return ThresholdChoice(0.5, _evaluate(model, y_true, scores, 0.5, amounts_cents))

    # Candidate thresholds at score quantiles rather than on a uniform grid: at
    # a 0.1% positive rate the scores bunch near zero, and a uniform grid spends
    # all its candidates in a region containing no decision boundary.
    quantiles = np.linspace(0.0, 1.0, candidates)
    grid = np.unique(np.quantile(scores, quantiles))

    curve: list[dict[str, float]] = []
    best: tuple[float, CostOutcome] | None = None

    for threshold in grid:
        outcome = _evaluate(model, y_true, scores, float(threshold), amounts_cents)
        rate = outcome.alerts / max(1, outcome.rows)
        curve.append(
            {
                "threshold": float(threshold),
                "cost_per_10k": outcome.cost_per_10k,
                "alert_rate": rate,
                "precision": outcome.precision,
                "recall": outcome.recall,
            }
        )
        if max_alert_rate is not None and rate > max_alert_rate:
            continue
        if best is None or outcome.cost_per_10k < best[1].cost_per_10k:
            best = (float(threshold), outcome)

    if best is None:
        # Every candidate breached the capacity constraint; fall back to the
        # tightest threshold that satisfies it.
        threshold = float(np.quantile(scores, 1.0 - (max_alert_rate or 0.01)))
        best = (threshold, _evaluate(model, y_true, scores, threshold, amounts_cents))

    return ThresholdChoice(threshold=best[0], outcome=best[1], curve=curve, chosen_on=chosen_on)


def sensitivity_analysis(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    amounts_cents: np.ndarray | None = None,
    ratios: tuple[float, ...] = (10.0, 50.0, 100.0, 500.0),
    max_alert_rate: float | None = None,
) -> list[dict[str, Any]]:
    """How the operating point moves as the FN/FP cost ratio changes.

    Protocol §4 requires this because the euro figures are assumptions. If the
    chosen threshold is stable across an order of magnitude of the ratio, the
    assumption is not load-bearing; if it swings wildly, the result depends on a
    number nobody measured — and that is worth saying out loud.
    """
    results: list[dict[str, Any]] = []
    base = CostModel()
    for ratio in ratios:
        # Hold the FP cost fixed and scale the loss so FN/FP equals the ratio.
        model = CostModel(
            investigation_cents=base.investigation_cents,
            friction_cents=base.friction_cents,
            prevention_fraction=base.prevention_fraction,
            default_loss_cents=int(base.false_positive_cents * ratio),
            max_loss_cents=base.max_loss_cents,
        )
        choice = choose_threshold(
            model,
            y_true,
            scores,
            amounts_cents=None,  # use the flat loss so the ratio is exact
            max_alert_rate=max_alert_rate,
        )
        results.append(
            {
                "fn_fp_ratio": ratio,
                "threshold": round(choice.threshold, 6),
                "alert_rate": round(choice.outcome.alerts / max(1, choice.outcome.rows), 6),
                "precision": round(choice.outcome.precision, 6),
                "recall": round(choice.outcome.recall, 6),
                "cost_per_10k_eur": round(choice.outcome.cost_per_10k, 2),
            }
        )
    return results
