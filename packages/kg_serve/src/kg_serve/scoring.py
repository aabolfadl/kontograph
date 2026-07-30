"""The decision path: one payment in, one scored and explained decision out.

Everything here runs inside the 200 ms budget, so the ordering is chosen for
latency as much as for correctness:

    1. feature_fetch    read this account's online state, compute the vector
    2. graph_assemble   pull the temporal neighbourhood (bounded, not unbounded)
    3. inference        the LightGBM booster in-process, no network hop (ADR-0008)
    4. explain          the amortised head, one forward pass (ADR-0007)
    5. counterfactual   bounded beam search, wall-clock budgeted

Stages 4 and 5 only run when the score crosses the threshold. That is not a
shortcut — an explanation for a payment nobody will look at has no consumer, and
spending 20 ms on it would be spending the budget of the alerts that *do* get
looked at.

**The feature vector comes from the same online executor the equivalence tests
use.** Not a reimplementation "for speed": a reimplementation is exactly the
training-serving skew M3 exists to eliminate, and the property suite would stop
covering the code that actually runs.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kg_core.contracts import Decision
from kg_core.fmt import format_score
from kg_core.logging import get_logger
from kg_features.compilers.online import OnlineFeatureExecutor
from kg_serve.latency import StageTimer

log = get_logger(__name__)


@dataclass(slots=True)
class ScoredDecision:
    """What the gateway returns and what the decision log records."""

    alert_id: str
    tx_id: str
    score: float
    threshold: float
    decision: Decision
    latency: dict[str, float]
    features: dict[str, float] = field(default_factory=dict)
    explanation: dict[str, Any] | None = None

    @property
    def alerted(self) -> bool:
        return self.decision is not Decision.ALLOW

    def as_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "tx_id": self.tx_id,
            "score": round(self.score, 6),
            "threshold": round(self.threshold, 6),
            "decision": self.decision.value,
            "latency_ms": {k: round(v, 3) for k, v in self.latency.items()},
            "explanation": self.explanation,
        }


def decide(score: float, threshold: float) -> Decision:
    """Map a calibrated score to an action.

    Four outcomes rather than two, because a real gate does more than "yes/no".
    The bands above the threshold are widely spaced: under SCT Inst a BLOCK is
    an irrevocable customer-facing action taken in under ten seconds, so the bar
    for it is deliberately far above the bar for raising an alert somebody will
    read later.
    """
    if score < threshold:
        return Decision.ALLOW
    if score < threshold * 1.5:
        return Decision.ALERT
    if score < threshold * 2.0:
        return Decision.HOLD
    return Decision.BLOCK


class ScoringEngine:
    """Holds the model, the feature state and the explainer for the hot path."""

    def __init__(
        self,
        feature_set: Any,
        model: Any,
        *,
        threshold: float | None = None,
        store: Any = None,
        explainer: Any = None,
        model_name: str = "lightgbm",
        model_version: str = "unknown",
        code_commit: str = "unknown",
        explain_alerts_only: bool = True,
    ) -> None:
        self.feature_set = feature_set
        self.model = model
        self.threshold = threshold if threshold is not None else getattr(model, "threshold", 0.5)
        self.executor = OnlineFeatureExecutor(feature_set, store=store)
        self.explainer = explainer
        self.model_name = model_name
        self.model_version = model_version
        self.code_commit = code_commit
        self.explain_alerts_only = explain_alerts_only
        self.feature_set_hash = feature_set.fingerprint()
        self._last_entity_count = 0

    def score(self, event: dict[str, Any], *, observe: bool = True) -> ScoredDecision:
        """Score one payment.

        ``observe=False`` re-scores without folding the event into state, which
        is what the explanation-stability metric needs: the same alert must
        produce the same story when asked twice.
        """
        timer = StageTimer()

        with timer.stage("feature_fetch"):
            vector = self.executor.compute(event)
            matrix = np.asarray(
                [[vector[name] for name in self.feature_set.column_names]], dtype=np.float64
            )

        with timer.stage("graph_assemble"):
            # Bounded by construction: the online executor already holds only
            # the retained window per entity, so there is no unbounded traversal
            # on this path, and this stage is currently near-free.
            #
            # Timed anyway, and deliberately kept in the budget table as a
            # near-zero row: the moment the TGN's neighbour fetch lands here it
            # stops being free, and a stage that only appears once it becomes
            # expensive is a stage nobody budgeted for.
            self._last_entity_count = self.executor.tracked_entities

        with timer.stage("inference"):
            score = float(self.model.predict(matrix)[0])

        decision = decide(score, self.threshold)
        explanation: dict[str, Any] | None = None

        should_explain = self.explainer is not None and (
            decision is not Decision.ALLOW or not self.explain_alerts_only
        )
        if should_explain:
            with timer.stage("explain"):
                explanation = self.explainer.explain(event, vector, score)
            with timer.stage("counterfactual"):
                counterfactual = self.explainer.counterfactual(event, vector, score, self.threshold)
                if explanation is not None:
                    explanation["counterfactual"] = counterfactual

        if observe:
            self.executor.observe(event)

        return ScoredDecision(
            alert_id=str(uuid.uuid4()),
            tx_id=event["tx_id"],
            score=score,
            threshold=self.threshold,
            decision=decision,
            latency=timer.as_dict(),
            features=vector,
            explanation=explanation,
        )

    def warm(self, events: list[dict[str, Any]]) -> int:
        """Fold historical events into state without scoring them.

        Serving from a cold state is not merely inaccurate, it is *silently*
        inaccurate: every feature reads its default, the model scores against a
        world where nobody has ever transacted, and nothing in the output says
        so. A gateway that starts cold should be warmed or should refuse
        traffic.
        """
        for event in sorted(events, key=lambda e: (int(e["acceptance_ts"]), str(e["tx_id"]))):
            self.executor.observe(event)
        log.info("warmed", events=len(events), entities=self.executor.tracked_entities)
        return self.executor.tracked_entities


class FeatureAttributionExplainer:
    """A tabular explainer for the LightGBM baseline.

    Not the graph explainer — that is the TGN's, and it lives in
    ``kg_models.explain``. This one exists so the gateway is complete and
    measurable before the GPU-trained student lands, and so the latency budget
    has a real explanation stage in it rather than a zero.

    It is honest about what it is: a per-feature contribution from the booster's
    own SHAP output, which explains the *model*, not the *graph*.
    """

    def __init__(self, model: Any, feature_names: tuple[str, ...], top_k: int = 5) -> None:
        self.model = model
        self.feature_names = feature_names
        self.top_k = top_k

    def explain(
        self, event: dict[str, Any], vector: dict[str, float], score: float
    ) -> dict[str, Any]:
        booster = getattr(self.model, "booster", None)
        if booster is None:
            return {"kind": "none", "reason": "the model has no booster to attribute through"}

        matrix = np.asarray([[vector[n] for n in self.feature_names]], dtype=np.float64)
        contributions = booster.predict(matrix, pred_contrib=True)[0]
        # The last element is the expected value, not a feature.
        weights = contributions[:-1]
        order = np.argsort(-np.abs(weights))[: self.top_k]

        return {
            "kind": "feature_attribution",
            "note": "explains the model's use of features, not the graph structure",
            "top_features": [
                {
                    "feature": self.feature_names[i],
                    "value": round(float(vector[self.feature_names[i]]), 4),
                    "contribution": round(float(weights[i]), 6),
                }
                for i in order
            ],
        }

    def counterfactual(
        self, event: dict[str, Any], vector: dict[str, float], score: float, threshold: float
    ) -> dict[str, Any]:
        """A feature-space counterfactual, clearly marked as the weaker kind.

        design principle P5.3 wants a *subgraph* counterfactual — "remove these two
        hops" — which needs the temporal graph and lands with the TGN. This one
        answers "which single feature, set to its population median, would drop
        the score below threshold?" That is genuinely less useful, and saying so
        is better than presenting it as the real thing.
        """
        booster = getattr(self.model, "booster", None)
        if booster is None:
            return {"found": False, "reason": "no booster"}

        names = list(self.feature_names)
        base = np.asarray([[vector[n] for n in names]], dtype=np.float64)
        contributions = booster.predict(base, pred_contrib=True)[0][:-1]

        # Try the highest-contributing features first: neutralising one of those
        # is the likeliest single edit to move the score.
        for index in np.argsort(-contributions)[:8]:
            probe = base.copy()
            probe[0, index] = 0.0
            probed = float(self.model.predict(probe)[0])
            if probed < threshold:
                return {
                    "found": True,
                    "kind": "feature_space",
                    "feature": names[index],
                    "from": round(float(base[0, index]), 4),
                    "to": 0.0,
                    "score_after": round(probed, 6),
                    "crosses_threshold": True,
                    "narrative_en": (
                        f"The alert rests on {names[index]}: with that feature at its "
                        f"baseline the score falls from {format_score(score)} to "
                        f"{format_score(probed)}, below the "
                        f"{format_score(threshold)} threshold."
                    ),
                    "narrative_de": (
                        f"Der Alarm beruht auf {names[index]}: bei Basiswert dieses Merkmals "
                        f"faellt der Score von {format_score(score)} auf "
                        f"{format_score(probed)} und damit unter die Schwelle "
                        f"von {format_score(threshold)}."
                    ),
                }

        return {
            "found": False,
            "kind": "feature_space",
            "reason": "no single feature explains this alert on its own",
        }
