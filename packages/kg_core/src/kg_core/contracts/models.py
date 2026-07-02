"""Pydantic mirrors of the Avro contracts.

Two representations of the same contract is a drift risk, so we take the risk
deliberately and then close it: :func:`kg_core.contracts.registry.field_parity`
compares the two and CI fails if they diverge. The alternative — generating
dataclasses from Avro at import time — costs readable type hints and IDE
completion everywhere in the codebase, which is a worse trade.

Avro remains the *source of truth on the wire*. These models are the in-process
shape.
"""

from __future__ import annotations

import enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from kg_core.ids import normalise_iban

# --------------------------------------------------------------- enums -----


class VopResult(enum.StrEnum):
    """Verification of Payee outcome (Reg. EU 2024/886, mandatory since 2025-10-09)."""

    MATCH = "MATCH"
    CLOSE_MATCH = "CLOSE_MATCH"
    NO_MATCH = "NO_MATCH"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNAVAILABLE = "UNAVAILABLE"


class Channel(enum.StrEnum):
    ONLINE_BANKING = "ONLINE_BANKING"
    MOBILE_APP = "MOBILE_APP"
    POS = "POS"
    STANDING_ORDER = "STANDING_ORDER"
    BATCH_UPLOAD = "BATCH_UPLOAD"
    OPEN_BANKING_API = "OPEN_BANKING_API"
    BRANCH = "BRANCH"


class InitiationSource(enum.StrEnum):
    ASPSP_DIRECT = "ASPSP_DIRECT"
    PISP = "PISP"
    AISP_TRIGGERED = "AISP_TRIGGERED"


class SettlementStatus(enum.StrEnum):
    ACCP = "ACCP"
    ACSC = "ACSC"
    RJCT = "RJCT"
    PDNG = "PDNG"


class SubjectType(enum.StrEnum):
    TRANSACTION = "TRANSACTION"
    ACCOUNT = "ACCOUNT"


class LabelValue(enum.StrEnum):
    CONFIRMED_FRAUD = "CONFIRMED_FRAUD"
    CONFIRMED_LAUNDERING = "CONFIRMED_LAUNDERING"
    SUSPICIOUS_UNPROVEN = "SUSPICIOUS_UNPROVEN"
    BENIGN = "BENIGN"
    UNKNOWN = "UNKNOWN"

    @property
    def is_positive(self) -> bool:
        """The binary target. SUSPICIOUS_UNPROVEN is deliberately NOT positive.

        Treating unproven suspicion as fraud is how a model learns to reproduce
        the incumbent system's alerting behaviour instead of detecting crime.
        """
        return self in (LabelValue.CONFIRMED_FRAUD, LabelValue.CONFIRMED_LAUNDERING)


class LabelSource(enum.StrEnum):
    ANALYST_REVIEW = "ANALYST_REVIEW"
    CUSTOMER_COMPLAINT = "CUSTOMER_COMPLAINT"
    CHARGEBACK = "CHARGEBACK"
    LEA_REQUEST = "LEA_REQUEST"
    FIU_FEEDBACK = "FIU_FEEDBACK"
    CORRESPONDENT_BANK = "CORRESPONDENT_BANK"


class Decision(enum.StrEnum):
    ALLOW = "ALLOW"
    ALERT = "ALERT"
    HOLD = "HOLD"
    BLOCK = "BLOCK"


class Verdict(enum.StrEnum):
    ESCALATE_SAR = "ESCALATE_SAR"
    ESCALATE_INTERNAL = "ESCALATE_INTERNAL"
    DISMISS_FALSE_POSITIVE = "DISMISS_FALSE_POSITIVE"
    NEEDS_MORE_INFO = "NEEDS_MORE_INFO"


# -------------------------------------------------------------- records ----

Cents = Annotated[int, Field(ge=0, description="Amount in minor units; never a float")]
Millis = Annotated[int, Field(description="Epoch milliseconds, UTC")]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class _Contract(BaseModel):
    """Base for wire records: strict, immutable, and enum-serialising."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


class SctInstEvent(_Contract):
    """One SEPA Instant Credit Transfer. The unit of everything downstream.

    design principle P1: *a transaction is not a row, it is an edge.* This model is
    the edge payload; ``debtor_account_id`` and ``creditor_account_id`` are its
    endpoints.
    """

    schema_version: int = 1
    tx_id: str
    end_to_end_id: str

    acceptance_ts: Millis
    ingest_ts: Millis

    debtor_account_id: str
    debtor_iban: str
    debtor_bic: str
    debtor_name: str
    debtor_country: str = "DE"

    creditor_account_id: str
    creditor_iban: str
    creditor_bic: str
    creditor_name: str
    creditor_country: str = "DE"

    creditor_name_claimed: str
    vop_result: VopResult = VopResult.MATCH
    vop_name_similarity: float = 1.0
    vop_overridden: bool = False

    amount_cents: Cents
    currency: str = "EUR"

    purpose_code: str | None = None
    remittance_info: str | None = None

    channel: Channel = Channel.ONLINE_BANKING
    initiation_source: InitiationSource = InitiationSource.ASPSP_DIRECT

    device_id: str | None = None
    session_ip_prefix: str | None = None
    is_new_beneficiary: bool = False

    settlement_status: SettlementStatus = SettlementStatus.ACSC

    @field_validator("debtor_iban", "creditor_iban", mode="before")
    @classmethod
    def _norm_iban(cls, value: str) -> str:
        return normalise_iban(value)

    @field_validator("ingest_ts")
    @classmethod
    def _ingest_not_before_acceptance(cls, value: int, info: ValidationInfo) -> int:
        """Ingest can lag acceptance; it can never precede it.

        A negative lag means a clock problem, and a clock problem in a system
        whose core invariant is point-in-time correctness is a stop-the-line
        defect, not a warning.
        """
        acceptance = info.data.get("acceptance_ts")
        if acceptance is not None and value < acceptance:
            raise ValueError(
                f"ingest_ts ({value}) precedes acceptance_ts ({acceptance}): "
                "an event cannot be observed before it happened"
            )
        return value

    @property
    def amount_eur(self) -> float:
        """Display helper only. Never use for arithmetic that feeds a feature."""
        return self.amount_cents / 100.0


class LabelOutcome(_Contract):
    """A confirmed outcome, arriving late and only for investigated cases."""

    schema_version: int = 1
    label_id: str
    subject_type: SubjectType
    subject_id: str
    event_ts: Millis
    decided_ts: Millis
    label: LabelValue
    source: LabelSource
    sar_filed: bool = False
    confidence: float = 1.0
    notes: str | None = None

    @property
    def verification_latency_ms(self) -> int:
        """How long the truth took to arrive. The quantity design principle P7 is about."""
        return self.decided_ts - self.event_ts


class LatencyBreakdown(_Contract):
    """Per-stage timings for one decision, in milliseconds."""

    feature_fetch: float = 0.0
    graph_assemble: float = 0.0
    inference: float = 0.0
    explain: float = 0.0
    counterfactual: float = 0.0
    total: float = 0.0


class SalientEdge(_Contract):
    """One transaction the explainer considers load-bearing for the score."""

    tx_id: str
    src_account_id: str
    dst_account_id: str
    ts: Millis
    importance: float
    hop: int = 1


class Counterfactual(_Contract):
    """The actionable half of the explanation.

    Not "feature 17 had high attribution" but "remove these transactions and
    the score falls below threshold" — a claim an investigator can check.
    """

    removed_tx_ids: list[str] = Field(default_factory=list)
    score_after: Probability = 0.0
    crosses_threshold: bool = False
    narrative_de: str = ""
    narrative_en: str = ""


class Explanation(_Contract):
    """The salient temporal subgraph plus its measured faithfulness."""

    edges: list[SalientEdge] = Field(default_factory=list)
    comprehensiveness: float | None = None
    sufficiency: float | None = None
    teacher_agreement: float | None = None
    counterfactual: Counterfactual | None = None


class Alert(_Contract):
    """A scoring decision. Also the GDPR Art. 22 decision-log record."""

    schema_version: int = 1
    alert_id: str
    tx_id: str
    scored_ts: Millis
    score: Probability
    threshold: Probability
    decision: Decision
    model_name: str
    model_version: str
    feature_set_hash: str
    code_commit: str
    latency_ms: LatencyBreakdown
    explanation: Explanation | None = None


class AnalystFeedback(_Contract):
    """Investigator response, published back onto Kafka as a training signal."""

    schema_version: int = 1
    feedback_id: str
    alert_id: str
    analyst_id: str
    submitted_ts: Millis
    verdict: Verdict
    explanation_helpful: bool | None = None
    explanation_rating: int | None = Field(default=None, ge=1, le=5)
    review_seconds: float | None = None
    rationale: str | None = None


#: Maps each Avro schema file stem to the model that mirrors it. The parity
#: test iterates this, so adding a contract without a mirror fails CI.
CONTRACT_MODELS: dict[str, type[_Contract]] = {
    "sct_inst_event.v1": SctInstEvent,
    "label_outcome.v1": LabelOutcome,
    "alert.v1": Alert,
    "analyst_feedback.v1": AnalystFeedback,
}
