"""Simulation configuration.

Design principle P9: the simulator is a feature, described honestly. Everything that
shapes the generated ecosystem lives here, in one serialisable object, so that a
dataset is a pure function of ``(config, seed, code commit)`` — which is the
premise ADR-0009 rests on when it replaces DVC with content hashing.

Calibration amplitudes are **approximations of public aggregates**, not licensed
microdata. Each carries its source in ``calibration.yaml``. Where a figure is a
judgement call rather than a published number, it says so.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from kg_core.hashing import content_hash

CALIBRATION_PATH = Path(__file__).parent / "calibration.yaml"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PopulationConfig(_Frozen):
    """How many of each agent archetype exist.

    Ratios matter more than absolute counts: a payment graph with too few
    merchants collapses into a star, and one with too many never forms the
    repeated debtor→creditor pairs that make a *temporal* model worth having.
    Defaults give roughly one merchant per 17 households, which keeps merchant
    in-degree high enough to be interesting without saturating.
    """

    households: int = Field(default=5_000, ge=10)
    merchants: int = Field(default=300, ge=2)
    smes: int = Field(default=150, ge=1)

    @property
    def total_accounts(self) -> int:
        return self.households + self.merchants + self.smes


class HorizonConfig(_Frozen):
    """The simulated period.

    Defaults start on 9 October 2025 — the day SEPA Instant became mandatory
    across the euro area, which is the regulatory event the whole project is a
    response to.
    """

    start_date: _dt.date = _dt.date(2025, 10, 9)
    days: int = Field(default=90, ge=7)

    @property
    def end_date(self) -> _dt.date:
        return self.start_date + _dt.timedelta(days=self.days)


class TypologyConfig(_Frozen):
    """How much crime to inject, and how much of it is evasive.

    Counts are *campaigns*, not transactions: one mule network is a coordinated
    structure spanning many accounts and days. The resulting positive rate lands
    near 0.1 %, which is the realistic order of magnitude and the reason
    design principle P6 rejects ROC-AUC as a headline.
    """

    mule_networks: int = Field(default=12, ge=0)
    smurfing_campaigns: int = Field(default=15, ge=0)
    app_fraud_cases: int = Field(default=40, ge=0)
    circular_flows: int = Field(default=8, ge=0)
    layering_chains: int = Field(default=10, ge=0)

    #: Share of campaigns generated in their evasive variant — deliberate dwell
    #: time, sub-threshold structuring, fan-out through freshly opened accounts.
    #: The evaluation protocol reports standard and evasive metrics separately,
    #: so this fraction directly controls how hard the adversarial table is.
    evasive_fraction: float = Field(default=0.35, ge=0.0, le=1.0)

    @property
    def total_campaigns(self) -> int:
        return (
            self.mule_networks
            + self.smurfing_campaigns
            + self.app_fraud_cases
            + self.circular_flows
            + self.layering_chains
        )


class LabelConfig(_Frozen):
    """The label-generating process — design principle P7, modelled explicitly.

    Real confirmed-laundering labels are late, incomplete and selection-biased.
    Rather than hand-waving that, the simulator *produces* those pathologies so
    they can be measured:

    - ``detection_probability`` encodes incompleteness: most crime is never
      confirmed, so most positives never receive a label at all.
    - ``incumbent_alert_recall`` encodes the selection bias: analyst review only
      ever happens on cases the *old* rules engine flagged, so labels are drawn
      from a biased subset rather than at random.
    - the latency parameters encode verification delay, which the protocol then
      simulates at 14/30/60 days.
    """

    #: P(a genuinely criminal transaction is ever confirmed as such).
    detection_probability: float = Field(default=0.35, ge=0.0, le=1.0)

    #: P(the incumbent rules engine flags a criminal case at all). Cases it
    #: misses are almost never investigated, and so almost never labelled.
    incumbent_alert_recall: float = Field(default=0.55, ge=0.0, le=1.0)

    #: P(a *legitimate* transaction is investigated and confirmed benign).
    #: Small, because nobody investigates ordinary payments — which is why the
    #: negative class is overwhelmingly unlabelled rather than confirmed.
    benign_confirmation_probability: float = Field(default=0.002, ge=0.0, le=1.0)

    #: Verification latency, lognormal in days.
    latency_days_median: float = Field(default=21.0, gt=0)
    latency_days_sigma: float = Field(default=0.8, gt=0)

    #: P(a confirmed case results in a Verdachtsmeldung to the FIU).
    sar_filing_probability: float = Field(default=0.7, ge=0.0, le=1.0)


class SimulationConfig(_Frozen):
    """The complete recipe for one synthetic ecosystem."""

    seed: int = 20251009
    population: PopulationConfig = PopulationConfig()
    horizon: HorizonConfig = HorizonConfig()
    typologies: TypologyConfig = TypologyConfig()
    labels: LabelConfig = LabelConfig()

    #: Behavioural amplitudes, loaded from calibration.yaml unless overridden.
    calibration: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _load_calibration(self) -> SimulationConfig:
        if not self.calibration:
            # frozen model: bypass the setattr guard for this one lazy default
            object.__setattr__(self, "calibration", load_calibration())
        return self

    @property
    def fingerprint(self) -> str:
        """Content hash of the entire recipe.

        Stamped onto the dataset manifest. Two datasets with the same
        fingerprint are byte-identical; two results computed against different
        fingerprints are not comparable, and the evaluation code refuses to
        pretend otherwise.
        """
        return content_hash(self.model_dump(mode="json"), prefix="kontograph-sim")

    def profile_summary(self) -> dict[str, Any]:
        """Human-readable shape of what this config will generate."""
        return {
            "accounts": self.population.total_accounts,
            "days": self.horizon.days,
            "start": self.horizon.start_date.isoformat(),
            "end": self.horizon.end_date.isoformat(),
            "campaigns": self.typologies.total_campaigns,
            "evasive_fraction": self.typologies.evasive_fraction,
            "seed": self.seed,
            "fingerprint": self.fingerprint,
        }


def load_calibration(path: Path | None = None) -> dict[str, Any]:
    """Load the calibration amplitudes."""
    return yaml.safe_load((path or CALIBRATION_PATH).read_text(encoding="utf-8"))


# Named profiles. `kg sim generate --profile smoke` needs to finish in seconds
# during development; `full` is the dataset that goes to Kaggle.
PROFILES: dict[str, SimulationConfig] = {
    "smoke": SimulationConfig(
        population=PopulationConfig(households=200, merchants=25, smes=12),
        # 35 days, not 14: the window must span a month boundary or it contains
        # no payroll and no rent -- the two strongest periodic patterns in the
        # data, and the ones the confusable-fan-out story depends on.
        horizon=HorizonConfig(days=35),
        typologies=TypologyConfig(
            mule_networks=2,
            smurfing_campaigns=2,
            app_fraud_cases=4,
            circular_flows=1,
            layering_chains=2,
        ),
    ),
    "dev": SimulationConfig(
        population=PopulationConfig(households=1_000, merchants=80, smes=40),
        horizon=HorizonConfig(days=30),
        typologies=TypologyConfig(
            mule_networks=5,
            smurfing_campaigns=6,
            app_fraud_cases=15,
            circular_flows=3,
            layering_chains=4,
        ),
    ),
    # 270 days, not 90, and campaigns scaled with it to hold the base rate.
    #
    # Not an arbitrary choice. With a realistic 0.18% base rate, 15% label
    # coverage and a 21-day median verification latency, a 90-day horizon leaves
    # roughly FIFTY matured labelled positives in the training fold — the model
    # loses to a constant predictor, and does so for a completely legitimate
    # reason. Labels for events late in the training window have not been
    # decided by the time the split boundary arrives.
    #
    # The honest fix is more history, not a higher fraud rate or a shorter
    # verification delay. Those would make the numbers look better by making the
    # problem easier than the real one. A bank has years of history; 90 days was
    # the artificial constraint.
    "full": SimulationConfig(
        horizon=HorizonConfig(days=270),
        typologies=TypologyConfig(
            mule_networks=36,
            smurfing_campaigns=45,
            app_fraud_cases=120,
            circular_flows=24,
            layering_chains=30,
        ),
    ),
}


def get_profile(name: str) -> SimulationConfig:
    """Look up a named profile, listing the alternatives on a miss."""
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(f"unknown profile {name!r}; available: {sorted(PROFILES)}") from None
