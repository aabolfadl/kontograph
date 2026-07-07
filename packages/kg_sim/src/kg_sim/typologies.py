"""Financial-crime typologies, injected under known ground truth.

Design principle P9: *"it gives us something no public dataset does: the true causal
structure of the fraud."* That is what this module produces. Every criminal
transaction is emitted together with the campaign it belongs to, the role the
account plays in it, and its position in the chain — so an explanation can be
scored against the actual mechanism rather than against a proxy.

**Each typology has an evasive variant**, and the difference is not cosmetic.
The evasive variants attack the specific signals the standard variants leak:

| Standard leaks | Evasive countermeasure |
|---|---|
| burst velocity (minutes between hops) | dwell time of hours to days |
| fresh accounts | aged or compromised legitimate accounts |
| round amounts | amounts drawn from the legitimate distribution |
| amounts near the reporting threshold | amounts far below it, more of them |
| empty remittance text | plausible invoice references |

The evaluation protocol reports standard and evasive metrics separately, so the
degradation between them is a published number.

**Ground truth never touches the event.** Tags are returned alongside events in
a parallel structure. The wire contract has no label field and a test enforces
that, which is what makes leakage structurally impossible rather than merely
avoided.
"""

from __future__ import annotations

import datetime as _dt
import enum
from dataclasses import dataclass

import numpy as np

from kg_core.contracts import Channel
from kg_sim.config import SimulationConfig
from kg_sim.economy import Clock, RawEvent
from kg_sim.names import company_name, introduce_typo, person_name
from kg_sim.population import AccountKind, Population


class Typology(enum.StrEnum):
    """The five injected patterns."""

    MULE_NETWORK = "MULE_NETWORK"
    SMURFING = "SMURFING"
    APP_FRAUD = "APP_FRAUD"
    CIRCULAR_FLOW = "CIRCULAR_FLOW"
    LAYERING_CHAIN = "LAYERING_CHAIN"


class Role(enum.StrEnum):
    """What an account is doing inside a campaign.

    Roles matter for explanation evaluation: an explainer that highlights the
    collector is more useful than one that highlights a random victim, even if
    both edges belong to the campaign.
    """

    VICTIM = "VICTIM"  # defrauded, did not consent to the loss
    SOURCE = "SOURCE"  # supplies the value, knowingly or not
    COLLECTOR = "COLLECTOR"  # concentrates inbound value
    MULE = "MULE"  # receives and forwards, usually recruited
    HOP = "HOP"  # intermediate node in a layering chain
    SMURF = "SMURF"  # makes sub-threshold deposits
    BENEFICIARY = "BENEFICIARY"  # final destination of the value
    ORIGINATOR = "ORIGINATOR"  # instigates the campaign


@dataclass(slots=True, frozen=True)
class TypologyTag:
    """Ground truth for one criminal transaction.

    Stored parallel to the event, never inside it.
    """

    campaign_id: str
    typology: Typology
    src_role: Role
    dst_role: Role
    hop: int
    is_evasive: bool


#: One criminal payment plus its ground truth.
TaggedEvent = tuple[RawEvent, TypologyTag]


def _amount_cents(rng: np.random.Generator, low: float, high: float) -> int:
    return round(float(rng.uniform(low, high)) * 100)


def _round_amount_cents(rng: np.random.Generator, low: float, high: float) -> int:
    """A conspicuously round figure — the standard-variant tell."""
    step = int(rng.choice([50, 100, 500, 1000]))
    value = int(rng.uniform(low, high) / step) * step
    return max(step, value) * 100


class TypologyEngine:
    """Generates criminal campaigns against an existing population."""

    def __init__(self, config: SimulationConfig, population: Population) -> None:
        self.config = config
        self.population = population
        self.cal = config.calibration
        self.clock = Clock(self.cal)
        self.thresholds = self.cal["thresholds"]
        self._campaign_counter = 0

    # ------------------------------------------------------------- driver --

    def run(self, rng: np.random.Generator) -> list[TaggedEvent]:
        """Inject every configured campaign."""
        spec = self.config.typologies
        tagged: list[TaggedEvent] = []
        builders = (
            (spec.mule_networks, self._mule_network),
            (spec.smurfing_campaigns, self._smurfing),
            (spec.app_fraud_cases, self._app_fraud),
            (spec.circular_flows, self._circular_flow),
            (spec.layering_chains, self._layering_chain),
        )
        for count, builder in builders:
            for _ in range(count):
                evasive = bool(rng.random() < spec.evasive_fraction)
                tagged.extend(builder(rng, evasive))
        return tagged

    # ------------------------------------------------------------ helpers --

    def _campaign_id(self, typology: Typology, evasive: bool) -> str:
        self._campaign_counter += 1
        suffix = "E" if evasive else "S"
        return f"{typology.value[:4]}-{suffix}-{self._campaign_counter:04d}"

    def _random_day(self, rng: np.random.Generator, *, margin: int = 0) -> _dt.date:
        """A start date leaving *margin* days of room for the campaign to run."""
        span = max(1, self.config.horizon.days - margin)
        return self.config.horizon.start_date + _dt.timedelta(days=int(rng.integers(span)))

    def _ts(self, day: _dt.date, rng: np.random.Generator, *, business: bool = False) -> int:
        return self.clock.stamp(day, rng, business=business)

    def _advance(
        self, ts: int, rng: np.random.Generator, *, evasive: bool, base_minutes: float
    ) -> int:
        """Time between consecutive hops.

        The standard variant moves in minutes, which is the burst-velocity tell
        every rules engine already looks for. The evasive variant deliberately
        dwells for hours or days, which defeats velocity rules entirely and is
        the reason a model with temporal *memory* is needed rather than a
        fixed-window aggregate.
        """
        if evasive:
            hours = float(rng.gamma(shape=2.0, scale=9.0)) + 1.0
            return ts + int(hours * 3_600_000)
        minutes = float(rng.gamma(shape=2.0, scale=base_minutes / 2.0)) + 0.5
        return ts + int(minutes * 60_000)

    def _recruit(self, rng: np.random.Generator, count: int, *, evasive: bool) -> list[int]:
        """Obtain accounts for a network to use.

        Standard campaigns open fresh accounts, which is fast but leaves account
        age as a giveaway. Evasive campaigns instead *compromise existing
        legitimate households* — no new account, ordinary age, real transaction
        history. That is materially harder to detect and is what actually
        happens in mule recruitment.
        """
        # Distinct by construction. Evasive campaigns compromise *existing*
        # accounts, so the same account can be drawn twice; where adjacency
        # matters (a ring, a chain) that produces a payment from an account to
        # itself, which is not a transfer between two parties.
        accounts: list[int] = []
        seen: set[int] = set()
        attempts = 0
        while len(accounts) < count and attempts < count * 20:
            attempts += 1
            if evasive and self.population.households and rng.random() < 0.7:
                household = self.population.households[
                    int(rng.integers(len(self.population.households)))
                ]
                candidate = household.account_index
            else:
                candidate = self.population.open_account(
                    AccountKind.MULE, person_name(rng), rng, fresh=not evasive
                )
            if candidate not in seen:
                seen.add(candidate)
                accounts.append(candidate)
        return accounts

    # ------------------------------------------------------- mule network --

    def _mule_network(self, rng: np.random.Generator, evasive: bool) -> list[TaggedEvent]:
        """Fan-in to a collector, then fan-out to mules, then consolidation.

        The shape that motivates the whole project: no single transaction here
        is remarkable on its own. A €900 transfer between two private accounts
        is invisible in isolation. It is only anomalous *given who these parties
        are to each other and to everyone else* — design principle P1.
        """
        campaign = self._campaign_id(Typology.MULE_NETWORK, evasive)
        pop = self.population
        events: list[TaggedEvent] = []

        n_sources = int(rng.integers(6, 18))
        n_mules = int(rng.integers(4, 11))
        sources = [
            pop.households[int(rng.integers(len(pop.households)))].account_index
            for _ in range(n_sources)
        ]
        collector = self._recruit(rng, 1, evasive=evasive)[0]
        mules = self._recruit(rng, n_mules, evasive=evasive)
        beneficiary = (
            int(rng.choice(pop.foreign))
            if pop.foreign and rng.random() < 0.6
            else self._recruit(rng, 1, evasive=evasive)[0]
        )

        day = self._random_day(rng, margin=7 if not evasive else 20)
        ts = self._ts(day, rng)

        # --- fan-in ---------------------------------------------------------
        collected = 0
        for source in sources:
            amount = (
                _amount_cents(rng, 200, 4_500) if evasive else _round_amount_cents(rng, 500, 5_000)
            )
            collected += amount
            events.append(
                (
                    RawEvent(
                        ts=ts,
                        src_index=source,
                        dst_index=collector,
                        amount_cents=amount,
                        category="mule_fan_in",
                        channel=Channel.MOBILE_APP,
                        remittance=_criminal_remittance(rng, evasive),
                    ),
                    TypologyTag(
                        campaign, Typology.MULE_NETWORK, Role.SOURCE, Role.COLLECTOR, 0, evasive
                    ),
                )
            )
            ts = self._advance(ts, rng, evasive=evasive, base_minutes=25)

        # --- fan-out --------------------------------------------------------
        # Split the collected value across the mules, keeping a margin so the
        # collector is not emptied to exactly zero — an exact drain is itself a
        # tell, and a careful launderer leaves residue.
        share = int(collected * float(rng.uniform(0.82, 0.94)) / max(1, n_mules))
        for hop_index, mule in enumerate(mules, start=1):
            amount = int(share * float(rng.uniform(0.85, 1.15)))
            if amount <= 0:
                continue
            events.append(
                (
                    RawEvent(
                        ts=ts,
                        src_index=collector,
                        dst_index=mule,
                        amount_cents=amount,
                        category="mule_fan_out",
                        channel=Channel.ONLINE_BANKING,
                        remittance=_criminal_remittance(rng, evasive),
                    ),
                    TypologyTag(
                        campaign,
                        Typology.MULE_NETWORK,
                        Role.COLLECTOR,
                        Role.MULE,
                        hop_index,
                        evasive,
                    ),
                )
            )
            ts = self._advance(ts, rng, evasive=evasive, base_minutes=12)

        # --- consolidation --------------------------------------------------
        for mule in mules:
            amount = int(share * float(rng.uniform(0.80, 0.95)))
            if amount <= 0:
                continue
            events.append(
                (
                    RawEvent(
                        ts=ts,
                        src_index=mule,
                        dst_index=beneficiary,
                        amount_cents=amount,
                        category="mule_consolidation",
                        channel=Channel.ONLINE_BANKING,
                        remittance=_criminal_remittance(rng, evasive),
                    ),
                    TypologyTag(
                        campaign, Typology.MULE_NETWORK, Role.MULE, Role.BENEFICIARY, 2, evasive
                    ),
                )
            )
            ts = self._advance(ts, rng, evasive=evasive, base_minutes=18)

        return events

    # ------------------------------------------------------------ smurfing --

    def _smurfing(self, rng: np.random.Generator, evasive: bool) -> list[TaggedEvent]:
        """Structuring: one large sum split to stay under a reporting threshold.

        The standard variant crowds just below €10 000, which is detectable with
        a histogram. The evasive variant sits far below the threshold and uses
        more smurfs — defeating threshold rules at the cost of more edges, which
        is precisely the trade a *graph* model is positioned to catch.
        """
        campaign = self._campaign_id(Typology.SMURFING, evasive)
        events: list[TaggedEvent] = []
        threshold = float(self.thresholds["cash_report_eur"])

        total = float(rng.uniform(45_000, 220_000))
        if evasive:
            # Well clear of the threshold: a launderer who has read the rules.
            per_payment = float(rng.uniform(0.15, 0.45)) * threshold
        else:
            # Hugging the threshold: the classic, and the detectable, pattern.
            per_payment = float(rng.uniform(0.88, 0.985)) * threshold

        n_payments = max(3, int(total / per_payment))
        n_smurfs = int(rng.integers(3, 9))
        smurfs = self._recruit(rng, n_smurfs, evasive=evasive)
        destination = self._recruit(rng, 1, evasive=evasive)[0]

        day = self._random_day(rng, margin=self.config.horizon.days // 3)
        ts = self._ts(day, rng)

        for index in range(n_payments):
            amount = round(per_payment * float(rng.uniform(0.93, 1.0)) * 100)
            events.append(
                (
                    RawEvent(
                        ts=ts,
                        src_index=smurfs[index % n_smurfs],
                        dst_index=destination,
                        amount_cents=amount,
                        category="smurfing",
                        channel=Channel.ONLINE_BANKING,
                        remittance=_criminal_remittance(rng, evasive),
                    ),
                    TypologyTag(
                        campaign, Typology.SMURFING, Role.SMURF, Role.COLLECTOR, index, evasive
                    ),
                )
            )
            if evasive:
                # Spread across days so no single day looks unusual.
                day = day + _dt.timedelta(days=int(rng.integers(0, 3)))
                if day >= self.config.horizon.end_date:
                    break
                ts = self._ts(day, rng)
            else:
                ts = self._advance(ts, rng, evasive=False, base_minutes=45)
        return events

    # ----------------------------------------------------------- APP fraud --

    def _app_fraud(self, rng: np.random.Generator, evasive: bool) -> list[TaggedEvent]:
        """Authorised push payment fraud.

        The typology Verification of Payee was introduced to stop, and the one
        where VoP features earn their place. A socially engineered victim sends
        a large payment to a brand-new beneficiary whose account name does not
        match what they were told.

        The evasive variant is the important one: the fraudster registers an
        account under a name *close* to a real business, so VoP returns
        CLOSE_MATCH rather than NO_MATCH — which is indistinguishable from an
        honest typo, and is why VoP cannot be used as a rule.
        """
        campaign = self._campaign_id(Typology.APP_FRAUD, evasive)
        pop = self.population
        events: list[TaggedEvent] = []

        victim = pop.households[int(rng.integers(len(pop.households)))]
        # Victims are drained relative to their own means, not to an absolute
        # figure — a €40 000 transfer is a catastrophe for one household and a
        # routine week for another, and the model has to learn that difference.
        multiple = float(rng.uniform(1.5, 8.0))
        amount = int(min(victim.monthly_income_cents * multiple, 90_000_00))

        # The fraudster impersonates a real business the victim plausibly deals
        # with. This is what gives Verification of Payee something to compare,
        # and it is where the standard and evasive variants diverge most sharply:
        #
        #   standard -> the account name is unrelated to the impersonated business,
        #               so VoP returns NO_MATCH and the warning is explicit
        #   evasive  -> the account is registered under a near-identical name, so
        #               VoP returns CLOSE_MATCH, which is indistinguishable from
        #               an honest typo and is exactly why VoP cannot be a rule
        impersonation_pool = [m.account_index for m in pop.merchants] + [
            s.account_index for s in pop.smes
        ]
        target_index = int(rng.choice(impersonation_pool))
        claimed_name = pop.accounts[target_index].holder_name

        if evasive:
            fraudster_name = introduce_typo(claimed_name, rng)
        else:
            fraudster_name = company_name(rng) if rng.random() < 0.5 else person_name(rng)

        fraudster = pop.open_account(AccountKind.MULE, fraudster_name, rng, fresh=not evasive)

        day = self._random_day(rng, margin=3)
        ts = self._ts(day, rng)
        events.append(
            (
                RawEvent(
                    ts=ts,
                    src_index=victim.account_index,
                    dst_index=fraudster,
                    amount_cents=amount,
                    category="app_fraud",
                    channel=Channel.MOBILE_APP if rng.random() < 0.7 else Channel.ONLINE_BANKING,
                    remittance=str(
                        rng.choice(
                            [
                                "Sicherheitsuebertrag",
                                "Rechnung dringend",
                                "Anzahlung",
                                "Handwerker Rechnung",
                                "Kaution",
                                "Investition",
                            ]
                        )
                    ),
                    claimed_name=claimed_name,
                ),
                TypologyTag(
                    campaign, Typology.APP_FRAUD, Role.VICTIM, Role.BENEFICIARY, 0, evasive
                ),
            )
        )

        # The proceeds are moved on within the settlement window — the whole
        # reason a 10-second irrevocable rail changes the threat model.
        remaining, hop = amount, 1
        onward = self._recruit(rng, int(rng.integers(2, 5)), evasive=evasive)
        previous = fraudster
        for destination in onward:
            ts = self._advance(ts, rng, evasive=evasive, base_minutes=6)
            slice_amount = int(remaining * float(rng.uniform(0.3, 0.6)))
            if slice_amount < 1000:
                break
            remaining -= slice_amount
            events.append(
                (
                    RawEvent(
                        ts=ts,
                        src_index=previous,
                        dst_index=destination,
                        amount_cents=slice_amount,
                        category="app_fraud_dispersal",
                        channel=Channel.ONLINE_BANKING,
                        remittance=_criminal_remittance(rng, evasive),
                    ),
                    TypologyTag(
                        campaign, Typology.APP_FRAUD, Role.BENEFICIARY, Role.MULE, hop, evasive
                    ),
                )
            )
            previous, hop = destination, hop + 1
        return events

    # ------------------------------------------------------ circular flow --

    def _circular_flow(self, rng: np.random.Generator, evasive: bool) -> list[TaggedEvent]:
        """Value that returns to its origin, manufacturing apparent turnover.

        Used to give a shell company a credible trading history — for a loan
        application, or to justify later inflows. The legitimate economy also
        contains cycles (merchant → SME supplier → employee → merchant), which
        is why a cycle detector alone is not enough and why this is worth
        including as a distinct typology.
        """
        campaign = self._campaign_id(Typology.CIRCULAR_FLOW, evasive)
        pop = self.population
        events: list[TaggedEvent] = []

        length = int(rng.integers(3, 7)) if not evasive else int(rng.integers(5, 10))
        ring = self._recruit(rng, length, evasive=evasive)
        # Anchor the ring on a real SME so it has legitimate traffic around it,
        # unless that SME is already in the ring -- which would collapse two
        # positions into one and create a self-edge.
        if pop.smes and rng.random() < 0.6:
            anchor = pop.smes[int(rng.integers(len(pop.smes)))].account_index
            if anchor not in ring[1:]:
                ring[0] = anchor

        amount = int(float(rng.uniform(8_000, 60_000)) * 100)
        day = self._random_day(rng, margin=length + 2)
        ts = self._ts(day, rng, business=True)

        for position in range(length):
            src = ring[position]
            dst = ring[(position + 1) % length]
            # Value decays around the ring: each hop skims a margin, so the
            # returning amount is smaller than the outgoing one. An exactly
            # conserved cycle would be trivially detectable.
            decay = (
                float(rng.uniform(0.90, 0.985)) if not evasive else float(rng.uniform(0.94, 0.998))
            )
            amount = int(amount * decay)
            events.append(
                (
                    RawEvent(
                        ts=ts,
                        src_index=src,
                        dst_index=dst,
                        amount_cents=amount,
                        category="circular_flow",
                        channel=Channel.ONLINE_BANKING,
                        remittance=f"Rechnung {day.year}-{int(rng.integers(1000, 9999))}",
                    ),
                    TypologyTag(
                        campaign, Typology.CIRCULAR_FLOW, Role.HOP, Role.HOP, position, evasive
                    ),
                )
            )
            if evasive:
                day = day + _dt.timedelta(days=int(rng.integers(1, 4)))
                if day >= self.config.horizon.end_date:
                    break
                ts = self._ts(day, rng, business=True)
            else:
                ts = self._advance(ts, rng, evasive=False, base_minutes=90)
        return events

    # ----------------------------------------------------- layering chain --

    def _layering_chain(self, rng: np.random.Generator, evasive: bool) -> list[TaggedEvent]:
        """A rapid sequential chain, usually terminating cross-border.

        This is the typology the *explanation* work is aimed at. When an alert
        fires on the last hop, the useful answer is not "this account is risky"
        but "this payment is the seventh hop of a chain that started here" —
        which is a subgraph, and is exactly what a counterfactual can act on:
        remove two hops through one intermediary and the score collapses.
        """
        campaign = self._campaign_id(Typology.LAYERING_CHAIN, evasive)
        pop = self.population
        events: list[TaggedEvent] = []

        length = int(rng.integers(4, 8)) if not evasive else int(rng.integers(6, 12))
        chain = self._recruit(rng, length, evasive=evasive)
        if pop.foreign and rng.random() < 0.75:
            chain.append(int(rng.choice(pop.foreign)))

        amount = int(float(rng.uniform(20_000, 150_000)) * 100)
        day = self._random_day(rng, margin=3 if not evasive else 15)
        ts = self._ts(day, rng)

        origin = chain[0]
        for hop_index in range(len(chain) - 1):
            src, dst = chain[hop_index], chain[hop_index + 1]
            # Each hop skims a fee. Standard chains skim visibly; evasive ones
            # skim little, so the amount trail is harder to follow.
            amount = int(
                amount
                * float(rng.uniform(0.88, 0.97) if not evasive else rng.uniform(0.965, 0.995))
            )
            src_role = Role.ORIGINATOR if src == origin else Role.HOP
            dst_role = Role.BENEFICIARY if hop_index == len(chain) - 2 else Role.HOP
            events.append(
                (
                    RawEvent(
                        ts=ts,
                        src_index=src,
                        dst_index=dst,
                        amount_cents=amount,
                        category="layering",
                        channel=Channel.ONLINE_BANKING,
                        remittance=_criminal_remittance(rng, evasive),
                    ),
                    TypologyTag(
                        campaign, Typology.LAYERING_CHAIN, src_role, dst_role, hop_index, evasive
                    ),
                )
            )
            ts = self._advance(ts, rng, evasive=evasive, base_minutes=4)
        return events


def _criminal_remittance(rng: np.random.Generator, evasive: bool) -> str:
    """Remittance text on a criminal payment.

    Standard campaigns leave it blank or uniform — a genuine and well-known AML
    red flag. Evasive campaigns fabricate plausible invoice references, which
    removes the signal and forces the model to rely on structure instead.
    """
    if evasive:
        return str(
            rng.choice(
                [
                    f"RE-{int(rng.integers(10000, 99999))}",
                    f"Rechnung {int(rng.integers(1000, 9999))}",
                    "Beratungsleistung",
                    "Warenlieferung",
                    f"Auftrag {int(rng.integers(100, 999))}",
                    "Teilzahlung",
                ]
            )
        )
    return str(rng.choice(["", "", "Transfer", "Ueberweisung", ""]))
