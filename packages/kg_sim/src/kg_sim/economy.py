"""Legitimate payment behaviour.

This module is where most of the project's realism budget goes, and it is worth
saying why: **the fraud is only as interesting as the background it hides in.**

A simulator that emits uniform random payments makes every typology trivially
separable — fan-out, velocity and round amounts all become giveaways, the model
scores 0.99 PR-AUC, and the number means nothing. So the legitimate economy
deliberately generates the *confusable* cases:

- payroll fan-out on the last banking day, which looks like mule fan-out
- utility providers with enormous in-degree, which looks like a collector
- salary→rent chains hours apart, which look like layering
- large one-off purchases from long-tail income households, which look like
  structuring failures

Every one of those is a false positive waiting to happen, and that is the point.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any

import numpy as np

from kg_core.contracts import Channel
from kg_core.timeutil import UTC, to_millis
from kg_sim.calendar_de import (
    is_business_day,
    is_weekend,
    last_business_day_of_month,
    seasonal_spending_multiplier,
)
from kg_sim.config import SimulationConfig
from kg_sim.population import Population

#: ISO 20022 ExternalPurpose1Code values, mapped to our spend categories.
PURPOSE_CODES: dict[str, str] = {
    "salary": "SALA",
    "pension": "PENS",
    "rent": "RENT",
    "utilities": "UBIL",
    "insurance": "INSU",
    "subscription": "SCVE",
    "groceries": "GDDS",
    "retail": "GDDS",
    "ecommerce": "GDDS",
    "dining": "SCVE",
    "p2p": "OTHR",
    "supplier": "SUPP",
}


@dataclass(slots=True)
class RawEvent:
    """A payment before it is dressed as a contract record.

    Deliberately minimal and *free of any label*: the typology machinery adds
    ground truth in a separate structure keyed by ``tx_id``, so a criminal
    transaction and a legitimate one are indistinguishable at this layer. That
    is a structural guarantee, not a convention — see the wire-contract leakage
    test in ``tests/unit/test_contracts.py``.
    """

    ts: int
    src_index: int
    dst_index: int
    amount_cents: int
    category: str
    channel: Channel
    remittance: str

    #: The payee name the payer *typed*, when it is not simply derived from the
    #: creditor's account name. Only impersonation sets this: in APP fraud the
    #: victim types the name of the business they believe they are paying, which
    #: is precisely the comparison Verification of Payee performs.
    claimed_name: str | None = None


class Clock:
    """Turns a date plus a behavioural profile into an intraday timestamp.

    Under SCT Inst there is no nightly batch window, so intraday shape survives
    into the data instead of being flattened into an overnight run. That is
    precisely the structure a continuous-time model can exploit and a snapshot
    model averages away (ADR-0006), so it is worth generating properly.
    """

    def __init__(self, calibration: dict[str, Any]) -> None:
        diurnal = calibration["diurnal"]
        self._consumer = np.asarray(diurnal["consumer_hour_weights"], dtype=float)
        self._consumer /= self._consumer.sum()
        self._business = np.asarray(diurnal["business_hour_weights"], dtype=float)
        self._business /= self._business.sum()

    def stamp(self, day: _dt.date, rng: np.random.Generator, *, business: bool = False) -> int:
        """A UTC millisecond timestamp somewhere inside *day*."""
        weights = self._business if business else self._consumer
        hour = int(rng.choice(24, p=weights))
        minute, second = int(rng.integers(60)), int(rng.integers(60))
        millis = int(rng.integers(1000))
        moment = _dt.datetime.combine(
            day, _dt.time(hour, minute, second, millis * 1000), tzinfo=UTC
        )
        return to_millis(moment)


def _amount(rng: np.random.Generator, spec: dict[str, Any], scale: float = 1.0) -> int:
    value = rng.lognormal(mean=np.log(spec["median"]), sigma=spec["sigma"]) * scale
    return round(float(np.clip(value, spec["min"], spec["max"])) * 100)


def _consumer_channel(
    rng: np.random.Generator, mix: dict[str, float], online_only: bool
) -> Channel:
    """Pick an initiation channel.

    An online-only merchant can never be paid at a point of sale, so the POS
    share is redistributed rather than silently produced — a POS payment to a
    pure e-commerce merchant would be an impossible row that a data-quality
    check should catch, and shipping impossible rows undermines the whole
    quality-gate story.
    """
    names = list(mix)
    weights = np.asarray([mix[name] for name in names], dtype=float)
    if online_only and "POS" in names:
        weights[names.index("POS")] = 0.0
    weights /= weights.sum()
    return Channel(str(rng.choice(names, p=weights)))


class Economy:
    """Generates the legitimate transaction stream day by day."""

    def __init__(self, config: SimulationConfig, population: Population) -> None:
        self.config = config
        self.population = population
        self.cal = config.calibration
        self.clock = Clock(self.cal)
        self._channel_mix = self.cal["channel_mix"]
        self._paid_months: set[tuple[str, int, int]] = set()

    # ------------------------------------------------------------- driver --

    def run(self, rng: np.random.Generator) -> list[RawEvent]:
        """Generate every legitimate payment across the horizon."""
        events: list[RawEvent] = []
        start = self.config.horizon.start_date
        for offset in range(self.config.horizon.days):
            day = start + _dt.timedelta(days=offset)
            events.extend(self._day(day, rng))
        return events

    def _day(self, day: _dt.date, rng: np.random.Generator) -> list[RawEvent]:
        events: list[RawEvent] = []
        events.extend(self._payroll(day, rng))
        events.extend(self._rent(day, rng))
        events.extend(self._recurring_bills(day, rng))
        events.extend(self._consumer_spending(day, rng))
        events.extend(self._business_payments(day, rng))
        return events

    # ------------------------------------------------------------ payroll --

    def _payroll(self, day: _dt.date, rng: np.random.Generator) -> list[RawEvent]:
        """Salaries, on the last banking day of the month.

        This produces the single largest fan-out in the dataset: one employer
        paying dozens of accounts within a few hours. Structurally it is a mule
        fan-out. The difference is that it repeats every month to the same set,
        the amounts are stable, and the recipients then behave like people. A
        model that cannot tell these apart will alert on every employer in
        Germany on the 30th — which is exactly the failure the simulator exists
        to expose.
        """
        if day != last_business_day_of_month(day.year, day.month):
            return []

        events: list[RawEvent] = []
        pop = self.population
        for household in pop.households:
            if household.employer_index is None:
                continue
            # Small month-to-month variation: overtime, bonuses, deductions.
            gross = int(household.monthly_income_cents * float(rng.normal(1.0, 0.035)))
            events.append(
                RawEvent(
                    ts=self.clock.stamp(day, rng, business=True),
                    src_index=household.employer_index,
                    dst_index=household.account_index,
                    amount_cents=max(gross, 50_000),
                    category="salary",
                    channel=Channel.BATCH_UPLOAD,
                    remittance=f"Lohn/Gehalt {day.month:02d}/{day.year}",
                )
            )

        # Households with no employer still receive income — pensions, grants,
        # invoices. Same day-of-month clustering, different counterparty shape.
        for household in pop.households:
            if household.employer_index is not None:
                continue
            payer = int(rng.choice(pop.utilities + [s.account_index for s in pop.smes]))
            events.append(
                RawEvent(
                    ts=self.clock.stamp(day, rng, business=True),
                    src_index=payer,
                    dst_index=household.account_index,
                    amount_cents=int(household.monthly_income_cents * float(rng.normal(1.0, 0.06))),
                    category="pension",
                    channel=Channel.BATCH_UPLOAD,
                    remittance=f"Rente/Bezuege {day.month:02d}/{day.year}",
                )
            )
        return events

    # --------------------------------------------------------------- rent --

    def _rent(self, day: _dt.date, rng: np.random.Generator) -> list[RawEvent]:
        """Rent, in the first few days of the month.

        Modelled as a standing order to a *stable* counterparty. This is the
        strongest repeated edge in the graph: the same debtor→creditor pair,
        the same amount, every month. It is the thing a temporal model should
        learn to expect, and its absence or change is informative.
        """
        low, high = self.cal["household"]["rent_day_range"]
        if not (low <= day.day <= high):
            return []

        events: list[RawEvent] = []
        for household in self.population.households:
            if not household.is_renter or household.landlord_index is None:
                continue
            # Each household pays on its own fixed day within the window,
            # derived from its account index so it does not drift month to month.
            if household.account_index % (high - low + 1) != (day.day - low):
                continue
            events.append(
                RawEvent(
                    ts=self.clock.stamp(day, rng),
                    src_index=household.account_index,
                    dst_index=household.landlord_index,
                    amount_cents=household.rent_cents,
                    category="rent",
                    channel=Channel.STANDING_ORDER,
                    remittance=f"Miete {day.month:02d}/{day.year}",
                )
            )
        return events

    # ------------------------------------------------------ utilities etc. --

    def _recurring_bills(self, day: _dt.date, rng: np.random.Generator) -> list[RawEvent]:
        """Utilities, insurance and subscriptions.

        Utility providers accumulate very high in-degree — thousands of
        households paying the same Stadtwerke. Any pure degree- or
        in-degree-based heuristic will flag them, which is the point: they are
        the legitimate high-degree nodes a graph model must learn to excuse.
        """
        low, high = self.cal["household"]["utilities_day_range"]
        if not (low <= day.day <= high):
            return []

        amounts = self.cal["household"]["amounts"]
        events: list[RawEvent] = []
        span = high - low + 1
        for household in self.population.households:
            if household.account_index % span != (day.day - low):
                continue
            events.append(
                RawEvent(
                    ts=self.clock.stamp(day, rng),
                    src_index=household.account_index,
                    dst_index=household.utility_index,
                    amount_cents=_amount(rng, amounts["utilities"]),
                    category="utilities",
                    channel=Channel.STANDING_ORDER,
                    remittance=f"Abschlag {day.month:02d}/{day.year} KdNr {household.account_index:06d}",
                )
            )
            if rng.random() < 0.55:
                events.append(
                    RawEvent(
                        ts=self.clock.stamp(day, rng),
                        src_index=household.account_index,
                        dst_index=int(rng.choice(self.population.utilities)),
                        amount_cents=_amount(rng, amounts["insurance"]),
                        category="insurance",
                        channel=Channel.STANDING_ORDER,
                        remittance=f"Versicherungsbeitrag {day.month:02d}/{day.year}",
                    )
                )
        return events

    # -------------------------------------------------- consumer spending --

    def _consumer_spending(self, day: _dt.date, rng: np.random.Generator) -> list[RawEvent]:
        """Groceries, retail, e-commerce, dining and P2P.

        Poisson counts are drawn for the whole population at once, then expanded
        — with 5 000 households over 90 days a per-household Python loop per
        category would dominate the runtime for no benefit.
        """
        pop = self.population
        cal = self.cal["household"]
        rates, amounts = cal["weekly_rates"], cal["amounts"]
        seasonal = seasonal_spending_multiplier(day)
        weekend = self.cal["diurnal"]["weekend_consumer_multiplier"] if is_weekend(day) else 1.0
        scales = np.asarray([h.activity_scale for h in pop.households], dtype=float)

        events: list[RawEvent] = []
        for category, weekly_rate in rates.items():
            lam = (weekly_rate / 7.0) * scales * seasonal * weekend
            counts = rng.poisson(lam)
            for position in np.nonzero(counts)[0]:
                household = pop.households[int(position)]
                for _ in range(int(counts[position])):
                    event = self._one_consumer_event(household, category, day, rng, amounts)
                    if event is not None:
                        events.append(event)
        return events

    def _one_consumer_event(
        self,
        household: Any,
        category: str,
        day: _dt.date,
        rng: np.random.Generator,
        amounts: dict[str, Any],
    ) -> RawEvent | None:
        pop = self.population

        if category == "p2p":
            if not household.social_circle:
                return None
            counterparty_position = int(rng.choice(household.social_circle))
            dst = pop.households[counterparty_position].account_index
            online_only = False
        else:
            candidates = pop.merchants_by_category.get(category)
            if not candidates:
                return None
            # Loyalty: usually one of this household's regular merchants, but
            # sometimes somewhere new. That mixture is what gives
            # is_new_beneficiary a realistic base rate instead of making it
            # either always-false or always-true.
            loyal = [m for m in household.loyal_merchants if m in candidates]
            if loyal and rng.random() < 0.78:
                merchant_position = int(rng.choice(loyal))
            else:
                merchant_position = int(rng.choice(candidates))
            merchant = pop.merchants[merchant_position]
            dst, online_only = merchant.account_index, merchant.online_only

        return RawEvent(
            ts=self.clock.stamp(day, rng),
            src_index=household.account_index,
            dst_index=dst,
            amount_cents=_amount(rng, amounts[category]),
            category=category,
            channel=_consumer_channel(rng, self._channel_mix, online_only),
            remittance=_remittance_for(category, day, rng),
        )

    # -------------------------------------------------- business payments --

    def _business_payments(self, day: _dt.date, rng: np.random.Generator) -> list[RawEvent]:
        """Supplier settlement between merchants, SMEs and foreign counterparties.

        Business-to-business flow is what closes the economic loop: consumer
        money reaching merchants comes back out to SMEs, and SME money goes out
        as salary. Without it the graph is a bipartite consumer→merchant star
        with no cycles, and circular-flow laundering would be detectable by the
        mere existence of a cycle.
        """
        if not is_business_day(day):
            return []

        events: list[RawEvent] = []
        pop = self.population

        for merchant in pop.merchants:
            if not merchant.supplier_indices:
                continue
            count = rng.poisson(self.cal["merchant"]["weekly_supplier_payments"] / 5.0)
            for _ in range(int(count)):
                events.append(
                    RawEvent(
                        ts=self.clock.stamp(day, rng, business=True),
                        src_index=merchant.account_index,
                        dst_index=int(rng.choice(merchant.supplier_indices)),
                        amount_cents=_amount(rng, self.cal["merchant"]["supplier_amount"]),
                        category="supplier",
                        channel=Channel.ONLINE_BANKING,
                        remittance=f"Rechnung {day.year}-{int(rng.integers(1000, 9999))}",
                    )
                )

        for sme in pop.smes:
            if not sme.supplier_indices:
                continue
            count = rng.poisson(self.cal["sme"]["weekly_supplier_payments"] / 5.0)
            for _ in range(int(count)):
                events.append(
                    RawEvent(
                        ts=self.clock.stamp(day, rng, business=True),
                        src_index=sme.account_index,
                        dst_index=int(rng.choice(sme.supplier_indices)),
                        amount_cents=_amount(rng, self.cal["sme"]["supplier_amount"]),
                        category="supplier",
                        channel=Channel.ONLINE_BANKING,
                        remittance=f"Rechnung {day.year}-{int(rng.integers(1000, 9999))}",
                    )
                )
        return events


def _remittance_for(category: str, day: _dt.date, rng: np.random.Generator) -> str:
    """A plausible Verwendungszweck.

    Free-text remittance information is a genuine AML signal — structuring
    campaigns often carry conspicuously uniform or empty references — so it is
    generated with realistic variety rather than left blank.
    """
    if category == "groceries":
        return f"Einkauf {day.strftime('%d.%m.%Y')}"
    if category == "ecommerce":
        return f"Bestellung {int(rng.integers(100000, 999999))}"
    if category == "dining":
        return f"Rechnung {day.strftime('%d.%m.')} Tisch {int(rng.integers(1, 40))}"
    if category == "p2p":
        return str(
            rng.choice(
                [
                    "Danke!",
                    "Essen gestern",
                    "Anteil Geschenk",
                    "Miete Anteil",
                    "Zurueck von gestern",
                    "Kino",
                    "",
                ]
            )
        )
    if category == "retail":
        return f"Kauf Beleg {int(rng.integers(10000, 99999))}"
    return ""
