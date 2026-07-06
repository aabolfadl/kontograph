"""The synthetic population: accounts and the agents that own them.

The structure here is what makes the graph worth modelling. Three properties are
deliberate:

1. **Households have small, stable counterparty sets.** Real people pay the same
   handful of merchants repeatedly. That stability is what makes a *departure*
   from it informative, and it is why ``is_new_beneficiary`` carries signal.
2. **The economy circulates.** SMEs pay salaries to households, households pay
   merchants, merchants pay SME suppliers. Value flows in loops, so the graph
   has genuine community structure rather than being a bipartite star.
3. **Employers fan out to many accounts on one day.** This is the confusable
   twin of mule fan-out, and including it is the difference between a dataset
   where fan-out detection is trivial and one where it is a real problem.
"""

from __future__ import annotations

import datetime as _dt
import enum
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kg_core.ids import IBAN_LENGTHS, build_iban, german_iban
from kg_core.timeutil import UTC, to_millis
from kg_sim.config import SimulationConfig
from kg_sim.institutions import BANKS, FOREIGN_BANKS, Bank, retail_weights
from kg_sim.names import company_name, merchant_name, person_name, utility_name


class AccountKind(enum.StrEnum):
    HOUSEHOLD = "HOUSEHOLD"
    MERCHANT = "MERCHANT"
    SME = "SME"
    UTILITY = "UTILITY"
    FOREIGN = "FOREIGN"
    #: Opened by a mule network specifically to receive and forward value.
    #: Kept as a distinct kind so ground truth can be reconstructed exactly,
    #: never as a feature — see the leakage test in tests/unit/test_contracts.py.
    MULE = "MULE"


@dataclass(frozen=True, slots=True)
class Account:
    """A payment account: the node type of the graph."""

    account_id: str
    iban: str
    bic: str
    holder_name: str
    kind: AccountKind
    bank_blz: str
    country: str
    opened_ts: int

    def age_days(self, at_ts: int) -> float:
        """Account age in days at *at_ts*. Never negative."""
        return max(0.0, (at_ts - self.opened_ts) / 86_400_000)


@dataclass(slots=True)
class Household:
    """A private individual with an income and recurring obligations."""

    account_index: int
    monthly_income_cents: int
    is_renter: bool
    rent_cents: int
    employer_index: int | None
    landlord_index: int | None
    utility_index: int
    loyal_merchants: list[int] = field(default_factory=list)
    social_circle: list[int] = field(default_factory=list)
    #: Per-category weekly Poisson intensities, jittered per household so the
    #: population has heterogeneous activity rather than one shared rate.
    activity_scale: float = 1.0


@dataclass(slots=True)
class Merchant:
    """A retailer or service business that receives consumer payments."""

    account_index: int
    categories: tuple[str, ...]
    online_only: bool
    supplier_indices: list[int] = field(default_factory=list)
    employee_indices: list[int] = field(default_factory=list)


@dataclass(slots=True)
class Sme:
    """A small or medium business: the main employer and supplier archetype."""

    account_index: int
    employee_indices: list[int] = field(default_factory=list)
    supplier_indices: list[int] = field(default_factory=list)


@dataclass(slots=True)
class Population:
    """Everything the economy needs to run."""

    accounts: list[Account]
    households: list[Household]
    merchants: list[Merchant]
    smes: list[Sme]
    utilities: list[int]
    foreign: list[int]

    #: Merchant indices grouped by the spend category they serve, so the
    #: economy can sample "a grocer" without scanning every merchant.
    merchants_by_category: dict[str, list[int]] = field(default_factory=dict)

    #: Next Kontonummer to issue. Carried on the population rather than being
    #: local to the builder, because the typology engine opens accounts too —
    #: mule networks recruit fresh accounts mid-simulation, and those must draw
    #: from the same sequence or two accounts could share an IBAN.
    next_kontonummer: int = 1_000_000

    #: Horizon start, so accounts opened later can compute a plausible age.
    horizon_start: _dt.date = _dt.date(2025, 10, 9)

    def account(self, index: int) -> Account:
        return self.accounts[index]

    def __len__(self) -> int:
        return len(self.accounts)

    def open_account(
        self,
        kind: AccountKind,
        holder_name: str,
        rng: np.random.Generator,
        *,
        fresh: bool = False,
        opened_ts: int | None = None,
    ) -> int:
        """Open a new account mid-simulation and return its index.

        ``fresh=True`` forces a recently opened account. Mule networks depend on
        this: the recruited accounts are days old, which is the single most
        reliable structural signal available against them — and also why the
        evasive variant deliberately uses older accounts instead.
        """
        bank = _sample_bank(rng)
        self.next_kontonummer += 1
        if opened_ts is None:
            opened_ts = (
                int(
                    to_millis(_dt.datetime.combine(self.horizon_start, _dt.time(0, 0), tzinfo=UTC))
                    - float(rng.uniform(1, 45)) * 86_400_000
                )
                if fresh
                else _opened_ts(rng, bank, self.horizon_start)
            )
        self.accounts.append(
            Account(
                account_id=f"ACC-{len(self.accounts):08d}",
                iban=german_iban(bank.blz, self.next_kontonummer),
                bic=bank.bic,
                holder_name=holder_name,
                kind=kind,
                bank_blz=bank.blz,
                country="DE",
                opened_ts=opened_ts,
            )
        )
        return len(self.accounts) - 1


def _lognormal_cents(rng: np.random.Generator, spec: dict[str, Any], scale: float = 1.0) -> int:
    """Draw a monetary amount in cents from a calibrated lognormal.

    Clipped to the configured bounds. Money is integer cents throughout —
    the same discipline: a float euro amount that round-trips through
    a feature aggregation is how reconciliation breaks.
    """
    value = rng.lognormal(mean=np.log(spec["median"]), sigma=spec["sigma"]) * scale
    value = float(np.clip(value, spec["min"], spec["max"]))
    return round(value * 100)


def _sample_bank(rng: np.random.Generator) -> Bank:
    banks, weights = retail_weights()
    return banks[int(rng.choice(len(banks), p=weights))]


def _opened_ts(rng: np.random.Generator, bank: Bank, horizon_start: _dt.date) -> int:
    """When an account was opened.

    Most accounts are years old. A bank-specific minority are recent, and direct
    banks skew far younger because remote onboarding is fast — which is exactly
    the property mule networks exploit, and why account age is a real feature
    rather than a curiosity.
    """
    start_ts = to_millis(_dt.datetime.combine(horizon_start, _dt.time(0, 0), tzinfo=UTC))
    if rng.random() < bank.fresh_account_propensity:
        age_days = float(rng.uniform(1, 120))
    else:
        # Long tail: median around 4 years, some accounts decades old.
        age_days = float(np.clip(rng.lognormal(np.log(1400), 0.9), 130, 12_000))
    return int(start_ts - age_days * 86_400_000)


def _next_account_number(counter: list[int]) -> int:
    counter[0] += 1
    return counter[0]


def build_population(config: SimulationConfig, rng: np.random.Generator) -> Population:
    """Construct the whole ecosystem.

    Built in a fixed order so the result depends only on ``(config, seed)``, not
    on iteration or hashing order — the determinism ADR-0009 depends on.
    """
    cal = config.calibration
    horizon_start = config.horizon.start_date
    accounts: list[Account] = []
    kontonummer = [1_000_000]

    def add_account(kind: AccountKind, holder_name: str) -> int:
        bank = _sample_bank(rng)
        number = _next_account_number(kontonummer)
        accounts.append(
            Account(
                account_id=f"ACC-{len(accounts):08d}",
                iban=german_iban(bank.blz, number),
                bic=bank.bic,
                holder_name=holder_name,
                kind=kind,
                bank_blz=bank.blz,
                country="DE",
                opened_ts=_opened_ts(rng, bank, horizon_start),
            )
        )
        return len(accounts) - 1

    # ---------------------------------------------------------- utilities --
    # A small set of utility providers, because in reality a whole city pays the
    # same Stadtwerke. This creates high-in-degree nodes that are entirely
    # legitimate — useful negative examples for any degree-based heuristic.
    utility_indices = [
        add_account(
            AccountKind.UTILITY, utility_name(rng, str(rng.choice([b.city for b in BANKS])))
        )
        for _ in range(max(3, config.population.merchants // 25))
    ]

    # ------------------------------------------------------------- SMEs ----
    sme_objects: list[Sme] = []
    for _ in range(config.population.smes):
        index = add_account(AccountKind.SME, company_name(rng))
        sme_objects.append(Sme(account_index=index))

    # --------------------------------------------------------- merchants ---
    merchant_objects: list[Merchant] = []
    merchants_by_category: dict[str, list[int]] = {}
    for position in range(config.population.merchants):
        trading, categories = merchant_name(rng)
        index = add_account(AccountKind.MERCHANT, trading)
        merchant = Merchant(
            account_index=index,
            categories=categories,
            online_only=bool(rng.random() < cal["merchant"]["online_only_probability"]),
        )
        merchant_objects.append(merchant)
        for category in categories:
            merchants_by_category.setdefault(category, []).append(position)

    # -------------------------------------------------------- households ---
    household_objects: list[Household] = []
    hh_cal = cal["household"]
    for _ in range(config.population.households):
        index = add_account(AccountKind.HOUSEHOLD, person_name(rng))
        income_cents = _lognormal_cents(rng, hh_cal["income_eur"])
        is_renter = bool(rng.random() < hh_cal["renter_probability"])
        rent_share = float(
            np.clip(
                rng.normal(hh_cal["rent_share"]["mean"], hh_cal["rent_share"]["sigma"]),
                hh_cal["rent_share"]["min"],
                hh_cal["rent_share"]["max"],
            )
        )
        household_objects.append(
            Household(
                account_index=index,
                monthly_income_cents=income_cents,
                is_renter=is_renter,
                rent_cents=int(income_cents * rent_share) if is_renter else 0,
                employer_index=None,  # assigned below
                landlord_index=None,
                utility_index=int(rng.choice(utility_indices)),
                activity_scale=float(np.clip(rng.gamma(shape=4.0, scale=0.25), 0.25, 3.0)),
            )
        )

    # ---------------------------------------------------- foreign accounts --
    # Cross-border counterparties. Without them, "creditor is abroad" would be a
    # perfect crime indicator, and every laundering typology would be findable
    # with one WHERE clause.
    foreign_indices: list[int] = []
    for _ in range(max(8, config.population.smes // 6)):
        country, bank_code, bic, _bank_name = FOREIGN_BANKS[int(rng.integers(len(FOREIGN_BANKS)))]
        number = _next_account_number(kontonummer)
        # Derive the account-number width from the country's total IBAN length
        # rather than a hand-kept table. A hand-kept table was wrong for ES on
        # the first run, and would have gone wrong again the next time a country
        # was added -- IBAN_LENGTHS is the single authority for this.
        account_width = IBAN_LENGTHS[country] - 4 - len(bank_code)
        account_number = str(number).zfill(account_width)[-account_width:]
        accounts.append(
            Account(
                account_id=f"ACC-{len(accounts):08d}",
                iban=build_iban(country, bank_code, account_number),
                bic=bic,
                holder_name=company_name(rng),
                kind=AccountKind.FOREIGN,
                bank_blz="",
                country=country,
                opened_ts=_opened_ts(rng, BANKS[0], horizon_start),
            )
        )
        foreign_indices.append(len(accounts) - 1)

    population = Population(
        accounts=accounts,
        households=household_objects,
        merchants=merchant_objects,
        smes=sme_objects,
        utilities=utility_indices,
        foreign=foreign_indices,
        merchants_by_category=merchants_by_category,
        next_kontonummer=kontonummer[0],
        horizon_start=horizon_start,
    )
    _wire_relationships(population, config, rng)
    return population


def _wire_relationships(
    population: Population, config: SimulationConfig, rng: np.random.Generator
) -> None:
    """Connect agents into a circulating economy.

    Employment is assigned SME-first so that every SME has a payroll to run.
    Leftover households are employed by merchants or treated as self-employed —
    the residual matters, because an account with income from no fixed employer
    is a genuinely different behavioural profile.
    """
    cal = config.calibration
    n_households = len(population.households)
    order = list(range(n_households))
    rng.shuffle(order)

    # Reserve the non-employed up front, so employers cannot absorb everyone.
    # Without this every household received a regular monthly salary from a
    # single payer, which made "has a stable employer" universally true and
    # therefore worthless as a feature.
    reserved = round(n_households * cal["household"]["no_employer_share"])
    unassigned = order[reserved:]
    cursor = 0

    # --- SMEs employ first -------------------------------------------------
    sme_spec = cal["sme"]["employees"]
    for sme in population.smes:
        headcount = int(rng.integers(sme_spec["min"], sme_spec["max"] + 1))
        take = unassigned[cursor : cursor + headcount]
        cursor += len(take)
        sme.employee_indices = take
        for household_position in take:
            population.households[household_position].employer_index = sme.account_index

    # --- merchants employ the remainder ------------------------------------
    merchant_spec = cal["merchant"]["employees"]
    for merchant in population.merchants:
        if cursor >= n_households:
            break
        headcount = int(rng.integers(merchant_spec["min"], merchant_spec["max"] + 1))
        take = unassigned[cursor : cursor + headcount]
        cursor += len(take)
        merchant.employee_indices = take
        for household_position in take:
            population.households[household_position].employer_index = merchant.account_index

    # Whoever is left has no employer: pensioners, students, the self-employed.
    # Their income arrives on a different schedule, which the economy handles.

    # --- landlords ---------------------------------------------------------
    # A landlord is another household or an SME. Modelling landlords as real
    # nodes (rather than a sink) is what makes rent a repeated edge between the
    # same pair every month — the single strongest recurring pattern in the data.
    landlord_pool = [h.account_index for h in population.households] + [
        s.account_index for s in population.smes
    ]
    for household in population.households:
        if household.is_renter:
            candidate = int(rng.choice(len(landlord_pool)))
            if landlord_pool[candidate] != household.account_index:
                household.landlord_index = landlord_pool[candidate]

    # --- loyalty and social circles ----------------------------------------
    hh_cal = cal["household"]
    loyal_spec, social_spec = hh_cal["loyal_merchants"], hh_cal["social_circle"]
    n_merchants = len(population.merchants)
    for position, household in enumerate(population.households):
        count = int(rng.integers(loyal_spec["min"], loyal_spec["max"] + 1))
        household.loyal_merchants = [
            int(x) for x in rng.choice(n_merchants, size=min(count, n_merchants), replace=False)
        ]
        friends = int(rng.integers(social_spec["min"], social_spec["max"] + 1))
        # social_circle holds household POSITIONS, so the self-exclusion must
        # compare against this household's position -- not against its
        # account_index, which indexes a different list entirely (accounts,
        # offset by the utilities/SMEs/merchants created before households).
        # That type confusion let a household appear in its own social circle
        # and pay itself. Caught by the bronze->silver quality gate, which
        # blocked the promotion on debtor_and_creditor_differ.
        household.social_circle = [
            int(x)
            for x in rng.choice(n_households, size=min(friends, n_households), replace=False)
            if int(x) != position
        ]

    # --- supply chains -----------------------------------------------------
    sme_positions = list(range(len(population.smes)))
    for merchant in population.merchants:
        count = int(rng.integers(1, 6))
        merchant.supplier_indices = [
            population.smes[int(p)].account_index
            for p in rng.choice(sme_positions, size=min(count, len(sme_positions)), replace=False)
        ]
    for position, sme in enumerate(population.smes):
        others = [p for p in sme_positions if p != position]
        if not others:
            continue
        count = int(rng.integers(1, 5))
        picks = rng.choice(others, size=min(count, len(others)), replace=False)
        sme.supplier_indices = [population.smes[int(p)].account_index for p in picks]
        # A minority import from abroad. Legitimate cross-border business
        # payments, so that foreign edges are not a crime giveaway.
        if population.foreign and rng.random() < 0.18:
            sme.supplier_indices.append(int(rng.choice(population.foreign)))
