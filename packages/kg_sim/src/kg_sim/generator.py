"""Simulation orchestration.

Merges the legitimate economy with the injected typologies, dresses each raw
event as a wire-contract record, and derives the label stream.

Three things happen here that matter to the rest of the platform:

1. **Global temporal ordering.** Events from every source are merged and sorted
   by acceptance time before anything derived is computed. Any state that
   depends on history — ``is_new_beneficiary`` above all — is then accumulated
   in that order, so it is point-in-time correct *by construction* rather than
   by a later join that has to be trusted.

2. **Deterministic identifiers.** ``tx_id`` is derived from the event's content
   and position, not from ``uuid4()``. Re-running the simulator produces
   byte-identical output, which is what lets ADR-0009 replace DVC with a content
   hash.

3. **The label-generating process is applied last, and separately.** Labels
   arrive late, only for detected cases, and only for cases the incumbent system
   flagged. Modelling that here — rather than joining ground truth directly —
   is what makes design principle P7 an experiment instead of a caveat.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kg_core.contracts import (
    Channel,
    LabelSource,
    LabelValue,
    SctInstEvent,
    SettlementStatus,
    SubjectType,
)
from kg_core.hashing import content_hash
from kg_sim.config import SimulationConfig
from kg_sim.economy import PURPOSE_CODES, Economy, RawEvent
from kg_sim.population import build_population
from kg_sim.typologies import TaggedEvent, TypologyEngine, TypologyTag
from kg_sim.vop import VopEngine

#: Criminal categories, mapped to the purpose code a launderer would plausibly
#: claim. They pick innocuous codes precisely because the code is self-declared
#: and unverified — a detail worth reproducing, since it means purpose_code is
#: weak evidence rather than a label in disguise.
CRIMINAL_PURPOSE_CODES: dict[str, str] = {
    "mule_fan_in": "OTHR",
    "mule_fan_out": "OTHR",
    "mule_consolidation": "SUPP",
    "smurfing": "OTHR",
    "app_fraud": "OTHR",
    "app_fraud_dispersal": "OTHR",
    "circular_flow": "SUPP",
    "layering": "SUPP",
}


@dataclass(slots=True)
class SimulationResult:
    """Everything one run produces."""

    config: SimulationConfig
    accounts: list[dict[str, Any]]
    events: list[dict[str, Any]]
    ground_truth: list[dict[str, Any]]
    account_roles: list[dict[str, Any]]
    labels: list[dict[str, Any]]
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def positive_rate(self) -> float:
        """Share of transactions that are genuinely criminal (oracle truth)."""
        return len(self.ground_truth) / len(self.events) if self.events else 0.0

    @property
    def labelled_rate(self) -> float:
        """Share of transactions that ever receive a confirmed label."""
        return len(self.labels) / len(self.events) if self.events else 0.0


def _deterministic_tx_id(index: int, event: RawEvent) -> str:
    """A UETR-shaped identifier derived from content, not from a random source.

    Formatted as a UUID because that is what a pacs.008 UETR is, but generated
    by hashing so the dataset is reproducible. ``index`` is included because two
    genuinely identical payments can occur in the same millisecond, and they
    still need distinct references.
    """
    digest = content_hash(
        (index, event.ts, event.src_index, event.dst_index, event.amount_cents, event.category),
        prefix="kontograph-tx",
    )
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def _ingest_lag_millis(rng: np.random.Generator) -> int:
    """Delay between a payment being accepted and reaching our platform.

    Small and right-skewed. Present in the data because it is real, and
    deliberately never used as a feature: ingest lag correlates with system load
    at scoring time, not with the payment, and a model that learns it has learnt
    an artefact of the pipeline.
    """
    if rng.random() < 0.985:
        return int(rng.gamma(shape=2.0, scale=18.0))
    return int(rng.gamma(shape=2.0, scale=900.0))  # occasional retry or backlog


def _device_and_ip(
    rng: np.random.Generator, channel: Channel, account_index: int
) -> tuple[str | None, str | None]:
    """Device and network context for consumer-initiated payments.

    The IP prefix is truncated to a /24 *at the producer*, which is data
    minimisation under DSGVO Art. 5(1)(c) rather than a modelling convenience —
    the full address is never collected, so it cannot leak later.
    """
    if channel in (Channel.STANDING_ORDER, Channel.BATCH_UPLOAD):
        return None, None
    # Most accounts use a stable device; a minority rotate.
    device = f"dev-{content_hash((account_index, int(rng.integers(3))), prefix='dev')[:8]}"
    octets = (
        int(rng.choice([84, 91, 78, 217, 46, 5])),
        int(rng.integers(256)),
        int(rng.integers(256)),
    )
    return device, f"{octets[0]}.{octets[1]}.{octets[2]}.0/24"


def simulate(config: SimulationConfig) -> SimulationResult:
    """Run one complete simulation."""
    # Independent, reproducible streams per subsystem. Spawning from a single
    # SeedSequence means adding a subsystem later cannot perturb the draws of an
    # existing one — without this, introducing the label model would silently
    # change every previously generated transaction.
    root = np.random.SeedSequence(config.seed)
    pop_seed, econ_seed, typo_seed, vop_seed, label_seed, wire_seed = root.spawn(6)
    rng_pop = np.random.default_rng(pop_seed)
    rng_econ = np.random.default_rng(econ_seed)
    rng_typo = np.random.default_rng(typo_seed)
    rng_vop = np.random.default_rng(vop_seed)
    rng_label = np.random.default_rng(label_seed)
    rng_wire = np.random.default_rng(wire_seed)

    population = build_population(config, rng_pop)
    legitimate = Economy(config, population).run(rng_econ)
    criminal: list[TaggedEvent] = TypologyEngine(config, population).run(rng_typo)

    merged: list[tuple[RawEvent, TypologyTag | None]] = [(e, None) for e in legitimate]
    merged.extend(criminal)

    # Total order. The sort key is fully determined by event content so that
    # ties break identically on every run and on every platform - Python's sort
    # is stable, but the input order across two sources is not something to rely
    # on when reproducibility is the whole claim.
    merged.sort(
        key=lambda pair: (
            pair[0].ts,
            pair[0].src_index,
            pair[0].dst_index,
            pair[0].amount_cents,
            pair[0].category,
        )
    )

    vop_engine = VopEngine(config.calibration)
    horizon_end_ts = int(
        _dt.datetime.combine(config.horizon.end_date, _dt.time(0, 0), tzinfo=_dt.UTC).timestamp()
        * 1000
    )

    events: list[dict[str, Any]] = []
    ground_truth: list[dict[str, Any]] = []
    seen_pairs: set[tuple[int, int]] = set()

    for index, (raw, tag) in enumerate(merged):
        # Campaigns are scheduled against the horizon but can run past its end;
        # truncating here keeps the dataset's time bounds honest.
        if raw.ts >= horizon_end_ts:
            continue

        debtor = population.accounts[raw.src_index]
        creditor = population.accounts[raw.dst_index]
        tx_id = _deterministic_tx_id(index, raw)

        # Accumulated in temporal order, so it can only ever reflect the past.
        pair = (raw.src_index, raw.dst_index)
        is_new_beneficiary = pair not in seen_pairs
        seen_pairs.add(pair)

        claimed = raw.claimed_name or vop_engine.perturb_claimed_name(creditor.holder_name, rng_vop)
        vop_result, similarity, overridden = vop_engine.check(
            claimed,
            creditor.holder_name,
            rng_vop,
            is_app_fraud=raw.category == "app_fraud",
        )

        device_id, ip_prefix = _device_and_ip(rng_wire, raw.channel, raw.src_index)
        purpose = PURPOSE_CODES.get(raw.category) or CRIMINAL_PURPOSE_CODES.get(raw.category)

        events.append(
            {
                "schema_version": 1,
                "tx_id": tx_id,
                "end_to_end_id": f"E2E-{index:010d}",
                "acceptance_ts": raw.ts,
                "ingest_ts": raw.ts + _ingest_lag_millis(rng_wire),
                "debtor_account_id": debtor.account_id,
                "debtor_iban": debtor.iban,
                "debtor_bic": debtor.bic,
                "debtor_name": debtor.holder_name,
                "debtor_country": debtor.country,
                "creditor_account_id": creditor.account_id,
                "creditor_iban": creditor.iban,
                "creditor_bic": creditor.bic,
                "creditor_name": creditor.holder_name,
                "creditor_country": creditor.country,
                "creditor_name_claimed": claimed,
                "vop_result": vop_result.value,
                "vop_name_similarity": round(similarity, 6),
                "vop_overridden": overridden,
                "amount_cents": raw.amount_cents,
                "currency": "EUR",
                "purpose_code": purpose,
                "remittance_info": raw.remittance or None,
                "channel": raw.channel.value,
                "initiation_source": _initiation_source(rng_wire, raw.channel),
                "device_id": device_id,
                "session_ip_prefix": ip_prefix,
                "is_new_beneficiary": is_new_beneficiary,
                "settlement_status": _settlement_status(rng_wire).value,
            }
        )

        if tag is not None:
            ground_truth.append(
                {
                    "tx_id": tx_id,
                    "acceptance_ts": raw.ts,
                    "campaign_id": tag.campaign_id,
                    "typology": tag.typology.value,
                    "src_account_id": debtor.account_id,
                    "dst_account_id": creditor.account_id,
                    "src_role": tag.src_role.value,
                    "dst_role": tag.dst_role.value,
                    "hop": tag.hop,
                    "is_evasive": tag.is_evasive,
                    "category": raw.category,
                }
            )

    account_roles = _account_roles(ground_truth)
    labels = _generate_labels(config, events, ground_truth, rng_label)
    accounts = [
        {
            "account_id": account.account_id,
            "iban": account.iban,
            "bic": account.bic,
            "holder_name": account.holder_name,
            "kind": account.kind.value,
            "bank_blz": account.bank_blz,
            "country": account.country,
            "opened_ts": account.opened_ts,
        }
        for account in population.accounts
    ]

    result = SimulationResult(
        config=config,
        accounts=accounts,
        events=events,
        ground_truth=ground_truth,
        account_roles=account_roles,
        labels=labels,
    )
    result.manifest = _build_manifest(config, result)
    return result


def _initiation_source(rng: np.random.Generator, channel: Channel) -> str:
    """PSD2 initiation source. Only API-channel payments come from a PISP."""
    if channel is Channel.OPEN_BANKING_API:
        return "PISP" if rng.random() < 0.85 else "AISP_TRIGGERED"
    return "ASPSP_DIRECT"


def _settlement_status(rng: np.random.Generator) -> SettlementStatus:
    """Almost everything settles. A small tail is rejected or pending.

    Rejections matter: under SCT Inst a rejected payment still leaves a trace,
    and a burst of rejections followed by a success is itself a probing pattern.
    """
    draw = rng.random()
    if draw < 0.988:
        return SettlementStatus.ACSC
    if draw < 0.996:
        return SettlementStatus.RJCT
    return SettlementStatus.PDNG


def _account_roles(ground_truth: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Account-level membership, derived from the edge-level ground truth.

    Kept as a separate table because account-level and transaction-level
    evaluation answer different questions: "did we catch the mule?" is not the
    same as "did we score this payment highly?", and the protocol reports both.
    """
    seen: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in ground_truth:
        for account_id, role in (
            (row["src_account_id"], row["src_role"]),
            (row["dst_account_id"], row["dst_role"]),
        ):
            key = (account_id, row["campaign_id"], role)
            entry = seen.get(key)
            if entry is None:
                seen[key] = {
                    "account_id": account_id,
                    "campaign_id": row["campaign_id"],
                    "typology": row["typology"],
                    "role": role,
                    "is_evasive": row["is_evasive"],
                    "first_ts": row["acceptance_ts"],
                    "last_ts": row["acceptance_ts"],
                    "edge_count": 1,
                }
            else:
                entry["last_ts"] = max(entry["last_ts"], row["acceptance_ts"])
                entry["first_ts"] = min(entry["first_ts"], row["acceptance_ts"])
                entry["edge_count"] += 1
    return list(seen.values())


def _generate_labels(
    config: SimulationConfig,
    events: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Apply the label-generating process — design principle P7.

    Three pathologies are produced deliberately:

    **Incompleteness.** Only ``detection_probability`` of criminal transactions
    are ever confirmed. The rest stay unlabelled and will be counted as negative
    by any naive pipeline, which is why the evaluation protocol reports both
    label-based and oracle metrics and publishes the gap.

    **Selection bias.** Confirmation requires the incumbent rules engine to have
    flagged the case first. Since that engine is threshold-based, it
    preferentially catches the *standard* variants — so the labelled positives
    systematically under-represent evasive typologies. A model trained naively on
    these labels learns to reproduce the old engine.

    **Latency.** ``decided_ts`` is drawn lognormally weeks after the event, so a
    chronological split can enforce "this label did not exist yet".
    """
    spec = config.labels
    by_tx = {row["tx_id"]: row for row in ground_truth}
    labels: list[dict[str, Any]] = []

    for event in events:
        truth = by_tx.get(event["tx_id"])
        event_ts = event["acceptance_ts"]

        if truth is not None:
            # The incumbent engine is threshold-based, so evasive campaigns —
            # built specifically to defeat thresholds — are caught far less
            # often. This is the mechanism behind the selection bias, not an
            # assumption bolted on afterwards.
            recall = spec.incumbent_alert_recall * (0.45 if truth["is_evasive"] else 1.0)
            if rng.random() > recall:
                continue
            if rng.random() > spec.detection_probability:
                continue
            value = (
                LabelValue.CONFIRMED_FRAUD
                if truth["typology"] == "APP_FRAUD"
                else LabelValue.CONFIRMED_LAUNDERING
            )
            source = (
                LabelSource.CUSTOMER_COMPLAINT
                if truth["typology"] == "APP_FRAUD"
                else LabelSource.ANALYST_REVIEW
            )
            sar_filed = bool(rng.random() < spec.sar_filing_probability)
        else:
            if rng.random() > spec.benign_confirmation_probability:
                continue
            value, source, sar_filed = LabelValue.BENIGN, LabelSource.ANALYST_REVIEW, False

        latency_days = float(
            rng.lognormal(np.log(spec.latency_days_median), spec.latency_days_sigma)
        )
        labels.append(
            {
                "schema_version": 1,
                "label_id": content_hash((event["tx_id"], "label"), prefix="kontograph-label")[:24],
                "subject_type": SubjectType.TRANSACTION.value,
                "subject_id": event["tx_id"],
                "event_ts": event_ts,
                "decided_ts": event_ts + int(latency_days * 86_400_000),
                "label": value.value,
                "source": source.value,
                "sar_filed": sar_filed,
                "confidence": round(float(rng.uniform(0.75, 1.0)), 4),
                "notes": None,
            }
        )
    return labels


def _build_manifest(config: SimulationConfig, result: SimulationResult) -> dict[str, Any]:
    """The dataset's identity card.

    This is the DVC replacement from ADR-0009: a result is only comparable to
    another result carrying the same ``config_fingerprint``. Row-level digests
    let a consumer verify that the data on disk is the data a number was
    computed from.
    """
    import collections

    typology_counts = collections.Counter(row["typology"] for row in result.ground_truth)
    evasive_counts = collections.Counter(
        (row["typology"], row["is_evasive"]) for row in result.ground_truth
    )
    label_counts = collections.Counter(row["label"] for row in result.labels)

    acceptance = [event["acceptance_ts"] for event in result.events]
    return {
        "config_fingerprint": config.fingerprint,
        "seed": config.seed,
        "profile": config.profile_summary(),
        "counts": {
            "accounts": len(result.accounts),
            "events": len(result.events),
            "criminal_events": len(result.ground_truth),
            "campaigns": len({row["campaign_id"] for row in result.ground_truth}),
            "labels": len(result.labels),
            "account_roles": len(result.account_roles),
        },
        "rates": {
            "positive_rate": round(result.positive_rate, 6),
            "labelled_rate": round(result.labelled_rate, 6),
            # The share of genuinely criminal transactions that ever get a label.
            # This number is the honest headline of design principle P7 and belongs in the
            # manifest so nobody has to recompute it to be reminded of it.
            "criminal_label_coverage": round(
                (label_counts["CONFIRMED_FRAUD"] + label_counts["CONFIRMED_LAUNDERING"])
                / max(1, len(result.ground_truth)),
                6,
            ),
        },
        "typologies": dict(typology_counts),
        "evasive_split": {
            f"{typology}:{'evasive' if evasive else 'standard'}": count
            for (typology, evasive), count in evasive_counts.items()
        },
        "labels": dict(label_counts),
        "time_bounds": {
            "min_acceptance_ts": min(acceptance) if acceptance else None,
            "max_acceptance_ts": max(acceptance) if acceptance else None,
        },
        "content_hashes": {
            "events": content_hash(result.events, prefix="events"),
            "ground_truth": content_hash(result.ground_truth, prefix="ground-truth"),
            "labels": content_hash(result.labels, prefix="labels"),
            "accounts": content_hash(result.accounts, prefix="accounts"),
        },
    }


def validate_sample(result: SimulationResult, sample_size: int = 2_000) -> int:
    """Validate a sample of generated events against the pydantic contract.

    A sample rather than the whole set: at full scale there are hundreds of
    thousands of events, and materialising a pydantic model for each costs both
    time and roughly a gigabyte of memory for no additional guarantee. The
    generator emits every field from one code path, so a representative sample
    proves the shape. Returns the number validated.
    """
    if not result.events:
        return 0
    step = max(1, len(result.events) // sample_size)
    checked = 0
    for event in result.events[::step]:
        SctInstEvent(**event)
        checked += 1
    return checked
