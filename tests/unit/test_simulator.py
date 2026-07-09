"""Simulator invariants.

These tests defend the claims M1 makes. Three of them are load-bearing for
decisions taken elsewhere in the project:

- ``test_simulation_is_byte_reproducible`` keeps ADR-0009 honest. Content-hashing
  a dataset only replaces DVC if the recipe genuinely regenerates the bytes.
- ``test_is_new_beneficiary_is_point_in_time_correct`` is the first instance of
  the design principle P2 invariant: derived state may only reflect the past.
- ``test_events_table_contains_no_ground_truth`` is the structural anti-leakage
  guarantee at the dataset level, complementing the wire-contract test.
"""

from __future__ import annotations

import collections
import itertools

import pytest

from kg_core.ids import validate_iban
from kg_sim.config import get_profile
from kg_sim.dataset import ORACLE_TABLES, OracleAccessError, read_table, write_dataset
from kg_sim.generator import simulate
from kg_sim.typologies import Typology


@pytest.fixture(scope="module")
def result():
    """One smoke-profile simulation, shared across the module."""
    return simulate(get_profile("smoke"))


# ------------------------------------------------------- reproducibility ----


@pytest.mark.slow
def test_simulation_is_byte_reproducible() -> None:
    """The premise ADR-0009 rests on when it drops DVC.

    If this ever fails, the right response is not to weaken the test — it is to
    reinstate dataset versioning, because the "just regenerate it" story has
    stopped being true.
    """
    first, second = simulate(get_profile("smoke")), simulate(get_profile("smoke"))
    assert first.manifest["content_hashes"] == second.manifest["content_hashes"]
    assert first.events == second.events
    assert first.labels == second.labels


def test_config_fingerprint_changes_with_the_seed() -> None:
    base = get_profile("smoke")
    altered = base.model_copy(update={"seed": base.seed + 1})
    assert base.fingerprint != altered.fingerprint


def test_different_seeds_produce_different_data() -> None:
    base = get_profile("smoke")
    other = simulate(base.model_copy(update={"seed": base.seed + 1}))
    assert (
        simulate(base).manifest["content_hashes"]["events"]
        != (other.manifest["content_hashes"]["events"])
    )


# ------------------------------------------------ point-in-time correctness --


def test_events_are_in_ascending_acceptance_time(result) -> None:
    stamps = [event["acceptance_ts"] for event in result.events]
    assert stamps == sorted(stamps)


def test_is_new_beneficiary_is_point_in_time_correct(result) -> None:
    """True exactly on the first payment between a pair, false thereafter.

    Recomputed here from scratch in temporal order. If the generator ever
    derived this from a global set instead of an accumulating one, every
    payment between a repeat pair would be flagged new — or worse, none would,
    because the pair already existed in the future.
    """
    seen: set[tuple[str, str]] = set()
    for event in result.events:
        pair = (event["debtor_account_id"], event["creditor_account_id"])
        assert event["is_new_beneficiary"] == (pair not in seen), (
            f"is_new_beneficiary wrong for {pair} at {event['acceptance_ts']}"
        )
        seen.add(pair)


def test_ingest_never_precedes_acceptance(result) -> None:
    assert all(e["ingest_ts"] >= e["acceptance_ts"] for e in result.events)


def test_labels_are_decided_after_the_event(result) -> None:
    """Verification latency is strictly positive — labels arrive later, always."""
    assert all(label["decided_ts"] > label["event_ts"] for label in result.labels)


def test_label_latency_is_material(result) -> None:
    """Median latency is weeks, not hours.

    If labels arrived instantly the label-delay experiments in the evaluation
    protocol would be measuring nothing.
    """
    latencies = sorted(
        (label["decided_ts"] - label["event_ts"]) / 86_400_000 for label in result.labels
    )
    assert latencies, "the smoke profile should produce some labels"
    assert latencies[len(latencies) // 2] > 5.0


# --------------------------------------------------------------- leakage ----


def test_events_table_contains_no_ground_truth(result) -> None:
    """No column in the payment stream may reveal the answer."""
    forbidden = {
        "typology",
        "campaign_id",
        "is_evasive",
        "label",
        "is_fraud",
        "role",
        "src_role",
        "dst_role",
        "hop",
        "y",
        "target",
        "is_criminal",
    }
    present = set(result.events[0])
    assert not (present & forbidden), f"ground truth leaked into events: {present & forbidden}"


def test_account_kind_mule_is_not_in_the_event_stream(result) -> None:
    """The MULE account kind is oracle metadata, not a transaction field.

    It exists so ground truth can be reconstructed, and it lives on the account
    table. A feature that reads it would be reading the answer.
    """
    assert "kind" not in result.events[0]
    assert "debtor_kind" not in result.events[0]


def test_oracle_tables_are_guarded(tmp_path, result) -> None:
    """Reading ground truth must be a deliberate, greppable act."""
    target = write_dataset(result, tmp_path, profile="smoke")
    for name in ORACLE_TABLES:
        with pytest.raises(OracleAccessError, match="evaluation path only"):
            read_table(target, name)
        assert read_table(target, name, allow_oracle=True).num_rows >= 0
    # Non-oracle tables need no ceremony.
    assert read_table(target, "events").num_rows == len(result.events)


# --------------------------------------------------------- ground truth ----


def test_every_campaign_is_exactly_recoverable(result) -> None:
    """Ground truth must reconstruct each campaign without loss.

    This is what design principle P9 means by "the true causal structure of the
    fraud": not just which transactions were criminal, but which campaign each
    belonged to and what role each account played.
    """
    campaigns = collections.defaultdict(list)
    for row in result.ground_truth:
        campaigns[row["campaign_id"]].append(row)

    assert len(campaigns) == result.manifest["counts"]["campaigns"]
    for campaign_id, rows in campaigns.items():
        assert len({row["typology"] for row in rows}) == 1, f"{campaign_id} mixes typologies"
        assert len({row["is_evasive"] for row in rows}) == 1, f"{campaign_id} mixes variants"
        assert all(row["tx_id"] for row in rows)


def test_ground_truth_tx_ids_all_exist_in_the_event_stream(result) -> None:
    event_ids = {event["tx_id"] for event in result.events}
    truth_ids = {row["tx_id"] for row in result.ground_truth}
    assert truth_ids <= event_ids, "ground truth references transactions that do not exist"


def test_all_five_typologies_are_present(result) -> None:
    produced = {row["typology"] for row in result.ground_truth}
    assert produced == {t.value for t in Typology}


def test_account_roles_agree_with_edge_level_truth(result) -> None:
    edge_accounts = {(row["src_account_id"], row["campaign_id"]) for row in result.ground_truth} | {
        (row["dst_account_id"], row["campaign_id"]) for row in result.ground_truth
    }
    role_accounts = {(row["account_id"], row["campaign_id"]) for row in result.account_roles}
    assert role_accounts == edge_accounts


# --------------------------------------------- the label-generating process --


def test_most_crime_is_never_labelled(result) -> None:
    """Design principle P7, made measurable.

    If coverage were near 1.0 the dataset would be describing a world where
    every laundering case is eventually confirmed, and the whole discussion of
    label bias would be decoration.
    """
    coverage = result.manifest["rates"]["criminal_label_coverage"]
    assert 0.0 < coverage < 0.6, f"criminal label coverage {coverage} is implausible"


def test_labels_are_biased_against_evasive_campaigns() -> None:
    """The selection bias is real and directional.

    Confirmation requires the incumbent threshold-based engine to have flagged
    the case. Evasive campaigns are built to defeat thresholds, so they are
    labelled less often — which means a model trained naively on these labels
    learns to reproduce the old engine rather than to detect crime.
    """
    result = simulate(get_profile("dev"))
    labelled = {label["subject_id"] for label in result.labels}
    counts = collections.Counter(
        (row["is_evasive"], row["tx_id"] in labelled) for row in result.ground_truth
    )
    standard_rate = counts[(False, True)] / max(1, counts[(False, True)] + counts[(False, False)])
    evasive_rate = counts[(True, True)] / max(1, counts[(True, True)] + counts[(True, False)])
    assert evasive_rate < standard_rate, (
        f"evasive label rate {evasive_rate:.3f} should be below standard {standard_rate:.3f}"
    )


def test_benign_labels_exist_but_are_rare(result) -> None:
    """Nobody investigates ordinary payments, so negatives are mostly unlabelled."""
    by_value = collections.Counter(label["label"] for label in result.labels)
    assert by_value["BENIGN"] < len(result.events) * 0.01


def test_unproven_suspicion_is_never_emitted_as_a_positive(result) -> None:
    """The simulator only confirms what it actually knows to be criminal."""
    values = {label["label"] for label in result.labels}
    assert values <= {"CONFIRMED_FRAUD", "CONFIRMED_LAUNDERING", "BENIGN"}


# ----------------------------------------------------------- realism -------


def test_every_generated_iban_is_valid(result) -> None:
    for account in result.accounts:
        validate_iban(account["iban"], expect_country=account["country"])


def test_ibans_are_unique(result) -> None:
    ibans = [account["iban"] for account in result.accounts]
    assert len(set(ibans)) == len(ibans), "two accounts share an IBAN"


def test_amounts_are_positive_integer_cents(result) -> None:
    for event in result.events:
        assert isinstance(event["amount_cents"], int)
        assert event["amount_cents"] > 0


def test_positive_rate_is_realistically_small() -> None:
    """A believable AML base rate, which is what makes design principle P6 true.

    At this rate ROC-AUC is dominated by the negative class and says nothing,
    which is exactly why the protocol leads with PR-AUC and precision@k.
    """
    manifest = simulate(get_profile("dev")).manifest
    assert 0.0001 < manifest["rates"]["positive_rate"] < 0.03


def test_payroll_produces_a_large_legitimate_fan_out(result) -> None:
    """The confusable twin of mule fan-out must actually be present.

    Without a legitimate high-fan-out day the dataset would make fan-out
    detection trivial and the reported metrics meaningless.
    """
    criminal_ids = {row["tx_id"] for row in result.ground_truth}
    out_degree = collections.Counter(
        event["debtor_account_id"] for event in result.events if event["tx_id"] not in criminal_ids
    )
    assert max(out_degree.values()) >= 20, (
        "no legitimate account has substantial out-degree; payroll is missing"
    )


def test_vop_produces_all_three_outcome_bands(result) -> None:
    """VoP must not be a fraud oracle.

    Legitimate payments have to generate CLOSE_MATCH and NO_MATCH too —
    transliterated umlauts, married names, trading names — or the model would
    learn a rule that does not survive contact with reality.
    """
    criminal_ids = {row["tx_id"] for row in result.ground_truth}
    legit = collections.Counter(
        event["vop_result"] for event in result.events if event["tx_id"] not in criminal_ids
    )
    assert legit["MATCH"] > 0
    assert legit["CLOSE_MATCH"] > 0, "no legitimate near-miss: VoP would be a giveaway"
    assert legit["NO_MATCH"] > 0, "no legitimate mismatch: VoP would be a giveaway"


def test_evasive_campaigns_dwell_longer_than_standard_ones() -> None:
    """The evasive countermeasure against velocity rules must be real."""
    import statistics

    result = simulate(get_profile("dev"))
    gaps: dict[bool, list[float]] = {False: [], True: []}
    by_campaign: dict[str, list[tuple[int, bool]]] = collections.defaultdict(list)
    for row in result.ground_truth:
        by_campaign[row["campaign_id"]].append((row["acceptance_ts"], row["is_evasive"]))

    for rows in by_campaign.values():
        stamps = sorted(ts for ts, _ in rows)
        evasive = rows[0][1]
        gaps[evasive].extend(
            (later - earlier) / 60_000 for earlier, later in itertools.pairwise(stamps)
        )

    assert gaps[False] and gaps[True]
    assert statistics.median(gaps[True]) > 5 * statistics.median(gaps[False]), (
        "evasive campaigns must dwell materially longer, or they are not evasive"
    )


def test_criminal_and_legitimate_amounts_overlap(result) -> None:
    """Crime must not be separable by amount alone.

    If criminal amounts occupied a distinct range, a single threshold would
    solve the problem and the graph model would be pointless.
    """
    criminal_ids = {row["tx_id"] for row in result.ground_truth}
    criminal = [e["amount_cents"] for e in result.events if e["tx_id"] in criminal_ids]
    legit = [e["amount_cents"] for e in result.events if e["tx_id"] not in criminal_ids]
    assert min(criminal) < max(legit) and min(legit) < max(criminal)
    overlap = sum(1 for amount in criminal if min(legit) <= amount <= max(legit))
    assert overlap / len(criminal) > 0.9


# ------------------------------------------------------------ persistence ---


def test_dataset_round_trips_through_parquet(tmp_path, result) -> None:
    target = write_dataset(result, tmp_path, profile="smoke")
    events = read_table(target, "events")
    assert events.num_rows == len(result.events)
    assert events.column("acceptance_ts").to_pylist() == [e["acceptance_ts"] for e in result.events]
    # Explicit schemas: nullable text columns must survive as strings even when
    # every value in a small profile happens to be null.
    assert events.schema.field("purpose_code").type == "string"
    assert events.schema.field("remittance_info").type == "string"


def test_digest_detects_tampering(tmp_path, result) -> None:
    """A number can never be attributed to data that has since changed."""
    import pyarrow.parquet as pq

    from kg_sim.dataset import verify_digest

    target = write_dataset(result, tmp_path, profile="smoke")
    assert verify_digest(target)

    table = pq.read_table(target / "events.parquet")
    pq.write_table(table.slice(0, table.num_rows - 1), target / "events.parquet")
    assert not verify_digest(target)
