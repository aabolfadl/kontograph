"""The streaming spine and the lakehouse.

Everything here runs without a broker or object store — the wire format is pure
functions, and the Iceberg tests use a local filesystem catalog. The tests that
genuinely need the stack are marked ``integration`` and live at the bottom.
"""

from __future__ import annotations

import copy
import datetime as _dt
import struct

import pyarrow as pa
import pytest

from kg_core.contracts import load_schema, load_schema_dict
from kg_stream.wire import HEADER_SIZE, MAGIC_BYTE, WireFormatError, decode, encode, peek_schema_id

# ====================================================== the wire format =====


def test_frame_layout_is_magic_byte_then_schema_id(sample_event_dict: dict) -> None:
    """The five bytes that decide whether data stays readable in 2030."""
    schema = load_schema("sct_inst_event.v1")
    payload = encode(schema, 1234, sample_event_dict)

    magic, schema_id = struct.unpack(">bI", payload[:HEADER_SIZE])
    assert magic == MAGIC_BYTE == 0
    assert schema_id == 1234
    assert HEADER_SIZE == 5


def test_round_trip_preserves_every_field(sample_event_dict: dict) -> None:
    schema = load_schema("sct_inst_event.v1")
    restored = decode(encode(schema, 7, sample_event_dict), schema)
    for key, expected in sample_event_dict.items():
        assert restored[key] == expected, f"{key} did not survive the round trip"


def test_timestamps_decode_as_integer_millis(sample_event_dict: dict) -> None:
    """The invariant a logical-type conversion silently broke once already.

    fastavro converts `timestamp-millis` to a datetime by default. That
    reintroduces exactly the timezone ambiguity kg_core.timeutil exists to
    eliminate, and it does so invisibly — a comparison against an int simply
    starts returning False. kg_core.contracts opts out of the conversion; this
    test is what stops it coming back.
    """
    schema = load_schema("sct_inst_event.v1")
    restored = decode(encode(schema, 1, sample_event_dict), schema)
    assert isinstance(restored["acceptance_ts"], int)
    assert not isinstance(restored["acceptance_ts"], _dt.datetime)
    assert restored["acceptance_ts"] == sample_event_dict["acceptance_ts"]


def test_peek_reads_the_id_without_decoding(sample_event_dict: dict) -> None:
    schema = load_schema("sct_inst_event.v1")
    assert peek_schema_id(encode(schema, 99, sample_event_dict)) == 99


def test_unframed_payload_is_rejected_with_a_useful_message() -> None:
    with pytest.raises(WireFormatError, match="not produced with Schema Registry framing"):
        peek_schema_id(b"\x01\x00\x00\x00\x01rest")


def test_truncated_payload_is_rejected() -> None:
    with pytest.raises(WireFormatError, match="too short"):
        peek_schema_id(b"\x00\x00")


def test_schema_id_must_fit_in_four_bytes(sample_event_dict: dict) -> None:
    schema = load_schema("sct_inst_event.v1")
    with pytest.raises(WireFormatError, match="4 bytes"):
        encode(schema, 2**33, sample_event_dict)


def test_a_consumer_on_a_newer_schema_reads_older_messages(sample_event_dict: dict) -> None:
    """Schema resolution across a rolling upgrade.

    This is the behaviour the backward-compatibility gate in
    kg_core.contracts.registry is a proxy for, so it is exercised directly
    rather than trusted.
    """
    import fastavro

    old_dict = load_schema_dict("sct_inst_event.v1")
    new_dict = copy.deepcopy(old_dict)
    new_dict["fields"].append({"name": "risk_band", "type": "string", "default": "UNKNOWN"})

    old_schema = fastavro.parse_schema(old_dict)
    new_schema = fastavro.parse_schema(new_dict)

    payload = encode(old_schema, 42, sample_event_dict)
    restored = decode(payload, new_schema, old_schema)

    assert restored["risk_band"] == "UNKNOWN"
    assert restored["tx_id"] == sample_event_dict["tx_id"]


# ========================================================== the DQ gate =====


@pytest.fixture()
def payments_table(sample_event_dict: dict) -> pa.Table:
    """A small, valid batch shaped like bronze."""
    from kg_lake.medallion import _event_date

    rows = []
    for index in range(20):
        row = dict(sample_event_dict)
        row["tx_id"] = f"tx-{index:04d}"
        row["acceptance_ts"] = sample_event_dict["acceptance_ts"] + index * 1000
        row["ingest_ts"] = row["acceptance_ts"] + 40
        row["event_date"] = _event_date(row["acceptance_ts"])
        rows.append(row)
    return pa.Table.from_pylist(rows)


def test_a_clean_batch_passes_every_expectation(payments_table: pa.Table) -> None:
    from kg_lake.quality import validate

    report = validate(payments_table, "payments")
    failures = [r.name for r in report.results if not r.passed]
    assert report.ok and not failures, f"unexpected failures: {failures}"


def test_self_payment_blocks_the_promotion(payments_table: pa.Table) -> None:
    """The expectation that caught a real simulator bug on its first run.

    3 of 34,769 rows were payments from an account to itself, caused by a type
    confusion in the simulator's social-circle wiring. This is the regression
    test for that class of defect reaching silver.
    """
    from kg_lake.quality import DataQualityError, validate

    rows = payments_table.to_pylist()
    rows[3]["creditor_account_id"] = rows[3]["debtor_account_id"]
    report = validate(pa.Table.from_pylist(rows), "payments")

    assert not report.ok
    assert [r.name for r in report.blocking_failures] == ["debtor_and_creditor_differ"]
    with pytest.raises(DataQualityError, match="debtor_and_creditor_differ"):
        report.raise_if_blocked()


def test_a_clock_fault_blocks_the_promotion(payments_table: pa.Table) -> None:
    """ingest_ts before acceptance_ts means an event observed before it happened."""
    from kg_lake.quality import validate

    rows = payments_table.to_pylist()
    rows[0]["ingest_ts"] = rows[0]["acceptance_ts"] - 5
    report = validate(pa.Table.from_pylist(rows), "payments")
    assert "ingest_not_before_acceptance" in [r.name for r in report.blocking_failures]


def test_an_undeclared_enum_value_blocks_the_promotion(payments_table: pa.Table) -> None:
    """A value outside the contract's domain means the producer drifted."""
    from kg_lake.quality import validate

    rows = payments_table.to_pylist()
    rows[1]["vop_result"] = "PROBABLY_FINE"
    report = validate(pa.Table.from_pylist(rows), "payments")
    assert "vop_result_in_domain" in [r.name for r in report.blocking_failures]


def test_a_bad_iban_is_quarantined_not_blocked(payments_table: pa.Table) -> None:
    """Row-level defects divert the row and let the batch through."""
    from kg_lake.quality import partition_valid_rows, validate

    rows = payments_table.to_pylist()
    rows[2]["debtor_iban"] = "DE00370400440532013000"  # fails MOD-97-10
    table = pa.Table.from_pylist(rows)

    report = validate(table, "payments")
    assert report.ok, "an invalid IBAN must not block the whole batch"
    assert "debtor_iban_valid" in [r.name for r in report.quarantine_failures]

    clean, quarantined, reasons = partition_valid_rows(table, "payments")
    assert quarantined.num_rows == 1
    assert clean.num_rows == len(rows) - 1
    assert "debtor_iban_valid" in reasons


def test_duplicates_are_quarantined_keeping_the_first(payments_table: pa.Table) -> None:
    from kg_lake.quality import partition_valid_rows

    rows = payments_table.to_pylist()
    rows.append(dict(rows[0]))  # redelivery of the same transaction
    clean, quarantined, _ = partition_valid_rows(pa.Table.from_pylist(rows), "payments")

    assert quarantined.num_rows == 1
    assert clean.num_rows == len(rows) - 1
    assert clean.column("tx_id").to_pylist().count(rows[0]["tx_id"]) == 1


def test_warnings_do_not_block(payments_table: pa.Table) -> None:
    """A VoP mismatch is information, not an error."""
    from kg_lake.quality import validate

    rows = payments_table.to_pylist()
    for row in rows:
        row["vop_result"] = "NO_MATCH"
    report = validate(pa.Table.from_pylist(rows), "payments")

    assert report.ok, "distributional warnings must never block the pipeline"
    assert "vop_mostly_matches" in [r.name for r in report.warnings]


def test_nulls_count_as_failures(payments_table: pa.Table) -> None:
    """ "Unknown" is not "valid" — a null must never pass an expectation."""
    from kg_lake.quality import validate

    rows = payments_table.to_pylist()
    rows[0]["tx_id"] = None
    report = validate(pa.Table.from_pylist(rows), "payments")
    assert "tx_id_present" in [r.name for r in report.blocking_failures]


def test_expectation_domains_match_the_avro_contract() -> None:
    """The quality suite's enum domains must not drift from the contract.

    Two places listing the same values is a drift risk; this is how it is
    closed, the same way Avro and pydantic parity is enforced.
    """
    from kg_lake.quality import _CHANNELS, _SETTLEMENT, _VOP_RESULTS

    fields = {f["name"]: f for f in load_schema_dict("sct_inst_event.v1")["fields"]}
    for column, expected in (
        ("vop_result", _VOP_RESULTS),
        ("channel", _CHANNELS),
        ("settlement_status", _SETTLEMENT),
    ):
        symbols = set(fields[column]["type"]["symbols"])
        assert symbols == expected, f"{column}: contract has {symbols}, suite has {expected}"


def test_empty_batch_is_vacuously_valid() -> None:
    from kg_lake.quality import validate

    report = validate(pa.Table.from_pylist([], schema=pa.schema([("tx_id", pa.string())])))
    assert report.ok and report.total_rows == 0


# ========================================================= the lakehouse ====


@pytest.fixture()
def lake(tmp_path):
    """A local-filesystem Iceberg catalog with the tables created."""
    from kg_lake.catalog import local_catalog
    from kg_lake.medallion import ensure_tables

    catalog = local_catalog(tmp_path / "warehouse")
    ensure_tables(catalog)
    return catalog


def test_every_managed_table_is_created(lake) -> None:
    from kg_lake.schemas import TABLES

    for identifier in TABLES:
        assert lake.load_table(identifier) is not None


def test_bronze_is_append_only_and_carries_kafka_coordinates(lake, sample_event_dict) -> None:
    """The provenance that makes bronze an audit substrate rather than a copy."""
    from dataclasses import dataclass

    from kg_lake.medallion import bronze_rows_from_messages, write_bronze

    @dataclass
    class FakeMessage:
        value: dict
        topic: str
        partition: int
        offset: int
        timestamp_ms: int
        schema_id: int

    messages = [
        FakeMessage(
            value={**sample_event_dict, "tx_id": f"tx-{i}"},
            topic="kontograph.payments.sctinst.v1",
            partition=i % 3,
            offset=100 + i,
            timestamp_ms=sample_event_dict["acceptance_ts"],
            schema_id=4,
        )
        for i in range(5)
    ]
    rows = list(bronze_rows_from_messages(messages))
    write_bronze(rows, lake)

    stored = lake.load_table("bronze.payments").scan().to_arrow()
    assert stored.num_rows == 5
    assert set(stored.column("kafka_offset").to_pylist()) == {100, 101, 102, 103, 104}
    assert stored.column("writer_schema_id").to_pylist() == [4] * 5


def test_partition_date_is_utc_not_host_local() -> None:
    """A late-evening payment must not land in tomorrow's partition.

    date.fromtimestamp() takes no tz argument and silently uses the host
    timezone; east of UTC that shifts events across a partition boundary. This
    is a leakage bug wearing a partitioning costume.
    """
    from kg_core.timeutil import UTC, to_millis
    from kg_lake.medallion import _event_date

    late = to_millis(_dt.datetime(2025, 10, 9, 23, 30, tzinfo=UTC))
    assert _event_date(late) == _dt.date(2025, 10, 9)

    just_after = to_millis(_dt.datetime(2025, 10, 10, 0, 1, tzinfo=UTC))
    assert _event_date(just_after) == _dt.date(2025, 10, 10)


def test_iceberg_records_a_snapshot_per_write(lake, sample_event_dict) -> None:
    """Snapshots are the time-travel handle the lineage report depends on."""
    from dataclasses import dataclass

    from kg_lake.medallion import bronze_rows_from_messages, snapshot_history, write_bronze

    @dataclass
    class FakeMessage:
        value: dict
        topic: str = "t"
        partition: int = 0
        offset: int = 0
        timestamp_ms: int = 0
        schema_id: int = 4

    for batch in range(3):
        write_bronze(
            list(
                bronze_rows_from_messages(
                    [
                        FakeMessage(value={**sample_event_dict, "tx_id": f"b{batch}-{i}"})
                        for i in range(2)
                    ]
                )
            ),
            lake,
        )

    history = snapshot_history("bronze.payments", lake)
    assert len(history) == 3
    assert history[0]["parent_id"] is None
    assert all(history[i]["parent_id"] == history[i - 1]["snapshot_id"] for i in range(1, 3))


def test_simulated_events_get_the_same_silver_columns_as_the_lake(sample_event_dict) -> None:
    """Features are defined against silver; an in-memory stream must match it.

    ``kg features verify`` falls back to a freshly simulated stream when no
    dataset exists, which is exactly the situation in CI. Feeding it raw events
    instead of silver rows fails on the first derived column a feature reads.
    """
    from kg_lake.medallion import silver_rows_from_events

    accounts = [
        {"account_id": "ACC-000001", "bank_blz": "37040044", "country": "DE", "opened_ts": 0},
        {"account_id": "ACC-000002", "bank_blz": "12030000", "country": "AT", "opened_ts": 0},
    ]
    event = {**sample_event_dict, "creditor_country": "AT"}
    (row,) = silver_rows_from_events([event], accounts)

    assert row["is_cross_border"] is True
    assert row["is_cross_institution"] is True
    assert row["debtor_account_age_days"] > 0
    assert row["ingest_lag_ms"] == 40
    assert silver_rows_from_events([], accounts) == []


# ======================================================== integration =======


@pytest.mark.integration
def test_schema_registry_round_trip() -> None:
    """Register a contract with the live registry and decode by resolved id."""
    from kg_core.config import get_settings
    from kg_stream.registry import SchemaRegistry

    settings = get_settings()
    with SchemaRegistry(settings.schema_registry_url) as registry:
        assert registry.ping(), "the Schema Registry is not reachable; run `make up`"
        registered = registry.register_contract("sct_inst_event.v1")
        assert registered.schema_id > 0
        assert registry.schema_by_id(registered.schema_id) is not None

        # Registering the identical schema again must be idempotent.
        assert registry.register(registered.subject, load_schema_dict("sct_inst_event.v1")) == (
            registered.schema_id
        )


@pytest.mark.integration
def test_registry_rejects_an_incompatible_schema() -> None:
    """The server-side half of the compatibility gate."""
    from kg_core.config import get_settings
    from kg_stream.registry import SchemaRegistry

    settings = get_settings()
    with SchemaRegistry(settings.schema_registry_url) as registry:
        registered = registry.register_contract("sct_inst_event.v1")

        breaking = copy.deepcopy(load_schema_dict("sct_inst_event.v1"))
        breaking["fields"].append({"name": "mandatory_new_field", "type": "string"})
        assert not registry.check_compatibility(registered.subject, breaking), (
            "adding a field with no default must be rejected under BACKWARD compatibility"
        )
