"""Compiler-level tests, including the Flink streaming compiler.

The Flink compiler cannot be executed here without a JVM cluster, so its
*algebra* is tested where it can be: the same OVER-window correction is emitted
in a DuckDB-compatible dialect and diffed against the join-based backfill. That
covers the part most likely to be wrong — the tie-group subtraction — and leaves
only dialect syntax to the integration test in ``make flink-up``.
"""

from __future__ import annotations

import duckdb
import pyarrow as pa
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from kg_features.compilers.duckdb_sql import compile_feature_set as compile_duckdb
from kg_features.compilers.flink import (
    compile_correction,
    compile_for_duckdb,
)
from kg_features.compilers.flink import (
    compile_feature_set as compile_flink,
)
from kg_features.dsl import count, maximum, mean, stddev, total, when
from kg_features.ir import Entity, FeatureSet, UnsupportedOnBackendError
from kg_features.registry import PAYMENTS_V1
from kg_features.verify import run_duckdb

# Only subtractable aggregations: this is exactly the set Flink can express.
STREAMABLE = FeatureSet(
    name="streamable",
    features=(
        count("c_2h", Entity.DEBTOR, "2h"),
        count("c_filtered", Entity.DEBTOR, "2h", where=(when("vop_result", "!=", "MATCH"),)),
        count("c_unbounded", Entity.DEBTOR),
        total("s_2h", Entity.DEBTOR, "amount_cents", "2h"),
        mean("m_2h", Entity.DEBTOR, "amount_cents", "2h"),
        stddev("sd_2h", Entity.DEBTOR, "amount_cents", "2h"),
        count("cr_2h", Entity.CREDITOR, "2h"),
        total("pair_s_2h", Entity.PAIR, "amount_cents", "2h"),
    ),
)


@st.composite
def streams(draw):
    """Streams with frequent duplicate timestamps on the same entity.

    Duplicates are the whole point: a naive ``COUNT(*) OVER w - 1`` correction
    is right until two payments share a millisecond, which is exactly the case a
    realistic fixture would never produce.
    """
    size = draw(st.integers(min_value=2, max_value=25))
    stamps = sorted(
        draw(st.lists(st.integers(min_value=0, max_value=30), min_size=size, max_size=size))
    )
    events = []
    for index, tick in enumerate(stamps):
        debtor = draw(st.sampled_from(["A", "B", "C"]))
        creditor = draw(st.sampled_from([a for a in ["A", "B", "C"] if a != debtor]))
        events.append(
            {
                "tx_id": f"tx{index:03d}",
                # Coarse ticks -> many exact collisions.
                "acceptance_ts": 1_000_000 + tick * 600_000,
                "debtor_account_id": debtor,
                "creditor_account_id": creditor,
                "amount_cents": draw(st.integers(min_value=1, max_value=100_000)),
                "vop_result": draw(st.sampled_from(["MATCH", "NO_MATCH"])),
            }
        )
    return events


def _run_over_window(feature_set: FeatureSet, events: list[dict]) -> dict[str, dict]:
    """Execute the streaming compiler's algebra through DuckDB."""
    connection = duckdb.connect(":memory:")
    try:
        connection.register("ei", pa.Table.from_pylist(events))
        connection.execute("CREATE OR REPLACE TEMP VIEW events AS SELECT * FROM ei")
        rows = connection.execute(compile_for_duckdb(feature_set)).to_arrow_table().to_pylist()
    finally:
        connection.close()
    return {row["tx_id"]: row for row in rows}


@given(streams())
@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_over_window_correction_matches_the_join_backfill(events: list[dict]) -> None:
    """The streaming algebra must produce the backfill's numbers.

    If this diverges, a model trained on the backfill is served different
    numbers by the streaming job — training-serving skew, arriving through the
    one backend that is hardest to observe in production.
    """
    over = _run_over_window(STREAMABLE, events)
    join = run_duckdb(STREAMABLE, events)

    for tx_id, expected in join.items():
        for name in STREAMABLE.column_names:
            got = float(over[tx_id][name])
            want = float(expected[name])
            assert abs(got - want) <= 1e-6 * max(1.0, abs(want)), (
                f"{name} @ {tx_id}: OVER-window gave {got}, join gave {want}"
            )


def test_tie_group_subtraction_is_what_makes_it_correct() -> None:
    """A naive '- 1' correction is wrong, and here is the case that proves it.

    Three payments from the same account in the same millisecond. A RANGE frame
    is value-based, so all three are inside each other's frame; subtracting one
    row leaves two, when the right answer is zero.
    """
    events = [
        {
            "tx_id": f"tx{i}",
            "acceptance_ts": 2_000_000,
            "debtor_account_id": "A",
            "creditor_account_id": "B",
            "amount_cents": 100,
            "vop_result": "MATCH",
        }
        for i in range(3)
    ]
    feature_set = FeatureSet(name="ties", features=(count("c", Entity.DEBTOR, "2h"),))

    over = _run_over_window(feature_set, events)
    assert {float(row["c"]) for row in over.values()} == {0.0}, (
        "simultaneous events must not see each other; a '- 1' correction would give 2"
    )

    # And the inclusive frame really does contain all three, which is what makes
    # the correction necessary rather than merely defensive.
    connection = duckdb.connect(":memory:")
    connection.register("ei", pa.Table.from_pylist(events))
    connection.execute("CREATE OR REPLACE TEMP VIEW events AS SELECT * FROM ei")
    inclusive = (
        connection.execute(
            "SELECT COUNT(*) OVER (PARTITION BY debtor_account_id ORDER BY acceptance_ts "
            "RANGE BETWEEN 7200000 PRECEDING AND CURRENT ROW) AS c FROM events"
        )
        .to_arrow_table()
        .to_pylist()
    )
    connection.close()
    assert {row["c"] for row in inclusive} == {3}, "the inclusive frame holds the whole tie group"


def test_non_subtractable_aggregations_are_refused_not_approximated() -> None:
    """MAX cannot be corrected, so the compiler must fail rather than guess.

    Emitting nearly-correct streaming SQL would be precisely the skew this layer
    exists to eliminate, so the failure is deliberate and the message says what
    to do instead.
    """
    feature = maximum("mx", Entity.DEBTOR, "amount_cents", "2h")
    with pytest.raises(UnsupportedOnBackendError, match="must end at CURRENT ROW"):
        compile_correction(feature)


def test_the_streaming_split_is_visible_up_front() -> None:
    """A feature that cannot stream should be known when it is defined."""
    streamable = PAYMENTS_V1.flink_compatible()
    assert 0 < len(streamable) < len(PAYMENTS_V1)

    compiled = compile_flink(PAYMENTS_V1)
    assert set(compiled.skipped) == {
        f.name for f in PAYMENTS_V1 if not f.aggregation.is_subtractable
    }
    # Every skipped feature is MAX / MIN / DISTINCT_COUNT / TIME_SINCE_LAST.
    for name in compiled.skipped:
        feature = next(f for f in PAYMENTS_V1 if f.name == name)
        assert not feature.aggregation.is_subtractable


def test_strict_mode_raises_on_the_first_unsupported_feature() -> None:
    with pytest.raises(UnsupportedOnBackendError):
        compile_flink(PAYMENTS_V1, strict=True)


def test_generated_flink_ddl_uses_event_time_with_a_watermark() -> None:
    """Processing time would make the stream disagree with the backfill on replay."""
    compiled = compile_flink(STREAMABLE, topic="t", bootstrap="b:9092", registry="http://r")
    assert "WATERMARK FOR rowtime AS rowtime" in compiled.ddl
    assert "TO_TIMESTAMP_LTZ(acceptance_ts, 3)" in compiled.ddl
    assert "PROCTIME" not in compiled.ddl.upper()
    assert "'connector' = 'kafka'" in compiled.ddl
    assert "avro-confluent" in compiled.ddl


def test_flink_windows_order_on_rowtime_not_on_the_raw_column() -> None:
    compiled = compile_flink(STREAMABLE)
    assert "ORDER BY rowtime" in compiled.sql
    assert "INTERVAL" in compiled.sql


# ------------------------------------------------------- the DuckDB compiler --


def test_backfill_uses_a_strict_inequality() -> None:
    """The single most important character in the generated SQL."""
    sql = compile_duckdb(PAYMENTS_V1).sql
    assert "h.acceptance_ts < e.acceptance_ts" in sql
    assert "h.acceptance_ts <= e.acceptance_ts" not in sql


def test_backfill_never_counts_star_over_a_left_join() -> None:
    """COUNT(*) over a LEFT JOIN counts the unmatched row.

    That bug made every event with no history score 1 instead of 0. Counting a
    NOT NULL column from the right-hand side counts matches, which is what was
    meant.
    """
    sql = compile_duckdb(PAYMENTS_V1).sql
    assert "COUNT(*)" not in sql
    assert "COUNT(h.tx_id)" in sql


def test_features_are_grouped_into_one_scan_per_entity_and_window() -> None:
    compiled = compile_duckdb(PAYMENTS_V1)
    assert compiled.groups < len(PAYMENTS_V1), "grouping should beat one join per feature"
    assert compiled.groups == len(PAYMENTS_V1.grouped())


def test_output_column_order_follows_the_declared_order() -> None:
    """The model consumes the vector positionally."""
    compiled = compile_duckdb(PAYMENTS_V1)
    assert compiled.columns[-len(PAYMENTS_V1) :] == PAYMENTS_V1.column_names


def test_compiled_sql_parses() -> None:
    """Syntax check without executing, so a malformed query fails fast."""
    connection = duckdb.connect(":memory:")
    try:
        connection.execute(
            "CREATE TABLE events (tx_id VARCHAR, acceptance_ts BIGINT, "
            "debtor_account_id VARCHAR, creditor_account_id VARCHAR, amount_cents BIGINT, "
            "vop_result VARCHAR, vop_name_similarity DOUBLE, vop_overridden BOOLEAN, "
            "is_new_beneficiary BOOLEAN, is_cross_border BOOLEAN, channel VARCHAR)"
        )
        connection.execute(f"EXPLAIN {compile_duckdb(PAYMENTS_V1).sql}")
    finally:
        connection.close()


def test_feature_set_validates_against_a_real_schema() -> None:
    """A schema change that removes a column must fail loudly, not silently."""
    from kg_features.ir import validate_against_schema

    columns = {
        "tx_id",
        "acceptance_ts",
        "debtor_account_id",
        "creditor_account_id",
        "amount_cents",
        "vop_result",
        "vop_name_similarity",
        "vop_overridden",
        "is_new_beneficiary",
        "is_cross_border",
        "channel",
    }
    assert validate_against_schema(PAYMENTS_V1, columns) == []

    problems = validate_against_schema(PAYMENTS_V1, columns - {"vop_result"})
    assert problems and any("vop_result" in p for p in problems)


def test_silver_table_can_actually_supply_every_feature() -> None:
    """The feature set must be computable from the table it will really read.

    Cheap to check, and it closes the gap where a feature references a column
    that exists in the simulator's output but was never promoted to silver.
    """
    from kg_features.ir import validate_against_schema
    from kg_lake.schemas import SILVER_PAYMENTS

    columns = {field.name for field in SILVER_PAYMENTS.fields}
    assert validate_against_schema(PAYMENTS_V1, columns) == []
