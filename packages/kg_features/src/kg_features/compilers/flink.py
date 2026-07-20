"""The streaming compiler: feature IR to Flink SQL.

# The constraint that shapes everything here

A Flink SQL ``OVER`` window's frame **must end at ``CURRENT ROW``**. There is no
``... AND 1 PRECEDING``, and no way to write "everything up to but excluding the
current row". Our semantics are exactly that exclusion (see
:mod:`kg_features.ir`), so the compiler has to recover it.

# The correction

Compute the aggregate over the inclusive frame, then subtract the contribution
of the rows at the current timestamp:

    strictly_before = agg OVER (RANGE BETWEEN W PRECEDING AND CURRENT ROW)
                    - agg OVER (RANGE BETWEEN 0 PRECEDING AND CURRENT ROW)

The second frame is the *tie group*: every row sharing the current row's
ordering value. Subtracting the whole tie group rather than just "one row"
matters, and is not a detail — ``RANGE`` frames are value-based, so two payments
from the same account in the same millisecond are both inside the frame. A naive
``COUNT(*) OVER w - 1`` is correct only while timestamps are unique per entity,
which is an assumption that holds in every test fixture and fails in production.

The property suite generates duplicate timestamps deliberately for this reason,
and :func:`kg_features.compilers.flink.compile_correction` is validated against
the join-based backfill on those streams.

# What cannot be compiled, and why we refuse rather than approximate

The correction requires the aggregate to be *subtractable*: COUNT, SUM, MEAN and
STDDEV are (the last two via running sums of x and x²). MAX, MIN and
DISTINCT_COUNT are not — removing a value from a maximum needs the
second-largest, which the aggregate does not carry.

For those, this compiler raises :class:`~kg_features.ir.UnsupportedOnBackendError`.
Emitting *nearly* correct streaming SQL would be the training-serving skew this
entire layer exists to eliminate, so the honest move is to fail loudly and let
the feature set decide: either drop the feature from the streaming path, or
compute it in the online executor, which has no such restriction.

``FeatureSet.flink_compatible()`` reports the split up front, so this is visible
when features are defined rather than when they are deployed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from kg_features.ir import (
    EVENT_TIME_COLUMN,
    Aggregation,
    Entity,
    FeatureSet,
    FeatureSpec,
    UnsupportedOnBackendError,
)

Dialect = Literal["flink", "duckdb"]


@dataclass(frozen=True, slots=True)
class CompiledStreamQuery:
    """Generated streaming SQL plus what it covers."""

    ddl: str
    sql: str
    columns: tuple[str, ...]
    skipped: tuple[str, ...]

    def __str__(self) -> str:
        return f"{self.ddl}\n\n{self.sql}"


def _partition_by(entity: Entity) -> str:
    return ", ".join(entity.key_columns)


def _range_bound(window_millis: int | None, dialect: Dialect) -> str:
    """The lower bound of a RANGE frame, WITHOUT the trailing PRECEDING.

    The caller appends PRECEDING, so this must not include it -- an earlier
    version returned "UNBOUNDED PRECEDING" here and produced
    "RANGE BETWEEN UNBOUNDED PRECEDING PRECEDING", which DuckDB rejects and
    Flink would too.
    """
    if window_millis is None:
        return "UNBOUNDED"
    if dialect == "flink":
        # Flink orders on a TIMESTAMP rowtime attribute, so the bound is an
        # INTERVAL rather than a number.
        return f"INTERVAL '{window_millis / 1000:.3f}' SECOND"
    # DuckDB orders on the raw BIGINT millis, so the bound is numeric.
    return f"{window_millis}"


def _window_clause(feature: FeatureSpec, dialect: Dialect, *, ties_only: bool) -> str:
    order_column = "rowtime" if dialect == "flink" else EVENT_TIME_COLUMN
    if ties_only:
        bound = "INTERVAL '0.000' SECOND" if dialect == "flink" else "0"
    else:
        bound = _range_bound(feature.window_millis, dialect)
    return (
        f"PARTITION BY {_partition_by(feature.entity)} "
        f"ORDER BY {order_column} "
        f"RANGE BETWEEN {bound} PRECEDING AND CURRENT ROW"
    )


def _assert_frame_is_well_formed(clause: str) -> None:
    """Guard against the double-PRECEDING class of bug."""
    assert "PRECEDING PRECEDING" not in clause, clause


def _filtered(expression: str, feature: FeatureSpec) -> str:
    """Apply a feature's filters inside the aggregate.

    ``FILTER (WHERE ...)`` is not available on Flink OVER aggregates, so the
    predicate is pushed into a CASE expression instead. Same semantics: a
    non-matching row contributes NULL, and aggregates ignore NULLs.
    """
    if not feature.filters:
        return expression
    condition = feature.filters_sql("")
    condition = condition.replace(".", "")  # unqualified inside the CASE
    return f"CASE WHEN {condition} THEN {expression} END"


def compile_correction(feature: FeatureSpec, dialect: Dialect = "flink") -> str:
    """The strictly-before aggregate for one feature, as an expression.

    Split out from the full query so the *algebra* — the part that is easy to
    get subtly wrong — can be executed in DuckDB and diffed against the
    join-based backfill, without needing a Flink cluster. The remaining risk is
    dialect syntax, which the integration test against a real cluster covers.
    """
    if not feature.aggregation.is_subtractable:
        raise UnsupportedOnBackendError(
            f"{feature.name}: {feature.aggregation.value} cannot be compiled to Flink SQL. "
            "A Flink OVER frame must end at CURRENT ROW, so the current row's "
            "contribution has to be subtracted afterwards -- which is impossible for "
            f"{feature.aggregation.value} (removing a value from it requires "
            "information the aggregate does not retain). Compute this feature in the "
            "online executor, or drop it from the streaming path. See "
            "FeatureSet.flink_compatible()."
        )

    full = _window_clause(feature, dialect, ties_only=False)
    ties = _window_clause(feature, dialect, ties_only=True)
    default = feature.default

    if feature.aggregation is Aggregation.COUNT:
        inner = _filtered("1", feature)
        counted = f"COUNT({inner})"
        # Subtract the whole tie group, not 1: RANGE frames are value-based, so
        # every row sharing this timestamp is inside the frame.
        strictly_before = f"({counted} OVER ({full}) - {counted} OVER ({ties}))"
        return f"CASE WHEN {strictly_before} = 0 THEN {default} ELSE CAST({strictly_before} AS DOUBLE) END"

    column = feature.source_column
    # Guaranteed by FeatureSpec.__post_init__: every subtractable aggregation
    # except COUNT needs a source column, and COUNT returned above.
    assert column is not None
    value = _filtered(column, feature)
    counted = f"COUNT({value})"
    summed = f"SUM({value})"
    n = f"({counted} OVER ({full}) - {counted} OVER ({ties}))"
    s1 = f"({summed} OVER ({full}) - {summed} OVER ({ties}))"

    if feature.aggregation is Aggregation.SUM:
        return f"CASE WHEN {n} = 0 THEN {default} ELSE CAST({s1} AS DOUBLE) END"

    if feature.aggregation is Aggregation.MEAN:
        return f"CASE WHEN {n} = 0 THEN {default} ELSE CAST({s1} AS DOUBLE) / {n} END"

    if feature.aggregation is Aggregation.STDDEV:
        # Population stddev from running sums of x and x^2:
        #     var = E[x^2] - E[x]^2
        # GREATEST(..., 0) guards the catastrophic-cancellation case where
        # floating point drives a mathematically non-negative variance slightly
        # below zero and SQRT would return NaN.
        squared = _filtered(f"CAST({column} AS DOUBLE) * CAST({column} AS DOUBLE)", feature)
        s2 = f"(SUM({squared}) OVER ({full}) - SUM({squared}) OVER ({ties}))"
        variance = f"({s2} / {n} - ({s1} / CAST({n} AS DOUBLE)) * ({s1} / CAST({n} AS DOUBLE)))"
        # GREATEST is spelled the same in both dialects.
        return f"CASE WHEN {n} = 0 THEN {default} ELSE SQRT(GREATEST({variance}, 0.0)) END"

    raise AssertionError(f"unreachable: {feature.aggregation} claimed subtractable")


SOURCE_DDL = """-- Source table: the payments topic, decoded through the Schema Registry.
--
-- rowtime is derived from acceptance_ts and carries the watermark, so every
-- window in this job is defined on EVENT time. Using processing time here would
-- make the streaming features disagree with the backfill on replay, which is
-- exactly the skew this layer exists to prevent.
CREATE TABLE payments (
    tx_id                 STRING,
    acceptance_ts         BIGINT,
    debtor_account_id     STRING,
    creditor_account_id   STRING,
    amount_cents          BIGINT,
    currency              STRING,
    vop_result            STRING,
    vop_name_similarity   DOUBLE,
    vop_overridden        BOOLEAN,
    is_new_beneficiary    BOOLEAN,
    channel               STRING,
    initiation_source     STRING,
    settlement_status     STRING,
    rowtime AS TO_TIMESTAMP_LTZ(acceptance_ts, 3),
    -- 5 seconds of lateness tolerance. Under SCT Inst the whole settlement
    -- window is ten seconds, so a larger allowance would mean holding results
    -- past the point at which they could still act on the payment.
    WATERMARK FOR rowtime AS rowtime - INTERVAL '5' SECOND
) WITH (
    'connector' = 'kafka',
    'topic' = '{topic}',
    'properties.bootstrap.servers' = '{bootstrap}',
    'properties.group.id' = 'kontograph-features',
    'scan.startup.mode' = 'earliest-offset',
    'format' = 'avro-confluent',
    'avro-confluent.url' = '{registry}'
);"""


def compile_feature_set(
    feature_set: FeatureSet,
    *,
    dialect: Dialect = "flink",
    topic: str = "kontograph.payments.sctinst.v1",
    bootstrap: str = "redpanda:9092",
    registry: str = "http://redpanda:8081",
    source_table: str = "payments",
    strict: bool = False,
) -> CompiledStreamQuery:
    """Compile the streaming half of a feature set.

    ``strict=True`` raises on the first feature Flink cannot express.
    ``strict=False`` (the default) skips them and reports them in ``skipped`` —
    which is the useful behaviour, because a feature set is expected to contain
    features that only the online executor computes.
    """
    projections: list[str] = [
        "tx_id",
        EVENT_TIME_COLUMN,
        "debtor_account_id",
        "creditor_account_id",
    ]
    columns: list[str] = list(projections)
    skipped: list[str] = []

    for feature in feature_set:
        try:
            expression = compile_correction(feature, dialect)
        except UnsupportedOnBackendError:
            if strict:
                raise
            skipped.append(feature.name)
            continue
        projections.append(f"{expression} AS {feature.name}")
        columns.append(feature.name)

    ddl = SOURCE_DDL.format(topic=topic, bootstrap=bootstrap, registry=registry)
    body = ",\n    ".join(projections)
    sql = f"SELECT\n    {body}\nFROM {source_table}"

    return CompiledStreamQuery(
        ddl=ddl,
        sql=sql,
        columns=tuple(columns),
        skipped=tuple(skipped),
    )


def compile_for_duckdb(feature_set: FeatureSet) -> str:
    """The same OVER-window algebra, in a dialect DuckDB can execute.

    This exists so the *correction* can be verified without a Flink cluster.
    DuckDB implements standard SQL ``RANGE`` frames with the same value-based
    tie semantics, so running this and diffing against the join-based backfill
    tests the part of the streaming compiler most likely to be wrong.
    """
    return compile_feature_set(feature_set, dialect="duckdb", source_table="events").sql
