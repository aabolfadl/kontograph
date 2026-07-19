"""The backfill compiler: feature IR to DuckDB SQL.

This is the path that builds training data, so it is the one where a leak would
be most damaging and least visible — a model trained on leaked features scores
beautifully offline and collapses in production.

**How the strict bound is achieved.** A self-join on the entity key with

    h.acceptance_ts <  e.acceptance_ts
    h.acceptance_ts >= e.acceptance_ts - W

Not a window function. DuckDB's ``OVER (... RANGE BETWEEN INTERVAL 'W'
PRECEDING AND CURRENT ROW)`` would include the current row *and* every other row
sharing its timestamp, and correcting for that is where the Flink compiler has
to do real work (see :mod:`kg_features.compilers.flink`). The join expresses the
definition directly, and a reader can check it against the IR docstring by eye —
which matters more here than shaving a few seconds off a backfill.

**One join per (entity, window) group, not one per feature.** Thirty features
over four distinct groups become four scans.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kg_features.ir import (
    EVENT_TIME_COLUMN,
    Aggregation,
    Entity,
    FeatureSet,
    FeatureSpec,
    UnsupportedOnBackendError,
)


@dataclass(frozen=True, slots=True)
class CompiledQuery:
    """Generated SQL plus what it produces."""

    sql: str
    columns: tuple[str, ...]
    groups: int

    def __str__(self) -> str:
        return self.sql


def _key_join(entity: Entity) -> str:
    return " AND ".join(f"h.{column} = e.{column}" for column in entity.key_columns)


def _filtered(expression: str, feature: FeatureSpec) -> str:
    """Wrap an aggregate in a FILTER clause when the feature has predicates."""
    if not feature.filters:
        return expression
    return f"{expression} FILTER (WHERE {feature.filters_sql('h')})"


def _aggregate_sql(feature: FeatureSpec) -> str:
    """The aggregate expression for one feature, over the joined history rows."""
    column = f"h.{feature.source_column}" if feature.source_column else None
    default = feature.default

    if feature.aggregation is Aggregation.COUNT:
        # COUNT(h.tx_id), never COUNT(*). With a LEFT JOIN an unmatched event
        # still produces one row whose right-hand columns are NULL, and COUNT(*)
        # counts that row -- so every event with no history scored 1 instead of
        # 0. Counting a NOT NULL column from the right side counts matches.
        expression = _filtered("COUNT(h.tx_id)", feature)
        # Zero matches means an empty window, which is when the declared default
        # applies. NULLIF routes that through COALESCE; with the usual default
        # of 0 it is a no-op, but it keeps a non-zero default honest.
        return f"COALESCE(NULLIF(CAST({expression} AS DOUBLE), 0.0), {default})"

    if feature.aggregation is Aggregation.SUM:
        expression = _filtered(f"SUM({column})", feature)
        return f"COALESCE(CAST({expression} AS DOUBLE), {default})"

    if feature.aggregation is Aggregation.MEAN:
        expression = _filtered(f"AVG({column})", feature)
        return f"COALESCE(CAST({expression} AS DOUBLE), {default})"

    if feature.aggregation is Aggregation.MAX:
        expression = _filtered(f"MAX({column})", feature)
        return f"COALESCE(CAST({expression} AS DOUBLE), {default})"

    if feature.aggregation is Aggregation.MIN:
        expression = _filtered(f"MIN({column})", feature)
        return f"COALESCE(CAST({expression} AS DOUBLE), {default})"

    if feature.aggregation is Aggregation.STDDEV:
        # Population, not sample: STDDEV_SAMP over a single prior event is
        # undefined and each engine invents its own answer. STDDEV_POP is
        # defined for n >= 1 and returns 0, identically everywhere.
        expression = _filtered(f"STDDEV_POP({column})", feature)
        return f"COALESCE(CAST({expression} AS DOUBLE), {default})"

    if feature.aggregation is Aggregation.DISTINCT_COUNT:
        # Same NULL-vs-zero distinction as COUNT: COUNT(DISTINCT ...) returns 0
        # rather than NULL when nothing matches.
        expression = _filtered(f"COUNT(DISTINCT {column})", feature)
        return f"COALESCE(NULLIF(CAST({expression} AS DOUBLE), 0.0), {default})"

    if feature.aggregation is Aggregation.TIME_SINCE_LAST:
        # The most recent prior matching event. MAX over a strictly-earlier set
        # is the previous event by definition, so no ordering is needed.
        expression = _filtered(f"MAX(h.{EVENT_TIME_COLUMN})", feature)
        return (
            f"COALESCE(CAST(e.{EVENT_TIME_COLUMN} AS DOUBLE) "
            f"- CAST({expression} AS DOUBLE), {default})"
        )

    raise AssertionError(f"unhandled aggregation {feature.aggregation}")


def compile_group(
    features: list[FeatureSpec],
    entity: Entity,
    window_millis: int | None,
    *,
    source_table: str = "events",
    cte_name: str = "g0",
) -> str:
    """SQL for one (entity, window) group as a CTE."""
    window_clause = (
        f"\n     AND h.{EVENT_TIME_COLUMN} >= e.{EVENT_TIME_COLUMN} - {window_millis}"
        if window_millis is not None
        else ""
    )
    projections = ",\n        ".join(
        f"{_aggregate_sql(feature)} AS {feature.name}" for feature in features
    )
    return f"""{cte_name} AS (
    SELECT
        e.tx_id,
        {projections}
    FROM {source_table} e
    LEFT JOIN {source_table} h
      ON {_key_join(entity)}
     -- STRICT: the event being scored never contributes to its own features.
     AND h.{EVENT_TIME_COLUMN} < e.{EVENT_TIME_COLUMN}{window_clause}
    GROUP BY e.tx_id, e.{EVENT_TIME_COLUMN}
)"""


def compile_feature_set(
    feature_set: FeatureSet,
    *,
    source_table: str = "events",
    include_keys: bool = True,
) -> CompiledQuery:
    """Compile a whole feature set into one DuckDB query.

    Emits one CTE per ``(entity, window)`` group, then joins them on ``tx_id``.
    Column order follows the feature set's declared order, because the model
    consumes the vector positionally.
    """
    groups = feature_set.grouped()
    ctes: list[str] = []
    names: list[str] = []

    for index, ((entity_value, _window), features) in enumerate(groups.items()):
        cte = f"g{index}"
        names.append(cte)
        ctes.append(
            compile_group(
                features,
                Entity(entity_value),
                features[0].window_millis,
                source_table=source_table,
                cte_name=cte,
            )
        )

    # Map each feature back to the CTE that produced it, so the final projection
    # can be emitted in the feature set's order rather than in group order.
    owner: dict[str, str] = {}
    for cte, (_, features) in zip(names, groups.items(), strict=True):
        for feature in features:
            owner[feature.name] = cte

    key_columns = (
        [
            "base.tx_id",
            f"base.{EVENT_TIME_COLUMN}",
            "base.debtor_account_id",
            "base.creditor_account_id",
        ]
        if include_keys
        else ["base.tx_id"]
    )
    projections = key_columns + [f"{owner[name]}.{name}" for name in feature_set.column_names]
    joins = "\n".join(f"    LEFT JOIN {cte} ON {cte}.tx_id = base.tx_id" for cte in names)

    sql = (
        "WITH\n"
        + ",\n".join(ctes)
        + "\nSELECT\n    "
        + ",\n    ".join(projections)
        + f"\nFROM {source_table} base\n"
        + joins
        + f"\nORDER BY base.{EVENT_TIME_COLUMN}, base.tx_id"
    )

    columns = tuple(([c.split(".", 1)[1] for c in key_columns]) + list(feature_set.column_names))
    return CompiledQuery(sql=sql, columns=columns, groups=len(groups))


def execute(
    feature_set: FeatureSet,
    events: Any,
    *,
    connection: Any = None,
) -> Any:
    """Compute features for an Arrow table of events. Returns an Arrow table.

    ``events`` must be sorted by event time or not — the SQL does not care,
    because the join is defined on values rather than on row order. That is
    itself worth noting: a backfill whose correctness depends on input ordering
    is a backfill that will eventually be run on unsorted input.
    """
    import duckdb

    own_connection = connection is None
    connection = connection or duckdb.connect(":memory:")
    try:
        connection.register("events_input", events)
        connection.execute("CREATE OR REPLACE TEMP VIEW events AS SELECT * FROM events_input")
        query = compile_feature_set(feature_set)
        return _fetch_arrow(connection.execute(query.sql))
    finally:
        if own_connection:
            connection.close()


def _fetch_arrow(result: Any) -> Any:
    """Materialise a DuckDB result as an Arrow table.

    ``fetch_arrow_table`` is deprecated in current DuckDB in favour of
    ``to_arrow_table``; older versions have only the former. Probing keeps this
    working across both rather than pinning a DuckDB version for a method name.
    """
    if hasattr(result, "to_arrow_table"):
        return result.to_arrow_table()
    return result.fetch_arrow_table()


# ---------------------------------------------------------------- chunking --

#: Days of events processed per chunk. Each chunk additionally loads the
#: feature set's longest window as lookback, so a 30-day chunk with a 30-day
#: window holds 60 days of data at a time.
DEFAULT_CHUNK_DAYS = 30


def execute_chunked(
    feature_set: FeatureSet,
    events: Any,
    *,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
    connection: Any = None,
    on_progress: Any = None,
) -> Any:
    """Compute features in time slices, each with enough history to be exact.

    **Why chunking is necessary, not merely tidy.** The backfill is a self-join
    keyed by entity, so its intermediate size is O(rows x events-in-window). That
    is fine for a household with twenty payments a month. It is not fine for a
    utility provider receiving from thousands of households: a 30-day window on
    such an account holds tens of thousands of events, and the join for that one
    key alone produces hundreds of millions of rows. On the 270-day profile
    (1.56 M events) DuckDB exhausted its temp-spill budget and failed outright.

    **Why the result is still exact.** A feature at time ``t`` depends only on
    ``[t - W, t)``. So a chunk covering ``[c0, c1)`` needs the events in
    ``[c0 - W, c1)`` to compute its own rows correctly — and the lookback rows
    are used as *history only*, never emitted. This is precisely how an
    incremental backfill works in production, and the equivalence is checked by
    ``test_chunked_backfill_matches_the_single_pass``.

    Unbounded features are the exception and are handled by falling back to a
    single pass: with no window there is no bounded lookback that would be
    correct, and quietly truncating their history would be the leakage this
    layer exists to prevent.
    """
    import duckdb
    import pyarrow as pa

    from kg_core.timeutil import MILLIS_PER_DAY

    windowed = [f for f in feature_set if f.window_millis is not None]
    unbounded = [f for f in feature_set if f.window_millis is None]

    # Unbounded features have no bounded lookback, so they cannot be chunked.
    # TIME_SINCE_LAST does not need to be: it is a window function, not a
    # self-join, and runs in one O(n log n) pass over the whole stream. Any
    # OTHER unbounded aggregation would need the full self-join and is refused
    # rather than silently computed against a truncated history.
    unsupported = [f.name for f in unbounded if f.aggregation is not Aggregation.TIME_SINCE_LAST]
    if unsupported:
        raise UnsupportedOnBackendError(
            f"unbounded {unsupported} cannot be chunked: a bounded lookback would "
            "truncate their history and silently change their value. Give them a "
            "window, or compute them with execute() on a dataset small enough for "
            "a single pass."
        )

    stamps = events.column(EVENT_TIME_COLUMN).to_numpy(zero_copy_only=False)
    if len(stamps) == 0:
        return execute(feature_set, events, connection=connection)

    windowed_set = FeatureSet(
        name=f"{feature_set.name}-windowed", features=tuple(windowed), version=feature_set.version
    )
    lookback = feature_set.max_window_millis
    start, end = int(stamps.min()), int(stamps.max()) + 1
    step = chunk_days * MILLIS_PER_DAY

    own_connection = connection is None
    connection = connection or duckdb.connect(":memory:")
    query = compile_feature_set(windowed_set)
    pieces = []

    try:
        chunk_index = 0
        chunk_count = max(1, -(-(end - start) // step))
        for chunk_start in range(start, end, step):
            chunk_end = min(chunk_start + step, end)
            window_mask = (stamps >= chunk_start - lookback) & (stamps < chunk_end)
            if not window_mask.any():
                continue

            slice_table = events.filter(pa.array(window_mask))
            connection.register("events_input", slice_table)
            connection.execute("CREATE OR REPLACE TEMP VIEW events AS SELECT * FROM events_input")
            computed = _fetch_arrow(connection.execute(query.sql))

            # Emit only this chunk's own rows; the lookback was history.
            emitted = computed.column(EVENT_TIME_COLUMN).to_numpy(zero_copy_only=False)
            pieces.append(
                computed.filter(pa.array((emitted >= chunk_start) & (emitted < chunk_end)))
            )
            connection.unregister("events_input")

            chunk_index += 1
            if on_progress is not None:
                on_progress(chunk_index, chunk_count, pieces[-1].num_rows)
    finally:
        if own_connection:
            connection.close()

    if not pieces:
        return execute(feature_set, events)

    result = pa.concat_tables(pieces)
    if unbounded:
        result = _attach_unbounded(result, events, unbounded, connection=None)
    # Restore the declared column order: the model consumes the vector
    # positionally, so the order is part of the contract.
    keys = [c for c in result.column_names if c not in feature_set.column_names]
    return result.select(keys + list(feature_set.column_names))


def _attach_unbounded(
    computed: Any, events: Any, features: list[FeatureSpec], *, connection: Any
) -> Any:
    """Add unbounded TIME_SINCE_LAST columns, computed in one pass.

    The gap since an entity's previous event is a window function, not an
    aggregation over a window, so it costs a sort rather than a join. The
    subtlety is ties: LAG over the raw rows would return an event sharing the
    current timestamp, which the semantics forbid. Taking LAG over the DISTINCT
    (key, timestamp) pairs gives the largest timestamp STRICTLY less than the
    current one, which is exactly the definition.
    """
    import duckdb
    import pyarrow as pa

    own = connection is None
    connection = connection or duckdb.connect(":memory:")
    try:
        connection.register("events_input", events)
        connection.register("computed_input", computed)
        result = computed

        for feature in features:
            keys = ", ".join(feature.entity.key_columns)
            where = f"WHERE {feature.filters_sql('e')}" if feature.filters else ""
            sql = f"""
                WITH moments AS (
                    SELECT DISTINCT {keys}, {EVENT_TIME_COLUMN}
                    FROM events_input e {where}
                ),
                gaps AS (
                    SELECT {keys}, {EVENT_TIME_COLUMN},
                           LAG({EVENT_TIME_COLUMN}) OVER (
                               PARTITION BY {keys} ORDER BY {EVENT_TIME_COLUMN}
                           ) AS previous_ts
                    FROM moments
                )
                SELECT e.tx_id,
                       COALESCE(
                           CAST(e.{EVENT_TIME_COLUMN} AS DOUBLE) - CAST(g.previous_ts AS DOUBLE),
                           {feature.default}
                       ) AS {feature.name}
                FROM events_input e
                LEFT JOIN gaps g
                  ON {" AND ".join(f"g.{c} = e.{c}" for c in feature.entity.key_columns)}
                 AND g.{EVENT_TIME_COLUMN} = e.{EVENT_TIME_COLUMN}
            """
            column = _fetch_arrow(connection.execute(sql))
            lookup = dict(
                zip(
                    column.column("tx_id").to_pylist(),
                    column.column(feature.name).to_pylist(),
                    strict=True,
                )
            )
            values = [lookup.get(tx, feature.default) for tx in result.column("tx_id").to_pylist()]
            result = result.append_column(feature.name, pa.array(values, type=pa.float64()))

        return result
    finally:
        if own:
            connection.close()
