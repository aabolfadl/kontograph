"""The feature intermediate representation.

This module is the single definition of what a feature *means*. Three compilers
lower it — DuckDB for backfill, Flink SQL for streaming, and an incremental
Python executor for serving — and all three are obliged to produce the number
defined here. That obligation is not a convention: the property suite generates
random event streams and diffs the backends against each other.

# The semantics, stated once

For an event ``e`` with entity key ``k = key(e)`` and event time ``t``:

    f(e) = AGG over  H = { h in events :
                              key(h) == k
                          AND t - W <= h.acceptance_ts < t
                          AND filter(h) }

with ``f(e) = default`` when ``H`` is empty.

Three details in that definition carry the entire anti-leakage guarantee, and
each of them is a bug somebody has shipped:

**``h.acceptance_ts < t`` is strict.** The event being scored never contributes
to its own features. An inclusive bound leaks the present into the past: a
"count of transfers in the last hour" that counts *this* transfer is trained on
information the model would not have at decision time. Strictness is the
difference between a backtest and a time machine.

**The bound is on ``acceptance_ts``, never on ingest or processing time.**
Processing time is a property of the pipeline, not of the payment, and a feature
computed against it produces a different value on replay than it did live.

**The window is left-closed, right-open: ``[t - W, t)``.** Stated explicitly so
the three compilers cannot each pick a different convention for the boundary —
which is exactly the kind of one-millisecond disagreement that never shows up in
a smoke test and always shows up in production.

# Why an IR at all

Because point-in-time correctness has to be a property we can *test*, not one we
promise to remember (ADR-0005). Owning the representation is what makes it
possible to assert that every compiled query orders on event time with a strict
inequality, to generate adversarial event streams, and to diff three backends
against one another. With a vendor feature store the strongest available claim
is "we configured it correctly".
"""

from __future__ import annotations

import enum
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

from kg_core.hashing import content_hash
from kg_core.timeutil import format_window, parse_window


class Entity(enum.StrEnum):
    """What a feature is keyed by.

    The choice of entity is the modelling decision, not a technicality. The same
    aggregation keyed differently answers a different question: keyed by DEBTOR
    it asks "is this account behaving unusually?", keyed by CREDITOR it asks "is
    this account receiving unusually?", and keyed by PAIR it asks "is this
    relationship unusual?" — which is the one that catches a mule collector,
    because each individual relationship looks fine.
    """

    DEBTOR = "debtor"
    CREDITOR = "creditor"
    PAIR = "pair"

    @property
    def key_columns(self) -> tuple[str, ...]:
        return {
            Entity.DEBTOR: ("debtor_account_id",),
            Entity.CREDITOR: ("creditor_account_id",),
            Entity.PAIR: ("debtor_account_id", "creditor_account_id"),
        }[self]


class Aggregation(enum.StrEnum):
    """The supported aggregations."""

    COUNT = "count"
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    STDDEV = "stddev"
    DISTINCT_COUNT = "distinct_count"
    TIME_SINCE_LAST = "time_since_last"

    @property
    def needs_source_column(self) -> bool:
        return self not in (Aggregation.COUNT, Aggregation.TIME_SINCE_LAST)

    @property
    def is_subtractable(self) -> bool:
        """Whether the aggregate can be corrected by removing one contribution.

        This determines Flink support and is not a detail — it is a hard
        constraint of Flink SQL. A Flink ``OVER`` window's frame **must** end at
        ``CURRENT ROW``; there is no way to express "up to but excluding the
        current row". So the streaming compiler computes the inclusive aggregate
        and subtracts the current event's own contribution.

        That works for COUNT, SUM, MEAN and STDDEV (via running sums of x and
        x²). It cannot work for MAX, MIN or DISTINCT_COUNT: removing a value
        from a maximum requires knowing the second-largest, which the aggregate
        does not carry.

        Rather than emitting subtly-wrong Flink SQL for those, the compiler
        refuses. See :mod:`kg_features.compilers.flink`.
        """
        return self in (
            Aggregation.COUNT,
            Aggregation.SUM,
            Aggregation.MEAN,
            Aggregation.STDDEV,
        )


#: Comparison operators a filter may use. A closed set, compiled to SQL and to
#: Python by the same table — so a filter cannot mean one thing offline and
#: another online, and no user-supplied string ever reaches a SQL string.
FilterOp = Literal["==", "!=", ">", ">=", "<", "<=", "in", "not in"]

_SQL_OPS: dict[str, str] = {
    "==": "=",
    "!=": "<>",
    ">": ">",
    ">=": ">=",
    "<": "<",
    "<=": "<=",
}


@dataclass(frozen=True, slots=True)
class Filter:
    """A single predicate restricting which historical events count.

    Deliberately not an arbitrary expression. A closed algebra can be compiled
    faithfully to three targets and proven equivalent; free-form SQL cannot, and
    would put a string from a feature definition directly into a query.
    """

    column: str
    op: FilterOp
    value: Any

    def to_sql(self, alias: str = "h") -> str:
        column = f"{alias}.{self.column}"
        if self.op in ("in", "not in"):
            values = ", ".join(_sql_literal(v) for v in self.value)
            return f"{column} {'IN' if self.op == 'in' else 'NOT IN'} ({values})"
        return f"{column} {_SQL_OPS[self.op]} {_sql_literal(self.value)}"

    def evaluate(self, row: dict[str, Any]) -> bool:
        """The same predicate in Python, for the online executor."""
        actual = row.get(self.column)
        if actual is None:
            # NULL comparisons are false in SQL; matching that here is what
            # keeps the online executor equal to the backfill on rows with
            # missing values.
            return False
        if self.op == "in":
            return actual in self.value
        if self.op == "not in":
            return actual not in self.value
        if self.op == "==":
            return bool(actual == self.value)
        if self.op == "!=":
            return bool(actual != self.value)
        if self.op == ">":
            return bool(actual > self.value)
        if self.op == ">=":
            return bool(actual >= self.value)
        if self.op == "<":
            return bool(actual < self.value)
        return bool(actual <= self.value)


def _sql_literal(value: Any) -> str:
    """Render a Python value as a SQL literal.

    Strings are single-quoted with internal quotes doubled. Filter values come
    from feature definitions in the repository rather than from user input, but
    a compiler that builds SQL by concatenation should escape regardless — the
    day it stops being true is not the day you want to discover it.
    """
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One feature. See the module docstring for the semantics."""

    name: str
    entity: Entity
    aggregation: Aggregation
    #: Window as a compact spec (``"7d"``, ``"30m"``). ``None`` means unbounded:
    #: the entity's whole history strictly before the current event.
    window: str | None = None
    source_column: str | None = None
    filters: tuple[Filter, ...] = ()
    #: Value when no historical event matches. Chosen per feature: 0 for counts
    #: and sums, but for TIME_SINCE_LAST a large sentinel, because "never seen
    #: before" is behaviourally closer to "very long ago" than to "just now".
    default: float = 0.0
    description: str = ""

    def __post_init__(self) -> None:
        if self.aggregation.needs_source_column and not self.source_column:
            raise ValueError(f"{self.name}: {self.aggregation} needs a source_column")
        if not self.aggregation.needs_source_column and self.source_column:
            raise ValueError(f"{self.name}: {self.aggregation} takes no source_column")
        if self.window is not None:
            parse_window(self.window)  # validates, raises with a useful message
        if not self.name.isidentifier():
            raise ValueError(f"{self.name!r} is not a valid identifier")

    @property
    def window_millis(self) -> int | None:
        return parse_window(self.window) if self.window else None

    @property
    def group_key(self) -> tuple[str, str | None]:
        """Features sharing this key can be computed from one scan.

        The compilers group by ``(entity, window)`` and emit a single join per
        group rather than one per feature. On the full profile that is the
        difference between four joins and forty.
        """
        return (self.entity.value, self.window)

    def filters_sql(self, alias: str = "h") -> str:
        return " AND ".join(f.to_sql(alias) for f in self.filters)

    def matches(self, row: dict[str, Any]) -> bool:
        return all(f.evaluate(row) for f in self.filters)

    def fingerprint(self) -> str:
        """Stable hash of the definition, for provenance stamping."""
        return content_hash(
            {
                "name": self.name,
                "entity": self.entity.value,
                "aggregation": self.aggregation.value,
                "window": self.window,
                "source_column": self.source_column,
                "filters": [(f.column, f.op, f.value) for f in self.filters],
                "default": self.default,
            },
            prefix="kontograph-feature",
        )

    def describe(self) -> str:
        window = format_window(self.window_millis) if self.window_millis else "all history"
        target = f"({self.source_column})" if self.source_column else "()"
        where = f" where {self.filters_sql('')}" if self.filters else ""
        return f"{self.aggregation.value}{target} over {window} per {self.entity.value}{where}"


@dataclass(frozen=True, slots=True)
class FeatureSet:
    """A named, ordered collection of features.

    Ordered because the compiled output is a fixed-width vector and the model
    consumes it positionally. A reordering that silently changed the column
    order would swap two features under a trained model — which fails no test
    and degrades quietly.
    """

    name: str
    features: tuple[FeatureSpec, ...]
    version: int = 1
    description: str = ""

    def __post_init__(self) -> None:
        names = [f.name for f in self.features]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"duplicate feature names in {self.name!r}: {sorted(duplicates)}")

    def __len__(self) -> int:
        return len(self.features)

    def __iter__(self) -> Iterator[FeatureSpec]:
        return iter(self.features)

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.features)

    @property
    def max_window_millis(self) -> int:
        """Longest window in the set.

        Two things depend on this number. The online executor uses it to decide
        how much per-entity state to retain, and the evaluation protocol uses it
        as the **purge width** around a chronological split boundary: a training
        event whose longest window still overlaps the validation period has to
        be dropped, or the split leaks.
        """
        windows = [f.window_millis for f in self.features if f.window_millis]
        return max(windows) if windows else 0

    def grouped(self) -> dict[tuple[str, str | None], list[FeatureSpec]]:
        """Features bucketed by ``(entity, window)`` for single-scan compilation."""
        groups: dict[tuple[str, str | None], list[FeatureSpec]] = {}
        for feature in self.features:
            groups.setdefault(feature.group_key, []).append(feature)
        return groups

    def fingerprint(self) -> str:
        """Hash of the whole set. Stamped onto models, decisions and artifacts.

        A score is only comparable to another score computed under the same
        feature-set fingerprint. The Alert contract carries this value for
        exactly that reason.
        """
        return content_hash(
            {
                "name": self.name,
                "version": self.version,
                "features": [f.fingerprint() for f in self.features],
            },
            prefix="kontograph-feature-set",
        )

    def flink_compatible(self) -> tuple[FeatureSpec, ...]:
        """The subset Flink SQL can express exactly.

        See :attr:`Aggregation.is_subtractable`. Surfaced as a first-class query
        rather than discovered at compile time, so a feature that cannot stream
        is visible when the set is defined rather than when it is deployed.
        """
        return tuple(f for f in self.features if f.aggregation.is_subtractable)

    def summary(self) -> list[dict[str, Any]]:
        return [
            {
                "name": f.name,
                "entity": f.entity.value,
                "aggregation": f.aggregation.value,
                "window": f.window or "unbounded",
                "source": f.source_column,
                "streams": f.aggregation.is_subtractable,
                "description": f.description or f.describe(),
            }
            for f in self.features
        ]


#: Columns the compilers rely on. Named here so a schema change that removes one
#: fails loudly at import rather than producing an empty feature.
EVENT_TIME_COLUMN = "acceptance_ts"
ENTITY_COLUMNS = ("debtor_account_id", "creditor_account_id")
REQUIRED_COLUMNS: tuple[str, ...] = (EVENT_TIME_COLUMN, "tx_id", *ENTITY_COLUMNS)


class UnsupportedOnBackendError(NotImplementedError):
    """Raised when a feature cannot be compiled exactly for a target.

    Deliberately an error rather than a silent approximation. A backend that
    quietly computes something *close* to the definition is the training-serving
    skew this whole layer exists to eliminate.
    """


def validate_against_schema(feature_set: FeatureSet, columns: set[str]) -> list[str]:
    """Return the problems that would make *feature_set* uncomputable on a table."""
    problems: list[str] = []
    for column in REQUIRED_COLUMNS:
        if column not in columns:
            problems.append(f"missing required column {column!r}")
    for feature in feature_set:
        for column in feature.entity.key_columns:
            if column not in columns:
                problems.append(f"{feature.name}: entity column {column!r} not in the table")
        if feature.source_column and feature.source_column not in columns:
            problems.append(
                f"{feature.name}: source column {feature.source_column!r} not in the table"
            )
        for filter_ in feature.filters:
            if filter_.column not in columns:
                problems.append(
                    f"{feature.name}: filter column {filter_.column!r} not in the table"
                )
    return problems
