"""The declarative surface.

Thin constructors over :mod:`kg_features.ir`. The point of the DSL is that a
feature definition reads like the question it answers, so a reviewer can check
the *intent* in a pull request without reconstructing it from SQL:

    count("outbound_1h", Entity.DEBTOR, "1h",
          description="transfers this account has sent in the last hour")

Everything below returns a plain :class:`FeatureSpec`. There is no runtime
machinery here — the DSL is a convenience layer, and the IR is the contract.
"""

from __future__ import annotations

from typing import Any

from kg_features.ir import Aggregation, Entity, FeatureSpec, Filter, FilterOp


def _spec(
    name: str,
    entity: Entity,
    aggregation: Aggregation,
    window: str | None,
    source_column: str | None,
    where: tuple[Filter, ...],
    default: float,
    description: str,
) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        entity=entity,
        aggregation=aggregation,
        window=window,
        source_column=source_column,
        filters=where,
        default=default,
        description=description,
    )


def when(column: str, op: FilterOp, value: Any) -> Filter:
    """A single predicate: ``when("vop_result", "!=", "MATCH")``."""
    return Filter(column=column, op=op, value=value)


def count(
    name: str,
    entity: Entity,
    window: str | None = None,
    *,
    where: tuple[Filter, ...] = (),
    description: str = "",
) -> FeatureSpec:
    """How many prior events match."""
    return _spec(name, entity, Aggregation.COUNT, window, None, where, 0.0, description)


def total(
    name: str,
    entity: Entity,
    column: str,
    window: str | None = None,
    *,
    where: tuple[Filter, ...] = (),
    description: str = "",
) -> FeatureSpec:
    """Sum of a column over prior events."""
    return _spec(name, entity, Aggregation.SUM, window, column, where, 0.0, description)


def mean(
    name: str,
    entity: Entity,
    column: str,
    window: str | None = None,
    *,
    where: tuple[Filter, ...] = (),
    default: float = 0.0,
    description: str = "",
) -> FeatureSpec:
    return _spec(name, entity, Aggregation.MEAN, window, column, where, default, description)


def maximum(
    name: str,
    entity: Entity,
    column: str,
    window: str | None = None,
    *,
    where: tuple[Filter, ...] = (),
    description: str = "",
) -> FeatureSpec:
    return _spec(name, entity, Aggregation.MAX, window, column, where, 0.0, description)


def minimum(
    name: str,
    entity: Entity,
    column: str,
    window: str | None = None,
    *,
    where: tuple[Filter, ...] = (),
    description: str = "",
) -> FeatureSpec:
    return _spec(name, entity, Aggregation.MIN, window, column, where, 0.0, description)


def stddev(
    name: str,
    entity: Entity,
    column: str,
    window: str | None = None,
    *,
    where: tuple[Filter, ...] = (),
    description: str = "",
) -> FeatureSpec:
    """Population standard deviation of a column over prior events.

    Population rather than sample: with one prior event the sample form is
    undefined (division by n-1 = 0) and every backend picks a different answer —
    NULL, NaN, or an error. Population is defined for n >= 1 and gives 0, which
    is both correct and identical across the three compilers.
    """
    return _spec(name, entity, Aggregation.STDDEV, window, column, where, 0.0, description)


def distinct(
    name: str,
    entity: Entity,
    column: str,
    window: str | None = None,
    *,
    where: tuple[Filter, ...] = (),
    description: str = "",
) -> FeatureSpec:
    """Number of distinct values of a column over prior events."""
    return _spec(name, entity, Aggregation.DISTINCT_COUNT, window, column, where, 0.0, description)


#: Returned by TIME_SINCE_LAST when the entity has no prior event.
#:
#: 30 days rather than 0 or -1. Zero would read as "this happened just now",
#: which is the opposite of the truth and is exactly backwards for a model that
#: treats small gaps as bursty behaviour. A large finite value keeps the feature
#: monotone and usable by a tree split, where NaN or -1 would not be.
NEVER_SEEN_MILLIS: float = 30 * 24 * 60 * 60 * 1000.0


def time_since_last(
    name: str,
    entity: Entity,
    window: str | None = None,
    *,
    where: tuple[Filter, ...] = (),
    default: float = NEVER_SEEN_MILLIS,
    description: str = "",
) -> FeatureSpec:
    """Milliseconds since this entity's previous matching event.

    The velocity primitive. Standard laundering typologies move in minutes; the
    evasive variants deliberately dwell for hours, which is precisely the
    distinction this feature exposes and a fixed-window count cannot.
    """
    return _spec(
        name, entity, Aggregation.TIME_SINCE_LAST, window, None, where, default, description
    )


__all__ = [
    "NEVER_SEEN_MILLIS",
    "Entity",
    "count",
    "distinct",
    "maximum",
    "mean",
    "minimum",
    "stddev",
    "time_since_last",
    "total",
    "when",
]
