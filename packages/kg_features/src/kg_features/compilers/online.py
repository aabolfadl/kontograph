"""The serving compiler: an incremental executor over per-entity state.

This is the code that runs inside the 200 ms budget, so it gets the most
scrutiny of the three backends — and it is also the one where point-in-time
correctness is easiest to guarantee, because of the order of operations:

    1. compute the feature vector from the state as it currently stands
    2. *then* fold the current event into the state

Step 1 physically cannot see the current event, because step 2 has not happened
yet. The strict ``h.ts < t`` bound that the SQL compilers achieve with an
explicit inequality is here a consequence of causality. That is worth stating
plainly: the online path is correct by construction, and the offline paths have
to work to match it.

**State is a bounded, time-ordered history per entity key.** Aggregations like
MAX and DISTINCT_COUNT over a sliding window are not incrementally maintainable
from a summary — removing a value from a maximum needs the second-largest — so
the executor keeps the events themselves and evicts what has aged out. Memory is
bounded by ``max_window`` rather than by total volume, which is what makes it
viable at serving time.
"""

from __future__ import annotations

import bisect
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol

from kg_features.ir import (
    EVENT_TIME_COLUMN,
    Aggregation,
    Entity,
    FeatureSet,
    FeatureSpec,
    UnsupportedOnBackendError,
)

#: Separates the composite key parts (a PAIR has two).
SEP_VALUE = chr(0x1F)  # ASCII unit separator
#: Separates the entity name from the key, so DEBTOR "B" and CREDITOR "B"
#: cannot collide. Distinct from SEP_VALUE so no account id can forge one.
SEP_ENTITY = chr(0x1E)  # ASCII record separator


def entity_key(entity: Entity, row: dict[str, Any]) -> str:
    """The state key for an event under a given entity.

    The entity name is part of the key, and that is not cosmetic. Without it,
    account "B" as a DEBTOR and account "B" as a CREDITOR collapse to the same
    string, and the store hands an account's *inbound* history to a feature
    asking about its *outbound* history. On a two-event stream

        tx0:  A -> B
        tx1:  B -> A

    every debtor-keyed feature for tx1 returned 1 instead of 0, because tx0 had
    been filed under "B" as a creditor.

    Found by the cross-backend property test. That is the whole argument for
    diffing the backends on random streams rather than trusting that two
    implementations written from the same docstring agree.
    """
    values = SEP_VALUE.join(str(row.get(column)) for column in entity.key_columns)
    return f"{entity.value}{SEP_ENTITY}{values}"


@dataclass(slots=True)
class Accumulator:
    """Running summary for one UNBOUNDED feature on one entity.

    Unbounded features cannot be served from the retained window, because the
    window is pruned and their history is not. Keeping every event forever would
    make serving memory grow without bound, so each unbounded feature maintains
    the constant-size summary its aggregation needs:

        COUNT / SUM / MEAN / STDDEV   n, sum(x), sum(x^2)
        MAX / MIN                     the running extreme -- monotone, so an
                                      unbounded window never needs to remove a
                                      value (a sliding one would)
        TIME_SINCE_LAST               the last matching timestamp

    DISTINCT_COUNT is the exception: an exact unbounded distinct count needs a
    set that grows with cardinality, so it is refused rather than silently
    unbounded. See OnlineFeatureExecutor.__init__.

    # The pending bucket, and why it exists

    The summary must represent events **strictly before** the event being
    scored. A naive accumulator cannot: by the time the second of two payments
    sharing a millisecond is scored, the first has already been folded in, and
    the feature reports it -- exactly the right-closed window the whole layer
    forbids.

    So updates land in a *pending* bucket tagged with their timestamp, and are
    committed into the main summary only once event time advances past it.
    ``snapshot(now)`` then includes pending only when ``pending_ts < now``.

    Events arrive in non-decreasing event-time order (enforced by
    ``OnlineFeatureExecutor.run``), so at most one pending timestamp is ever
    open, and the bucket stays constant-size.

    Found by the cross-backend property test: online reported a
    ``time_since_last`` of 0 where the backfill reported 7200001, because two
    events shared a timestamp.
    """

    # Committed: every event strictly earlier than ``pending_ts``.
    n: int = 0
    total: float = 0.0
    total_squared: float = 0.0
    maximum: float | None = None
    minimum: float | None = None
    last_ts: int | None = None

    # Pending: events at exactly ``pending_ts``, not yet visible to a scorer at
    # that same instant.
    pending_ts: int | None = None
    pending_n: int = 0
    pending_total: float = 0.0
    pending_total_squared: float = 0.0
    pending_maximum: float | None = None
    pending_minimum: float | None = None

    def update(self, value: float | None, timestamp: int) -> None:
        if self.pending_ts is not None and timestamp > self.pending_ts:
            self._commit()
        self.pending_ts = timestamp
        self.pending_n += 1
        if value is None:
            return
        self.pending_total += value
        self.pending_total_squared += value * value
        self.pending_maximum = (
            value if self.pending_maximum is None else max(self.pending_maximum, value)
        )
        self.pending_minimum = (
            value if self.pending_minimum is None else min(self.pending_minimum, value)
        )

    def _commit(self) -> None:
        """Fold the pending bucket into the committed summary."""
        self.n += self.pending_n
        self.total += self.pending_total
        self.total_squared += self.pending_total_squared
        self.maximum = _combine(self.maximum, self.pending_maximum, max)
        self.minimum = _combine(self.minimum, self.pending_minimum, min)
        if self.pending_n:
            self.last_ts = self.pending_ts
        self.pending_n = 0
        self.pending_total = 0.0
        self.pending_total_squared = 0.0
        self.pending_maximum = None
        self.pending_minimum = None

    def snapshot(self, now: int) -> AccumulatorSnapshot:
        """The summary over events strictly before *now*."""
        include_pending = self.pending_ts is not None and self.pending_ts < now
        if not include_pending:
            return AccumulatorSnapshot(
                self.n, self.total, self.total_squared, self.maximum, self.minimum, self.last_ts
            )
        return AccumulatorSnapshot(
            self.n + self.pending_n,
            self.total + self.pending_total,
            self.total_squared + self.pending_total_squared,
            _combine(self.maximum, self.pending_maximum, max),
            _combine(self.minimum, self.pending_minimum, min),
            self.pending_ts if self.pending_n else self.last_ts,
        )


@dataclass(frozen=True, slots=True)
class AccumulatorSnapshot:
    """An accumulator as it stood strictly before some instant."""

    n: int
    total: float
    total_squared: float
    maximum: float | None
    minimum: float | None
    last_ts: int | None


def _combine(left: float | None, right: float | None, chooser: Any) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return chooser(left, right)


@dataclass(slots=True)
class EntityHistory:
    """Time-ordered events for one entity key, pruned to the longest window.

    ``accumulators`` holds the unbounded features' running summaries, keyed by
    feature name. They are deliberately never pruned -- that is their purpose.
    """

    timestamps: list[int] = field(default_factory=list)
    rows: deque[dict[str, Any]] = field(default_factory=deque)
    accumulators: dict[str, Accumulator] = field(default_factory=dict)

    def accumulator(self, name: str) -> Accumulator:
        acc = self.accumulators.get(name)
        if acc is None:
            acc = Accumulator()
            self.accumulators[name] = acc
        return acc

    def append(self, timestamp: int, row: dict[str, Any]) -> None:
        self.timestamps.append(timestamp)
        self.rows.append(row)

    def evict_before(self, cutoff: int) -> None:
        """Drop events older than *cutoff*, keeping memory bounded."""
        dropped = 0
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.pop(0)
            self.rows.popleft()
            dropped += 1

    def window(self, low: int, high: int) -> list[dict[str, Any]]:
        """Events with ``low <= ts < high``.

        Binary search on the timestamp list, which stays sorted because events
        are appended in event-time order. ``bisect_left(high)`` gives the strict
        upper bound directly, which is where the point-in-time guarantee lives:
        an event at exactly ``high`` — the event being scored — is excluded.
        """
        start = bisect.bisect_left(self.timestamps, low)
        end = bisect.bisect_left(self.timestamps, high)
        return list(self.rows)[start:end]

    def __len__(self) -> int:
        return len(self.timestamps)


class StateStore(Protocol):
    """Where per-entity history lives."""

    def get(self, key: str) -> EntityHistory: ...
    def put(self, key: str, history: EntityHistory) -> None: ...
    def clear(self) -> None: ...
    def size(self) -> int: ...


class InMemoryStateStore:
    """Process-local state. Used by the backfill-equivalence tests and by tests."""

    def __init__(self) -> None:
        self._data: dict[str, EntityHistory] = {}

    def get(self, key: str) -> EntityHistory:
        """Return the history for *key*, or an empty one.

        Deliberately does NOT insert on a miss. `compute` reads state for every
        entity it touches, so a get-that-inserts made scoring allocate an empty
        history per unseen account -- state growing on a read path that is
        supposed to be side-effect free. `observe` is the only writer.
        """
        return self._data.get(key) or EntityHistory()

    def put(self, key: str, history: EntityHistory) -> None:
        self._data[key] = history

    def clear(self) -> None:
        self._data.clear()

    def size(self) -> int:
        return len(self._data)


def _aggregate(feature: FeatureSpec, rows: list[dict[str, Any]], now: int) -> float:
    """Evaluate one aggregation over the matching historical rows.

    Mirrors :mod:`kg_features.compilers.duckdb_sql` exactly, including the
    default when nothing matches. Any divergence here is training-serving skew,
    which is why the property suite diffs the two on random streams rather than
    trusting that these two functions were written from the same docstring.
    """
    matching = [row for row in rows if feature.matches(row)] if feature.filters else rows

    if feature.aggregation is Aggregation.COUNT:
        return float(len(matching)) if matching else feature.default

    if feature.aggregation is Aggregation.TIME_SINCE_LAST:
        if not matching:
            return feature.default
        # Rows are in event-time order, so the last one is the most recent.
        return float(now - int(matching[-1][EVENT_TIME_COLUMN]))

    column = feature.source_column
    assert column is not None  # guaranteed by FeatureSpec.__post_init__

    if feature.aggregation is Aggregation.DISTINCT_COUNT:
        # SQL's COUNT(DISTINCT x) ignores NULLs; match that.
        values = {row.get(column) for row in matching}
        values.discard(None)
        return float(len(values)) if values else feature.default

    numbers = [float(row[column]) for row in matching if row.get(column) is not None]
    if not numbers:
        return feature.default

    if feature.aggregation is Aggregation.SUM:
        return float(sum(numbers))
    if feature.aggregation is Aggregation.MEAN:
        return float(sum(numbers) / len(numbers))
    if feature.aggregation is Aggregation.MAX:
        return float(max(numbers))
    if feature.aggregation is Aggregation.MIN:
        return float(min(numbers))
    if feature.aggregation is Aggregation.STDDEV:
        # Population standard deviation, matching STDDEV_POP.
        average = sum(numbers) / len(numbers)
        variance = sum((value - average) ** 2 for value in numbers) / len(numbers)
        return float(math.sqrt(variance))

    raise AssertionError(f"unhandled aggregation {feature.aggregation}")


def _from_accumulator(feature: FeatureSpec, acc: Accumulator, now: int) -> float:
    """Evaluate an unbounded feature from its running summary.

    Reads a snapshot taken strictly before *now*, so events sharing this
    instant are excluded exactly as they are in the SQL compilers. Must agree
    with :func:`_aggregate` and with the DuckDB compiler; the cross-backend
    property test enforces that rather than hoping for it.
    """
    snap = acc.snapshot(now)
    if snap.n == 0:
        return feature.default

    if feature.aggregation is Aggregation.COUNT:
        return float(snap.n)
    if feature.aggregation is Aggregation.TIME_SINCE_LAST:
        return float(now - snap.last_ts) if snap.last_ts is not None else feature.default
    if feature.aggregation is Aggregation.SUM:
        return snap.total
    if feature.aggregation is Aggregation.MEAN:
        return snap.total / snap.n
    if feature.aggregation is Aggregation.MAX:
        return snap.maximum if snap.maximum is not None else feature.default
    if feature.aggregation is Aggregation.MIN:
        return snap.minimum if snap.minimum is not None else feature.default
    if feature.aggregation is Aggregation.STDDEV:
        # Population variance from running sums. max(..., 0) guards the
        # catastrophic-cancellation case where floating point drives a
        # mathematically non-negative variance just below zero.
        mean_value = snap.total / snap.n
        variance = max(snap.total_squared / snap.n - mean_value * mean_value, 0.0)
        return math.sqrt(variance)

    raise AssertionError(f"unhandled unbounded aggregation {feature.aggregation}")


class OnlineFeatureExecutor:
    """Computes the feature vector for one event, then absorbs it.

    Used identically by the serving gateway and by the default local stream
    processor, so the code path that produces a production score is the same one
    the equivalence tests exercise.
    """

    def __init__(self, feature_set: FeatureSet, store: StateStore | None = None) -> None:
        self.feature_set = feature_set
        self.store = store if store is not None else InMemoryStateStore()
        # Only the entities the set actually uses; no point maintaining PAIR
        # history for a set that never keys by it.
        self._entities = sorted({feature.entity for feature in feature_set}, key=str)
        self._max_window = feature_set.max_window_millis

        # Windowed features are served by scanning retained rows; unbounded ones
        # by a running accumulator. Splitting them is what fixes the bug where
        # eviction -- sized by the longest *window* -- silently truncated the
        # history an unbounded feature depends on. The cross-backend property
        # test caught it as a disagreement on c_unbounded and the
        # time_since_last features, once a gap exceeded the longest window.
        self._windowed: dict[Entity, list[FeatureSpec]] = {}
        self._unbounded: dict[Entity, list[FeatureSpec]] = {}
        for feature in feature_set:
            target = self._windowed if feature.window_millis is not None else self._unbounded
            target.setdefault(feature.entity, []).append(feature)

        unbounded_distinct = [
            f.name
            for features in self._unbounded.values()
            for f in features
            if f.aggregation is Aggregation.DISTINCT_COUNT
        ]
        if unbounded_distinct:
            raise UnsupportedOnBackendError(
                f"unbounded DISTINCT_COUNT cannot be served online: {unbounded_distinct}. "
                "An exact distinct count over all history needs a set that grows with "
                "cardinality, which is unbounded serving memory. Give the feature a "
                "window, or add an approximate sketch as a separate aggregation."
            )

    def compute(self, event: dict[str, Any]) -> dict[str, float]:
        """Feature vector for *event*, from history strictly before it.

        Does **not** mutate state. Separated from :meth:`observe` so the serving
        path can score without committing, and so a re-score of the same event
        (which the explanation-stability metric does) returns the same answer.
        """
        now = int(event[EVENT_TIME_COLUMN])
        vector: dict[str, float] = {}

        for entity in self._entities:
            history = self.store.get(entity_key(entity, event))

            for feature in self._windowed.get(entity, ()):
                window = feature.window_millis
                assert window is not None
                rows = history.window(now - window, now)  # strict upper bound
                vector[feature.name] = _aggregate(feature, rows, now)

            for feature in self._unbounded.get(entity, ()):
                acc = history.accumulators.get(feature.name)
                vector[feature.name] = (
                    _from_accumulator(feature, acc, now) if acc else feature.default
                )

        # Emit in declared order: the model consumes this vector positionally.
        return {name: vector[name] for name in self.feature_set.column_names}

    def observe(self, event: dict[str, Any]) -> None:
        """Fold *event* into the state. Call after :meth:`compute`."""
        now = int(event[EVENT_TIME_COLUMN])
        cutoff = now - self._max_window if self._max_window else None

        for entity in self._entities:
            key = entity_key(entity, event)
            history = self.store.get(key)

            # Accumulators are updated first and never pruned: they ARE the
            # unbounded history, held in constant space.
            for feature in self._unbounded.get(entity, ()):
                if feature.filters and not feature.matches(event):
                    continue
                raw = event.get(feature.source_column) if feature.source_column else None
                history.accumulator(feature.name).update(
                    float(raw) if raw is not None else None, now
                )

            # Rows are retained only if some windowed feature needs them.
            if self._windowed.get(entity):
                history.append(now, event)
                if cutoff is not None:
                    history.evict_before(cutoff)
            self.store.put(key, history)

    def process(self, event: dict[str, Any]) -> dict[str, float]:
        """Compute then observe — the streaming path, in the correct order.

        The ordering here *is* the point-in-time guarantee. Swapping these two
        lines would make every feature include the event it is describing, and
        every offline metric would improve.
        """
        vector = self.compute(event)
        self.observe(event)
        return vector

    def run(self, events: Any) -> list[dict[str, Any]]:
        """Process a stream in event-time order, returning keyed feature rows.

        The caller is responsible for the ordering. An out-of-order stream would
        silently produce features from a partly-future state, so this is checked
        rather than assumed.
        """
        results: list[dict[str, Any]] = []
        previous = -(2**62)
        for event in events:
            timestamp = int(event[EVENT_TIME_COLUMN])
            if timestamp < previous:
                raise ValueError(
                    f"events must arrive in non-decreasing event-time order; "
                    f"saw {timestamp} after {previous}. Out-of-order arrival would "
                    "compute features against a state containing later events."
                )
            previous = timestamp
            vector = self.process(event)
            results.append(
                {
                    "tx_id": event["tx_id"],
                    EVENT_TIME_COLUMN: timestamp,
                    "debtor_account_id": event.get("debtor_account_id"),
                    "creditor_account_id": event.get("creditor_account_id"),
                    **vector,
                }
            )
        return results

    def reset(self) -> None:
        self.store.clear()

    @property
    def tracked_entities(self) -> int:
        return self.store.size()
