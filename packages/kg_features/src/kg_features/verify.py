"""Cross-backend equivalence and the point-in-time guarantee.

Design principle P3 claims offline and online are the same computation, and design principle P2
claims no feature can see the future. Both are claims about behaviour, so both
are checked here by *running* the backends rather than by reasoning about them.

Two different properties, checked two different ways:

**Equivalence** — :func:`compare_backends` runs the same event stream through
the DuckDB backfill and the incremental online executor and diffs every value.
A disagreement is training-serving skew, caught as a number rather than as a
production surprise months later.

**No lookahead** — :func:`find_lookahead` is the stronger and more interesting
test. It does not inspect the SQL for a suspicious ``<=``; it *perturbs the
future* and checks that the past does not move. For a stream of n events it
recomputes the features of event ``i`` against a stream whose later events have
been altered, and asserts nothing changed. A feature that can see forward is
caught by construction, whatever mechanism produced it.

The second test is the one that would have caught every leakage bug I have seen
in production ML, because it makes no assumption about *how* the leak happens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from kg_features.compilers.duckdb_sql import execute as duckdb_execute
from kg_features.compilers.online import OnlineFeatureExecutor
from kg_features.ir import EVENT_TIME_COLUMN, FeatureSet

#: Values closer than this are treated as equal. Both backends do float
#: arithmetic in a different order (SQL aggregates vs. a Python sum), so exact
#: bit equality is not a reasonable expectation for MEAN or STDDEV.
TOLERANCE = 1e-6


@dataclass(slots=True)
class Divergence:
    """One disagreement between two backends."""

    tx_id: str
    feature: str
    left: float
    right: float

    @property
    def delta(self) -> float:
        return abs(self.left - self.right)

    def __str__(self) -> str:
        return f"{self.feature} @ {self.tx_id}: {self.left!r} vs {self.right!r}"


@dataclass(slots=True)
class ComparisonReport:
    """The outcome of diffing two backends over one event stream."""

    events: int
    features: int
    compared: int = 0
    divergences: list[Divergence] = field(default_factory=list)

    @property
    def agrees(self) -> bool:
        return not self.divergences

    def summary(self) -> dict[str, Any]:
        by_feature: dict[str, int] = {}
        for divergence in self.divergences:
            by_feature[divergence.feature] = by_feature.get(divergence.feature, 0) + 1
        return {
            "events": self.events,
            "features": self.features,
            "values_compared": self.compared,
            "divergences": len(self.divergences),
            "agrees": self.agrees,
            "by_feature": by_feature,
            "worst": max((d.delta for d in self.divergences), default=0.0),
        }

    def raise_if_diverged(self, left: str = "duckdb", right: str = "online") -> None:
        if self.agrees:
            return
        shown = "\n".join(f"    {d}" for d in self.divergences[:10])
        more = (
            f"\n    ... and {len(self.divergences) - 10} more" if len(self.divergences) > 10 else ""
        )
        raise AssertionError(
            f"{left} and {right} disagree on {len(self.divergences)} of "
            f"{self.compared} values. This is training-serving skew:\n{shown}{more}"
        )


def _equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    left, right = float(left), float(right)
    if math.isnan(left) or math.isnan(right):
        return math.isnan(left) and math.isnan(right)
    return abs(left - right) <= TOLERANCE * max(1.0, abs(left), abs(right))


def run_online(
    feature_set: FeatureSet, events: list[dict[str, Any]]
) -> dict[str, dict[str, float]]:
    """Feature vectors from the incremental executor, keyed by ``tx_id``."""
    executor = OnlineFeatureExecutor(feature_set)
    ordered = sorted(events, key=lambda e: (int(e[EVENT_TIME_COLUMN]), str(e["tx_id"])))
    return {
        row["tx_id"]: {name: row[name] for name in feature_set.column_names}
        for row in executor.run(ordered)
    }


def run_duckdb(
    feature_set: FeatureSet, events: list[dict[str, Any]]
) -> dict[str, dict[str, float]]:
    """Feature vectors from the backfill compiler, keyed by ``tx_id``."""
    import pyarrow as pa

    table = duckdb_execute(feature_set, pa.Table.from_pylist(events))
    return {
        row["tx_id"]: {name: row[name] for name in feature_set.column_names}
        for row in table.to_pylist()
    }


def compare_backends(feature_set: FeatureSet, events: list[dict[str, Any]]) -> ComparisonReport:
    """Diff the DuckDB backfill against the online executor, value by value."""
    report = ComparisonReport(events=len(events), features=len(feature_set))
    if not events:
        return report

    offline = run_duckdb(feature_set, events)
    online = run_online(feature_set, events)

    missing = set(offline) ^ set(online)
    if missing:
        raise AssertionError(
            f"backends produced different transaction sets; symmetric difference: "
            f"{sorted(missing)[:5]}"
        )

    for tx_id, offline_row in offline.items():
        online_row = online[tx_id]
        for name in feature_set.column_names:
            report.compared += 1
            if not _equal(offline_row[name], online_row[name]):
                report.divergences.append(
                    Divergence(tx_id, name, offline_row[name], online_row[name])
                )
    return report


# ------------------------------------------------------- lookahead ----------


@dataclass(slots=True)
class LookaheadFinding:
    """Evidence that a feature depends on an event at or after its own time."""

    tx_id: str
    feature: str
    baseline: float
    #: None when the transaction vanished from the output entirely, which is
    #: itself a violation worth reporting rather than a crash.
    after_future_changed: float | None
    event_time: int

    def __str__(self) -> str:
        return (
            f"{self.feature} @ {self.tx_id} (t={self.event_time}) changed from "
            f"{self.baseline!r} to {self.after_future_changed!r} when a LATER "
            "event was modified"
        )


def find_lookahead(
    feature_set: FeatureSet,
    events: list[dict[str, Any]],
    *,
    backend: str = "duckdb",
    amount_multiplier: float = 1000.0,
) -> list[LookaheadFinding]:
    """Detect any dependence on the future, by perturbing the future.

    Method: compute features for the whole stream, then for each cut point
    ``i`` build a variant in which every event **at or after** ``events[i]``'s
    timestamp has been altered — amount inflated, entity keys rewritten, VoP
    result flipped. Recompute, and compare the features of the events strictly
    *before* the cut. Nothing may have moved.

    Events sharing a timestamp with the cut are perturbed too, deliberately: the
    window is right-open, so a simultaneous event must not contribute either.
    That boundary is where an off-by-one lives.

    Returns every violation found; empty means the property held on this stream.
    """
    if len(events) < 2:
        return []

    ordered = sorted(events, key=lambda e: (int(e[EVENT_TIME_COLUMN]), str(e["tx_id"])))
    compute = run_duckdb if backend == "duckdb" else run_online
    baseline = compute(feature_set, ordered)
    findings: list[LookaheadFinding] = []

    # A handful of cut points rather than every index: the property is uniform
    # in i, and n recomputations of an n-event stream is quadratic.
    cut_points = sorted({len(ordered) // 4, len(ordered) // 2, 3 * len(ordered) // 4, 1})
    for cut in cut_points:
        if not 0 < cut < len(ordered):
            continue
        cut_time = int(ordered[cut][EVENT_TIME_COLUMN])

        perturbed: list[dict[str, Any]] = []
        for event in ordered:
            if int(event[EVENT_TIME_COLUMN]) >= cut_time:
                altered = dict(event)
                altered["amount_cents"] = int(
                    abs(int(event.get("amount_cents", 1))) * amount_multiplier + 7
                )
                altered["debtor_account_id"] = f"FUTURE-{event['debtor_account_id']}"
                altered["creditor_account_id"] = f"FUTURE-{event['creditor_account_id']}"
                altered["vop_result"] = (
                    "NO_MATCH" if event.get("vop_result") == "MATCH" else "MATCH"
                )
                altered["is_new_beneficiary"] = not event.get("is_new_beneficiary", False)
                perturbed.append(altered)
            else:
                perturbed.append(event)

        after = compute(feature_set, perturbed)

        for event in ordered[:cut]:
            tx_id = event["tx_id"]
            if int(event[EVENT_TIME_COLUMN]) >= cut_time:
                continue  # this event was itself perturbed
            for name in feature_set.column_names:
                before_value = baseline[tx_id][name]
                after_value = after.get(tx_id, {}).get(name)
                if not _equal(before_value, after_value):
                    findings.append(
                        LookaheadFinding(
                            tx_id=tx_id,
                            feature=name,
                            baseline=before_value,
                            after_future_changed=after_value,
                            event_time=int(event[EVENT_TIME_COLUMN]),
                        )
                    )
    return findings


def assert_no_lookahead(
    feature_set: FeatureSet, events: list[dict[str, Any]], *, backend: str = "duckdb"
) -> None:
    """Raise if any feature depends on the future."""
    findings = find_lookahead(feature_set, events, backend=backend)
    if findings:
        shown = "\n".join(f"    {f}" for f in findings[:10])
        raise AssertionError(
            f"LEAKAGE: {len(findings)} feature value(s) on the {backend} backend "
            f"changed when a later event was modified:\n{shown}"
        )
