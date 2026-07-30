"""Per-stage latency measurement and the budget table.

Design principle P4: *"Latency is a modelling constraint, not an engineering
afterthought. The SLA belongs in the architecture diagram and in the loss
function's feasible set."*

The SLA is **p99 ≤ 200 ms end to end**, and it lives in configuration
(``KG_SLA_P99_MILLIS``) rather than in prose, so the load test can fail the
build against it.

**Per stage, not just end to end.** A p99 of 180 ms is a pass; a p99 of 180 ms
of which 140 is the counterfactual search is a pass that will stop being one the
moment the graph gets denser. The budget table is what makes the headroom
visible, and it is emitted from the same run that produces the total — measured,
never modelled.

Quantiles come from the raw samples rather than from a pre-averaged number: an
average latency cannot produce a p99, and a system that reports one has already
lost the ability to see its own tail.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np


#: The stages a decision passes through, in order. Named here so the budget
#: table has a fixed shape and a missing stage is visible as a gap.
class UnknownStageError(ValueError):
    """Raised when timing a stage that is not declared in :data:`STAGES`."""


STAGES: tuple[str, ...] = (
    "feature_fetch",
    "graph_assemble",
    "inference",
    "explain",
    "counterfactual",
)


@dataclass(slots=True)
class StageTimer:
    """Accumulates per-stage timings for one decision."""

    timings: dict[str, float] = field(default_factory=dict)
    _started: float = 0.0

    def __post_init__(self) -> None:
        self._started = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time one stage. Nested stages are not supported and not needed.

        An unknown stage name raises rather than being recorded. ``as_dict``
        reports the declared stages, so a typo would otherwise silently drop
        that stage from the budget table while its cost stayed folded into the
        total - the one failure mode a latency investigation cannot see.
        """
        if name not in STAGES:
            raise UnknownStageError(
                f"{name!r} is not a declared stage; expected one of {', '.join(STAGES)}"
            )
        began = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = (time.perf_counter() - began) * 1000.0

    def total(self) -> float:
        """Wall clock for the whole decision, not the sum of the stages.

        Deliberately measured independently: if the total exceeds the sum, time
        is being spent somewhere unaccounted for, and that gap is exactly what a
        latency investigation needs to see.
        """
        return (time.perf_counter() - self._started) * 1000.0

    def as_dict(self) -> dict[str, float]:
        payload = {stage: self.timings.get(stage, 0.0) for stage in STAGES}
        payload["total"] = self.total()
        payload["unaccounted"] = max(0.0, payload["total"] - sum(self.timings.values()))
        return payload


@dataclass(slots=True)
class LatencyBudget:
    """Aggregated per-stage quantiles across many decisions."""

    samples: dict[str, list[float]] = field(default_factory=dict)
    sla_p99_ms: float = 200.0

    def record(self, timings: dict[str, float]) -> None:
        for stage, value in timings.items():
            self.samples.setdefault(stage, []).append(float(value))

    @property
    def count(self) -> int:
        return len(self.samples.get("total", []))

    def quantiles(self, stage: str) -> dict[str, float]:
        values = np.asarray(self.samples.get(stage, []), dtype=np.float64)
        if values.size == 0:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0}
        # Nearest-rank ("higher"), not the numpy default of linear
        # interpolation. Interpolation reports a latency no request actually
        # experienced, and it does so on the optimistic side: with 100 samples
        # where one takes 900 ms, linear interpolation puts p99 at 13.9 ms. For
        # a number that gates an SLA claim, the reported figure should be one a
        # real request saw, and should round against us.
        return {
            "p50": float(np.percentile(values, 50, method="higher")),
            "p95": float(np.percentile(values, 95, method="higher")),
            "p99": float(np.percentile(values, 99, method="higher")),
            "max": float(values.max()),
            "mean": float(values.mean()),
        }

    @property
    def meets_sla(self) -> bool:
        return self.quantiles("total")["p99"] <= self.sla_p99_ms

    def table(self) -> list[dict[str, Any]]:
        """The budget table, per stage, with each stage's share of the p99."""
        total_p99 = self.quantiles("total")["p99"] or 1.0
        rows = []
        for stage in (*STAGES, "unaccounted", "total"):
            if stage not in self.samples:
                continue
            q = self.quantiles(stage)
            rows.append(
                {
                    "stage": stage,
                    "p50_ms": round(q["p50"], 3),
                    "p95_ms": round(q["p95"], 3),
                    "p99_ms": round(q["p99"], 3),
                    "max_ms": round(q["max"], 3),
                    "share_of_p99": round(q["p99"] / total_p99, 4) if stage != "total" else 1.0,
                }
            )
        return rows

    def as_dict(self) -> dict[str, Any]:
        return {
            "decisions": self.count,
            "sla_p99_ms": self.sla_p99_ms,
            "meets_sla": self.meets_sla,
            "headroom_ms": round(self.sla_p99_ms - self.quantiles("total")["p99"], 3),
            "budget": self.table(),
        }

    def render_markdown(self) -> str:
        """The table as it appears in the README. Generated, never typed."""
        lines = [
            f"**SLA: p99 ≤ {self.sla_p99_ms:.0f} ms end to end** — "
            f"measured over {self.count:,} decisions.",
            "",
            "| Stage | p50 | p95 | p99 | share of p99 |",
            "|---|---|---|---|---|",
        ]
        for row in self.table():
            share = "" if row["stage"] == "total" else f"{row['share_of_p99']:.0%}"
            emphasis = "**" if row["stage"] == "total" else ""
            lines.append(
                f"| {emphasis}{row['stage']}{emphasis} | {row['p50_ms']:.2f} ms | "
                f"{row['p95_ms']:.2f} ms | {emphasis}{row['p99_ms']:.2f} ms{emphasis} | {share} |"
            )

        total = self.quantiles("total")
        verdict = (
            f"\n**PASS** — p99 of {total['p99']:.1f} ms leaves "
            f"{self.sla_p99_ms - total['p99']:.1f} ms of headroom."
            if self.meets_sla
            else f"\n**FAIL** — p99 of {total['p99']:.1f} ms exceeds the "
            f"{self.sla_p99_ms:.0f} ms budget by {total['p99'] - self.sla_p99_ms:.1f} ms."
        )
        return "\n".join(lines) + "\n" + verdict
