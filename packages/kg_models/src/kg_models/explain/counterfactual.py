"""Minimal-edit counterfactual search.

Design principle P5.3: not *"feature 17 had high attribution"* but *"remove these two
hops through this intermediary and the score drops below threshold."*

# Searched, never predicted

A *predicted* counterfactual is unfalsifiable — the model asserts an edit would
help and nobody checks. Every candidate here is **re-scored**, and a
counterfactual is emitted only if the model genuinely falls below threshold
without it. Validity is therefore a property of each individual answer, not an
average over a benchmark.

# Constrained to edits that mean something

The search only *removes* edges, and only edges in the target's own temporal
neighbourhood. That restriction is deliberate: an unconstrained search finds
minimal perturbations that are mathematically valid and operationally absurd —
"if this payment had been €4,000.01 instead of €4,000.00" tells an investigator
nothing. Removing a payment corresponds to a real question ("what if this
transfer had been held?") and to a real intervention.

# Budgeted in milliseconds

Beam search over a bounded candidate set, ordered by the student's importances
so the promising edits are tried first. The budget is a wall-clock deadline, not
an iteration count, because the serving path has 200 ms total and the
counterfactual gets a slice of it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kg_core.logging import get_logger

log = get_logger(__name__)

#: Given a set of edge rows to KEEP, return the target's score.
ScoreFn = Callable[[np.ndarray], float]


@dataclass(slots=True)
class CounterfactualConfig:
    """Search budget and shape."""

    beam_width: int = 3
    max_edits: int = 4
    #: Wall-clock budget in milliseconds. The serving path has 200 ms in total;
    #: this is the counterfactual's slice of it.
    budget_ms: float = 25.0
    #: Only consider this many of the highest-importance candidates.
    max_candidates: int = 20

    def as_dict(self) -> dict[str, Any]:
        return {
            "beam_width": self.beam_width,
            "max_edits": self.max_edits,
            "budget_ms": self.budget_ms,
            "max_candidates": self.max_candidates,
        }


@dataclass(slots=True)
class Counterfactual:
    """A verified minimal edit, or an honest failure to find one."""

    found: bool
    removed_rows: np.ndarray
    removed_tx_ids: list[str]
    score_before: float
    score_after: float
    threshold: float
    edits: int
    searched: int
    elapsed_ms: float
    #: Why no counterfactual exists, when none was found.
    reason: str = ""

    @property
    def crosses_threshold(self) -> bool:
        return self.found and self.score_after < self.threshold

    def as_dict(self) -> dict[str, Any]:
        return {
            "found": self.found,
            "removed_tx_ids": self.removed_tx_ids,
            "score_before": round(self.score_before, 6),
            "score_after": round(self.score_after, 6),
            "threshold": round(self.threshold, 6),
            "crosses_threshold": self.crosses_threshold,
            "edits": self.edits,
            "candidates_searched": self.searched,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "reason": self.reason,
        }


def search(
    score_fn: ScoreFn,
    candidates: np.ndarray,
    importances: np.ndarray,
    tx_ids: np.ndarray,
    *,
    threshold: float,
    config: CounterfactualConfig | None = None,
) -> Counterfactual:
    """Find the smallest set of removals that drops the score below threshold.

    Breadth-first over edit sets of increasing size, keeping the ``beam_width``
    most promising partial edits at each depth. Increasing size means the first
    solution found is minimal in *count*, which is the property an investigator
    cares about — "these two payments" is actionable, "these eleven payments" is
    a shrug.
    """
    config = config or CounterfactualConfig()
    started = time.perf_counter()

    base_score = score_fn(candidates)
    if base_score < threshold:
        return Counterfactual(
            found=False,
            removed_rows=np.empty(0, dtype=np.int64),
            removed_tx_ids=[],
            score_before=base_score,
            score_after=base_score,
            threshold=threshold,
            edits=0,
            searched=0,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            reason="the alert is already below threshold; nothing to explain away",
        )

    ranked = candidates[np.argsort(-importances, kind="stable")][: config.max_candidates]
    tx_by_row = dict(zip(candidates.tolist(), tx_ids.tolist(), strict=False))

    beam: list[tuple[float, tuple[int, ...]]] = [(base_score, ())]
    searched = 0

    for depth in range(1, config.max_edits + 1):
        next_beam: list[tuple[float, tuple[int, ...]]] = []

        for _, removed in beam:
            for row in ranked:
                if row in removed:
                    continue
                if (time.perf_counter() - started) * 1000 > config.budget_ms:
                    return _exhausted(
                        beam,
                        base_score,
                        threshold,
                        searched,
                        started,
                        tx_by_row,
                        reason=f"budget of {config.budget_ms:.0f} ms exhausted at depth {depth}",
                    )

                candidate_removal = (*removed, int(row))
                keep = np.setdiff1d(candidates, np.asarray(candidate_removal), assume_unique=False)
                score = score_fn(keep)
                searched += 1

                if score < threshold:
                    # Verified: this edit really does move the model below the
                    # operating point. Nothing is asserted that was not checked.
                    return Counterfactual(
                        found=True,
                        removed_rows=np.asarray(candidate_removal, dtype=np.int64),
                        removed_tx_ids=[str(tx_by_row.get(r, r)) for r in candidate_removal],
                        score_before=base_score,
                        score_after=score,
                        threshold=threshold,
                        edits=len(candidate_removal),
                        searched=searched,
                        elapsed_ms=(time.perf_counter() - started) * 1000,
                    )
                next_beam.append((score, candidate_removal))

        # Keep the edits that pushed the score down most; those are the ones
        # most likely to reach the threshold with one more removal.
        next_beam.sort(key=lambda entry: entry[0])
        beam = next_beam[: config.beam_width]
        if not beam:
            break

    return _exhausted(
        beam,
        base_score,
        threshold,
        searched,
        started,
        tx_by_row,
        reason=(
            f"no removal of up to {config.max_edits} payments brings the score below "
            "threshold; the alert does not rest on a small number of edges"
        ),
    )


def _exhausted(
    beam: list[tuple[float, tuple[int, ...]]],
    base_score: float,
    threshold: float,
    searched: int,
    started: float,
    tx_by_row: dict[int, Any],
    *,
    reason: str,
) -> Counterfactual:
    """Report the best edit found, clearly marked as not reaching the threshold.

    Returning the near-miss rather than nothing is deliberate: "removing these
    two drops the score from 0.81 to 0.62, still above the 0.55 threshold" is
    useful to an investigator. Presenting it as a *success* would not be.
    """
    best_score, best_removal = min(beam, key=lambda entry: entry[0]) if beam else (base_score, ())
    return Counterfactual(
        found=False,
        removed_rows=np.asarray(best_removal, dtype=np.int64),
        removed_tx_ids=[str(tx_by_row.get(r, r)) for r in best_removal],
        score_before=base_score,
        score_after=best_score,
        threshold=threshold,
        edits=len(best_removal),
        searched=searched,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        reason=reason,
    )


@dataclass(slots=True)
class CounterfactualReport:
    """Aggregate validity and sparsity — protocol §8."""

    attempted: int
    found: int
    median_edits: float
    median_ms: float
    mean_score_drop: float
    samples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def validity(self) -> float:
        """Share of alerts for which a verified counterfactual was found.

        Not "share that looked plausible" — every one of these was re-scored.
        """
        return self.found / self.attempted if self.attempted else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "found": self.found,
            "validity": round(self.validity, 6),
            "median_edits": self.median_edits,
            "median_ms": round(self.median_ms, 3),
            "mean_score_drop": round(self.mean_score_drop, 6),
        }


def summarise(results: list[Counterfactual]) -> CounterfactualReport:
    """Aggregate a batch of counterfactual searches."""
    found = [r for r in results if r.crosses_threshold]
    edits = [r.edits for r in found] or [0]
    times = [r.elapsed_ms for r in results] or [0.0]
    drops = [r.score_before - r.score_after for r in results] or [0.0]

    return CounterfactualReport(
        attempted=len(results),
        found=len(found),
        median_edits=float(np.median(edits)),
        median_ms=float(np.median(times)),
        mean_score_drop=float(np.mean(drops)),
        samples=[r.as_dict() for r in results[:20]],
    )


# --------------------------------------------------------------- narrative --


def narrate(
    counterfactual: Counterfactual,
    edge_descriptions: dict[str, str],
    *,
    language: str = "en",
) -> str:
    """A sentence an investigator can put in a SAR draft.

    Bilingual because the reporting mart and the console are DE/EN (the regulatory context), and
    because a Verdachtsmeldung is filed in German.
    """
    if not counterfactual.crosses_threshold:
        if language == "de":
            return (
                "Kein einzelner Zahlungsvorgang erklärt diesen Alarm: das Entfernen von bis zu "
                f"{counterfactual.edits} Transaktionen senkt den Score nur von "
                f"{counterfactual.score_before:.2f} auf {counterfactual.score_after:.2f} "
                f"(Schwelle {counterfactual.threshold:.2f})."
            )
        return (
            "No small set of payments explains this alert: removing up to "
            f"{counterfactual.edits} transactions moves the score only from "
            f"{counterfactual.score_before:.2f} to {counterfactual.score_after:.2f} "
            f"(threshold {counterfactual.threshold:.2f})."
        )

    described = [edge_descriptions.get(tx, tx) for tx in counterfactual.removed_tx_ids]
    listed = "; ".join(described)

    if language == "de":
        return (
            f"Der Alarm beruht auf {counterfactual.edits} Zahlung(en): {listed}. "
            f"Ohne diese fällt der Score von {counterfactual.score_before:.2f} auf "
            f"{counterfactual.score_after:.2f} und damit unter die Schwelle von "
            f"{counterfactual.threshold:.2f}."
        )
    return (
        f"This alert rests on {counterfactual.edits} payment(s): {listed}. "
        f"Without them the score falls from {counterfactual.score_before:.2f} to "
        f"{counterfactual.score_after:.2f}, below the {counterfactual.threshold:.2f} threshold."
    )
