"""Measuring whether an explanation is honest.

§8 of the evaluation protocol, committed before any explainer existed.

Design principle P5 and ADR-0007: the central risk of distilling an expensive
explainer into a fast one is that the student learns to produce *plausible*
masks rather than *load-bearing* ones. Plausible masks are worse than no
explanation, because an investigator builds a case on them.

So faithfulness is defined operationally — by intervening on the model — rather
than by asking whether the explanation looks reasonable:

``comprehensiveness``
    Remove the salient edges and re-score. If the score barely moves, the
    explanation was not describing what the model used.

``sufficiency``
    Keep *only* the salient edges and re-score. If the score collapses, the
    explanation was incomplete.

``deletion`` / ``insertion`` curves
    The same two ideas swept over the whole ranking rather than at one cut
    point, which is harder to game by tuning k.

Both are needed, and they pull in opposite directions: a mask containing every
edge is perfectly sufficient and completely uninformative; a mask containing one
critical edge may be comprehensive but insufficient. Reporting only one is how
an explainer gets to look good.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Signature of a re-scoring function: given a set of edge indices to KEEP,
#: return the model's score for the target event.
ScoreFn = Callable[[np.ndarray], float]


@dataclass(slots=True)
class FaithfulnessScores:
    """How well one explanation describes what the model actually did."""

    comprehensiveness: float
    sufficiency: float
    deletion_auc: float
    insertion_auc: float
    salient_edges: int
    candidate_edges: int
    base_score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "comprehensiveness": round(self.comprehensiveness, 6),
            "sufficiency": round(self.sufficiency, 6),
            "deletion_auc": round(self.deletion_auc, 6),
            "insertion_auc": round(self.insertion_auc, 6),
            "salient_edges": self.salient_edges,
            "candidate_edges": self.candidate_edges,
            "base_score": round(self.base_score, 6),
        }


def comprehensiveness(
    score_fn: ScoreFn, candidates: np.ndarray, salient: np.ndarray, base_score: float
) -> float:
    """Score drop when the salient edges are removed. Higher is more faithful.

    Normalised by the base score so it is comparable across alerts: a drop of
    0.1 means something different at a base score of 0.15 than at 0.95.
    """
    remaining = np.setdiff1d(candidates, salient, assume_unique=False)
    without = score_fn(remaining)
    return float((base_score - without) / max(base_score, 1e-9))


def sufficiency(score_fn: ScoreFn, salient: np.ndarray, base_score: float) -> float:
    """Score retained using only the salient edges. Higher is more complete."""
    if salient.size == 0:
        return 0.0
    return float(score_fn(salient) / max(base_score, 1e-9))


def deletion_curve(
    score_fn: ScoreFn, candidates: np.ndarray, ranking: np.ndarray, steps: int = 10
) -> list[tuple[float, float]]:
    """Score as edges are removed in descending importance. Lower AUC is better.

    Sweeping rather than cutting at a single k: an explainer tuned to look good
    at exactly top-5 will show it here as a curve that only dips at one point.
    """
    points: list[tuple[float, float]] = []
    total = len(ranking)
    if total == 0:
        return points
    for step in range(steps + 1):
        take = round(total * step / steps)
        removed = ranking[:take]
        remaining = np.setdiff1d(candidates, removed, assume_unique=False)
        points.append((step / steps, score_fn(remaining)))
    return points


def insertion_curve(
    score_fn: ScoreFn, ranking: np.ndarray, steps: int = 10
) -> list[tuple[float, float]]:
    """Score as edges are added in descending importance. Higher AUC is better."""
    points: list[tuple[float, float]] = []
    total = len(ranking)
    if total == 0:
        return points
    for step in range(steps + 1):
        take = round(total * step / steps)
        points.append((step / steps, score_fn(ranking[:take])))
    return points


def curve_auc(points: list[tuple[float, float]]) -> float:
    """Trapezoidal area under a (fraction, score) curve."""
    if len(points) < 2:
        return 0.0
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    # np.trapezoid in numpy 2.x, np.trapz before it. Probed rather than
    # pinned so this works on both Kaggle's image and a local install.
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]
    return float(trapezoid(ys, xs))


def evaluate_explanation(
    score_fn: ScoreFn,
    candidates: np.ndarray,
    importances: np.ndarray,
    *,
    top_k: int = 5,
    steps: int = 10,
) -> FaithfulnessScores:
    """Full faithfulness assessment of one ranked explanation."""
    base_score = score_fn(candidates)
    ranking = candidates[np.argsort(-importances, kind="stable")]
    salient = ranking[: min(top_k, len(ranking))]

    deletion = deletion_curve(score_fn, candidates, ranking, steps=steps)
    insertion = insertion_curve(score_fn, ranking, steps=steps)

    return FaithfulnessScores(
        comprehensiveness=comprehensiveness(score_fn, candidates, salient, base_score),
        sufficiency=sufficiency(score_fn, salient, base_score),
        deletion_auc=curve_auc(deletion),
        insertion_auc=curve_auc(insertion),
        salient_edges=int(salient.size),
        candidate_edges=int(candidates.size),
        base_score=base_score,
    )


# ------------------------------------------------------------- agreement ----


def top_k_jaccard(left: np.ndarray, right: np.ndarray, k: int = 5) -> float:
    """Overlap between two explanations' top-k sets."""
    a = set(left[:k].tolist())
    b = set(right[:k].tolist())
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman correlation between two importance vectors.

    Computed directly to avoid a scipy dependency on the serving path, and
    because the tie handling matters: an explainer that assigns the same
    importance to everything should score 0, not 1.
    """
    if left.size < 2 or left.size != right.size:
        return 0.0
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_centred = left_ranks - left_ranks.mean()
    right_centred = right_ranks - right_ranks.mean()
    denominator = np.sqrt((left_centred**2).sum() * (right_centred**2).sum())
    if denominator == 0:
        return 0.0
    return float((left_centred * right_centred).sum() / denominator)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1)
    sorted_values = values[order]
    start = 0
    for index in range(1, values.size + 1):
        if index == values.size or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return ranks


@dataclass(slots=True)
class TeacherAgreement:
    """How closely the fast student tracks the slow teacher.

    The teacher is the fidelity ceiling. ADR-0007 commits to publishing this gap
    rather than assuming distillation preserved faithfulness — so the number
    exists whether it is flattering or not.
    """

    jaccard_at_5: float
    spearman: float
    instances: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "top5_jaccard": round(self.jaccard_at_5, 6),
            "spearman": round(self.spearman, 6),
            "instances": self.instances,
        }


def compare_to_teacher(
    student_importances: list[np.ndarray],
    teacher_importances: list[np.ndarray],
    *,
    k: int = 5,
) -> TeacherAgreement:
    """Aggregate agreement across many explained instances."""
    jaccards, correlations = [], []
    for student, teacher in zip(student_importances, teacher_importances, strict=True):
        if student.size == 0:
            continue
        student_ranking = np.argsort(-student, kind="stable")
        teacher_ranking = np.argsort(-teacher, kind="stable")
        jaccards.append(top_k_jaccard(student_ranking, teacher_ranking, k))
        correlations.append(rank_correlation(student, teacher))

    return TeacherAgreement(
        jaccard_at_5=float(np.mean(jaccards)) if jaccards else 0.0,
        spearman=float(np.mean(correlations)) if correlations else 0.0,
        instances=len(jaccards),
    )


# ------------------------------------------------------------- stability ----


@dataclass(slots=True)
class StabilityReport:
    """Whether the same alert tells the same story a moment later.

    design principle P5.4: *"an explainer that contradicts itself is worse than no
    explainer at all."* The protocol commits in advance to a top-5 Jaccard
    threshold of 0.6 and to reporting the share below it.
    """

    mean_jaccard: float
    unstable_share: float
    threshold: float
    instances: int
    samples: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean_top5_jaccard": round(self.mean_jaccard, 6),
            "unstable_share": round(self.unstable_share, 6),
            "threshold": self.threshold,
            "instances": self.instances,
        }


#: Committed in protocol §8 before any explainer was built.
STABILITY_THRESHOLD = 0.6


def measure_stability(
    explanations_before: list[np.ndarray],
    explanations_after: list[np.ndarray],
    *,
    k: int = 5,
    threshold: float = STABILITY_THRESHOLD,
) -> StabilityReport:
    """Compare explanations of the same alerts one event tick apart."""
    jaccards: list[float] = []
    for before, after in zip(explanations_before, explanations_after, strict=True):
        if before.size == 0 or after.size == 0:
            continue
        jaccards.append(
            top_k_jaccard(np.argsort(-before, kind="stable"), np.argsort(-after, kind="stable"), k)
        )
    if not jaccards:
        return StabilityReport(0.0, 0.0, threshold, 0)

    values = np.asarray(jaccards)
    return StabilityReport(
        mean_jaccard=float(values.mean()),
        unstable_share=float((values < threshold).mean()),
        threshold=threshold,
        instances=len(jaccards),
        samples=jaccards[:50],
    )
