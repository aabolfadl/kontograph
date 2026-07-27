"""The perturbation teacher — deliberately slow, deliberately trusted.

This is the expensive explainer the fast one is distilled from. It follows the
GNNExplainer family: learn a continuous mask over the candidate edges by
gradient descent, so that keeping the masked subgraph reproduces the model's
score while the mask stays small.

    maximise   score(masked subgraph) ≈ score(full)      (fidelity)
    subject to the mask being sparse and confident        (regularisers)

**It costs seconds per instance, and that is the point.** ADR-0007's whole
argument is that this cost structure is wrong for a 200 ms budget, so the cost
is moved to training time by distillation. Nothing here is optimised for speed —
optimising it would weaken the teacher, and the teacher is the fidelity ceiling
everything else is measured against.

Two regularisers, both standard and both load-bearing:

*Size* — without it the optimiser keeps every edge, which is perfectly faithful
and explains nothing.

*Entropy* — pushes mask values towards 0 or 1. Without it the mask settles into
a grey soup where every edge is "somewhat important", which is unrankable and
therefore useless to an investigator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from kg_core.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class TeacherConfig:
    """Optimisation settings for the perturbation search."""

    #: Gradient steps per instance. This is the cost: ~100 forward+backward
    #: passes for ONE explanation, which is the two-orders-of-magnitude problem
    #: ADR-0007 is about.
    steps: int = 100
    learning_rate: float = 0.05
    size_weight: float = 0.05
    entropy_weight: float = 0.1
    #: Candidate edges considered per instance — the target's temporal
    #: neighbourhood. Bounded so the teacher terminates on dense accounts.
    max_candidates: int = 40
    seed: int = 20251009

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "size_weight": self.size_weight,
            "entropy_weight": self.entropy_weight,
            "max_candidates": self.max_candidates,
        }


@dataclass(slots=True)
class Explanation:
    """One explained alert: which edges mattered, and how much."""

    tx_id: str
    candidate_tx_ids: np.ndarray
    importances: np.ndarray
    base_score: float
    seconds: float = 0.0

    def top(self, k: int = 5) -> list[tuple[str, float]]:
        order = np.argsort(-self.importances, kind="stable")[:k]
        return [(str(self.candidate_tx_ids[i]), float(self.importances[i])) for i in order]

    def as_dict(self) -> dict[str, Any]:
        return {
            "tx_id": self.tx_id,
            "base_score": round(self.base_score, 6),
            "candidates": int(self.candidate_tx_ids.size),
            "seconds": round(self.seconds, 4),
            "top": [{"tx_id": t, "importance": round(v, 6)} for t, v in self.top()],
        }


def candidate_neighbourhood(graph: Any, target_row: int, *, max_candidates: int = 40) -> np.ndarray:
    """Rows in the temporal neighbourhood of the target event.

    Strictly earlier events touching either endpoint, most recent first. The
    strictness is not decoration: an explanation that cited a *later* payment
    would be describing information the model did not have, which is the same
    leak the feature layer forbids, wearing a different hat.
    """
    src = graph.src[target_row]
    dst = graph.dst[target_row]
    t = graph.t[target_row]

    earlier = np.flatnonzero(graph.t < t)
    if earlier.size == 0:
        return np.empty(0, dtype=np.int64)

    touches = (
        (graph.src[earlier] == src)
        | (graph.dst[earlier] == src)
        | (graph.src[earlier] == dst)
        | (graph.dst[earlier] == dst)
    )
    candidates = earlier[touches]
    if candidates.size > max_candidates:
        # Most recent first: a two-hop chain from last week matters less to this
        # decision than one from ten minutes ago, and the budget has to go
        # somewhere.
        candidates = candidates[np.argsort(-graph.t[candidates], kind="stable")][:max_candidates]
    return np.sort(candidates)


def explain_instance(
    score_masked: Any,
    tx_id: str,
    candidates: np.ndarray,
    candidate_tx_ids: np.ndarray,
    base_score: float,
    config: TeacherConfig,
) -> Explanation:
    """Learn an edge mask for one instance by gradient descent.

    ``score_masked(mask)`` must be differentiable in ``mask`` — it applies the
    mask as edge weights and returns the target's score as a torch scalar.
    """
    import time

    import torch

    started = time.perf_counter()
    if candidates.size == 0:
        return Explanation(tx_id, candidate_tx_ids, np.empty(0), base_score, 0.0)

    torch.manual_seed(config.seed)
    # Initialised near zero in logit space -> mask ~0.5, so the search starts
    # undecided rather than committed to keeping or dropping everything.
    logits = torch.nn.Parameter(torch.randn(candidates.size) * 0.1)
    optimiser = torch.optim.Adam([logits], lr=config.learning_rate)
    target = torch.tensor(float(base_score))

    for _ in range(config.steps):
        optimiser.zero_grad()
        mask = torch.sigmoid(logits)
        masked_score = score_masked(mask)

        fidelity = (masked_score - target) ** 2
        size = config.size_weight * mask.mean()
        # Binary entropy: minimised when every value is 0 or 1.
        entropy = (
            -config.entropy_weight
            * (mask * torch.log(mask + 1e-9) + (1 - mask) * torch.log(1 - mask + 1e-9)).mean()
        )

        (fidelity + size + entropy).backward()
        optimiser.step()

    with torch.no_grad():
        importances = torch.sigmoid(logits).cpu().numpy()

    return Explanation(
        tx_id=tx_id,
        candidate_tx_ids=candidate_tx_ids,
        importances=importances,
        base_score=base_score,
        seconds=time.perf_counter() - started,
    )


def distillation_targets(
    explanations: list[Explanation],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split explanations into (candidate rows, importance targets).

    The student is trained to reproduce these in a single forward pass. The
    teacher's masks are the supervision signal; the student never sees the
    label, only what the teacher concluded — which is why the student can be
    faithful to the *model* even where the model is wrong about the world.
    """
    candidates = [e.candidate_tx_ids for e in explanations]
    targets = [e.importances for e in explanations]
    return candidates, targets
