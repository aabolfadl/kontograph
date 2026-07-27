"""The amortised explainer — one forward pass instead of a search.

This is the vertical spike (ADR-0007). The teacher runs ~100 gradient steps per
instance and costs seconds; the student reads the same temporal neighbourhood
the detector already computed and emits edge importances in **one forward pass**,
reusing embeddings the detector produced anyway. Explanation cost collapses from
seconds to a fraction of a millisecond.

# Why this can work at all

The teacher solves an optimisation problem per instance. But that problem has
enormous shared structure across instances: "an edge is important if it is
recent, if it moved a large amount relative to this account's history, if it
connects to a node whose memory is unusual". Those are learnable regularities,
and amortised inference is precisely the technique for replacing a repeated
optimisation with a learned map.

# Why it might not, and how we would know

The student could learn to produce masks that *look* like the teacher's without
being causally load-bearing — mimicking the teacher's style rather than its
substance. Distillation loss alone cannot distinguish those cases, because it
only compares mask values.

So the loss has a second term: a **comprehensiveness regulariser** that scores
the model with the student's top edges removed and pushes that score down. The
student is rewarded for finding edges whose removal actually changes the
decision, not merely edges the teacher happened to like.

And it is measured against the teacher afterwards regardless. The gap is
published (protocol §8) whether flattering or not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from kg_core.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class StudentConfig:
    """Architecture and distillation settings."""

    hidden_dim: int = 64
    dropout: float = 0.1
    learning_rate: float = 1e-3
    epochs: int = 30
    batch_size: int = 32

    #: Weight on the comprehensiveness regulariser, relative to the
    #: distillation loss. Non-zero on purpose: with it at zero the student
    #: optimises only for looking like the teacher, and a plausible-but-inert
    #: mask satisfies that completely.
    comprehensiveness_weight: float = 0.3
    #: How many top edges the regulariser removes when scoring.
    regulariser_top_k: int = 5
    seed: int = 20251009

    def as_dict(self) -> dict[str, Any]:
        return {
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "comprehensiveness_weight": self.comprehensiveness_weight,
            "regulariser_top_k": self.regulariser_top_k,
        }


def build_student(config: StudentConfig, embedding_dim: int, edge_dim: int) -> Any:
    """The explainer head.

    Deliberately small. It runs inside the serving budget alongside the
    detector, and its inputs are things the detector has already computed — the
    target's endpoint embeddings and each candidate edge's features. Recomputing
    anything here would give back the speed the whole design exists to win.
    """
    import torch
    from torch.nn import Dropout, Linear, ReLU, Sequential

    torch.manual_seed(config.seed)

    class ExplainerHead(torch.nn.Module):
        """Scores each candidate edge against the target's context.

        Input per candidate edge:
            target embedding (src, dst)   what is being explained
            candidate edge features       what this edge is
            time delta                    how long before the target it happened
            endpoint-sharing indicators   how it connects to the target

        Output: one importance logit per candidate.
        """

        def __init__(self) -> None:
            super().__init__()
            # 2 target embeddings + candidate edge features + 1 time delta + 3
            # structural indicators.
            in_dim = embedding_dim * 2 + edge_dim + 4
            self.net = Sequential(
                Linear(in_dim, config.hidden_dim),
                ReLU(),
                Dropout(config.dropout),
                Linear(config.hidden_dim, config.hidden_dim // 2),
                ReLU(),
                Linear(config.hidden_dim // 2, 1),
            )

        def forward(
            self,
            target_src_z: Any,
            target_dst_z: Any,
            candidate_features: Any,
            time_deltas: Any,
            structure: Any,
        ) -> Any:
            n = candidate_features.size(0)
            context = torch.cat([target_src_z, target_dst_z], dim=-1).expand(n, -1)
            inputs = torch.cat(
                [context, candidate_features, time_deltas.unsqueeze(-1), structure], dim=-1
            )
            return self.net(inputs).squeeze(-1)

    return ExplainerHead()


def structural_indicators(graph: Any, target_row: int, candidates: np.ndarray) -> np.ndarray:
    """How each candidate edge connects to the target, as three flags.

    Shares-source, shares-destination, and forms-a-chain (this edge's
    destination is the target's source — i.e. money arrived here and then left).
    The chain indicator is the one that matters for layering: it is what turns a
    list of payments into a *path*.
    """
    src, dst = graph.src[target_row], graph.dst[target_row]
    candidate_src, candidate_dst = graph.src[candidates], graph.dst[candidates]

    shares_source = (candidate_src == src) | (candidate_dst == src)
    shares_dest = (candidate_src == dst) | (candidate_dst == dst)
    forms_chain = candidate_dst == src

    return np.stack([shares_source, shares_dest, forms_chain], axis=-1).astype(np.float32)


def time_deltas(graph: Any, target_row: int, candidates: np.ndarray) -> np.ndarray:
    """Log-scaled seconds between each candidate and the target.

    Log-scaled because the interesting range spans seconds to weeks, and a
    linear encoding would make everything inside an hour indistinguishable —
    which is exactly the range where laundering velocity lives.
    """
    deltas = graph.t[target_row] - graph.t[candidates]
    return np.log1p(np.maximum(deltas, 0.0)).astype(np.float32)


@dataclass(slots=True)
class DistillationResult:
    """What one distillation run produced."""

    epochs: int
    final_loss: float
    distillation_loss: float
    comprehensiveness_loss: float
    history: list[dict[str, float]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "final_loss": round(self.final_loss, 6),
            "distillation_loss": round(self.distillation_loss, 6),
            "comprehensiveness_loss": round(self.comprehensiveness_loss, 6),
            "history": self.history[-10:],
        }


def distill(
    student: Any,
    examples: list[dict[str, Any]],
    config: StudentConfig,
    *,
    score_masked: Any = None,
) -> DistillationResult:
    """Train the student to reproduce the teacher's masks in one pass.

    ``examples`` each carry the student's inputs and the teacher's importances.
    ``score_masked`` is optional: when supplied, the comprehensiveness
    regulariser is active and the student is additionally rewarded for picking
    edges whose removal moves the score.
    """
    import torch

    torch.manual_seed(config.seed)
    optimiser = torch.optim.Adam(student.parameters(), lr=config.learning_rate)
    history: list[dict[str, float]] = []
    distillation_value = comprehensiveness_value = 0.0

    for epoch in range(config.epochs):
        student.train()
        epoch_distill, epoch_comprehensive, batches = 0.0, 0.0, 0

        for start in range(0, len(examples), config.batch_size):
            batch = examples[start : start + config.batch_size]
            optimiser.zero_grad()
            total = torch.tensor(0.0, requires_grad=True)

            for example in batch:
                if example["teacher"].size == 0:
                    continue
                predicted = student(
                    example["target_src_z"],
                    example["target_dst_z"],
                    example["candidate_features"],
                    example["time_deltas"],
                    example["structure"],
                )
                teacher = torch.as_tensor(example["teacher"], dtype=torch.float32)
                # Binary cross-entropy against the teacher's mask, which is a
                # value in [0,1] — a soft target, so this is distillation rather
                # than classification.
                distillation = torch.nn.functional.binary_cross_entropy_with_logits(
                    predicted, teacher
                )
                total = total + distillation
                epoch_distill += float(distillation.item())

                if score_masked is not None and config.comprehensiveness_weight > 0:
                    # Soft top-k: a differentiable stand-in for "remove the top
                    # edges". A hard top-k has no gradient, so the student would
                    # get no signal from this term at all.
                    keep = 1.0 - torch.sigmoid(predicted)
                    dropped_score = score_masked(example, keep)
                    penalty = config.comprehensiveness_weight * dropped_score
                    total = total + penalty
                    epoch_comprehensive += float(penalty.item())

                batches += 1

            if batches:
                total.backward()
                optimiser.step()

        distillation_value = epoch_distill / max(1, batches)
        comprehensiveness_value = epoch_comprehensive / max(1, batches)
        history.append(
            {
                "epoch": epoch,
                "distillation": round(distillation_value, 6),
                "comprehensiveness": round(comprehensiveness_value, 6),
            }
        )
        if epoch % 5 == 0:
            log.info("distillation", epoch=epoch, loss=round(distillation_value, 5))

    return DistillationResult(
        epochs=config.epochs,
        final_loss=distillation_value + comprehensiveness_value,
        distillation_loss=distillation_value,
        comprehensiveness_loss=comprehensiveness_value,
        history=history,
    )


def predict_importances(student: Any, example: dict[str, Any]) -> np.ndarray:
    """One forward pass. This is the number ADR-0007 is about."""
    import torch

    student.eval()
    with torch.no_grad():
        logits = student(
            example["target_src_z"],
            example["target_dst_z"],
            example["candidate_features"],
            example["time_deltas"],
            example["structure"],
        )
        return torch.sigmoid(logits).cpu().numpy()
