"""Training and scoring the temporal GNN.

The loop that keeps the memory module honest.

**Chronological, always.** Batches are consecutive slices of the event stream in
event-time order, never shuffled. Shuffling a TGN's training stream is not a
variance-reduction trick, it is a leak: node memory at batch *i* would reflect
events from batch *i+k*.

**Memory is reset before every evaluation.** Otherwise the test fold is scored
against a memory that has already absorbed the test fold. This is the single
easiest way to produce a spectacular and meaningless test number, and it leaves
no trace in the metrics.

**Memory is detached between batches.** Backpropagating through the entire
history would blow up both memory and time; detaching keeps the gradient local
to the batch while the *state* carries forward, which is what TGN prescribes.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from kg_core.logging import get_logger
from kg_models.tgn import TemporalGraph, TGNBundle, TGNConfig, build_model

log = get_logger(__name__)


@dataclass(slots=True)
class TrainingHistory:
    """Per-epoch record, for the model card and the write-up."""

    epochs: list[dict[str, float]] = field(default_factory=list)
    seconds: float = 0.0

    def add(self, **values: float) -> None:
        self.epochs.append({k: round(float(v), 6) for k, v in values.items()})

    def as_dict(self) -> dict[str, Any]:
        return {"epochs": self.epochs, "seconds": round(self.seconds, 2)}


def _batches(count: int, size: int) -> Iterator[tuple[int, int]]:
    for start in range(0, count, size):
        yield start, min(start + size, count)


def train(
    graph: TemporalGraph,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    config: TGNConfig,
) -> tuple[TGNBundle, TrainingHistory]:
    """Train the TGN over the event stream in chronological order."""
    import torch

    bundle = build_model(config, graph)
    bundle.account_ids = graph.account_ids
    optimiser = torch.optim.Adam(bundle.parameters(), lr=config.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(config.positive_weight, device=bundle.device)
    )
    history = TrainingHistory()
    started = time.perf_counter()

    src = torch.as_tensor(graph.src, dtype=torch.long, device=bundle.device)
    dst = torch.as_tensor(graph.dst, dtype=torch.long, device=bundle.device)
    # Long, to match TGNMemory's last_update buffer.
    stamps = torch.as_tensor(graph.t, dtype=torch.long, device=bundle.device)
    features = torch.as_tensor(graph.edge_features, dtype=torch.float, device=bundle.device)
    labels = torch.as_tensor(graph.y, dtype=torch.float, device=bundle.device)

    best_validation = float("inf")
    for epoch in range(config.epochs):
        bundle.train_mode()
        # Both memory and the neighbour cache start empty each epoch: the model
        # must relearn the stream from its beginning, exactly as it will at
        # deployment.
        bundle.reset_state()
        total_loss, seen = 0.0, 0

        for start, end in _batches(len(train_rows), config.batch_size):
            rows = train_rows[start:end]
            if rows.size == 0:
                continue
            optimiser.zero_grad()

            batch_src, batch_dst = src[rows], dst[rows]
            nodes = torch.cat([batch_src, batch_dst]).unique()
            assoc, _node_ids, edge_index, event_ids, z, last_update = bundle.embed(
                nodes, stamps[rows]
            )

            embeddings = bundle.embedding(
                z,
                last_update,
                edge_index,
                stamps[rows][0].expand(edge_index.size(1)),
                features[event_ids] if event_ids.numel() else features[:0],
            )
            logits = bundle.decoder(
                embeddings[assoc[batch_src]], embeddings[assoc[batch_dst]], features[rows]
            )
            loss = criterion(logits, labels[rows])
            loss.backward()
            optimiser.step()

            # Update state AFTER the gradient step, so this batch's own events
            # never contributed to the embedding that predicted them.
            if bundle.memory is not None:
                bundle.memory.update_state(batch_src, batch_dst, stamps[rows], features[rows])
                bundle.memory.detach()
            bundle.neighbor_loader.insert(batch_src, batch_dst)

            total_loss += float(loss.item()) * rows.size
            seen += rows.size

        validation_loss = _evaluate_loss(bundle, graph, validation_rows, criterion, config)
        history.add(
            epoch=epoch, train_loss=total_loss / max(1, seen), validation_loss=validation_loss
        )
        best_validation = min(best_validation, validation_loss)
        log.info(
            "tgn epoch",
            epoch=epoch,
            train_loss=round(total_loss / max(1, seen), 5),
            validation_loss=round(validation_loss, 5),
        )

    history.seconds = time.perf_counter() - started
    return bundle, history


def _evaluate_loss(
    bundle: TGNBundle,
    graph: TemporalGraph,
    rows: np.ndarray,
    criterion: Any,
    config: TGNConfig,
) -> float:
    """Validation loss, replayed from a cold memory."""
    scores = score(bundle, graph, rows, config, return_logits=True)
    if scores.size == 0:
        return float("nan")
    import torch

    return float(
        criterion(
            torch.as_tensor(scores, dtype=torch.float),
            torch.as_tensor(graph.y[rows], dtype=torch.float),
        ).item()
    )


def score(
    bundle: TGNBundle,
    graph: TemporalGraph,
    rows: np.ndarray,
    config: TGNConfig,
    *,
    return_logits: bool = False,
    replay_from: np.ndarray | None = None,
) -> np.ndarray:
    """Score *rows*, replaying the stream to build the memory they need.

    ``replay_from`` is the crucial argument. Scoring the test fold requires node
    memory as it stood at the start of that fold, which means replaying the
    *earlier* events first without scoring them. Skipping the replay and scoring
    from a cold memory understates the model; scoring without resetting first
    overstates it. Both are wrong, in opposite directions, and neither is
    visible in the output.
    """
    import torch

    bundle.eval_mode()
    bundle.reset_state()

    src = torch.as_tensor(graph.src, dtype=torch.long, device=bundle.device)
    dst = torch.as_tensor(graph.dst, dtype=torch.long, device=bundle.device)
    # Long, to match TGNMemory's last_update buffer.
    stamps = torch.as_tensor(graph.t, dtype=torch.long, device=bundle.device)
    features = torch.as_tensor(graph.edge_features, dtype=torch.float, device=bundle.device)

    with torch.no_grad():
        if replay_from is not None and replay_from.size:
            for start, end in _batches(len(replay_from), config.batch_size):
                warm = replay_from[start:end]
                if bundle.memory is not None:
                    bundle.memory.update_state(src[warm], dst[warm], stamps[warm], features[warm])
                bundle.neighbor_loader.insert(src[warm], dst[warm])

        outputs: list[np.ndarray] = []
        for start, end in _batches(len(rows), config.batch_size):
            batch = rows[start:end]
            if batch.size == 0:
                continue
            batch_src, batch_dst = src[batch], dst[batch]
            nodes = torch.cat([batch_src, batch_dst]).unique()
            assoc, _node_ids, edge_index, event_ids, z, last_update = bundle.embed(
                nodes, stamps[batch]
            )
            embeddings = bundle.embedding(
                z,
                last_update,
                edge_index,
                stamps[batch][0].expand(edge_index.size(1)),
                features[event_ids] if event_ids.numel() else features[:0],
            )
            logits = bundle.decoder(
                embeddings[assoc[batch_src]], embeddings[assoc[batch_dst]], features[batch]
            )
            outputs.append(logits.cpu().numpy())

            if bundle.memory is not None:
                bundle.memory.update_state(batch_src, batch_dst, stamps[batch], features[batch])
            bundle.neighbor_loader.insert(batch_src, batch_dst)

    if not outputs:
        return np.empty(0, dtype=np.float32)
    stacked = np.concatenate(outputs)
    if return_logits:
        return stacked
    return 1.0 / (1.0 + np.exp(-np.clip(stacked, -50, 50)))


def save(bundle: TGNBundle, history: TrainingHistory, directory: Path) -> Path:
    """Persist weights, config, node vocabulary and the memory state.

    The node vocabulary and memory travel with the weights deliberately. A
    restored model with a fresh memory silently scores its first several
    thousand transactions against an empty history, and a restored model with a
    different node ordering scores them against the wrong accounts entirely.
    """
    import json

    import torch

    directory.mkdir(parents=True, exist_ok=True)
    state = {
        "embedding": bundle.embedding.state_dict(),
        "decoder": bundle.decoder.state_dict(),
    }
    if bundle.memory is not None:
        state["memory"] = bundle.memory.state_dict()
    torch.save(state, directory / "tgn.pt")

    (directory / "tgn.json").write_text(
        json.dumps(
            {
                "config": bundle.config.as_dict(),
                "num_nodes": bundle.num_nodes,
                "edge_dim": bundle.edge_dim,
                "account_ids": bundle.account_ids,
                "history": history.as_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("tgn saved", path=str(directory))
    return directory


def load(directory: Path, graph: TemporalGraph) -> TGNBundle:
    """Restore a trained TGN against a graph with the same node vocabulary."""
    import json

    import torch

    payload = json.loads((directory / "tgn.json").read_text(encoding="utf-8"))
    config = TGNConfig(**payload["config"])
    if payload["account_ids"] != graph.account_ids:
        raise ValueError(
            "the saved model's node vocabulary does not match this graph. Node indices "
            "are positional, so scoring would attribute each account's memory to a "
            "different account."
        )

    bundle = build_model(config, graph)
    bundle.account_ids = payload["account_ids"]
    state = torch.load(directory / "tgn.pt", map_location=bundle.device, weights_only=False)
    bundle.embedding.load_state_dict(state["embedding"])
    bundle.decoder.load_state_dict(state["decoder"])
    if bundle.memory is not None and "memory" in state:
        bundle.memory.load_state_dict(state["memory"])
    return bundle
