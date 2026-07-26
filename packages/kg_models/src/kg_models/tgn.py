"""The continuous-time temporal graph network — rungs 3 and 4 of the ladder.

ADR-0006 explains why a continuous-time model rather than snapshots. This module
is the implementation, built on PyTorch Geometric's ``TGNMemory`` and
``LastNeighborLoader`` following the structure of the canonical ``examples/tgn.py``.

# The architecture, and what each piece is for

**Memory.** A vector per account, updated by a GRU every time that account
transacts. This is what makes an account's whole history available at scoring
time without re-traversing it — and therefore what makes a graph model feasible
inside a 200 ms budget at all. The memory module is the difference between rungs
3 and 4 of the ablation ladder, and it is switchable precisely so that question
can be answered rather than assumed.

**Temporal attention.** An embedding for the two endpoints, computed by attending
over their most recent neighbours with a time encoding on each edge. "Twelve
transfers in nine minutes" and "twelve transfers over three weeks" produce
different embeddings, which is the entire point of continuous time.

**A supervised edge decoder.** Most TGN literature trains on link prediction.
We have labels, so the decoder is a binary classifier over the concatenated
endpoint embeddings plus the transaction's own features.

# Two things that are easy to get wrong

**Memory leaks the future if the stream is not processed in order.** Node memory
at time t must reflect only events before t. The training loop therefore walks
the stream chronologically and calls ``detach_memory`` between batches, and
evaluation *resets* memory before replaying — otherwise the test fold is scored
against a memory that already absorbed it.

**Memory state must be checkpointed with the weights.** A restored model with
cold memory silently scores the first several thousand transactions against an
empty history. Serving depends on this; see ADR-0006's "bad" list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kg_core.logging import get_logger

log = get_logger(__name__)


def torch_available() -> bool:
    """Whether the optional torch extra is installed."""
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(slots=True)
class TGNConfig:
    """Architecture and training settings.

    Defaults are sized for the ``full`` profile on a single Kaggle GPU: a few
    minutes per epoch, comfortably inside the 12-hour session cap and the weekly
    quota.
    """

    memory_dim: int = 100
    time_dim: int = 100
    embedding_dim: int = 100
    #: Most recent neighbours attended over per node. 10 is the TGN paper's
    #: default and is also a latency decision: this bounds the work done per
    #: score, and the serving budget is 200 ms.
    neighbours: int = 10
    heads: int = 2
    dropout: float = 0.1

    #: Rung 3 vs rung 4. With memory disabled the model still has temporal
    #: attention over recent neighbours, so the ablation isolates exactly what
    #: per-node state contributes.
    use_memory: bool = True

    learning_rate: float = 1e-4
    batch_size: int = 200
    epochs: int = 5
    #: Positives are ~0.2% of the stream. Without reweighting the loss is
    #: minimised by predicting zero everywhere.
    positive_weight: float = 50.0
    seed: int = 20251009
    device: str = "auto"

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def as_dict(self) -> dict[str, Any]:
        return {
            "memory_dim": self.memory_dim,
            "time_dim": self.time_dim,
            "embedding_dim": self.embedding_dim,
            "neighbours": self.neighbours,
            "heads": self.heads,
            "dropout": self.dropout,
            "use_memory": self.use_memory,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "positive_weight": self.positive_weight,
            "seed": self.seed,
        }


@dataclass(slots=True)
class TemporalGraph:
    """The payment stream as a continuous-time dynamic graph.

    Nodes are accounts, edges are payments, and each edge carries the
    transaction's own feature vector. Node *indices* are dense integers because
    the memory module is an embedding table; the mapping back to account ids is
    kept so an explanation can name real accounts.
    """

    src: np.ndarray
    dst: np.ndarray
    t: np.ndarray
    edge_features: np.ndarray
    y: np.ndarray
    tx_ids: np.ndarray
    account_ids: list[str]
    num_nodes: int

    def __len__(self) -> int:
        return len(self.src)

    def slice(self, rows: np.ndarray) -> TemporalGraph:
        return TemporalGraph(
            src=self.src[rows],
            dst=self.dst[rows],
            t=self.t[rows],
            edge_features=self.edge_features[rows],
            y=self.y[rows],
            tx_ids=self.tx_ids[rows],
            account_ids=self.account_ids,
            num_nodes=self.num_nodes,
        )


def build_temporal_graph(
    debtor_ids: np.ndarray,
    creditor_ids: np.ndarray,
    timestamps: np.ndarray,
    edge_features: np.ndarray,
    y: np.ndarray,
    tx_ids: np.ndarray,
) -> TemporalGraph:
    """Encode account ids as dense node indices, preserving event-time order.

    Sorted by ``(t, tx_id)`` — the same total order the feature layer uses — so
    that memory updates happen in exactly the order the features were computed
    under. A different order here would mean the GNN and the tabular features
    disagree about what "the past" was.
    """
    order = np.lexsort((tx_ids, timestamps))
    debtor_ids, creditor_ids = debtor_ids[order], creditor_ids[order]
    timestamps, edge_features = timestamps[order], edge_features[order]
    y, tx_ids = y[order], tx_ids[order]

    accounts = sorted({*debtor_ids.tolist(), *creditor_ids.tolist()})
    index = {account: position for position, account in enumerate(accounts)}

    return TemporalGraph(
        src=np.fromiter((index[a] for a in debtor_ids), dtype=np.int64, count=len(debtor_ids)),
        dst=np.fromiter((index[a] for a in creditor_ids), dtype=np.int64, count=len(creditor_ids)),
        # Integer SECONDS. Two constraints meet here:
        #   - PyG's TGNMemory stores `last_update` as a Long buffer, so a float
        #     timestamp fails on assignment with a dtype mismatch.
        #   - The time encoder learns a Fourier basis over the delta, and
        #     millisecond magnitudes push it into a numerically unhappy regime.
        # Seconds satisfy both, and one-second resolution is far finer than any
        # window in the feature registry.
        t=(timestamps // 1000).astype(np.int64),
        edge_features=edge_features.astype(np.float32),
        y=y.astype(np.float32),
        tx_ids=tx_ids,
        account_ids=accounts,
        num_nodes=len(accounts),
    )


# ---------------------------------------------------------------------------
# Everything below needs torch. Imported inside functions so that `kg_models`
# stays importable — and the whole CPU pipeline stays runnable — without the
# optional extra.
# ---------------------------------------------------------------------------


def build_model(config: TGNConfig, graph: TemporalGraph) -> Any:
    """Construct the TGN. Returns a module bundle; see :class:`TGNBundle`."""
    import torch
    from torch.nn import Dropout, Linear, ReLU, Sequential
    from torch_geometric.nn import TransformerConv
    from torch_geometric.nn.models.tgn import (
        IdentityMessage,
        LastAggregator,
        LastNeighborLoader,
        TGNMemory,
        TimeEncoder,
    )

    torch.manual_seed(config.seed)
    device = torch.device(config.resolve_device())
    edge_dim = graph.edge_features.shape[1]

    class GraphAttentionEmbedding(torch.nn.Module):
        """Temporal attention over a node's most recent neighbours.

        The time delta on each edge is encoded and concatenated to the edge
        features, so the attention can distinguish "this neighbour paid me a
        minute ago" from "a month ago". Without it this collapses to a static
        GNN over a recency-truncated neighbourhood, which is rung 2 of the
        ladder, not rung 3.
        """

        def __init__(
            self, in_channels: int, out_channels: int, msg_dim: int, time_enc: Any
        ) -> None:
            super().__init__()
            self.time_enc = time_enc
            attention_edge_dim = msg_dim + time_enc.out_channels
            self.conv = TransformerConv(
                in_channels,
                out_channels // config.heads,
                heads=config.heads,
                dropout=config.dropout,
                edge_dim=attention_edge_dim,
            )

        def forward(self, x: Any, last_update: Any, edge_index: Any, t: Any, msg: Any) -> Any:
            relative_t = last_update[edge_index[0]] - t
            relative_t_enc = self.time_enc(relative_t.to(x.dtype))
            edge_attr = torch.cat([relative_t_enc, msg], dim=-1)
            return self.conv(x, edge_index, edge_attr)

    class EdgeDecoder(torch.nn.Module):
        """Binary classifier over the two endpoint embeddings plus edge features.

        The transaction's own features are concatenated deliberately. The graph
        contributes context; the payment itself still carries the amount, the
        channel and the VoP outcome, and a decoder that ignored them would be
        throwing away the strongest single-transaction signals in order to look
        more graph-native.
        """

        def __init__(self, embedding_dim: int, edge_dim: int) -> None:
            super().__init__()
            self.net = Sequential(
                Linear(embedding_dim * 2 + edge_dim, embedding_dim),
                ReLU(),
                Dropout(config.dropout),
                Linear(embedding_dim, embedding_dim // 2),
                ReLU(),
                Linear(embedding_dim // 2, 1),
            )

        def forward(self, z_src: Any, z_dst: Any, edge_features: Any) -> Any:
            return self.net(torch.cat([z_src, z_dst, edge_features], dim=-1)).squeeze(-1)

    memory = None
    if config.use_memory:
        memory = TGNMemory(
            graph.num_nodes,
            edge_dim,
            config.memory_dim,
            config.time_dim,
            message_module=IdentityMessage(edge_dim, config.memory_dim, config.time_dim),
            aggregator_module=LastAggregator(),
        ).to(device)

    time_encoder = TimeEncoder(config.time_dim).to(device)
    embedding = GraphAttentionEmbedding(
        in_channels=config.memory_dim if config.use_memory else edge_dim,
        out_channels=config.embedding_dim,
        msg_dim=edge_dim,
        time_enc=memory.time_enc if memory is not None else time_encoder,
    ).to(device)
    decoder = EdgeDecoder(config.embedding_dim, edge_dim).to(device)
    neighbor_loader = LastNeighborLoader(graph.num_nodes, size=config.neighbours, device=device)

    return TGNBundle(
        config=config,
        memory=memory,
        embedding=embedding,
        decoder=decoder,
        neighbor_loader=neighbor_loader,
        device=device,
        num_nodes=graph.num_nodes,
        edge_dim=edge_dim,
        fallback_time_encoder=time_encoder,
    )


@dataclass(slots=True)
class TGNBundle:
    """The trained modules plus the state that has to travel with them."""

    config: TGNConfig
    memory: Any
    embedding: Any
    decoder: Any
    neighbor_loader: Any
    device: Any
    num_nodes: int
    edge_dim: int
    fallback_time_encoder: Any = None
    account_ids: list[str] = field(default_factory=list)

    def modules(self) -> list[Any]:
        return [m for m in (self.memory, self.embedding, self.decoder) if m is not None]

    def parameters(self) -> Any:
        import itertools

        return itertools.chain(*(m.parameters() for m in self.modules()))

    def train_mode(self) -> None:
        for module in self.modules():
            module.train()

    def eval_mode(self) -> None:
        for module in self.modules():
            module.eval()

    def reset_state(self) -> None:
        """Clear memory and the neighbour cache.

        Called before every evaluation replay. Without it the test fold is
        scored against a memory that has already absorbed the test fold — a
        leak that produces excellent numbers and no warning.
        """
        if self.memory is not None:
            self.memory.reset_state()
        self.neighbor_loader.reset_state()

    def embed(self, nodes: Any, t: Any) -> Any:
        """Embeddings for *nodes* at time *t*, via their recent neighbourhood."""
        import torch

        node_ids, edge_index, event_ids = self.neighbor_loader(nodes)
        # Dense re-indexing so the attention operates on a small subgraph rather
        # than on the full node set. This is the mechanism that keeps per-event
        # cost bounded, and therefore what makes the latency budget reachable.
        assoc = torch.empty(self.num_nodes, dtype=torch.long, device=self.device)
        assoc[node_ids] = torch.arange(node_ids.size(0), device=self.device)

        if self.memory is not None:
            z, last_update = self.memory(node_ids)
        else:
            # Rung 3: no per-node state. Start from zeros so the embedding comes
            # purely from the attended neighbourhood.
            z = torch.zeros(
                node_ids.size(0), self.edge_dim, device=self.device, dtype=torch.float32
            )
            last_update = torch.zeros(node_ids.size(0), device=self.device)

        return assoc, node_ids, edge_index, event_ids, z, last_update
