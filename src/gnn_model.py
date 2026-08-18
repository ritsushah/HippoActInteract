"""GraphSAGE encoder and MLP decoder for undirected PPI link prediction."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv

from src.config import (
    DATA_PROCESSED,
    GNN_ARCHITECTURE_JSON,
    GNN_DROPOUT,
    GNN_HIDDEN_SIZE,
    GNN_NUM_LAYERS,
)
from src.device import get_device
from src.graph_data import LinkPredictionGraph, load_link_graph, make_pair_loader

logger = logging.getLogger(__name__)


class GraphSAGEEncoder(nn.Module):
    """Two-layer GraphSAGE (mean aggregation) over ESM-2 node features."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = GNN_HIDDEN_SIZE,
        num_layers: int = GNN_NUM_LAYERS,
        dropout: float = GNN_DROPOUT,
    ) -> None:
        super().__init__()
        if num_layers != 2:
            raise ValueError("this encoder is fixed at 2 SAGEConv layers")
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.dropout = dropout
        self.hidden_channels = hidden_channels

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        hidden = F.relu(self.conv1(x, edge_index))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.conv2(hidden, edge_index)


class MLPDecoder(nn.Module):
    """Score a pair from concatenated states, including absolute difference."""

    def __init__(self, hidden_channels: int = GNN_HIDDEN_SIZE, dropout: float = GNN_DROPOUT) -> None:
        super().__init__()
        pair_dim = hidden_channels * 3
        self.net = nn.Sequential(
            nn.Linear(pair_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, z: Tensor, src: Tensor, dst: Tensor) -> Tensor:
        left = z[src]
        right = z[dst]
        pair = torch.cat([left, right, (left - right).abs()], dim=-1)
        return self.net(pair).squeeze(-1)


class LinkPredictor(nn.Module):
    """Encode the train graph, then classify candidate undirected edges."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = GNN_HIDDEN_SIZE,
        dropout: float = GNN_DROPOUT,
    ) -> None:
        super().__init__()
        self.encoder = GraphSAGEEncoder(in_channels, hidden_channels, dropout=dropout)
        self.decoder = MLPDecoder(hidden_channels, dropout=dropout)

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.encoder(x, edge_index)

    def decode(self, z: Tensor, src: Tensor, dst: Tensor) -> Tensor:
        return self.decoder(z, src, dst)

    def forward(self, x: Tensor, edge_index: Tensor, src: Tensor, dst: Tensor) -> Tensor:
        z = self.encode(x, edge_index)
        return self.decode(z, src, dst)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


def architecture_dict(model: LinkPredictor, graph: LinkPredictionGraph) -> dict[str, Any]:
    """JSON-serializable description for Phase 3 review."""
    in_channels = int(graph.data.x.size(1))
    hidden = model.encoder.hidden_channels
    return {
        "encoder": {
            "type": "GraphSAGE",
            "aggregation": "mean",
            "layers": [
                f"SAGEConv({in_channels} -> {hidden})",
                "ReLU",
                f"Dropout({model.encoder.dropout})",
                f"SAGEConv({hidden} -> {hidden})",
            ],
        },
        "decoder": {
            "type": "MLP",
            "pair_features": "[z_i || z_j || |z_i - z_j|]",
            "layers": [
                f"Linear({hidden * 3} -> {hidden})",
                "ReLU",
                f"Dropout({model.encoder.dropout})",
                f"Linear({hidden} -> 1)",
            ],
            "loss": "BCEWithLogits",
        },
        "trainable_parameters": count_parameters(model),
        "graph": {
            "num_nodes": graph.num_nodes,
            "feature_dim": in_channels,
            "message_passing_edges": int(graph.data.edge_index.size(1)),
            "train_pos": int(graph.train_pos.size(1)),
            "train_neg": int(graph.train_neg.size(1)),
            "val_pos": int(graph.val_pos.size(1)),
            "val_neg": int(graph.val_neg.size(1)),
            "test_pos": int(graph.test_pos.size(1)),
            "test_neg": int(graph.test_neg.size(1)),
        },
    }


def dummy_forward(model: LinkPredictor, graph: LinkPredictionGraph, device: torch.device) -> dict[str, Any]:
    """One eval-mode pass on a train mini-batch; used to verify shapes."""
    model = model.to(device)
    model.eval()
    data = graph.data
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    loader = make_pair_loader(graph.train_pos, graph.train_neg, shuffle=False)
    src, dst, labels = next(iter(loader))
    src = src.to(device)
    dst = dst.to(device)
    with torch.inference_mode():
        z = model.encode(x, edge_index)
        logits = model.decode(z, src, dst)
    if logits.shape != labels.shape:
        raise RuntimeError(f"logit shape {tuple(logits.shape)} != label shape {tuple(labels.shape)}")
    if not torch.isfinite(logits).all():
        raise RuntimeError("dummy forward produced non-finite logits")
    return {
        "device": str(device),
        "z_shape": [int(z.size(0)), int(z.size(1))],
        "batch_size": int(logits.numel()),
        "logits_shape": [int(logits.numel())],
        "logits_finite": True,
        "logit_mean": float(logits.mean().cpu()),
    }


def run() -> dict[str, Any]:
    """Load the real graph, instantiate the GNN, and write an architecture report."""
    device = get_device()
    graph = load_link_graph()
    model = LinkPredictor(in_channels=int(graph.data.x.size(1)))
    report = architecture_dict(model, graph)
    report["dummy_forward"] = dummy_forward(model, graph, device)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    GNN_ARCHITECTURE_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("architecture:\n%s", json.dumps(report, indent=2))
    logger.info("wrote %s", GNN_ARCHITECTURE_JSON)
    return report


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def main() -> None:
    configure_logging()
    report = run()
    logger.info(
        "GNN ready params=%s z_shape=%s",
        report["trainable_parameters"],
        report["dummy_forward"]["z_shape"],
    )


if __name__ == "__main__":
    main()
