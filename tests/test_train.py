"""Unit tests for training metrics and Hippo–actin candidate filtering."""

from __future__ import annotations

import numpy as np
import torch

from src.gnn_model import LinkPredictor
from src.graph_data import build_link_graph
from src.train import compute_binary_metrics, hippo_actin_candidates, score_pairs


def test_compute_binary_metrics_perfect_ranking() -> None:
    y_true = np.array([0.0, 0.0, 1.0, 1.0])
    logits = np.array([-4.0, -2.0, 2.0, 5.0])
    metrics = compute_binary_metrics(y_true, logits)
    assert metrics["auroc"] == 1.0
    assert metrics["ap"] == 1.0


def test_hippo_actin_candidates_filters_known_and_cross_species() -> None:
    compartments = ["hippo", "actin", "partner", "hippo", "actin"]
    species = [9606, 9606, 9606, 4932, 4932]
    known = {(0, 1)}
    pairs = hippo_actin_candidates(compartments, species, known)
    assert (0, 1) not in pairs
    assert (3, 4) in pairs
    assert (0, 4) not in pairs
    assert (0, 2) not in pairs
    assert all(a < b for a, b in pairs)


def test_both_compartment_pairs_with_hippo_and_actin() -> None:
    compartments = ["both", "hippo", "actin"]
    species = [9606, 9606, 9606]
    pairs = hippo_actin_candidates(compartments, species, known=set())
    assert pairs == [(0, 1), (0, 2), (1, 2)]


def test_score_pairs_is_symmetric() -> None:
    torch.manual_seed(0)
    model = LinkPredictor(in_channels=8, hidden_channels=4, dropout=0.0)
    model.eval()
    z = torch.randn(4, 4)
    pairs = [(0, 1), (0, 2)]
    with torch.inference_mode():
        probs = score_pairs(model, z, pairs)
        flipped = score_pairs(model, z, [(1, 0), (2, 0)])
    assert probs.shape == (2,)
    assert torch.allclose(probs, flipped)
    assert torch.all((probs >= 0) & (probs <= 1))


def test_tiny_train_step_finite_loss() -> None:
    graph = build_link_graph(
        protein_ids=[f"p{i}" for i in range(8)],
        features=torch.randn(8, 8),
        species_ids=torch.tensor([9606, 9606, 9606, 9606, 4932, 4932, 4932, 4932]),
        undirected_edges=torch.tensor([[0, 0, 1, 4, 4, 5], [1, 2, 3, 5, 6, 7]], dtype=torch.long),
        seed=0,
    )
    model = LinkPredictor(in_channels=8, hidden_channels=4, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.BCEWithLogitsLoss()
    model.train()
    src, dst, y = graph.train_pos[0], graph.train_pos[1], torch.ones(graph.train_pos.size(1))
    neg_src, neg_dst = graph.train_neg[0], graph.train_neg[1]
    src = torch.cat([src, neg_src])
    dst = torch.cat([dst, neg_dst])
    y = torch.cat([y, torch.zeros(graph.train_neg.size(1))])
    optimizer.zero_grad()
    logits = model(graph.data.x, graph.data.edge_index, src, dst)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
