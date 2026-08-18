"""Unit tests for GraphSAGE link prediction and leak-safe splits."""

from __future__ import annotations

import torch
from torch import nn

from src.gnn_model import LinkPredictor, count_parameters
from src.graph_data import (
    build_link_graph,
    canonicalize_undirected,
    edge_pairs_set,
    make_pair_loader,
    sample_negative_pairs,
    to_bidirectional,
)
import numpy as np


def _toy_graph():
    """Two species, 4 nodes each, sparse within-species edges (room for negatives)."""
    protein_ids = [f"p{i}" for i in range(8)]
    features = torch.randn(8, 8)
    species = torch.tensor([9606, 9606, 9606, 9606, 4932, 4932, 4932, 4932])
    undirected = torch.tensor(
        [
            [0, 0, 1, 4, 4, 5],
            [1, 2, 3, 5, 6, 7],
        ],
        dtype=torch.long,
    )
    return build_link_graph(
        protein_ids=protein_ids,
        features=features,
        species_ids=species,
        undirected_edges=undirected,
        seed=0,
        negative_ratio=1.0,
    )


def test_canonicalize_drops_loops_and_duplicates() -> None:
    raw = torch.tensor([[0, 1, 2, 0], [1, 0, 2, 1]], dtype=torch.long)
    out = canonicalize_undirected(raw)
    pairs = edge_pairs_set(out)
    assert pairs == {(0, 1)}


def test_bidirectional_is_symmetric() -> None:
    undirected = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    both = to_bidirectional(undirected)
    assert both.size(1) == 4
    pairs = {(int(s), int(d)) for s, d in both.T.tolist()}
    assert pairs == {(0, 1), (1, 0), (1, 2), (2, 1)}


def test_negatives_avoid_known_and_self_loops() -> None:
    nodes = torch.arange(4)
    forbidden = {(0, 1), (0, 2)}
    rng = np.random.default_rng(1)
    neg = sample_negative_pairs(3, nodes, forbidden, rng)
    sampled = edge_pairs_set(neg)
    assert sampled.isdisjoint(forbidden)
    assert all(a != b for a, b in sampled)
    assert all(a < b for a, b in sampled)


def test_split_does_not_leak_into_message_passing() -> None:
    graph = _toy_graph()
    mp = edge_pairs_set(graph.data.edge_index)
    train = edge_pairs_set(graph.train_pos)
    val = edge_pairs_set(graph.val_pos)
    test = edge_pairs_set(graph.test_pos)
    assert mp == train
    assert val.isdisjoint(train)
    assert test.isdisjoint(train)
    assert val.isdisjoint(test)
    assert train | val | test == graph.known_undirected


def test_no_cross_species_pairs() -> None:
    graph = _toy_graph()
    species = graph.data.species
    for split in (graph.train_pos, graph.train_neg, graph.val_pos, graph.val_neg, graph.test_pos, graph.test_neg):
        if split.numel() == 0:
            continue
        src_sp = species[split[0]]
        dst_sp = species[split[1]]
        assert torch.equal(src_sp, dst_sp)


def test_negatives_not_in_known_edges() -> None:
    graph = _toy_graph()
    for split in (graph.train_neg, graph.val_neg, graph.test_neg):
        assert edge_pairs_set(split).isdisjoint(graph.known_undirected)


def test_link_predictor_forward_shapes() -> None:
    graph = _toy_graph()
    model = LinkPredictor(in_channels=8, hidden_channels=4, dropout=0.0)
    model.eval()
    loader = make_pair_loader(graph.train_pos, graph.train_neg, batch_size=4, shuffle=False)
    src, dst, y = next(iter(loader))
    with torch.inference_mode():
        logits = model(graph.data.x, graph.data.edge_index, src, dst)
        z = model.encode(graph.data.x, graph.data.edge_index)
    assert z.shape == (8, 4)
    assert logits.shape == y.shape
    assert torch.isfinite(logits).all()
    assert count_parameters(model) > 0


def test_decoder_pair_dim_is_three_times_hidden() -> None:
    decoder = LinkPredictor(in_channels=8, hidden_channels=4, dropout=0.0).decoder
    z = torch.randn(3, 4)
    logits = decoder(z, torch.tensor([0, 1]), torch.tensor([1, 2]))
    assert logits.shape == (2,)
    assert isinstance(decoder.net[0], nn.Linear)
    assert decoder.net[0].in_features == 12
