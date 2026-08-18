"""Unit tests for physical-label splits and topology scores (tiny graphs)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.physical_benchmark import (
    build_protocol_split,
    load_physical_index_pairs,
    metrics_from_scores,
    neighbor_sets,
    precision_at_k,
    sample_degree_matched_negatives,
    split_edge_random,
    split_node_disjoint,
    topology_scores,
)


def test_edge_random_split_covers_all_positives() -> None:
    positives = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7)]
    rng = np.random.default_rng(0)
    train, val, test = split_edge_random(positives, rng)
    assert sorted(train + val + test) == sorted(positives)
    assert train and test


def test_node_disjoint_does_not_cross_buckets() -> None:
    positives = [(0, 1), (2, 3), (4, 5)]
    by_species = {9606: [0, 1, 2, 3, 4, 5]}
    species = torch.tensor([9606] * 6)
    rng = np.random.default_rng(1)
    train, val, test = split_node_disjoint(positives, by_species, species, rng)
    buckets = [set(train), set(val), set(test)]
    nodes = [{n for pair in group for n in pair} for group in (train, val, test)]
    for i in range(3):
        for j in range(i + 1, 3):
            assert nodes[i].isdisjoint(nodes[j]) or not buckets[i] or not buckets[j]


def test_jaccard_and_l3_on_path_graph() -> None:
    context = {(0, 1), (1, 2), (2, 3)}
    neighbors = neighbor_sets(4, context)
    degrees = np.array([len(n) for n in neighbors])
    adj = np.zeros((4, 4))
    for src, dst in context:
        adj[src, dst] = 1
        adj[dst, src] = 1
    jac = topology_scores("jaccard", [(0, 2), (0, 3)], neighbors, degrees, adj)
    assert jac[0] > jac[1]
    a3 = adj @ adj @ adj
    l3 = topology_scores("l3", [(0, 3)], neighbors, degrees, a3)
    assert l3[0] > 0


def test_precision_at_k_and_metrics() -> None:
    y = np.array([1.0, 0.0, 1.0, 0.0])
    scores = np.array([0.9, 0.8, 0.1, 0.0])
    assert precision_at_k(y, scores, 2) == 0.5
    metrics = metrics_from_scores(y, scores)
    assert 0.0 <= metrics["auroc"] <= 1.0


def test_degree_matched_negatives_avoid_positives() -> None:
    positives = [(0, 1), (2, 3)]
    forbidden = set(positives)
    nodes = torch.arange(8)
    degrees = np.array([3, 3, 2, 2, 4, 4, 1, 1])
    rng = np.random.default_rng(0)
    neg = sample_degree_matched_negatives(positives, nodes, forbidden, degrees, rng)
    pairs = {(int(a), int(b)) for a, b in neg.T.tolist()}
    assert pairs.isdisjoint(forbidden)
    assert neg.size(1) == 2


def test_build_protocol_split_edge_random() -> None:
    proteins = pd.DataFrame(
        {
            "string_id": [f"n{i}" for i in range(8)],
            "species_id": [9606] * 8,
        }
    )
    physical = pd.DataFrame(
        {
            "string_id_a": ["n0", "n0", "n1", "n4", "n4", "n5"],
            "string_id_b": ["n1", "n2", "n3", "n5", "n6", "n7"],
        }
    )
    positives, by_species, species_ids = load_physical_index_pairs(proteins, physical)
    string_pairs = set(positives) | {(0, 3), (4, 7)}
    rng = np.random.default_rng(2)
    split = build_protocol_split(
        protocol="edge_random",
        positives=positives,
        by_species=by_species,
        species_ids=species_ids,
        string_pairs=string_pairs,
        rng=rng,
        n_nodes=8,
    )
    assert split["n_train_pos"] + split["n_val_pos"] + split["n_test_pos"] == len(positives)
    assert split["mp_edges"].size(0) == 2
