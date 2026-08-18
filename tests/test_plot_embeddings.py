"""Unit tests for t-SNE seed-label disambiguation."""

from __future__ import annotations

import numpy as np

from src.plot_embeddings import rightward_positions, unique_seed_labels


def test_rightward_positions_use_empty_side() -> None:
    all_coords = np.vstack(
        [np.column_stack([-15.0 * np.ones(20), np.linspace(-10, 20, 20)]), np.array([[18.0, 0.0]])]
    )
    points = np.array([[-12.0, 5.0], [-11.0, 8.0], [-10.0, -2.0]])
    placed = rightward_positions(points, all_coords)
    assert np.all(placed[:, 0] > points[:, 0])
    assert float(placed[:, 0].min()) > 0.0
    assert np.array_equal(np.argsort(points[:, 1]), np.argsort(placed[:, 1]))


def test_unique_seed_labels_tags_shared_gene_by_species() -> None:
    labels = unique_seed_labels(
        ["CDC42", "YAP1", "CDC42", "VCL"],
        [
            "Homo sapiens",
            "Homo sapiens",
            "Saccharomyces cerevisiae",
            "Homo sapiens",
        ],
    )
    assert labels[0] == "CDC42 (H)"
    assert labels[1] == "YAP1"
    assert labels[2] == "CDC42 (Y)"
    assert 3 not in labels


def test_unique_seed_labels_skips_non_seeds() -> None:
    labels = unique_seed_labels(
        ["CDH1", "ACTB"],
        ["Homo sapiens", "Homo sapiens"],
    )
    assert labels == {1: "ACTB"}
