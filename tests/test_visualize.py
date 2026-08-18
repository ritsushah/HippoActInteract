"""Unit tests for Hippo–actin subgraph construction."""

from __future__ import annotations

from io import StringIO

import pandas as pd

from src.visualize import build_subgraph, top_hits_for_species


def test_top_hits_for_species_respects_limit() -> None:
    hits = pd.DataFrame(
        {
            "species": ["Homo sapiens"] * 4 + ["Saccharomyces cerevisiae"],
            "probability": [0.2, 0.9, 0.5, 0.7, 0.8],
            "string_id_a": ["h1", "h2", "h3", "h4", "y1"],
            "string_id_b": ["a1", "a2", "a3", "a4", "ya1"],
        }
    )
    top = top_hits_for_species(hits, "Homo sapiens", top_n=2)
    assert list(top["probability"]) == [0.9, 0.7]


def test_build_subgraph_marks_predicted_and_known() -> None:
    hits = pd.DataFrame(
        {
            "string_id_a": ["9606.A"],
            "string_id_b": ["9606.B"],
            "probability": [0.91],
        }
    )
    interactions = pd.DataFrame(
        {
            "source_string_id": ["9606.A", "9606.A"],
            "target_string_id": ["9606.C", "9606.B"],
            "combined_score": [0.8, 0.99],
        }
    )
    proteins = pd.read_csv(
        StringIO(
            "string_id,preferred_name,compartment,species_name\n"
            "9606.A,YAP1,hippo,Homo sapiens\n"
            "9606.B,ACTB,actin,Homo sapiens\n"
            "9606.C,VCL,actin,Homo sapiens\n"
        )
    )
    graph = build_subgraph(hits, interactions, proteins)
    assert set(graph.nodes) == {"9606.A", "9606.B"}
    assert graph["9606.A"]["9606.B"]["kind"] == "predicted"
    assert graph["9606.A"]["9606.B"]["probability"] == 0.91
    assert not graph.has_edge("9606.A", "9606.C")
    assert graph.nodes["9606.A"]["name"] == "YAP1"


def test_known_edge_added_when_both_nodes_present() -> None:
    hits = pd.DataFrame(
        {
            "string_id_a": ["A", "A"],
            "string_id_b": ["B", "C"],
            "probability": [0.8, 0.7],
        }
    )
    interactions = pd.DataFrame(
        {
            "source_string_id": ["B"],
            "target_string_id": ["C"],
            "combined_score": [0.85],
        }
    )
    proteins = pd.DataFrame(
        {
            "string_id": ["A", "B", "C"],
            "preferred_name": ["YAP1", "ACTB", "VCL"],
            "compartment": ["hippo", "actin", "actin"],
            "species_name": ["Homo sapiens"] * 3,
        }
    )
    graph = build_subgraph(hits, interactions, proteins)
    assert graph["B"]["C"]["kind"] == "known"
    predicted = [(u, v) for u, v, d in graph.edges(data=True) if d["kind"] == "predicted"]
    assert len(predicted) == 2
