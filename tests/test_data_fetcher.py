"""Unit tests for graph construction helpers (no live API)."""

from __future__ import annotations

from pathlib import Path

from src.config import HUMAN
from src.data_fetcher import (
    InteractionRecord,
    ProteinRecord,
    build_graph,
    dedupe_undirected_edges,
    label_compartment,
    normalize_score,
    write_fasta,
)
from src.uniprot_client import UniProtSequence, _select_best_sequences


def test_label_compartment_hippo_actin_partner() -> None:
    assert label_compartment("YAP1", HUMAN) == "hippo"
    assert label_compartment("actb", HUMAN) == "actin"
    assert label_compartment("AMOT", HUMAN) == "partner"


def test_normalize_score_accepts_string_milli_units() -> None:
    assert normalize_score(0.9) == 0.9
    assert normalize_score(900) == 0.9


def test_dedupe_undirected_edges_keeps_higher_score() -> None:
    low = InteractionRecord("B", "A", "b", "a", 9606, 0.4)
    high = InteractionRecord("A", "B", "a", "b", 9606, 0.8)
    loop = InteractionRecord("A", "A", "a", "a", 9606, 1.0)
    out = dedupe_undirected_edges([low, high, loop])
    assert len(out) == 1
    assert out[0].source_string_id == "A"
    assert out[0].target_string_id == "B"
    assert out[0].combined_score == 0.8


def test_build_graph_drops_edges_without_nodes() -> None:
    proteins = [
        ProteinRecord("A", "YAP1", 9606, "Homo sapiens", "hippo", "P1", "MKT"),
        ProteinRecord("B", "ACTB", 9606, "Homo sapiens", "actin", "P2", "AAA"),
    ]
    edges = [
        InteractionRecord("A", "B", "YAP1", "ACTB", 9606, 0.7),
        InteractionRecord("A", "Z", "YAP1", "MISSING", 9606, 0.9),
    ]
    graph = build_graph(proteins, edges)
    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1
    assert graph["A"]["B"]["combined_score"] == 0.7


def test_write_fasta_wraps_and_includes_header(tmp_path: Path) -> None:
    proteins = [
        ProteinRecord("9606.X", "YAP1", 9606, "Homo sapiens", "hippo", "P46937", "A" * 70),
    ]
    path = tmp_path / "proteins.fasta"
    write_fasta(path, proteins, width=60)
    text = path.read_text(encoding="utf-8")
    assert text.startswith(">9606.X YAP1")
    assert "compartment=hippo" in text
    lines = text.strip().splitlines()
    assert lines[1] == "A" * 60
    assert lines[2] == "A" * 10


def test_uniprot_selects_longest_sequence_for_gene() -> None:
    results = [
        {
            "primaryAccession": "SHORT",
            "genes": [{"geneName": {"value": "YAP1"}}],
            "sequence": {"value": "AAA"},
        },
        {
            "primaryAccession": "LONG",
            "genes": [{"geneName": {"value": "YAP1"}}],
            "sequence": {"value": "AAAAAA"},
        },
        {
            "primaryAccession": "OTHER",
            "genes": [{"geneName": {"value": "TP53"}}],
            "sequence": {"value": "MEEPQ"},
        },
    ]
    selected = _select_best_sequences(results, {"YAP1"})
    assert selected["YAP1"] == UniProtSequence(gene="YAP1", accession="LONG", sequence="AAAAAA")
    assert "TP53" not in selected
