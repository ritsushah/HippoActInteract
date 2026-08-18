"""Download Hippo–actin STRING networks and UniProt sequences, then write graph files."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd

from src.config import (
    DATA_PROCESSED,
    DATA_RAW,
    INGEST_SUMMARY_JSON,
    INTERACTIONS_CSV,
    MAX_PARTNERS_PER_SEED,
    NETWORK_TYPE,
    PROTEINS_CSV,
    PROTEINS_FASTA,
    REQUIRED_SCORE,
    SPECIES,
    SpeciesSeeds,
)
from src.string_client import StringClient
from src.uniprot_client import UniProtClient, UniProtSequence

logger = logging.getLogger(__name__)

FASTA_LINE_WIDTH: int = 60


@dataclass(frozen=True)
class ProteinRecord:
    string_id: str
    preferred_name: str
    species_id: int
    species_name: str
    compartment: str
    uniprot_accession: str
    sequence: str

    @property
    def sequence_length(self) -> int:
        return len(self.sequence)


@dataclass(frozen=True)
class InteractionRecord:
    source_string_id: str
    target_string_id: str
    source_name: str
    target_name: str
    species_id: int
    combined_score: float


def label_compartment(preferred_name: str, seeds: SpeciesSeeds) -> str:
    """Classify a protein as hippo, actin, both, or partner (STRING expansion)."""
    key = preferred_name.upper()
    in_hippo = key in {s.upper() for s in seeds.hippo}
    in_actin = key in {s.upper() for s in seeds.actin}
    if in_hippo and in_actin:
        return "both"
    if in_hippo:
        return "hippo"
    if in_actin:
        return "actin"
    return "partner"


def normalize_score(raw: float) -> float:
    """STRING TSV scores are 0–1; required_score inputs are 0–1000."""
    if raw > 1.0:
        return raw / 1000.0
    return raw


def dedupe_undirected_edges(edges: Iterable[InteractionRecord]) -> list[InteractionRecord]:
    """Keep one undirected edge per protein pair; drop self-loops."""
    best: dict[tuple[str, str], InteractionRecord] = {}
    for edge in edges:
        if edge.source_string_id == edge.target_string_id:
            continue
        pair = tuple(sorted((edge.source_string_id, edge.target_string_id)))
        current = best.get(pair)
        if current is None or edge.combined_score > current.combined_score:
            if edge.source_string_id <= edge.target_string_id:
                canonical = edge
            else:
                canonical = InteractionRecord(
                    source_string_id=edge.target_string_id,
                    target_string_id=edge.source_string_id,
                    source_name=edge.target_name,
                    target_name=edge.source_name,
                    species_id=edge.species_id,
                    combined_score=edge.combined_score,
                )
            best[pair] = canonical
    return list(best.values())


def build_graph(
    proteins: list[ProteinRecord],
    edges: list[InteractionRecord],
) -> nx.Graph:
    """Build an undirected PPI graph; nodes without sequence are already excluded."""
    graph: nx.Graph = nx.Graph()
    for protein in proteins:
        graph.add_node(
            protein.string_id,
            preferred_name=protein.preferred_name,
            species_id=protein.species_id,
            species_name=protein.species_name,
            compartment=protein.compartment,
            uniprot_accession=protein.uniprot_accession,
            sequence_length=protein.sequence_length,
        )
    skipped = 0
    for edge in edges:
        if edge.source_string_id not in graph or edge.target_string_id not in graph:
            skipped += 1
            continue
        graph.add_edge(
            edge.source_string_id,
            edge.target_string_id,
            combined_score=edge.combined_score,
            species_id=edge.species_id,
        )
    if skipped:
        logger.warning("dropped %s edges incident to proteins without sequence", skipped)
    return graph


def write_fasta(path: Path, proteins: list[ProteinRecord], *, width: int = FASTA_LINE_WIDTH) -> None:
    """Write ESM-2-ready FASTA with STRING IDs as record names."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for protein in proteins:
            handle.write(
                f">{protein.string_id} {protein.preferred_name} "
                f"OX={protein.species_id} GN={protein.preferred_name} "
                f"compartment={protein.compartment} AC={protein.uniprot_accession}\n"
            )
            seq = protein.sequence
            for i in range(0, len(seq), width):
                handle.write(seq[i : i + width] + "\n")


def _file_stats(path: Path) -> dict[str, float | int | str]:
    size = path.stat().st_size
    return {"path": str(path), "bytes": size, "mb": round(size / (1024 * 1024), 4)}


class DataFetcher:
    """Orchestrate STRING + UniProt downloads for all configured species."""

    def __init__(
        self,
        string_client: StringClient | None = None,
        uniprot_client: UniProtClient | None = None,
    ) -> None:
        self.string = string_client or StringClient()
        self.uniprot = uniprot_client or UniProtClient()

    def fetch_species(
        self,
        seeds: SpeciesSeeds,
    ) -> tuple[list[ProteinRecord], list[InteractionRecord]]:
        """Resolve seeds, expand partners, induce the network, attach sequences."""
        mapped = self.string.get_string_ids(list(seeds.all_symbols()), seeds.taxon_id)
        if mapped.empty:
            raise RuntimeError(f"STRING resolved 0 seeds for {seeds.name}")

        seed_ids = mapped["stringId"].astype(str).tolist()
        id_to_name = dict(
            zip(mapped["stringId"].astype(str), mapped["preferredName"].astype(str), strict=True)
        )

        partners = self.string.interaction_partners(
            seed_ids,
            seeds.taxon_id,
            required_score=REQUIRED_SCORE,
            limit=MAX_PARTNERS_PER_SEED,
            network_type=NETWORK_TYPE,
        )
        protein_ids = set(seed_ids)
        if not partners.empty:
            protein_ids.update(partners["stringId_A"].astype(str))
            protein_ids.update(partners["stringId_B"].astype(str))
            for _, row in partners.iterrows():
                id_to_name[str(row["stringId_A"])] = str(row["preferredName_A"])
                id_to_name[str(row["stringId_B"])] = str(row["preferredName_B"])

        ordered_ids = sorted(protein_ids)
        logger.info("%s unique proteins after partner expansion: %s", seeds.name, len(ordered_ids))

        network = self.string.network(
            ordered_ids,
            seeds.taxon_id,
            required_score=REQUIRED_SCORE,
            network_type=NETWORK_TYPE,
        )
        if not network.empty:
            for _, row in network.iterrows():
                id_to_name[str(row["stringId_A"])] = str(row["preferredName_A"])
                id_to_name[str(row["stringId_B"])] = str(row["preferredName_B"])

        names = [id_to_name[pid] for pid in ordered_ids if pid in id_to_name]
        sequences = self.uniprot.fetch_sequences(names, seeds.uniprot_taxon_id)

        proteins = self._attach_sequences(ordered_ids, id_to_name, sequences, seeds)
        have_seq = {p.string_id for p in proteins}
        raw_edges = self._rows_to_edges(network, seeds.taxon_id)
        edges = [e for e in dedupe_undirected_edges(raw_edges) if e.source_string_id in have_seq and e.target_string_id in have_seq]
        logger.info(
            "%s retained proteins=%s edges=%s (dropped seq-less nodes=%s)",
            seeds.name,
            len(proteins),
            len(edges),
            len(ordered_ids) - len(proteins),
        )
        return proteins, edges

    def run(self) -> nx.Graph:
        """Fetch all species, write artifacts, and return the combined graph."""
        DATA_RAW.mkdir(parents=True, exist_ok=True)
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

        all_proteins: list[ProteinRecord] = []
        all_edges: list[InteractionRecord] = []
        per_species: dict[str, dict[str, int]] = {}

        for seeds in SPECIES:
            proteins, edges = self.fetch_species(seeds)
            all_proteins.extend(proteins)
            all_edges.extend(edges)
            per_species[seeds.name] = {"nodes": len(proteins), "edges": len(edges)}

        graph = build_graph(all_proteins, all_edges)
        self._write_outputs(all_proteins, all_edges, graph, per_species)
        return graph

    @staticmethod
    def _attach_sequences(
        string_ids: list[str],
        id_to_name: dict[str, str],
        sequences: dict[str, UniProtSequence],
        seeds: SpeciesSeeds,
    ) -> list[ProteinRecord]:
        records: list[ProteinRecord] = []
        for string_id in string_ids:
            name = id_to_name.get(string_id)
            if name is None:
                logger.warning("missing preferred name for %s", string_id)
                continue
            seq = sequences.get(name.upper())
            if seq is None:
                continue
            records.append(
                ProteinRecord(
                    string_id=string_id,
                    preferred_name=name,
                    species_id=seeds.taxon_id,
                    species_name=seeds.name,
                    compartment=label_compartment(name, seeds),
                    uniprot_accession=seq.accession,
                    sequence=seq.sequence,
                )
            )
        return records

    @staticmethod
    def _rows_to_edges(network: pd.DataFrame, species_id: int) -> list[InteractionRecord]:
        if network.empty:
            return []
        edges: list[InteractionRecord] = []
        for _, row in network.iterrows():
            edges.append(
                InteractionRecord(
                    source_string_id=str(row["stringId_A"]),
                    target_string_id=str(row["stringId_B"]),
                    source_name=str(row["preferredName_A"]),
                    target_name=str(row["preferredName_B"]),
                    species_id=species_id,
                    combined_score=normalize_score(float(row["score"])),
                )
            )
        return edges

    @staticmethod
    def _write_outputs(
        proteins: list[ProteinRecord],
        edges: list[InteractionRecord],
        graph: nx.Graph,
        per_species: dict[str, dict[str, int]],
    ) -> None:
        protein_rows = [
            {
                "string_id": p.string_id,
                "preferred_name": p.preferred_name,
                "species_id": p.species_id,
                "species_name": p.species_name,
                "compartment": p.compartment,
                "uniprot_accession": p.uniprot_accession,
                "sequence_length": p.sequence_length,
            }
            for p in proteins
        ]
        edge_rows = [
            {
                "source_string_id": e.source_string_id,
                "target_string_id": e.target_string_id,
                "source_name": e.source_name,
                "target_name": e.target_name,
                "species_id": e.species_id,
                "combined_score": e.combined_score,
            }
            for e in edges
        ]
        pd.DataFrame(protein_rows).to_csv(PROTEINS_CSV, index=False)
        pd.DataFrame(edge_rows).to_csv(INTERACTIONS_CSV, index=False)
        write_fasta(PROTEINS_FASTA, proteins)

        components = nx.number_connected_components(graph) if graph.number_of_nodes() else 0
        isolates = list(nx.isolates(graph))
        summary = {
            "species": per_species,
            "totals": {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "connected_components": components,
                "isolates": len(isolates),
            },
            "parameters": {
                "required_score": REQUIRED_SCORE,
                "max_partners_per_seed": MAX_PARTNERS_PER_SEED,
                "network_type": NETWORK_TYPE,
                "sequence_source": "UniProt REST (STRING has no sequence endpoint)",
            },
            "files": {
                "proteins_csv": _file_stats(PROTEINS_CSV),
                "interactions_csv": _file_stats(INTERACTIONS_CSV),
                "proteins_fasta": _file_stats(PROTEINS_FASTA),
            },
        }
        INGEST_SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        logger.info("wrote %s", PROTEINS_CSV)
        logger.info("wrote %s", INTERACTIONS_CSV)
        logger.info("wrote %s", PROTEINS_FASTA)
        logger.info("ingest summary: %s", json.dumps(summary, indent=2))


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def main() -> None:
    configure_logging()
    graph = DataFetcher().run()
    logger.info(
        "graph ready nodes=%s edges=%s",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )


if __name__ == "__main__":
    main()
