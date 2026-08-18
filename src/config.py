"""Pipeline constants: species, Hippo/actin seeds, STRING cutoffs, paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"

STRING_API_URL: str = "https://version-12-0.string-db.org/api"
STRING_CALLER_IDENTITY: str = "HippoActInteract"
UNIPROT_SEARCH_URL: str = "https://rest.uniprot.org/uniprotkb/search"

REQUIRED_SCORE: int = 700
MAX_PARTNERS_PER_SEED: int = 20
NETWORK_TYPE: str = "functional"
GENE_QUERY_BATCH_SIZE: int = 40

HTTP_TIMEOUT_SECONDS: float = 30.0
HTTP_MAX_ATTEMPTS: int = 5
HTTP_BACKOFF_SECONDS: float = 1.0
HTTP_REQUEST_PAUSE_SECONDS: float = 0.35

PROTEINS_CSV: Path = DATA_RAW / "proteins.csv"
INTERACTIONS_CSV: Path = DATA_RAW / "interactions.csv"
PROTEINS_FASTA: Path = DATA_RAW / "proteins.fasta"
INGEST_SUMMARY_JSON: Path = DATA_PROCESSED / "ingest_summary.json"

ESM_MODEL_NAME: str = "facebook/esm2_t12_35M_UR50D"
ESM_HIDDEN_SIZE: int = 480
ESM_MAX_RESIDUES: int = 1022
ESM_MAX_LENGTH: int = 1024
ESM_BATCH_SIZE: int = int(os.environ.get("ESM_BATCH_SIZE", "1"))
NODE_EMBEDDINGS_PT: Path = DATA_PROCESSED / "node_embeddings.pt"
EMBED_SUMMARY_JSON: Path = DATA_PROCESSED / "embed_summary.json"

GNN_HIDDEN_SIZE: int = 64
GNN_NUM_LAYERS: int = 2
GNN_DROPOUT: float = 0.2
LINK_TRAIN_RATIO: float = 0.8
LINK_VAL_RATIO: float = 0.1
LINK_TEST_RATIO: float = 0.1
NEGATIVE_RATIO: float = 1.0
LINK_BATCH_SIZE: int = 256
GNN_RANDOM_SEED: int = 42
GNN_ARCHITECTURE_JSON: Path = DATA_PROCESSED / "gnn_architecture.json"

TRAIN_EPOCHS: int = 80
TRAIN_LR: float = 1e-3
TRAIN_WEIGHT_DECAY: float = 1e-4
TRAIN_PATIENCE: int = 10
HIT_LIST_TOP_K: int = 50
GNN_CHECKPOINT_PT: Path = DATA_PROCESSED / "gnn_best.pt"
TRAIN_METRICS_JSON: Path = DATA_PROCESSED / "train_metrics.json"
HIT_LIST_CSV: Path = DATA_PROCESSED / "top_predicted_interactions.csv"

VIZ_TOP_PER_SPECIES: int = 15
SUBGRAPH_PNG: Path = DATA_PROCESSED / "top_predicted_subgraph.png"
SUBGRAPH_SVG: Path = DATA_PROCESSED / "top_predicted_subgraph.svg"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"
FIGURES_PNG: Path = FIGURES_DIR / "top_predicted_subgraph.png"
TSNE_PNG: Path = FIGURES_DIR / "figure1_tsne_embeddings.png"

HIPPO_LIKE_COMPARTMENTS: frozenset[str] = frozenset({"hippo", "both"})
ACTIN_LIKE_COMPARTMENTS: frozenset[str] = frozenset({"actin", "both"})

EVIDENCE_DIR: Path = DATA_RAW / "evidence"
BIOGRID_ORGANISM_ZIP_URL: str = (
    "https://downloads.thebiogrid.org/Download/BioGRID/Latest-Release/"
    "BIOGRID-ORGANISM-LATEST.tab3.zip"
)
BIOGRID_ZIP_PATH: Path = EVIDENCE_DIR / "BIOGRID-ORGANISM-LATEST.tab3.zip"
INTACT_PSIQUIC_URL: str = (
    "https://www.ebi.ac.uk/Tools/webservices/psicquic/intact/webservices/"
    "current/search/query"
)
INTACT_MITAB_PATH: Path = EVIDENCE_DIR / "intact_proteins.mitab25.txt"
STRING_CHANNELS_CSV: Path = EVIDENCE_DIR / "string_channels.csv"
UNIPROT_LOCATIONS_CSV: Path = EVIDENCE_DIR / "uniprot_locations.csv"
EVIDENCE_MANIFEST_JSON: Path = EVIDENCE_DIR / "release_manifest.json"
ATLAS_CSV: Path = DATA_PROCESSED / "hippo_actin_atlas.csv"
ATLAS_QC_JSON: Path = DATA_PROCESSED / "atlas_qc.json"
ATLAS_STATS_JSON: Path = DATA_PROCESSED / "atlas_stats.json"
PHYSICAL_EDGES_CSV: Path = DATA_PROCESSED / "physical_edges.csv"
EVIDENCE_FIGURE_PNG: Path = FIGURES_DIR / "figure4_evidence_classes.png"
STRING_CHANNEL_REQUIRED_SCORE: int = 150
STRING_FUNCTIONAL_CUTOFF: float = REQUIRED_SCORE / 1000.0
DOWNLOAD_TIMEOUT_SECONDS: float = 600.0
DEGREE_PRODUCT_QUANTILE: float = 0.90
PHASE2_PHYSICAL_FRACTION_STOP: float = 0.90
PHASE2_UNREPORTED_FRACTION_RUN: float = 0.15
PHASE2_DEGREE_SPEARMAN_RUN: float = 0.30

BENCHMARK_N_SEEDS: int = 20
BENCHMARK_EPOCHS: int = 20
BENCHMARK_PATIENCE: int = 5
BENCHMARK_METRICS_JSON: Path = DATA_PROCESSED / "benchmark_metrics.json"
BENCHMARK_STABILITY_CSV: Path = DATA_PROCESSED / "benchmark_stability.csv"
BENCHMARK_SPLITS_DIR: Path = DATA_PROCESSED / "benchmark_splits"
BENCHMARK_FIGURE_PNG: Path = FIGURES_DIR / "figure5_physical_benchmark.png"


@dataclass(frozen=True)
class SpeciesSeeds:
    """Seed gene symbols for one organism."""

    taxon_id: int
    uniprot_taxon_id: int
    name: str
    hippo: tuple[str, ...]
    actin: tuple[str, ...]

    def all_symbols(self) -> tuple[str, ...]:
        """Return Hippo then actin seeds, preserving order and uniqueness."""
        seen: set[str] = set()
        ordered: list[str] = []
        for symbol in (*self.hippo, *self.actin):
            key = symbol.upper()
            if key not in seen:
                seen.add(key)
                ordered.append(symbol)
        return tuple(ordered)


HUMAN: SpeciesSeeds = SpeciesSeeds(
    taxon_id=9606,
    uniprot_taxon_id=9606,
    name="Homo sapiens",
    hippo=("YAP1", "WWTR1", "LATS1", "LATS2", "STK3", "STK4", "SAV1", "MOB1A", "NF2", "TEAD1"),
    actin=("ACTB", "CFL1", "PFN1", "GSN", "DIAPH1", "ACTN1", "VCL", "RHOA", "CDC42", "RAC1"),
)

YEAST: SpeciesSeeds = SpeciesSeeds(
    taxon_id=4932,
    uniprot_taxon_id=559292,
    name="Saccharomyces cerevisiae",
    hippo=("CBK1", "KIC1", "MOB2", "TAO3", "SOG2", "HYM1"),
    actin=("ACT1", "COF1", "PFY1", "BNI1", "BNR1", "CDC42", "RHO1", "SAC6", "TPM1", "ABP1"),
)

SPECIES: tuple[SpeciesSeeds, ...] = (HUMAN, YEAST)
