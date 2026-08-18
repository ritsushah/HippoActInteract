"""Evidence atlas for same-species Hippo/RAM × actin pairs.

Downloads BioGRID and IntAct into ``data/raw/evidence/``, joins on UniProt
accession, overlays STRING channel scores and GraphSAGE ranks, and writes
``hippo_actin_atlas.csv`` plus a QC JSON. Unit tests use local fixtures and
never hit the network.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import sys
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import requests

from src.config import (
    ACTIN_LIKE_COMPARTMENTS,
    ATLAS_CSV,
    ATLAS_QC_JSON,
    ATLAS_STATS_JSON,
    BIOGRID_ORGANISM_ZIP_URL,
    BIOGRID_ZIP_PATH,
    DATA_PROCESSED,
    DEGREE_PRODUCT_QUANTILE,
    DOWNLOAD_TIMEOUT_SECONDS,
    EVIDENCE_DIR,
    EVIDENCE_FIGURE_PNG,
    EVIDENCE_MANIFEST_JSON,
    FIGURES_DIR,
    GNN_CHECKPOINT_PT,
    HIPPO_LIKE_COMPARTMENTS,
    HTTP_BACKOFF_SECONDS,
    HTTP_MAX_ATTEMPTS,
    HTTP_REQUEST_PAUSE_SECONDS,
    HTTP_TIMEOUT_SECONDS,
    INTACT_MITAB_PATH,
    INTACT_PSIQUIC_URL,
    INTERACTIONS_CSV,
    NODE_EMBEDDINGS_PT,
    PHASE2_DEGREE_SPEARMAN_RUN,
    PHASE2_PHYSICAL_FRACTION_STOP,
    PHASE2_UNREPORTED_FRACTION_RUN,
    PHYSICAL_EDGES_CSV,
    PROTEINS_CSV,
    STRING_CHANNEL_REQUIRED_SCORE,
    STRING_CHANNELS_CSV,
    STRING_FUNCTIONAL_CUTOFF,
    UNIPROT_LOCATIONS_CSV,
    UNIPROT_SEARCH_URL,
)
from src.data_fetcher import normalize_score
from src.http_util import APIError, header_map, request_with_retry
from src.string_client import StringClient

logger = logging.getLogger(__name__)

UNIPROT_ACCESSION_RE = re.compile(
    r"\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\b"
)
MITAB_UNIPROT_RE = re.compile(r"(?:uniprotkb|uniprot|uniprot\.isoform):([A-Z0-9\-]+)", re.I)
PMID_RE = re.compile(r"(?:pubmed:|PMID:)?(\d{4,9})", re.I)
MI_RE = re.compile(r"MI:\d{4}")

STRING_CHANNEL_NAMES: tuple[str, ...] = (
    "nscore",
    "fscore",
    "pscore",
    "ascore",
    "escore",
    "dscore",
    "tscore",
)
CHANNEL_LABELS: dict[str, str] = {
    "nscore": "neighborhood",
    "fscore": "fusion",
    "pscore": "cooccurrence",
    "ascore": "coexpression",
    "escore": "experiments",
    "dscore": "databases",
    "tscore": "textmining",
}
BIOGRID_HUMAN_TOKEN = "Homo_sapiens"
BIOGRID_YEAST_TOKEN = "Saccharomyces_cerevisiae"
YEAST_TAXON_IDS: frozenset[int] = frozenset({4932, 559292})
ATLAS_TAXON_IDS: frozenset[int] = frozenset({9606, 4932, 559292})

LOCATION_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("nucleus", ("nucleus", "nucleoplasm", "nucleolus", "nuclear speckle")),
    ("cytoplasm", ("cytoplasm", "cytosol", "cytoplasmic")),
    (
        "cortex",
        (
            "cell cortex",
            "plasma membrane",
            "cell membrane",
            "cell periphery",
            "bud neck",
            "site of polarized growth",
            "cell septum",
        ),
    ),
    ("cytoskeleton", ("cytoskeleton", "actin filament", "actin cytoskeleton", "microfilament")),
    ("extracellular", ("extracellular", "secreted", "cell wall", "external side")),
    ("mitochondrion", ("mitochondrion", "mitochondrial")),
)

EVIDENCE_CLASS_COLORS: dict[str, str] = {
    "physical_curated": "#2E5A88",
    "string_functional_only": "#7B68A6",
    "unreported": "#E07A5F",
    "artifact_risk": "#9AA0A6",
}

TIER1_BIOGRID_SYSTEMS: frozenset[str] = frozenset(
    {
        "Two-hybrid",
        "Two-hybrid Array",
        "Two-hybrid Pooling",
        "Reconstituted Complex",
        "PCA",
        "FRET",
        "BiFC",
        "Far Western",
        "Protein-peptide",
        "Co-crystal Structure",
        "Biochemical Activity",
    }
)
TIER1_MI: frozenset[str] = frozenset(
    {
        "MI:0018",
        "MI:0019",
        "MI:0397",
        "MI:0398",
        "MI:0399",
        "MI:0096",
        "MI:0412",
        "MI:0055",
        "MI:0809",
        "MI:0114",
        "MI:0107",
        "MI:0047",
        "MI:0407",
        "MI:0408",
    }
)


@dataclass(frozen=True)
class CuratedHit:
    """One database record supporting a UniProt pair."""

    accession_a: str
    accession_b: str
    source: str
    experimental_system: str
    experimental_system_type: str
    interaction_type: str
    detection_method: str
    pubmed: str
    throughput: str
    taxon_a: int | None = None
    taxon_b: int | None = None


@dataclass
class PairEvidence:
    """Aggregated BioGRID/IntAct support for one undirected UniProt pair."""

    biogrid_physical: bool = False
    biogrid_genetic: bool = False
    intact: bool = False
    assays: list[str] = field(default_factory=list)
    detection_methods: list[str] = field(default_factory=list)
    interaction_types: list[str] = field(default_factory=list)
    pubmeds: set[str] = field(default_factory=set)
    throughputs: list[str] = field(default_factory=list)

    def add(self, hit: CuratedHit) -> None:
        if hit.source == "biogrid":
            kind = hit.experimental_system_type.strip().lower()
            if kind == "physical":
                self.biogrid_physical = True
            elif kind == "genetic":
                self.biogrid_genetic = True
            if hit.experimental_system:
                self.assays.append(hit.experimental_system)
            if hit.throughput:
                self.throughputs.append(hit.throughput)
        elif hit.source == "intact":
            self.intact = True
            if hit.detection_method:
                self.detection_methods.append(hit.detection_method)
            if hit.interaction_type:
                self.interaction_types.append(hit.interaction_type)
        for pmid in extract_pmids(hit.pubmed):
            self.pubmeds.add(pmid)


def pair_key(left: str, right: str) -> tuple[str, str]:
    """Canonical undirected identifier pair."""
    return (left, right) if left <= right else (right, left)


def extract_uniprot_accessions(text: str) -> set[str]:
    """Pull UniProt accessions from a BioGRID or MITAB cell."""
    if not text or text.strip() in {"-", "NA", "na", "."}:
        return set()
    found: set[str] = set()
    for match in MITAB_UNIPROT_RE.finditer(text):
        acc = match.group(1).split("-")[0]
        if UNIPROT_ACCESSION_RE.fullmatch(acc):
            found.add(acc)
    for match in UNIPROT_ACCESSION_RE.finditer(text.replace("|", " ")):
        found.add(match.group(1))
    return found


def extract_pmids(text: str) -> set[str]:
    """Unique PubMed IDs from a publication cell."""
    if not text or text.strip() in {"-", "NA"}:
        return set()
    return {match.group(1) for match in PMID_RE.finditer(text)}


def extract_mi_ids(text: str) -> list[str]:
    """PSI-MI identifiers in appearance order."""
    return MI_RE.findall(text or "")


def _column_index(header: Sequence[str], *candidates: str) -> int:
    normalized = {name.strip().lstrip("#").lower(): i for i, name in enumerate(header)}
    for candidate in candidates:
        idx = normalized.get(candidate.lower())
        if idx is not None:
            return idx
    raise KeyError(f"none of {candidates} found in BioGRID header")


def parse_biogrid_tab3(path: Path, wanted: set[str]) -> list[CuratedHit]:
    """Keep BioGRID rows whose Swiss-Prot/TrEMBL IDs both sit in ``wanted``."""
    hits: list[CuratedHit] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        header_line = handle.readline()
        if not header_line:
            return hits
        header = header_line.rstrip("\n").split("\t")
        idx_swiss_a = _column_index(header, "SWISS-PROT Accessions Interactor A")
        idx_swiss_b = _column_index(header, "SWISS-PROT Accessions Interactor B")
        try:
            idx_trembl_a = _column_index(header, "TREMBL Accessions Interactor A")
            idx_trembl_b = _column_index(header, "TREMBL Accessions Interactor B")
        except KeyError:
            idx_trembl_a = idx_trembl_b = None
        idx_system = _column_index(header, "Experimental System")
        idx_type = _column_index(header, "Experimental System Type")
        idx_pub = _column_index(header, "Publication Source")
        try:
            idx_throughput = _column_index(header, "Throughput")
        except KeyError:
            idx_throughput = None
        try:
            idx_tax_a = _column_index(header, "Organism ID Interactor A")
            idx_tax_b = _column_index(header, "Organism ID Interactor B")
        except KeyError:
            idx_tax_a = idx_tax_b = None

        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(idx_swiss_a, idx_swiss_b, idx_system, idx_type, idx_pub):
                continue
            accs_a = extract_uniprot_accessions(parts[idx_swiss_a])
            accs_b = extract_uniprot_accessions(parts[idx_swiss_b])
            if idx_trembl_a is not None and len(parts) > idx_trembl_a:
                accs_a |= extract_uniprot_accessions(parts[idx_trembl_a])
            if idx_trembl_b is not None and len(parts) > idx_trembl_b:
                accs_b |= extract_uniprot_accessions(parts[idx_trembl_b])
            accs_a &= wanted
            accs_b &= wanted
            if not accs_a or not accs_b:
                continue
            taxon_a = _optional_int(parts[idx_tax_a]) if idx_tax_a is not None and len(parts) > idx_tax_a else None
            taxon_b = _optional_int(parts[idx_tax_b]) if idx_tax_b is not None and len(parts) > idx_tax_b else None
            pubmed = parts[idx_pub] if len(parts) > idx_pub else ""
            throughput = parts[idx_throughput] if idx_throughput is not None and len(parts) > idx_throughput else ""
            for acc_a in accs_a:
                for acc_b in accs_b:
                    if acc_a == acc_b:
                        continue
                    left, right = pair_key(acc_a, acc_b)
                    hits.append(
                        CuratedHit(
                            accession_a=left,
                            accession_b=right,
                            source="biogrid",
                            experimental_system=parts[idx_system],
                            experimental_system_type=parts[idx_type],
                            interaction_type="",
                            detection_method="",
                            pubmed=pubmed,
                            throughput=throughput,
                            taxon_a=taxon_a,
                            taxon_b=taxon_b,
                        )
                    )
    logger.info("BioGRID %s kept %s intra-universe records", path.name, len(hits))
    return hits


def parse_intact_mitab(path: Path, wanted: set[str]) -> list[CuratedHit]:
    """Parse PSI-MI TAB 2.5/2.7 rows; keep pairs inside ``wanted``."""
    hits: list[CuratedHit] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 12:
                continue
            ids_a = extract_uniprot_accessions(" ".join(parts[0:4:2])) | extract_uniprot_accessions(parts[0])
            ids_b = extract_uniprot_accessions(" ".join(parts[1:4:2])) | extract_uniprot_accessions(parts[1])
            if len(parts) > 3:
                ids_a |= extract_uniprot_accessions(parts[2])
                ids_b |= extract_uniprot_accessions(parts[3])
            ids_a &= wanted
            ids_b &= wanted
            if not ids_a or not ids_b:
                continue
            taxon_a = _parse_taxid(parts[9] if len(parts) > 9 else "")
            taxon_b = _parse_taxid(parts[10] if len(parts) > 10 else "")
            if taxon_a is not None and taxon_a not in ATLAS_TAXON_IDS:
                continue
            if taxon_b is not None and taxon_b not in ATLAS_TAXON_IDS:
                continue
            detection = parts[6] if len(parts) > 6 else ""
            pubmed = parts[8] if len(parts) > 8 else ""
            itype = parts[11] if len(parts) > 11 else ""
            for acc_a in ids_a:
                for acc_b in ids_b:
                    if acc_a == acc_b:
                        continue
                    left, right = pair_key(acc_a, acc_b)
                    hits.append(
                        CuratedHit(
                            accession_a=left,
                            accession_b=right,
                            source="intact",
                            experimental_system="",
                            experimental_system_type="",
                            interaction_type=itype,
                            detection_method=detection,
                            pubmed=pubmed,
                            throughput="",
                            taxon_a=taxon_a,
                            taxon_b=taxon_b,
                        )
                    )
    logger.info("IntAct %s kept %s intra-universe records", path.name, len(hits))
    return hits


def aggregate_hits(hits: Iterable[CuratedHit]) -> dict[tuple[str, str], PairEvidence]:
    """Collapse curated records onto undirected UniProt pairs."""
    bundled: dict[tuple[str, str], PairEvidence] = defaultdict(PairEvidence)
    for hit in hits:
        bundled[pair_key(hit.accession_a, hit.accession_b)].add(hit)
    return dict(bundled)


def is_tier1(evidence: PairEvidence) -> bool:
    """Binary-ish assays (Y2H, reconstituted complex, FRET, crystals) vs AP-MS co-complex."""
    for assay in evidence.assays:
        if assay.strip() in TIER1_BIOGRID_SYSTEMS:
            return True
    blob = " ".join(evidence.detection_methods + evidence.interaction_types)
    return any(mi in TIER1_MI for mi in extract_mi_ids(blob))


def physical_edges_from_curated(
    proteins: pd.DataFrame,
    curated: dict[tuple[str, str], PairEvidence],
) -> pd.DataFrame:
    """Same-species physical BioGRID/IntAct edges whose both ends are in proteins.csv."""
    acc_to_ids: dict[str, list[pd.Series]] = defaultdict(list)
    for row in proteins.itertuples(index=False):
        acc_to_ids[str(row.uniprot_accession)].append(row)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for (acc_a, acc_b), evidence in curated.items():
        if not (evidence.biogrid_physical or evidence.intact):
            continue
        for left in acc_to_ids.get(acc_a, []):
            for right in acc_to_ids.get(acc_b, []):
                if int(left.species_id) != int(right.species_id):
                    continue
                sid_key = pair_key(str(left.string_id), str(right.string_id))
                if sid_key in seen:
                    continue
                seen.add(sid_key)
                rows.append(
                    {
                        "string_id_a": sid_key[0],
                        "string_id_b": sid_key[1],
                        "uniprot_a": acc_a if acc_a <= acc_b else acc_b,
                        "uniprot_b": acc_b if acc_a <= acc_b else acc_a,
                        "species_id": int(left.species_id),
                        "species": str(left.species_name),
                        "tier1": is_tier1(evidence),
                        "biogrid_physical": evidence.biogrid_physical,
                        "intact": evidence.intact,
                        "assays": _unique_join(evidence.assays),
                        "pubmed_count": len(evidence.pubmeds),
                    }
                )
    return pd.DataFrame(rows)


def parse_string_channel_frame(frame: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    """Map STRING stringId pairs to combined + channel scores in 0–1 units."""
    if frame.empty:
        return {}
    id_a_col = _first_present(frame, "stringId_A", "stringId_a")
    id_b_col = _first_present(frame, "stringId_B", "stringId_b")
    score_col = _first_present(frame, "score", "combined_score")
    out: dict[tuple[str, str], dict[str, float]] = {}
    for row in frame.itertuples(index=False):
        mapping = row._asdict() if hasattr(row, "_asdict") else dict(zip(frame.columns, row, strict=True))
        left = str(mapping[id_a_col])
        right = str(mapping[id_b_col])
        if left == right:
            continue
        key = pair_key(left, right)
        scores = {"combined_score": normalize_score(float(mapping[score_col]))}
        for channel in STRING_CHANNEL_NAMES:
            if channel in mapping and pd.notna(mapping[channel]):
                scores[channel] = normalize_score(float(mapping[channel]))
            else:
                scores[channel] = 0.0
        current = out.get(key)
        if current is None or scores["combined_score"] > current["combined_score"]:
            out[key] = scores
    return out


def dominant_channel(scores: dict[str, float]) -> str:
    """Highest STRING evidence channel; empty if all channels are zero."""
    ranked = [(scores.get(name, 0.0), name) for name in STRING_CHANNEL_NAMES]
    value, name = max(ranked)
    if value <= 0.0:
        return ""
    return CHANNEL_LABELS[name]


def location_buckets(location_text: str, go_cc: str) -> set[str]:
    """Map UniProt location / GO CC text onto coarse compartments."""
    blob = f"{location_text} {go_cc}".lower()
    found: set[str] = set()
    for bucket, keywords in LOCATION_BUCKETS:
        if any(keyword in blob for keyword in keywords):
            found.add(bucket)
    return found


def localization_overlap(buckets_a: set[str], buckets_b: set[str]) -> str:
    """compatible / unclear / conflicting. Nuclear vs cortical is unclear, not conflict."""
    if not buckets_a or not buckets_b:
        return "unclear"
    if buckets_a & buckets_b:
        return "compatible"
    exclusive = ({"extracellular"}, {"nucleus"}, {"mitochondrion"})
    for left in exclusive:
        for right in exclusive:
            if left == right:
                continue
            if buckets_a <= left and buckets_b <= right:
                return "conflicting"
            if buckets_b <= left and buckets_a <= right:
                return "conflicting"
    return "unclear"


def classify_pair(
    *,
    biogrid_physical: bool,
    intact: bool,
    string_combined: float,
    dominant: str,
    localization: str,
    degree_product: float,
    degree_product_cutoff: float,
    biogrid_genetic: bool = False,
) -> str:
    """Assign one exclusive literature class. Physical curated wins over hubs."""
    physical = biogrid_physical or intact
    textmining_only = (
        string_combined >= STRING_FUNCTIONAL_CUTOFF
        and dominant == "textmining"
        and not physical
    )
    hub_risk = degree_product >= degree_product_cutoff and not physical
    loc_risk = localization == "conflicting" and not physical
    if physical:
        return "physical_curated"
    if loc_risk or textmining_only or hub_risk:
        return "artifact_risk"
    if string_combined >= STRING_FUNCTIONAL_CUTOFF:
        return "string_functional_only"
    _ = biogrid_genetic
    return "unreported"


def hippo_actin_protein_frame(proteins: pd.DataFrame) -> pd.DataFrame:
    """Rows whose compartment is Hippo-like or actin-like."""
    mask = proteins["compartment"].astype(str).isin(
        HIPPO_LIKE_COMPARTMENTS | ACTIN_LIKE_COMPARTMENTS
    )
    return proteins.loc[mask].copy()


def enumerate_atlas_pairs(proteins: pd.DataFrame) -> list[dict[str, Any]]:
    """Every same-species Hippo × actin pair in the current graph."""
    rows: list[dict[str, Any]] = []
    hippo_actin = hippo_actin_protein_frame(proteins)
    for species_id, group in hippo_actin.groupby("species_id"):
        records = list(group.itertuples(index=False))
        hippo = [r for r in records if str(r.compartment) in HIPPO_LIKE_COMPARTMENTS]
        actin = [r for r in records if str(r.compartment) in ACTIN_LIKE_COMPARTMENTS]
        seen: set[tuple[str, str]] = set()
        for left in hippo:
            for right in actin:
                if left.string_id == right.string_id:
                    continue
                key = pair_key(str(left.string_id), str(right.string_id))
                if key in seen:
                    continue
                seen.add(key)
                if str(left.string_id) <= str(right.string_id):
                    a, b = left, right
                else:
                    a, b = right, left
                rows.append(
                    {
                        "species_id": int(species_id),
                        "species": str(a.species_name),
                        "protein_a": str(a.preferred_name),
                        "protein_b": str(b.preferred_name),
                        "string_id_a": str(a.string_id),
                        "string_id_b": str(b.string_id),
                        "uniprot_a": str(a.uniprot_accession),
                        "uniprot_b": str(b.uniprot_accession),
                        "compartment_a": str(a.compartment),
                        "compartment_b": str(b.compartment),
                    }
                )
    rows.sort(key=lambda r: (r["species"], r["protein_a"], r["protein_b"]))
    return rows


def string_degree_and_neighbors(
    proteins: pd.DataFrame,
    interactions: pd.DataFrame,
) -> tuple[dict[str, int], dict[str, set[str]]]:
    """Undirected degree and neighbor sets on the STRING ≥ 700 graph."""
    graph = nx.Graph()
    graph.add_nodes_from(proteins["string_id"].astype(str))
    for row in interactions.itertuples(index=False):
        src = str(row.source_string_id)
        dst = str(row.target_string_id)
        if src == dst:
            continue
        if src in graph and dst in graph:
            graph.add_edge(src, dst)
    degrees = dict(graph.degree())
    neighbors = {node: set(graph.neighbors(node)) for node in graph.nodes}
    return degrees, neighbors


def build_atlas_table(
    proteins: pd.DataFrame,
    interactions: pd.DataFrame,
    curated: dict[tuple[str, str], PairEvidence],
    string_channels: dict[tuple[str, str], dict[str, float]],
    locations: pd.DataFrame,
    graphsage: pd.DataFrame | None = None,
    *,
    degree_quantile: float = DEGREE_PRODUCT_QUANTILE,
) -> pd.DataFrame:
    """One row per Hippo × actin pair with provenance and class."""
    pairs = enumerate_atlas_pairs(proteins)
    degrees, neighbors = string_degree_and_neighbors(proteins, interactions)
    loc_map = _location_map(locations)
    known_string = {
        pair_key(str(row.source_string_id), str(row.target_string_id))
        for row in interactions.itertuples(index=False)
    }
    score_map = _graphsage_map(graphsage) if graphsage is not None else {}

    draft: list[dict[str, Any]] = []
    for pair in pairs:
        sid_key = pair_key(pair["string_id_a"], pair["string_id_b"])
        acc_key = pair_key(pair["uniprot_a"], pair["uniprot_b"])
        evidence = curated.get(acc_key, PairEvidence())
        channels = string_channels.get(sid_key, {})
        combined = float(channels.get("combined_score", 0.0))
        if sid_key in known_string and combined < STRING_FUNCTIONAL_CUTOFF:
            combined = STRING_FUNCTIONAL_CUTOFF
        deg_a = int(degrees.get(pair["string_id_a"], 0))
        deg_b = int(degrees.get(pair["string_id_b"], 0))
        shared = neighbors.get(pair["string_id_a"], set()) & neighbors.get(pair["string_id_b"], set())
        buckets_a = loc_map.get(pair["uniprot_a"], set())
        buckets_b = loc_map.get(pair["uniprot_b"], set())
        loc = localization_overlap(buckets_a, buckets_b)
        gnn = score_map.get(sid_key, {})
        assays = _unique_join(evidence.assays)
        methods = _unique_join(evidence.detection_methods)
        types = _unique_join(evidence.interaction_types)
        draft.append(
            {
                **pair,
                "degree_a": deg_a,
                "degree_b": deg_b,
                "degree_product": deg_a * deg_b,
                "shared_neighbors": len(shared),
                "string_combined": combined,
                "string_in_graph": sid_key in known_string or combined >= STRING_FUNCTIONAL_CUTOFF,
                "nscore": float(channels.get("nscore", 0.0)),
                "fscore": float(channels.get("fscore", 0.0)),
                "pscore": float(channels.get("pscore", 0.0)),
                "ascore": float(channels.get("ascore", 0.0)),
                "escore": float(channels.get("escore", 0.0)),
                "dscore": float(channels.get("dscore", 0.0)),
                "tscore": float(channels.get("tscore", 0.0)),
                "dominant_channel": dominant_channel(channels) if channels else "",
                "biogrid_physical": evidence.biogrid_physical,
                "biogrid_genetic": evidence.biogrid_genetic,
                "intact": evidence.intact,
                "genetic_only": evidence.biogrid_genetic
                and not evidence.biogrid_physical
                and not evidence.intact,
                "assays": assays,
                "intact_methods": methods,
                "intact_types": types,
                "pubmed_count": len(evidence.pubmeds),
                "pubmeds": ";".join(sorted(evidence.pubmeds, key=int) if _all_int(evidence.pubmeds) else sorted(evidence.pubmeds)),
                "throughput": _unique_join(evidence.throughputs),
                "location_a": ";".join(sorted(buckets_a)),
                "location_b": ";".join(sorted(buckets_b)),
                "localization_overlap": loc,
                "graphsage_probability": gnn.get("probability"),
                "graphsage_rank": gnn.get("rank"),
            }
        )

    products = [row["degree_product"] for row in draft] or [0]
    cutoff = float(pd.Series(products).quantile(degree_quantile))
    for row in draft:
        row["degree_product_cutoff"] = cutoff
        row["evidence_class"] = classify_pair(
            biogrid_physical=bool(row["biogrid_physical"]),
            intact=bool(row["intact"]),
            string_combined=float(row["string_combined"]),
            dominant=str(row["dominant_channel"]),
            localization=str(row["localization_overlap"]),
            degree_product=float(row["degree_product"]),
            degree_product_cutoff=cutoff,
            biogrid_genetic=bool(row["biogrid_genetic"]),
        )
        row["string_absent"] = not bool(row["string_in_graph"])
    table = pd.DataFrame(draft)
    if table.empty:
        return table
    absent = table["string_absent"].astype(bool)
    if absent.any() and table.loc[absent, "graphsage_probability"].notna().any():
        ranked = table.loc[absent].sort_values(
            "graphsage_probability", ascending=False, kind="mergesort"
        )
        rank_map = {idx: rank for rank, idx in enumerate(ranked.index, start=1)}
        table["graphsage_rank_among_absent"] = table.index.map(lambda i: rank_map.get(i))
    else:
        table["graphsage_rank_among_absent"] = pd.NA
    return table.reset_index(drop=True)


def atlas_stats(table: pd.DataFrame) -> dict[str, Any]:
    """Missingness, species mix, and score-vs-topology correlations from the atlas."""
    if table.empty:
        return {"n_pairs": 0}
    absent = table[table["string_absent"].astype(bool)].copy()
    counts = table["evidence_class"].value_counts().to_dict()
    absent_counts = absent["evidence_class"].value_counts().to_dict() if not absent.empty else {}
    by_species: dict[str, Any] = {}
    for species, group in table.groupby("species"):
        g_absent = group[group["string_absent"].astype(bool)]
        by_species[str(species)] = {
            "n_pairs": int(len(group)),
            "n_string_absent": int(len(g_absent)),
            "classes": group["evidence_class"].value_counts().to_dict(),
            "absent_classes": g_absent["evidence_class"].value_counts().to_dict() if len(g_absent) else {},
            "n_physical": int(
                ((group["biogrid_physical"].astype(bool)) | (group["intact"].astype(bool))).sum()
            ),
            "n_genetic_only": int(group["genetic_only"].astype(bool).sum()),
        }
    scored_absent = absent[absent["graphsage_probability"].notna()]
    correlations: dict[str, float | None] = {
        "spearman_prob_vs_degree_product": None,
        "spearman_prob_vs_shared_neighbors": None,
        "n_scored_absent": int(len(scored_absent)),
    }
    if len(scored_absent) >= 5:
        correlations["spearman_prob_vs_degree_product"] = _spearman(
            scored_absent["graphsage_probability"], scored_absent["degree_product"]
        )
        correlations["spearman_prob_vs_shared_neighbors"] = _spearman(
            scored_absent["graphsage_probability"], scored_absent["shared_neighbors"]
        )
    n_absent = int(len(absent))
    n_physical_absent = int(
        ((absent["biogrid_physical"].astype(bool)) | (absent["intact"].astype(bool))).sum()
    ) if n_absent else 0
    n_unreported_absent = int((absent["evidence_class"] == "unreported").sum()) if n_absent else 0
    frac_physical_absent = (n_physical_absent / n_absent) if n_absent else 0.0
    frac_unreported_absent = (n_unreported_absent / n_absent) if n_absent else 0.0
    spearman_deg = correlations["spearman_prob_vs_degree_product"]
    run_benchmark, gate_reason = phase2_gate(
        n_absent=n_absent,
        frac_physical_absent=frac_physical_absent,
        frac_unreported_absent=frac_unreported_absent,
        spearman_degree=spearman_deg,
    )
    top_absent = (
        scored_absent.sort_values("graphsage_probability", ascending=False, kind="mergesort")
        .head(10)
        .to_dict(orient="records")
        if not scored_absent.empty
        else []
    )
    return {
        "n_pairs": int(len(table)),
        "n_string_absent": n_absent,
        "n_string_present": int((~table["string_absent"].astype(bool)).sum()),
        "classes": {k: int(v) for k, v in counts.items()},
        "absent_classes": {k: int(v) for k, v in absent_counts.items()},
        "n_physical_curated": int((table["evidence_class"] == "physical_curated").sum()),
        "n_physical_among_absent": n_physical_absent,
        "frac_physical_among_absent": frac_physical_absent,
        "frac_unreported_among_absent": frac_unreported_absent,
        "n_genetic_only": int(table["genetic_only"].astype(bool).sum()),
        "by_species": by_species,
        "correlations": correlations,
        "phase2_gate": {
            "run_benchmark": run_benchmark,
            "reason": gate_reason,
            "n_string_absent": n_absent,
            "frac_physical_among_absent": frac_physical_absent,
            "frac_unreported_among_absent": frac_unreported_absent,
            "spearman_prob_vs_degree_product": spearman_deg,
        },
        "top_absent_with_evidence": _compact_top(top_absent),
    }


def phase2_gate(
    *,
    n_absent: int,
    frac_physical_absent: float,
    frac_unreported_absent: float,
    spearman_degree: float | None,
) -> tuple[bool, str]:
    """Stop at a missingness resource if almost every STRING-absent hit is already curated."""
    if n_absent == 0:
        return False, "no STRING-absent Hippo×actin pairs"
    if frac_physical_absent >= PHASE2_PHYSICAL_FRACTION_STOP:
        return (
            False,
            "≥90% of STRING-absent pairs already have BioGRID/IntAct physical evidence; "
            "write a database-missingness resource paper",
        )
    abs_spearman = abs(spearman_degree) if spearman_degree is not None else 0.0
    if frac_unreported_absent >= PHASE2_UNREPORTED_FRACTION_RUN:
        return True, "substantial unreported set remains among STRING-absent pairs"
    if abs_spearman >= PHASE2_DEGREE_SPEARMAN_RUN:
        return True, "GraphSAGE scores track STRING degree product more than curated evidence"
    return True, "disagreement between STRING, GraphSAGE, and curated physical catalogs"


def plot_evidence_figure(table: pd.DataFrame, path: Path) -> None:
    """Stacked evidence-class bars plus GraphSAGE score vs degree product."""
    path.parent.mkdir(parents=True, exist_ok=True)
    classes = list(EVIDENCE_CLASS_COLORS)
    species_order = sorted(table["species"].astype(str).unique())
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)

    bottoms = [0.0] * len(species_order)
    x = list(range(len(species_order)))
    for cls in classes:
        heights = [
            float(((table["species"] == sp) & (table["evidence_class"] == cls)).sum())
            for sp in species_order
        ]
        axes[0].bar(
            x,
            heights,
            bottom=bottoms,
            color=EVIDENCE_CLASS_COLORS[cls],
            label=cls.replace("_", " "),
            width=0.55,
        )
        bottoms = [b + h for b, h in zip(bottoms, heights, strict=True)]
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([_short_species(s) for s in species_order])
    axes[0].set_ylabel("Hippo × actin pairs")
    axes[0].set_title("Evidence class")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    scored = table[table["graphsage_probability"].notna()]
    if scored.empty:
        axes[1].text(0.5, 0.5, "no GraphSAGE scores", ha="center", va="center")
    else:
        for cls, color in EVIDENCE_CLASS_COLORS.items():
            subset = scored[scored["evidence_class"] == cls]
            if subset.empty:
                continue
            marker = "o" if cls != "artifact_risk" else "x"
            axes[1].scatter(
                subset["degree_product"].astype(float).clip(lower=1),
                subset["graphsage_probability"].astype(float),
                c=color,
                label=cls.replace("_", " "),
                alpha=0.85,
                s=28,
                marker=marker,
                linewidths=0.8,
            )
        axes[1].set_xscale("log")
        axes[1].set_xlabel("STRING degree product")
        axes[1].set_ylabel("GraphSAGE probability")
        axes[1].set_title("Score vs hub-ness")
        axes[1].legend(frameon=False, fontsize=8)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    logger.info("wrote %s", path)


def stream_download(url: str, dest: Path, *, timeout: float = DOWNLOAD_TIMEOUT_SECONDS) -> None:
    """Download ``url`` to ``dest`` without loading the whole body into memory."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    session = requests.Session()
    session.headers.update(header_map())
    logger.info("GET %s -> %s", url, dest.name)
    with session.get(url, stream=True, timeout=timeout) as response:
        if response.status_code >= 400:
            snippet = response.text[:200].replace("\n", " ") if response.text else ""
            raise APIError(f"HTTP {response.status_code} for {url}: {snippet}")
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp.replace(dest)


def extract_biogrid_organisms(zip_path: Path, dest_dir: Path) -> dict[str, Path]:
    """Extract human and yeast TAB3 files from the BioGRID organism zip."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted: dict[str, Path] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            base = Path(name).name
            if BIOGRID_HUMAN_TOKEN in base and base.endswith(".tab3.txt"):
                target = dest_dir / base
                target.write_bytes(archive.read(name))
                extracted["human"] = target
            elif BIOGRID_YEAST_TOKEN in base and base.endswith(".tab3.txt"):
                target = dest_dir / base
                target.write_bytes(archive.read(name))
                extracted["yeast"] = target
    if "human" not in extracted or "yeast" not in extracted:
        raise FileNotFoundError(
            f"BioGRID zip {zip_path} did not contain human and yeast TAB3 files; "
            f"found {list(extracted)}"
        )
    return extracted


def download_intact_psicquic(accessions: Sequence[str], dest: Path, *, batch_size: int = 20) -> Path:
    """Query IntAct PSIQUIC for UniProt IDs and cache MITAB 2.5 locally."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    unique = _unique(accessions)
    session = requests.Session()
    session.headers.update(header_map())
    lines: list[str] = []
    for start in range(0, len(unique), batch_size):
        batch = unique[start : start + batch_size]
        query = " OR ".join(f"id:{acc}" for acc in batch)
        first = 0
        page_size = 500
        while True:
            url = f"{INTACT_PSIQUIC_URL}/{quote(query)}"
            logger.info("IntAct PSIQUIC batch=%s first=%s", len(batch), first)
            response = request_with_retry(
                session,
                "GET",
                url,
                timeout=HTTP_TIMEOUT_SECONDS,
                max_attempts=HTTP_MAX_ATTEMPTS,
                backoff_seconds=HTTP_BACKOFF_SECONDS,
                params={"format": "tab25", "firstResult": first, "maxResults": page_size},
            )
            text = response.text.strip()
            if not text:
                break
            page = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
            lines.extend(page)
            if len(page) < page_size:
                break
            first += page_size
        if HTTP_REQUEST_PAUSE_SECONDS > 0:
            import time

            time.sleep(HTTP_REQUEST_PAUSE_SECONDS)
    dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    logger.info("wrote %s (%s rows)", dest, len(lines))
    return dest


def fetch_string_channels(proteins: pd.DataFrame, dest: Path) -> pd.DataFrame:
    """STRING v12 channel scores among Hippo/actin proteins (not the whole proteome)."""
    client = StringClient()
    frames: list[pd.DataFrame] = []
    hippo_actin = hippo_actin_protein_frame(proteins)
    for species_id, group in hippo_actin.groupby("species_id"):
        ids = group["string_id"].astype(str).tolist()
        if len(ids) < 2:
            continue
        frame = client.network(
            ids,
            int(species_id),
            required_score=STRING_CHANNEL_REQUIRED_SCORE,
            network_type="functional",
        )
        if not frame.empty:
            frames.append(frame)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    dest.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(dest, index=False)
    logger.info("wrote %s rows=%s", dest, len(combined))
    return combined


def fetch_uniprot_locations(accessions: Sequence[str], dest: Path, *, batch_size: int = 50) -> pd.DataFrame:
    """UniProt subcellular location and GO cellular component for atlas proteins."""
    unique = _unique(accessions)
    session = requests.Session()
    session.headers.update(header_map({"Accept": "text/plain"}))
    chunks: list[pd.DataFrame] = []
    import time

    for start in range(0, len(unique), batch_size):
        batch = unique[start : start + batch_size]
        query = " OR ".join(f"accession:{acc}" for acc in batch)
        response = request_with_retry(
            session,
            "GET",
            UNIPROT_SEARCH_URL,
            timeout=HTTP_TIMEOUT_SECONDS,
            max_attempts=HTTP_MAX_ATTEMPTS,
            backoff_seconds=HTTP_BACKOFF_SECONDS,
            params={
                "query": query,
                "fields": "accession,cc_subcellular_location,go_c",
                "format": "tsv",
                "size": "500",
            },
        )
        text = response.text.strip()
        if text:
            chunks.append(pd.read_csv(io.StringIO(text), sep="\t"))
        if HTTP_REQUEST_PAUSE_SECONDS > 0:
            time.sleep(HTTP_REQUEST_PAUSE_SECONDS)
    table = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
        columns=["Entry", "Subcellular location [CC]", "Gene Ontology (cellular component)"]
    )
    table = table.rename(
        columns={
            "Entry": "uniprot_accession",
            "Subcellular location [CC]": "subcellular_location",
            "Gene Ontology (cellular component)": "go_cc",
        }
    )
    if "uniprot_accession" not in table.columns and "Entry" not in table.columns:
        if not table.empty:
            table.columns = ["uniprot_accession", "subcellular_location", "go_cc"][: len(table.columns)]
    dest.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(dest, index=False)
    logger.info("wrote %s rows=%s", dest, len(table))
    return table


def overlay_graphsage_from_checkpoint(
    proteins: pd.DataFrame,
    interactions: pd.DataFrame,
    *,
    checkpoint: Path = GNN_CHECKPOINT_PT,
    embeddings: Path = NODE_EMBEDDINGS_PT,
) -> pd.DataFrame:
    """Score every Hippo × actin pair with the saved GraphSAGE checkpoint."""
    import torch

    from src.device import get_device
    from src.gnn_model import LinkPredictor
    from src.graph_data import load_link_graph, to_bidirectional
    from src.train import score_pairs

    if not checkpoint.is_file() or not embeddings.is_file():
        logger.warning("skip GraphSAGE overlay; missing %s or %s", checkpoint, embeddings)
        return pd.DataFrame()

    device = get_device()
    graph = load_link_graph(proteins_csv=PROTEINS_CSV, interactions_csv=INTERACTIONS_CSV, embeddings_pt=embeddings)
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    model = LinkPredictor(in_channels=int(payload["in_channels"]), hidden_channels=int(payload["hidden_channels"]))
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()

    atlas_rows = enumerate_atlas_pairs(proteins)
    id_to_index = {pid: i for i, pid in enumerate(graph.protein_ids)}
    pairs: list[tuple[int, int]] = []
    keys: list[tuple[str, str]] = []
    for row in atlas_rows:
        src = id_to_index[row["string_id_a"]]
        dst = id_to_index[row["string_id_b"]]
        pairs.append((src, dst) if src < dst else (dst, src))
        keys.append(pair_key(row["string_id_a"], row["string_id_b"]))

    x = graph.data.x.to(device)
    full_pos = torch.cat([graph.train_pos, graph.val_pos, graph.test_pos], dim=1)
    full_edges = to_bidirectional(full_pos).to(device)
    with torch.inference_mode():
        z = model.encode(x, full_edges)
        probs = score_pairs(model, z, pairs).cpu().numpy()

    table = pd.DataFrame(
        {
            "string_id_a": [k[0] for k in keys],
            "string_id_b": [k[1] for k in keys],
            "probability": probs,
        }
    )
    table = table.sort_values("probability", ascending=False, kind="mergesort").reset_index(drop=True)
    table.insert(0, "rank", range(1, len(table) + 1))
    _ = interactions
    return table


def ensure_evidence_files(proteins: pd.DataFrame, *, skip_download: bool = False) -> dict[str, Any]:
    """Download BioGRID/IntAct/STRING/UniProt evidence unless cached files exist."""
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    wanted = set(proteins["uniprot_accession"].astype(str))
    hippo_actin_acc = set(hippo_actin_protein_frame(proteins)["uniprot_accession"].astype(str))
    manifest: dict[str, Any] = {"skip_download": skip_download}

    biogrid_paths: list[Path] = sorted(EVIDENCE_DIR.glob("BIOGRID-ORGANISM-*.tab3.txt"))
    if not biogrid_paths:
        if skip_download:
            raise FileNotFoundError("no BioGRID TAB3 files and skip_download=True")
        if not BIOGRID_ZIP_PATH.is_file():
            stream_download(BIOGRID_ORGANISM_ZIP_URL, BIOGRID_ZIP_PATH)
        extracted = extract_biogrid_organisms(BIOGRID_ZIP_PATH, EVIDENCE_DIR)
        biogrid_paths = [extracted["human"], extracted["yeast"]]
        manifest["biogrid_zip"] = BIOGRID_ZIP_PATH.name
        manifest["biogrid_files"] = [p.name for p in biogrid_paths]
    else:
        manifest["biogrid_files"] = [p.name for p in biogrid_paths]
        manifest["biogrid_cached"] = True

    if not INTACT_MITAB_PATH.is_file():
        if skip_download:
            raise FileNotFoundError(f"missing {INTACT_MITAB_PATH} and skip_download=True")
        download_intact_psicquic(sorted(wanted), INTACT_MITAB_PATH)
        manifest["intact"] = INTACT_MITAB_PATH.name
    else:
        manifest["intact"] = INTACT_MITAB_PATH.name
        manifest["intact_cached"] = True

    if not STRING_CHANNELS_CSV.is_file():
        if skip_download:
            string_frame = pd.DataFrame()
        else:
            string_frame = fetch_string_channels(proteins, STRING_CHANNELS_CSV)
        manifest["string_channels_rows"] = int(len(string_frame))
    else:
        string_frame = pd.read_csv(STRING_CHANNELS_CSV)
        manifest["string_channels_cached"] = True
        manifest["string_channels_rows"] = int(len(string_frame))

    if not UNIPROT_LOCATIONS_CSV.is_file():
        if skip_download:
            locations = pd.DataFrame(columns=["uniprot_accession", "subcellular_location", "go_cc"])
        else:
            locations = fetch_uniprot_locations(sorted(hippo_actin_acc | wanted), UNIPROT_LOCATIONS_CSV)
        manifest["uniprot_location_rows"] = int(len(locations))
    else:
        locations = pd.read_csv(UNIPROT_LOCATIONS_CSV)
        manifest["uniprot_locations_cached"] = True
        manifest["uniprot_location_rows"] = int(len(locations))

    EVIDENCE_MANIFEST_JSON.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "manifest": manifest,
        "biogrid_paths": biogrid_paths,
        "intact_path": INTACT_MITAB_PATH,
        "string_frame": string_frame,
        "locations": locations,
        "wanted": wanted,
    }


def run_atlas(
    *,
    proteins_csv: Path = PROTEINS_CSV,
    interactions_csv: Path = INTERACTIONS_CSV,
    skip_download: bool = False,
    skip_graphsage: bool = False,
) -> dict[str, Any]:
    """Build the atlas CSV, QC JSON, stats JSON, and evidence figure."""
    proteins = pd.read_csv(proteins_csv)
    interactions = pd.read_csv(interactions_csv)
    bundle = ensure_evidence_files(proteins, skip_download=skip_download)
    wanted: set[str] = bundle["wanted"]
    hits: list[CuratedHit] = []
    for path in bundle["biogrid_paths"]:
        hits.extend(parse_biogrid_tab3(path, wanted))
    hits.extend(parse_intact_mitab(bundle["intact_path"], wanted))
    curated = aggregate_hits(hits)
    string_channels = parse_string_channel_frame(bundle["string_frame"])
    graphsage = pd.DataFrame()
    if not skip_graphsage:
        graphsage = overlay_graphsage_from_checkpoint(proteins, interactions)

    atlas = build_atlas_table(
        proteins,
        interactions,
        curated,
        string_channels,
        bundle["locations"],
        graphsage if not graphsage.empty else None,
    )
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    atlas.to_csv(ATLAS_CSV, index=False)
    physical = physical_edges_from_curated(proteins, curated)
    physical.to_csv(PHYSICAL_EDGES_CSV, index=False)
    stats = atlas_stats(atlas)
    if not atlas.empty:
        plot_evidence_figure(atlas, EVIDENCE_FIGURE_PNG)
        stats["figure"] = str(EVIDENCE_FIGURE_PNG)
    qc = {
        "n_proteins": int(len(proteins)),
        "n_hippo_actin_proteins": int(len(hippo_actin_protein_frame(proteins))),
        "n_atlas_pairs": int(len(atlas)),
        "n_biogrid_intact_records": int(len(hits)),
        "n_curated_pairs_in_universe": int(len(curated)),
        "n_physical_edges_in_graph": int(len(physical)),
        "n_tier1_physical_edges": int(physical["tier1"].astype(bool).sum()) if len(physical) else 0,
        "n_string_channel_pairs": int(len(string_channels)),
        "n_graphsage_scored": int(atlas["graphsage_probability"].notna().sum()) if len(atlas) else 0,
        "manifest": bundle["manifest"],
        "outputs": {
            "atlas_csv": str(ATLAS_CSV),
            "qc_json": str(ATLAS_QC_JSON),
            "stats_json": str(ATLAS_STATS_JSON),
        },
        "phase2_gate": stats.get("phase2_gate", {}),
    }
    ATLAS_QC_JSON.write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
    ATLAS_STATS_JSON.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s (%s rows)", ATLAS_CSV, len(atlas))
    logger.info("wrote %s", ATLAS_QC_JSON)
    logger.info("phase2_gate %s", json.dumps(stats.get("phase2_gate", {})))
    return {"atlas": atlas, "qc": qc, "stats": stats}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def main() -> None:
    configure_logging()
    result = run_atlas()
    stats = result["stats"]
    logger.info(
        "atlas pairs=%s string_absent=%s physical_among_absent=%s unreported_frac=%.3f",
        stats.get("n_pairs"),
        stats.get("n_string_absent"),
        stats.get("n_physical_among_absent"),
        stats.get("frac_unreported_among_absent", 0.0),
    )
    gate = stats.get("phase2_gate", {})
    skip_benchmark = os.environ.get("HIPPO_SKIP_BENCHMARK", "").strip() in {"1", "true", "True"}
    if gate.get("run_benchmark") and not skip_benchmark:
        logger.info("Phase 2 gate open: %s", gate.get("reason"))
        from src.physical_benchmark import run_benchmark

        run_benchmark()
    elif skip_benchmark:
        logger.info("Phase 2 skipped via HIPPO_SKIP_BENCHMARK: %s", gate.get("reason"))
    else:
        logger.info("Phase 2 skipped: %s", gate.get("reason"))


def _optional_int(text: str) -> int | None:
    try:
        return int(str(text).strip())
    except (TypeError, ValueError):
        return None


def _parse_taxid(text: str) -> int | None:
    match = re.search(r"taxid:(\d+)", text or "", flags=re.I)
    if match:
        return int(match.group(1))
    return _optional_int(text)


def _first_present(frame: pd.DataFrame, *names: str) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise KeyError(f"none of {names} in columns {list(frame.columns)}")


def _unique(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _unique_join(values: Sequence[str]) -> str:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in values:
        text = str(raw).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ";".join(ordered)


def _all_int(values: Iterable[str]) -> bool:
    try:
        return all(str(v).isdigit() for v in values)
    except TypeError:
        return False


def _location_map(locations: pd.DataFrame) -> dict[str, set[str]]:
    if locations.empty:
        return {}
    acc_col = "uniprot_accession" if "uniprot_accession" in locations.columns else locations.columns[0]
    loc_col = "subcellular_location" if "subcellular_location" in locations.columns else None
    go_col = "go_cc" if "go_cc" in locations.columns else None
    mapping: dict[str, set[str]] = {}
    for row in locations.itertuples(index=False):
        data = row._asdict() if hasattr(row, "_asdict") else {}
        acc = str(data.get(acc_col, getattr(row, acc_col)))
        loc_text = str(data.get(loc_col, "")) if loc_col else ""
        go_text = str(data.get(go_col, "")) if go_col else ""
        if loc_text == "nan":
            loc_text = ""
        if go_text == "nan":
            go_text = ""
        mapping[acc] = location_buckets(loc_text, go_text)
    return mapping


def _graphsage_map(scores: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    mapping: dict[tuple[str, str], dict[str, Any]] = {}
    if scores.empty:
        return mapping
    a_col = "string_id_a"
    b_col = "string_id_b"
    p_col = "probability" if "probability" in scores.columns else "graphsage_probability"
    r_col = "rank" if "rank" in scores.columns else None
    for row in scores.itertuples(index=False):
        data = row._asdict() if hasattr(row, "_asdict") else dict(zip(scores.columns, row, strict=True))
        key = pair_key(str(data[a_col]), str(data[b_col]))
        mapping[key] = {
            "probability": float(data[p_col]) if pd.notna(data[p_col]) else None,
            "rank": int(data[r_col]) if r_col and pd.notna(data.get(r_col)) else None,
        }
    return mapping


def _spearman(left: pd.Series, right: pd.Series) -> float:
    if left.nunique() < 2 or right.nunique() < 2:
        return float("nan")
    corr = left.rank().corr(right.rank())
    if corr is None or (isinstance(corr, float) and math.isnan(corr)):
        return float("nan")
    return float(corr)


def _short_species(name: str) -> str:
    if "sapiens" in name.lower():
        return "H. sapiens"
    if "cerevisiae" in name.lower():
        return "S. cerevisiae"
    return name


def _compact_top(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = (
        "species",
        "protein_a",
        "protein_b",
        "evidence_class",
        "graphsage_probability",
        "graphsage_rank_among_absent",
        "string_combined",
        "dominant_channel",
        "biogrid_physical",
        "intact",
        "pubmed_count",
        "assays",
        "localization_overlap",
        "degree_product",
        "shared_neighbors",
    )
    compact = []
    for row in rows:
        compact.append({k: row.get(k) for k in keep})
    return compact


if __name__ == "__main__":
    main()
