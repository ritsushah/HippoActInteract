"""UniProt REST client for canonical amino-acid sequences."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import requests

from src.config import (
    GENE_QUERY_BATCH_SIZE,
    HTTP_BACKOFF_SECONDS,
    HTTP_MAX_ATTEMPTS,
    HTTP_REQUEST_PAUSE_SECONDS,
    HTTP_TIMEOUT_SECONDS,
    UNIPROT_SEARCH_URL,
)
from src.http_util import header_map, request_with_retry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UniProtSequence:
    """One reviewed (or fallback) UniProt sequence for a gene symbol."""

    gene: str
    accession: str
    sequence: str


class UniProtClient:
    """Fetch Swiss-Prot sequences by gene symbol, with TrEMBL fallback."""

    def __init__(
        self,
        *,
        search_url: str = UNIPROT_SEARCH_URL,
        session: requests.Session | None = None,
        pause_seconds: float = HTTP_REQUEST_PAUSE_SECONDS,
        batch_size: int = GENE_QUERY_BATCH_SIZE,
    ) -> None:
        self.search_url = search_url
        self.session = session or requests.Session()
        self.session.headers.update(
            header_map({"Accept": "application/json"})
        )
        self.pause_seconds = pause_seconds
        self.batch_size = batch_size

    def fetch_sequences(
        self,
        gene_symbols: list[str],
        organism_id: int,
    ) -> dict[str, UniProtSequence]:
        """Return a map of UPPERCASE gene symbol -> sequence record."""
        unique_genes = _unique_preserve(gene_symbols)
        reviewed = self._search_batches(unique_genes, organism_id, reviewed_only=True)
        missing = [g for g in unique_genes if g.upper() not in reviewed]
        if missing:
            logger.warning(
                "no reviewed UniProt hit for %s genes (organism=%s); trying TrEMBL",
                len(missing),
                organism_id,
            )
            unreviewed = self._search_batches(missing, organism_id, reviewed_only=False)
            reviewed.update(unreviewed)
        still_missing = [g for g in unique_genes if g.upper() not in reviewed]
        if still_missing:
            logger.warning("no UniProt sequence for: %s", still_missing)
        logger.info(
            "UniProt sequences organism=%s resolved=%s/%s",
            organism_id,
            len(reviewed),
            len(unique_genes),
        )
        return reviewed

    def _search_batches(
        self,
        genes: list[str],
        organism_id: int,
        *,
        reviewed_only: bool,
    ) -> dict[str, UniProtSequence]:
        found: dict[str, UniProtSequence] = {}
        for start in range(0, len(genes), self.batch_size):
            batch = genes[start : start + self.batch_size]
            found.update(self._search(batch, organism_id, reviewed_only=reviewed_only))
        return found

    def _search(
        self,
        genes: list[str],
        organism_id: int,
        *,
        reviewed_only: bool,
    ) -> dict[str, UniProtSequence]:
        if not genes:
            return {}
        query = _build_query(genes, organism_id, reviewed_only=reviewed_only)
        params = {
            "query": query,
            "fields": "accession,gene_names,sequence",
            "format": "json",
            "size": "500",
        }
        logger.info(
            "UniProt search organism=%s reviewed=%s batch=%s",
            organism_id,
            reviewed_only,
            len(genes),
        )
        response = request_with_retry(
            self.session,
            "GET",
            self.search_url,
            timeout=HTTP_TIMEOUT_SECONDS,
            max_attempts=HTTP_MAX_ATTEMPTS,
            backoff_seconds=HTTP_BACKOFF_SECONDS,
            params=params,
        )
        if self.pause_seconds > 0:
            time.sleep(self.pause_seconds)
        payload = response.json()
        results = payload.get("results", [])
        if not isinstance(results, list):
            logger.error("unexpected UniProt payload keys=%s", list(payload)[:8])
            return {}
        return _select_best_sequences(results, set(g.upper() for g in genes))


def _build_query(genes: list[str], organism_id: int, *, reviewed_only: bool) -> str:
    gene_clause = " OR ".join(f"gene_exact:{gene}" for gene in genes)
    query = f"({gene_clause}) AND organism_id:{organism_id}"
    if reviewed_only:
        query += " AND reviewed:true"
    return query


def _unique_preserve(symbols: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for symbol in symbols:
        key = symbol.upper()
        if key not in seen:
            seen.add(key)
            ordered.append(symbol)
    return ordered


def _primary_gene_name(entry: dict[str, object]) -> str | None:
    genes = entry.get("genes")
    if not isinstance(genes, list) or not genes:
        return None
    first = genes[0]
    if not isinstance(first, dict):
        return None
    gene_name = first.get("geneName")
    if not isinstance(gene_name, dict):
        return None
    value = gene_name.get("value")
    return str(value) if value else None


def _select_best_sequences(
    results: list[object],
    wanted: set[str],
) -> dict[str, UniProtSequence]:
    """Keep the longest sequence per requested gene symbol."""
    best: dict[str, UniProtSequence] = {}
    for raw in results:
        if not isinstance(raw, dict):
            continue
        gene = _primary_gene_name(raw)
        accession = raw.get("primaryAccession")
        sequence_block = raw.get("sequence")
        if not gene or not isinstance(accession, str) or not isinstance(sequence_block, dict):
            continue
        key = gene.upper()
        if key not in wanted:
            continue
        seq_value = sequence_block.get("value")
        if not isinstance(seq_value, str) or not seq_value:
            continue
        current = best.get(key)
        if current is None or len(seq_value) > len(current.sequence):
            best[key] = UniProtSequence(gene=gene, accession=accession, sequence=seq_value)
    return best
