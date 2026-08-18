"""STRING REST client for ID mapping, partners, and induced networks."""

from __future__ import annotations

import io
import logging
import time
from typing import Any

import pandas as pd
import requests

from src.config import (
    HTTP_BACKOFF_SECONDS,
    HTTP_MAX_ATTEMPTS,
    HTTP_REQUEST_PAUSE_SECONDS,
    HTTP_TIMEOUT_SECONDS,
    STRING_API_URL,
    STRING_CALLER_IDENTITY,
)
from src.http_util import APIError, header_map, join_identifiers, request_with_retry

logger = logging.getLogger(__name__)


class StringClient:
    """Thin STRING API wrapper. All calls use POST to avoid URL length limits."""

    def __init__(
        self,
        *,
        base_url: str = STRING_API_URL,
        caller_identity: str = STRING_CALLER_IDENTITY,
        session: requests.Session | None = None,
        pause_seconds: float = HTTP_REQUEST_PAUSE_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.caller_identity = caller_identity
        self.session = session or requests.Session()
        self.session.headers.update(header_map())
        self.pause_seconds = pause_seconds

    def get_string_ids(self, symbols: list[str], species: int) -> pd.DataFrame:
        """Map common gene names to STRING IDs. One best hit is kept per query."""
        if not symbols:
            return pd.DataFrame()
        text = self._post(
            "tsv",
            "get_string_ids",
            {
                "identifiers": join_identifiers(symbols),
                "species": species,
                "echo_query": 1,
            },
        )
        frame = self._read_tsv(text)
        if frame.empty:
            return frame
        if "queryItem" not in frame.columns:
            raise APIError("STRING get_string_ids response missing queryItem column")
        deduped = frame.drop_duplicates(subset=["queryItem"], keep="first")
        logger.info(
            "mapped %s/%s symbols for species=%s",
            len(deduped),
            len(symbols),
            species,
        )
        mapped_queries = set(deduped["queryItem"].astype(str).str.upper())
        unresolved = [s for s in symbols if s.upper() not in mapped_queries]
        if unresolved:
            logger.warning("unresolved STRING identifiers species=%s: %s", species, unresolved)
        return deduped.reset_index(drop=True)

    def interaction_partners(
        self,
        string_ids: list[str],
        species: int,
        *,
        required_score: int,
        limit: int,
        network_type: str,
    ) -> pd.DataFrame:
        """Return high-confidence partners for each protein (most confident first)."""
        if not string_ids:
            return pd.DataFrame()
        text = self._post(
            "tsv",
            "interaction_partners",
            {
                "identifiers": join_identifiers(string_ids),
                "species": species,
                "required_score": required_score,
                "limit": limit,
                "network_type": network_type,
            },
        )
        frame = self._read_tsv(text)
        logger.info("interaction_partners species=%s rows=%s", species, len(frame))
        return frame

    def network(
        self,
        string_ids: list[str],
        species: int,
        *,
        required_score: int,
        network_type: str,
    ) -> pd.DataFrame:
        """Return edges among the given proteins (no extra neighborhood nodes)."""
        if len(string_ids) < 2:
            return pd.DataFrame()
        text = self._post(
            "tsv",
            "network",
            {
                "identifiers": join_identifiers(string_ids),
                "species": species,
                "required_score": required_score,
                "network_type": network_type,
                "add_nodes": 0,
            },
        )
        frame = self._read_tsv(text)
        logger.info("network species=%s edges=%s", species, len(frame))
        return frame

    def _post(self, output_format: str, method: str, data: dict[str, Any]) -> str:
        url = f"{self.base_url}/{output_format}/{method}"
        payload: dict[str, Any] = {**data, "caller_identity": self.caller_identity}
        logger.info("STRING POST %s n_ids=%s", method, _identifier_count(data.get("identifiers")))
        response = request_with_retry(
            self.session,
            "POST",
            url,
            timeout=HTTP_TIMEOUT_SECONDS,
            max_attempts=HTTP_MAX_ATTEMPTS,
            backoff_seconds=HTTP_BACKOFF_SECONDS,
            data=payload,
        )
        if self.pause_seconds > 0:
            time.sleep(self.pause_seconds)
        return response.text

    @staticmethod
    def _read_tsv(text: str) -> pd.DataFrame:
        stripped = text.strip()
        if not stripped:
            return pd.DataFrame()
        if stripped.lower().startswith("error"):
            raise APIError(f"STRING API error: {stripped[:300]}")
        frame = pd.read_csv(io.StringIO(stripped), sep="\t")
        return frame


def _identifier_count(raw: Any) -> int:
    if not isinstance(raw, str) or not raw:
        return 0
    return len([part for part in raw.replace("\n", "\r").split("\r") if part])
