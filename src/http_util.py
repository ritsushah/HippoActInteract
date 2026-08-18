"""Shared HTTP helpers with retry/backoff for public REST APIs."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

import requests

logger = logging.getLogger(__name__)

RETRY_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


class APIError(RuntimeError):
    """Raised when an HTTP API call fails after retries."""


def _retry_after_seconds(response: requests.Response, fallback: float) -> float:
    """Parse Retry-After if present; otherwise use the exponential fallback."""
    raw: str | None = response.headers.get("Retry-After")
    if raw is None:
        return fallback
    try:
        return max(float(raw), 0.0)
    except ValueError:
        logger.warning("unparseable Retry-After=%r; using fallback=%.1fs", raw, fallback)
        return fallback


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    timeout: float,
    max_attempts: int,
    backoff_seconds: float,
    **kwargs: Any,
) -> requests.Response:
    """Send an HTTP request, retrying rate limits and transient server errors."""
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.request(method, url, timeout=timeout, **kwargs)
        except requests.RequestException as exc:
            last_error = exc
            sleep_for = backoff_seconds * (2 ** (attempt - 1))
            logger.warning(
                "request error attempt=%s/%s url=%s err=%s; sleep=%.1fs",
                attempt,
                max_attempts,
                url,
                exc,
                sleep_for,
            )
            if attempt == max_attempts:
                break
            time.sleep(sleep_for)
            continue

        if response.status_code in RETRY_STATUSES:
            sleep_for = _retry_after_seconds(
                response, fallback=backoff_seconds * (2 ** (attempt - 1))
            )
            logger.warning(
                "retryable HTTP %s attempt=%s/%s url=%s; sleep=%.1fs",
                response.status_code,
                attempt,
                max_attempts,
                url,
                sleep_for,
            )
            last_error = APIError(f"HTTP {response.status_code} for {url}")
            if attempt == max_attempts:
                break
            time.sleep(sleep_for)
            continue

        if response.status_code >= 400:
            snippet = response.text[:300].replace("\n", " ")
            raise APIError(f"HTTP {response.status_code} for {url}: {snippet}")

        return response

    raise APIError(f"Failed {method} {url} after {max_attempts} attempts") from last_error


def join_identifiers(identifiers: list[str]) -> str:
    """Join protein identifiers the way STRING POST bodies expect."""
    return "\r".join(identifiers)


def header_map(headers: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return default User-Agent headers merged with optional extras."""
    merged = {"User-Agent": "HippoActInteract/0.1 (bioinformatics research pipeline)"}
    if headers:
        merged.update(dict(headers))
    return merged
