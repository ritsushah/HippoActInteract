"""Unit tests for STRING HTTP behavior (mocked; no live API)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest
import requests

from src.http_util import APIError, request_with_retry
from src.string_client import StringClient

GET_STRING_IDS_TSV = (
    "queryItem\tqueryIndex\tstringId\tncbiTaxonId\ttaxonName\tpreferredName\tannotation\n"
    "YAP1\t0\t9606.ENSP00000282458\t9606\tHomo sapiens\tYAP1\tYes-associated protein\n"
    "YAP1\t0\t9606.ENSP000DUPLICATE\t9606\tHomo sapiens\tYAP1\tsecond hit dropped\n"
    "ACTB\t1\t9606.ENSP00000349960\t9606\tHomo sapiens\tACTB\tActin cytoplasmic 1\n"
)

PARTNERS_TSV = (
    "stringId_A\tstringId_B\tpreferredName_A\tpreferredName_B\tncbiTaxonId\tscore\n"
    "9606.ENSP00000282458\t9606.ENSP000AMOT\tYAP1\tAMOT\t9606\t0.95\n"
)


def _ok_response(text: str, status: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.text = text
    response.headers = {}
    response.raise_for_status = MagicMock()
    return response


def test_request_with_retry_recovers_from_429() -> None:
    session = MagicMock()
    busy = MagicMock()
    busy.status_code = 429
    busy.headers = {"Retry-After": "0"}
    session.request.side_effect = [busy, _ok_response("ok")]

    response = request_with_retry(
        session,
        "GET",
        "https://example.test/api",
        timeout=5,
        max_attempts=3,
        backoff_seconds=0.01,
    )
    assert response.text == "ok"
    assert session.request.call_count == 2


def test_request_with_retry_raises_after_exhausted_5xx() -> None:
    session = MagicMock()
    fail = MagicMock()
    fail.status_code = 503
    fail.headers = {}
    session.request.return_value = fail

    with pytest.raises(APIError, match="after 2 attempts"):
        request_with_retry(
            session,
            "POST",
            "https://example.test/api",
            timeout=5,
            max_attempts=2,
            backoff_seconds=0.0,
        )


def test_get_string_ids_keeps_first_hit_per_query() -> None:
    session = MagicMock()
    session.headers = {}
    session.request.return_value = _ok_response(GET_STRING_IDS_TSV)
    client = StringClient(session=session, pause_seconds=0.0)

    frame = client.get_string_ids(["YAP1", "ACTB"], 9606)

    assert list(frame["queryItem"]) == ["YAP1", "ACTB"]
    assert "9606.ENSP000DUPLICATE" not in set(frame["stringId"])
    posted = session.request.call_args
    assert posted.args[0] == "POST"
    assert posted.kwargs["data"]["species"] == 9606
    assert posted.kwargs["data"]["echo_query"] == 1


def test_interaction_partners_sends_limit_and_score() -> None:
    session = MagicMock()
    session.headers = {}
    session.request.return_value = _ok_response(PARTNERS_TSV)
    client = StringClient(session=session, pause_seconds=0.0)

    frame = client.interaction_partners(
        ["9606.ENSP00000282458"],
        9606,
        required_score=700,
        limit=20,
        network_type="functional",
    )

    assert len(frame) == 1
    payload = session.request.call_args.kwargs["data"]
    assert payload["required_score"] == 700
    assert payload["limit"] == 20
    assert payload["network_type"] == "functional"


def test_string_error_body_raises() -> None:
    session = MagicMock()
    session.headers = {}
    session.request.return_value = _ok_response("Error: invalid species")
    client = StringClient(session=session, pause_seconds=0.0)

    with pytest.raises(APIError, match="STRING API error"):
        client.get_string_ids(["YAP1"], 9606)


def test_request_exception_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    session = MagicMock()
    session.request.side_effect = [
        requests.ConnectionError("down"),
        _ok_response("recovered"),
    ]
    response = request_with_retry(
        session,
        "GET",
        "https://example.test/api",
        timeout=5,
        max_attempts=3,
        backoff_seconds=0.0,
    )
    assert response.text == "recovered"
    assert isinstance(pd.DataFrame(), pd.DataFrame)
