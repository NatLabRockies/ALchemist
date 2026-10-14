"""Unit tests for the example API client workflow helper."""

from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import pytest

from alchemist_core.session import ANALYTIC_STRATEGIES
from api import example_client


def _mock_response(payload: Dict[str, Any]) -> MagicMock:
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    response.status_code = 200
    response.text = str(payload)
    return response


def test_main_happy_path(monkeypatch, capsys):
    posted: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    fetched: List[str] = []

    def fake_post(url: str, json: Dict[str, Any] | None = None, params: Dict[str, Any] | None = None):
        posted.append((url, json or {}, params or {}))
        if url.endswith("/sessions"):
            return _mock_response({"session_id": "session-123"})
        if url.endswith("/variables"):
            return _mock_response({"message": "ok"})
        if url.endswith("/experiments/batch"):
            assert json is not None
            return _mock_response({"n_experiments": len(json["experiments"]), "model_trained": False})
        if url.endswith("/model/train"):
            return _mock_response({
                "success": True,
                "backend": json["backend"],
                "kernel": json["kernel"],
                "metrics": {"r2_score": 0.91},
                "hyperparameters": {"kernel": json["kernel"]},
            })
        if url.endswith("/acquisition/suggest"):
            return _mock_response({
                "suggestions": [
                    {"temperature": 320.0, "pressure": 4.2, "catalyst": "Pt"},
                    {"temperature": 340.0, "pressure": 5.0, "catalyst": "Pd"},
                ]
            })
        if url.endswith("/model/predict"):
            assert json is not None
            predictions = [
                {"inputs": point, "prediction": 0.8, "uncertainty": 0.1}
                for point in json["inputs"]
            ]
            return _mock_response({"predictions": predictions})
        raise AssertionError(f"Unexpected POST URL: {url}")

    def fake_get(url: str):
        fetched.append(url)
        if url.endswith("/experiments/summary"):
            return _mock_response({
                "n_experiments": 8,
                "has_data": True,
                "output_range": [0.65, 0.95],
            })
        if url.endswith("/sessions/session-123"):
            return _mock_response({
                "session_id": "session-123",
                "created_at": "2024-01-01T00:00:00",
                "expires_at": "2024-01-02T00:00:00",
            })
        raise AssertionError(f"Unexpected GET URL: {url}")

    monkeypatch.setattr(example_client.requests, "post", fake_post)
    monkeypatch.setattr(example_client.requests, "get", fake_get)

    example_client.main()

    output = capsys.readouterr().out
    assert "Workflow complete" in output
    assert posted[0][0].endswith("/sessions")
    assert len([item for item in posted if item[0].endswith("/variables")]) == 3
    assert any(item[0].endswith("/model/train") for item in posted)
    assert any(item[0].endswith("/acquisition/suggest") for item in posted)

    # A1: a batch (n_suggestions > 1) needs the BoTorch backend and a q-strategy,
    # otherwise the real API answers 400. The mocks above would not notice.
    train = next(item[1] for item in posted if item[0].endswith("/model/train"))
    suggest = next(item[1] for item in posted if item[0].endswith("/acquisition/suggest"))
    if suggest["n_suggestions"] > 1:
        assert train["backend"] == "botorch"
        assert suggest["strategy"].lower() not in ANALYTIC_STRATEGIES
    assert fetched == [
        f"{example_client.BASE_URL}/sessions/session-123/experiments/summary",
        f"{example_client.BASE_URL}/sessions/session-123",
    ]


def test_main_propagates_connection_errors(monkeypatch):
    def failing_post(*_args, **_kwargs):  # pragma: no cover - control flow guard
        raise example_client.requests.exceptions.ConnectionError()

    monkeypatch.setattr(example_client.requests, "post", failing_post)

    with pytest.raises(example_client.requests.exceptions.ConnectionError):
        example_client.main()
