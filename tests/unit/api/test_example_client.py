"""Unit tests for the example API client workflow helper.

The fakes validate every request body against the real request model and
build every response from the real response model, so the test fails when
the script and the API disagree (EX-1: it used to send `type: continuous`
with `bounds`, which the API refuses with a 422, and read an `output_range`
key the summary does not have, while the old fakes accepted anything).
"""

from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, TypeAdapter

from alchemist_core.session import ANALYTIC_STRATEGIES
from api import example_client
from api.models.requests import (
    AcquisitionRequest,
    AddExperimentsBatchRequest,
    AddVariableRequest,
    PredictionRequest,
    TrainModelRequest,
)
from api.models.responses import (
    AcquisitionResponse,
    DataSummary,
    ExperimentResponse,
    ExperimentsSummaryResponse,
    PredictionResponse,
    SessionCreateResponse,
    SessionInfoResponse,
    TrainModelResponse,
    VariableResponse,
)

_variable_request = TypeAdapter(AddVariableRequest)


def _mock_response(payload: BaseModel | Dict[str, Any]) -> MagicMock:
    if isinstance(payload, BaseModel):
        payload = payload.model_dump(mode="json")
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
            return _mock_response(SessionCreateResponse(session_id="session-123",
                                                        created_at="2024-01-01T00:00:00"))
        if url.endswith("/variables"):
            variable = _variable_request.validate_python(json)  # 422 in the real API
            return _mock_response(VariableResponse(message="ok", variable={"name": variable.name}))
        if url.endswith("/experiments/batch"):
            batch = AddExperimentsBatchRequest.model_validate(json)
            return _mock_response(ExperimentResponse(n_experiments=len(batch.experiments)))
        if url.endswith("/model/train"):
            train = TrainModelRequest.model_validate(json)
            return _mock_response(TrainModelResponse(
                success=True, backend=train.backend, kernel=train.kernel,
                # the keys train_model returns (TrainModelResponse example)
                metrics={"rmse": 0.05, "mae": 0.03, "r2": 0.91}, hyperparameters={"kernel": train.kernel}))
        if url.endswith("/acquisition/suggest"):
            AcquisitionRequest.model_validate(json)
            return _mock_response(AcquisitionResponse(suggestions=[
                {"temperature": 320.0, "pressure": 4.2, "catalyst": "Pt"},
                {"temperature": 340.0, "pressure": 5.0, "catalyst": "Pd"},
            ], n_suggestions=2))
        if url.endswith("/model/predict"):
            predict = PredictionRequest.model_validate(json)
            return _mock_response(PredictionResponse(predictions=[
                {"inputs": point, "prediction": 0.8, "uncertainty": 0.1}
                for point in predict.inputs
            ], n_predictions=len(predict.inputs)))
        raise AssertionError(f"Unexpected POST URL: {url}")

    def fake_get(url: str):
        fetched.append(url)
        if url.endswith("/experiments/summary"):
            # The keys get_data_summary() returns: no "output_range".
            return _mock_response(ExperimentsSummaryResponse(
                n_experiments=8, has_data=True, has_noise=False,
                target_stats={"min": 0.65, "max": 0.95, "mean": 0.8, "std": 0.1},
                feature_names=["temperature", "pressure", "catalyst"]))
        if url.endswith("/sessions/session-123"):
            return _mock_response(SessionInfoResponse(
                session_id="session-123", created_at="2024-01-01T00:00:00",
                last_accessed="2024-01-01T00:00:00", search_space={"n_variables": 3},
                data=DataSummary(n_experiments=8, has_data=True), model=None))
        raise AssertionError(f"Unexpected GET URL: {url}")

    monkeypatch.setattr(example_client.requests, "post", fake_post)
    monkeypatch.setattr(example_client.requests, "get", fake_get)

    example_client.main()

    output = capsys.readouterr().out
    assert "Workflow complete" in output
    assert "Output range: [0.65, 0.95]" in output
    assert "R² Score: 0.910" in output
    assert posted[0][0].endswith("/sessions")
    variables = [item[1] for item in posted if item[0].endswith("/variables")]
    assert [v["type"] for v in variables] == ["real", "real", "categorical"]
    assert any(item[0].endswith("/model/train") for item in posted)
    assert any(item[0].endswith("/acquisition/suggest") for item in posted)

    # A1: a batch (n_suggestions > 1) needs the BoTorch backend and a q-strategy,
    # otherwise the real API answers 400. The request models would not notice.
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
