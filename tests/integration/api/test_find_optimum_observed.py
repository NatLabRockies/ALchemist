"""POST /acquisition/find-optimum conditions on observed variables."""
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

pytestmark = pytest.mark.filterwarnings("ignore::SyntaxWarning")


@pytest.fixture
def trained_sid():
    r = client.post("/api/v1/sessions", json={"ttl_hours": 1})
    assert r.status_code == 201
    sid = r.json()["session_id"]
    base = f"/api/v1/sessions/{sid}"
    assert client.post(f"{base}/variables", json={"name": "x1", "type": "real", "min": 0, "max": 10}).status_code == 200
    assert client.post(f"{base}/variables", json={"name": "z", "type": "real", "role": "observed"}).status_code == 200
    for i in range(8):
        x1 = 1.0 + i
        z = 20.0 + i
        r = client.post(f"{base}/experiments", json={
            "inputs": {"x1": x1, "z": z}, "output": -(x1 - 6) ** 2 + 0.5 * z,
        })
        assert r.status_code == 200, r.text
    r = client.post(f"{base}/model/train", json={"backend": "botorch", "kernel": "Matern"})
    assert r.status_code == 200, r.text
    yield sid
    client.delete(base)


def test_find_optimum_is_conditioned_on_the_supplied_observed_value(trained_sid):
    r = client.post(f"/api/v1/sessions/{trained_sid}/acquisition/find-optimum",
                    json={"goal": "maximize", "observed": {"z": 22.0}})
    assert r.status_code == 200, r.text
    assert r.json()["optimum"]["z"] == 22.0


def test_find_optimum_defaults_to_the_latest_experiment(trained_sid):
    r = client.post(f"/api/v1/sessions/{trained_sid}/acquisition/find-optimum",
                    json={"goal": "maximize"})
    assert r.status_code == 200, r.text
    assert r.json()["optimum"]["z"] == 27.0


def test_find_optimum_rejects_an_invalid_observed_value(trained_sid):
    r = client.post(f"/api/v1/sessions/{trained_sid}/acquisition/find-optimum",
                    json={"goal": "maximize", "observed": {"z": "abc"}})
    assert r.status_code == 400
    assert "z" in r.json()["detail"]
