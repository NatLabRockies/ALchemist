"""The contour grid's features come from the model, not from "all data
columns minus metadata": a CSV with an extra non-variable column 500'd the
contour plot although the model trained on the variables alone.
"""
import io

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

CSV = "temperature,pressure,Operator,Output\n" + "\n".join(
    f"{t},{p},op{i % 2},{y}" for i, (t, p, y) in enumerate(
        [(200, 3, 0.65), (250, 5, 0.85), (300, 7, 0.92), (350, 4, 0.78), (400, 6, 0.88), (450, 8, 0.81)]))


@pytest.fixture
def sid():
    s = client.post("/api/v1/sessions", json={}).json()["session_id"]
    for name, lo, hi in (("temperature", 100, 500), ("pressure", 1, 10)):
        assert client.post(f"/api/v1/sessions/{s}/variables",
                           json={"name": name, "type": "real", "min": lo, "max": hi}).status_code == 200
    r = client.post(f"/api/v1/sessions/{s}/experiments/upload",
                    files={"file": ("data.csv", io.BytesIO(CSV.encode()), "text/csv")})
    assert r.status_code == 200, r.text
    yield s
    client.delete(f"/api/v1/sessions/{s}")


@pytest.mark.parametrize("backend", ["sklearn", "botorch"])
def test_contour_ignores_an_extra_csv_column(sid, backend):
    r = client.post(f"/api/v1/sessions/{sid}/model/train", json={"backend": backend, "kernel": "RBF"})
    assert r.status_code == 200, r.text
    r = client.post(f"/api/v1/sessions/{sid}/visualizations/contour",
                    json={"x_var": "temperature", "y_var": "pressure", "grid_resolution": 10})
    assert r.status_code == 200, r.text
    assert len(r.json()["predictions"]) == 10
