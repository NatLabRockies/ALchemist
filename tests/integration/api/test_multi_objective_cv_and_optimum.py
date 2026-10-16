"""Multi-objective sessions over REST: find-optimum answers 400, not 500 (A19),
and the parity, Q-Q and calibration plots use the per-objective CV results
with a ``target`` query parameter (A20).

``session.find_optimum`` returns a Pareto frontier for a multi-objective
session; the route read ``result['x_opt']`` and the KeyError became a 500.
The three CV routes looked only at ``model.cv_cached_results``, which the
core fills for single-objective models; the per-objective results live in
``cv_cached_results_multi``.
"""
import io

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

CSV = """x1,x2,conversion,selectivity
1,2,0.50,0.90
3,4,0.70,0.80
5,6,0.90,0.60
2,7,0.55,0.85
6,1,0.80,0.70
4,5,0.75,0.75
7,3,0.95,0.50
"""


@pytest.fixture(scope="module")
def sid():
    s = client.post("/api/v1/sessions", json={}).json()["session_id"]
    for name in ("x1", "x2"):
        assert client.post(f"/api/v1/sessions/{s}/variables",
                           json={"name": name, "type": "real", "min": 0, "max": 10}).status_code == 200
    r = client.post(f"/api/v1/sessions/{s}/experiments/upload",
                    files={"file": ("data.csv", io.BytesIO(CSV.encode()), "text/csv")},
                    params={"target_columns": "conversion,selectivity"})
    assert r.status_code == 200, r.text
    r = client.post(f"/api/v1/sessions/{s}/model/train", json={"backend": "botorch", "kernel": "rbf"})
    assert r.status_code == 200, r.text
    yield s
    client.delete(f"/api/v1/sessions/{s}")


def test_find_optimum_is_a_clear_400(sid):
    r = client.post(f"/api/v1/sessions/{sid}/acquisition/find-optimum", json={"goal": "maximize"})
    assert r.status_code == 400, r.text
    assert "multi-objective" in r.json()["detail"].lower()


@pytest.mark.parametrize("plot", ["parity", "qq-plot", "calibration-curve"])
def test_cv_plots_default_to_the_first_objective(sid, plot):
    r = client.get(f"/api/v1/sessions/{sid}/visualizations/{plot}")
    assert r.status_code == 200, r.text
    body = r.json()
    if plot == "parity":
        assert len(body["y_true"]) == 7
        # cross-validation folds reorder the rows
        assert sorted(body["y_true"]) == pytest.approx(sorted([0.50, 0.70, 0.90, 0.55, 0.80, 0.75, 0.95]))


@pytest.mark.parametrize("plot", ["parity", "qq-plot", "calibration-curve"])
def test_cv_plots_take_a_target(sid, plot):
    r = client.get(f"/api/v1/sessions/{sid}/visualizations/{plot}", params={"target": "selectivity"})
    assert r.status_code == 200, r.text
    if plot == "parity":
        assert sorted(r.json()["y_true"]) == pytest.approx(sorted([0.90, 0.80, 0.60, 0.85, 0.70, 0.75, 0.50]))


def test_an_unknown_target_is_a_400(sid):
    r = client.get(f"/api/v1/sessions/{sid}/visualizations/parity", params={"target": "yield"})
    assert r.status_code == 400, r.text
    assert "selectivity" in r.json()["detail"]
