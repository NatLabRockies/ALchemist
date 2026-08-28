"""One design request must not deny the API to everyone else.

``POST /initial-design`` and ``POST /optimal-design`` are ``async def`` and
used to call their core generator **synchronously**, so the work ran on the
ASGI event loop itself. A single request therefore stalled every other client
-- health checks, WebSocket pings, other sessions -- for as long as the
generator ran, which for a constrained space-filling design could be minutes.

Three routes in this API already avoid that with ``run_in_threadpool``
(``models.py`` for training, ``visualizations.py`` for predictions and for
metrics); the design routes simply never adopted it.

The starvation probe is the discriminating part. A test that only asserted
"the response came back" passes either way -- blocking the loop still returns
the right answer, just late and at everyone else's expense. So a sampler task
is started *first* and left running across the request, and what is asserted
is the worst delay it observed. Ordering matters: if the request were started
first, a synchronous handler would run to completion before the sampler ever
got a turn, and the probe would measure nothing.
"""

import asyncio
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from api.main import app
from alchemist_core.session import OptimizationSession

client = TestClient(app)

# How long the stand-in generator occupies. Long enough that a blocked loop is
# unmistakable, short enough to keep the suite quick.
BUSY_SECONDS = 1.0

# The sampler asks for 50 ms naps. On a free loop it overshoots by ~1 ms; on a
# loop blocked by BUSY_SECONDS it overshoots by ~1 s. 400 ms sits far from both.
LAG_BUDGET = 0.4
NAP = 0.05


@pytest.fixture
def session_id():
    response = client.post("/api/v1/sessions", json={"ttl_hours": 1})
    response.raise_for_status()
    sid = response.json()["session_id"]
    yield sid
    client.delete(f"/api/v1/sessions/{sid}")


def _add_variables(sid):
    for payload in (
        {"name": "x1", "type": "real", "min": 0.0, "max": 10.0},
        {"name": "x2", "type": "real", "min": 0.0, "max": 10.0},
    ):
        client.post(f"/api/v1/sessions/{sid}/variables", json=payload).raise_for_status()


# ============================================================
# Defect 1 at the endpoint: an impossible design must fail fast
# ============================================================

class TestAnInfeasibleDesignRequestReturnsQuickly:

    def test_initial_design_returns_400_in_well_under_a_second(self, session_id):
        """The reported symptom: 480 s and a 400. Now: a 400, promptly."""
        _add_variables(session_id)
        client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x1": 2.5, "x2": 1.5},
            "rhs": -8.0,
            "name": "impossible",
        }).raise_for_status()

        started = time.perf_counter()
        response = client.post(
            f"/api/v1/sessions/{session_id}/initial-design",
            json={"method": "lhs", "n_points": 8, "random_seed": 3},
        )
        elapsed = time.perf_counter() - started

        assert response.status_code == 400
        assert elapsed < 5.0, f"infeasible design took {elapsed:.1f}s"
        detail = response.json()["detail"]
        assert "no feasible point" in detail
        assert "relax the constraints" in detail

    def test_a_small_feasible_region_still_returns_a_design(self, session_id):
        """The failure mode the fast path must not introduce."""
        _add_variables(session_id)
        client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x1": -1.0, "x2": -1.0},
            "rhs": -18.0,
            "name": "narrow_corner",
        }).raise_for_status()

        response = client.post(
            f"/api/v1/sessions/{session_id}/initial-design",
            json={"method": "lhs", "n_points": 8, "random_seed": 3},
        )

        assert response.status_code == 200, response.text
        points = response.json()["points"]
        assert len(points) == 8
        for point in points:
            assert point["x1"] + point["x2"] >= 18.0 - 1e-9, point


# ============================================================
# Defect 2: the event loop stays free while a design is generated
# ============================================================

def _busy_initial_design(self, **kwargs):
    time.sleep(BUSY_SECONDS)
    return [{"x1": 1.0, "x2": 2.0}]


def _busy_optimal_design(self, **kwargs):
    time.sleep(BUSY_SECONDS)
    return [{"x1": 1.0, "x2": 2.0}], {"D_eff": 88.0, "model_terms": ["x1", "x2"]}


async def _worst_lag_during(path, payload):
    """Post to ``path`` while timing naps taken on the same event loop.

    Returns ``(status_code, worst_overshoot_seconds)``.
    """
    lags = []
    stop = asyncio.Event()

    async def sampler():
        while not stop.is_set():
            started = time.perf_counter()
            await asyncio.sleep(NAP)
            lags.append(time.perf_counter() - started - NAP)

    sampler_task = asyncio.create_task(sampler())
    # Let the sampler get into its nap before anything can block the loop.
    await asyncio.sleep(NAP * 3)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://probe") as ac:
        response = await ac.post(path, json=payload, timeout=60.0)

    stop.set()
    await sampler_task
    assert lags, "the sampler never ran"
    return response.status_code, max(lags)


class TestADesignRequestDoesNotStarveTheEventLoop:

    def test_initial_design_leaves_the_loop_responsive(self, session_id, monkeypatch):
        _add_variables(session_id)
        monkeypatch.setattr(
            OptimizationSession, "generate_initial_design", _busy_initial_design
        )

        status, worst_lag = asyncio.run(_worst_lag_during(
            f"/api/v1/sessions/{session_id}/initial-design",
            {"method": "lhs", "n_points": 8},
        ))

        assert status == 200
        assert worst_lag < LAG_BUDGET, (
            f"the event loop stalled {worst_lag:.2f}s while /initial-design ran "
            f"for {BUSY_SECONDS}s; the core call is back on the loop"
        )

    def test_optimal_design_leaves_the_loop_responsive(self, session_id, monkeypatch):
        _add_variables(session_id)
        monkeypatch.setattr(
            OptimizationSession, "generate_optimal_design", _busy_optimal_design
        )

        status, worst_lag = asyncio.run(_worst_lag_during(
            f"/api/v1/sessions/{session_id}/optimal-design",
            {"model_type": "linear", "n_points": 8},
        ))

        assert status == 200
        assert worst_lag < LAG_BUDGET, (
            f"the event loop stalled {worst_lag:.2f}s while /optimal-design ran "
            f"for {BUSY_SECONDS}s; the core call is back on the loop"
        )

    def test_the_probe_can_see_a_blocked_loop(self, session_id, monkeypatch):
        """Calibration: the same probe against a deliberately blocking route.

        Without this, a probe that measured nothing at all would look like a
        passing result above. The endpoint is patched to do its sleeping in the
        handler's own coroutine, which is exactly what the defect did.
        """
        _add_variables(session_id)

        from api.routers import experiments as experiments_router

        async def blocking_run_in_threadpool(func, *args, **kwargs):
            return func(*args, **kwargs)

        monkeypatch.setattr(
            OptimizationSession, "generate_initial_design", _busy_initial_design
        )
        monkeypatch.setattr(
            experiments_router, "run_in_threadpool", blocking_run_in_threadpool
        )

        status, worst_lag = asyncio.run(_worst_lag_during(
            f"/api/v1/sessions/{session_id}/initial-design",
            {"method": "lhs", "n_points": 8},
        ))

        assert status == 200
        assert worst_lag > LAG_BUDGET, (
            f"the probe recorded only {worst_lag:.2f}s of stall against a "
            f"knowingly blocking handler, so it cannot detect the defect"
        )


# ============================================================
# The seed contract has to survive the concurrency the fix enables
# ============================================================

async def _designs_for(session_id, seeds, rounds):
    """Post one seeded /initial-design per seed per round, all concurrently."""
    transport = httpx.ASGITransport(app=app)
    path = f"/api/v1/sessions/{session_id}/initial-design"
    out = {seed: [] for seed in seeds}
    async with httpx.AsyncClient(transport=transport, base_url="http://probe") as ac:
        for _ in range(rounds):
            responses = await asyncio.gather(*[
                ac.post(path, json={"method": "lhs", "n_points": 64,
                                    "random_seed": seed}, timeout=60.0)
                for seed in seeds
            ])
            for seed, response in zip(seeds, responses):
                assert response.status_code == 200, response.text
                out[seed].append(response.json()["points"])
    return out


class TestASeededDesignIsReproducibleWhileOthersRun:
    """`random_seed` must name one design, whoever else is mid-request.

    ``generate_initial_design`` used to call the process-global
    ``np.random.seed`` and let the samplers draw from global state. That was
    atomic only because this endpoint ran the whole generation on the event
    loop. Moving it to a worker thread -- the entire point of the fix above --
    lets two seeded requests interleave their draws, so neither receives the
    design its seed names. The remedy is a generator owned by the call; this
    pins the property that remedy exists for.
    """

    def test_two_seeds_in_flight_together_each_get_their_own_design(self, session_id):
        _add_variables(session_id)

        # Reference: each seed generated with nothing else running.
        solo = asyncio.run(_designs_for(session_id, [7], rounds=1))[7][0]
        other = asyncio.run(_designs_for(session_id, [99], rounds=1))[99][0]
        assert solo != other, "the two seeds must not coincide, or this proves nothing"

        # Now both seeds, repeatedly, all in flight at once.
        concurrent = asyncio.run(_designs_for(session_id, [7, 99], rounds=6))

        for design in concurrent[7]:
            assert design == solo, "seed=7 did not get its own design under concurrency"
        for design in concurrent[99]:
            assert design == other, "seed=99 did not get its own design under concurrency"
