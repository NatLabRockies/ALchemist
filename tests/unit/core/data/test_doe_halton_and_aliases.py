"""Halton is its own sequence (T12); the CCD and optimal-design aliases say
what they do (T9, T10).

``method="halton"`` ran the Hammersly sampler, so the two designs were
identical. It now runs scikit-optimize's ``Halton`` sampler, seeded.
``ccd_face="circumscribed"`` is scaled into the bounds and so equals the
inscribed design; ``detmax`` and ``modified_fedorov`` run the Fedorov
exchange. Neither behaviour changes here, but both are now said out loud
in a log warning.
"""
import logging

import pytest

from alchemist_core import OptimizationSession
from alchemist_core.data.search_space import SearchSpace
from alchemist_core.utils.doe import generate_initial_design


def _space():
    s = SearchSpace()
    s.add_variable("x", "real", min=0.0, max=1.0)
    s.add_variable("y", "real", min=0.0, max=1.0)
    s.add_variable("z", "real", min=0.0, max=1.0)
    return s


def test_halton_differs_from_hammersly():
    halton = generate_initial_design(_space(), method="halton", n_points=8, random_seed=1)
    hammersly = generate_initial_design(_space(), method="hammersly", n_points=8, random_seed=1)
    assert halton != hammersly
    assert len(halton) == 8 and all(0.0 <= p["x"] <= 1.0 for p in halton)


def test_halton_is_reproducible():
    # Like Hammersly, skopt's Halton is a fixed sequence (random_state only
    # draws a skip when a skip range is set), so a seed reproduces it.
    a = generate_initial_design(_space(), method="halton", n_points=8, random_seed=3)
    b = generate_initial_design(_space(), method="halton", n_points=8, random_seed=3)
    assert a == b
    # The first point after the origin is the classic (1/2, 1/3, 1/5).
    assert a[1] == {"x": 0.5, "y": pytest.approx(1 / 3), "z": 0.2}


def test_circumscribed_ccd_warns_that_it_is_inscribed(caplog):
    with caplog.at_level(logging.WARNING):
        circ = generate_initial_design(_space(), method="ccd", ccd_face="circumscribed")
    assert "inscribed" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        insc = generate_initial_design(_space(), method="ccd", ccd_face="inscribed")
    assert "inscribed" not in caplog.text
    assert circ == insc


@pytest.mark.parametrize("alias", ["detmax", "modified_fedorov"])
def test_optimal_design_aliases_warn_that_they_run_fedorov(alias, caplog):
    s = OptimizationSession()
    s.add_variable("x", "real", bounds=(0.0, 1.0))
    s.add_variable("y", "real", bounds=(0.0, 1.0))
    with caplog.at_level(logging.WARNING):
        pts = s.generate_optimal_design(n_points=6, criterion="D", algorithm=alias,
                                        model_type="quadratic", random_seed=1)
    assert alias in caplog.text and "fedorov" in caplog.text.lower()
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        ref = s.generate_optimal_design(n_points=6, criterion="D", algorithm="fedorov",
                                        model_type="quadratic", random_seed=1)
    assert "detmax" not in caplog.text and "modified_fedorov" not in caplog.text
    # (points, info): the points agree, only info['algorithm'] differs.
    assert pts[0] == ref[0]
