"""
Round-trip tests for session save/load covering transient state that historically
was not persisted: staged experiments and last suggestions.

These exist because users generate suggestions / DoE points in one ALchemist
frontend (e.g. web app), save the session, then reopen it in another frontend
(e.g. desktop GUI). Anything in `session.staged_experiments` or
`session.last_suggestions` at save time must survive the round-trip.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from alchemist_core import OptimizationSession


def _make_session_with_staged():
    """Session with 2 vars and 3 staged experiments (e.g. from a DoE)."""
    session = OptimizationSession()
    session.add_variable("temperature", "real", bounds=(100, 300))
    session.add_variable("pressure", "real", bounds=(1, 10))
    session.add_staged_experiment({"temperature": 150.0, "pressure": 2.0})
    session.add_staged_experiment({"temperature": 200.0, "pressure": 5.0})
    session.add_staged_experiment({"temperature": 250.0, "pressure": 8.0})
    return session


def _make_session_with_suggestions():
    """Session with last_suggestions populated as if suggest_next() had been called."""
    session = OptimizationSession()
    session.add_variable("temperature", "real", bounds=(100, 300))
    # Simulate a previous suggest_next() result. Stored as list of dicts is the
    # path used by acquisition; DataFrame is also a valid in-memory shape.
    session.last_suggestions = [
        {"temperature": 175.0, "Acquisition": 0.42},
        {"temperature": 225.0, "Acquisition": 0.38},
    ]
    return session


# ---- staged_experiments round-trip ----

def test_save_session_persists_staged_experiments():
    """Staged experiments are written to the session JSON file."""
    session = _make_session_with_staged()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        session.save_session(path)
        with open(path) as fh:
            data = json.load(fh)
        assert "staged_experiments" in data, (
            "save_session must include 'staged_experiments' key so DoE points "
            "generated in the web app survive when reloaded in the desktop GUI"
        )
        assert len(data["staged_experiments"]) == 3
        assert data["staged_experiments"][0]["temperature"] == 150.0
        assert data["staged_experiments"][0]["pressure"] == 2.0
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_session_restores_staged_experiments():
    """Loaded session has the same staged experiments as the saved one."""
    session = _make_session_with_staged()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        session.save_session(path)
        loaded = OptimizationSession.load_session(path, retrain_on_load=False)
        staged = loaded.get_staged_experiments()
        assert len(staged) == 3
        assert staged[0]["temperature"] == 150.0
        assert staged[1]["pressure"] == 5.0
        assert staged[2]["temperature"] == 250.0
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_session_with_no_staged_experiments_works():
    """Backward compatibility: sessions saved before staged persistence existed
    must still load cleanly (no KeyError)."""
    session = OptimizationSession()
    session.add_variable("x", "real", bounds=(0, 1))
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        session.save_session(path)
        # Simulate an older file: rewrite without staged_experiments
        with open(path) as fh:
            data = json.load(fh)
        data.pop("staged_experiments", None)
        with open(path, "w") as fh:
            json.dump(data, fh)

        loaded = OptimizationSession.load_session(path, retrain_on_load=False)
        assert loaded.get_staged_experiments() == []
    finally:
        Path(path).unlink(missing_ok=True)


# ---- last_suggestions round-trip ----

def test_save_session_persists_last_suggestions():
    """last_suggestions are written to the session JSON file."""
    session = _make_session_with_suggestions()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        session.save_session(path)
        with open(path) as fh:
            data = json.load(fh)
        assert "last_suggestions" in data
        assert len(data["last_suggestions"]) == 2
        assert data["last_suggestions"][0]["temperature"] == 175.0
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_session_restores_last_suggestions():
    session = _make_session_with_suggestions()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        session.save_session(path)
        loaded = OptimizationSession.load_session(path, retrain_on_load=False)
        assert len(loaded.last_suggestions) == 2
        assert loaded.last_suggestions[0]["temperature"] == 175.0
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_session_with_no_last_suggestions_works():
    """Sessions saved before last_suggestions persistence must still load cleanly."""
    session = OptimizationSession()
    session.add_variable("x", "real", bounds=(0, 1))
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        session.save_session(path)
        with open(path) as fh:
            data = json.load(fh)
        data.pop("last_suggestions", None)
        with open(path, "w") as fh:
            json.dump(data, fh)

        loaded = OptimizationSession.load_session(path, retrain_on_load=False)
        assert loaded.last_suggestions == []
    finally:
        Path(path).unlink(missing_ok=True)


# ---- end-to-end: web-saved session round-trip to desktop ----

def test_web_session_with_all_variable_types_roundtrips_to_desktop_sync():
    """
    Reproduces the intern's workflow:
      1. Build a session with all four SearchSpace variable types and stage a
         DoE (the way the web app does it).
      2. Save the session JSON.
      3. Load it back (simulating "Open Session" in the desktop GUI).
      4. Format every variable for the desktop variable sheet using the same
         helper the desktop UI uses.
      5. Confirm no KeyError, all four rows render, and staged experiments are
         restored.
    """
    pytest.importorskip("customtkinter")
    from ui.ui import _variable_to_sheet_row

    # 1. Build a session that exercises every variable shape
    session = OptimizationSession()
    session.add_variable("temperature", "real", bounds=(100, 300))
    session.add_variable("n_cycles", "integer", bounds=(1, 10))
    session.add_variable("catalyst", "categorical", categories=["Pt", "Pd", "Ru"])
    session.add_variable("SAR", "discrete", allowed_values=[80, 180, 280])
    session.add_variable("batch_quality", "context")

    # Stage some DoE points (this is what web-app DoE generation produces)
    session.add_staged_experiment(
        {"temperature": 150.0, "n_cycles": 2, "catalyst": "Pt", "SAR": 80, "batch_quality": 1.0}
    )
    session.add_staged_experiment(
        {"temperature": 250.0, "n_cycles": 5, "catalyst": "Pd", "SAR": 280, "batch_quality": 1.2}
    )

    # 2. Save (web app or desktop "Save" -- same code path)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        session.save_session(path)

        # 3. Load (simulates desktop "Open Session")
        loaded = OptimizationSession.load_session(path, retrain_on_load=False)

        # 4. Format every variable for the variable sheet (the exact step that
        #    used to crash with KeyError: 'values' on discrete or context vars)
        rows = [
            _variable_to_sheet_row(var_dict)
            for var_dict in loaded.search_space.variables
        ]
        assert len(rows) == 5
        names = [r[0] for r in rows]
        assert names == ["temperature", "n_cycles", "catalyst", "SAR", "batch_quality"]

        # Spot-check the categorical and discrete rows contain their values
        cat_row = next(r for r in rows if r[0] == "catalyst")
        assert "Pt" in cat_row[4] and "Pd" in cat_row[4] and "Ru" in cat_row[4]

        sar_row = next(r for r in rows if r[0] == "SAR")
        assert "80" in sar_row[4] and "180" in sar_row[4] and "280" in sar_row[4]

        # 5. Staged experiments survived the round-trip
        staged = loaded.get_staged_experiments()
        assert len(staged) == 2
        assert staged[0]["catalyst"] == "Pt"
        assert staged[1]["SAR"] == 280
    finally:
        Path(path).unlink(missing_ok=True)
