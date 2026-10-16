"""``save_session`` writes strict JSON (W13).

A row with no value in a column, such as a result with no ``ProvenanceId``
or no ``Noise``, was written as a bare ``NaN``, which Python's ``json``
reads but ``JSON.parse`` and other strict parsers refuse. Empty cells are
now ``null`` everywhere in the file, the dump runs with ``allow_nan=False``
so a regression fails loudly, and the loader reads ``null`` back as NaN
where the frame expects a float.
"""
import json

import numpy as np
import pytest

from alchemist_core import OptimizationSession
from alchemist_core.data.experiment_manager import PROVENANCE_COL


def _refuse(constant):
    raise ValueError(f"non-strict JSON constant {constant}")


@pytest.fixture
def gappy(tmp_path):
    """One hand-added row (no Noise, no ProvenanceId) and one queue row with both."""
    s = OptimizationSession()
    s.add_variable("x", "real", bounds=(0.0, 1.0))
    s.add_experiment({"x": 0.2}, 1.0)
    s.add_staged_experiment({"x": 0.8})
    item = s.queue.pending_items()[0]
    s.complete_experiment(item.id, {"x": 0.8}, output=2.0, noise=0.01)
    df = s.experiment_manager.df
    assert np.isnan(df["Noise"].iloc[0]) and df[PROVENANCE_COL].isna().iloc[0]
    path = tmp_path / "s.json"
    s.save_session(str(path))
    return s, path


def test_file_has_no_nan_constants(gappy):
    _, path = gappy
    text = path.read_text()
    data = json.loads(text, parse_constant=_refuse)
    first = data["experiments"]["data"][0]
    assert first["Noise"] is None and first[PROVENANCE_COL] is None
    # The audit log's data snapshot is written through the same path.
    assert data["audit_log"]["experiment_data"][0]["Noise"] is None


def test_load_reads_null_back_as_nan(gappy):
    original, path = gappy
    loaded = OptimizationSession.load_session(str(path), retrain_on_load=False)
    df = loaded.experiment_manager.df
    assert len(df) == 2
    assert np.isnan(df["Noise"].iloc[0]) and df["Noise"].iloc[1] == 0.01
    assert df[PROVENANCE_COL].isna().iloc[0]
    assert df[PROVENANCE_COL].iloc[1] == original.experiment_manager.df[PROVENANCE_COL].iloc[1]
    assert loaded.experiment_manager.has_noise_data() == original.experiment_manager.has_noise_data()


def test_export_session_json_is_strict_too(gappy):
    s, _ = gappy
    json.loads(s.export_session_json(), parse_constant=_refuse)
