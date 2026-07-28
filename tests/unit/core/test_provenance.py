"""Provenance: ProvenanceId must never become a model feature."""
import pandas as pd
from alchemist_core.data.experiment_manager import ExperimentManager, PROVENANCE_COL


def _manager_with_provenance():
    em = ExperimentManager(target_columns=["Output"])
    df = pd.DataFrame({
        "x": [0.1, 0.2, 0.3],
        "Output": [1.0, 2.0, 3.0],
        "Iteration": [0, 1, 2],
        "Reason": ["Manual", "qEI", "qEI"],
        PROVENANCE_COL: ["id-a", "id-b", "id-c"],
    })
    em.df = df
    return em


def test_provenance_col_constant():
    assert PROVENANCE_COL == "ProvenanceId"


def test_provenance_excluded_from_model_inputs():
    em = _manager_with_provenance()
    X, _ = em.get_features_and_target()
    assert PROVENANCE_COL not in X.columns
    assert "Output" not in X.columns
    assert "Iteration" not in X.columns
    assert "Reason" not in X.columns
    assert list(X.columns) == ["x"]


def test_provenance_excluded_from_all_x_accessors():
    em = _manager_with_provenance()
    X1, _ = em.get_features_and_target()
    X2, _, _ = em.get_features_target_and_noise()
    X3, _, _ = em.get_features_and_targets_multi()
    for X in (X1, X2, X3):
        assert PROVENANCE_COL not in X.columns
        assert list(X.columns) == ["x"]
