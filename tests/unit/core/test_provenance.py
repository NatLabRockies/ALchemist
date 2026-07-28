"""Provenance: ProvenanceId must never become a model feature."""
import pandas as pd
from alchemist_core.data.experiment_manager import ExperimentManager, PROVENANCE_COL
from alchemist_core import OptimizationSession


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


def test_provenance_col_does_not_leak_into_botorch_eval():
    """A ProvenanceId column present in the dataset must not reach the model
    during botorch evaluate()/predictions (regression for the leak the
    metadata_columns() helper guards)."""
    from alchemist_core import OptimizationSession
    s = OptimizationSession()
    s.add_variable("t", "real", bounds=(0.0, 10.0))
    # Stage + complete a handful of points; ProvenanceId gets stamped in a later
    # task, so here we stamp it directly to simulate its presence.
    for i in range(6):
        s.add_experiment({"t": float(i)}, output=float(i) * 0.5)
    s.experiment_manager.df[PROVENANCE_COL] = [f"id-{i}" for i in range(6)]
    # Train + evaluate must not crash on a non-numeric ProvenanceId feature.
    s.train_model(backend="botorch")
    # get_features_and_target must exclude it
    X, _ = s.experiment_manager.get_features_and_target()
    assert PROVENANCE_COL not in X.columns


def _session_with_staged_suggestion():
    s = OptimizationSession()
    s.add_variable("temperature", "real", bounds=(100, 1000))
    s.add_variable("catalyst", "categorical", categories=["A", "B"])
    item = s.queue.stage({"temperature": 500.0, "catalyst": "A", "_reason": "qEI"})
    return s, item


def test_complete_records_actual_and_delta():
    s, item = _session_with_staged_suggestion()
    s.complete_experiment(
        item.id,
        actual_inputs={"temperature": 505.0, "catalyst": "A"},
        output=0.42,
    )
    df = s.experiment_manager.get_data()
    assert len(df) == 1
    assert df.iloc[0]["temperature"] == 505.0
    assert df.iloc[0][PROVENANCE_COL] == item.id

    recs = s.get_provenance()
    assert len(recs) == 1
    r = recs[0]
    assert r["id"] == item.id
    assert r["strategy"] == "qEI"
    assert r["suggested"] == {"temperature": 500.0, "catalyst": "A"}
    assert r["actual"] == {"temperature": 505.0, "catalyst": "A"}
    assert r["delta"]["temperature"] == 5.0
    assert r["delta"]["catalyst"] == "unchanged"
    assert r["output"] == 0.42
