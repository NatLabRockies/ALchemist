"""
OptimizationSession records the arguments of its last successful training,
so a retrain (the web app's "Retrain model" on Add Point) can rebuild the same
model, and keeps them across save/load.
"""

import numpy as np
import pytest

from alchemist_core import OptimizationSession

USER_CONFIG = dict(
    backend="botorch",
    kernel="Matern",
    kernel_params={"nu": 2.5},
    input_transform_type="normalize",
    output_transform_type="standardize",
    calibration_enabled=True,
)


def _session(n=8, seed=0):
    session = OptimizationSession()
    session.add_variable("x1", "real", bounds=(0.0, 1.0))
    session.add_variable("x2", "real", bounds=(0.0, 1.0))
    rng = np.random.default_rng(seed)
    for _ in range(n):
        a, b = rng.uniform(size=2)
        session.add_experiment(inputs={"x1": float(a), "x2": float(b)}, output=float(a**2 + b))
    return session


def _assert_user_model(session):
    model = session.model
    assert session.model_backend == "botorch"
    assert model.cont_kernel_type == "Matern"
    assert model.model.covar_module.base_kernel.nu == 2.5
    assert type(model.model.input_transform).__name__ == "Normalize"
    assert type(model.model.outcome_transform).__name__ == "Standardize"


def test_no_config_before_training():
    assert _session().get_last_training_config() is None


def test_records_arguments_as_passed():
    session = _session()
    session.train_model(**USER_CONFIG)
    assert session.get_last_training_config() == USER_CONFIG


def test_records_explicit_none_transforms():
    # For BoTorch an explicit None means "no transform"; omitting the key means
    # "apply the normalize/standardize defaults". The record must keep the None.
    session = _session()
    session.train_model(backend="botorch", kernel="RBF",
                        input_transform_type=None, output_transform_type=None)
    config = session.get_last_training_config()
    assert config["input_transform_type"] is None
    assert config["output_transform_type"] is None

    session.train_model(**config)
    assert getattr(session.model.model, "input_transform", None) is None
    assert getattr(session.model.model, "outcome_transform", None) is None


def test_replaying_config_rebuilds_the_same_model():
    session = _session()
    session.train_model(**USER_CONFIG)
    session.add_experiment(inputs={"x1": 0.5, "x2": 0.5}, output=0.75)
    session.train_model(**session.get_last_training_config())
    _assert_user_model(session)


def test_returns_a_copy():
    session = _session()
    session.train_model(**USER_CONFIG)
    config = session.get_last_training_config()
    config["kernel_params"]["nu"] = 0.5
    assert session.get_last_training_config()["kernel_params"] == {"nu": 2.5}


def test_failed_training_keeps_previous_config():
    session = _session()
    session.train_model(**USER_CONFIG)
    with pytest.raises(ValueError):
        session.train_model(backend="no_such_backend", kernel="RBF")
    assert session.get_last_training_config() == USER_CONFIG


def test_config_survives_save_and_load_with_retrain(tmp_path):
    session = _session()
    session.train_model(**USER_CONFIG)
    path = tmp_path / "session.json"
    session.save_session(str(path))

    loaded = OptimizationSession.load_session(str(path))
    assert loaded.get_last_training_config() == USER_CONFIG
    # Load-time retrain now uses the full config (transforms included).
    _assert_user_model(loaded)


def test_instance_load_without_retrain_restores_config(tmp_path):
    session = _session()
    session.train_model(**USER_CONFIG)
    path = tmp_path / "session.json"
    session.save_session(str(path))

    target = OptimizationSession()
    target.load_session(str(path), retrain_on_load=False)
    assert target.model is None
    # A later auto-train can still rebuild the user's model.
    assert target.get_last_training_config() == USER_CONFIG
