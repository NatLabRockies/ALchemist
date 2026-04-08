"""Additional unit tests targeting BoTorch acquisition edge cases."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from alchemist_core.acquisition import botorch_acquisition as botorch_module
from alchemist_core.acquisition.botorch_acquisition import BoTorchAcquisition
from alchemist_core.data.search_space import SearchSpace

from tests.unit.core.acquisition.test_botorch_acquisition_extended import (
    trained_session_botorch,
)


def _make_acquisition(session, strategy="ei", batch_size=1, **kwargs):
    return BoTorchAcquisition(
        session.search_space,
        model=session.model,
        acq_func=strategy,
        batch_size=batch_size,
        acq_func_kwargs=kwargs or None,
    )


def _simple_search_space():
    space = SearchSpace()
    space.add_variable("x", "real", min=0.0, max=1.0)
    return space


def test_invalid_acquisition_function_rejected():
    space = _simple_search_space()
    with pytest.raises(ValueError):
        BoTorchAcquisition(space, model=None, acq_func="not-a-real-acq")


@pytest.mark.parametrize(
    "strategy, expected_kwargs",
    [
        ("ucb", {"beta": 0.5}),
        ("qucb", {"beta": 0.5, "mc_samples": 128}),
        ("qei", {"mc_samples": 128}),
        ("qipv", {"mc_samples": 128}),
    ],
)
def test_default_acq_kwargs_populated(strategy, expected_kwargs):
    space = _simple_search_space()
    acquisition = BoTorchAcquisition(space, model=None, acq_func=strategy)

    for key, value in expected_kwargs.items():
        assert acquisition.acq_func_kwargs[key] == value


@pytest.mark.parametrize(
    "strategy, kwargs, expected_cls",
    [
        ("ei", {}, botorch_module.ExpectedImprovement),
        ("logei", {}, botorch_module.LogExpectedImprovement),
        ("pi", {}, botorch_module.ProbabilityOfImprovement),
        ("logpi", {}, botorch_module.LogProbabilityOfImprovement),
        ("ucb", {"beta": 1.0}, botorch_module.UpperConfidenceBound),
        ("qei", {"mc_samples": 4}, botorch_module.qExpectedImprovement),
        ("qucb", {"beta": 0.7, "mc_samples": 4}, botorch_module.qUpperConfidenceBound),
        ("qipv", {"n_mc_points": 8}, botorch_module.qNegIntegratedPosteriorVariance),
    ],
)
def test_acquisition_function_types(strategy, kwargs, expected_cls, trained_session_botorch):
    session = trained_session_botorch
    acquisition = BoTorchAcquisition(
        session.search_space,
        model=session.model,
        acq_func=strategy,
        acq_func_kwargs=kwargs,
    )

    assert isinstance(acquisition.acq_function, expected_cls)


def test_create_acq_function_uses_train_targets_when_no_y_orig(trained_session_botorch):
    session = trained_session_botorch
    acquisition = _make_acquisition(session, strategy="ei")
    acquisition.model.Y_orig = None

    acquisition._create_acquisition_function()

    assert isinstance(acquisition.acq_function, botorch_module.ExpectedImprovement)


def test_create_acq_function_handles_missing_train_targets():
    space = _simple_search_space()
    acquisition = BoTorchAcquisition(space, model=None, acq_func="ucb")
    acquisition.model = SimpleNamespace(model=SimpleNamespace(), is_trained=True)
    acquisition.acq_func_name = "unsupported"

    with pytest.raises(ValueError, match="Unsupported acquisition function"):
        acquisition._create_acquisition_function()


def test_select_next_requires_acquisition_function(monkeypatch):
    space = _simple_search_space()
    acquisition = BoTorchAcquisition(space, model=None)
    acquisition.model = SimpleNamespace(is_trained=False)
    with pytest.raises(ValueError, match="Could not create acquisition function"):
        acquisition.select_next()


def test_update_model_rejects_non_botorch(trained_session_botorch):
    session = trained_session_botorch
    acquisition = _make_acquisition(session, strategy="ucb")

    with pytest.raises(ValueError):
        acquisition.update_model(object())


def test_select_next_with_dataframe_candidates(monkeypatch, trained_session_botorch):
    session = trained_session_botorch
    acquisition = _make_acquisition(session, strategy="ei")

    # Replace heavy BoTorch acquisition function with deterministic stub.
    def fake_acq(points: torch.Tensor) -> torch.Tensor:
        # points shape: (n, 1, d); return increasing values so last row is chosen.
        n = points.shape[0]
        return torch.arange(1, n + 1, dtype=torch.double).view(n, 1)

    acquisition.acq_function = fake_acq

    original_argmax = botorch_module.torch.argmax

    def argmax_python(tensor, *args, **kwargs):
        return int(original_argmax(tensor, *args, **kwargs).item())

    monkeypatch.setattr(botorch_module.torch, "argmax", argmax_python)

    feature_names = session.model.original_feature_names
    candidates = pd.DataFrame(
        [
            {feature_names[0]: 360.0, feature_names[1]: "High SAR", feature_names[2]: 1.0, feature_names[3]: 0.2},
            {feature_names[0]: 365.0, feature_names[1]: "Low SAR", feature_names[2]: 1.5, feature_names[3]: 0.3},
        ]
    )

    best = acquisition.select_next(candidate_points=candidates)

    assert best[feature_names[0]] == 365.0
    assert best[feature_names[1]] == "Low SAR"


def test_select_next_falls_back_to_standard_optimization(monkeypatch, trained_session_botorch):
    session = trained_session_botorch
    acquisition = _make_acquisition(session, strategy="qei", batch_size=2)

    def fake_acq(points: torch.Tensor) -> torch.Tensor:
        n = points.shape[0]
        return torch.zeros(n, 1, dtype=torch.double)

    acquisition.acq_function = fake_acq

    # Force both mixed optimizers to fail so the standard fallback path executes.
    def raise_mixed(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(botorch_module, "optimize_acqf_mixed_alternating", raise_mixed)
    monkeypatch.setattr(botorch_module, "optimize_acqf_mixed", raise_mixed)

    def fake_optimize_acqf(**_kwargs):
        tensor = torch.tensor(
            [[360.0, 0.0, 1.7, 0.1], [370.0, 1.0, 2.4, 0.2]],
            dtype=torch.double,
        )
        return tensor, torch.tensor([0.0, 0.0], dtype=torch.double)

    monkeypatch.setattr(botorch_module, "optimize_acqf", fake_optimize_acqf)
    monkeypatch.setattr(session.search_space, "get_integer_variables", lambda: ["Metal Loading"])

    results = acquisition.select_next()

    assert isinstance(results, list)
    assert {r["Catalyst"] for r in results} <= {"High SAR", "Low SAR"}


def test_find_optimum_uses_grid_search(trained_session_botorch):
    """Test that find_optimum uses grid search and model.predict() correctly."""
    session = trained_session_botorch
    acquisition = _make_acquisition(session, strategy="ucb")

    # find_optimum should use model.predict() which handles encoding properly
    result = acquisition.find_optimum(model=session.model, maximize=False, random_state=7)

    assert set(result.keys()) == {"x_opt", "value", "std"}
    assert isinstance(result["x_opt"], pd.DataFrame)
    assert not result["x_opt"].empty
    # Check that categorical variables are in original space (not encoded)
    assert result["x_opt"]["Catalyst"].iloc[0] in {"High SAR", "Low SAR"}


def test_select_next_mixed_success_path(monkeypatch, trained_session_botorch):
    session = trained_session_botorch
    acquisition = _make_acquisition(session, strategy="qei", batch_size=2)

    def fixed_mixed(**_kwargs):
        tensor = torch.tensor(
            [
                [360.4, 0.3, 1.9, 0.15],
                [365.2, 1.7, 2.2, 0.25],
            ],
            dtype=torch.double,
        )
        return tensor, torch.tensor([0.1, 0.2], dtype=torch.double)

    def forbid_fallback(**_kwargs):
        raise AssertionError("optimize_acqf fallback should not run")

    monkeypatch.setattr(botorch_module, "optimize_acqf_mixed", fixed_mixed)
    monkeypatch.setattr(botorch_module, "optimize_acqf", forbid_fallback)
    monkeypatch.setattr(session.search_space, "get_integer_variables", lambda: ["Metal Loading"])

    results = acquisition.select_next()

    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(entry["Metal Loading"], int) for entry in results)
    assert all(entry["Catalyst"] in {"High SAR", "Low SAR"} for entry in results)


def test_select_next_qipv_uses_custom_options(monkeypatch, trained_session_botorch):
    session = trained_session_botorch
    acquisition = _make_acquisition(session, strategy="qipv", batch_size=1, n_mc_points=6)

    def fake_optimize_acqf_mixed(**kwargs):
        assert kwargs["num_restarts"] == 20
        assert kwargs["raw_samples"] == 100
        assert kwargs["options"]["maxiter"] == 150
        tensor = torch.tensor([[360.0, 0.3, 1.2, 0.15]], dtype=torch.double)
        return tensor, torch.tensor([0.3], dtype=torch.double)

    monkeypatch.setattr(botorch_module, "optimize_acqf_mixed", fake_optimize_acqf_mixed)

    def forbid_standard(**_kwargs):
        raise AssertionError("standard optimizer should not run")

    monkeypatch.setattr(botorch_module, "optimize_acqf", forbid_standard)

    result = acquisition.select_next()

    assert isinstance(result, dict)


def test_select_next_with_numpy_candidates(monkeypatch, trained_session_botorch):
    session = trained_session_botorch
    acquisition = _make_acquisition(session, strategy="ei")

    def fake_acq(points: torch.Tensor) -> torch.Tensor:
        n = points.shape[0]
        return torch.linspace(1.0, float(n), steps=n, dtype=torch.double).view(n, 1)

    acquisition.acq_function = fake_acq

    candidates = np.array(
        [
            [360.0, 0.0, 1.0, 0.1],
            [366.0, 1.0, 1.5, 0.2],
        ],
        dtype=float,
    )

    best = acquisition.select_next(candidate_points=candidates)

    assert pytest.approx(best[0], rel=1e-6) == 366.0


def test_select_next_with_tensor_candidates(monkeypatch, trained_session_botorch):
    session = trained_session_botorch
    acquisition = _make_acquisition(session, strategy="ei")

    def fake_acq(points: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[0.1], [0.9]], dtype=torch.double)

    acquisition.acq_function = fake_acq

    candidates = torch.tensor(
        [[360.0, 0.0, 1.0, 0.1], [362.0, 1.0, 1.2, 0.2]],
        dtype=torch.double,
    )

    best = acquisition.select_next(candidate_points=candidates)

    assert torch.allclose(best, candidates[1])


def test_select_next_continuous_variables(monkeypatch, trained_session_botorch):
    session = trained_session_botorch
    acquisition = _make_acquisition(session, strategy="ei")

    monkeypatch.setattr(session.search_space, "get_categorical_variables", lambda: [])
    monkeypatch.setattr(session.search_space, "get_integer_variables", lambda: ["Temperature"])

    def fake_optimize_acqf(**_kwargs):
        tensor = torch.tensor([[351.6, 0.0, 1.0, 0.2]], dtype=torch.double)
        return tensor, torch.tensor([0.0], dtype=torch.double)

    monkeypatch.setattr(botorch_module, "optimize_acqf", fake_optimize_acqf)

    result = acquisition.select_next()

    assert isinstance(result, dict)
    assert result["Temperature"] == 352


def test_select_next_single_point_categorical_conversion(monkeypatch, trained_session_botorch):
    session = trained_session_botorch
    acquisition = _make_acquisition(session, strategy="qei", batch_size=1)

    def fake_optimize_acqf_mixed(**_kwargs):
        tensor = torch.tensor([[360.0, 1.0, 1.0, 0.1]], dtype=torch.double)
        return tensor, torch.tensor([0.3], dtype=torch.double)

    def forbid_fallback(**_kwargs):
        raise AssertionError("fallback should not run")

    monkeypatch.setattr(botorch_module, "optimize_acqf_mixed", fake_optimize_acqf_mixed)
    monkeypatch.setattr(botorch_module, "optimize_acqf", forbid_fallback)

    result = acquisition.select_next()

    assert isinstance(result, dict)
    assert result["Catalyst"] in {"High SAR", "Low SAR"}


def test_get_bounds_uses_tensor_directly(trained_session_botorch):
    session = trained_session_botorch
    acquisition = _make_acquisition(session)

    tensor_bounds = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=torch.double)
    acquisition.search_space_obj = SimpleNamespace(to_botorch_bounds=lambda: tensor_bounds)

    bounds = acquisition._get_bounds_from_search_space()

    assert torch.equal(bounds, tensor_bounds)


def test_get_bounds_requires_original_feature_names():
    """Falls back to original_feature_names when search space has no get_tunable_variable_names."""
    from types import SimpleNamespace as SN
    # Minimal search space: has variables dict but no get_tunable_variable_names helper
    minimal_space = SN(
        to_botorch_bounds=lambda: None,
        variables=[{"name": "x", "type": "real", "min": 0.0, "max": 1.0}],
        get_context_variable_names=lambda: [],
        # deliberately no get_tunable_variable_names
    )
    acq = BoTorchAcquisition.__new__(BoTorchAcquisition)
    acq.search_space_obj = minimal_space
    acq.model = SimpleNamespace()  # no original_feature_names

    with pytest.raises(ValueError, match="original_feature_names"):
        acq._get_bounds_from_search_space()


def test_get_bounds_handles_mixed_variable_metadata(monkeypatch, trained_session_botorch):
    session = trained_session_botorch
    acquisition = _make_acquisition(session)

    custom_variables = [
        {"name": "Temperature", "type": "real", "min": 300, "max": 400},
        {"name": "Catalyst", "type": "categorical"},
        {"name": "Metal Loading", "type": "integer", "bounds": [0, 3]},
    ]

    def get_categorical_variables():
        return ["Catalyst"]

    acquisition.model.categorical_encodings.pop("Catalyst", None)
    acquisition.search_space_obj = SimpleNamespace(
        to_botorch_bounds=lambda: None,
        variables=custom_variables,
        get_categorical_variables=get_categorical_variables,
    )

    bounds = acquisition._get_bounds_from_search_space()

    assert bounds.shape == (2, len(acquisition.model.original_feature_names))


def test_update_invokes_model_update(monkeypatch, trained_session_botorch):
    session = trained_session_botorch
    acquisition = _make_acquisition(session)

    calls = {}

    def fake_update(X, y):
        calls["args"] = (X, y)

    recreated = {"called": False}

    def mark_recreated():
        recreated["called"] = True

    monkeypatch.setattr(acquisition.model, "update", fake_update, raising=False)
    monkeypatch.setattr(acquisition, "_create_acquisition_function", mark_recreated)

    X = pd.DataFrame({"Temperature": [350.0]})
    y = pd.Series([0.2])

    result = acquisition.update(X=X, y=y)

    assert result is acquisition
    assert "args" in calls
    assert recreated["called"]


def test_find_optimum_maximize_returns_valid_result(trained_session_botorch):
    """Test that find_optimum returns valid results for maximization."""
    session = trained_session_botorch
    acquisition = _make_acquisition(session, strategy="ucb")

    result = acquisition.find_optimum()

    assert isinstance(result["x_opt"], pd.DataFrame)
    assert isinstance(result["value"], float)
    assert isinstance(result["std"], float)
    # Check that result is in original variable space
    assert "Catalyst" in result["x_opt"].columns
    assert result["x_opt"]["Catalyst"].iloc[0] in {"High SAR", "Low SAR"}


def test_find_optimum_handles_minimization(trained_session_botorch):
    """Test that find_optimum correctly handles minimization."""
    session = trained_session_botorch
    acquisition = _make_acquisition(session, strategy="ucb")

    result = acquisition.find_optimum(maximize=False)

    assert isinstance(result["x_opt"], pd.DataFrame)
    assert isinstance(result["value"], float)
    assert isinstance(result["std"], float)
    # Categorical values should be in original space
    assert result["x_opt"]["Catalyst"].iloc[0] in {"High SAR", "Low SAR"}


def test_find_optimum_generates_grid_correctly(trained_session_botorch):
    """Test that find_optimum generates a proper grid across the search space."""
    session = trained_session_botorch
    acquisition = _make_acquisition(session, strategy="ucb")

    result = acquisition.find_optimum()

    # Check result structure
    assert isinstance(result["x_opt"], pd.DataFrame)
    assert not result["x_opt"].empty
    
    # Verify all variables are present
    expected_vars = {var['name'] for var in session.search_space.variables}
    assert set(result["x_opt"].columns) == expected_vars
    
    # Verify categorical variables are not encoded (should be strings)
    assert isinstance(result["x_opt"]["Catalyst"].iloc[0], str)
    assert result["x_opt"]["Catalyst"].iloc[0] in {"High SAR", "Low SAR"}