"""Unit tests for BoTorchAcquisition context variable support."""
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch


def _make_mock_model(n_tunable=2, n_context=1):
    """Create a minimal mock BoTorchModel for testing."""
    from alchemist_core.models.botorch_model import BoTorchModel
    mock = MagicMock(spec=BoTorchModel)
    mock.is_trained = True
    # feature names: tunable first, then context
    all_names = [f"x{i}" for i in range(n_tunable)] + [f"ctx{i}" for i in range(n_context)]
    mock.original_feature_names = all_names
    mock.feature_names = all_names
    mock.categorical_encodings = {}
    mock.Y_orig = torch.tensor([[0.5], [0.8]], dtype=torch.float64)
    # Mock the internal GP model
    mock_gp = MagicMock()
    mock_gp.train_targets = torch.tensor([0.5, 0.8], dtype=torch.float64)
    mock.model = mock_gp
    return mock


def _make_search_space(n_tunable=2, n_context=1):
    from alchemist_core.data.search_space import SearchSpace
    ss = SearchSpace()
    for i in range(n_tunable):
        ss.add_variable(f"x{i}", "real", min=0.0, max=1.0)
    for i in range(n_context):
        ss.add_variable(f"ctx{i}", "context")
    return ss


def test_get_bounds_excludes_context_vars():
    """_get_bounds_from_search_space() returns (2, n_tunable) tensor, not (2, n_total)."""
    from alchemist_core.acquisition.botorch_acquisition import BoTorchAcquisition

    ss = _make_search_space(n_tunable=2, n_context=1)

    acq = BoTorchAcquisition.__new__(BoTorchAcquisition)
    acq.search_space_obj = ss
    acq.model = _make_mock_model(n_tunable=2, n_context=1)
    acq.acq_func_name = "ei"

    bounds = acq._get_bounds_from_search_space()
    assert bounds.shape == (2, 2), f"Expected (2, 2), got {bounds.shape}"


def test_get_bounds_no_context_unchanged():
    """Without context vars, bounds shape is (2, n_tunable) as before."""
    from alchemist_core.acquisition.botorch_acquisition import BoTorchAcquisition

    from alchemist_core.data.search_space import SearchSpace
    ss = SearchSpace()
    ss.add_variable("T", "real", min=200.0, max=320.0)
    ss.add_variable("P", "real", min=20.0, max=80.0)

    acq = BoTorchAcquisition.__new__(BoTorchAcquisition)
    acq.search_space_obj = ss
    mock_model = MagicMock()
    mock_model.original_feature_names = ["T", "P"]
    mock_model.feature_names = ["T", "P"]
    mock_model.categorical_encodings = {}
    acq.model = mock_model
    acq.acq_func_name = "ei"

    bounds = acq._get_bounds_from_search_space()
    assert bounds.shape == (2, 2)


def test_select_next_raises_without_context_when_context_vars_present():
    """select_next() raises ValueError when context vars are registered but context_values is None."""
    from alchemist_core.acquisition.botorch_acquisition import BoTorchAcquisition

    ss = _make_search_space(n_tunable=2, n_context=1)
    acq = BoTorchAcquisition.__new__(BoTorchAcquisition)
    acq.search_space_obj = ss
    acq.model = _make_mock_model()
    acq.acq_func_name = "ei"
    acq.acq_function = MagicMock()
    acq.batch_size = 1

    with pytest.raises(ValueError, match="context_values"):
        acq.select_next(context_values=None)


def test_select_next_raises_with_missing_context_values():
    """select_next() raises ValueError when context_values is missing a registered context var."""
    from alchemist_core.acquisition.botorch_acquisition import BoTorchAcquisition

    ss = _make_search_space(n_tunable=2, n_context=2)  # ctx0 and ctx1
    acq = BoTorchAcquisition.__new__(BoTorchAcquisition)
    acq.search_space_obj = ss
    acq.model = _make_mock_model(n_tunable=2, n_context=2)
    acq.acq_func_name = "ei"
    acq.acq_function = MagicMock()
    acq.batch_size = 1

    with pytest.raises(ValueError, match="ctx1"):
        acq.select_next(context_values={"ctx0": 0.5})  # missing ctx1


def test_select_next_raises_for_qipv_with_context():
    """select_next() raises ValueError for qIPV when context vars are present."""
    from alchemist_core.acquisition.botorch_acquisition import BoTorchAcquisition

    ss = _make_search_space(n_tunable=2, n_context=1)
    acq = BoTorchAcquisition.__new__(BoTorchAcquisition)
    acq.search_space_obj = ss
    acq.model = _make_mock_model()
    acq.acq_func_name = "qipv"
    acq.acq_function = MagicMock()
    acq.batch_size = 1

    with pytest.raises(ValueError, match="qIPV"):
        acq.select_next(context_values={"ctx0": 0.42})


def test_fixed_feature_acq_function_is_constructed():
    """FixedFeatureAcquisitionFunction is created when context_values provided."""
    from botorch.acquisition import FixedFeatureAcquisitionFunction
    from alchemist_core.acquisition.botorch_acquisition import BoTorchAcquisition

    ss = _make_search_space(n_tunable=2, n_context=1)

    acq = BoTorchAcquisition.__new__(BoTorchAcquisition)
    acq.search_space_obj = ss
    acq.model = _make_mock_model()
    acq.acq_func_name = "ei"
    acq.batch_size = 1

    # Capture FixedFeatureAcquisitionFunction construction
    constructed = []

    original_cls = FixedFeatureAcquisitionFunction

    class CapturingFFAF(original_cls):
        def __init__(self, *args, **kwargs):
            constructed.append(kwargs)
            super().__init__(*args, **kwargs)

    with patch(
        "alchemist_core.acquisition.botorch_acquisition.FixedFeatureAcquisitionFunction",
        CapturingFFAF,
    ):
        mock_acq = MagicMock()
        mock_acq.model = MagicMock()
        acq.acq_function = mock_acq
        try:
            acq.select_next(context_values={"ctx0": 0.42})
        except Exception:
            pass  # optimization will fail with mocks; we only need to check construction

    assert len(constructed) == 1
    assert constructed[0]["d"] == 3  # 2 tunable + 1 context
    assert constructed[0]["columns"] == [2]  # ctx0 is at index 2 (tunable-first)
    assert constructed[0]["values"] == [0.42]
