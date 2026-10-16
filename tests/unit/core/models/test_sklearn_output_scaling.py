"""scikit-learn output scaling: every listed scaler is built (T17) and the
noise variance is scaled with the output (T18).

``_initialize_scalers`` built a scaler for ``output_transform_type='standard'``
only; ``'minmax'`` and ``'robust'`` fell through to ``None`` while the CV
fold path built them, so training and CV disagreed. And the Noise column
was handed to ``GaussianProcessRegressor`` as ``alpha`` in the output's raw
units while ``y`` was standardised: with y spanning thousands, a noise of
4.0 was passed where 7e-5 was right, telling the GP the noise was 5e4 times
larger than it was.
"""
import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from alchemist_core import OptimizationSession


def _session(noise=None, scale=1000.0):
    s = OptimizationSession()
    s.add_variable("x", "real", bounds=(0.0, 1.0))
    s.add_variable("y", "real", bounds=(0.0, 1.0))
    pts = [(0.1, 0.2), (0.4, 0.8), (0.7, 0.3), (0.9, 0.9), (0.3, 0.6), (0.5, 0.1), (0.8, 0.5)]
    for x, y in pts:
        s.add_experiment({"x": x, "y": y}, scale * ((x - 0.3) ** 2 + (y - 0.7) ** 2), noise=noise)
    return s


@pytest.mark.parametrize("transform,cls", [
    ("standard", StandardScaler), ("minmax", MinMaxScaler), ("robust", RobustScaler),
])
def test_output_scaler_is_built_for_every_listed_transform(transform, cls):
    s = _session()
    s.train_model(backend="sklearn", output_transform_type=transform)
    assert isinstance(s.model.output_scaler, cls)
    # Predictions come back in the output's own units.
    X, y = s.experiment_manager.get_features_and_target()
    pred = np.asarray(s.model.predict(X))
    np.testing.assert_allclose(pred, y.to_numpy(), rtol=0.05, atol=5.0)


def test_none_leaves_no_output_scaler():
    s = _session()
    s.train_model(backend="sklearn", output_transform_type="none")
    assert s.model.output_scaler is None


def test_alpha_is_scaled_by_the_standard_scaler_variance():
    s = _session(noise=4.0)
    s.train_model(backend="sklearn", output_transform_type="standard")
    scale = float(s.model.output_scaler.scale_[0])
    np.testing.assert_allclose(s.model.model.alpha, np.full(7, 4.0 / scale**2))
    assert s.model.model.alpha[0] < 1e-3


def test_alpha_is_scaled_by_the_robust_scaler_variance():
    s = _session(noise=4.0)
    s.train_model(backend="sklearn", output_transform_type="robust")
    scale = float(s.model.output_scaler.scale_[0])
    np.testing.assert_allclose(s.model.model.alpha, np.full(7, 4.0 / scale**2))


def test_alpha_is_scaled_by_the_minmax_scaler_variance():
    # MinMaxScaler multiplies by scale_ (= 1 / range), so the variance is
    # multiplied by scale_**2.
    s = _session(noise=4.0)
    s.train_model(backend="sklearn", output_transform_type="minmax")
    scale = float(s.model.output_scaler.scale_[0])
    np.testing.assert_allclose(s.model.model.alpha, np.full(7, 4.0 * scale**2))


def test_alpha_is_raw_with_no_output_scaling():
    s = _session(noise=4.0)
    s.train_model(backend="sklearn", output_transform_type="none")
    np.testing.assert_allclose(s.model.model.alpha, np.full(7, 4.0))
