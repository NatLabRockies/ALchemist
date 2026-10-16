"""BoTorch ``input_transform_type='standardize'`` builds ``InputStandardize`` (T6).

Both ``'normalize'`` and ``'standardize'`` used to build ``Normalize``, which
rescales to [0, 1]; standardize is zero mean, unit variance.
"""
import pytest

from alchemist_core import OptimizationSession


def _session():
    s = OptimizationSession()
    s.add_variable("x", "real", bounds=(0.0, 1.0))
    s.add_variable("y", "real", bounds=(0.0, 1.0))
    for x, y in [(0.1, 0.2), (0.4, 0.8), (0.7, 0.3), (0.9, 0.9), (0.3, 0.6), (0.5, 0.1)]:
        s.add_experiment({"x": x, "y": y}, (x - 0.3) ** 2 + (y - 0.7) ** 2)
    return s


@pytest.mark.parametrize("transform,cls_name", [
    ("normalize", "Normalize"), ("standardize", "InputStandardize"),
])
def test_input_transform_class(transform, cls_name):
    s = _session()
    s.train_model(backend="botorch", input_transform_type=transform)
    assert type(s.model.model.input_transform).__name__ == cls_name
    s.suggest_next(strategy="LogEI", goal="minimize")
