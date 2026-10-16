"""``load_search_space`` hands the new space to the ExperimentManager (A28).

It set ``self.search_space`` and left ``experiment_manager.search_space`` on
the old, empty one, so rows added afterwards with ``add_experiment`` had no
inputs as far as training was concerned: scikit-learn failed with "0
feature(s)" and BoTorch with a misleading ``active_dims`` state-dict error.
"""
import pytest

from alchemist_core import OptimizationSession


@pytest.fixture
def space_file(tmp_path):
    s = OptimizationSession()
    s.add_variable("x", "real", bounds=(0.0, 1.0))
    s.add_variable("y", "real", bounds=(0.0, 1.0))
    path = tmp_path / "space.json"
    s.search_space.save_to_json(str(path))
    return str(path)


def test_experiment_manager_sees_the_loaded_space(space_file):
    s = OptimizationSession()
    s.load_search_space(space_file)
    assert s.experiment_manager.search_space is s.search_space
    assert s.experiment_manager.search_space.get_variable_names() == ["x", "y"]


@pytest.mark.parametrize("backend", ["sklearn", "botorch"])
def test_add_experiment_then_train_after_load_search_space(space_file, backend):
    s = OptimizationSession()
    s.load_search_space(space_file)
    for i, (x, y) in enumerate([(0.1, 0.2), (0.4, 0.8), (0.7, 0.3), (0.9, 0.9), (0.3, 0.6), (0.5, 0.1)]):
        s.add_experiment({"x": x, "y": y}, (x - 0.3) ** 2 + (y - 0.7) ** 2)
    s.train_model(backend=backend)
    assert s.model is not None
    assert list(s.model.original_feature_names) == ["x", "y"]
