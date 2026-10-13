"""WholePlot / Subplot / RunOrder travel from staging to the dataset as metadata."""
import pytest

from alchemist_core.session import OptimizationSession


def _session():
    s = OptimizationSession()
    s.add_variable("x1", "real", min=0, max=10)
    s.add_variable("c", "categorical", values=["a", "b"], changes="hard")
    s.add_variable("z", "real", role="observed")
    return s


def test_staging_needs_only_controllable_variables():
    s = _session()
    item = s.add_staged_experiment({"x1": 1.0, "c": "a"})
    assert item.inputs == {"x1": 1.0, "c": "a"}


def test_staging_still_requires_every_controllable_variable():
    with pytest.raises(ValueError, match="Missing"):
        _session().add_staged_experiment({"x1": 1.0})


def test_plot_keys_move_into_item_plot_and_back():
    s = _session()
    item = s.add_staged_experiment(
        {"x1": 1.0, "c": "a", "WholePlot": 2, "RunOrder": 5, "_reason": "Initial DoE (split-plot)"}
    )
    assert item.inputs == {"x1": 1.0, "c": "a"}
    assert item.plot == {"WholePlot": 2, "RunOrder": 5}
    staged = s.get_staged_experiments()[0]
    assert staged["WholePlot"] == 2 and staged["RunOrder"] == 5
    assert staged["_reason"] == "Initial DoE (split-plot)"


def test_completion_writes_plot_columns_and_they_are_not_model_inputs():
    s = _session()
    item = s.add_staged_experiment({"x1": 1.0, "c": "a", "WholePlot": 3, "RunOrder": 7})
    s.complete_experiment(item.id, actual_inputs={"x1": 1.1, "c": "a", "z": 25.0}, output=0.5)
    df = s.experiment_manager.df
    assert int(df["WholePlot"].iloc[-1]) == 3 and int(df["RunOrder"].iloc[-1]) == 7
    for col in ("WholePlot", "RunOrder"):
        assert col in s.experiment_manager.metadata_columns()
    X, _ = s.experiment_manager.get_features_and_target()
    assert list(X.columns) == ["x1", "c", "z"]


def test_completion_without_observed_value_fails():
    s = _session()
    item = s.add_staged_experiment({"x1": 1.0, "c": "a"})
    with pytest.raises(ValueError, match="z"):
        s.complete_experiment(item.id, actual_inputs={"x1": 1.0, "c": "a"}, output=0.5)


def test_add_experiment_plot_argument():
    s = _session()
    s.add_experiment({"x1": 1.0, "c": "b", "z": 20.0}, output=1.0, plot={"WholePlot": 1, "Subplot": 2})
    row = s.experiment_manager.df.iloc[-1]
    assert int(row["WholePlot"]) == 1 and int(row["Subplot"]) == 2


def test_add_experiment_rejects_unknown_plot_keys():
    with pytest.raises(ValueError, match="plot"):
        _session().add_experiment({"x1": 1.0, "c": "b", "z": 20.0}, output=1.0, plot={"Block": 1})


def test_queue_item_round_trips_with_and_without_plot():
    from alchemist_core.queue import QueueItem
    old = {"id": "q1", "inputs": {"x1": 1.0}}
    assert QueueItem.from_dict(old).plot is None
    item = QueueItem(id="q2", inputs={"x1": 1.0}, plot={"WholePlot": 1})
    assert QueueItem.from_dict(item.to_dict()).plot == {"WholePlot": 1}
