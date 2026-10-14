"""Posterior/prior std at a suggestion, and the warning it drives."""
import numpy as np
import pytest

from alchemist_core import OptimizationSession
from alchemist_core.acquisition.diagnostics import knowledge_ratio

# Library noise, not behaviour under test: BoTorch warns that qEI is the legacy
# form (the session's qEI strategy builds it), and pyro (imported by the
# GP stack) has an invalid escape in a docstring.
pytestmark = [
    pytest.mark.filterwarnings("ignore:qExpectedImprovement has known numerical issues"),
    pytest.mark.filterwarnings("ignore:invalid escape sequence:SyntaxWarning"),
]


def _six_input_session(n=40, seed=5):
    """Three inputs held per campaign plus three free ones; data covers a corner."""
    rng = np.random.default_rng(seed)
    s = OptimizationSession()
    for name in ("h1", "h2", "h3"):
        s.add_variable(name, "real", min=0, max=100, changes="very_hard")
    for name in ("x1", "x2", "x3"):
        s.add_variable(name, "real", min=0, max=1)
    for _ in range(n):
        h = rng.uniform(0, 20, 3)
        x = rng.uniform(0, 1, 3)
        # The held inputs need real curvature: a near-linear response fits with
        # very long lengthscales, and the model then extrapolates (correctly, by
        # its own account) with a low ratio even far from the data.
        y = -np.sum((x - 0.5) ** 2) + 0.3 * np.sin(h / 4).sum()
        s.add_experiment({"h1": h[0], "h2": h[1], "h3": h[2], "x1": x[0], "x2": x[1], "x3": x[2]},
                         float(y + rng.normal(0, 0.01)))
    s.train_model(backend="botorch")
    return s


def test_ratio_is_low_at_training_data_and_near_one_far_away():
    s = _six_input_session()
    X = s.experiment_manager.get_features_and_target()[0].to_numpy(dtype=float)
    near = knowledge_ratio(s.model, X[:3])
    far = knowledge_ratio(s.model, np.array([[95, 95, 95, 0.5, 0.5, 0.5]], dtype=float))
    assert max(near) < 0.6
    assert far[0] > 0.9


def test_holding_three_inputs_keeps_them_and_varies_the_rest():
    s = _six_input_session()
    out = s.suggest_next(strategy="EI", hold={"h1": 10.0, "h2": 5.0, "h3": 12.0})
    row = out.iloc[0]
    assert (row["h1"], row["h2"], row["h3"]) == (10.0, 5.0, 12.0)
    diag = s.last_suggestion_diagnostics[0]
    assert diag["held"] == ["h1", "h2", "h3"]
    assert diag["knowledge_ratio"] < 0.9 and diag["warnings"] == []


def test_held_values_far_from_data_warn():
    s = _six_input_session()
    s.suggest_next(strategy="EI", hold={"h1": 95.0, "h2": 95.0, "h3": 95.0})
    diag = s.last_suggestion_diagnostics[0]
    assert diag["knowledge_ratio"] > 0.9
    assert any("little information" in w for w in diag["warnings"])


def test_no_warning_without_holds_even_when_exploring():
    s = _six_input_session()
    s.suggest_next(strategy="UCB")
    diag = s.last_suggestion_diagnostics[0]
    assert diag["held"] == [] and diag["warnings"] == []
    assert diag["knowledge_ratio"] is not None


def test_holding_an_unseen_categorical_level_warns_but_works():
    rng = np.random.default_rng(0)
    s = OptimizationSession()
    s.add_variable("x1", "real", min=0, max=1)
    s.add_variable("c", "categorical", values=["a", "b", "c"], changes="hard")
    for i in range(12):
        c = "ab"[i % 2]
        x1 = float(rng.uniform())
        s.add_experiment({"x1": x1, "c": c}, float(x1 + (1 if c == "b" else 0)))
    s.train_model(backend="botorch")
    s.suggest_next(strategy="EI", hold={"c": "b"})
    seen = s.last_suggestion_diagnostics[0]["knowledge_ratio"]
    out = s.suggest_next(strategy="EI", hold={"c": "c"})
    assert out.iloc[0]["c"] == "c"
    unseen = s.last_suggestion_diagnostics[0]["knowledge_ratio"]
    assert unseen is not None and unseen > seen


def test_batch_diagnostics_align_with_rows():
    s = _six_input_session()
    out = s.suggest_next(strategy="qEI", n_suggestions=3, hold={"h1": 10.0})
    assert len(s.last_suggestion_diagnostics) == len(out) == 3
