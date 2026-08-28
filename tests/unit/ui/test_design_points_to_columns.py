"""The desktop UI's pending-suggestions frame builder must skip context variables.

``_generate_points`` built its column dict by iterating
``session.search_space.variables`` and doing ``p[name]`` on every design point.
That raises ``KeyError`` for any space carrying a ``context`` variable, and it
did so *both* before and after the core ``doe.py`` fix, with different keys --
which is why the two had to be fixed together:

    today (broken core)  points {x1, c1}, names [x1, c1, x2] -> KeyError: 'x2'
    core fixed alone     points {x1, x2}, names [x1, c1, x2] -> KeyError: 'c1'

Both were reproduced by executing ``ui/ui.py``'s two lines verbatim before the
fix. The ``try/except ValueError`` around the design call in ``_generate_points``
does not cover either one.

The helper is module-level for the same reason ``_variable_to_sheet_row`` is:
so it can be executed under test without instantiating CustomTkinter.
"""

import pytest

# CustomTkinter must initialise a Tk root on import, which fails in headless
# CI. Skip the whole module if it isn't importable in this environment.
pytest.importorskip("customtkinter")

from alchemist_core.data.search_space import SearchSpace
from alchemist_core.utils.doe import generate_initial_design
from ui.ui import _design_points_to_columns


POSITIONS = ["first", "middle", "last"]


def _space(position):
    """``x1(real)``, ``x2(integer)``, ``x4(categorical)`` with ``c1`` at ``position``."""
    order = {
        "first": ["c1", "x1", "x2", "x4"],
        "middle": ["x1", "c1", "x2", "x4"],
        "last": ["x1", "x2", "x4", "c1"],
    }[position]
    space = SearchSpace()
    for name in order:
        if name == "c1":
            space.add_variable("c1", "context")
        elif name == "x1":
            space.add_variable("x1", "real", min=0.0, max=5.0)
        elif name == "x2":
            space.add_variable("x2", "integer", min=100, max=200)
        else:
            space.add_variable("x4", "categorical", values=["a", "b", "c"])
    return space


class TestDesignPointsToColumns:

    @pytest.mark.parametrize("position", POSITIONS)
    @pytest.mark.parametrize("method", ["random", "lhs", "sobol", "halton", "hammersly"])
    def test_a_context_variable_does_not_raise(self, position, method):
        space = _space(position)
        points = generate_initial_design(
            space, method=method, n_points=4, random_seed=7
        )
        data = _design_points_to_columns(space, points)
        assert set(data) == {"x1", "x2", "x4"}
        assert all(len(col) == 4 for col in data.values())

    @pytest.mark.parametrize("position", POSITIONS)
    def test_the_context_variable_gets_no_column(self, position):
        """It carries no design value, so it must not become a sheet column."""
        space = _space(position)
        points = generate_initial_design(space, method="lhs", n_points=3, random_seed=1)
        assert "c1" not in _design_points_to_columns(space, points)

    @pytest.mark.parametrize("position", POSITIONS)
    def test_values_are_transposed_in_row_order(self, position):
        """Column j of the frame is variable j's value from each point, in order."""
        space = _space(position)
        points = generate_initial_design(space, method="random", n_points=5, random_seed=4)
        data = _design_points_to_columns(space, points)
        for name, column in data.items():
            assert column == [p[name] for p in points]

    def test_a_space_with_no_context_variable_is_unchanged(self):
        """The substitution must be a no-op wherever the two lists coincide."""
        space = SearchSpace()
        space.add_variable("x1", "real", min=0.0, max=5.0)
        space.add_variable("x3", "discrete", allowed_values=[1.0, 2.0, 4.0])
        points = generate_initial_design(space, method="lhs", n_points=3, random_seed=2)
        data = _design_points_to_columns(space, points)
        assert list(data) == ["x1", "x3"]
        assert data == {name: [p[name] for p in points] for name in ("x1", "x3")}

    def test_a_constrained_design_reaches_the_frame_intact(self):
        """End to end from the core call the UI actually makes.

        The constraint is computed by hand rather than through
        ``filter_feasible`` -- see the core test module's docstring for why.
        """
        space = SearchSpace()
        space.add_variable("x1", "real", min=0.0, max=5.0)
        space.add_variable("c1", "context")
        space.add_variable("x5", "real", min=0.0, max=5.0)
        space.add_constraint("inequality", {"x1": 1.0, "x5": 1.0}, 5.0)

        points = generate_initial_design(space, method="lhs", n_points=4, random_seed=7)
        data = _design_points_to_columns(space, points)
        assert set(data) == {"x1", "x5"}
        for a, b in zip(data["x1"], data["x5"]):
            assert a + b <= 5.0 + 1e-9, (a, b)
