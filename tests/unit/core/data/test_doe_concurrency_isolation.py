"""``generate_initial_design`` must not share mutable state between calls.

``POST /initial-design`` used to run its generator on the ASGI event loop, so
two designs were never built at once and shared state could not be observed.
Task 12B moved it to a worker thread, which is the point of the fix -- and
which turned two pieces of shared state into defects:

1. the **process-global** numpy RNG, which ``np.random.seed(random_seed)``
   configured and the samplers then drew from, so concurrent seeded requests
   consumed each other's draws;
2. the **SearchSpace's own skopt ``Dimension`` objects**, which skopt's
   samplers mutate (``set_transformer("normalize")`` ... ``inverse_transform``
   ... restore) and ``Space(dimensions)`` references rather than copies. An
   unlucky interleaving returns a design still normalized -- squashed into
   ``[0,1]`` instead of spanning the declared bounds. It is in range and the
   right shape, so nothing downstream can notice.

Both are races, and the end-to-end guard in
``tests/integration/api/test_design_endpoint_concurrency.py`` catches (2)
only probabilistically. These tests pin the same two properties
**deterministically**, by observing what the call touches rather than by
racing it.
"""

import numpy as np
import pytest

from alchemist_core.data.search_space import SearchSpace
from alchemist_core.utils import doe
from alchemist_core.utils.doe import SPACE_FILLING_METHODS, generate_initial_design

METHODS = sorted(SPACE_FILLING_METHODS)


def _space():
    space = SearchSpace()
    space.add_variable("x1", "real", min=0.0, max=10.0)
    space.add_variable("x2", "integer", min=0, max=20)
    space.add_variable("x3", "discrete", allowed_values=[1.0, 2.0, 4.0])
    return space


class TestTheProcessGlobalRngIsLeftAlone:
    """A seeded design must not be built out of, or reach into, global state."""

    @pytest.mark.parametrize("method", METHODS)
    def test_generating_a_seeded_design_does_not_move_the_global_rng(self, method):
        space = _space()
        np.random.seed(20260827)
        before = np.random.get_state()

        generate_initial_design(space, method=method, n_points=8, random_seed=7)

        after = np.random.get_state()
        assert before[0] == after[0]
        assert np.array_equal(before[1], after[1]), (
            f"method={method}: generating a design moved the process-global "
            f"RNG, so a concurrent request drawing from it gets different "
            f"numbers than its own seed names"
        )
        assert before[2:] == after[2:]

    def test_an_unseeded_design_also_leaves_the_global_rng_alone(self):
        space = _space()
        np.random.seed(11)
        before = np.random.get_state()

        generate_initial_design(space, method="lhs", n_points=8)

        after = np.random.get_state()
        assert np.array_equal(before[1], after[1])


class TestTheSamplersNeverTouchTheSearchSpacesOwnDimensions:
    """skopt mutates what it is handed, so it must not be handed the original."""

    @pytest.mark.parametrize("method", METHODS)
    def test_the_sampler_receives_copies_not_the_shared_objects(self, method, monkeypatch):
        space = _space()
        original = space.skopt_dimensions
        seen = {}

        def _capture(fn):
            def wrapper(skopt_space, *args, **kwargs):
                seen["dims"] = skopt_space
                return fn(skopt_space, *args, **kwargs)
            return wrapper

        for name in ("_random_sampling", "_lhs_sampling",
                     "_sobol_sampling", "_hammersly_sampling"):
            monkeypatch.setattr(doe, name, _capture(getattr(doe, name)))

        generate_initial_design(space, method=method, n_points=8, random_seed=3)

        handed = seen["dims"]
        assert handed is not original, (
            f"method={method}: the sampler was handed the SearchSpace's own "
            f"dimension list"
        )
        assert len(handed) == len(original)
        for i, (copy_dim, shared_dim) in enumerate(zip(handed, original)):
            assert copy_dim is not shared_dim, (
                f"method={method}: dimension {i} is the SearchSpace's own "
                f"object; skopt's set_transformer() mutates it, so a "
                f"concurrent design can be left normalized"
            )

    @pytest.mark.parametrize("method", METHODS)
    def test_the_dimensions_still_describe_the_same_space(self, method):
        """Copying must not have changed what is sampled."""
        space = _space()
        points = generate_initial_design(
            space, method=method, n_points=8, random_seed=3
        )
        assert len(points) == 8
        for point in points:
            assert 0.0 <= point["x1"] <= 10.0, point
            assert 0 <= point["x2"] <= 20, point
            assert point["x3"] in (1.0, 2.0, 4.0), point
