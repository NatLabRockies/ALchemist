"""Unit tests for SearchSpace input constraints."""

import pytest
import json
import tempfile
import os
import numpy as np
from alchemist_core.data.search_space import SearchSpace


class TestInputConstraints:
    """Tests for SearchSpace.add_constraint() and related methods."""

    def setup_method(self):
        self.space = SearchSpace()
        self.space.add_variable('x1', 'real', min=0.0, max=1.0)
        self.space.add_variable('x2', 'real', min=0.0, max=1.0)
        self.space.add_variable('x3', 'real', min=0.0, max=2.0)

    def test_add_inequality_constraint(self):
        self.space.add_constraint('inequality', {'x1': 1.0, 'x2': 1.0}, rhs=1.5)
        constraints = self.space.get_constraints()
        assert len(constraints) == 1
        assert constraints[0]['type'] == 'inequality'
        assert constraints[0]['rhs'] == 1.5

    def test_add_equality_constraint(self):
        self.space.add_constraint('equality', {'x1': 1.0, 'x2': -1.0}, rhs=0.0)
        constraints = self.space.get_constraints()
        assert len(constraints) == 1
        assert constraints[0]['type'] == 'equality'

    def test_add_named_constraint(self):
        self.space.add_constraint('inequality', {'x1': 1.0}, rhs=0.5, name='upper_x1')
        constraints = self.space.get_constraints()
        assert constraints[0]['name'] == 'upper_x1'

    def test_invalid_constraint_type_raises(self):
        with pytest.raises(ValueError, match="constraint_type"):
            self.space.add_constraint('invalid', {'x1': 1.0}, rhs=1.0)

    def test_invalid_variable_raises(self):
        with pytest.raises(ValueError, match="Variable 'z'"):
            self.space.add_constraint('inequality', {'z': 1.0}, rhs=1.0)

    def test_multiple_constraints(self):
        self.space.add_constraint('inequality', {'x1': 1.0, 'x2': 1.0}, rhs=1.5)
        self.space.add_constraint('inequality', {'x2': 1.0, 'x3': 1.0}, rhs=2.0)
        self.space.add_constraint('equality', {'x1': 1.0, 'x2': -1.0}, rhs=0.0)
        assert len(self.space.get_constraints()) == 3

    def test_get_constraints_returns_copies(self):
        self.space.add_constraint('inequality', {'x1': 1.0}, rhs=1.0)
        constraints = self.space.get_constraints()
        constraints[0]['rhs'] = 999.0
        # Original should be unmodified
        assert self.space.constraints[0]['rhs'] == 1.0


class TestBotorchConstraintConversion:
    """Tests for to_botorch_constraints()."""

    def setup_method(self):
        self.space = SearchSpace()
        self.space.add_variable('x1', 'real', min=0.0, max=1.0)
        self.space.add_variable('x2', 'real', min=0.0, max=1.0)

    def test_no_constraints_returns_none(self):
        ineq, eq = self.space.to_botorch_constraints(['x1', 'x2'])
        assert ineq is None
        assert eq is None

    def test_inequality_conversion(self):
        self.space.add_constraint('inequality', {'x1': 1.0, 'x2': 1.0}, rhs=1.5)
        ineq, eq = self.space.to_botorch_constraints(['x1', 'x2'])
        assert ineq is not None
        assert eq is None
        assert len(ineq) == 1

        indices, coeffs, rhs = ineq[0]
        # ALchemist convention is coeff·x <= rhs; BoTorch expects coeff·x >= rhs,
        # so coefficients and rhs must both be negated by to_botorch_constraints.
        assert indices.tolist() == [0, 1]
        assert coeffs.tolist() == [-1.0, -1.0]
        assert rhs == -1.5

    def test_equality_conversion(self):
        self.space.add_constraint('equality', {'x1': 1.0, 'x2': -1.0}, rhs=0.0)
        ineq, eq = self.space.to_botorch_constraints(['x1', 'x2'])
        assert ineq is None
        assert eq is not None
        assert len(eq) == 1

    def test_feature_order_respected(self):
        """Ensure indices match the feature_names order, not search space order."""
        self.space.add_constraint('inequality', {'x2': 2.0, 'x1': 3.0}, rhs=5.0)
        # Reversed feature order
        ineq, _ = self.space.to_botorch_constraints(['x2', 'x1'])
        indices, coeffs, _ = ineq[0]
        # x2 is at index 0, x1 is at index 1 in ['x2', 'x1']; coeffs are negated
        # for BoTorch's >= convention.
        assert indices.tolist() == [0, 1]
        assert coeffs.tolist() == [-2.0, -3.0]


    def test_equality_not_sign_flipped(self):
        """Equality constraints are sign-symmetric; pass through unchanged."""
        self.space.add_constraint('equality', {'x1': 1.0, 'x2': -1.0}, rhs=0.5)
        _, eq = self.space.to_botorch_constraints(['x1', 'x2'])
        indices, coeffs, rhs = eq[0]
        assert indices.tolist() == [0, 1]
        assert coeffs.tolist() == [1.0, -1.0]
        assert rhs == 0.5


class TestConstraintSerialization:
    """Tests for constraint serialization in save/load JSON."""

    def test_save_and_load_with_constraints(self):
        space = SearchSpace()
        space.add_variable('x1', 'real', min=0.0, max=1.0)
        space.add_variable('x2', 'real', min=0.0, max=1.0)
        space.add_constraint('inequality', {'x1': 1.0, 'x2': 1.0}, rhs=1.5, name='sum_bound')

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            filepath = f.name

        try:
            space.save_to_json(filepath)
            loaded_space = SearchSpace.from_json(filepath)

            assert len(loaded_space.variables) == 2
            assert len(loaded_space.constraints) == 1
            assert loaded_space.constraints[0]['name'] == 'sum_bound'
            assert loaded_space.constraints[0]['rhs'] == 1.5
        finally:
            os.unlink(filepath)

    def test_load_legacy_format_without_constraints(self):
        """Old JSON format (list of variables) should still work."""
        space = SearchSpace()
        space.add_variable('x1', 'real', min=0.0, max=1.0)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            # Write old format (just a list)
            json.dump(space.to_dict(), f)
            filepath = f.name

        try:
            loaded_space = SearchSpace.from_json(filepath)
            assert len(loaded_space.variables) == 1
            assert len(loaded_space.constraints) == 0
        finally:
            os.unlink(filepath)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class TestFeasibility:
    """Tests for SearchSpace.is_feasible / filter_feasible."""

    def setup_method(self):
        self.space = SearchSpace()
        self.space.add_variable('H2', 'real', min=0.0, max=100.0)
        self.space.add_variable('CO', 'real', min=0.0, max=100.0)
        self.space.add_variable('CO2', 'real', min=0.0, max=100.0)

    def test_no_constraints_all_feasible(self):
        import pandas as pd
        df = pd.DataFrame({'H2': [10, 90], 'CO': [10, 90], 'CO2': [10, 90]})
        mask = self.space.filter_feasible(df)
        assert mask.tolist() == [True, True]

    def test_equality_within_relative_tolerance(self):
        import pandas as pd
        self.space.add_constraint('equality', {'H2': 1.0, 'CO': 1.0, 'CO2': 1.0}, rhs=100.0)
        df = pd.DataFrame({
            'H2':  [50.0, 50.0, 0.0],
            'CO':  [30.0, 30.0, 0.0],
            'CO2': [20.0, 25.0, 0.0],
        })
        mask = self.space.filter_feasible(df)
        assert mask.tolist() == [True, False, False]

    def test_inequality_feasibility(self):
        import pandas as pd
        self.space.add_constraint('inequality', {'H2': 1.0, 'CO': 1.0}, rhs=50.0)
        df = pd.DataFrame({'H2': [20.0, 40.0], 'CO': [20.0, 40.0], 'CO2': [0, 0]})
        mask = self.space.filter_feasible(df)
        assert mask.tolist() == [True, False]

    def test_multiple_constraints_and(self):
        import pandas as pd
        self.space.add_constraint('equality', {'H2': 1.0, 'CO': 1.0, 'CO2': 1.0}, rhs=100.0)
        self.space.add_constraint('inequality', {'H2': 1.0, 'CO': -1.0}, rhs=20.0)
        df = pd.DataFrame({
            'H2':  [60.0, 80.0],
            'CO':  [30.0, 10.0],
            'CO2': [10.0, 10.0],
        })
        mask = self.space.filter_feasible(df)
        assert mask.tolist() == [False, False]

    def test_is_feasible_single_dict(self):
        self.space.add_constraint('equality', {'H2': 1.0, 'CO': 1.0, 'CO2': 1.0}, rhs=100.0)
        assert self.space.is_feasible({'H2': 50.0, 'CO': 30.0, 'CO2': 20.0}) is True
        assert self.space.is_feasible({'H2': 50.0, 'CO': 30.0, 'CO2': 30.0}) is False

    def test_subset_constraint_ignores_missing_columns(self):
        import pandas as pd
        self.space.add_constraint('inequality', {'H2': 1.0, 'CO': 1.0}, rhs=50.0)
        df = pd.DataFrame({'H2': [20.0], 'CO': [20.0], 'CO2': [999.0]})
        assert self.space.filter_feasible(df).tolist() == [True]


class TestConstraintNameUniqueness:
    """Auto-generated constraint names must stay unique across removals.

    Names are the identity used to remove a constraint (the REST DELETE route
    takes a name). The old generator was ``constraint_{len(self.constraints)}``,
    which is collision-free only while nothing is ever removed. Once a removal
    path exists, the counter revisits an index already in use and two
    constraints end up sharing a name -- at which point a delete-by-name filter
    removes both.
    """

    def setup_method(self):
        self.space = SearchSpace()
        self.space.add_variable('x1', 'real', min=0.0, max=10.0)
        self.space.add_variable('x2', 'integer', min=0, max=10)
        self.space.add_variable('x3', 'discrete', allowed_values=[0.0, 2.5, 5.0])

    def _names(self):
        return [c['name'] for c in self.space.constraints]

    def test_auto_names_stay_unique_after_a_removal(self):
        self.space.add_constraint('inequality', {'x1': 3.0}, rhs=5.0)
        self.space.add_constraint('inequality', {'x2': -2.0}, rhs=4.0)
        self.space.add_constraint('equality', {'x3': 1.5}, rhs=2.5)
        assert self._names() == ['constraint_0', 'constraint_1', 'constraint_2']

        # Removal by name, as the DELETE route performs it.
        self.space.constraints = [
            c for c in self.space.constraints if c['name'] != 'constraint_0'
        ]

        self.space.add_constraint('inequality', {'x1': 3.0, 'x2': -2.0}, rhs=7.0)
        assert len(set(self._names())) == len(self._names()), self._names()
        assert self._names() == ['constraint_1', 'constraint_2', 'constraint_3']

    def test_delete_by_name_removes_exactly_one_constraint(self):
        for _ in range(3):
            self.space.add_constraint('inequality', {'x1': 3.0}, rhs=5.0)
        self.space.constraints = [
            c for c in self.space.constraints if c['name'] != 'constraint_0'
        ]
        self.space.add_constraint('inequality', {'x2': -2.0}, rhs=4.0)

        before = len(self.space.constraints)
        self.space.constraints = [
            c for c in self.space.constraints if c['name'] != 'constraint_2'
        ]
        assert len(self.space.constraints) == before - 1
        assert self._names() == ['constraint_1', 'constraint_3']

    def test_auto_name_survives_a_save_load_round_trip(self):
        """The generator must be stateless: nothing but the names is persisted.

        ``save_to_json`` writes ``self.constraints`` as raw data and
        ``load_from_json`` restores it with a bare assignment, so a counter
        attribute would come back at zero and immediately collide.
        """
        for _ in range(3):
            self.space.add_constraint('inequality', {'x1': 3.0}, rhs=5.0)
        self.space.constraints = [
            c for c in self.space.constraints if c['name'] != 'constraint_0'
        ]

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            filepath = f.name
        try:
            self.space.save_to_json(filepath)
            loaded = SearchSpace.from_json(filepath)
        finally:
            os.unlink(filepath)

        assert [c['name'] for c in loaded.constraints] == ['constraint_1', 'constraint_2']

        loaded.add_constraint('inequality', {'x2': -2.0}, rhs=4.0)
        names = [c['name'] for c in loaded.constraints]
        assert len(set(names)) == len(names), names
        assert names == ['constraint_1', 'constraint_2', 'constraint_3']

    def test_existing_names_are_never_renumbered(self):
        """Names are stable identifiers; adding must not disturb the ones held."""
        self.space.add_constraint('inequality', {'x1': 3.0}, rhs=5.0, name='keep_me')
        self.space.add_constraint('inequality', {'x2': -2.0}, rhs=4.0)
        first = dict(self.space.constraints[0])

        self.space.constraints = [
            c for c in self.space.constraints if c['name'] != 'constraint_0'
        ]
        self.space.add_constraint('equality', {'x3': 1.5}, rhs=2.5)

        assert self.space.constraints[0] == first

    def test_duplicate_explicit_name_raises(self):
        self.space.add_constraint('inequality', {'x1': 3.0}, rhs=5.0, name='half_plane')
        with pytest.raises(ValueError, match="already registered"):
            self.space.add_constraint('inequality', {'x2': -2.0}, rhs=4.0, name='half_plane')
        assert len(self.space.constraints) == 1

    def test_explicit_name_colliding_with_an_auto_name_raises(self):
        self.space.add_constraint('inequality', {'x1': 3.0}, rhs=5.0)
        with pytest.raises(ValueError, match="already registered"):
            self.space.add_constraint('inequality', {'x2': -2.0}, rhs=4.0,
                                      name='constraint_0')

    def test_explicit_name_still_takes_precedence(self):
        self.space.add_constraint('inequality', {'x1': 3.0}, rhs=5.0)
        self.space.add_constraint('inequality', {'x2': -2.0}, rhs=4.0, name='named')
        assert self.space.constraints[1]['name'] == 'named'

    def test_auto_name_skips_an_index_claimed_by_an_explicit_name(self):
        self.space.add_constraint('inequality', {'x1': 3.0}, rhs=5.0,
                                  name='constraint_7')
        self.space.add_constraint('inequality', {'x2': -2.0}, rhs=4.0)
        assert self._names() == ['constraint_7', 'constraint_8']

    def test_non_indexed_names_do_not_break_auto_naming(self):
        self.space.add_constraint('inequality', {'x1': 3.0}, rhs=5.0, name='alpha')
        self.space.add_constraint('inequality', {'x2': -2.0}, rhs=4.0, name='constraint_x')
        self.space.add_constraint('equality', {'x3': 1.5}, rhs=2.5)
        assert self._names() == ['alpha', 'constraint_x', 'constraint_0']


class TestNonFiniteConstraintValues:
    """A constraint value must be finite, at the core method.

    ``OptimizationSession.add_outcome_constraint`` has rejected non-finite
    values all along (session.py); input constraints never learned it. The
    check lives here rather than only in the API request model because this
    method is reachable from Python and the desktop GUI, not just over REST.

    A NaN rhs makes every point infeasible, which drives the DoE into a
    pathological resampling path with no infeasible geometry involved, and a
    non-finite value is not JSON-representable -- it serializes to ``null``,
    so a constraint carrying one cannot round-trip.
    """

    def setup_method(self):
        self.space = SearchSpace()
        self.space.add_variable('x1', 'real', min=0.0, max=10.0)
        self.space.add_variable('x2', 'integer', min=0, max=10)
        self.space.add_variable('x3', 'discrete', allowed_values=[0.0, 2.5, 5.0])

    @pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')])
    def test_non_finite_rhs_raises(self, bad):
        with pytest.raises(ValueError, match='rhs must be finite'):
            self.space.add_constraint('inequality', {'x1': 3.0}, rhs=bad)
        assert self.space.constraints == []

    @pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')])
    @pytest.mark.parametrize('var', ['x1', 'x2', 'x3'])
    def test_non_finite_coefficient_raises(self, var, bad):
        """Covers all three constraint-eligible variable types."""
        with pytest.raises(ValueError, match=f"coefficient for '{var}' must be finite"):
            self.space.add_constraint('equality', {var: bad}, rhs=5.0)
        assert self.space.constraints == []

    def test_one_non_finite_coefficient_rejects_the_whole_constraint(self):
        with pytest.raises(ValueError, match='must be finite'):
            self.space.add_constraint(
                'inequality', {'x1': 3.0, 'x2': float('nan'), 'x3': 0.5}, rhs=-4.5
            )
        assert self.space.constraints == []

    def test_numpy_non_finite_is_rejected(self):
        """np.nan/np.inf arrive from array code, not just from JSON."""
        with pytest.raises(ValueError, match='rhs must be finite'):
            self.space.add_constraint('inequality', {'x1': 3.0}, rhs=np.inf)
        with pytest.raises(ValueError, match='must be finite'):
            self.space.add_constraint('inequality', {'x2': np.nan}, rhs=5.0)
        assert self.space.constraints == []

    def test_large_but_finite_values_are_accepted(self):
        """The check rejects non-finite, not merely large."""
        self.space.add_constraint(
            'inequality', {'x1': 1e308, 'x2': -2.0}, rhs=-1e308, name='huge'
        )
        stored = self.space.get_constraints()[0]
        assert stored['coefficients'] == {'x1': 1e308, 'x2': -2.0}
        assert stored['rhs'] == -1e308

    def test_zero_and_negative_values_are_accepted(self):
        """0.0 and -0.0 are finite; the check must not confuse falsiness."""
        self.space.add_constraint('equality', {'x1': 0.0, 'x3': -0.0}, rhs=0.0)
        assert len(self.space.constraints) == 1

    def test_rejected_constraint_does_not_consume_an_auto_name(self):
        """A failed add must leave no trace, auto-name counter included."""
        self.space.add_constraint('inequality', {'x1': 3.0}, rhs=5.0)
        with pytest.raises(ValueError, match='must be finite'):
            self.space.add_constraint('inequality', {'x2': -2.0}, rhs=float('nan'))
        self.space.add_constraint('equality', {'x3': 0.5}, rhs=2.5)
        assert [c['name'] for c in self.space.constraints] == [
            'constraint_0', 'constraint_1'
        ]


class TestConstraintDocstringsAgree:
    """Fix 4: the session delegate's docstring drifted from the core method.

    ``SearchSpace.add_constraint`` documents what it raises;
    ``OptimizationSession.add_input_constraint`` -- the delegate the REST
    router actually calls -- documented none of it, so the contract a Python
    caller reads depended on which of the two they happened to open. This test
    lives in the constraint test module because it is about the constraint
    contract, though the subject is the session class.
    """

    def test_delegate_documents_what_the_core_method_raises(self):
        from alchemist_core.session import OptimizationSession
        doc = OptimizationSession.add_input_constraint.__doc__
        assert doc is not None
        assert 'Raises:' in doc, 'delegate must document its failure modes'
        lowered = doc.lower()
        # The four rejection causes the core method actually has.
        assert 'constraint_type' in lowered
        assert 'numeric' in lowered
        assert 'finite' in lowered
        assert 'duplicate' in lowered

    def test_delegate_documents_the_auto_naming_and_uniqueness_rule(self):
        from alchemist_core.session import OptimizationSession
        doc = OptimizationSession.add_input_constraint.__doc__
        assert 'constraint_N' in doc, 'auto-naming is part of the contract'
        assert 'unique' in doc.lower() or 'duplicate' in doc.lower()
