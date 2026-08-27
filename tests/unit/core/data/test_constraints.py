"""Unit tests for SearchSpace input constraints."""

import pytest
import json
import tempfile
import os
import sys
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


class TestNonNumericConstraintValues:
    """Ruling 33 Item B: a non-numeric value raises the documented ValueError.

    ``np.isfinite(None)`` raises ``TypeError: ufunc 'isfinite' not supported
    for the input types``. The docstrings on ``SearchSpace.add_constraint`` and
    ``OptimizationSession.add_input_constraint`` both promise ``ValueError``,
    and ``api/routers/variables.py`` catches only ``ValueError`` -- so the
    documented contract and the caller both disagreed with the behavior.

    It was not reachable through ``POST /constraints``, where ``rhs: float``
    coerces first. It became reachable the moment ``/variables/load`` started
    registering constraints out of a JSON file, where ``"rhs": null`` is an
    entirely ordinary thing to find.
    """

    def setup_method(self):
        self.space = SearchSpace()
        self.space.add_variable('x1', 'real', min=0.0, max=10.0)
        self.space.add_variable('x2', 'integer', min=0, max=10)
        self.space.add_variable('x3', 'discrete', allowed_values=[0.0, 2.5, 5.0])

    @pytest.mark.parametrize('bad', [None, 'abc', '', [1.0], {'a': 1}, object()])
    def test_non_numeric_rhs_raises_value_error(self, bad):
        with pytest.raises(ValueError, match='rhs must be a finite number'):
            self.space.add_constraint('inequality', {'x1': 2.5}, rhs=bad)
        assert self.space.constraints == []

    @pytest.mark.parametrize('bad', [None, 'abc', [1.0], {'a': 1}, object()])
    @pytest.mark.parametrize('var', ['x1', 'x2', 'x3'])
    def test_non_numeric_coefficient_raises_value_error(self, var, bad):
        """All three constraint-eligible variable types, not just real."""
        with pytest.raises(ValueError, match=f"coefficient for '{var}' must be a finite number"):
            self.space.add_constraint('equality', {var: bad}, rhs=5.0)
        assert self.space.constraints == []

    @pytest.mark.parametrize('bad', [None, 'abc', [1.0]])
    def test_non_numeric_never_raises_type_error(self, bad):
        """The point of the guard: callers catching ValueError must not be bypassed."""
        for call in (
            lambda: self.space.add_constraint('inequality', {'x1': 2.5}, rhs=bad),
            lambda: self.space.add_constraint('inequality', {'x1': bad}, rhs=2.5),
        ):
            try:
                call()
            except ValueError:
                pass
            except TypeError as exc:  # pragma: no cover - the defect being fixed
                pytest.fail(f'add_constraint raised TypeError, not ValueError: {exc}')
            else:
                pytest.fail(f'add_constraint accepted a non-numeric value: {bad!r}')

    def test_bool_is_still_accepted_as_a_number(self):
        """bools are ints in Python; the guard must reject non-numbers, not
        narrow the accepted numeric tower."""
        self.space.add_constraint('inequality', {'x1': True}, rhs=False, name='b')
        assert self.space.get_constraints()[0]['rhs'] is False

    def test_numeric_strings_are_not_silently_coerced(self):
        """'3.0' is not a number. Coercing it would let a whole file of quoted
        numbers through and store types the DoE does not expect."""
        with pytest.raises(ValueError, match='must be a finite number'):
            self.space.add_constraint('inequality', {'x1': '3.0'}, rhs=5.0)
        with pytest.raises(ValueError, match='must be a finite number'):
            self.space.add_constraint('inequality', {'x1': 3.0}, rhs='5.0')
        assert self.space.constraints == []

    def test_numpy_scalars_are_still_accepted(self):
        """Array code passes np.float64, which is not a Python float."""
        self.space.add_constraint(
            'inequality', {'x1': np.float64(3.0), 'x2': np.int64(-2)},
            rhs=np.float32(8.0), name='npy',
        )
        assert len(self.space.constraints) == 1

    def test_the_session_delegate_also_raises_value_error(self):
        """``variables.py`` catches ValueError around the session delegate."""
        from alchemist_core.session import OptimizationSession
        session = OptimizationSession(search_space=self.space)
        with pytest.raises(ValueError, match='must be a finite number'):
            session.add_input_constraint('inequality', {'x1': 2.5}, rhs=None)


class TestAddVariableIsAtomic:
    """A failed add_variable must leave no trace.

    ``self.variables`` was appended to before the skopt dimension was built, so
    every failure path left a half-registered variable behind: present in
    ``variables``, absent from ``skopt_dimensions``.

    The invariant that breaks is positional pairing, and it is narrower than
    "the two lists line up". ``context`` is the one type that registers a
    variable and deliberately builds no dimension, so the lists are paired
    positionally only across the *dimension-bearing* types (real, integer,
    categorical, discrete) -- which is precisely the assumption
    ``update_variable`` and ``delete_variable`` in ``api/routers/variables.py``
    make when they index one list by the other. A ``context`` variable
    therefore breaks those two routes as well (filed as branch item B7); this
    class does not test the routes, and
    ``test_a_context_variable_still_registers_without_a_dimension`` below pins
    the legitimate exception rather than contradicting the rule.

    So: a failed add must add to neither list, and a ``context`` add must add
    to ``variables`` only.

    Reachable over REST through the bare-list branch of
    ``POST /variables/load``, which appends into the session with no dry run:
    a malformed variable rejected the file with 400 and kept the fragment. The
    dict branch is protected by its dry run.
    """

    BAD = [
        ('real', {}),                                   # missing bounds
        ('real', {'min': 10.0, 'max': 0.0}),            # inverted bounds
        ('integer', {}),                                # missing bounds
        ('categorical', {}),                            # missing values
        ('discrete', {'allowed_values': [1.0]}),        # too few values
        ('discrete', {'allowed_values': [1.0, 1.0]}),   # duplicate values
        ('wat', {}),                                    # unknown type
    ]

    @pytest.mark.parametrize('var_type,kwargs', BAD)
    def test_a_failed_add_registers_nothing(self, var_type, kwargs):
        space = SearchSpace()
        with pytest.raises((ValueError, KeyError)):
            space.add_variable('x1', var_type, **kwargs)
        assert space.variables == []
        assert space.skopt_dimensions == []
        assert space.categorical_variables == []
        assert space.discrete_variables == []

    @pytest.mark.parametrize('var_type,kwargs', BAD)
    def test_a_failed_add_does_not_disturb_earlier_variables(self, var_type, kwargs):
        space = SearchSpace()
        space.add_variable('x1', 'real', min=0.0, max=10.0)
        space.add_variable('x2', 'categorical', values=['A', 'B'])
        space.add_variable('x3', 'discrete', allowed_values=[0.0, 2.5, 5.0])
        with pytest.raises((ValueError, KeyError)):
            space.add_variable('bad', var_type, **kwargs)
        assert [v['name'] for v in space.variables] == ['x1', 'x2', 'x3']
        assert len(space.skopt_dimensions) == 3
        assert space.categorical_variables == ['x2']
        assert space.discrete_variables == ['x3']

    def test_variables_and_dimensions_stay_paired_positionally(self):
        """The invariant the desync broke, over dimension-bearing types only."""
        space = SearchSpace()
        space.add_variable('x1', 'real', min=0.0, max=10.0)
        with pytest.raises(KeyError):
            space.add_variable('x2', 'integer')
        space.add_variable('x3', 'categorical', values=['A', 'B'])
        assert [v['name'] for v in space.variables] == ['x1', 'x3']
        assert [d.name for d in space.skopt_dimensions] == ['x1', 'x3']

    def test_a_context_variable_still_registers_without_a_dimension(self):
        """'context' is the one type that legitimately has no dimension."""
        space = SearchSpace()
        space.add_variable('x1', 'real', min=0.0, max=10.0)
        space.add_variable('note', 'context')
        assert [v['name'] for v in space.variables] == ['x1', 'note']
        assert [d.name for d in space.skopt_dimensions] == ['x1']

    def test_a_duplicate_name_still_leaves_the_original_intact(self):
        space = SearchSpace()
        space.add_variable('x1', 'real', min=0.0, max=10.0)
        with pytest.raises(ValueError, match='already registered'):
            space.add_variable('x1', 'integer', min=0, max=3)
        assert len(space.variables) == 1
        assert space.variables[0]['type'] == 'real'
        assert len(space.skopt_dimensions) == 1

    def test_discrete_values_are_still_sorted_and_float_coerced(self):
        """Moving the append must not lose the normalization done on the way in."""
        space = SearchSpace()
        space.add_variable('x1', 'discrete', allowed_values=[5, 1, 3.0])
        assert space.variables[0]['allowed_values'] == [1.0, 3.0, 5.0]
        assert list(space.skopt_dimensions[0].categories) == [1.0, 3.0, 5.0]


class TestNumpyBoolIsANumber:
    """Fix 4: ``np.bool_`` fell through the numeric tower and lied about it.

    ``np.bool_`` is a subclass of neither ``bool`` nor ``np.integer``, so it
    was refused by a guard whose own comment says bool is accepted and whose
    other branch accepts every numpy scalar. The message made that worse:
    ``type(np.True_).__name__`` is the bare string ``'bool'``, so the
    diagnostic read "must be a finite number, got np.True_ of type bool" --
    naming, as the reason for rejection, the exact type documented as accepted.
    """

    def setup_method(self):
        self.space = SearchSpace()
        self.space.add_variable('x1', 'real', min=0.0, max=10.0)
        self.space.add_variable('x2', 'integer', min=0, max=8)

    @pytest.mark.parametrize('value', [np.True_, np.False_, np.bool_(True)])
    def test_numpy_bool_is_accepted_like_python_bool(self, value):
        self.space.add_constraint(
            'inequality', {'x1': 3.0, 'x2': -2.5}, rhs=value, name=f'c_{value}'
        )
        assert len(self.space.constraints) == 1

    def test_numpy_bool_is_accepted_as_a_coefficient_too(self):
        self.space.add_constraint(
            'equality', {'x1': np.True_, 'x2': -1.5}, rhs=4.0, name='coef'
        )
        assert self.space.get_constraints()[0]['coefficients']['x1'] == np.True_

    def test_numpy_bool_and_python_bool_agree(self):
        """The hole was that these two behaved differently at all."""
        self.space.add_constraint('inequality', {'x1': 3.0}, rhs=True, name='py')
        self.space.add_constraint('inequality', {'x1': 3.0}, rhs=np.True_, name='np')
        assert len(self.space.constraints) == 2

    def test_a_rejected_numpy_scalar_is_named_as_numpy(self):
        """The message must not pass a numpy type off as its builtin namesake.

        ``np.str_`` reports ``__name__ == 'str_'``; the type that actually
        shadowed a builtin name was ``np.bool_``, now accepted. The rule is
        stated on whatever numpy scalar is still refused: the module qualifies
        the name, so "numpy.<something>" can never be read as the builtin.
        """
        with pytest.raises(ValueError, match=r'of type numpy\.'):
            self.space.add_constraint(
                'inequality', {'x1': 3.0}, rhs=np.str_('5.0'), name='bad'
            )
        assert self.space.constraints == []

    def test_a_rejected_builtin_is_still_named_unqualified(self):
        """Qualifying numpy must not make ordinary messages worse."""
        with pytest.raises(ValueError, match='of type NoneType'):
            self.space.add_constraint('inequality', {'x1': 3.0}, rhs=None)
        with pytest.raises(ValueError, match='of type str'):
            self.space.add_constraint('inequality', {'x1': 3.0}, rhs='5.0')


class TestVariableBoundsMustBeFinite:
    """Fix 1: a non-finite bound loaded cleanly and then poisoned every export.

    skopt's ``low >= high`` check is ``False`` for ``NaN``, so
    ``Real(nan, 10.0)`` builds without complaint. The variable registered, the
    session returned 200, and from then on *both* export shapes returned 400
    ("Out of range float values are not JSON compliant: nan") because
    ``JSONResponse`` serializes with ``allow_nan=False``. There was no way to
    get the search space back out of the session in either format.

    ``json.load`` accepts the bare ``NaN``/``Infinity``/``-Infinity`` literals,
    so the file that does this is an ordinary upload.

    The same guard the constraint values use is applied here, and the two agree
    on the deliberate decisions recorded with it: bool accepted, numpy scalars
    accepted, numeric strings rejected.
    """

    @pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')])
    @pytest.mark.parametrize('var_type,other', [('real', 10.0), ('integer', 8)])
    def test_a_non_finite_min_is_rejected(self, var_type, other, bad):
        space = SearchSpace()
        with pytest.raises(ValueError, match=r"Variable 'x1' min must be finite"):
            space.add_variable('x1', var_type, min=bad, max=other)
        assert space.variables == []
        assert space.skopt_dimensions == []

    @pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')])
    @pytest.mark.parametrize('var_type,other', [('real', 0.0), ('integer', 0)])
    def test_a_non_finite_max_is_rejected(self, var_type, other, bad):
        space = SearchSpace()
        with pytest.raises(ValueError, match=r"Variable 'x1' max must be finite"):
            space.add_variable('x1', var_type, min=other, max=bad)
        assert space.variables == []
        assert space.skopt_dimensions == []

    @pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')])
    def test_a_non_finite_discrete_value_is_rejected(self, bad):
        space = SearchSpace()
        with pytest.raises(
            ValueError, match=r"Variable 'x3' allowed_values\[1\] must be finite"
        ):
            space.add_variable('x3', 'discrete', allowed_values=[0.5, bad, 7.25])
        assert space.variables == []
        assert space.skopt_dimensions == []
        assert space.discrete_variables == []

    def test_the_reported_index_is_the_position_in_the_file(self):
        """Not the position after sorting -- the value is checked before sort."""
        space = SearchSpace()
        with pytest.raises(ValueError, match=r"allowed_values\[2\] must be finite"):
            space.add_variable(
                'x3', 'discrete', allowed_values=[7.25, 0.5, float('nan')]
            )

    @pytest.mark.parametrize('bad', [None, '0.0', [1.0], {'a': 1}, object()])
    def test_a_non_numeric_bound_raises_value_error_not_type_error(self, bad):
        """The bounds guard agrees with the constraint guard, including on
        numeric strings: skopt raised a bare TypeError for these, which the
        loader reported without naming the variable or the key."""
        space = SearchSpace()
        try:
            space.add_variable('x1', 'real', min=bad, max=10.0)
        except ValueError as exc:
            assert "Variable 'x1' min must be a finite number" in str(exc)
        except TypeError as exc:  # pragma: no cover - the defect being fixed
            pytest.fail(f'add_variable raised TypeError, not ValueError: {exc}')
        else:
            pytest.fail(f'add_variable accepted a non-numeric bound: {bad!r}')
        assert space.variables == []

    def test_a_missing_bound_still_raises_key_error(self):
        """The loader's message names the missing key off this KeyError."""
        space = SearchSpace()
        with pytest.raises(KeyError):
            space.add_variable('x1', 'real', max=10.0)

    def test_finite_bounds_of_every_shape_are_still_accepted(self):
        """Including the numeric tower the constraint guard documents.

        ``2**70`` is here because this test is what should have caught the
        round-2 regression and did not: it exercised ``1e12`` and
        ``np.int64(97)``, so every bound it offered was inside the range numpy
        can coerce, and the suite stayed green over an input that 500'd.
        """
        space = SearchSpace()
        space.add_variable('x1', 'real', min=-273.15, max=1e12)
        space.add_variable('x2', 'integer', min=np.int64(0), max=np.int64(97))
        space.add_variable('x3', 'discrete', allowed_values=[0.5, 7.25, -3.75])
        space.add_variable('x4', 'categorical', values=['A', 'B', 'C'])
        space.add_variable('x5', 'context')
        space.add_variable('x6', 'integer', min=-2**70, max=2**70)
        space.add_variable('x7', 'integer', min=False, max=True)
        assert [v['name'] for v in space.variables] == [
            'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'
        ]
        assert space.variables[2]['allowed_values'] == [-3.75, 0.5, 7.25]
        assert space.variables[5]['max'] == 2**70

    def test_a_rejected_bound_does_not_disturb_earlier_variables(self):
        space = SearchSpace()
        space.add_variable('x1', 'real', min=0.0, max=10.0)
        space.add_variable('x4', 'categorical', values=['A', 'B'])
        with pytest.raises(ValueError, match='must be finite'):
            space.add_variable('x2', 'integer', min=0, max=float('inf'))
        assert [v['name'] for v in space.variables] == ['x1', 'x4']
        assert [d.name for d in space.skopt_dimensions] == ['x1', 'x4']

    @pytest.mark.parametrize('bad', [float('nan'), float('inf')])
    def test_the_space_stays_json_serializable(self, bad):
        """The property the guard exists to protect, stated as itself."""
        space = SearchSpace()
        with pytest.raises(ValueError):
            space.add_variable('x1', 'real', min=0.0, max=bad)
        space.add_variable('x1', 'real', min=0.0, max=10.0)
        json.dumps({'variables': space.to_dict()}, allow_nan=False)


class TestABoundBeyondTheNumpyRangeIsStillAFiniteNumber:
    """Fix round 2: ``np.isfinite`` cannot be asked about every Python int.

    Round 1 introduced ``_validate_bound``, which called ``np.isfinite(value)``
    on whatever survived the type check. A Python ``int`` outside the uint64
    range fits no numpy dtype, so ``np.isfinite(2**64)`` raises
    ``TypeError: ufunc 'isfinite' not supported for the input types`` instead of
    returning True. ``POST /variables`` with ``integer max=2**64`` had returned
    200 before round 1 and returned 500 after it -- the endpoint has no
    try/except and only ValueError has a global handler -- while the load
    branches, which do catch TypeError, produced a 400 whose whole text was the
    numpy ufunc string, naming neither the variable nor the key.

    This is the defect class Fix 5 was opened to remove: an exception type
    outside the caught set, arriving where a 400 is documented. Fix 5 closed it
    for ``ZeroDivisionError`` from ``Categorical([])``; Fix 1 reopened it for
    ``TypeError``. The fix is not to catch TypeError more widely but to stop
    generating one: a Python ``int`` is finite by construction, so the
    finiteness test runs only on the types that can be non-finite.

    ``math.isfinite`` is not the fix either. It accepts ``2**64`` and raises
    ``OverflowError: int too large to convert to float`` on ``2**10000`` --
    the same defect with a different exception type, one door further along.
    """

    # 2**64 is where numpy stops coercing; 2**2000 is past the float64 range,
    # where anything routing through ``float()`` -- ``math.isfinite`` included
    # -- starts raising OverflowError instead. Both boundaries are needed: a
    # list that stopped at 2**200 accepts ``math.isfinite`` as a fix, and
    # ``math.isfinite`` is the same defect with a different exception type.
    HUGE = [
        2**63, 2**64 - 1, 2**64, 2**70, 2**200, 2**1024, 2**2000,
        -(2**64), -(2**70), -(2**2000),
    ]

    @pytest.mark.parametrize('bound', HUGE)
    def test_it_is_accepted_as_a_max(self, bound):
        """``min`` is put a full magnitude below so skopt's ``low < high``
        check is satisfied for negative bounds too."""
        space = SearchSpace()
        space.add_variable('x2', 'integer', min=bound - abs(bound) - 1, max=bound)
        space.add_variable('x1', 'real', min=-1.0, max=float(2**64))
        assert [v['name'] for v in space.variables] == ['x2', 'x1']
        assert [d.name for d in space.skopt_dimensions] == ['x2', 'x1']
        assert space.variables[0]['max'] == bound

    @pytest.mark.parametrize('bound', HUGE)
    def test_it_is_accepted_as_a_min(self, bound):
        space = SearchSpace()
        space.add_variable('x2', 'integer', min=-abs(bound), max=abs(bound))
        assert space.variables[0]['min'] == -abs(bound)
        assert space.skopt_dimensions[0].high == abs(bound)

    def test_the_variable_is_usable_afterwards(self):
        """Registering is not enough -- the space has to still work."""
        space = SearchSpace()
        space.add_variable('x2', 'integer', min=0, max=2**70)
        space.add_variable('x1', 'real', min=0.0, max=1.0)
        space.add_constraint('inequality', {'x1': 1.0}, rhs=0.5, name='c_a')
        assert len(space.skopt_dimensions) == 2
        assert len(space.get_constraints()) == 1
        json.dumps({'variables': space.to_dict()}, allow_nan=False)

    @pytest.mark.parametrize('bound', HUGE)
    def test_it_raises_nothing_at_all(self, bound):
        """The door class, stated as itself rather than as one exception type.

        Round 1 fixed a ValueError-shaped hole and opened a TypeError-shaped
        one; ``math.isfinite`` would close that and open an OverflowError-shaped
        one. Naming the specific type is what let each successor through, so the
        assertion is that a legitimate bound raises nothing whatsoever.
        """
        space = SearchSpace()
        try:
            space.add_variable('x2', 'integer', min=0, max=abs(bound))
        except Exception as exc:  # pragma: no cover - the defect being fixed
            pytest.fail(
                f'add_variable raised {type(exc).__name__} for a legitimate '
                f'bound {bound!r}: {exc}'
            )

    def test_a_huge_discrete_value_survives_the_float_coercion(self):
        """``allowed_values`` is coerced by ``float()`` before the guard sees
        it, so the int never reaches ``np.isfinite`` as an int. Pinned so that
        a later change to that coercion cannot reintroduce the fault here."""
        space = SearchSpace()
        space.add_variable('x3', 'discrete', allowed_values=[0.5, float(2**70)])
        assert space.variables[0]['allowed_values'] == [0.5, float(2**70)]

    def test_a_huge_int_is_still_refused_as_a_constraint_rhs_of_wrong_type(self):
        """The same guard backs constraint values, so it must not raise there
        either -- and a huge int is a legitimate rhs, not an error."""
        space = SearchSpace()
        space.add_variable('x1', 'real', min=0.0, max=1.0)
        space.add_constraint('inequality', {'x1': 2**70}, rhs=2**64, name='c_a')
        assert space.get_constraints()[0]['rhs'] == 2**64


# ------------------------------------------------------------------
# Ruling 38 -- fix round 3: the tower, swept by instance
# ------------------------------------------------------------------

def _concrete_types_the_tower_admits():
    """Every type an ``isinstance`` check against ``_FINITE_NUMBER_TYPES`` accepts.

    Walked out of ``np.generic``'s subclass tree rather than listed, so a numpy
    release that adds a scalar type inside the accepted tower is swept without
    anyone editing this file. Abstract entries (``np.integer``,
    ``np.floating``) come along harmlessly: nothing can be constructed from
    them, so they contribute no instances.
    """
    from alchemist_core.data.search_space import _FINITE_NUMBER_TYPES
    candidates = {int, float, bool}
    stack = [np.generic]
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        candidates.add(cls)
    return sorted(
        (t for t in candidates if issubclass(t, _FINITE_NUMBER_TYPES)),
        key=lambda t: (t.__module__, t.__name__),
    )


# Ways a non-finite value of *some* type can be spelled. Each is tried against
# every admitted type; the ones that cannot express one (every integer type,
# both bool types) simply raise and are skipped, which is the answer the sweep
# wants from them.
_NON_FINITE_RECIPES = (
    lambda t: t('nan'),
    lambda t: t('NaT'),
    lambda t: t('inf'),
    lambda t: t(float('inf')),
    lambda t: t(float('-inf')),
)


def _non_finite_instances(t):
    """Non-finite instances of exactly type ``t``, by construction not by name.

    ``np.isfinite`` is the oracle, and a value it cannot even be asked about
    counts as non-finite: whatever such a thing is, it is not a finite number,
    which is the only claim the guard is allowed to wave through.
    """
    found = []
    for recipe in _NON_FINITE_RECIPES:
        try:
            value = recipe(t)
        except Exception:
            continue
        if type(value) is not t:
            continue
        try:
            finite = bool(np.isfinite(value))
        except Exception:
            finite = False
        if not finite:
            found.append(value)
    return found


class TestTheFinitenessTestRunsOnlyWhereItCanSucceed:
    """The invariant that keeps a third door from opening.

    ``_MAY_BE_NON_FINITE`` narrows which accepted types get asked about their
    finiteness. That is only safe while the types it leaves out genuinely have
    no non-finite values -- if someone widens ``_FINITE_NUMBER_TYPES`` with
    ``Decimal`` (which has ``Decimal('NaN')``) or ``np.complexfloating``
    without revisiting the narrower tuple, a non-finite value would be waved
    through. Both halves of the relationship are asserted here rather than
    left as a comment.
    """

    def test_every_type_it_tests_is_a_type_the_tower_admits(self):
        """Subset by ``issubclass``, not by tuple membership.

        ``np.timedelta64`` is admitted through ``np.integer`` without being an
        entry of ``_FINITE_NUMBER_TYPES`` itself, so the entry-level form of
        this assertion called it a stranger. What matters is that nothing is
        tested for finiteness which the type check would have rejected anyway
        -- that would be dead code hiding a widening nobody made.
        """
        from alchemist_core.data.search_space import (
            _FINITE_NUMBER_TYPES, _MAY_BE_NON_FINITE,
        )
        assert _MAY_BE_NON_FINITE
        for t in _MAY_BE_NON_FINITE:
            assert issubclass(t, _FINITE_NUMBER_TYPES), (
                f'{t!r} is tested for finiteness but is not accepted at all'
            )

    def test_the_sweep_finds_the_non_finite_instances_it_is_meant_to(self):
        """The sweep below is only worth anything if it discovers things.

        Pinned separately so that a sweep which silently stopped constructing
        anything -- a renamed recipe, a numpy release that rejects
        ``np.float16('nan')`` -- fails as itself rather than as a vacuous pass
        of the invariant.
        """
        found = {t for t in _concrete_types_the_tower_admits()
                 if _non_finite_instances(t)}
        assert {float, np.float64, np.timedelta64} <= found, sorted(
            t.__name__ for t in found
        )

    def test_no_non_finite_instance_the_tower_admits_escapes_the_test(self):
        """The invariant, asserted over admitted *instances* not tuple entries.

        The previous form of this test asked whether every exempted tuple entry
        was an integer or bool type. ``issubclass(np.integer, (int, np.integer,
        np.bool_))`` is True, so the invariant passed -- while
        ``np.timedelta64('NaT')``, which is an ``np.integer`` and is not
        finite, walked straight through it and registered as a bound. An
        entry-level assertion cannot see inside a type; only an instance can.

        So: construct a non-finite instance of every concrete type an
        ``isinstance`` check against ``_FINITE_NUMBER_TYPES`` admits, and
        require the guard to refuse it with a labelled ValueError. Any future
        widening of the tower that brings a new non-finite value with it fails
        here without anyone having to think of the value first.
        """
        from alchemist_core.data.search_space import _MAY_BE_NON_FINITE
        checked = 0
        for t in _concrete_types_the_tower_admits():
            for value in _non_finite_instances(t):
                checked += 1
                assert issubclass(t, _MAY_BE_NON_FINITE), (
                    f'{t.__name__} admits the non-finite value {value!r} but '
                    f'is exempt from the finiteness test'
                )
                for var_type, key in (
                    ('real', 'min'), ('real', 'max'),
                    ('integer', 'min'), ('integer', 'max'),
                ):
                    space = SearchSpace()
                    bounds = {'min': 0, 'max': 10}
                    bounds[key] = value
                    with pytest.raises(ValueError) as exc:
                        space.add_variable('x1', var_type, **bounds)
                    detail = str(exc.value)
                    assert 'x1' in detail and key in detail, detail
                    assert space.variables == []
                    assert space.skopt_dimensions == []
        assert checked >= 3, 'the sweep checked nothing'

    def test_a_nat_bound_is_refused_on_every_numeric_variable_type(self):
        """The counterexample the entry-level invariant could not see.

        ``np.timedelta64`` subclasses ``np.signedinteger``, so it is inside the
        accepted tower; ``np.timedelta64('NaT')`` is a non-finite value of it.
        With the finiteness test skipped for the whole of ``np.integer`` it
        registered as an ordinary integer bound, and on a ``real`` bound it
        reached skopt and came back as ``UFuncTypeError`` -- a TypeError
        subclass out of a guard documented to raise nothing but a labelled
        ValueError.
        """
        nat = np.timedelta64('NaT')
        for var_type in ('real', 'integer'):
            space = SearchSpace()
            with pytest.raises(ValueError, match='must be finite'):
                space.add_variable('x1', var_type, min=0, max=nat)
            assert space.variables == []
        space = SearchSpace()
        with pytest.raises(ValueError, match='must be finite'):
            space.add_variable('x3', 'discrete', allowed_values=[0.5, nat])
        assert space.variables == []

    def test_a_finite_timedelta_is_refused_on_a_float_backed_bound(self):
        """The other half of taking np.timedelta64 seriously.

        The representability test answers by comparison, and that comparison is
        total only because np.timedelta64 is taken out of its way first:
        ``np.timedelta64(5, 's') <= 1.79e308`` raises UFuncTypeError rather
        than answering. A *finite* timedelta is past the finiteness test, so
        without ``_NOT_REAL_VALUED`` it would leave this guard by a TypeError
        subclass -- the same shape of defect as the one being fixed, reached
        from the other side.
        """
        td = np.timedelta64(5, 's')
        for var_type, kwargs in (
            ('real', {'min': 0.0, 'max': td}),
            ('discrete', {'allowed_values': [0.5, td]}),
        ):
            space = SearchSpace()
            with pytest.raises(ValueError, match='real number'):
                space.add_variable('x1', var_type, **kwargs)
            assert space.variables == []

    def test_a_non_finite_float_is_still_caught_by_it(self):
        """The narrowing must not have narrowed away the original guard."""
        space = SearchSpace()
        for bad in (float('nan'), float('inf'), np.float64('inf')):
            with pytest.raises(ValueError, match='must be finite'):
                space.add_variable('x1', 'real', min=0.0, max=bad)
        assert space.variables == []


class TestEmptyCategoricalIsRejectedByTheCore:
    """Fix 5: skopt divides by the category count to build its prior.

    ``Categorical([])`` raised ``ZeroDivisionError`` from
    ``1.0 / len(self.categories)``, which is outside the
    ``(ValueError, KeyError, TypeError)`` tuple the API loader catches, so an
    empty list was a 500 on both load branches against an endpoint that
    documents 400. Refused in the core rather than the router so the desktop
    loader gets the same answer.
    """

    @pytest.mark.parametrize('empty', [[], ()])
    def test_an_empty_category_list_raises_value_error(self, empty):
        space = SearchSpace()
        with pytest.raises(ValueError, match="at least 1 value"):
            space.add_variable('x4', 'categorical', values=empty)
        assert space.variables == []
        assert space.skopt_dimensions == []
        assert space.categorical_variables == []

    def test_it_is_not_a_zero_division_error(self):
        space = SearchSpace()
        try:
            space.add_variable('x4', 'categorical', values=[])
        except ValueError:
            pass
        except ZeroDivisionError as exc:  # pragma: no cover - the defect
            pytest.fail(f'add_variable raised ZeroDivisionError: {exc}')

    def test_a_single_category_is_still_accepted(self):
        """The guard rejects empty, not small. One category is degenerate but
        legal, and narrowing that is a change to what files load."""
        space = SearchSpace()
        space.add_variable('x4', 'categorical', values=['A'])
        assert space.categorical_variables == ['x4']
        assert list(space.skopt_dimensions[0].categories) == ['A']

    def test_a_missing_values_key_still_raises_key_error(self):
        space = SearchSpace()
        with pytest.raises(KeyError):
            space.add_variable('x4', 'categorical')
        assert space.variables == []


class TestFromDictPreservesVariableMetadata:
    """Fix 2: the dict path dropped ``unit`` and ``description``.

    ``from_dict`` forwarded only the fields each type needs to build its skopt
    dimension. ``POST /variables`` and the bare-list branch of
    ``POST /variables/load`` both call ``add_variable`` with the whole entry
    and keep the metadata, so the newly advertised round-trip format was
    strictly worse than the legacy shape beside it: ``export -> load ->
    export`` lost fields the first export had emitted.
    """

    ANNOTATED = [
        {'name': 'x1', 'type': 'real', 'min': 0.0, 'max': 10.0,
         'unit': 'kPa', 'description': 'first axis'},
        {'name': 'x2', 'type': 'integer', 'min': 0, 'max': 8,
         'unit': 'counts', 'description': 'second axis'},
        {'name': 'x3', 'type': 'discrete', 'allowed_values': [0.5, 7.25],
         'unit': 'mm', 'description': 'third axis'},
        {'name': 'x4', 'type': 'categorical', 'values': ['A', 'B', 'C'],
         'unit': '-', 'description': 'fourth axis'},
        {'name': 'x5', 'type': 'context',
         'unit': 'degC', 'description': 'observed only'},
    ]

    def test_every_variable_type_keeps_unit_and_description(self):
        space = SearchSpace().from_dict([dict(v) for v in self.ANNOTATED])
        by_name = {v['name']: v for v in space.variables}
        assert set(by_name) == {'x1', 'x2', 'x3', 'x4', 'x5'}
        for original in self.ANNOTATED:
            loaded = by_name[original['name']]
            assert loaded['unit'] == original['unit'], original['name']
            assert loaded['description'] == original['description'], original['name']

    def test_the_categories_alias_also_carries_metadata(self):
        """The alias branch is a separate add_variable call."""
        space = SearchSpace().from_dict([
            {'name': 'x4', 'type': 'categorical', 'categories': ['A', 'B'],
             'unit': '-', 'description': 'aliased'},
        ])
        assert space.variables[0]['unit'] == '-'
        assert space.variables[0]['description'] == 'aliased'

    def test_a_file_without_metadata_is_unchanged(self):
        """The desktop loader shares this method; files that lack these fields
        must produce exactly the variable dicts they always did -- no keys
        defaulted in."""
        space = SearchSpace().from_dict([
            {'name': 'x1', 'type': 'real', 'min': 0.0, 'max': 10.0},
            {'name': 'x4', 'type': 'categorical', 'values': ['A', 'B']},
            {'name': 'x3', 'type': 'discrete', 'allowed_values': [0.5, 7.25]},
            {'name': 'x5', 'type': 'context'},
        ])
        assert space.variables == [
            {'name': 'x1', 'type': 'real', 'min': 0.0, 'max': 10.0},
            {'name': 'x4', 'type': 'categorical', 'values': ['A', 'B']},
            {'name': 'x3', 'type': 'discrete', 'allowed_values': [0.5, 7.25]},
            {'name': 'x5', 'type': 'context'},
        ]

    def test_only_one_of_the_two_fields_is_forwarded_when_only_one_is_there(self):
        space = SearchSpace().from_dict([
            {'name': 'x1', 'type': 'real', 'min': 0.0, 'max': 10.0, 'unit': 'kPa'},
        ])
        assert space.variables[0]['unit'] == 'kPa'
        assert 'description' not in space.variables[0]

    def test_metadata_survives_save_to_json_and_back(self):
        """The advertised round trip, at the level the dict format is defined."""
        space = SearchSpace().from_dict([dict(v) for v in self.ANNOTATED])
        space.add_constraint('inequality', {'x1': 3.0, 'x2': -2.5}, rhs=8.0, name='c_a')
        fd, path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        try:
            space.save_to_json(path)
            reloaded = SearchSpace.from_json(path)
        finally:
            os.unlink(path)
        assert reloaded.to_dict() == space.to_dict()
        assert reloaded.get_constraints() == space.get_constraints()


class TestBareListIsTheUnprotectedPath:
    """Fix 3: which path made the atomicity defect reachable.

    The comment in ``add_variable`` attributed reachability to the dict branch
    of ``POST /variables/load``. It is the *bare-list* branch that carried it:
    it calls ``session.add_variable`` straight into the live session, one entry
    at a time, with no dry run, so a fragment survived a rejected file. The
    dict branch applies the same file to a throwaway ``SearchSpace`` first and
    only touches the session once that has succeeded.

    These two tests are the difference, at the level the core can state it: a
    failed add against a live space leaves a desync unless ``add_variable``
    itself is atomic, while a failed add against a throwaway leaves the live
    space untouched no matter what ``add_variable`` does.
    """

    FILE = [
        {'name': 'x1', 'type': 'real', 'min': 0.0, 'max': 10.0},
        {'name': 'x2', 'type': 'real', 'min': 9.0, 'max': 1.0},   # inverted
    ]

    def test_appending_into_a_live_space_needs_add_variable_to_be_atomic(self):
        """The bare-list branch, reduced to its core calls."""
        live = SearchSpace()
        with pytest.raises(ValueError):
            for entry in self.FILE:
                var = dict(entry)
                live.add_variable(var.pop('name'), var.pop('type'), **var)
        assert [v['name'] for v in live.variables] == ['x1']
        assert [d.name for d in live.skopt_dimensions] == ['x1']

    def test_a_dry_run_protects_the_live_space_by_itself(self):
        """The dict branch, reduced the same way: the live space is never
        reached, so its state does not depend on add_variable's atomicity."""
        live = SearchSpace()
        live.add_variable('x9', 'integer', min=0, max=8)
        with pytest.raises(ValueError):
            SearchSpace().from_dict([dict(v) for v in self.FILE])
        assert [v['name'] for v in live.variables] == ['x9']
        assert [d.name for d in live.skopt_dimensions] == ['x9']


class TestThePairingInvariantAsStated:
    """Fix 8: the pairing holds for dimension-bearing types; context is out.

    ``TestAddVariableIsAtomic``'s docstring said the two lists "are paired
    positionally" while a test in the same class asserted a ``context``
    variable registers without a dimension. Both were true of different
    subsets and had never been joined. This states the joined rule once.
    """

    def _space(self):
        space = SearchSpace()
        space.add_variable('x1', 'real', min=0.0, max=10.0)
        space.add_variable('x5', 'context')
        space.add_variable('x4', 'categorical', values=['A', 'B'])
        space.add_variable('x3', 'discrete', allowed_values=[0.5, 7.25])
        space.add_variable('x2', 'integer', min=0, max=8)
        return space

    def test_dimension_bearing_variables_pair_in_order(self):
        space = self._space()
        bearing = [v['name'] for v in space.variables if v['type'] != 'context']
        assert bearing == [d.name for d in space.skopt_dimensions]

    def test_context_is_the_only_type_without_a_dimension(self):
        space = self._space()
        without = [
            v['name'] for v in space.variables
            if v['name'] not in {d.name for d in space.skopt_dimensions}
        ]
        assert without == ['x5']
        assert [v['type'] for v in space.variables if v['name'] in without] == [
            'context'
        ]

    def test_context_shifts_the_positional_index(self):
        """Why B7 exists: the routers index skopt_dimensions by the position
        in variables, which is only correct up to the first context variable.
        Pinned as an observation, not endorsed -- the router is not fixed here.
        """
        space = self._space()
        assert space.variables[0]['name'] == space.skopt_dimensions[0].name
        assert space.variables[2]['name'] != space.skopt_dimensions[2].name
        assert len(space.skopt_dimensions) == len(space.variables) - 1

    def test_the_atomicity_class_states_the_narrowed_invariant(self):
        """Ruling 27: the prose and the code must not disagree.

        ``TestAddVariableIsAtomic`` asserted the pairing unqualified while
        holding a test that legitimately breaks it. Its docstring must scope
        the claim and account for the exception, not restate the flat version.
        """
        doc = TestAddVariableIsAtomic.__doc__
        assert doc is not None
        lowered = ' '.join(doc.lower().split())
        assert 'context' in lowered, 'the exception must be named'
        assert 'dimension-bearing' in lowered, 'the pairing must be scoped'
        # The unqualified claim the class used to make.
        assert 'the two lists are paired positionally' not in lowered


class TestTheRoundTwoStatementsAreTrueOfTheCode:
    """Ruling 27: a comment that says something false about the code is fixed.

    The round-2 regression falsified three sentences written in round 1 -- all
    three described a guard that could only raise a labelled ValueError, while
    the guard could in fact raise a bare TypeError. Each is pinned here against
    the behaviour it claims, so the sentence cannot outlive the code again.
    """

    def test_the_type_tuple_states_a_rule_the_code_actually_follows(self):
        """Statement 1 (``search_space.py`` module comment).

        It states the accepted set as one sentence. ``2**64`` is a finite
        Python integer and was refused, so the sentence was false. The pin is
        the behaviour, not the wording: whatever sentence stands there, a bound
        of any magnitude has to be accepted for it to be true.
        """
        import inspect
        import alchemist_core.data.search_space as mod
        # The comment block that introduces the tuple, i.e. everything above
        # the assignment. Sliced rather than line-numbered so it survives edits
        # elsewhere in the module.
        source = inspect.getsource(mod)
        comment = source.split('_FINITE_NUMBER_TYPES = ')[0]
        assert 'magnitude' in comment, (
            'the rule must say that magnitude is not what it screens on'
        )
        space = SearchSpace()
        space.add_variable('x2', 'integer', min=-(2**200), max=2**200)
        assert len(space.skopt_dimensions) == 1

    def test_validate_bound_documents_a_promise_it_keeps(self):
        """Statement 2 (``_validate_bound`` docstring).

        "The label names both the variable and the key" was false whenever the
        guard raised TypeError, which carried no label at all. Both the claim
        and the fact are checked.
        """
        from alchemist_core.data.search_space import _validate_bound
        doc = _validate_bound.__doc__
        assert doc is not None
        lowered = ' '.join(doc.lower().split())
        assert 'names both the variable and the key' in lowered
        assert 'every rejection' in lowered, (
            'the claim must be stated as universal, since it now is'
        )

        # And it is: the only exit is a ValueError carrying both.
        for kwargs, key in [
            (dict(min=None, max=10.0), 'min'),
            (dict(min=0.0, max=float('inf')), 'max'),
            (dict(min=0.0, max='9'), 'max'),
        ]:
            with pytest.raises(ValueError) as exc:
                SearchSpace().add_variable('x1', 'real', **kwargs)
            assert "Variable 'x1'" in str(exc.value)
            assert key in str(exc.value)

    def test_the_loader_docstring_no_longer_blames_an_unenumerated_shape(self):
        """Statement 3 (``_load_error_detail`` docstring in the API router).

        It claimed the TypeError branch only ever saw shapes nobody had
        enumerated. The guard itself was feeding that branch with ``2**64`` --
        an enumerated shape -- so the sentence was false. The branch must now
        describe what actually reaches it.
        """
        from api.routers.variables import _load_error_detail
        doc = _load_error_detail.__doc__
        assert doc is not None
        lowered = ' '.join(doc.lower().split())
        assert 'no bound reaches this branch' in lowered, (
            'the docstring must say that bounds no longer feed the branch'
        )
        assert 'the guard does not enumerate' in lowered, (
            'the remaining shapes must be scoped to the guard, not to "nobody"'
        )
        # The unqualified claim it used to make.
        assert 'shape nobody enumerated' not in lowered

    def test_the_shape_the_loader_docstring_names_is_a_real_type_error(self):
        """The replacement sentence names ``"allowed_values": 5``. If that
        stopped raising TypeError the sentence would be false again."""
        from api.routers.variables import _load_error_detail
        with pytest.raises(TypeError) as exc:
            SearchSpace().add_variable('x3', 'discrete', allowed_values=5)
        detail = _load_error_detail(exc.value)
        assert detail.startswith('Search space file could not be loaded:')

    def test_the_rationale_admits_what_it_accepts_but_cannot_serialize(self):
        """Reviewer's Minor 1: the guard's stated reason was
        JSON-representability, but ``np.int64`` and ``np.True_`` are accepted
        and ``json.dumps`` refuses both. The rationale must name that gap
        rather than imply the tower is JSON-clean."""
        from alchemist_core.data.search_space import _validate_finite_number
        doc = _validate_finite_number.__doc__
        assert doc is not None
        lowered = ' '.join(doc.lower().split())
        assert 'finiteness, not json-representability' in lowered
        assert 'np.integer' in doc and 'np.bool_' in doc, (
            'the two accepted-but-unserializable types must be named'
        )

        # The fact behind the correction, in both directions.
        space = SearchSpace()
        space.add_variable('x2', 'integer', min=np.int64(0), max=np.int64(9))
        with pytest.raises(TypeError):
            json.dumps(np.int64(9))
        json.dumps(2**200)  # the Python half really is serializable at any width


class TestTheGuardDoesNotRouteThroughFloat:
    """Why the finiteness test is narrowed rather than swapped for another one.

    ``math.isfinite`` looks like the obvious one-line fix: it takes ``2**64``,
    where ``np.isfinite`` does not. It converts through ``float()`` to do it,
    so it raises ``OverflowError: int too large to convert to float`` at
    ``2**1024`` -- the same defect as the round-2 regression with a different
    exception type, and equally a 500 on ``POST /variables``.

    Fix 5 removed this defect class for ``ZeroDivisionError``, Fix 1 reopened
    it for ``TypeError``, and a ``math.isfinite`` fix would reopen it for
    ``OverflowError``. The class is closed by not asking the question of values
    that cannot answer it, rather than by picking a function whose failure mode
    starts further out.
    """

    def test_the_boundary_that_distinguishes_the_two_is_real(self):
        """If this stops holding, the test below stops discriminating."""
        import math
        assert math.isfinite(2**1023)
        with pytest.raises(OverflowError):
            math.isfinite(2**1024)
        with pytest.raises(TypeError):
            np.isfinite(2**64)

    @pytest.mark.parametrize('bound', [2**1024, 2**2000, -(2**2000)])
    def test_a_bound_past_the_float64_range_is_accepted(self, bound):
        space = SearchSpace()
        space.add_variable('x2', 'integer', min=-abs(bound), max=abs(bound))
        assert space.variables[0]['max'] == abs(bound)
        json.dumps({'variables': space.to_dict()}, allow_nan=False)

    def test_it_holds_for_a_constraint_value_too(self):
        """The same guard backs rhs and coefficients."""
        space = SearchSpace()
        space.add_variable('x1', 'real', min=0.0, max=1.0)
        space.add_constraint(
            'inequality', {'x1': 2**2000}, rhs=2**1024, name='c_a'
        )
        assert space.get_constraints()[0]['rhs'] == 2**1024


# ============================================================
# Ruling 38 -- fix round 3
# ============================================================

# Boundaries a bound guard has to get right, and why each one is here:
#   sys.float_info.max        the last finite float64
#   int(sys.float_info.max)   the same value spelled as a Python int, which is
#                             the only spelling POST /variables/load can carry
#   2**1023                   comfortably inside, and the value the controller
#                             observed returning 200
#   2**1024                   the first power of two with no float64 image; the
#                             magnitude that made Real() raise OverflowError
#   2**2000                   far past it, where any conversion overflows
_INSIDE_FLOAT64 = [2**63, 2**64, 2**70, 2**200, 2**1023, int(sys.float_info.max)]
_OUTSIDE_FLOAT64 = [2**1024, 2**2000, int(sys.float_info.max) + 1]


class TestARealBoundMustBeRepresentableByTheDimensionItBuilds:
    """Fix round 3: ``Real(0, 2**1024)`` raised OverflowError three frames down.

    Three rounds each removed one exception type from this guard and revealed
    the next -- ``ZeroDivisionError`` from ``Categorical([])``, then
    ``TypeError`` from ``np.isfinite(2**64)``, then ``OverflowError`` -- and
    every one arrived as a 500 where the endpoint documents 400. Round 2's own
    comment named the OverflowError hazard and avoided ``math.isfinite``, but
    the identical ``float()`` conversion happens inside
    ``skopt.Real.set_transformer``, which ``Real.__init__`` calls: a ``real``
    bound of ``2**1024`` therefore reached numpy's normalizer and raised
    ``OverflowError: int too large to convert to float``, an ``ArithmeticError``
    outside both the loaders' ``(ValueError, KeyError, TypeError)`` tuple and
    the global ``ValueError`` handler.

    The fix is not a fourth exception type. It is to ask, before constructing
    anything, whether the dimension *about to be built* can represent the
    bound: ``skopt.Real`` is float64 all the way down, so a bound outside the
    float64 range is refused here with the variable and key on it, and skopt is
    never asked a question it cannot answer. ``skopt.Integer`` converts
    nothing, so an integer bound stays legitimate at every magnitude -- which
    is the half of the rule that a blanket magnitude limit would have broken.
    """

    @pytest.mark.parametrize('bound', _OUTSIDE_FLOAT64)
    @pytest.mark.parametrize('key', ['min', 'max'])
    def test_it_is_a_labelled_value_error(self, bound, key):
        space = SearchSpace()
        bounds = {'min': 0.0, 'max': 10.0}
        bounds[key] = -bound if key == 'min' else bound
        with pytest.raises(ValueError) as exc:
            space.add_variable('x1', 'real', **bounds)
        detail = str(exc.value)
        assert 'x1' in detail, 'the message must name the variable'
        assert key in detail, 'the message must name the key'
        assert 'float64' in detail
        assert space.variables == []
        assert space.skopt_dimensions == []

    @pytest.mark.parametrize('bound', _OUTSIDE_FLOAT64)
    def test_it_is_not_an_arithmetic_error(self, bound):
        """The door class stated as itself, the way round 2 learned to state it.

        Naming ``OverflowError`` is what let its predecessor through, so the
        assertion is that nothing but a ValueError leaves the guard.
        """
        space = SearchSpace()
        try:
            space.add_variable('x1', 'real', min=0.0, max=bound)
        except ValueError:
            pass
        except Exception as exc:  # pragma: no cover - the defect being fixed
            pytest.fail(
                f'add_variable raised {type(exc).__name__} rather than a '
                f'labelled ValueError for a real bound of {bound.bit_length()} '
                f'bits: {exc}'
            )

    @pytest.mark.parametrize('bound', _INSIDE_FLOAT64)
    @pytest.mark.parametrize('key', ['min', 'max'])
    def test_a_bound_inside_the_float64_range_is_still_accepted(self, bound, key):
        """The guard rejects unrepresentable, not large. ``2**1023`` returned
        200 before this round and has to keep doing so."""
        space = SearchSpace()
        bounds = {'min': -1.0, 'max': 1.0}
        bounds[key] = -bound if key == 'min' else bound
        space.add_variable('x1', 'real', **bounds)
        assert space.variables[0][key] == bounds[key]
        assert len(space.skopt_dimensions) == 1

    @pytest.mark.parametrize('bound', _OUTSIDE_FLOAT64 + _INSIDE_FLOAT64)
    @pytest.mark.parametrize('key', ['min', 'max'])
    def test_an_integer_variable_takes_the_same_bound_at_any_magnitude(
        self, bound, key
    ):
        """``skopt.Integer`` keeps Python ints at full width and converts
        nothing, so the float64 rule does not apply to it and must not be
        allowed to leak across."""
        space = SearchSpace()
        bounds = {'min': -1, 'max': 1}
        bounds[key] = -bound if key == 'min' else bound
        space.add_variable('x2', 'integer', **bounds)
        assert space.variables[0][key] == bounds[key]
        assert len(space.skopt_dimensions) == 1

    def test_the_boundary_that_makes_this_test_discriminate_is_real(self):
        """If ``float()`` stopped overflowing, the fix would be unmotivated."""
        assert float(2**1023) == 2.0**1023
        with pytest.raises(OverflowError):
            float(2**1024)
        assert 2**1023 <= sys.float_info.max
        assert not 2**1024 <= sys.float_info.max

    def test_the_rejected_variable_is_not_half_registered(self):
        """``add_variable`` builds the dimension before appending anything, and
        a bound rejected here must not leave ``variables`` and
        ``skopt_dimensions`` out of step -- they are positionally paired."""
        space = SearchSpace()
        space.add_variable('x1', 'real', min=0.0, max=10.0)
        with pytest.raises(ValueError):
            space.add_variable('x2', 'real', min=0.0, max=2**1024)
        assert [v['name'] for v in space.variables] == ['x1']
        assert [d.name for d in space.skopt_dimensions] == ['x1']

    def test_a_huge_bound_is_not_printed_digit_by_digit(self):
        """The message itself must not raise.

        CPython 3.11+ caps ``int.__str__`` at 4300 digits and raises ValueError
        past it, which would replace the labelled rejection with an unlabelled
        one from inside its own f-string.
        """
        space = SearchSpace()
        with pytest.raises(ValueError) as exc:
            space.add_variable('x1', 'real', min=0.0, max=2**60000)
        detail = str(exc.value)
        assert 'x1' in detail and 'max' in detail
        assert '60001 bits' in detail, 'the magnitude is described, not printed'
        assert len(detail) < 400


class TestDiscreteValuesAnswerToTheSameRule:
    """``allowed_values`` becomes a Categorical of ``float()``-coerced values.

    The coercion ran before any validation, so ``float()`` -- the very
    conversion the guard exists to stand in front of -- raised OverflowError
    first for an entry outside the float64 range. Identical defect, identical
    500, one variable type over: exactly the uniform-variable-type blind spot
    that let the previous three rounds through, so it is closed in the same
    round rather than left to be the fourth.
    """

    @pytest.mark.parametrize('bad', _OUTSIDE_FLOAT64)
    def test_an_entry_outside_the_float64_range_is_a_labelled_value_error(self, bad):
        space = SearchSpace()
        with pytest.raises(ValueError) as exc:
            space.add_variable('x3', 'discrete', allowed_values=[0.5, bad, 7.25])
        detail = str(exc.value)
        assert 'x3' in detail
        assert 'allowed_values[1]' in detail, 'the message must name the index'
        assert space.variables == []

    @pytest.mark.parametrize('bad', _OUTSIDE_FLOAT64)
    def test_it_is_not_an_arithmetic_error(self, bad):
        space = SearchSpace()
        try:
            space.add_variable('x3', 'discrete', allowed_values=[0.5, bad])
        except ValueError:
            pass
        except Exception as exc:  # pragma: no cover - the defect being fixed
            pytest.fail(f'add_variable raised {type(exc).__name__}: {exc}')

    @pytest.mark.parametrize('good', _INSIDE_FLOAT64)
    def test_an_entry_inside_it_still_loads(self, good):
        space = SearchSpace()
        space.add_variable('x3', 'discrete', allowed_values=[0.5, good])
        assert space.variables[0]['allowed_values'] == [0.5, float(good)]

    def test_a_quoted_entry_still_survives_the_coercion(self):
        """Branch item M5, deliberately left alone: an ``allowed_values`` entry
        may arrive as a string and be coerced, while a quoted bound is refused.
        Validating before the coercion must not have quietly narrowed that."""
        space = SearchSpace()
        space.add_variable('x3', 'discrete', allowed_values=['3.0', 7.0])
        assert space.variables[0]['allowed_values'] == [3.0, 7.0]

    @pytest.mark.parametrize('bad', ['nan', 'inf', '-inf'])
    def test_a_quoted_non_finite_entry_is_still_caught_after_it(self, bad):
        """The check behind the coercion is still doing its job."""
        space = SearchSpace()
        with pytest.raises(ValueError, match='must be finite'):
            space.add_variable('x3', 'discrete', allowed_values=[0.5, bad])
        assert space.variables == []


class TestConstraintValuesAreUnchangedAtEveryMagnitude:
    """The rule is about dimensions, and a constraint builds none.

    ``rhs`` and the coefficients are checked by ``_validate_finite_number``
    directly, never by ``_validate_bound``, and they are stored as given and
    used in arithmetic that Python performs at full width. Narrowing them to
    match the ``real`` rule would refuse working files for a reason constraints
    do not have, so this pins that fix round 3 did not touch them.
    """

    @pytest.mark.parametrize('value', _OUTSIDE_FLOAT64 + _INSIDE_FLOAT64)
    def test_a_rhs_of_any_magnitude_is_accepted(self, value):
        space = SearchSpace()
        space.add_variable('x1', 'real', min=0.0, max=1.0)
        space.add_constraint('inequality', {'x1': 1.0}, rhs=value, name='c_a')
        assert space.get_constraints()[0]['rhs'] == value

    @pytest.mark.parametrize('value', _OUTSIDE_FLOAT64 + _INSIDE_FLOAT64)
    def test_a_coefficient_of_any_magnitude_is_accepted(self, value):
        space = SearchSpace()
        space.add_variable('x1', 'real', min=0.0, max=1.0)
        space.add_constraint('inequality', {'x1': -value}, rhs=0.0, name='c_a')
        assert space.get_constraints()[0]['coefficients']['x1'] == -value

    @pytest.mark.parametrize('value', _OUTSIDE_FLOAT64)
    def test_a_non_finite_constraint_value_is_still_refused(self, value):
        """Unchanged does not mean unguarded."""
        space = SearchSpace()
        space.add_variable('x1', 'real', min=0.0, max=1.0)
        with pytest.raises(ValueError, match='must be finite'):
            space.add_constraint(
                'inequality', {'x1': 1.0}, rhs=float('nan'), name='c_a'
            )


class TestEveryBoundTakingVariableTypeDeclaresItsDomain:
    """What a bound may be is a property of the dimension, so it is declared.

    ``_FLOAT64_BACKED_VAR_TYPES`` and ``_ARBITRARY_PRECISION_VAR_TYPES``
    partition the variable types that validate a bound. A new bound-taking type
    added to ``add_variable`` without an entry in one of them fails here rather
    than silently inheriting whichever branch of the ``if`` was written first
    -- which, given that this defect has now been rediscovered on a second
    variable type, is the failure mode worth a test of its own.
    """

    # Every type add_variable understands. Listed because there is no registry
    # to read it out of; the test below is what notices when the list and
    # add_variable disagree.
    ALL_VAR_TYPES = ('real', 'integer', 'discrete', 'categorical', 'context')

    _NAN_KWARGS = {
        'real': {'min': 0.0, 'max': float('nan')},
        'integer': {'min': 0, 'max': float('nan')},
        'discrete': {'allowed_values': [0.5, float('nan')]},
        'categorical': {'values': ['A', float('nan')]},
        'context': {},
    }

    def _validates_numbers(self, var_type):
        try:
            SearchSpace().add_variable(
                'x1', var_type, **self._NAN_KWARGS[var_type]
            )
        except ValueError as exc:
            return 'must be finite' in str(exc)
        return False

    def test_the_two_domains_are_disjoint(self):
        from alchemist_core.data.search_space import (
            _ARBITRARY_PRECISION_VAR_TYPES, _FLOAT64_BACKED_VAR_TYPES,
        )
        assert not (_FLOAT64_BACKED_VAR_TYPES & _ARBITRARY_PRECISION_VAR_TYPES)

    def test_they_cover_exactly_the_types_that_validate_a_bound(self):
        from alchemist_core.data.search_space import (
            _ARBITRARY_PRECISION_VAR_TYPES, _FLOAT64_BACKED_VAR_TYPES,
        )
        declared = _FLOAT64_BACKED_VAR_TYPES | _ARBITRARY_PRECISION_VAR_TYPES
        validating = {t for t in self.ALL_VAR_TYPES if self._validates_numbers(t)}
        assert validating == declared, (
            f'types that validate a bound: {sorted(validating)}; '
            f'types that declare a domain: {sorted(declared)}'
        )

    @pytest.mark.parametrize('var_type', ['real', 'discrete'])
    def test_a_float64_backed_type_refuses_an_unrepresentable_bound(self, var_type):
        kwargs = ({'min': 0.0, 'max': 2**1024} if var_type == 'real'
                  else {'allowed_values': [0.5, 2**1024]})
        with pytest.raises(ValueError, match='float64'):
            SearchSpace().add_variable('x1', var_type, **kwargs)

    def test_an_arbitrary_precision_type_accepts_one(self):
        space = SearchSpace()
        space.add_variable('x2', 'integer', min=0, max=2**1024)
        assert space.variables[0]['max'] == 2**1024
