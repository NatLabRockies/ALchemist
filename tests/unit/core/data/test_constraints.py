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
