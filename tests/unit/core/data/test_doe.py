"""
Test Design of Experiments (DoE) functionality.
"""

import pytest
import numpy as np
from alchemist_core.data.search_space import SearchSpace
from alchemist_core.utils.doe import generate_initial_design
from alchemist_core.session import OptimizationSession


class TestDoE:
    """Test initial design generation methods."""
    
    def setup_method(self):
        """Create a simple search space for testing."""
        self.space = SearchSpace()
        self.space.add_variable('temperature', 'real', min=300, max=500)
        self.space.add_variable('pressure', 'real', min=1, max=10)
        self.space.add_variable('catalyst', 'categorical', values=['A', 'B', 'C'])
    
    def test_random_sampling(self):
        """Test random sampling method."""
        points = generate_initial_design(
            self.space,
            method='random',
            n_points=10,
            random_seed=42
        )
        
        assert len(points) == 10
        assert all('temperature' in p for p in points)
        assert all('pressure' in p for p in points)
        assert all('catalyst' in p for p in points)
        
        # Check bounds
        for point in points:
            assert 300 <= point['temperature'] <= 500
            assert 1 <= point['pressure'] <= 10
            assert point['catalyst'] in ['A', 'B', 'C']
    
    def test_lhs_sampling(self):
        """Test Latin Hypercube Sampling."""
        points = generate_initial_design(
            self.space,
            method='lhs',
            n_points=10,
            random_seed=42,
            lhs_criterion='maximin'
        )
        
        assert len(points) == 10
        # LHS should provide good space coverage
        temps = [p['temperature'] for p in points]
        assert max(temps) - min(temps) > 100  # Should span the space
    
    def test_sobol_sampling(self):
        """Test Sobol sequence sampling."""
        points = generate_initial_design(
            self.space,
            method='sobol',
            n_points=10
        )
        
        assert len(points) == 10
    
    def test_halton_sampling(self):
        """Test Halton sequence sampling."""
        points = generate_initial_design(
            self.space,
            method='halton',
            n_points=10
        )
        
        assert len(points) == 10
    
    def test_hammersly_sampling(self):
        """Test Hammersly sequence sampling."""
        points = generate_initial_design(
            self.space,
            method='hammersly',
            n_points=10
        )
        
        assert len(points) == 10
    
    def test_invalid_method(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown sampling method"):
            generate_initial_design(self.space, method='invalid_method')
    
    def test_empty_search_space(self):
        """Test that empty search space raises error."""
        empty_space = SearchSpace()
        with pytest.raises(ValueError, match="no variables"):
            generate_initial_design(empty_space, method='lhs')
    
    def test_reproducibility_with_seed(self):
        """Test that random seed produces reproducible results."""
        points1 = generate_initial_design(
            self.space,
            method='random',
            n_points=5,
            random_seed=123
        )
        
        points2 = generate_initial_design(
            self.space,
            method='random',
            n_points=5,
            random_seed=123
        )
        
        # Should be identical
        for p1, p2 in zip(points1, points2):
            assert p1['temperature'] == p2['temperature']
            assert p1['pressure'] == p2['pressure']
            assert p1['catalyst'] == p2['catalyst']


class TestClassicalDoE:
    """Test classical RSM design methods."""

    def _make_space(self, n_real=3, n_int=0, n_cat=0):
        """Helper: build a SearchSpace with n_real + n_int + n_cat variables."""
        space = SearchSpace()
        for i in range(n_real):
            space.add_variable(f'x{i+1}', 'real', min=0, max=10)
        for i in range(n_int):
            space.add_variable(f'n{i+1}', 'integer', min=1, max=20)
        for i in range(n_cat):
            space.add_variable(f'cat{i+1}', 'categorical', values=['A', 'B', 'C'])
        return space

    # ---- Full Factorial ----

    def test_full_factorial_2level(self):
        """2^3 = 8 factorial runs + 1 center = 9 total."""
        space = self._make_space(n_real=3)
        points = generate_initial_design(space, method='full_factorial',
                                         n_levels=2, n_center=1)
        assert len(points) == 2**3 + 1
        # All factorial points at bounds
        for p in points[:-1]:
            for name in ('x1', 'x2', 'x3'):
                assert p[name] in (0.0, 10.0)

    def test_full_factorial_3level(self):
        """3^2 = 9 factorial runs + 1 center = 10 total."""
        space = self._make_space(n_real=2)
        points = generate_initial_design(space, method='full_factorial',
                                         n_levels=3, n_center=1)
        assert len(points) == 3**2 + 1
        # 3-level values should be {0, 5, 10}
        vals = {p['x1'] for p in points[:-1]}
        assert vals == {0.0, 5.0, 10.0}

    def test_full_factorial_with_categorical(self):
        """Mixed real + categorical: 2 real levels x 3 categories = 12."""
        space = self._make_space(n_real=2, n_cat=1)
        points = generate_initial_design(space, method='full_factorial',
                                         n_levels=2, n_center=0)
        assert len(points) == 2 * 2 * 3
        cats = {p['cat1'] for p in points}
        assert cats == {'A', 'B', 'C'}

    def test_full_factorial_no_center(self):
        """n_center=0 produces only factorial points."""
        space = self._make_space(n_real=2)
        points = generate_initial_design(space, method='full_factorial',
                                         n_levels=2, n_center=0)
        assert len(points) == 2**2

    # ---- Fractional Factorial ----

    def test_fractional_factorial(self):
        """4 factors with default generator: should produce 2^(4-1)=8 + 1 center."""
        space = self._make_space(n_real=4)
        points = generate_initial_design(space, method='fractional_factorial',
                                         n_center=1)
        assert len(points) == 8 + 1
        # All factorial points at bounds
        for p in points[:-1]:
            for name in ('x1', 'x2', 'x3', 'x4'):
                assert p[name] in (0.0, 10.0)

    def test_fractional_factorial_explicit_generator(self):
        """Explicit generator for 3 factors: 'a b ab' gives 2^(3-1)=4 runs."""
        space = self._make_space(n_real=3)
        points = generate_initial_design(space, method='fractional_factorial',
                                         generators='a b ab', n_center=0)
        assert len(points) == 4

    def test_fractional_factorial_auto_generator(self):
        """5 factors with no generator should still produce a valid design."""
        space = self._make_space(n_real=5)
        points = generate_initial_design(space, method='fractional_factorial',
                                         n_center=1)
        # Should produce some reasonable number of runs
        assert len(points) >= 5  # at least as many runs as factors
        assert all('x1' in p for p in points)

    # ---- CCD ----

    def test_ccd_circumscribed(self):
        """3 factors: 2^3 + 2*3 + 2 center = 16 runs."""
        space = self._make_space(n_real=3)
        points = generate_initial_design(space, method='ccd', n_center=1,
                                         ccd_face='circumscribed')
        # 8 factorial + 6 axial + 2 center (1 in factorial block, 1 in axial block)
        assert len(points) == 16

    def test_ccd_faced(self):
        """Faced CCD: axial points should be at bounds."""
        space = self._make_space(n_real=3)
        points = generate_initial_design(space, method='ccd', n_center=1,
                                         ccd_face='faced')
        # Check all points within bounds
        for p in points:
            for name in ('x1', 'x2', 'x3'):
                assert 0 <= p[name] <= 10

    def test_ccd_rejects_categoricals(self):
        """CCD should reject categorical variables."""
        space = self._make_space(n_real=3, n_cat=1)
        with pytest.raises(ValueError, match="does not support categorical"):
            generate_initial_design(space, method='ccd')

    # ---- Box-Behnken ----

    def test_box_behnken(self):
        """3 factors: 12 edge + 1 center = 13 runs."""
        space = self._make_space(n_real=3)
        points = generate_initial_design(space, method='box_behnken', n_center=1)
        assert len(points) == 13

    def test_box_behnken_rejects_2_factors(self):
        """Box-Behnken requires >= 3 continuous factors."""
        space = self._make_space(n_real=2)
        with pytest.raises(ValueError, match="at least 3"):
            generate_initial_design(space, method='box_behnken')

    def test_box_behnken_rejects_categoricals(self):
        """Box-Behnken should reject categorical variables."""
        space = self._make_space(n_real=3, n_cat=1)
        with pytest.raises(ValueError, match="does not support categorical"):
            generate_initial_design(space, method='box_behnken')

    # ---- Cross-cutting ----

    def test_classical_bounds_respected(self):
        """All classical designs should keep points within variable bounds."""
        space = self._make_space(n_real=3)
        for method_name in ('full_factorial', 'ccd', 'box_behnken', 'plackett_burman'):
            points = generate_initial_design(space, method=method_name, n_center=1)
            for p in points:
                for name in ('x1', 'x2', 'x3'):
                    assert 0 <= p[name] <= 10, \
                        f"{method_name}: {name}={p[name]} out of bounds [0, 10]"

    def test_classical_integer_rounding(self):
        """Integer variables should produce integer values."""
        space = self._make_space(n_real=2, n_int=1)
        points = generate_initial_design(space, method='full_factorial',
                                         n_levels=2, n_center=1)
        for p in points:
            assert isinstance(p['n1'], int), f"n1={p['n1']} is not int"

    def test_center_points(self):
        """Center points should be at variable midpoints."""
        space = self._make_space(n_real=3)
        points = generate_initial_design(space, method='full_factorial',
                                         n_levels=2, n_center=2)
        # Last 2 points are center points
        for cp in points[-2:]:
            for name in ('x1', 'x2', 'x3'):
                assert cp[name] == 5.0, f"Center point {name}={cp[name]} != 5.0"

    def test_n_points_ignored_for_classical(self):
        """Passing n_points to a classical method should not affect the result."""
        space = self._make_space(n_real=3)
        points = generate_initial_design(space, method='box_behnken',
                                         n_points=999, n_center=1)
        # Box-Behnken for 3 factors = 13, not 999
        assert len(points) == 13

    # ---- Plackett-Burman ----

    def test_plackett_burman(self):
        """5 factors: next multiple of 4 above 5 is 8 runs + 1 center = 9."""
        space = self._make_space(n_real=5)
        points = generate_initial_design(space, method='plackett_burman', n_center=1)
        assert len(points) == 8 + 1
        # All screening points at bounds
        for p in points[:-1]:
            for name in ('x1', 'x2', 'x3', 'x4', 'x5'):
                assert p[name] in (0.0, 10.0)

    def test_plackett_burman_3_factors(self):
        """3 factors: 4 PB runs + 0 center."""
        space = self._make_space(n_real=3)
        points = generate_initial_design(space, method='plackett_burman', n_center=0)
        assert len(points) == 4
        for p in points:
            for name in ('x1', 'x2', 'x3'):
                assert p[name] in (0.0, 10.0)

    def test_plackett_burman_rejects_categoricals(self):
        """Plackett-Burman should reject categorical variables."""
        space = self._make_space(n_real=3, n_cat=1)
        with pytest.raises(ValueError, match="does not support categorical"):
            generate_initial_design(space, method='plackett_burman')

    def test_plackett_burman_integer_rounding(self):
        """Integer variables should produce integer values in PB designs."""
        space = self._make_space(n_real=2, n_int=1)
        points = generate_initial_design(space, method='plackett_burman', n_center=0)
        for p in points:
            assert isinstance(p['n1'], int), f"n1={p['n1']} is not int"

    # ---- GSD ----

    def test_gsd_continuous_only(self):
        """GSD with 3 continuous 2-level factors, reduction=2."""
        space = self._make_space(n_real=3)
        points = generate_initial_design(space, method='gsd',
                                          n_levels=2, gsd_reduction=2)
        # Full factorial would be 2^3=8, GSD should be ~8/2=4
        assert len(points) < 8
        assert len(points) >= 2
        for p in points:
            for name in ('x1', 'x2', 'x3'):
                assert p[name] in (0.0, 10.0)

    def test_gsd_with_categorical(self):
        """GSD supports mixed real + categorical variables."""
        space = self._make_space(n_real=2, n_cat=1)
        points = generate_initial_design(space, method='gsd',
                                          n_levels=2, gsd_reduction=2)
        # Full factorial: 2*2*3=12, GSD ~12/2=6
        assert len(points) < 12
        assert len(points) >= 2
        cats = {p['cat1'] for p in points}
        assert cats.issubset({'A', 'B', 'C'})

    def test_gsd_3level(self):
        """GSD with 3 levels per continuous factor."""
        space = self._make_space(n_real=3)
        points = generate_initial_design(space, method='gsd',
                                          n_levels=3, gsd_reduction=3)
        # Full factorial would be 3^3=27, GSD should be ~27/3=9
        assert len(points) < 27
        assert len(points) >= 3
        # 3-level values should be in {0, 5, 10}
        vals = {p['x1'] for p in points}
        assert vals.issubset({0.0, 5.0, 10.0})

    def test_gsd_bounds_respected(self):
        """GSD points should stay within variable bounds."""
        space = self._make_space(n_real=3, n_cat=1)
        points = generate_initial_design(space, method='gsd',
                                          n_levels=3, gsd_reduction=2)
        for p in points:
            for name in ('x1', 'x2', 'x3'):
                assert 0 <= p[name] <= 10, f"GSD: {name}={p[name]} out of bounds"
            assert p['cat1'] in ('A', 'B', 'C')

    def test_gsd_integer_rounding(self):
        """Integer variables should produce integer values in GSD designs."""
        space = self._make_space(n_real=1, n_int=1)
        points = generate_initial_design(space, method='gsd',
                                          n_levels=3, gsd_reduction=2)
        for p in points:
            assert isinstance(p['n1'], int), f"n1={p['n1']} is not int"


class TestSessionDoE:
    """Test DoE integration with OptimizationSession."""
    
    def test_session_generate_initial_design(self):
        """Test generate_initial_design() method on Session API."""
        session = OptimizationSession()
        session.add_variable('temp', 'real', bounds=(300, 500))
        session.add_variable('flow', 'real', bounds=(1, 10))
        
        points = session.generate_initial_design(method='lhs', n_points=8)
        
        assert len(points) == 8
        assert all('temp' in p and 'flow' in p for p in points)
    
    def test_session_no_variables_error(self):
        """Test that session raises error if no variables defined."""
        session = OptimizationSession()
        
        with pytest.raises(ValueError, match="No variables defined"):
            session.generate_initial_design()
    
    def test_workflow_initial_design_then_add(self):
        """Test complete workflow: generate design, add experiments, train."""
        session = OptimizationSession()
        session.add_variable('x', 'real', min=0, max=10)
        session.add_variable('y', 'real', min=0, max=10)

        # Generate initial design
        points = session.generate_initial_design('lhs', n_points=10, random_seed=42)

        # Simulate experiments (use simple function: z = x + y)
        for point in points:
            output = point['x'] + point['y']
            session.add_experiment(point, output=output)

        # Verify data was added
        assert len(session.experiment_manager.df) == 10

        # Train model
        result = session.train_model(backend='sklearn', kernel='rbf')
        assert result is not None
        assert session.model is not None

    def test_ccd_session_integration(self):
        """Full CCD workflow via OptimizationSession."""
        session = OptimizationSession()
        session.add_variable('x1', 'real', min=0, max=10)
        session.add_variable('x2', 'real', min=0, max=10)
        session.add_variable('x3', 'real', min=0, max=10)

        points = session.generate_initial_design(
            method='ccd', n_center=1, ccd_face='circumscribed'
        )
        assert len(points) == 16

        # Add experiments with simple response surface
        for p in points:
            output = p['x1']**2 + p['x2'] + p['x3']
            session.add_experiment(p, output=output)

        assert len(session.experiment_manager.df) == 16

        # Train model
        result = session.train_model(backend='sklearn', kernel='rbf')
        assert result is not None

    def test_plackett_burman_session_integration(self):
        """Plackett-Burman workflow via OptimizationSession."""
        session = OptimizationSession()
        for i in range(5):
            session.add_variable(f'x{i+1}', 'real', min=0, max=10)

        points = session.generate_initial_design(method='plackett_burman', n_center=1)
        assert len(points) == 9  # 8 PB runs + 1 center

        for p in points:
            output = sum(p.values())
            session.add_experiment(p, output=output)

        assert len(session.experiment_manager.df) == 9

    def test_gsd_session_integration(self):
        """GSD workflow with mixed variables via OptimizationSession."""
        session = OptimizationSession()
        session.add_variable('temp', 'real', bounds=(300, 500))
        session.add_variable('pressure', 'real', bounds=(1, 10))
        session.add_variable('catalyst', 'categorical', categories=['Pt', 'Pd', 'Ru'])

        points = session.generate_initial_design(
            method='gsd', n_levels=2, gsd_reduction=2
        )
        assert len(points) >= 2
        assert len(points) < 2 * 2 * 3  # less than full factorial

        for p in points:
            assert 'temp' in p and 'pressure' in p and 'catalyst' in p
            assert p['catalyst'] in ('Pt', 'Pd', 'Ru')
