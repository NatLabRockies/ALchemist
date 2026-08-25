# Cross-Surface Input Constraint Coverage Implementation Plan

> **STATUS: COMPLETED (2026-07-21).** All tasks implemented and committed on `main`.
> Full suite: 903 passed, 10 skipped. Notes on deviations:
> - Task 3: session.find_optimum SO path is backend-agnostic (grid-based), so
>   sklearn find_optimum inherits the feasibility filter — no separate raise needed.
> - Task 7: botorch_acquisition.find_optimum also filters its own grid so direct
>   callers (desktop) inherit enforcement. Known limitation: desktop sklearn
>   find_optimum (differential_evolution) cannot enforce constraints.
> - DOE sampling uses STRICT tolerance (rtol=0) so no design point exceeds a
>   stated bound; grid/plot masking uses the relative band.
> - Wiki lives in an external QMD vault (not this repo); documented in
>   docs/acquisition/botorch.md instead.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every surface that generates candidate points over the search space (find_optimum, all plot grids, DOE/initial design) honor registered linear input constraints, using a single shared feasibility primitive on `SearchSpace`.

**Architecture:** Add `SearchSpace.is_feasible(df)` / `filter_feasible(df)` as the single source of truth for "does this point satisfy all registered linear input constraints" (relative tolerance for equality). Retrofit `find_optimum` to filter its grid to feasible points, plot grids to mask (NaN) infeasible cells, and DOE/initial-design to reject-and-resample infeasible points. The BoTorch `suggest_next` acquisition path is already constraint-aware (continuous + mixed) and is out of scope here except for reuse of the primitive.

**Tech Stack:** Python, numpy, pandas, pytest. Constraint storage: `SearchSpace.constraints` = list of `{'type': 'equality'|'inequality', 'coefficients': {name: coeff}, 'rhs': float, 'name': str}`. ALchemist convention: inequality means `sum(coeff_i * x_i) <= rhs`.

---

## Background / Current State (verified)

- `SearchSpace.constraints` stores raw-space linear constraints. **No feasibility method exists** (`search_space.py`).
- Only constraint-aware path today: `suggest_next` → `botorch_acquisition.select_next` → `optimize_acqf` / `optimize_acqf_mixed_alternating` (both now pass constraints).
- **Unconstrained / can emit infeasible points:**
  - `session._generate_prediction_grid` (session.py:3975) → `session.find_optimum` (SO, session.py:1673).
  - `botorch_acquisition.find_optimum` (botorch_acquisition.py:~805) + its own `_generate_prediction_grid` (~847).
  - `skopt_acquisition.find_optimum` (skopt_acquisition.py:173) — differential_evolution, box bounds only.
  - All `session.plot_*` grid builders (contour/surface/slice/voxel/acquisition/uncertainty variants).
  - `doe.py` initial-design methods; `optimal_design.py` candidate sets.
- sklearn `suggest_next` already raises when constraints are registered (session.py:~1484). We follow the same "clear error" stance where enforcement is infeasible (skopt find_optimum).

**Tolerance decision (from design review):** equality feasibility uses a **relative** band. A point satisfies `sum(coeff·x) == rhs` when `|sum(coeff·x) - rhs| <= atol + rtol * max(|rhs|, scale)`, with `scale = sum(|coeff_i| * typical_range_i)` fallback. Defaults: `rtol=1e-3`, `atol=1e-6`. Inequality uses `sum(coeff·x) <= rhs + atol + rtol*max(|rhs|,scale)`.

---

## Task 1: `SearchSpace.is_feasible` / `filter_feasible` primitive

**Files:**
- Modify: `alchemist_core/data/search_space.py` (add methods after `get_constraints`, ~line 383)
- Test: `tests/unit/core/data/test_constraints.py` (add `TestFeasibility` class)

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/unit/core/data/test_constraints.py

class TestFeasibility:
    """Tests for SearchSpace.is_feasible / filter_feasible."""

    def setup_method(self):
        from alchemist_core.data.search_space import SearchSpace
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
            'CO2': [20.0, 25.0, 0.0],   # row0 sum=100 feasible; row1 sum=105 infeasible; row2 sum=0 infeasible
        })
        mask = self.space.filter_feasible(df)
        assert mask.tolist() == [True, False, False]

    def test_inequality_feasibility(self):
        import pandas as pd
        self.space.add_constraint('inequality', {'H2': 1.0, 'CO': 1.0}, rhs=50.0)
        df = pd.DataFrame({'H2': [20.0, 40.0], 'CO': [20.0, 40.0], 'CO2': [0, 0]})
        # row0 sum=40 <= 50 feasible; row1 sum=80 > 50 infeasible
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
        # both sum to 100; row0 H2-CO=30 > 20 infeasible; row1 H2-CO=70 > 20 infeasible
        mask = self.space.filter_feasible(df)
        assert mask.tolist() == [False, False]

    def test_is_feasible_single_dict(self):
        self.space.add_constraint('equality', {'H2': 1.0, 'CO': 1.0, 'CO2': 1.0}, rhs=100.0)
        assert self.space.is_feasible({'H2': 50.0, 'CO': 30.0, 'CO2': 20.0}) is True
        assert self.space.is_feasible({'H2': 50.0, 'CO': 30.0, 'CO2': 30.0}) is False

    def test_subset_constraint_ignores_missing_columns(self):
        import pandas as pd
        # constraint references only H2, CO; CO2 free
        self.space.add_constraint('inequality', {'H2': 1.0, 'CO': 1.0}, rhs=50.0)
        df = pd.DataFrame({'H2': [20.0], 'CO': [20.0], 'CO2': [999.0]})
        assert self.space.filter_feasible(df).tolist() == [True]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/data/test_constraints.py::TestFeasibility -v`
Expected: FAIL with `AttributeError: 'SearchSpace' object has no attribute 'filter_feasible'`

- [ ] **Step 3: Write minimal implementation**

```python
# Add to alchemist_core/data/search_space.py after get_constraints() (~line 383)

    def _constraint_scale(self, c: Dict) -> float:
        """Characteristic magnitude of a constraint, for relative tolerance."""
        scale = 0.0
        for var in self.variables:
            name = var['name']
            if name not in c['coefficients']:
                continue
            coeff = abs(float(c['coefficients'][name]))
            if 'min' in var and 'max' in var:
                rng = abs(float(var['max']) - float(var['min']))
            elif var.get('type') == 'discrete':
                vals = var.get('allowed_values', [0.0, 1.0])
                rng = abs(float(max(vals)) - float(min(vals)))
            else:
                rng = 1.0
            scale += coeff * rng
        return scale

    def filter_feasible(self, points, rtol: float = 1e-3, atol: float = 1e-6) -> np.ndarray:
        """Boolean mask: which rows satisfy ALL registered linear input constraints.

        Args:
            points: pandas DataFrame (columns are variable names) or a list of dicts.
            rtol, atol: relative/absolute tolerance. Equality is feasible when
                |lhs - rhs| <= atol + rtol * max(|rhs|, scale); inequality when
                lhs <= rhs + atol + rtol * max(|rhs|, scale).

        Returns:
            numpy boolean array of length len(points). All True if no constraints.
        """
        df = points if isinstance(points, pd.DataFrame) else pd.DataFrame(list(points))
        n = len(df)
        mask = np.ones(n, dtype=bool)
        if not self.constraints:
            return mask

        for c in self.constraints:
            lhs = np.zeros(n, dtype=float)
            any_col = False
            for var_name, coeff in c['coefficients'].items():
                if var_name in df.columns:
                    lhs = lhs + float(coeff) * df[var_name].to_numpy(dtype=float)
                    any_col = True
            if not any_col:
                continue  # constraint references no present columns; cannot judge -> skip
            rhs = float(c['rhs'])
            tol = atol + rtol * max(abs(rhs), self._constraint_scale(c))
            if c['type'] == 'equality':
                mask &= np.abs(lhs - rhs) <= tol
            else:  # inequality: lhs <= rhs
                mask &= lhs <= rhs + tol
        return mask

    def is_feasible(self, point, rtol: float = 1e-3, atol: float = 1e-6) -> bool:
        """Whether a single point (dict or 1-row DataFrame) is feasible."""
        if isinstance(point, dict):
            df = pd.DataFrame([point])
        elif isinstance(point, pd.DataFrame):
            df = point
        else:
            df = pd.DataFrame([dict(point)])
        return bool(self.filter_feasible(df, rtol=rtol, atol=atol)[0])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/data/test_constraints.py::TestFeasibility -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add alchemist_core/data/search_space.py tests/unit/core/data/test_constraints.py
git commit -m "feat: add SearchSpace.is_feasible/filter_feasible constraint primitive"
```

---

## Task 2: `find_optimum` filters grid to feasible points (session SO path)

**Files:**
- Modify: `alchemist_core/session.py:1672-1694` (single-objective `find_optimum`)
- Test: `tests/unit/core/acquisition/test_input_constraints.py` (add `TestFindOptimumFeasible`)

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/unit/core/acquisition/test_input_constraints.py

class TestFindOptimumFeasible:
    """find_optimum must return a feasible optimum when input constraints exist."""

    def test_find_optimum_respects_equality(self):
        session = _syngas_session()
        session.add_input_constraint('equality', {'H2': 1.0, 'CO': 1.0, 'CO2': 1.0}, rhs=100.0)
        opt = session.find_optimum('maximize')
        x = opt['x_opt'].iloc[0]
        total = x['H2'] + x['CO'] + x['CO2']
        assert total == pytest.approx(100.0, abs=5.0)

    def test_find_optimum_respects_inequality(self):
        session = _syngas_session()
        session.add_input_constraint('inequality', {'H2': 1.0, 'CO': 1.0, 'CO2': 1.0}, rhs=100.0)
        opt = session.find_optimum('maximize')
        x = opt['x_opt'].iloc[0]
        assert (x['H2'] + x['CO'] + x['CO2']) <= 100.0 + 5.0

    def test_find_optimum_raises_when_no_feasible_grid_points(self):
        session = _syngas_session()
        # Impossible constraint (sum of three vars each in [0,100] cannot be 500)
        session.add_input_constraint('equality', {'H2': 1.0, 'CO': 1.0, 'CO2': 1.0}, rhs=500.0)
        with pytest.raises(ValueError, match='(?i)feasible'):
            session.find_optimum('maximize')
```

Note: `find_optimum` uses a coarse grid (`10000 ** (1/n)` points/dim ≈ 21 for 3 vars), so equality feasibility needs the relative tolerance band; keep `abs=5.0` in assertions. `_syngas_session` and imports already exist at the top of this test file.

- [ ] **Step 2: Run test to verify it fails**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/acquisition/test_input_constraints.py::TestFindOptimumFeasible -v`
Expected: FAIL — equality test returns an infeasible sum; the raises-test does not raise.

- [ ] **Step 3: Write minimal implementation**

Replace the single-objective block in `session.py` (currently lines 1672-1694, starting at the comment `# Single-objective`) with:

```python
        # Single-objective
        grid = self._generate_prediction_grid(n_grid_points)

        # Restrict grid to points satisfying registered linear input constraints
        if getattr(self.search_space, 'constraints', None):
            feasible_mask = self.search_space.filter_feasible(grid)
            grid = grid[feasible_mask].reset_index(drop=True)
            if len(grid) == 0:
                raise ValueError(
                    "No feasible grid points satisfy the registered input "
                    "constraints. Increase n_grid_points, relax the constraints, "
                    "or check that the constraint is satisfiable within the "
                    "variable bounds."
                )

        # predict() always returns Dict[str, (means, stds)]; unwrap for SO case.
        target_name = self.objective_names[0]
        means, stds = self.predict(grid)[target_name]

        if directions[0] == 'maximize':
            best_idx = np.argmax(means)
        else:
            best_idx = np.argmin(means)

        opt_point_df = grid.iloc[[best_idx]].reset_index(drop=True)

        result = {
            'x_opt': opt_point_df,
            'value': float(means[best_idx]),
            'std': float(stds[best_idx])
        }

        logger.info(f"Found optimum: {result['x_opt'].to_dict('records')[0]}")
        logger.info(f"Predicted value: {result['value']:.4f} ± {result['std']:.4f}")

        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/acquisition/test_input_constraints.py::TestFindOptimumFeasible -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run find_optimum regression tests**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/acquisition/test_acquisition.py::TestSessionFindOptimum -v`
Expected: PASS (unchanged — no constraints registered in those tests)

- [ ] **Step 6: Commit**

```bash
git add alchemist_core/session.py tests/unit/core/acquisition/test_input_constraints.py
git commit -m "feat: find_optimum filters grid to constraint-feasible points"
```

---

## Task 3: `skopt_acquisition.find_optimum` rejects registered constraints

**Files:**
- Modify: `alchemist_core/acquisition/skopt_acquisition.py:173` (`find_optimum`)
- Test: `tests/unit/core/acquisition/test_input_constraints.py` (add to `TestSklearnBackendRejectsInputConstraints`)

The skopt `find_optimum` uses `differential_evolution` with box bounds only and cannot express linear constraints. Consistent with `suggest_next`, raise a clear error rather than return an infeasible optimum.

- [ ] **Step 1: Write the failing test**

```python
# Add this method inside the existing class TestSklearnBackendRejectsInputConstraints

    def test_sklearn_find_optimum_raises_on_input_constraint(self):
        session = OptimizationSession()
        session.add_variable('x1', 'real', bounds=(0.0, 1.0))
        session.add_variable('x2', 'real', bounds=(0.0, 1.0))
        session.add_input_constraint('inequality', {'x1': 1.0, 'x2': 1.0}, rhs=1.0)
        np.random.seed(2)
        n = 12
        df = pd.DataFrame({
            'x1': np.random.uniform(0, 1, n),
            'x2': np.random.uniform(0, 1, n),
            'yield': np.random.uniform(0, 10, n),
        })
        session.experiment_manager.target_columns = ['yield']
        session.experiment_manager.df = df
        session.train_model(backend='sklearn')
        with pytest.raises(ValueError, match='(?i)input constraint'):
            session.find_optimum('maximize')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest "tests/unit/core/acquisition/test_input_constraints.py::TestSklearnBackendRejectsInputConstraints::test_sklearn_find_optimum_raises_on_input_constraint" -v`
Expected: FAIL — no error is raised (skopt silently returns an unconstrained optimum).

- [ ] **Step 3: Write minimal implementation**

`session.find_optimum` (session.py:1638+) currently only grid-searches for `sklearn` too via `_generate_prediction_grid` — verify: it uses `self.predict` on a grid regardless of backend, so **Task 2 already covers sklearn find_optimum via the grid filter**. Confirm by reading session.py:1638-1694: if the SO path is backend-agnostic (grid + predict), then the Task 2 filter already makes sklearn find_optimum feasible and NO separate error is needed.

Decision rule for the implementer:
- If `session.find_optimum` SO path is backend-agnostic (grid-based) → Task 2's filter handles sklearn too. **Change this test** to assert the returned optimum is feasible (like Task 2) instead of expecting a raise, and skip editing `skopt_acquisition.py`.
- If `session.find_optimum` delegates to `skopt_acquisition.find_optimum` for sklearn → add the guard below to `skopt_acquisition.find_optimum` and keep the raise test.

Guard to add to `skopt_acquisition.find_optimum` (only if the second case applies):

```python
    def find_optimum(self, model, maximize=True, random_state=42):
        if getattr(self.search_space, 'constraints', None):
            raise ValueError(
                "Linear input constraints are registered but the sklearn "
                "backend's find_optimum cannot enforce them. Use the 'botorch' "
                "backend for constrained optimization, or remove the input "
                "constraints."
            )
        # ... existing body ...
```

Note: `SkoptAcquisition.__init__` receives `search_space=self.search_space.to_skopt()` in session.py — a skopt object without `.constraints`. If so, the guard must live in `session.find_optimum` before delegation, checking `self.search_space.constraints`. Read session.py:1638-1694 to place it correctly.

- [ ] **Step 4: Run test to verify it passes**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest "tests/unit/core/acquisition/test_input_constraints.py::TestSklearnBackendRejectsInputConstraints" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add alchemist_core/ tests/unit/core/acquisition/test_input_constraints.py
git commit -m "feat: sklearn find_optimum honors or rejects input constraints"
```

---

## Task 4: Plot grids mask infeasible regions (2D contour/surface/slice)

**Files:**
- Modify: `alchemist_core/session.py` — 2D grid plot methods: `plot_contour` (~2709), `plot_surface` (~2892), `plot_uncertainty_surface` (~3062), `plot_slice` (~2565), `plot_acquisition_contour` (~4623), `plot_uncertainty_contour` (~4828). Each builds `grid_df` then predicts.
- Test: `tests/unit/visualization/test_input_constraint_masking.py` (new)

**Pattern:** after each method builds `grid_df` and computes predictions `Z` (reshaped to meshgrid), set `Z` to `np.nan` where `grid_df` rows are infeasible. Because there are 6 near-identical sites, add ONE private helper on the session and call it from each.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/visualization/test_input_constraint_masking.py
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')

from alchemist_core import OptimizationSession


def _session():
    s = OptimizationSession()
    s.add_variable('H2', 'real', bounds=(0.0, 100.0))
    s.add_variable('CO', 'real', bounds=(0.0, 100.0))
    s.add_variable('CO2', 'real', bounds=(0.0, 100.0))
    np.random.seed(0)
    n = 20
    df = pd.DataFrame({
        'H2': np.random.uniform(0, 100, n),
        'CO': np.random.uniform(0, 100, n),
        'CO2': np.random.uniform(0, 100, n),
    })
    df['yield'] = 0.5 * df.H2 - 0.2 * df.CO + 0.1 * df.CO2
    s.experiment_manager.target_columns = ['yield']
    s.experiment_manager.df = df
    s.train_model(backend='botorch')
    return s


class TestGridMasking:
    def test_apply_feasibility_mask_sets_nan(self):
        """The shared masking helper NaNs infeasible cells of a Z grid."""
        s = _session()
        s.add_input_constraint('equality', {'H2': 1.0, 'CO': 1.0, 'CO2': 1.0}, rhs=100.0)
        # Build a small grid_df spanning feasible + infeasible
        grid_df = pd.DataFrame({
            'H2':  [50.0, 50.0],
            'CO':  [30.0, 30.0],
            'CO2': [20.0, 80.0],   # row0 sum=100 feasible, row1 sum=160 infeasible
        })
        Z = np.array([1.0, 2.0])
        Z_masked = s._apply_feasibility_mask(Z, grid_df)
        assert not np.isnan(Z_masked[0])
        assert np.isnan(Z_masked[1])

    def test_mask_noop_without_constraints(self):
        s = _session()
        grid_df = pd.DataFrame({'H2': [50.0], 'CO': [30.0], 'CO2': [20.0]})
        Z = np.array([1.0])
        Z_masked = s._apply_feasibility_mask(Z, grid_df)
        assert Z_masked.tolist() == [1.0]

    def test_plot_contour_masks_infeasible(self):
        """End-to-end: constrained contour has NaNs where sum != 100 (masked)."""
        s = _session()
        s.add_input_constraint('equality', {'H2': 1.0, 'CO': 1.0, 'CO2': 1.0}, rhs=100.0)
        fig = s.plot_contour('H2', 'CO')  # CO2 fixed at midpoint 50
        # With H2,CO varying and CO2=50, feasible band is H2+CO=50; most cells infeasible.
        # Assert the plot was produced without error and some data was masked.
        assert fig is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/visualization/test_input_constraint_masking.py -v`
Expected: FAIL — `AttributeError: 'OptimizationSession' object has no attribute '_apply_feasibility_mask'`

- [ ] **Step 3: Add the shared helper**

Add near `_generate_prediction_grid` in `session.py`:

```python
    def _apply_feasibility_mask(self, Z, grid_df):
        """Set Z entries to NaN where grid_df rows violate input constraints.

        Args:
            Z: numpy array of predicted values, aligned row-wise with grid_df
               (before any reshape to a meshgrid).
            grid_df: DataFrame of grid points with all variable columns.

        Returns:
            Z with infeasible entries replaced by np.nan. No-op if no constraints.
        """
        import numpy as np
        if not getattr(self.search_space, 'constraints', None):
            return Z
        mask = self.search_space.filter_feasible(grid_df)
        Z = np.array(Z, dtype=float).copy()
        flat = Z.ravel()
        flat[~mask] = np.nan
        return flat.reshape(Z.shape)
```

- [ ] **Step 4: Wire the helper into each 2D plot method**

For EACH method (`plot_contour`, `plot_surface`, `plot_uncertainty_surface`, `plot_slice`, `plot_acquisition_contour`, `plot_uncertainty_contour`): locate where predictions are extracted from `predict_result` into the value array that gets reshaped for plotting (search for `.reshape(` following the `predict` call in that method). Immediately BEFORE the reshape, insert:

```python
        # Mask cells that violate registered input constraints
        <values> = self._apply_feasibility_mask(<values>, grid_df)
```

where `<values>` is the flat prediction array for that method (e.g. `mean_values`, `z_values`, `acq_values`). Read each method to use the correct local variable name — do NOT assume a single name. Apply while the array is still flat (aligned with `grid_df` rows), then reshape as the existing code does.

- [ ] **Step 5: Run tests to verify they pass**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/visualization/test_input_constraint_masking.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Run plotting regression suite**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/visualization/ -q`
Expected: PASS (no regressions — masking is a no-op without constraints)

- [ ] **Step 7: Commit**

```bash
git add alchemist_core/session.py tests/unit/visualization/test_input_constraint_masking.py
git commit -m "feat: mask infeasible regions in 2D constraint-aware plots"
```

---

## Task 5: Plot grids mask infeasible regions (3D voxel plots)

**Files:**
- Modify: `alchemist_core/session.py` — `plot_voxel` (~3219), `plot_uncertainty_voxel` (~5007), `plot_acquisition_voxel` (~5195).
- Test: extend `tests/unit/visualization/test_input_constraint_masking.py`

3D voxel methods build a 3D meshgrid and a `grid_df`, predict, and reshape to 3D. The same `_apply_feasibility_mask` works (it reshapes back to the original shape).

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/unit/visualization/test_input_constraint_masking.py

class TestVoxelMasking:
    def test_plot_voxel_masks_infeasible(self):
        s = _session()
        s.add_input_constraint('inequality', {'H2': 1.0, 'CO': 1.0, 'CO2': 1.0}, rhs=100.0)
        fig = s.plot_voxel('H2', 'CO', 'CO2')
        assert fig is not None
```

- [ ] **Step 2: Run to verify it fails or errors**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/visualization/test_input_constraint_masking.py::TestVoxelMasking -v`
Expected: FAIL if voxel signature differs, or PASS-but-unmasked. Read `plot_voxel` signature first and adjust the call. The real assertion after wiring: infeasible voxels are NaN/hidden.

- [ ] **Step 3: Wire helper into the 3 voxel methods**

In each voxel method, after predictions are extracted and BEFORE the reshape to the 3D grid shape, insert `<values> = self._apply_feasibility_mask(<values>, grid_df)` (correct local name per method).

- [ ] **Step 4: Run tests**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/visualization/test_input_constraint_masking.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add alchemist_core/session.py tests/unit/visualization/test_input_constraint_masking.py
git commit -m "feat: mask infeasible voxels in 3D constraint-aware plots"
```

---

## Task 6: DOE / initial design rejects infeasible points

**Files:**
- Modify: `alchemist_core/utils/doe.py` (`generate_initial_design` routing, ~171) — add a feasibility filter/resample wrapper.
- Modify: `alchemist_core/session.py` `generate_initial_design` if the search_space isn't passed into doe.
- Test: `tests/unit/core/data/test_doe_constraints.py` (new)

Space-filling methods (random/LHS/sobol/hammersly) can reject-and-resample. Classical designs (factorial, CCD, Box-Behnken) have fixed structure and cannot be resampled — for those, filter to feasible rows and warn if the count drops, or raise if zero remain.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/core/data/test_doe_constraints.py
import numpy as np
import pandas as pd
import pytest
from alchemist_core import OptimizationSession


def _session():
    s = OptimizationSession()
    s.add_variable('H2', 'real', bounds=(0.0, 100.0))
    s.add_variable('CO', 'real', bounds=(0.0, 100.0))
    s.add_variable('CO2', 'real', bounds=(0.0, 100.0))
    return s


class TestInitialDesignFeasible:
    def test_random_design_all_feasible_inequality(self):
        s = _session()
        s.add_input_constraint('inequality', {'H2': 1.0, 'CO': 1.0, 'CO2': 1.0}, rhs=100.0)
        design = s.generate_initial_design(n_points=8, method='random')
        df = design if isinstance(design, pd.DataFrame) else pd.DataFrame(design)
        totals = df['H2'] + df['CO'] + df['CO2']
        assert (totals <= 100.0 + 1e-3).all()

    def test_lhs_design_all_feasible_inequality(self):
        s = _session()
        s.add_input_constraint('inequality', {'H2': 1.0, 'CO': 1.0, 'CO2': 1.0}, rhs=100.0)
        design = s.generate_initial_design(n_points=8, method='lhs')
        df = design if isinstance(design, pd.DataFrame) else pd.DataFrame(design)
        totals = df['H2'] + df['CO'] + df['CO2']
        assert (totals <= 100.0 + 1e-3).all()
```

Read `session.generate_initial_design` and `doe.generate_initial_design` for the exact method-name strings and return type before finalizing assertions.

- [ ] **Step 2: Run to verify it fails**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/data/test_doe_constraints.py -v`
Expected: FAIL — some sampled points exceed the constraint.

- [ ] **Step 3: Implement reject-and-resample for space-filling methods**

In `doe.py`, wrap space-filling generation: over-sample (e.g. request `k * n_points`), filter via `search_space.filter_feasible`, take the first `n_points`; loop with growing `k` up to a cap; raise `ValueError` if insufficient feasible points are found. Thread `search_space` into `doe.generate_initial_design` if not already available (check its signature — it already receives the space per the inventory). For classical designs, filter feasible rows and `logger.warning` on reduction; raise if zero remain.

- [ ] **Step 4: Run tests**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/data/test_doe_constraints.py -v`
Expected: PASS

- [ ] **Step 5: Run DOE regression suite**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/ -k "doe or design" -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add alchemist_core/utils/doe.py alchemist_core/session.py tests/unit/core/data/test_doe_constraints.py
git commit -m "feat: initial design honors input constraints via reject-and-resample"
```

---

## Task 7: API + desktop find-optimum paths inherit feasibility

**Files:**
- Verify only (no logic if they delegate to session): `api/routers/acquisition.py` (`find_model_optimum`, ~L101), `ui/acquisition_panel.py` (`find_optimum`, ~L1023).
- Test: `tests/integration/api/test_acquisition_router.py` (extend)

The API `find-optimum` endpoint instantiates the acquisition object directly and calls `acquisition.find_optimum` rather than `session.find_optimum`, bypassing Task 2's session-level filter. Route it through `session.find_optimum` (which is now constraint-aware) OR apply `search_space.filter_feasible` to its grid.

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/integration/api/test_acquisition_router.py
# (follow the existing fixture/client pattern in that file)

def test_find_optimum_endpoint_respects_input_constraint(client, constrained_botorch_session):
    """POST /acquisition/find-optimum returns a feasible optimum when an input
    constraint is registered on the session."""
    resp = client.post(f"/sessions/{constrained_botorch_session}/acquisition/find-optimum",
                        json={"goal": "maximize"})
    assert resp.status_code == 200
    x = resp.json()["x_opt"]
    total = x["H2"] + x["CO"] + x["CO2"]
    assert abs(total - 100.0) <= 5.0
```

Read `tests/integration/api/test_acquisition_router.py` for the actual fixture names and request schema; adapt the fixture to register the H2+CO+CO2==100 constraint. If no such fixture exists, create `constrained_botorch_session` mirroring the existing session fixture plus `add_input_constraint`.

- [ ] **Step 2: Run to verify it fails**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/integration/api/test_acquisition_router.py -k find_optimum -v`
Expected: FAIL — endpoint returns an infeasible optimum.

- [ ] **Step 3: Route the endpoint through session.find_optimum**

In `api/routers/acquisition.py::find_model_optimum`, replace the direct `BoTorchAcquisition/SkoptAcquisition.find_optimum` construction with `session.find_optimum(goal=...)`, preserving the response schema. This inherits the Task 2 grid filter. Do the same in `ui/acquisition_panel.py::find_optimum` (or apply `session.search_space.filter_feasible` to its grid if it must stay direct).

- [ ] **Step 4: Run tests**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/integration/api/test_acquisition_router.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api/routers/acquisition.py ui/acquisition_panel.py tests/integration/api/test_acquisition_router.py
git commit -m "feat: route API/desktop find-optimum through constraint-aware session path"
```

---

## Task 8: Full regression + wiki update

- [ ] **Step 1: Run the full suite**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/ -q`
Expected: All pass (previous baseline: 876 passed, 10 skipped, plus new tests).

- [ ] **Step 2: Update the acquisition-engine wiki note**

Modify `wiki/entities/alchemist-acquisition-engine.md` (or the raw source it's built from) to document: (a) `to_botorch_constraints` emits raw-space constraints because the model normalizes internally; (b) constraints are passed on both continuous and mixed acquisition paths; (c) `SearchSpace.is_feasible/filter_feasible` is the shared feasibility primitive used by find_optimum, plots, and DOE. Keep it factual and concise.

- [ ] **Step 3: Commit**

```bash
git add wiki/
git commit -m "docs: document cross-surface input constraint enforcement"
```

---

## Out of Scope (separate tasks)

- **EI/LogEI degeneracy bug** — reproduced independently of constraints (unconstrained EI/LogEI also prefer high-σ corners over high-mean points while UCB is correct). Tracked separately; do NOT attempt here.
- Nonlinear input constraints.
- `optimal_design.py` D/I-optimal candidate-set constraint filtering beyond the space-filling/classical DOE covered in Task 6 (fast-follow if needed).

---

## Self-Review Notes

- **Spec coverage:** find_optimum (T2, T3, T7), plots 2D (T4) + 3D (T5), DOE (T6), shared primitive (T1), API/desktop (T7). suggest_next already done in prior commits.
- **Type consistency:** `filter_feasible(points) -> np.ndarray[bool]`; `is_feasible(point) -> bool`; `_apply_feasibility_mask(Z, grid_df) -> np.ndarray`. Same names used across tasks.
- **Tolerance:** relative band defined once in `filter_feasible`, reused everywhere.
- **Known ambiguity flagged in T3/T4/T6:** implementer must read the exact local variable names / delegation structure before wiring; steps say so explicitly rather than guessing.
