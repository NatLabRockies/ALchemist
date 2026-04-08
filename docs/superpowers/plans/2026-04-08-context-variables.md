# Context Variables Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `"context"` variable type to ALchemist so observed, non-optimized input columns (e.g. catalyst lot, ambient humidity, pre-computed physical property) are included in the GP feature matrix but excluded from acquisition optimization; at suggestion time the user provides context values explicitly.

**Architecture:** A new `"context"` var_type is added to `SearchSpace.add_variable()`. `ExperimentManager.get_features_and_target()` uses `search_space.get_variable_names()` (tunable-first ordering) to build `train_X` when context vars are registered. At suggestion time, `BoTorchAcquisition.select_next(context_values={...})` wraps the acquisition function with BoTorch's `FixedFeatureAcquisitionFunction`, locking context dims to the user-provided values and optimizing only in the tunable subspace. Session validates that context values are provided and complete before each call to `suggest_next()`.

**Tech Stack:** Python 3.11+, BoTorch (`botorch.acquisition.FixedFeatureAcquisitionFunction`), PyTorch, pandas, pytest.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| **Modify** | `alchemist_core/data/search_space.py` | Accept `"context"` var_type; add `get_tunable_variable_names()`, `get_context_variable_names()`, `has_context_variables()`; reorder `get_variable_names()` to return tunable-first |
| **Modify** | `alchemist_core/data/experiment_manager.py` | Explicit column ordering via search_space when context vars are registered; validation that context columns exist in data |
| **Modify** | `alchemist_core/acquisition/botorch_acquisition.py` | `select_next(context_values=None)` — wraps acq with `FixedFeatureAcquisitionFunction`; uses tunable-only bounds; returns tunable-only result |
| **Modify** | `alchemist_core/session.py` | `suggest_next(context=None)` param; validates context completeness; raises for sklearn backend; passes context to acquisition |
| **Create** | `tests/unit/core/data/test_context_variables.py` | Unit tests for SearchSpace context variable methods |
| **Create** | `tests/unit/core/acquisition/test_context_variable_acquisition.py` | Unit tests for acquisition-layer context var errors |
| **Create** | `tests/integration/workflows/test_context_variable_workflow.py` | End-to-end workflow tests |

---

## Task 1: SearchSpace context variable support

**Files:**
- Modify: `alchemist_core/data/search_space.py`
- Create: `tests/unit/core/data/test_context_variables.py`

- [ ] **Step 1.1: Write the failing tests**

Create `tests/unit/core/data/test_context_variables.py`:

```python
"""Unit tests for SearchSpace context variable support."""
import pytest
from alchemist_core.data.search_space import SearchSpace


def _make_space():
    ss = SearchSpace()
    ss.add_variable("T", "real", min=200.0, max=320.0)
    ss.add_variable("P", "real", min=20.0, max=80.0)
    return ss


def test_add_context_variable_stores_entry():
    ss = _make_space()
    ss.add_variable("humidity", "context")
    assert any(v["name"] == "humidity" and v["type"] == "context" for v in ss.variables)


def test_context_variable_not_in_skopt_dimensions():
    ss = _make_space()
    ss.add_variable("humidity", "context")
    names = [d.name for d in ss.skopt_dimensions]
    assert "humidity" not in names


def test_context_variable_not_in_tunable_names():
    ss = _make_space()
    ss.add_variable("humidity", "context")
    assert "humidity" not in ss.get_tunable_variable_names()


def test_context_variable_in_context_names():
    ss = _make_space()
    ss.add_variable("humidity", "context")
    assert "humidity" in ss.get_context_variable_names()


def test_get_variable_names_returns_tunable_first():
    """Even if context var registered between tunable vars, tunable come first."""
    ss = SearchSpace()
    ss.add_variable("T", "real", min=200.0, max=320.0)
    ss.add_variable("batch", "context")
    ss.add_variable("P", "real", min=20.0, max=80.0)
    names = ss.get_variable_names()
    assert names == ["T", "P", "batch"]


def test_context_variable_not_in_botorch_bounds():
    ss = _make_space()
    ss.add_variable("humidity", "context")
    bounds = ss.to_botorch_bounds()
    assert "humidity" not in bounds


def test_has_context_variables_false_by_default():
    ss = _make_space()
    assert ss.has_context_variables() is False


def test_has_context_variables_true_after_add():
    ss = _make_space()
    ss.add_variable("batch", "context")
    assert ss.has_context_variables() is True


def test_context_variable_name_conflict_with_tunable():
    ss = _make_space()
    with pytest.raises(ValueError, match="already registered"):
        ss.add_variable("T", "context")


def test_from_dict_restores_context_variable():
    ss = SearchSpace()
    ss.from_dict([
        {"name": "T", "type": "real", "min": 200.0, "max": 320.0},
        {"name": "batch", "type": "context"},
    ])
    assert ss.has_context_variables()
    assert "batch" in ss.get_context_variable_names()
    assert "batch" not in [d.name for d in ss.skopt_dimensions]
```

- [ ] **Step 1.2: Run tests to verify they fail**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/Active learning code development/ALchemist/.worktrees/feature-derived-variables"
~/miniforge3/envs/alchemist-env/bin/pytest tests/unit/core/data/test_context_variables.py -v 2>&1 | head -30
```

Expected: `AttributeError` — `get_tunable_variable_names` and `get_context_variable_names` do not exist yet; `add_variable("humidity", "context")` raises `ValueError: Unknown variable type`.

- [ ] **Step 1.3: Update `add_variable()` to accept `"context"` type**

In `alchemist_core/data/search_space.py`, in `add_variable()`, find the `else: raise ValueError` block at line 61–62:

```python
        else:
            raise ValueError(f"Unknown variable type: {var_type}")
```

Replace with:

```python
        elif var_type.lower() == "context":
            # Context variables are observed but not optimized — no skopt dimension needed.
            # Conflicts with existing tunable variables are rejected.
            if name in [v["name"] for v in self.variables if v["type"] != "context"]:
                raise ValueError(
                    f"'{name}' is already registered as a tunable variable. "
                    f"Cannot re-register it as a context variable."
                )
        else:
            raise ValueError(f"Unknown variable type: {var_type}")
```

- [ ] **Step 1.4: Add helper methods to SearchSpace**

After `get_discrete_variables()` (line 212), insert:

```python
    def get_tunable_variable_names(self) -> List[str]:
        """Return names of variables that the acquisition function optimizes (non-context)."""
        return [var["name"] for var in self.variables if var["type"] != "context"]

    def get_context_variable_names(self) -> List[str]:
        """Return names of context (observed, non-optimized) variables."""
        return [var["name"] for var in self.variables if var["type"] == "context"]

    def has_context_variables(self) -> bool:
        """Return True if any context variables are registered."""
        return any(var["type"] == "context" for var in self.variables)
```

- [ ] **Step 1.5: Update `get_variable_names()` to return tunable-first**

Replace the existing `get_variable_names()` at line 198–200:

```python
    def get_variable_names(self) -> List[str]:
        """Get list of all variable names. Tunable variables come first, context variables last."""
        tunable = [var["name"] for var in self.variables if var["type"] != "context"]
        context = [var["name"] for var in self.variables if var["type"] == "context"]
        return tunable + context
```

- [ ] **Step 1.6: Update `from_dict()` to handle `"context"` type**

In `from_dict()`, find the final `elif var_type == "discrete":` block (lines 86–91), and add after it (before `return self`):

```python
            elif var_type == "context":
                self.add_variable(name=var["name"], var_type="context")
```

- [ ] **Step 1.7: Run tests to verify they pass**

```bash
~/miniforge3/envs/alchemist-env/bin/pytest tests/unit/core/data/test_context_variables.py -v
```

Expected: all 10 tests pass.

- [ ] **Step 1.8: Run full data test suite to check for regressions**

```bash
~/miniforge3/envs/alchemist-env/bin/pytest tests/unit/core/data/ -v --tb=short 2>&1 | tail -20
```

Expected: all existing tests pass.

- [ ] **Step 1.9: Commit**

```bash
git add alchemist_core/data/search_space.py tests/unit/core/data/test_context_variables.py
git commit -m "feat: add context variable type to SearchSpace

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: ExperimentManager column ordering for context variables

**Files:**
- Modify: `alchemist_core/data/experiment_manager.py`

- [ ] **Step 2.1: Write failing tests**

Add a new test file `tests/unit/core/data/test_context_var_experiment_manager.py`:

```python
"""Tests for ExperimentManager behavior with context variables."""
import pytest
import pandas as pd
from alchemist_core.data.search_space import SearchSpace
from alchemist_core.data.experiment_manager import ExperimentManager


def _make_em_with_context():
    """ExperimentManager with 2 tunable + 1 context var and 5 experiments."""
    ss = SearchSpace()
    ss.add_variable("T", "real", min=200.0, max=320.0)
    ss.add_variable("P", "real", min=20.0, max=80.0)
    ss.add_variable("batch", "context")

    em = ExperimentManager(search_space=ss, target_columns=["Output"])
    # Intentionally add columns in non-registration order (batch before P in dict)
    for i in range(5):
        em.add_experiment({"T": 250.0 + i, "batch": float(i % 2), "P": 50.0 + i}, 1.0)
    return em, ss


def test_get_features_and_target_orders_tunable_first():
    """Columns come back as [T, P, batch], not [T, batch, P] (DataFrame insert order)."""
    em, ss = _make_em_with_context()
    X, y = em.get_features_and_target()
    assert list(X.columns) == ["T", "P", "batch"]


def test_get_features_and_target_context_column_present():
    em, ss = _make_em_with_context()
    X, y = em.get_features_and_target()
    assert "batch" in X.columns


def test_get_features_and_target_missing_context_column_raises():
    ss = SearchSpace()
    ss.add_variable("T", "real", min=200.0, max=320.0)
    ss.add_variable("batch", "context")

    em = ExperimentManager(search_space=ss, target_columns=["Output"])
    # Add experiments WITHOUT the context column
    em.add_experiment({"T": 250.0}, 1.0)

    with pytest.raises(ValueError, match="batch"):
        em.get_features_and_target()


def test_get_features_target_and_noise_orders_tunable_first():
    em, ss = _make_em_with_context()
    X, y, noise = em.get_features_target_and_noise()
    assert list(X.columns) == ["T", "P", "batch"]


def test_get_features_and_targets_multi_orders_tunable_first():
    ss = SearchSpace()
    ss.add_variable("T", "real", min=200.0, max=320.0)
    ss.add_variable("batch", "context")
    ss.add_variable("P", "real", min=20.0, max=80.0)

    em = ExperimentManager(search_space=ss, target_columns=["Y1", "Y2"])
    for i in range(5):
        em.add_experiment({"T": 250.0 + i, "batch": float(i % 2), "P": 50.0 + i},
                          output_value=None)
        em.df.loc[em.df.index[-1], "Y1"] = float(i)
        em.df.loc[em.df.index[-1], "Y2"] = float(i * 2)

    X, Y, noise = em.get_features_and_targets_multi()
    assert list(X.columns) == ["T", "P", "batch"]
```

- [ ] **Step 2.2: Run tests to verify they fail**

```bash
~/miniforge3/envs/alchemist-env/bin/pytest tests/unit/core/data/test_context_var_experiment_manager.py -v 2>&1 | head -30
```

Expected: the ordering tests fail (columns return in DataFrame order, not tunable-first); missing-column test may pass or fail.

- [ ] **Step 2.3: Update `get_features_and_target()` to use search_space ordering**

In `experiment_manager.py`, replace `get_features_and_target()` (lines 114–147):

```python
    def get_features_and_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get features (X) and target (y) separated.

        When a search_space with context variables is set, columns are ordered
        tunable-first then context-last (matching search_space.get_variable_names()).
        This ordering is required for FixedFeatureAcquisitionFunction index consistency.

        Returns:
            X: Features DataFrame
            y: Target Series

        Raises:
            ValueError: If configured target column is not found in data, or if a
                        context variable column is missing from the data.
        """
        target_col = self.target_columns[0]

        if target_col not in self.df.columns:
            raise ValueError(
                f"DataFrame doesn't contain target column '{target_col}'. "
                f"Available columns: {list(self.df.columns)}"
            )

        if self.variable_columns is not None:
            X = self.df[self.variable_columns]
        elif (self.search_space is not None
              and hasattr(self.search_space, 'has_context_variables')
              and self.search_space.has_context_variables()):
            var_names = self.search_space.get_variable_names()
            missing = [n for n in var_names if n not in self.df.columns]
            if missing:
                ctx_names = self.search_space.get_context_variable_names()
                ctx_missing = [n for n in missing if n in ctx_names]
                if ctx_missing:
                    raise ValueError(
                        f"Context variable column(s) {ctx_missing} not found in experiment "
                        f"data. Add these columns when recording experiments. "
                        f"Available columns: {list(self.df.columns)}"
                    )
                raise ValueError(
                    f"Variable column(s) {missing} not found in experiment data. "
                    f"Available columns: {list(self.df.columns)}"
                )
            X = self.df[var_names]
        else:
            metadata_cols = self.target_columns.copy()
            if 'Noise' in self.df.columns:
                metadata_cols.append('Noise')
            if 'Iteration' in self.df.columns:
                metadata_cols.append('Iteration')
            if 'Reason' in self.df.columns:
                metadata_cols.append('Reason')
            X = self.df.drop(columns=metadata_cols)

        y = self.df[target_col]
        return X, y
```

- [ ] **Step 2.4: Update `get_features_target_and_noise()` with the same ordering logic**

Replace `get_features_target_and_noise()` (lines 149–184):

```python
    def get_features_target_and_noise(self) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """
        Get features (X), target (y), and noise values if available.
        Uses search_space ordering (tunable-first) when context vars are registered.
        """
        target_col = self.target_columns[0]

        if target_col not in self.df.columns:
            raise ValueError(
                f"DataFrame doesn't contain target column '{target_col}'. "
                f"Available columns: {list(self.df.columns)}"
            )

        if self.variable_columns is not None:
            X = self.df[self.variable_columns]
        elif (self.search_space is not None
              and hasattr(self.search_space, 'has_context_variables')
              and self.search_space.has_context_variables()):
            var_names = self.search_space.get_variable_names()
            missing = [n for n in var_names if n not in self.df.columns]
            if missing:
                raise ValueError(
                    f"Variable column(s) {missing} not found in experiment data. "
                    f"Available columns: {list(self.df.columns)}"
                )
            X = self.df[var_names]
        else:
            metadata_cols = self.target_columns.copy()
            if 'Noise' in self.df.columns:
                metadata_cols.append('Noise')
            if 'Iteration' in self.df.columns:
                metadata_cols.append('Iteration')
            if 'Reason' in self.df.columns:
                metadata_cols.append('Reason')
            X = self.df.drop(columns=metadata_cols)

        y = self.df[target_col]
        noise = self.df['Noise'] if 'Noise' in self.df.columns else None
        return X, y, noise
```

- [ ] **Step 2.5: Update `get_features_and_targets_multi()` with the same ordering logic**

Replace `get_features_and_targets_multi()` (lines 186–222):

```python
    def get_features_and_targets_multi(self) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Get features (X), all targets (Y), and optional noise for multi-objective optimization.
        Uses search_space ordering (tunable-first) when context vars are registered.
        """
        missing_cols = [col for col in self.target_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(
                f"Target column(s) {missing_cols} not found in data. "
                f"Available columns: {list(self.df.columns)}"
            )

        if self.variable_columns is not None:
            X = self.df[self.variable_columns]
        elif (self.search_space is not None
              and hasattr(self.search_space, 'has_context_variables')
              and self.search_space.has_context_variables()):
            var_names = self.search_space.get_variable_names()
            missing = [n for n in var_names if n not in self.df.columns]
            if missing:
                raise ValueError(
                    f"Variable column(s) {missing} not found in experiment data. "
                    f"Available columns: {list(self.df.columns)}"
                )
            X = self.df[var_names]
        else:
            metadata_cols = self.target_columns.copy()
            if 'Noise' in self.df.columns:
                metadata_cols.append('Noise')
            if 'Iteration' in self.df.columns:
                metadata_cols.append('Iteration')
            if 'Reason' in self.df.columns:
                metadata_cols.append('Reason')
            X = self.df.drop(columns=metadata_cols)

        Y = self.df[self.target_columns].copy()
        noise = self.df[['Noise']] if 'Noise' in self.df.columns else None
        return X, Y, noise
```

- [ ] **Step 2.6: Run tests to verify they pass**

```bash
~/miniforge3/envs/alchemist-env/bin/pytest tests/unit/core/data/test_context_var_experiment_manager.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 2.7: Run full data test suite to check for regressions**

```bash
~/miniforge3/envs/alchemist-env/bin/pytest tests/unit/core/data/ -v --tb=short 2>&1 | tail -20
```

Expected: all existing tests pass.

- [ ] **Step 2.8: Commit**

```bash
git add alchemist_core/data/experiment_manager.py tests/unit/core/data/test_context_var_experiment_manager.py
git commit -m "feat: use search_space ordering in ExperimentManager for context vars

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: BoTorchAcquisition — FixedFeatureAcquisitionFunction

**Files:**
- Modify: `alchemist_core/acquisition/botorch_acquisition.py`
- Create: `tests/unit/core/acquisition/test_context_variable_acquisition.py`

- [ ] **Step 3.1: Write failing tests**

Create `tests/unit/core/acquisition/test_context_variable_acquisition.py`:

```python
"""Tests for BoTorchAcquisition context variable handling."""
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch
from alchemist_core.data.search_space import SearchSpace
from alchemist_core.acquisition.botorch_acquisition import BoTorchAcquisition


def _make_mock_model(feature_names):
    """Return a mock BoTorchModel with the given original_feature_names."""
    model = MagicMock()
    model.original_feature_names = feature_names
    model.feature_names = feature_names
    model.categorical_encodings = {}
    model.model = MagicMock()
    return model


def _make_search_space_with_context():
    ss = SearchSpace()
    ss.add_variable("x1", "real", min=0.0, max=1.0)
    ss.add_variable("x2", "real", min=0.0, max=1.0)
    ss.add_variable("batch", "context")
    return ss


def test_select_next_raises_if_context_values_missing():
    """select_next() with context vars registered but no context_values raises."""
    ss = _make_search_space_with_context()
    model = _make_mock_model(["x1", "x2", "batch"])

    acq = BoTorchAcquisition(
        model=model,
        search_space=ss,
        acq_func="EI",
        maximize=True,
    )
    acq.acq_function = MagicMock()

    with pytest.raises(ValueError, match="context"):
        acq.select_next(context_values=None)


def test_select_next_raises_if_context_values_incomplete():
    """select_next() raises if context_values dict is missing a registered context var."""
    ss = _make_search_space_with_context()
    model = _make_mock_model(["x1", "x2", "batch"])

    acq = BoTorchAcquisition(
        model=model,
        search_space=ss,
        acq_func="EI",
        maximize=True,
    )
    acq.acq_function = MagicMock()

    with pytest.raises(ValueError, match="Missing context"):
        acq.select_next(context_values={"wrong_key": 1.0})
```

- [ ] **Step 3.2: Run tests to verify they fail**

```bash
~/miniforge3/envs/alchemist-env/bin/pytest tests/unit/core/acquisition/test_context_variable_acquisition.py -v 2>&1 | head -20
```

Expected: `TypeError` — `select_next()` doesn't accept `context_values` yet.

- [ ] **Step 3.3: Update `_get_bounds_from_search_space()` to handle context variable type**

In `botorch_acquisition.py`, in `_get_bounds_from_search_space()`, find the block at lines ~682–697 that handles `categorical`, `discrete`, and numeric bounds. After the `elif var.get('type') == 'discrete':` branch, add:

```python
                    elif var.get('type') == 'context':
                        # Placeholder bounds — these dims are fixed by FixedFeatureAcquisitionFunction
                        # and never seen by the optimizer directly.
                        lower_bounds.append(0.0)
                        upper_bounds.append(1.0)
```

This makes the intent explicit rather than relying on the `else` fallback.

- [ ] **Step 3.4: Add `context_values` parameter and FFAF logic to `select_next()`**

Change the method signature at line 336 from:

```python
    def select_next(self, candidate_points=None):
```

to:

```python
    def select_next(self, candidate_points=None, context_values=None):
```

Then immediately after `bounds_tensor = self._get_bounds_from_search_space()` (line 353), insert:

```python
        # --- Context variable handling ---
        context_var_names = []
        if hasattr(self.search_space_obj, 'get_context_variable_names'):
            context_var_names = self.search_space_obj.get_context_variable_names()

        effective_acq = self.acq_function
        result_feature_names = list(self.model.original_feature_names)

        if context_var_names:
            # Validate context_values is provided and complete
            if context_values is None:
                ctx_list = ", ".join(f"'{n}'" for n in context_var_names)
                raise ValueError(
                    f"Context variables are registered ({ctx_list}). "
                    f"Provide context_values={{name: value, ...}} with a value for each."
                )
            missing = [n for n in context_var_names if n not in context_values]
            if missing:
                raise ValueError(
                    f"Missing context values for: {missing}. "
                    f"Provide a value for every registered context variable."
                )

            from botorch.acquisition import FixedFeatureAcquisitionFunction
            all_var_names = self.search_space_obj.get_variable_names()
            ctx_indices = [all_var_names.index(n) for n in context_var_names]
            ctx_vals = [float(context_values[n]) for n in context_var_names]
            D_total = bounds_tensor.shape[1]
            tunable_mask = [i for i in range(D_total) if i not in ctx_indices]

            effective_acq = FixedFeatureAcquisitionFunction(
                acq_function=self.acq_function,
                d=D_total,
                columns=ctx_indices,
                values=ctx_vals,
            )
            bounds_tensor = bounds_tensor[:, tunable_mask]
            result_feature_names = [
                n for n in self.model.original_feature_names if n not in context_var_names
            ]
```

- [ ] **Step 3.5: Replace all `self.acq_function` with `effective_acq` in the optimization paths**

In `select_next()`, replace every occurrence of `self.acq_function` inside the optimization block with `effective_acq`. There are five occurrences:

**Line ~421** (inside `ma_kwargs` dict):
```python
                    ma_kwargs = dict(
                        acq_function=effective_acq,
                        bounds=bounds_tensor,
                        ...
                    )
```

**Line ~478** (`optimize_acqf_mixed` call):
```python
                        batch_candidates, batch_acq_values = optimize_acqf_mixed(
                            acq_function=effective_acq,
                            bounds=bounds_tensor,
                            ...
                        )
```

**Line ~508** (fallback `optimize_acqf` inside the mixed exception handler):
```python
                    batch_candidates, batch_acq_values = optimize_acqf(
                        acq_function=effective_acq,
                        bounds=bounds_tensor,
                        ...
                    )
```

**Lines ~528–529** (continuous-only path `optim_kwargs`):
```python
                optim_kwargs = dict(
                    acq_function=effective_acq,
                    bounds=bounds_tensor,
                    ...
                )
```

Note: `bounds_tensor` is already reassigned to tunable-only bounds above, so passing it directly is correct.

- [ ] **Step 3.6: Update `feat_to_idx` and result-building to use `result_feature_names`**

There are two `feat_to_idx` assignments in `select_next()`:

**Mixed path (~line 407):**
```python
                feat_to_idx = {name: i for i, name in enumerate(result_feature_names)}
```

**Continuous integer-rounding path (~line 547):**
```python
                    feat_to_idx = {name: i for i, name in enumerate(result_feature_names)}
```

In the **batch result-building block** (~lines 580–611), change:
```python
            feature_names = result_feature_names
```
(replacing the existing `feature_names = self.model.original_feature_names`)

In the **single-point result-building block** (~line 615), change:
```python
        feature_names = result_feature_names
```
(replacing the existing `feature_names = self.model.original_feature_names`)

- [ ] **Step 3.7: Run tests to verify they pass**

```bash
~/miniforge3/envs/alchemist-env/bin/pytest tests/unit/core/acquisition/test_context_variable_acquisition.py -v
```

Expected: both tests pass.

- [ ] **Step 3.8: Run full acquisition test suite to check for regressions**

```bash
~/miniforge3/envs/alchemist-env/bin/pytest tests/unit/core/acquisition/ -v --tb=short 2>&1 | tail -20
```

Expected: all existing acquisition tests pass.

- [ ] **Step 3.9: Commit**

```bash
git add alchemist_core/acquisition/botorch_acquisition.py tests/unit/core/acquisition/test_context_variable_acquisition.py
git commit -m "feat: add FixedFeatureAcquisitionFunction support for context variables

select_next(context_values={...}) fixes context dims and optimizes over
tunable dims only. Result dict contains only tunable variable names.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Session API — `suggest_next(context=...)` and `fit_model()` validation

**Files:**
- Modify: `alchemist_core/session.py`

- [ ] **Step 4.1: Write failing tests**

Add `tests/unit/core/test_context_session.py`:

```python
"""Session-level tests for context variable validation."""
import pytest
import numpy as np
from alchemist_core import OptimizationSession


def _make_session(n=20, seed=42):
    rng = np.random.default_rng(seed)
    session = OptimizationSession()
    session.add_variable("x1", "real", min=0.0, max=1.0)
    session.add_variable("x2", "real", min=0.0, max=1.0)
    session.add_variable("batch", "context")
    for i in range(n):
        x1, x2 = rng.uniform(0, 1), rng.uniform(0, 1)
        batch = float(i % 2)
        y = x1**2 + x2**2 + 0.1 * batch + rng.normal(0, 0.05)
        session.add_experiment({"x1": x1, "x2": x2, "batch": batch}, float(y))
    return session


def test_fit_model_with_context_var_succeeds():
    session = _make_session()
    result = session.fit_model(backend="botorch")
    assert result["success"] is True


def test_fit_model_raises_if_context_column_missing():
    session = OptimizationSession()
    session.add_variable("x1", "real", min=0.0, max=1.0)
    session.add_variable("batch", "context")
    # Add experiments WITHOUT the batch column
    session.add_experiment({"x1": 0.5}, 1.0)
    session.add_experiment({"x1": 0.3}, 0.8)
    session.add_experiment({"x1": 0.7}, 1.2)
    with pytest.raises(ValueError, match="batch"):
        session.fit_model(backend="botorch")


def test_suggest_next_raises_without_context_kwarg():
    session = _make_session()
    session.fit_model(backend="botorch")
    with pytest.raises(ValueError, match="context"):
        session.suggest_next(strategy="EI", goal="maximize")


def test_suggest_next_raises_with_incomplete_context():
    session = _make_session()
    session.fit_model(backend="botorch")
    with pytest.raises(ValueError, match="Missing context"):
        session.suggest_next(strategy="EI", goal="maximize", context={"wrong": 0.0})


def test_suggest_next_with_context_returns_tunable_only():
    session = _make_session()
    session.fit_model(backend="botorch")
    suggestions = session.suggest_next(strategy="EI", goal="maximize", context={"batch": 1.0})
    assert "batch" not in suggestions.columns
    assert "x1" in suggestions.columns
    assert "x2" in suggestions.columns


def test_fit_model_with_context_raises_for_sklearn():
    session = OptimizationSession()
    session.add_variable("x1", "real", min=0.0, max=1.0)
    session.add_variable("batch", "context")
    for i in range(5):
        session.add_experiment({"x1": float(i) / 5, "batch": float(i % 2)}, float(i))
    with pytest.raises(ValueError, match="BoTorch"):
        session.fit_model(backend="sklearn")
```

- [ ] **Step 4.2: Run tests to verify they fail**

```bash
~/miniforge3/envs/alchemist-env/bin/pytest tests/unit/core/test_context_session.py -v 2>&1 | head -30
```

Expected: `test_fit_model_with_context_var_succeeds` may pass, others fail with missing context param or no validation.

- [ ] **Step 4.3: Add `context` parameter to `suggest_next()`**

In `session.py`, find `suggest_next()` signature at line 1316:

```python
    def suggest_next(self, strategy: str = 'EI', goal: Union[str, List[str]] = 'maximize',
                    n_suggestions: int = 1, ref_point: Optional[List[float]] = None,
                    **kwargs) -> pd.DataFrame:
```

Replace with:

```python
    def suggest_next(self, strategy: str = 'EI', goal: Union[str, List[str]] = 'maximize',
                    n_suggestions: int = 1, ref_point: Optional[List[float]] = None,
                    context: Optional[Dict] = None,
                    **kwargs) -> pd.DataFrame:
```

(The `Dict` type is already imported via `from typing import ... Dict ...` in session.py.)

- [ ] **Step 4.4: Add context validation at the start of `suggest_next()`**

After the `if self.model is None:` check (line ~1358), insert:

```python
        # Validate context values for context variables
        if self.search_space.has_context_variables():
            ctx_names = self.search_space.get_context_variable_names()
            if context is None:
                ctx_list = ", ".join(f"'{n}'" for n in ctx_names)
                raise ValueError(
                    f"Context variables are registered ({ctx_list}). "
                    f"Provide context={{name: value, ...}} with a value for each."
                )
            missing = [n for n in ctx_names if n not in context]
            if missing:
                raise ValueError(f"Missing context values for: {missing}")
```

- [ ] **Step 4.5: Add sklearn backend rejection for context variables in `fit_model()`**

In `fit_model()`, find the existing derived variable sklearn rejection block (~lines 1152–1157):

```python
                if self.model_backend != "botorch":
                    raise ValueError(
                        "Derived variables are currently only supported with the BoTorch backend. "
                        ...
                    )
```

Add the context variable check directly after the derived variable block (after its closing brace), before the `DerivedFeatureTransform` construction:

```python
            # Reject sklearn backend if context variables are registered
            if self.search_space.has_context_variables() and self.model_backend != "botorch":
                raise ValueError(
                    "Context variables are only supported with the BoTorch backend. "
                    f"Switch to backend='botorch' or remove context variables before "
                    f"using '{self.model_backend}'."
                )
```

- [ ] **Step 4.6: Pass `context_values` to `acquisition.select_next()`**

In `suggest_next()`, find line 1466:

```python
        next_point = self.acquisition.select_next()
```

Replace with:

```python
        if self.model_backend == 'botorch':
            next_point = self.acquisition.select_next(context_values=context)
        else:
            next_point = self.acquisition.select_next()
```

- [ ] **Step 4.7: Run tests to verify they pass**

```bash
~/miniforge3/envs/alchemist-env/bin/pytest tests/unit/core/test_context_session.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 4.8: Commit**

```bash
git add alchemist_core/session.py tests/unit/core/test_context_session.py
git commit -m "feat: add context parameter to suggest_next() and fit_model() validation

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Integration tests + full regression check

**Files:**
- Create: `tests/integration/workflows/test_context_variable_workflow.py`

- [ ] **Step 5.1: Write integration tests**

Create `tests/integration/workflows/test_context_variable_workflow.py`:

```python
"""
Integration tests for context variable support in OptimizationSession.

Uses a synthetic batch effect (batch_id ∈ {0.0, 1.0}) as the context variable
to avoid requiring any domain-specific external data sources.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from alchemist_core import OptimizationSession


def _make_session_with_context(n=20, seed=42):
    """Session with 2 tunable vars + 1 context var (batch_id) and n experiments."""
    rng = np.random.default_rng(seed)
    session = OptimizationSession()
    session.add_variable("x1", "real", min=0.0, max=1.0)
    session.add_variable("x2", "real", min=0.0, max=1.0)
    session.add_variable("batch_id", "context")
    for i in range(n):
        x1, x2 = rng.uniform(0, 1), rng.uniform(0, 1)
        batch = float(i % 2)
        y = x1**2 + x2**2 + 0.1 * batch + rng.normal(0, 0.05)
        session.add_experiment({"x1": x1, "x2": x2, "batch_id": batch}, float(y))
    return session


class TestContextVariableRegistration:
    def test_context_var_not_in_tunable_vars(self):
        session = _make_session_with_context()
        tunable = session.search_space.get_tunable_variable_names()
        assert "batch_id" not in tunable
        assert "x1" in tunable
        assert "x2" in tunable

    def test_context_var_in_context_names(self):
        session = _make_session_with_context()
        assert "batch_id" in session.search_space.get_context_variable_names()


class TestFitModelWithContextVariable:
    def test_fit_model_succeeds(self):
        session = _make_session_with_context()
        result = session.fit_model(backend="botorch")
        assert result["success"] is True

    def test_gp_train_inputs_include_context_dim(self):
        """GP sees all 3 dims (x1, x2, batch_id) in its training data."""
        session = _make_session_with_context()
        session.fit_model(backend="botorch")
        # original_feature_names includes all 3 features
        assert len(session.model.original_feature_names) == 3
        assert "batch_id" in session.model.original_feature_names

    def test_suggest_next_returns_only_tunable_columns(self):
        session = _make_session_with_context()
        session.fit_model(backend="botorch")
        suggestions = session.suggest_next(
            strategy="EI", goal="maximize", context={"batch_id": 1.0}
        )
        assert "batch_id" not in suggestions.columns
        assert "x1" in suggestions.columns
        assert "x2" in suggestions.columns

    def test_suggest_next_without_context_raises(self):
        session = _make_session_with_context()
        session.fit_model(backend="botorch")
        with pytest.raises(ValueError, match="context"):
            session.suggest_next(strategy="EI", goal="maximize")

    def test_suggest_next_with_incomplete_context_raises(self):
        session = _make_session_with_context()
        session.fit_model(backend="botorch")
        with pytest.raises(ValueError, match="Missing context"):
            session.suggest_next(strategy="EI", goal="maximize", context={})

    def test_sklearn_backend_with_context_var_raises(self):
        session = _make_session_with_context()
        with pytest.raises(ValueError, match="BoTorch"):
            session.fit_model(backend="sklearn")

    def test_missing_context_column_in_data_raises(self):
        session = OptimizationSession()
        session.add_variable("x1", "real", min=0.0, max=1.0)
        session.add_variable("batch_id", "context")
        for _ in range(5):
            session.add_experiment({"x1": 0.5}, 1.0)
        with pytest.raises(ValueError, match="batch_id"):
            session.fit_model(backend="botorch")


class TestContextVariableSerialisation:
    def test_save_session_includes_context_var(self):
        session = _make_session_with_context()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        session.save_session(path)
        with open(path) as f:
            data = json.load(f)
        var_types = {v["name"]: v["type"] for v in data["search_space"]["variables"]}
        assert var_types.get("batch_id") == "context"
        Path(path).unlink()

    def test_load_session_restores_context_var(self):
        session = _make_session_with_context()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        session.save_session(path)
        loaded = OptimizationSession.load_session(path, retrain_on_load=False)
        assert loaded.search_space.has_context_variables()
        assert "batch_id" in loaded.search_space.get_context_variable_names()
        Path(path).unlink()

    def test_load_session_then_fit_and_suggest(self):
        session = _make_session_with_context()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        session.save_session(path)
        loaded = OptimizationSession.load_session(path, retrain_on_load=False)
        result = loaded.fit_model(backend="botorch")
        assert result["success"] is True
        suggestions = loaded.suggest_next(
            strategy="EI", goal="maximize", context={"batch_id": 0.0}
        )
        assert "batch_id" not in suggestions.columns
        Path(path).unlink()
```

- [ ] **Step 5.2: Run integration tests to verify they pass**

```bash
~/miniforge3/envs/alchemist-env/bin/pytest tests/integration/workflows/test_context_variable_workflow.py -v --tb=short
```

Expected: all tests pass.

- [ ] **Step 5.3: Run the full test suite to confirm no regressions**

```bash
~/miniforge3/envs/alchemist-env/bin/pytest tests/ --cov=alchemist_core --cov=api --cov-report=term-missing -x --tb=short 2>&1 | tail -40
```

Expected: all previously-passing tests still pass; new tests add to coverage.

- [ ] **Step 5.4: Commit**

```bash
git add tests/integration/workflows/test_context_variable_workflow.py
git commit -m "feat: add integration tests for context variable workflow

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Self-Review Against Spec

| Spec requirement | Covered by |
|---|---|
| `add_variable(name, "context")` stores with type="context" | Task 1, Step 1.3 |
| Context var absent from `skopt_dimensions` and `to_botorch_bounds()` | Task 1, Step 1.3 |
| `get_tunable_variable_names()` | Task 1, Step 1.4 |
| `get_context_variable_names()` | Task 1, Step 1.4 |
| `has_context_variables()` | Task 1, Step 1.4 |
| `get_variable_names()` returns tunable-first | Task 1, Step 1.5 |
| `from_dict()` restores context vars | Task 1, Step 1.6 |
| `get_features_and_target()` uses search_space ordering | Task 2, Step 2.3 |
| Missing context column in data raises at `get_features_and_target()` | Task 2, Step 2.3 |
| `get_features_target_and_noise()` and multi updated | Task 2, Steps 2.4–2.5 |
| `_get_bounds_from_search_space()` handles context type | Task 3, Step 3.3 |
| `select_next(context_values=None)` signature | Task 3, Step 3.4 |
| `FixedFeatureAcquisitionFunction` wraps acq when context present | Task 3, Step 3.4 |
| Tunable-only bounds passed to optimizer | Task 3, Step 3.4 |
| `ValueError` if `context_values` missing | Task 3, Step 3.4 |
| `ValueError` if `context_values` incomplete | Task 3, Step 3.4 |
| `feat_to_idx` and result-building use tunable names only | Task 3, Step 3.6 |
| `suggest_next(context=None)` signature | Task 4, Step 4.3 |
| Session validates context completeness | Task 4, Step 4.4 |
| sklearn backend + context vars raises | Task 4, Step 4.5 |
| `context_values` passed to `acquisition.select_next()` | Task 4, Step 4.6 |
| `save_session()` serializes context vars | Automatic — stored in `search_space.variables` |
| `_load_session_impl()` restores context vars | Automatic — `from_dict()` handles it |
| Integration: fit, suggest, save/load round-trip | Task 5 |
