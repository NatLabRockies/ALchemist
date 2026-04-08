# Context Variables Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-class `context` variable support to ALchemist so observed, non-optimized inputs (e.g. catalyst lot, ambient humidity) are included in the GP's feature matrix while the acquisition optimizer continues to work in the original base variable space.

**Architecture:** Context variables are registered on `SearchSpace` with `type="context"` and stored in the same `variables` list as tunable vars. `get_variable_names()` guarantees tunable-first ordering so context var column indices are deterministic. At suggestion time, user-supplied context values are fixed into the acquisition function via BoTorch's `FixedFeatureAcquisitionFunction`, which lets the optimizer work in the reduced tunable-only space while the GP sees the full feature vector.

**Tech Stack:** Python 3.11+, BoTorch (`FixedFeatureAcquisitionFunction`), pandas, pytest. Activate `conda activate alchemist-env` before every test command.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| **Modify** | `alchemist_core/data/search_space.py` | Handle `type="context"` in `add_variable()` / `from_dict()`; tunable-first `get_variable_names()`; new `get_tunable_variable_names()` / `get_context_variable_names()` helpers |
| **Modify** | `alchemist_core/data/experiment_manager.py` | Explicit column ordering via `search_space.get_variable_names()` in all three feature-extraction methods |
| **Modify** | `alchemist_core/acquisition/botorch_acquisition.py` | Skip context vars in bounds; wrap acq with `FixedFeatureAcquisitionFunction`; return tunable-only results |
| **Modify** | `alchemist_core/session.py` | Validate context vars in data at fit time; sklearn backend guard; `context` param on `suggest_next()` |
| **Create** | `tests/unit/core/data/test_context_variables.py` | Unit tests for `SearchSpace` context variable methods |
| **Create** | `tests/unit/core/acquisition/test_context_variable_acquisition.py` | Unit tests for `BoTorchAcquisition` context wrapping |
| **Create** | `tests/integration/workflows/test_context_variable_workflow.py` | End-to-end integration tests |

---

## Task 1: `SearchSpace` — context variable type

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


# --- add_variable("context") ---

def test_add_context_variable_stores_entry():
    ss = _make_space()
    ss.add_variable("humidity", "context")
    ctx = [v for v in ss.variables if v["name"] == "humidity"]
    assert len(ctx) == 1
    assert ctx[0] == {"name": "humidity", "type": "context"}


def test_context_variable_not_in_skopt_dimensions():
    ss = _make_space()
    ss.add_variable("humidity", "context")
    names_in_skopt = [d.name for d in ss.skopt_dimensions]
    assert "humidity" not in names_in_skopt
    assert len(ss.skopt_dimensions) == 2  # only T and P


def test_context_variable_not_in_categorical_or_discrete():
    ss = _make_space()
    ss.add_variable("humidity", "context")
    assert "humidity" not in ss.categorical_variables
    assert "humidity" not in ss.discrete_variables


def test_add_context_variable_rejects_duplicate_name():
    ss = _make_space()
    ss.add_variable("humidity", "context")
    with pytest.raises(ValueError, match="already registered"):
        ss.add_variable("humidity", "context")


def test_add_context_variable_rejects_existing_tunable_name():
    ss = _make_space()
    with pytest.raises(ValueError, match="already registered"):
        ss.add_variable("T", "context")


# --- get_variable_names() tunable-first ordering ---

def test_get_variable_names_tunable_first():
    """Tunable vars appear before context vars regardless of registration order."""
    ss = SearchSpace()
    ss.add_variable("humidity", "context")   # context registered first
    ss.add_variable("T", "real", min=200.0, max=320.0)
    ss.add_variable("P", "real", min=20.0, max=80.0)
    names = ss.get_variable_names()
    assert names.index("T") < names.index("humidity")
    assert names.index("P") < names.index("humidity")


def test_get_variable_names_no_context_unchanged():
    """Without context vars, ordering is unchanged (registration order)."""
    ss = _make_space()
    assert ss.get_variable_names() == ["T", "P"]


# --- get_tunable_variable_names() ---

def test_get_tunable_variable_names_excludes_context():
    ss = _make_space()
    ss.add_variable("humidity", "context")
    assert ss.get_tunable_variable_names() == ["T", "P"]


def test_get_tunable_variable_names_all_tunable():
    ss = _make_space()
    assert ss.get_tunable_variable_names() == ["T", "P"]


# --- get_context_variable_names() ---

def test_get_context_variable_names():
    ss = _make_space()
    ss.add_variable("humidity", "context")
    ss.add_variable("lot", "context")
    assert ss.get_context_variable_names() == ["humidity", "lot"]


def test_get_context_variable_names_empty():
    ss = _make_space()
    assert ss.get_context_variable_names() == []


# --- to_botorch_bounds() excludes context ---

def test_to_botorch_bounds_excludes_context():
    ss = _make_space()
    ss.add_variable("humidity", "context")
    bounds = ss.to_botorch_bounds()
    assert "humidity" not in bounds
    assert "T" in bounds
    assert "P" in bounds


# --- from_dict() round-trip ---

def test_from_dict_restores_context_variable():
    ss = _make_space()
    ss.add_variable("humidity", "context")
    data = ss.variables  # list of dicts including context
    ss2 = SearchSpace()
    ss2.from_dict(data)
    assert ss2.get_context_variable_names() == ["humidity"]
    assert "humidity" not in [d.name for d in ss2.skopt_dimensions]


def test_from_dict_context_not_in_skopt():
    ss = SearchSpace()
    ss.from_dict([
        {"name": "T", "type": "real", "min": 200.0, "max": 320.0},
        {"name": "humidity", "type": "context"},
    ])
    assert len(ss.skopt_dimensions) == 1
    assert ss.skopt_dimensions[0].name == "T"
```

- [ ] **Step 1.2: Run tests to verify they fail**

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate alchemist-env
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/Active learning code development/ALchemist"
pytest tests/unit/core/data/test_context_variables.py -v 2>&1 | head -40
```

Expected: `ValueError: Unknown variable type: context` on add_variable tests, plus failures on get_tunable_variable_names / get_context_variable_names.

- [ ] **Step 1.3: Modify `add_variable()` to handle `"context"` type**

In `alchemist_core/data/search_space.py`, find the `else: raise ValueError(f"Unknown variable type: {var_type}")` at the end of `add_variable()` (line 58–59). Replace:

```python
        else:
            raise ValueError(f"Unknown variable type: {var_type}")
```

with:

```python
        elif var_type.lower() == "context":
            # Context variables are observed inputs, not optimized.
            # They live in self.variables but have no skopt dimension or bounds.
            if name in [v["name"] for v in self.variables]:
                raise ValueError(
                    f"Variable '{name}' is already registered."
                )
            # Already appended above via var_dict, so just return here.
            # (We re-check because we need to guard before the append at line 33)
        else:
            raise ValueError(f"Unknown variable type: {var_type}")
```

Wait — `var_dict` is appended to `self.variables` at line 33 unconditionally before the type branching. So the context check needs to happen differently. The full corrected `add_variable()` method:

```python
    def add_variable(self, name: str, var_type: str, **kwargs):
        """
        Add a variable to the search space.

        Args:
            name: Variable name
            var_type: "real", "integer", "categorical", "discrete", or "context"
            **kwargs: Additional parameters:
                - real/integer: min, max
                - categorical: values (list of strings)
                - discrete: allowed_values (list of numbers, at least 2, no duplicates)
                - context: no additional parameters required
        """
        var_type_lower = var_type.lower()

        # Guard: reject duplicate names across all variable types
        if name in [v["name"] for v in self.variables]:
            raise ValueError(
                f"Variable '{name}' is already registered."
            )

        var_dict = {"name": name, "type": var_type_lower}
        var_dict.update(kwargs)
        self.variables.append(var_dict)

        if var_type_lower == "real":
            self.skopt_dimensions.append(Real(kwargs["min"], kwargs["max"], name=name))
        elif var_type_lower == "integer":
            self.skopt_dimensions.append(Integer(kwargs["min"], kwargs["max"], name=name))
        elif var_type_lower == "categorical":
            self.skopt_dimensions.append(Categorical(kwargs["values"], name=name))
            self.categorical_variables.append(name)
        elif var_type_lower == "discrete":
            allowed = kwargs.get("allowed_values")
            if allowed is None or len(allowed) < 2:
                raise ValueError(
                    f"Discrete variable '{name}' requires 'allowed_values' with at least 2 values."
                )
            if len(allowed) != len(set(allowed)):
                raise ValueError(
                    f"Discrete variable '{name}' has duplicate values in 'allowed_values'."
                )
            sorted_vals = sorted(float(v) for v in allowed)
            var_dict["allowed_values"] = sorted_vals
            self.skopt_dimensions.append(Categorical(sorted_vals, name=name))
            self.discrete_variables.append(name)
        elif var_type_lower == "context":
            pass  # No skopt dimension; no bounds; just lives in self.variables
        else:
            raise ValueError(f"Unknown variable type: {var_type}")
```

Replace the entire `add_variable` method (lines 19–59) with the above.

- [ ] **Step 1.4: Modify `get_variable_names()` for tunable-first ordering**

Replace the `get_variable_names()` method (lines 195–197):

```python
    def get_variable_names(self) -> List[str]:
        """Get all variable names, tunable variables first, then context variables."""
        tunable = [v["name"] for v in self.variables if v.get("type") != "context"]
        context = [v["name"] for v in self.variables if v.get("type") == "context"]
        return tunable + context
```

- [ ] **Step 1.5: Add `get_tunable_variable_names()` and `get_context_variable_names()` helpers**

Add immediately after `get_variable_names()` (after the updated line ~198):

```python
    def get_tunable_variable_names(self) -> List[str]:
        """Get names of all non-context (tunable) variables in registration order."""
        return [v["name"] for v in self.variables if v.get("type") != "context"]

    def get_context_variable_names(self) -> List[str]:
        """Get names of all context (observed, non-optimized) variables in registration order."""
        return [v["name"] for v in self.variables if v.get("type") == "context"]
```

- [ ] **Step 1.6: Update `from_dict()` to handle `type="context"`**

In `from_dict()` (line 61), the loop body has branches for real/integer/categorical/discrete but no else — unknown types are silently skipped. Replace the loop body at lines 69–88:

```python
        for var in data:
            var_type = var["type"].lower()
            if var_type in ["real", "integer"]:
                self.add_variable(
                    name=var["name"],
                    var_type=var_type,
                    min=var["min"],
                    max=var["max"]
                )
            elif var_type == "categorical":
                self.add_variable(
                    name=var["name"],
                    var_type=var_type,
                    values=var["values"]
                )
            elif var_type == "discrete":
                self.add_variable(
                    name=var["name"],
                    var_type=var_type,
                    allowed_values=var["allowed_values"]
                )
            elif var_type == "context":
                self.add_variable(name=var["name"], var_type="context")
```

- [ ] **Step 1.7: Run tests to verify they pass**

```bash
pytest tests/unit/core/data/test_context_variables.py -v
```

Expected: all 17 tests pass.

- [ ] **Step 1.8: Run the full data test suite to confirm no regressions**

```bash
pytest tests/unit/core/data/ -v --tb=short
```

Expected: all previously-passing tests pass.

- [ ] **Step 1.9: Commit**

```bash
git add alchemist_core/data/search_space.py tests/unit/core/data/test_context_variables.py
git commit -m "feat: add context variable type to SearchSpace

add_variable() now accepts type='context'. Context vars are stored in
self.variables but excluded from skopt_dimensions, categorical_variables,
discrete_variables, and to_botorch_bounds(). get_variable_names() returns
tunable vars first, then context vars. New helpers:
get_tunable_variable_names() and get_context_variable_names().
from_dict() round-trips context vars without creating skopt dimensions.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 2: `ExperimentManager` — ordered column selection

**Files:**
- Modify: `alchemist_core/data/experiment_manager.py`

The three feature-extraction methods currently drop metadata columns from the raw DataFrame. When a `SearchSpace` with context variables is linked, we must select columns in the deterministic order returned by `search_space.get_variable_names()` (tunable first, then context) so that column indices in `train_X` match what `FixedFeatureAcquisitionFunction` expects.

- [ ] **Step 2.1: Update `get_features_and_target()` to use search space ordering**

In `alchemist_core/data/experiment_manager.py`, replace `get_features_and_target()` (lines 114–147) with:

```python
    def get_features_and_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get features (X) and target (y) separated.

        When a search space is linked, X columns are selected and ordered
        by search_space.get_variable_names() (tunable first, then context).

        Returns:
            X: Features DataFrame
            y: Target Series

        Raises:
            ValueError: If configured target column is not found in data
            ValueError: If a context variable column is missing from the data
        """
        target_col = self.target_columns[0]

        if target_col not in self.df.columns:
            raise ValueError(
                f"DataFrame doesn't contain target column '{target_col}'. "
                f"Available columns: {list(self.df.columns)}"
            )

        if self.variable_columns is not None:
            X = self.df[self.variable_columns]
        elif self.search_space is not None and hasattr(self.search_space, 'get_variable_names'):
            ordered_cols = self.search_space.get_variable_names()
            missing = [c for c in ordered_cols if c not in self.df.columns]
            if missing:
                raise ValueError(
                    f"Variable column(s) {missing} not found in experiment data. "
                    f"Available columns: {list(self.df.columns)}"
                )
            X = self.df[ordered_cols]
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

- [ ] **Step 2.2: Update `get_features_target_and_noise()` with same ordering**

Replace `get_features_target_and_noise()` (lines 149–184) with:

```python
    def get_features_target_and_noise(self) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """
        Get features (X), target (y), and noise values if available.

        When a search space is linked, X columns are selected and ordered
        by search_space.get_variable_names() (tunable first, then context).

        Returns:
            X: Features DataFrame
            y: Target Series
            noise: Noise Series if available, otherwise None

        Raises:
            ValueError: If configured target column is not found in data
            ValueError: If a context variable column is missing from the data
        """
        target_col = self.target_columns[0]

        if target_col not in self.df.columns:
            raise ValueError(
                f"DataFrame doesn't contain target column '{target_col}'. "
                f"Available columns: {list(self.df.columns)}"
            )

        if self.variable_columns is not None:
            X = self.df[self.variable_columns]
        elif self.search_space is not None and hasattr(self.search_space, 'get_variable_names'):
            ordered_cols = self.search_space.get_variable_names()
            missing = [c for c in ordered_cols if c not in self.df.columns]
            if missing:
                raise ValueError(
                    f"Variable column(s) {missing} not found in experiment data. "
                    f"Available columns: {list(self.df.columns)}"
                )
            X = self.df[ordered_cols]
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

- [ ] **Step 2.3: Update `get_features_and_targets_multi()` with same ordering**

Replace the feature-extraction portion of `get_features_and_targets_multi()` (lines 186–222). The only part that changes is the `X` selection block. Replace lines 206–217:

```python
        if self.variable_columns is not None:
            X = self.df[self.variable_columns]
        elif self.search_space is not None and hasattr(self.search_space, 'get_variable_names'):
            ordered_cols = self.search_space.get_variable_names()
            missing = [c for c in ordered_cols if c not in self.df.columns]
            if missing:
                raise ValueError(
                    f"Variable column(s) {missing} not found in experiment data. "
                    f"Available columns: {list(self.df.columns)}"
                )
            X = self.df[ordered_cols]
        else:
            metadata_cols = self.target_columns.copy()
            if 'Noise' in self.df.columns:
                metadata_cols.append('Noise')
            if 'Iteration' in self.df.columns:
                metadata_cols.append('Iteration')
            if 'Reason' in self.df.columns:
                metadata_cols.append('Reason')
            X = self.df.drop(columns=metadata_cols)
```

- [ ] **Step 2.4: Run the full test suite to confirm no regressions**

```bash
pytest tests/ -v --tb=short -q 2>&1 | tail -20
```

Expected: all previously-passing tests pass. (No new tests in this task — ordering changes are transparent for sessions without context vars since `get_variable_names()` returns the same order as before when all vars are tunable.)

- [ ] **Step 2.5: Commit**

```bash
git add alchemist_core/data/experiment_manager.py
git commit -m "feat: use search_space.get_variable_names() ordering in ExperimentManager

get_features_and_target(), get_features_target_and_noise(), and
get_features_and_targets_multi() now select and order X columns using
search_space.get_variable_names() when a search space is linked.
This ensures context variable column indices in train_X are deterministic
and consistent with what FixedFeatureAcquisitionFunction expects.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 3: `BoTorchAcquisition` — fixed-feature context wrapping

**Files:**
- Modify: `alchemist_core/acquisition/botorch_acquisition.py`
- Create: `tests/unit/core/acquisition/test_context_variable_acquisition.py`

- [ ] **Step 3.1: Write the failing tests**

Create `tests/unit/core/acquisition/test_context_variable_acquisition.py`:

```python
"""Unit tests for BoTorchAcquisition context variable support."""
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch


def _make_mock_model(n_tunable=2, n_context=1):
    """Create a minimal mock BoTorchModel for testing."""
    from alchemist_core.models.botorch_model import BoTorchModel
    mock = MagicMock(spec=BoTorchModel)
    mock.is_trained = True
    # feature names: tunable first, then context
    all_names = [f"x{i}" for i in range(n_tunable)] + [f"ctx{i}" for i in range(n_context)]
    mock.original_feature_names = all_names
    mock.feature_names = all_names
    mock.categorical_encodings = {}
    mock.Y_orig = torch.tensor([[0.5], [0.8]], dtype=torch.float64)
    # Mock the internal GP model
    mock_gp = MagicMock()
    mock_gp.train_targets = torch.tensor([0.5, 0.8], dtype=torch.float64)
    mock.model = mock_gp
    return mock


def _make_search_space(n_tunable=2, n_context=1):
    from alchemist_core.data.search_space import SearchSpace
    ss = SearchSpace()
    for i in range(n_tunable):
        ss.add_variable(f"x{i}", "real", min=0.0, max=1.0)
    for i in range(n_context):
        ss.add_variable(f"ctx{i}", "context")
    return ss


def test_get_bounds_excludes_context_vars():
    """_get_bounds_from_search_space() returns (2, n_tunable) tensor, not (2, n_total)."""
    from alchemist_core.acquisition.botorch_acquisition import BoTorchAcquisition

    ss = _make_search_space(n_tunable=2, n_context=1)

    acq = BoTorchAcquisition.__new__(BoTorchAcquisition)
    acq.search_space_obj = ss
    acq.model = _make_mock_model(n_tunable=2, n_context=1)
    acq.acq_func_name = "ei"

    bounds = acq._get_bounds_from_search_space()
    assert bounds.shape == (2, 2), f"Expected (2, 2), got {bounds.shape}"


def test_get_bounds_no_context_unchanged():
    """Without context vars, bounds shape is (2, n_tunable) as before."""
    from alchemist_core.acquisition.botorch_acquisition import BoTorchAcquisition

    from alchemist_core.data.search_space import SearchSpace
    ss = SearchSpace()
    ss.add_variable("T", "real", min=200.0, max=320.0)
    ss.add_variable("P", "real", min=20.0, max=80.0)

    acq = BoTorchAcquisition.__new__(BoTorchAcquisition)
    acq.search_space_obj = ss
    mock_model = MagicMock()
    mock_model.original_feature_names = ["T", "P"]
    mock_model.feature_names = ["T", "P"]
    mock_model.categorical_encodings = {}
    acq.model = mock_model
    acq.acq_func_name = "ei"

    bounds = acq._get_bounds_from_search_space()
    assert bounds.shape == (2, 2)


def test_select_next_raises_without_context_when_context_vars_present():
    """select_next() raises ValueError when context vars are registered but context_values is None."""
    from alchemist_core.acquisition.botorch_acquisition import BoTorchAcquisition

    ss = _make_search_space(n_tunable=2, n_context=1)
    acq = BoTorchAcquisition.__new__(BoTorchAcquisition)
    acq.search_space_obj = ss
    acq.model = _make_mock_model()
    acq.acq_func_name = "ei"
    acq.acq_function = MagicMock()
    acq.batch_size = 1

    with pytest.raises(ValueError, match="context_values"):
        acq.select_next(context_values=None)


def test_select_next_raises_with_missing_context_values():
    """select_next() raises ValueError when context_values is missing a registered context var."""
    from alchemist_core.acquisition.botorch_acquisition import BoTorchAcquisition

    ss = _make_search_space(n_tunable=2, n_context=2)  # ctx0 and ctx1
    acq = BoTorchAcquisition.__new__(BoTorchAcquisition)
    acq.search_space_obj = ss
    acq.model = _make_mock_model(n_tunable=2, n_context=2)
    acq.acq_func_name = "ei"
    acq.acq_function = MagicMock()
    acq.batch_size = 1

    with pytest.raises(ValueError, match="ctx1"):
        acq.select_next(context_values={"ctx0": 0.5})  # missing ctx1


def test_select_next_raises_for_qipv_with_context():
    """select_next() raises ValueError for qIPV when context vars are present."""
    from alchemist_core.acquisition.botorch_acquisition import BoTorchAcquisition

    ss = _make_search_space(n_tunable=2, n_context=1)
    acq = BoTorchAcquisition.__new__(BoTorchAcquisition)
    acq.search_space_obj = ss
    acq.model = _make_mock_model()
    acq.acq_func_name = "qipv"
    acq.acq_function = MagicMock()
    acq.batch_size = 1

    with pytest.raises(ValueError, match="qIPV"):
        acq.select_next(context_values={"ctx0": 0.42})


def test_fixed_feature_acq_function_is_constructed():
    """FixedFeatureAcquisitionFunction is created when context_values provided."""
    from botorch.acquisition import FixedFeatureAcquisitionFunction
    from alchemist_core.acquisition.botorch_acquisition import BoTorchAcquisition

    ss = _make_search_space(n_tunable=2, n_context=1)

    acq = BoTorchAcquisition.__new__(BoTorchAcquisition)
    acq.search_space_obj = ss
    acq.model = _make_mock_model()
    acq.acq_func_name = "ei"
    acq.batch_size = 1

    # Capture FixedFeatureAcquisitionFunction construction
    constructed = []

    original_cls = FixedFeatureAcquisitionFunction

    class CapturingFFAF(original_cls):
        def __init__(self, *args, **kwargs):
            constructed.append(kwargs)
            super().__init__(*args, **kwargs)

    with patch(
        "alchemist_core.acquisition.botorch_acquisition.FixedFeatureAcquisitionFunction",
        CapturingFFAF,
    ):
        # Create a mock acq function that won't crash when called
        mock_acq = MagicMock()
        mock_acq.model = MagicMock()
        acq.acq_function = mock_acq
        try:
            acq.select_next(context_values={"ctx0": 0.42})
        except Exception:
            pass  # optimization will fail with mocks; we only need to check construction

    assert len(constructed) == 1
    assert constructed[0]["d"] == 3  # 2 tunable + 1 context
    assert constructed[0]["columns"] == [2]  # ctx0 is at index 2 (tunable-first)
    assert constructed[0]["values"] == [0.42]
```

- [ ] **Step 3.2: Run tests to verify they fail**

```bash
pytest tests/unit/core/acquisition/test_context_variable_acquisition.py -v 2>&1 | head -40
```

Expected: `test_get_bounds_excludes_context_vars` fails (context var gets default [0,1] bounds), `test_select_next_raises_*` and `test_fixed_feature_acq_function_is_constructed` fail (feature not implemented).

- [ ] **Step 3.3: Update `_get_bounds_from_search_space()` to skip context variables**

In `alchemist_core/acquisition/botorch_acquisition.py`, find `_get_bounds_from_search_space()` at line 642. Replace the entire method:

```python
    def _get_bounds_from_search_space(self):
        """Extract bounds for tunable variables only (context variables are excluded)."""
        # Get context variable names to exclude from optimizer bounds
        context_var_names = set()
        if hasattr(self.search_space_obj, 'get_context_variable_names'):
            context_var_names = set(self.search_space_obj.get_context_variable_names())

        # Try the to_botorch_bounds() dict path first
        if hasattr(self.search_space_obj, 'to_botorch_bounds'):
            bounds_dict = self.search_space_obj.to_botorch_bounds()
            if isinstance(bounds_dict, dict) and bounds_dict:
                # to_botorch_bounds() already excludes context vars (no min/max defined)
                lower_bounds = []
                upper_bounds = []
                # Preserve ordering from search space
                if hasattr(self.search_space_obj, 'get_tunable_variable_names'):
                    tunable_names = self.search_space_obj.get_tunable_variable_names()
                else:
                    tunable_names = [n for n in (self.model.original_feature_names or [])
                                     if n not in context_var_names]
                for name in tunable_names:
                    if name in bounds_dict:
                        lower_bounds.append(float(bounds_dict[name][0]))
                        upper_bounds.append(float(bounds_dict[name][1]))
                if lower_bounds:
                    return torch.tensor(
                        [lower_bounds, upper_bounds], dtype=torch.double
                    )

        # Fallback: build bounds from search space variables and model feature names
        if not hasattr(self.model, 'original_feature_names'):
            raise ValueError("Model doesn't have original_feature_names attribute")

        feature_names = self.model.original_feature_names
        categorical_variables = []
        if hasattr(self.search_space_obj, 'get_categorical_variables'):
            categorical_variables = self.search_space_obj.get_categorical_variables()

        lower_bounds = []
        upper_bounds = []

        if hasattr(self.search_space_obj, 'variables'):
            var_dict = {var['name']: var for var in self.search_space_obj.variables}
            for name in feature_names:
                if name in context_var_names:
                    continue  # context vars have no optimizer bounds
                if name in var_dict:
                    var = var_dict[name]
                    if var.get('type') == 'categorical':
                        if hasattr(self.model, 'categorical_encodings') and name in self.model.categorical_encodings:
                            encodings = self.model.categorical_encodings[name]
                            lower_bounds.append(0.0)
                            upper_bounds.append(float(max(encodings.values())))
                        else:
                            lower_bounds.append(0.0)
                            upper_bounds.append(1.0)
                    elif var.get('type') == 'discrete':
                        allowed = var.get('allowed_values', [0.0, 1.0])
                        lower_bounds.append(float(min(allowed)))
                        upper_bounds.append(float(max(allowed)))
                    elif 'min' in var and 'max' in var:
                        lower_bounds.append(float(var['min']))
                        upper_bounds.append(float(var['max']))
                    elif 'bounds' in var:
                        lower_bounds.append(float(var['bounds'][0]))
                        upper_bounds.append(float(var['bounds'][1]))
                else:
                    lower_bounds.append(0.0)
                    upper_bounds.append(1.0)

        if not lower_bounds or not upper_bounds:
            raise ValueError("Could not extract bounds from search space")

        return torch.tensor([lower_bounds, upper_bounds], dtype=torch.double)
```

- [ ] **Step 3.4: Update `select_next()` signature and add context validation + wrapping**

In `alchemist_core/acquisition/botorch_acquisition.py`, find `def select_next(self, candidate_points=None):` at line 336. Change the signature:

```python
    def select_next(self, candidate_points=None, context_values=None):
```

Then add context validation and FixedFeatureAcquisitionFunction wrapping immediately after the `if self.acq_function is None:` guard block (around line 351, after the guard that raises if acq_function is still None). Add this block before `bounds_tensor = self._get_bounds_from_search_space()`:

```python
        # --- Context variable validation and acquisition wrapping ---
        ctx_names = []
        if hasattr(self.search_space_obj, 'get_context_variable_names'):
            ctx_names = self.search_space_obj.get_context_variable_names()

        if ctx_names:
            # qIPV/qNIPV are not compatible with FixedFeatureAcquisitionFunction
            # because their mc_points live in the full feature space.
            if self.acq_func_name in ('qipv', 'qnipv'):
                raise ValueError(
                    "Context variables are not supported with qIPV/qNIPV acquisition "
                    "functions. Use EI, LogEI, PI, UCB, qEI, or qUCB instead."
                )

            if context_values is None:
                registered = ", ".join(f"'{n}'" for n in ctx_names)
                raise ValueError(
                    f"Context variables are registered ({registered}). "
                    f"Provide context_values={{name: value, ...}} with a value for each."
                )

            missing = [n for n in ctx_names if n not in context_values]
            if missing:
                raise ValueError(
                    f"Missing context_values for: {missing}. "
                    f"Provide a value for every registered context variable."
                )
```

- [ ] **Step 3.5: Build `acq_to_optimize` before the optimization block**

Continuing in `select_next()`, immediately after the validation block and before `bounds_tensor = self._get_bounds_from_search_space()`, add:

```python
        # Build wrapped acquisition function that fixes context columns
        if ctx_names:
            from botorch.acquisition import FixedFeatureAcquisitionFunction
            all_var_names = self.search_space_obj.get_variable_names()  # tunable first
            n_total = len(all_var_names)
            ctx_indices = [all_var_names.index(n) for n in ctx_names]
            ctx_vals = [float(context_values[n]) for n in ctx_names]
            acq_to_optimize = FixedFeatureAcquisitionFunction(
                acq_function=self.acq_function,
                d=n_total,
                columns=ctx_indices,
                values=ctx_vals,
            )
        else:
            acq_to_optimize = self.acq_function
```

- [ ] **Step 3.6: Replace `self.acq_function` with `acq_to_optimize` in all `optimize_acqf*` calls**

There are 4 sites in `select_next()` where `self.acq_function` is passed to an optimizer. Replace each with `acq_to_optimize`:

**Site 1** — `ma_kwargs` dict (mixed path, BoTorch >= 0.17):
```python
                    ma_kwargs = dict(
                        acq_function=acq_to_optimize,
                        bounds=bounds_tensor,
```

**Site 2** — `optimize_acqf_mixed` call (fallback mixed path):
```python
                        batch_candidates, batch_acq_values = optimize_acqf_mixed(
                            acq_function=acq_to_optimize,
                            bounds=bounds_tensor,
```

**Site 3** — last-resort `optimize_acqf` fallback inside the mixed error handler:
```python
                    batch_candidates, batch_acq_values = optimize_acqf(
                        acq_function=acq_to_optimize,
                        bounds=bounds_tensor,
```

**Site 4** — `optim_kwargs` dict (pure continuous path):
```python
                optim_kwargs = dict(
                    acq_function=acq_to_optimize,
                    bounds=bounds_tensor,
```

- [ ] **Step 3.7: Return only tunable variable columns in the result**

In the result-building section at the bottom of `select_next()`, `feature_names` is currently always `self.model.original_feature_names` (which includes context vars). Replace both occurrences of `feature_names = self.model.original_feature_names` with:

```python
            feature_names = (
                self.search_space_obj.get_tunable_variable_names()
                if ctx_names
                else self.model.original_feature_names
            )
```

There are two such assignments:
1. Inside `if self.batch_size > 1:` block (~line 581)
2. In the single-point path (~line 615)

Both must be replaced.

- [ ] **Step 3.8: Run the unit tests**

```bash
pytest tests/unit/core/acquisition/test_context_variable_acquisition.py -v
```

Expected: all 6 tests pass. (The `test_fixed_feature_acq_function_is_constructed` test may require adjustment if the mock setup triggers the validation guard before reaching construction — verify it passes by confirming the FFAF construction kwarg assertions.)

- [ ] **Step 3.9: Run the full acquisition test suite to confirm no regressions**

```bash
pytest tests/unit/core/acquisition/ -v --tb=short
```

Expected: all previously-passing tests pass.

- [ ] **Step 3.10: Commit**

```bash
git add alchemist_core/acquisition/botorch_acquisition.py \
        tests/unit/core/acquisition/test_context_variable_acquisition.py
git commit -m "feat: add FixedFeatureAcquisitionFunction support for context variables

_get_bounds_from_search_space() now skips context variables, returning
(2, n_tunable) bounds. select_next() accepts context_values={name: val},
validates all registered context vars are provided, wraps the acquisition
function with FixedFeatureAcquisitionFunction so the optimizer works in
the tunable subspace, and returns only tunable variable columns.
qIPV/qNIPV raise ValueError when context vars are present (unsupported).

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 4: `Session` — validation and `suggest_next(context=...)` parameter

**Files:**
- Modify: `alchemist_core/session.py`

- [ ] **Step 4.1: Add context variable validation to `fit_model()` / `train_model()`**

In `alchemist_core/session.py`, find `self.model.train(self.experiment_manager)` at line 1093. Just before that line, add:

```python
            # Validate context variables before training
            if self.search_space.get_context_variable_names():
                if self.model_backend != 'botorch':
                    raise ValueError(
                        "Context variables are only supported with the BoTorch backend. "
                        f"Switch to backend='botorch' or remove context variables before "
                        f"using '{self.model_backend}'."
                    )
                ctx_names = self.search_space.get_context_variable_names()
                data_cols = set(self.experiment_manager.df.columns)
                missing = [n for n in ctx_names if n not in data_cols]
                if missing:
                    raise ValueError(
                        f"Context variable(s) {missing} not found in experiment data columns: "
                        f"{sorted(data_cols)}. Add a column for each context variable to your "
                        f"experiment data before calling fit_model()."
                    )
```

- [ ] **Step 4.2: Add `context` parameter to `suggest_next()`**

In `session.py`, find `def suggest_next(self, strategy: str = 'EI', goal: Union[str, List[str]] = 'maximize', n_suggestions: int = 1, ref_point: Optional[List[float]] = None, **kwargs) -> pd.DataFrame:` at line 1234. Add `context: Optional[Dict[str, Any]] = None` before `**kwargs`:

```python
    def suggest_next(self, strategy: str = 'EI', goal: Union[str, List[str]] = 'maximize',
                    n_suggestions: int = 1, ref_point: Optional[List[float]] = None,
                    context: Optional[Dict[str, Any]] = None,
                    **kwargs) -> pd.DataFrame:
```

Also update the docstring Args section (find `**kwargs: Strategy-specific parameters:`) to add:

```
            context: Dict mapping context variable names to their current values.
                Required when any context variables are registered on the search space.
                Extra keys beyond registered context vars are silently ignored.
                Example: context={"catalyst_lot": "B", "humidity": 0.42}
```

- [ ] **Step 4.3: Validate and pass context down to acquisition**

In `suggest_next()`, find the block that validates `self.model is None` and the kwargs check (around lines 1276–1330). After that block (just before `if self.model_backend == 'sklearn':` at line 1332), add context validation:

```python
        # Validate and normalise context values
        validated_context = None
        registered_ctx = self.search_space.get_context_variable_names()
        if registered_ctx:
            if context is None:
                registered_str = ", ".join(f"'{n}'" for n in registered_ctx)
                raise ValueError(
                    f"Context variables are registered ({registered_str}). "
                    f"Provide context={{name: value, ...}} with a value for each."
                )
            missing = [n for n in registered_ctx if n not in context]
            if missing:
                raise ValueError(
                    f"Missing context values for: {missing}"
                )
            # Keep only registered context vars (ignore extras per spec)
            validated_context = {n: context[n] for n in registered_ctx}
```

- [ ] **Step 4.4: Thread `validated_context` into the acquisition call**

Find `next_point = self.acquisition.select_next()` at line 1384. Replace with:

```python
        next_point = self.acquisition.select_next(context_values=validated_context)
```

- [ ] **Step 4.5: Run the full test suite to check no regressions**

```bash
pytest tests/ -v --tb=short -q 2>&1 | tail -30
```

Expected: all previously-passing tests still pass.

- [ ] **Step 4.6: Commit**

```bash
git add alchemist_core/session.py
git commit -m "feat: add context parameter to suggest_next() and validate context vars

fit_model() validates that context variable columns exist in experiment
data and raises ValueError for sklearn backend with context vars.
suggest_next(context={...}) validates all registered context vars are
provided and passes them down to BoTorchAcquisition.select_next().

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 5: Integration tests

**Files:**
- Create: `tests/integration/workflows/test_context_variable_workflow.py`

- [ ] **Step 5.1: Write the integration tests**

Create `tests/integration/workflows/test_context_variable_workflow.py`:

```python
"""
Integration tests for context variable support in OptimizationSession.

Uses a synthetic 2-tunable + 1-context setup. The context variable is a
measured batch quality score (continuous), known before each run but not
controlled by the optimizer.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from alchemist_core import OptimizationSession


def _make_session_with_data(n=20, seed=42):
    """
    Return a session with 2 tunable vars, 1 context var, n experiments.
    y = x1**2 + x2**2 + 0.3 * batch_quality + noise
    """
    rng = np.random.default_rng(seed)
    session = OptimizationSession()
    session.add_variable("x1", "real", min=0.0, max=1.0)
    session.add_variable("x2", "real", min=0.0, max=1.0)
    session.add_variable("batch_quality", "context")
    for _ in range(n):
        x1, x2 = rng.uniform(0, 1), rng.uniform(0, 1)
        bq = rng.uniform(0.5, 1.5)
        y = x1 ** 2 + x2 ** 2 + 0.3 * bq + rng.normal(0, 0.05)
        session.add_experiment({"x1": x1, "x2": x2, "batch_quality": bq}, float(y))
    return session


# ---- Registration ----

def test_context_variable_registered_on_search_space():
    session = _make_session_with_data()
    assert session.search_space.get_context_variable_names() == ["batch_quality"]


def test_context_variable_not_in_tunable_vars():
    session = _make_session_with_data()
    assert "batch_quality" not in session.search_space.get_tunable_variable_names()


# ---- fit_model with context variable ----

def test_fit_model_with_context_variable_succeeds():
    session = _make_session_with_data()
    result = session.fit_model(backend="botorch")
    assert result["success"] is True


def test_gp_train_x_has_three_columns():
    """GP train_inputs should include tunable + context = 3 columns."""
    import torch
    session = _make_session_with_data()
    session.fit_model(backend="botorch")
    train_X = session.model.model.train_inputs[0]
    assert train_X.shape[-1] == 3, f"Expected 3 columns, got {train_X.shape[-1]}"


def test_fit_model_raises_missing_context_column():
    """fit_model() raises if context variable column is absent from data."""
    session = OptimizationSession()
    session.add_variable("x1", "real", min=0.0, max=1.0)
    session.add_variable("batch_quality", "context")
    rng = np.random.default_rng(0)
    for _ in range(15):
        session.add_experiment({"x1": rng.uniform(0, 1)}, float(rng.uniform()))
    with pytest.raises(ValueError, match="batch_quality"):
        session.fit_model(backend="botorch")


def test_fit_model_raises_for_sklearn_with_context_vars():
    """sklearn backend does not support context variables."""
    session = _make_session_with_data()
    with pytest.raises(ValueError, match="BoTorch"):
        session.fit_model(backend="sklearn")


# ---- suggest_next with context ----

def test_suggest_next_returns_only_tunable_columns():
    """Suggestions contain only x1, x2 — not batch_quality."""
    session = _make_session_with_data()
    session.fit_model(backend="botorch")
    suggestions = session.suggest_next(
        n_suggestions=1, strategy="EI", goal="maximize",
        context={"batch_quality": 1.0}
    )
    assert "batch_quality" not in suggestions.columns
    assert "x1" in suggestions.columns
    assert "x2" in suggestions.columns


def test_suggest_next_raises_without_context():
    """suggest_next() raises when context vars are registered but context is not provided."""
    session = _make_session_with_data()
    session.fit_model(backend="botorch")
    with pytest.raises(ValueError, match="batch_quality"):
        session.suggest_next(n_suggestions=1, strategy="EI", goal="maximize")


def test_suggest_next_raises_with_missing_context_key():
    """suggest_next() raises when context dict is missing a registered context var."""
    session = _make_session_with_data()
    session.fit_model(backend="botorch")
    with pytest.raises(ValueError, match="batch_quality"):
        session.suggest_next(
            n_suggestions=1, strategy="EI", goal="maximize",
            context={"wrong_name": 0.5}
        )


def test_suggest_next_extra_context_keys_ignored():
    """Extra keys in context dict beyond registered context vars are silently ignored."""
    session = _make_session_with_data()
    session.fit_model(backend="botorch")
    # Should not raise
    suggestions = session.suggest_next(
        n_suggestions=1, strategy="EI", goal="maximize",
        context={"batch_quality": 1.0, "extra_key": 999.0}
    )
    assert "x1" in suggestions.columns


# ---- save / load round-trip ----

def test_save_session_persists_context_variable():
    """Context variable is saved in search_space.variables with type='context'."""
    session = _make_session_with_data()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    session.save_session(path)
    with open(path) as f:
        data = json.load(f)
    var_types = {v["name"]: v["type"] for v in data["search_space"]["variables"]}
    assert var_types.get("batch_quality") == "context"
    Path(path).unlink()


def test_load_session_restores_context_variable():
    """Loaded session has context variable in search space."""
    session = _make_session_with_data()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    session.save_session(path)

    loaded = OptimizationSession.load_session(path, retrain_on_load=False)
    assert loaded.search_space.get_context_variable_names() == ["batch_quality"]
    assert "batch_quality" not in loaded.search_space.get_tunable_variable_names()
    Path(path).unlink()


def test_load_and_fit_after_save():
    """Round-trip: save → load → fit_model → suggest_next works end-to-end."""
    session = _make_session_with_data()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    session.save_session(path)

    loaded = OptimizationSession.load_session(path, retrain_on_load=False)
    result = loaded.fit_model(backend="botorch")
    assert result["success"] is True

    suggestions = loaded.suggest_next(
        n_suggestions=1, strategy="EI", goal="maximize",
        context={"batch_quality": 1.0}
    )
    assert "x1" in suggestions.columns
    assert "batch_quality" not in suggestions.columns
    Path(path).unlink()
```

- [ ] **Step 5.2: Run integration tests to verify they fail before implementation is complete**

```bash
pytest tests/integration/workflows/test_context_variable_workflow.py -v 2>&1 | head -50
```

Expected: several tests fail if any prior task was not fully completed. If all tasks 1–4 are done, all tests should pass. If you see unexpected failures, address the underlying task first.

- [ ] **Step 5.3: Run the full test suite**

```bash
pytest tests/ --cov=alchemist_core --cov=api --cov-report=term-missing -x --tb=short 2>&1 | tail -40
```

Expected: all previously-passing tests pass and the new tests add coverage.

- [ ] **Step 5.4: Commit**

```bash
git add tests/integration/workflows/test_context_variable_workflow.py
git commit -m "test: add integration tests for context variable workflow

End-to-end tests covering: registration, fit_model (success and error
cases), suggest_next (tunable-only output, missing context errors, extra
key tolerance), and save/load round-trip with context variables.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Self-Review Against Spec

| Spec requirement | Covered by |
|---|---|
| `add_variable(name, "context")` stores `{"name": name, "type": "context"}` | Task 1, Step 1.3 |
| Context vars absent from `skopt_dimensions`, `categorical_variables`, `discrete_variables` | Task 1, Step 1.3 |
| `get_tunable_variable_names()` | Task 1, Step 1.5 |
| `get_context_variable_names()` | Task 1, Step 1.5 |
| `get_variable_names()` returns tunable-first | Task 1, Step 1.4 |
| `to_botorch_bounds()` skips context vars | Task 3, Step 3.3 (via dict path) and Step 1.3 (no bounds in dict) |
| `from_dict()` handles `type="context"` | Task 1, Step 1.6 |
| `get_features_and_target()` orders by `get_variable_names()` | Task 2, Step 2.1 |
| `get_features_target_and_noise()` same | Task 2, Step 2.2 |
| `get_features_and_targets_multi()` same | Task 2, Step 2.3 |
| Context var column validation at `fit_model()` time | Task 4, Step 4.1 |
| sklearn backend guard | Task 4, Step 4.1 |
| `FixedFeatureAcquisitionFunction` wraps acq when context present | Task 3, Steps 3.4–3.6 |
| Bounds tensor covers only tunable dims | Task 3, Step 3.3 |
| All optimizer paths receive wrapped acq transparently | Task 3, Step 3.6 |
| `suggest_next(context={...})` parameter | Task 4, Steps 4.2–4.4 |
| Missing context → `ValueError` with clear message | Task 3 Step 3.4, Task 4 Step 4.3 |
| Extra context keys silently ignored | Task 4, Step 4.3 |
| Result DataFrame: only tunable columns | Task 3, Step 3.7 |
| Serialization: context vars auto-serialized via `variables` | Covered by Task 1 (context type in `variables`); no separate save/load step needed |
| `_load_session_impl()` restores context vars via `add_variable()` | Covered by Task 1 Step 1.3 (`add_variable` handles "context") |
| Duplicate name rejection | Task 1, Step 1.3 |
| Conflict with existing tunable var rejected | Task 1, Step 1.3 |
