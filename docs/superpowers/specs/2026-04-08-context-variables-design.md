# Context Variables Design Spec

**Date:** 2026-04-08
**Feature:** Context (covariate/environmental) variable support
**Branch:** `feature/derived-variables` (ships alongside derived variables)

---

## Goal

Allow users to include observed, non-optimized input columns in the GP's feature matrix. Context variables are measured/known quantities that vary across experiments (e.g. catalyst lot, ambient humidity, pre-computed physical property) but are not controlled by the acquisition optimizer. At suggestion time, the user explicitly provides values for them.

This is distinct from **derived variables**, which are computed on-the-fly from base inputs via a Python callable. Context variables come from the data; no callable is needed.

---

## API Surface

### Registration

```python
session.add_variable("catalyst_lot", "context")
session.add_variable("humidity", "context")
```

`"context"` is a new valid `var_type` in `add_variable()`. No bounds or values are required — the variable has a name and type only.

### Data

Context variable values live in the experiment data as regular columns:

```python
session.add_experiment(
    {"T": 300.0, "P": 50.0, "humidity": 0.42},
    output_value=0.87
)
```

Or loaded from CSV with a `humidity` column alongside the tunable variable columns.

### Fitting

```python
session.fit_model(backend="botorch")
# context vars are automatically included in train_X
```

### Suggestion

```python
suggestions = session.suggest_next(
    strategy="EI",
    goal="maximize",
    context={"catalyst_lot": "B", "humidity": 0.42}
)
# Returns DataFrame with only tunable variable columns (T, P)
```

`context` is required when any context variables are registered.

---

## Architecture

### 1. SearchSpace (`alchemist_core/data/search_space.py`)

- `add_variable(name, "context")` stores `{"name": name, "type": "context"}` in `self.variables`
- Context vars are **not** added to `self.skopt_dimensions`, `categorical_variables`, or `discrete_variables`
- Two new helper methods:
  - `get_tunable_variable_names()` → names of all non-context variables
  - `get_context_variable_names()` → names of context variables only
- `get_variable_names()` returns **tunable vars first** (in registration order among themselves), then **context vars** (in registration order among themselves) — this ordering is guaranteed and used to derive fixed column indices in the acquisition layer
- `to_botorch_bounds()` skips context vars (no bounds defined)
- `from_dict()` handles `type="context"` by skipping skopt dimension creation

### 2. ExperimentManager (`alchemist_core/data/experiment_manager.py`)

`get_data()` selects and orders columns explicitly via `search_space.get_variable_names()` (which guarantees tunable-first ordering) rather than inferring from DataFrame column order. This ensures context var column indices in `train_X` are deterministic and match what `FixedFeatureAcquisitionFunction` expects.

Validation at `fit_model()` time: if a context variable registered on the search space is absent from the experiment data, raise a clear `ValueError` before training begins.

### 3. BoTorchModel (`alchemist_core/models/botorch_model.py`)

No structural changes required. Context vars are already in `train_X` from the data. The ARD kernel's `num_dims` is derived from `train_X.shape[-1]` and naturally includes context dims. `to_botorch_bounds()` already excludes context vars since they have no bounds.

### 4. BoTorchAcquisition (`alchemist_core/acquisition/botorch_acquisition.py`)

`suggest()` accepts a `context_values` dict `{name: value}`. When context vars are present:

1. Look up context var column indices from `search_space.get_variable_names()`
2. Wrap the acquisition function with `FixedFeatureAcquisitionFunction`:
   ```python
   from botorch.acquisition import FixedFeatureAcquisitionFunction

   fixed_acq = FixedFeatureAcquisitionFunction(
       acq_function=acq_func,
       d=total_dims,           # N_tunable + N_context
       columns=[idx_ctx1, ...],
       values=[val1, ...],
   )
   ```
3. Pass `fixed_acq` to `optimize_acqf` with bounds covering only tunable dims

The wrapping happens before the optimizer path is selected, so all paths (standard, batch, mixed/discrete) receive it transparently.

### 5. Session (`alchemist_core/session.py`)

- `suggest_next()` gains a `context` parameter (default `None`)
- If context vars are registered, validates that `context` is provided and contains all registered context var names
- Passes context values down to the acquisition `suggest()` call
- The returned suggestion DataFrame contains only tunable variable columns

---

## Serialization

Context vars live in `search_space.variables` with `type="context"`. The existing `save_session()` / `_load_session_impl()` machinery handles them automatically — no special stubs or callable re-registration needed. The only required change: `from_dict()` must handle `type="context"` without creating a skopt dimension.

---

## Error Handling

| Situation | Error |
|---|---|
| `suggest_next()` with context vars registered but no `context` kwarg | `ValueError`: "Context variables are registered [...]. Provide context={...} with a value for each." |
| `context` dict missing one or more registered context var names | `ValueError`: "Missing context values for: ['humidity']" |
| `context` dict contains extra names not registered as context vars | Silent ignore (lenient) |
| Context var column missing from experiment data at `fit_model()` | `ValueError`: "Context variable 'humidity' not found in experiment data columns: [...]" |
| `add_variable("T", "context")` when "T" is already registered | `ValueError`: "'T' is already registered as a tunable variable" |
| sklearn backend with context vars registered | `ValueError`: "Context variables are only supported with the BoTorch backend." |

---

## Testing

### Unit: `tests/unit/core/data/test_context_variables.py`

- `add_variable("x", "context")` stores entry with `type="context"`
- Context var absent from `get_tunable_variable_names()` and `skopt_dimensions`
- Context var present in `get_context_variable_names()` and `get_variable_names()`
- Context var absent from `to_botorch_bounds()`
- Duplicate name rejected; conflict with existing tunable var rejected
- `from_dict()` round-trip restores context vars without creating skopt dimensions

### Unit: `tests/unit/core/models/test_context_variable_acquisition.py`

- `FixedFeatureAcquisitionFunction` is constructed when context vars present
- Bounds tensor covers only tunable dims
- `suggest()` without `context_values` raises `ValueError`
- `suggest()` with incomplete `context_values` raises `ValueError`

### Integration: `tests/integration/workflows/test_context_variable_workflow.py`

- `fit_model()` succeeds with context var column in experiment data
- `suggest_next(context={...})` returns only tunable variable columns
- `suggest_next()` without `context` raises clear error when context vars registered
- Save/load round-trip: context var restored, session fits and suggests correctly
- Missing context column in data raises error at `fit_model()` time
- sklearn backend with context var raises `ValueError`

---

## Out of Scope

- Categorical context variables (v1: numeric only)
- Default context value inference (e.g. last observed value) — user must always provide explicitly
- Context variable support in the sklearn backend
- Exposing context variables in the Web UI or desktop GUI (API/core only for now)
