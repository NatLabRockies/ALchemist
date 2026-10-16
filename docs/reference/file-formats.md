# File formats

The files ALchemist reads and writes, key by key. The examples were produced by running the code on a synthetic session, with long numbers and lists shortened.

| File | Written by | Read by |
|---|---|---|
| [Variables file](#variables-file) (JSON) | `search_space.save_to_json`; **Save File**; `GET /api/v1/sessions/{session_id}/variables/export` | `load_search_space`; **Load File**; `POST /api/v1/sessions/{session_id}/variables/load` |
| [Session file](#session-file) (JSON) | `save_session`, `export_session_json`; **Save**; `GET /api/v1/sessions/{session_id}/export` and `/download` | `load_session`; **Load Session**; `POST /api/v1/sessions/import` |
| [Experiment CSV](#experiment-csv) | `experiment_manager.save_to_csv` | `load_data`; **Load CSV**; `POST /api/v1/sessions/{session_id}/experiments/upload` |
| [Design CSV](#design-csv) | The **CSV** and **Export CSV** buttons on the design panels | Any tool; add a result column and use `load_data` |
| [Audit-log Markdown](#audit-log-markdown) | `export_audit_markdown`; **Export Log**; `GET /api/v1/sessions/{session_id}/audit/export` | People |

Files written by Python (`save_to_json`, `save_session`) and the session export use two-space indentation and escape characters outside ASCII, so `°C` is stored as `\u00b0C`. The variables export from the web app and REST is compact UTF-8 and keeps `°C` as it is. The loaders read both.

## Variables file

A variables file defines the search space. It comes in two forms, and the loaders read both.

| Form | Top level | Holds |
|---|---|---|
| Bare list | An array | The variables only. |
| Object | An object with `variables` and `constraints` | The variables and the linear input constraints. |

`search_space.save_to_json` always writes the object form. The web app's **Save File** writes the object form when the session has input constraints and the bare list when it has none. `GET .../variables/export` writes the bare list unless you add `?include_constraints=true`.

```json
{
  "variables": [
    {"name": "temperature", "type": "real", "unit": "°C", "description": "Reactor temperature",
     "min": 300.0, "max": 500.0, "changes": "hard"},
    {"name": "batches", "type": "integer", "min": 1, "max": 10},
    {"name": "catalyst", "type": "categorical", "values": ["A", "B", "C"]},
    {"name": "SAR", "type": "discrete", "allowed_values": [80.0, 280.0, 450.0]},
    {"name": "ambient", "type": "real", "min": 15.0, "max": 30.0, "role": "observed"}
  ],
  "constraints": [
    {"type": "inequality", "coefficients": {"temperature": 1.0, "batches": 20.0},
     "rhs": 650.0, "name": "heat_budget"}
  ]
}
```

### Variables

Every variable has `name` and `type`. The other keys depend on the type, and a key a type does not use is ignored.

| Key | Used by | Content |
|---|---|---|
| `type` | all | `real`, `integer`, `discrete` or `categorical`. The old type `context` is read as an observed `real`. |
| `min`, `max` | `real`, `integer` | The bounds. Required, except on an observed variable, which may leave them out. |
| `allowed_values` | `discrete` | At least two numbers, no duplicates. |
| `values` | `categorical` | The category names. The loaders also accept `categories`, the name the REST request uses. |
| `unit`, `description` | all | Labels. They appear in the app and in saved files, and no model uses them. |
| `role` | all | `observed` for a variable you measure and do not set. Left out for the default, `controllable`. |
| `changes` | controllable | How hard the variable is to change between runs: `easy` (the default and left out), `hard` or `very_hard`. It drives [split-plot designs](../guide/more/split-plot.md). An observed variable cannot have it. |

Names are unique. `WholePlot`, `Subplot` and `RunOrder` are reserved for split-plot columns and cannot be variable names. The [variables page](../guide/variables.md) explains what each type means.

### Constraints

Each entry of `constraints` is a linear relation over numeric variables, described in [Constraints](../guide/constraints.md).

| Key | Content |
|---|---|
| `type` | `inequality` (`sum(coefficient * variable) <= rhs`) or `equality`. |
| `coefficients` | An object from variable name to a finite number. |
| `rhs` | A finite number. |
| `name` | Optional. A missing name becomes `constraint_0`, `constraint_1` and so on. |

### Loading

`load_search_space` builds a new search space from the file. Whatever was registered before, derived variables included, is gone, and a bare list leaves the new search space with no constraints. With the object form the constraints come from the file, after the shape of every constraint is checked, so a file with a bad entry is refused whole. The file may list a constraint that names a variable the file does not define, because a session can reach that state by deleting a variable (A14), and the loader keeps it as written.

!!! warning "Known issue"
    A28: `load_search_space` replaces the session's search space but not the one its results table points to. Results added afterwards with `add_experiment` cannot be trained on: `train_model` fails with "Found array with 0 feature(s)". Load results with `load_data`, which builds its table from the new search space, or call `session.experiment_manager.set_search_space(session.search_space)` right after `load_search_space`.

The web app and `POST .../variables/load` treat the two forms differently. The object form replaces the whole search space and is atomic. A bare list adds its variables to the existing ones, one at a time.

!!! warning "Known issue"
    W36: with a bare list, a name that already exists stops the load with an error, and the variables listed before it stay added. Load into an empty session, or use the object form.

Derived variables are not in this file. They are in the session file, without the function that computes them.

## Session file

`save_session` writes one JSON file that holds everything a session needs to be opened again, and the web app's **Save** and the REST export write the same content. The top-level keys are these.

| Key | Holds |
|---|---|
| `version` | The format version, `"1.1.0"`. The loader reads any `1.x` file, warns about others, and treats a missing version as `1.0.0`. |
| `metadata` | `session_id`, `name`, `created_at`, `last_modified`, `description`, `author` and `tags`. |
| `audit_log` | `entries`, the [audit entries](../python/events-and-audit.md#the-audit-log), and two snapshots: `search_space` (the variables as they were at the first `lock_data`, `lock_model` or `lock_acquisition` call on the session; designs and the REST suggest and train routes write entries without setting it, so it can be missing) and `experiment_data` (the results table, refreshed on every save). |
| `search_space` | `variables` as in the variables file, `derived_variables` (name, `input_cols` and description of each, with no function), and `constraints`. |
| `experiments` | `data`, one object per row; `n_total`; and `target_columns`, the names of the target columns. |
| `staged_experiments` | Every item of the [work queue](../automation/queue.md), whatever its status. The key keeps its name from before the queue existed. |
| `objective_metadata` | A `label` and `unit` for each target column, kept as the text you gave. |
| `control` | The [control record](../guide/more/monitoring.md): `requested`, `requested_at`, `requested_by`, `reported`, `reported_at`, `reported_by` and `detail`. |
| `last_suggestions` | The rows of the most recent suggestion, as the web app shows them. |
| `provenance` | The [provenance records](../automation/provenance.md), in the order the results were recorded. |
| `config` | Session settings: `random_state`, `verbose`, `auto_train`, `auto_train_threshold`, and `initial_design_method` and `initial_design_n_points` once a design has been generated. |
| `model_config` | Only when a model has been trained, or a training call is on record: `backend`, `kernel`, `kernel_params`, `hyperparameters`, `metrics` and `training_config`. `fitted` is `false` when the block keeps the call but no model is currently fitted. |

The first keys, trimmed:

```json
{
  "version": "1.1.0",
  "metadata": {
    "session_id": "763f1e56-f924-4f39-a76a-619156f1c3e3",
    "name": "Pd screening",
    "created_at": "2026-10-14T18:46:13.855281",
    "last_modified": "2026-10-14T18:46:15.023835",
    "description": "Synthetic example",
    "author": "",
    "tags": ["demo"]
  },
  "search_space": {
    "variables": [
      {"name": "temperature", "type": "real", "unit": "°C", "min": 200.0, "max": 300.0},
      {"name": "pressure", "type": "real", "unit": "bar", "min": 20.0, "max": 80.0}
    ],
    "derived_variables": [],
    "constraints": [
      {"type": "inequality", "coefficients": {"temperature": 1.0, "pressure": 2.0},
       "rhs": 380.0, "name": "budget"}
    ]
  }
}
```

The results, with two of six rows shown. Each row has the variables, the target, and the bookkeeping columns described under [Experiment CSV](#experiment-csv).

```json
{
  "experiments": {
    "data": [
      {"temperature": 210.0, "pressure": 30.0, "Output": -0.95, "Noise": 0.0004,
       "Iteration": 0, "Reason": "Initial Design", "ProvenanceId": NaN},
      {"temperature": 246.2, "pressure": 41.4, "Output": -0.12, "Noise": 0.0004,
       "Iteration": 1, "Reason": "EI", "ProvenanceId": "c41ca4c8-ad99-4332-a2ef-4f1e73d653b0"}
    ],
    "n_total": 6,
    "target_columns": ["Output"]
  }
}
```

!!! warning "Known issue"
    W13: a row with no value in a column, such as a result with no `ProvenanceId` or no `Noise`, is written as a bare `NaN`, which is not valid JSON. Python's `json` module reads it. `JSON.parse` and other strict parsers do not, so replace `NaN` with `null` before you hand the file to one.

Two audit entries. Each entry's `hash` is described on the [audit log page](../python/events-and-audit.md#the-audit-log).

```json
{
  "audit_log": {
    "entries": [
      {
        "timestamp": "2026-10-14T18:46:13.885817",
        "entry_type": "data_locked",
        "parameters": {"n_experiments": 5, "data_hash": "849f458ca8d3b588",
                       "initial_design_method": "lhs", "initial_design_n_points": 5},
        "hash": "b90a7310a57c48cd",
        "notes": "Initial design (lhs)"
      },
      {
        "timestamp": "2026-10-14T18:46:15.023808",
        "entry_type": "acquisition_locked",
        "parameters": {"strategy": "EI", "parameters": {"goal": "maximize"},
                       "suggestions": [{"temperature": 246.1756, "pressure": 41.388}], "iteration": 1},
        "hash": "db25c3d72e3d0873",
        "notes": "Round 1"
      }
    ],
    "search_space": {"variables": "..."},
    "experiment_data": "..."
  }
}
```

One queue item, which is what each element of `staged_experiments` looks like. The fields are explained under [Staging and the work queue](../automation/queue.md#item-states).

```json
{
  "staged_experiments": [
    {
      "id": "c41ca4c8-ad99-4332-a2ef-4f1e73d653b0",
      "inputs": {"temperature": 246.1756, "pressure": 41.388},
      "reason": "EI",
      "status": "done",
      "output": -0.12,
      "noise": 0.0004,
      "error": null,
      "dataset_ref": 5,
      "actual_inputs": {"temperature": 246.2, "pressure": 41.4},
      "staged_at": "2026-10-14T18:46:15.023889",
      "started_at": null,
      "completed_at": "2026-10-14T18:46:15.024525",
      "plot": null
    }
  ]
}
```

The small sections:

```json
{
  "objective_metadata": {"Output": {"label": "Yield", "unit": "%"}},
  "control": {
    "requested": "pause", "requested_at": "2026-10-14T18:46:15.024541", "requested_by": "analyst",
    "reported": "idle", "reported_at": null, "reported_by": null, "detail": null
  },
  "last_suggestions": [{"temperature": 246.1756, "pressure": 41.388}],
  "config": {
    "random_state": 42, "verbose": true, "auto_train": false, "auto_train_threshold": 5,
    "initial_design_method": "lhs", "initial_design_n_points": 5
  }
}
```

### The model

`model_config` records which model was trained and how, not the trained model. `backend`, `kernel` and `kernel_params` describe it, and `hyperparameters` and `metrics` are what the fit gave when the file was saved, kept for reading. `training_config` is the one the loader uses: it holds the arguments of the last successful `train_model` call exactly as passed, including transforms and calibration when you set them, and `load_session` calls `train_model` with them.

```json
{
  "model_config": {
    "backend": "botorch",
    "kernel": "Matern",
    "kernel_params": {},
    "hyperparameters": {"model_type": "SingleTaskGP", "covar_module_type": "ScaleKernel",
                        "lengthscale": "[0.2968, 4321.2102]", "outputscale": 1.08},
    "metrics": {"rmse": 0.147, "mae": 0.0931, "r2": 0.7039},
    "training_config": {"backend": "botorch", "kernel": "Matern", "kernel_params": null}
  }
}
```

A session that has a recorded training call but no fitted model, such as one whose retrain failed on the last load, still writes `model_config`, with `"fitted": false`, empty `hyperparameters` and `metrics`, and the same `training_config`, so the next load can replay the call. The key is absent from files saved with a fitted model, and a missing `fitted` means true.

A `training_config` that leaves out the transforms means the backend's defaults: BoTorch then normalizes the inputs and standardizes the outputs, as it did when the model was first trained. Files saved before `training_config` existed carry only `backend`, `kernel` and `kernel_params`, and the loader uses those.

### What loading does

- Variables, constraints and rows are added in order, and a row that fails is skipped with a warning that names it.
- A missing key is normal in an older file. A file with no `constraints` key restores none, and one with no `target_columns` infers the target as the column that is neither a variable nor a bookkeeping column, falling back to `Output`.
- The model is trained again from `training_config` unless you ask otherwise with `retrain_on_load=False` (see [Recipes](../python/recipes.md#save-and-load-a-session)). A failed retrain is logged and the session loads without a model.
- Over REST, `POST /api/v1/sessions/import` and `POST /api/v1/sessions/upload` rebuild the model the same way, by repeating the session's last training call. If that retrain fails, the session loads without a model. An import gets a new ID, and the imported session's `metadata.session_id` is set to the new ID.
- The server does the same when it reads one of its own session files the first time that session's ID is used.
- The server's own copies of its sessions use this format with one more key, `_session_store_metadata`, holding `created_at` and `last_accessed`. Files you download do not have it.

### What the file does not hold

- The fitted model. Only its training arguments are saved, so a loaded model is a new fit on the same data.
- Random seeds you did not set. The `random_state` in `config` is the session's own default; a design's `random_seed` is not recorded.
- The functions behind derived variables. Re-register each with `register_derived_variable` after loading.
- Outcome constraints added with `add_outcome_constraint`. They are lost on save and load, so add them again (A29).

## Experiment CSV

The results table is a CSV with one row per experiment. `experiment_manager.save_to_csv` writes it, and the example below is three of the six rows of the table from the session above.

```csv
temperature,pressure,Output,Noise,Iteration,Reason,ProvenanceId
210.0,30.0,-0.95,0.0004,0,Initial Design,
240.0,45.0,-0.18,0.0004,0,Initial Design,
246.2,41.4,-0.12,0.0004,1,EI,c41ca4c8-ad99-4332-a2ef-4f1e73d653b0
```

| Column | Content |
|---|---|
| One per variable, named as the variable | The input value. A categorical value is its text. A file with no column for a variable cannot train a model. |
| Target column | The measured result. `Output` by default, or any name you pass as `target_columns`. A multi-objective table has one column per objective. |
| `Noise` | The variance of that measurement, optional. Give it on every row or on none. The models use it as a variance, not a standard deviation (T19). A column that is not numeric is replaced by 1e-10 on every row, with a printed warning. |
| `Iteration` | A whole number: the round of the campaign. `0` is the initial design. A missing column is filled with `0`. When you add a result in Python without one, it takes the highest existing number plus one. |
| `Reason` | Why the row was run, such as `Initial Design` or `EI`. A missing column is filled with `Initial Design`, and `add_experiment` with no reason writes `Manual`. |
| `ProvenanceId` | The ID of the queue item that produced the row, which finds its [provenance record](../automation/provenance.md). Empty for rows that were added directly. |
| `WholePlot`, `Subplot`, `RunOrder` | Whole numbers that place a split-plot run. They are never model inputs. |
| Any other column | Kept in the table and ignored by the model. |

!!! warning "Known issue"
    T19: the REST descriptions call `Noise` "measurement uncertainty", which suggests a standard deviation. The code gives the values to the model as variances: BoTorch as `train_Yvar` and scikit-learn as `alpha`. Square a standard deviation before you enter it.

### Choosing the target columns

`load_data` and the upload route do not guess: they default to `Output` unless you name the target. (Loading a session file is different: it infers the target, as described above.)

| Where | How |
|---|---|
| Python | `load_data(path, target_columns='Output', noise_column=None)`. Pass a string for one target or a list for several. A name the file lacks raises a `ValueError` that lists the columns it has. `noise_column` names a column to read as `Noise`. |
| REST | `POST /api/v1/sessions/{session_id}/experiments/upload?target_columns=yield`, with the file as multipart form data. Separate several names with commas. The default is `Output`. |
| Web app | **Load CSV** previews the columns first. A file with an `Output` column loads at once with that target. Otherwise a dialog lists the columns and asks which are targets. |

With several targets, the first one listed is the primary objective. On a multi-objective session `add_experiment` takes its `output` as a dict `{objective: value}` or a sequence in target order, one finite value per objective, and refuses a bare float with a message naming the missing objectives, so no row is added with an objective empty. **Add Point...** in the web app and the REST `output` field still take one value, so on a multi-objective session they are refused; use `add_experiment` from Python or load a CSV.

Rows are read as written. A CSV from `save_to_csv` loads back with `load_data(path, target_columns=...)` and the same bookkeeping columns.

## Design CSV

The design panels in the web app save their points as a CSV, so you can run them in another tool. **CSV** on the initial-design and optimal-design panels opens a save dialog where the browser has one, and downloads the file elsewhere; **Export CSV** on the split-plot panel always downloads.

| File name | Panel |
|---|---|
| `initial_design_<method>_<N>pts.csv` | Initial design, for example `initial_design_lhs_10pts.csv`. |
| `optimal_design_<criterion>_<N>pts.csv` | Optimal design, for example `optimal_design_D_12pts.csv`. |
| `split_plot_design_<N>runs.csv` | Split-plot design. |

The header is the names of the keys of the first point, and every row follows it. A design of controllable variables has one column per variable.

```csv
temperature,pressure
249.6659,39.1523
228.932,44.2638
```

A split-plot design adds the columns that describe its structure, between the variables, so address the columns by name and not by position.

```csv
furnace_temp,WholePlot,flow,Reason,RunOrder
600.0,1,3.0832569770184954,Initial DoE (split-plot),1
600.0,1,4.976122190878276,Initial DoE (split-plot),2
900.0,2,1.1267486974116248,Initial DoE (split-plot),4
```

A design with three levels also has `Subplot`. Numbers are written at full precision, and values are joined with commas without quoting, so a categorical value that contains a comma would break its row.

A design CSV has no result column. Add one, named `Output` or anything you pass as `target_columns`, and load it with `load_data`: the split-plot columns and `Reason` are kept, and `Iteration` is filled with `0`.

## Audit-log Markdown

The audit log exports to a Markdown document for a methods section. The same session as above gives:

```markdown
# Optimization Audit Trail

## Session Metadata

- **Name**: Pd screening
- **Created At**: 2026-10-14T18:46:13.855281
- **Last Modified**: 2026-10-14T18:46:15.025816
- **Description**: Synthetic example
- **Tags**: demo

## Search Space Definition

- **temperature** (real): [200.0, 300.0]
- **pressure** (real): [20.0, 80.0]

## Experimental Data

| Iteration | Reason | temperature | pressure | Output |
|---|---|---|---|---|
| 0 | Initial Design | 210.0000 | 30.0000 | -0.9500 |
| 1 | EI | 246.2000 | 41.4000 | -0.1200 |

## Optimization Iterations

### Iteration 0

**Initial Design**: lhs (5 points)

**Notes**: Initial design (lhs)

### Iteration 1

**Timestamp**: 2026-10-14T18:46:14.575165

**Model**: botorch, Matern kernel

**Metrics**: R²=0.7039, RMSE=0.1470

**Input Scaling**: normalize
**Output Scaling**: standardize

**Notes**: Default Matern kernel

**Acquisition**: EI

**Parameters**: goal=maximize

**Suggested Next**: [{'temperature': 246.1756, 'pressure': 41.388}]

**Notes**: Round 1
```

The data table is cut to two of six rows here, and the suggestion's numbers are shortened. The table leaves out the `Noise` and `ProvenanceId` columns and prints numbers to four decimals. Each iteration block comes from the entries that share an iteration number, with the initial design as iteration 0. [Events and audit log](../python/events-and-audit.md#export-the-audit-log-as-markdown) lists which entry types reach the Markdown, and [Sessions](../guide/more/sessions.md#3-export-the-audit-log) shows the export in the web app.

!!! warning "Known issue"
    W23: a discrete variable appears in the search-space part as `(categorical): {}`, with no values. Also, every entry whose `parameters` carry an `iteration` opens an iteration group, `config_changed` (a running count) and `experiment_completed` included, so the document can show empty "Iteration N" headings.
