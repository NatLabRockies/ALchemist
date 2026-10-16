# Recipes

These are short, runnable answers to jobs that come up when you script a campaign with `OptimizationSession`. The blocks share one namespace and run top to bottom, so each recipe builds on the names the earlier ones made.

## Load results from a CSV

Read a results file into a session and name its target column.

Define the variables first. `load_data` matches the file's columns to them by name, takes `target_columns` for the result (default `Output`), and takes `noise_column` for a column of measurement noise, which it renames to `Noise`. A column called `Noise` is read without `noise_column`. Other columns stay in the table and the model ignores them. A file without the named target is refused with the list of columns it has. The columns ALchemist reads and writes are listed in [File formats](../reference/file-formats.md#experiment-csv).

```python
import numpy as np
import pandas as pd
from alchemist_core import OptimizationSession

rng = np.random.default_rng(0)
runs = pd.DataFrame({
    "temperature": rng.uniform(200, 300, 12).round(1),
    "pressure": rng.uniform(20, 80, 12).round(1),
})
runs["yield"] = (
    -((runs["temperature"] - 250) / 50) ** 2 - ((runs["pressure"] - 50) / 30) ** 2
    + rng.normal(0, 0.02, 12)
).round(4)
runs["yield_variance"] = 0.0004
runs.to_csv("runs.csv", index=False)

session = OptimizationSession()
session.add_variable("temperature", "real", bounds=(200.0, 300.0), unit="°C")
session.add_variable("pressure", "real", bounds=(20.0, 80.0), unit="bar")
session.load_data("runs.csv", target_columns="yield", noise_column="yield_variance")
print(session.experiment_manager.get_data().columns.tolist())
```

The models use noise as a variance, the square of a standard deviation, and need it on every row or on none. A file with noise on only some rows loads, and `train_model` then refuses it and names the rows.

!!! warning "Known issue"
    T19: the REST descriptions call the noise column "measurement uncertainty", which reads as a standard deviation. The code passes the values to the model as variances. Square a standard deviation before you load it.

## Train a model and ask for the next point

Fit a model to the loaded rows and get a suggestion.

```python
session.train_model(backend="botorch")
suggestion = session.suggest_next(strategy="EI", goal="maximize")
print(suggestion.round(1).to_dict("records"))
```

`suggest_next` returns a one-row DataFrame. The [Train](../guide/train.md) and [Suggest](../guide/suggest.md) pages list the backends, kernels and strategies.

## Load several targets

Give `target_columns` a list and the session becomes multi-objective.

```python
multi_runs = pd.DataFrame({
    "x1": rng.uniform(0, 1, 10).round(3),
    "x2": rng.uniform(0, 1, 10).round(3),
})
multi_runs["yield"] = (multi_runs["x1"] + rng.normal(0, 0.05, 10)).round(3)
multi_runs["selectivity"] = (1 - multi_runs["x1"] + 0.2 * multi_runs["x2"]).round(3)
multi_runs.to_csv("multi_runs.csv", index=False)

multi = OptimizationSession()
multi.add_variable("x1", "real", bounds=(0.0, 1.0))
multi.add_variable("x2", "real", bounds=(0.0, 1.0))
multi.load_data("multi_runs.csv", target_columns=["yield", "selectivity"])
print(multi.is_multi_objective, multi.experiment_manager.target_columns)
```

Multi-objective models need the BoTorch backend; see [Multi-objective](../guide/more/multi-objective.md).

On a multi-objective session, give `add_experiment` every objective: `session.add_experiment({"x": 0.3}, {"conversion": 42.0, "selectivity": 88.0})`, or a list in target order. A bare float is refused with the missing objectives named, so a row is never recorded with an objective empty.

## Add a constraint

Keep designs and suggestions inside a linear limit, such as two flows that share a pump.

```python
capped = OptimizationSession()
for name in ("x1", "x2", "x3"):
    capped.add_variable(name, "real", bounds=(0.0, 10.0))
capped.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=12.0, name="budget")

points = capped.generate_initial_design("lhs", n_points=8, random_seed=3)
print(all(p["x1"] + p["x2"] <= 12.0 for p in points))
```

Every design method and a BoTorch suggestion respect the constraint. What each one does with it, and the equality form, are on the [Constraints](../guide/constraints.md) page.

## Generate an optimal design

Choose runs for a model you name, such as a quadratic.

```python
design_points, design = capped.generate_optimal_design(
    model_type="quadratic", n_points=12, criterion="D", random_seed=3
)
print(len(design_points), round(design["D_eff"], 1))
```

The call returns the points and a dictionary describing the design. Here it also respects the constraint above. Criteria, algorithms and the `effects` form are on the [Optimal design](../guide/more/optimal-design.md) page.

## Save and load a session

Keep a campaign in one file and open it again.

```python
session.save_session("campaign.json")

restored = OptimizationSession()
restored.load_session("campaign.json", retrain_on_load=False)
print(len(restored.experiment_manager.get_data()), restored.model is None)
```

The file holds the variables, input and outcome constraints, results, queue, provenance, audit log and the model's training arguments, not the fitted model. By default `load_session` trains the model again from those arguments; `retrain_on_load=False` skips that. You can call `load_session` on a session you already have, as above, or on the class, as in `OptimizationSession.load_session("campaign.json")`, which returns a new session. Either way the session ends up with the file's variables, results, queue and provenance. Pass a file path, not a directory.

!!! warning "Known issue"
    A8: `save_session("results/")` treats the trailing slash as part of a file name and writes a file called `results`, or fails if a directory of that name exists.

The variables and the results table can travel on their own.

```python
session.search_space.save_to_json("variables.json")
session.experiment_manager.save_to_csv("results.csv")

fresh = OptimizationSession()
fresh.load_search_space("variables.json")
fresh.load_data("results.csv", target_columns="yield")
print(len(fresh.experiment_manager.get_data()))
```

`variables.json` also carries the constraints, and `results.csv` the `Noise`, `Iteration` and `Reason` columns. After `load_search_space`, bring the results in with `load_data` as above, or add them one by one with `add_experiment`. `session.export_session_json()` returns the session file as a string when you want to store it somewhere other than a path. The keys of every file are in [File formats](../reference/file-formats.md), and the web app's side of saving is on the [Sessions](../guide/more/sessions.md) page.

## Plot the model

Draw the model's predictions and its cross-validation check, and save the figure.

Set the `Agg` backend if the script runs without a display. Each plot method returns a matplotlib `Figure`. `plot_contour`, `plot_surface`, `plot_parity` and `plot_regret` also take an `ax` to draw on, so two plots can share one figure; the others always make their own.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
session.plot_contour("temperature", "pressure", ax=left, subplot_labels=True)
session.plot_parity(ax=right)
fig.tight_layout()
fig.savefig("model.png", dpi=150)
```

Most session plots read the current model, so call `train_model` first, and again after new data. The ones you will use most are below; all of them take `figsize`, `dpi`, `title` and `formatters` for tick labels.

| Method | Shows |
|---|---|
| `plot_contour(x_var, y_var)` | Predicted response over two variables, with the experiments. |
| `plot_surface(x_var, y_var)` | The same as a 3D surface. |
| `plot_slice(x_var)` | Predicted response along one variable, with an uncertainty band. |
| `plot_voxel(x_var, y_var, z_var)` | Predicted response over three variables. |
| `plot_parity()` | Cross-validated prediction against measurement. |
| `plot_regret()` | The best value so far after each experiment; for a multi-objective session, hypervolume instead, which needs `ref_point`. |
| `plot_acquisition_contour(x_var, y_var)` | An acquisition function over two variables, `acq_func='ei'` by default. |
| `plot_pareto_frontier()` | The non-dominated points of a multi-objective session. |

For the full list of `plot_*` methods and their arguments, see the [OptimizationSession reference](../reference/python/session.md).

The plotting functions underneath are plain functions of arrays. They need no session and return a figure and an axes, so you can use them with predictions you computed yourself.

```python
from alchemist_core.visualization import create_parity_plot

cv = session.model.cv_cached_results
fig, ax = create_parity_plot(cv["y_true"], cv["y_pred"], cv["y_std"], title="Yield, cross-validation")
fig.savefig("parity.png")
```

!!! info "How ALchemist does it"
    The functions in `alchemist_core/visualization/plots.py` take arrays and return matplotlib objects, with no session or web code in them. The `plot_*` methods on the session build the grid, ask the model for predictions and call them; see the [visualization reference](../reference/python/visualization.md).
