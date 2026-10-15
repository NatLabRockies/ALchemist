# Quickstart

This page runs a whole optimization campaign from one Python script, on a synthetic response, so you can see every step of the loop in code before you point it at your own experiments.

Install ALchemist first (`pip install alchemist-nrel`; see [Install](../get-started/install.md)). Nothing has to run in the background: the script uses the library directly.

## The script

The function `run_experiment` stands in for the lab. It returns a yield, in percent, that peaks at 255 °C, 60 bar and an H₂ : CO₂ ratio of 3, with a little noise. Replace it with your own measurements.

```python
import numpy as np
from alchemist_core import OptimizationSession


def run_experiment(point, rng):
    """Stands in for the lab: a yield (%) that peaks at 255 °C, 60 bar and 3 mol/mol, with noise."""
    t = (point["temperature"] - 255) / 40
    p = (point["pressure"] - 60) / 35
    r = (point["h2_co2"] - 3) / 1.5
    return round(20 - 8 * t**2 - 5 * p**2 - 3 * r**2 + 2 * t * p + rng.normal(0, 0.3), 2)


rng = np.random.default_rng(0)

# 1. The variables: a name, a type and a range for each
session = OptimizationSession()
session.update_metadata(name="Quickstart", description="Synthetic three-variable example")
session.add_variable("temperature", "real", bounds=(200.0, 300.0), unit="°C")
session.add_variable("pressure", "real", bounds=(20.0, 80.0), unit="bar")
session.add_variable("h2_co2", "real", bounds=(2.0, 4.0), unit="mol/mol")

# 2. An initial design of ten points, run and recorded
for point in session.generate_initial_design("lhs", n_points=10, random_seed=42):
    session.add_experiment(point, output=run_experiment(point, rng), reason="Initial DoE (LHS)")

# 3. Three rounds: train the model, suggest a batch of three, run and record them
for round_number in range(1, 4):
    session.train_model(backend="botorch")
    batch = session.suggest_next(strategy="qEI", goal="maximize", n_suggestions=3)
    for point in batch.to_dict("records"):
        session.add_experiment(point, output=run_experiment(point, rng), reason="qEI")
    best = session.experiment_manager.get_data()["Output"].max()
    print(f"round {round_number}: {len(session.experiment_manager.get_data())} results, best {best:.2f} %")

# 4. The model's best guess, and the session saved to a file
session.train_model(backend="botorch")
optimum = session.find_optimum(goal="maximize")
print(optimum["x_opt"].round(1).to_dict("records"), round(optimum["value"], 2))
session.save_session("quickstart_session.json")
```

It takes a few seconds. The best yield rises round by round, from about 19.2 % in the design to about 19.8 %, and the optimum it reports lies within a few units of 255 °C, 60 bar and 3 mol/mol, where the synthetic response peaks. BoTorch prints a `NumericsWarning` that recommends qLogEI over qEI; it does not stop the script.

!!! warning "Known issue"
    N2: importing `alchemist_core` sets up Python's logging at the INFO level and replaces any configuration you made before, so every step prints a log line. To keep only warnings, call `set_verbosity(False)` after the import (`from alchemist_core import set_verbosity`).

## What each step does

| Step | Call | More |
|---|---|---|
| Define the variables | `add_variable(name, type, bounds=...)`; also `integer`, `discrete` and `categorical` types | [Define variables](../guide/variables.md) |
| Generate the first experiments | `generate_initial_design("lhs", n_points=10, random_seed=42)` returns a list of dictionaries | [Initial design](../guide/initial-design.md) |
| Record a result | `add_experiment(point, output=..., reason=...)`; `load_data("file.csv")` reads many at once | [Run and record](../guide/run-and-record.md) |
| Fit the model | `train_model(backend="botorch")`, which uses a Matérn kernel and normalizes inputs and standardizes outputs | [Train a model](../guide/train.md) |
| Check it | `plot_parity()`, `plot_qq()` and the metrics `train_model` returns | [Check the model](../guide/check-model.md) |
| Suggest the next batch | `suggest_next(strategy="qEI", goal="maximize", n_suggestions=3)` returns a DataFrame, one row per point | [Suggest next](../guide/suggest.md) |
| Report the best point | `find_optimum(goal="maximize")` returns `x_opt`, `value` and `std` | [Suggest next](../guide/suggest.md#3-find-the-models-optimum) |
| Keep the campaign | `save_session(path)`; `load_session(path)` opens it again, and the web app's **Load Session** opens the same file | [Save and load a session](recipes.md#save-and-load-a-session) |

The model is trained again before each suggestion: Python does not retrain by itself when results arrive.

## Open the result

The saved file holds the variables, the 19 results, the audit log and the model's settings. Load it back, here without retraining:

```python
restored = OptimizationSession()
restored.load_session("quickstart_session.json", retrain_on_load=False)
print(len(restored.experiment_manager.get_data()), restored.metadata.name)
```

## From here

- `add_experiment` records a result with no link to the suggestion it came from. To keep that link, the suggested point beside the conditions actually run, stage the points and complete them: see [Staging and the work queue](../automation/queue.md) and [Provenance](../automation/provenance.md).
- [Recipes](recipes.md) covers CSV files, several targets, constraints, optimal designs and plots.
- [The autonomous loop](../automation/loop.md) runs the same loop against the REST API, the way an instrument controller does.
- Every method and argument is in the [Python API reference](../reference/python/index.md).
