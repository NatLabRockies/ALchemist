# 2. Constraints

Variable bounds describe a box, but real experiments often cannot use the whole box: two flows may share a pump, a ratio may have to hold, one corner may be unsafe. A linear input constraint cuts the box down to the combinations you can run. ALchemist then keeps every design and suggestion inside the cut.

!!! note "No constraints screen in the web app"
    The web app has no screen for input constraints, so there is no screenshot on this page. Set constraints from Python, over REST, or in a variables file loaded with **Load File**. Once a session has constraints, designs and suggestions made in the web app respect them, but nothing in the app lists them.

## What a constraint is

A constraint is a linear relation over numeric variables.

| Type | Meaning |
|---|---|
| `inequality` | `sum(coefficient * variable) <= rhs` |
| `equality` | `sum(coefficient * variable) == rhs` |

Only real, integer and discrete variables can appear. Variables you leave out are unconstrained. A constraint has a name, which is also how you delete it; names are unique, and ALchemist names an unnamed one `constraint_0`, `constraint_1` and so on. The coefficients and `rhs` must be finite numbers.

## 1. Add a constraint

=== "Web app"

    The app reads constraints from a variables file whose top level is an object with `variables` and `constraints`. This file defines three variables and one constraint, `x1 + x2 <= 12`:

    ```json
    {
      "variables": [
        {"name": "x1", "type": "real", "min": 0.0, "max": 10.0},
        {"name": "x2", "type": "real", "min": 0.0, "max": 10.0},
        {"name": "x3", "type": "real", "min": 0.0, "max": 10.0}
      ],
      "constraints": [
        {
          "type": "inequality",
          "coefficients": {"x1": 1.0, "x2": 1.0},
          "rhs": 12.0,
          "name": "budget"
        }
      ]
    }
    ```

    1. Save the text as a `.json` file.
    2. In **Variable Management**, choose **Load File** and pick the file.
    3. A message reads "Loaded 3 variables and 1 constraints successfully". The list shows the three variables and no constraint.

    Loading this form replaces the session's variables and constraints, so use a new session. The server checks the whole file first, and a file with a bad variable or constraint is refused and changes nothing.

=== "Python"

    ```python
    from alchemist_core import OptimizationSession

    session = OptimizationSession()
    for name in ("x1", "x2", "x3"):
        session.add_variable(name, "real", bounds=(0.0, 10.0))
    session.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=12.0, name="budget")
    print(session.search_space.get_constraints())
    print(session.search_space.is_feasible({"x1": 5.0, "x2": 5.0, "x3": 5.0}))
    print(session.search_space.is_feasible({"x1": 9.0, "x2": 9.0, "x3": 5.0}))
    ```

    A constraint on a categorical variable is refused when you add it:

    ```python
    mixed = OptimizationSession()
    mixed.add_variable("x1", "real", bounds=(0.0, 10.0))
    mixed.add_variable("c1", "categorical", categories=["a", "b"])
    try:
        mixed.add_input_constraint("inequality", {"x1": 1.0, "c1": 1.0}, rhs=5.0)
    except ValueError as err:
        print(err)
    ```

=== "REST"

    ```http
    POST /api/v1/sessions/{session_id}/constraints
    Content-Type: application/json

    {
      "constraint_type": "inequality",
      "coefficients": {"x1": 1.0, "x2": 1.0},
      "rhs": 12.0,
      "name": "budget"
    }
    ```

    The response holds the registered constraint under `constraint`. `GET /api/v1/sessions/{session_id}/constraints` lists them, and the request below removes one by name.

    ```http
    DELETE /api/v1/sessions/{session_id}/constraints/budget
    ```

!!! warning "Known issue"
    A14: deleting a variable does not delete the constraints that name it. The constraint then applies without that variable, so after you delete `x2`, `x1 + x2 <= 12` is enforced as `x1 <= 12`, with no warning. Delete the constraint yourself and check the constraint list after any variable deletion.

## 2. What a constraint changes

| Operation | With a constraint registered |
|---|---|
| Space-filling designs (`random`, `lhs`, `sobol`, `halton`, `hammersly`) | Draws more points than needed and keeps the feasible ones. |
| Classical designs | Drops the infeasible points, then checks that the rest can still estimate the design's model. |
| Optimal design | Chooses from the feasible region and from points on its boundary. |
| `suggest_next` with BoTorch | Searches only the feasible region. |
| `suggest_next` with scikit-learn | Raises a `ValueError`; over REST the call returns 400. |
| `find_optimum` | Searches the feasible part of the grid, with either backend. |
| `plot_contour` | Leaves infeasible cells blank. |

### Designs

```python
points = session.generate_initial_design("lhs", n_points=10, random_seed=7)
print(len(points), all(p["x1"] + p["x2"] <= 12.0 for p in points))
```

A classical design is a fixed structure, and a constraint that cuts through it removes points its model needs. ALchemist raises `DesignNotEstimableError` and names the model terms you lost, instead of returning a remnant that only looks like the design.

```python
from alchemist_core.utils.doe import DesignNotEstimableError

tight = OptimizationSession()
for name in ("x1", "x2", "x3"):
    tight.add_variable(name, "real", bounds=(0.0, 10.0))
tight.add_input_constraint("inequality", {"x1": 1.0, "x2": 0.8}, rhs=9.3)
try:
    tight.generate_initial_design(method="ccd", random_seed=7)
except DesignNotEstimableError as err:
    print(str(err)[:80])
remnant = tight.generate_initial_design(method="ccd", random_seed=7, allow_infeasible=True)
print(len(remnant))
```

`allow_infeasible=True` returns the surviving points anyway, with a warning in the log. Over REST the same switch is `allow_infeasible` in the initial-design request. A constraint that only trims harmless points returns normally. For a design that fits the feasible region, use a space-filling method or an optimal design.

!!! warning "Known issue"
    T11: ALchemist treats `fractional_factorial` as a design for an interaction model, which none of its default generators supports. A constraint that drops any point of a fractional factorial design therefore raises `DesignNotEstimableError`. Use a space-filling or optimal design, or `allow_infeasible=True`.

An optimal design picks its runs from the feasible region and reports what it did under `feasibility`:

```python
points, info = session.generate_optimal_design(
    model_type="quadratic", n_points=12, criterion="D", random_seed=7
)
print(len(points), info["feasibility"]["constraints_applied"], info["feasibility"]["n_candidates_feasible"])
```

An equality constraint over real variables leaves a region with no volume, which a sampler draws past and never lands on. Space-filling methods refuse it with `InfeasibleRegionError`. An optimal design can serve it, but the tied variables are collinear, so name the effects and leave one of them out of the model:

```python
tie = OptimizationSession()
for name in ("x1", "x2", "x3"):
    tie.add_variable(name, "real", bounds=(0.0, 10.0))
tie.add_input_constraint("equality", {"x1": 1.0, "x2": 1.0}, rhs=10.0, name="tie")
points, info = tie.generate_optimal_design(effects=["x1", "x3"], n_points=6, criterion="D", random_seed=7)
print(len(points), all(abs(p["x1"] + p["x2"] - 10.0) < 1e-6 for p in points))
```

Over REST, `POST /api/v1/sessions/{session_id}/initial-design` returns a `feasibility` object next to the points. It is `null` when no constraint is registered. For a classical design, `n_points_dropped` counts the structural points removed and `estimability` is `passed`, or `waived` when you set `allow_infeasible`. The web app does not show this object. If no feasible design exists, the call fails with `InfeasibleRegionError`, and the web app shows the message in a notification. The designs are described in [Initial design](initial-design.md) and [Optimal design](more/optimal-design.md).

### Suggestions

Constrained suggestions need the BoTorch backend. In the web app, open the **Model** tab, choose **BoTorch**, and choose **Train Model**; then use **Run Acquisition Strategy** on the **Acquisition** tab. With **scikit-learn** the same click shows the error below in a notification.

```python
def response(p):
    return -((p["x1"] - 6) / 5) ** 2 - ((p["x2"] - 4) / 5) ** 2 - ((p["x3"] - 5) / 5) ** 2

for p in session.generate_initial_design("lhs", n_points=10, random_seed=7):
    session.add_experiment(p, output=response(p))

session.train_model(backend="sklearn")
try:
    session.suggest_next(strategy="EI", goal="maximize")
except ValueError as err:
    print(err)

session.train_model(backend="botorch")
suggestion = session.suggest_next(strategy="EI", goal="maximize")
print(suggestion)
print(session.search_space.is_feasible(suggestion.iloc[0].to_dict()))
```

## Save and load constraints

A saved session keeps its constraints. A variables file keeps them only in the object form shown above. `save_to_json` writes that form, and `load_search_space` reads it:

```python
session.search_space.save_to_json("space.json")

restored = OptimizationSession()
restored.load_search_space("space.json")
print([c["name"] for c in restored.search_space.get_constraints()])
```

Over REST, `GET /api/v1/sessions/{session_id}/variables/export` returns a bare list of variables by default. Ask for the object form with the constraints included:

```http
GET /api/v1/sessions/{session_id}/variables/export?include_constraints=true
```

`POST /api/v1/sessions/{session_id}/variables/load` accepts either form, so a file exported this way loads back with its constraints.

!!! warning "Known issue"
    W35: the web app's **Save File** writes only the variables, as a bare list. If you opened a file with constraints using **Load File**, **Save File** can overwrite that linked file and drop its `constraints`. Keep the original file, or export with `include_constraints=true` from Python or REST.

!!! info "How ALchemist does it"
    Feasibility is decided in one place, `SearchSpace.filter_feasible`, so designs, suggestions and plots agree on what is feasible. The estimability check, the boundary points added for optimal designs and the references behind them are in [Constraints and optimal designs](../concepts/design-of-experiments.md#constraints-and-optimal-designs).
