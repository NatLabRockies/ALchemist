# 3. Initial design

Before the optimizer has any results to learn from, it needs a first set of experiments. This page generates that set, queues it so you can record the results, and exports it.

![The Initial Design (DoE) panel on four simulated reactor variables: Method LHS, Points 10, Seed 42, LHS Criterion Maximin, a Generate Design button, the text "10 points generated" beside the Stage and CSV buttons, and a table of Temperature, Pressure, H2_CO2 and WHSV values.](../images/guide/initial-design.png)

*A Latin hypercube design of 10 points with seed 42. Stage queues the points for recording; CSV saves them.*

## Choose a method

Most sessions start with a space-filling design, which spreads points over the whole space without assuming a model. Classical and screening designs have a fixed structure for fitting a polynomial or finding the variables that matter. The theory is in [Design of experiments](../concepts/design-of-experiments.md). The **Method** menu groups the methods as shown.

| Group | **Method** menu | `method` | You set |
|---|---|---|---|
| Space-Filling | **LHS** | `lhs` | **Points**, **Seed (opt)**, **LHS Criterion** |
| | **Sobol** | `sobol` | **Points**, **Seed (opt)** |
| | **Halton** | `halton` | **Points** |
| | **Hammersly** | `hammersly` | **Points** |
| | **Random** | `random` | **Points**, **Seed (opt)** |
| Classical RSM | **Full Factorial** | `full_factorial` | **Levels per factor**, **Center points** |
| | **Fractional Factorial** | `fractional_factorial` | **Generators (opt)**, **Center points** |
| | **CCD** | `ccd` | **Alpha**, **Face**, **Center points** |
| | **Box-Behnken** | `box_behnken` | **Center points** |
| Screening | **Plackett-Burman** | `plackett_burman` | **Center points** |
| | **GSD** | `gsd` | **Levels per factor**, **Reduction** |

For the space-filling methods you choose the number of points. For the others the design decides: with four real variables and the default settings, **Full Factorial** gives 17 runs, **Fractional Factorial** 9, **CCD** 26, **Box-Behnken** 25, **Plackett-Burman** 9 and **GSD** 8. **Box-Behnken** needs at least three continuous variables. **Fractional Factorial**, **CCD**, **Box-Behnken** and **Plackett-Burman** do not support categorical variables; the panel warns you and disables **Generate Design**. Use **Full Factorial**, **GSD** or a space-filling method when a variable is categorical.

An optimal design, which targets a model you name, has its own panel: see [Optimal design](more/optimal-design.md). If a variable is hard to change, a **Split-plot** toggle appears above **Method**; see [Split-plot designs](more/split-plot.md).

## 1. Generate a design

1. In **Initial Design (DoE)**, pick a **Method**.
2. For a space-filling method, set **Points**. **Seed (opt)** is shown for **LHS**, **Sobol** and **Random**; leave it blank and each click draws a new design, or type a whole number to get the same design every time. **Halton** and **Hammersly** are fixed sequences and have no seed.
3. For **LHS**, **LHS Criterion** is **Maximin**, **Correlation** or **Ratio**. For **CCD**, **Alpha** is **Orthogonal** or **Rotatable**, and **Face** is **Circumscribed**, **Inscribed** or **Faced**.
4. Choose **Generate Design**. The panel reports "10 points generated" and lists them in a table.
5. Choose **Stage** to queue the points. The **Add Point...** button in the **Experiment Data** panel then shows the count, and steps through the staged points so you can enter each result; see [Run and record](run-and-record.md). Choose **CSV** to save the points as `initial_design_<method>_<N>pts.csv`, for example `initial_design_lhs_10pts.csv`. In Chrome and Edge on https or localhost a save dialog lets you pick the folder; other browsers, and plain http addresses, download the file.

Generating a design does not add anything to the session's data. **Stage** records each point with a reason such as "Initial DoE (LHS)", and writes the method and its settings to the audit log. Linear input constraints, if you set any, apply to every method; see [Constraints](constraints.md).

## Defaults

| Setting | Web app | Python and REST |
|---|---|---|
| **Method** | **LHS** | `method="lhs"` |
| **Points** | 10; the number field steps between 5 and 100 | `n_points` of `None` becomes 10 for space-filling methods; REST accepts 1 to 1000 |
| **Seed (opt)** | blank, shown as Auto | `random_seed=None`, a new design every call |
| **LHS Criterion** | **Maximin** | `lhs_criterion="maximin"` |
| **Center points** | 1 | `n_center=1`; REST accepts 0 to 10 |
| **Levels per factor** | 2 | `n_levels=2`; the web menu offers 2 or 3 for **Full Factorial** and 2 to 5 for **GSD** |
| **Generators (opt)** | blank, chosen from the number of variables | `generators=None` |
| **Alpha** | **Orthogonal** | `ccd_alpha="orthogonal"` |
| **Face** | **Circumscribed** | `ccd_face="circumscribed"` |
| **Reduction** | ÷2 | `gsd_reduction=2`; REST accepts 2 to 10 |

!!! warning "Known issue"
    T12: **Halton** is not a Halton sequence. It runs the same sampler as **Hammersly**, so the two give the same points. Use **Sobol** or **LHS** if you want a different sequence.

!!! warning "Known issue"
    T9: **Circumscribed** and **Inscribed** give the same **CCD**. The axial points land on the variable bounds and the factorial points sit inside them, for example at 2.6 and 7.4 with four variables on a range of 0 to 10 and the default **Orthogonal** alpha. **Faced** puts every point on the bounds or the center.

## From Python or REST

=== "Python"

    ```python
    from alchemist_core import OptimizationSession

    session = OptimizationSession()
    session.add_variable("temperature", "real", bounds=(200.0, 300.0), unit="°C")
    session.add_variable("pressure", "real", bounds=(20.0, 80.0), unit="bar")
    session.add_variable("h2_co2", "real", bounds=(2.0, 4.0))
    session.add_variable("whsv", "real", bounds=(1.5, 18.0))

    points = session.generate_initial_design("lhs", n_points=10, random_seed=42)
    print(len(points))
    print(points[0])
    ```

    Each point is a dictionary of variable values. Run it, then record the result with `session.add_experiment(point, output=...)`. Classical methods take their size from the design:

    ```python
    for method in ("full_factorial", "ccd", "plackett_burman"):
        print(method, len(session.generate_initial_design(method)))
    ```

=== "REST"

    ```http
    POST /api/v1/sessions/{session_id}/initial-design
    Content-Type: application/json

    {"method": "lhs", "n_points": 10, "random_seed": 42, "lhs_criterion": "maximin"}
    ```

    The response holds `points`, `method`, `n_points`, `design_info` (for classical designs) and `feasibility` (when constraints are set). The other request keys are the Python arguments above: `n_levels`, `n_center`, `generators`, `ccd_alpha`, `ccd_face`, `gsd_reduction` and `allow_infeasible`. Nothing is added to the session's data.

    To queue the points as the app's **Stage** does, post them to the staged-experiments route:

    ```http
    POST /api/v1/sessions/{session_id}/experiments/staged/batch
    Content-Type: application/json

    {"experiments": [{"temperature": 271.2, "pressure": 55.3, "h2_co2": 3.3, "whsv": 7.4}], "reason": "Initial DoE (LHS)"}
    ```

    This route is deprecated in the schema, and the web app still calls it. New code should stage points with `POST .../experiments/queue`; see [Stage points](../automation/queue.md#stage-points).

!!! info "How ALchemist does it"
    `generate_initial_design` in `alchemist_core/utils/doe.py` builds each design from its own random generator, so one seed gives one design even when requests overlap. Points for the space-filling methods come from scikit-optimize, and the classical designs from pyDOE. scikit-optimize warns that **Sobol** sequences lose their balance when the number of points is not a power of two. How each method works, and what its runs can estimate, is in [Design of experiments](../concepts/design-of-experiments.md). The arguments are listed in the [design reference](../reference/python/design.md).
