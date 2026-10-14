# Constraining the Variable Space

Variable bounds describe a box. Real experimental spaces are often not boxes:
two settings may have a combined budget, a ratio may have to hold, a corner of
the box may be unreachable. A **linear input constraint** carves that box down
to the region you can actually run.

A constraint is a linear relation over numeric variables:

- **inequality** — `sum(coeff_i * x_i) <= rhs`
- **equality** — `sum(coeff_i * x_i) == rhs`

Every surface that proposes a point in the input space honors the constraints
that are registered when it runs, so a suggestion is not generated inside the
excluded region. One lifecycle operation can change what "registered" means
without telling you — see the warning under [Names](#names).

!!! note "Running the examples"
    The Python examples run top to bottom as one script. The first block
    imports and builds a session, and the two that immediately follow continue
    it. From the estimability gate onward, **each block that opens with
    `session = OptimizationSession()` rebuilds its session from scratch** —
    constraints accumulate on a search space, and those examples need a
    different constraint set than the one before. Two blocks raise on purpose,
    and say so. The last block writes `space.json` into your current
    directory.

---

## Registering a Constraint

```python
from alchemist_core import OptimizationSession

session = OptimizationSession()
session.add_variable("x1", "real", min=0.0, max=10.0)
session.add_variable("x2", "real", min=0.0, max=10.0)
session.add_variable("x3", "real", min=0.0, max=10.0)

# x1 + x2 must not exceed 12
session.add_input_constraint(
    "inequality", {"x1": 1.0, "x2": 1.0}, rhs=12.0, name="budget"
)
```

Variables not named in `coefficients` are unconstrained — `x3` above is free
over its full range.

Over REST:

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

```json
{
  "message": "Constraint added successfully",
  "constraint": {
    "type": "inequality",
    "coefficients": {"x1": 1.0, "x2": 1.0},
    "rhs": 12.0,
    "name": "budget"
  }
}
```

`GET /api/v1/sessions/{session_id}/constraints` lists them;
`DELETE /api/v1/sessions/{session_id}/constraints/{name}` removes exactly one.

### Names

A constraint's **name is its delete identity**, so names are unique. Omit
`name` and you get an auto-generated `constraint_0`, `constraint_1`, …; supply
one that is already registered and the call is rejected. Deletion is by name
rather than by index, because an index shifts as soon as an earlier constraint
is removed and a client holding one would delete the wrong constraint.

!!! warning "Deleting a variable does not delete the constraints that name it"
    A constraint outlives the variable it references, and silently becomes a
    **different constraint**: feasibility filtering drops the absent variable
    from the sum rather than skipping the constraint, so after deleting `x2`,
    `x1 + x2 <= 2` is enforced as `x1 <= 2`. Designs are then generated over a
    region you never specified, with a `200` and no warning.

    Delete the constraint yourself whenever you delete a variable it names, and
    check `GET /constraints` after any variable deletion. Tracked as an open
    issue in [Troubleshooting](../ISSUES_LOG.md).

### Which variables may appear

Only **numeric** variables — `real`, `integer`, `discrete`. A `categorical` or
`context` variable in a constraint is rejected at registration:

```python
session.add_variable("c1", "categorical", values=["a", "b"])
session.add_input_constraint("inequality", {"x1": 1.0, "c1": 1.0}, rhs=5.0)
# ValueError: Variable 'c1' is not numeric (type 'categorical') and cannot
# appear in a linear constraint. Constraints may only reference variables of
# type real, integer, discrete.
```

There is no arithmetic that would make `"a" * 1.0` meaningful. Such
constraints were previously accepted and then failed later, deep inside
feasibility filtering; they now fail at the point you write them.

`rhs` and every coefficient must be a finite number. `NaN` and infinity are
refused: a `NaN` right-hand side makes every point infeasible, and neither
value survives a JSON round trip, so the constraint the API emits could not be
posted back.

---

## Where Constraints Are Honored

| Surface | Behavior |
|---|---|
| Acquisition (`suggest_next`) | Passed to the optimizer directly, in raw variable units |
| `find_optimum` | Search grid filtered to the feasible region |
| Contour / surface / slice / 3D plots | Infeasible cells masked (rendered blank) |
| Space-filling DoE (`random`, `lhs`, `sobol`, `halton`, `hammersly`) | Reject-and-resample until enough strictly feasible points are found |
| Classical DoE (`ccd`, `box_behnken`, `full_factorial`, …) | Infeasible structural points dropped, then [gated on estimability](#classical-designs-the-estimability-gate) |
| Optimal DoE (`optimal`) | Candidate set [filtered *and* augmented on the feasible boundary](#optimal-designs-the-augmented-candidate-set) |

Feasibility is decided in one place — `SearchSpace.filter_feasible` /
`is_feasible` — so no two surfaces can disagree about what is feasible. You
can call it yourself:

```python
# Judged against x1 + x2 <= 12, registered above.
session.search_space.is_feasible({"x1": 5.0, "x2": 5.0, "x3": 5.0})   # True
session.search_space.is_feasible({"x1": 9.0, "x2": 9.0, "x3": 5.0})   # False
```

Plot and grid judgments use a relative tolerance band, because an equality
constraint never lands exactly on a discrete grid. DoE sampling uses a strict
tolerance, so no returned design point exceeds a stated bound.

!!! note "sklearn backend"
    The scikit-optimize optimizer cannot express linear input constraints.
    Registering one and calling `suggest_next` raises a clear error. Use the
    `botorch` backend for constrained input optimization.

---

## Classical Designs: the Estimability Gate

A classical design is a **structure**: a CCD's factorial corners, axial points
and center replicates exist together to estimate a quadratic model. A
constraint that cuts through that structure removes points the model needs.

Dropping them and returning the remnant produces something that still looks
like a design and is not one. ALchemist therefore checks whether the surviving
points can still estimate the design's implied model, and raises when they
cannot:

```python
# A fresh session: x1, x2, x3 over [0, 10], one constraint.
session = OptimizationSession()
for name in ("x1", "x2", "x3"):
    session.add_variable(name, "real", min=0.0, max=10.0)
session.add_input_constraint("inequality", {"x1": 1.0, "x2": 0.8}, rhs=9.3)

session.generate_initial_design(method="ccd", random_seed=7)
```

```text
DesignNotEstimableError: 6 of 16 'ccd' design points violate the registered
input constraints and were dropped. The remaining 10 points can no longer
estimate the design's implied quadratic model — these terms became
inestimable: x1*x2. A classical design's value comes from its structure, so the
remnant is not the design it claims to be. Use method='optimal' for a genuine
constrained optimal design, or a space-filling method (random, lhs, sobol).
Pass allow_infeasible=True to return the remnant anyway.
```

The message names the terms you lost, so you know what the remnant could not
have told you.

**A constraint that only trims harmless points still returns normally.** The
gate fires on estimability, not on whether anything was dropped:

```python
# Again a fresh session over the same three variables, with x1 + x2 <= 11.
session = OptimizationSession()
for name in ("x1", "x2", "x3"):
    session.add_variable(name, "real", min=0.0, max=10.0)
session.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=11.0)

points = session.generate_initial_design(method="ccd", random_seed=7)
len(points)   # 12, from a 16-run CCD — the quadratic model is still estimable
```

### What to do instead

| Situation | Do this |
|---|---|
| You want a design tailored to the feasible region | `method="optimal"` — it is built for exactly this |
| You are exploring, not fitting a fixed polynomial | A space-filling method (`lhs`, `sobol`) |
| You genuinely want the remnant | `allow_infeasible=True` (below) |
| The constraint is tighter than you meant | Relax the constraint or widen the bounds |

### The escape hatch

```python
# The failing session from above: x1 + 0.8*x2 <= 9.3.
session = OptimizationSession()
for name in ("x1", "x2", "x3"):
    session.add_variable(name, "real", min=0.0, max=10.0)
session.add_input_constraint("inequality", {"x1": 1.0, "x2": 0.8}, rhs=9.3)

remnant = session.generate_initial_design(
    method="ccd", random_seed=7, allow_infeasible=True
)
len(remnant)   # 10 — the surviving points, quadratic model no longer estimable
```

This returns the surviving points and logs a warning naming the inestimable
terms. It is the pre-gate behavior, kept for callers who want the rows for
some other purpose. It is not available over REST — a design that cannot
estimate its own model is not something to make one query parameter away.

---

## Optimal Designs: the Augmented Candidate Set

An optimal design chooses runs from a candidate set. Under a constraint, the
naive move is to filter a lattice down to its feasible members — but the
extreme points an optimal design wants are precisely the ones a filter
removes, because they sit on the boundary that the constraint just created.

ALchemist filters the lattice **and augments it** with points lying on the
feasible region's boundary, including its vertices. The exchange algorithm
then selects from candidates it is actually allowed to keep, so a constrained
D-, A- or I-optimal design is genuinely optimal over its region.

```python
# A fresh session over x1, x2, x3 in [0, 10], with x1 + x2 <= 12.
session = OptimizationSession()
for name in ("x1", "x2", "x3"):
    session.add_variable(name, "real", min=0.0, max=10.0)
session.add_input_constraint(
    "inequality", {"x1": 1.0, "x2": 1.0}, rhs=12.0, name="budget"
)

points, info = session.generate_optimal_design(
    model_type="quadratic", n_points=12, criterion="D", random_seed=7
)
info["feasibility"]
# {'constraints_applied': ['budget'], 'n_candidates_total': 125,
#  'n_candidates_feasible': 75, 'n_boundary_added': 35,
#  'n_vertices_added': 4, 'vertex_enumeration_skipped': False}
```

This is also the only method that can serve an equality constraint over
`real` variables. Such a constraint defines a zero-volume slice, which a
continuous sampler draws past and never lands on; an optimal design places
points on constraint boundaries by construction. (Over `integer` or `discrete`
variables the slice contains reachable lattice points, so space-filling
methods work there too.)

!!! warning "An equality constraint restricts which models you can fit"
    An equality ties its variables together on every feasible candidate, so the
    intercept and the tied main effects become **exactly collinear** and the
    design matrix is rank-deficient. `model_type="linear"` and
    `model_type="quadratic"` therefore raise for `x1 + x2 == rhs` — the error
    names this case and its remedy. Pass an `effects` list that drops one of
    the tied variables:

    ```python
    session = OptimizationSession()
    for name in ("x1", "x2", "x3"):
        session.add_variable(name, "real", min=0.0, max=10.0)
    session.add_input_constraint(
        "equality", {"x1": 1.0, "x2": 1.0}, rhs=10.0, name="tie"
    )
    points, info = session.generate_optimal_design(
        effects=["x1", "x3"], n_points=6, criterion="D", random_seed=7
    )
    len(points)              # 6, every one exactly on x1 + x2 == 10
    round(info["D_eff"], 1)  # 92.5
    ```

    This is a property of the mathematics, not a limitation of the
    implementation: `x2` is not dropped from the design, only from the model —
    its values still vary, determined by `x1`.

!!! warning "Constrained optimal designs changed"
    A constrained optimal design now returns **different points for the same
    seed** than earlier versions, because it selects from the augmented
    candidate set rather than from a plain filtered lattice. This is the fix,
    not a regression. Unconstrained designs are unchanged at every seed.

---

## Reading the `feasibility` Block

Both design endpoints return a `feasibility` object next to the points. It is
`null` when no constraints are registered — that alone tells a client whether
a design was constrained at all, which used to be visible only in a server log
line.

```json
{
  "method": "ccd",
  "n_points": 12,
  "design_info": {"factorial_runs": 8, "axial_runs": 6, "center_runs": 2,
                  "total_runs": 16, "alpha": "orthogonal",
                  "face": "circumscribed"},
  "feasibility": {
    "constraints_applied": ["budget"],
    "n_candidates_total": null,
    "n_candidates_feasible": null,
    "n_boundary_added": null,
    "n_vertices_added": null,
    "vertex_enumeration_skipped": null,
    "n_points_dropped": 4,
    "estimability": "passed"
  }
}
```

| Field | Meaning |
|---|---|
| `constraints_applied` | Names of the constraints in force for this design |
| `n_candidates_total` | Lattice size before filtering (optimal designs only) |
| `n_candidates_feasible` | How many survived the filter (optimal designs only) |
| `n_boundary_added` | Points added on the feasible boundary (optimal designs only) |
| `n_vertices_added` | Of those, how many are region vertices (optimal designs only) |
| `vertex_enumeration_skipped` | `true` when the space has too many numeric variables to enumerate region vertices (optimal designs only) |
| `n_points_dropped` | Structural points removed by the constraint (classical designs only) |
| `estimability` | `"passed"`, `"waived"`, or `"not_applicable"` |

`estimability` reports the gate above, which applies to exactly one of the
three method classes:

- **classical** (`ccd`, `full_factorial`, …) — the gate runs, so a successful
  response means the design survived it: `"passed"`. With
  `"allow_infeasible": true` the gate is suppressed rather than passed, and
  neither outcome is claimed: `"waived"`.
- **`optimal`** — exempt, since its candidate set is already constrained and
  its model is user-specified: `"not_applicable"`.
- **space-filling** (`lhs`, `sobol`, …) — no implied model, no gate:
  `"not_applicable"`.

A field that does not apply to the method you called is `null`. Note that
`n_points_dropped` is `null` for a constrained *space-filling* design even
though its resample loop discarded draws; it counts structural points, not
rejected samples.

---

## Errors

Both constraint errors are `400`, with the class name in `error_type` so you
can branch on them:

| `error_type` | Cause | Remedy |
|---|---|---|
| `DesignNotEstimableError` | A classical design lost points its implied model needs | `method="optimal"`, a space-filling method, or `allow_infeasible=True` |
| `InfeasibleRegionError` | No feasible point exists, or the region is a zero-volume slice a sampler cannot reach | Relax the constraints, widen the bounds, or use `method="optimal"` for an equality slice |

```json
{
  "detail": "The registered input constraints leave no feasible point anywhere within the variable bounds, so no 'lhs' design can be generated. This is the constraint set itself, not the value of n_points: relax the constraints or widen the bounds.",
  "error_type": "InfeasibleRegionError",
  "status_code": 400
}
```

A provably empty region is refused immediately rather than discovered by
exhausting a resampling budget.

---

## Saving and Loading

Constraints persist with the session (`save_session` / `load_session`) and
with a search space saved on its own:

```python
session = OptimizationSession()
for name in ("x1", "x2", "x3"):
    session.add_variable(name, "real", min=0.0, max=10.0)
session.add_input_constraint(
    "inequality", {"x1": 1.0, "x2": 1.0}, rhs=12.0, name="budget"
)

session.search_space.save_to_json("space.json")   # written to the cwd

from alchemist_core.data.search_space import SearchSpace
restored = SearchSpace()
restored.load_from_json("space.json")
[c["name"] for c in restored.get_constraints()]   # ['budget']
```

Over REST, `GET /variables/export` returns a **bare array of variables** by
default — the shape `SearchSpace.from_dict` and the desktop loader consume, and
an existing contract that would break if it changed. Constraints are not in
that shape. Ask for them explicitly:

```http
GET /api/v1/sessions/{session_id}/variables/export?include_constraints=true
```

```json
{
  "variables": [
    {"name": "x1", "type": "real", "min": 0.0, "max": 10.0},
    {"name": "x2", "type": "real", "min": 0.0, "max": 10.0}
  ],
  "constraints": [
    {"type": "inequality", "coefficients": {"x1": 1.0, "x2": 1.0},
     "rhs": 12.0, "name": "budget"}
  ]
}
```

`POST /variables/load` accepts that same `{variables, constraints}` document as
well as the bare list, so `load → export → load` round-trips constraints:

```json
{
  "message": "Loaded 2 variables and 1 constraints successfully",
  "n_variables": 2,
  "n_constraints": 1
}
```

The dict form is validated **atomically**: a bad variable or a bad constraint
anywhere in the file rejects the whole document and leaves the session exactly
as it was. Loaded constraints go through the same validation as
`POST /constraints`, so a file cannot install a constraint the API would
refuse.

---

## See Also

- [Classical & Screening Designs](doe_classical.md)
- [Optimal Experimental Design](doe_optimal.md)
- [Setting Up the Variable Space](variable_space.md)
- [BoTorch Acquisition](../acquisition/botorch.md)
