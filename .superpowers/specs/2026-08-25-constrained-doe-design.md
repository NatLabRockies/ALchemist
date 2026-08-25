# Constrained DoE — design

**Status:** Approved (design). Implementation plan not yet written.
**Date:** 2026-08-25
**Scope:** `alchemist_core/utils/`, `alchemist_core/data/search_space.py`, `api/`
**Not in scope:** the web app (`alchemist-web/`) — see §10.

---

## 1. Summary

ALchemist supports linear constraints on input variables. They are honored in the
acquisition path and in space-filling DoE, but **not** in classical or optimal
DoE, and they **cannot be registered over the REST API at all**.

This design closes both gaps:

1. Optimal (D/A/I) designs select from a candidate set that is filtered *and*
   augmented with points on the feasible region's boundary, making them genuine
   constrained optimal designs.
2. Classical fixed-structure designs stop silently returning a degraded design
   when the constraint removes structural points; they raise unless the surviving
   points can still estimate the design's implied model.
3. Post-hoc variable spreading can no longer reintroduce infeasible values.
4. Constraints become settable over REST, so a non-Python consumer can use them.

---

## 2. The domain-agnostic boundary (non-negotiable)

ALchemist is a general-purpose optimization toolkit with multiple consumers. Per
§2.1 of the autonomous-monitoring-control-UX decision, **no consumer-specific
domain concept may enter this repository** — not in code, comments, docstrings,
test names, fixtures, or in this document.

A constraint is a generic linear relation over input variables, supplied as
configuration exactly as variable bounds are. ALchemist never learns what any
constraint *means*.

Throughout this spec the motivating case is described neutrally: **a consumer
needs to exclude a corner of a 3-variable continuous input space using a single
half-plane.** All fixtures use `x1`, `x2`, `x3`.

---

## 3. Current behavior (verified 2026-08-25)

| Path | Location | Behavior |
|---|---|---|
| Acquisition | `search_space.py:448` → `botorch_acquisition.py` | ✅ Correct — emits BoTorch `inequality_constraints` in raw variable space |
| Space-filling DoE | `doe.py:192-214` | ✅ Correct — reject-and-resample, oversampling factor 4 → 4096, clear error if the feasible region is too small |
| Classical DoE | `doe.py:261-291` | ❌ Generates ignoring constraints, drops infeasible points with `logger.warning`, returns the remnant |
| Optimal DoE | `optimal_design.py:988` → `:309` | ❌ No constraint awareness at all; the exchange algorithm optimizes over a candidate set containing infeasible points |
| Variable spreading | `optimal_design.py:1056-1075` | ❌ Overwrites non-model variables across their full range *after* selection, writing straight through any constraint |
| REST registration | — | ❌ No endpoint. `/variables/load` (`variables.py:97`) parses only the legacy bare-list format, so the `{variables, constraints}` dict that `SearchSpace.from_dict` supports (`search_space.py:339-344`) never reaches core |
| Categorical guard | `search_space.py:353` | ❌ `add_constraint` validates that a variable exists, not that it is numeric; a categorical coefficient fails later inside `filter_feasible` on `float(str)` |

### Why the classical case is a real defect

A classical design's statistical properties come from its *structure*. A CCD's
axial points are what let it estimate quadratic terms; its factorial corners are
what let it estimate interactions. Dropping the infeasible ones does not yield
"a CCD minus two runs" — it yields a design that is rank-deficient for the model
it claims to fit, returned with a log line the user will not see.

Meanwhile dropping a replicated center point is close to harmless. Current
behavior treats those two cases identically. That is the distinction this design
introduces.

### Why the optimal case is a sharper defect

The exchange algorithm spends its whole budget selecting points, then the
selected design is filtered afterward. It optimizes for points it will not be
allowed to keep.

---

## 4. Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | Candidate sets are **filtered and augmented with boundary points**, not merely filtered | `generate_mixed_candidate_set` builds a regular lattice. Filtering it leaves no candidates *on* the constraint boundary, and a D-optimal design wants precisely the extremes of the feasible region. Filter-only would produce a design hugging the box's corners with a dead band along the active constraint |
| D2 | Classical designs **raise unless the surviving points keep the implied model estimable** | Distinguishes a fatal structural loss from a harmless one, instead of treating every drop alike |
| D3 | Non-model variable spreading draws from **each row's feasible interval** | Preserves the feature's intent (unused variables look distributed, not clumped) while making infeasible output structurally impossible |
| D4 | Equality constraints are supported | The API already advertises `'equality'`; it is currently broken for optimal design because a lattice point satisfies an equality only by coincidence. Projection generates the feasible set exactly, so support falls out of D1 |
| D5 | Geometry lives in a **new module**, not in `SearchSpace` or the DoE files | The geometry is distinct from both "what variables exist" and "what design to generate", and is testable in isolation with neutral fixtures. Constraint-awareness is currently drifting independently across three files |
| D6 | Constraint **REST CRUD is in scope** | The motivating consumer speaks REST. Without registration, correct constrained DoE is unreachable by the consumer that needs it |

---

## 5. Architecture

### 5.1 New module — `alchemist_core/utils/constrained_region.py`

One unit, one purpose: **the geometry of a feasible region defined by linear
constraints and variable bounds.** Pure functions; no model, no design, no I/O.

```python
def project_onto_constraint(x: np.ndarray,
                            coeffs: np.ndarray,
                            rhs: float) -> np.ndarray:
    """Orthogonal projection of x onto the hyperplane c·x == rhs.

    x' = x - c * (c·x - rhs) / ||c||^2
    """


def feasible_interval(search_space, var_name: str,
                      fixed_values: Dict[str, float]) -> Optional[Tuple[float, float]]:
    """Interval of feasible values for one variable, others held fixed.

    Each constraint sum(c_j x_j) <= rhs collapses to a one-sided bound on x_v:
        c_v > 0  ->  upper bound
        c_v < 0  ->  lower bound
        c_v == 0 ->  no information
    An equality contributes both. Intersected with the variable's own bounds.
    Returns None when the intersection is empty.
    """


def feasible_vertices(search_space, *,
                      fixed: Optional[Dict[str, Any]] = None,
                      max_vars: int = 5) -> pd.DataFrame:
    """Vertices of the feasible polytope over the continuous variables.

    Intersections of n hyperplanes drawn from (constraints ∪ bound faces),
    taken n at a time, retained when feasible. Returns empty above max_vars.
    """


def augment_with_boundary(search_space, points: pd.DataFrame, *,
                          max_vertex_vars: int = 5,
                          rtol: float = 0.0,
                          atol: float = 1e-9) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Feasible candidate set: filtered lattice + boundary points + vertices.

    Returns the augmented frame and a provenance dict (§7.3).
    """
```

**Coordinate space.** All functions operate in **raw variable space**, matching
`filter_feasible` and `to_botorch_constraints`. `generate_mixed_candidate_set`
returns *coded* `[-1, +1]` values, so the optimal-design pipeline decodes,
augments, and re-encodes (§5.2). `filter_feasible` therefore remains the single
definition of "feasible" in the codebase — nothing re-implements the predicate.

**Combinatorial cap.** `feasible_vertices` intersects `C(n_planes, n_vars)`
combinations. For 3 continuous variables with 1 constraint that is
`C(7, 3) = 35` — trivial. It grows badly, so above `max_vars` (default 5)
vertex enumeration is skipped, projection-derived points are still added, and
the omission is recorded in the provenance dict and logged. **A reduced
candidate set must never be silent.**

### 5.2 `augment_with_boundary` algorithm

Run once **per categorical/discrete combination present in the input**;
categorical columns are held fixed while the continuous sub-vector is projected.

1. **Filter.** `search_space.filter_feasible(points, rtol=0.0, atol=1e-9)` —
   the strict tolerance already used at `doe.py:200-203`. Keep the survivors.
2. **Project.** For each infeasible point, for each constraint it violates,
   project onto that constraint's hyperplane. Then:
   - clip to variable bounds,
   - round integer variables, snap `discrete` variables to their nearest
     allowed value,
   - re-test against **all** constraints; keep only if now feasible.

   Snapping and clipping can push a projected point back out of the region;
   the re-test is what makes that safe.
3. **Add vertices.** `feasible_vertices(...)`, subject to the cap.
4. **Deduplicate** at tolerance (round to a decimal grid derived from `atol`).
5. **Empty check.** If nothing survives, raise `InfeasibleRegionError` (§7.4)
   naming the constraints, rather than returning an empty design.

Equality constraints need no special case: step 2 projects every point onto the
equality hyperplane, and that hyperplane *is* the feasible set.

### 5.3 Optimal design integration

In `run_optimal_design`, between candidate generation (`:988`) and design-matrix
construction (`:999`):

```python
candidates_coded, column_map = generate_mixed_candidate_set(search_space, n_levels=n_levels)

if getattr(search_space, 'constraints', None):
    raw = _decode_all_candidates(candidates_coded, column_map, variables)
    raw, feas_info = constrained_region.augment_with_boundary(search_space, raw)
    candidates_coded = _encode_candidates(raw, column_map, variables)
else:
    feas_info = None
```

Everything downstream — design matrix, exchange algorithm, decode — is unchanged.

**Two new private helpers in `optimal_design.py`:**

- `_decode_all_candidates(...)` — `_decode_candidates` (`:680`) restricted to
  selected indices; this variant decodes every row. Refactor the existing
  function to take an optional index set rather than duplicating the body.
- `_encode_candidates(...)` — the inverse. Does not currently exist. Continuous:
  `coded = (actual - mid) / half_range`. Categorical: one-hot from the category
  name. Must round-trip exactly for lattice points, which is a test in its own
  right (§9).

### 5.4 Classical design estimability gate

Replacing the block at `doe.py:261-291`.

**Implied model per method:**

| Method | Implied model |
|---|---|
| `ccd`, `box_behnken` | quadratic |
| `full_factorial` with `n_levels >= 3` | quadratic |
| `full_factorial` with `n_levels == 2` | interaction |
| `fractional_factorial` | interaction |
| `plackett_burman`, `gsd` | linear (main effects) |

**Applies to `method in CLASSICAL_METHODS - {"optimal"}` when constraints are
registered.** `optimal` is a member of `CLASSICAL_METHODS` but is excluded: its
points are already guaranteed feasible by §5.2/§5.3, and its model is
user-specified rather than implied.

**Procedure:**

1. Filter. If nothing is feasible → raise (existing behavior, message retained).
2. If nothing was dropped → return unchanged. **No behavior change whatsoever.**
3. If something was dropped, build the model matrix from the *survivors* using
   the existing `parse_model_spec` + `build_custom_design_matrix` from
   `optimal_design.py`, with the implied model above.
4. `numpy.linalg.matrix_rank(X) == p` →
   - **pass**: log at INFO that points were dropped and the model remains
     estimable, return the remnant;
   - **fail**: raise `DesignNotEstimableError` (§7.4) naming the method, how
     many of how many points were dropped, which terms became inestimable, and
     steering the user toward `optimal` or a space-filling method.
5. `allow_infeasible: bool = False` on `generate_initial_design` restores the
   old warn-and-drop for callers that need it.

Reusing `optimal_design`'s machinery from `doe.py` is not a new coupling —
`get_design_info` already does exactly this at `doe.py:760`, importing
`parse_model_spec` and `get_model_term_names` inside the `optimal` branch. The
gate follows that established pattern, keeping one definition of "build a model
matrix for these terms" rather than a second, drifting copy.
`get_model_term_names` also gives the raise message its list of inestimable
terms for free.

### 5.5 Feasible-interval spreading

Replacing the loop at `optimal_design.py:1056-1075`.

- A variable named in **no** constraint keeps today's behavior **bit-for-bit** —
  shuffled `linspace` over its full range.
- A variable named in a constraint is assigned per row:
  1. Take the other variables' values for that row as fixed.
  2. `feasible_interval(...)` → `[lo_i, hi_i]`.
  3. Value = `lo_i + u_i * (hi_i - lo_i)`, where `u` is an evenly-spaced
     sequence in `[0, 1]` of length `n`, shuffled by the same
     `np.random.default_rng(random_seed)` stream the current code already uses
     at `optimal_design.py:1061`, so seeded runs stay reproducible.
  4. Empty interval (should be unreachable — the selected point was feasible) →
     keep the selected value and log a warning.
- Process constrained variables **sequentially**, treating already-assigned ones
  as fixed, so variables sharing a constraint cannot jointly violate it.
- Final assertion: `filter_feasible` over the returned design is all-True. A
  constrained optimal design returning an infeasible point is a bug, and it
  should fail loudly rather than reach a consumer.

### 5.6 Categorical guard

`SearchSpace.add_constraint` gains a type check: every variable named in
`coefficients` must be `real`, `integer`, or `discrete`. A `categorical` raises
`ValueError` at registration, listing the offending name — instead of a
`float(str)` failure deep inside `filter_feasible` at design time.

This is a **breaking change** for any caller currently registering such a
constraint, but that caller is already broken; it simply fails later and less
clearly.

---

## 6. Component boundaries

| Unit | Does | Depends on |
|---|---|---|
| `constrained_region` | Geometry of a linear-constrained region: projection, vertices, intervals, augmentation | `numpy`, `pandas`, `SearchSpace` (reads `.variables`, `.constraints`, calls `filter_feasible`) |
| `SearchSpace` | Defines/serializes variables + constraints; **is** the definition of feasibility | unchanged |
| `optimal_design` | Model terms, design matrices, optimality criteria, exchange algorithms | `constrained_region` (new) |
| `doe` | Design generation + method routing + estimability gate | `constrained_region`, `optimal_design` (new) |
| `api/routers/variables` | Constraint CRUD, search-space load/export | `SearchSpace` |

`constrained_region` can be understood, used, and tested without reading any
DoE code — the isolation test for this boundary.

---

## 7. API surface

All additions are **additive**; no existing field or route changes shape.

### 7.1 Constraint CRUD — `api/routers/variables.py`

```
POST   /sessions/{session_id}/constraints    -> ConstraintResponse
GET    /sessions/{session_id}/constraints    -> ConstraintsListResponse
DELETE /sessions/{session_id}/constraints/{name} -> 204
```

Request body:

```json
{
  "constraint_type": "inequality",
  "coefficients": {"x1": 0.5, "x2": -1.0},
  "rhs": -10.0,
  "name": "half_plane_1"
}
```

Mirrors `SearchSpace.add_constraint` exactly. The categorical guard (§5.6) is
enforced here too, so a bad constraint is rejected at registration with a 400.

`DELETE` requires a constraint name. `add_constraint` auto-names as
`constraint_{n}`, which is positional and shifts when an earlier one is removed
— so deletion is by the stored name, and `GET` returns those names.

### 7.2 `/variables/load` dict format

`load_variables_from_file` (`variables.py:97`) currently iterates a bare list.
It gains a branch: a `dict` payload with a `variables` key is routed through
`SearchSpace.from_dict`, carrying `constraints` through. A `list` payload keeps
today's path exactly. `/variables/export` already emits whatever `to_dict`
produces (`search_space.py:330`), which includes constraints, so export needs no
change — worth an explicit round-trip test.

### 7.3 Feasibility reporting

New optional `feasibility` field on `InitialDesignResponse`
(`responses.py:233`) and `OptimalDesignResponse` (`responses.py:284`). The two
design families populate different subsets, so unused keys are `null`.

On `OptimalDesignResponse` — candidate-set provenance:

```json
{
  "constraints_applied": ["half_plane_1"],
  "n_candidates_total": 125,
  "n_candidates_feasible": 74,
  "n_boundary_added": 31,
  "n_vertices_added": 6,
  "vertex_enumeration_skipped": false,
  "n_points_dropped": null,
  "estimability": "not_applicable"
}
```

On `InitialDesignResponse` for a classical method — drop + estimability:

```json
{
  "constraints_applied": ["half_plane_1"],
  "n_candidates_total": null,
  "n_candidates_feasible": null,
  "n_boundary_added": null,
  "n_vertices_added": null,
  "vertex_enumeration_skipped": null,
  "n_points_dropped": 2,
  "estimability": "ok"
}
```

The whole `feasibility` field is `null` when no constraints are registered, so
unconstrained responses stay byte-identical. `estimability` is `"ok"`,
`"not_applicable"` (optimal and space-filling methods), or `null`.
`vertex_enumeration_skipped` is how the `max_vars` cap (§5.1) reaches the
caller instead of hiding in a log.

### 7.4 Errors

Two new exceptions in `api/middleware/error_handlers.py`, following the existing
`NoDataError` / `NoVariablesError` pattern (`:14-31`), both mapped to **400**:

- `DesignNotEstimableError` — §5.4 step 4 failure.
- `InfeasibleRegionError` — §5.2 step 5 failure.

The core library raises plain `ValueError` subclasses defined in
`constrained_region` / `doe`; the API layer translates. Core must not import
from `api/`.

---

## 8. Data flow

```
SearchSpace (variables + constraints)
        |
        +-- classical method --> generate design (structure-determined)
        |                         -> filter_feasible
        |                         -> estimability gate (§5.4)  --> raise | return
        |
        +-- space-filling ------> reject-and-resample (unchanged, §3)
        |
        +-- optimal ------------> generate_mixed_candidate_set (coded lattice)
                                  -> decode to raw
                                  -> augment_with_boundary (§5.2)
                                       filter | project | vertices | dedup
                                  -> encode to coded
                                  -> build_custom_design_matrix
                                  -> exchange algorithm
                                  -> decode selected
                                  -> feasible-interval spreading (§5.5)
                                  -> final feasibility assertion
```

---

## 9. Testing

**TDD, red-green-refactor.** All fixtures neutral: `x1`, `x2`, `x3`, and a
generic half-plane such as `0.5*x1 - x2 <= -10`.

### 9.1 Back-compat golden tests — written FIRST

Before any production code changes, capture golden outputs at a fixed seed for
every method in `SPACE_FILLING_METHODS | CLASSICAL_METHODS` with **no
constraints registered**, and assert equality thereafter. Unconstrained
behavior must be bit-for-bit unchanged, and this makes that a verified property
rather than a claim.

### 9.2 `constrained_region` unit tests

- `project_onto_constraint`: projection lands on the hyperplane; is idempotent;
  is a no-op for an already-satisfied equality.
- `feasible_interval`: correct one-sided bounds for positive/negative
  coefficients; both-sided for equality; `None` on empty intersection; ignores
  zero coefficients.
- `feasible_vertices`: for a known 2-D triangle, returns exactly its 3 vertices;
  empty above `max_vars`.
- `augment_with_boundary`: every returned row is feasible; boundary points
  exist *on* the constraint within tolerance; integer/discrete snapping never
  yields an infeasible row; runs per categorical combination.

### 9.3 Optimal design

- Every point of a constrained design is feasible.
- The augmented candidate set contains points on the active constraint that the
  filtered lattice does not.
- D-efficiency of the constrained design ≥ that of a filter-only design on the
  same problem (the quantitative justification for D1).
- `_encode_candidates` ∘ `_decode_all_candidates` is the identity on a lattice.
- Equality constraint yields a design lying on the hyperplane.
- Empty feasible region raises `InfeasibleRegionError`.

### 9.4 Classical designs

- Constrained CCD losing an axial point raises `DesignNotEstimableError`.
- Constrained design losing only a replicated center point returns normally.
- `allow_infeasible=True` restores warn-and-drop.
- Unconstrained CCD is unchanged (covered by §9.1).

### 9.5 Spreading

- Unused-and-unconstrained variable: output identical to current at fixed seed.
- Unused-and-constrained variable: all rows feasible, values distributed rather
  than clumped.
- Two unused variables sharing one constraint: jointly feasible.

### 9.6 API

- Constraint CRUD round-trip; delete by name.
- `add_constraint` with a categorical → 400.
- `/variables/load` with the dict format registers constraints; bare-list format
  still works; `load` → `export` → `load` round-trips constraints.
- `feasibility` is `null` when unconstrained, populated when constrained.
- Estimability failure surfaces as 400 with an actionable message.

### 9.7 Commands

```
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/ -q
```

Frontend is untouched by this design; if any response model changes in a way
that regenerates types, run `npx tsc --noEmit && npm test && npm run build` in
`alchemist-web/`.

---

## 10. Back-compat and breaking changes

**Unchanged (guaranteed by §9.1):** all unconstrained DoE, every space-filling
method constrained or not, all acquisition behavior, every existing API route
and response shape.

**Breaking, intentional, to be flagged in `CHANGELOG.md`:**

1. A constrained classical design that loses structural points now **raises**
   instead of returning a degraded design. Escape hatch: `allow_infeasible=True`.
2. A constrained optimal design **returns different points than before** for the
   same seed — selected from an augmented candidate set. This is the fix, not a
   regression.
3. `add_constraint` now **rejects categorical variables** at registration. Such
   constraints were already non-functional.

**Deliberately out of scope:**

- **Web UI.** The web app has no constraint surface at all (zero occurrences of
  "constraint" in `alchemist-web/src`). Expressing a linear constraint in a form
  is its own design problem and deserves its own brainstorm. A read-only
  feasibility display in `InitialDesignPanel` / `OptimalDesignPanel` is a small
  follow-on once §7.3 exists.
- **Non-linear constraints.** `SearchSpace` models linear relations only.
- **Constraint-aware acquisition changes.** Already correct (§3).
- **The orphaned pause/stop control plane.** Owned by no step; unrelated to this
  work.

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| Vertex enumeration blows up on many continuous variables | `max_vars` cap (default 5), projection-only fallback, surfaced via `vertex_enumeration_skipped` rather than a silent reduction |
| Projected points violate a *different* constraint after clipping/snapping | Step 2 re-tests against all constraints and discards failures |
| `_encode_candidates` round-trip drift on discrete variables | Explicit identity test (§9.3); snap-to-nearest-allowed already exists at `optimal_design.py:741-747` |
| Estimability gate false-positives on a design users were happy with | Rank check fires only when points were actually dropped; `allow_infeasible` escape hatch; the raise message names what was lost |
| `doe.py` importing `optimal_design.py` creates a cycle | Not a new risk: `optimal_design` imports `SearchSpace`, never `doe`, and `doe` already imports from `optimal_design` lazily in two places (`doe.py:240`, `doe.py:760`). Keep the estimability import function-local, matching those |

---

## 12. Implementation sequencing

Two phases, with a review checkpoint between them:

**Phase 1 — core.** Golden back-compat tests → `constrained_region` module →
optimal-design integration (§5.3) + encode/decode helpers → estimability gate
(§5.4) → interval spreading (§5.5) → categorical guard (§5.6).

**Phase 2 — API.** Constraint CRUD (§7.1) → `/variables/load` dict format
(§7.2) → feasibility reporting (§7.3) → error mapping (§7.4).

Phase 1 is independently valuable and independently verifiable. Phase 2 is what
makes it reachable by a non-Python consumer.
