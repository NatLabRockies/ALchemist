# Multi-Objective Bayesian Optimization

Most optimization problems involve a single measurable objective (yield, selectivity, conversion). However, many real experimental systems require balancing **multiple competing objectives simultaneously** — for example, maximizing yield while minimizing cost, or maximizing selectivity while maintaining throughput.

ALchemist supports multi-objective Bayesian optimization (MOBO), providing Pareto frontier visualization and hypervolume-based convergence tracking.

---

## Core Concepts

### The Pareto Frontier

When optimizing multiple objectives, there is rarely a single "best" solution. Instead, there is a **Pareto frontier** — the set of solutions where improving one objective necessarily worsens another.

A solution is **Pareto-optimal** if no other feasible solution is better on at least one objective without being worse on any other.

**Example:** Maximizing both yield and selectivity simultaneously. Higher yield often comes at the cost of lower selectivity (e.g., broader operating conditions). The Pareto frontier maps the achievable trade-off curve.

### Hypervolume

The **hypervolume** (or S-metric) measures the volume of the objective space dominated by the current Pareto frontier, relative to a reference point. It is a scalar convergence metric:

- **Increasing hypervolume** → the Pareto frontier is expanding (more experiments finding better trade-offs)
- **Plateauing hypervolume** → the frontier is well-characterized; further experiments offer diminishing returns

---

## Setting Up Multi-Objective Optimization

### Step 1: Define Your Variables

Set up your variable space as usual. Multi-objective workflows use the same `Real`, `Integer`, `Categorical`, and `Discrete` variable types.

### Step 2: Load Data with Multiple Target Columns

Your CSV file should have **one column per objective**, each containing numeric measurements.

**Example CSV:**
```
Temperature,Pressure,Catalyst,Yield,Selectivity
250,10,Ni,0.72,0.91
300,15,Pt,0.85,0.78
...
```

When uploading data in the web UI:

1. Click **Load Experiments**.
2. After selecting your CSV, a **Target Column Selection** panel appears showing all available columns.
3. Select the columns that represent your optimization objectives (e.g., `Yield` and `Selectivity`).
4. Columns not selected and not matching variable names are treated as **dropped columns** (ignored).
5. Click **Load**.

### Step 3: Configure Target Column Roles

Each column in your dataset can be assigned one of three roles:

| Role | Description |
|---|---|
| **Variable** | Controllable input (matches your variable space definition) |
| **Target** | Measured output to optimize — one per objective |
| **Drop** | Metadata column to ignore (e.g., experiment date, run ID) |

In the web UI, column roles are assigned automatically where names match your variable space, and you select targets from the remaining columns.

---

## Python API

```python
from alchemist_core import OptimizationSession

session = OptimizationSession()
session.add_variable("Temperature", "real", min=200, max=400)
session.add_variable("Pressure", "real", min=5, max=30)
session.add_variable("Catalyst", "categorical", categories=["Ni", "Pt", "Pd"])

# Load data with multiple target columns
session.load_data(
    "my_experiments.csv",
    target_columns=["Yield", "Selectivity"]
)
```

---

## REST API

```http
POST /api/v1/sessions/{session_id}/experiments/upload?target_columns=Yield,Selectivity
Content-Type: multipart/form-data
```

The `target_columns` parameter accepts a single column name or comma-separated names for multi-objective.

---

## Visualizations

### Pareto Plot

The Pareto plot shows the experimental data projected onto two objective dimensions, highlighting which experiments lie on the Pareto frontier.

- **Pareto-optimal points** are marked distinctly from dominated solutions
- Helps visualize the current trade-off curve and identify gaps to explore

Access in the web UI from the **Visualizations** panel → **Pareto Plot**.

### Hypervolume Convergence Plot

Tracks the hypervolume of the Pareto frontier across iterations. A rising curve indicates the active learning loop is making progress; a plateau suggests convergence.

Access from **Visualizations** → **Hypervolume Convergence**.

---

## Desktop GUI

The desktop application includes a dedicated **Multi-Objective** panel that becomes active when multiple target columns are loaded. From this panel you can:

- View the current Pareto frontier
- See which experiments are Pareto-optimal
- Inspect trade-off relationships between objectives

---

## Current Scope

MOBO in ALchemist is currently **Phase 1 (visualization and data infrastructure)**:

- ✅ Multi-target column loading
- ✅ Pareto frontier computation and visualization
- ✅ Hypervolume convergence tracking
- ✅ Desktop and web UI panels for MOBO data

Multi-objective acquisition functions (e.g., qEHVI, qNEHVI) for active Pareto improvement are planned for a future release.

---

## Tips

- **Choose objectives carefully:** Conflicting objectives (negatively correlated) produce meaningful Pareto frontiers. Aligned objectives (positively correlated) do not require MOBO — optimize one as a proxy.
- **Normalize objectives:** If objectives have very different scales, consider normalizing before interpretation, though ALchemist handles this internally for hypervolume computation.
- **Reference point:** The hypervolume is computed relative to an anti-ideal reference point derived from the worst observed values. Results are comparable within a session but not across sessions with different data scales.
