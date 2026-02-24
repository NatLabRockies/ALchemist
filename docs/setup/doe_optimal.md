# Optimal Experimental Design

**Optimal designs** generate the most information-rich set of experiments for a user-specified statistical model. Unlike space-filling or classical RSM designs, optimal designs let you say "I expect these specific effects to matter — give me the minimum runs that estimate them as precisely as possible."

---

## Core Concept

An optimal design maximizes (or minimizes) a mathematical criterion on the **information matrix** $\mathbf{X}'\mathbf{X}$, where $\mathbf{X}$ is the design matrix. The three supported criteria are:

| Criterion | Optimizes | Best for |
|---|---|---|
| **D-optimal** | Maximize $|\mathbf{X}'\mathbf{X}|$ | Parameter estimation precision |
| **A-optimal** | Minimize $\text{tr}[(\mathbf{X}'\mathbf{X})^{-1}]$ | Average parameter variance |
| **I-optimal** | Minimize integrated prediction variance | Prediction accuracy over the space |

**D-optimal is the recommended default** for most applications — it maximizes the information in the design for estimating all model parameters simultaneously.

---

## Specifying the Model

You specify which effects to include in the model. ALchemist builds the design matrix accordingly. The intercept is always included automatically.

### Model Type Shortcuts

The quickest way to specify a model — use a shortcut name:

| Shortcut | Includes |
|---|---|
| `"linear"` | Intercept + all main effects |
| `"interaction"` | Intercept + main effects + all two-factor interactions |
| `"quadratic"` | Intercept + main effects + interactions + quadratic terms (continuous only) |

### Custom Effects

For full control, pass an explicit list of effects using variable names:

```
Main effect:        "Temperature"
Two-factor interaction: "Temperature*Pressure"
Quadratic term:     "Temperature**2"
```

**Example — a selective model:**
```python
effects = [
    "Temperature",           # main effect
    "Pressure",              # main effect
    "Catalyst",              # main effect (categorical)
    "Temperature*Pressure",  # interaction
    "Temperature**2",        # quadratic (continuous only)
]
```

**Rules:**
- Use `*` for interactions: `"Temperature*Pressure"`
- Use `**2` for quadratic: `"Temperature**2"`
- Quadratic terms are only valid for Real and Integer variables (not Categorical or Discrete with exactly 2 values)
- Categorical variables can have main effects and interactions, but not quadratic terms

> **Tip:** Use the **Preview** button in the web UI to see how many model columns your specification produces, and the recommended run count, before generating the design.

---

## Run Count

You can specify run count in two ways:

| Mode | Parameter | When to use |
|---|---|---|
| **Absolute** | `n_points` | Fixed budget of experiments |
| **Multiplier** | `p_multiplier` | Proportional to model complexity |

The **p_multiplier** approach is recommended: `p_multiplier=2.0` gives 2× as many runs as model columns (`p`). This ensures the design is estimable with some degrees of freedom for lack-of-fit testing.

**Rule of thumb:** Use at least `p_multiplier=1.5` for reliable estimation; `2.0` is a solid default; `3.0` for high-quality estimation with outlier robustness.

---

## Exchange Algorithms

Optimal designs are found by iteratively exchanging candidate points to improve the criterion. Five algorithms are available:

| Algorithm | Speed | Quality | When to use |
|---|---|---|---|
| `sequential` (Dykstra) | Fastest | Lower | Quick prototyping, large spaces |
| `simple_exchange` | Fast | Moderate | General use |
| `fedorov` | Moderate | High | **Recommended default** |
| `modified_fedorov` | Moderate | High | Similar to Fedorov |
| `detmax` (Mitchell) | Slowest | Highest | Final design, critical applications |

**Fedorov** is the default — it performs full pairwise candidate exchanges and reliably finds high-quality designs. Use **DetMax** when you want the best possible design and runtime is not a concern.

---

## Web UI Walkthrough

1. Define your variable space.
2. In the **Initial Design** panel, switch to the **Optimal Design** tab.
3. Choose **Quick** (model type shortcut) or **Custom** (effect checkboxes + text input).
4. Click **Preview** to see the model terms and recommended run count.
5. Optionally adjust `n_points` to the recommended value, or set a `p_multiplier`.
6. Choose criterion (D/A/I) and algorithm.
7. Click **Generate**.
8. Review design quality metrics (D-efficiency, model terms).
9. Click **Stage Suggestions** to queue the design.

> **AI-Assisted Effect Selection:** The Optimal Design panel includes an AI suggestion tool. See [LLM-Assisted Effect Suggestion](llm_effects.md) for details.

---

## Python API

### Preview before generating

```python
from alchemist_core import OptimizationSession

session = OptimizationSession()
session.add_variable("Temperature", "real", min=200, max=350)
session.add_variable("Pressure", "real", min=1, max=10)
session.add_variable("Catalyst", "categorical", categories=["Ni", "Pt", "Pd"])

# Get model info (dry run — no design generated yet)
info = session.get_optimal_design_info(model_type="quadratic")
print(f"Model has {info['p_columns']} terms: {info['model_terms']}")
print(f"Recommended runs: {info['n_points_recommended']}")
```

### Generate design

```python
# Using a model type shortcut
points, info = session.generate_optimal_design(
    model_type="quadratic",
    p_multiplier=2.0,
    criterion="D",
    algorithm="fedorov",
    random_seed=42,
)

# Using a custom effects list
points, info = session.generate_optimal_design(
    effects=["Temperature", "Pressure", "Temperature*Pressure", "Temperature**2"],
    n_points=20,
    criterion="D",
    algorithm="detmax",
)

# Design quality metrics
print(f"D-efficiency: {info['D_eff']:.1f}%")
print(f"A-efficiency: {info['A_eff']:.1f}%")
print(f"Model terms: {info['model_terms']}")
```

---

## REST API

### Preview model terms

```http
POST /api/v1/sessions/{session_id}/optimal-design/info
Content-Type: application/json

{
  "model_type": "quadratic"
}
```

Response:
```json
{
  "p_columns": 9,
  "model_terms": ["Intercept", "Temperature", "Pressure", "Catalyst[Pt]", "Catalyst[Pd]", "Temperature*Pressure", "Temperature**2", "Pressure**2", "..."],
  "n_points_minimum": 9,
  "n_points_recommended": 18
}
```

### Generate design

```http
POST /api/v1/sessions/{session_id}/optimal-design
Content-Type: application/json

{
  "model_type": "quadratic",
  "p_multiplier": 2.0,
  "criterion": "D",
  "algorithm": "fedorov",
  "random_seed": 42
}
```

Custom effects:

```http
POST /api/v1/sessions/{session_id}/optimal-design
{
  "effects": ["Temperature", "Pressure", "Temperature*Pressure"],
  "n_points": 15,
  "criterion": "D",
  "algorithm": "fedorov"
}
```

---

## D-efficiency

The design response includes a **D-efficiency** percentage:

$$D\text{-eff} = \frac{100}{p} \cdot |\mathbf{X}'\mathbf{X}|^{1/p}$$

A D-efficiency of 100% is a theoretical ideal (rarely achieved in practice). Values above 90% are generally excellent; above 80% is acceptable.

---

## Limitations

!!! warning "Categorical variables (experimental)"
    Optimal design with categorical variables uses dummy coding (k−1 indicator columns per variable). The statistical properties are well-understood but designs may be harder to interpret than classical factorial methods. For categorical screening with well-established properties, consider [GSD or Full Factorial](doe_classical.md) instead.

- **Discrete variables** with exactly 2 allowed values cannot include quadratic terms.
- Very large candidate sets (many variables × many levels) may be slow. Reduce `n_levels` (default 5) to speed up.

---

For the conceptual background on optimality criteria and design matrices, see [Background: Design of Experiments](../background/doe_theory.md).
