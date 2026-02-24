# Classical & Screening Designs

In addition to space-filling methods (LHS, Sobol, etc.), ALchemist supports a full suite of **classical Response Surface Methodology (RSM)** designs and **screening designs**. These structured methods follow decades of established statistical practice and are often preferred when variable effects are expected to follow polynomial trends or when you need to screen many factors with minimal runs.

---

## When to Use Classical Designs

| Situation | Recommended design |
|---|---|
| Need to estimate main effects + interactions/curvature | CCD or Box-Behnken |
| Maximum coverage of all factor combinations | Full Factorial |
| Many factors (≥5), need to screen efficiently | Plackett-Burman or Fractional Factorial |
| Mixed categorical + continuous factors, efficient screening | GSD |
| Want maximum statistical freedom (custom model) | [Optimal Design](doe_optimal.md) |
| Exploratory, no prior model structure assumed | [Space-filling](initial_sampling.md) |

---

## Method Overview

### Full Factorial

Evaluates **every combination** of factor levels. Provides complete information about all main effects and interactions.

- **Run count:** `n_levels^k` where `k` = number of continuous variables
- **Use when:** You have few factors (2–4) and need complete information
- **Supports:** Continuous and integer variables; categorical variables use their full category set
- **Key parameter:** `n_levels` — number of evenly-spaced levels per factor (default 2; max 5)

**Example (2 variables, 3 levels):** 3² = 9 runs at all combinations of (low, mid, high) × (low, mid, high)

---

### Fractional Factorial

A **subset of a full factorial** selected using generator strings. Reduces run count at the cost of some higher-order interaction estimates, which are "aliased" (confounded) with main effects or lower-order interactions.

- **Run count:** `2^(k-p)` where `p` determines the fraction (automatically selected)
- **Use when:** 3–7 continuous factors, want to screen main effects efficiently
- **Default generators:** Built-in Resolution III+ generators for 3–7 factors
- **Custom generators:** Specify your own via the `generators` parameter (e.g., `"a b ab"`)
- **Requires:** Continuous and integer variables only; no categorical

**Resolution** describes which effects are confounded:
- **Resolution III:** Main effects aliased with two-factor interactions
- **Resolution IV:** Main effects clear of two-factor interactions
- **Resolution V+:** Two-factor interactions estimable

---

### Central Composite Design (CCD)

The most common Response Surface Methodology design. Combines a **2-level factorial core** with **axial (star) points** and **center replicates** to estimate curvature (quadratic terms).

- **Run count:** `2^k + 2k + n_center` where `k` = number of continuous variables
- **Use when:** 2–5 continuous factors, anticipate curvature, need to fit a full quadratic model
- **Requires:** Continuous and integer variables only; no categorical

**Key parameters:**

| Parameter | Options | Description |
|---|---|---|
| `ccd_alpha` | `orthogonal` (default), `rotatable` | Positions axial points for orthogonality or equal prediction variance |
| `ccd_face` | `circumscribed` (default), `inscribed`, `faced` | Where axial points sit relative to factorial points |
| `n_center` | integer (default 1) | Center point replicates (helps estimate pure error) |

- **Circumscribed (CCC):** Axial points extend outside the factorial cube — gives better prediction but requires a wider variable range
- **Inscribed (CCI):** Keeps all points within the original bounds — useful when extremes are infeasible
- **Faced (CCF):** Axial points at ±1 on each axis — cube-shaped region, no extension

---

### Box-Behnken

A **spherical design** that avoids corner points. Runs are placed at edge midpoints and center, forming a roughly spherical design region.

- **Run count:** Typically ~3k² - 3k + 3 (varies by k)
- **Use when:** 3–5 continuous factors, extremes (corner combinations) are physically infeasible or dangerous
- **Requires:** Exactly 3+ continuous variables; no categorical or integer variables
- **Does not require extreme corners**, making it safer for processes where simultaneous extreme conditions are problematic

---

### Plackett-Burman

An ultra-efficient **2-level screening design** with run counts that are multiples of 4 (e.g., 12, 20, 24 runs). Designed to estimate main effects with maximum efficiency given the run count.

- **Run count:** Next multiple of 4 ≥ k+1 (e.g., 5 factors → 8 runs)
- **Use when:** Many continuous factors (5–20), want to quickly screen which matter
- **Requires:** Continuous and integer variables only; no categorical
- **Does not estimate interactions** — complex aliasing structure

---

### Generalized Subset Design (GSD)

A **fractional design for mixed or multi-level factors**, including categorical. A GSD with reduction factor `r` evaluates approximately `1/r` of the full factorial.

- **Run count:** `total_combinations / gsd_reduction` (approximately)
- **Use when:** Mixed categorical + continuous factors, full factorial is too expensive
- **Supports:** Categorical, continuous, and integer variables
- **Key parameter:** `gsd_reduction` — reduction factor (2 = ~half of full factorial; default 2)

> **Note:** GSD is the recommended classical screening method when your variable space includes categorical variables.

---

## Variable Type Compatibility

| Method | Real | Integer | Categorical | Discrete |
|---|---|---|---|---|
| Full Factorial | ✅ | ✅ | ✅ | ✅ |
| Fractional Factorial | ✅ | ✅ | ❌ | ❌ |
| CCD | ✅ | ✅ | ❌ | ❌ |
| Box-Behnken | ✅ | ❌ | ❌ | ❌ |
| Plackett-Burman | ✅ | ✅ | ❌ | ❌ |
| GSD | ✅ | ✅ | ✅ | ✅ |

> If your design includes incompatible variable types, ALchemist will raise an informative error. Use GSD or Full Factorial for mixed-type spaces.

---

## Web UI Walkthrough

1. Define your variable space and load the **Initial Design** panel.
2. Select a method from the **Design Method** dropdown (e.g., *CCD*).
3. Adjust method-specific parameters that appear below the dropdown.
4. Click **Generate** — the run count is determined automatically from the design structure.
5. Review the design table, then click **Stage Suggestions** to queue them for execution.

> Unlike space-filling methods, classical designs do **not** require you to specify a number of points — the run count is determined by the design structure.

---

## Python API

```python
from alchemist_core import OptimizationSession

session = OptimizationSession()
session.add_variable("Temperature", "real", min=200, max=350)
session.add_variable("Pressure", "real", min=1, max=10)
session.add_variable("Catalyst", "categorical", categories=["Ni", "Pt", "Pd"])

# Central Composite Design (continuous variables only)
ccd_points = session.generate_initial_design(
    method="ccd",
    ccd_alpha="orthogonal",
    ccd_face="circumscribed",
    n_center=1,
)

# Generalized Subset Design (supports categorical)
gsd_points = session.generate_initial_design(
    method="gsd",
    gsd_reduction=2,
)

# Fractional factorial with default generators
ff_points = session.generate_initial_design(
    method="fractional_factorial",
)
```

---

## REST API

```http
POST /api/v1/sessions/{session_id}/initial-design
Content-Type: application/json

{
  "method": "ccd",
  "ccd_alpha": "orthogonal",
  "ccd_face": "circumscribed",
  "n_center": 1
}
```

```http
POST /api/v1/sessions/{session_id}/initial-design
{
  "method": "gsd",
  "gsd_reduction": 2
}
```

---

## Comparison: Space-filling vs. Classical

| Property | Space-filling (LHS, Sobol) | Classical (CCD, Box-Behnken) |
|---|---|---|
| **Run count** | User-specified | Fixed by design structure |
| **Variable types** | All | Restricted (most exclude categoricals) |
| **Statistical model** | Agnostic | Polynomial (RSM) |
| **Best for** | Exploratory BO | Confirmatory RSM, curvature estimation |
| **Requires prior knowledge** | No | Somewhat (know which model to fit) |

---

For designs where you specify exactly which model terms to estimate, see [Optimal Experimental Design](doe_optimal.md).
