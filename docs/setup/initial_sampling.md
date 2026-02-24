# Generating Initial Experiments

When starting an active learning workflow, it's important to generate an initial set of experiments that cover your variable space efficiently. ALchemist provides three families of design methods:

- **Space-filling** — exploratory coverage with no model assumptions (LHS, Sobol, etc.)
- **Classical RSM** — structured designs for polynomial model fitting (CCD, Box-Behnken, etc.)
- **Optimal** — user-specified model terms, statistically efficient run selection

---

## Why Generate Initial Points?

- **No Prior Data:** If you are starting from scratch, you need a set of initial experiments to train your first surrogate model.

- **Supplement Existing Data:** If you have some data, but it is sparse or not well-distributed, you can generate additional points to improve coverage.

- **Efficient Model Convergence:** Good initial coverage of the variable space helps the model learn faster and reduces the risk of missing important regions.

---

## Space-Filling Methods (Recommended for Exploratory BO)

Space-filling methods distribute points uniformly across the variable space with no assumptions about the response surface. They are the best default choice when you don't have prior knowledge about which model terms matter.

**Available methods:**
- **Random:** Uniformly samples points at random.
- **LHS (Latin Hypercube Sampling):** Ensures each variable is sampled evenly across its range. Recommended default.
- **Sobol, Halton, Hammersly:** Quasi-random low-discrepancy sequences for more uniform coverage in high dimensions.

**How to generate:**

1. Define your variable space.
2. Open the **Initial Design** panel and click **Generate Initial Points**.
3. Choose a space-filling method.
4. Set the **Number of Points**.
5. Click **Generate**.

**LHS criterion options:**
- `maximin` (default) — maximizes minimum pairwise distance between points
- `correlation` — minimizes correlation between columns
- `ratio` — balances between the above

---

## Classical & Screening Designs

For structured experimental programs following Response Surface Methodology (RSM), ALchemist supports Full Factorial, Fractional Factorial, Central Composite Design (CCD), Box-Behnken, Plackett-Burman, and Generalized Subset Design (GSD).

These methods determine run count automatically from the design structure — you do not specify a number of points.

> See [Classical & Screening Designs](doe_classical.md) for full documentation.

---

## Optimal Experimental Design

Optimal designs let you specify which model terms to estimate (main effects, interactions, quadratic terms) and generate the most information-rich set of runs for that specific model.

The **Optimal Design** tab in the Initial Design panel provides a UI for:
- Choosing model terms via shortcut (linear/interaction/quadratic) or checkboxes
- AI-assisted effect suggestion (OpenAI, Ollama, or Edison Scientific)
- Criterion selection (D/A/I-optimal) and algorithm selection
- Run count as a multiple of model complexity (`p_multiplier`)

> See [Optimal Experimental Design](doe_optimal.md) for full documentation.

> See [AI-Assisted Effect Selection](llm_effects.md) to use LLMs for model term recommendations.

---

## Choosing a Method

| Situation | Recommended method |
|---|---|
| No prior knowledge, exploratory BO | LHS |
| Fitting a response surface, expecting curvature | CCD |
| Many factors (5+), need to screen | Plackett-Burman or Fractional Factorial |
| Mixed categorical + continuous | GSD or Full Factorial |
| Know which effects matter, want efficient design | Optimal Design |
| High dimensions (4+), quasi-random coverage | Sobol |

---

## Tips

- **Coverage Matters:** More points give better coverage, but also require more experiments. Balance your resources and modeling needs.
- **Quasi-Random vs. Random:** Quasi-random methods (LHS, Sobol, etc.) are generally preferred for initial sampling, especially in higher dimensions.
- **Classical designs and n_points:** Classical designs ignore `n_points` — the run count is fixed by the design structure.
- **Supplementing Data:** You can generate initial points even if you already have some data, to fill gaps or improve distribution.

---

For more details on experiment management and data loading, see the next section of the workflow documentation.