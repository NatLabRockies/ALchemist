# Background: Design of Experiments

Design of Experiments (DoE) is the systematic planning of experiments to efficiently extract information about a process. Rather than changing one variable at a time (OVAT), DoE methods vary multiple inputs simultaneously in structured patterns, enabling the estimation of main effects, interactions, and curvature with far fewer experiments.

---

## Why DoE Matters for Active Learning

Active learning begins with an **initial dataset**. The quality of that initial dataset directly affects:

- **Model convergence speed** — well-spread initial data trains more accurate surrogates with fewer experiments
- **Bias avoidance** — clustered or correlated initial points create blind spots in the model
- **Estimability** — if you intend to fit a specific model (e.g., quadratic RSM), the initial design must support all model terms

A good initial design reduces the number of total experiments needed to find the optimum.

---

## Coded vs. Actual Variables

Most DoE literature works in **coded space** — a normalized representation where each variable ranges from −1 (low) to +1 (high). ALchemist automatically converts between coded and actual variable values, so you always work in physical units.

**Benefit of coding:** Allows designs from different variable ranges to be compared and combined consistently.

---

## Design Matrix

The **design matrix** $\mathbf{X}$ is an $n \times p$ matrix where each row is an experimental run and each column is a model term (intercept, main effects, interactions, quadratic terms, and dummy-coded categorical indicators). Optimal design algorithms work directly on this matrix.

---

## Space-Filling vs. Classical vs. Optimal

| Property | Space-Filling | Classical RSM | Optimal |
|---|---|---|---|
| **Prior model structure needed?** | No | Implicit (polynomial) | Yes (explicit) |
| **Run count** | User-specified | Fixed by design rules | User-controlled |
| **Supports categoricals** | Yes | GSD/Full Factorial only | Yes (experimental) |
| **Interaction estimation** | Indirect | Built-in for RSM | Explicit |
| **When to prefer** | Exploratory BO | Known RSM context | Known model, custom design |

**Space-filling** methods (LHS, Sobol) make no assumptions about the response surface — they maximize coverage of the variable space and are ideal for exploratory Bayesian optimization.

**Classical RSM methods** (CCD, Box-Behnken) are designed for fitting second-order polynomial response surfaces and are the standard in chemistry and chemical engineering for process optimization.

**Optimal designs** generalize RSM by letting you specify exactly which terms to estimate, then finding the run configuration that estimates those terms most precisely.

---

## Optimality Criteria

### D-optimality

Maximizes the determinant of the information matrix $|\mathbf{X}'\mathbf{X}|$:

$$\text{maximize } |\mathbf{X}'\mathbf{X}|$$

This minimizes the volume of the joint confidence ellipsoid for all parameters — the best criterion when your goal is precise **parameter estimation**.

### A-optimality

Minimizes the trace of the inverse information matrix:

$$\text{minimize } \text{tr}[(\mathbf{X}'\mathbf{X})^{-1}]$$

This minimizes the **average variance** of parameter estimates. Appropriate when all parameters are of equal interest and you want to minimize their average uncertainty.

### I-optimality

Minimizes the average prediction variance integrated over the design region:

$$\text{minimize } \int_{\mathcal{X}} \text{Var}[\hat{y}(\mathbf{x})] \, d\mathbf{x}$$

Appropriate when the goal is **accurate prediction** across the entire variable space, rather than precise parameter estimation. Often preferred for response surface prediction.

---

## Exchange Algorithms

Optimal designs cannot be computed analytically in general. Instead, **exchange algorithms** iteratively improve a starting design by swapping candidate points in and out:

1. Start with a random set of $n$ candidate points
2. Try replacing each point with every other candidate
3. Accept swaps that improve the criterion
4. Repeat until no swap improves the criterion (local optimum)

**Fedorov's algorithm** performs full pairwise exchanges and is the most widely used; **DetMax** adds random "excursions" to escape local optima at the cost of more compute.

---

## Fractional Factorials and Aliasing

A fractional factorial design uses a **fraction** of the full factorial. The trade-off is **aliasing**: some higher-order effects are statistically confounded with lower-order effects. The **resolution** of a design describes the severity of this confounding:

| Resolution | Meaning |
|---|---|
| III | Main effects aliased with 2-factor interactions |
| IV | Main effects clear; 2FIs aliased with other 2FIs |
| V | Main effects and 2FIs all estimable |

For screening (identifying which factors matter), Resolution III is often acceptable. For estimating interactions, Resolution IV or higher is required.

---

## Further Reading

- Montgomery, D. C. (2017). *Design and Analysis of Experiments* (9th ed.). Wiley.
- Atkinson, A. C., & Donev, A. N. (1992). *Optimum Experimental Designs*. Oxford University Press.
- Fedorov, V. V. (1972). *Theory of Optimal Experiments*. Academic Press.
- Box, G. E. P., Hunter, J. S., & Hunter, W. G. (2005). *Statistics for Experimenters* (2nd ed.). Wiley.
