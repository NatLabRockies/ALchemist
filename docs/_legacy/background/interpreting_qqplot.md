# Interpreting Q-Q Plots for Uncertainty Calibration

A **Q-Q plot** (quantile-quantile plot) is a diagnostic tool for assessing whether the uncertainty estimates from your Gaussian Process model are well-calibrated. In ALchemist, the Q-Q plot helps you determine if your model's predicted uncertainties accurately reflect the true prediction errors.

---

## What is Uncertainty Calibration?

When a Gaussian Process predicts an output value, it also provides a measure of uncertainty (standard deviation). **Well-calibrated uncertainty** means that:

- If the model says there's a 68% chance the true value is within ±1σ, then approximately 68% of predictions should fall within that range

- If the model says there's a 95% chance the true value is within ±2σ, then approximately 95% should fall within that range

Calibration is critical for:

- **Decision-making**: Reliable uncertainties help determine when to trust predictions vs. run more experiments

- **Acquisition functions**: Methods like Expected Improvement and UCB rely on accurate uncertainty estimates

- **Risk assessment**: Understanding prediction confidence for safety-critical applications

---

## Understanding the Q-Q Plot

### What the Plot Shows

The Q-Q plot in ALchemist displays:

- **X-axis**: Theoretical quantiles from a standard normal distribution $\mathcal{N}(0, 1)$

- **Y-axis**: Standardized residuals (z-scores) from cross-validation predictions

- **Diagonal reference line**: Perfect calibration (y = x)

- **Confidence band** (for small samples, N < 100): Expected deviation range due to finite sample size

- **Diagnostic metrics**: Mean(z) and Std(z) displayed on the plot

### Standardized Residuals (Z-scores)

For each cross-validation prediction, the z-score is calculated as:

$$
z_i = \frac{y_i^{\text{true}} - y_i^{\text{pred}}}{\sigma_i}
$$

Where:

- $y_i^{\text{true}}$ = actual experimental value

- $y_i^{\text{pred}}$ = model prediction

- $\sigma_i$ = predicted standard deviation

If uncertainties are well-calibrated, these z-scores should follow a standard normal distribution $\mathcal{N}(0, 1)$.

---

## Interpreting the Plot

### Perfect Calibration 🎯

**What it looks like:**

- Points closely follow the diagonal line (y = x)

- Mean(z) ≈ 0.0

- Std(z) ≈ 1.0

- Points within confidence band

**What it means:**

- Model uncertainties accurately reflect prediction errors

- 68% of predictions within ±1σ, 95% within ±2σ (as expected)

- Acquisition functions will make optimal decisions

**Example:**
```
Mean(z) = 0.02
Std(z) = 0.98
Status: ✓ Well-calibrated
```

---

### Over-Confident Predictions

**What it looks like:**

- Points systematically **above** the diagonal line

- Std(z) > 1.0 (e.g., 1.5, 2.0, or higher)

- Residuals are larger than predicted uncertainties

**What it means:**

- Model is **too confident** in its predictions

- Actual errors are larger than the model thinks they are

- The model predicts σ = 2, but actual error is σ = 4

- Risk of over-exploiting regions that may not be optimal

**Why it happens:**

- Insufficient model complexity (kernel too simple)

- Underestimated noise in the data

- Not enough data for the problem complexity

- Overfitting to training data

**How to fix:**

- Try a more flexible kernel (e.g., Matern with lower ν)

- Increase model noise parameter (if using noise column)

- Collect more training data

- Apply uncertainty calibration (ALchemist does this automatically)

**Example:**
```
Mean(z) = -0.05
Std(z) = 1.45
Status: Over-confident (model uncertainties too small)
```

---

### Under-Confident Predictions

**What it looks like:**

- Points systematically **below** the diagonal line

- Std(z) < 1.0 (e.g., 0.6, 0.7, 0.8)

- Residuals are smaller than predicted uncertainties

**What it means:**

- Model is **too uncertain** about its predictions

- The model predicts σ = 4, but actual error is σ = 2

- Predictions are more accurate than the model believes

- Risk of over-exploring, wasting experiments on unnecessary regions

**Why it happens:**

- Model is overly conservative

- Noise parameter set too high

- Kernel lengthscales too large (oversmoothing)

- Small dataset with conservative priors

**How to fix:**

- Try a more restrictive kernel (e.g., Matern with higher ν)

- Reduce model noise parameter

- Optimize kernel hyperparameters more aggressively

- Collect more data to reduce inherent uncertainty

**Example:**
```
Mean(z) = 0.08
Std(z) = 0.72
Status: Under-confident (model uncertainties too large)
```

---

### Systematic Bias 🔴

**What it looks like:**

- Mean(z) significantly different from 0 (e.g., |Mean(z)| > 0.3)

- Points shifted up or down from the diagonal

- Consistent over- or under-prediction

**What it means:**

- Model has systematic bias in predictions

- Not just a calibration issue—predictions are consistently off

- Mean(z) > 0: Model consistently under-predicts

- Mean(z) < 0: Model consistently over-predicts

**How to fix:**

- Check data quality and units

- Try different kernel types

- Check for data preprocessing issues

- Ensure input/output transforms are appropriate

- Consider adding a mean function or trend

**Example:**
```
Mean(z) = 0.45
Std(z) = 1.02
Status: 🔴 Systematic bias (consistent under-prediction)
```

---

## Sample Size Considerations

### Small Datasets (N < 30)

- Expect more scatter around the diagonal

- Confidence bands are wider

- Std(z) can deviate from 1.0 more easily

- Don't over-interpret minor deviations

### Medium Datasets (30 < N < 100)

- Narrower confidence bands

- More reliable calibration assessment

- Moderate deviations indicate real issues

### Large Datasets (N > 100)

- Tight confidence bands

- High confidence in calibration assessment

- Even small deviations may indicate issues

---

## Practical Guidelines

### When to Worry 🚨

Take action if you see:

- Std(z) > 1.3 or < 0.7 (with N > 30)

- |Mean(z)| > 0.3

- Clear systematic pattern in deviations

- Points consistently outside confidence band

### When Not to Worry

Don't be concerned if:

- Minor scatter with Std(z) between 0.9 and 1.1

- Mean(z) between -0.1 and 0.1

- Points within confidence band (especially for N < 30)

- Random scatter without systematic pattern

---

## Relationship to Calibration Curve

The Q-Q plot and [Calibration Curve](interpreting_calibration.md) are complementary:

- **Q-Q Plot**: Tests if residuals follow normal distribution (are z-scores ~ N(0,1)?)

- **Calibration Curve**: Tests if confidence intervals have correct coverage (do 95% intervals contain 95% of points?)

Use both together for comprehensive uncertainty assessment:

- Q-Q plot reveals over/under-confidence and bias

- Calibration curve quantifies coverage at specific confidence levels

---

## Summary

| Observation | Mean(z) | Std(z) | Interpretation | Action |
|-------------|---------|--------|----------------|--------|
| Points on diagonal | ≈ 0 | ≈ 1.0 | ✓ Well-calibrated | None needed |
| Points above diagonal | ≈ 0 | > 1.0 | Over-confident | Increase uncertainty |
| Points below diagonal | ≈ 0 | < 1.0 | Under-confident | Reduce uncertainty |
| Points shifted up | > 0 | any | 🔴 Under-predicting | Check data/model |
| Points shifted down | < 0 | any | 🔴 Over-predicting | Check data/model |

---

## Further Reading

- [Calibration Curve Interpretation](interpreting_calibration.md) - Complementary calibration diagnostic

- [Model Performance](../modeling/performance.md) - General model assessment guidance

- [Metrics Evolution Plot](../visualizations/metrics_plot.md) - RMSE, MAE, R² interpretation and convergence tracking

For the mathematical foundations of uncertainty calibration in Gaussian Processes, see:

- Kuleshov et al. (2018), "Accurate Uncertainties for Deep Learning Using Calibrated Regression"

- Gneiting et al. (2007), "Strictly Proper Scoring Rules, Prediction, and Estimation"
