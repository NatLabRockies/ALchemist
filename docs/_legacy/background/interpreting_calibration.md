# Interpreting Calibration Curves for Uncertainty Assessment

A **calibration curve** (also called a reliability diagram) is a diagnostic tool that helps you evaluate whether your Gaussian Process model's predicted confidence intervals have the correct coverage. In ALchemist, the calibration curve complements the Q-Q plot to provide a comprehensive view of your model's uncertainty quality.

---

## What is Coverage Calibration?

When a Gaussian Process provides a prediction with uncertainty, it defines confidence intervals at various levels:

- **68% confidence interval**: μ ± 1σ should contain 68% of true values

- **95% confidence interval**: μ ± 1.96σ should contain 95% of true values

- **99% confidence interval**: μ ± 2.58σ should contain 99% of true values

**Well-calibrated coverage** means that the empirical (observed) coverage matches the nominal (claimed) coverage. For example, if your model claims 95% confidence, then 95% of experimental observations should actually fall within those bounds.

---

## Why Coverage Calibration Matters

Accurate coverage is essential for:

- **Reliable decision-making**: Knowing when predictions are trustworthy

- **Risk management**: Avoiding over-confident predictions in safety-critical applications

- **Efficient exploration**: Balancing exploration and exploitation in optimization

- **Experimental planning**: Determining when more data is needed vs. when to act on predictions

---

## Understanding the Calibration Curve

### Components of the Visualization

ALchemist's calibration curve display includes:

1. **Line plot**: Nominal coverage (x-axis) vs. empirical coverage (y-axis)
2. **Perfect calibration line**: Diagonal reference (y = x)
3. **Metrics table**: Coverage values at standard confidence levels
4. **Color-coded status**: Visual indicators for calibration quality

### Reading the Plot

**X-axis (Nominal Coverage)**: The confidence level claimed by the model (e.g., 0.68 = 68% confidence)

**Y-axis (Empirical Coverage)**: The actual fraction of observations that fall within the predicted intervals

**Perfect calibration**: Points lie on the diagonal line (y = x)

---

## Interpreting Coverage Patterns

### Well-Calibrated Model

**What it looks like:**

- Points closely follow the diagonal line

- Empirical ≈ Nominal at all confidence levels

- Status indicators show "Good" (green)

**Metrics example:**
```
Confidence   Nominal   Empirical   Status
68%          0.68      0.67        ✓ Good
95%          0.95      0.94        ✓ Good
99%          0.99      0.98        ✓ Good
99.7%        0.997     0.995       ✓ Good
```

**What it means:**

- Model uncertainties accurately reflect prediction errors

- Confidence intervals have correct coverage

- Safe to trust model predictions and uncertainties

- Acquisition functions will make optimal decisions

---

### Over-Confident Model

**What it looks like:**

- Curve **below** the diagonal line

- Empirical coverage < Nominal coverage

- Status indicators show "Under-conf" (orange/red)

**Metrics example:**
```
Confidence   Nominal   Empirical   Status
68%          0.68      0.55        Under-conf
95%          0.95      0.82        Under-conf
99%          0.99      0.91        Under-conf
99.7%        0.997     0.95        Under-conf
```

**What it means:**

- Model is **too confident** in its predictions

- Claimed "95% confidence" only captures 82% of observations

- Prediction intervals are too narrow

- Risk of missing optimal regions by over-exploiting

**Why it happens:**

- Model underestimates noise in the data

- Kernel is too restrictive (overfit to training data)

- Insufficient data for problem complexity

- Lengthscales too small (overfitting local variations)

**How to fix:**
1. **Increase noise parameter**: If using noise column, increase values
2. **Regularization**: Add explicit noise term to model
3. **Change kernel**: Try more flexible kernel (Matern ν=1.5 instead of ν=2.5)
4. **Collect more data**: Especially in high-variance regions
5. **Apply calibration**: ALchemist automatically applies calibration corrections
6. **Check data quality**: Look for outliers or measurement errors

---

### Under-Confident Model

**What it looks like:**

- Curve **above** the diagonal line

- Empirical coverage > Nominal coverage

- Status indicators show "Over-conf" (blue)

**Metrics example:**
```
Confidence   Nominal   Empirical   Status
68%          0.68      0.78        Over-conf
95%          0.95      0.99        Over-conf
99%          0.99      1.00        Over-conf
99.7%        0.997     1.00        Over-conf
```

**What it means:**

- Model is **too cautious** with its predictions

- Claimed "95% confidence" actually captures 99% of observations

- Prediction intervals are too wide

- Risk of over-exploring, wasting experiments on unnecessary regions

**Why it happens:**

- Model overestimates noise in the data

- Kernel is too flexible (underfit)

- Lengthscales too large (over-smoothing)

- Prior distributions too broad

**How to fix:**
1. **Decrease noise parameter**: Reduce explicit noise values
2. **Tighter kernel**: Try less flexible kernel (Matern ν=2.5 or RBF)
3. **Optimize hyperparameters**: Ensure lengthscales are optimized, not fixed too large
4. **Check preprocessing**: Ensure data is properly scaled
5. **More aggressive optimization**: Increase training iterations

**Note**: Under-confidence is generally less problematic than over-confidence, but wastes experimental resources.

---

### Mixed Calibration Issues 🔄

**What it looks like:**

- Curve crosses the diagonal line

- Some confidence levels over-confident, others under-confident

- Inconsistent status indicators

**Metrics example:**
```
Confidence   Nominal   Empirical   Status
68%          0.68      0.62        Under-conf
95%          0.95      0.96        Good
99%          0.99      1.00        Over-conf
99.7%        0.997     1.00        Over-conf
```

**What it means:**

- Uncertainty estimates have non-normal distribution

- May indicate model misspecification

- Different behavior in tails vs. center of distribution

**How to fix:**
1. **Check data distribution**: Look for outliers or bimodality
2. **Transform outputs**: Consider log or Box-Cox transformation
3. **Different kernel**: Experiment with alternative kernel families
4. **Stratified sampling**: Ensure training data covers full range

---

## Sample Size Considerations

### Small Datasets (N < 30) 🔍

- High variability in empirical coverage estimates

- Coverage metrics less reliable

- ±10% deviation from nominal is common

- Focus on overall trends rather than exact values

**Interpretation guidance:**
```
Empirical coverage of 0.85 for nominal 0.95 is acceptable with N=20
Same coverage would be concerning with N=100
```

### Medium Datasets (30 < N < 100)

- Moderate reliability in coverage estimates

- ±5% deviation becoming significant

- Clear patterns indicate real issues

### Large Datasets (N > 100) 🎯

- High confidence in calibration assessment

- Even small deviations (±3%) may indicate issues

- Coverage metrics are highly reliable

**ALchemist displays a warning for N < 30:**
```
Note: Small sample size (N=25). Coverage estimates may be unreliable.
```

---

## Calibration Status Indicators

ALchemist uses color-coded status indicators for quick assessment:

| Status | Color | Criterion | Interpretation |
|--------|-------|-----------|----------------|
| ✓ Good | Green | \|Empirical - Nominal\| < 0.05 | Well-calibrated |
| Under-conf | Orange | Empirical < Nominal - 0.05 | Too confident (narrow intervals) |
| Over-conf | Blue | Empirical > Nominal + 0.05 | Too cautious (wide intervals) |

These thresholds are adjustable based on sample size and application requirements.

---

## Using Calibration with Q-Q Plots

The calibration curve and [Q-Q plot](interpreting_qqplot.md) provide complementary information:

### Q-Q Plot Strengths:

- Tests normality assumption

- Detects bias (Mean(z) ≠ 0)

- Shows over/under-confidence via Std(z)

- Visual pattern recognition

### Calibration Curve Strengths:

- Quantifies coverage at specific confidence levels

- Easier to interpret numerically

- Less sensitive to distributional assumptions

- Direct link to decision-making thresholds

### Combined Analysis:

**Both good**: Model is well-calibrated and uncertainties are reliable

**Q-Q bad, Calibration good**: Non-normal errors but coverage is correct (acceptable for many applications)

**Q-Q good, Calibration bad**: Distribution is normal but variance is miscalibrated (systematic scaling issue)

**Both bad**: Significant model misspecification (investigate data and model choices)

---

## Practical Guidelines

### When to Trust Your Model

Proceed with confidence if:

- All coverage metrics within ±5% of nominal (for N > 30)

- Status indicators show "Good" or mild "Over-conf"

- Calibration curve closely follows diagonal

- Q-Q plot also shows good calibration

### When to Improve Calibration

Take corrective action if:

- Multiple "Under-conf" indicators (model too confident)

- Large deviations (>10%) from nominal coverage

- Consistent pattern across all confidence levels

- Sample size is adequate (N > 30) for reliable assessment

### When to Collect More Data

Consider more experiments if:

- Sample size is small (N < 30) with unclear patterns

- High variance in coverage estimates

- Model appears underfit (high RMSE, low R²)

- Coverage is acceptable but prediction accuracy is poor

---

## Calibration in Active Learning Context

During Bayesian optimization, calibration affects:

### Acquisition Function Performance:

- **Expected Improvement**: Relies on accurate σ for exploration/exploitation balance

- **UCB**: Directly uses σ in the acquisition formula

- **Probability of Improvement**: Needs correct uncertainty quantification

### Optimization Strategy:

- **Over-confident model**: May converge prematurely to local optima (too much exploitation)

- **Under-confident model**: May waste experiments exploring known regions (too much exploration)

- **Well-calibrated model**: Optimal balance, efficient convergence

### Stopping Criteria:

- Calibrated uncertainties help determine when optimization has converged

- Under-confident models may never reach stopping criteria

- Over-confident models may stop too early

---

## ALchemist's Automatic Calibration

ALchemist implements automatic uncertainty calibration:

1. **Cross-validation**: Computes z-scores from CV predictions
2. **Calibration factor**: Calculates `s = Std(z)` from residuals
3. **Scaling**: Multiplies predicted σ by calibration factor
4. **Application**: Automatically applied to future predictions

**Effect:**

- If Std(z) = 1.5 (over-confident), future σ predictions are scaled by 1.5×

- If Std(z) = 0.7 (under-confident), future σ predictions are scaled by 0.7×

- Brings model toward better calibration without retraining

**Toggle:**

- Calibrated vs. uncalibrated results viewable in visualization panel

- Compare to see calibration impact on your specific dataset

---

## Summary

| Pattern | Coverage vs. Diagonal | Issue | Primary Fix |
|---------|----------------------|-------|-------------|
| On diagonal | Aligned | ✓ None | - |
| Below diagonal | Empirical < Nominal | Over-confident | Increase noise/uncertainty |
| Above diagonal | Empirical > Nominal | Under-confident | Reduce noise, tighter kernel |
| Crosses diagonal | Mixed | Model misspecification | Check data, try different kernel |
| High scatter | Variable | Small sample | Collect more data |

---

## Further Reading

- [Q-Q Plot Interpretation](interpreting_qqplot.md) - Complementary diagnostic for normality

- [Model Performance](../modeling/performance.md) - Overall model quality assessment

- [Calibration Curve Visualization](../visualizations/calibration_curve.md) - How to generate and use the plot in ALchemist

For theoretical background on uncertainty calibration:

- Guo et al. (2017), "On Calibration of Modern Neural Networks"

- Kuleshov et al. (2018), "Accurate Uncertainties for Deep Learning Using Calibrated Regression"

- DeGroot & Fienberg (1983), "The Comparison and Evaluation of Forecasters"
