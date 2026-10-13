# Q-Q Plot

The **Q-Q plot** (quantile-quantile plot) in ALchemist is a specialized diagnostic tool that helps you assess whether your Gaussian Process model's uncertainty estimates are well-calibrated. It compares the distribution of standardized residuals from cross-validation against the theoretical normal distribution.

---

## What the Q-Q Plot Shows

**X-axis**: Theoretical quantiles from standard normal distribution $\mathcal{N}(0,1)$  
**Y-axis**: Observed standardized residuals (z-scores) from cross-validation predictions

**Key elements:**

- **Scatter points**: Each point represents one cross-validation prediction

- **Diagonal line**: Perfect calibration reference (y = x)

- **Confidence band**: Expected deviation range for finite samples (shown when N < 100)

- **Diagnostic text**: Mean(z) and Std(z) with calibration status

---

## Quick Interpretation Guide

| Pattern | Mean(z) | Std(z) | Status | What It Means |
|---------|---------|--------|--------|---------------|
| Points on diagonal | ≈0 | ≈1.0 | ✓ Well-calibrated | Uncertainties are accurate |
| Points above diagonal | ≈0 | >1.0 | Over-confident | Intervals too narrow |
| Points below diagonal | ≈0 | <1.0 | Under-confident | Intervals too wide |
| Shifted upward | >0 | any | 🔴 Under-predicting | Systematic bias |
| Shifted downward | <0 | any | 🔴 Over-predicting | Systematic bias |

---

## Understanding Standardized Residuals

For each cross-validation prediction, the z-score is:

$$
z_i = \frac{y_i^{\text{true}} - y_i^{\text{pred}}}{\sigma_i}
$$

Where:

- $y_i^{\text{true}}$ = actual experimental value

- $y_i^{\text{pred}}$ = model prediction

- $\sigma_i$ = predicted standard deviation

**If well-calibrated**: z-scores should follow $\mathcal{N}(0,1)$ distribution

---

## When to Use the Q-Q Plot

### Essential Situations

**Before optimization decisions:**

- Verify uncertainty estimates are reliable

- Check if acquisition functions can be trusted

- Assess risk of over-confident predictions

**After model training:**

- Initial calibration check

- Compare different backends (sklearn vs BoTorch)

- Evaluate impact of kernel choices

**During active learning:**

- Monitor calibration as data accumulates

- Detect if model becomes over/under-confident

- Ensure continued reliability

### Combined with Other Diagnostics

Use Q-Q plot alongside:

- **[Parity plot](parity_plot.md)**: Check prediction accuracy (R², RMSE)

- **[Calibration curve](../background/interpreting_calibration.md)**: Verify coverage at confidence levels

- **Metrics plot**: Monitor performance trends

---

## Accessing the Q-Q Plot

### In Web Application

1. Train a model in the GPR Panel
2. Click "Show Model Visualizations"
3. Select "Q-Q Plot" from plot type buttons
4. Toggle between calibrated/uncalibrated results

### In Desktop Application

1. Train model in Model panel
2. Open Visualizations dialog
3. Q-Q plot available in visualization options
4. Can customize and save for publications

---

## Interpreting Diagnostic Metrics

### Mean(z): Bias Assessment

$$
\text{Mean}(z) = \frac{1}{n}\sum_{i=1}^{n} z_i
$$

**Ideal**: Mean(z) ≈ 0 (within ±0.1)

**Problematic**:

- Mean(z) > 0.3: Model consistently under-predicts

- Mean(z) < -0.3: Model consistently over-predicts

**Actions**:

- Check data preprocessing and units

- Verify output transforms are appropriate

- Try different kernel or backend

- Investigate data quality issues

### Std(z): Calibration Assessment

$$
\text{Std}(z) = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(z_i - \bar{z})^2}
$$

**Ideal**: Std(z) ≈ 1.0 (within 0.9-1.1)

**Over-confident** (Std(z) > 1.1):

- Model uncertainties too small

- Actual errors larger than predicted

- Risk of over-exploitation in optimization

**Under-confident** (Std(z) < 0.9):

- Model uncertainties too large

- Actual errors smaller than predicted

- Risk of over-exploration, wasted experiments

---

## Calibration Status Messages

ALchemist automatically interprets Q-Q plot results:

### ✓ Well-Calibrated
```
Mean(z) = 0.02, Std(z) = 0.98
Status: ✓ Well-calibrated uncertainties
```
**Action**: None needed, model is ready for optimization

### Over-Confident
```
Mean(z) = -0.05, Std(z) = 1.45
Status: Over-confident (model uncertainties too small)
```
**Actions**:

- Apply automatic calibration (built-in)

- Increase noise parameter

- Try more flexible kernel (Matern ν=1.5)

- Collect more training data

### Under-Confident
```
Mean(z) = 0.08, Std(z) = 0.72
Status: Under-confident (model uncertainties too large)
```
**Actions**:

- May be acceptable (conservative is safe)

- Try less flexible kernel (Matern ν=2.5, RBF)

- Reduce explicit noise values

- Optimize kernel hyperparameters more aggressively

### 🔴 Systematic Bias
```
Mean(z) = 0.45, Std(z) = 1.02
Status: 🔴 Systematic bias (consistent under-prediction)
```
**Actions**:

- Critical issue requiring attention

- Check data units and scaling

- Verify preprocessing steps

- Consider different kernel family

- Investigate data quality

---

## Sample Size Considerations

### Small Datasets (N < 30)

- High variability expected

- Wider confidence bands

- Don't over-interpret moderate deviations

- Focus on overall trend rather than exact values

### Medium Datasets (30 < N < 100)

- Moderate reliability

- Confidence bands still shown

- Deviations >0.2 in Std(z) indicate issues

- Patterns become meaningful

### Large Datasets (N > 100)

- High confidence in assessment

- No confidence bands (not needed)

- Even small deviations meaningful

- Std(z) should be within 0.95-1.05

---

## Automatic Calibration in ALchemist

When miscalibration is detected, ALchemist automatically applies correction:

**Calibration Process**:

1. Calculate Std(z) from cross-validation
2. Use as scaling factor: $\sigma_{\text{calibrated}} = \sigma_{\text{raw}} \times \text{Std}(z)$
3. Apply to future predictions

**Effect**:

- Std(z) = 1.5 → Future uncertainties scaled up 1.5×

- Std(z) = 0.7 → Future uncertainties scaled down 0.7×

- Brings model toward better calibration

**Toggle**:

- Compare calibrated vs uncalibrated in visualization panel

- See immediate impact of calibration

- Verify improvement in Q-Q plot

---

## Common Patterns and Solutions

### Pattern: S-curve (Sigmoid Shape)
**What it means**: Heavier tails than normal distribution  
**Actions**: Check for outliers, consider robust scaling

### Pattern: Points Fan Out at Extremes
**What it means**: Heteroscedastic errors (variance changes)  
**Actions**: Try log transform on outputs, check data range

### Pattern: Multiple Clusters
**What it means**: Multiple modes or subpopulations  
**Actions**: Check for categorical effects, investigate data stratification

### Pattern: Systematic Curve but Std(z) ≈ 1
**What it means**: Non-normal but correct variance  
**Actions**: Usually acceptable, functional form is more important

---

## Integration with Bayesian Optimization

Q-Q plot calibration directly impacts optimization:

### Expected Improvement (EI)

- Relies on σ for exploration/exploitation balance

- Over-confident → premature convergence

- Under-confident → excessive exploration

### Upper Confidence Bound (UCB)

- Uses σ directly in formula: UCB = μ + κσ

- Miscalibration affects all decisions

- Calibrated σ ensures optimal trade-off

### Probability of Improvement (PI)

- Depends on σ for probability calculation

- Correct calibration critical for thresholds

**Bottom line**: Well-calibrated uncertainty is essential for efficient optimization.

---

## Troubleshooting

ALchemist's automatic calibration (enabled by default) handles most calibration issues. For over-confident models (Std(z) > 1.3), try a more flexible kernel like Matern ν=1.5. For under-confident models (Std(z) < 0.7), this is often acceptable as it's conservative. If Mean(z) shows significant bias, check the [parity plot](parity_plot.md) for systematic patterns.

---

## Further Reading

- [Interpreting Q-Q Plots (Educational Guide)](../background/interpreting_qqplot.md) - Comprehensive theory and examples
- [Calibration Curve](../background/interpreting_calibration.md) - Complementary coverage diagnostic
- [Parity Plot](parity_plot.md) - Prediction accuracy assessment
- [Model Performance](../modeling/performance.md) - Overall model quality guide

---

**Key Takeaway**: The Q-Q plot reveals whether your model "knows what it doesn't know." Well-calibrated uncertainty is as important as accurate predictions for successful Bayesian optimization.
