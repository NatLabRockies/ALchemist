# Calibration Curve

The **calibration curve** in ALchemist measures whether your Gaussian Process model's predicted confidence intervals have the correct **coverage**. It answers the question: "When the model says a measurement is 90% likely to fall within an interval, does it actually fall within that interval 90% of the time?"

---

## What the Calibration Curve Shows

**X-axis**: Expected confidence level (0 to 1, or 0% to 100%)  
**Y-axis**: Observed coverage from cross-validation predictions

**Key elements:**

- **Blue curve**: Actual coverage at each confidence level

- **Diagonal line**: Perfect calibration reference (y = x)

- **Shaded regions**: 95% and 68% confidence bands (Clopper-Pearson)

- **Diagnostic text**: Summary statistics and calibration status

---

## Quick Interpretation Guide

| Pattern | Shape | Status | What It Means |
|---------|-------|--------|---------------|
| On diagonal | y ≈ x | ✓ Well-calibrated | Coverage matches expectations |
| Below diagonal | y < x | Over-confident | Intervals too narrow, poor coverage |
| Above diagonal | y > x | Under-confident | Intervals too wide, conservative |
| Matches band | Within shaded area | ✓ Acceptable | Within statistical uncertainty |

---

## Understanding Coverage Calibration

For each confidence level α (e.g., 0.90 for 90%), the coverage is:

$$
\text{Coverage}(\alpha) = \frac{1}{n}\sum_{i=1}^{n} \mathbb{1}\left[y_i \in [\hat{\mu}_i - z_{\alpha}\hat{\sigma}_i,\ \hat{\mu}_i + z_{\alpha}\hat{\sigma}_i]\right]
$$

Where:

- $y_i$ = true experimental value

- $\hat{\mu}_i$ = predicted mean from cross-validation

- $\hat{\sigma}_i$ = predicted standard deviation

- $z_{\alpha}$ = z-score for confidence level α (e.g., 1.96 for 95%)

- $\mathbb{1}[\cdot]$ = indicator function (1 if true, 0 if false)

**Perfect calibration**: Coverage(α) = α for all α ∈ [0, 1]

---

## When to Use the Calibration Curve

### Essential Situations

**Before making optimization decisions:**

- Verify confidence intervals are trustworthy

- Assess risk tolerance (safety-critical applications)

- Validate uncertainty-based acquisition functions

**After model training:**

- Check calibration across all confidence levels

- Compare different modeling backends

- Evaluate kernel choices

**During active learning:**

- Monitor if calibration degrades as data grows

- Ensure reliability of new predictions

- Detect if recalibration is needed

### Combined with Other Diagnostics

Use calibration curve alongside:

- **[Q-Q plot](qq_plot.md)**: Check z-score distribution (Std(z) ≈ 1)

- **[Parity plot](parity_plot.md)**: Assess prediction accuracy

- **Metrics plot**: Track overall performance

---

## Accessing the Calibration Curve

### In Web Application

1. Train a model in the GPR Panel
2. Click "Show Model Visualizations"
3. Select "Calibration Curve" from plot type buttons
4. Toggle between calibrated/uncalibrated results

### In Desktop Application

1. Train model in Model panel
2. Open Visualizations dialog
3. Calibration curve available in visualization options
4. Customize and export for reports

---

## Interpreting Calibration Patterns

### Perfect Calibration
```
Curve follows diagonal within confidence bands
All predicted confidence levels match observed coverage
```
**Example**: 95% intervals contain true value 94-96% of the time  
**Action**: Model is ready for optimization, no changes needed

### Over-Confident Model
```
Curve below diagonal across multiple confidence levels
Observed coverage < expected confidence level
```
**Example**: 90% intervals only contain true value 75% of the time  
**Impact**:

- Higher risk of missing optimal regions

- Acquisition functions overly exploitative

- May converge prematurely

**Actions**:

1. Use automatic calibration (enabled by default)
2. Increase noise parameter if applicable
3. Try more flexible kernel (Matern ν=1.5)
4. Collect more diverse training data

### Under-Confident Model
```
Curve above diagonal across multiple confidence levels
Observed coverage > expected confidence level
```
**Example**: 90% intervals contain true value 98% of the time  
**Impact**:

- Wasted experimental budget (over-exploration)

- Slower convergence to optimum

- Conservative but safer

**Actions**:

1. Often acceptable (conservative is safer than aggressive)
2. If inefficient, reduce explicit noise values
3. Try less flexible kernel (Matern ν=2.5, RBF)
4. Check that lengthscales aren't manually fixed too large

---

## Confidence Bands (Statistical Uncertainty)

The shaded regions show expected variability due to finite sample size.

### Clopper-Pearson Intervals

For each confidence level α with n samples and k successes:

$$
\text{Lower bound} = \text{Beta}^{-1}\left(\frac{\alpha_{\text{band}}}{2}; k, n-k+1\right)
$$

$$
\text{Upper bound} = \text{Beta}^{-1}\left(1 - \frac{\alpha_{\text{band}}}{2}; k+1, n-k\right)
$$

**Shaded regions**:

- Dark band: 68% confidence (≈1σ)

- Light band: 95% confidence (≈2σ)

**Interpretation**:

- If curve is **within bands**: Deviations likely due to chance

- If curve is **outside bands**: Genuine calibration issue

---

## Sample Size Considerations

### Small Datasets (N < 30)

- Very wide confidence bands

- High variability expected

- Difficult to distinguish poor calibration from sampling noise

- Focus on overall trend, don't over-interpret

### Medium Datasets (30 < N < 100)

- Moderate confidence bands

- Systematic deviations become detectable

- Curves outside 95% band indicate real issues

- Sufficient for calibration assessment

### Large Datasets (N > 100)

- Narrow confidence bands

- High confidence in calibration assessment

- Even small deviations from diagonal are meaningful

- Clear detection of calibration problems

---

## Automatic Calibration in ALchemist

When miscalibration is detected, ALchemist automatically applies correction:

### Calibration Method

1. Compute standardized residuals from cross-validation: $z_i = \frac{y_i - \hat{\mu}_i}{\hat{\sigma}_i}$
2. Calculate empirical standard deviation: $\text{Std}(z)$
3. Apply scaling to future predictions: $\sigma_{\text{calibrated}} = \sigma_{\text{raw}} \times \text{Std}(z)$

### Effect on Calibration Curve

**Over-confident (Std(z) > 1)**:

- Raw curve below diagonal

- Calibrated curve shifts upward toward diagonal

- Intervals widened by factor Std(z)

**Under-confident (Std(z) < 1)**:

- Raw curve above diagonal

- Calibrated curve shifts downward toward diagonal

- Intervals narrowed by factor Std(z)

### Verification

Toggle between calibrated/uncalibrated views to see:

- Raw model performance

- Impact of automatic correction

- Improvement in coverage

---

## Relationship to Q-Q Plot

Calibration curve and Q-Q plot are complementary:

| Diagnostic | What It Checks | Key Metric |
|------------|----------------|------------|
| **Q-Q Plot** | Distribution of z-scores | Std(z) ≈ 1.0 |
| **Calibration Curve** | Coverage at confidence levels | Coverage(α) ≈ α |

**Connection**:

- If Std(z) = 1.0 → Calibration curve should be near diagonal

- If Std(z) > 1.0 → Calibration curve below diagonal (over-confident)

- If Std(z) < 1.0 → Calibration curve above diagonal (under-confident)

**Why use both?**

- Q-Q plot: Global assessment, single metrics

- Calibration curve: Level-specific assessment, shows where issues occur

---

## Integration with Bayesian Optimization

Calibration directly impacts optimization efficiency:

### Expected Improvement (EI)

- Relies on correct σ for exploration/exploitation balance

- Poor calibration → suboptimal decisions

### Upper Confidence Bound (UCB)

- Formula: $\text{UCB} = \mu + \kappa \sigma$

- Miscalibrated σ → wrong balance between mean and uncertainty

### Safety-Constrained Optimization

- Often requires 95% or 99% confidence intervals

- Poor calibration at high confidence levels → safety violations or excessive conservatism

**Bottom line**: Well-calibrated intervals are critical for successful and safe optimization.

---

## Interpreting Specific Confidence Levels

### Low Confidence (50%-70%)
**Region**: Central part of curve  
**Importance**: Typical working range for acquisition functions  
**Good calibration here**: Essential for efficient exploration

### Medium Confidence (80%-90%)
**Region**: Upper-middle of curve  
**Importance**: Safety margins for constraints  
**Deviations**: Impact risk assessment in constrained optimization

### High Confidence (95%-99%)
**Region**: Far right of curve  
**Importance**: Critical for safety-critical applications  
**Statistical note**: Fewer samples at extremes, wider confidence bands

---

## Common Calibration Issues

### Issue: Curve Below Diagonal at All Levels
**Diagnosis**: Systematically over-confident  
**Root causes**:

- Insufficient training data diversity

- Overfitting to training data

- Noise parameter too small

- Overly complex kernel

**Solutions**:

1. Use automatic calibration
2. Collect more varied training data
3. Increase noise constraints
4. Simplify kernel or regularize hyperparameters

### Issue: Curve Above Diagonal at All Levels
**Diagnosis**: Systematically under-confident  
**Root causes**:

- Noise parameter too large

- Overly conservative kernel

- Lengthscales fixed too large

**Solutions**:

1. Assess if this is acceptable (conservative is safer)
2. Reduce explicit noise if set manually
3. Allow lengthscale optimization
4. Try less flexible kernel

### Issue: Good at Center, Poor at Extremes
**Diagnosis**: Non-uniform calibration  
**Root causes**:

- Sample size effects (fewer points at extremes)

- Non-Gaussian error distribution

- Heteroscedastic noise

**Solutions**:

1. Check if deviations are within confidence bands (may be statistical noise)
2. Try output transforms (log, Box-Cox)
3. Consider heteroscedastic GP if available
4. Collect more data to reduce uncertainty

### Issue: Sudden Jumps or Non-Monotonic Curve
**Diagnosis**: Small sample size or data artifacts  
**Root causes**:

- Insufficient cross-validation samples

- Outliers or data quality issues

- Too few unique predictions

**Solutions**:

1. Increase dataset size
2. Check for and address outliers
3. Verify data quality and preprocessing
4. Use smoothing or binning for visualization only

---

## Practical Guidelines

### Acceptable Calibration

**Strict (safety-critical)**:

- Curve within 68% confidence band at all levels

- Maximum deviation < 5% from diagonal

**Moderate (standard optimization)**:

- Curve within 95% confidence band at most levels

- Maximum deviation < 10% from diagonal

**Relaxed (exploratory)**:

- Overall trend near diagonal

- No systematic bias > 15%

### When to Recalibrate

**During active learning**:

- After adding 20-30% more data

- If acquisition functions seem unreliable

- When optimization stagnates unexpectedly

**After model changes**:

- Switching kernels or backends

- Changing hyperparameter constraints

- Applying new preprocessing

---

## Advanced Topics

### Coverage vs. Sharpness Trade-off

**Coverage**: Frequency of intervals containing true value  
**Sharpness**: Width of confidence intervals

**Ideal**: High coverage with narrow intervals  
**Trade-off**: Can always increase coverage by widening intervals, but this reduces utility

**ALchemist approach**:

1. Optimize model for best predictions (sharpness)
2. Apply calibration to ensure correct coverage
3. Balance achieved automatically

### Bayesian Confidence Intervals

GP predictions naturally provide Bayesian credible intervals:

$$
y_{\text{new}} \sim \mathcal{N}(\mu_*, \sigma_*^2)
$$

**Interpretation**:

- 95% credible interval: $[\mu_* - 1.96\sigma_*, \mu_* + 1.96\sigma_*]$

- Probability that true value is in interval (given model assumptions)

**Calibration check**: Do these intervals have frequentist coverage?

---

## Troubleshooting

If calibration is poor, ALchemist's automatic calibration (enabled by default) will adjust confidence intervals. For persistent issues, try different kernels (Matern ν=1.5, ν=2.5, RBF) or collect more diverse data. Check the [Q-Q plot](qq_plot.md) and [parity plot](parity_plot.md) for additional diagnostics.

---

## Further Reading

- [Interpreting Calibration Curves (Educational Guide)](../background/interpreting_calibration.md) - Comprehensive theory and examples
- [Q-Q Plot](qq_plot.md) - Complementary z-score distribution diagnostic
- [Parity Plot](parity_plot.md) - Prediction accuracy assessment
- [Model Performance](../modeling/performance.md) - Overall model quality guide

---

**Key Takeaway**: The calibration curve tells you whether you can trust your model's confidence intervals. Well-calibrated uncertainties enable confident decision-making and efficient Bayesian optimization.
