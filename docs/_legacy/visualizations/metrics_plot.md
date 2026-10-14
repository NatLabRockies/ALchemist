# Metrics Evolution Plot

The **metrics evolution plot** in ALchemist tracks how your Gaussian Process model's predictive performance changes as you collect more experimental data during the active learning loop. It provides a visual record of model improvement and helps you decide when to stop optimization.

---

## What the Metrics Plot Shows

**X-axis**: Number of observations (training data points)  
**Y-axis**: Performance metric value(s)

**Key elements:**

- **Line plot(s)**: Evolution of selected metric(s) over data collection

- **Metric options**: RMSE, MAE, MAPE, R²

- **Multiple metrics**: Can display several metrics simultaneously

- **Trends**: Visual indication of convergence or degradation

---

## Available Metrics

### Root Mean Squared Error (RMSE)

$$
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$

**Interpretation**:

- Units match your response variable

- Lower is better (0 = perfect predictions)

- Penalizes large errors more than small ones

- Most common metric for regression

**Typical values**:

- RMSE < 5% of response range: Excellent

- RMSE 5-10% of response range: Good

- RMSE > 20% of response range: Poor or insufficient data

### Mean Absolute Error (MAE)

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
$$

**Interpretation**:

- Units match your response variable

- Lower is better (0 = perfect predictions)

- Less sensitive to outliers than RMSE

- Average magnitude of errors

**Comparison to RMSE**:

- MAE < RMSE always (equality only if all errors identical)

- RMSE/MAE ratio indicates error distribution

- Ratio ≈ 1: Uniform errors

- Ratio > 1.5: Some large errors (outliers)

### Mean Absolute Percentage Error (MAPE)

$$
\text{MAPE} = \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|
$$

**Interpretation**:

- Percentage units (scale-independent)

- Lower is better (0% = perfect predictions)

- Useful for comparing across different response ranges

- Can be unstable if $y_i$ near zero

**Typical values**:

- MAPE < 5%: Excellent

- MAPE 5-10%: Good

- MAPE 10-20%: Acceptable

- MAPE > 20%: Poor

**Warning**: Undefined if any true value is exactly zero. ALchemist skips MAPE calculation in this case.

### Coefficient of Determination (R²)

$$
R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}
$$

Where $\bar{y}$ is the mean of observed values.

**Interpretation**:

- Dimensionless (0 to 1 for good models)

- Higher is better (1 = perfect predictions)

- Proportion of variance explained by model

- Can be negative for very poor models

**Typical values**:

- R² > 0.95: Excellent

- R² 0.90-0.95: Good

- R² 0.80-0.90: Acceptable

- R² < 0.80: Poor or insufficient data

**Note**: R² from cross-validation (ALchemist default) is more reliable than training R².

---

## When to Use the Metrics Plot

### During Active Learning

**Essential for**:

- Monitoring model improvement as data accumulates

- Deciding when to stop optimization

- Detecting convergence or plateaus

- Identifying data quality issues

**Check metrics plot**:

- After each batch of experiments

- Before requesting new candidates

- When considering stopping criteria

### Comparing Models

**Use metrics plot to**:

- Compare different kernels (Matern ν=1.5 vs RBF)

- Evaluate different backends (sklearn vs BoTorch)

- Test impact of transforms

- Assess hyperparameter choices

### Diagnosing Issues

**Metrics plot reveals**:

- Degrading performance (possible overfitting)

- Stagnant metrics (plateau reached)

- Erratic behavior (data quality problems)

- Unexpected trends (check preprocessing)

---

## Accessing the Metrics Plot

### In Web Application

1. Train a model in the GPR Panel
2. Click "Show Model Visualizations"
3. Select "Metrics Plot" from plot type buttons
4. Choose which metrics to display (checkboxes)

### In Desktop Application

1. Train model in Model panel
2. Open Visualizations dialog
3. Metrics evolution available in plot options
4. Customize display and export

---

## Interpreting Common Patterns

### Ideal Pattern: Steady Improvement

```
Metrics improve (RMSE/MAE/MAPE decrease, R² increases) as data accumulates
Rate of improvement slows as model converges
Eventually plateaus at acceptable performance
```

**What this means**: Active learning is working as expected  
**Action**: Continue until plateau, then stop optimization

**Example**:

- Start: RMSE = 15, R² = 0.60 (10 samples)

- Mid: RMSE = 8, R² = 0.88 (25 samples)

- End: RMSE = 5, R² = 0.94 (40 samples) → Plateau

### Warning Pattern: Degrading Performance

```
Metrics worsen (RMSE/MAE increase, R² decreases) as data accumulates
Performance peaks early then declines
May indicate overfitting or data issues
```

**What this means**: Problem with data quality or model  
**Actions**:

1. Check for outliers in recent data
2. Verify experimental measurements
3. Inspect data preprocessing
4. Try different kernel or regularization
5. Check if hyperparameter optimization failing

### Warning Pattern: No Improvement

```
Metrics flat or erratic across all data sizes
No clear trend of improvement
High variance between evaluations
```

**What this means**: Model not learning from data  
**Actions**:

1. Verify variable space covers response range
2. Check data is actually varying (not constant)
3. Ensure preprocessing appropriate
4. Try different kernel family
5. Inspect initial sampling distribution

### Expected Pattern: Early Volatility

```
Metrics fluctuate significantly with very few samples (< 15)
Behavior stabilizes as data accumulates
Trends become clear after 20-30 samples
```

**What this means**: Normal statistical noise with small samples  
**Action**: Don't over-interpret early behavior, wait for more data

---

## Deciding When to Stop

### Convergence Criteria

**Metrics-based**:

- RMSE/MAE plateau (< 5% change over 10 samples)

- R² > target threshold (e.g., 0.90 or 0.95)

- Absolute performance acceptable for application

**Practical**:

- Budget exhausted (time, cost, materials)

- Acceptable optimum found (target performance reached)

- Diminishing returns (effort exceeds benefit)

### Common Stopping Rules

**Conservative**:

- R² > 0.95 AND no improvement in last 15 samples

- RMSE < 2% of response range for 20 consecutive samples

- Validation metrics stable across 3 CV folds

**Moderate**:

- R² > 0.90 AND plateau for 10 samples

- RMSE < 5% of response range

- Acquisition function values < threshold

**Aggressive**:

- R² > 0.85 achieved

- RMSE better than baseline/literature

- Optimization objective reached

---

## Understanding Cross-Validation Metrics

### K-Fold Cross-Validation

ALchemist uses k-fold cross-validation (default: 5 folds) for all datasets:

**Process**:

1. Split data into k groups (typically 5)
2. For each group:
   - Train on other k-1 groups
   - Predict on held-out group
3. Aggregate predictions across all folds

**Advantages**:

- Good balance of bias/variance

- Computationally efficient

- Standard practice in machine learning

- Reliable estimates across dataset sizes

**Interpretation**:

- Reflects generalization to new data

- More pessimistic than training metrics

- More realistic for decision-making

---

## Metric Selection Guidelines

### Use RMSE when:

-  Response units meaningful (e.g., temperature °C, yield %)

-  Large errors particularly problematic

-  Standard metric expected in your field

-  Comparing models on same response

### Use MAE when:

-  Outliers present (more robust)

-  All errors treated equally important

-  Easier interpretation needed (average error)

-  RMSE vs MAE comparison informative (error distribution)

### Use MAPE when:

-  Comparing across different response scales

-  Percentage errors more interpretable

-  Response values far from zero (avoid division issues)

-  Scale-independent comparison needed

### Use R² when:

-  Variance explanation important

-  Comparing to baseline (R² = 0)

-  Standard metric in your field (common in chemistry/materials)

-  Want single dimensionless metric

### Display multiple metrics when:

-  Want comprehensive view

-  Different stakeholders prefer different metrics

-  Checking consistency across metrics

-  Diagnosing specific issues

---

## Integration with Optimization

Metrics evolution informs optimization strategy:

### Acquisition Function Choice

**Poor metrics (R² < 0.75)**:

- Favor exploration (UCB with high κ)

- Collect diverse data first

- Consider space-filling designs

**Good metrics (R² > 0.90)**:

- Allow exploitation (EI, PI)

- Trust model predictions

- Focus on promising regions

### Batch Size Decisions

**Rapidly improving metrics**:

- Smaller batches (adapt quickly)

- Re-train frequently

- Stay responsive to learning

**Plateaued metrics**:

- Larger batches acceptable

- Less frequent re-training

- Efficiency over responsiveness

### Stopping Criteria

**Metrics-driven stopping**:

- Set R² or RMSE threshold

- Monitor plateau duration

- Balance performance vs cost

---

## Practical Tips

### Displaying Multiple Metrics

**Recommended combinations**:

- **Standard**: RMSE + R² (accuracy and variance explained)

- **Comprehensive**: RMSE + MAE + R² (multiple perspectives)

- **Scale-independent**: MAPE + R² (for broad comparisons)

**Avoid**:

- Too many metrics (cluttered plot)

- Redundant pairs (RMSE + MAE without reason)

### Interpreting Trends

**Smooth trends**: Good model stability  
**Erratic jumps**: Check data quality or hyperparameter optimization  
**Sudden drops**: Possible outlier added or CV issue  
**Linear improvement**: Still learning, more data beneficial

### Comparing Sessions

**When comparing**:

- Use same metrics across sessions

- Account for different data sizes

- Consider response scale differences

- Check cross-validation method consistency

---

## Troubleshooting

If metrics aren't improving, check the [parity plot](parity_plot.md) for systematic bias and try different kernels. If metrics worsen, inspect recent data for outliers. High variance is expected with small datasets (n < 25). For persistent issues, see [Model Performance](../modeling/performance.md).

---

## Further Reading

- [Parity Plot](parity_plot.md) - Visual accuracy assessment
- [Model Performance](../modeling/performance.md) - Comprehensive model evaluation
- [Q-Q Plot](qq_plot.md) - Uncertainty calibration diagnostic
- [Calibration Curve](calibration_curve.md) - Coverage verification

---

**Key Takeaway**: The metrics evolution plot is your guide to tracking model improvement during active learning. Use it to decide when your model is good enough and when to stop collecting data.
