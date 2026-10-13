# Parity Plot

The **parity plot** (also called actual vs. predicted plot) is a fundamental diagnostic tool for evaluating your surrogate model's prediction accuracy. It shows how well the model's predictions match the actual experimental measurements.

---

## What the Parity Plot Shows

**X-axis**: Actual (true) experimental values  
**Y-axis**: Model predicted values

**Perfect predictions**: All points lie on the diagonal line (y = x), indicating predictions exactly match observations.

**Additional information displayed**:

- **Error bars**: Optional uncertainty visualization (±1σ, ±1.96σ, ±2σ, ±2.58σ, or ±3σ)

- **Performance metrics**: RMSE, MAE, and R² displayed in the plot title

- **Parity line**: Diagonal reference (y = x) for perfect predictions

---

## Interpreting the Plot

### Excellent Model Fit

**What it looks like:**

- Points tightly clustered along diagonal line

- Minimal scatter

- R² > 0.9

- RMSE and MAE are small relative to output range

**What it means:**

- Model predictions are highly accurate

- Strong confidence in optimization decisions

- Safe to trust acquisition function suggestions

### Good Model Fit 👍

**What it looks like:**

- Points generally follow diagonal with moderate scatter

- R² between 0.7-0.9

- No systematic bias

**What it means:**

- Model captures main trends

- Acceptable for optimization

- Some uncertainty in predictions

### Poor Model Fit

**What it looks like:**

- Large scatter around diagonal

- R² < 0.5

- High RMSE relative to output range

**What it means:**

- Model has difficulty predicting outcomes

- Consider collecting more data

- Try different kernel or backend

- Check data quality

### Systematic Bias 🔴

**What it looks like:**

- Points systematically above or below diagonal

- Clear pattern rather than random scatter

**What it means:**

- **Above diagonal**: Model consistently under-predicts (predicts lower than actual)

- **Below diagonal**: Model consistently over-predicts (predicts higher than actual)

- Check data preprocessing and transforms

- May indicate model misspecification

---

## Cross-Validation Approach

ALchemist's parity plot uses **k-fold cross-validation** to provide unbiased estimates:

1. Data is split into k folds (typically 5)
2. For each fold:
   - Train model on remaining k-1 folds
   - Predict on held-out fold
3. Aggregate all predictions for complete dataset coverage

**Benefits:**

- Predictions for every point without using that point in training

- Unbiased estimate of generalization performance

- More reliable than training set predictions

---

## Error Bars and Uncertainty

### Selecting Confidence Intervals

Choose from standard statistical confidence intervals:

- **±1σ (68%)**: Standard deviation, 68% of true values should fall within

- **±1.96σ (95%)**: Most common, 95% confidence interval

- **±2σ (95.4%)**: Approximately 2-sigma interval

- **±2.58σ (99%)**: High confidence, 99% of true values

- **±3σ (99.7%)**: Very high confidence, three-sigma interval

### Interpreting Error Bars

**Well-calibrated uncertainty:**

- Error bars cross the diagonal line for most points

- About 68% of points within ±1σ, 95% within ±2σ

**Under-confident predictions:**

- Error bars are much larger than actual deviations

- Most points fall well within error bars

- Model is too cautious

**Over-confident predictions:**

- Error bars are smaller than actual deviations

- Many points fall outside error bars

- Model underestimates uncertainty (see [Q-Q plot](../background/interpreting_qqplot.md) for more)

---

## Calibrated vs. Uncalibrated Results

ALchemist provides both calibrated and uncalibrated predictions:

### Uncalibrated (Raw Model Output)

- Direct predictions from Gaussian Process

- May have over/under-confident uncertainty estimates

- Useful for comparing with calibrated results

### Calibrated (Adjusted Uncertainty)

- Uncertainty scaled based on cross-validation residuals

- Corrects systematic over/under-confidence

- Recommended for decision-making

- Toggle available in visualization panel

For more on calibration, see [Interpreting Calibration Curves](../background/interpreting_calibration.md).

---

## Performance Metrics

### RMSE (Root Mean Squared Error)

$$
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$

- Measures average prediction error magnitude

- Same units as output variable

- Sensitive to large errors (squared term)

- Lower is better

**Interpretation:**

- RMSE = 0: Perfect predictions

- RMSE << output range: Excellent fit

- RMSE ≈ output std dev: Poor fit (no better than mean prediction)

### MAE (Mean Absolute Error)

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
$$

- Average absolute difference between predictions and actual

- Same units as output variable

- Less sensitive to outliers than RMSE

- Lower is better

**Interpretation:**

- MAE typically < RMSE (due to no squaring)

- If MAE ≈ RMSE: Errors are consistently sized

- If MAE << RMSE: Some large outlier errors

### R² (Coefficient of Determination)

$$
R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}
$$

- Fraction of variance explained by model

- Dimensionless (0 to 1 for good models)

- Can be negative for very poor fits

**Interpretation:**

- R² = 1.0: Perfect predictions

- R² > 0.9: Excellent fit

- R² = 0.7-0.9: Good fit

- R² = 0.5-0.7: Moderate fit

- R² < 0.5: Poor fit

- R² < 0: Model worse than predicting mean

---

## Practical Guidelines

### When to Be Satisfied

Proceed with optimization if:

- R² > 0.7 with no systematic bias

- Error bars reasonable (not too wide or narrow)

- No obvious outliers or patterns

- Metrics improve as data is added

### When to Improve Model

Take action if:

- R² < 0.5 or negative

- Clear systematic bias visible

- Many points outside error bars (over-confident)

- Error bars much wider than scatter (under-confident)

### Remediation Strategies

**For poor R²:**
1. Collect more training data
2. Try different kernel (RBF ↔ Matern, adjust ν)
3. Switch backend (sklearn ↔ BoTorch)
4. Check data quality (outliers, measurement errors)
5. Apply input/output transforms

**For systematic bias:**
1. Check data preprocessing
2. Verify units and scales
3. Try different kernel
4. Check for missing variables or physical constraints

**For miscalibrated uncertainty:**
1. Use calibration feature (automatic in ALchemist)
2. Adjust noise parameter
3. See [Q-Q plot](../background/interpreting_qqplot.md) for diagnosis

---

## Using the Parity Plot in Workflows

### During Initial Modeling

- Generate after first model training

- Check R² > 0.5 before proceeding

- Identify if more initial data needed

### During Active Learning

- Monitor after each iteration

- Watch for degradation (may indicate overfitting)

- R² should generally improve with more data

### Before Final Optimization

- Ensure R² > 0.7

- Verify calibration quality

- Confirm no systematic bias

- Check that best experiments are well-predicted

---

## Desktop vs. Web UI

**Desktop Application:**

- Access via Visualizations dialog after training model

- Full Matplotlib controls for zoom, pan, save

- Customization options for publication-quality figures

**Web Application:**

- Embedded in visualizations panel

- Interactive Recharts visualization

- Theme-aware (light/dark mode)

- Select error bar confidence levels

- Toggle calibrated/uncalibrated results

---

## Example Interpretations

### Case 1: Excellent Fit
```
R² = 0.94, RMSE = 1.2, MAE = 0.9
Points tightly along diagonal, error bars appropriate
→ Model ready for optimization, trust suggestions
```

### Case 2: Under-Predicting
```
R² = 0.72, RMSE = 3.1, MAE = 2.8
Points systematically above diagonal
→ Check data units, try transforms, more data needed
```

### Case 3: High Uncertainty
```
R² = 0.81, RMSE = 2.0, MAE = 1.5
Large error bars, but points within them
→ Under-confident, consider calibration or tighter kernel
```

### Case 4: Poor Fit
```
R² = 0.28, RMSE = 8.5, MAE = 7.2
Large scatter, no clear pattern
→ Collect more data, check data quality, try different kernel
```

---

## Further Reading

- [Metrics Evolution](metrics_plot.md) - Track RMSE, MAE, R² over iterations

- [Q-Q Plot](../background/interpreting_qqplot.md) - Uncertainty calibration diagnostic

- [Calibration Curve](../background/interpreting_calibration.md) - Coverage assessment

- [Model Performance](../modeling/performance.md) - Troubleshooting poor fits

---

The parity plot is your primary tool for assessing model quality. Combined with [Q-Q plots](../background/interpreting_qqplot.md) and [calibration curves](../background/interpreting_calibration.md), you get a complete picture of both prediction accuracy and uncertainty calibration.
