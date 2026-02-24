# Model Performance

Evaluating the performance of your surrogate model is a critical step in the active learning workflow. ALchemist provides several tools and visualizations to help you assess model quality and guide your next steps.

---

## Cross-Validation and Error Metrics

After training a model, ALchemist automatically computes cross-validation metrics such as RMSE, MAE, MAPE, and R². These metrics are visualized in the **Visualizations** dialog, where you can see how model error changes as more data points are added.

**General expectation:**  

- As you increase the number of observations, cross-validation error (e.g., RMSE) should generally decrease. This indicates that the model is learning from the data and improving its predictions.

---

## What if Error Doesn't Decrease?

If you notice that error metrics do not decrease with more data, consider the following:

1. **Small Data Regime (<10 points):**  
   With very few data points, high error or flat trends are common. This is not necessarily a problem—acquisition functions will naturally suggest new experiments in regions of high uncertainty, helping the model converge as more data is collected.

2. **Try a Different Backend:**  
   Switch between the scikit-optimize and BoTorch backends. Sometimes one backend may fit your data better, especially depending on the variable types and dimensionality.

3. **Tweak the Kernel:**  
   Experiment with different kernel types (RBF, Matern, RationalQuadratic) or adjust the Matern `nu` parameter. The choice of kernel can significantly affect model flexibility and fit.

---

## Additional Tips and Considerations

- **Overfitting:**  
  Overfitting may appear as jagged or unrealistic response surfaces in contour plots. If you see this, try increasing regularization (e.g., by specifying higher noise values) or collecting more data.

- **Data Quality:**  
  Poor model performance can result from poor data quality. Check for outliers or inconsistent measurements. Consider populating the `Noise` column with an appropriate metric (such as variance or signal-to-noise ratio) to help regularize the model. See the [BoTorch Backend](botorch.md) page for details on noise handling.

- **Model Diagnostics:**  
  Use parity plots and error metric trends to diagnose underfitting, overfitting, or data issues. Ideally, parity plots should show points close to the diagonal (y = x), indicating good agreement between predicted and actual values.

- **Variable Importance:**  
  Both backends use anisotropic (ARD) kernels by default, allowing the model to learn a separate lengthscale for each variable. This can help identify which variables are most relevant to the output.

---

## Summary

- Expect error to decrease as more data is added.

- Use backend and kernel options to improve fit.

- Watch for signs of overfitting or poor data quality.

- Use regularization and noise estimates to stabilize the model.

For more on error metrics and visualization, see the [Metrics Evolution Plot](../visualizations/metrics_plot.md) page.