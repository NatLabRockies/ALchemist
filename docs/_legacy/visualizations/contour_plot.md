# Contour Plot Visualization

The **Contour Plot** feature in ALchemist lets you visualize your surrogate model’s predicted output as a 2D contour plot, providing insight into the model’s response surface across your variable space. This tool is designed for interpreting model predictions, identifying trends, and creating publication-quality figures.

---

## How to Create a Contour Plot

1. **Open the Visualizations Dialog:**  
   After training a model, open the Visualizations dialog from the main application window.

2. **Choose X and Y Axes:**  
   Use the dropdown menus in the "Contour Plot Options" panel to select which two real-valued variables to display on the X and Y axes.

3. **Set Fixed Values for Other Variables:**  
   Set a value for each remaining variable using the provided controls. This lets you view a “slice” of the model’s prediction at a specific cross-section.

4. **Plot the Contour:**  
   Click **Plot Contour** to generate the plot for your selected axes and fixed values. The predicted output will be shown as filled contours.

---

## Customization and Export

- **Customize Appearance:**  
  Click **Customize Plot** to adjust the plot title, axis labels, colormap, font, font size, axis limits, font weight, tick style, and whether to show experimental data points (white circles) and the next suggested point (red diamond).

- **Save Figures:**  
  Use the Matplotlib toolbar below the plot to save your figure as a high-resolution image (PNG, SVG, PDF, etc.) for publication or presentations.

---

## Additional Features

- **Interactive Controls:**  
  Changing the X or Y axis or adjusting fixed values for other variables will update the plot in real time.

- **Legend:**  
  The plot includes a legend for experimental points and the next suggested point if displayed.

- **Hyperparameters Display:**  
  The bottom of the Visualizations dialog shows the learned kernel hyperparameters from the trained model.

---

## Web Application

The contour plot is also available in the **ALchemist web application**:

1. Navigate to **GPR Panel**
2. Train a model
3. Click **Show Model Visualizations**
4. Select **Contour Plot** from plot type buttons
5. Use dropdown menus to select X and Y axis variables
6. Adjust sliders for fixed values (if > 2 variables)
7. Plot updates dynamically

**Web UI features**:

- Interactive variable selection

- Real-time updates

- Responsive design

- Hover tooltips showing values

- Download button for export

---

## Tips

- Use customization options to match your figure style to journal or presentation requirements.

- Generate multiple contour plots for different variable pairs and fixed values to explore the model's predictions.

- Contour plots are useful for diagnosing model behavior, visualizing optima, and communicating results.

- In web UI, create multiple plots to compare different variable combinations side-by-side.

---

## Further Reading

- [Parity Plot](parity_plot.md) - Verify model prediction accuracy
- [Q-Q Plot](qq_plot.md) - Check model calibration
- [Model Performance](../modeling/performance.md) - Comprehensive model evaluation
- [Web Application Guide](../setup/web_app.md) - Complete web UI documentation

For more on model evaluation and error metrics, see the [Metrics Evolution](metrics_plot.md) section.