# scikit-optimize Acquisition Functions

The **scikit-optimize** (skopt) backend in ALchemist provides a range of acquisition functions for Bayesian optimization using the [scikit-optimize](https://scikit-optimize.github.io/) library. This guide explains the available options, how to use them, and what each setting means.

---

## Overview

The Acquisition panel in ALchemist allows you to:

- Choose from several acquisition functions, each balancing exploration and exploitation in different ways.

- Customize parameters such as exploration/exploitation trade-offs.

- Run the selected strategy to suggest the next experiment based on your trained model.

---

## Important Note

You must first train your model using the scikit-optimize backend before running any skopt acquisition functions.  
See [scikit-optimize Backend](../modeling/skopt.md) for details on model training.

---

## Acquisition Functions

- **Expected Improvement (EI):**  
  Balances exploration and exploitation by selecting points with the highest expected improvement over the current best value.  
  **Parameter:** ξ (xi) — higher values favor exploration.

- **Upper Confidence Bound (UCB):**  
  Selects points with the highest upper confidence bound, balancing exploration and exploitation.  
  **Parameter:** κ (kappa) — higher values increase exploration.

- **Probability of Improvement (PI):**  
  Selects points with the highest probability of improving over the current best value.  
  **Parameter:** ξ (xi) — higher values favor exploration.

- **GP Hedge (Auto-balance):**  
  Automatically balances between EI, UCB, and PI by adaptively selecting the best-performing strategy during optimization.  
  **Parameters:** ξ (xi) and κ (kappa).

**Customization:**  

- Choose to **maximize** or **minimize** your objective.

- Adjust ξ (xi) and κ (kappa) parameters using sliders as appropriate for the selected acquisition function.

---

## How to Use

1. **Train Model:**  
   Train your model using the scikit-optimize backend. See [scikit-optimize Backend](../modeling/skopt.md) for instructions.

2. **Open Acquisition Panel:**  
   Go to the Acquisition panel. The scikit-optimize options will appear automatically.

3. **Select Acquisition Function:**  
   Use the dropdown menu to select the acquisition function (EI, UCB, PI, or GP Hedge).

4. **Configure Options:**  
   - Adjust ξ (xi) and κ (kappa) parameters as needed.
   - Choose whether to maximize or minimize.

5. **Run Acquisition:**  
   Click **Run Acquisition Strategy** to suggest the next experiment. Results, including predicted value and uncertainty, will be shown in a notification window and highlighted in the data table and plots.

---

## Model Optimum Finder

In addition to acquisition functions, you can use the **Model Prediction Optimum** tool to find the point where the model predicts the best value (maximum or minimum).  
**Note:** This does not balance exploration and exploitation—it simply finds the model's optimum prediction.

---

## Tips & Notes

- **Parameter Tuning:**  
  - Increase ξ (xi) or κ (kappa) for more exploration; decrease for more exploitation.
  - GP Hedge is useful if you are unsure which acquisition function to use.

- **Publication Quality:**  
  All results and suggested points are integrated with ALchemist's visualization tools for easy analysis and export.

---

For more details on the underlying algorithms,