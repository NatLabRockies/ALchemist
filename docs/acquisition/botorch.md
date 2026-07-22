# BoTorch Acquisition Functions

The **BoTorch** backend in ALchemist provides a flexible and powerful interface for selecting the next experiment(s) using a variety of acquisition functions from the [BoTorch](https://botorch.org/) library. This guide explains the available options, how to use them, and what each setting means.

---

## Overview

The Acquisition panel in ALchemist allows you to:

- Choose between **Regular**, **Batch**, and **Exploratory** acquisition strategies.

- Select from several acquisition functions, each balancing exploration and exploitation in different ways.

- Customize parameters such as batch size and Monte Carlo integration points.

- Run the selected strategy to suggest the next experiment(s) based on your trained model.

---

## Important Note

You must first train your model using the BoTorch backend before running any BoTorch acquisition functions.  
See [BoTorch Backend](../modeling/botorch.md) for details on model training.

---

## Acquisition Types

### 1. Regular Acquisition

- **Expected Improvement (EI):**  
  Suggests points with the highest expected improvement over the current best observed value.

- **Log Expected Improvement (LogEI):**  
  Numerically stable version of EI.

- **Probability of Improvement (PI):**  
  Selects points with the highest probability of improving over the current best value.

- **Log Probability of Improvement (LogPI):**  
  Numerically stable version of PI.

- **Upper Confidence Bound (UCB):**  
  Balances exploration and exploitation by selecting points with the highest upper confidence bound.

**Customization:**  

- Choose to **maximize** or **minimize** your objective.

### 2. Batch Acquisition

- **q-Expected Improvement (qEI):**  
  Selects a batch of points that together maximize expected improvement.

- **q-Upper Confidence Bound (qUCB):**  
  Batch version of UCB.

**Customization:**  

- Set **batch size** (number of points to suggest at once, q).

- Monte Carlo samples (mc_samples) are used internally for batch methods.

### 3. Exploratory Acquisition

- **Integrated Posterior Variance (qNIPV):**  
  Selects points to reduce overall model uncertainty, focusing on exploration rather than optimization.

**Customization:**  

- Set the number of **Monte Carlo integration points** (higher values improve accuracy but increase computation time; 500–2000 is typical).

---

## How to Use

1. **Train Model:**  
   Train your model using the BoTorch backend. See [BoTorch Backend](../modeling/botorch.md) for instructions.

2. **Open Acquisition Panel:**  
   Go to the Acquisition panel. The BoTorch options will appear automatically.

3. **Choose Acquisition Type:**  
   Use the segmented button to select Regular, Batch, or Exploratory.

4. **Configure Options:**  
   - Select the acquisition function from the dropdown.
   - Adjust parameters (batch size, MC points) as needed.
   - Choose whether to maximize or minimize.

5. **Run Acquisition:**  
   Click **Run Acquisition Strategy** to suggest the next experiment(s). Results, including predicted value and uncertainty, will be shown in a notification window and highlighted in the data table and plots.

---

## Model Optimum Finder

In addition to acquisition functions, you can use the **Model Prediction Optimum** tool to find the point where the model predicts the best value (maximum or minimum).  
**Note:** This does not balance exploration and exploitation—it simply finds the model's optimum prediction.

---

## Linear Input Constraints

Registered linear input constraints (via `add_input_constraint`) are honored across every surface that proposes points in the input space:

- **Acquisition (`suggest_next`)** — constraints are passed to `optimize_acqf` on both the continuous and mixed-variable optimizer paths, in the model's raw variable space. The BoTorch model normalizes inputs internally, so constraints are expressed in raw units (not the unit cube).
- **`find_optimum`** — the search grid is filtered to the feasible region before selecting the optimum; the API and desktop optimum finders inherit this.
- **Plots** — contour, surface, slice, and 3D voxel plots (including acquisition/uncertainty variants) mask infeasible cells (rendered blank). If a slice leaves no feasible cells, the unmasked plot is shown with a warning rather than failing.
- **Initial design (DOE)** — space-filling methods (random/LHS/Sobol/Halton/Hammersly) reject-and-resample until enough strictly-feasible points are found; classical/optimal designs drop infeasible rows with a warning.

Feasibility is evaluated by `SearchSpace.filter_feasible` / `is_feasible`, the single shared primitive. Grid/plot judgments use a relative tolerance band (equality constraints never hit a discrete grid exactly); DOE sampling uses strict tolerance so no design point exceeds a stated bound.

**sklearn backend:** the skopt optimizer cannot express linear input constraints, so registering one and calling `suggest_next` raises a clear error. Use the `botorch` backend for constrained input optimization.

---

## Tips & Notes

- **Batch Acquisition:** Use batch mode to suggest multiple experiments at once, useful for parallel experimentation.

- **Exploratory Mode:** Use qNIPV when you want to reduce model uncertainty rather than optimize the objective.

- **Parameter Tuning:** Increase MC points for more accurate but slower exploratory acquisition.

- **Publication Quality:** All results and suggested points are integrated with ALchemist's visualization tools for easy analysis and export.

---

For more details on the underlying algorithms, see the [BoTorch documentation](https://botorch.org/docs/acquisition.html)