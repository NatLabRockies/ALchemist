# Issues & Troubleshooting Log

This log tracks known issues, user-reported bugs, and observations from internal testing for ALchemist. It is maintained by the development team.

---

## How to Report an Issue

If you encounter a problem or have feedback, please [open an issue on GitHub](https://github.com/NatLabRockies/ALchemist/issues) or email [ccoatney@nrel.gov](mailto:ccoatney@nrel.gov) with the following information:

- **Brief description of the issue**

- **Steps to reproduce (if applicable)**

- **Your operating system and environment**

- **Any error messages or screenshots**

- **Date observed**

---

## Known Issues

| Issue                                                                                         | Date Reported | Status      | Notes / Workarounds                                                                                 |
|-----------------------------------------------------------------------------------------------|---------------|-------------|-----------------------------------------------------------------------------------------------------|
| None currently - see resolved issues below                                                   | -             | -           | -                                                                                                   |

---

## Resolved Issues

| Issue                                                                 | Date Reported | Date Resolved | Notes                                                                                               |
|-----------------------------------------------------------------------|---------------|---------------|-----------------------------------------------------------------------------------------------------|
| **BoTorch kernel hyperparameters not shown in "Next Point" dialog**  | **2024-06-16** | **2025-08-20** | **✅ RESOLVED**: Enhanced hyperparameter extraction with recursive kernel traversal. Now properly displays ARD lengthscales, kernel types, noise parameters, and transform information for both SingleTaskGP and MixedSingleTaskGP models. Handles complex AdditiveKernel structures with categorical/continuous variables. |
| GUI not displaying fully on macOS; windows may be cut off             | 2024-06-16    | 2025-06-29    | Resolved as of latest testing; GUI now displays correctly on Mac without external monitor.           |
| Loading variables from CSV does not work; only JSON loads correctly   | 2025-06-29    | 2025-07-15    | Fixed CSV parsing for integer min/max values and categorical value parsing.                         |
| Saving variables as CSV and reloading does not restore variables      | 2025-06-29    | 2025-07-15    | Fixed Integer variable population and main UI update after variable definition.                     |
| Main UI "Load Variables" button fails with JSON error when loading CSV files | 2025-07-15    | 2025-07-15    | Fixed load_variables() function to properly detect and parse both JSON and CSV file formats.        |
| Categorical variables losing values when editing in variables setup   | 2025-07-15    | 2025-07-15    | Enhanced categorical editor data filtering and improved Sheet widget data handling.                 |
| Model Prediction Optimum tool: suggested experiment gives fractional value for integer variable (BoTorch backend) | 2025-06-29    | 2025-07-15    | Fixed by implementing integer rounding in optimization results. Note: BoTorch likely has native integer constraints - investigate optimize_acqf with integer_indices parameter for future improvement. |
| Model Prediction Optimum tool: optimizing to maximum or minimum gives same suggested values (BoTorch backend) | 2025-06-29    | 2025-07-15    | Fixed by correcting acquisition panel to use find_optimum() method instead of select_next() method. |
| **EI/LogEI/PI return degenerate max-variance suggestions (BoTorch backend)** | **2026-07-21** | **2026-07-21** | **✅ RESOLVED**: Improvement-family acquisitions (EI, LogEI, PI, LogPI) collapsed to pure exploration (low predicted mean, max σ) while UCB was fine. Root cause: `best_f` used the raw observed max of a noisy target, which the smoothed GP posterior mean cannot reach, making improvement negative everywhere. Fixed by setting the incumbent to the best posterior mean at the training points (BoTorch's noisy-observation convention), and routing `ei` to the numerically stable `LogExpectedImprovement` to avoid analytic EI's vanishing-gradient degeneracy. Not constraint-specific. |

---

This log is updated as issues are reported and resolved.