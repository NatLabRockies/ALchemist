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
| **Loading a session restored 0 experiments when the objective column was not named `Output`** | **2026-07-28** | **2026-07-28** | **✅ RESOLVED**: Long-standing bug (present since ≤ v0.3.3). The loader hardcoded `output = row.get('Output')`, so a session whose target was named e.g. `Methane umol/g/Wh` silently restored 0 rows (each row failed `np.isfinite(None)` and was swallowed by a per-row try/except — "loaded without error" but with no data). Loader now resolves the target from a persisted `target_columns` key or infers it from non-variable/non-metadata columns; `save_session` records `target_columns`. Backward compatible. |
| **Batch acquisition points all appeared identical in the Add Point dialog** | **2026-07-28** | **2026-07-28** | **✅ RESOLVED**: Stepping through a qEI batch showed the first point's values for all N points. The dialog initialized form state from props at mount only and was not remounted when the suggestion prop changed. Fixed by keying the dialog per suggestion. The acquisition/staging were always correct (verified 5 distinct points staged/persisted) — display-only bug. |
| **Staged suggestions persisted across a new session ("ghost points")** | **2026-07-28** | **2026-07-28** | **✅ RESOLVED**: The staged-suggestions restore effect only added to state and never cleared, so a new/switched session kept stale suggestions in the UI. It now always resolves to a definitive value and clears when the session has none, with a race guard. |
| **Add Point dialog showed `Iteration: N/A`** | **2026-07-28** | **2026-07-28** | **✅ RESOLVED**: Suggested points never carried an iteration; the API computed it only for the audit log. `AcquisitionResponse` now returns `iteration`, tagged onto each suggestion and written on save (whole batch = one iteration). |
| **"Retrain model" checkbox ignored when recording results via the work queue** | **2026-07-28** | **2026-07-28** | **✅ RESOLVED**: The queue-complete endpoint had no auto-train, so the default record path never retrained even when requested. Added `auto_train` to the endpoint, threaded through the web client. |
| **Web app served a stale prebuilt bundle (fixes appeared not to work)** | **2026-07-28** | **2026-07-28** | **✅ RESOLVED (process)**: The FastAPI server serves the prebuilt `alchemist-web/dist/`, which was months stale; merged source fixes weren't in the browser. Compounded by `npm run build` silently failing on test files. Fixed the build (test files excluded from the app tsconfig) and documented the rebuild-and-restart requirement in `AGENTS.md`. |
| **Regret / hypervolume plots showed NaN gaps when a subset GP failed to fit** | **2026-07-28** | **2026-08-25** | **✅ RESOLVED**: `_compute_posterior_predictions` refits a fresh GP on each data prefix; small / ill-conditioned prefixes could raise `ModelFittingError` and the handler left that iteration NaN, producing a visible gap. Most reproducible with categorical variables, where the `AdditiveKernel` structure skips the hyperparameter-reuse fast path so every iteration re-optimizes. Both the single-objective and MOBO paths now walk a fallback ladder (requested transforms → transforms disabled → jitter 1e-3 → 1e-2). Every plotted value is a genuine fitted-GP prediction; NaN remains the last resort if the whole ladder fails. |

---

This log is updated as issues are reported and resolved.