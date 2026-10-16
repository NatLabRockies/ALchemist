# Troubleshooting

This page lists the problems people meet most often, what causes each, and the fix. Where the cause is a known bug, its ID is given, as on the other pages; the [Changelog](../reference/changelog.md) lists what each release fixed.

## Installing and starting

| Symptom | Cause | Fix |
|---|---|---|
| `pip install alchemist-nrel` finds no matching version, or says it needs a newer Python. | The environment's Python is older than 3.11. | Create an environment with Python 3.11 or later; see [Install](install.md#requirements). |
| `alchemist-web` stops at once with an error that reads "Address already in use" on Linux and macOS, or "[WinError 10048] Only one usage of each socket address" on Windows. | Another program, often an `alchemist-web` started earlier, holds port 8000. `alchemist-web` always uses port 8000 and has no option to change it (W37). | Stop the other program, or start the server on another port with uvicorn, as below. |
| `http://localhost:8000` shows a few lines of JSON (name, version, docs) instead of the app. | The web app is not built. Installs from GitHub and fresh clones do not include it. | Install from PyPI, or build it with `npm run build` in `alchemist-web`; see [Install from source](install.md#install-from-source). |
| The welcome screen's **Join** does not find a session you used yesterday. | The server reads its sessions from `cache/sessions` under the folder it was started from. | Start `alchemist-web` from the same folder as before, or open the session's file with **Load Session**. |

To find what holds port 8000, and to run on another port:

```bash
lsof -i :8000                       # macOS and Linux
netstat -ano | findstr :8000        # Windows
python -m uvicorn api.main:app --host 127.0.0.1 --port 8001
```

Then open `http://localhost:8001`. The app sends its requests to the address it was loaded from, so any port works.

## In the web app

| Symptom | Cause | Fix |
|---|---|---|
| "Session not found. Please create a new session." and the app returns to the welcome screen. | The server no longer has that session: it was started from another folder, or its `cache/` folder was deleted. | Open your saved file with **Load Session**. |
| The terminal running `alchemist-web` prints "Autosave failed for session ...". | The server saves a session to `cache/sessions` after every change, and the write failed, for example because the disk is full or the folder is not writable. The change is still in the running session, but the session's file lacks it. | Free the space or fix the permissions; the next change saves the session again. Until then, choose **Save** to keep a copy as a file. |
| **Save** downloads a new file each time, and there is no **Save As…**. The same goes for **Save File** and **Export Log**. | Only Chrome and Edge, on `localhost` or https, let a page write to a file you chose. Safari, Firefox and plain http addresses such as `http://labserver:8000` get downloads. | Use Chrome or Edge on `localhost`, or keep the downloaded files; see [Sessions and audit log](../guide/more/sessions.md#2-save). |
| **Train Model** stays greyed out. | The session has fewer than five results. | Record at least five; see [Train a model](../guide/train.md#1-train). |
| After **Save N & close**: "Model not retrained", "Retraining failed on the server; see the API log". | The retrain raised an error. The terminal running `alchemist-web` shows it; a noise value on only some rows is a common one. | Fix the cause, then choose **Train Model** in the **Model** tab, which shows the message. |
| "Give every row a noise value or none." with the rows that lack one. | Some results have a noise value and others do not. | Give every row a noise value, or clear the column; see [Noise, Iteration and Reason](../guide/run-and-record.md#noise-iteration-and-reason). |
| "Observed (context) variables are only supported with the BoTorch backend." | The session has an observed variable and the model is scikit-learn. | Choose **BoTorch** and train again. |
| "Linear input constraints are registered but the sklearn backend cannot enforce them during acquisition." | The session has input constraints and the model is scikit-learn. | Train with **BoTorch**; see [Constraints](../guide/constraints.md). |
| "multi-objective optimization requires backend='botorch'". | The data have several target columns and the model is scikit-learn. | Train with **BoTorch**; see [Multi-objective](../guide/more/multi-objective.md). |
| "Enter the measured value of ..." when you choose **Run Acquisition Strategy**. | An observed variable needs its current value before every suggestion. | Type it under **Measured now**; see [Hold variables](../guide/suggest.md#2-hold-variables). |
| **Add Point...** shows a count before you choose **Stage Suggestions**, and those points are gone after a reload. | W29: the suggestions live only in the browser until they are staged. | Choose **Stage Suggestions**, once, after every suggestion. |
| After **Join** or **Load Session**, the **Model** tab shows scikit-learn, not the model you trained. | W31: the tab shows its starting settings, not the session's. | Choose the backend and settings again before **Train Model**. |
| The plots still show the previous session after **Clear**. | W24: **Model Visualizations** stays on the session it was opened in. | Close the panel and choose **Show Visualizations** again. |
| **Load File** stops at a name that exists, and some variables were added. | W36: a plain list is added one variable at a time. | Load into an empty session, or use a file with a `variables` key; see [Define variables](../guide/variables.md#2-save-and-load-variables). |
| Ticking **Calibrate Uncertainty** changes nothing. | F1: calibration runs on every train either way. | None needed; see [Reading the diagnostics](../concepts/diagnostics.md). |

## From Python

| Symptom | Cause | Fix |
|---|---|---|
| "'EI' scores one point at a time, so it cannot suggest a batch of 3. Use 'qEI', 'qUCB' or 'qIPV' for a batch, or set n_suggestions=1." | Only the q-strategies of the BoTorch backend suggest a batch. | `suggest_next(strategy="qEI", n_suggestions=3)`; see [Suggest next](../guide/suggest.md#one-point-or-a-batch). |
| "suggest_next() got unexpected keyword argument(s) ..." | A keyword `suggest_next` does not know, such as `n_candidates`. | Use `n_suggestions` for the batch size; the message lists the accepted parameters. |
| "Missing observed (context) values for: [...]", or "Provide observed={name: value, ...} with a value for each." | The session has observed variables, and each suggestion needs their current values. | `suggest_next(..., observed={"ambient": 22.5})`. |
| "Target column(s) [...] not found in CSV file. Available columns: [...]" | `load_data` looks for a column named `Output` unless told otherwise. | Pass `target_columns="your_column"`. |
| "Missing search space variables in inputs: {...}" | `add_experiment` needs a value for every variable. | Add the missing names to the dictionary. |
| Every call prints INFO log lines, and your own logging settings stop working after the import. | N2: importing `alchemist_core` sets up logging at INFO and replaces the existing configuration. | Call `set_verbosity(False)` after the import (`from alchemist_core import set_verbosity`) to keep only warnings. |
| Outcome constraints are gone after `load_session`. | The file was saved by a version before 0.4.0, which did not write them. | Add them again and save; files saved by 0.4.0 carry an `outcome_constraints` key. |
| `save_session("results/")` writes a file called `results`. | A8: the argument is a file path. | Pass a file name, such as `"results/campaign.json"`. |

## Reporting a problem

Report a bug at [github.com/NatLabRockies/ALchemist/issues](https://github.com/NatLabRockies/ALchemist/issues). Give the ALchemist version (`python -c "import alchemist_core; print(alchemist_core.__version__)"`), the browser, what you did, the message, and the lines the terminal running `alchemist-web` printed at that moment. Leave out data you cannot share; a session file with made-up numbers that shows the problem is the most useful thing you can attach.
