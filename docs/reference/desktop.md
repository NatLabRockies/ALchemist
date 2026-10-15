# Desktop app (legacy)

The desktop app, `alchemist`, is the original CustomTkinter interface to ALchemist. This page says how to install it, what it still does that the web app does not, and why it refuses to save some sessions.

!!! warning "Deprecated"
    The desktop app is deprecated and will be removed in a future release. It gets no new features. The web app, started with `alchemist-web`, is the supported interface; start with [Your first loop](../get-started/first-loop.md).

## Install and start

The desktop app is no longer part of the default install. Its packages (customtkinter, tksheet, ctkmessagebox, mplcursors and tabulate) come with the `desktop` extra, and it needs a Python build that includes Tk:

```bash
pip install "alchemist-nrel[desktop]"
alchemist
```

`alchemist` first prints that the desktop app is deprecated and points to the web app. Without the extra it stops with the name of the missing package and the command above.

## What it still does

Most of the workflow is in both apps: variables, the initial and optimal designs, training with BoTorch or scikit-learn, the diagnostic plots, suggestions and **Find Model Optimum**. A few things are only in one of them.

| Only in the desktop app | Only in the web app |
|---|---|
| Multi-objective suggestions (qEHVI and qNEHVI) with a direction per objective and a reference point | The work queue, provenance and the monitoring dashboard ([Lab automation](../automation/loop.md), [Monitoring](../guide/more/monitoring.md)) |
| Column roles when loading data (Variable, Target or Drop) and changing the targets after loading | Observed variables, hard-to-change variables and split-plot designs ([Split-plot designs](../guide/more/split-plot.md)) |
| An editable experiment table, and **Save Experiments** to CSV | Input constraints in designs and suggestions ([Constraints](../guide/constraints.md)) |
| Variable files in CSV as well as JSON | AI effect suggestion for optimal designs ([Optimal design](../guide/more/optimal-design.md)) |
| Matplotlib figures with a customize dialog, saved in any Matplotlib format | The **Hold** list for suggestions, and the 30-second recovery copy |

In Python, two of the desktop's extras are available: multi-objective suggestions with `suggest_next(strategy="qNEHVI", goal=[...], ref_point=[...])` ([Multi-objective](../guide/more/multi-objective.md)), and target columns with `load_data(path, target_columns=[...])`. Variables can be saved and loaded as JSON with `SearchSpace.save_to_json` and `load_from_json`, but there are no CSV variable files, no editable experiment table and no figure customize dialog; the plot methods return Matplotlib figures that you save yourself.

!!! warning "Known issue"
    D3: the desktop app ignores input constraints in its designs and suggestions, although **Find Model Optimum** respects them. D4: its calibration checkbox has no effect (F1 in the web app). D5: **Help**, **About** and **Settings** open placeholder dialogs.

## Opening a session from the web app

The desktop app keeps its own copy of the variables and the experiment table, and writes that copy back into the session before it saves, trains or logs. Its copy has no place for several things a web or Python session can hold, so writing it back would delete them.

Instead, the desktop app checks first and refuses. A dialog titled "Can't sync this session" lists what would be lost, ends with "Nothing was changed.", and leaves the session as it was. It refuses when the session has:

- an observed variable, or a variable marked hard or very hard to change;
- derived variables;
- input constraints;
- the columns that provenance and split-plot designs add to the experiment table (`ProvenanceId`, `WholePlot`, `Subplot`, `RunOrder`), as any session recorded through the work queue or the web app's **Add Point...** has;
- rows with a result whose inputs are empty or not numbers, which it would drop.

The check runs on **Save Session**, **Save Session As...**, saving when you quit, **Train Model**, **Log to Audit Trail**, and **Save & Close** in the variable editor. For the first kinds, the dialog says to use the web app for that session. For rows with empty inputs, fill in or delete the empty cells in the experiment table and try again. A session the desktop app made itself is never refused.

Rows that have no result yet, such as suggestions you have not run, are still dropped from the table when it syncs, as in earlier versions.

## Moving to the web app

The two apps use the same session file. In the web app, choose **Load Session** on the welcome screen and pick the file the desktop app saved; see [Sessions and audit log](../guide/more/sessions.md). The variables and results come across, and so do the model settings in the session file, but after loading the **Model** tab shows its starting settings, not the saved ones (W31): set the backend and kernel again before you choose **Train Model**; see [Sessions and audit log](../guide/more/sessions.md). The desktop app's text logs in `./logs` are not read; the session's audit log replaces them.
