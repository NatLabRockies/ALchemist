"""What the desktop app's session sync would lose (D1, D2).

The desktop app rebuilds the session's search space and experiment table from
its own widgets, which only know names, types and ranges. Anything else in the
session would be dropped silently: observed variables, change difficulty,
derived variables, input constraints, work-queue provenance and
split-plot bookkeeping. The cleaning step also drops any row it cannot coerce
to numbers; only a dropped row that carries a result counts as a loss. This module lists those losses so the sync can refuse instead.

It must stay free of tkinter / customtkinter so it can be tested headless.

A session the desktop app made itself is never refused. The columns the desktop
writes into its experiment table are the targets, ``Noise``, ``Iteration`` and
``Reason`` (see ``save_new_point``, ``toggle_noise`` and the initial design in
``ui/ui.py``), plus whatever the user's own CSV carried in; none of them is a
loss. A variable missing from the desktop list is not a loss either: desktop
load fills that list from the session, so a missing one was deleted on purpose.
"""
from __future__ import annotations

import pandas as pd

from alchemist_core.data.experiment_manager import PLOT_COLUMNS, PROVENANCE_COL
from alchemist_core.data.search_space import changes_of, role_of

# Columns only the web app or the work queue write; the desktop has no place
# to keep them.
_NON_DESKTOP_COLUMNS = (PROVENANCE_COL,) + tuple(PLOT_COLUMNS)

# Columns the cleaning in ``_sync_data_to_session`` treats specially.
_META = {"Noise", "Iteration", "Reason"}

# Reasons are plain strings (the interface the caller shows). Two kinds need
# their own advice in the dialog, so they carry a marker format_blockers reads.
ROW_REASON_MARK = " (fixable here)"
CHECK_FAILED_PREFIX = "the desktop app could not check the session"


def _would_be_dropped(exp_df: pd.DataFrame, target_columns, categorical) -> int:
    """Rows ``_sync_data_to_session`` would drop. Mirrors its cleaning exactly:
    numeric coercion of the targets and of every non-categorical feature
    column, then ``dropna`` over all columns except Noise and Reason."""
    df = exp_df.copy()
    targets = set(target_columns)
    metadata = targets | _META
    if "Iteration" in df.columns:
        df["Iteration"] = pd.to_numeric(df["Iteration"], errors="coerce").fillna(0)
    for col in targets:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in df.columns:
        if col in metadata:
            continue
        if col in categorical:
            df[col] = df[col].astype(str)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    required = [c for c in df.columns if c not in ("Noise", "Reason")]
    dropped = df[required].isna().any(axis=1)
    # A row with no result at all is an unmeasured suggestion (the desktop
    # commits those with an empty Output); dropping it loses no data, and it
    # is what the cleaning has always done. Only a row that carries a result
    # is a loss.
    present = [c for c in targets if c in df.columns]
    measured = df[present].notna().any(axis=1) if present else pd.Series(False, index=df.index)
    return int((dropped & measured).sum())


def sync_blockers(session, exp_df, ui_variable_names, target_columns, ui_variables=None) -> list[str]:
    """Reasons the sync would lose data (empty list = safe to sync).

    ``ui_variables`` is the desktop's variable list (dicts); the sync turns its
    categorical entries into string columns, so the row check needs them.
    Without it the session's own categorical variables, restricted to
    ``ui_variable_names``, stand in.
    """
    reasons: list[str] = []
    space = session.search_space

    for var in space.variables:
        name = var["name"]
        if role_of(var) == "observed":
            reasons.append(f"'{name}' is an observed variable")
        elif changes_of(var) != "easy":
            level = str(changes_of(var)).replace("_", " ")
            reasons.append(f"'{name}' is marked {level} to change")

    if space.has_derived_variables():
        reasons.append(
            "the session has derived variables: " + ", ".join(space.get_derived_variable_names()))
    if getattr(space, "constraints", None):
        reasons.append("the session has input constraints")
    # Outcome constraints live on the session, not the search space; the sync
    # never touches them, so they are not a loss.

    if exp_df is not None and len(exp_df):
        extra = [c for c in _NON_DESKTOP_COLUMNS if c in exp_df.columns]
        if extra:
            reasons.append(
                "the experiment table has columns the desktop app cannot keep: " + ", ".join(extra))

        if ui_variables is not None:
            categorical = {v["name"] for v in ui_variables if v.get("type") == "categorical"}
        else:
            ui_names = set(ui_variable_names)
            categorical = {v["name"] for v in space.variables
                           if v.get("type") == "categorical" and v["name"] in ui_names}
        n_drop = _would_be_dropped(exp_df, target_columns, categorical)
        if n_drop:
            reasons.append(
                f"{n_drop} experiment row(s) have a result but an empty or non-numeric "
                f"input, so they would be dropped{ROW_REASON_MARK}")
    return reasons


def format_blockers(reasons: list[str]) -> str:
    """The dialog text for a refused sync, with advice that fits each kind."""
    failed = [r for r in reasons if r.startswith(CHECK_FAILED_PREFIX)]
    rows = [r.replace(ROW_REASON_MARK, "") for r in reasons if r.endswith(ROW_REASON_MARK)]
    features = [r for r in reasons if r not in failed and not r.endswith(ROW_REASON_MARK)]

    def bullets(items):
        return "\n".join(f"- {r}" for r in items)

    parts = []
    if features:
        parts.append(
            "This session uses features the desktop app can't keep:\n\n"
            f"{bullets(features)}\n\n"
            "Use the web app (alchemist-web) for this session instead.")
    if rows:
        parts.append(
            f"{bullets(rows)}\n\n"
            "Fill in or delete the empty cells in the experiment table, then try again.")
    if failed:
        parts.append(
            f"{bullets(failed)}\n\n"
            "This is a bug in the desktop app. Try again; if it keeps happening, "
            "save a copy of your data from the experiment table and report it.")
    return "\n\n".join(parts) + "\n\nNothing was changed."
