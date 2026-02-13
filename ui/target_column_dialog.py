"""
Target Column Selection Dialog

Allows users to select which column(s) in their CSV should be treated as optimization targets.
Supports both single-objective and multi-objective optimization.

In multi-objective mode, users assign each column a role:
- Variable: input feature (X) for the surrogate model
- Target: objective (Y) to optimize
- Drop: column is ignored entirely
"""

import customtkinter as ctk
from typing import Dict, List, Optional, Tuple, Union
import tkinter as tk


class TargetColumnDialog(ctk.CTkToplevel):
    """
    Dialog for selecting target columns when loading experimental data.

    Features:
    - Single/Multi-objective mode toggle
    - Column selection (dropdown for single, segmented role assignment for multi)
    - Validation before confirming
    """

    def __init__(self, parent, available_columns: List[str], default_column: str = None):
        """
        Initialize the target column selection dialog.

        Args:
            parent: Parent window
            available_columns: List of column names available in the CSV
            default_column: Default column to select (if it exists in available_columns)
        """
        super().__init__(parent)

        self.title("Select Target Column(s)")
        self.geometry("550x550")
        self.resizable(True, True)
        self.minsize(500, 450)

        # Make dialog modal
        self.transient(parent)
        self.grab_set()

        # Store data
        self.available_columns = available_columns
        self.default_column = default_column if default_column in available_columns else None
        self.result = None  # Will store selected column(s) when confirmed

        # UI state
        self.mode = "single"  # "single" or "multi"
        self.checkbox_vars = {}  # For multi-objective mode (legacy, unused in new UI)
        self.role_vars = {}  # For multi-objective per-column role assignment

        self._create_ui()

        # Center the dialog
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - (self.winfo_width() // 2)
        y = parent.winfo_y() + (parent.winfo_height() // 2) - (self.winfo_height() // 2)
        self.geometry(f"+{x}+{y}")
        
    def _create_ui(self):
        """Create the dialog UI elements."""
        # Header
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        ctk.CTkLabel(
            header_frame,
            text="Select Target Column(s)",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w")
        
        ctk.CTkLabel(
            header_frame,
            text="Choose which column(s) to optimize:",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        ).pack(anchor="w", pady=(5, 0))
        
        # Mode selector (Single vs Multi-objective)
        mode_frame = ctk.CTkFrame(self)
        mode_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            mode_frame,
            text="Optimization Mode:",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(side="left", padx=(10, 20))
        
        self.mode_var = ctk.StringVar(value="single")
        
        self.single_radio = ctk.CTkRadioButton(
            mode_frame,
            text="Single-Objective",
            variable=self.mode_var,
            value="single",
            command=self._on_mode_change
        )
        self.single_radio.pack(side="left", padx=10)
        
        self.multi_radio = ctk.CTkRadioButton(
            mode_frame,
            text="Multi-Objective",
            variable=self.mode_var,
            value="multi",
            command=self._on_mode_change
        )
        self.multi_radio.pack(side="left", padx=10)
        
        # Column selection area (content changes based on mode)
        self.selection_frame = ctk.CTkFrame(self)
        self.selection_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self._update_selection_ui()
        
        # Buttons
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack(fill="x", padx=20, pady=(10, 20))
        
        ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=self._on_cancel,
            width=100
        ).pack(side="right", padx=(10, 0))
        
        ctk.CTkButton(
            button_frame,
            text="Confirm",
            command=self._on_confirm,
            width=100
        ).pack(side="right")
        
    def _on_mode_change(self):
        """Handle mode change between single and multi-objective."""
        self.mode = self.mode_var.get()
        self._update_selection_ui()
        
    def _update_selection_ui(self):
        """Update the column selection UI based on current mode."""
        # Clear existing widgets
        for widget in self.selection_frame.winfo_children():
            widget.destroy()
            
        if self.mode == "single":
            self._create_single_objective_ui()
        else:
            self._create_multi_objective_ui()
            
    def _create_single_objective_ui(self):
        """Create UI for single-objective mode (dropdown)."""
        ctk.CTkLabel(
            self.selection_frame,
            text="Select target column:",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", padx=20, pady=(20, 10))
        
        # Dropdown menu
        self.column_var = ctk.StringVar(value=self.default_column or self.available_columns[0])
        
        self.column_dropdown = ctk.CTkOptionMenu(
            self.selection_frame,
            variable=self.column_var,
            values=self.available_columns,
            width=400
        )
        self.column_dropdown.pack(padx=20, pady=10)
        
        # Info text
        info_frame = ctk.CTkFrame(self.selection_frame, fg_color="transparent")
        info_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        ctk.CTkLabel(
            info_frame,
            text="💡 Tip: This column will be maximized or minimized during optimization.",
            font=ctk.CTkFont(size=11),
            text_color="gray",
            wraplength=400,
            justify="left"
        ).pack(anchor="w")
        
    def _create_multi_objective_ui(self):
        """Create UI for multi-objective mode (per-column role assignment)."""
        ctk.CTkLabel(
            self.selection_frame,
            text="Assign each column a role:",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", padx=20, pady=(15, 5))

        ctk.CTkLabel(
            self.selection_frame,
            text="Variable = input feature, Target = objective to optimize, Drop = ignore",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        ).pack(anchor="w", padx=20, pady=(0, 10))

        # Scrollable frame for per-column role rows
        role_frame = ctk.CTkScrollableFrame(
            self.selection_frame,
            height=250
        )
        role_frame.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        # Column header row
        header_row = ctk.CTkFrame(role_frame, fg_color="transparent")
        header_row.pack(fill="x", pady=(0, 5))
        ctk.CTkLabel(
            header_row, text="Column", font=ctk.CTkFont(size=11, weight="bold"), width=160, anchor="w"
        ).pack(side="left", padx=(10, 20))
        ctk.CTkLabel(
            header_row, text="Role", font=ctk.CTkFont(size=11, weight="bold"), anchor="w"
        ).pack(side="left")

        # Create a segmented button per column
        self.role_vars = {}
        for col in self.available_columns:
            row = ctk.CTkFrame(role_frame, fg_color="transparent")
            row.pack(fill="x", pady=3)

            ctk.CTkLabel(
                row, text=col, font=ctk.CTkFont(size=12), width=160, anchor="w"
            ).pack(side="left", padx=(10, 20))

            var = ctk.StringVar(value="Variable")
            seg = ctk.CTkSegmentedButton(
                row,
                values=["Variable", "Target", "Drop"],
                variable=var,
                width=240
            )
            seg.pack(side="left")
            self.role_vars[col] = var

        # Validation hint (updated dynamically would add complexity; keep static)
        info_frame = ctk.CTkFrame(self.selection_frame, fg_color="transparent")
        info_frame.pack(fill="x", padx=20, pady=(5, 10))

        ctk.CTkLabel(
            info_frame,
            text="Requires at least 2 Targets and 1 Variable.",
            font=ctk.CTkFont(size=11),
            text_color="gray",
            wraplength=450,
            justify="left"
        ).pack(anchor="w")
        
    def _show_error(self, title: str, message: str):
        """Show an error popup centered on this dialog."""
        error_dialog = ctk.CTkToplevel(self)
        error_dialog.title(title)
        error_dialog.geometry("380x150")
        error_dialog.transient(self)
        error_dialog.grab_set()

        ctk.CTkLabel(
            error_dialog,
            text=f"⚠️ {title}",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(20, 10))

        ctk.CTkLabel(
            error_dialog,
            text=message,
            font=ctk.CTkFont(size=12)
        ).pack(pady=10)

        ctk.CTkButton(
            error_dialog,
            text="OK",
            command=error_dialog.destroy,
            width=100
        ).pack(pady=10)

        error_dialog.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() // 2) - (error_dialog.winfo_width() // 2)
        y = self.winfo_y() + (self.winfo_height() // 2) - (error_dialog.winfo_height() // 2)
        error_dialog.geometry(f"+{x}+{y}")

    def _on_confirm(self):
        """Handle confirm button click."""
        if self.mode == "single":
            # Single-objective: return selected column as string
            selected = self.column_var.get()
            if selected:
                self.result = selected
                self.destroy()
        else:
            # Multi-objective: collect roles from segmented buttons
            targets = [col for col, var in self.role_vars.items() if var.get() == "Target"]
            variables = [col for col, var in self.role_vars.items() if var.get() == "Variable"]

            if len(targets) < 2:
                self._show_error(
                    "Invalid Selection",
                    "Please assign at least 2 columns as Target\nfor multi-objective optimization."
                )
                return

            if len(variables) < 1:
                self._show_error(
                    "Invalid Selection",
                    "Please assign at least 1 column as Variable\n(input feature for the model)."
                )
                return

            self.result = {"targets": targets, "variables": variables}
            self.destroy()
            
    def _on_cancel(self):
        """Handle cancel button click."""
        self.result = None
        self.destroy()
        
    def get_result(self) -> Optional[Union[str, Dict[str, List[str]]]]:
        """
        Get the user's selection.

        Returns:
            - str for single-objective (column name)
            - dict ``{"targets": [...], "variables": [...]}`` for multi-objective
            - None if cancelled
        """
        return self.result


def show_target_column_dialog(parent, available_columns: List[str],
                              default_column: str = None) -> Optional[Union[str, Dict[str, List[str]]]]:
    """
    Show target column selection dialog and return user's choice.

    Args:
        parent: Parent window
        available_columns: List of column names available in the CSV
        default_column: Default column to select (if it exists)

    Returns:
        - str for single-objective
        - dict ``{"targets": [...], "variables": [...]}`` for multi-objective
        - None if cancelled
    """
    dialog = TargetColumnDialog(parent, available_columns, default_column)
    parent.wait_window(dialog)
    return dialog.get_result()
