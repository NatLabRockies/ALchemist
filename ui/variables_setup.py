import customtkinter as ctk
from customtkinter import filedialog
import json, csv, os
from skopt.space import Real, Integer, Categorical
from alchemist_core.data.search_space import SearchSpace
from tkinter import StringVar
import tkinter.messagebox as messagebox
from tksheet import Sheet  # requires tksheet package

# -----------------------------------------------------------------------------
# A new toplevel for editing categorical values using tksheet.
class CategoricalEditorWindow(ctk.CTkToplevel):
    def __init__(self, parent, initial_values=None, callback=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.title("Edit Categorical Values")
        self.geometry("300x300")
        self.lift()
        self.focus_force()
        self.grab_set()
        self.callback = callback  # function to call with the new list
        
        # If there are initial values, use them; otherwise, create 10 blank rows.
        if initial_values and len(initial_values) > 0:
            data = [[v] for v in initial_values] + [[""] for _ in range(10 - len(initial_values))]
        else:
            data = [[""] for _ in range(10)]

        
        # Create a Sheet widget with one column.
        self.sheet = Sheet(self, data=data, headers=["Value"])
        self.sheet.enable_bindings()  # enable editing and navigation
        self.sheet.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # A save button to commit changes.
        btn_save = ctk.CTkButton(self, text="Save", command=self.on_save)
        btn_save.grid(row=1, column=0, pady=10)
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
    
    def on_save(self):
        # Get all data from the sheet; expect a list of lists (one column each)
        raw_data = self.sheet.get_sheet_data()
        
        # More robust filtering to handle different data formats
        values = []
        if raw_data:
            for row in raw_data:
                if row and len(row) > 0:
                    val = str(row[0]).strip() if row[0] is not None else ""
                    if val:
                        values.append(val)
        
        if self.callback:
            self.callback(values)
        self.destroy()

# -----------------------------------------------------------------------------
# Row widget representing a single variable.
class SpaceVariableRow(ctk.CTkFrame):
    def __init__(self, master, select_callback=None, **kwargs):
        super().__init__(master, **kwargs)
        self.select_callback = select_callback
        self.configure(border_color="gray", border_width=1, corner_radius=5)
        
        # Initialize storage for categorical values and discrete allowed values.
        self.categorical_values = []

        # Entry for variable name.
        self.var_name_entry = ctk.CTkEntry(self, placeholder_text="Variable Name")
        self.var_name_entry.grid(row=0, column=0, padx=5, pady=5)
        self.var_name_entry.bind("<FocusIn>", lambda e: self.on_click(e))
        
        # Dropdown for selecting type.
        self.type_var = StringVar()
        self.type_dropdown = ctk.CTkOptionMenu(
            self,
            variable=self.type_var,
            values=["Real", "Integer", "Categorical", "Discrete"],
            command=self.on_type_change
        )
        self.type_dropdown.grid(row=0, column=1, padx=5, pady=5)
        self.type_var.set("Real")
        # (Do not bind the OptionMenu so its dropdown works correctly.)
        
        # For Real type: entries for min and max.
        self.min_entry = ctk.CTkEntry(self, placeholder_text="Min")
        self.max_entry = ctk.CTkEntry(self, placeholder_text="Max")
        self.min_entry.grid(row=0, column=2, padx=5, pady=5)
        self.max_entry.grid(row=0, column=3, padx=5, pady=5)
        self.min_entry.bind("<FocusIn>", lambda e: self.on_click(e))
        self.max_entry.bind("<FocusIn>", lambda e: self.on_click(e))
        
        # For Categorical type: a button to open the editor.
        self.edit_button = ctk.CTkButton(self, text="Edit Values", command=self.open_categorical_editor)
        self.edit_button.bind("<FocusIn>", lambda e: self.on_click(e))
        # Not gridded by default.

        # For Discrete type: an entry for comma-separated allowed values.
        self.discrete_entry = ctk.CTkEntry(self, placeholder_text="e.g. 80, 280", width=200)
        self.discrete_entry.bind("<FocusIn>", lambda e: self.on_click(e))
        # Not gridded by default.

        # Bind the row itself (empty areas) so that clicking selects the row.
        self.bind("<Button-1>", self.on_click)
    
    def on_click(self, event):
        if self.select_callback:
            self.select_callback(self)
    
    def on_type_change(self, selection):
        if selection in ["Real", "Integer"]:
            self.min_entry.grid(row=0, column=2, padx=5, pady=5)
            self.max_entry.grid(row=0, column=3, padx=5, pady=5)
            self.edit_button.grid_forget()
            self.discrete_entry.grid_forget()
        elif selection == "Categorical":
            self.min_entry.grid_forget()
            self.max_entry.grid_forget()
            self.edit_button.grid(row=0, column=2, padx=5, pady=5, columnspan=2)
            self.discrete_entry.grid_forget()
        elif selection == "Discrete":
            self.min_entry.grid_forget()
            self.max_entry.grid_forget()
            self.edit_button.grid_forget()
            self.discrete_entry.grid(row=0, column=2, padx=5, pady=5, columnspan=2)
    
    def open_categorical_editor(self):
        # Open the CategoricalEditorWindow, passing current values.
        CategoricalEditorWindow(self, initial_values=self.categorical_values, callback=self.set_categorical_values)
    
    def set_categorical_values(self, values):
        self.categorical_values = values
        # Update the button text to show how many values are set.
        self.edit_button.configure(text=f"Edit Values ({len(values)})")
    
    def get_data(self):
        name = self.var_name_entry.get().strip()
        if not name:
            return None
        typ = self.type_var.get()
        if typ in ["Real", "Integer"]:
            try:
                min_val = float(self.min_entry.get())
                max_val = float(self.max_entry.get())
            except ValueError:
                return None
            return {"name": name, "type": typ, "min": min_val, "max": max_val}
        elif typ == "Categorical":
            if not self.categorical_values:
                return None
            return {"name": name, "type": "Categorical", "values": self.categorical_values}
        elif typ == "Discrete":
            raw = self.discrete_entry.get().strip()
            if not raw:
                return None
            try:
                allowed = [float(v.strip()) for v in raw.split(",") if v.strip()]
            except ValueError:
                return None
            if len(allowed) < 2:
                return None
            allowed = sorted(set(allowed))
            return {"name": name, "type": "Discrete", "allowed_values": allowed}
    
    def populate(self, data):
        self.var_name_entry.delete(0, ctk.END)
        self.var_name_entry.insert(0, data.get("name", ""))
        self.type_var.set(data.get("type", "Real"))
        self.on_type_change(data.get("type", "Real"))
        if data.get("type") in ["Real", "Integer"]:
            self.min_entry.delete(0, ctk.END)
            self.min_entry.insert(0, str(data.get("min", "")))
            self.max_entry.delete(0, ctk.END)
            self.max_entry.insert(0, str(data.get("max", "")))
        elif data.get("type") == "Categorical":
            self.categorical_values = data.get("values", [])
            self.edit_button.configure(text=f"Edit Values ({len(self.categorical_values)})")
        elif data.get("type") == "Discrete":
            self.discrete_entry.delete(0, ctk.END)
            self.discrete_entry.insert(0, ", ".join(str(v) for v in data.get("allowed_values", [])))
    
    def clear(self):
        self.var_name_entry.delete(0, ctk.END)
        self.min_entry.delete(0, ctk.END)
        self.max_entry.delete(0, ctk.END)
        self.categorical_values = []
        self.edit_button.configure(text="Edit Values")
        self.discrete_entry.delete(0, ctk.END)
    
    def set_selected(self, selected: bool):
        if selected:
            self.configure(border_color="blue", border_width=2)
        else:
            self.configure(border_color="gray", border_width=1)

# -----------------------------------------------------------------------------
# Top-level window for setting up the variable space.
class SpaceSetupWindow(ctk.CTkToplevel):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.title("Search Space Setup")
        self.geometry("800x500")
        self.selected_row = None  # currently selected row
        self.categorical_variables = []  # Store categorical variable names

        # Main container with left (rows) and right (controls) panels.
        main_container = ctk.CTkFrame(self)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left panel: header and scrollable area.
        left_frame = ctk.CTkFrame(main_container)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Header using pack.
        header = ctk.CTkFrame(left_frame)
        header.pack(fill="x", padx=5, pady=(0, 5))
        lbl_name = ctk.CTkLabel(header, text="Variable Name", width=120)
        lbl_name.pack(side="left", padx=5)
        lbl_type = ctk.CTkLabel(header, text="Type", width=120)
        lbl_type.pack(side="left", padx=5)
        lbl_params = ctk.CTkLabel(header, text="Parameters", width=240)
        lbl_params.pack(side="left", padx=5)
        
        # Scrollable frame for variable rows.
        self.scrollable_frame = ctk.CTkScrollableFrame(left_frame, height=350)
        self.scrollable_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.rows_frame = ctk.CTkFrame(self.scrollable_frame)
        self.rows_frame.pack(fill="x", expand=True)
        self.rows = []
        
        # Right panel: control buttons.
        control_panel = ctk.CTkFrame(main_container, width=150)
        control_panel.pack(side="right", fill="y")
        btn_add = ctk.CTkButton(control_panel, text="Add Variable", command=self.add_row)
        btn_add.pack(pady=5, fill="x")
        btn_delete = ctk.CTkButton(control_panel, text="Delete Row", command=self.delete_selected_row)
        btn_delete.pack(pady=5, fill="x")
        btn_clear = ctk.CTkButton(control_panel, text="Clear Row", command=self.clear_selected_row)
        btn_clear.pack(pady=5, fill="x")
        btn_move_up = ctk.CTkButton(control_panel, text="Move Up", command=self.move_selected_up)
        btn_move_up.pack(pady=5, fill="x")
        btn_move_down = ctk.CTkButton(control_panel, text="Move Down", command=self.move_selected_down)
        btn_move_down.pack(pady=5, fill="x")
        btn_load = ctk.CTkButton(control_panel, text="Load from File", command=self.load_from_file)
        btn_load.pack(pady=5, fill="x")
        btn_save_file = ctk.CTkButton(control_panel, text="Save to File", command=self.save_to_file)
        btn_save_file.pack(pady=5, fill="x")
        btn_save_close = ctk.CTkButton(control_panel, text="Save & Close", command=self.save_and_close)
        btn_save_close.pack(pady=5, fill="x")
        
        # Load existing data if available; otherwise, add one row.
        if hasattr(self.master, "variable_space_data") and self.master.variable_space_data:
            self.load_data(self.master.variable_space_data)
        else:
            self.add_row()
        
        # Bring this window to the front.
        self.lift()
        self.focus_force()
        self.grab_set()
    
    def add_row(self):
        row = SpaceVariableRow(self.rows_frame, select_callback=self.select_row)
        row.pack(fill="x", pady=2)
        self.rows.append(row)
    
    def select_row(self, row_widget):
        if self.selected_row:
            self.selected_row.set_selected(False)
        self.selected_row = row_widget
        self.selected_row.set_selected(True)
    
    def refresh_rows(self):
        for row in self.rows:
            row.pack_forget()
        for row in self.rows:
            row.pack(fill="x", pady=2)
    
    def move_selected_up(self):
        if self.selected_row is None:
            return
        idx = self.rows.index(self.selected_row)
        if idx > 0:
            self.rows[idx], self.rows[idx-1] = self.rows[idx-1], self.rows[idx]
            self.refresh_rows()
    
    def move_selected_down(self):
        if self.selected_row is None:
            return
        idx = self.rows.index(self.selected_row)
        if idx < len(self.rows) - 1:
            self.rows[idx], self.rows[idx+1] = self.rows[idx+1], self.rows[idx]
            self.refresh_rows()
    
    def delete_selected_row(self):
        if self.selected_row is None:
            return
        self.rows.remove(self.selected_row)
        self.selected_row.destroy()
        self.selected_row = None
    
    def clear_selected_row(self):
        if self.selected_row is None:
            return
        self.selected_row.clear()
    
    def get_data(self):
        data_list = []
        for row in self.rows:
            d = row.get_data()
            if d is not None:
                data_list.append(d)
        return data_list
    
    def load_data(self, data_list):
        for row in self.rows:
            row.destroy()
        self.rows = []
        for d in data_list:
            row = SpaceVariableRow(self.rows_frame, select_callback=self.select_row)
            row.populate(d)
            row.pack(fill="x", pady=2)
            self.rows.append(row)
    
    def save_to_file(self):
        data = self.get_data()
        if not data:
            messagebox.showerror("Error", "No valid variable data to save.")
            return
        file_path = filedialog.asksaveasfilename(
            title="Save Variable Space",
            defaultextension=".json",
            filetypes=(("JSON files", "*.json"), ("CSV files", "*.csv"))
        )
        if not file_path:
            return
        try:
            if file_path.lower().endswith(".json"):
                with open(file_path, "w") as f:
                    json.dump(data, f, indent=2)
            elif file_path.lower().endswith(".csv"):
                with open(file_path, "w", newline="") as f:
                    fieldnames = ["Variable", "Type", "Min", "Max", "Values"]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for d in data:
                        if d.get("type") == "Categorical":
                            values_str = ", ".join(d.get("values", []))
                        elif d.get("type") == "Discrete":
                            values_str = ", ".join(str(v) for v in d.get("allowed_values", []))
                        else:
                            values_str = ""
                        row_data = {
                            "Variable": d.get("name", ""),
                            "Type": d.get("type", ""),
                            "Min": d.get("min", ""),
                            "Max": d.get("max", ""),
                            "Values": values_str
                        }
                        writer.writerow(row_data)
            messagebox.showinfo("Success", f"Saved variable space to:\n{os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))
    
    def load_from_file(self):
        file_path = filedialog.askopenfilename(
            title="Load Variable Space",
            filetypes=(("JSON files", "*.json"), ("CSV files", "*.csv"))
        )
        if not file_path:
            return
        try:
            if file_path.lower().endswith(".json"):
                with open(file_path, "r") as f:
                    data = json.load(f)
            elif file_path.lower().endswith(".csv"):
                data = []
                with open(file_path, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        typ = row.get("Type", "").strip()
                        variable_name = row.get("Variable", "").strip()
                        
                        # Skip rows with empty variable names
                        if not variable_name:
                            continue
                            
                        if typ == "Real":
                            try:
                                min_val = float(row.get("Min", "0").strip() or "0")
                                max_val = float(row.get("Max", "1").strip() or "1")
                            except (ValueError, TypeError):
                                print(f"Warning: Invalid min/max values for Real variable '{variable_name}'. Using defaults 0, 1.")
                                min_val, max_val = 0.0, 1.0
                            d = {
                                "name": variable_name,
                                "type": "Real",
                                "min": min_val,
                                "max": max_val
                            }
                        elif typ == "Integer":
                            try:
                                min_val = int(float(row.get("Min", "0").strip() or "0"))
                                max_val = int(float(row.get("Max", "1").strip() or "1"))
                            except (ValueError, TypeError):
                                print(f"Warning: Invalid min/max values for Integer variable '{variable_name}'. Using defaults 0, 1.")
                                min_val, max_val = 0, 1
                            d = {
                                "name": variable_name,
                                "type": "Integer",
                                "min": min_val,
                                "max": max_val
                            }
                        elif typ == "Categorical":
                            values_str = row.get("Values", "").strip()
                            if values_str:
                                values = [v.strip() for v in values_str.split(",") if v.strip()]
                            else:
                                values = []
                            
                            if not values:
                                print(f"Warning: No values found for Categorical variable '{variable_name}'. Skipping.")
                                continue
                                
                            d = {
                                "name": variable_name,
                                "type": "Categorical",
                                "values": values
                            }
                        elif typ == "Discrete":
                            values_str = row.get("Values", "").strip()
                            if not values_str:
                                print(f"Warning: No allowed_values found for Discrete variable '{variable_name}'. Skipping.")
                                continue
                            try:
                                allowed = sorted(set(float(v.strip()) for v in values_str.split(",") if v.strip()))
                            except ValueError:
                                print(f"Warning: Invalid allowed_values for Discrete variable '{variable_name}'. Skipping.")
                                continue
                            if len(allowed) < 2:
                                print(f"Warning: Discrete variable '{variable_name}' needs at least 2 allowed values. Skipping.")
                                continue
                            d = {
                                "name": variable_name,
                                "type": "Discrete",
                                "allowed_values": allowed
                            }
                        else:
                            print(f"Warning: Unknown variable type '{typ}' for variable '{variable_name}'. Skipping.")
                            continue
                        data.append(d)
            self.load_data(data)
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
    
    def save_and_close(self):
        """Save the search space and update the categorical variables list."""
        data = self.get_data()
        if not data:
            messagebox.showerror("Error", "No valid variable data to save.")
            return
        self.master.variable_space_data = data
        self.search_space = []
        self.categorical_variables = []  # Reset the list

        for d in data:
            if d["type"] == "Real":
                self.search_space.append(Real(d["min"], d["max"], name=d["name"]))
            elif d["type"] == "Integer":
                self.search_space.append(Integer(d["min"], d["max"], name=d["name"]))
            elif d["type"] == "Categorical":
                self.search_space.append(Categorical(d["values"], name=d["name"]))
                self.categorical_variables.append(d["name"])  # Add to categorical list
            elif d["type"] == "Discrete":
                sorted_vals = sorted(float(v) for v in d["allowed_values"])
                self.search_space.append(Categorical(sorted_vals, name=d["name"]))

        print("Saved search space:")
        for obj in self.search_space:
            print(obj)
        print("Categorical variables:", self.categorical_variables)
        
        # Update the main UI if it has the necessary methods
        if hasattr(self.master, 'search_space'):
            # Store the skopt-compatible dimensions for legacy UI code
            self.master.search_space = self.search_space
        # Always create/update the SearchSpace manager on the master so the
        # OptimizationSession and ExperimentManager get a consistent object.
        try:
            ss = SearchSpace().from_dict(self.master.variable_space_data)
            self.master.search_space_manager = ss
            # Keep the skopt-compatible reference too
            try:
                self.master.search_space = ss.to_skopt()
            except Exception:
                self.master.search_space = ss
            # Ensure experiment manager is updated
            if hasattr(self.master, 'experiment_manager'):
                try:
                    self.master.experiment_manager.set_search_space(ss)
                except Exception:
                    pass
            # If there's an active OptimizationSession, update it as well
            if hasattr(self.master, 'session') and self.master.session is not None:
                try:
                    self.master.session.search_space = ss
                    self.master.session.experiment_manager.set_search_space(ss)
                    # emit an event so other UI components can react
                    try:
                        self.master.session.events.emit('search_space_loaded', {'source': 'ui'})
                    except Exception:
                        pass
                except Exception:
                    pass
        except Exception:
            pass

        # Let the main app know the search space changed so it can sync to session
        if hasattr(self.master, '_sync_data_to_session'):
            try:
                self.master._sync_data_to_session()
            except Exception:
                pass

        if hasattr(self.master, '_update_ui_state'):
            try:
                self.master._update_ui_state()
            except Exception:
                pass

        if hasattr(self.master, 'update_variables_display'):
            self.master.update_variables_display()
        elif hasattr(self.master, 'var_sheet'):
            # Directly update the variable sheet in the main UI
            self._update_main_ui_variables()
            
        self.destroy()
    
    def _update_main_ui_variables(self):
        """Update the main UI variable sheet with current data."""
        try:
            # Prepare data for the main UI sheet
            sheet_data = []
            for d in self.master.variable_space_data:
                if d["type"] in ["Real", "Integer"]:
                    row = [
                        d.get("name", ""),
                        d.get("type", ""),
                        str(d.get("min", "")),
                        str(d.get("max", "")),
                        ""  # Values column empty for Real/Integer
                    ]
                elif d["type"] == "Categorical":
                    row = [
                        d.get("name", ""),
                        d.get("type", ""),
                        "",  # Min column empty for Categorical
                        "",  # Max column empty for Categorical
                        ", ".join(d.get("values", []))  # Values column
                    ]
                elif d["type"] == "Discrete":
                    row = [
                        d.get("name", ""),
                        d.get("type", ""),
                        "",  # Min column empty for Discrete
                        "",  # Max column empty for Discrete
                        ", ".join(str(v) for v in d.get("allowed_values", []))  # Values column
                    ]
                sheet_data.append(row)
            
            # Update the main UI sheet
            self.master.var_sheet.set_sheet_data(sheet_data)
            self.master.var_sheet.set_all_column_widths()
            
            # Enable buttons that depend on variables being loaded
            if hasattr(self.master, 'load_exp_button'):
                self.master.load_exp_button.configure(state='normal')
            if hasattr(self.master, 'gen_template_button'):
                self.master.gen_template_button.configure(state='normal')
            # Clustering UI removed; no cluster_switch configuration needed
                
        except Exception as e:
            print(f"Error updating main UI variables: {e}")

# -----------------------------------------------------------------------------
# Main application window.
if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    app = ctk.CTk()
    app.geometry("600x400")
    app.title("Main Application")
    
    # The main app holds the variable space data.
    app.variable_space_data = None
    
    def open_space_setup():
        SpaceSetupWindow(app)
    
    open_btn = ctk.CTkButton(app, text="Define Search Space", command=open_space_setup)
    open_btn.pack(pady=20)
    
    app.mainloop()
