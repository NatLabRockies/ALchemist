import customtkinter as ctk
from customtkinter import filedialog
import mplcursors
from tabulate import tabulate
import tksheet
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from skopt.space import Categorical, Integer, Real
import numpy as np
from skopt.sampler import Lhs, Sobol, Hammersly
import tkinter as tk
import os

from ui.variables_setup import SpaceSetupWindow
from ui.gpr_panel import GaussianProcessPanel
from ui.acquisition_panel import AcquisitionPanel

# DEPRECATED: Pool visualization (will be removed in v0.3.0)
from ui.pool_viz import generate_pool, plot_pool

# Deprecated imports - these functions are no longer used in the modern UI
# from logic.clustering import cluster_pool
# from logic.emoc import select_EMOC
# from logic.optimization import select_optimize

from alchemist_core.data.search_space import SearchSpace
from alchemist_core.data.experiment_manager import ExperimentManager

# UI-layer utilities
from ui.experiment_logger import ExperimentLogger

# Import new session API
from alchemist_core.session import OptimizationSession
from alchemist_core.events import EventEmitter

plt.rcParams['savefig.dpi'] = 600
 


# ============================================================
# UI Helper Functions
# ============================================================

# ============================================================
# Main Application
# ============================================================

class ALchemistApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self._configure_window()

        # Create the menu bar
        self._create_menu_bar()

        # State variables for data management using our new classes
        self.search_space_manager = SearchSpace()
        self.experiment_manager = ExperimentManager()
        self.next_point = None  # Keep as DataFrame for visualization
        
        # Legacy variables for compatibility during transition
        self.var_df = None
        self.exp_df = pd.DataFrame()
        self.search_space = None
        self.pool = None  # DEPRECATED: Pool visualization (initialized when variables loaded)
        # Clustering removed; no kmeans placeholder
        
        # NEW: Create OptimizationSession for session-based API
        # This provides a parallel code path alongside the existing direct logic calls
        self.session = OptimizationSession()
        
        # Connect session events to UI updates
        self.session.events.on('progress', self._on_session_progress)
        self.session.events.on('model_trained', self._on_session_model_trained)
        self.session.events.on('model_retrained', self._on_session_model_retrained)
        self.session.events.on('suggestions_ready', self._on_session_suggestions)

        # Build essential UI sections
        self._create_vertical_frame()
        self._create_variable_management_frame()
        self._create_experiment_management_frame()
        self._create_visualization_frame()
        
        # Start in tabbed layout by default
        self.using_tabs = True
        
        # Create tabbed interface
        self.tab_view = ctk.CTkTabview(self)
        self.tab_view.pack(side='right', fill='both', padx=10, pady=10)
        self.tab_view.configure(width=300)
        
        # Add tabs
        self.tab_view.add("Model")
        self.tab_view.add("Acquisition")
        
        # Set the default tab
        self.tab_view.set("Model")
        
        # Create panels inside tabs
        self.model_frame = GaussianProcessPanel(self.tab_view.tab("Model"), self)
        self.model_frame.pack(fill='both', expand=True)
        
        self.acquisition_panel = AcquisitionPanel(self.tab_view.tab("Acquisition"), self)
        self.acquisition_panel.pack(fill='both', expand=True)
        
        # Set initial UI state based on data load
        self._update_ui_state()
        
        # Initialize the experiment logger
        self.experiment_logger = ExperimentLogger()
        self.experiment_logger.start_experiment("ALchemist_Experiment")

        # UI state: pending acquisition suggestions (persist across dialogs)
        self.pending_suggestions = []
        self.current_suggestion_index = 0
    
    def _configure_window(self):
        ctk.set_appearance_mode('dark')
        ctk.set_default_color_theme('blue')
        self.title('Active Learning Experiment Planner')
        self.geometry('1450x800')
        self.minsize(1300, 600)  # Increase minimum width to accommodate all panels
        self.protocol('WM_DELETE_WINDOW', self._quit)

    def _create_menu_bar(self):
        menu_bar = tk.Menu(self)
        
        # File menu - NEW: Session management
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="New Session", command=self.new_session, accelerator="Cmd+N")
        file_menu.add_command(label="Open Session...", command=self.open_session, accelerator="Cmd+O")
        file_menu.add_command(label="Save Session", command=self.save_session_cmd, accelerator="Cmd+S")
        file_menu.add_command(label="Save Session As...", command=self.save_session_as)
        file_menu.add_separator()
        file_menu.add_command(label="Export Audit Log...", command=self.export_audit_log)
        file_menu.add_separator()
        file_menu.add_command(label="Session Metadata...", command=self.edit_session_metadata)
        menu_bar.add_cascade(label="File", menu=file_menu)
        
        # Help menu
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="Help", command=self.show_help)
        help_menu.add_command(label="About", command=self.show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        # Documentation menu
        doc_menu = tk.Menu(menu_bar, tearoff=0)
        doc_menu.add_command(label="User Guide", command=self.show_user_guide)
        doc_menu.add_command(label="API Reference", command=self.show_api_reference)
        menu_bar.add_cascade(label="Documentation", menu=doc_menu)
        # Preferences menu
        pref_menu = tk.Menu(menu_bar, tearoff=0)
        pref_menu.add_command(label="Settings", command=self.show_settings)
        pref_menu.add_command(label="Toggle Tabbed Layout", command=self.toggle_tabbed_layout)
        pref_menu.add_command(label="Toggle Noise Column", command=self.toggle_noise_column)
        pref_menu.add_command(label="Configure Target Columns...", command=self.configure_target_columns)
        pref_menu.add_separator()
    # Removed: Toggle Session API menu item (session API is now always enabled)
        menu_bar.add_cascade(label="Preferences", menu=pref_menu)
        
        # Bind keyboard shortcuts
        self.bind_all("<Command-s>", lambda e: self.save_session_cmd())
        self.bind_all("<Command-n>", lambda e: self.new_session())
        self.bind_all("<Control-n>", lambda e: self.new_session())
        # Open session with Cmd+O (macOS) or Ctrl+O (Windows/Linux)
        self.bind_all("<Command-o>", lambda e: self.open_session())
        self.bind_all("<Control-o>", lambda e: self.open_session())
        
        self.config(menu=menu_bar)

    def show_help(self):
        tk.messagebox.showinfo("Help", "This is the help dialog.")

    def show_about(self):
        tk.messagebox.showinfo("About", "This is the about dialog.")

    def show_user_guide(self):
        tk.messagebox.showinfo("User Guide", "This is the user guide.")

    def show_api_reference(self):
        tk.messagebox.showinfo("API Reference", "This is the API reference.")

    def show_settings(self):
        tk.messagebox.showinfo("Settings", "This is the settings dialog.")

    
    
    # ============================================================
    # Session Event Handlers
    # ============================================================
    
    def _on_session_progress(self, event_data):
        """Handle progress events from the session."""
        message = event_data.get('message', '')
        # UI components can listen to this for progress updates
    
    def _on_session_model_trained(self, event_data):
        """Handle model training completion from session."""
        metrics = event_data.get('metrics', {})
        print(f"Session: Model trained successfully")
        print(f"  R² = {metrics.get('mean_R²', 'N/A'):.3f}")
        print(f"  RMSE = {metrics.get('mean_RMSE', 'N/A'):.3f}")
        # Sync session model to main_app.gpr_model for visualization compatibility
        self.gpr_model = self.session.model
    
    def _on_session_suggestions(self, event_data):
        """Handle acquisition suggestions from session."""
        suggestions = event_data.get('suggestions', None)
        if suggestions is not None:
            print(f"Session: {len(suggestions)} new experiments suggested")
            # Sync to main_app.next_point for visualization compatibility
            self.next_point = suggestions

    def _create_vertical_frame(self):
        # LEFT COLUMN: Fixed-width frame for variable and experiment management.
        self.vertical_frame = ctk.CTkFrame(self, width=450)
        self.vertical_frame.pack(side='left', fill='y', padx=10, pady=10)
        self.vertical_frame.pack_propagate(False)  # Prevent automatic resizing

    def _create_variable_management_frame(self):
        self.frame_vars = ctk.CTkFrame(self.vertical_frame)
        self.frame_vars.pack(side='top', fill='both', padx=5, pady=5)

        ctk.CTkLabel(self.frame_vars, text='Variable Management', font=('Arial', 16)).pack(pady=5)
        self.var_sheet = tksheet.Sheet(self.frame_vars, height=200, header=['Variables', 'Type', 'Min', 'Max', 'Values'])
        self.var_sheet.pack(fill='both', expand=True, padx=5, pady=5)
        self.var_sheet.set_all_column_widths()
        self.var_sheet.enable_bindings()

        self.frame_vars_buttons = ctk.CTkFrame(self.frame_vars)
        self.frame_vars_buttons.pack(fill='x', pady=5)
        self.load_var_button = ctk.CTkButton(self.frame_vars_buttons, text='Load Variables', command=self.load_variables)
        self.load_var_button.pack(side='left', padx=5, pady=5)
        def open_space_setup():
            self.var_space_editor = SpaceSetupWindow(self)
            self.var_space_editor.grab_set()
        self.gen_var_button = ctk.CTkButton(self.frame_vars_buttons, text='Generate Variables File', command=open_space_setup)
        self.gen_var_button.pack(side='left', padx=5, pady=5)

    def _create_experiment_management_frame(self):
        self.frame_exp = ctk.CTkFrame(self.vertical_frame)
        self.frame_exp.pack(side='top', fill='both', padx=5, pady=5)

        ctk.CTkLabel(self.frame_exp, text='Experiment Data', font=('Arial', 16)).pack(pady=5)
        self.exp_sheet = tksheet.Sheet(self.frame_exp)
        self.exp_sheet.pack(fill='both', expand=True, padx=5, pady=5)
        self.exp_sheet.enable_bindings()

        self.frame_exp_buttons_top = ctk.CTkFrame(self.frame_exp)
        self.frame_exp_buttons_top.pack(fill='x', pady=5)
        self.load_exp_button = ctk.CTkButton(self.frame_exp_buttons_top, text='Load Experiments', command=self.load_experiments, state='disabled')
        self.load_exp_button.pack(side='left', padx=5, pady=5)
        self.save_exp_button = ctk.CTkButton(self.frame_exp_buttons_top, text='Save Experiments', command=self.save_experiments)
        self.save_exp_button.pack(side='left', padx=5, pady=5)

        self.frame_exp_buttons_bottom = ctk.CTkFrame(self.frame_exp)
        self.frame_exp_buttons_bottom.pack(fill='x', pady=5)
        self.gen_template_button = ctk.CTkButton(self.frame_exp_buttons_bottom, text='Generate Initial Points', command=self.generate_initial_points, state='disabled')
        self.gen_template_button.pack(side='left', padx=5, pady=5)
        self.add_point_button = ctk.CTkButton(self.frame_exp_buttons_bottom, text='Add Point', command=self.add_point)
        self.add_point_button.pack(side='left', padx=5, pady=5)

    def _create_visualization_frame(self):
        # MIDDLE COLUMN: Visualization frame expands but maintains aspect ratio
        self.frame_viz = ctk.CTkFrame(self)
        self.frame_viz.pack(side='left', fill='both', expand=True, padx=(10, 5), pady=10)
        # Set size constraints
        self.frame_viz.pack_propagate(False)
        self.frame_viz.configure(width=500, height=600)  # Fixed width and height for better aspect ratio

        ctk.CTkLabel(self.frame_viz, text='Visualization', font=('Arial', 16)).pack(pady=5)

        # Frame for variable dropdowns and clustering switch
        self.frame_viz_options = ctk.CTkFrame(self.frame_viz)
        self.frame_viz_options.pack(pady=5)

        # Add variable dropdowns
        self._create_variables_dropdown()

        # Clustering switch
        # NOTE: Clustering switch removed (deprecated)

        # Visualization canvas - use square figure for better aspect ratio
        # Fix macOS Retina scaling by using the Tk root pixels-per-inch as DPI
        try:
            dpi = int(self.winfo_fpixels('1i'))
        except Exception:
            dpi = plt.rcParams.get('figure.dpi', 100)

        # Apply DPI to matplotlib runtime settings so text and markers scale correctly
        plt.rcParams['figure.dpi'] = dpi

        # Create figure with explicit DPI and cap font sizes to avoid oversized rendering
        try:
            plt.rcParams['axes.titlesize'] = min(12, plt.rcParams.get('axes.titlesize', 12))
            plt.rcParams['axes.labelsize'] = min(10, plt.rcParams.get('axes.labelsize', 10))
            plt.rcParams['legend.fontsize'] = min(9, plt.rcParams.get('legend.fontsize', 9))
            plt.rcParams['xtick.labelsize'] = min(9, plt.rcParams.get('xtick.labelsize', 9))
            plt.rcParams['ytick.labelsize'] = min(9, plt.rcParams.get('ytick.labelsize', 9))
            plt.rcParams['lines.markersize'] = min(6, plt.rcParams.get('lines.markersize', 6))
        except Exception:
            pass

        try:
            self.fig, self.ax = plt.subplots(figsize=(5, 5), dpi=dpi)
            # Ensure compact layout
            try:
                self.fig.tight_layout(pad=0.6)
            except Exception:
                pass
        except Exception:
            # Fallback to defaults
            self.fig, self.ax = plt.subplots(figsize=(5, 5))

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_viz)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame_viz)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=2, pady=2)

    def _create_variables_dropdown(self):
        '''Creates dropdowns for selecting variables for 2D visualization.'''
        # Always use the skopt-compatible version for iteration
        if self.search_space is None:
            variables = ['Variable 1', 'Variable 2']
        else:
            # Make sure we're using an iterable version
            if hasattr(self.search_space, 'to_skopt'):
                skopt_space = self.search_space.to_skopt()
            else:
                skopt_space = self.search_space
            variables = [dim.name for dim in skopt_space]

        # Dropdown for the first variable
        self.var1_dropdown = ctk.CTkComboBox(self.frame_viz_options, values=variables, command=self.update_pool_plot)
        self.var1_dropdown.set(variables[0] if variables else 'Variable 1')
        self.var1_dropdown.pack(side='left', padx=5, pady=5)

        # Dropdown for the second variable
        self.var2_dropdown = ctk.CTkComboBox(self.frame_viz_options, values=variables, command=self.update_pool_plot)
        self.var2_dropdown.set(variables[1] if len(variables) > 1 else 'Variable 2')
        self.var2_dropdown.pack(side='left', padx=5, pady=5)

    def _create_model_frame(self):
        # RIGHT COLUMN: Model frame (GPR) placed on the right.
        self.model_frame = GaussianProcessPanel(self)
        self.model_frame.pack(side='right', fill='both', padx=10, pady=10)
        # Ensure the panel doesn't collapse below minimum width
        self.model_frame.configure(width=300, height=600)
        self.model_frame.pack_propagate(False)  # Prevent automatic resizing

    def _create_acquisition_frame(self):
        # NEW PANEL: Acquisition function panel on far right
        self.acq_frame = ctk.CTkFrame(self)
        self.acq_frame.pack(side='right', fill='both', padx=(5, 10), pady=10)
        self.acq_frame.configure(width=280)
        self.acq_frame.pack_propagate(False)
        
        # Create the acquisition panel (no title in the frame as the panel has its own)
        self.acquisition_panel = AcquisitionPanel(self.acq_frame, self)
        self.acquisition_panel.pack(fill='both', expand=True, padx=5, pady=5)

    def _update_ui_state(self):
        """Updates the UI state based on the loaded data."""
        if self.search_space is not None:
            self.load_exp_button.configure(state='normal')
            self.gen_template_button.configure(state='normal')
            variables = [dim.name for dim in self.search_space]
            if list(self.var1_dropdown.cget('values')) != list(variables):
                self.var1_dropdown.configure(values=variables)
                self.var1_dropdown.set(variables[0])
                self.var2_dropdown.configure(values=variables)
                self.var2_dropdown.set(variables[1])
        else:
            self.load_exp_button.configure(state='disabled')
            self.gen_template_button.configure(state='disabled')
        # Clustering functionality is deprecated and removed from the UI

    def load_variables(self):
        """Loads a search space from a file using a file dialog."""
        file_path = filedialog.askopenfilename(
            title='Select Variable Space File',
            filetypes=[('JSON Files', '*.json'), ('CSV Files', '*.csv')]
        )
        if file_path:
            try:
                # Determine file type and load accordingly
                if file_path.lower().endswith('.json'):
                    # Load JSON file
                    self.search_space_manager.load_from_json(file_path)
                elif file_path.lower().endswith('.csv'):
                    # Load CSV file using the same logic as variables_setup.py
                    data = self._load_variables_from_csv(file_path)
                    self.search_space_manager.from_dict(data)
                else:
                    raise ValueError("Unsupported file format. Please use .json or .csv files.")
                
                # CRITICAL: Update the legacy variable with skopt format
                self.search_space = self.search_space_manager.to_skopt()
                
                # Update the experiment manager with the new search space
                self.experiment_manager.set_search_space(self.search_space_manager)

                # Update the variable sheet with the loaded search space
                data = []
                for var in self.search_space_manager.variables:
                    if var["type"] == "categorical":
                        # For categorical variables, include the categories
                        row = [
                            var["name"],               # Variable Name
                            'Categorical',             # Type of the variable
                            '',                        # No min for categorical
                            '',                        # No max for categorical
                            ', '.join(map(str, var["values"]))  # List the possible categories as a string
                        ]
                    else:
                        # For Integer and Real variables
                        row = [
                            var["name"],               # Variable Name
                            var["type"].capitalize(),  # Type of the variable ('Integer' or 'Real')
                            var["min"],                # Minimum Value
                            var["max"],                # Maximum Value
                            ''                         # No values for Integer/Real
                        ]
                    data.append(row)

                # Insert the data into the tksheet
                self.var_sheet.set_sheet_data(data)
                self.var_sheet.set_all_column_widths()

                # Update the experiment sheet headers
                variables = self.search_space_manager.get_variable_names()
                exp_sheet_headers = variables + ['Output']
                self.exp_sheet.set_header_data(exp_sheet_headers)

                # Update the model frame with the search space and categorical variables
                self.model_frame.update_search_space(
                    self.search_space_manager,  # Pass the manager object
                    self.search_space_manager.get_categorical_variables()
                )

                # Update the UI state
                self._update_ui_state()
                print('Search space loaded successfully.')
                
                # Ensure we're using the skopt-compatible version for pool generation
                self.pool = generate_pool(self.search_space, lhs_iterations=20)

                # Reset kmeans and update plot
                # Clustering removed; just update plot
                self.update_pool_plot()
            except Exception as e:
                print('Error loading search space:', e)
    
    def _load_variables_from_csv(self, file_path):
        """Load variables from CSV file using the same logic as variables_setup.py"""
        import csv
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
                        "type": "real",  # lowercase for SearchSpace compatibility
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
                        "type": "integer",  # lowercase for SearchSpace compatibility
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
                        "type": "categorical",  # lowercase for SearchSpace compatibility
                        "values": values
                    }
                else:
                    print(f"Warning: Unknown variable type '{typ}' for variable '{variable_name}'. Skipping.")
                    continue
                data.append(d)
        return data

    def load_experiments(self, file_path=None):
        '''Loads experimental data from a CSV file using a file dialog.'''
        if file_path is None:
            file_path = filedialog.askopenfilename(title='Select Experiments CSV', filetypes=[('CSV Files', '*.csv')])
        
        if file_path:
            try:
                # First, read the CSV to check for target column
                import pandas as pd
                preview_df = pd.read_csv(file_path)
                
                # Check if any configured target column exists
                # Default to looking for 'Output' if no target_columns configured
                expected_targets = getattr(self.experiment_manager, 'target_columns', ['Output'])
                missing_targets = [col for col in expected_targets if col not in preview_df.columns]
                
                # If target column(s) missing, show selection dialog
                target_columns_to_use = None
                if missing_targets:
                    # Get non-metadata columns that could be targets
                    metadata_cols = {'Iteration', 'Reason', 'Noise'}
                    available_cols = [col for col in preview_df.columns if col not in metadata_cols]
                    
                    if not available_cols:
                        raise ValueError("CSV file contains no columns that could be target columns.")
                    
                    # Show target selection dialog
                    from ui.target_column_dialog import show_target_column_dialog
                    selected = show_target_column_dialog(
                        parent=self,
                        available_columns=available_cols,
                        default_column='output' if 'output' in available_cols else None
                    )
                    
                    if selected is None:
                        # User cancelled
                        print("Data loading cancelled by user.")
                        return

                    # Multi-objective returns dict with targets and variables
                    if isinstance(selected, dict):
                        target_columns_to_use = selected["targets"]
                        self.experiment_manager.variable_columns = selected["variables"]
                        print(f"User selected targets: {target_columns_to_use}, variables: {selected['variables']}")
                    else:
                        target_columns_to_use = selected if isinstance(selected, list) else [selected]
                        self.experiment_manager.variable_columns = None
                        print(f"User selected target column(s): {target_columns_to_use}")
                else:
                    target_columns_to_use = expected_targets

                # Configure experiment manager with selected target columns
                self.experiment_manager.target_columns = target_columns_to_use
                
                # Load experiments using the ExperimentManager
                self.experiment_manager.load_from_csv(file_path)
                
                # Update the main DataFrame from the experiment manager
                self.exp_df = self.experiment_manager.get_data()
                
                # Get the headers and data for the tksheet
                headers = self.exp_df.columns.tolist()
                data = self.exp_df.values.tolist()
                
                # Update the experiment sheet
                self.exp_sheet.set_sheet_data(data)
                self.exp_sheet.set_header_data(headers)
                self.exp_sheet.set_all_column_widths()
                
                # Log the data loading
                print(f"Loaded {len(self.exp_df)} experiment points from {file_path}")
                print(f"Target column(s): {target_columns_to_use}")
                if 'Noise' in self.exp_df.columns:
                    print("Notice: Noise column detected. This will be used for model regularization if available.")
                
                # Enable UI elements that require experiment data
                self._update_ui_state()
                
                # Propagate multi-objective state to panels
                is_mobo = len(target_columns_to_use) > 1
                if hasattr(self, 'model_frame') and self.model_frame is not None:
                    self.model_frame.set_multi_objective_mode(
                        is_mobo, 
                        objective_names=target_columns_to_use if is_mobo else []
                    )
                
                # Reset any existing model
                if hasattr(self, 'model_frame') and self.model_frame is not None:
                    self.model_frame.reset_model()
                    
                # Update plot if available
                self.update_pool_plot()
            except Exception as e:
                print(f"Error loading experiments: {e}")
                import traceback
                traceback.print_exc()

    def configure_target_columns(self):
        """Open the target column dialog to change objective configuration at any time."""
        # Determine available columns from current data
        if self.exp_df is None or self.exp_df.empty:
            # Fall back to session experiment manager or show error
            if hasattr(self.session.experiment_manager, 'df') and not self.session.experiment_manager.df.empty:
                all_cols = self.session.experiment_manager.df.columns.tolist()
            else:
                import tkinter.messagebox
                tkinter.messagebox.showinfo(
                    "No Data", "Load experiment data first before configuring target columns.")
                return
        else:
            all_cols = self.exp_df.columns.tolist()
        
        metadata_cols = {'Iteration', 'Reason', 'Noise'}
        available_cols = [col for col in all_cols if col not in metadata_cols]
        
        if not available_cols:
            return
        
        from ui.target_column_dialog import show_target_column_dialog
        selected = show_target_column_dialog(
            parent=self,
            available_columns=available_cols,
            default_column=self.experiment_manager.target_columns[0]
                if hasattr(self.experiment_manager, 'target_columns') and self.experiment_manager.target_columns
                else None
        )
        
        if selected is None:
            return

        # Multi-objective returns dict with targets and variables
        if isinstance(selected, dict):
            target_columns_to_use = selected["targets"]
            self.experiment_manager.variable_columns = selected["variables"]
            print(f"Target column(s) updated to: {target_columns_to_use}, variables: {selected['variables']}")
        else:
            target_columns_to_use = selected if isinstance(selected, list) else [selected]
            self.experiment_manager.variable_columns = None
            print(f"Target column(s) updated to: {target_columns_to_use}")

        # Update experiment manager
        self.experiment_manager.target_columns = target_columns_to_use
        
        # Propagate MOBO state to panels
        is_mobo = len(target_columns_to_use) > 1
        if hasattr(self, 'model_frame') and self.model_frame is not None:
            self.model_frame.set_multi_objective_mode(
                is_mobo,
                objective_names=target_columns_to_use if is_mobo else []
            )
            self.model_frame.reset_model()

    def update_exp_df_from_sheet(self):
        '''Updates the exp_df DataFrame with the current data from the exp_sheet.'''
        sheet_data = self.exp_sheet.get_sheet_data(get_header=False)
        headers = self.exp_sheet.headers()
        
        # If headers are empty, use exp_df columns if available
        if not headers and self.exp_df is not None:
            headers = self.exp_df.columns.tolist()
        
        if headers:
            self.exp_df = pd.DataFrame(sheet_data, columns=headers)
        else:
            print("Warning: No headers available for sheet data")
            self.exp_df = pd.DataFrame(sheet_data)

    def save_experiments(self):
        '''Saves the experimental data to a CSV file using a file dialog.'''
        self.update_exp_df_from_sheet()  # Update the DataFrame with the current data from the sheet
        if self.exp_df is not None:
            # If we have an associated experiments file path, overwrite it silently
            target_path = None
            if hasattr(self, 'exp_file_path') and self.exp_file_path:
                target_path = self.exp_file_path
            elif hasattr(self.experiment_manager, 'filepath') and self.experiment_manager.filepath:
                target_path = self.experiment_manager.filepath

            if target_path:
                try:
                    self.exp_df.to_csv(target_path, index=False)
                    print(f'Experiments auto-saved to {target_path}')
                except Exception as e:
                    print('Error saving experiments to existing path:', e)
            else:
                file_path = filedialog.asksaveasfilename(
                    title='Save Experiments CSV',
                    defaultextension='.csv',
                    filetypes=[('CSV Files', '*.csv')]
                )
                if file_path:
                    try:
                        self.exp_df.to_csv(file_path, index=False)
                        self.exp_file_path = file_path
                        print('Experiments saved successfully.')
                    except Exception as e:
                        print('Error saving experiments:', e)
        else:
            print('No experimental data to save.')

    def save_and_commit_all_pending(self):
        """Commit all pending suggestions to the experiment table and session.

        This method persists any user edits that were made while navigating suggestions,
        appends all pending suggestions to `self.exp_df` and to the session's ExperimentManager,
        optionally retrains once, and clears the pending list.
        """
        # Save current dialog state first
        try:
            self._save_current_dialog_state()
        except Exception:
            pass

        if not self.pending_suggestions:
            # Nothing staged; behave like save_new_point
            self.save_new_point()
            if hasattr(self, 'add_point_window'):
                self.add_point_window.destroy()
            return

        # Build DataFrame from pending suggestions
        pending_df = pd.DataFrame(self.pending_suggestions)

        # Ensure we have Iteration and Reason columns
        # If there are acquisition suggestions (not initial design), we should
        # increment the iteration counter first so committed rows receive the
        # next iteration number. This keeps the experiment table in-sync with
        # the audit log which increments on lock_acquisition.
        if 'Iteration' not in pending_df.columns:
            pending_df['Iteration'] = int(getattr(self.experiment_manager, '_current_iteration', 0))
        if 'Reason' not in pending_df.columns:
            pending_df['Reason'] = pending_df.get('_reason', 'Acquisition')

        # If committing acquisition suggestions, advance the iteration counter
        try:
            # Determine if any pending suggestion is not an initial design
            has_acquisition = any(not str(row.get('_reason', row.get('Reason', ''))).lower().startswith('initial')
                                  for _, row in pending_df.iterrows())
        except Exception:
            has_acquisition = False

        if has_acquisition:
            # Use the session's ExperimentManager iteration as authoritative.
            # `lock_acquisition` should have already incremented it during
            # audit logging; do not modify `_current_iteration` here to avoid
            # double increments.
            try:
                sess_iter = int(getattr(self.session.experiment_manager, '_current_iteration',
                                         getattr(self.experiment_manager, '_current_iteration', 0)))
            except Exception:
                sess_iter = int(getattr(self.experiment_manager, '_current_iteration', 0))

            # Overwrite Iteration column for all non-initial-design pending rows
            def assign_iter(row):
                r_reason = str(row.get('_reason', row.get('Reason', ''))).lower()
                if r_reason.startswith('initial'):
                    return int(row.get('Iteration', 0))
                return int(sess_iter)

            pending_df['Iteration'] = pending_df.apply(assign_iter, axis=1).astype(int)

        # Now that Iteration column has been finalized, append to the visible experiment
        # table so the UI shows the correct (post-increment) iteration numbers.
        if self.exp_df is None or self.exp_df.empty:
            self.exp_df = pending_df.drop(columns=['_reason'], errors='ignore').copy()
        else:
            self.exp_df = pd.concat([self.exp_df, pending_df.drop(columns=['_reason'], errors='ignore')], ignore_index=True)

        # Update experiment manager and session
        for _, row in pending_df.iterrows():
            inputs = {c: row[c] for c in pending_df.columns if c not in ['Output', 'Noise', 'Iteration', 'Reason', '_reason']}
            output_val = row.get('Output', None)
            noise_val = row.get('Noise', None)
            # Determine reason (prefer internal _reason tag)
            reason_val = row.get('Reason', row.get('_reason', 'Acquisition'))

            # If this is an acquisition (committed after lock_acquisition),
            # prefer the experiment manager's current iteration which was
            # incremented when the acquisition was locked. For initial design
            # keep the recorded iteration (usually 0).
            if str(reason_val).lower().startswith('initial'):
                iter_val = int(row.get('Iteration', 0))
            else:
                iter_val = int(getattr(self.experiment_manager, '_current_iteration', 0))
            try:
                self.session.add_experiment(inputs=inputs, output=float(output_val) if output_val is not None else None,
                                            noise=noise_val, iteration=iter_val, reason=reason_val)
            except Exception:
                # Fallback: add without casting
                self.session.add_experiment(inputs=inputs, output=output_val, noise=noise_val, iteration=iter_val, reason=reason_val)

        # Update sheet and headers
        headers = self.exp_df.columns.tolist()
        self.exp_sheet.set_sheet_data(self.exp_df.values.tolist())
        try:
            self.exp_sheet.set_header_data(headers)
        except Exception:
            pass

        # Clear pending suggestions
        n_committed = len(self.pending_suggestions)
        self.pending_suggestions = []
        self.current_suggestion_index = 0

        # Optionally retrain model once if user checked retrain
        if hasattr(self, 'retrain_checkbox') and self.retrain_checkbox.get():
            try:
                self.retrain_model()
            except Exception as e:
                print('Warning: retrain failed after committing pending suggestions:', e)

        # Auto-save experiments if we have a path
        if hasattr(self, 'exp_file_path') and self.exp_file_path:
            try:
                self.exp_df.to_csv(self.exp_file_path, index=False)
                print(f'Committed {n_committed} pending suggestions and auto-saved to {self.exp_file_path}')
            except Exception as e:
                print('Warning: failed to auto-save committed suggestions:', e)

        # Auto-save session if available
        if hasattr(self, 'current_session_file') and self.current_session_file:
            try:
                self.session.save_session(self.current_session_file)
                print(f'Session auto-saved to {self.current_session_file}')
            except Exception as e:
                print('Warning: Failed to auto-save session:', e)

        # Close dialog
        if hasattr(self, 'add_point_window'):
            self.add_point_window.destroy()
    
    # ============================================================
    # Lock-in Methods (Audit Trail)
    # ============================================================
    
    def log_optimization_to_audit(self, next_point_df=None, strategy_info=None):
        """Log complete optimization decision (data + model + acquisition) to audit trail.
        
        Args:
            next_point_df: DataFrame with suggested points
            strategy_info: Dict with strategy details (name, params, notes, etc.)
        """
        try:
            # Ensure data is synced to session
            self._sync_data_to_session()
            
            # Extract notes from strategy_info
            notes = strategy_info.get('notes', '') if strategy_info else ''
            
            # Log data snapshot (include initial design metadata if present in session.config)
            extra = {}
            if hasattr(self.session, 'config'):
                method = self.session.config.get('initial_design_method')
                n_pts = self.session.config.get('initial_design_n_points')
                if method:
                    extra['initial_design_method'] = method
                    extra['initial_design_n_points'] = n_pts

            try:
                data_entry = self.session.lock_data(notes=notes, extra_parameters=extra)
            except TypeError:
                # Older session API may not accept extra_parameters; fall back and merge metadata into last entry
                data_entry = self.session.lock_data(notes=notes)
                try:
                    if extra and hasattr(self.session.audit_log, 'entries') and len(self.session.audit_log.entries) > 0:
                        last = self.session.audit_log.entries[-1]
                        if last.entry_type == 'data_locked':
                            last.parameters.update(extra)
                except Exception:
                    pass

            print(f"Data logged to audit trail: {len(self.session.experiment_manager.df)} experiments")
            
            # Log model
            if hasattr(self, 'gpr_model') and self.gpr_model is not None:
                self.session.model = self.gpr_model
                model_entry = self.session.lock_model(notes=notes)
                print(f"Model logged to audit trail")
            
            # Log acquisition
            if next_point_df is not None and strategy_info is not None:
                # Convert DataFrame to list of dicts
                suggestions = next_point_df.to_dict('records')
                
                # Extract strategy information
                strategy_type = strategy_info.get('type', 'Unknown')
                parameters = strategy_info.get('params', {})
                
                # Add goal to parameters
                if 'maximize' in strategy_info:
                    parameters['goal'] = 'maximize' if strategy_info['maximize'] else 'minimize'
                
                acq_entry = self.session.lock_acquisition(
                    strategy=strategy_type,
                    parameters=parameters,
                    suggestions=suggestions,
                    notes=notes
                )
                print(f"Acquisition logged to audit trail: {strategy_type}")
            
            # Auto-save session after audit log update
            if hasattr(self, 'current_session_file') and self.current_session_file:
                try:
                    self.session.save_session(self.current_session_file)
                    print(f"Session auto-saved to {self.current_session_file}")
                except Exception as e:
                    print(f"Warning: Failed to auto-save session: {e}")
            
            return True
            
        except Exception as e:
            print(f"Error logging to audit trail: {e}")
            import traceback
            traceback.print_exc()
            return False


    def generate_template(self):
        '''Generates a blank template with 10 starter points based on loaded variables.'''
        if self.var_df is not None:
            num_points = 10
            # Assume the variable names are in a column called 'Variables'
            if 'Variables' in self.var_df.columns:
                var_names = self.var_df['Variables'].tolist()
            else:
                var_names = self.var_df.columns.tolist()

            # Create a DataFrame with a column for each variable and an 'Output' column
            data = {var: [None] * num_points for var in var_names}
            data['Output'] = [None] * num_points
            self.exp_df = pd.DataFrame(data)
            self.exp_sheet.set_sheet_data(self.exp_df.values.tolist())
            self._update_ui_state()
            print('Experiment template generated.')
        else:
            print('Please load variables before generating a template.')

    def _get_skopt_space(self):
        """Helper to always return the skopt-compatible version of the search space"""
        if hasattr(self.search_space, 'to_skopt'):
            return self.search_space.to_skopt()
        return self.search_space

    def update_pool_plot(self, event=None):
        self.ax.cla()

        var1 = self.var1_dropdown.get()
        var2 = self.var2_dropdown.get()
        
        # Safety check: pool visualization is deprecated and may not be initialized
        if self.pool is None:
            print("Pool not initialized - skipping pool visualization (deprecated feature)")
            self.canvas.draw()
            return

        # Clustering was deprecated — always use non-clustered visualization
        plot_pool(self.pool, var1, var2, self.ax, kmeans=None, experiments=self.exp_df)

        if self.exp_df is not None and not self.exp_df.empty:
            self.ax.plot(self.exp_df[var1], self.exp_df[var2], 'go', markeredgecolor='k')

        if self.next_point is not None:

            if hasattr(self, 'tooltip'):
                self.tooltip.remove()

            # Plot the points
            self.ax.plot(self.next_point[var1], self.next_point[var2],
                        'bD', markeredgecolor='k', markersize=10)
            scatter = self.ax.scatter(self.next_point[var1], self.next_point[var2])
            
            # Create the tooltip
            self.tooltip = mplcursors.cursor(scatter, hover=True)
            
            # Format the numeric values to 1 decimal place, but leave strings as they are
            next_point_formatted = self.next_point.T.apply(lambda x: x.map(lambda v: f'{v:.1f}' if isinstance(v, (int, float)) else v))
            
            # Create the tooltip text with formatted values
            tooltip_text = tabulate(next_point_formatted, tablefmt='plain')

            # Set up the tooltip appearance
            self.tooltip.connect('add', lambda sel: sel.annotation.set_bbox(
                dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')))
            
            # Set the tooltip text to the formatted text
            self.tooltip.connect('add', lambda sel: sel.annotation.set_text(tooltip_text))

        self.canvas.draw()



    def next_explore_point(self):
        # DEPRECATED: This method uses legacy clustering and EMOC acquisition
        # Use the modern AcquisitionPanel with session API instead
        print("WARNING: next_explore_point() is deprecated and no longer functional")
        print("Please use the Acquisition Panel in the UI for next point selection")
        return
        # # Use skopt-compatible version
        # skopt_space = self._get_skopt_space()
        # # cluster_pool now returns the new clustering (with an added cluster) and kmeans.
        # labels, largest_empty_cluster, kmeans = cluster_pool(self.pool, self.exp_df, skopt_space, add_cluster=True)
        # # Update the stored kmeans object.
        # self.kmeans = kmeans
        #
        # largest_empty_cluster_points = self.pool[labels == largest_empty_cluster]
        #
        # X = self.exp_df.drop(columns='Output')
        # y = self.exp_df['Output']
        #
        # self.next_point = select_EMOC(largest_empty_cluster_points, X, y, self.search_space, model=self.gpr_model, verbose=False)
        # self.update_pool_plot()



    
    def pool_mode(self):
        self.update_pool_plot()
        print('Pool mode activated.')
    
    def explore_mode(self):
        '''Placeholder for exploration mode functionality.'''
        print('Exploration mode activated.')
        self.ax.clear()
        self.ax.text(0.5, 0.5, 'Explore mode visualization',
                     horizontalalignment='center', verticalalignment='center',
                     transform=self.ax.transAxes)
        self.canvas.draw()

    def optimize_mode(self, use_dataframe=True):
        '''DEPRECATED: Use AcquisitionPanel with OptimizationSession API instead.'''
        print("WARNING: optimize_mode() is deprecated and no longer functional")
        print("Please use the Acquisition Panel in the UI for optimization")
        return
        # '''Optimizes the next experiment point based on the loaded data.'''
        # if self.exp_df is not None and self.search_space is not None:
        #     if use_dataframe:
        #         next_point = select_optimize(self.search_space, self.exp_df, base_estimator=self.gpr_model)
        #     else:
        #         if hasattr(self, 'encoded_X') and hasattr(self, 'gpr_model'):
        #             X = self.encoded_X.values
        #             y = self.exp_df['Output'].values
        #             next_point = select_optimize(self.search_space, (X, y), base_estimator=self.gpr_model)
        #         else:
        #             print('Encoded data or GPR model not found. Please train the model first.')
        #             return
        #
        #     # Convert the next_point to a DataFrame for consistency
        #     next_point_df = pd.DataFrame([next_point], columns=self.exp_df.drop(columns='Output').columns)
        #     
        #     # Store the next point for visualization
        #     self.next_point = next_point_df
        #     
        #     # Update the plot with the new point
        #     self.update_pool_plot()
        #     
        #     print('Optimization mode activated.')
        # else:
        #     print('Please load experiments and variables before optimizing.')


    def generate_initial_points(self):
        """Opens a window to select the sampling strategy and number of points."""
        if not self.search_space:
            print('Please load variables before generating initial points.')
            return

        self.initial_points_window = ctk.CTkToplevel(self)
        self.initial_points_window.title("Generate Initial Points")
        self.initial_points_window.geometry("300x200")
        self.initial_points_window.grab_set()

        ctk.CTkLabel(self.initial_points_window, text="Select Strategy:").pack(pady=5)
        self.strategy_var = ctk.StringVar(value="random")
        strategies = ["random", "LHS", "Sobol", "Halton", "Hammersly"]
        self.strategy_dropdown = ctk.CTkComboBox(self.initial_points_window, values=strategies, variable=self.strategy_var)
        self.strategy_dropdown.pack(pady=5)

        ctk.CTkLabel(self.initial_points_window, text="Number of Points:").pack(pady=5)
        self.num_points_entry = ctk.CTkEntry(self.initial_points_window)
        self.num_points_entry.insert(0, "10")
        self.num_points_entry.pack(pady=5)

        ctk.CTkButton(self.initial_points_window, text="Generate", command=self._generate_points).pack(pady=10)

    def _generate_points(self):
        """Generates initial points based on the selected strategy and number of points."""
        strategy = self.strategy_var.get()
        try:
            num_points = int(self.num_points_entry.get())
        except ValueError:
            print("Invalid number of points.")
            return

        if not self.search_space:
            print('Search space is not loaded.')
            return

        # Generate samples based on the chosen strategy
        if strategy == "random":
            # Manually sample for each dimension based on its type
            samples_list = []
            for dim in self.search_space:
                if isinstance(dim, Categorical):
                    samples = np.random.choice(dim.categories, size=num_points)
                elif isinstance(dim, Integer):
                    # np.random.randint is [low, high), so add 1 to include the upper bound
                    samples = np.random.randint(dim.low, dim.high + 1, size=num_points)
                elif isinstance(dim, Real):
                    samples = np.random.uniform(dim.low, dim.high, size=num_points)
                else:
                    raise ValueError(f"Unknown dimension type: {type(dim)}")
                samples_list.append(samples)
            # Combine the samples into a 2D numpy array
            samples = np.column_stack(samples_list)
        else:
            # Use the appropriate skopt sampler
            if strategy == "LHS":
                sampler = Lhs(lhs_type="classic", criterion="maximin")
            elif strategy == "Sobol":
                sampler = Sobol()
            elif strategy == "Halton":
                sampler = Hammersly()
            elif strategy == "Hammersly":
                sampler = Hammersly()
            else:
                print("Unknown sampling strategy.")
                return

            samples = sampler.generate(self.search_space, num_points)
            # Convert list of samples to a NumPy array for slicing
            samples = np.array(samples)

        # Build a DataFrame with the generated points and an 'Output' column.
        data = {dim.name: samples[:, i].tolist() for i, dim in enumerate(self.search_space)}
        # Add workflow metadata columns: Output (empty), Iteration, Reason
        current_iter = getattr(self.experiment_manager, '_current_iteration', 0)
        data['Output'] = [None] * num_points
        data['Iteration'] = [int(current_iter)] * num_points
        data['Reason'] = ['Initial Design'] * num_points

        # Stage as pending suggestions (do not commit to session until user confirms)
        pending_df = pd.DataFrame(data)

        # Store pending suggestions as list of dicts (used by Add Point dialog)
        pending = []
        for _, row in pending_df.iterrows():
            rec = {col: row[col] for col in pending_df.columns}
            # Tag with internal reason key for dialog logic
            rec['_reason'] = 'Initial Design'
            pending.append(rec)

        # Replace current pending suggestions and reset index
        self.pending_suggestions = pending
        self.current_suggestion_index = 0

        # Display pending suggestions in the experiment sheet (so user can edit or save)
        headers = pending_df.columns.tolist()
        self.exp_sheet.set_sheet_data(pending_df.values.tolist())
        try:
            self.exp_sheet.set_header_data(headers)
        except Exception:
            pass
        try:
            self.exp_sheet.set_all_column_widths()
        except Exception:
            pass

        # Do not commit to self.exp_df yet; user must confirm via Add Point or Save
        self._update_ui_state()
        # Record initial design metadata in the audit log as a snapshot of planned points
        try:
            extra = {'initial_design_method': strategy, 'initial_design_n_points': num_points}
            # Use a copy to avoid accidental mutation
            self.session.audit_log.lock_data(pending_df.copy(), notes=f"Initial design staged ({strategy})", extra_parameters=extra)
        except Exception as e:
            print(f"Warning: failed to record initial design in audit log: {e}")

        print(f'Initial points generated and staged as {len(pending)} pending suggestions.')
        self.initial_points_window.destroy()

    def add_point(self):
        '''Opens a window to add a new experiment point, with support for pending suggestions.'''
        if not self.search_space:
            print('Please load variables before adding a point.')
            return

        self.add_point_window = ctk.CTkToplevel(self)
        self.add_point_window.title("Add Experimental Result")
        # Taller default to avoid vertical clipping; constrain min size
        try:
            screen_h = self.add_point_window.winfo_screenheight()
            default_h = min(800, int(screen_h * 0.75))
            self.add_point_window.geometry(f"560x{default_h}")
            self.add_point_window.minsize(520, 520)
        except Exception:
            self.add_point_window.geometry("560x700")
        self.add_point_window.grab_set()
        
        # Header with suggestion info
        header_frame = ctk.CTkFrame(self.add_point_window)
        header_frame.pack(pady=10, padx=10, fill='x')
        
        if self.pending_suggestions and self.current_suggestion_index < len(self.pending_suggestions):
            # Show which suggestion we're on
            suggestion_label = ctk.CTkLabel(
                header_frame,
                text=f"Pending Suggestion {self.current_suggestion_index + 1} of {len(self.pending_suggestions)}",
                font=('Arial', 14, 'bold'),
                text_color='#2B8A3E'
            )
            suggestion_label.pack(pady=5)
            
            # Navigation buttons
            nav_frame = ctk.CTkFrame(header_frame)
            nav_frame.pack(pady=5)
            
            prev_btn = ctk.CTkButton(
                nav_frame,
                text="← Previous",
                width=100,
                command=lambda: self._save_current_and_load(self.current_suggestion_index - 1)
            )
            if self.current_suggestion_index > 0:
                prev_btn.pack(side='left', padx=5)
            
            next_btn = ctk.CTkButton(
                nav_frame,
                text="Next →",
                width=100,
                command=lambda: self._save_current_and_load(self.current_suggestion_index + 1)
            )
            if self.current_suggestion_index < len(self.pending_suggestions) - 1:
                next_btn.pack(side='left', padx=5)
        else:
            # Manual entry (no suggestions)
            manual_label = ctk.CTkLabel(
                header_frame,
                text="Manual Entry",
                font=('Arial', 14, 'bold')
            )
            manual_label.pack(pady=5)

        # Variable entries
        entries_frame = ctk.CTkFrame(self.add_point_window)
        entries_frame.pack(pady=10, padx=10, fill='both', expand=True)
        
        self.var_entries = {}
        for var in self.search_space:
            ctk.CTkLabel(entries_frame, text=var.name).pack(pady=5)
            entry = ctk.CTkEntry(entries_frame)
            entry.pack(pady=5)
            self.var_entries[var.name] = entry
        
        # Pre-fill if we have a pending suggestion
        if self.pending_suggestions and self.current_suggestion_index < len(self.pending_suggestions):
            current_suggestion = self.pending_suggestions[self.current_suggestion_index]
            for var_name, entry in self.var_entries.items():
                if var_name in current_suggestion:
                    entry.insert(0, str(current_suggestion[var_name]))

        ctk.CTkLabel(entries_frame, text='Output').pack(pady=5)
        self.output_entry = ctk.CTkEntry(entries_frame)
        self.output_entry.pack(pady=5)
        self.output_entry.focus()  # Focus on output field

        # Display iteration and reason (read-only)
        iter_frame = ctk.CTkFrame(entries_frame)
        iter_frame.pack(fill='x', pady=(6, 2))
        ctk.CTkLabel(iter_frame, text='Iteration:', width=100, anchor='w').pack(side='left', padx=5)
        self.iteration_label = ctk.CTkLabel(iter_frame, text='N/A')
        self.iteration_label.pack(side='left', padx=5)

        reason_frame = ctk.CTkFrame(entries_frame)
        reason_frame.pack(fill='x', pady=(2, 8))
        ctk.CTkLabel(reason_frame, text='Reason:', width=100, anchor='w').pack(side='left', padx=5)
        self.reason_label = ctk.CTkLabel(reason_frame, text='Manual')
        self.reason_label.pack(side='left', padx=5)

        # If pending suggestion exists, populate iteration and reason
        if self.pending_suggestions and self.current_suggestion_index < len(self.pending_suggestions):
            cs = self.pending_suggestions[self.current_suggestion_index]
            # Prefer the session's experiment manager as the authoritative source
            iter_val = cs.get('Iteration', getattr(self.session.experiment_manager, '_current_iteration',
                                                  getattr(self.experiment_manager, '_current_iteration', 0))) + 1
            self.iteration_label.configure(text=str(int(iter_val)))
            self.reason_label.configure(text=str(cs.get('_reason', cs.get('Reason', 'Acquisition'))))
        else:
            # Default values for manual entry
            self.iteration_label.configure(text=str(getattr(self.experiment_manager, '_current_iteration', 0)))
            self.reason_label.configure(text='Manual')
        
        # Add noise field
        ctk.CTkLabel(entries_frame, text='Noise (optional)').pack(pady=5)
        self.noise_entry = ctk.CTkEntry(entries_frame)
        self.noise_entry.pack(pady=5)
        
        # Add info tooltip about noise
        ctk.CTkLabel(
            entries_frame, 
            text='Noise represents measurement uncertainty',
            font=('Arial', 10),
            text_color='grey'
        ).pack(pady=0)

        # Options
        self.add_point_button_frame = ctk.CTkFrame(self.add_point_window)
        self.add_point_button_frame.pack(pady=10, padx=10, fill='x')

        self.save_checkbox = ctk.CTkCheckBox(self.add_point_button_frame, text='Save to file')
        self.save_checkbox.select()
        self.save_checkbox.pack(side='left', padx=5, pady=5)

        self.retrain_checkbox = ctk.CTkCheckBox(self.add_point_button_frame, text='Retrain model')
        self.retrain_checkbox.select()
        self.retrain_checkbox.pack(side='left', padx=5, pady=5)

        save_btn = ctk.CTkButton(self.add_point_button_frame, text='Save & Close', command=self.save_and_commit_all_pending)
        save_btn.pack(side='right', padx=5, pady=5)
    
    def load_suggestion(self, index):
        '''Load a specific suggestion into the add point dialog.'''
        if 0 <= index < len(self.pending_suggestions):
            self.current_suggestion_index = index
            # Close and reopen dialog with new suggestion
            if hasattr(self, 'add_point_window'):
                self.add_point_window.destroy()
            self.add_point()

    def _save_current_and_load(self, index):
        """Save current dialog edits back to pending_suggestions then load another suggestion."""
        try:
            # Save current edits
            self._save_current_dialog_state()
        except Exception:
            pass
        # Load requested suggestion
        self.load_suggestion(index)

    def _save_current_dialog_state(self):
        """Persist current Add Point dialog fields into pending_suggestions[current_index]."""
        if not (hasattr(self, 'var_entries') and self.var_entries):
            return
        if not (self.pending_suggestions and 0 <= self.current_suggestion_index < len(self.pending_suggestions)):
            return

        # Read current values
        cs = self.pending_suggestions[self.current_suggestion_index]
        for var_name, entry in self.var_entries.items():
            try:
                cs[var_name] = entry.get()
            except Exception:
                cs[var_name] = ''

        # Output and noise
        try:
            cs['Output'] = self.output_entry.get()
        except Exception:
            pass
        try:
            noise_val = self.noise_entry.get().strip()
            if noise_val:
                cs['Noise'] = float(noise_val)
        except Exception:
            # leave as-is if parse fails
            pass

    def save_new_point(self):
        '''Saves the new point with proper iteration and reason tracking.'''
        new_point = {var: entry.get() for var, entry in self.var_entries.items()}
        new_point['Output'] = self.output_entry.get()
        
        # Add noise if provided
        noise_value = self.noise_entry.get().strip()
        if noise_value:
            try:
                new_point['Noise'] = float(noise_value)
            except ValueError:
                print(f"Invalid noise value '{noise_value}'. Using default.")
                new_point['Noise'] = 1e-6
        elif 'Noise' in self.exp_df.columns:
            # If noise column exists but no value provided, use default
            new_point['Noise'] = 1e-6

        # If this corresponds to a pending suggestion, preserve Iteration and Reason
        if self.pending_suggestions and self.current_suggestion_index < len(self.pending_suggestions):
            ps = self.pending_suggestions[self.current_suggestion_index]
            new_point['Iteration'] = int(ps.get('Iteration', getattr(self.experiment_manager, '_current_iteration', 0)))
            new_point['Reason'] = ps.get('_reason', ps.get('Reason', 'Acquisition'))
            # Remove this suggestion from pending list
            self.pending_suggestions.pop(self.current_suggestion_index)
            if self.current_suggestion_index >= len(self.pending_suggestions):
                self.current_suggestion_index = max(0, len(self.pending_suggestions) - 1)
        else:
            # Manual entry uses current iteration and Manual reason
            new_point['Iteration'] = int(getattr(self.experiment_manager, '_current_iteration', 0))
            new_point['Reason'] = 'Manual'

        # Add the new point to the exp_df and update sheet
        new_point_df = pd.DataFrame([new_point])
        # Ensure exp_df has the right columns (merge if empty)
        if self.exp_df is None or self.exp_df.empty:
            self.exp_df = new_point_df.copy()
        else:
            self.exp_df = pd.concat([self.exp_df, new_point_df], ignore_index=True)

        # Update the tksheet and headers
        headers = self.exp_df.columns.tolist()
        self.exp_sheet.set_sheet_data(self.exp_df.values.tolist())
        try:
            self.exp_sheet.set_header_data(headers)
        except Exception:
            pass
        
        # Sync to session (adds iteration/reason and saves session)
        if hasattr(self, 'session') and self.session:
            try:
                # Extract inputs dict (remove Output and Noise)
                inputs = {k: v for k, v in new_point.items() if k not in ['Output', 'Noise']}
                output_val = float(new_point['Output'])
                noise_val = new_point.get('Noise', None)
                
                # Add to session with determined reason
                self.session.add_experiment(
                    inputs=inputs,
                    output=output_val,
                    noise=noise_val,
                    reason=new_point.get('Reason', 'Manual')
                )

                # Auto-save session if we have a file path
                if hasattr(self, 'current_session_file') and self.current_session_file:
                    try:
                        self.session.save_session(self.current_session_file)
                        print(f"Point added (Iteration {self.session.experiment_manager._current_iteration}, Reason: {new_point.get('Reason', 'Manual')})")
                        print(f"Session auto-saved to {self.current_session_file}")
                    except Exception as e:
                        print(f"Warning: Failed to auto-save session: {e}")
            except Exception as e:
                print(f"Warning: Failed to sync point to session: {e}")

        # Save to file if checkbox is checked
        if self.save_checkbox.get():
            if hasattr(self, 'exp_file_path') and self.exp_file_path:
                # Auto-save to existing file without prompting
                self.exp_df.to_csv(self.exp_file_path, index=False)
                print(f"Experiments saved to {self.exp_file_path}")
            else:
                # Ask for file path if not set
                file_path = filedialog.asksaveasfilename(
                    title='Save Experiments CSV',
                    defaultextension='.csv',
                    filetypes=[('CSV Files', '*.csv')]
                )
                if file_path:
                    self.exp_df.to_csv(file_path, index=False)
                    self.exp_file_path = file_path
                    print(f"Experiments saved to {file_path}")
        
        if self.retrain_checkbox.get():
            self.retrain_model()

        self.add_point_window.destroy()

    def retrain_model(self):
        print('Retraining model with new data...')
        self.model_frame.train_model_threaded()

    def run_selected_strategy(self):
        """DEPRECATED: Use AcquisitionPanel.run_selected_strategy() instead."""
        print("WARNING: run_selected_strategy() is deprecated and no longer functional")
        print("Please use the Acquisition Panel in the UI")
        return
        # """Executes the selected acquisition strategy."""
        # strategy = self.strategy_var.get()
        # try:
        #     if strategy == "Expected Improvement (EI)":
        #         self.next_point = select_EMOC(
        #             self.pool,
        #             self.exp_df.drop(columns='Output'),
        #             self.exp_df['Output'],
        #             self.search_space,
        #             model=self.gpr_model
        #         )
        #     elif strategy == "Upper Confidence Bound (UCB)":
        #         self.next_point = select_optimize(
        #             self.search_space,
        #             self.exp_df,
        #             base_estimator=self.gpr_model,
        #             acq_func="ucb"
        #         )
        #     elif strategy == "Probability of Improvement (PI)":
        #         self.next_point = select_optimize(
        #             self.search_space,
        #             self.exp_df,
        #             base_estimator=self.gpr_model,
        #             acq_func="pi"
        #         )
        #     elif strategy == "Thompson Sampling":
        #         # Implement Thompson Sampling logic here
        #         print("Thompson Sampling is not yet implemented.")
        #         return
        #     elif strategy == "Entropy Search":
        #         # Implement Entropy Search logic here
        #         print("Entropy Search is not yet implemented.")
        #         return
        #     elif strategy == "Custom Strategy":
        #         # Allow the user to define a custom strategy
        #         print("Custom Strategy is not yet implemented.")
        #         return
        #     elif strategy == "EMOC (Exploration)":
        #         # Comment out import and implementation
        #         # from logic.acquisition.emoc_acquisition import EMOCAcquisition
        #         
        #         # Placeholder message
        #         print("EMOC acquisition function not implemented in this version.")
        #         return
        #         
        #         # # Create acquisition function using trained model
        #         # acquisition = EMOCAcquisition(
        #         #     search_space=self.main_app.search_space,
        #         #     model=self.main_app.gpr_model,
        #         #     random_state=42
        #         # )
        #         # 
        #         # # Update with existing data
        #         # if hasattr(acquisition, 'update'):
        #         #     acquisition.update(
        #         #         self.main_app.exp_df.drop(columns='Output'),
        #         #         self.main_app.exp_df['Output']
        #         #     )
        #         # 
        #         # # Generate a pool if needed
        #         # if not hasattr(self.main_app, 'pool') or self.main_app.pool is None:
        #         #     from logic.pool import generate_pool
        #         #     self.main_app.pool = generate_pool(
        #         #         self.main_app.search_space, 
        #         #         self.main_app.exp_df, 
        #         #         pool_size=5000
        #         #     )
        #         # 
        #         # # Get next point
        #         # next_point = acquisition.select_next(self.main_app.pool)
        #         # 
        #         # # acq_func_kwargs for result data
        #         # acq_func_kwargs = {}
        #         
        #     elif strategy == "GandALF (Clustering + EMOC)":
        #         # Comment out import and implementation
        #         # from logic.acquisition.gandalf_acquisition import GandALFAcquisition
        #         
        #         # Placeholder message
        #         print("GandALF acquisition function not implemented in this version.")
        #         return
        #         
        #         # # Create acquisition instance
        #         # acquisition = GandALFAcquisition(
        #         #     search_space=self.main_app.search_space,
        #         #     model=self.main_app.gpr_model,
        #         #     random_state=42
        #         # )
        #         # 
        #         # # Update with existing data
        #         # acquisition.update(
        #         #     self.main_app.exp_df.drop(columns='Output'),
        #         #     self.main_app.exp_df['Output']
        #         # )
        #         # 
        #         # # Generate a pool if needed
        #         # if not hasattr(self.main_app, 'pool') or self.main_app.pool is None:
        #         #     from logic.pool import generate_pool
        #         #     self.main_app.pool = generate_pool(
        #         #         self.main_app.search_space, 
        #         #         self.main_app.exp_df,
        #         #         pool_size=5000
        #         #     )
        #         # 
        #         # # Get next point
        #         # next_point = acquisition.select_next(self.main_app.pool)
        #         # 
        #         # # acq_func_kwargs for result data
        #         # acq_func_kwargs = {'clustering': True}
        #     else:
        #         print("Unknown strategy selected.")
        #         return
        # 
        #     self.update_pool_plot()
        #     print(f"Strategy '{strategy}' executed successfully.")
        # except Exception as e:
        #     print(f"Error executing strategy '{strategy}': {e}")

    def toggle_tabbed_layout(self):
        """Toggle between side-by-side and tabbed layout"""
        if self.using_tabs:
            # Switch to side-by-side layout
            self.using_tabs = False
            
            # Store trained model and UI state for transfer
            trained_model = getattr(self, 'gpr_model', None)
            visualizations = None
            acq_enabled = False
            advanced_enabled = False
            current_backend = "scikit-learn"
            
            if hasattr(self, 'model_frame'):
                if hasattr(self.model_frame, 'visualizations'):
                    visualizations = self.model_frame.visualizations
                if hasattr(self.model_frame, 'advanced_var'):
                    advanced_enabled = self.model_frame.advanced_var.get()
                if hasattr(self.model_frame, 'backend_var'):
                    current_backend = self.model_frame.backend_var.get()
            
            if hasattr(self, 'acquisition_panel'):
                acq_enabled = self.acquisition_panel.run_button.cget("state") == "normal"
            
            # Remove the tabbed interface
            if hasattr(self, 'tab_view'):
                self.tab_view.destroy()
            
            # Create side-by-side layout
            self._create_model_frame()
            self._create_acquisition_frame()
            
            # Transfer the model and state
            if trained_model:
                self.model_frame.gpr_model = trained_model
                if visualizations:
                    self.model_frame.visualizations = visualizations
                    self.model_frame.visualize_button.configure(state="normal")
            
            # Set advanced options state
            self.model_frame.advanced_var.set(advanced_enabled)
            self.model_frame.toggle_advanced_options()
            
            # Set backend
            self.model_frame.backend_var.set(current_backend)
            self.model_frame.load_backend_options()
            
            # Set acquisition panel state
            if acq_enabled:
                self.acquisition_panel.enable()
            
            print("Switched to side-by-side layout")
        else:
            # Switch to tabbed layout
            self.using_tabs = True
            
            # Store trained model and UI state for transfer
            trained_model = getattr(self, 'gpr_model', None)
            visualizations = None
            acq_enabled = False
            advanced_enabled = False
            current_backend = "scikit-learn"
            
            if hasattr(self, 'model_frame'):
                if hasattr(self.model_frame, 'visualizations'):
                    visualizations = self.model_frame.visualizations
                if hasattr(self.model_frame, 'advanced_var'):
                    advanced_enabled = self.model_frame.advanced_var.get()
                if hasattr(self.model_frame, 'backend_var'):
                    current_backend = self.model_frame.backend_var.get()
            
            if hasattr(self, 'acquisition_panel'):
                acq_enabled = self.acquisition_panel.run_button.cget("state") == "normal"
            
            # Unpack existing frames
            if hasattr(self, 'model_frame'):
                self.model_frame.pack_forget()
            if hasattr(self, 'acq_frame'):
                self.acq_frame.pack_forget()
                
            # Create tabbed interface
            self.tab_view = ctk.CTkTabview(self)
            self.tab_view.pack(side='right', fill='both', padx=10, pady=10)
            self.tab_view.configure(width=300)
            
            # Add tabs
            self.tab_view.add("Model")
            self.tab_view.add("Acquisition")
            
            # Set the default tab
            self.tab_view.set("Model")
            
            # Create panels inside tabs
            self.model_frame = GaussianProcessPanel(self.tab_view.tab("Model"), self)
            self.model_frame.pack(fill='both', expand=True)
            
            self.acquisition_panel = AcquisitionPanel(self.tab_view.tab("Acquisition"), self)
            self.acquisition_panel.pack(fill='both', expand=True)
            
            # Transfer the model and state
            if trained_model:
                self.model_frame.gpr_model = trained_model
                if visualizations:
                    self.model_frame.visualizations = visualizations
                    self.model_frame.visualize_button.configure(state="normal")
            
            # Set advanced options state
            self.model_frame.advanced_var.set(advanced_enabled)
            self.model_frame.toggle_advanced_options()
            
            # Set backend
            self.model_frame.backend_var.set(current_backend)
            self.model_frame.load_backend_options()
            
            # Set acquisition panel state
            if acq_enabled:
                self.acquisition_panel.enable()
            
            print("Switched to tabbed layout for small screens")
            
    def switch_tab(self, tab_name):
        """Switch between model and acquisition tabs"""
        if tab_name == "Model":
            if hasattr(self, 'acquisition_panel'):
                self.acquisition_panel.pack_forget()
            if hasattr(self, 'model_frame'):
                self.model_frame.pack(in_=self.right_panel, fill='both', expand=True)
        else:  # Acquisition
            if hasattr(self, 'model_frame'):
                self.model_frame.pack_forget()
            if hasattr(self, 'acquisition_panel'):
                self.acquisition_panel.pack(in_=self.right_panel, fill='both', expand=True)

    def toggle_noise_column(self):
        """Show or hide the noise column from view without deleting the data"""
        # Update dataframe from UI
        self.update_exp_df_from_sheet()
        
        has_noise = 'Noise' in self.exp_df.columns
        
        if has_noise:
            # Instead of removing, just hide from view
            visible_df = self.exp_df.drop(columns=['Noise'])
            print("Noise column hidden from view (data is preserved).")
            self.noise_column_hidden = True
        else:
            # Add noise column if it doesn't exist
            if hasattr(self, 'noise_column_hidden') and self.noise_column_hidden:
                # Restore from backup if we were hiding it
                visible_df = self.exp_df
                self.noise_column_hidden = False
                print("Noise column restored to view.")
            else:
                # Add new noise column with default value
                self.exp_df['Noise'] = 1e-6
                visible_df = self.exp_df
                self.noise_column_hidden = False
                print("Noise column added with default value 1e-6.")
        
        # Update UI with visible columns (not modifying actual data)
        self.exp_sheet.set_sheet_data(visible_df.values.tolist())
        self.exp_sheet.set_header_data(visible_df.columns.tolist())
        self.exp_sheet.set_all_column_widths()
        
        # No need to reset model since the actual data structure isn't changing
        print("Note: Toggle only affects display. Model training will use noise if present.")
    
    # Removed: toggle_session_api method (session API is always enabled)
    
    def _sync_data_to_session(self):
        """Synchronize current UI state (variables and experiments) to the session."""
        try:
            # Sync search space from search_space_manager
            if hasattr(self, 'search_space_manager') and len(self.search_space_manager.variables) > 0:
                # Import the core SearchSpace to avoid confusion
                from alchemist_core.data.search_space import SearchSpace as CoreSearchSpace
                
                # Clear session search space and rebuild
                self.session.search_space = CoreSearchSpace()
                
                # Variables is a LIST of dictionaries, not a dict
                for var_dict in self.search_space_manager.variables:
                    var_name = var_dict['name']
                    var_type = var_dict['type']
                    
                    if var_type in ['real', 'integer']:
                        self.session.add_variable(
                            var_name, 
                            var_type, 
                            bounds=(var_dict['min'], var_dict['max'])
                        )
                    elif var_type == 'categorical':
                        self.session.add_variable(
                            var_name,
                            var_type,
                            categories=var_dict['values']
                        )
                print(f"Synced {len(self.session.search_space.variables)} variables to session")
            
            # Sync experiment data
            if hasattr(self, 'exp_df') and len(self.exp_df) > 0:
                # Ensure metadata columns have correct types
                exp_df_clean = self.exp_df.copy()
                
                # Define metadata columns (including configured target columns)
                target_cols = set(self.experiment_manager.target_columns) if hasattr(self.experiment_manager, 'target_columns') else {'Output'}
                metadata_cols = target_cols | {'Noise', 'Iteration', 'Reason'}
                
                # Ensure Iteration is numeric
                if 'Iteration' in exp_df_clean.columns:
                    exp_df_clean['Iteration'] = pd.to_numeric(exp_df_clean['Iteration'], errors='coerce').fillna(0).astype(int)
                
                # Ensure Reason is string
                if 'Reason' in exp_df_clean.columns:
                    exp_df_clean['Reason'] = exp_df_clean['Reason'].astype(str).replace('nan', 'Manual')
                
                # Ensure target columns are numeric
                for target_col in target_cols:
                    if target_col in exp_df_clean.columns:
                        exp_df_clean[target_col] = pd.to_numeric(exp_df_clean[target_col], errors='coerce')
                
                # Ensure Noise is numeric if present
                if 'Noise' in exp_df_clean.columns:
                    exp_df_clean['Noise'] = pd.to_numeric(exp_df_clean['Noise'], errors='coerce')
                
                # Get categorical variable names from search space
                categorical_vars = []
                if hasattr(self.session, 'search_space') and hasattr(self.session.search_space, 'variables'):
                    categorical_vars = [v['name'] for v in self.session.search_space.variables if v.get('type') == 'categorical']
                
                # Ensure all feature columns are numeric (except categoricals)
                for col in exp_df_clean.columns:
                    if col not in metadata_cols:
                        if col in categorical_vars:
                            # Keep as string for categorical variables
                            exp_df_clean[col] = exp_df_clean[col].astype(str)
                        else:
                            # Convert to numeric for real/integer variables  
                            exp_df_clean[col] = pd.to_numeric(exp_df_clean[col], errors='coerce')
                
                # Verify no NaN in non-nullable columns and drop bad rows
                required_cols = [col for col in exp_df_clean.columns if col not in ['Noise', 'Reason']]
                n_before = len(exp_df_clean)
                exp_df_clean = exp_df_clean.dropna(subset=required_cols)
                n_after = len(exp_df_clean)
                
                if n_after < n_before:
                    print(f"WARNING: Dropped {n_before - n_after} rows with invalid/missing data")
                
                if len(exp_df_clean) == 0:
                    print("ERROR: No valid experiment data after cleaning!")
                    return
                
                # Copy cleaned data to session's experiment manager
                self.session.experiment_manager.df = exp_df_clean
                
                # Copy target_columns configuration to session's experiment manager
                if hasattr(self.experiment_manager, 'target_columns'):
                    self.session.experiment_manager.target_columns = self.experiment_manager.target_columns
                
                # Update local exp_df with cleaned version
                self.exp_df = exp_df_clean
                
                # Set search space in experiment manager
                self.session.experiment_manager.set_search_space(self.session.search_space)
                
                print(f"Synced {len(self.exp_df)} experiments to session")
                
        except Exception as e:
            print(f"Error syncing data to session: {e}")
            import traceback
            traceback.print_exc()
    
    # ============================================================
    # Session Management Methods
    # ============================================================
    
    def new_session(self):
        """Create a new session."""
        # Ask if user wants to save current session
        if hasattr(self.session, 'audit_log') and len(self.session.audit_log.entries) > 0:
            response = tk.messagebox.askyesnocancel("Save Current Session?", 
                "Would you like to save the current session before creating a new one?")
            if response is None:  # Cancel
                return
            elif response:  # Yes
                self.save_session_cmd()
        
        # Reset session
        self.session = OptimizationSession()
        self.session.events.on('progress', self._on_session_progress)
        self.session.events.on('model_trained', self._on_session_model_trained)
        self.session.events.on('model_retrained', self._on_session_model_retrained)
        self.session.events.on('suggestions_ready', self._on_session_suggestions)
        
        # Clear UI
        self.search_space_manager = SearchSpace()
        self.experiment_manager = ExperimentManager()
        self.exp_df = pd.DataFrame()
        self.var_sheet.set_sheet_data([])
        self.exp_sheet.set_sheet_data([])
        
        print("New session created")
        # Prompt user to name the session immediately
        dialog = SessionMetadataDialog(self, self.session)
        dialog.grab_set()
        self.wait_window(dialog)

        if getattr(dialog, 'saved', False):
            tk.messagebox.showinfo("New Session", "New session created successfully.")
    
    def open_session(self):
        """Open a session from a JSON file."""
        filepath = filedialog.askopenfilename(
            title="Open Session",
            filetypes=[("JSON Session Files", "*.json"), ("All Files", "*.*")],
            defaultextension=".json"
        )
        
        if not filepath:
            return
        
        try:
            # Load session from file
            self.session = OptimizationSession.load_session(filepath)
            
            # Reconnect event handlers
            self.session.events.on('progress', self._on_session_progress)
            self.session.events.on('model_trained', self._on_session_model_trained)
            self.session.events.on('model_retrained', self._on_session_model_retrained)
            self.session.events.on('suggestions_ready', self._on_session_suggestions)
            
            # Sync session data to UI
            self._sync_session_to_ui()
            
            # Store the current filepath for "Save" command
            self.current_session_file = filepath

            # If the loaded session has no meaningful name, default to the
            # filename (basename without extension).
            current_name = (self.session.metadata.name or "").strip()
            if not current_name or current_name.lower().startswith("untitled"):
                try:
                    basename = os.path.splitext(os.path.basename(filepath))[0]
                    # Update session metadata via provided API (keeps modified timestamp)
                    self.session.update_metadata(name=basename)
                except Exception:
                    # Fall back to leaving whatever the session provided
                    pass

            # Update application title to reflect the opened session
            try:
                self.title(f"ALchemist - {self.session.metadata.name}")
            except Exception:
                pass

            print(f"Session loaded from {filepath}")
            tk.messagebox.showinfo("Session Loaded", 
                f"Session '{self.session.metadata.name}' loaded successfully.")
            
        except Exception as e:
            tk.messagebox.showerror("Error Loading Session", 
                f"Failed to load session:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def _on_session_model_retrained(self, event_data):
        """Handle model retraining completion after session load."""
        backend = event_data.get('backend', 'unknown')
        print(f"Session: Model retrained successfully ({backend})")
        
        # Sync session model to main_app.gpr_model
        self.gpr_model = self.session.model
        
        # Enable UI elements
        if hasattr(self, 'gpr_panel'):
            self.gpr_panel.visualize_button.configure(state="normal")
        
        if hasattr(self, 'acq_panel'):
            self.acq_panel.enable()
        
        # Show notification
        tk.messagebox.showinfo("Model Retrained", 
            f"Model retrained successfully using {backend} backend.")

    
    def save_session_cmd(self):
        """Save the current session (use existing file or prompt for new)."""
        # Sync current UI state to session
        self._sync_data_to_session()
        
        if hasattr(self, 'current_session_file') and self.current_session_file:
            # Use existing file
            try:
                self.session.save_session(self.current_session_file)
                print(f"Session saved to {self.current_session_file}")
                tk.messagebox.showinfo("Session Saved", 
                    f"Session saved successfully.")
            except Exception as e:
                tk.messagebox.showerror("Error Saving Session", 
                    f"Failed to save session:\n{str(e)}")
        else:
            # No existing file: this is effectively an autosave point. If the
            # session is unnamed, ask the user if they'd like to create/save a
            # named session first.
            if self.maybe_prompt_create_session():
                # User either created/confirmed a session name; prompt for save-as
                self.save_session_as()
            else:
                # User declined to create a session; still offer Save As as optional
                # but don't force it. We return early.
                return

    
    def save_session_as(self):
        """Save the current session to a new file."""
        # Sync current UI state to session
        self._sync_data_to_session()
        
        # Suggest filename from session metadata
        default_name = self.session.metadata.name or "alchemist_session"
        # Sanitize filename
        import re
        default_name = re.sub(r'[^\w\s-]', '', default_name).strip().replace(' ', '_')
        default_name = f"{default_name}.json"
        
        filepath = filedialog.asksaveasfilename(
            title="Save Session As",
            filetypes=[("JSON Session Files", "*.json"), ("All Files", "*.*")],
            defaultextension=".json",
            initialfile=default_name
        )
        
        if not filepath:
            return
        
        try:
            self.session.save_session(filepath)
            self.current_session_file = filepath
            print(f"Session saved to {filepath}")
            tk.messagebox.showinfo("Session Saved", 
                f"Session saved successfully to:\n{filepath}")
        except Exception as e:
            tk.messagebox.showerror("Error Saving Session", 
                f"Failed to save session:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def export_audit_log(self):
        """Export the audit log to a markdown file."""
        if not hasattr(self.session, 'audit_log') or len(self.session.audit_log.entries) == 0:
            tk.messagebox.showwarning("No Audit Log", 
                "No audit log entries to export.")
            return
        
        # Suggest filename
        default_name = f"audit_log_{self.session.metadata.session_id[:8]}.md"
        
        filepath = filedialog.asksaveasfilename(
            title="Export Audit Log",
            filetypes=[("Markdown Files", "*.md"), ("All Files", "*.*")],
            defaultextension=".md",
            initialfile=default_name
        )
        
        if not filepath:
            return
        
        try:
            markdown = self.session.audit_log.to_markdown()
            with open(filepath, 'w') as f:
                f.write(markdown)
            print(f"Audit log exported to {filepath}")
            tk.messagebox.showinfo("Audit Log Exported", 
                f"Audit log exported successfully to:\n{filepath}")
        except Exception as e:
            tk.messagebox.showerror("Error Exporting Audit Log", 
                f"Failed to export audit log:\n{str(e)}")
    
    def edit_session_metadata(self):
        """Open a dialog to edit session metadata."""
        dialog = SessionMetadataDialog(self, self.session)
        dialog.grab_set()  # Make modal
        self.wait_window(dialog)
    
    def _sync_session_to_ui(self):
        """Sync session data to UI components."""
        try:
            # Sync search space
            if self.session.search_space and len(self.session.search_space.variables) > 0:
                self.search_space_manager = self.session.search_space
                self.search_space = self.session.search_space.to_skopt()
                
                # Update variable sheet
                var_data = []
                for var_dict in self.session.search_space.variables:
                    if var_dict['type'] in ['real', 'integer']:
                        var_data.append([
                            var_dict['name'],
                            var_dict['type'],
                            var_dict['min'],
                            var_dict['max'],
                            ''
                        ])
                    else:  # categorical
                        var_data.append([
                            var_dict['name'],
                            var_dict['type'],
                            '',
                            '',
                            ', '.join(map(str, var_dict['values']))
                        ])
                
                self.var_sheet.set_sheet_data(var_data)
                self.var_sheet.set_all_column_widths()
            
            # Sync experiment data
            if hasattr(self.session.experiment_manager, 'df') and len(self.session.experiment_manager.df) > 0:
                self.experiment_manager = self.session.experiment_manager
                self.exp_df = self.session.experiment_manager.df.copy()
                
                # Update experiment sheet
                self.exp_sheet.set_sheet_data(self.exp_df.values.tolist())
                self.exp_sheet.set_header_data(self.exp_df.columns.tolist())
                self.exp_sheet.set_all_column_widths()
            
            # Update UI state
            self._update_ui_state()
            
            # Detect multi-objective from session's experiment manager
            target_cols = getattr(self.session.experiment_manager, 'target_columns', ['Output'])
            is_mobo = len(target_cols) > 1
            if is_mobo:
                print(f"Session loaded with multi-objective targets: {target_cols}")
            if hasattr(self, 'model_frame') and self.model_frame is not None:
                self.model_frame.set_multi_objective_mode(
                    is_mobo,
                    objective_names=target_cols if is_mobo else []
                )
            
        except Exception as e:
            print(f"Error syncing session to UI: {e}")
            import traceback
            traceback.print_exc()

    def maybe_prompt_create_session(self) -> bool:
        """If the current session is unnamed, prompt the user to create session metadata.

        Returns True if the user created/confirmed a session name (or one already existed),
        False if the user declined.
        """
        try:
            name = (self.session.metadata.name or "").strip()
            if name and not name.lower().startswith('untitled'):
                return True

            resp = tk.messagebox.askyesno(
                "Create Session?",
                "This workspace is not associated with a saved session. Would you like to create a session now?"
            )
            if not resp:
                return False

            dialog = SessionMetadataDialog(self, self.session)
            dialog.grab_set()
            self.wait_window(dialog)
            # Return True if the session now has a name
            return bool((self.session.metadata.name or "").strip())
        except Exception:
            return False

    def _quit(self):
        """Handle window close with Save / Don't Save / Cancel dialog.

        - Yes: prompt Save As if no file, then quit on success
        - No: quit without saving
        - Cancel: abort close
        """
        try:
            if hasattr(self.session, 'audit_log') and len(self.session.audit_log.entries) > 0:
                resp = tk.messagebox.askyesnocancel(
                    "Save Session?",
                    "Would you like to save the current session before quitting?"
                )
                if resp is None:
                    # Cancel
                    return
                if resp:
                    # Yes -> save
                    if hasattr(self, 'current_session_file') and self.current_session_file:
                        try:
                            self._sync_data_to_session()
                            self.session.save_session(self.current_session_file)
                        except Exception as e:
                            tk.messagebox.showerror("Save Failed", f"Failed to save session:\n{e}")
                            return
                    else:
                        # No existing file -> prompt for save-as
                        if not self.maybe_prompt_create_session():
                            # User declined to create a session, abort quit
                            return
                        self.save_session_as()
            # If we get here, either user chose No or save succeeded
            self.destroy()
        except Exception:
            # Fallback: destroy the window
            try:
                self.destroy()
            except Exception:
                pass


# ============================================================
# Session Metadata Dialog
# ============================================================

class SessionMetadataDialog(ctk.CTkToplevel):
    """Dialog for editing session metadata."""
    
    def __init__(self, parent, session):
        super().__init__(parent)
        self.session = session
        self.title("Session Metadata")
        # Taller dialog to accommodate fields on smaller displays
        self.geometry("560x560")
        self.minsize(480, 420)
        self.saved = False
        
        # Name field
        ctk.CTkLabel(self, text="Session Name:", font=('Arial', 12)).pack(pady=(10, 0))
        self.name_entry = ctk.CTkEntry(self, width=400)
        self.name_entry.pack(pady=5)
        self.name_entry.insert(0, session.metadata.name or "")
        
        # Description field
        ctk.CTkLabel(self, text="Description:", font=('Arial', 12)).pack(pady=(10, 0))
        self.desc_text = ctk.CTkTextbox(self, width=400, height=100)
        self.desc_text.pack(pady=5)
        self.desc_text.insert("1.0", session.metadata.description or "")
        
        # Tags field
        ctk.CTkLabel(self, text="Tags (comma-separated):", font=('Arial', 12)).pack(pady=(10, 0))
        self.tags_entry = ctk.CTkEntry(self, width=400)
        self.tags_entry.pack(pady=5)
        if session.metadata.tags:
            self.tags_entry.insert(0, ", ".join(session.metadata.tags))
        
        # Author field (editable)
        ctk.CTkLabel(self, text="Author:", font=('Arial', 12)).pack(pady=(10, 0))
        self.author_entry = ctk.CTkEntry(self, width=400)
        self.author_entry.pack(pady=5)
        author_text = getattr(session.metadata, 'author', None)
        if author_text:
            self.author_entry.insert(0, author_text)
        
        # Session ID (read-only)
        ctk.CTkLabel(self, text="Session ID:", font=('Arial', 12)).pack(pady=(10, 0))
        session_id = getattr(getattr(session, 'metadata', None), 'session_id', None)
        id_text = (session_id[:16] + "...") if session_id else "(no id)"
        id_label = ctk.CTkLabel(self, text=id_text)
        id_label.pack(pady=5)
        
        # Buttons
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(pady=20)
        
        save_btn = ctk.CTkButton(button_frame, text="Save", command=self.save_metadata)
        save_btn.pack(side='left', padx=5)
        
        cancel_btn = ctk.CTkButton(button_frame, text="Cancel", command=self._on_cancel)
        cancel_btn.pack(side='left', padx=5)
    
    def save_metadata(self):
        """Save the metadata changes."""
        name = self.name_entry.get().strip()
        description = self.desc_text.get("1.0", "end-1c").strip()
        tags_str = self.tags_entry.get().strip()
        tags = [t.strip() for t in tags_str.split(',') if t.strip()]
        
        # Update session metadata
        self.session.update_metadata(
            name=name if name else None,
            description=description if description else None,
            tags=tags if tags else None,
            author=self.author_entry.get().strip() if hasattr(self, 'author_entry') else None
        )
        
        print(f"Session metadata updated: {name}")
        self.saved = True
        self.destroy()

    def _on_cancel(self):
        """Handle cancel/close without saving."""
        self.saved = False
        self.destroy()


# ============================================================
# Lock Decision Confirmation Dialog
# ============================================================

class LockDecisionDialog(ctk.CTkToplevel):
    """Dialog for confirming lock decisions with optional notes."""
    
    def __init__(self, parent, decision_type: str):
        super().__init__(parent)
        self.result = None  # Will be None (cancelled) or notes string (confirmed)
        self.decision_type = decision_type
        
        self.title(f"Lock {decision_type}")
        self.geometry("450x300")
        
        # Message
        message_text = f"Lock this {decision_type.lower()} decision to the audit log?\n\n" \
                      f"This will create an immutable record of your {decision_type.lower()} configuration."
        
        msg_label = ctk.CTkLabel(
            self, 
            text=message_text,
            font=('Arial', 12),
            wraplength=400
        )
        msg_label.pack(pady=(20, 10))
        
        # Notes field
        ctk.CTkLabel(self, text="Optional Notes:", font=('Arial', 12, 'bold')).pack(pady=(10, 5))
        
        self.notes_text = ctk.CTkTextbox(self, width=400, height=100)
        self.notes_text.pack(pady=5)
        self.notes_text.focus()
        
        # Buttons
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(pady=20)
        
        ok_btn = ctk.CTkButton(
            button_frame, 
            text="Lock Decision", 
            command=self.confirm,
            fg_color="#2B8A3E",  # Green
            hover_color="#228B22"
        )
        ok_btn.pack(side='left', padx=5)
        
        cancel_btn = ctk.CTkButton(
            button_frame, 
            text="Cancel", 
            command=self.cancel
        )
        cancel_btn.pack(side='left', padx=5)
        
        # Make modal
        self.transient(parent)
        self.grab_set()
    
    def confirm(self):
        """Confirm the lock decision."""
        self.result = self.notes_text.get("1.0", "end-1c").strip()
        self.destroy()
    
    def cancel(self):
        """Cancel the lock decision."""
        self.result = None
        self.destroy()

