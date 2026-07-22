import customtkinter as ctk
import numpy as np
import pandas as pd

from ui.notifications import ResultNotificationWindow

class AcquisitionPanel(ctk.CTkScrollableFrame):  # Changed from CTkFrame to CTkScrollableFrame
    def __init__(self, parent, main_app):
        super().__init__(parent, height=600, width=300)  # Added height and width parameters
        
        self.main_app = main_app  # Reference to main application
        
        # Title at the very top
        ctk.CTkLabel(self, text='Acquisition Functions', font=('Arial', 16)).pack(pady=5)
        
        # Create frames for different backends
        self.acq_sklearn_frame = ctk.CTkFrame(self)
        self.acq_botorch_frame = ctk.CTkFrame(self)
        self.acq_mobo_frame = ctk.CTkFrame(self)
        
        # MOBO state
        self._is_mobo = False
        self._objective_names = []
        
        # Create the widgets for each backend
        self.create_sklearn_widgets()
        self.create_botorch_widgets()
        self.create_mobo_widgets()
        
        # Initial load - always starts with scikit-learn
        self.acq_sklearn_frame.pack(fill="x", expand=True, padx=10, pady=5)
        
        # Run strategy button right after the acquisition functions (moved up)
        self.run_button = ctk.CTkButton(
            self, 
            text="Run Acquisition Strategy",
            command=self.run_selected_strategy,
            state="disabled"  # Disabled by default until model is trained
        )
        self.run_button.pack(pady=10, padx=10, fill="x")
        
        # Add separator
        separator = ctk.CTkFrame(self, height=2, fg_color="gray30")
        separator.pack(fill="x", padx=20, pady=15)
        
        # Add "Model Maximum" section
        self.create_model_maximum_section()
        
    def create_sklearn_widgets(self):
        """Create acquisition function widgets for scikit-learn logic."""
        self.acq_sklearn_var = ctk.StringVar(value="Expected Improvement (EI)")
        ctk.CTkLabel(self.acq_sklearn_frame, text="Acquisition Strategy:").pack(pady=2)
        self.acq_sklearn_menu = ctk.CTkOptionMenu(
            self.acq_sklearn_frame,
            values=["Expected Improvement (EI)", "Upper Confidence Bound (UCB)", 
                   "Probability of Improvement (PI)", "GP Hedge (Auto-balance)"],
            variable=self.acq_sklearn_var,
            command=self.update_acq_param_visibility
        )
        self.acq_sklearn_menu.pack(pady=5)
        
        # Add maximize/minimize segmented button
        ctk.CTkLabel(self.acq_sklearn_frame, text="Optimization Goal:").pack(pady=2)
        self.goal_options = ["Maximize", "Minimize"]
        self.goal_var = ctk.StringVar(value="Maximize")
        self.goal_segmented = ctk.CTkSegmentedButton(
            self.acq_sklearn_frame,
            values=self.goal_options,
            variable=self.goal_var,
        )
        self.goal_segmented.pack(pady=5)
        
        # Add acquisition function parameter widgets
        ctk.CTkLabel(self.acq_sklearn_frame, text="Acquisition Parameters:").pack(pady=2)
        
        # xi parameter frame (for EI and PI) - Updated to go down to 0.0001
        self.xi_frame = ctk.CTkFrame(self.acq_sklearn_frame)
        self.xi_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(self.xi_frame, text="ξ (xi):").pack(side="left", padx=5)
        self.xi_value_label = ctk.CTkLabel(self.xi_frame, text="0.01")
        self.xi_value_label.pack(side="right", padx=5)
        self.xi_slider = ctk.CTkSlider(
            self.xi_frame, 
            from_=0.0001,  # Changed from 0.001 to 0.0001
            to=0.1, 
            number_of_steps=999,  # Increased steps for finer control
            command=self.update_xi_value
        )
        self.xi_slider.pack(fill="x", padx=5)
        self.xi_slider.set(0.01)  # Default value
        
        # kappa parameter frame (for UCB)
        self.kappa_frame = ctk.CTkFrame(self.acq_sklearn_frame)
        self.kappa_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(self.kappa_frame, text="κ (kappa):").pack(side="left", padx=5)
        self.kappa_value_label = ctk.CTkLabel(self.kappa_frame, text="1.96")
        self.kappa_value_label.pack(side="right", padx=5)
        self.kappa_slider = ctk.CTkSlider(
            self.kappa_frame, 
            from_=0.1, 
            to=5.0, 
            number_of_steps=49,
            command=self.update_kappa_value
        )
        self.kappa_slider.pack(fill="x", padx=5)
        self.kappa_slider.set(1.96)  # Default value
        
        # Parameter info/help labels
        self.xi_info_label = ctk.CTkLabel(
            self.acq_sklearn_frame, 
            text="ξ (xi): Higher values favor exploration over exploitation", 
            font=("Arial", 10), 
            text_color="gray",
            wraplength=250
        )
        self.xi_info_label.pack(after=self.xi_frame, pady=(0, 5))

        self.kappa_info_label = ctk.CTkLabel(
            self.acq_sklearn_frame, 
            text="κ (kappa): Higher values increase exploration", 
            font=("Arial", 10), 
            text_color="gray",
            wraplength=250
        )
        self.kappa_info_label.pack(after=self.kappa_frame, pady=(0, 5))

        # Set initial visibility
        self.update_acq_param_visibility(self.acq_sklearn_var.get())

    def create_botorch_widgets(self):
        """Create acquisition function widgets for BoTorch logic."""
        # Create a frame for the segmented button that will always be visible
        self.botorch_type_frame = ctk.CTkFrame(self.acq_botorch_frame)
        self.botorch_type_frame.pack(fill="x", expand=True, pady=5)
        
        # Create tabs for regular, batch, and exploratory acquisition functions in this separate frame
        ctk.CTkLabel(self.botorch_type_frame, text="Acquisition Type:").pack(pady=2)
        self.botorch_acq_type_var = ctk.StringVar(value="Regular")
        self.botorch_acq_type_seg = ctk.CTkSegmentedButton(
            self.botorch_type_frame,
            values=["Regular", "Batch", "Exploratory"],
            variable=self.botorch_acq_type_var,
            command=self.update_botorch_acq_type
        )
        self.botorch_acq_type_seg.pack(pady=5)
        
        # Create subframes for different acquisition function types
        self.botorch_acq_frame = ctk.CTkFrame(self.acq_botorch_frame)
        self.botorch_batch_frame = ctk.CTkFrame(self.acq_botorch_frame)
        self.botorch_exploratory_frame = ctk.CTkFrame(self.acq_botorch_frame)
        
        # Regular acquisition functions
        self.acq_botorch_var = ctk.StringVar(value="Expected Improvement")  # Changed default
        ctk.CTkLabel(self.botorch_acq_frame, text="Acquisition Function:").pack(pady=2)
        self.acq_botorch_menu = ctk.CTkOptionMenu(
            self.botorch_acq_frame,
            values=[
                "Expected Improvement",
                "Log Expected Improvement",
                "Probability of Improvement",
                "Log Probability of Improvement",
                "Upper Confidence Bound"
            ],
            variable=self.acq_botorch_var,
            command=self.update_botorch_description
        )
        self.acq_botorch_menu.pack(pady=5)
        
        # Batch acquisition functions
        self.acq_botorch_batch_var = ctk.StringVar(value="q-Expected Improvement")
        ctk.CTkLabel(self.botorch_batch_frame, text="Batch Acquisition Function:").pack(pady=2)
        self.acq_botorch_batch_menu = ctk.CTkOptionMenu(
            self.botorch_batch_frame,
            values=[
                "q-Expected Improvement",
                "q-Upper Confidence Bound"
            ],
            variable=self.acq_botorch_batch_var,
            command=self.update_botorch_description
        )
        self.acq_botorch_batch_menu.pack(pady=5)
        
        # Batch size selection
        ctk.CTkLabel(self.botorch_batch_frame, text="Batch Size (q):").pack(pady=2)
        self.botorch_batch_size_var = ctk.IntVar(value=2)
        batch_sizes = [str(i) for i in range(2, 11)]  # 2 to 10
        self.botorch_batch_size_menu = ctk.CTkOptionMenu(
            self.botorch_batch_frame,
            values=batch_sizes,
            command=lambda v: self.botorch_batch_size_var.set(int(v))
        )
        self.botorch_batch_size_menu.pack(pady=5)
        
        # Beta parameter for q-UCB
        self.botorch_beta_frame = ctk.CTkFrame(self.botorch_batch_frame)
        self.botorch_beta_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(self.botorch_beta_frame, text="β (beta):").pack(side="left", padx=5)
        self.botorch_beta_value_label = ctk.CTkLabel(self.botorch_beta_frame, text="0.5")
        self.botorch_beta_value_label.pack(side="right", padx=5)
        self.botorch_beta_slider = ctk.CTkSlider(
            self.botorch_beta_frame, 
            from_=0.1, 
            to=2.0, 
            number_of_steps=19,
            command=self.update_botorch_beta_value
        )
        self.botorch_beta_slider.pack(fill="x", padx=5)
        self.botorch_beta_slider.set(0.5)  # Default value
        
        # Exploratory acquisition functions
        ctk.CTkLabel(
            self.botorch_exploratory_frame, 
            text="Integrated Posterior Variance",
            font=("Arial", 12, "bold")
        ).pack(pady=5)
        
        ctk.CTkLabel(
            self.botorch_exploratory_frame,
            text="This purely exploratory acquisition function selects points to reduce model uncertainty, "
                 "without considering optimization goals.",
            wraplength=250,
            justify="left"
        ).pack(pady=5)
        
        # Number of MC points for exploratory acquisition
        ctk.CTkLabel(self.botorch_exploratory_frame, text="MC Integration Points:").pack(pady=2)
        self.botorch_mc_points_var = ctk.IntVar(value=1000)
        
        def validate_mc_points(action, value_if_allowed):
            """Validate MC points entry to prevent empty values"""
            if value_if_allowed == "":
                return False
            try:
                int(value_if_allowed)
                return True
            except ValueError:
                return False

        # Use validation for the MC points entry
        vcmd = (self.register(validate_mc_points), '%d', '%P')
        self.botorch_mc_points_entry = ctk.CTkEntry(
            self.botorch_exploratory_frame,
            textvariable=self.botorch_mc_points_var,
            width=100,
            validate="key",
            validatecommand=vcmd
        )
        self.botorch_mc_points_entry.pack(pady=5)
        
        # Add explanation for MC points
        ctk.CTkLabel(
            self.botorch_exploratory_frame, 
            text="Higher values improve accuracy but increase computation time.\nTypically 500-2000 points works well.",
            font=("Arial", 10), 
            text_color="gray",
            wraplength=250
        ).pack(pady=(0, 5))
        
        # Add maximize/minimize option for regular and batch acquisitions
        ctk.CTkLabel(self.acq_botorch_frame, text="Optimization Goal:").pack(pady=2)
        self.botorch_goal_options = ["Maximize", "Minimize"]
        self.botorch_goal_var = ctk.StringVar(value="Maximize")
        self.botorch_goal_segmented = ctk.CTkSegmentedButton(
            self.acq_botorch_frame,
            values=self.botorch_goal_options,
            variable=self.botorch_goal_var,
        )
        self.botorch_goal_segmented.pack(pady=5)
        
        # Add description of selected acquisition function
        descrip_frame = ctk.CTkFrame(self.acq_botorch_frame)
        descrip_frame.pack(fill="x", padx=10, pady=5)
        
        # Strategy descriptions dictionary
        self.strategy_info = {
            "Expected Improvement": (
                "Expected Improvement (EI) balances exploration and exploitation by calculating "
                "the expected improvement over the current best observed value."
            ),
            "Log Expected Improvement": (
                "LogEI is a numerically stable version of EI. It balances exploration and exploitation "
                "by calculating the log of the expected improvement over the current best value."
            ),
            "Probability of Improvement": (
                "Probability of Improvement (PI) selects points based on the probability that they "
                "will improve over the current best observed value."
            ),
            "Log Probability of Improvement": (
                "LogPI is a numerically stable version of PI. It calculates the log of the probability "
                "of improving over the current best value."
            ),
            "Upper Confidence Bound": (
                "Upper Confidence Bound (UCB) balances exploration and exploitation by selecting "
                "points where the upper confidence bound is highest (for maximization)."
            ),
            "q-Expected Improvement": (
                "q-Expected Improvement (qEI) is a batch version of EI that selects multiple points "
                "simultaneously while accounting for the interactions between selections."
            ),
            "q-Upper Confidence Bound": (
                "q-Upper Confidence Bound (qUCB) is a batch version of UCB that selects multiple points "
                "simultaneously while balancing exploration and exploitation."
            ),
            "Integrated Posterior Variance": (
                "q-Negative Integrated Posterior Variance (qNIPV) selects points to reduce overall model "
                "uncertainty. This is a purely exploratory function for active learning, not optimization."
            )
        }
        
        self.acq_descrip_label = ctk.CTkLabel(
            descrip_frame,
            text=self.strategy_info["Expected Improvement"],
            wraplength=250,
            justify="left",
            text_color="light gray"
        )
        self.acq_descrip_label.pack(pady=5, fill="x")
        
        # Show the regular acquisition frame by default
        self.botorch_acq_frame.pack(fill="x", expand=True, pady=5)

    def create_mobo_widgets(self):
        """Create acquisition function widgets for multi-objective optimization."""
        # Title
        ctk.CTkLabel(
            self.acq_mobo_frame,
            text="Multi-Objective Acquisition",
            font=("Arial", 12, "bold")
        ).pack(pady=(10, 5))
        
        # Strategy selector
        ctk.CTkLabel(self.acq_mobo_frame, text="MOBO Strategy:").pack(pady=2)
        self.mobo_strategy_var = ctk.StringVar(value="qNEHVI")
        self.mobo_strategy_menu = ctk.CTkOptionMenu(
            self.acq_mobo_frame,
            values=["qEHVI", "qNEHVI"],
            variable=self.mobo_strategy_var,
            command=self._update_mobo_description
        )
        self.mobo_strategy_menu.pack(pady=5)
        
        # Strategy descriptions
        self.mobo_strategy_info = {
            "qEHVI": (
                "q-Expected Hypervolume Improvement selects points that maximize "
                "the expected improvement in the Pareto hypervolume."
            ),
            "qNEHVI": (
                "q-Noisy Expected Hypervolume Improvement is a noise-robust variant "
                "of qEHVI. Recommended as the default MOBO strategy."
            ),
        }
        
        self.mobo_descrip_label = ctk.CTkLabel(
            self.acq_mobo_frame,
            text=self.mobo_strategy_info["qNEHVI"],
            wraplength=250,
            justify="left",
            text_color="light gray"
        )
        self.mobo_descrip_label.pack(pady=5, padx=10)
        
        # Batch size
        batch_frame = ctk.CTkFrame(self.acq_mobo_frame)
        batch_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(batch_frame, text="Batch Size (q):").pack(side="left", padx=5)
        self.mobo_batch_size_var = ctk.IntVar(value=1)
        self.mobo_batch_size_menu = ctk.CTkOptionMenu(
            batch_frame,
            values=[str(i) for i in range(1, 11)],
            command=lambda v: self.mobo_batch_size_var.set(int(v))
        )
        self.mobo_batch_size_menu.pack(side="right", padx=5)
        
        # Per-objective directions
        ctk.CTkLabel(
            self.acq_mobo_frame,
            text="Objective Directions:",
            font=("Arial", 11, "bold")
        ).pack(pady=(10, 5), anchor="w", padx=10)
        
        self.mobo_directions_frame = ctk.CTkFrame(self.acq_mobo_frame)
        self.mobo_directions_frame.pack(fill="x", padx=10, pady=5)
        self.mobo_direction_vars = {}
        
        # Reference point (optional)
        ctk.CTkLabel(
            self.acq_mobo_frame,
            text="Reference Point (optional):",
            font=("Arial", 11, "bold")
        ).pack(pady=(10, 2), anchor="w", padx=10)
        
        ctk.CTkLabel(
            self.acq_mobo_frame,
            text="Leave blank for auto-computed defaults.",
            font=("Arial", 10),
            text_color="gray"
        ).pack(anchor="w", padx=10)
        
        self.mobo_refpoint_frame = ctk.CTkFrame(self.acq_mobo_frame)
        self.mobo_refpoint_frame.pack(fill="x", padx=10, pady=5)
        self.mobo_refpoint_entries = {}

    def _update_mobo_description(self, selection):
        """Update the MOBO strategy description text."""
        self.mobo_descrip_label.configure(
            text=self.mobo_strategy_info.get(selection, "")
        )

    def _populate_mobo_objectives(self, objective_names):
        """Populate per-objective direction selectors and ref point entries."""
        # Clear existing widgets
        for widget in self.mobo_directions_frame.winfo_children():
            widget.destroy()
        for widget in self.mobo_refpoint_frame.winfo_children():
            widget.destroy()
        
        self.mobo_direction_vars = {}
        self.mobo_refpoint_entries = {}
        
        for obj_name in objective_names:
            # Direction row
            row = ctk.CTkFrame(self.mobo_directions_frame)
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=obj_name, width=120, anchor="w").pack(side="left", padx=5)
            dir_var = ctk.StringVar(value="Maximize")
            dir_seg = ctk.CTkSegmentedButton(
                row,
                values=["Maximize", "Minimize"],
                variable=dir_var,
                width=160
            )
            dir_seg.pack(side="right", padx=5)
            self.mobo_direction_vars[obj_name] = dir_var
            
            # Reference point row
            ref_row = ctk.CTkFrame(self.mobo_refpoint_frame)
            ref_row.pack(fill="x", pady=2)
            ctk.CTkLabel(ref_row, text=obj_name, width=120, anchor="w").pack(side="left", padx=5)
            ref_entry = ctk.CTkEntry(ref_row, placeholder_text="auto", width=100)
            ref_entry.pack(side="right", padx=5)
            self.mobo_refpoint_entries[obj_name] = ref_entry

    def update_botorch_acq_type(self, selection):
        """Update the visible botorch acquisition function type frame."""
        # Hide all content frames first
        self.botorch_acq_frame.pack_forget()
        self.botorch_batch_frame.pack_forget()
        self.botorch_exploratory_frame.pack_forget()
        
        # Show the selected frame
        if selection == "Regular":
            self.botorch_acq_frame.pack(fill="x", expand=True, pady=5)
            self.update_botorch_description(self.acq_botorch_var.get())
        elif selection == "Batch":
            self.botorch_batch_frame.pack(fill="x", expand=True, pady=5)
            self.update_botorch_description(self.acq_botorch_batch_var.get())
        elif selection == "Exploratory":
            self.botorch_exploratory_frame.pack(fill="x", expand=True, pady=5)
            self.update_botorch_description("Integrated Posterior Variance")

    def update_botorch_description(self, selection):
        """Update the description text for the selected BoTorch acquisition function."""
        self.acq_descrip_label.configure(text=self.strategy_info.get(selection, ""))
        
        # Show/hide beta slider for UCB functions
        if selection == "q-Upper Confidence Bound":
            self.botorch_beta_frame.pack(fill="x", pady=5)
        else:
            # Only hide if we're in batch mode
            if self.botorch_acq_type_var.get() == "Batch":
                self.botorch_beta_frame.pack_forget()

    def update_botorch_beta_value(self, value):
        """Update the beta parameter value label for BoTorch UCB functions."""
        self.botorch_beta_value_label.configure(text=f"{value:.2f}")

    def create_ax_widgets(self):
        """Create acquisition function widgets for Ax logic."""
        info_text = (
            "Ax is designed to work autonomously with minimal configuration.\n\n"
            "It automatically selects appropriate acquisition functions and optimization strategies "
            "based on your problem characteristics and available data."
        )
        ctk.CTkLabel(
            self.acq_ax_frame, 
            text=info_text,
            wraplength=250,
            justify="left"
        ).pack(pady=10, padx=10)
        
        # Add maximize/minimize option
        ctk.CTkLabel(self.acq_ax_frame, text="Optimization Goal:").pack(pady=2)
        self.ax_goal_options = ["Maximize", "Minimize"]
        self.ax_goal_var = ctk.StringVar(value="Maximize")
        self.ax_goal_segmented = ctk.CTkSegmentedButton(
            self.acq_ax_frame,
            values=self.ax_goal_options,
            variable=self.ax_goal_var,
        )
        self.ax_goal_segmented.pack(pady=5)

    def load_backend_options(self, event=None):
        """Show acquisition options based on selected logic."""
        self.acq_sklearn_frame.pack_forget()
        self.acq_botorch_frame.pack_forget()
        
        backend = self.backend_var.get()
        if backend == "scikit-learn":
            self.acq_sklearn_frame.pack(fill="x", expand=True, padx=10, pady=5)
        elif backend == "botorch":
            self.acq_botorch_frame.pack(fill="x", expand=True, padx=10, pady=5)
        # Removed the Ax backend option
        
        # Sync the backend selection with the model panel
        if hasattr(self.main_app, 'model_frame') and self.main_app.model_frame is not None:
            self.main_app.model_frame.backend_var.set(backend)
            self.main_app.model_frame.load_backend_options()

    def toggle_gp_hedge(self):
        """Toggle between manual strategy selection and automatic GP hedge."""
        is_gp_hedge = self.use_gp_hedge_var.get()
        
        # Enable/disable the strategy dropdown
        self.acq_sklearn_menu.configure(state="disabled" if is_gp_hedge else "normal")
        
        # Show/hide parameter sliders appropriately
        if is_gp_hedge:
            # GP Hedge can use both parameters
            self.xi_frame.pack(fill="x", pady=5)
            self.xi_info_label.pack(after=self.xi_frame, pady=(0, 5))
            self.kappa_frame.pack(fill="x", pady=5)
            self.kappa_info_label.pack(after=self.kappa_frame, pady=(0, 5))
        else:
            # Regular strategy selection - update based on current selection
            self.update_acq_param_visibility()

    def update_xi_value(self, value):
        """Update the xi parameter value label."""
        self.xi_value_label.configure(text=f"{value:.3f}")

    def update_kappa_value(self, value):
        """Update the kappa parameter value label."""
        self.kappa_value_label.configure(text=f"{value:.2f}")

    def update_acq_param_visibility(self, selection=None):
        """Show/hide acquisition function parameters based on selection."""
        if selection is None:
            selection = self.acq_sklearn_var.get()
            
        if selection in ["Expected Improvement (EI)", "Probability of Improvement (PI)"]:
            self.xi_frame.pack(fill="x", pady=5)
            self.xi_info_label.pack(after=self.xi_frame, pady=(0, 5))
            self.kappa_frame.pack_forget()
            self.kappa_info_label.pack_forget()
        elif selection == "Upper Confidence Bound (UCB)":
            self.xi_frame.pack_forget()
            self.xi_info_label.pack_forget()
            self.kappa_frame.pack(fill="x", pady=5)
            self.kappa_info_label.pack(after=self.kappa_frame, pady=(0, 5))
        elif selection == "GP Hedge (Auto-balance)":
            # GP Hedge shows both parameters
            self.xi_frame.pack(fill="x", pady=5)
            self.xi_info_label.pack(after=self.xi_frame, pady=(0, 5))
            self.kappa_frame.pack(fill="x", pady=5)
            self.kappa_info_label.pack(after=self.kappa_frame, pady=(0, 5))
    
    def update_for_backend(self, backend):
        """Update acquisition options based on GPR backend selection."""
        self.acq_sklearn_frame.pack_forget()
        self.acq_botorch_frame.pack_forget()
        self.acq_mobo_frame.pack_forget()
        
        if self._is_mobo:
            # MOBO always shows the MOBO frame
            self.acq_mobo_frame.pack(fill="x", expand=True, padx=10, pady=5)
        elif backend == "scikit-learn":
            self.acq_sklearn_frame.pack(fill="x", expand=True, padx=10, pady=5)
        elif backend == "botorch":
            self.acq_botorch_frame.pack(fill="x", expand=True, padx=10, pady=5)

    def run_selected_strategy(self):
        """Execute the selected acquisition strategy."""
        # Access GPR model from main app
        if not hasattr(self.main_app, 'gpr_model') or self.main_app.gpr_model is None:
            print("Error: Model not trained. Please train the model first.")
            return
        
        # Get backend from the main app's model panel
        if hasattr(self.main_app, 'model_frame'):
            backend = self.main_app.model_frame.backend_var.get()
        else:
            backend = "scikit-learn"  # Default
        
        try:
            # ============================================================
            # Session-based acquisition path
            # ============================================================
            print("Using OptimizationSession API for acquisition...")
            
            # ============================================================
            # Multi-objective path
            # ============================================================
            if self._is_mobo:
                acq_strategy = self.mobo_strategy_var.get()
                n_suggestions = self.mobo_batch_size_var.get()
                
                # Build per-objective directions list
                goal_list = []
                for obj_name in self._objective_names:
                    dir_var = self.mobo_direction_vars.get(obj_name)
                    goal_list.append('maximize' if dir_var and dir_var.get() == "Maximize" else 'minimize')
                
                # Build reference point (None entries → let core compute)
                ref_point = None
                ref_values = []
                all_provided = True
                for obj_name in self._objective_names:
                    entry = self.mobo_refpoint_entries.get(obj_name)
                    val_str = entry.get().strip() if entry else ""
                    if val_str and val_str != "auto":
                        try:
                            ref_values.append(float(val_str))
                        except ValueError:
                            print(f"Warning: Invalid reference point for {obj_name}: '{val_str}'. Using auto.")
                            all_provided = False
                            break
                    else:
                        all_provided = False
                        break
                if all_provided and len(ref_values) == len(self._objective_names):
                    ref_point = ref_values
                
                # Call session API
                next_point_df = self.main_app.session.suggest_next(
                    strategy=acq_strategy,
                    goal=goal_list,
                    n_suggestions=n_suggestions,
                    ref_point=ref_point,
                    mc_samples=128
                )
                
                # Get per-objective predictions
                mobo_predictions = {}
                if hasattr(self.main_app.gpr_model, '_predict_multi_objective'):
                    pred_dict = self.main_app.gpr_model._predict_multi_objective(
                        next_point_df, return_std=True
                    )
                    mobo_predictions = pred_dict
                
                # Store and update plot
                self.main_app.next_point = next_point_df
                self.main_app.update_pool_plot()
                
                # Build result_data for notification
                display_strategy = f"{acq_strategy}"
                if n_suggestions > 1:
                    display_strategy += f" (q={n_suggestions})"
                
                directions_str = ", ".join(
                    f"{name}: {d}" for name, d in zip(self._objective_names, goal_list)
                )
                
                result_data = {
                    'point_df': next_point_df,
                    'value': None,
                    'std': None,
                    'maximize': True,
                    'strategy_type': display_strategy,
                    'strategy_params': {
                        'directions': directions_str,
                        'ref_point': str(ref_point) if ref_point else "auto",
                    },
                    'strategy_description': (
                        f"Multi-objective acquisition using {acq_strategy}. "
                        f"Optimizing {len(self._objective_names)} objectives to improve the Pareto frontier."
                    ),
                    'is_batch': n_suggestions > 1,
                    'batch_size': n_suggestions,
                    'is_mobo': True,
                    'mobo_predictions': mobo_predictions,
                    'objective_names': self._objective_names,
                }
                
                hyperparams = self.main_app.gpr_model.get_hyperparameters()
                model_data = {
                    'backend': 'botorch',
                    'kernel': hyperparams.get('kernel_type', 'Unknown'),
                    'hyperparameters': hyperparams,
                    'metrics': {
                        'RMSE': self.main_app.rmse_values[-1] if hasattr(self.main_app, 'rmse_values') and self.main_app.rmse_values else None,
                        'R²': self.main_app.r2_values[-1] if hasattr(self.main_app, 'r2_values') and self.main_app.r2_values else None
                    }
                }
                
                ResultNotificationWindow(self, result_data, model_data)
                print(f"{acq_strategy} executed successfully")
                return
            
            if backend == "scikit-learn":
                strategy = self.acq_sklearn_var.get()
                maximize = self.goal_var.get() == "Maximize"
                
                # Map UI strategy names to session API names
                strategy_map = {
                    "Expected Improvement (EI)": "EI",
                    "Upper Confidence Bound (UCB)": "UCB",
                    "Probability of Improvement (PI)": "PI",
                    "GP Hedge (Auto-balance)": "gp_hedge"
                }
                acq_strategy = strategy_map.get(strategy, "EI")
                
                # Prepare acquisition parameters
                acq_kwargs = {}
                if acq_strategy in ["EI", "PI"]:
                    acq_kwargs['xi'] = float(self.xi_slider.get())
                elif acq_strategy == "UCB":
                    acq_kwargs['kappa'] = float(self.kappa_slider.get())
                elif acq_strategy == "gp_hedge":
                    acq_kwargs['xi'] = float(self.xi_slider.get())
                    acq_kwargs['kappa'] = float(self.kappa_slider.get())
                
                # Use session API to get suggestions
                goal = 'maximize' if maximize else 'minimize'
                next_point_df = self.main_app.session.suggest_next(
                    strategy=acq_strategy,
                    goal=goal,
                    n_suggestions=1,
                    **acq_kwargs
                )
                
                # Get prediction
                pred_value, pred_std = None, None
                if hasattr(self.main_app.gpr_model, 'predict_with_std'):
                    pred_values, pred_stds = self.main_app.gpr_model.predict_with_std(next_point_df)
                    pred_value = pred_values[0]
                    pred_std = pred_stds[0]
                else:
                    pred_value = self.main_app.gpr_model.predict(next_point_df)[0]
                
                # Store and update plot
                self.main_app.next_point = next_point_df
                self.main_app.update_pool_plot()
                
                # Show notification
                result_data = {
                    'point_df': next_point_df,
                    'value': pred_value,
                    'std': pred_std,
                    'maximize': maximize,
                    'strategy_type': f"{strategy}",
                    'strategy_params': acq_kwargs,
                    'strategy_description': f"Using OptimizationSession API with {acq_strategy}"
                }
                
                model_data = {
                    'backend': 'sklearn',
                    'kernel': str(self.main_app.gpr_model.kernel) if hasattr(self.main_app.gpr_model, 'kernel') else 'Unknown',
                    'hyperparameters': self.main_app.gpr_model.get_hyperparameters(),
                    'metrics': {
                        'RMSE': self.main_app.rmse_values[-1] if hasattr(self.main_app, 'rmse_values') and self.main_app.rmse_values else None,
                        'R²': self.main_app.r2_values[-1] if hasattr(self.main_app, 'r2_values') and self.main_app.r2_values else None
                    }
                }
                
                ResultNotificationWindow(self, result_data, model_data)
                print(f"{strategy} executed successfully")
            
            elif backend == "botorch":
                maximize = self.botorch_goal_var.get() == "Maximize"
                acq_type = self.botorch_acq_type_var.get()
                
                # Map BoTorch acquisition functions to session API
                if acq_type == "Regular":
                    strategy = self.acq_botorch_var.get()
                    strategy_map = {
                        "Expected Improvement": "EI",
                        "Log Expected Improvement": "logEI",
                        "Probability of Improvement": "PI",
                        "Log Probability of Improvement": "logPI",
                        "Upper Confidence Bound": "UCB"
                    }
                    acq_strategy = strategy_map.get(strategy, "logEI")
                    n_suggestions = 1
                    acq_kwargs = {}
                    if acq_strategy == "UCB":
                        acq_kwargs['beta'] = 0.5
                
                elif acq_type == "Batch":
                    strategy = self.acq_botorch_batch_var.get()
                    strategy_map = {
                        "q-Expected Improvement": "qEI",
                        "q-Upper Confidence Bound": "qUCB"
                    }
                    acq_strategy = strategy_map.get(strategy, "qEI")
                    n_suggestions = self.botorch_batch_size_var.get()
                    acq_kwargs = {"mc_samples": 128}
                    if acq_strategy == "qUCB":
                        acq_kwargs['beta'] = float(self.botorch_beta_slider.get())
                
                elif acq_type == "Exploratory":
                    strategy = "Integrated Posterior Variance"
                    acq_strategy = "qIPV"
                    n_suggestions = 1
                    acq_kwargs = {
                        "n_mc_points": self.botorch_mc_points_var.get(),
                        "mc_samples": 128
                    }
                
                # Use session API
                goal = 'maximize' if maximize else 'minimize'
                next_point_df = self.main_app.session.suggest_next(
                    strategy=acq_strategy,
                    goal=goal,
                    n_suggestions=n_suggestions,
                    **acq_kwargs
                )
                
                # Get predictions
                if hasattr(self.main_app.gpr_model, 'predict_with_std'):
                    pred_values, pred_stds = self.main_app.gpr_model.predict_with_std(next_point_df)
                    if n_suggestions == 1:
                        pred_value = pred_values[0]
                        pred_std = pred_stds[0]
                    else:
                        pred_value = pred_values
                        pred_std = pred_stds
                else:
                    pred_value = self.main_app.gpr_model.predict(next_point_df)
                    pred_std = None
                
                # Store and update plot
                self.main_app.next_point = next_point_df
                self.main_app.update_pool_plot()
                
                # Show notification
                display_strategy = f"{strategy}"
                if n_suggestions > 1:
                    display_strategy += f" (q={n_suggestions})"
                
                result_data = {
                    'point_df': next_point_df,
                    'value': pred_value,
                    'std': pred_std,
                    'maximize': maximize,
                    'strategy_type': display_strategy,
                    'strategy_params': acq_kwargs,
                    'strategy_description': f"Using OptimizationSession API with {acq_strategy}",
                    'is_batch': n_suggestions > 1,
                    'batch_size': n_suggestions
                }
                
                hyperparams = self.main_app.gpr_model.get_hyperparameters()
                kernel_display = hyperparams.get('kernel_type', 'Unknown')
                
                model_data = {
                    'backend': 'botorch',
                    'kernel': kernel_display,
                    'hyperparameters': hyperparams,
                    'metrics': {
                        'RMSE': self.main_app.rmse_values[-1] if hasattr(self.main_app, 'rmse_values') and self.main_app.rmse_values else None,
                        'R²': self.main_app.r2_values[-1] if hasattr(self.main_app, 'r2_values') and self.main_app.r2_values else None
                    }
                }
                
                ResultNotificationWindow(self, result_data, model_data)
                print(f"{strategy} executed successfully")
        
        except Exception as e:
            print(f"Error executing strategy: {e}")
            import traceback
            traceback.print_exc()
        '''
        elif backend == "ax":
            maximize = self.ax_goal_var.get() == "Maximize"
            
            # Implementation for Ax will go here in the future
            print("Ax acquisition strategies not fully implemented yet.")
            
            # For now, show a notification with basic info
            result_data = {
                'point_df': None,
                'value': None,
                'std': None,
                'maximize': maximize,
                'strategy_type': "Ax Bayesian Optimization",
                'strategy_params': {},
                'strategy_description': "Ax acquisition strategies will be fully implemented in a future update."
            }
            
            model_data = {
                'backend': backend,
                'kernel': "Ax model",
                'hyperparameters': self.main_app.gpr_model.get_hyperparameters() if hasattr(self.main_app.gpr_model, 'get_hyperparameters') else {},
                'metrics': {}
            }
            
            ResultNotificationWindow(self, result_data, model_data)
            '''
            
    def create_model_maximum_section(self):
        """Create a section for finding the model's predicted optimum."""
        # Create a frame for the model optimum section
        self.opt_frame = ctk.CTkFrame(self)
        self.opt_frame.pack(fill="x", expand=True, padx=10, pady=5)
        
        # Add title
        ctk.CTkLabel(self.opt_frame, text="Model Prediction Optimum", font=('Arial', 14)).pack(pady=5)
        
        # Add explanatory text
        self.opt_description_label = ctk.CTkLabel(
            self.opt_frame, 
            text="Find the point where the model predicts the optimal value.",
            wraplength=250,
            justify="left"
        )
        self.opt_description_label.pack(pady=5)
        
        # MOBO objective selector (hidden by default)
        self.opt_objective_frame = ctk.CTkFrame(self.opt_frame)
        ctk.CTkLabel(self.opt_objective_frame, text="Objective to Optimize:").pack(pady=2)
        self.opt_objective_var = ctk.StringVar(value="")
        self.opt_objective_menu = ctk.CTkOptionMenu(
            self.opt_objective_frame,
            values=[""],
            variable=self.opt_objective_var,
            width=200
        )
        self.opt_objective_menu.pack(pady=5)
        
        # Add maximize/minimize option
        ctk.CTkLabel(self.opt_frame, text="Optimization Goal:").pack(pady=2)
        self.opt_goal_options = ["Maximize", "Minimize"]
        self.opt_goal_var = ctk.StringVar(value="Maximize")
        self.opt_goal_segmented = ctk.CTkSegmentedButton(
            self.opt_frame,
            values=self.opt_goal_options,
            variable=self.opt_goal_var,
        )
        self.opt_goal_segmented.pack(pady=5)
        
        # Add button to find the optimum
        self.find_opt_button = ctk.CTkButton(
            self.opt_frame,
            text="Find Model Optimum",
            command=self.find_model_optimum,
            state="disabled"  # Disabled until model is trained
        )
        self.find_opt_button.pack(pady=10, fill="x")
        
        # Warning label about model reliability
        ctk.CTkLabel(
            self.opt_frame,
            text="Note: This relies entirely on the model's prediction, "
                 "not on acquisition functions that balance exploration and exploitation.",
            wraplength=250,
            text_color="gray",
            justify="left",
            font=("Arial", 10)
        ).pack(pady=5)

    def find_model_optimum(self):
        """Find the point where the model predicts the optimal value."""
        if not hasattr(self.main_app, 'gpr_model') or self.main_app.gpr_model is None:
            print("Error: Model not trained. Please train the model first.")
            return
            
        try:
            maximize = self.opt_goal_var.get() == "Maximize"
            backend = self.main_app.model_frame.backend_var.get()
            
            # For MOBO, find optimum for the selected single objective
            if self._is_mobo:
                selected_obj = self.opt_objective_var.get()
                
                # Build grid from model's feature names so dimensions match
                from itertools import product
                import pandas as pd
                model = self.main_app.gpr_model
                feature_names = getattr(model, 'original_feature_names', None) or getattr(model, 'feature_names', [])
                var_dict = {v['name']: v for v in self.main_app.search_space_manager.variables}
                
                n_vars = len(feature_names)
                n_per_dim = max(2, int(10000 ** (1/n_vars)))
                
                grid_1d = []
                for name in feature_names:
                    if name in var_dict:
                        var = var_dict[name]
                        if var['type'] == 'real':
                            grid_1d.append(np.linspace(var['min'], var['max'], n_per_dim))
                        elif var['type'] == 'integer':
                            vals = np.arange(var['min'], var['max'] + 1)
                            if len(vals) > n_per_dim:
                                vals = np.unique(np.linspace(var['min'], var['max'], n_per_dim).astype(int))
                            grid_1d.append(vals)
                        elif var['type'] == 'categorical':
                            grid_1d.append(var['values'])
                    else:
                        # Extra feature not in search space — derive range from training data
                        exp_data = getattr(self.main_app.experiment_manager, 'df', None)
                        if exp_data is not None and name in exp_data.columns:
                            col = exp_data[name]
                            grid_1d.append(np.linspace(float(col.min()), float(col.max()), n_per_dim))
                        else:
                            grid_1d.append(np.linspace(0.0, 1.0, n_per_dim))
                
                mesh = list(product(*grid_1d))
                grid = pd.DataFrame(mesh, columns=feature_names)

                # Restrict to constraint-feasible points so the reported optimum
                # honors registered linear input constraints.
                _ssm = self.main_app.search_space_manager
                if getattr(_ssm, 'constraints', None) and hasattr(_ssm, 'filter_feasible'):
                    _mask = _ssm.filter_feasible(grid)
                    grid = grid[_mask].reset_index(drop=True)
                    if len(grid) == 0:
                        raise ValueError(
                            "No feasible grid points satisfy the registered input "
                            "constraints. Relax the constraints or increase resolution."
                        )

                print(f"Searching for model's predicted {'maximum' if maximize else 'minimum'} of '{selected_obj}'...")
                pred_dict = model._predict_multi_objective(grid, return_std=True)
                means, stds = pred_dict[selected_obj]
                
                best_idx = int(np.argmax(means) if maximize else np.argmin(means))
                opt_point_df = grid.iloc[[best_idx]].reset_index(drop=True)
                opt_value = float(means[best_idx])
                opt_std = float(stds[best_idx]) if stds is not None else None
                
                result = {'x_opt': opt_point_df, 'value': opt_value, 'std': opt_std}
            
            elif backend == "scikit-learn":
                # TEMPORARY: Direct acquisition import until Session API supports find_optimum
                from alchemist_core.acquisition.skopt_acquisition import SkoptAcquisition
                
                # Create acquisition instance
                acquisition = SkoptAcquisition(
                    search_space=self.main_app.search_space,
                    model=self.main_app.gpr_model,
                    maximize=maximize,
                    random_state=42
                )
                
                # Find the optimum
                print(f"Searching for model's predicted {'maximum' if maximize else 'minimum'}...")
                result = acquisition.find_optimum(
                    model=self.main_app.gpr_model,
                    maximize=maximize,
                    random_state=42
                )
                
            elif backend == "botorch":
                # TEMPORARY: Direct acquisition import until Session API supports find_optimum
                from alchemist_core.acquisition.botorch_acquisition import BoTorchAcquisition
                
                # Create acquisition instance
                acquisition = BoTorchAcquisition(
                    search_space=self.main_app.search_space_manager,
                    model=self.main_app.gpr_model,
                    acq_func="ucb",  # This doesn't matter for find_optimum
                    maximize=maximize,
                    random_state=42
                )
                
                # Find the optimum using BoTorch's find_optimum method
                print(f"Searching for model's predicted {'maximum' if maximize else 'minimum'}...")
                result = acquisition.find_optimum(
                    model=self.main_app.gpr_model,
                    maximize=maximize,
                    random_state=42
                )
                
            else:
                print(f"Finding model optimum not implemented for {backend} backend")
                return
                
            # Extract results
            opt_point_df = result['x_opt']
            
            # Add this check:
            if 'Noise' in self.main_app.exp_df.columns and 'Noise' not in opt_point_df.columns:
                opt_point_df['Noise'] = 1e-6  # Add default noise
            
            # Get predicted value and std at optimum (ensure we drop Noise for prediction)
            pred_df = opt_point_df.drop(columns=['Noise']) if 'Noise' in opt_point_df.columns else opt_point_df
            opt_value = float(result['value'])
            opt_std = float(result['std']) if result.get('std') is not None else None
            
            # Store the optimum point
            self.main_app.next_point = opt_point_df
            
            # Update the plot
            self.main_app.update_pool_plot()
            
            # Prepare result data for the notification window
            strategy_desc = (
                "This strategy directly finds the point where the model "
                "predicts the best value, without considering exploration. "
                "It's best used when you're confident in your model's accuracy "
                "and want to exploit its predictions."
            )
            strategy_type = "Model Optimum Finder"
            if self._is_mobo:
                selected_obj = self.opt_objective_var.get()
                strategy_type = f"Model Optimum: {selected_obj}"
                strategy_desc = (
                    f"Finds the point where the model predicts the best value for "
                    f"'{selected_obj}', independent of other objectives. "
                    "This does not consider the Pareto frontier."
                )
            
            result_data = {
                'point_df': opt_point_df,
                'value': opt_value,
                'std': opt_std,
                'maximize': maximize,
                'strategy_type': strategy_type,
                'strategy_params': {'random_state': 42},
                'strategy_description': strategy_desc,
            }
            
            # Get metrics if available
            metrics = {
                'RMSE': self.main_app.rmse_values[-1] if hasattr(self.main_app, 'rmse_values') and self.main_app.rmse_values else None,
                'R²': self.main_app.r2_values[-1] if hasattr(self.main_app, 'r2_values') and self.main_app.r2_values else None
            }
            
            # Use the backend-agnostic model data collector
            from ui.utils import get_model_data
            model_data = {
                'backend': backend,
                'kernel': str(self.main_app.gpr_model.kernel if hasattr(self.main_app.gpr_model, 'kernel') else "Unknown"),
                'hyperparameters': self.main_app.gpr_model.get_hyperparameters(),
                'metrics': metrics
            }
            
            # Log the acquisition if a logger exists
            if hasattr(self.main_app, 'experiment_logger'):
                self.main_app.experiment_logger.log_acquisition(result_data)
            
            # Show the notification window
            ResultNotificationWindow(self, result_data, model_data)
            
        except Exception as e:
            print(f"Error finding model optimum: {e}")
            import traceback
            traceback.print_exc()

    def _find_botorch_optimum(self, acquisition, maximize):
        """Find the optimum point using BoTorch."""
        # Get model from the acquisition
        model = acquisition.model
        
        # Use the select_next method with the appropriate acquisition function
        # This is a simplification - in a real implementation you might want to try multiple starts
        next_point = acquisition.select_next()
        
        # Convert to DataFrame for predictions
        if isinstance(next_point, dict):
            next_point_df = pd.DataFrame([next_point])
        else:
            next_point_df = pd.DataFrame([next_point], columns=model.feature_names)
        
        # Get predicted value and std
        pred_mean, pred_std = model.predict_with_std(next_point_df)
        
        # Return in the same format as SkoptAcquisition.find_optimum
        return {
            'x_opt': next_point_df,
            'value': float(pred_mean[0]),
            'std': float(pred_std[0])
        }
    
    def enable(self, is_mobo=False, objective_names=None):
        """
        Enable the acquisition panel buttons after model training is complete.
        Called by the GPR panel after successful model training.
        
        Args:
            is_mobo: Whether multi-objective optimization is active
            objective_names: List of objective column names (for MOBO)
        """
        self._is_mobo = is_mobo
        self._objective_names = objective_names or []
        
        # Enable the run button
        self.run_button.configure(state="normal")
        
        # Enable the find optimum button
        self.find_opt_button.configure(state="normal")
        
        if is_mobo:
            # Switch to MOBO acquisition frame
            self.acq_sklearn_frame.pack_forget()
            self.acq_botorch_frame.pack_forget()
            self.acq_mobo_frame.pack(fill="x", expand=True, padx=10, pady=5)
            self._populate_mobo_objectives(self._objective_names)
            
            # Show objective selector for model optimum
            self.opt_objective_menu.configure(values=self._objective_names)
            self.opt_objective_var.set(self._objective_names[0])
            self.opt_objective_frame.pack(fill="x", pady=5, after=self.opt_description_label)
            self.opt_description_label.configure(
                text="Find the optimal value for a specific objective."
            )
        else:
            # Hide MOBO objective selector
            self.opt_objective_frame.pack_forget()
            self.opt_description_label.configure(
                text="Find the point where the model predicts the optimal value."
            )
        
        print("Acquisition panel activated - you can now run acquisition strategies or find model optimum")