import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.metrics import r2_score
from skopt.space import Real, Integer, Categorical
from skopt.learning.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ConstantKernel as C
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict, cross_validate, train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
import pandas as pd
import threading
from ui.custom_widgets import CTkSpinbox
from scipy import stats

plt.rcParams['savefig.dpi'] = 600


class Visualizations:
    def __init__(self, parent, search_space, gpr_model, exp_df, encoded_X, encoded_y, session=None):
        self.parent = parent  # Store the parent widget
        self.session = session
        self.search_space = search_space
        
        # Pre-convert search space to skopt if it's a SearchSpace object
        if hasattr(search_space, 'to_skopt'):
            self.skopt_space = search_space.to_skopt()
        else:
            self.skopt_space = search_space
            
        self.gpr_model = gpr_model
        self.exp_df = exp_df
        self.encoded_X = encoded_X
        self.encoded_y = encoded_y

        # Initialize metrics as None; they will be set later
        self.rmse_values = None
        self.mae_values = None
        self.mape_values = None
        self.r2_values = None

        # Delay Matplotlib figure initialization
        self.fig = None
        self.ax = None
        self.canvas = None
        self.colorbar = None

        self.customization_options = {
            'title': "",
            'xlabel': "",
            'ylabel': "",
            'colormap': "viridis",
            'font': "Arial",
            'title_fontsize': 12,
            'label_fontsize': 10,
            'title_fontweight': "normal",
            'label_fontweight': "normal",
            'tick_style': "x and y",  # Set default to "x and y"
            'tick_direction': "out",  # Set default to "out"
            'show_experiments': False,  # Disabled by default
            'show_next_point': False,    # Disabled by default
            'x_number_format': "auto",  # X-axis number formatting
            'y_number_format': "auto",  # Y-axis number formatting
            'colorbar_format': "auto"   # Colorbar number formatting
        }
        self.last_plot_method = None

    def _try_session_plot(self, method_name, **kwargs):
        """Try plotting via Session API. Returns True on success, False for fallback."""
        if self.session is None:
            return False
        try:
            getattr(self.session, method_name)(ax=self.ax, **kwargs)
            return True
        except Exception as e:
            print(f"Session API error ({method_name}): {e}, falling back to legacy code")
            self.ax.clear()  # Clean up partial draws
            return False

    def _get_number_formatter(self, format_type):
        """Create a matplotlib formatter based on the format type."""
        from matplotlib.ticker import FuncFormatter, ScalarFormatter, PercentFormatter
        import matplotlib.ticker as ticker
        
        if format_type == "auto":
            return None  # Use matplotlib's default
        elif format_type == "decimal":
            return FuncFormatter(lambda x, p: f"{x:.2f}")
        elif format_type == "scientific":
            return ticker.ScientificFormatter(precision=2)
        elif format_type == "percent":
            # Handle both 0-1 range and 0-100 range automatically
            def percent_formatter(x, p):
                if abs(x) <= 1.0:  # Assume 0-1 range
                    return f"{x*100:.1f}%"
                else:  # Assume already in percentage
                    return f"{x:.1f}%"
            return FuncFormatter(percent_formatter)
        elif format_type == "integer":
            return FuncFormatter(lambda x, p: f"{int(round(x))}")
        elif format_type == "custom":
            return FuncFormatter(lambda x, p: f"{x:.3g}")  # General format with 3 significant digits
        else:
            return None

    def initialize_figure(self):
        """Initialize the Matplotlib figure and axes."""
        if self.fig is None or self.ax is None:
            # Use current rcParam DPI when creating the figure so text sizes
            # are consistent with the active window's DPI settings.
            dpi = plt.rcParams.get('figure.dpi', 100)
            self.fig, self.ax = plt.subplots(figsize=(6, 4), dpi=dpi)

    def open_window(self):
        """Open the visualization window."""
        # Probe DPI from the parent window (avoids creating a new Tk root).
        try:
            dpi = int(self.parent.winfo_fpixels('1i'))
        except Exception:
            dpi = plt.rcParams.get('figure.dpi', 100)

        # Apply DPI to matplotlib runtime settings so text and markers scale correctly
        plt.rcParams['figure.dpi'] = dpi

        # Cap font sizes to avoid oversized rendering on HiDPI displays
        try:
            fs = plt.rcParams.get('font.size', 10)
            if fs > 14:
                plt.rcParams['font.size'] = 10
            if plt.rcParams.get('xtick.labelsize', 10) > 12:
                plt.rcParams['xtick.labelsize'] = 8
            if plt.rcParams.get('ytick.labelsize', 10) > 12:
                plt.rcParams['ytick.labelsize'] = 8
        except Exception:
            pass

        self.initialize_figure()  # Ensure the figure is initialized with corrected DPI
        # Enforce additional rcParams to keep the figure compact and fonts reasonable
        try:
            plt.rcParams['axes.titlesize'] = min(12, plt.rcParams.get('axes.titlesize', 12))
            plt.rcParams['axes.labelsize'] = min(10, plt.rcParams.get('axes.labelsize', 10))
            plt.rcParams['legend.fontsize'] = min(9, plt.rcParams.get('legend.fontsize', 9))
            plt.rcParams['xtick.labelsize'] = min(9, plt.rcParams.get('xtick.labelsize', 9))
            plt.rcParams['ytick.labelsize'] = min(9, plt.rcParams.get('ytick.labelsize', 9))
            plt.rcParams['lines.markersize'] = min(6, plt.rcParams.get('lines.markersize', 6))
        except Exception:
            pass

        # Ensure the figure uses a compact size (in inches) and the detected DPI.
        try:
            # Prefer a slightly smaller figure to avoid huge white margins
            self.fig.set_size_inches(5, 4, forward=True)
            self.fig.set_dpi(dpi)
        except Exception:
            pass

        if self.search_space is None:
            print("Error: Search space is not defined.")
            return

        # Convert search space to skopt dimensions if it's a SearchSpace object
        skopt_space = self.search_space
        if hasattr(self.search_space, 'to_skopt'):
            skopt_space = self.search_space.to_skopt()
    
        # Now use skopt_space instead of self.search_space
        real_dims = [dim.name for dim in self.skopt_space if isinstance(dim, Real)]
        if not real_dims or len(real_dims) < 2:
            print("Error: Need at least two Real dimensions for contour plotting.")
            return

        # Proceed with the rest of the method
        visualization_window = ctk.CTkToplevel(self.parent)
        visualization_window.title("Model Visualizations")
        visualization_window.geometry("900x700")
        visualization_window.lift()
        visualization_window.focus_force()
        visualization_window.grab_set()

        # Fix macOS Retina scaling by probing the new window's DPI and applying it
        try:
            dpi = int(visualization_window.winfo_fpixels('1i'))
        except Exception:
            dpi = plt.rcParams.get('figure.dpi', 100)

        # Apply DPI to matplotlib runtime settings so text and markers scale correctly
        plt.rcParams['figure.dpi'] = dpi

        # Top control frame for other visualizations (metrics, parity, hyperparameters)
        # Use two rows to avoid crowding
        control_container = ctk.CTkFrame(visualization_window)
        control_container.pack(side="top", fill="x", padx=10, pady=10)
        
        # First row: Plot type controls
        control_frame_row1 = ctk.CTkFrame(control_container, fg_color="transparent")
        control_frame_row1.pack(side="top", fill="x", pady=(0, 5))

        self.error_metric_var = ctk.StringVar(value="RMSE")
        self.error_metric_menu = ctk.CTkOptionMenu(
            control_frame_row1,
            values=["RMSE", "MAE", "MAPE", "R2"],
            variable=self.error_metric_var,
            width=100
        )
        self.error_metric_menu.pack(side="left", padx=5)
        self.plot_metrics_button = ctk.CTkButton(control_frame_row1, text="Plot Metrics", command=self.plot_metrics, width=100)
        self.plot_metrics_button.pack(side="left", padx=5)
        self.plot_parity_button = ctk.CTkButton(control_frame_row1, text="Plot Parity", command=self.plot_parity, width=100)
        self.plot_parity_button.pack(side="left", padx=5)
        
        # Add sigma multiplier control for parity plot error bars
        ctk.CTkLabel(control_frame_row1, text="Error bars:").pack(side="left", padx=(15, 5))
        self.sigma_multiplier_var = ctk.StringVar(value="1.96")
        self.sigma_multiplier_menu = ctk.CTkOptionMenu(
            control_frame_row1,
            values=["None", "1.0", "1.96", "2.0", "2.58", "3.0"],
            variable=self.sigma_multiplier_var,
            width=80,
            command=self._on_sigma_changed  # Auto-replot when sigma changes
        )
        self.sigma_multiplier_menu.pack(side="left", padx=5)
        
        # Second row: Calibration controls
        control_frame_row2 = ctk.CTkFrame(control_container, fg_color="transparent")
        control_frame_row2.pack(side="top", fill="x")
        
        # Add calibration analysis buttons
        self.plot_qq_button = ctk.CTkButton(control_frame_row2, text="Plot Q-Q", command=self.plot_qq_plot, width=100)
        self.plot_qq_button.pack(side="left", padx=5)
        self.plot_calibration_button = ctk.CTkButton(control_frame_row2, text="Plot Calibration", command=self.plot_calibration_curve, width=120)
        self.plot_calibration_button.pack(side="left", padx=5)
        
        # Add toggle for calibrated/uncalibrated results
        # Default to True if calibrated results exist, False otherwise
        has_calibrated = (hasattr(self.gpr_model, 'cv_cached_results_calibrated') and 
                         self.gpr_model.cv_cached_results_calibrated is not None)
        self.use_calibrated_var = ctk.BooleanVar(value=has_calibrated)
        self.use_calibrated_checkbox = ctk.CTkCheckBox(
            control_frame_row2,
            text="Use Calibrated",
            variable=self.use_calibrated_var,
            command=self._on_calibration_toggle
        )
        self.use_calibrated_checkbox.pack(side="left", padx=(15, 5))
        
        # Disable checkbox if no calibrated results available
        if not has_calibrated:
            self.use_calibrated_checkbox.configure(state="disabled")

        # Main container holds the plot and the contour controls side by side.
        main_container = ctk.CTkFrame(visualization_window)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Plot frame on the left with reduced height.
        self.plot_frame = ctk.CTkFrame(main_container, height=450)
        self.plot_frame.pack(side="left", fill='both', expand=True, padx=5, pady=5)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        # Draw and tighten layout before packing to reduce excess margins
        try:
            self.fig.tight_layout(pad=0.6)
        except Exception:
            pass
        self.canvas.draw()
        # Use smaller padding so the canvas doesn't create excess whitespace
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=2, pady=2)
        self.metrics_toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.metrics_toolbar.config(background='#2E2E2E')
        self.metrics_toolbar._message_label.config(background='#2E2E2E')
        for button in self.metrics_toolbar.winfo_children():
            button.config(background='#D0D0D0')
        self.metrics_toolbar.winfo_children()[-1].config(background='#2E2E2E')
        self.metrics_toolbar.winfo_children()[-2].config(background='#2E2E2E')
        self.metrics_toolbar.update()

        # Scrollable contour controls frame on the right.
        self.contour_controls_frame = ctk.CTkScrollableFrame(main_container, width=250, height=600)
        self.contour_controls_frame.pack(side="right", fill="y", padx=5, pady=5)

        # Title for the contour controls
        ctk.CTkLabel(self.contour_controls_frame, text="Contour Plot Options", font=("Arial", 14, "bold")).pack(pady=(0, 10))

        # Two dropdowns for selecting the contour plot axes.
        ctk.CTkLabel(self.contour_controls_frame, text="X-Axis Variable:").pack(pady=(5, 2))
        self.contour_axis_x_dropdown = ctk.CTkOptionMenu(
            self.contour_controls_frame, values=real_dims, variable=ctk.StringVar(value=real_dims[0])
        )
        self.contour_axis_x_dropdown.pack(pady=(0, 5))

        ctk.CTkLabel(self.contour_controls_frame, text="Y-Axis Variable:").pack(pady=(5, 2))
        self.contour_axis_y_dropdown = ctk.CTkOptionMenu(
            self.contour_controls_frame, values=real_dims, variable=ctk.StringVar(value=real_dims[1])
        )
        self.contour_axis_y_dropdown.pack(pady=(0, 5))

        # REMOVE: The toggle switches for next point and experimental points
        # They will be added to the customization dialog instead

        # Bind changes in axis selection to update fixed controls.
        self.contour_axis_x_dropdown.configure(command=self.update_fixed_controls)
        self.contour_axis_y_dropdown.configure(command=self.update_fixed_controls)

        # Container for the fixed-value controls for the non-axis dimensions.
        self.fixed_controls_container = ctk.CTkFrame(self.contour_controls_frame)
        self.fixed_controls_container.pack(fill="x", pady=5)
        self.fixed_controls = {}

        # Initially populate the fixed controls.
        self.update_fixed_controls()

        # Button to generate the contour plot.
        self.contour_plot_button = ctk.CTkButton(self.contour_controls_frame, text="Plot Contour", command=self.plot_contour)
        self.contour_plot_button.pack(pady=10)

        self.customize_plot_button = ctk.CTkButton(self.contour_controls_frame, text="Customize Plot", command=self.customize_plot)
        self.customize_plot_button.pack(side='bottom', pady=10)

        # Frame at the bottom to display hyperparameters
        hyperparameters_frame = ctk.CTkFrame(visualization_window)
        hyperparameters_frame.pack(side="bottom", fill="x", padx=10, pady=10)

        hyperparameters = self.get_hyperparameters()
        if hyperparameters:
            hyperparameters_str = "\n".join([f"{key}: {value}" for key, value in hyperparameters.items()])
            ctk.CTkLabel(hyperparameters_frame, text="Learned Hyperparameters", font=('Arial', 16)).pack(pady=5)
            ctk.CTkLabel(hyperparameters_frame, text=hyperparameters_str, justify='left').pack(pady=5)

    def update_fixed_controls(self, event=None):
        """Dynamically create controls to fix values for dimensions not selected as axes."""
        # Clear any existing fixed controls.
        for widget in self.fixed_controls_container.winfo_children():
            widget.destroy()
        self.fixed_controls.clear()

        axis_x = self.contour_axis_x_dropdown.get()
        axis_y = self.contour_axis_y_dropdown.get()

        # Convert search space to skopt dimensions if it's a SearchSpace object
        skopt_space = self.search_space
        if hasattr(self.search_space, 'to_skopt'):
            skopt_space = self.search_space.to_skopt()

        # For every dimension in the search space that is not one of the chosen axes, add a fixed value control.
        for dim in skopt_space:
            if dim.name in [axis_x, axis_y]:
                continue

            label = ctk.CTkLabel(self.fixed_controls_container, text=f"Fixed {dim.name}:")
            label.pack(pady=2)

            if isinstance(dim, Real):
                slider = ctk.CTkSlider(
                    self.fixed_controls_container,
                    from_=dim.bounds[0],
                    to=dim.bounds[1]
                )
                slider.set((dim.bounds[0] + dim.bounds[1]) / 2)
                slider.pack(pady=5)
                self.fixed_controls[dim.name] = slider

            elif isinstance(dim, Integer):
                spinbox = CTkSpinbox(
                    self.fixed_controls_container,
                    step_size=1
                )
                spinbox.set(0)
                spinbox.pack(pady=5)
                self.fixed_controls[dim.name] = spinbox

            elif isinstance(dim, Categorical):
                # CTkOptionMenu requires string values; convert in case categories are numeric
                str_cats = [str(c) for c in dim.categories]
                option = ctk.CTkOptionMenu(
                    self.fixed_controls_container,
                    values=str_cats
                )
                option.set(str_cats[0])
                option.pack(pady=5)
                self.fixed_controls[dim.name] = option

    def plot_contour(self):
        """Generate a 2D contour plot for the two selected Real dimensions using fixed values for the others."""
        if self.exp_df is None or len(self.exp_df) < 2:
            print("Error: Not enough observations to plot contour.")
            return

        model = self.gpr_model
        if model is None:
            print("Error: Model is not trained.")
            return

        # Clear user customization flags when switching plot types
        if self.last_plot_method != self.plot_contour:
            self._clear_user_set_flags()
        # Clear previous plot and colorbar.
        self.clear_current_plot()

        # Get the selected axes from the dropdowns.
        axis_x_name = self.contour_axis_x_dropdown.get()
        axis_y_name = self.contour_axis_y_dropdown.get()

        # Gather fixed values from UI controls
        fixed_values = {}
        for dim_name, control in self.fixed_controls.items():
            raw = control.get()
            # CTkOptionMenu values were converted to str for numeric discrete dims;
            # try converting back to float so the model receives the correct type.
            try:
                fixed_values[dim_name] = float(raw)
            except (ValueError, TypeError):
                fixed_values[dim_name] = raw

        # Use the colormap from user customizations if set; otherwise default to 'viridis'.
        cmap = (self.customization_options.get('colormap') if hasattr(self, 'customization_options') and
                self.customization_options.get('colormap') else 'viridis')

        show_experiments = self.customization_options.get('show_experiments', False)
        show_suggestions = self.customization_options.get('show_next_point', False)

        if not self._try_session_plot('plot_contour',
                x_var=axis_x_name,
                y_var=axis_y_name,
                fixed_values=fixed_values,
                show_experiments=show_experiments,
                show_suggestions=show_suggestions,
                cmap=cmap):
            # === Legacy fallback ===
            # Convert search space to skopt dimensions if needed
            skopt_space = self.search_space
            if hasattr(self.search_space, 'to_skopt'):
                skopt_space = self.search_space.to_skopt()

            # Find the corresponding dimension objects.
            axis_x_dim = next(dim for dim in skopt_space if dim.name == axis_x_name)
            axis_y_dim = next(dim for dim in skopt_space if dim.name == axis_y_name)

            # Check if this is a BoTorch model
            if hasattr(model, 'generate_contour_data') and callable(model.generate_contour_data):
                if hasattr(model, 'original_feature_names') and model.original_feature_names:
                    feature_names = model.original_feature_names
                else:
                    feature_names = model.feature_names

                if feature_names:
                    x_idx = feature_names.index(axis_x_name)
                    y_idx = feature_names.index(axis_y_name)

                    idx_fixed = {}
                    for i, dim_name in enumerate(feature_names):
                        if dim_name not in [axis_x_name, axis_y_name] and dim_name in self.fixed_controls:
                            idx_fixed[i] = self.fixed_controls[dim_name].get()

                    X1, X2, y_pred = model.generate_contour_data(
                        x_range=axis_x_dim.bounds,
                        y_range=axis_y_dim.bounds,
                        fixed_values=idx_fixed,
                        x_idx=x_idx,
                        y_idx=y_idx
                    )
                else:
                    X1, X2 = np.meshgrid(
                        np.linspace(axis_x_dim.bounds[0], axis_x_dim.bounds[1], 100),
                        np.linspace(axis_y_dim.bounds[0], axis_y_dim.bounds[1], 100)
                    )
                    y_pred = np.zeros_like(X1)
            else:
                x1 = np.linspace(axis_x_dim.bounds[0], axis_x_dim.bounds[1], 100)
                x2 = np.linspace(axis_y_dim.bounds[0], axis_y_dim.bounds[1], 100)
                X1, X2 = np.meshgrid(x1, x2)

                grid_df = pd.DataFrame({axis_x_name: X1.ravel(), axis_y_name: X2.ravel()})

                all_dims = skopt_space
                for dim in all_dims:
                    if dim.name not in [axis_x_name, axis_y_name]:
                        if dim.name in self.fixed_controls:
                            control = self.fixed_controls[dim.name]
                            grid_df[dim.name] = control.get()
                        else:
                            if hasattr(dim, 'bounds'):
                                grid_df[dim.name] = (dim.bounds[0] + dim.bounds[1]) / 2
                            elif hasattr(dim, 'categories'):
                                grid_df[dim.name] = dim.categories[0]

                if hasattr(model, '_preprocess_X') and callable(model._preprocess_X):
                    processed_grid = model._preprocess_X(grid_df)
                    y_pred = model.predict(processed_grid)
                else:
                    y_pred = model.predict(grid_df)

                y_pred = y_pred.reshape(X1.shape)

            contour = self.ax.contourf(X1, X2, y_pred, levels=20, cmap=cmap)
            self.colorbar = self.fig.colorbar(contour, ax=self.ax)

            if (self.exp_df is not None and
                not self.exp_df.empty and show_experiments):
                if axis_x_name in self.exp_df.columns and axis_y_name in self.exp_df.columns:
                    self.ax.scatter(
                        self.exp_df[axis_x_name],
                        self.exp_df[axis_y_name],
                        c='white', edgecolors='black', s=50, label='Experiments'
                    )

            main_app = self.parent.main_app if hasattr(self.parent, 'main_app') else self.parent
            if (hasattr(main_app, 'next_point') and
                main_app.next_point is not None and show_suggestions):
                next_point = main_app.next_point
                if axis_x_name in next_point.columns and axis_y_name in next_point.columns:
                    self.ax.scatter(
                        next_point[axis_x_name],
                        next_point[axis_y_name],
                        c='red', marker='D', s=80, label='Next Point', zorder=10
                    )

            if self.ax.get_legend_handles_labels()[1]:
                self.ax.legend()

        # Set default labels only if not customized by user
        if not self.customization_options.get('xlabel_user_set'):
            self.ax.set_xlabel(axis_x_name)
            self.customization_options['xlabel'] = axis_x_name
        if not self.customization_options.get('ylabel_user_set'):
            self.ax.set_ylabel(axis_y_name)
            self.customization_options['ylabel'] = axis_y_name
        if not self.customization_options.get('title_user_set'):
            self.ax.set_title("Contour Plot of Model Predictions")
            self.customization_options['title'] = "Contour Plot of Model Predictions"

        self.customization_options['xlim'] = self.ax.get_xlim()
        self.customization_options['ylim'] = self.ax.get_ylim()

        self.apply_customizations_to_axes()
        self.fig.tight_layout()
        self.canvas.draw()

        self.last_plot_method = self.plot_contour

    def plot_metrics(self):
        """Generate the plots for model evaluation metrics using precomputed values."""
        if self.last_plot_method != self.plot_metrics:
            self._clear_user_set_flags()
        self.clear_current_plot()

        error_metric = self.error_metric_var.get()
        # Map UI metric names to session API names
        metric_map = {'RMSE': 'rmse', 'MAE': 'mae', 'MAPE': 'mape', 'R2': 'r2'}
        session_metric = metric_map.get(error_metric)

        if session_metric and self._try_session_plot('plot_metrics', metric=session_metric):
            pass  # Session API succeeded
        else:
            # === Legacy fallback ===
            required_metrics = ['rmse_values', 'mae_values', 'mape_values', 'r2_values']
            if not all(hasattr(self, attr) for attr in required_metrics):
                print("Error: Metrics have not been computed. Please train the model first.")
                return

            rmse_values = self.rmse_values
            mae_values = self.mae_values
            mape_values = self.mape_values
            r2_values = self.r2_values
            x_range = range(5, len(rmse_values) + 5)

            if error_metric == "RMSE":
                self.ax.plot(x_range, rmse_values, marker='o')
                if not self.customization_options.get('title_user_set'):
                    self.ax.set_title("RMSE vs Number of Observations")
                    self.customization_options['title'] = "RMSE vs Number of Observations"
                if not self.customization_options.get('ylabel_user_set'):
                    self.ax.set_ylabel("RMSE")
                    self.customization_options['ylabel'] = "RMSE"
            elif error_metric == "MAE":
                self.ax.plot(x_range, mae_values, marker='o')
                if not self.customization_options.get('title_user_set'):
                    self.ax.set_title("MAE vs Number of Observations")
                    self.customization_options['title'] = "MAE vs Number of Observations"
                if not self.customization_options.get('ylabel_user_set'):
                    self.ax.set_ylabel("MAE")
                    self.customization_options['ylabel'] = "MAE"
            elif error_metric == "MAPE":
                self.ax.plot(x_range, mape_values, marker='o')
                if not self.customization_options.get('title_user_set'):
                    self.ax.set_title("MAPE vs Number of Observations")
                    self.customization_options['title'] = "MAPE vs Number of Observations"
                if not self.customization_options.get('ylabel_user_set'):
                    self.ax.set_ylabel("MAPE (%)")
                    self.customization_options['ylabel'] = "MAPE (%)"
            elif error_metric == "R2":
                self.ax.plot(x_range, r2_values, marker='o')
                if not self.customization_options.get('title_user_set'):
                    self.ax.set_title("R² vs Number of Observations")
                    self.customization_options['title'] = "R² vs Number of Observations"
                if not self.customization_options.get('ylabel_user_set'):
                    self.ax.set_ylabel("R²")
                    self.customization_options['ylabel'] = "R²"
            else:
                print("Error: Unknown error metric selected.")
                return

            if not self.customization_options.get('xlabel_user_set'):
                self.ax.set_xlabel("Number of Observations")
                self.customization_options['xlabel'] = "Number of Observations"

        self.customization_options['xlim'] = self.ax.get_xlim()
        self.customization_options['ylim'] = self.ax.get_ylim()
        self.apply_customizations_to_axes()
        self.fig.tight_layout()
        self.canvas.draw()

        self.last_plot_method = self.plot_metrics
    
    def _on_sigma_changed(self, *args):
        """Callback when sigma dropdown changes - auto-replot parity if it was the last plot."""
        # Only auto-replot if parity plot was the last thing plotted
        if hasattr(self, 'last_plot_method') and self.last_plot_method == self.plot_parity:
            self.plot_parity()

    def plot_parity(self, debug=False):
        """Generate and display a parity plot of actual vs predicted values using cross-validation."""
        if self.exp_df is None or len(self.exp_df) < 5:  # Need at least 5 points for 5-fold CV
            print("Error: Not enough observations for a parity plot (need at least 5).")
            return

        # Clear user customization flags when switching plot types
        if self.last_plot_method != self.plot_parity:
            self._clear_user_set_flags()
        # Clear the current plot
        self.clear_current_plot()

        # Get sigma multiplier from UI
        sigma_str = self.sigma_multiplier_var.get() if hasattr(self, 'sigma_multiplier_var') else "1.96"
        use_error_bars = sigma_str != "None"
        sigma_multiplier = float(sigma_str) if use_error_bars else 1.96

        use_calibrated = (hasattr(self, 'use_calibrated_var') and self.use_calibrated_var.get())

        if not self._try_session_plot('plot_parity',
                sigma_multiplier=sigma_multiplier,
                use_calibrated=use_calibrated,
                show_error_bars=use_error_bars,
                show_metrics=True):
            # === Legacy fallback ===
            if use_calibrated and hasattr(self.gpr_model, 'cv_cached_results_calibrated') and self.gpr_model.cv_cached_results_calibrated is not None:
                cv_results = self.gpr_model.cv_cached_results_calibrated
                y_true_all = cv_results['y_true']
                y_pred_all = cv_results['y_pred']
                y_std_all = cv_results.get('y_std', None)
                print("Using calibrated cross-validation results")
            elif hasattr(self.gpr_model, 'cv_cached_results') and self.gpr_model.cv_cached_results is not None:
                cv_results = self.gpr_model.cv_cached_results
                y_true_all = cv_results['y_true']
                y_pred_all = cv_results['y_pred']
                y_std_all = cv_results.get('y_std', None)
                print("Using cached cross-validation results")
            else:
                print("No cached results found, performing cross-validation...")
                X = self.exp_df.drop(columns=["Output"])
                y = self.exp_df["Output"]
                backend = self.gpr_model.__class__.__name__
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                y_true_all = []
                y_pred_all = []
                y_std_all = []

                if backend == "SklearnModel":
                    for train_idx, test_idx in kf.split(X):
                        X_train = X.iloc[train_idx]
                        y_train = y.iloc[train_idx]
                        X_test = X.iloc[test_idx]
                        y_test = y.iloc[test_idx]
                        optimized_kernel = self.gpr_model.optimized_kernel
                        cv_model = GaussianProcessRegressor(
                            kernel=optimized_kernel,
                            optimizer=None,
                            random_state=self.gpr_model.random_state
                        )
                        if hasattr(self.gpr_model, '_preprocess_X'):
                            X_train_processed = self.gpr_model._preprocess_X(X_train)
                            cv_model.fit(X_train_processed, y_train.values)
                            X_test_processed = self.gpr_model._preprocess_X(X_test)
                            y_pred, y_std = cv_model.predict(X_test_processed, return_std=True)
                        else:
                            cv_model.fit(X_train.values, y_train.values)
                            y_pred, y_std = cv_model.predict(X_test.values, return_std=True)
                        y_true_all.extend(y_test.values)
                        y_pred_all.extend(y_pred)
                        y_std_all.extend(y_std)

                elif backend == "BoTorchModel":
                    for train_idx, test_idx in kf.split(X):
                        X_train = X.iloc[train_idx]
                        y_train = y.iloc[train_idx]
                        X_test = X.iloc[test_idx]
                        y_test = y.iloc[test_idx]
                        from alchemist_core.data.experiment_manager import ExperimentManager
                        temp_exp_manager = ExperimentManager()
                        train_df = pd.concat([X_train, y_train.rename("Output")], axis=1)
                        temp_exp_manager.df = train_df
                        from alchemist_core.models.botorch_model import BoTorchModel
                        temp_model = BoTorchModel(
                            training_iter=self.gpr_model.training_iter,
                            random_state=self.gpr_model.random_state,
                            kernel_options=self.gpr_model.kernel_options,
                            cat_dims=self.gpr_model.cat_dims
                        )
                        temp_model.train(temp_exp_manager)
                        if hasattr(self.gpr_model, 'fitted_state_dict') and self.gpr_model.fitted_state_dict is not None:
                            temp_model.model.load_state_dict(self.gpr_model.fitted_state_dict, strict=False)
                        y_pred, y_std = temp_model.predict(X_test, return_std=True)
                        y_true_all.extend(y_test.values)
                        y_pred_all.extend(y_pred)
                        y_std_all.extend(y_std)
                else:
                    print(f"Error: Unsupported backend type '{backend}' for parity plot.")
                    return

            y_true_all = np.array(y_true_all)
            y_pred_all = np.array(y_pred_all)
            y_std_all = np.array(y_std_all) if y_std_all is not None and len(y_std_all) > 0 else None

            rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
            mae = mean_absolute_error(y_true_all, y_pred_all)
            try:
                r2 = r2_score(y_true_all, y_pred_all)
            except:
                r2 = float('nan')

            if use_error_bars and y_std_all is not None:
                yerr = sigma_multiplier * y_std_all
                self.ax.errorbar(y_true_all, y_pred_all, yerr=yerr,
                               fmt='o', alpha=0.7, capsize=3, capthick=1,
                               elinewidth=1, markersize=5)
            else:
                self.ax.scatter(y_true_all, y_pred_all, alpha=0.7)

            min_val = min(np.min(y_true_all), np.min(y_pred_all))
            max_val = max(np.max(y_true_all), np.max(y_pred_all))
            self.ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Parity line')

            ci_labels = {
                1.0: "68% CI",
                1.96: "95% CI",
                2.0: "95.4% CI",
                2.58: "99% CI",
                3.0: "99.7% CI"
            }
            ci_label = ci_labels.get(sigma_multiplier, f"{sigma_multiplier}σ")

            if use_error_bars and y_std_all is not None:
                title_str = f"Cross-Validation Parity Plot\nRMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}\nError bars: ±{sigma_multiplier}σ ({ci_label})"
            else:
                title_str = f"Cross-Validation Parity Plot\nRMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}"

            if not self.customization_options.get('title_user_set'):
                self.ax.set_title(title_str)
                self.customization_options['title'] = title_str
            if not self.customization_options.get('xlabel_user_set'):
                self.ax.set_xlabel("Actual Values")
                self.customization_options['xlabel'] = "Actual Values"
            if not self.customization_options.get('ylabel_user_set'):
                self.ax.set_ylabel("Predicted Values")
                self.customization_options['ylabel'] = "Predicted Values"

        self.customization_options['xlim'] = self.ax.get_xlim()
        self.customization_options['ylim'] = self.ax.get_ylim()

        self.apply_customizations_to_axes()
        self.fig.tight_layout()
        self.canvas.draw()

        self.last_plot_method = self.plot_parity

    def compute_calibration_metrics(self):
        """
        Compute empirical calibration metrics for the GP model.
        Uses calibrated CV results if toggle is enabled AND available, otherwise uses uncalibrated.
        
        Returns:
            dict: Dictionary containing coverage fractions at different sigma levels
                  and sample size, or None if insufficient data
        """
        # Check if we have CV results with standard deviations
        if not hasattr(self.gpr_model, 'cv_cached_results') or self.gpr_model.cv_cached_results is None:
            print("Error: No cached cross-validation results available. Train a model first.")
            return None
        
        # Use calibrated results if toggle is ON and calibrated results exist
        use_calibrated = (hasattr(self, 'use_calibrated_var') and self.use_calibrated_var.get())
        if use_calibrated and hasattr(self.gpr_model, 'cv_cached_results_calibrated') and self.gpr_model.cv_cached_results_calibrated is not None:
            cv_results = self.gpr_model.cv_cached_results_calibrated
            results_type = "calibrated"
        else:
            cv_results = self.gpr_model.cv_cached_results
            results_type = "uncalibrated"
            
        y_true = cv_results.get('y_true')
        y_pred = cv_results.get('y_pred')
        y_std = cv_results.get('y_std')
        
        if y_true is None or y_pred is None or y_std is None:
            print("Error: Cross-validation results missing standard deviations.")
            return None
        
        # Compute empirical coverage at different sigma levels
        sigma_levels = [1.0, 1.96, 2.0, 2.58, 3.0]
        nominal_coverage = {
            1.0: 0.683,   # ~68.3%
            1.96: 0.950,  # ~95.0%
            2.0: 0.954,   # ~95.4%
            2.58: 0.990,  # ~99.0%
            3.0: 0.997    # ~99.7%
        }
        
        coverage_results = {}
        for sigma in sigma_levels:
            # Check if true value falls within predicted interval
            lower_bound = y_pred - sigma * y_std
            upper_bound = y_pred + sigma * y_std
            within_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
            empirical_coverage = np.mean(within_interval)
            
            coverage_results[sigma] = {
                'empirical': empirical_coverage,
                'nominal': nominal_coverage[sigma],
                'n_samples': len(y_true),
                'results_type': results_type
            }
        
        return coverage_results

    def compute_zscore_diagnostics(self):
        """
        Compute z-score diagnostics (standardized residuals).
        Uses calibrated CV results if available, otherwise uses uncalibrated.
        
        Returns:
            dict: Dictionary containing z-scores, mean, std, and sample size,
                  or None if insufficient data
        """
        # Check if we have CV results with standard deviations
        if not hasattr(self.gpr_model, 'cv_cached_results') or self.gpr_model.cv_cached_results is None:
            print("Error: No cached cross-validation results available. Train a model first.")
            return None
        
        # Use calibrated results if toggle is ON and calibrated results exist
        use_calibrated = (hasattr(self, 'use_calibrated_var') and self.use_calibrated_var.get())
        if use_calibrated and hasattr(self.gpr_model, 'cv_cached_results_calibrated') and self.gpr_model.cv_cached_results_calibrated is not None:
            cv_results = self.gpr_model.cv_cached_results_calibrated
            results_type = "calibrated"
        else:
            cv_results = self.gpr_model.cv_cached_results
            results_type = "uncalibrated"
            
        y_true = cv_results.get('y_true')
        y_pred = cv_results.get('y_pred')
        y_std = cv_results.get('y_std')
        
        if y_true is None or y_pred is None or y_std is None:
            print("Error: Cross-validation results missing standard deviations.")
            return None
        
        # Compute standardized residuals (z-scores)
        z_scores = (y_true - y_pred) / y_std
        
        return {
            'z_scores': z_scores,
            'mean': np.mean(z_scores),
            'std': np.std(z_scores, ddof=1),  # Sample std with N-1
            'n_samples': len(z_scores),
            'results_type': results_type
        }

    def plot_qq_plot(self):
        """Generate and display a Q-Q plot of standardized residuals vs. normal distribution."""
        # Clear user customization flags when switching plot types
        if self.last_plot_method != self.plot_qq_plot:
            self._clear_user_set_flags()
        # Clear the current plot
        self.clear_current_plot()

        use_calibrated = (hasattr(self, 'use_calibrated_var') and self.use_calibrated_var.get())

        if not self._try_session_plot('plot_qq', use_calibrated=use_calibrated):
            # === Legacy fallback ===
            z_diagnostics = self.compute_zscore_diagnostics()

            if z_diagnostics is None:
                return

            z_scores = z_diagnostics['z_scores']
            z_mean = z_diagnostics['mean']
            z_std = z_diagnostics['std']
            n_samples = z_diagnostics['n_samples']
            results_type = z_diagnostics.get('results_type', 'uncalibrated')

            z_sorted = np.sort(z_scores)
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(z_scores)))

            self.ax.scatter(theoretical_quantiles, z_sorted, alpha=0.7, s=30, edgecolors='k', linewidth=0.5)

            min_val = min(theoretical_quantiles.min(), z_sorted.min())
            max_val = max(theoretical_quantiles.max(), z_sorted.max())
            self.ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect calibration')

            if n_samples < 100:
                se = 1.96 / np.sqrt(n_samples)
                self.ax.fill_between([min_val, max_val],
                                    [min_val - se, max_val - se],
                                    [min_val + se, max_val + se],
                                    alpha=0.2, color='red', label='Approximate 95% CI')

            cal_label = " (Calibrated)" if results_type == "calibrated" else " (Uncalibrated)"
            title_str = f"Q-Q Plot: Standardized Residuals vs. Normal Distribution{cal_label}\n"
            title_str += f"Mean(z) = {z_mean:.3f}, Std(z) = {z_std:.3f}, N = {n_samples}"

            if not self.customization_options.get('title_user_set'):
                self.ax.set_title(title_str, fontsize=11)
                self.customization_options['title'] = title_str
            if not self.customization_options.get('xlabel_user_set'):
                self.ax.set_xlabel("Theoretical Quantiles (Standard Normal)")
                self.customization_options['xlabel'] = "Theoretical Quantiles (Standard Normal)"
            if not self.customization_options.get('ylabel_user_set'):
                self.ax.set_ylabel("Observed Quantiles (Standardized Residuals)")
                self.customization_options['ylabel'] = "Observed Quantiles (Standardized Residuals)"

            self.ax.legend(loc='best')
            self.ax.grid(True, alpha=0.3)

            # Print diagnostics to console
            cal_status = "CALIBRATED" if results_type == "calibrated" else "UNCALIBRATED"
            print("\n" + "="*60)
            print(f"Z-SCORE DIAGNOSTICS ({cal_status})")
            print("="*60)
            print(f"Mean(z):     {z_mean:.4f}  (expected ≈ 0 for well-calibrated)")
            print(f"Std(z):      {z_std:.4f}  (expected ≈ 1 for well-calibrated)")
            print(f"Sample size: {n_samples}")
            print("="*60)
            if abs(z_mean) < 0.1 and abs(z_std - 1.0) < 0.2:
                print("✓ Model appears well-calibrated (unbiased, good uncertainty)")
            elif abs(z_mean) > 0.2:
                print("⚠ Model may be biased (mean(z) far from 0)")
            elif abs(z_std - 1.0) > 0.3:
                if z_std < 1.0:
                    print("⚠ Model may be under-confident (std(z) < 1)")
                else:
                    print("⚠ Model may be over-confident (std(z) > 1)")
            print("="*60 + "\n")

        self.customization_options['xlim'] = self.ax.get_xlim()
        self.customization_options['ylim'] = self.ax.get_ylim()

        self.apply_customizations_to_axes()
        self.fig.tight_layout()
        self.canvas.draw()

        self.last_plot_method = self.plot_qq_plot

    def plot_calibration_curve(self):
        """Generate and display a calibration curve (reliability diagram)."""
        # Clear user customization flags when switching plot types
        if self.last_plot_method != self.plot_calibration_curve:
            self._clear_user_set_flags()
        # Clear the current plot
        self.clear_current_plot()

        use_calibrated = (hasattr(self, 'use_calibrated_var') and self.use_calibrated_var.get())

        if not self._try_session_plot('plot_calibration', use_calibrated=use_calibrated):
            # === Legacy fallback ===
            calibration_metrics = self.compute_calibration_metrics()

            if calibration_metrics is None:
                return

            results_type = list(calibration_metrics.values())[0].get('results_type', 'uncalibrated')

            if use_calibrated and hasattr(self.gpr_model, 'cv_cached_results_calibrated') and self.gpr_model.cv_cached_results_calibrated is not None:
                cv_results = self.gpr_model.cv_cached_results_calibrated
            else:
                cv_results = self.gpr_model.cv_cached_results

            y_true = cv_results['y_true']
            y_pred = cv_results['y_pred']
            y_std = cv_results['y_std']
            n_samples = len(y_true)

            nominal_probs = np.arange(0.10, 1.00, 0.05)
            empirical_coverage = []

            for prob in nominal_probs:
                sigma = stats.norm.ppf((1 + prob) / 2)
                lower_bound = y_pred - sigma * y_std
                upper_bound = y_pred + sigma * y_std
                within_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
                empirical_coverage.append(np.mean(within_interval))

            empirical_coverage = np.array(empirical_coverage)

            self.ax.plot(nominal_probs, empirical_coverage, 'o-', linewidth=2,
                        markersize=6, label='Empirical coverage', color='steelblue')
            self.ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect calibration')

            self.ax.fill_between(nominal_probs, nominal_probs, empirical_coverage,
                                where=(empirical_coverage < nominal_probs),
                                alpha=0.2, color='orange', label='Over-confident')
            self.ax.fill_between(nominal_probs, nominal_probs, empirical_coverage,
                                where=(empirical_coverage >= nominal_probs),
                                alpha=0.2, color='blue', label='Under-confident')

            cal_label = " (Calibrated)" if results_type == "calibrated" else " (Uncalibrated)"
            title_str = f"Calibration Curve (Reliability Diagram){cal_label}\nN = {n_samples}"

            if not self.customization_options.get('title_user_set'):
                self.ax.set_title(title_str)
                self.customization_options['title'] = title_str
            if not self.customization_options.get('xlabel_user_set'):
                self.ax.set_xlabel("Nominal Coverage Probability")
                self.customization_options['xlabel'] = "Nominal Coverage Probability"
            if not self.customization_options.get('ylabel_user_set'):
                self.ax.set_ylabel("Empirical Coverage Probability")
                self.customization_options['ylabel'] = "Empirical Coverage Probability"

            self.ax.legend(loc='best')
            self.ax.grid(True, alpha=0.3)
            self.ax.set_xlim([0, 1])
            self.ax.set_ylim([0, 1])

            # Print calibration metrics to console
            cal_status = "(CALIBRATED)" if results_type == "calibrated" else "(UNCALIBRATED)"
            print("\n" + "="*70)
            print(f"CALIBRATION METRICS (Empirical Coverage) {cal_status}")
            print("="*70)
            print(f"{'Confidence':<15} {'Nominal':<12} {'Empirical':<12} {'Difference':<12} {'Status'}")
            print("-"*70)

            for sigma, metrics in sorted(calibration_metrics.items()):
                nominal = metrics['nominal']
                empirical = metrics['empirical']
                diff = empirical - nominal

                if abs(diff) < 0.05:
                    status = "✓ Good"
                elif diff > 0.1:
                    status = "⚠ Under-conf"
                elif diff < -0.1:
                    status = "⚠ Over-conf"
                else:
                    status = "~ Acceptable"

                if sigma == 1.0:
                    conf_label = "±1.0σ (68%)"
                elif sigma == 1.96:
                    conf_label = "±1.96σ (95%)"
                elif sigma == 2.0:
                    conf_label = "±2.0σ (95%)"
                elif sigma == 2.58:
                    conf_label = "±2.58σ (99%)"
                elif sigma == 3.0:
                    conf_label = "±3.0σ (99.7%)"
                else:
                    conf_label = f"±{sigma}σ"

                print(f"{conf_label:<15} {nominal:>6.3f} ({nominal*100:>4.1f}%)  {empirical:>6.3f} ({empirical*100:>4.1f}%)  "
                      f"{diff:>+6.3f} ({diff*100:>+5.1f}%)  {status}")

            print("-"*70)
            print(f"Sample size: N = {n_samples}")
            if n_samples < 30:
                print("⚠ WARNING: Small sample size (N < 30). Coverage estimates may be noisy.")
                print("           Consider reporting binomial confidence intervals.")
            print("="*70 + "\n")

        self.customization_options['xlim'] = self.ax.get_xlim()
        self.customization_options['ylim'] = self.ax.get_ylim()

        self.apply_customizations_to_axes()
        self.fig.tight_layout()
        self.canvas.draw()

        self.last_plot_method = self.plot_calibration_curve

    
    def customize_plot(self, event=None):
        """Open a window to customize plot parameters, prepopulated with current settings."""
        customization_window = ctk.CTkToplevel(self.parent)
        customization_window.title("Customize Plot")
        customization_window.geometry("300x650")  # Increased height for number formatting controls
        customization_window.resizable(False, False)
        customization_window.lift()
        customization_window.focus_force()
        customization_window.grab_set()

        # Retrieve current settings
        current_title = self.ax.get_title()
        current_xlabel = self.ax.get_xlabel()
        current_ylabel = self.ax.get_ylabel()
        current_cmap = "viridis"
        if self.ax.collections:
            try:
                current_cmap = self.ax.collections[0].get_cmap().name
            except Exception:
                current_cmap = "viridis"
        current_font = self.customization_options.get('font', 'Arial') if hasattr(self, 'customization_options') else "Arial"
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        current_title_fontsize = int(self.ax.title.get_fontsize())
        current_label_fontsize = int(self.ax.xaxis.label.get_fontsize())
        current_title_fontstyle = 'bold' if self.ax.title.get_fontweight() == 'bold' else 'normal'
        current_label_fontstyle = 'bold' if self.ax.xaxis.label.get_fontweight() == 'bold' else 'normal'

        frame = ctk.CTkScrollableFrame(customization_window)
        frame.pack(pady=5, padx=5, fill="both", expand=True)

        ctk.CTkLabel(frame, text="Plot Title:").pack(pady=5)
        title_entry = ctk.CTkEntry(frame)
        title_entry.insert(0, current_title)
        title_entry.pack(pady=5)

        ctk.CTkLabel(frame, text="X-axis Label:").pack(pady=5)
        x_label_entry = ctk.CTkEntry(frame)
        x_label_entry.insert(0, current_xlabel)
        x_label_entry.pack(pady=5)

        ctk.CTkLabel(frame, text="Y-axis Label:").pack(pady=5)
        y_label_entry = ctk.CTkEntry(frame)
        y_label_entry.insert(0, current_ylabel)
        y_label_entry.pack(pady=5)

        ctk.CTkLabel(frame, text="Colormap:").pack(pady=5)
        colormap_var = ctk.StringVar(value=current_cmap)
        colormap_menu = ctk.CTkOptionMenu(frame, values=plt.colormaps(), variable=colormap_var)
        colormap_menu.pack(pady=5)

        ctk.CTkLabel(frame, text="Font:").pack(pady=5)
        font_var = ctk.StringVar(value=current_font)
        fonts = [
            "DejaVu Sans",
            "DejaVu Serif",
            "Arial",
            "Times New Roman",
            "Courier New",
            "Verdana",
            "Georgia",
            "Palatino Linotype",
            "STIXGeneral",
            "Segoe UI"
        ]
        font_menu = ctk.CTkOptionMenu(frame, values=fonts, variable=font_var)
        font_menu.pack(pady=5)

        ctk.CTkLabel(frame, text="X-axis Limit Min:").pack(pady=5)
        x_min_entry = ctk.CTkEntry(frame)
        x_min_entry.insert(0, str(current_xlim[0]))
        x_min_entry.pack(pady=5)

        ctk.CTkLabel(frame, text="X-axis Limit Max:").pack(pady=5)
        x_max_entry = ctk.CTkEntry(frame)
        x_max_entry.insert(0, str(current_xlim[1]))
        x_max_entry.pack(pady=5)

        ctk.CTkLabel(frame, text="Y-axis Limit Min:").pack(pady=5)
        y_min_entry = ctk.CTkEntry(frame)
        y_min_entry.insert(0, str(current_ylim[0]))
        y_min_entry.pack(pady=5)

        ctk.CTkLabel(frame, text="Y-axis Limit Max:").pack(pady=5)
        y_max_entry = ctk.CTkEntry(frame)
        y_max_entry.insert(0, str(current_ylim[1]))
        y_max_entry.pack(pady=5)

        ctk.CTkLabel(frame, text="Title Font Size:").pack(pady=5)
        title_fontsize_entry = ctk.CTkEntry(frame)
        title_fontsize_entry.insert(0, str(current_title_fontsize))
        title_fontsize_entry.pack(pady=5)

        ctk.CTkLabel(frame, text="Label Font Size:").pack(pady=5)
        label_fontsize_entry = ctk.CTkEntry(frame)
        label_fontsize_entry.insert(0, str(current_label_fontsize))
        label_fontsize_entry.pack(pady=5)

        ctk.CTkLabel(frame, text="Title Font Style:").pack(pady=5)
        title_fontstyle_var = ctk.StringVar(value=current_title_fontstyle)
        title_fontstyle_menu = ctk.CTkOptionMenu(frame, values=["normal", "bold"], variable=title_fontstyle_var)
        title_fontstyle_menu.pack(pady=5)

        ctk.CTkLabel(frame, text="Label Font Style:").pack(pady=5)
        label_fontstyle_var = ctk.StringVar(value=current_label_fontstyle)
        label_fontstyle_menu = ctk.CTkOptionMenu(frame, values=["normal", "bold"], variable=label_fontstyle_var)
        label_fontstyle_menu.pack(pady=5)

        ctk.CTkLabel(frame, text="Tick Style:").pack(pady=5)
        tick_style_var = ctk.StringVar(value=self.customization_options.get('tick_style', "x and y"))
        tick_style_menu = ctk.CTkOptionMenu(frame, values=["all", "x and y", "none"], variable=tick_style_var)
        tick_style_menu.pack(pady=5)

        ctk.CTkLabel(frame, text="Tick Direction:").pack(pady=5)
        tick_direction_var = ctk.StringVar(value=self.customization_options.get('tick_direction', "out"))
        tick_direction_menu = ctk.CTkOptionMenu(frame, values=["in", "out"], variable=tick_direction_var)
        tick_direction_menu.pack(pady=5)

        # Add a section title for number formatting options
        ctk.CTkLabel(frame, text="Number Formatting", font=("Arial", 12, "bold")).pack(pady=(15, 5))
        
        # Add a small help text
        help_text = "Format options: auto, decimal (0.00), scientific (1e-3), percent (50%), integer (1), custom (3 sig figs)"
        help_label = ctk.CTkLabel(frame, text=help_text, wraplength=280, font=("Arial", 9))
        help_label.pack(pady=(0, 10))
        
        # X-axis number formatting
        ctk.CTkLabel(frame, text="X-Axis Number Format:").pack(pady=5)
        x_format_var = ctk.StringVar(value=self.customization_options.get('x_number_format', "auto"))
        x_format_options = ["auto", "decimal", "scientific", "percent", "integer", "custom"]
        x_format_menu = ctk.CTkOptionMenu(frame, values=x_format_options, variable=x_format_var)
        x_format_menu.pack(pady=5)
        
        # Y-axis number formatting
        ctk.CTkLabel(frame, text="Y-Axis Number Format:").pack(pady=5)
        y_format_var = ctk.StringVar(value=self.customization_options.get('y_number_format', "auto"))
        y_format_menu = ctk.CTkOptionMenu(frame, values=x_format_options, variable=y_format_var)
        y_format_menu.pack(pady=5)
        
        # Colorbar number formatting
        ctk.CTkLabel(frame, text="Colorbar Number Format:").pack(pady=5)
        colorbar_format_var = ctk.StringVar(value=self.customization_options.get('colorbar_format', "auto"))
        colorbar_format_menu = ctk.CTkOptionMenu(frame, values=x_format_options, variable=colorbar_format_var)
        colorbar_format_menu.pack(pady=5)

        # Add a section title for data display options
        ctk.CTkLabel(frame, text="Data Display Options", font=("Arial", 12, "bold")).pack(pady=(15, 5))
        
        # Add toggle for experimental points
        show_exp_points_var = ctk.BooleanVar(
            value=self.customization_options.get('show_experiments', False)
        )
        show_exp_switch = ctk.CTkSwitch(
            frame, 
            text="Show Experimental Points",
            variable=show_exp_points_var
        )
        show_exp_switch.pack(pady=5)
        
        # Add toggle for next point
        show_next_point_var = ctk.BooleanVar(
            value=self.customization_options.get('show_next_point', False)
        )
        show_next_switch = ctk.CTkSwitch(
            frame, 
            text="Show Next Point",
            variable=show_next_point_var
        )
        show_next_switch.pack(pady=5)

        button_frame = ctk.CTkFrame(customization_window)
        button_frame.pack(pady=10, fill="x")

        def apply_customizations():
            try:
                # Store user customizations
                user_customizations = {
                    'title': title_entry.get(),
                    'xlabel': x_label_entry.get(),
                    'ylabel': y_label_entry.get(),
                    'colormap': colormap_var.get(),
                    'font': font_var.get(),
                    'xlim': [float(x_min_entry.get()), float(x_max_entry.get())],
                    'ylim': [float(y_min_entry.get()), float(y_max_entry.get())],
                    'title_fontsize': int(title_fontsize_entry.get()),
                    'label_fontsize': int(label_fontsize_entry.get()),
                    'title_fontstyle': title_fontstyle_var.get(),
                    'label_fontstyle': label_fontstyle_var.get(),
                    'tick_style': tick_style_var.get(),
                    'tick_direction': tick_direction_var.get(),
                    'show_experiments': show_exp_points_var.get(),
                    'show_next_point': show_next_point_var.get(),
                    'x_number_format': x_format_var.get(),
                    'y_number_format': y_format_var.get(),
                    'colorbar_format': colorbar_format_var.get(),
                    # Set flags to indicate these were set by user
                    'title_user_set': True,
                    'xlabel_user_set': True,
                    'ylabel_user_set': True,
                    'xlim_user_set': True,
                    'ylim_user_set': True
                }
                
                # Update the customization options with user values
                self.customization_options.update(user_customizations)
                
                # Apply customizations directly without regenerating the plot
                self.apply_customizations_to_axes()
                self.fig.tight_layout()
                self.canvas.draw()

                # If we're changing data display options (experiments/next point),
                # we need to redraw the plot to show/hide those elements
                if (self.last_plot_method == self.plot_contour and
                    ('show_experiments' in user_customizations or 'show_next_point' in user_customizations)):
                    self.last_plot_method()

                # Briefly change button text to show success
                apply_button.configure(text="Applied!")
                customization_window.after(1000, lambda: apply_button.configure(text="Apply"))
                
            except Exception as e:
                print(f"Error applying customizations: {e}")
                # Change button text to show error
                apply_button.configure(text="Error!")
                customization_window.after(2000, lambda: apply_button.configure(text="Apply"))

        apply_button = ctk.CTkButton(button_frame, text="Apply", command=apply_customizations)
        apply_button.pack(side="left", padx=5, pady=10)
        
        close_button = ctk.CTkButton(button_frame, text="Close", command=customization_window.destroy)
        close_button.pack(side="right", padx=5, pady=10)

    def apply_customizations_to_axes(self):
        """
        Apply user customization options stored in self.customization_options to the axes.
        Title/xlabel/ylabel text is only overwritten when the user has explicitly set them
        via the Customize Plot dialog (_user_set flags). Font/style properties are always applied.
        """
        custom = getattr(self, 'customization_options', {})
        font = custom.get('font', None)
        # Only override text content when the user explicitly set it via the dialog
        if custom.get('title_user_set') and custom.get('title'):
            self.ax.set_title(custom['title'], fontname=font if font else None, fontsize=custom.get('title_fontsize', None), fontweight=custom.get('title_fontstyle', None))
        elif font or custom.get('title_fontsize') or custom.get('title_fontstyle'):
            # Apply font/style to whatever title is already on the axes
            self.ax.set_title(self.ax.get_title(), fontname=font if font else None, fontsize=custom.get('title_fontsize', None), fontweight=custom.get('title_fontstyle', None))
        if custom.get('xlabel_user_set') and custom.get('xlabel'):
            self.ax.set_xlabel(custom['xlabel'], fontname=font if font else None, fontsize=custom.get('label_fontsize', None), fontweight=custom.get('label_fontstyle', None))
        elif font or custom.get('label_fontsize') or custom.get('label_fontstyle'):
            self.ax.set_xlabel(self.ax.get_xlabel(), fontname=font if font else None, fontsize=custom.get('label_fontsize', None), fontweight=custom.get('label_fontstyle', None))
        if custom.get('ylabel_user_set') and custom.get('ylabel'):
            self.ax.set_ylabel(custom['ylabel'], fontname=font if font else None, fontsize=custom.get('label_fontsize', None), fontweight=custom.get('label_fontstyle', None))
        elif font or custom.get('label_fontsize') or custom.get('label_fontstyle'):
            self.ax.set_ylabel(self.ax.get_ylabel(), fontname=font if font else None, fontsize=custom.get('label_fontsize', None), fontweight=custom.get('label_fontstyle', None))
        if font:
            for label in (self.ax.get_xticklabels() + self.ax.get_yticklabels()):
                label.set_fontname(font)
                if custom.get('label_fontstyle') == 'bold':
                    label.set_fontweight('bold')
        # Only apply axis limits if the user explicitly set them via the dialog
        if custom.get('xlim_user_set') and custom.get('xlim'):
            self.ax.set_xlim(custom['xlim'])
        if custom.get('ylim_user_set') and custom.get('ylim'):
            self.ax.set_ylim(custom['ylim'])
        
        # Apply tick styling (use defaults if not explicitly set)
        tick_style = custom.get('tick_style', 'x and y')
        tick_direction = custom.get('tick_direction', 'out')
        if tick_style == 'all':
            self.ax.tick_params(axis='both', which='both', direction=tick_direction,
                                bottom=True, top=True, left=True, right=True)
        elif tick_style == 'x and y':
            self.ax.tick_params(axis='x', which='both', direction=tick_direction,
                                bottom=True, top=False)
            self.ax.tick_params(axis='y', which='both', direction=tick_direction,
                                left=True, right=False)
        elif tick_style == 'none':
            self.ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

        if custom.get('colormap') and self.ax.collections:
            try:
                self.ax.collections[0].set_cmap(custom['colormap'])
            except Exception as e:
                print("Error updating colormap:", e)
        if hasattr(self, 'colorbar') and self.colorbar is not None:
            self.colorbar.ax.tick_params(labelsize=10)
            if font:
                for label in self.colorbar.ax.get_yticklabels():
                    label.set_fontname(font)
                    if custom.get('label_fontstyle') == 'bold':
                        label.set_fontweight('bold')
            
            # Apply colorbar number formatting
            colorbar_format = custom.get('colorbar_format', 'auto')
            colorbar_formatter = self._get_number_formatter(colorbar_format)
            if colorbar_formatter:
                self.colorbar.ax.yaxis.set_major_formatter(colorbar_formatter)
        
        # Apply axis number formatting (apply defaults if not set)
        x_format = custom.get('x_number_format', 'auto')
        x_formatter = self._get_number_formatter(x_format)
        if x_formatter:
            self.ax.xaxis.set_major_formatter(x_formatter)
        
        y_format = custom.get('y_number_format', 'auto')
        y_formatter = self._get_number_formatter(y_format)
        if y_formatter:
            self.ax.yaxis.set_major_formatter(y_formatter)
        
        # Apply tight layout to prevent label cutoff
        self.fig.tight_layout()
        self.canvas.draw()


    def get_hyperparameters(self):
        """Extract the hyperparameters from the trained model."""
        if self.gpr_model is None:
            print("Error: Model is not trained.")
            return None

        # Access the kernel from the SklearnModel
        kernel = self.gpr_model.kernel
        if kernel is None:
            print("Error: Kernel is not defined in the model.")
            return None

        # Extract hyperparameters
        hyperparameters = kernel.get_params()
        lengthscale = hyperparameters.get('k2__length_scale', 'N/A')

        return {
            'Kernel': kernel,
            'Lengthscale': lengthscale
        }

    def _clear_user_set_flags(self):
        """Clear per-plot user customization flags (title, labels, limits).

        Called when switching to a different plot type so that customizations
        set for one plot (e.g. contour) don't bleed into another (e.g. parity).
        Font/style settings persist across plots.
        """
        for flag in ('title_user_set', 'xlabel_user_set', 'ylabel_user_set',
                     'xlim_user_set', 'ylim_user_set'):
            self.customization_options.pop(flag, None)

    def clear_current_plot(self):
        """Clears the current axes and removes any extra axes (e.g. colorbars)."""
        if self.fig is None or self.ax is None:
            self.initialize_figure()
            return
        # Remove all extra axes (colorbars, make_axes_locatable dividers) from figure
        for fig_ax in list(self.fig.axes):
            if fig_ax is not self.ax:
                try:
                    fig_ax.remove()
                except Exception:
                    pass
        self.colorbar = None
        self.ax.clear()
        # Reset the axes locator so make_axes_locatable dividers from the
        # previous contour plot don't keep reserving colorbar space.
        self.ax.set_axes_locator(None)

    def _on_calibration_toggle(self):
        """Callback when calibrated/uncalibrated toggle is changed. Re-plots the last plot."""
        if self.last_plot_method is not None:
            print(f"\nSwitching to {'calibrated' if self.use_calibrated_var.get() else 'uncalibrated'} results...")
            self.last_plot_method()  # Re-run the last plot method
        else:
            print("No plot to refresh yet. Generate a plot first.")


