"""
Optimization Session API - High-level interface for Bayesian optimization workflows.

This module provides the main entry point for using ALchemist as a headless library.
"""

from typing import Optional, Dict, Any, List, Tuple, Callable, Union, Literal
import pandas as pd
import numpy as np
import json
import hashlib
from pathlib import Path
from alchemist_core.data.search_space import SearchSpace
from alchemist_core.data.experiment_manager import ExperimentManager
from alchemist_core.events import EventEmitter
from alchemist_core.config import get_logger
from alchemist_core.audit_log import AuditLog, SessionMetadata, AuditEntry

# Optional matplotlib import for visualization methods
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False
    Figure = None  # Type hint placeholder

# Import visualization functions (delegates to visualization module)
try:
    from alchemist_core.visualization import (
        create_parity_plot,
        create_contour_plot,
        create_slice_plot,
        create_metrics_plot,
        create_qq_plot,
        create_calibration_plot,
        create_pareto_plot,
        check_matplotlib
    )
    _HAS_VISUALIZATION = True
except ImportError:
    _HAS_VISUALIZATION = False

logger = get_logger(__name__)


class OptimizationSession:
    """
    High-level interface for Bayesian optimization workflows.
    
    This class orchestrates the complete optimization loop:
    1. Define search space
    2. Load/add experimental data
    3. Train surrogate model
    4. Run acquisition to suggest next experiments
    5. Iterate
    
    Example:
        > from alchemist_core import OptimizationSession
        > 
        > # Create session with search space
        > session = OptimizationSession()
        > session.add_variable('temperature', 'real', bounds=(300, 500))
        > session.add_variable('pressure', 'real', bounds=(1, 10))
        > session.add_variable('catalyst', 'categorical', categories=['A', 'B', 'C'])
        > 
        > # Load experimental data
        > session.load_data('experiments.csv', target_columns='yield')
        > 
        > # Train model
        > session.train_model(backend='botorch', kernel='Matern')
        > 
        > # Suggest next experiment
        > next_point = session.suggest_next(strategy='EI', goal='maximize')
        > print(next_point)
    """
    
    def __init__(self, search_space: Optional[SearchSpace] = None, 
                 experiment_manager: Optional[ExperimentManager] = None,
                 event_emitter: Optional[EventEmitter] = None,
                 session_metadata: Optional[SessionMetadata] = None):
        """
        Initialize optimization session.
        
        Args:
            search_space: Pre-configured SearchSpace object (optional)
            experiment_manager: Pre-configured ExperimentManager (optional)
            event_emitter: EventEmitter for progress notifications (optional)
            session_metadata: Pre-configured session metadata (optional)
        """
        self.search_space = search_space if search_space is not None else SearchSpace()
        self.experiment_manager = experiment_manager if experiment_manager is not None else ExperimentManager()
        self.events = event_emitter if event_emitter is not None else EventEmitter()
        
        # Session metadata and audit log
        self.metadata = session_metadata if session_metadata is not None else SessionMetadata.create()
        self.audit_log = AuditLog()
        
        # Link search_space to experiment_manager
        self.experiment_manager.set_search_space(self.search_space)
        
        # Model and acquisition state
        self.model = None
        self.model_backend = None
        self.acquisition = None
        
        # Staged experiments (for workflow management)
        self.staged_experiments = []  # List of experiment dicts awaiting evaluation
        self.last_suggestions = []  # Most recent acquisition suggestions (for UI)

        # Outcome constraints for constrained optimization
        self._outcome_constraints = []  # List of {objective_name, bound_type, value}
        
        # Configuration
        self.config = {
            'random_state': 42,
            'verbose': True,
            'auto_train': False,  # Auto-train model after adding experiments
            'auto_train_threshold': 5  # Minimum experiments before auto-train
        }
        
        logger.info(f"OptimizationSession initialized: {self.metadata.session_id}")
    
    # ============================================================
    # Search Space Management
    # ============================================================
    
    def add_variable(self, name: str, var_type: str, **kwargs) -> None:
        """
        Add a variable to the search space.
        
        Args:
            name: Variable name
            var_type: Type ('real', 'integer', 'categorical')
            **kwargs: Type-specific parameters:
                - For 'real'/'integer': bounds=(min, max) or min=..., max=...
                - For 'categorical': categories=[list of values] or values=[list]
        
        Example:
            > session.add_variable('temp', 'real', bounds=(300, 500))
            > session.add_variable('catalyst', 'categorical', categories=['A', 'B'])
        """
        # Convert user-friendly API to internal format
        params = kwargs.copy()
        
        # Handle 'bounds' parameter for real/integer
        if 'bounds' in params and var_type.lower() in ['real', 'integer']:
            min_val, max_val = params.pop('bounds')
            params['min'] = min_val
            params['max'] = max_val
        
        # Handle 'categories' parameter for categorical
        if 'categories' in params and var_type.lower() == 'categorical':
            params['values'] = params.pop('categories')
        
        self.search_space.add_variable(name, var_type, **params)
        
        # Update the search_space reference in experiment_manager
        self.experiment_manager.set_search_space(self.search_space)
        
        logger.info(f"Added variable '{name}' ({var_type}) to search space")
        self.events.emit('variable_added', {'name': name, 'type': var_type})
    
    def load_search_space(self, filepath: str) -> None:
        """
        Load search space from JSON or CSV file.
        
        Args:
            filepath: Path to search space definition file
        """
        self.search_space = SearchSpace.from_json(filepath)
        logger.info(f"Loaded search space from {filepath}")
        self.events.emit('search_space_loaded', {'filepath': filepath})
    
    def get_search_space_summary(self) -> Dict[str, Any]:
        """
        Get summary of current search space.
        
        Returns:
            Dictionary with variable information
        """
        variables = []
        for var in self.search_space.variables:
            var_summary = {
                'name': var['name'],
                'type': var['type']
            }
            
            # Convert min/max to bounds for real/integer
            if var['type'] in ['real', 'integer']:
                if 'min' in var and 'max' in var:
                    var_summary['bounds'] = [var['min'], var['max']]
                else:
                    var_summary['bounds'] = None
            else:
                var_summary['bounds'] = None
            
            # Convert values to categories for categorical
            if var['type'] == 'categorical':
                var_summary['categories'] = var.get('values')
            else:
                var_summary['categories'] = None
            
            # Include optional fields
            if 'unit' in var:
                var_summary['unit'] = var['unit']
            if 'description' in var:
                var_summary['description'] = var['description']
            
            variables.append(var_summary)
        
        return {
            'n_variables': len(self.search_space.variables),
            'variables': variables,
            'categorical_variables': self.search_space.get_categorical_variables()
        }
    
    # ============================================================
    # Multi-Objective Properties
    # ============================================================

    @property
    def is_multi_objective(self) -> bool:
        """Whether this session has multiple target objectives."""
        return len(self.experiment_manager.target_columns) > 1

    @property
    def n_objectives(self) -> int:
        """Number of target objectives."""
        return len(self.experiment_manager.target_columns)

    @property
    def objective_names(self) -> List[str]:
        """Names of target objectives."""
        return list(self.experiment_manager.target_columns)

    # ============================================================
    # Constraint API
    # ============================================================

    def add_input_constraint(self, constraint_type: str, coefficients: Dict[str, float],
                             rhs: float, name: Optional[str] = None):
        """Add linear input constraint: sum(coeff_i * x_i) <= rhs or == rhs.

        Args:
            constraint_type: 'inequality' (<=) or 'equality' (==)
            coefficients: {variable_name: coefficient} mapping
            rhs: right-hand side value
            name: optional human-readable name

        Example:
            >>> session.add_input_constraint('inequality', {'x1': 1.0, 'x2': 1.0}, rhs=1.5)
        """
        self.search_space.add_constraint(constraint_type, coefficients, rhs, name)
        logger.info(f"Added input constraint: {constraint_type}, {coefficients} {'<=' if constraint_type == 'inequality' else '=='} {rhs}")

    def add_outcome_constraint(self, objective_name: str, bound_type: str, value: float):
        """Add outcome constraint on a modeled output.

        Args:
            objective_name: Name of the target column to constrain
            bound_type: 'lower' (output >= value) or 'upper' (output <= value)
            value: The constraint threshold

        Example:
            >>> session.add_outcome_constraint('selectivity', 'lower', 80.0)
        """
        if bound_type not in ('lower', 'upper'):
            raise ValueError(f"bound_type must be 'lower' or 'upper', got '{bound_type}'")

        self._outcome_constraints.append({
            'objective_name': objective_name,
            'bound_type': bound_type,
            'value': value
        })
        op = '>=' if bound_type == 'lower' else '<='
        logger.info(f"Added outcome constraint: {objective_name} {op} {value}")

    def get_outcome_constraints(self) -> List[Dict]:
        """Return list of outcome constraint dicts."""
        return [c.copy() for c in self._outcome_constraints]

    # ============================================================
    # Data Management
    # ============================================================
    
    def load_data(self, filepath: str, target_columns: Union[str, List[str]] = 'Output',
                  noise_column: Optional[str] = None) -> None:
        """
        Load experimental data from CSV file.
        
        Args:
            filepath: Path to CSV file
            target_columns: Target column name(s). Can be:
                - String for single-objective: 'yield'
                - List for multi-objective: ['yield', 'selectivity']
                Default: 'Output'
            noise_column: Optional column with measurement noise/uncertainty
        
        Examples:
            Single-objective:
            >>> session.load_data('experiments.csv', target_columns='yield')
            >>> session.load_data('experiments.csv', target_columns=['yield'])  # also works
            
            Multi-objective:
            >>> session.load_data('experiments.csv', target_columns=['yield', 'selectivity'])
        
        Note:
            If the CSV doesn't have columns matching target_columns, an error will be raised.
            Target columns will be preserved with their original names internally.
        """
        # Load the CSV
        import pandas as pd
        df = pd.read_csv(filepath)
        
        # Normalize target_columns to list
        if isinstance(target_columns, str):
            target_columns_list = [target_columns]
        else:
            target_columns_list = list(target_columns)
        
        # Validate that all target columns exist
        missing_cols = [col for col in target_columns_list if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Target column(s) {missing_cols} not found in CSV file. "
                f"Available columns: {list(df.columns)}. "
                f"Please specify the correct target column name(s) using the target_columns parameter."
            )
        
        # Warn if 'Output' column exists but user specified different target(s)
        if 'Output' in df.columns and 'Output' not in target_columns_list:
            logger.warning(
                f"CSV contains 'Output' column but you specified {target_columns_list}. "
                f"Using {target_columns_list} as specified."
            )
        
        # Store the target column names for ExperimentManager
        target_col_internal = target_columns_list
        
        # Rename noise column to 'Noise' if specified and different
        if noise_column and noise_column in df.columns and noise_column != 'Noise':
            df = df.rename(columns={noise_column: 'Noise'})
        
        # Save to temporary file and load via ExperimentManager
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as tmp:
            df.to_csv(tmp.name, index=False)
            temp_path = tmp.name
        
        try:
            # Create ExperimentManager with the specified target column(s)
            self.experiment_manager = ExperimentManager(
                search_space=self.search_space,
                target_columns=target_col_internal
            )
            self.experiment_manager.load_from_csv(temp_path)
        finally:
            # Clean up temp file
            import os
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        n_experiments = len(self.experiment_manager.df)
        logger.info(f"Loaded {n_experiments} experiments from {filepath}")
        self.events.emit('data_loaded', {'n_experiments': n_experiments, 'filepath': filepath})
    
    def add_experiment(self, inputs: Dict[str, Any], output: float, 
                      noise: Optional[float] = None, iteration: Optional[int] = None,
                      reason: Optional[str] = None) -> None:
        """
        Add a single experiment to the dataset.
        
        Args:
            inputs: Dictionary mapping variable names to values
            output: Target/output value
            noise: Optional measurement uncertainty
            iteration: Iteration number (auto-assigned if None)
            reason: Reason for this experiment (e.g., 'Manual', 'Expected Improvement')
        
        Example:
            > session.add_experiment(
            ...     inputs={'temperature': 350, 'catalyst': 'A'},
            ...     output=0.85,
            ...     reason='Manual'
            ... )
        """
        # Use ExperimentManager's add_experiment method
        self.experiment_manager.add_experiment(
            point_dict=inputs,
            output_value=output,
            noise_value=noise,
            iteration=iteration,
            reason=reason
        )
        
        logger.info(f"Added experiment: {inputs} → {output}")
        self.events.emit('experiment_added', {'inputs': inputs, 'output': output})
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of current experimental data.
        
        Returns:
            Dictionary with data statistics
        """
        df = self.experiment_manager.get_data()
        if df is None or df.empty:
            return {'n_experiments': 0, 'has_data': False}
        
        X, y = self.experiment_manager.get_features_and_target()
        return {
            'n_experiments': len(y),
            'has_data': True,
            'has_noise': self.experiment_manager.has_noise_data(),
            'target_stats': {
                'min': float(y.min()),
                'max': float(y.max()),
                'mean': float(y.mean()),
                'std': float(y.std())
            },
            'feature_names': list(X.columns)
        }
    
    # ============================================================
    # Staged Experiments (Workflow Management)
    # ============================================================
    
    def add_staged_experiment(self, inputs: Dict[str, Any]) -> None:
        """
        Add an experiment to the staging area (awaiting evaluation).
        
        Staged experiments are typically suggested by acquisition functions
        but not yet evaluated. They can be retrieved, evaluated externally,
        and then added to the dataset with add_experiment().
        
        Args:
            inputs: Dictionary mapping variable names to values
            
        Example:
            > # Generate suggestions and stage them
            > suggestions = session.suggest_next(n_suggestions=3)
            > for point in suggestions.to_dict('records'):
            >     session.add_staged_experiment(point)
            > 
            > # Later, evaluate and add
            > staged = session.get_staged_experiments()
            > for point in staged:
            >     output = run_experiment(**point)
            >     session.add_experiment(point, output=output)
            > session.clear_staged_experiments()
        """
        self.staged_experiments.append(inputs)
        logger.debug(f"Staged experiment: {inputs}")
        self.events.emit('experiment_staged', {'inputs': inputs})
    
    def get_staged_experiments(self) -> List[Dict[str, Any]]:
        """
        Get all staged experiments awaiting evaluation.
        
        Returns:
            List of experiment input dictionaries
        """
        return self.staged_experiments.copy()
    
    def clear_staged_experiments(self) -> int:
        """
        Clear all staged experiments.
        
        Returns:
            Number of experiments cleared
        """
        count = len(self.staged_experiments)
        self.staged_experiments.clear()
        if count > 0:
            logger.info(f"Cleared {count} staged experiments")
            self.events.emit('staged_experiments_cleared', {'count': count})
        return count
    
    def move_staged_to_experiments(self, outputs: List[float], 
                                   noises: Optional[List[float]] = None,
                                   iteration: Optional[int] = None,
                                   reason: Optional[str] = None) -> int:
        """
        Evaluate staged experiments and add them to the dataset in batch.
        
        Convenience method that pairs staged inputs with outputs and adds
        them all to the experiment manager, then clears the staging area.
        
        Args:
            outputs: List of output values (must match length of staged experiments)
            noises: Optional list of measurement uncertainties
            iteration: Iteration number for all experiments (auto-assigned if None)
            reason: Reason for these experiments (e.g., 'Expected Improvement')
            
        Returns:
            Number of experiments added
            
        Example:
            > # Stage some experiments
            > session.add_staged_experiment({'x': 1.0, 'y': 2.0})
            > session.add_staged_experiment({'x': 3.0, 'y': 4.0})
            > 
            > # Evaluate them
            > outputs = [run_experiment(**point) for point in session.get_staged_experiments()]
            > 
            > # Add to dataset and clear staging
            > session.move_staged_to_experiments(outputs, reason='LogEI')
        """
        if len(outputs) != len(self.staged_experiments):
            raise ValueError(
                f"Number of outputs ({len(outputs)}) must match "
                f"number of staged experiments ({len(self.staged_experiments)})"
            )
        
        if noises is not None and len(noises) != len(self.staged_experiments):
            raise ValueError(
                f"Number of noise values ({len(noises)}) must match "
                f"number of staged experiments ({len(self.staged_experiments)})"
            )
        
        # Add each experiment
        for i, inputs in enumerate(self.staged_experiments):
            noise = noises[i] if noises is not None else None
            
            # Strip any metadata fields (prefixed with _) from inputs
            # These are used for UI/workflow tracking but shouldn't be stored as variables
            clean_inputs = {k: v for k, v in inputs.items() if not k.startswith('_')}
            
            # Use per-experiment reason if stored in _reason, otherwise use batch reason
            exp_reason = inputs.get('_reason', reason)
            
            self.add_experiment(
                inputs=clean_inputs,
                output=outputs[i],
                noise=noise,
                iteration=iteration,
                reason=exp_reason
            )
        
        count = len(self.staged_experiments)
        self.clear_staged_experiments()
        
        logger.info(f"Moved {count} staged experiments to dataset")
        return count
    
    # ============================================================
    # Initial Design Generation
    # ============================================================
    
    def generate_initial_design(
        self,
        method: str = "lhs",
        n_points: Optional[int] = None,
        random_seed: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate initial experimental design (Design of Experiments).

        Creates a set of experimental conditions to evaluate before starting
        Bayesian optimization. This does NOT add the experiments to the session -
        you must evaluate them and add the results using add_experiment().

        Space-filling methods (take n_points as input):
        - 'random': Uniform random sampling
        - 'lhs': Latin Hypercube Sampling (recommended, good space-filling)
        - 'sobol': Sobol quasi-random sequences (low discrepancy)
        - 'halton': Halton sequences
        - 'hammersly': Hammersly sequences (low discrepancy)

        Classical RSM methods (run count determined by design structure):
        - 'full_factorial': All combinations of factor levels
        - 'fractional_factorial': Subset of full factorial using generators
        - 'ccd': Central Composite Design (factorial + axial + center)
        - 'box_behnken': Box-Behnken design (3+ continuous factors)

        Screening methods (run count determined by design structure):
        - 'plackett_burman': Ultra-efficient 2-level screening (continuous only)
        - 'gsd': Generalized Subset Design (mixed categorical/continuous)

        Args:
            method: Sampling strategy to use
            n_points: Number of points (required for space-filling; ignored for classical)
            random_seed: Random seed for reproducibility
            **kwargs: Additional method-specific parameters:
                - lhs_criterion: For LHS method ("maximin", "correlation", "ratio")
                - n_levels: Levels per factor for full factorial (2 or 3)
                - n_center: Center point replicates (classical designs)
                - generators: Fractional factorial generator string
                - ccd_alpha: CCD alpha ("orthogonal" or "rotatable")
                - ccd_face: CCD face ("circumscribed", "inscribed", "faced")
                - gsd_reduction: GSD reduction factor (>=2, larger = fewer runs)

        Returns:
            List of dictionaries with variable names and values (no outputs)

        Example:
            > # Generate initial design
            > points = session.generate_initial_design('lhs', n_points=10)
            >
            > # Run experiments and add results
            > for point in points:
            >     output = run_experiment(**point)  # Your experiment function
            >     session.add_experiment(point, output=output)
            >
            > # Now ready to train model
            > session.train_model()
        """
        if len(self.search_space.variables) == 0:
            raise ValueError(
                "No variables defined in search space. "
                "Use add_variable() to define variables before generating initial design."
            )
        
        from alchemist_core.utils.doe import generate_initial_design
        
        points = generate_initial_design(
            search_space=self.search_space,
            method=method,
            n_points=n_points,
            random_seed=random_seed,
            **kwargs
        )
        
        # Store sampler info in config for audit trail
        self.config['initial_design_method'] = method
        self.config['initial_design_n_points'] = len(points)
        
        logger.info(f"Generated {len(points)} initial design points using {method} method")
        self.events.emit('initial_design_generated', {
            'method': method,
            'n_points': len(points)
        })
        
        # Add a lightweight audit data_locked entry for the initial design metadata
        try:
            extra = {'initial_design_method': method, 'initial_design_n_points': len(points)}
            # Create an empty dataframe snapshot of the planned points
            import pandas as pd
            planned_df = pd.DataFrame(points)
            self.audit_log.lock_data(planned_df, notes=f"Initial design ({method})", extra_parameters=extra)
        except Exception:
            # Audit logging should not block design generation
            logger.debug("Failed to add initial design to audit log")

        return points
    
    # ============================================================
    # Model Training
    # ============================================================
    
    def train_model(self, backend: str = 'sklearn', kernel: str = 'Matern',
                   kernel_params: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """
        Train surrogate model on current data.
        
        Args:
            backend: 'sklearn' or 'botorch'
            kernel: Kernel type ('RBF', 'Matern', 'RationalQuadratic')
            kernel_params: Additional kernel parameters (e.g., {'nu': 2.5} for Matern)
            **kwargs: Backend-specific parameters
        
        Returns:
            Dictionary with training results and hyperparameters
        
        Example:
            > results = session.train_model(backend='botorch', kernel='Matern')
            > print(results['metrics'])
        """
        df = self.experiment_manager.get_data()
        if df is None or df.empty:
            raise ValueError("No experimental data available. Use load_data() or add_experiment() first.")

        # Guard: multi-objective requires BoTorch
        if self.is_multi_objective and backend.lower() == 'sklearn':
            raise ValueError(
                "multi-objective optimization requires backend='botorch'. The sklearn backend "
                "only supports single-objective. Change to: session.train_model(backend='botorch')"
            )

        self.model_backend = backend.lower()
        
        # Normalize kernel name to match expected case
        kernel_name_map = {
            'rbf': 'RBF',
            'matern': 'Matern',
            'rationalquadratic': 'RationalQuadratic',
            'rational_quadratic': 'RationalQuadratic'
        }
        kernel = kernel_name_map.get(kernel.lower(), kernel)
        
        # Extract calibration_enabled before passing kwargs to model constructor
        calibration_enabled = kwargs.pop('calibration_enabled', False)
        
        # Validate and map transform types based on backend
        # BoTorch uses: 'normalize', 'standardize'
        # Sklearn uses: 'minmax', 'standard', 'robust', 'none'
        if self.model_backend == 'sklearn':
            # Map BoTorch transform types to sklearn equivalents
            transform_map = {
                'normalize': 'minmax',      # BoTorch normalize → sklearn minmax
                'standardize': 'standard',  # BoTorch standardize → sklearn standard
                'none': 'none'
            }
            if 'input_transform_type' in kwargs:
                original = kwargs['input_transform_type']
                kwargs['input_transform_type'] = transform_map.get(original, original)
                if original != kwargs['input_transform_type']:
                    logger.debug(f"Mapped input transform '{original}' → '{kwargs['input_transform_type']}' for sklearn")
            if 'output_transform_type' in kwargs:
                original = kwargs['output_transform_type']
                kwargs['output_transform_type'] = transform_map.get(original, original)
                if original != kwargs['output_transform_type']:
                    logger.debug(f"Mapped output transform '{original}' → '{kwargs['output_transform_type']}' for sklearn")
        
        # Import appropriate model class
        if self.model_backend == 'sklearn':
            from alchemist_core.models.sklearn_model import SklearnModel
            
            # Build kernel options
            kernel_options = {'kernel_type': kernel}
            if kernel_params:
                kernel_options.update(kernel_params)
            
            self.model = SklearnModel(
                kernel_options=kernel_options,
                random_state=self.config['random_state'],
                **kwargs
            )
            
        elif self.model_backend == 'botorch':
            from alchemist_core.models.botorch_model import BoTorchModel
            
            # Apply sensible defaults for BoTorch if not explicitly overridden
            # Input normalization and output standardization are critical for performance
            if 'input_transform_type' not in kwargs:
                kwargs['input_transform_type'] = 'normalize'
                logger.debug("Auto-applying input normalization for BoTorch model")
            if 'output_transform_type' not in kwargs:
                kwargs['output_transform_type'] = 'standardize'
                logger.debug("Auto-applying output standardization for BoTorch model")
            
            # Build kernel options - BoTorch uses 'cont_kernel_type' not 'kernel_type'
            kernel_options = {'cont_kernel_type': kernel}
            if kernel_params:
                # Add matern_nu if provided
                if 'nu' in kernel_params:
                    kernel_options['matern_nu'] = kernel_params['nu']
                # Add any other kernel params
                for k, v in kernel_params.items():
                    if k != 'nu':  # Already handled above
                        kernel_options[k] = v
            
            # Identify categorical variable indices for BoTorch
            # Only compute if not already provided in kwargs (e.g., from UI)
            if 'cat_dims' not in kwargs:
                cat_dims = []
                categorical_var_names = self.search_space.get_categorical_variables()
                if categorical_var_names:
                    # Get the column order from search space
                    all_var_names = self.search_space.get_variable_names()
                    cat_dims = [i for i, name in enumerate(all_var_names) if name in categorical_var_names]
                    logger.debug(f"Categorical dimensions for BoTorch: {cat_dims} (variables: {categorical_var_names})")
                kwargs['cat_dims'] = cat_dims if cat_dims else None
            
            self.model = BoTorchModel(
                kernel_options=kernel_options,
                random_state=self.config['random_state'],
                **kwargs
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'sklearn' or 'botorch'")
        
        # Train model
        logger.info(f"Training {backend} model with {kernel} kernel...")
        self.events.emit('training_started', {'backend': backend, 'kernel': kernel})
        
        self.model.train(self.experiment_manager)
        
        # Apply calibration if requested (sklearn only)
        if calibration_enabled and self.model_backend == 'sklearn':
            if hasattr(self.model, '_compute_calibration_factors'):
                self.model._compute_calibration_factors()
                logger.info("Uncertainty calibration enabled")
        
        # Get hyperparameters
        hyperparams = self.model.get_hyperparameters()
        
        # Convert hyperparameters to JSON-serializable format
        # (kernel objects can't be serialized directly)
        json_hyperparams = {}
        for key, value in hyperparams.items():
            if isinstance(value, (int, float, str, bool, type(None))):
                json_hyperparams[key] = value
            elif isinstance(value, np.ndarray):
                json_hyperparams[key] = value.tolist()
            else:
                # Convert complex objects to their string representation
                json_hyperparams[key] = str(value)
        
        # Compute metrics from CV results if available (single-objective only)
        metrics = {}
        if not self.is_multi_objective and hasattr(self.model, 'cv_cached_results') and self.model.cv_cached_results is not None:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            y_true = self.model.cv_cached_results['y_true']
            y_pred = self.model.cv_cached_results['y_pred']

            metrics = {
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'r2': float(r2_score(y_true, y_pred))
            }

        results = {
            'backend': backend,
            'kernel': kernel,
            'hyperparameters': json_hyperparams,
            'metrics': metrics,
            'success': True,
            'n_objectives': self.n_objectives,
        }

        if self.is_multi_objective:
            logger.info(f"multi-objective model trained successfully ({self.n_objectives} objectives)")
        else:
            logger.info(f"Model trained successfully. R²: {metrics.get('r2', 'N/A')}")
        self.events.emit('training_completed', results)
        
        return results
    
    def get_model_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get summary of trained model.
        
        Returns:
            Dictionary with model information, or None if no model trained
        """
        if self.model is None:
            return None
        
        # Compute metrics if available
        metrics = {}
        if hasattr(self.model, 'cv_cached_results') and self.model.cv_cached_results is not None:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            y_true = self.model.cv_cached_results['y_true']
            y_pred = self.model.cv_cached_results['y_pred']
            
            metrics = {
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'r2': float(r2_score(y_true, y_pred))
            }
        
        # Get hyperparameters and make them JSON-serializable
        hyperparams = self.model.get_hyperparameters()
        json_hyperparams = {}
        for key, value in hyperparams.items():
            if isinstance(value, (int, float, str, bool, type(None))):
                json_hyperparams[key] = value
            elif isinstance(value, np.ndarray):
                json_hyperparams[key] = value.tolist()
            else:
                # Convert complex objects to their string representation
                json_hyperparams[key] = str(value)
        
        # Extract kernel name and parameters
        kernel_name = 'unknown'
        if self.model_backend == 'sklearn':
            # First try kernel_options
            if hasattr(self.model, 'kernel_options') and 'kernel_type' in self.model.kernel_options:
                kernel_name = self.model.kernel_options['kernel_type']
                # Add nu parameter for Matern kernels
                if kernel_name == 'Matern' and 'matern_nu' in self.model.kernel_options:
                    json_hyperparams['matern_nu'] = self.model.kernel_options['matern_nu']
            # Then try trained kernel
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'kernel_'):
                kernel_obj = self.model.model.kernel_
                # Navigate through Product/Sum kernels to find base kernel
                if hasattr(kernel_obj, 'k2'):  # Product kernel (Constant * BaseKernel)
                    base_kernel = kernel_obj.k2
                else:
                    base_kernel = kernel_obj
                
                kernel_class = type(base_kernel).__name__
                if 'Matern' in kernel_class:
                    kernel_name = 'Matern'
                    # Extract nu parameter if available
                    if hasattr(base_kernel, 'nu'):
                        json_hyperparams['matern_nu'] = float(base_kernel.nu)
                elif 'RBF' in kernel_class:
                    kernel_name = 'RBF'
                elif 'RationalQuadratic' in kernel_class:
                    kernel_name = 'RationalQuadratic'
                else:
                    kernel_name = kernel_class
        elif self.model_backend == 'botorch':
            if hasattr(self.model, 'cont_kernel_type'):
                kernel_name = self.model.cont_kernel_type
            elif 'kernel_type' in json_hyperparams:
                kernel_name = json_hyperparams['kernel_type']
        
        return {
            'backend': self.model_backend,
            'kernel': kernel_name,
            'hyperparameters': json_hyperparams,
            'metrics': metrics,
            'is_trained': True
        }
    
    # ============================================================
    # Acquisition and Suggestions
    # ============================================================
    
    def suggest_next(self, strategy: str = 'EI', goal: Union[str, List[str]] = 'maximize',
                    n_suggestions: int = 1, ref_point: Optional[List[float]] = None,
                    **kwargs) -> pd.DataFrame:
        """
        Suggest next experiment(s) using acquisition function.

        Args:
            strategy: Acquisition strategy
                - 'EI': Expected Improvement
                - 'PI': Probability of Improvement
                - 'UCB': Upper Confidence Bound
                - 'LogEI': Log Expected Improvement (BoTorch only)
                - 'LogPI': Log Probability of Improvement (BoTorch only)
                - 'qEI', 'qUCB', 'qIPV': Batch acquisition (BoTorch only)
                - 'qEHVI', 'qNEHVI': Multi-objective acquisition (BoTorch only)
            goal: 'maximize' or 'minimize' (str), or list of per-objective directions
            n_suggestions: Number of suggestions (batch acquisition)
            ref_point: Reference point for MOBO hypervolume (list of floats, optional)
            **kwargs: Strategy-specific parameters:

                **Sklearn backend:**
                - xi (float): Exploration parameter for EI/PI (default: 0.01)
                - kappa (float): Exploration parameter for UCB (default: 1.96)

                **BoTorch backend:**
                - beta (float): Exploration parameter for UCB (default: 0.5)
                - mc_samples (int): Monte Carlo samples for batch acquisition (default: 128)

        Returns:
            DataFrame with suggested experiment(s)

        Examples:
            >>> # Single-objective
            >>> next_point = session.suggest_next(strategy='EI', goal='maximize')

            >>> # Multi-objective
            >>> suggestions = session.suggest_next(
            ...     strategy='qNEHVI',
            ...     goal=['maximize', 'maximize'],
            ...     ref_point=[0.0, 0.0]
            ... )
        """
        if self.model is None:
            raise ValueError("No trained model available. Use train_model() first.")

        # Handle multi-objective goal/direction
        if self.is_multi_objective:
            strategy_lower = strategy.lower()
            valid_mobo_strategies = {'qehvi', 'qnehvi'}
            if strategy_lower not in valid_mobo_strategies:
                raise ValueError(
                    f"multi-objective optimization requires 'qEHVI' or 'qNEHVI' acquisition strategy. "
                    f"Got '{strategy}'."
                )

            # Convert goal to directions list
            if isinstance(goal, str):
                directions = [goal.lower()] * self.n_objectives
            else:
                directions = [g.lower() for g in goal]
                if len(directions) != self.n_objectives:
                    raise ValueError(
                        f"goal list length ({len(directions)}) must match number of objectives ({self.n_objectives})"
                    )

            # Build outcome constraint callables from stored constraints
            outcome_constraint_callables = None
            if self._outcome_constraints:
                outcome_constraint_callables = []
                obj_names = self.objective_names
                for oc in self._outcome_constraints:
                    obj_idx = obj_names.index(oc['objective_name'])
                    threshold = oc['value']
                    if oc['bound_type'] == 'lower':
                        # feasible when Y >= threshold, i.e. threshold - Y <= 0
                        outcome_constraint_callables.append(
                            lambda Y, t=threshold, i=obj_idx: t - Y[..., i]
                        )
                    else:
                        # feasible when Y <= threshold, i.e. Y - threshold <= 0
                        outcome_constraint_callables.append(
                            lambda Y, t=threshold, i=obj_idx: Y[..., i] - t
                        )

        # Validate and log kwargs
        supported_kwargs = self._get_supported_kwargs(strategy, self.model_backend)
        if kwargs:
            unsupported = set(kwargs.keys()) - supported_kwargs
            if unsupported:
                logger.warning(
                    f"Unsupported parameters for {strategy} with {self.model_backend} backend: "
                    f"{unsupported}. Supported parameters: {supported_kwargs or 'none'}"
                )
            used_kwargs = {k: v for k, v in kwargs.items() if k in supported_kwargs}
            if used_kwargs:
                logger.info(f"Using acquisition parameters: {used_kwargs}")

        # Import appropriate acquisition class
        if self.model_backend == 'sklearn':
            from alchemist_core.acquisition.skopt_acquisition import SkoptAcquisition

            self.acquisition = SkoptAcquisition(
                search_space=self.search_space.to_skopt(),
                model=self.model,
                acq_func=strategy.lower(),
                maximize=(goal.lower() == 'maximize') if isinstance(goal, str) else True,
                random_state=self.config['random_state'],
                acq_func_kwargs=kwargs
            )

            # Update acquisition with existing experimental data (un-encoded)
            X, y = self.experiment_manager.get_features_and_target()
            self.acquisition.update(X, y)

        elif self.model_backend == 'botorch':
            from alchemist_core.acquisition.botorch_acquisition import BoTorchAcquisition

            acq_kwargs = dict(
                model=self.model,
                search_space=self.search_space,
                acq_func=strategy,
                maximize=(goal.lower() == 'maximize') if isinstance(goal, str) else True,
                batch_size=n_suggestions,
                acq_func_kwargs=kwargs,
            )

            # Add MOBO-specific parameters
            if self.is_multi_objective:
                # Validate ref_point if provided
                if ref_point is not None and len(ref_point) != self.n_objectives:
                    raise ValueError(
                        f"ref_point length ({len(ref_point)}) must match number of objectives ({self.n_objectives})"
                    )
                
                acq_kwargs['ref_point'] = ref_point
                acq_kwargs['directions'] = directions
                acq_kwargs['objective_names'] = self.objective_names
                acq_kwargs['outcome_constraints'] = outcome_constraint_callables if self._outcome_constraints else None

            self.acquisition = BoTorchAcquisition(**acq_kwargs)
        
        # Check if this is a pure exploration acquisition (doesn't use best_f)
        is_exploratory = strategy.lower() in ['qnipv', 'qipv']
        goal_desc = 'pure exploration' if is_exploratory else goal
        logger.info(f"Running acquisition: {strategy} ({goal_desc})")
        self.events.emit('acquisition_started', {'strategy': strategy, 'goal': goal})
        
        # Get suggestion
        next_point = self.acquisition.select_next()
        
        # Robustly handle output type and convert to DataFrame
        if isinstance(next_point, pd.DataFrame):
            suggestion_dict = next_point.to_dict('records')[0]
            result_df = next_point
        elif isinstance(next_point, list):
            # Get variable names from search space
            var_names = [var['name'] for var in self.search_space.variables]
            
            # Check if it's a list of dicts or a list of values
            if len(next_point) > 0 and isinstance(next_point[0], dict):
                # List of dicts
                result_df = pd.DataFrame(next_point)
                suggestion_dict = next_point[0]
            else:
                # List of values - create dict with variable names
                suggestion_dict = dict(zip(var_names, next_point))
                result_df = pd.DataFrame([suggestion_dict])
        else:
            # Fallback: wrap in DataFrame
            result_df = pd.DataFrame([next_point])
            suggestion_dict = result_df.to_dict('records')[0]
        
        logger.info(f"Suggested point: {suggestion_dict}")
        self.events.emit('acquisition_completed', {'suggestion': suggestion_dict})
        
        # Store suggestions for UI/API access
        self.last_suggestions = result_df.to_dict('records')
        
        # Cache suggestion info for audit log and visualization
        self._last_acquisition_info = {
            'strategy': strategy,
            'goal': goal,
            'parameters': kwargs
        }
        self._last_acq_func = strategy.lower()
        self._last_goal = goal.lower() if isinstance(goal, str) else [g.lower() for g in goal]
        
        return result_df
    
    def _get_supported_kwargs(self, strategy: str, backend: str) -> set:
        """
        Return supported kwargs for given acquisition strategy and backend.
        
        Args:
            strategy: Acquisition strategy name
            backend: Model backend ('sklearn' or 'botorch')
            
        Returns:
            Set of supported kwarg names
        """
        strategy_lower = strategy.lower()
        
        if backend == 'sklearn':
            if strategy_lower in ['ei', 'pi', 'expectedimprovement', 'probabilityofimprovement']:
                return {'xi'}
            elif strategy_lower in ['ucb', 'lcb', 'upperconfidencebound', 'lowerconfidencebound']:
                return {'kappa'}
            elif strategy_lower == 'gp_hedge':
                return {'xi', 'kappa'}
        elif backend == 'botorch':
            if strategy_lower in ['ei', 'logei', 'pi', 'logpi', 'expectedimprovement', 'probabilityofimprovement']:
                return set()  # No additional parameters for these
            elif strategy_lower in ['ucb', 'upperconfidencebound']:
                return {'beta'}
            elif strategy_lower in ['qei', 'qucb']:
                return {'mc_samples', 'beta'}
            elif strategy_lower in ['qipv', 'qnipv']:
                return {'mc_samples', 'n_mc_points'}
            elif strategy_lower in ['qehvi', 'qnehvi']:
                return {'mc_samples', 'ref_point', 'eta'}

        return set()
    
    def find_optimum(self, goal: Union[str, List[str]] = 'maximize',
                     n_grid_points: int = 10000) -> Dict[str, Any]:
        """
        Find the point where the model predicts the optimal value.

        For single-objective: returns the predicted optimum via grid search.
        For multi-objective: returns the predicted Pareto frontier from observed data.

        Args:
            goal: 'maximize' or 'minimize' (str), or list of per-objective directions
            n_grid_points: Target number of grid points for search (default: 10000)

        Returns:
            Single-objective dict with 'x_opt', 'value', 'std'.
            Multi-objective dict with 'pareto_frontier', 'predicted_values',
            'objective_names', 'n_pareto'.
        """
        if self.model is None:
            raise ValueError("No trained model available. Use train_model() first.")

        # Multi-objective: return Pareto frontier
        if self.is_multi_objective:
            if isinstance(goal, str):
                directions = [goal.lower()] * self.n_objectives
            else:
                directions = [g.lower() for g in goal]
                # Validate directions length
                if len(directions) != self.n_objectives:
                    raise ValueError(
                        f"goal list length ({len(directions)}) must match number of objectives ({self.n_objectives})"
                    )

            pareto_df = self.experiment_manager.get_pareto_frontier(directions)
            if len(pareto_df) == 0:
                return {
                    'pareto_frontier': pd.DataFrame(),
                    'predicted_values': np.array([]),
                    'objective_names': self.objective_names,
                    'n_pareto': 0,
                }

            # Get predicted values for Pareto points
            feature_cols = [c for c in pareto_df.columns
                          if c not in self.objective_names + ['Noise', 'Iteration', 'Reason']]
            X_pareto = pareto_df[feature_cols]
            pred_results = self.model.predict(X_pareto, return_std=True)

            # pred_results is a dict[str, (mean, std)] for multi-objective
            predicted_values = np.column_stack([pred_results[name][0] for name in self.objective_names])

            result = {
                'pareto_frontier': pareto_df,
                'predicted_values': predicted_values,
                'objective_names': self.objective_names,
                'n_pareto': len(pareto_df),
            }
            logger.info(f"Found {result['n_pareto']} Pareto-optimal points")
            return result

        # Single-objective
        grid = self._generate_prediction_grid(n_grid_points)
        means, stds = self.predict(grid)

        if isinstance(goal, list):
            goal = goal[0]
        if goal.lower() == 'maximize':
            best_idx = np.argmax(means)
        else:
            best_idx = np.argmin(means)

        opt_point_df = grid.iloc[[best_idx]].reset_index(drop=True)

        result = {
            'x_opt': opt_point_df,
            'value': float(means[best_idx]),
            'std': float(stds[best_idx])
        }

        logger.info(f"Found optimum: {result['x_opt'].to_dict('records')[0]}")
        logger.info(f"Predicted value: {result['value']:.4f} ± {result['std']:.4f}")

        return result
    
    # ============================================================
    # Predictions
    # ============================================================
    
    def predict(self, inputs: pd.DataFrame) -> Union[Tuple[np.ndarray, np.ndarray], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        Make predictions at new points.

        Args:
            inputs: DataFrame with input features

        Returns:
            Single-objective: Tuple of (predictions, uncertainties)
            Multi-objective: dict[str, tuple[ndarray, ndarray]] keyed by objective name
        """
        if self.model is None:
            raise ValueError("No trained model available. Use train_model() first.")

        # Multi-objective returns dict from model
        if self.is_multi_objective:
            return self.model.predict(inputs, return_std=True)

        # Single-objective
        if self.model_backend == 'sklearn':
            return self.model.predict(inputs, return_std=True)
        elif self.model_backend == 'botorch':
            return self.model.predict(inputs, return_std=True)
        else:
            try:
                return self.model.predict(inputs, return_std=True)
            except TypeError:
                preds = self.model.predict(inputs)
                return preds, np.zeros_like(preds)
    
    # ============================================================
    # Event Handling
    # ============================================================
    
    def on(self, event: str, callback: Callable) -> None:
        """
        Register event listener.
        
        Args:
            event: Event name
            callback: Callback function
        
        Example:
            > def on_training_done(data):
            ...     print(f"Training completed with R² = {data['metrics']['r2']}")
            > session.on('training_completed', on_training_done)
        """
        self.events.on(event, callback)
    
    # ============================================================
    # Configuration
    # ============================================================
    
    def set_config(self, **kwargs) -> None:
        """
        Update session configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        
        Example:
            > session.set_config(random_state=123, verbose=False)
        """
        self.config.update(kwargs)
        logger.info(f"Updated config: {kwargs}")
    
    # ============================================================
    # Audit Log & Session Management
    # ============================================================
    
    def lock_data(self, notes: str = "", extra_parameters: Optional[Dict[str, Any]] = None) -> AuditEntry:
        """
        Lock in current experimental data configuration.
        
        Creates an immutable audit log entry capturing the current data state.
        This should be called when you're satisfied with your experimental dataset
        and ready to proceed with modeling.
        
        Args:
            notes: Optional user notes about this data configuration
            
        Returns:
            Created AuditEntry
            
        Example:
            > session.add_experiment({'temp': 100, 'pressure': 5}, output=85.2)
            > session.lock_data(notes="Initial screening dataset")
        """
        # Set search space in audit log (once)
        if self.audit_log.search_space_definition is None:
            self.audit_log.set_search_space(self.search_space.variables)
        
        # Get current experimental data
        df = self.experiment_manager.get_data()
        
        # Lock data in audit log
        entry = self.audit_log.lock_data(
            experiment_data=df,
            notes=notes,
            extra_parameters=extra_parameters
        )
        
        self.metadata.update_modified()
        logger.info(f"Locked data: {len(df)} experiments")
        self.events.emit('data_locked', {'entry': entry.to_dict()})
        
        return entry
    
    def lock_model(self, notes: str = "") -> AuditEntry:
        """
        Lock in current trained model configuration.
        
        Creates an immutable audit log entry capturing the trained model state.
        This should be called when you're satisfied with your model performance
        and ready to use it for acquisition.
        
        Args:
            notes: Optional user notes about this model
            
        Returns:
            Created AuditEntry
            
        Raises:
            ValueError: If no model has been trained
            
        Example:
            > session.train_model(backend='sklearn', kernel='matern')
            > session.lock_model(notes="Best cross-validation performance")
        """
        if self.model is None:
            raise ValueError("No trained model available. Use train_model() first.")
        
        # Set search space in audit log (once)
        if self.audit_log.search_space_definition is None:
            self.audit_log.set_search_space(self.search_space.variables)
        
        # Get model info
        model_info = self.get_model_summary()
        
        # Extract hyperparameters
        hyperparameters = model_info.get('hyperparameters', {})
        
        # Get kernel name from model_info (which extracts it properly)
        kernel_name = model_info.get('kernel', 'unknown')
        
        # Get CV metrics if available - use model_info metrics which are already populated
        cv_metrics = model_info.get('metrics', None)
        if cv_metrics and all(k in cv_metrics for k in ['rmse', 'r2']):
            # Metrics already in correct format from get_model_summary
            pass
        elif hasattr(self.model, 'cv_cached_results') and self.model.cv_cached_results:
            # Fallback to direct access
            cv_metrics = {
                'rmse': float(self.model.cv_cached_results.get('rmse', 0)),
                'r2': float(self.model.cv_cached_results.get('r2', 0)),
                'mae': float(self.model.cv_cached_results.get('mae', 0))
            }
        else:
            cv_metrics = None
        
        # Get current iteration number
        # Use the next iteration number for the model lock so model+acquisition share the same iteration
        iteration = self.experiment_manager._current_iteration + 1
        
        # Include scaler information if available in hyperparameters
        try:
            if hasattr(self.model, 'input_transform_type'):
                hyperparameters['input_transform_type'] = self.model.input_transform_type
            if hasattr(self.model, 'output_transform_type'):
                hyperparameters['output_transform_type'] = self.model.output_transform_type
        except Exception:
            pass

        # Try to extract Matern nu for sklearn models if not already present
        try:
            if self.model_backend == 'sklearn' and 'matern_nu' not in hyperparameters:
                # Try to navigate fitted kernel object for sklearn GaussianProcessRegressor
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'kernel_'):
                    kernel_obj = self.model.model.kernel_
                    base_kernel = getattr(kernel_obj, 'k2', kernel_obj)
                    if hasattr(base_kernel, 'nu'):
                        hyperparameters['matern_nu'] = float(base_kernel.nu)
        except Exception:
            pass

        entry = self.audit_log.lock_model(
            backend=self.model_backend,
            kernel=kernel_name,
            hyperparameters=hyperparameters,
            cv_metrics=cv_metrics,
            iteration=iteration,
            notes=notes
        )
        
        self.metadata.update_modified()
        logger.info(f"Locked model: {self.model_backend}/{model_info.get('kernel')}, iteration {iteration}")
        self.events.emit('model_locked', {'entry': entry.to_dict()})
        
        return entry
    
    def lock_acquisition(self, strategy: str, parameters: Dict[str, Any],
                        suggestions: List[Dict[str, Any]], notes: str = "") -> AuditEntry:
        """
        Lock in acquisition function decision and suggested experiments.
        
        Creates an immutable audit log entry capturing the acquisition decision.
        This should be called when you've reviewed the suggestions and are ready
        to run the recommended experiments.
        
        Args:
            strategy: Acquisition strategy name ('EI', 'PI', 'UCB', etc.)
            parameters: Acquisition function parameters (xi, kappa, etc.)
            suggestions: List of suggested experiment dictionaries
            notes: Optional user notes about this decision
            
        Returns:
            Created AuditEntry
            
        Example:
            > suggestions = session.suggest_next(strategy='EI', n_suggestions=3)
            > session.lock_acquisition(
            ...     strategy='EI',
            ...     parameters={'xi': 0.01, 'goal': 'maximize'},
            ...     suggestions=suggestions,
            ...     notes="Top 3 candidates for next batch"
            ... )
        """
        # Set search space in audit log (once)
        if self.audit_log.search_space_definition is None:
            self.audit_log.set_search_space(self.search_space.variables)
        
        # Increment iteration counter first so this acquisition is logged as the next iteration
        self.experiment_manager._current_iteration += 1
        iteration = self.experiment_manager._current_iteration

        entry = self.audit_log.lock_acquisition(
            strategy=strategy,
            parameters=parameters,
            suggestions=suggestions,
            iteration=iteration,
            notes=notes
        )
        
        self.metadata.update_modified()
        logger.info(f"Locked acquisition: {strategy}, {len(suggestions)} suggestions")
        self.events.emit('acquisition_locked', {'entry': entry.to_dict()})
        
        return entry
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """
        Get complete audit log as list of dictionaries.
        
        Returns:
            List of audit entry dictionaries
        """
        return self.audit_log.to_dict()
    
    def export_audit_markdown(self) -> str:
        """
        Export audit log as markdown for publications.
        
        Returns:
            Markdown-formatted audit trail
        """
        # Pass session metadata to markdown exporter so user-entered metadata appears
        try:
            metadata_dict = self.metadata.to_dict()
        except Exception:
            metadata_dict = None

        return self.audit_log.to_markdown(session_metadata=metadata_dict)
    
    def save_session(self, filepath: str):
        """
        Save complete session state to JSON file.
        
        Saves all session data including:
        - Session metadata (name, description, tags)
        - Search space definition
        - Experimental data
        - Trained model state (if available)
        - Complete audit log
        
        Args:
            filepath: Path to save session file (.json extension recommended)
            
        Example:
            > session.save_session("~/ALchemist_Sessions/catalyst_study_nov2025.json")
        """
        filepath = Path(filepath)
        
        # Update audit log's experimental data snapshot to reflect current state
        # This ensures the data table in the audit log markdown is always up-to-date
        current_data = self.experiment_manager.get_data()
        if current_data is not None and len(current_data) > 0:
            self.audit_log.experiment_data = current_data.copy()
        
        # Prepare session data
        session_data = {
            'version': '1.0.0',
            'metadata': self.metadata.to_dict(),
            'audit_log': self.audit_log.to_dict(),
            'search_space': {
                'variables': self.search_space.variables
            },
            'experiments': {
                'data': self.experiment_manager.get_data().to_dict(orient='records'),
                'n_total': len(self.experiment_manager.df)
            },
            'config': self.config
        }
        
        # Add model state if available
        if self.model is not None:
            model_info = self.get_model_summary()
            
            # Get kernel name from model_info which properly extracts it
            kernel_name = model_info.get('kernel', 'unknown')
            
            # Extract kernel parameters if available
            kernel_params = {}
            if self.model_backend == 'sklearn' and hasattr(self.model, 'model'):
                kernel_obj = self.model.model.kernel
                # Extract kernel-specific parameters
                if hasattr(kernel_obj, 'get_params'):
                    kernel_params = kernel_obj.get_params()
            elif self.model_backend == 'botorch':
                # For BoTorch, parameters are in hyperparameters
                hyperparams = model_info.get('hyperparameters', {})
                if 'matern_nu' in hyperparams:
                    kernel_params['nu'] = hyperparams['matern_nu']
            
            session_data['model_config'] = {
                'backend': self.model_backend,
                'kernel': kernel_name,
                'kernel_params': kernel_params,
                'hyperparameters': model_info.get('hyperparameters', {}),
                'metrics': model_info.get('metrics', {})
            }
        
        # Create directory if needed
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Write JSON
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        self.metadata.update_modified()
        logger.info(f"Saved session to {filepath}")
        self.events.emit('session_saved', {'filepath': str(filepath)})

    def export_session_json(self) -> str:
        """
        Export current session state as a JSON string (no filesystem side-effects for caller).

        Returns:
            JSON string of session data
        """
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name
            # Use existing save_session logic to write a complete JSON
            self.save_session(tmp_path)

        try:
            with open(tmp_path, 'r') as f:
                content = f.read()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return content
    
    def load_session(self, filepath: str = None, retrain_on_load: bool = True) -> 'OptimizationSession':
        """
        Load session from JSON file.
        
        This method works both as a static method (creating a new session) and as an
        instance method (loading into existing session):
        
        Static usage (returns new session):
            > session = OptimizationSession.load_session("my_session.json")
        
        Instance usage (loads into existing session):
            > session = OptimizationSession()
            > session.load_session("my_session.json")
            > # session.experiment_manager.df is now populated
        
        Args:
            filepath: Path to session file (required when called as static method,
                     can be self when called as instance method)
            retrain_on_load: Whether to retrain model if config exists (default: True)
            
        Returns:
            OptimizationSession (new or modified instance)
        """
        # Detect if called as instance method or static method
        # When called as static method: self is actually the filepath string
        # When called as instance method: self is an OptimizationSession instance
        if isinstance(self, OptimizationSession):
            # Instance method: load into this session
            if filepath is None:
                raise ValueError("filepath is required when calling as instance method")
            
            # Load from static implementation
            loaded_session = OptimizationSession._load_session_impl(filepath, retrain_on_load)
            
            # Copy all attributes from loaded session to this instance
            self.search_space = loaded_session.search_space
            self.experiment_manager = loaded_session.experiment_manager
            self.metadata = loaded_session.metadata
            self.audit_log = loaded_session.audit_log
            self.config = loaded_session.config
            self.model = loaded_session.model
            self.model_backend = loaded_session.model_backend
            self.acquisition = loaded_session.acquisition
            self.staged_experiments = loaded_session.staged_experiments
            self.last_suggestions = loaded_session.last_suggestions
            
            # Don't copy events emitter - keep the original
            logger.info(f"Loaded session data into current instance from {filepath}")
            self.events.emit('session_loaded', {'filepath': str(filepath)})
            
            return self
        else:
            # Static method: self is actually the filepath, retrain_on_load is in filepath param
            actual_filepath = self
            actual_retrain = filepath if filepath is not None else True
            return OptimizationSession._load_session_impl(actual_filepath, actual_retrain)
    
    @staticmethod
    def _load_session_impl(filepath: str, retrain_on_load: bool = True) -> 'OptimizationSession':
        """
        Internal implementation for loading session from file.
        This always creates and returns a new session.
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            session_data = json.load(f)
        
        # Check version compatibility
        version = session_data.get('version', '1.0.0')
        if not version.startswith('1.'):
            logger.warning(f"Session file version {version} may not be fully compatible")
        
        # Create session
        session = OptimizationSession()
        
        # Restore metadata
        if 'metadata' in session_data:
            session.metadata = SessionMetadata.from_dict(session_data['metadata'])
        
        # Restore audit log
        if 'audit_log' in session_data:
            session.audit_log.from_dict(session_data['audit_log'])
        
        # Restore search space
        if 'search_space' in session_data:
            for var in session_data['search_space']['variables']:
                session.search_space.add_variable(
                    var['name'],
                    var['type'],
                    **{k: v for k, v in var.items() if k not in ['name', 'type']}
                )
        
        # Restore experimental data
        if 'experiments' in session_data and session_data['experiments']['data']:
            df = pd.DataFrame(session_data['experiments']['data'])
            
            # Metadata columns to exclude from inputs
            metadata_cols = {'Output', 'Noise', 'Iteration', 'Reason'}
            
            # Add experiments one by one
            for _, row in df.iterrows():
                # Only include actual input variables, not metadata
                inputs = {col: row[col] for col in df.columns if col not in metadata_cols}
                output = row.get('Output')
                noise = row.get('Noise') if pd.notna(row.get('Noise')) else None
                iteration = row.get('Iteration') if pd.notna(row.get('Iteration')) else None
                reason = row.get('Reason') if pd.notna(row.get('Reason')) else None
                
                session.add_experiment(inputs, output, noise=noise, iteration=iteration, reason=reason)
        
        # Restore config
        if 'config' in session_data:
            session.config.update(session_data['config'])
        
        # Auto-retrain model if configuration exists (optional)
        if 'model_config' in session_data and retrain_on_load:
            model_config = session_data['model_config']
            logger.info(f"Auto-retraining model: {model_config['backend']} with {model_config.get('kernel', 'default')} kernel")
            
            try:
                # Trigger model training with saved configuration
                session.train_model(
                    backend=model_config['backend'],
                    kernel=model_config.get('kernel', 'Matern'),
                    kernel_params=model_config.get('kernel_params', {})
                )
                logger.info("Model retrained successfully")
                session.events.emit('model_retrained', {'backend': model_config['backend']})
            except Exception as e:
                logger.warning(f"Failed to retrain model: {e}")
                session.events.emit('model_retrain_failed', {'error': str(e)})
        
        logger.info(f"Loaded session from {filepath}")
        session.events.emit('session_loaded', {'filepath': str(filepath)})
        
        return session
    
    def update_metadata(self, name: Optional[str] = None, 
                       description: Optional[str] = None,
                       tags: Optional[List[str]] = None,
                       author: Optional[str] = None):
        """
        Update session metadata.
        
        Args:
            name: New session name (optional)
            description: New description (optional)
            tags: New tags (optional)
            
        Example:
            > session.update_metadata(
            ...     name="Catalyst Screening - Final",
            ...     description="Optimized Pt/Pd ratios",
            ...     tags=["catalyst", "platinum", "palladium", "final"]
            ... )
        """
        if name is not None:
            self.metadata.name = name
        if description is not None:
            self.metadata.description = description
        if author is not None:
            # Backwards compatible: store author if provided
            setattr(self.metadata, 'author', author)
        if tags is not None:
            self.metadata.tags = tags
        
        self.metadata.update_modified()
        logger.info("Updated session metadata")
        self.events.emit('metadata_updated', self.metadata.to_dict())
    
    # ============================================================
    # Legacy Configuration
    # ============================================================
    
    def set_config(self, **kwargs) -> None:
        """
        Update session configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        
        Example:
            > session.set_config(random_state=123, verbose=False)
        """
        self.config.update(kwargs)
        logger.info(f"Updated config: {kwargs}")
    
    # ============================================================
    # Visualization Methods (Notebook Support)
    # ============================================================
    
    # ============================================================
    # MOBO-aware plotting helpers
    # ============================================================

    def _resolve_target_column(self, target_columns: Optional[str]) -> Union[str, List[str]]:
        """Validate & resolve target_columns for MOBO-aware plot methods.

        Args:
            target_columns: Objective name, 'all', or None.

        Returns:
            Single objective name (str) or list of all objective names.
        """
        if not self.is_multi_objective:
            return self.experiment_manager.target_columns[0]
        if target_columns is None:
            raise ValueError(
                f"multi-objective session requires target_columns parameter. "
                f"Use one of {self.objective_names} or 'all'."
            )
        if target_columns == 'all':
            return list(self.objective_names)
        if target_columns not in self.objective_names:
            raise ValueError(
                f"Unknown objective '{target_columns}'. "
                f"Available objectives: {self.objective_names}"
            )
        return target_columns

    def _get_predictions_for_objective(self, predict_result, target_columns: str) -> Tuple[np.ndarray, np.ndarray]:
        """Extract single-objective (predictions, std) from predict() result.

        Args:
            predict_result: Output of self.predict() — either (mean, std) tuple
                or dict[str, (mean, std)] for MOBO.
            target_columns: Objective name to extract.

        Returns:
            Tuple of (predictions, std) ndarrays.
        """
        if isinstance(predict_result, dict):
            return predict_result[target_columns]
        return predict_result

    def _build_grid_df(self, grid_data: dict) -> 'pd.DataFrame':
        """Build a grid DataFrame with columns matching the model's training order.

        Any columns in ``original_feature_names`` that are not present in
        *grid_data* are filled with the median value from the experiment data
        (or 0 if the column is unavailable).  
        
        This prevents NaN columns when the training data contained extra feature 
        columns beyond the search space (e.g., derived features or auxiliary data).
        
        WARNING: If your search space is missing variables that were in the training data,
        predictions will use median values for those missing features, which may not be 
        appropriate for all use cases.
        """
        if hasattr(self.model, 'original_feature_names') and self.model.original_feature_names:
            column_order = self.model.original_feature_names
            for col in column_order:
                if col not in grid_data:
                    if col in self.experiment_manager.df.columns:
                        grid_data[col] = self.experiment_manager.df[col].median()
                    else:
                        grid_data[col] = 0
        else:
            column_order = self.search_space.get_variable_names()
        return pd.DataFrame(grid_data, columns=column_order)

    def _check_matplotlib(self) -> None:
        """Check if matplotlib is available for plotting."""
        if _HAS_VISUALIZATION:
            check_matplotlib()  # Use visualization module's check
        elif not _HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for visualization methods. "
                "Install with: pip install matplotlib"
            )
    
    def _check_model_trained(self) -> None:
        """Check if model is trained before plotting."""
        if self.model is None:
            raise ValueError(
                "Model not trained. Call train_model() before creating visualizations."
            )
    
    def _check_cv_results(self, use_calibrated: bool = False,
                          target_columns: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Get CV results from model, handling both calibrated and uncalibrated.

        Args:
            use_calibrated: Whether to use calibrated results if available
            target_columns: For MOBO, which objective's CV results to return.
                          If None in MOBO, raises ValueError.

        Returns:
            Dictionary with y_true, y_pred, y_std arrays
        """
        self._check_model_trained()

        # For multi-objective, use per-objective CV results
        if self.is_multi_objective:
            if not hasattr(self.model, 'cv_cached_results_multi') or not self.model.cv_cached_results_multi:
                raise ValueError(
                    "No per-objective CV results available for multi-objective model."
                )
            if target_columns is None:
                raise ValueError(
                    f"multi-objective session requires target_columns for CV-based plots. "
                    f"Use one of {self.objective_names}."
                )
            if target_columns not in self.model.cv_cached_results_multi:
                raise ValueError(
                    f"No CV results for objective '{target_columns}'. "
                    f"Available objectives: {list(self.model.cv_cached_results_multi.keys())}"
                )
            return self.model.cv_cached_results_multi[target_columns]

        # Check for calibrated results first if requested
        if use_calibrated and hasattr(self.model, 'cv_cached_results_calibrated'):
            if self.model.cv_cached_results_calibrated is not None:
                return self.model.cv_cached_results_calibrated

        # Fall back to uncalibrated results
        if hasattr(self.model, 'cv_cached_results'):
            if self.model.cv_cached_results is not None:
                return self.model.cv_cached_results

        raise ValueError(
            "No CV results available. Model must be trained with cross-validation."
        )
    
    def plot_parity(
        self,
        use_calibrated: bool = False,
        sigma_multiplier: float = 1.96,
        figsize: Tuple[float, float] = (8, 6),
        dpi: int = 100,
        title: Optional[str] = None,
        show_metrics: bool = True,
        show_error_bars: bool = True,
        target_columns: Optional[str] = None,
        ax: Optional['Axes'] = None
    ) -> Figure: # pyright: ignore[reportInvalidTypeForm]
        """
        Create parity plot of actual vs predicted values from cross-validation.

        This plot shows how well the model's predictions match the actual experimental
        values, with optional error bars indicating prediction uncertainty.

        Args:
            use_calibrated: Use calibrated uncertainty estimates if available
            sigma_multiplier: Error bar size (1.96 = 95% CI, 1.0 = 68% CI, 2.58 = 99% CI)
            figsize: Figure size as (width, height) in inches
            dpi: Dots per inch for figure resolution
            title: Custom title (default: auto-generated with metrics)
            show_metrics: Include RMSE, MAE, R² in title
            show_error_bars: Display uncertainty error bars
            target_columns: For multi-objective: objective name, 'all', or None.
                Single-objective sessions ignore this parameter.
            ax: Existing matplotlib Axes to draw on (creates new figure if None).
                Cannot be used with multi-objective 'all' mode.

        Returns:
            matplotlib Figure object (displays inline in Jupyter)

        Example:
            >>> fig = session.plot_parity()
            >>> fig.savefig('parity.png', bbox_inches='tight')

        Note:
            Requires model to be trained with cross-validation (default behavior).
            Error bars are only shown if model provides uncertainty estimates.
        """
        self._check_matplotlib()
        self._check_model_trained()

        resolved = self._resolve_target_column(target_columns)

        # Multi-objective 'all' → subplot grid
        if isinstance(resolved, list):
            if ax is not None:
                raise ValueError(
                    "Cannot use ax= with multi-objective 'all' mode. "
                    "Specify a single target_columns name instead."
                )
            n = len(resolved)
            fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n, figsize[1]), dpi=dpi)
            if n == 1:
                axes = [axes]
            for i, obj in enumerate(resolved):
                cv_results = self._check_cv_results(use_calibrated, target_columns=obj)
                create_parity_plot(
                    y_true=cv_results['y_true'],
                    y_pred=cv_results['y_pred'],
                    y_std=cv_results.get('y_std'),
                    sigma_multiplier=sigma_multiplier,
                    show_metrics=show_metrics,
                    show_error_bars=show_error_bars,
                    title=title or f"Parity: {obj}",
                    ax=axes[i]
                )
            fig.tight_layout()
            logger.info(f"Generated multi-objective parity plot ({n} objectives)")
            return fig

        # Single objective (or specific MOBO objective)
        cv_results = self._check_cv_results(
            use_calibrated,
            target_columns=resolved if self.is_multi_objective else None
        )
        y_true = cv_results['y_true']
        y_pred = cv_results['y_pred']
        y_std = cv_results.get('y_std', None)

        obj_title = title
        if obj_title is None and self.is_multi_objective:
            obj_title = f"Parity: {resolved}"

        fig, plot_ax = create_parity_plot(
            y_true=y_true,
            y_pred=y_pred,
            y_std=y_std,
            sigma_multiplier=sigma_multiplier,
            figsize=figsize,
            dpi=dpi,
            title=obj_title,
            show_metrics=show_metrics,
            show_error_bars=show_error_bars,
            ax=ax
        )

        logger.info("Generated parity plot")
        return fig
    
    def plot_slice(
        self,
        x_var: str,
        fixed_values: Optional[Dict[str, Any]] = None,
        n_points: int = 100,
        show_uncertainty: Union[bool, List[float]] = True,
        show_experiments: bool = True,
        figsize: Tuple[float, float] = (8, 6),
        dpi: int = 100,
        title: Optional[str] = None,
        target_columns: Optional[str] = None
    ) -> Figure: # pyright: ignore[reportInvalidTypeForm]
        """
        Create 1D slice plot showing model predictions along one variable.

        Args:
            x_var: Variable name to vary along X axis (must be 'real' or 'integer')
            fixed_values: Dict of {var_name: value} for other variables.
            n_points: Number of points to evaluate along the slice
            show_uncertainty: Show uncertainty bands (True, False, or list of sigma values)
            show_experiments: Plot experimental data points as scatter
            figsize: Figure size as (width, height) in inches
            dpi: Dots per inch for figure resolution
            title: Custom title (default: auto-generated)
            target_columns: For multi-objective: objective name, 'all', or None.

        Returns:
            matplotlib Figure object
        """
        self._check_matplotlib()
        self._check_model_trained()

        if fixed_values is None:
            fixed_values = {}

        # Get variable info
        var_names = self.search_space.get_variable_names()
        if x_var not in var_names:
            raise ValueError(f"Variable '{x_var}' not in search space")

        x_var_def = next(v for v in self.search_space.variables if v['name'] == x_var)
        if x_var_def['type'] not in ['real', 'integer']:
            raise ValueError(f"Variable '{x_var}' must be 'real' or 'integer' type for slice plot")

        # Create range for x variable
        x_values = np.linspace(x_var_def['min'], x_var_def['max'], n_points)

        # Build prediction grid
        slice_data = {x_var: x_values}
        for var in self.search_space.variables:
            var_name = var['name']
            if var_name == x_var:
                continue
            if var_name in fixed_values:
                slice_data[var_name] = fixed_values[var_name]
            else:
                if var['type'] in ['real', 'integer']:
                    slice_data[var_name] = (var['min'] + var['max']) / 2
                elif var['type'] == 'categorical':
                    slice_data[var_name] = var['values'][0]

        slice_df = self._build_grid_df(slice_data)

        # Get predictions
        predict_result = self.predict(slice_df)

        # Handle show_uncertainty conversion
        sigma_bands = None
        if show_uncertainty is not False:
            if isinstance(show_uncertainty, bool):
                sigma_bands = [1.0, 2.0] if show_uncertainty else None
            else:
                sigma_bands = show_uncertainty

        resolved = self._resolve_target_column(target_columns)

        def _get_exp_data(obj_name):
            if not show_experiments or len(self.experiment_manager.df) == 0:
                return None, None
            df = self.experiment_manager.df
            mask = pd.Series([True] * len(df))
            for vn, fv in fixed_values.items():
                if vn in df.columns:
                    if isinstance(fv, (int, float)):
                        mask &= np.abs(df[vn] - fv) < 1e-6
                    else:
                        mask &= df[vn] == fv
            if not mask.any():
                return None, None
            filtered = df[mask]
            return filtered[x_var].values, filtered[obj_name].values

        def _make_title(obj_name):
            if title:
                return title
            parts = f"1D Slice: {x_var}"
            if self.is_multi_objective:
                parts = f"1D Slice ({obj_name}): {x_var}"
            if fixed_values:
                fixed_str = ', '.join([f'{k}={v}' for k, v in fixed_values.items()])
                parts += f"\n({fixed_str})"
            return parts

        if isinstance(resolved, list):
            n = len(resolved)
            fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n, figsize[1]), dpi=dpi)
            if n == 1:
                axes = [axes]
            for i, obj in enumerate(resolved):
                preds, std = self._get_predictions_for_objective(predict_result, obj)
                ex, ey = _get_exp_data(obj)
                create_slice_plot(
                    x_values=x_values, predictions=preds, x_var=x_var,
                    std=std, sigma_bands=sigma_bands, exp_x=ex, exp_y=ey,
                    title=_make_title(obj), ax=axes[i]
                )
            fig.tight_layout()
            logger.info(f"Generated multi-objective slice plot for {x_var}")
            return fig

        # Single objective
        predictions, std = self._get_predictions_for_objective(predict_result, resolved)
        exp_x, exp_y = _get_exp_data(resolved)

        fig, ax = create_slice_plot(
            x_values=x_values, predictions=predictions, x_var=x_var,
            std=std, sigma_bands=sigma_bands, exp_x=exp_x, exp_y=exp_y,
            figsize=figsize, dpi=dpi, title=_make_title(resolved)
        )

        logger.info(f"Generated 1D slice plot for {x_var}")
        return fig
    
    def plot_contour(
        self,
        x_var: str,
        y_var: str,
        fixed_values: Optional[Dict[str, Any]] = None,
        grid_resolution: int = 50,
        show_experiments: bool = True,
        show_suggestions: bool = False,
        cmap: str = 'viridis',
        figsize: Tuple[float, float] = (8, 6),
        dpi: int = 100,
        title: Optional[str] = None,
        target_columns: Optional[str] = None,
        ax: Optional['Axes'] = None
    ) -> Figure: # pyright: ignore[reportInvalidTypeForm]
        """
        Create 2D contour plot of model predictions over a variable space.
        
        Visualizes the model's predicted response surface by varying two variables
        while holding others constant. Useful for understanding variable interactions
        and identifying optimal regions.
        
        Args:
            x_var: Variable name for X axis (must be 'real' type)
            y_var: Variable name for Y axis (must be 'real' type)
            fixed_values: Dict of {var_name: value} for other variables.
                         If not provided, uses midpoint for real/integer,
                         first category for categorical.
            grid_resolution: Grid density (NxN points)
            show_experiments: Plot experimental data points as scatter
            show_suggestions: Plot last suggested points (if available)
            cmap: Matplotlib colormap name (e.g., 'viridis', 'coolwarm', 'plasma')
            figsize: Figure size as (width, height) in inches
            dpi: Dots per inch for figure resolution
            title: Custom title (default: "Contour Plot of Model Predictions")
        
        Returns:
            matplotlib Figure object (displays inline in Jupyter)
        
        Example:
            >>> # Basic contour plot
            >>> fig = session.plot_contour('temperature', 'pressure')
            
            >>> # With fixed values for other variables
            >>> fig = session.plot_contour(
            ...     'temperature', 'pressure',
            ...     fixed_values={'catalyst': 'Pt', 'flow_rate': 50},
            ...     cmap='coolwarm',
            ...     grid_resolution=100
            ... )
            >>> fig.savefig('contour.png', dpi=300, bbox_inches='tight')
        
        Note:
            - Requires at least 2 'real' type variables
            - Model must be trained before plotting
            - Categorical variables are automatically encoded using model's encoding
        """
        self._check_matplotlib()
        self._check_model_trained()
        
        if fixed_values is None:
            fixed_values = {}
        
        # Get variable names
        var_names = self.search_space.get_variable_names()
        
        # Validate variables exist
        if x_var not in var_names:
            raise ValueError(f"Variable '{x_var}' not in search space")
        if y_var not in var_names:
            raise ValueError(f"Variable '{y_var}' not in search space")
        
        # Get variable info (search_space.variables is a list)
        x_var_info = next(v for v in self.search_space.variables if v['name'] == x_var)
        y_var_info = next(v for v in self.search_space.variables if v['name'] == y_var)
        
        if x_var_info['type'] != 'real':
            raise ValueError(f"X variable '{x_var}' must be 'real' type, got '{x_var_info['type']}'")
        if y_var_info['type'] != 'real':
            raise ValueError(f"Y variable '{y_var}' must be 'real' type, got '{y_var_info['type']}'")
        
        # Get bounds
        x_bounds = (x_var_info['min'], x_var_info['max'])
        y_bounds = (y_var_info['min'], y_var_info['max'])
        
        # Create meshgrid
        x = np.linspace(x_bounds[0], x_bounds[1], grid_resolution)
        y = np.linspace(y_bounds[0], y_bounds[1], grid_resolution)
        X_grid, Y_grid = np.meshgrid(x, y)
        
        # Build prediction dataframe with ALL variables in proper order
        # Start with grid variables
        grid_data = {
            x_var: X_grid.ravel(),
            y_var: Y_grid.ravel()
        }
        
        # Add fixed values for other variables
        for var in self.search_space.variables:
            var_name = var['name']
            if var_name in [x_var, y_var]:
                continue
            
            if var_name in fixed_values:
                grid_data[var_name] = fixed_values[var_name]
            else:
                # Use default value
                if var['type'] in ['real', 'integer']:
                    grid_data[var_name] = (var['min'] + var['max']) / 2
                elif var['type'] == 'categorical':
                    grid_data[var_name] = var['values'][0]
        
        grid_df = self._build_grid_df(grid_data)
        
        # Get predictions
        predict_result = self.predict(grid_df)

        # Prepare experimental data for overlay
        exp_x = None
        exp_y = None
        if show_experiments and not self.experiment_manager.df.empty:
            exp_df = self.experiment_manager.df
            if x_var in exp_df.columns and y_var in exp_df.columns:
                exp_x = exp_df[x_var].values
                exp_y = exp_df[y_var].values

        # Prepare suggestion data for overlay
        sugg_x = None
        sugg_y = None
        if show_suggestions and len(self.last_suggestions) > 0:
            if isinstance(self.last_suggestions, pd.DataFrame):
                sugg_df = self.last_suggestions
            else:
                sugg_df = pd.DataFrame(self.last_suggestions)
            if x_var in sugg_df.columns and y_var in sugg_df.columns:
                sugg_x = sugg_df[x_var].values
                sugg_y = sugg_df[y_var].values

        resolved = self._resolve_target_column(target_columns)

        def _contour_for_obj(obj_name, contour_ax=None):
            preds, _ = self._get_predictions_for_objective(predict_result, obj_name)
            preds_grid = preds.reshape(X_grid.shape)
            obj_title = title or (f"Contour: {obj_name}" if self.is_multi_objective
                                   else "Contour Plot of Model Predictions")
            return create_contour_plot(
                x_grid=X_grid, y_grid=Y_grid, predictions_grid=preds_grid,
                x_var=x_var, y_var=y_var, exp_x=exp_x, exp_y=exp_y,
                suggest_x=sugg_x, suggest_y=sugg_y, cmap=cmap,
                figsize=figsize, dpi=dpi, title=obj_title, ax=contour_ax
            )

        if isinstance(resolved, list):
            if ax is not None:
                raise ValueError(
                    "Cannot use ax= with multi-objective 'all' mode. "
                    "Specify a single target_columns name instead."
                )
            n = len(resolved)
            fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n, figsize[1]), dpi=dpi)
            if n == 1:
                axes = [axes]
            for i, obj in enumerate(resolved):
                _contour_for_obj(obj, contour_ax=axes[i])
            fig.tight_layout()
            logger.info(f"Generated multi-objective contour plot for {x_var} vs {y_var}")
            return fig

        fig, plot_ax, cbar = _contour_for_obj(resolved, contour_ax=ax)
        logger.info(f"Generated contour plot for {x_var} vs {y_var}")
        return fig
    
    def plot_voxel(
        self,
        x_var: str,
        y_var: str,
        z_var: str,
        fixed_values: Optional[Dict[str, Any]] = None,
        grid_resolution: int = 15,
        show_experiments: bool = True,
        show_suggestions: bool = False,
        cmap: str = 'viridis',
        alpha: float = 0.5,
        use_log_scale: bool = False,
        figsize: Tuple[float, float] = (10, 8),
        dpi: int = 100,
        title: Optional[str] = None,
        target_columns: Optional[str] = None
    ) -> Figure: # pyright: ignore[reportInvalidTypeForm]
        """
        Create 3D voxel plot of model predictions over a variable space.
        
        Visualizes the model's predicted response surface by varying three variables
        while holding others constant. Uses volumetric rendering to show the 3D
        prediction landscape with adjustable transparency.
        
        Args:
            x_var: Variable name for X axis (must be 'real' or 'integer' type)
            y_var: Variable name for Y axis (must be 'real' or 'integer' type)
            z_var: Variable name for Z axis (must be 'real' or 'integer' type)
            fixed_values: Dict of {var_name: value} for other variables.
                         If not provided, uses midpoint for real/integer,
                         first category for categorical.
            grid_resolution: Grid density (NxNxN points, default: 15)
                           Note: 15³ = 3375 points, scales as N³
            show_experiments: Plot experimental data points as scatter
            show_suggestions: Plot last suggested points (if available)
            cmap: Matplotlib colormap name (e.g., 'viridis', 'coolwarm', 'plasma')
            alpha: Transparency level (0.0=transparent, 1.0=opaque, default: 0.5)
                  Lower values reveal interior structure better
            use_log_scale: Use logarithmic color scale for values spanning orders of magnitude
            figsize: Figure size as (width, height) in inches
            dpi: Dots per inch for figure resolution
            title: Custom title (default: "3D Voxel Plot of Model Predictions")
        
        Returns:
            matplotlib Figure object with 3D axes
        
        Example:
            >>> # Basic 3D voxel plot
            >>> fig = session.plot_voxel('temperature', 'pressure', 'flow_rate')
            
            >>> # With transparency to see interior
            >>> fig = session.plot_voxel(
            ...     'temperature', 'pressure', 'flow_rate',
            ...     alpha=0.3,
            ...     grid_resolution=20
            ... )
            >>> fig.savefig('voxel_plot.png', dpi=150, bbox_inches='tight')
            
            >>> # With fixed values for other variables
            >>> fig = session.plot_voxel(
            ...     'temperature', 'pressure', 'flow_rate',
            ...     fixed_values={'catalyst': 'Pt', 'pH': 7.0},
            ...     cmap='coolwarm'
            ... )
        
        Raises:
            ValueError: If search space doesn't have at least 3 continuous variables
        
        Note:
            - Requires at least 3 'real' or 'integer' type variables
            - Model must be trained before plotting
            - Computationally expensive: O(N³) evaluations
            - Lower grid_resolution for faster rendering
            - Use alpha < 0.5 to see interior structure
            - Interactive rotation available in some backends (notebook)
        """
        self._check_matplotlib()
        self._check_model_trained()
        
        if fixed_values is None:
            fixed_values = {}
        
        # Get all variable names and check for continuous variables
        var_names = self.search_space.get_variable_names()
        
        # Count continuous variables (real or integer)
        continuous_vars = []
        for var in self.search_space.variables:
            if var['type'] in ['real', 'integer']:
                continuous_vars.append(var['name'])
        
        # Check if we have at least 3 continuous variables
        if len(continuous_vars) < 3:
            raise ValueError(
                f"3D voxel plot requires at least 3 continuous (real or integer) variables. "
                f"Found only {len(continuous_vars)}: {continuous_vars}. "
                f"Use plot_slice() for 1D or plot_contour() for 2D visualization instead."
            )
        
        # Validate that the requested variables exist and are continuous
        for var_name, var_label in [(x_var, 'X'), (y_var, 'Y'), (z_var, 'Z')]:
            if var_name not in var_names:
                raise ValueError(f"{var_label} variable '{var_name}' not in search space")
            
            var_def = next(v for v in self.search_space.variables if v['name'] == var_name)
            if var_def['type'] not in ['real', 'integer']:
                raise ValueError(
                    f"{var_label} variable '{var_name}' must be 'real' or 'integer' type for voxel plot, "
                    f"got '{var_def['type']}'"
                )
        
        # Get variable definitions
        x_var_def = next(v for v in self.search_space.variables if v['name'] == x_var)
        y_var_def = next(v for v in self.search_space.variables if v['name'] == y_var)
        z_var_def = next(v for v in self.search_space.variables if v['name'] == z_var)
        
        # Get bounds
        x_bounds = (x_var_def['min'], x_var_def['max'])
        y_bounds = (y_var_def['min'], y_var_def['max'])
        z_bounds = (z_var_def['min'], z_var_def['max'])
        
        # Create 3D meshgrid
        x = np.linspace(x_bounds[0], x_bounds[1], grid_resolution)
        y = np.linspace(y_bounds[0], y_bounds[1], grid_resolution)
        z = np.linspace(z_bounds[0], z_bounds[1], grid_resolution)
        X_grid, Y_grid, Z_grid = np.meshgrid(x, y, z, indexing='ij')
        
        # Build prediction dataframe with ALL variables in proper order
        grid_data = {
            x_var: X_grid.ravel(),
            y_var: Y_grid.ravel(),
            z_var: Z_grid.ravel()
        }
        
        # Add fixed values for other variables
        for var in self.search_space.variables:
            var_name = var['name']
            if var_name in [x_var, y_var, z_var]:
                continue
            
            if var_name in fixed_values:
                grid_data[var_name] = fixed_values[var_name]
            else:
                # Use default value
                if var['type'] in ['real', 'integer']:
                    grid_data[var_name] = (var['min'] + var['max']) / 2
                elif var['type'] == 'categorical':
                    grid_data[var_name] = var['values'][0]
        
        grid_df = self._build_grid_df(grid_data)

        # Get predictions
        predict_result = self.predict(grid_df)

        # Prepare experimental data for overlay
        exp_x = None
        exp_y = None
        exp_z = None
        if show_experiments and not self.experiment_manager.df.empty:
            exp_df = self.experiment_manager.df
            if x_var in exp_df.columns and y_var in exp_df.columns and z_var in exp_df.columns:
                exp_x = exp_df[x_var].values
                exp_y = exp_df[y_var].values
                exp_z = exp_df[z_var].values

        # Prepare suggestion data for overlay
        sugg_x = None
        sugg_y = None
        sugg_z = None
        if show_suggestions and len(self.last_suggestions) > 0:
            if isinstance(self.last_suggestions, pd.DataFrame):
                sugg_df = self.last_suggestions
            else:
                sugg_df = pd.DataFrame(self.last_suggestions)
            if x_var in sugg_df.columns and y_var in sugg_df.columns and z_var in sugg_df.columns:
                sugg_x = sugg_df[x_var].values
                sugg_y = sugg_df[y_var].values
                sugg_z = sugg_df[z_var].values

        from alchemist_core.visualization.plots import create_voxel_plot

        resolved = self._resolve_target_column(target_columns)

        def _voxel_for_obj(obj_name):
            preds, _ = self._get_predictions_for_objective(predict_result, obj_name)
            preds_grid = preds.reshape(X_grid.shape)
            obj_title = title or (f"3D Voxel: {obj_name}" if self.is_multi_objective
                                   else "3D Voxel Plot of Model Predictions")
            return create_voxel_plot(
                x_grid=X_grid, y_grid=Y_grid, z_grid=Z_grid,
                predictions_grid=preds_grid, x_var=x_var, y_var=y_var, z_var=z_var,
                exp_x=exp_x, exp_y=exp_y, exp_z=exp_z,
                suggest_x=sugg_x, suggest_y=sugg_y, suggest_z=sugg_z,
                cmap=cmap, alpha=alpha, use_log_scale=use_log_scale,
                figsize=figsize, dpi=dpi, title=obj_title
            )

        if isinstance(resolved, list):
            # Voxel plots don't support subplots well; create separate figures
            # Return first objective's figure and log a warning
            logger.warning("Voxel 'all' mode creates separate figures per objective; returning last")
            fig = None
            for obj in resolved:
                fig, ax = _voxel_for_obj(obj)
            return fig

        fig, ax = _voxel_for_obj(resolved)
        logger.info(f"Generated 3D voxel plot for {x_var} vs {y_var} vs {z_var}")
        return fig
    
    def plot_metrics(
        self,
        metric: Literal['rmse', 'mae', 'r2', 'mape'] = 'rmse',
        cv_splits: int = 5,
        figsize: Tuple[float, float] = (8, 6),
        dpi: int = 100,
        use_cached: bool = True,
        target_columns: Optional[str] = None,
        ax: Optional['Axes'] = None
    ) -> Figure: # pyright: ignore[reportInvalidTypeForm]
        """
        Plot cross-validation metrics as a function of training set size.

        Args:
            metric: Which metric to plot ('rmse', 'mae', 'r2', or 'mape')
            cv_splits: Number of cross-validation folds (default: 5)
            figsize: Figure size as (width, height) in inches
            dpi: Dots per inch for figure resolution
            use_cached: Use cached metrics if available (default: True)
            target_columns: For multi-objective: objective name, 'all', or None.

        Returns:
            matplotlib Figure object

        Note:
            Calls model.evaluate() if metrics not cached, which can be computationally
            expensive for large datasets. Set use_cached=False to force recomputation.
        """
        self._check_matplotlib()
        self._check_model_trained()

        if self.is_multi_objective:
            raise ValueError(
                "plot_metrics() is not yet supported for multi-objective sessions. "
                "Use plot_parity(target_columns=...) for per-objective model diagnostics."
            )
        
        # Need at least 5 observations for CV
        n_total = len(self.experiment_manager.df)
        if n_total < 5:
            raise ValueError(f"Need at least 5 observations for metrics plot (have {n_total})")
        
        # Check for cached metrics first
        cache_key = f'_cached_cv_metrics_{cv_splits}'
        if use_cached and hasattr(self.model, cache_key):
            cv_metrics = getattr(self.model, cache_key)
            logger.info(f"Using cached CV metrics for {metric.upper()}")
        else:
            # Call model's evaluate method to get metrics over training sizes
            logger.info(f"Computing {metric.upper()} over training set sizes (this may take a moment)...")
            cv_metrics = self.model.evaluate(
                self.experiment_manager,
                cv_splits=cv_splits,
                debug=False
            )
            # Cache the results
            setattr(self.model, cache_key, cv_metrics)
        
        # Extract the requested metric
        metric_key_map = {
            'rmse': 'RMSE',
            'mae': 'MAE',
            'r2': 'R²',
            'mape': 'MAPE'
        }
        
        if metric not in metric_key_map:
            raise ValueError(f"Unknown metric '{metric}'. Choose from: {list(metric_key_map.keys())}")
        
        metric_key = metric_key_map[metric]
        metric_values = cv_metrics.get(metric_key, [])
        
        if not metric_values:
            raise RuntimeError(f"Model did not return {metric_key} values from evaluate()")
        
        # X-axis: training set sizes (starts at 5)
        x_range = np.arange(5, len(metric_values) + 5)
        metric_array = np.array(metric_values)
        
        # Delegate to visualization module
        fig, plot_ax = create_metrics_plot(
            training_sizes=x_range,
            metric_values=metric_array,
            metric_name=metric,
            figsize=figsize,
            dpi=dpi,
            ax=ax
        )

        logger.info(f"Generated {metric} metrics plot with {len(metric_values)} points")
        return fig
    
    def plot_qq(
        self,
        use_calibrated: bool = False,
        figsize: Tuple[float, float] = (8, 6),
        dpi: int = 100,
        title: Optional[str] = None,
        target_columns: Optional[str] = None,
        ax: Optional['Axes'] = None
    ) -> Figure: # pyright: ignore[reportInvalidTypeForm]
        """
        Create Q-Q (quantile-quantile) plot for model residuals normality check.

        Args:
            use_calibrated: Use calibrated uncertainty estimates if available
            figsize: Figure size as (width, height) in inches
            dpi: Dots per inch for figure resolution
            title: Custom title (default: auto-generated)
            target_columns: For multi-objective: objective name, 'all', or None.
            ax: Existing matplotlib Axes to draw on (creates new figure if None).
                Cannot be used with multi-objective 'all' mode.

        Returns:
            matplotlib Figure object
        """
        self._check_matplotlib()
        self._check_model_trained()

        resolved = self._resolve_target_column(target_columns)

        def _qq_for_obj(obj_name, qq_ax=None):
            cv_results = self._check_cv_results(
                use_calibrated,
                target_columns=obj_name if self.is_multi_objective else None
            )
            y_true = cv_results['y_true']
            y_pred = cv_results['y_pred']
            y_std = cv_results.get('y_std', None)
            residuals = y_true - y_pred
            if y_std is not None and len(y_std) > 0:
                z_scores = residuals / y_std
            else:
                z_scores = residuals / np.std(residuals)
            if title:
                obj_title = title
            elif self.is_multi_objective:
                obj_title = f"Q-Q Plot: {obj_name}"
            else:
                cal_label = " (Calibrated)" if use_calibrated else ""
                z_mean = float(np.mean(z_scores))
                z_std_val = float(np.std(z_scores, ddof=1))
                n = len(z_scores)
                obj_title = (
                    f"Q-Q Plot: Standardized Residuals vs. Normal Distribution{cal_label}\n"
                    f"Mean(z) = {z_mean:.3f}, Std(z) = {z_std_val:.3f}, N = {n}"
                )
            return create_qq_plot(z_scores=z_scores, figsize=figsize, dpi=dpi,
                                  title=obj_title, ax=qq_ax)

        if isinstance(resolved, list):
            if ax is not None:
                raise ValueError(
                    "Cannot use ax= with multi-objective 'all' mode. "
                    "Specify a single target_columns name instead."
                )
            n = len(resolved)
            fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n, figsize[1]), dpi=dpi)
            if n == 1:
                axes = [axes]
            for i, obj in enumerate(resolved):
                _qq_for_obj(obj, qq_ax=axes[i])
            fig.tight_layout()
            logger.info(f"Generated multi-objective Q-Q plot ({n} objectives)")
            return fig

        fig, plot_ax = _qq_for_obj(resolved, qq_ax=ax)
        logger.info("Generated Q-Q plot for residuals")
        return fig
    
    def plot_calibration(
        self,
        use_calibrated: bool = False,
        n_bins: int = 10,
        figsize: Tuple[float, float] = (8, 6),
        dpi: int = 100,
        title: Optional[str] = None,
        target_columns: Optional[str] = None,
        ax: Optional['Axes'] = None
    ) -> Figure: # pyright: ignore[reportInvalidTypeForm]
        """
        Create calibration plot showing reliability of uncertainty estimates.

        Args:
            use_calibrated: Use calibrated uncertainty estimates if available
            n_bins: Number of bins for grouping predictions (default: 10)
            figsize: Figure size as (width, height) in inches
            dpi: Dots per inch for figure resolution
            title: Custom title (default: auto-generated)
            target_columns: For multi-objective: objective name, 'all', or None.
            ax: Existing matplotlib Axes to draw on (creates new figure if None).
                Cannot be used with multi-objective 'all' mode.

        Returns:
            matplotlib Figure object
        """
        self._check_matplotlib()
        self._check_model_trained()

        from scipy import stats

        resolved = self._resolve_target_column(target_columns)

        def _cal_for_obj(obj_name, cal_ax=None):
            cv_results = self._check_cv_results(
                use_calibrated,
                target_columns=obj_name if self.is_multi_objective else None
            )
            y_true = cv_results['y_true']
            y_pred = cv_results['y_pred']
            y_std = cv_results.get('y_std', None)
            if y_std is None:
                raise ValueError(
                    "Model does not provide uncertainty estimates (y_std). "
                    "Calibration plot requires uncertainty predictions."
                )
            nominal_probs = np.arange(0.10, 1.00, 0.05)
            empirical_coverage = []
            for prob in nominal_probs:
                sigma = stats.norm.ppf((1 + prob) / 2)
                lower_bound = y_pred - sigma * y_std
                upper_bound = y_pred + sigma * y_std
                within_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
                empirical_coverage.append(np.mean(within_interval))
            if title:
                obj_title = title
            elif self.is_multi_objective:
                obj_title = f"Calibration: {obj_name}"
            else:
                cal_label = " (Calibrated)" if use_calibrated else " (Uncalibrated)"
                n = len(y_true)
                obj_title = (
                    f"Calibration Curve (Reliability Diagram){cal_label}\n"
                    f"N = {n}"
                )
            return create_calibration_plot(
                nominal_probs=nominal_probs,
                empirical_coverage=np.array(empirical_coverage),
                figsize=figsize, dpi=dpi, title=obj_title, ax=cal_ax
            )

        if isinstance(resolved, list):
            if ax is not None:
                raise ValueError(
                    "Cannot use ax= with multi-objective 'all' mode. "
                    "Specify a single target_columns name instead."
                )
            n = len(resolved)
            fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n, figsize[1]), dpi=dpi)
            if n == 1:
                axes = [axes]
            for i, obj in enumerate(resolved):
                _cal_for_obj(obj, cal_ax=axes[i])
            fig.tight_layout()
            logger.info(f"Generated multi-objective calibration plot ({n} objectives)")
            return fig

        fig, plot_ax = _cal_for_obj(resolved, cal_ax=ax)
        logger.info("Generated calibration plot for uncertainty estimates")
        return fig

    def plot_pareto_frontier(
        self,
        directions: Optional[List[str]] = None,
        ref_point: Optional[List[float]] = None,
        show_hypervolume: bool = True,
        constraint_boundaries: Optional[Dict[str, float]] = None,
        suggested_points_override: Optional[np.ndarray] = None,
        figsize=(8, 6), dpi=100, title=None
    ):
        """Plot the Pareto frontier for multi-objective optimization.
        
        Currently supports 2-objective optimization only.

        Args:
            directions: Per-objective direction ('maximize'/'minimize'). Default: all maximize.
            ref_point: Reference point for hypervolume shading.
            show_hypervolume: Whether to shade dominated hypervolume.
            constraint_boundaries: {objective_name: value} for constraint lines.
            suggested_points_override: (n, n_obj) array of points to overlay as suggestions.
            figsize: Figure size.
            dpi: Figure DPI.
            title: Optional plot title.

        Returns:
            matplotlib Figure
        """
        if not self.is_multi_objective:
            raise ValueError("plot_pareto_frontier requires multi-objective data (2+ target columns)")
        
        if self.n_objectives != 2:
            raise ValueError(
                f"plot_pareto_frontier currently only supports 2 objectives, but session has {self.n_objectives}. "
                "Use plot_parity() with target_columns to visualize individual objectives."
            )

        if not _HAS_VISUALIZATION:
            raise ImportError("matplotlib is required for visualization")

        Y = self.experiment_manager.df[self.objective_names].values

        if directions is None:
            directions = ['maximize'] * self.n_objectives

        pareto_df = self.experiment_manager.get_pareto_frontier(directions)
        pareto_mask = np.zeros(len(Y), dtype=bool)
        pareto_mask[pareto_df.index] = True

        fig, ax = create_pareto_plot(
            Y=Y,
            pareto_mask=pareto_mask,
            objective_names=self.objective_names,
            directions=directions,
            ref_point=ref_point,
            show_hypervolume=show_hypervolume,
            suggested_points=suggested_points_override,
            constraint_boundaries=constraint_boundaries,
            figsize=figsize,
            dpi=dpi,
            title=title,
        )

        logger.info("Generated Pareto frontier plot")
        return fig

    def plot_regret(
        self,
        goal: Union[str, List[str]] = 'maximize',
        include_predictions: bool = True,
        show_cumulative: bool = False,
        backend: Optional[str] = None,
        kernel: Optional[str] = None,
        n_grid_points: int = 1000,
        sigma_bands: Optional[List[float]] = None,
        start_iteration: int = 5,
        reuse_hyperparameters: bool = True,
        use_calibrated_uncertainty: bool = False,
        ref_point: Optional[List[float]] = None,
        figsize: Tuple[float, float] = (8, 6),
        dpi: int = 100,
        title: Optional[str] = None
    ) -> Figure: # pyright: ignore[reportInvalidTypeForm]
        """
        Plot optimization progress.

        For single-objective: regret (incumbent trajectory) plot.
        For multi-objective: hypervolume convergence plot.

        Args:
            goal: 'maximize' or 'minimize' (str), or list of per-objective directions.
            include_predictions: Overlay model predictions (single-obj only).
            show_cumulative: Show cumulative best line.
            backend: Model backend. Uses session default if None.
            kernel: Kernel type. Uses session default if None.
            n_grid_points: Grid points for prediction overlay.
            sigma_bands: Sigma values for uncertainty bands.
            start_iteration: First iteration for predictions.
            reuse_hyperparameters: Reuse final model's hyperparameters.
            use_calibrated_uncertainty: Use calibrated uncertainties.
            ref_point: Reference point for hypervolume (MOBO only, required).
            figsize: Figure size.
            dpi: DPI.
            title: Custom title.

        Returns:
            matplotlib Figure object
        """
        self._check_matplotlib()

        n_exp = len(self.experiment_manager.df)
        if n_exp < 2:
            raise ValueError(f"Need at least 2 experiments for regret plot (have {n_exp})")

        # ---- MOBO: hypervolume convergence ----
        if self.is_multi_objective:
            if ref_point is None:
                raise ValueError(
                    "ref_point is required for multi-objective hypervolume convergence. "
                    "Provide a list of reference values (one per objective)."
                )
            if isinstance(goal, str):
                directions = [goal.lower()] * self.n_objectives
            else:
                directions = [g.lower() for g in goal]

            iterations = np.arange(1, n_exp + 1)

            # Compute per-experiment HV contribution (delta HV), analogous to
            # raw Y values in single-objective regret.
            import torch
            from botorch.utils.multi_objective.hypervolume import Hypervolume
            from botorch.utils.multi_objective.pareto import is_non_dominated

            ref_t_base = torch.tensor(ref_point, dtype=torch.double)
            for j, d in enumerate(directions):
                if d == 'minimize':
                    ref_t_base[j] = -ref_t_base[j]

            cumulative_hv = np.zeros(n_exp)
            observed_hv = np.zeros(n_exp)

            for i in range(1, n_exp + 1):
                subset_df = self.experiment_manager.df.iloc[:i]
                Y_sub = subset_df[self.objective_names].values
                try:
                    Y_t = torch.tensor(Y_sub, dtype=torch.double)
                    for j, d in enumerate(directions):
                        if d == 'minimize':
                            Y_t[:, j] = -Y_t[:, j]
                    mask = is_non_dominated(Y_t)
                    if mask.any():
                        hv_obj = Hypervolume(ref_point=ref_t_base)
                        cumulative_hv[i - 1] = hv_obj.compute(Y_t[mask])
                    else:
                        cumulative_hv[i - 1] = 0.0
                except Exception:
                    cumulative_hv[i - 1] = cumulative_hv[i - 2] if i > 1 else 0.0

                # Delta HV = contribution of the i-th experiment
                prev_hv = cumulative_hv[i - 2] if i > 1 else 0.0
                observed_hv[i - 1] = cumulative_hv[i - 1] - prev_hv

            # Compute posterior predicted HV with uncertainty
            pred_hv = None
            pred_hv_std = None
            if include_predictions and n_exp >= start_iteration:
                try:
                    pred_hv, pred_hv_std = self._compute_posterior_hv_predictions(
                        ref_point=ref_point,
                        directions=directions,
                        n_grid_points=n_grid_points,
                        start_iteration=start_iteration,
                        reuse_hyperparameters=reuse_hyperparameters,
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not compute posterior HV predictions: {e}. "
                        "Plotting observations only."
                    )

            from alchemist_core.visualization.plots import create_hypervolume_convergence_plot
            fig, ax = create_hypervolume_convergence_plot(
                iterations=iterations,
                observed_hv=observed_hv,
                show_cumulative=show_cumulative,
                predicted_hv=pred_hv,
                predicted_hv_std=pred_hv_std,
                ref_point=ref_point,
                sigma_bands=sigma_bands,
                figsize=figsize,
                dpi=dpi,
                title=title,
            )
            logger.info(f"Generated hypervolume convergence plot with {n_exp} experiments")
            return fig

        # ---- Single-objective regret ----
        target_col = self.experiment_manager.target_columns[0]
        observed_values = self.experiment_manager.df[target_col].values
        iterations = np.arange(1, n_exp + 1)
        goal_str = goal if isinstance(goal, str) else goal[0]
        
        # Compute posterior predictions if requested
        predicted_means = None
        predicted_stds = None

        if include_predictions and n_exp >= start_iteration:
            try:
                predicted_means, predicted_stds = self._compute_posterior_predictions(
                    goal=goal_str,
                    backend=backend,
                    kernel=kernel,
                    n_grid_points=n_grid_points,
                    start_iteration=start_iteration,
                    reuse_hyperparameters=reuse_hyperparameters,
                    use_calibrated_uncertainty=use_calibrated_uncertainty
                )
            except Exception as e:
                logger.warning(f"Could not compute posterior predictions: {e}. Plotting observations only.")

        from alchemist_core.visualization.plots import create_regret_plot

        fig, ax = create_regret_plot(
            iterations=iterations,
            observed_values=observed_values,
            show_cumulative=show_cumulative,
            goal=goal_str,
            predicted_means=predicted_means,
            predicted_stds=predicted_stds,
            sigma_bands=sigma_bands,
            figsize=figsize,
            dpi=dpi,
            title=title
        )

        logger.info(f"Generated regret plot with {n_exp} experiments")
        return fig
    
    def _generate_prediction_grid(self, n_grid_points: int) -> pd.DataFrame:
        """
        Generate grid of test points across search space for predictions.
        
        Args:
            n_grid_points: Target number of grid points (actual number depends on dimensionality)
        
        Returns:
            DataFrame with columns for each variable
        """
        grid_1d = []
        var_names = []
        
        for var in self.search_space.variables:
            var_names.append(var['name'])
            
            if var['type'] == 'real':
                # Continuous: linspace
                n_per_dim = int(n_grid_points ** (1/len(self.search_space.variables)))
                grid_1d.append(np.linspace(var['min'], var['max'], n_per_dim))
            elif var['type'] == 'integer':
                # Integer: range of integers
                n_per_dim = int(n_grid_points ** (1/len(self.search_space.variables)))
                grid_1d.append(np.linspace(var['min'], var['max'], n_per_dim).astype(int))
            else:
                # Categorical: use actual category values
                grid_1d.append(var['values'])
        
        # Generate test points using Cartesian product
        from itertools import product
        X_test_tuples = list(product(*grid_1d))
        
        # Convert to DataFrame with proper variable names and types
        grid = pd.DataFrame(X_test_tuples, columns=var_names)
        
        # Ensure correct dtypes for categorical variables
        for var in self.search_space.variables:
            if var['type'] == 'categorical':
                grid[var['name']] = grid[var['name']].astype(str)
        
        return grid
    
    def _compute_posterior_predictions(
        self,
        goal: str,
        backend: Optional[str],
        kernel: Optional[str],
        n_grid_points: int,
        start_iteration: int,
        reuse_hyperparameters: bool,
        use_calibrated_uncertainty: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute max(posterior mean) and corresponding std at each iteration.
        
        Helper method for regret plot to overlay model predictions with uncertainty.
        
        IMPORTANT: When reuse_hyperparameters=True, this uses the final model's 
        hyperparameters for ALL iterations by creating fresh GP models with those
        hyperparameters and subsets of data. This avoids numerical instability from
        repeated MLE optimization.
        
        Returns:
            Tuple of (predicted_means, predicted_stds) arrays, same length as n_experiments
        """
        n_exp = len(self.experiment_manager.df)
        
        # Initialize arrays (NaN for iterations before start_iteration)
        predicted_means = np.full(n_exp, np.nan)
        predicted_stds = np.full(n_exp, np.nan)
        
        # Determine backend and kernel
        if backend is None:
            if self.model is None or not self.model.is_trained:
                raise ValueError("No trained model in session. Train a model first or specify backend/kernel.")
            backend = self.model_backend
        
        if kernel is None:
            if self.model is None or not self.model.is_trained:
                raise ValueError("No trained model in session. Train a model first or specify backend/kernel.")
            if backend == 'sklearn':
                kernel = self.model.kernel_options.get('kernel_type', 'RBF')
            elif backend == 'botorch':
                # BoTorchModel stores kernel type in cont_kernel_type
                kernel = getattr(self.model, 'cont_kernel_type', 'Matern')
        
        # Extract optimized state_dict for botorch or kernel params for sklearn
        optimized_state_dict = None
        optimized_kernel_params = None
        if reuse_hyperparameters and self.model is not None and self.model.is_trained:
            if backend == 'sklearn':
                optimized_kernel_params = self.model.optimized_kernel.get_params()
            elif backend == 'botorch':
                # Store the fitted state dict from the final model
                optimized_state_dict = self.model.fitted_state_dict
        
        # Generate grid for predictions
        grid = self._generate_prediction_grid(n_grid_points)
        
        # Get full dataset
        full_df = self.experiment_manager.df
        target_col = self.experiment_manager.target_columns[0]
        
        # Suppress INFO logging for temp sessions to avoid spam
        import logging
        original_session_level = logger.level
        original_model_level = logging.getLogger('alchemist_core.models.botorch_model').level
        logger.setLevel(logging.WARNING)
        logging.getLogger('alchemist_core.models.botorch_model').setLevel(logging.WARNING)
        
        # Loop through iterations
        for i in range(start_iteration, n_exp + 1):
            try:
                # Create temporary session with subset of data
                temp_session = OptimizationSession()
                
                # Directly assign search space to avoid logging spam
                temp_session.search_space = self.search_space
                temp_session.experiment_manager.set_search_space(self.search_space)
                
                # Add subset of experiments
                for idx in range(i):
                    row = full_df.iloc[idx]
                    inputs = {var['name']: row[var['name']] for var in self.experiment_manager.search_space.variables}
                    temp_session.add_experiment(inputs, output=row[target_col])
                
                # Train model on subset using SAME approach for all iterations
                if backend == 'sklearn':
                    # Create model instance
                    from alchemist_core.models.sklearn_model import SklearnModel
                    temp_model = SklearnModel(kernel_options={'kernel_type': kernel})
                    
                    if reuse_hyperparameters and optimized_kernel_params is not None:
                        # Override n_restarts to disable optimization
                        temp_model.n_restarts_optimizer = 0
                        temp_model._custom_optimizer = None
                        # Store the optimized kernel to use
                        from sklearn.base import clone
                        temp_model._reuse_kernel = clone(self.model.optimized_kernel)
                    
                    # Attach model and train
                    temp_session.model = temp_model
                    temp_session.model_backend = 'sklearn'
                    
                    # Train WITHOUT recomputing calibration (if reusing hyperparameters)
                    if reuse_hyperparameters:
                        temp_model.train(temp_session.experiment_manager, calibrate_uncertainty=False)
                        # Transfer calibration factor from final model
                        if hasattr(self.model, 'calibration_factor'):
                            temp_model.calibration_factor = self.model.calibration_factor
                            # Enable calibration only if user requested calibrated uncertainties
                            temp_model.calibration_enabled = use_calibrated_uncertainty
                    else:
                        temp_model.train(temp_session.experiment_manager)
                    
                    # Verify model was trained
                    if not temp_model.is_trained:
                        raise ValueError(f"Model training failed at iteration {i}")
                    if temp_session.model is None:
                        raise ValueError(f"temp_session.model is None after training at iteration {i}")
                    
                elif backend == 'botorch':
                    # For BoTorch: create a fresh model and load the fitted hyperparameters
                    from alchemist_core.models.botorch_model import BoTorchModel
                    import torch
                    
                    # Create model instance with same configuration as original model
                    kernel_opts = {'cont_kernel_type': kernel}
                    if hasattr(self.model, 'matern_nu'):
                        kernel_opts['matern_nu'] = self.model.matern_nu
                    
                    temp_model = BoTorchModel(
                        kernel_options=kernel_opts,
                        input_transform_type=self.model.input_transform_type if hasattr(self.model, 'input_transform_type') else 'normalize',
                        output_transform_type=self.model.output_transform_type if hasattr(self.model, 'output_transform_type') else 'standardize'
                    )
                    
                    # Train model on subset (this creates the GP with subset of data)
                    # Disable calibration computation if reusing hyperparameters
                    if reuse_hyperparameters:
                        temp_model.train(temp_session.experiment_manager, calibrate_uncertainty=False)
                    else:
                        temp_model.train(temp_session.experiment_manager)
                    
                    # Apply optimized hyperparameters from final model to trained subset model
                    # Only works for simple kernel structures (no categorical variables)
                    if reuse_hyperparameters and optimized_state_dict is not None:
                        try:
                            with torch.no_grad():
                                # Extract hyperparameters from final model
                                # This only works for ScaleKernel(base_kernel), not AdditiveKernel
                                final_lengthscale = self.model.model.covar_module.base_kernel.lengthscale.detach().clone()
                                final_outputscale = self.model.model.covar_module.outputscale.detach().clone()
                                final_noise = self.model.model.likelihood.noise.detach().clone()
                                
                                # Set hyperparameters in temp model (trained on subset)
                                temp_model.model.covar_module.base_kernel.lengthscale = final_lengthscale
                                temp_model.model.covar_module.outputscale = final_outputscale
                                temp_model.model.likelihood.noise = final_noise
                        except AttributeError:
                            # If kernel structure is complex (e.g., has categorical variables),
                            # skip hyperparameter reuse - fall back to each iteration's own optimization
                            pass
                    
                    # Transfer calibration factor from final model (even if hyperparameters couldn't be transferred)
                    # This ensures last iteration matches final model exactly
                    if reuse_hyperparameters and hasattr(self.model, 'calibration_factor'):
                        temp_model.calibration_factor = self.model.calibration_factor
                        # Enable calibration only if user requested calibrated uncertainties
                        temp_model.calibration_enabled = use_calibrated_uncertainty
                    
                    # Attach to session
                    temp_session.model = temp_model
                    temp_session.model_backend = 'botorch'
                
                # Predict on grid using temp_session.predict (consistent for all iterations)
                result = temp_session.predict(grid)
                if result is None:
                    raise ValueError(f"predict() returned None at iteration {i}")
                means, stds = result
                
                # Find max mean (or min for minimization)
                if goal.lower() == 'maximize':
                    best_idx = np.argmax(means)
                else:
                    best_idx = np.argmin(means)
                
                predicted_means[i - 1] = means[best_idx]
                predicted_stds[i - 1] = stds[best_idx]
                
            except Exception as e:
                import traceback
                logger.warning(f"Failed to compute predictions for iteration {i}: {e}")
                logger.debug(traceback.format_exc())
                # Leave as NaN
        
        # Restore original logging levels
        logger.setLevel(original_session_level)
        logging.getLogger('alchemist_core.models.botorch_model').setLevel(original_model_level)
        
        return predicted_means, predicted_stds

    def _compute_posterior_hv_predictions(
        self,
        ref_point: List[float],
        directions: List[str],
        n_grid_points: int = 1000,
        start_iteration: int = 5,
        reuse_hyperparameters: bool = True,
        n_mc_samples: int = 32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute model-predicted hypervolume at each iteration for MOBO.

        Mirrors ``_compute_posterior_predictions`` for single-objective.
        At each iteration *i* (from *start_iteration* to *n_exp*):

        1. Train a temporary ModelListGP on experiments ``0:i``.
        2. Draw *n_mc_samples* posterior samples on a prediction grid.
        3. For each sample, compute the hypervolume of its non-dominated set.
        4. Store the **mean** and **std** of these hypervolume samples.

        Returns:
            (predicted_hv, predicted_hv_std) — arrays of length n_exp (NaN
            for iterations before *start_iteration*).
        """
        import torch
        from botorch.utils.multi_objective.hypervolume import Hypervolume
        from botorch.utils.multi_objective.pareto import is_non_dominated
        from alchemist_core.models.botorch_model import BoTorchModel

        n_exp = len(self.experiment_manager.df)
        predicted_hv = np.full(n_exp, np.nan)
        predicted_hv_std = np.full(n_exp, np.nan)

        # Build ref_point tensor (convert to maximisation space)
        ref_t = torch.tensor(ref_point, dtype=torch.double)
        negate_idx = [j for j, d in enumerate(directions) if d == 'minimize']
        for j in negate_idx:
            ref_t[j] = -ref_t[j]

        # Generate prediction grid
        grid = self._generate_prediction_grid(n_grid_points)
        full_df = self.experiment_manager.df

        # Suppress INFO logging and scipy/botorch optimisation warnings
        import logging
        import warnings
        orig_session_lvl = logger.level
        orig_model_lvl = logging.getLogger('alchemist_core.models.botorch_model').level
        logger.setLevel(logging.WARNING)
        logging.getLogger('alchemist_core.models.botorch_model').setLevel(logging.WARNING)

        for i in range(start_iteration, n_exp + 1):
            try:
                # Create temp experiment manager with first i experiments
                temp_session = OptimizationSession()
                temp_session.search_space = self.search_space
                temp_session.experiment_manager.set_search_space(self.search_space)
                temp_session.experiment_manager.target_columns = list(self.experiment_manager.target_columns)
                # Directly assign DataFrame subset (avoids add_experiment single-output limitation)
                temp_session.experiment_manager.df = full_df.iloc[:i].copy().reset_index(drop=True)

                # Train temp MOBO model
                kernel_opts = {'cont_kernel_type': getattr(self.model, 'cont_kernel_type', 'Matern')}
                if hasattr(self.model, 'matern_nu'):
                    kernel_opts['matern_nu'] = self.model.matern_nu
                temp_model = BoTorchModel(
                    kernel_options=kernel_opts,
                    input_transform_type=getattr(self.model, 'input_transform_type', 'normalize'),
                    output_transform_type=getattr(self.model, 'output_transform_type', 'standardize'),
                )
                temp_model.n_objectives = self.n_objectives
                temp_model.objective_names = list(self.objective_names)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=Warning)
                    temp_model.train(temp_session.experiment_manager, cache_cv=False)
                temp_session.model = temp_model
                temp_session.model_backend = 'botorch'

                # Optionally transfer hyperparameters from final model
                if reuse_hyperparameters and self.model is not None:
                    try:
                        for src_m, dst_m in zip(
                            self.model.model.models, temp_model.model.models
                        ):
                            with torch.no_grad():
                                dst_m.covar_module.base_kernel.lengthscale = (
                                    src_m.covar_module.base_kernel.lengthscale.detach().clone()
                                )
                                dst_m.covar_module.outputscale = (
                                    src_m.covar_module.outputscale.detach().clone()
                                )
                                dst_m.likelihood.noise = (
                                    src_m.likelihood.noise.detach().clone()
                                )
                    except AttributeError:
                        pass  # complex kernel structure — use per-iter optimisation

                # Augment grid with any extra feature columns the model expects
                grid_aug = grid.copy()
                if temp_model.original_feature_names:
                    subset_df = full_df.iloc[:i]
                    for col in temp_model.original_feature_names:
                        if col not in grid_aug.columns:
                            if col in subset_df.columns:
                                grid_aug[col] = subset_df[col].median()
                            else:
                                grid_aug[col] = 0
                    grid_aug = grid_aug[temp_model.original_feature_names]

                # Encode grid and get posterior samples
                X_enc = temp_model._encode_categorical_data(grid_aug)
                if isinstance(X_enc, pd.DataFrame):
                    X_tensor = torch.tensor(X_enc.values, dtype=torch.double)
                else:
                    X_tensor = torch.tensor(X_enc, dtype=torch.double)

                # Stack posteriors from each sub-model into (n_mc, n_grid, n_obj)
                samples_per_obj = []
                for sub_m in temp_model.model.models:
                    sub_m.eval()
                    sub_m.likelihood.eval()
                    with torch.no_grad():
                        post = sub_m.posterior(X_tensor)
                        s = post.rsample(torch.Size([n_mc_samples]))  # (n_mc, n_grid, 1)
                        samples_per_obj.append(s.squeeze(-1))         # (n_mc, n_grid)

                Y_samples = torch.stack(samples_per_obj, dim=-1)  # (n_mc, n_grid, n_obj)

                # Negate minimisation objectives for hypervolume (maximisation convention)
                for j in negate_idx:
                    Y_samples[..., j] = -Y_samples[..., j]

                # Compute HV for each MC sample
                hv_obj = Hypervolume(ref_point=ref_t)
                hvs = []
                for s_idx in range(n_mc_samples):
                    Y_s = Y_samples[s_idx]                      # (n_grid, n_obj)
                    nd_mask = is_non_dominated(Y_s)
                    if nd_mask.any():
                        hvs.append(hv_obj.compute(Y_s[nd_mask]))
                    else:
                        hvs.append(0.0)
                hvs_t = torch.tensor(hvs, dtype=torch.double)
                predicted_hv[i - 1] = float(hvs_t.mean())
                predicted_hv_std[i - 1] = float(hvs_t.std())

            except Exception as e:
                import traceback as tb
                logger.debug(f"Failed posterior HV at iteration {i}: {e}")
                logger.debug(tb.format_exc())

        # Restore logging
        logger.setLevel(orig_session_lvl)
        logging.getLogger('alchemist_core.models.botorch_model').setLevel(orig_model_lvl)

        n_success = int(np.count_nonzero(~np.isnan(predicted_hv)))
        n_total = n_exp - start_iteration + 1
        if n_success < n_total:
            logger.info(
                f"Posterior HV computed for {n_success}/{n_total} iterations "
                f"(failures at early iterations are normal with small data subsets)."
            )

        return predicted_hv, predicted_hv_std

    def _evaluate_mobo_acquisition(self, grid_df: pd.DataFrame) -> np.ndarray:
        """Evaluate stored MOBO acquisition function on a grid.

        Returns scalar acquisition values (e.g. expected hypervolume improvement)
        for each row in grid_df.  Requires suggest_next() to have been called
        previously so that self.acquisition.acq_function is populated.
        """
        import torch

        if self.acquisition is None or not hasattr(self.acquisition, 'acq_function'):
            raise ValueError(
                "No MOBO acquisition function stored. Call suggest_next() "
                "with a MOBO strategy (qEHVI/qNEHVI) first."
            )
        acq_fn = self.acquisition.acq_function
        if acq_fn is None:
            raise ValueError(
                "Acquisition function is None. Call suggest_next() first."
            )

        logger.warning(
            "Evaluating MOBO acquisition function on grid — "
            "this may be slow for large grids."
        )

        # Encode and convert to tensor
        if hasattr(self.model, '_encode_categorical_data'):
            grid_encoded = self.model._encode_categorical_data(grid_df)
            X_tensor = torch.tensor(grid_encoded.values, dtype=torch.float64)
        else:
            X_tensor = torch.tensor(grid_df.values, dtype=torch.float64)

        # qEHVI/qNEHVI expect (batch, q, d) shape; we evaluate q=1
        X_tensor = X_tensor.unsqueeze(1)  # (n, 1, d)

        with torch.no_grad():
            acq_values = acq_fn(X_tensor)

        return acq_values.cpu().numpy()

    def plot_acquisition_slice(
        self,
        x_var: str,
        acq_func: str = 'ei',
        fixed_values: Optional[Dict[str, Any]] = None,
        n_points: int = 100,
        acq_func_kwargs: Optional[Dict[str, Any]] = None,
        goal: str = 'maximize',
        show_experiments: bool = True,
        show_suggestions: bool = True,
        figsize: Tuple[float, float] = (8, 6),
        dpi: int = 100,
        title: Optional[str] = None
    ) -> Figure: # pyright: ignore[reportInvalidTypeForm]
        """
        Create 1D slice plot showing acquisition function along one variable.
        
        Visualizes how the acquisition function value changes as one variable is varied
        while all other variables are held constant. This shows which regions along that
        variable axis are most promising for the next experiment.
        
        Args:
            x_var: Variable name to vary along X axis (must be 'real' or 'integer')
            acq_func: Acquisition function name ('ei', 'pi', 'ucb', 'logei', 'logpi')
            fixed_values: Dict of {var_name: value} for other variables.
                         If not provided, uses midpoint for real/integer,
                         first category for categorical.
            n_points: Number of points to evaluate along the slice
            acq_func_kwargs: Additional acquisition parameters (xi, kappa, beta)
            goal: 'maximize' or 'minimize' - optimization direction
            show_experiments: Plot experimental data points as scatter
            show_suggestions: Plot last suggested points (if available)
            figsize: Figure size as (width, height) in inches
            dpi: Dots per inch for figure resolution
            title: Custom title (default: auto-generated)
        
        Returns:
            matplotlib Figure object
        
        Example:
            >>> # Visualize Expected Improvement along temperature
            >>> fig = session.plot_acquisition_slice(
            ...     'temperature',
            ...     acq_func='ei',
            ...     fixed_values={'pressure': 5.0, 'catalyst': 'Pt'}
            ... )
            >>> fig.savefig('acq_slice.png', dpi=300)
            
            >>> # See where UCB is highest
            >>> fig = session.plot_acquisition_slice(
            ...     'pressure',
            ...     acq_func='ucb',
            ...     acq_func_kwargs={'beta': 0.5}
            ... )
        
        Note:
            - Model must be trained before plotting
            - Higher acquisition values indicate more promising regions
            - Use this to understand where the algorithm wants to explore next
        """
        self._check_matplotlib()
        self._check_model_trained()
        
        from alchemist_core.utils.acquisition_utils import evaluate_acquisition
        from alchemist_core.visualization.plots import create_slice_plot
        
        if fixed_values is None:
            fixed_values = {}
        
        # Get variable info
        var_names = self.search_space.get_variable_names()
        if x_var not in var_names:
            raise ValueError(f"Variable '{x_var}' not in search space")
        
        # Get x variable definition
        x_var_def = next(v for v in self.search_space.variables if v['name'] == x_var)
        
        if x_var_def['type'] not in ['real', 'integer']:
            raise ValueError(f"Variable '{x_var}' must be 'real' or 'integer' type for slice plot")
        
        # Create range for x variable
        x_min, x_max = x_var_def['min'], x_var_def['max']
        x_values = np.linspace(x_min, x_max, n_points)
        
        # Build acquisition evaluation grid
        slice_data = {x_var: x_values}
        
        for var in self.search_space.variables:
            var_name = var['name']
            if var_name == x_var:
                continue
            
            if var_name in fixed_values:
                slice_data[var_name] = fixed_values[var_name]
            else:
                # Use default value
                if var['type'] in ['real', 'integer']:
                    slice_data[var_name] = (var['min'] + var['max']) / 2
                elif var['type'] == 'categorical':
                    slice_data[var_name] = var['values'][0]
        
        slice_df = self._build_grid_df(slice_data)

        # Evaluate acquisition function
        if self.is_multi_objective:
            acq_values = self._evaluate_mobo_acquisition(slice_df)
        else:
            acq_values, _ = evaluate_acquisition(
                self.model,
                slice_df,
                acq_func=acq_func,
                acq_func_kwargs=acq_func_kwargs,
                goal=goal
            )

        # Prepare experimental data for plotting
        exp_x = None
        exp_y = None
        if show_experiments and len(self.experiment_manager.df) > 0:
            df = self.experiment_manager.df

            # Filter points that match the fixed values
            mask = pd.Series([True] * len(df))
            for var_name, fixed_val in fixed_values.items():
                if var_name in df.columns:
                    if isinstance(fixed_val, str):
                        mask &= (df[var_name] == fixed_val)
                    else:
                        mask &= np.isclose(df[var_name], fixed_val, atol=1e-6)
            
            if mask.any():
                filtered_df = df[mask]
                exp_x = filtered_df[x_var].values
                # For acquisition, we just mark where experiments exist (no y-value)
                exp_y = np.zeros_like(exp_x)
        
        # Prepare suggestion data
        sugg_x = None
        if show_suggestions and len(self.last_suggestions) > 0:
            if isinstance(self.last_suggestions, pd.DataFrame):
                sugg_df = self.last_suggestions
            else:
                sugg_df = pd.DataFrame(self.last_suggestions)
            
            if x_var in sugg_df.columns:
                sugg_x = sugg_df[x_var].values
        
        # Generate title if not provided
        if title is None:
            acq_name = acq_func.upper()
            if fixed_values:
                fixed_str = ', '.join([f'{k}={v}' for k, v in fixed_values.items()])
                title = f"Acquisition Function ({acq_name}): {x_var}\n({fixed_str})"
            else:
                title = f"Acquisition Function ({acq_name}): {x_var}"
        
        # Use create_slice_plot but with acquisition values
        # Note: We pass None for std since acquisition functions are deterministic
        fig, ax = create_slice_plot(
            x_values=x_values,
            predictions=acq_values,
            x_var=x_var,
            std=None,
            sigma_bands=None,  # No uncertainty for acquisition
            exp_x=exp_x,
            exp_y=None,  # Don't show experiment y-values for acquisition
            figsize=figsize,
            dpi=dpi,
            title=title,
            prediction_label=acq_func.upper(),
            line_color='darkgreen',
            line_width=1.5
        )
        
        # Add green fill under acquisition curve
        ax.fill_between(x_values, 0, acq_values, alpha=0.3, color='green', zorder=0)
        
        # Update y-label for acquisition
        ax.set_ylabel(f'{acq_func.upper()} Value')
        
        # Mark suggestions with star markers if present
        if sugg_x is not None and len(sugg_x) > 0:
            # Evaluate acquisition at suggested points
            for i, sx in enumerate(sugg_x):
                # Find acquisition value at this x
                idx = np.argmin(np.abs(x_values - sx))
                sy = acq_values[idx]
                label = 'Suggestion' if i == 0 else None  # Only label first marker
                ax.scatter([sx], [sy], color='black', s=102, marker='*', zorder=10, label=label)
        
        logger.info(f"Generated acquisition slice plot for {x_var} using {acq_func}")
        return fig
    
    def plot_acquisition_contour(
        self,
        x_var: str,
        y_var: str,
        acq_func: str = 'ei',
        fixed_values: Optional[Dict[str, Any]] = None,
        grid_resolution: int = 50,
        acq_func_kwargs: Optional[Dict[str, Any]] = None,
        goal: str = 'maximize',
        show_experiments: bool = True,
        show_suggestions: bool = True,
        cmap: str = 'viridis',
        use_log_scale: Optional[bool] = None,
        figsize: Tuple[float, float] = (8, 6),
        dpi: int = 100,
        title: Optional[str] = None
    ) -> Figure: # pyright: ignore[reportInvalidTypeForm]
        """
        Create 2D contour plot of acquisition function over variable space.
        
        Visualizes the acquisition function surface by varying two variables
        while holding others constant. Shows "hot spots" where the algorithm
        believes the next experiment should be conducted. Higher values indicate
        more promising regions to explore.
        
        Args:
            x_var: Variable name for X axis (must be 'real' type)
            y_var: Variable name for Y axis (must be 'real' type)
            acq_func: Acquisition function name ('ei', 'pi', 'ucb', 'logei', 'logpi')
            fixed_values: Dict of {var_name: value} for other variables.
                         If not provided, uses midpoint for real/integer,
                         first category for categorical.
            grid_resolution: Grid density (NxN points)
            acq_func_kwargs: Additional acquisition parameters (xi, kappa, beta)
            goal: 'maximize' or 'minimize' - optimization direction
            show_experiments: Plot experimental data points as scatter
            show_suggestions: Plot last suggested points (if available)
            cmap: Matplotlib colormap name (e.g., 'viridis', 'hot', 'plasma')
            use_log_scale: Use logarithmic color scale (default: auto-enable for logei/logpi)
            figsize: Figure size as (width, height) in inches
            dpi: Dots per inch for figure resolution
            title: Custom title (default: auto-generated)
        
        Returns:
            matplotlib Figure object
        
        Example:
            >>> # Visualize Expected Improvement surface
            >>> fig = session.plot_acquisition_contour(
            ...     'temperature', 'pressure',
            ...     acq_func='ei'
            ... )
            >>> fig.savefig('acq_contour.png', dpi=300)
            
            >>> # See UCB landscape with custom exploration
            >>> fig = session.plot_acquisition_contour(
            ...     'temperature', 'pressure',
            ...     acq_func='ucb',
            ...     acq_func_kwargs={'beta': 1.0},
            ...     cmap='hot'
            ... )
        
        Note:
            - Requires at least 2 'real' type variables
            - Model must be trained before plotting
            - Higher acquisition values = more promising regions
            - Suggestions are overlaid to show why they were chosen
        """
        self._check_matplotlib()
        self._check_model_trained()
        
        from alchemist_core.utils.acquisition_utils import evaluate_acquisition
        from alchemist_core.visualization.plots import create_contour_plot
        
        if fixed_values is None:
            fixed_values = {}
        
        # Get variable names
        var_names = self.search_space.get_variable_names()
        
        # Validate variables exist
        if x_var not in var_names:
            raise ValueError(f"Variable '{x_var}' not in search space")
        if y_var not in var_names:
            raise ValueError(f"Variable '{y_var}' not in search space")
        
        # Get variable info
        x_var_info = next(v for v in self.search_space.variables if v['name'] == x_var)
        y_var_info = next(v for v in self.search_space.variables if v['name'] == y_var)
        
        if x_var_info['type'] != 'real':
            raise ValueError(f"X variable '{x_var}' must be 'real' type, got '{x_var_info['type']}'")
        if y_var_info['type'] != 'real':
            raise ValueError(f"Y variable '{y_var}' must be 'real' type, got '{y_var_info['type']}'")
        
        # Get bounds
        x_bounds = (x_var_info['min'], x_var_info['max'])
        y_bounds = (y_var_info['min'], y_var_info['max'])
        
        # Create meshgrid
        x = np.linspace(x_bounds[0], x_bounds[1], grid_resolution)
        y = np.linspace(y_bounds[0], y_bounds[1], grid_resolution)
        X_grid, Y_grid = np.meshgrid(x, y)
        
        # Build acquisition evaluation grid
        grid_data = {
            x_var: X_grid.ravel(),
            y_var: Y_grid.ravel()
        }
        
        # Add fixed values for other variables
        for var in self.search_space.variables:
            var_name = var['name']
            if var_name in [x_var, y_var]:
                continue
            
            if var_name in fixed_values:
                grid_data[var_name] = fixed_values[var_name]
            else:
                # Use default value
                if var['type'] in ['real', 'integer']:
                    grid_data[var_name] = (var['min'] + var['max']) / 2
                elif var['type'] == 'categorical':
                    grid_data[var_name] = var['values'][0]
        
        grid_df = self._build_grid_df(grid_data)

        # Evaluate acquisition function
        if self.is_multi_objective:
            acq_values = self._evaluate_mobo_acquisition(grid_df)
        else:
            acq_values, _ = evaluate_acquisition(
                self.model,
                grid_df,
                acq_func=acq_func,
                acq_func_kwargs=acq_func_kwargs,
                goal=goal
            )

        # Reshape to grid
        acq_grid = acq_values.reshape(X_grid.shape)
        
        # Prepare experimental data for overlay
        exp_x = None
        exp_y = None
        if show_experiments and not self.experiment_manager.df.empty:
            exp_df = self.experiment_manager.df
            if x_var in exp_df.columns and y_var in exp_df.columns:
                exp_x = exp_df[x_var].values
                exp_y = exp_df[y_var].values
        
        # Prepare suggestion data for overlay
        sugg_x = None
        sugg_y = None
        if show_suggestions and len(self.last_suggestions) > 0:
            if isinstance(self.last_suggestions, pd.DataFrame):
                sugg_df = self.last_suggestions
            else:
                sugg_df = pd.DataFrame(self.last_suggestions)
            
            if x_var in sugg_df.columns and y_var in sugg_df.columns:
                sugg_x = sugg_df[x_var].values
                sugg_y = sugg_df[y_var].values
        
        # Auto-enable log scale for logei/logpi if not explicitly set
        if use_log_scale is None:
            use_log_scale = acq_func.lower() in ['logei', 'logpi']
        
        # Generate title if not provided
        if title is None:
            acq_name = acq_func.upper()
            title = f"Acquisition Function ({acq_name}): {x_var} vs {y_var}"
        
        # Delegate to visualization module
        fig, ax, cbar = create_contour_plot(
            x_grid=X_grid,
            y_grid=Y_grid,
            predictions_grid=acq_grid,
            x_var=x_var,
            y_var=y_var,
            exp_x=exp_x,
            exp_y=exp_y,
            suggest_x=sugg_x,
            suggest_y=sugg_y,
            cmap='Greens',  # Green colormap for acquisition
            use_log_scale=use_log_scale,
            figsize=figsize,
            dpi=dpi,
            title=title
        )
        
        # Update colorbar label for acquisition
        cbar.set_label(f'{acq_func.upper()} Value', rotation=270, labelpad=20)
        
        logger.info(f"Generated acquisition contour plot for {x_var} vs {y_var} using {acq_func}")
        return fig
    
    def plot_uncertainty_contour(
        self,
        x_var: str,
        y_var: str,
        fixed_values: Optional[Dict[str, Any]] = None,
        grid_resolution: int = 50,
        show_experiments: bool = True,
        show_suggestions: bool = False,
        cmap: str = 'Reds',
        figsize: Tuple[float, float] = (8, 6),
        dpi: int = 100,
        title: Optional[str] = None,
        target_columns: Optional[str] = None
    ) -> Figure: # pyright: ignore[reportInvalidTypeForm]
        """
        Create 2D contour plot of posterior uncertainty over a variable space.
        
        Visualizes where the model is most uncertain about predictions, showing
        regions that may benefit from additional sampling. Higher values indicate
        greater uncertainty (standard deviation).
        
        Args:
            x_var: Variable name for X axis (must be 'real' type)
            y_var: Variable name for Y axis (must be 'real' type)
            fixed_values: Dict of {var_name: value} for other variables.
                         If not provided, uses midpoint for real/integer,
                         first category for categorical.
            grid_resolution: Grid density (NxN points)
            show_experiments: Plot experimental data points as scatter
            show_suggestions: Plot last suggested points (if available)
            cmap: Matplotlib colormap name (default: 'Reds' - darker = more uncertain)
            figsize: Figure size as (width, height) in inches
            dpi: Dots per inch for figure resolution
            title: Custom title (default: auto-generated)
        
        Returns:
            matplotlib Figure object
        
        Example:
            >>> # Visualize uncertainty landscape
            >>> fig = session.plot_uncertainty_contour('temperature', 'pressure')
            
            >>> # Custom colormap
            >>> fig = session.plot_uncertainty_contour(
            ...     'temperature', 'pressure',
            ...     cmap='YlOrRd',
            ...     grid_resolution=100
            ... )
            >>> fig.savefig('uncertainty_contour.png', dpi=300)
        
        Note:
            - Requires at least 2 'real' type variables
            - Model must be trained and support std predictions
            - High uncertainty near data gaps is expected
            - Useful for planning exploration strategies
        """
        self._check_matplotlib()
        self._check_model_trained()
        
        from alchemist_core.visualization.plots import create_uncertainty_contour_plot
        
        if fixed_values is None:
            fixed_values = {}
        
        # Get variable names
        var_names = self.search_space.get_variable_names()
        
        # Validate variables exist
        if x_var not in var_names:
            raise ValueError(f"Variable '{x_var}' not in search space")
        if y_var not in var_names:
            raise ValueError(f"Variable '{y_var}' not in search space")
        
        # Get variable info
        x_var_info = next(v for v in self.search_space.variables if v['name'] == x_var)
        y_var_info = next(v for v in self.search_space.variables if v['name'] == y_var)
        
        if x_var_info['type'] != 'real':
            raise ValueError(f"X variable '{x_var}' must be 'real' type, got '{x_var_info['type']}'")
        if y_var_info['type'] != 'real':
            raise ValueError(f"Y variable '{y_var}' must be 'real' type, got '{y_var_info['type']}'")
        
        # Get bounds
        x_bounds = (x_var_info['min'], x_var_info['max'])
        y_bounds = (y_var_info['min'], y_var_info['max'])
        
        # Create meshgrid
        x = np.linspace(x_bounds[0], x_bounds[1], grid_resolution)
        y = np.linspace(y_bounds[0], y_bounds[1], grid_resolution)
        X_grid, Y_grid = np.meshgrid(x, y)
        
        # Build prediction grid
        grid_data = {
            x_var: X_grid.ravel(),
            y_var: Y_grid.ravel()
        }
        
        # Add fixed values for other variables
        for var in self.search_space.variables:
            var_name = var['name']
            if var_name in [x_var, y_var]:
                continue
            
            if var_name in fixed_values:
                grid_data[var_name] = fixed_values[var_name]
            else:
                # Use default value
                if var['type'] in ['real', 'integer']:
                    grid_data[var_name] = (var['min'] + var['max']) / 2
                elif var['type'] == 'categorical':
                    grid_data[var_name] = var['values'][0]
        
        grid_df = self._build_grid_df(grid_data)

        # Get predictions
        predict_result = self.predict(grid_df)

        # Prepare experimental data for overlay
        exp_x = None
        exp_y = None
        if show_experiments and not self.experiment_manager.df.empty:
            exp_df = self.experiment_manager.df
            if x_var in exp_df.columns and y_var in exp_df.columns:
                exp_x = exp_df[x_var].values
                exp_y = exp_df[y_var].values

        # Prepare suggestion data for overlay
        sugg_x = None
        sugg_y = None
        if show_suggestions and len(self.last_suggestions) > 0:
            if isinstance(self.last_suggestions, pd.DataFrame):
                sugg_df = self.last_suggestions
            else:
                sugg_df = pd.DataFrame(self.last_suggestions)
            if x_var in sugg_df.columns and y_var in sugg_df.columns:
                sugg_x = sugg_df[x_var].values
                sugg_y = sugg_df[y_var].values

        resolved = self._resolve_target_column(target_columns)

        def _unc_contour_for_obj(obj_name, ax=None):
            _, std = self._get_predictions_for_objective(predict_result, obj_name)
            unc_grid = std.reshape(X_grid.shape)
            obj_title = title or (f"Uncertainty: {obj_name} ({x_var} vs {y_var})"
                                   if self.is_multi_objective
                                   else f"Posterior Uncertainty: {x_var} vs {y_var}")
            return create_uncertainty_contour_plot(
                x_grid=X_grid, y_grid=Y_grid, uncertainty_grid=unc_grid,
                x_var=x_var, y_var=y_var, exp_x=exp_x, exp_y=exp_y,
                suggest_x=sugg_x, suggest_y=sugg_y, cmap=cmap,
                figsize=figsize, dpi=dpi, title=obj_title, ax=ax
            )

        if isinstance(resolved, list):
            n = len(resolved)
            fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n, figsize[1]), dpi=dpi)
            if n == 1:
                axes = [axes]
            for i, obj in enumerate(resolved):
                _unc_contour_for_obj(obj, ax=axes[i])
            fig.tight_layout()
            logger.info(f"Generated multi-objective uncertainty contour plot")
            return fig

        fig, ax, cbar = _unc_contour_for_obj(resolved)
        logger.info(f"Generated uncertainty contour plot for {x_var} vs {y_var}")
        return fig
    
    def plot_uncertainty_voxel(
        self,
        x_var: str,
        y_var: str,
        z_var: str,
        fixed_values: Optional[Dict[str, Any]] = None,
        grid_resolution: int = 15,
        show_experiments: bool = True,
        show_suggestions: bool = False,
        cmap: str = 'Reds',
        alpha: float = 0.5,
        figsize: Tuple[float, float] = (10, 8),
        dpi: int = 100,
        title: Optional[str] = None,
        target_columns: Optional[str] = None
    ) -> Figure: # pyright: ignore[reportInvalidTypeForm]
        """
        Create 3D voxel plot of posterior uncertainty over variable space.
        
        Visualizes where the model is most uncertain in 3D, helping identify
        under-explored regions that may benefit from additional sampling.
        Higher values indicate greater uncertainty (standard deviation).
        
        Args:
            x_var: Variable name for X axis (must be 'real' or 'integer' type)
            y_var: Variable name for Y axis (must be 'real' or 'integer' type)
            z_var: Variable name for Z axis (must be 'real' or 'integer' type)
            fixed_values: Dict of {var_name: value} for other variables.
                         If not provided, uses midpoint for real/integer,
                         first category for categorical.
            grid_resolution: Grid density (NxNxN points, default: 15)
            show_experiments: Plot experimental data points as scatter
            show_suggestions: Plot last suggested points (if available)
            cmap: Matplotlib colormap name (default: 'Reds')
            alpha: Transparency level (0=transparent, 1=opaque)
            figsize: Figure size as (width, height) in inches
            dpi: Dots per inch for figure resolution
            title: Custom title (default: auto-generated)
        
        Returns:
            matplotlib Figure object with 3D axes
        
        Example:
            >>> # Visualize uncertainty in 3D
            >>> fig = session.plot_uncertainty_voxel('temperature', 'pressure', 'flow_rate')
            
            >>> # With transparency to see interior
            >>> fig = session.plot_uncertainty_voxel(
            ...     'temperature', 'pressure', 'flow_rate',
            ...     alpha=0.3,
            ...     grid_resolution=20
            ... )
            >>> fig.savefig('uncertainty_voxel.png', dpi=150)
        
        Raises:
            ValueError: If search space doesn't have at least 3 continuous variables
        
        Note:
            - Requires at least 3 'real' or 'integer' type variables
            - Model must be trained and support std predictions
            - Computationally expensive: O(N³) evaluations
            - Useful for planning exploration in 3D space
        """
        self._check_matplotlib()
        self._check_model_trained()
        
        from alchemist_core.visualization.plots import create_uncertainty_voxel_plot
        
        if fixed_values is None:
            fixed_values = {}
        
        # Get all variable names
        var_names = self.search_space.get_variable_names()
        
        # Validate that the requested variables exist and are continuous
        for var_name, var_label in [(x_var, 'X'), (y_var, 'Y'), (z_var, 'Z')]:
            if var_name not in var_names:
                raise ValueError(f"{var_label} variable '{var_name}' not in search space")
            
            var_def = next(v for v in self.search_space.variables if v['name'] == var_name)
            if var_def['type'] not in ['real', 'integer']:
                raise ValueError(
                    f"{var_label} variable '{var_name}' must be 'real' or 'integer' type for voxel plot, "
                    f"got '{var_def['type']}'"
                )
        
        # Get variable definitions
        x_var_def = next(v for v in self.search_space.variables if v['name'] == x_var)
        y_var_def = next(v for v in self.search_space.variables if v['name'] == y_var)
        z_var_def = next(v for v in self.search_space.variables if v['name'] == z_var)
        
        # Get bounds
        x_bounds = (x_var_def['min'], x_var_def['max'])
        y_bounds = (y_var_def['min'], y_var_def['max'])
        z_bounds = (z_var_def['min'], z_var_def['max'])
        
        # Create 3D meshgrid
        x = np.linspace(x_bounds[0], x_bounds[1], grid_resolution)
        y = np.linspace(y_bounds[0], y_bounds[1], grid_resolution)
        z = np.linspace(z_bounds[0], z_bounds[1], grid_resolution)
        X_grid, Y_grid, Z_grid = np.meshgrid(x, y, z, indexing='ij')
        
        # Build prediction grid
        grid_data = {
            x_var: X_grid.ravel(),
            y_var: Y_grid.ravel(),
            z_var: Z_grid.ravel()
        }
        
        # Add fixed values for other variables
        for var in self.search_space.variables:
            var_name = var['name']
            if var_name in [x_var, y_var, z_var]:
                continue
            
            if var_name in fixed_values:
                grid_data[var_name] = fixed_values[var_name]
            else:
                # Use default value
                if var['type'] in ['real', 'integer']:
                    grid_data[var_name] = (var['min'] + var['max']) / 2
                elif var['type'] == 'categorical':
                    grid_data[var_name] = var['values'][0]
        
        grid_df = self._build_grid_df(grid_data)

        # Get predictions
        predict_result = self.predict(grid_df)

        # Prepare experimental data for overlay
        exp_x, exp_y, exp_z = None, None, None
        if show_experiments and not self.experiment_manager.df.empty:
            exp_df = self.experiment_manager.df
            if x_var in exp_df.columns and y_var in exp_df.columns and z_var in exp_df.columns:
                exp_x = exp_df[x_var].values
                exp_y = exp_df[y_var].values
                exp_z = exp_df[z_var].values

        # Prepare suggestion data for overlay
        sugg_x, sugg_y, sugg_z = None, None, None
        if show_suggestions and len(self.last_suggestions) > 0:
            if isinstance(self.last_suggestions, pd.DataFrame):
                sugg_df = self.last_suggestions
            else:
                sugg_df = pd.DataFrame(self.last_suggestions)
            if x_var in sugg_df.columns and y_var in sugg_df.columns and z_var in sugg_df.columns:
                sugg_x = sugg_df[x_var].values
                sugg_y = sugg_df[y_var].values
                sugg_z = sugg_df[z_var].values

        resolved = self._resolve_target_column(target_columns)

        def _unc_voxel_for_obj(obj_name):
            _, std = self._get_predictions_for_objective(predict_result, obj_name)
            unc_grid = std.reshape(X_grid.shape)
            obj_title = title or (f"3D Uncertainty: {obj_name}"
                                   if self.is_multi_objective
                                   else f"3D Posterior Uncertainty: {x_var} vs {y_var} vs {z_var}")
            return create_uncertainty_voxel_plot(
                x_grid=X_grid, y_grid=Y_grid, z_grid=Z_grid,
                uncertainty_grid=unc_grid, x_var=x_var, y_var=y_var, z_var=z_var,
                exp_x=exp_x, exp_y=exp_y, exp_z=exp_z,
                suggest_x=sugg_x, suggest_y=sugg_y, suggest_z=sugg_z,
                cmap=cmap, alpha=alpha, figsize=figsize, dpi=dpi, title=obj_title
            )

        if isinstance(resolved, list):
            logger.warning("Voxel 'all' mode creates separate figures per objective; returning last")
            fig = None
            for obj in resolved:
                fig, ax = _unc_voxel_for_obj(obj)
            return fig

        fig, ax = _unc_voxel_for_obj(resolved)
        logger.info(f"Generated 3D uncertainty voxel plot for {x_var} vs {y_var} vs {z_var}")
        return fig
    
    def plot_acquisition_voxel(
        self,
        x_var: str,
        y_var: str,
        z_var: str,
        acq_func: str = 'ei',
        fixed_values: Optional[Dict[str, Any]] = None,
        grid_resolution: int = 15,
        acq_func_kwargs: Optional[Dict[str, Any]] = None,
        goal: str = 'maximize',
        show_experiments: bool = True,
        show_suggestions: bool = True,
        cmap: str = 'hot',
        alpha: float = 0.5,
        use_log_scale: Optional[bool] = None,
        figsize: Tuple[float, float] = (10, 8),
        dpi: int = 100,
        title: Optional[str] = None
    ) -> Figure: # pyright: ignore[reportInvalidTypeForm]
        """
        Create 3D voxel plot of acquisition function over variable space.
        
        Visualizes the acquisition function in 3D, showing "hot spots" where
        the optimization algorithm believes the next experiment should be conducted.
        Higher values indicate more promising regions.
        
        Args:
            x_var: Variable name for X axis (must be 'real' or 'integer' type)
            y_var: Variable name for Y axis (must be 'real' or 'integer' type)
            z_var: Variable name for Z axis (must be 'real' or 'integer' type)
            acq_func: Acquisition function name ('ei', 'pi', 'ucb', 'logei', 'logpi')
            fixed_values: Dict of {var_name: value} for other variables.
                         If not provided, uses midpoint for real/integer,
                         first category for categorical.
            grid_resolution: Grid density (NxNxN points, default: 15)
            acq_func_kwargs: Additional acquisition parameters (xi, kappa, beta)
            goal: 'maximize' or 'minimize' - optimization direction
            show_experiments: Plot experimental data points as scatter
            show_suggestions: Plot last suggested points (if available)
            cmap: Matplotlib colormap name (default: 'hot')
            alpha: Transparency level (0=transparent, 1=opaque)
            use_log_scale: Use logarithmic color scale (default: auto for logei/logpi)
            figsize: Figure size as (width, height) in inches
            dpi: Dots per inch for figure resolution
            title: Custom title (default: auto-generated)
        
        Returns:
            matplotlib Figure object with 3D axes
        
        Example:
            >>> # Visualize Expected Improvement in 3D
            >>> fig = session.plot_acquisition_voxel(
            ...     'temperature', 'pressure', 'flow_rate',
            ...     acq_func='ei'
            ... )
            
            >>> # UCB with custom exploration
            >>> fig = session.plot_acquisition_voxel(
            ...     'temperature', 'pressure', 'flow_rate',
            ...     acq_func='ucb',
            ...     acq_func_kwargs={'beta': 1.0},
            ...     alpha=0.3
            ... )
            >>> fig.savefig('acq_voxel.png', dpi=150)
        
        Raises:
            ValueError: If search space doesn't have at least 3 continuous variables
        
        Note:
            - Requires at least 3 'real' or 'integer' type variables
            - Model must be trained before plotting
            - Computationally expensive: O(N³) evaluations
            - Higher values = more promising for next experiment
            - Suggestions should align with high-value regions
        """
        self._check_matplotlib()
        self._check_model_trained()
        
        from alchemist_core.utils.acquisition_utils import evaluate_acquisition
        from alchemist_core.visualization.plots import create_acquisition_voxel_plot
        
        if fixed_values is None:
            fixed_values = {}
        
        # Get all variable names
        var_names = self.search_space.get_variable_names()
        
        # Validate that the requested variables exist and are continuous
        for var_name, var_label in [(x_var, 'X'), (y_var, 'Y'), (z_var, 'Z')]:
            if var_name not in var_names:
                raise ValueError(f"{var_label} variable '{var_name}' not in search space")
            
            var_def = next(v for v in self.search_space.variables if v['name'] == var_name)
            if var_def['type'] not in ['real', 'integer']:
                raise ValueError(
                    f"{var_label} variable '{var_name}' must be 'real' or 'integer' type for voxel plot, "
                    f"got '{var_def['type']}'"
                )
        
        # Get variable definitions
        x_var_def = next(v for v in self.search_space.variables if v['name'] == x_var)
        y_var_def = next(v for v in self.search_space.variables if v['name'] == y_var)
        z_var_def = next(v for v in self.search_space.variables if v['name'] == z_var)
        
        # Get bounds
        x_bounds = (x_var_def['min'], x_var_def['max'])
        y_bounds = (y_var_def['min'], y_var_def['max'])
        z_bounds = (z_var_def['min'], z_var_def['max'])
        
        # Create 3D meshgrid
        x = np.linspace(x_bounds[0], x_bounds[1], grid_resolution)
        y = np.linspace(y_bounds[0], y_bounds[1], grid_resolution)
        z = np.linspace(z_bounds[0], z_bounds[1], grid_resolution)
        X_grid, Y_grid, Z_grid = np.meshgrid(x, y, z, indexing='ij')
        
        # Build acquisition evaluation grid
        grid_data = {
            x_var: X_grid.ravel(),
            y_var: Y_grid.ravel(),
            z_var: Z_grid.ravel()
        }
        
        # Add fixed values for other variables
        for var in self.search_space.variables:
            var_name = var['name']
            if var_name in [x_var, y_var, z_var]:
                continue
            
            if var_name in fixed_values:
                grid_data[var_name] = fixed_values[var_name]
            else:
                # Use default value
                if var['type'] in ['real', 'integer']:
                    grid_data[var_name] = (var['min'] + var['max']) / 2
                elif var['type'] == 'categorical':
                    grid_data[var_name] = var['values'][0]
        
        grid_df = self._build_grid_df(grid_data)

        # Evaluate acquisition function
        if self.is_multi_objective:
            acq_values = self._evaluate_mobo_acquisition(grid_df)
        else:
            acq_values, _ = evaluate_acquisition(
                self.model,
                grid_df,
                acq_func=acq_func,
                acq_func_kwargs=acq_func_kwargs,
                goal=goal
            )

        # Reshape to 3D grid
        acquisition_grid = acq_values.reshape(X_grid.shape)

        # Prepare experimental data for overlay
        exp_x = None
        exp_y = None
        exp_z = None
        if show_experiments and not self.experiment_manager.df.empty:
            exp_df = self.experiment_manager.df
            if x_var in exp_df.columns and y_var in exp_df.columns and z_var in exp_df.columns:
                exp_x = exp_df[x_var].values
                exp_y = exp_df[y_var].values
                exp_z = exp_df[z_var].values
        
        # Prepare suggestion data for overlay
        sugg_x = None
        sugg_y = None
        sugg_z = None
        if show_suggestions and len(self.last_suggestions) > 0:
            if isinstance(self.last_suggestions, pd.DataFrame):
                sugg_df = self.last_suggestions
            else:
                sugg_df = pd.DataFrame(self.last_suggestions)
            
            if x_var in sugg_df.columns and y_var in sugg_df.columns and z_var in sugg_df.columns:
                sugg_x = sugg_df[x_var].values
                sugg_y = sugg_df[y_var].values
                sugg_z = sugg_df[z_var].values
        
        # Auto-enable log scale for logei/logpi if not explicitly set
        if use_log_scale is None:
            use_log_scale = acq_func.lower() in ['logei', 'logpi']
        
        # Generate title if not provided
        if title is None:
            acq_name = acq_func.upper()
            title = f"3D Acquisition Function ({acq_name}): {x_var} vs {y_var} vs {z_var}"
        
        # Delegate to visualization module
        fig, ax = create_acquisition_voxel_plot(
            x_grid=X_grid,
            y_grid=Y_grid,
            z_grid=Z_grid,
            acquisition_grid=acquisition_grid,
            x_var=x_var,
            y_var=y_var,
            z_var=z_var,
            exp_x=exp_x,
            exp_y=exp_y,
            exp_z=exp_z,
            suggest_x=sugg_x,
            suggest_y=sugg_y,
            suggest_z=sugg_z,
            cmap=cmap,
            alpha=alpha,
            use_log_scale=use_log_scale,
            figsize=figsize,
            dpi=dpi,
            title=title
        )
        
        logger.info(f"Generated 3D acquisition voxel plot for {x_var} vs {y_var} vs {z_var} using {acq_func}")
        return fig
    
    def plot_suggested_next(
        self,
        x_var: str,
        y_var: Optional[str] = None,
        z_var: Optional[str] = None,
        acq_func: Optional[str] = None,
        fixed_values: Optional[Dict[str, Any]] = None,
        suggestion_index: int = 0,
        n_points: int = 100,
        grid_resolution: int = 50,
        show_uncertainty: Optional[Union[bool, List[float]]] = [1.0, 2.0],
        show_experiments: bool = True,
        acq_func_kwargs: Optional[Dict[str, Any]] = None,
        goal: Optional[str] = None,
        figsize: Tuple[float, float] = (10, 12),
        dpi: int = 100,
        title_prefix: Optional[str] = None
    ) -> Figure: # pyright: ignore[reportInvalidTypeForm]
        """
        Create visualization of suggested next experiment with posterior and acquisition.
        
        This creates a stacked subplot showing:
        - Top: Posterior mean prediction (slice/contour/voxel)
        - Bottom: Acquisition function with suggested point marked
        
        The fixed values for non-varying dimensions are automatically extracted from
        the suggested point coordinates, making it easy to visualize why that point
        was chosen.
        
        Args:
            x_var: Variable name for X axis (required)
            y_var: Variable name for Y axis (optional, creates 2D plot if provided)
            z_var: Variable name for Z axis (optional, creates 3D plot if provided with y_var)
            acq_func: Acquisition function used (if None, extracts from last run or defaults to 'ei')
            fixed_values: Override automatic fixed values from suggestion (optional)
            suggestion_index: Which suggestion to visualize if multiple (default: 0 = most recent)
            n_points: Points to evaluate for 1D slice (default: 100)
            grid_resolution: Grid density for 2D/3D plots (default: 50)
            show_uncertainty: For posterior plot - True, False, or list of sigma values (e.g., [1.0, 2.0])
            show_experiments: Overlay experimental data points
            acq_func_kwargs: Additional acquisition parameters (xi, kappa, beta)
            goal: 'maximize' or 'minimize' (if None, uses session's last goal)
            figsize: Figure size as (width, height) in inches
            dpi: Dots per inch
            title_prefix: Custom prefix for titles (default: auto-generated)
        
        Returns:
            matplotlib Figure object with 2 subplots
        
        Example:
            >>> # After running suggest_next()
            >>> session.suggest_next(strategy='ei')
            >>> 
            >>> # Visualize the suggestion in 1D
            >>> fig = session.plot_suggested_next('temperature')
            >>> 
            >>> # Visualize in 2D
            >>> fig = session.plot_suggested_next('temperature', 'pressure')
            >>> 
            >>> # Visualize in 3D
            >>> fig = session.plot_suggested_next('temperature', 'pressure', 'time')
            >>> fig.savefig('suggestion_3d.png', dpi=300)
        
        Note:
            - Must call suggest_next() before using this function
            - Automatically extracts fixed values from the suggested point
            - Creates intuitive visualization showing why the point was chosen
        """
        self._check_matplotlib()
        self._check_model_trained()

        # Check if we have suggestions
        if not self.last_suggestions or len(self.last_suggestions) == 0:
            raise ValueError("No suggestions available. Call suggest_next() first.")

        # Get the suggestion to visualize
        if isinstance(self.last_suggestions, pd.DataFrame):
            sugg_df = self.last_suggestions
        else:
            sugg_df = pd.DataFrame(self.last_suggestions)

        if suggestion_index >= len(sugg_df):
            raise ValueError(f"Suggestion index {suggestion_index} out of range (have {len(sugg_df)} suggestions)")

        # MOBO: route to Pareto plot with suggestions overlaid in objective space
        if self.is_multi_objective:
            # Predict objective values for all suggestions
            pred_dict = self.predict(sugg_df)
            suggested_points = np.column_stack(
                [pred_dict[obj][0] for obj in self.objective_names]
            )
            directions = None
            if goal is not None:
                if isinstance(goal, str):
                    directions = [goal.lower()] * self.n_objectives
                elif isinstance(goal, list):
                    directions = [g.lower() for g in goal]

            return self.plot_pareto_frontier(
                directions=directions,
                suggested_points_override=suggested_points,
                figsize=figsize,
                dpi=dpi,
                title=title_prefix or "Suggested Next Experiments (Pareto)",
            )

        suggestion = sugg_df.iloc[suggestion_index].to_dict()
        
        # Determine plot dimensionality
        if z_var is not None and y_var is None:
            raise ValueError("Must provide y_var if z_var is specified")
        
        is_1d = (y_var is None)
        is_2d = (y_var is not None and z_var is None)
        is_3d = (z_var is not None)
        
        # Cap 3D resolution to prevent kernel crashes
        if is_3d and grid_resolution > 30:
            logger.warning(f"3D voxel resolution capped at 30 (requested {grid_resolution})")
            grid_resolution = 30
        
        # Get variable names for the plot
        plot_vars = [x_var]
        if y_var is not None:
            plot_vars.append(y_var)
        if z_var is not None:
            plot_vars.append(z_var)
        
        # Extract fixed values from suggestion (for non-varying dimensions)
        if fixed_values is None:
            fixed_values = {}
            for var_name in self.search_space.get_variable_names():
                if var_name not in plot_vars and var_name in suggestion:
                    fixed_values[var_name] = suggestion[var_name]
        
        # Get acquisition function and goal from last run if not specified
        if acq_func is None:
            # Try to get from last acquisition run
            if hasattr(self, '_last_acq_func'):
                acq_func = self._last_acq_func
            else:
                acq_func = 'ei'  # Default fallback
        
        if goal is None:
            if hasattr(self, '_last_goal'):
                goal = self._last_goal
            else:
                goal = 'maximize'  # Default fallback
        
        # Create figure with 2 subplots (stacked vertically)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, dpi=dpi)
        
        # Generate titles
        if title_prefix is None:
            title_prefix = "Suggested Next Experiment"
        
        # Format fixed values with smart rounding (2 decimals for floats, no .00 for integers)
        def format_value(v):
            if isinstance(v, float):
                # Round to 2 decimals, but strip trailing zeros
                rounded = round(v, 2)
                # Check if it's effectively an integer
                if rounded == int(rounded):
                    return str(int(rounded))
                return f"{rounded:.2f}".rstrip('0').rstrip('.')
            return str(v)
        
        fixed_str = ', '.join([f'{k}={format_value(v)}' for k, v in fixed_values.items()])
        
        # Plot 1: Posterior Mean
        if is_1d:
            # 1D slice plot
            x_var_def = next(v for v in self.search_space.variables if v['name'] == x_var)
            x_values = np.linspace(x_var_def['min'], x_var_def['max'], n_points)
            
            # Build grid
            grid_data = {x_var: x_values}
            
            for var in self.search_space.variables:
                var_name = var['name']
                if var_name == x_var:
                    continue
                
                if var_name in fixed_values:
                    grid_data[var_name] = fixed_values[var_name]
                else:
                    if var['type'] in ['real', 'integer']:
                        grid_data[var_name] = (var['min'] + var['max']) / 2
                    elif var['type'] == 'categorical':
                        grid_data[var_name] = var['values'][0]
            
            grid_df = self._build_grid_df(grid_data)

            # Get predictions
            predictions, std = self.predict(grid_df)
            
            # Prepare experiment overlay
            exp_x, exp_y = None, None
            if show_experiments and not self.experiment_manager.df.empty:
                df = self.experiment_manager.df
                mask = pd.Series([True] * len(df))
                for var_name, fixed_val in fixed_values.items():
                    if var_name in df.columns:
                        if isinstance(fixed_val, str):
                            mask &= (df[var_name] == fixed_val)
                        else:
                            mask &= np.isclose(df[var_name], fixed_val, atol=1e-6)
                if mask.any():
                    filtered_df = df[mask]
                    exp_x = filtered_df[x_var].values
                    exp_y = filtered_df[self.experiment_manager.target_columns[0]].values
            
            # Determine sigma bands
            sigma_bands = None
            if show_uncertainty is not None:
                if isinstance(show_uncertainty, bool):
                    sigma_bands = [1.0, 2.0] if show_uncertainty else None
                else:
                    sigma_bands = show_uncertainty
            
            from alchemist_core.visualization.plots import create_slice_plot
            create_slice_plot(
                x_values=x_values,
                predictions=predictions,
                x_var=x_var,
                std=std,
                sigma_bands=sigma_bands,
                exp_x=exp_x,
                exp_y=exp_y,
                title=f"{title_prefix} - Posterior Mean\n({fixed_str})" if fixed_str else f"{title_prefix} - Posterior Mean",
                ax=ax1
            )
            
            # Mark the suggested point on posterior plot
            sugg_x = suggestion[x_var]
            sugg_y_pred, _ = self.predict(pd.DataFrame([suggestion]))
            ax1.scatter([sugg_x], sugg_y_pred, color='black', s=102, marker='*', zorder=10, 
                       linewidths=1.5, label='Suggested')
            ax1.legend()
            
        elif is_2d:
            # 2D contour plot
            x_var_def = next(v for v in self.search_space.variables if v['name'] == x_var)
            y_var_def = next(v for v in self.search_space.variables if v['name'] == y_var)
            
            x_values = np.linspace(x_var_def['min'], x_var_def['max'], grid_resolution)
            y_values = np.linspace(y_var_def['min'], y_var_def['max'], grid_resolution)
            X_grid, Y_grid = np.meshgrid(x_values, y_values)
            
            grid_data = {
                x_var: X_grid.ravel(),
                y_var: Y_grid.ravel()
            }
            
            for var in self.search_space.variables:
                var_name = var['name']
                if var_name in [x_var, y_var]:
                    continue
                
                if var_name in fixed_values:
                    grid_data[var_name] = fixed_values[var_name]
                else:
                    if var['type'] in ['real', 'integer']:
                        grid_data[var_name] = (var['min'] + var['max']) / 2
                    elif var['type'] == 'categorical':
                        grid_data[var_name] = var['values'][0]
            
            grid_df = self._build_grid_df(grid_data)

            predictions, _ = self.predict(grid_df)
            prediction_grid = predictions.reshape(X_grid.shape)

            # Prepare overlays
            exp_x, exp_y = None, None
            if show_experiments and not self.experiment_manager.df.empty:
                exp_df = self.experiment_manager.df
                if x_var in exp_df.columns and y_var in exp_df.columns:
                    exp_x = exp_df[x_var].values
                    exp_y = exp_df[y_var].values
            
            from alchemist_core.visualization.plots import create_contour_plot
            _, _, _ = create_contour_plot(
                x_grid=X_grid,
                y_grid=Y_grid,
                predictions_grid=prediction_grid,
                x_var=x_var,
                y_var=y_var,
                exp_x=exp_x,
                exp_y=exp_y,
                suggest_x=None,
                suggest_y=None,
                title=f"{title_prefix} - Posterior Mean\n({fixed_str})" if fixed_str else f"{title_prefix} - Posterior Mean",
                ax=ax1
            )
            
            # Mark the suggested point
            sugg_x = suggestion[x_var]
            sugg_y = suggestion[y_var]
            ax1.scatter([sugg_x], [sugg_y], color='black', s=102, marker='*', zorder=10, 
                       linewidths=1.5, label='Suggested')
            ax1.legend()
            
        else:  # 3D
            # 3D voxel plot
            x_var_def = next(v for v in self.search_space.variables if v['name'] == x_var)
            y_var_def = next(v for v in self.search_space.variables if v['name'] == y_var)
            z_var_def = next(v for v in self.search_space.variables if v['name'] == z_var)
            
            x_values = np.linspace(x_var_def['min'], x_var_def['max'], grid_resolution)
            y_values = np.linspace(y_var_def['min'], y_var_def['max'], grid_resolution)
            z_values = np.linspace(z_var_def['min'], z_var_def['max'], grid_resolution)
            X_grid, Y_grid, Z_grid = np.meshgrid(x_values, y_values, z_values, indexing='ij')
            
            grid_data = {
                x_var: X_grid.ravel(),
                y_var: Y_grid.ravel(),
                z_var: Z_grid.ravel()
            }
            
            for var in self.search_space.variables:
                var_name = var['name']
                if var_name in [x_var, y_var, z_var]:
                    continue
                
                if var_name in fixed_values:
                    grid_data[var_name] = fixed_values[var_name]
                else:
                    if var['type'] in ['real', 'integer']:
                        grid_data[var_name] = (var['min'] + var['max']) / 2
                    elif var['type'] == 'categorical':
                        grid_data[var_name] = var['values'][0]
            
            grid_df = self._build_grid_df(grid_data)

            predictions, _ = self.predict(grid_df)
            prediction_grid = predictions.reshape(X_grid.shape)

            # Prepare overlays
            exp_x, exp_y, exp_z = None, None, None
            if show_experiments and not self.experiment_manager.df.empty:
                exp_df = self.experiment_manager.df
                if all(v in exp_df.columns for v in [x_var, y_var, z_var]):
                    exp_x = exp_df[x_var].values
                    exp_y = exp_df[y_var].values
                    exp_z = exp_df[z_var].values
            
            from alchemist_core.visualization.plots import create_voxel_plot
            # Note: voxel plots don't support ax parameter yet, need to create separately
            # For now, we'll note this limitation
            logger.warning("3D voxel plots for suggestions not yet fully supported with subplots")
            ax1.text(0.5, 0.5, "3D voxel posterior visualization\n(use plot_voxel separately)",
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.axis('off')
        
        # Plot 2: Acquisition Function
        if is_1d:
            # 1D acquisition slice
            from alchemist_core.utils.acquisition_utils import evaluate_acquisition
            from alchemist_core.visualization.plots import create_slice_plot
            
            x_var_def = next(v for v in self.search_space.variables if v['name'] == x_var)
            x_values = np.linspace(x_var_def['min'], x_var_def['max'], n_points)
            
            grid_data = {x_var: x_values}
            
            for var in self.search_space.variables:
                var_name = var['name']
                if var_name == x_var:
                    continue
                
                if var_name in fixed_values:
                    grid_data[var_name] = fixed_values[var_name]
                else:
                    if var['type'] in ['real', 'integer']:
                        grid_data[var_name] = (var['min'] + var['max']) / 2
                    elif var['type'] == 'categorical':
                        grid_data[var_name] = var['values'][0]
            
            grid_df = self._build_grid_df(grid_data)

            acq_values, _ = evaluate_acquisition(
                self.model,
                grid_df,
                acq_func=acq_func,
                acq_func_kwargs=acq_func_kwargs,
                goal=goal
            )

            create_slice_plot(
                x_values=x_values,
                predictions=acq_values,
                x_var=x_var,
                std=None,
                sigma_bands=None,
                exp_x=None,
                exp_y=None,
                title=None,  # No title for acquisition subplot
                ax=ax2,
                prediction_label=acq_func.upper(),
                line_color='darkgreen',
                line_width=1.5
            )
            
            # Add green fill under acquisition curve
            ax2.fill_between(x_values, 0, acq_values, alpha=0.3, color='green', zorder=0)
            
            ax2.set_ylabel(f'{acq_func.upper()} Value')
            
            # Mark the suggested point
            sugg_x = suggestion[x_var]
            # Evaluate acquisition at the suggested point
            sugg_acq, _ = evaluate_acquisition(
                self.model,
                pd.DataFrame([suggestion]),
                acq_func=acq_func,
                acq_func_kwargs=acq_func_kwargs,
                goal=goal
            )
            ax2.scatter([sugg_x], sugg_acq, color='black', s=102, marker='*', zorder=10,
                       linewidths=1.5, label=f'{acq_func.upper()} (suggested)')
            ax2.legend()
            
        elif is_2d:
            # 2D acquisition contour
            from alchemist_core.utils.acquisition_utils import evaluate_acquisition
            from alchemist_core.visualization.plots import create_contour_plot
            
            x_var_def = next(v for v in self.search_space.variables if v['name'] == x_var)
            y_var_def = next(v for v in self.search_space.variables if v['name'] == y_var)
            
            x_values = np.linspace(x_var_def['min'], x_var_def['max'], grid_resolution)
            y_values = np.linspace(y_var_def['min'], y_var_def['max'], grid_resolution)
            X_grid, Y_grid = np.meshgrid(x_values, y_values)
            
            grid_data = {
                x_var: X_grid.ravel(),
                y_var: Y_grid.ravel()
            }
            
            for var in self.search_space.variables:
                var_name = var['name']
                if var_name in [x_var, y_var]:
                    continue
                
                if var_name in fixed_values:
                    grid_data[var_name] = fixed_values[var_name]
                else:
                    if var['type'] in ['real', 'integer']:
                        grid_data[var_name] = (var['min'] + var['max']) / 2
                    elif var['type'] == 'categorical':
                        grid_data[var_name] = var['values'][0]
            
            grid_df = self._build_grid_df(grid_data)

            acq_values, _ = evaluate_acquisition(
                self.model,
                grid_df,
                acq_func=acq_func,
                acq_func_kwargs=acq_func_kwargs,
                goal=goal
            )
            acquisition_grid = acq_values.reshape(X_grid.shape)
            
            _, _, _ = create_contour_plot(
                x_grid=X_grid,
                y_grid=Y_grid,
                predictions_grid=acquisition_grid,
                x_var=x_var,
                y_var=y_var,
                exp_x=None,
                exp_y=None,
                suggest_x=None,
                suggest_y=None,
                cmap='Greens',  # Green colormap for acquisition
                title=None,  # No title for acquisition subplot
                ax=ax2
            )
            
            # Mark the suggested point
            sugg_x = suggestion[x_var]
            sugg_y = suggestion[y_var]
            ax2.scatter([sugg_x], [sugg_y], color='black', s=102, marker='*', zorder=10,
                       linewidths=1.5, label=f'{acq_func.upper()} (suggested)')
            ax2.legend()
            
        else:  # 3D
            # 3D acquisition voxel
            logger.warning("3D voxel plots for acquisition not yet fully supported with subplots")
            ax2.text(0.5, 0.5, "3D voxel acquisition visualization\n(use plot_acquisition_voxel separately)",
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.axis('off')
        
        plt.tight_layout()
        
        logger.info(f"Generated suggested next experiment visualization ({len(plot_vars)}D)")
        return fig
    
    def plot_probability_of_improvement(
        self,
        goal: Literal['maximize', 'minimize'] = 'maximize',
        backend: Optional[str] = None,
        kernel: Optional[str] = None,
        n_grid_points: int = 1000,
        start_iteration: int = 5,
        reuse_hyperparameters: bool = True,
        xi: float = 0.01,
        figsize: Tuple[float, float] = (8, 6),
        dpi: int = 100,
        title: Optional[str] = None
    ) -> Figure: # pyright: ignore[reportInvalidTypeForm]
        """
        Plot maximum probability of improvement over optimization iterations.
        
        Retroactively computes how the probability of finding a better solution
        evolved during optimization. At each iteration:
        1. Trains GP on observations up to that point (reusing hyperparameters)
        2. Computes PI across the search space using native acquisition functions
        3. Records the maximum PI value
        
        Uses native PI implementations:
        - sklearn backend: skopt.acquisition.gaussian_pi
        - botorch backend: botorch.acquisition.ProbabilityOfImprovement
        
        Decreasing max(PI) indicates the optimization is converging and has
        less potential for improvement remaining.
        
        Args:
            goal: 'maximize' or 'minimize' - optimization direction
            backend: Model backend to use (defaults to session's model_backend)
            kernel: Kernel type for GP (defaults to session's kernel type)
            n_grid_points: Number of points to sample search space
            start_iteration: Minimum observations before computing PI (default: 5)
            reuse_hyperparameters: If True, use final model's optimized hyperparameters
                                   for all iterations (much faster, recommended)
            xi: PI parameter controlling improvement threshold (default: 0.01)
            figsize: Figure size as (width, height) in inches
            dpi: Dots per inch for figure resolution
            title: Custom plot title (auto-generated if None)
        
        Returns:
            matplotlib Figure object
        
        Example:
            >>> # After running optimization
            >>> fig = session.plot_probability_of_improvement(goal='maximize')
            >>> fig.savefig('pi_convergence.png')
        
        Note:
            - Requires at least `start_iteration` experiments
            - Use fewer n_grid_points for faster computation
            - PI values near 0 suggest little room for improvement
            - Reusing hyperparameters (default) is much faster and usually sufficient
            - Uses rigorous acquisition function implementations (not approximations)
        """
        self._check_matplotlib()

        if self.is_multi_objective:
            raise ValueError(
                "Probability of improvement plot is not available for multi-objective sessions. "
                "Use plot_regret(ref_point=...) for hypervolume convergence tracking."
            )

        # Check we have enough experiments
        n_exp = len(self.experiment_manager.df)
        if n_exp < start_iteration:
            raise ValueError(
                f"Need at least {start_iteration} experiments for PI plot "
                f"(have {n_exp}). Lower start_iteration if needed."
            )
        
        # Default to session's model configuration if not specified
        if backend is None:
            if self.model_backend is None:
                raise ValueError(
                    "No backend specified and session has no trained model. "
                    "Either train a model first or specify backend parameter."
                )
            backend = self.model_backend
        
        if kernel is None:
            if self.model is None:
                raise ValueError(
                    "No kernel specified and session has no trained model. "
                    "Either train a model first or specify kernel parameter."
                )
            # Extract kernel type from trained model
            if self.model_backend == 'sklearn' and hasattr(self.model, 'optimized_kernel'):
                # sklearn model
                kernel_obj = self.model.optimized_kernel
                if 'RBF' in str(type(kernel_obj)):
                    kernel = 'RBF'
                elif 'Matern' in str(type(kernel_obj)):
                    kernel = 'Matern'
                elif 'RationalQuadratic' in str(type(kernel_obj)):
                    kernel = 'RationalQuadratic'
                else:
                    kernel = 'RBF'  # fallback
            elif self.model_backend == 'botorch' and hasattr(self.model, 'cont_kernel_type'):
                # botorch model - use the stored kernel type
                kernel = self.model.cont_kernel_type
            else:
                # Final fallback if we can't determine kernel
                kernel = 'Matern'
        
        # Get optimized hyperparameters if reusing them
        optimized_kernel_params = None
        if reuse_hyperparameters and self.model is not None:
            if backend.lower() == 'sklearn' and hasattr(self.model, 'optimized_kernel'):
                # Extract the optimized kernel parameters
                optimized_kernel_params = self.model.optimized_kernel
                logger.info(f"Reusing optimized kernel hyperparameters from trained model")
            # Note: botorch hyperparameter reuse would go here if needed
        
        # Get data
        target_col = self.experiment_manager.target_columns[0]
        X_all, y_all = self.experiment_manager.get_features_and_target()
        
        # Generate grid of test points across search space
        X_test = self._generate_prediction_grid(n_grid_points)
        
        logger.info(f"Computing PI convergence from iteration {start_iteration} to {n_exp}...")
        logger.info(f"Using {len(X_test)} test points across search space")
        logger.info(f"Using native PI acquisition functions (xi={xi})")
        if reuse_hyperparameters and optimized_kernel_params is not None:
            logger.info("Using optimized hyperparameters from final model (faster)")
        else:
            logger.info("Optimizing hyperparameters at each iteration (slower but more accurate)")
        
        # Compute max PI at each iteration
        iterations = []
        max_pi_values = []
        
        for i in range(start_iteration, n_exp + 1):
            # Get data up to iteration i
            X_train = X_all.iloc[:i]
            y_train = y_all[:i]
            
            # Create temporary session for this iteration
            temp_session = OptimizationSession(
                search_space=self.search_space,
                experiment_manager=ExperimentManager(search_space=self.search_space)
            )
            temp_session.experiment_manager.df = self.experiment_manager.df.iloc[:i].copy()
            
            # Train model with optimized hyperparameters if available
            try:
                if reuse_hyperparameters and optimized_kernel_params is not None and backend.lower() == 'sklearn':
                    # For sklearn: directly access model and set optimized kernel
                    from alchemist_core.models.sklearn_model import SklearnModel
                    
                    # Create model instance with kernel options
                    model_kwargs = {
                        'kernel_options': {'kernel_type': kernel},
                        'n_restarts_optimizer': 0  # Don't optimize since we're using fixed hyperparameters
                    }
                    temp_model = SklearnModel(**model_kwargs)
                    
                    # Preprocess data
                    X_processed, y_processed = temp_model._preprocess_data(temp_session.experiment_manager)
                    
                    # Import sklearn's GP
                    from sklearn.gaussian_process import GaussianProcessRegressor
                    
                    # Create GP with the optimized kernel and optimizer=None to keep it fixed
                    gp_params = {
                        'kernel': optimized_kernel_params,
                        'optimizer': None,  # Keep hyperparameters fixed
                        'random_state': temp_model.random_state
                    }
                    
                    # Only add alpha if we have noise values
                    if temp_model.alpha is not None:
                        gp_params['alpha'] = temp_model.alpha
                    
                    temp_model.model = GaussianProcessRegressor(**gp_params)
                    
                    # Fit model (only computes GP weights, not hyperparameters)
                    temp_model.model.fit(X_processed, y_processed)
                    temp_model._is_trained = True
                    
                    # Set the model in the session
                    temp_session.model = temp_model
                    temp_session.model_backend = 'sklearn'
                else:
                    # Standard training with hyperparameter optimization
                    temp_session.train_model(backend=backend, kernel=kernel)
            except Exception as e:
                logger.warning(f"Failed to train model at iteration {i}: {e}")
                continue
            
            # Compute PI using native acquisition functions
            try:
                if backend.lower() == 'sklearn':
                    # Use skopt's gaussian_pi function
                    from skopt.acquisition import gaussian_pi
                    
                    # For maximization, negate y values so skopt treats it as minimization
                    if goal.lower() == 'maximize':
                        y_opt = -y_train.max()
                    else:
                        y_opt = y_train.min()
                    
                    # Preprocess X_test using the model's preprocessing pipeline
                    # This handles categorical encoding and scaling
                    X_test_processed = temp_session.model._preprocess_X(X_test)
                    
                    # Compute PI for all test points using skopt's implementation
                    # Note: gaussian_pi expects model with predict(X, return_std=True)
                    pi_values = gaussian_pi(
                        X=X_test_processed,
                        model=temp_session.model.model,  # sklearn GP model
                        y_opt=y_opt,
                        xi=xi
                    )
                    
                    max_pi = float(np.max(pi_values))
                    
                elif backend.lower() == 'botorch':
                    # Use BoTorch's ProbabilityOfImprovement
                    import torch
                    from botorch.acquisition import ProbabilityOfImprovement
                    
                    # Determine best value seen so far
                    if goal.lower() == 'maximize':
                        best_f = float(y_train.max())
                    else:
                        best_f = float(y_train.min())
                    
                    # Encode categorical variables if present
                    X_test_encoded = temp_session.model._encode_categorical_data(X_test)
                    
                    # Convert to torch tensor
                    X_tensor = torch.from_numpy(X_test_encoded.values).to(
                        dtype=temp_session.model.model.train_inputs[0].dtype,
                        device=temp_session.model.model.train_inputs[0].device
                    )
                    
                    # Create PI acquisition function
                    if goal.lower() == 'maximize':
                        pi_acq = ProbabilityOfImprovement(
                            model=temp_session.model.model,
                            best_f=best_f,
                            maximize=True
                        )
                    else:
                        pi_acq = ProbabilityOfImprovement(
                            model=temp_session.model.model,
                            best_f=best_f,
                            maximize=False
                        )
                    
                    # Evaluate PI on all test points
                    temp_session.model.model.eval()
                    with torch.no_grad():
                        pi_values = pi_acq(X_tensor.unsqueeze(-2))  # Add batch dimension
                    
                    max_pi = float(pi_values.max().item())
                    
                else:
                    raise ValueError(f"Unknown backend: {backend}")
                    
            except Exception as e:
                logger.warning(f"Failed to compute PI at iteration {i}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue
            
            # Record max PI
            iterations.append(i)
            max_pi_values.append(max_pi)
            
            if i % 5 == 0 or i == n_exp:
                logger.info(f"  Iteration {i}/{n_exp}: max(PI) = {max_pi:.4f}")
        
        if not iterations:
            raise RuntimeError("Failed to compute PI for any iterations")
        
        # Import visualization function
        from alchemist_core.visualization.plots import create_probability_of_improvement_plot
        
        # Create plot
        fig, ax = create_probability_of_improvement_plot(
            iterations=np.array(iterations),
            max_pi_values=np.array(max_pi_values),
            figsize=figsize,
            dpi=dpi,
            title=title
        )
        
        logger.info(f"Generated PI convergence plot with {len(iterations)} points")
        return fig
