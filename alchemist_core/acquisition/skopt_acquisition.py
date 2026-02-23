from skopt import Optimizer
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from .base_acquisition import BaseAcquisition
from alchemist_core.data.search_space import SearchSpace

class SkoptAcquisition(BaseAcquisition):
    """
    Simple acquisition function implementation using scikit-optimize (skopt).
    
    Supported acquisition functions:
    - 'ei' or 'EI': Expected Improvement
    - 'pi' or 'PI': Probability of Improvement  
    - 'ucb' or 'UCB': Upper Confidence Bound (mapped to LCB in skopt)
    - 'gp_hedge': GP-Hedge (portfolio of acquisition functions)
    """
    
    # Valid acquisition function names
    VALID_ACQ_FUNCS = {'ei', 'pi', 'ucb', 'gp_hedge', 'EI', 'PI', 'UCB', 'expectedimprovement', 'probabilityofimprovement', 'upperconfidencebound'}
    
    def __init__(self, search_space, model=None, acq_func='ei', maximize=True, random_state=42, acq_func_kwargs=None):
        """
        Initialize the acquisition function.
        
        Args:
            search_space: The search space (SearchSpace object or list of skopt dimensions)
            model: A trained model (SklearnModel or compatible)
            acq_func: Acquisition function ('ei', 'pi', 'ucb', or 'gp_hedge')
            maximize: Whether to maximize (True) or minimize (False) the objective
            random_state: Random state for reproducibility
            acq_func_kwargs: Dictionary of additional arguments for the acquisition function
            
        Raises:
            ValueError: If acq_func is not a valid acquisition function name
        """
        # Validate acquisition function before proceeding
        acq_func_lower = acq_func.lower()
        if acq_func_lower not in self.VALID_ACQ_FUNCS:
            valid_funcs = "', '".join(sorted(['ei', 'pi', 'ucb', 'gp_hedge']))
            raise ValueError(
                f"Invalid acquisition function '{acq_func}' for sklearn backend. "
                f"Valid options are: '{valid_funcs}'"
            )
        
        self.search_space_obj = search_space
        self.maximize = maximize
        self.random_state = random_state
        
        # Convert SearchSpace object to skopt dimensions if needed
        if isinstance(search_space, SearchSpace):
            self.search_space = search_space.to_skopt()
        else:
            self.search_space = search_space
        
        # Map lowercase acquisition function names to the expected uppercase versions
        acq_func_map = {
            'ei': 'EI',
            'expectedimprovement': 'EI',
            'pi': 'PI',
            'probabilityofimprovement': 'PI',
            'ucb': 'LCB',  # Note: ucb in our UI maps to LCB in skopt
            'upperconfidencebound': 'LCB',
            'gp_hedge': 'gp_hedge'
        }
        self.acq_func = acq_func_map.get(acq_func_lower, 'EI')
        
        # Process acquisition function kwargs
        if acq_func_kwargs is None:
            acq_func_kwargs = {}
            
        # Default values if not provided
        if self.acq_func in ['EI', 'PI'] and 'xi' not in acq_func_kwargs:
            acq_func_kwargs['xi'] = 0.01
        if self.acq_func == 'LCB' and 'kappa' not in acq_func_kwargs:
            acq_func_kwargs['kappa'] = 1.96
        if self.acq_func == 'gp_hedge':
            # For gp_hedge, we can set both parameters
            if 'xi' not in acq_func_kwargs:
                acq_func_kwargs['xi'] = 0.01
            if 'kappa' not in acq_func_kwargs:
                acq_func_kwargs['kappa'] = 1.96
                
        self.acq_func_kwargs = acq_func_kwargs
        
        # Create the optimizer
        if model is not None and hasattr(model, 'model'):
            # Use the model directly
            self.optimizer = Optimizer(
                dimensions=self.search_space,
                base_estimator=model.model,
                acq_func=self.acq_func,
                acq_func_kwargs=self.acq_func_kwargs,
                random_state=random_state
            )
        else:
            # Use default GP
            self.optimizer = Optimizer(
                dimensions=self.search_space,
                base_estimator='GP',
                acq_func=self.acq_func,
                acq_func_kwargs=self.acq_func_kwargs,
                random_state=random_state
            )
    
    def update(self, X, y, **kwargs):
        """
        Update the optimizer with new observations.
        """
        # Convert inputs to the format expected by skopt
        if isinstance(X, pd.DataFrame):
            # Make sure X doesn't include Noise column if present
            feature_cols = [col for col in X.columns if col != 'Noise']
            X_clean = X[feature_cols]
            X_list = X_clean.values.tolist()
        elif isinstance(X, np.ndarray):
            X_list = X.tolist()
        else:
            X_list = list(X)

        # For Categorical dimensions with all-numeric categories (discrete numeric
        # variables), snap each value to the nearest allowed value.  This is a
        # defensive guard against floating-point drift or out-of-set values that
        # would cause skopt's check_x_in_space to raise ValueError.
        for dim_idx, dim in enumerate(self.search_space):
            if hasattr(dim, 'categories') and all(
                isinstance(c, (int, float)) for c in dim.categories
            ):
                allowed = [float(c) for c in dim.categories]
                for row in X_list:
                    row[dim_idx] = min(allowed, key=lambda c, v=float(row[dim_idx]): abs(c - v))

        if isinstance(y, pd.Series) or isinstance(y, np.ndarray):
            y_list = y.tolist()
        else:
            y_list = y
    
        # If we're maximizing, negate y since skopt always minimizes
        if self.maximize:
            y_list = [-val for val in y_list]
    
        # Update the optimizer
        self.optimizer.tell(X_list, y_list)
        return self
    
    def select_next(self, candidate_points=None, **kwargs):
        """
        Suggest the next experiment point.
        """
        # Guard: skopt does not support multi-objective
        if hasattr(self, '_model_wrapper') and hasattr(self._model_wrapper, 'n_objectives') and self._model_wrapper.n_objectives > 1:
            raise ValueError(
                "Multi-objective optimization requires BoTorch backend. "
                "Use BoTorchAcquisition with qEHVI or qNEHVI strategies."
            )

        if candidate_points is None:
            # No candidates provided - let optimizer explore the full space
            return self.optimizer.ask()
        
        # With candidates, we'll have skopt evaluate each one
        # First convert to a list format if needed
        if isinstance(candidate_points, np.ndarray):
            candidates_list = candidate_points.tolist()
        elif isinstance(candidate_points, pd.DataFrame):
            candidates_list = candidate_points.values.tolist()
        else:
            candidates_list = candidate_points
        
        # Use skopt's built-in method to evaluate candidates
        return self.optimizer.ask(n_points=len(candidates_list), strategy="cl_min")
    
    def find_optimum(self, model, maximize=True, random_state=42):
        """
        Find the point where the model predicts the optimal value.
        
        Args:
            model: Trained model with predict method
            maximize: Whether to maximize (True) or minimize (False) the objective
            random_state: Random seed for reproducibility
            
        Returns:
            dict: Contains 'x_opt' (optimal point), 'value' (predicted value), 
                  'std' (standard deviation at optimum)
        """
        # Get the dimension names
        dimension_names = [dim.name for dim in self.search_space]
        
        # Initialize containers for different dimension types
        continuous_dims = []
        categorical_dims = {}
        
        # Categorize dimensions
        for i, dim in enumerate(self.search_space):
            if hasattr(dim, 'categories'):
                # Categorical dimension
                categorical_dims[i] = dim.categories
            elif hasattr(dim, 'bounds'):
                # Real or Integer dimension
                continuous_dims.append((i, dim))
                
        # If we have only categorical dimensions, use a simpler approach
        if not continuous_dims:
            return self._evaluate_categorical_only(model, dimension_names, categorical_dims, maximize)
        
        # Otherwise, we have a mix of continuous and categorical dimensions
        # Need to evaluate all categorical combinations with optimization for continuous
        
        # Get all categorical combinations
        cat_combinations = self._get_categorical_combinations(categorical_dims)
        
        # Best results tracking
        best_value = float('-inf') if maximize else float('inf')
        best_point = None
        best_std = None
        
        # For each categorical combination
        for cat_combo in cat_combinations:
            # Get bounds for ONLY the continuous dimensions
            bounds = []
            for idx, dim in continuous_dims:
                bounds.append(dim.bounds)
                
            # Define objective function for this categorical combination
            def objective(x_continuous):
                # Create point with continuous values from x and fixed categorical values
                point = [None] * len(dimension_names)
                
                # Fill continuous values at appropriate indices
                for i, (idx, _) in enumerate(continuous_dims):
                    point[idx] = x_continuous[i]
                    
                # Fill categorical values at appropriate indices
                for idx, val in cat_combo.items():
                    point[idx] = val
                    
                # Create DataFrame for prediction
                df = pd.DataFrame([point], columns=dimension_names)
                
                # Get prediction from model
                pred = model.predict(df)
                
                # Return negative for maximization since scipy minimizes
                return -pred[0] if maximize else pred[0]
                
            # Optimize continuous dimensions with differential evolution
            result = differential_evolution(
                objective, bounds, popsize=20, seed=random_state
            )
            
            # Create full optimal point
            opt_point = [None] * len(dimension_names)
            
            # Fill optimized continuous values
            for i, (idx, _) in enumerate(continuous_dims):
                opt_point[idx] = result.x[i]
                
            # Fill fixed categorical values
            for idx, val in cat_combo.items():
                opt_point[idx] = val
            
            # Create DataFrame for this point
            opt_df = pd.DataFrame([opt_point], columns=dimension_names)
            
            # Get predicted value and std
            if hasattr(model, 'predict_with_std'):
                pred_mean, pred_std = model.predict_with_std(opt_df)
                pred_value = pred_mean[0]
                pred_std = pred_std[0]
            else:
                pred_value = model.predict(opt_df)[0]
                pred_std = None
                
            # Check if this is the best point found so far
            is_better = (pred_value > best_value) if maximize else (pred_value < best_value)
            if best_point is None or is_better:
                best_value = pred_value
                best_point = opt_df
                best_std = pred_std
                
        return {
            'x_opt': best_point,
            'value': best_value,
            'std': best_std
        }

    def _get_categorical_combinations(self, categorical_dims):
        """Get all combinations of categorical variables."""
        import itertools
        
        # If no categorical dimensions, return empty dict
        if not categorical_dims:
            return [{}]
            
        # Get indices and categories
        indices = list(categorical_dims.keys())
        categories = [categorical_dims[idx] for idx in indices]
        
        # Generate all combinations
        combinations = []
        for combo in itertools.product(*categories):
            combo_dict = {}
            for i, idx in enumerate(indices):
                combo_dict[idx] = combo[i]
            combinations.append(combo_dict)
            
        return combinations

    def _evaluate_categorical_only(self, model, dimension_names, categorical_dims, maximize):
        """Evaluate all combinations when only categorical dimensions exist."""
        # Get all combinations
        combinations = self._get_categorical_combinations(categorical_dims)
        
        # Track best result
        best_value = float('-inf') if maximize else float('inf')
        best_point = None
        best_std = None
        
        # Evaluate each combination
        for combo in combinations:
            # Create point
            point = [None] * len(dimension_names)
            for idx, val in combo.items():
                point[idx] = val
                
            # Create DataFrame
            df = pd.DataFrame([point], columns=dimension_names)
            
            # Get prediction
            if hasattr(model, 'predict_with_std'):
                mean, std = model.predict_with_std(df)
                value = mean[0]
                std_val = std[0]
            else:
                value = model.predict(df)[0]
                std_val = None
                
            # Update best if better
            is_better = (value > best_value) if maximize else (value < best_value)
            if best_point is None or is_better:
                best_value = value
                best_point = df
                best_std = std_val
                
        return {
            'x_opt': best_point,
            'value': best_value,
            'std': best_std
        }
