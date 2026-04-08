from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import os
import json

class ExperimentManager:
    """
    Class for storing and managing experimental data in a consistent way across backends.
    Provides methods for data access, saving/loading, and conversion to formats needed by different backends.
    
    Supports both single-objective and multi-objective optimization:
    - Single-objective: Uses single target column (default: 'Output', but configurable)
    - Multi-objective: Uses multiple target columns specified in target_columns attribute
    
    The target_column parameter allows flexible column naming to support various CSV formats.
    """
    def __init__(self, search_space=None, target_columns: Optional[List[str]] = None,
                 variable_columns: Optional[List[str]] = None):
        self.df = pd.DataFrame()  # Raw experimental data
        self.search_space = search_space  # Reference to the search space
        self.filepath = None  # Path to saved experiment file
        self._current_iteration = 0  # Track current iteration for audit log
        # Support flexible target column naming for both single and multi-objective
        self.target_columns = target_columns or ['Output']  # Default to 'Output' for backward compatibility
        # Explicit input variable columns (if None, inferred by dropping metadata)
        self.variable_columns: Optional[List[str]] = variable_columns
        
    def set_search_space(self, search_space):
        """Set or update the search space reference."""
        self.search_space = search_space
        
    def add_experiment(self, point_dict: Dict[str, Union[float, str, int]], output_value: Optional[float] = None, 
                       noise_value: Optional[float] = None, iteration: Optional[int] = None, 
                       reason: Optional[str] = None):
        """
        Add a single experiment point.
        
        Args:
            point_dict: Dictionary with variable names as keys and values
            output_value: The experiment output/target value (if known)
            noise_value: Optional observation noise/uncertainty value for regularization
            iteration: Iteration number (auto-assigned if None)
            reason: Reason for this experiment (e.g., 'Initial Design (LHS)', 'Expected Improvement')
        """
        # Create a copy of the point_dict to avoid modifying the original
        new_point = point_dict.copy()
        
        # Add output value if provided (use first target column for single-objective)
        if output_value is not None:
            new_point[self.target_columns[0]] = output_value
            
        # Add noise value if provided
        if noise_value is not None:
            new_point['Noise'] = noise_value
        
        # Add iteration tracking
        if iteration is not None:
            # Use provided iteration explicitly
            new_point['Iteration'] = int(iteration)
        else:
            # Auto-calculate next iteration based on existing data
            # This ensures proper iteration tracking across all clients
            if len(self.df) > 0 and 'Iteration' in self.df.columns:
                max_iteration = int(self.df['Iteration'].max())
                new_point['Iteration'] = max_iteration + 1
            else:
                # First experiment defaults to iteration 0
                new_point['Iteration'] = 0
        
        # Keep _current_iteration in sync with latest iteration for backward compatibility
        try:
            self._current_iteration = int(new_point['Iteration'])
        except Exception:
            pass
        
        # Add reason
        new_point['Reason'] = reason if reason is not None else 'Manual'
            
        # Convert to DataFrame and append
        new_df = pd.DataFrame([new_point])
        self.df = pd.concat([self.df, new_df], ignore_index=True)
        
    def add_experiments_batch(self, data_df: pd.DataFrame):
        """Add multiple experiment points at once from a DataFrame."""
        # Ensure all required columns are present
        if self.search_space:
            required_cols = self.search_space.get_variable_names()
            missing_cols = [col for col in required_cols if col not in data_df.columns]
            if missing_cols:
                raise ValueError(f"DataFrame is missing required columns: {missing_cols}")
        
        # Ensure each row has an Iteration value; default to current iteration
        if 'Iteration' not in data_df.columns:
            data_df = data_df.copy()
            data_df['Iteration'] = int(self._current_iteration)
        else:
            # Fill missing iterations with current iteration
            data_df = data_df.copy()
            data_df['Iteration'] = pd.to_numeric(data_df['Iteration'], errors='coerce').fillna(self._current_iteration).astype(int)
            # Update _current_iteration to the max iteration present
            if len(data_df) > 0:
                max_iter = int(data_df['Iteration'].max())
                if max_iter > self._current_iteration:
                    self._current_iteration = max_iter

        # Append the data
        self.df = pd.concat([self.df, data_df], ignore_index=True)
    
    def get_data(self) -> pd.DataFrame:
        """Get the raw experiment data."""
        return self.df.copy()
    
    def get_features_and_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get features (X) and target (y) separated.

        Returns:
            X: Features DataFrame
            y: Target Series

        Raises:
            ValueError: If configured target column is not found in data
        """
        target_col = self.target_columns[0]  # Use first target column for single-objective

        if target_col not in self.df.columns:
            raise ValueError(
                f"DataFrame doesn't contain target column '{target_col}'. "
                f"Available columns: {list(self.df.columns)}"
            )

        if self.variable_columns is not None:
            X = self.df[self.variable_columns]
        elif self.search_space is not None and hasattr(self.search_space, 'get_variable_names'):
            ordered_cols = self.search_space.get_variable_names()
            missing = [c for c in ordered_cols if c not in self.df.columns]
            if missing:
                raise ValueError(
                    f"Variable column(s) {missing} not found in experiment data. "
                    f"Available columns: {list(self.df.columns)}"
                )
            X = self.df[ordered_cols]
        else:
            # Drop metadata columns (target, Noise, Iteration, Reason)
            metadata_cols = self.target_columns.copy()
            if 'Noise' in self.df.columns:
                metadata_cols.append('Noise')
            if 'Iteration' in self.df.columns:
                metadata_cols.append('Iteration')
            if 'Reason' in self.df.columns:
                metadata_cols.append('Reason')
            X = self.df.drop(columns=metadata_cols)

        y = self.df[target_col]
        return X, y
    
    def get_features_target_and_noise(self) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """
        Get features (X), target (y), and noise values if available.

        Returns:
            X: Features DataFrame
            y: Target Series
            noise: Noise Series if available, otherwise None

        Raises:
            ValueError: If configured target column is not found in data
        """
        target_col = self.target_columns[0]  # Use first target column for single-objective

        if target_col not in self.df.columns:
            raise ValueError(
                f"DataFrame doesn't contain target column '{target_col}'. "
                f"Available columns: {list(self.df.columns)}"
            )

        if self.variable_columns is not None:
            X = self.df[self.variable_columns]
        elif self.search_space is not None and hasattr(self.search_space, 'get_variable_names'):
            ordered_cols = self.search_space.get_variable_names()
            missing = [c for c in ordered_cols if c not in self.df.columns]
            if missing:
                raise ValueError(
                    f"Variable column(s) {missing} not found in experiment data. "
                    f"Available columns: {list(self.df.columns)}"
                )
            X = self.df[ordered_cols]
        else:
            # Drop metadata columns
            metadata_cols = self.target_columns.copy()
            if 'Noise' in self.df.columns:
                metadata_cols.append('Noise')
            if 'Iteration' in self.df.columns:
                metadata_cols.append('Iteration')
            if 'Reason' in self.df.columns:
                metadata_cols.append('Reason')
            X = self.df.drop(columns=metadata_cols)

        y = self.df[target_col]
        noise = self.df['Noise'] if 'Noise' in self.df.columns else None
        return X, y, noise
    
    def get_features_and_targets_multi(self) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Get features (X), all targets (Y), and optional noise values for multi-objective optimization.

        Returns:
            X: Features DataFrame (n_samples, n_features)
            Y: Targets DataFrame (n_samples, n_objectives) with columns = target names
            noise: Noise DataFrame if available, otherwise None

        Raises:
            ValueError: If any target column is not found in data
        """
        # Validate all target columns exist
        missing_cols = [col for col in self.target_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(
                f"Target column(s) {missing_cols} not found in data. "
                f"Available columns: {list(self.df.columns)}"
            )

        if self.variable_columns is not None:
            X = self.df[self.variable_columns]
        elif self.search_space is not None and hasattr(self.search_space, 'get_variable_names'):
            ordered_cols = self.search_space.get_variable_names()
            missing = [c for c in ordered_cols if c not in self.df.columns]
            if missing:
                raise ValueError(
                    f"Variable column(s) {missing} not found in experiment data. "
                    f"Available columns: {list(self.df.columns)}"
                )
            X = self.df[ordered_cols]
        else:
            # Drop all metadata columns (all targets, Noise, Iteration, Reason)
            metadata_cols = self.target_columns.copy()
            if 'Noise' in self.df.columns:
                metadata_cols.append('Noise')
            if 'Iteration' in self.df.columns:
                metadata_cols.append('Iteration')
            if 'Reason' in self.df.columns:
                metadata_cols.append('Reason')
            X = self.df.drop(columns=metadata_cols)

        Y = self.df[self.target_columns].copy()
        noise = self.df[['Noise']] if 'Noise' in self.df.columns else None

        return X, Y, noise

    def has_noise_data(self) -> bool:
        """Check if the experiment data includes noise values."""
        return 'Noise' in self.df.columns
    
    def save_to_csv(self, filepath: Optional[str] = None):
        """
        Save experiments to a CSV file.
        
        Args:
            filepath: Path to save the file. If None, uses the previously used path.
        """
        if filepath:
            self.filepath = filepath
        
        if not self.filepath:
            raise ValueError("No filepath specified and no previous filepath available")
            
        self.df.to_csv(self.filepath, index=False)
        
    def load_from_csv(self, filepath: str):
        """
        Load experiments from a CSV file.
        
        Args:
            filepath: Path to the CSV file
        """
        self.df = pd.read_csv(filepath)
        self.filepath = filepath
        
        # Ensure noise values are numeric if present
        if 'Noise' in self.df.columns:
            try:
                self.df['Noise'] = pd.to_numeric(self.df['Noise'])
                print(f"Loaded experiment data with noise column. Noise values will be used for model regularization.")
            except ValueError:
                print("Warning: Noise column contains non-numeric values. Converting to default noise level.")
                self.df['Noise'] = 1e-10  # Default small noise
        
        # Initialize iteration tracking from data
        if 'Iteration' in self.df.columns:
            self._current_iteration = int(self.df['Iteration'].max())
        else:
            # Add iteration column if missing (legacy data)
            self.df['Iteration'] = 0
            self._current_iteration = 0
        
        # Add reason column if missing (legacy data)
        if 'Reason' not in self.df.columns:
            self.df['Reason'] = 'Initial Design'
        
        return self
    
    @classmethod
    def from_csv(cls, filepath: str, search_space=None):
        """Class method to create an ExperimentManager from a CSV file."""
        instance = cls(search_space=search_space)
        return instance.load_from_csv(filepath)
    
    def clear(self):
        """Clear all experimental data."""
        self.df = pd.DataFrame()
    
    def get_full_history(self) -> pd.DataFrame:
        """Get the full experiment history."""
        return self.df.copy()
    
    def get_latest_experiment(self) -> pd.Series:
        """Get the most recently added experiment."""
        if len(self.df) == 0:
            return None
        return self.df.iloc[-1].copy()
    
    def __len__(self):
        return len(self.df)
    
    def get_pareto_frontier(self, directions: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compute Pareto-optimal solutions from experiments with multiple objectives.
        
        Uses BoTorch's fast non-dominated sorting algorithm to identify Pareto-optimal
        points. Works with both single-objective (returns all data) and multi-objective
        experiments.
        
        Args:
            directions: List of 'maximize' or 'minimize' for each target column.
                       If None, assumes all objectives are maximized.
                       Length must match number of target columns.
        
        Returns:
            DataFrame containing only Pareto-optimal experiments with all columns.
            
        Raises:
            ValueError: If directions length doesn't match target columns.
            ValueError: If target columns contain missing data.
            
        Example:
            >>> # For 2 objectives: maximize yield, minimize cost
            >>> pareto_df = exp_mgr.get_pareto_frontier(['maximize', 'minimize'])
        """
        import torch
        from botorch.utils.multi_objective.pareto import is_non_dominated
        
        if len(self.df) == 0:
            return pd.DataFrame()
        
        # Validate target columns exist
        missing_cols = [col for col in self.target_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Target columns {missing_cols} not found in experiment data")
        
        # Extract objective values
        Y = self.df[self.target_columns].values
        
        # Check for missing values
        if pd.isna(Y).any():
            raise ValueError("Target columns contain missing values (NaN). Cannot compute Pareto frontier.")
        
        # Single objective case: return all data
        if len(self.target_columns) == 1:
            return self.df.copy()
        
        # Set default directions if not provided
        if directions is None:
            directions = ['maximize'] * len(self.target_columns)
        
        # Validate directions
        if len(directions) != len(self.target_columns):
            raise ValueError(
                f"Number of directions ({len(directions)}) must match number of "
                f"target columns ({len(self.target_columns)})"
            )
        
        # Convert objectives to maximization form (BoTorch assumes maximization)
        Y_torch = torch.tensor(Y, dtype=torch.double)
        for i, direction in enumerate(directions):
            if direction.lower() == 'minimize':
                Y_torch[:, i] = -Y_torch[:, i]
        
        # Compute non-dominated mask
        nd_mask = is_non_dominated(Y_torch, maximize=True, deduplicate=True)
        
        # Return Pareto-optimal experiments
        return self.df[nd_mask.numpy()].copy()
    
    def compute_hypervolume(self, ref_point: Union[List[float], np.ndarray], 
                           directions: Optional[List[str]] = None) -> float:
        """
        Compute hypervolume indicator for multi-objective experiments.
        
        The hypervolume measures the volume of objective space dominated by the
        Pareto frontier relative to a reference point. Larger values indicate
        better overall performance.
        
        Args:
            ref_point: Reference point (worst acceptable values) for each objective.
                      Must have same length as target_columns.
                      For maximization: should be below minimum observed values.
                      For minimization: should be above maximum observed values.
            directions: List of 'maximize' or 'minimize' for each target column.
                       If None, assumes all objectives are maximized.
        
        Returns:
            Hypervolume value (float). Zero if no Pareto-optimal points exist.
            
        Raises:
            ValueError: If ref_point length doesn't match target columns.
            ValueError: If target columns contain missing data.
            
        Example:
            >>> # For 2 objectives (maximize yield, minimize cost)
            >>> # ref_point = [min_acceptable_yield, max_acceptable_cost]
            >>> hv = exp_mgr.compute_hypervolume([50.0, 100.0], ['maximize', 'minimize'])
        """
        import torch
        from botorch.utils.multi_objective.hypervolume import Hypervolume
        
        if len(self.df) == 0:
            return 0.0
        
        # Single objective case: not meaningful
        if len(self.target_columns) == 1:
            raise ValueError(
                "Hypervolume is only defined for multi-objective problems. "
                "For single-objective, use best observed value instead."
            )
        
        # Validate ref_point
        ref_point = np.array(ref_point)
        if len(ref_point) != len(self.target_columns):
            raise ValueError(
                f"Reference point length ({len(ref_point)}) must match number of "
                f"target columns ({len(self.target_columns)})"
            )
        
        # Get Pareto frontier
        pareto_df = self.get_pareto_frontier(directions)
        if len(pareto_df) == 0:
            return 0.0
        
        # Set default directions if not provided
        if directions is None:
            directions = ['maximize'] * len(self.target_columns)
        
        # Extract Pareto objectives and convert to torch tensors
        Y_pareto = pareto_df[self.target_columns].values
        Y_torch = torch.tensor(Y_pareto, dtype=torch.double)
        ref_torch = torch.tensor(ref_point, dtype=torch.double)
        
        # Convert to maximization form (BoTorch assumes maximization)
        for i, direction in enumerate(directions):
            if direction.lower() == 'minimize':
                Y_torch[:, i] = -Y_torch[:, i]
                ref_torch[i] = -ref_torch[i]
        
        # Compute hypervolume
        hv_calculator = Hypervolume(ref_point=ref_torch)
        hv = hv_calculator.compute(Y_torch)
        
        return float(hv)
