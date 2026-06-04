from .base_model import BaseModel
from alchemist_core.data.search_space import SearchSpace
from alchemist_core.data.experiment_manager import ExperimentManager
from alchemist_core.config import get_logger
import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
import scipy.optimize

from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ConstantKernel as C

logger = get_logger(__name__)

class SklearnModel(BaseModel):
    def __init__(self, kernel_options: dict, n_restarts_optimizer=30, random_state=42, 
                 optimizer="L-BFGS-B", input_transform_type: str = "none", 
                 output_transform_type: str = "none"):
        """
        Initialize the SklearnModel with kernel options and scaling transforms.
        
        Args:
            kernel_options: Dictionary with keys:
                - "kernel_type": one of "RBF", "Matern", "RationalQuadratic" 
                - If "Matern" is selected, a key "matern_nu" should be provided.
            n_restarts_optimizer: Number of restarts for the optimizer.
            random_state: Random state for reproducibility.
            optimizer: Optimization method for hyperparameter tuning.
            input_transform_type: Type of input scaling ("none", "standard", "minmax", "robust")
            output_transform_type: Type of output scaling ("none", "standard")
        """
        super().__init__(random_state=random_state, 
                        input_transform_type=input_transform_type,
                        output_transform_type=output_transform_type)
        self.kernel_options = kernel_options
        self.n_restarts_optimizer = n_restarts_optimizer
        self.optimizer = optimizer
        self.model = None
        self.optimized_kernel = None
        self.encoder = None  # For one-hot encoding
        self.categorical_variables = []
        self.cv_cached_results = None  # Will store y_true and y_pred from cross-validation
        
        # Calibration attributes
        self.calibration_enabled = False
        self.calibration_factor = 1.0  # Multiplicative factor for std (s = std(z))
        
        # Initialize transform objects
        self.input_scaler = None
        self.output_scaler = None
        self._initialize_scalers()

    def _initialize_scalers(self):
        """Initialize input and output scalers based on transform types."""
        # Initialize input scaler
        if self.input_transform_type == "standard":
            self.input_scaler = StandardScaler()
        elif self.input_transform_type == "minmax":
            self.input_scaler = MinMaxScaler()
        elif self.input_transform_type == "robust":
            self.input_scaler = RobustScaler()
        else:  # "none"
            self.input_scaler = None
            
        # Initialize output scaler
        if self.output_transform_type == "standard":
            self.output_scaler = StandardScaler()
        else:  # "none"
            self.output_scaler = None

    def _custom_optimizer(self, obj_func, initial_theta, bounds, args=(), **kwargs):
        result = scipy.optimize.minimize(
            obj_func,
            initial_theta,
            bounds=bounds if self.optimizer not in ['CG', 'BFGS'] else None,
            method=self.optimizer,
            jac=True,
            args=args,
            **kwargs
        )
        return result.x, result.fun

    def _build_kernel(self, X):
        """Build the kernel using training data X to initialize length scales."""
        kernel_type = self.kernel_options.get("kernel_type", "RBF")
        # Compute initial length scales from the data.
        # Use standard deviation (positive) as a robust length-scale initializer.
        try:
            ls_init = np.std(X, axis=0)
            ls_init = np.array(ls_init, dtype=float)
            # Replace non-finite or non-positive values with sensible defaults
            bad_mask = ~np.isfinite(ls_init) | (ls_init <= 0)
            if np.any(bad_mask):
                logger.debug("Replacing non-finite or non-positive length-scales with 1.0")
                ls_init[bad_mask] = 1.0

            # Build finite, positive bounds for each length-scale
            ls_bounds = []
            for l in ls_init:
                # Protect against extremely small or non-finite upper bounds
                upper = float(l * 1e5) if np.isfinite(l) else 1e5
                if not np.isfinite(upper) or upper <= 1e-8:
                    upper = 1e3
                ls_bounds.append((1e-5, upper))
        except Exception as e:
            logger.warning(f"Failed to compute sensible length-scales from data: {e}. Using safe defaults.")
            n_dims = X.shape[1] if hasattr(X, 'shape') else 1
            ls_init = np.ones(n_dims, dtype=float)
            ls_bounds = [(1e-5, 1e5) for _ in range(n_dims)]
        constant = C()
        if kernel_type == "RBF":
            kernel = constant * RBF(length_scale=ls_init, length_scale_bounds=ls_bounds)
        elif kernel_type == "Matern":
            matern_nu = self.kernel_options.get("matern_nu", 1.5)
            kernel = constant * Matern(length_scale=ls_init, length_scale_bounds=ls_bounds, nu=matern_nu)
        elif kernel_type == "RationalQuadratic":
            kernel = constant * RationalQuadratic()
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        return kernel

    def _preprocess_data(self, experiment_manager):
        """Preprocess the data for scikit-learn with one-hot encoding for categoricals and scaling."""
        # Get data with noise values if available
        X, y, noise = experiment_manager.get_features_target_and_noise()
        categorical_variables = experiment_manager.search_space.get_categorical_variables()
        self.categorical_variables = categorical_variables
        
        # Store noise values for later use in the model (use None if no noise provided)
        self.alpha = noise.values if noise is not None else None
        logger.info(f"{'Using provided noise values for regularization' if noise is not None else 'No noise values provided - using scikit-learn default regularization'}")
        
        # Separate categorical and numerical columns
        categorical_df = X[categorical_variables] if categorical_variables else None
        numerical_df = X.drop(columns=categorical_variables) if categorical_variables else X

        # One-hot-encode categorical variables if they exist, using drop='first' for skopt compatibility
        if categorical_df is not None and not categorical_df.empty:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop='first')
            encoded_categorical = self.encoder.fit_transform(categorical_df)
            encoded_categorical_df = pd.DataFrame(
                encoded_categorical,
                columns=self.encoder.get_feature_names_out(categorical_variables),
                index=X.index
            )
            # Merge numerical and encoded categorical data
            processed_X = pd.concat([numerical_df, encoded_categorical_df], axis=1)
        else:
            # If no categorical variables, use only numerical data
            processed_X = numerical_df
            self.encoder = None

        # Apply input scaling if enabled
        if self.input_scaler is not None:
            processed_X_scaled = self.input_scaler.fit_transform(processed_X.values)
            logger.info(f"Applied {self.input_transform_type} scaling to input features")
        else:
            processed_X_scaled = processed_X.values
            logger.info("No input scaling applied")

        # Apply output scaling if enabled
        y_processed = y.values.reshape(-1, 1)
        if self.output_scaler is not None:
            y_scaled = self.output_scaler.fit_transform(y_processed).ravel()
            logger.info(f"Applied {self.output_transform_type} scaling to output")
        else:
            y_scaled = y_processed.ravel()
            logger.info("No output scaling applied")

        # Save the feature names for debugging dimensional mismatches
        self.feature_names = processed_X.columns.tolist()
        logger.info(f"Model trained with {len(self.feature_names)} features: {self.feature_names}")
        
        return processed_X_scaled, y_scaled

    def _preprocess_subset(self, X_subset, categorical_variables, fit_scalers=True):
        """Preprocess a subset of data, optionally fitting scalers."""
        # Separate categorical and numerical columns
        categorical_df = X_subset[categorical_variables] if categorical_variables else None
        numerical_df = X_subset.drop(columns=categorical_variables) if categorical_variables else X_subset

        # One-hot-encode categorical variables if they exist
        if categorical_df is not None and not categorical_df.empty:
            if fit_scalers:
                # Create a new encoder for this fold
                fold_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop='first')
                encoded_categorical = fold_encoder.fit_transform(categorical_df)
                self._fold_encoder = fold_encoder  # Store for use in test set
            elif hasattr(self, '_fold_encoder'):
                encoded_categorical = self._fold_encoder.transform(categorical_df)
            else:
                # Fallback to original encoder if available
                encoded_categorical = self.encoder.transform(categorical_df) if self.encoder else categorical_df.values
                
            try:
                feature_names = self._fold_encoder.get_feature_names_out(categorical_variables) if hasattr(self, '_fold_encoder') else categorical_df.columns
                encoded_categorical_df = pd.DataFrame(
                    encoded_categorical,
                    columns=feature_names,
                    index=X_subset.index
                )
            except:
                # Fallback if feature names fail
                encoded_categorical_df = pd.DataFrame(
                    encoded_categorical,
                    index=X_subset.index
                )
            # Merge numerical and encoded categorical data
            processed_X = pd.concat([numerical_df, encoded_categorical_df], axis=1)
        else:
            processed_X = numerical_df

        # Apply input scaling with independent scaler for each fold
        if self.input_transform_type != "none":
            if fit_scalers:
                # Create a new scaler instance for this fold
                if self.input_transform_type == "minmax":
                    fold_input_scaler = MinMaxScaler()
                elif self.input_transform_type == "standard":
                    fold_input_scaler = StandardScaler()
                elif self.input_transform_type == "robust":
                    fold_input_scaler = RobustScaler()
                else:
                    fold_input_scaler = None
                
                if fold_input_scaler:
                    processed_X_scaled = fold_input_scaler.fit_transform(processed_X.values)
                    self._fold_input_scaler = fold_input_scaler  # Store for use in test set
                else:
                    processed_X_scaled = processed_X.values
            else:
                # Use the fold-specific scaler
                if hasattr(self, '_fold_input_scaler') and self._fold_input_scaler:
                    processed_X_scaled = self._fold_input_scaler.transform(processed_X.values)
                else:
                    processed_X_scaled = processed_X.values
        else:
            processed_X_scaled = processed_X.values

        return processed_X_scaled

    def _scale_output(self, y_values, fit_scaler=True):
        """Scale output values, optionally fitting the scaler."""
        if self.output_transform_type != "none":
            if fit_scaler:
                # Create a new scaler instance for this fold
                if self.output_transform_type == "minmax":
                    fold_output_scaler = MinMaxScaler()
                elif self.output_transform_type == "standard":
                    fold_output_scaler = StandardScaler()
                elif self.output_transform_type == "robust":
                    fold_output_scaler = RobustScaler()
                else:
                    fold_output_scaler = None
                
                if fold_output_scaler:
                    y_scaled = fold_output_scaler.fit_transform(y_values)
                    self._fold_output_scaler = fold_output_scaler  # Store for use in test set
                    return y_scaled
                else:
                    return y_values
            else:
                # Use the fold-specific scaler
                if hasattr(self, '_fold_output_scaler') and self._fold_output_scaler:
                    return self._fold_output_scaler.transform(y_values)
                else:
                    return y_values
        else:
            return y_values

    def _inverse_scale_output(self, y_scaled):
        """Inverse transform scaled output values using fold-specific scaler."""
        if hasattr(self, '_fold_output_scaler') and self._fold_output_scaler:
            return self._fold_output_scaler.inverse_transform(y_scaled)
        else:
            return y_scaled

    def _preprocess_X(self, X):
        """Preprocess X for prediction (apply the same transformations as in training)."""
        if isinstance(X, pd.DataFrame):
            if self.categorical_variables and self.encoder:
                # Check which categorical variables are actually present in the input
                available_categorical_vars = [var for var in self.categorical_variables if var in X.columns]
                
                if available_categorical_vars:
                    categorical_X = X[available_categorical_vars]
                    numerical_X = X.drop(columns=available_categorical_vars)
                    
                    if not categorical_X.empty:
                        encoded_categorical = self.encoder.transform(categorical_X)
                        encoded_categorical_df = pd.DataFrame(
                            encoded_categorical,
                            columns=self.encoder.get_feature_names_out(available_categorical_vars),
                            index=X.index
                        )
                        # Merge numerical and encoded categorical data
                        processed_X = pd.concat([numerical_X, encoded_categorical_df], axis=1).values
                    else:
                        processed_X = numerical_X.values
                else:
                    # No categorical variables present in input, treat all as numerical
                    processed_X = X.values
            else:
                processed_X = X.values
        else:
            # Assume it's already preprocessed if not a DataFrame
            processed_X = X
        
        # Apply input scaling if it was used during training
        if self.input_scaler is not None:
            processed_X = self.input_scaler.transform(processed_X)
            
        return processed_X

    def train(self, experiment_manager, **kwargs):
        """Train the model using the ExperimentManager."""
        if len(experiment_manager.target_columns) > 1:
            raise ValueError(
                "Multi-objective optimization requires backend='botorch'. "
                "The sklearn/scikit-optimize backend only supports single-objective optimization. "
                "Please re-train with: session.train_model(backend='botorch')"
            )

        # Store the original feature names before preprocessing
        X_orig, y_orig, noise = experiment_manager.get_features_target_and_noise()
        self.original_feature_names = X_orig.columns.tolist()
        self.X_orig = X_orig  # Store original data for contour generation
        
        X, y = self._preprocess_data(experiment_manager)
        
        # Check if we should reuse a pre-optimized kernel
        if hasattr(self, '_reuse_kernel') and self._reuse_kernel is not None:
            self.kernel = self._reuse_kernel
            logger.info("Reusing pre-optimized kernel hyperparameters")
        else:
            self.kernel = self._build_kernel(X)
        
        # Create base parameters dictionary
        params = {
            "kernel": self.kernel,
            "n_restarts_optimizer": self.n_restarts_optimizer,
            "random_state": self.random_state,
            "optimizer": self._custom_optimizer
        }
        
        # Only add alpha parameter when noise values are available
        if self.alpha is not None:
            params["alpha"] = self.alpha
        
        # Create model with appropriate parameters
        self.model = GaussianProcessRegressor(**params)

        # Store the raw training data for possible reuse with skopt
        self.X_train_ = X
        self.y_train_ = y

        # Fit the model, but be defensive: if sklearn complains about non-finite
        # bounds when n_restarts_optimizer>0, retry with no restarts.
        try:
            self.model.fit(X, y)
        except ValueError as e:
            msg = str(e)
            if 'requires that all bounds are finite' in msg or 'bounds' in msg.lower():
                logger.warning("GaussianProcessRegressor failed due to non-finite bounds. "
                               "Retrying without optimizer restarts (n_restarts_optimizer=0).")
                # Retry with safer parameters
                safe_params = params.copy()
                safe_params['n_restarts_optimizer'] = 0
                safe_params['optimizer'] = None
                self.model = GaussianProcessRegressor(**safe_params)
                self.model.fit(X, y)
            else:
                # Re-raise other value errors
                raise
        self.optimized_kernel = self.model.kernel_
        self._is_trained = True
        
        # After model is trained, cache CV results
        if kwargs.get('cache_cv', True):
            self._cache_cross_validation_results(experiment_manager)
        
        # Compute calibration factors if requested
        if kwargs.get('calibrate_uncertainty', True) and self.cv_cached_results is not None:
            self._compute_calibration_factors()
        
        return self

    def predict(self, X, return_std=False, **kwargs):
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            return_std: Whether to return standard deviations
            
        Returns:
            If return_std is False: numpy array of predictions
            If return_std is True: tuple of (predictions, standard deviations)
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
            
        X_processed = self._preprocess_X(X)
        predictions = self.model.predict(X_processed, return_std=return_std)
        
        # Handle output scaling inverse transform
        if return_std:
            pred_mean, pred_std = predictions
            
            # Safety check: replace invalid/negative std with small positive value
            # Sklearn GP can produce negative variances due to numerical issues
            pred_std = np.maximum(pred_std, 1e-6)
            
            # Apply calibration to standard deviation if enabled
            if self.calibration_enabled and np.isfinite(self.calibration_factor):
                pred_std = pred_std * self.calibration_factor
            
            # Inverse transform the mean predictions
            if self.output_scaler is not None:
                pred_mean_scaled = self.output_scaler.inverse_transform(pred_mean.reshape(-1, 1)).ravel()
                # For standard deviation, we need to scale by the output scaler's scale
                if hasattr(self.output_scaler, 'scale_'):
                    pred_std_scaled = pred_std * self.output_scaler.scale_[0]
                else:
                    # For MinMaxScaler, use the data range
                    pred_std_scaled = pred_std * (self.output_scaler.data_max_[0] - self.output_scaler.data_min_[0])
            else:
                pred_mean_scaled = pred_mean
                pred_std_scaled = pred_std
            return pred_mean_scaled, pred_std_scaled
        else:
            # Single prediction case
            if self.output_scaler is not None:
                predictions_scaled = self.output_scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()
            else:
                predictions_scaled = predictions
            return predictions_scaled

    def predict_with_std(self, X):
        """
        Make predictions with standard deviation.
        
        Args:
            X: Input features (DataFrame or array)
            
        Returns:
            Tuple of (predictions, standard deviations)
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
            
        # Use the main predict method with return_std=True for consistency
        return self.predict(X, return_std=True)

    def evaluate(self, experiment_manager, cv_splits=5, debug=False, progress_callback=None, **kwargs):
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation.")

        # Get ORIGINAL unscaled data for proper cross-validation
        X_orig, y_orig, noise = experiment_manager.get_features_target_and_noise()
        categorical_variables = experiment_manager.search_space.get_categorical_variables()
        
        # FIT SCALERS ON FULL DATASET to maintain consistent scaling like BoTorch
        # This is the key fix - we use the same scalers for all subset evaluations
        full_X_processed = self._preprocess_subset(X_orig, categorical_variables, fit_scalers=True)
        full_y_processed = self._scale_output(y_orig.values.reshape(-1, 1), fit_scaler=True).ravel()
        
        if debug:
            X_train, X_test, y_train, y_test = train_test_split(
                X_orig, y_orig, test_size=0.2, random_state=self.random_state
            )
            rmse_values, mae_values, mape_values, r2_values = [], [], [], []
            for i in range(5, len(X_train) + 1):
                subset_X_train = X_train.iloc[:i]
                subset_y_train = y_train.iloc[:i]
                
                # Fit scalers on this subset
                X_processed = self._preprocess_subset(subset_X_train, categorical_variables, fit_scalers=True)
                y_processed = self._scale_output(subset_y_train.values.reshape(-1, 1), fit_scaler=True).ravel()
                
                # Build kernel for this subset's dimensionality
                subset_kernel = self._build_kernel(X_processed)
                
                # Create model with subset-specific kernel but no re-optimization
                eval_model = GaussianProcessRegressor(
                    kernel=subset_kernel,
                    optimizer=None,  # Don't re-optimize
                    random_state=self.random_state
                )
                eval_model.fit(X_processed, y_processed)
                
                # Preprocess test data using the fitted scalers
                X_test_processed = self._preprocess_subset(X_test, categorical_variables, fit_scalers=False)
                y_pred_scaled = eval_model.predict(X_test_processed)
                
                # Inverse transform predictions to original scale
                y_pred_orig = self._inverse_scale_output(y_pred_scaled.reshape(-1, 1)).ravel()
                
                rmse = np.sqrt(mean_squared_error(y_test.values, y_pred_orig))
                mae = mean_absolute_error(y_test.values, y_pred_orig)
                with np.errstate(divide='ignore', invalid='ignore'):
                    mape = np.nanmean(np.abs((y_test.values - y_pred_orig) / (np.abs(y_test.values) + 1e-9))) * 100
                # Only calculate R² if we have at least 2 samples
                if len(y_test.values) >= 2:
                    r2 = r2_score(y_test.values, y_pred_orig)
                else:
                    r2 = np.nan
                rmse_values.append(rmse)
                mae_values.append(mae)
                mape_values.append(mape)
                r2_values.append(r2)
                if progress_callback:
                    progress_callback(i / len(X_train))
            return {"RMSE": rmse_values, "MAE": mae_values, "MAPE": mape_values, "R²": r2_values}
        else:
            kf = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
            rmse_values, mae_values, mape_values, r2_values = [], [], [], []
            
            for i in range(5, len(X_orig) + 1):
                subset_X = X_orig.iloc[:i]
                subset_y = y_orig.iloc[:i]
                
                # Perform manual cross-validation on original unscaled data
                fold_rmse, fold_mae, fold_mape, fold_r2 = [], [], [], []
                
                for train_idx, test_idx in kf.split(subset_X):
                    X_train_fold = subset_X.iloc[train_idx]
                    y_train_fold = subset_y.iloc[train_idx]
                    X_test_fold = subset_X.iloc[test_idx]
                    y_test_fold = subset_y.iloc[test_idx]
                    
                    # Fit scalers on this fold's training data
                    X_train_processed = self._preprocess_subset(X_train_fold, categorical_variables, fit_scalers=True)
                    y_train_processed = self._scale_output(y_train_fold.values.reshape(-1, 1), fit_scaler=True).ravel()
                    
                    # Build kernel for this fold's dimensionality
                    fold_kernel = self._build_kernel(X_train_processed)
                    
                    # Create model with fold-specific kernel but no re-optimization
                    eval_model = GaussianProcessRegressor(
                        kernel=fold_kernel,
                        optimizer=None,  # Don't re-optimize
                        random_state=self.random_state
                    )
                    eval_model.fit(X_train_processed, y_train_processed)
                    
                    # Preprocess test data using the fitted scalers (no refitting)
                    X_test_processed = self._preprocess_subset(X_test_fold, categorical_variables, fit_scalers=False)
                    y_pred_scaled = eval_model.predict(X_test_processed)
                    
                    # Inverse transform predictions to original scale
                    y_pred_orig = self._inverse_scale_output(y_pred_scaled.reshape(-1, 1)).ravel()
                    
                    # Calculate metrics on original scale
                    fold_rmse.append(np.sqrt(mean_squared_error(y_test_fold.values, y_pred_orig)))
                    fold_mae.append(mean_absolute_error(y_test_fold.values, y_pred_orig))
                    with np.errstate(divide='ignore', invalid='ignore'):
                        fold_mape.append(np.nanmean(np.abs((y_test_fold.values - y_pred_orig) / (np.abs(y_test_fold.values) + 1e-9))) * 100)
                    # Only calculate R² if we have at least 2 samples
                    if len(y_test_fold.values) >= 2:
                        fold_r2.append(r2_score(y_test_fold.values, y_pred_orig))
                    else:
                        fold_r2.append(np.nan)
                
                # Average across folds
                rmse_values.append(np.mean(fold_rmse))
                mae_values.append(np.mean(fold_mae))
                mape_values.append(np.mean(fold_mape))
                r2_values.append(np.mean(fold_r2))
                
                if progress_callback:
                    progress_callback(i / len(X_orig))
                    
            return {"RMSE": rmse_values, "MAE": mae_values, "MAPE": mape_values, "R²": r2_values}

    def get_hyperparameters(self):
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        return self.model.kernel_.get_params()
    

    def _cache_cross_validation_results(self, experiment_manager, n_splits=5):
        """
        Perform cross-validation and cache the results for faster parity plots.
        Uses original unscaled data with proper CV scaling.
        """
        # Get original unscaled data
        X_orig, y_orig, noise = experiment_manager.get_features_target_and_noise()
        categorical_variables = experiment_manager.search_space.get_categorical_variables()
        
        if len(X_orig) < n_splits:
            return  # Not enough data for CV
            
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        y_true_all = []
        y_pred_all = []
        y_std_all = []
        test_indices_all = []  # original sample index for each cached prediction
        fold_ids_all = []      # which fold each point was held out in

        for fold_id, (train_idx, test_idx) in enumerate(kf.split(X_orig)):
            # Split original data
            X_train_fold = X_orig.iloc[train_idx]
            y_train_fold = y_orig.iloc[train_idx]
            X_test_fold = X_orig.iloc[test_idx]
            y_test_fold = y_orig.iloc[test_idx]
            
            # Preprocess training data and fit scalers
            X_train_processed = self._preprocess_subset(X_train_fold, categorical_variables, fit_scalers=True)
            y_train_processed = self._scale_output(y_train_fold.values.reshape(-1, 1), fit_scaler=True).ravel()
            
            # Create a kernel for this fold's dimensionality
            # (categories might differ between folds, changing feature count)
            fold_kernel = self._build_kernel(X_train_processed)
            
            # Create model with fold-specific kernel but no re-optimization
            cv_model = GaussianProcessRegressor(
                kernel=fold_kernel,
                optimizer=None,  # Don't re-optimize
                random_state=self.random_state
            )
            
            # Train on this fold's training data
            cv_model.fit(X_train_processed, y_train_processed)
            
            # Preprocess test data using the fitted scalers (no refitting)
            X_test_processed = self._preprocess_subset(X_test_fold, categorical_variables, fit_scalers=False)
            y_pred_scaled, y_std_scaled = cv_model.predict(X_test_processed, return_std=True)
            
            # Inverse transform predictions to original scale
            y_pred_orig = self._inverse_scale_output(y_pred_scaled.reshape(-1, 1)).ravel()
            
            # Scale standard deviation to original scale
            if self.output_scaler is not None:
                y_std_orig = y_std_scaled * self.output_scaler.scale_[0]
            else:
                y_std_orig = y_std_scaled
            
            # Store results in original scale
            y_true_all.extend(y_test_fold.values)
            y_pred_all.extend(y_pred_orig)
            y_std_all.extend(y_std_orig)
            test_indices_all.append(np.asarray(test_idx, dtype=np.int64))
            fold_ids_all.append(np.full(len(test_idx), fold_id, dtype=np.int64))

        # Cache the results. ``test_indices[k]`` is the original sample-row index
        # corresponding to ``y_true[k]`` / ``y_pred[k]`` / ``y_std[k]``; ``fold_ids[k]``
        # is the KFold fold (0..n_splits-1) in which that point was held out.
        # The arrays are in fold-test-concatenation order, NOT original sample order.
        self.cv_cached_results = {
            'y_true': np.array(y_true_all),
            'y_pred': np.array(y_pred_all),
            'y_std': np.array(y_std_all),
            'test_indices': np.concatenate(test_indices_all),
            'fold_ids': np.concatenate(fold_ids_all),
        }

    def _compute_calibration_factors(self):
        """
        Compute calibration factor from CV results.
        The calibration_factor is the std of z-scores (standardized residuals).
        This factor will be used to scale predicted std in future predictions.
        Also creates a calibrated copy of CV results for plotting.
        """
        if self.cv_cached_results is None:
            logger.warning("No CV results available for calibration.")
            return
        
        y_true = self.cv_cached_results['y_true']
        y_pred = self.cv_cached_results['y_pred']
        y_std = self.cv_cached_results['y_std']
        
        # Check for numerical issues (zero/negative variances)
        if np.any(y_std <= 0) or np.any(~np.isfinite(y_std)):
            logger.warning("Sklearn GP produced invalid uncertainties (zero/negative/inf). Disabling calibration.")
            self.calibration_enabled = False
            self.calibration_factor = 1.0
            return
        
        # Compute standardized residuals (z-scores)
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        z_scores = (y_true - y_pred) / (y_std + epsilon)
        
        # Check for numerical validity
        if not np.all(np.isfinite(z_scores)):
            logger.warning("Z-scores contain NaN/inf. Disabling calibration.")
            self.calibration_enabled = False
            self.calibration_factor = 1.0
            return
        
        # Calibration factor = std(z)
        self.calibration_factor = np.std(z_scores, ddof=1)
        
        # Final check for valid calibration factor
        if not np.isfinite(self.calibration_factor) or self.calibration_factor <= 0:
            logger.warning(f"Invalid calibration factor: {self.calibration_factor}. Disabling calibration.")
            self.calibration_enabled = False
            self.calibration_factor = 1.0
            return
        
        self.calibration_enabled = True
        
        # Create calibrated copy of CV results for plotting.
        # Propagate ``test_indices`` / ``fold_ids`` from the source cache when
        # present so downstream consumers can map calibrated predictions back to
        # original sample rows. Older caches without these keys are tolerated.
        calibrated = {
            'y_true': y_true.copy(),
            'y_pred': y_pred.copy(),
            'y_std': y_std * self.calibration_factor,  # Apply calibration
        }
        if 'test_indices' in self.cv_cached_results:
            calibrated['test_indices'] = self.cv_cached_results['test_indices'].copy()
        if 'fold_ids' in self.cv_cached_results:
            calibrated['fold_ids'] = self.cv_cached_results['fold_ids'].copy()
        self.cv_cached_results_calibrated = calibrated
        
        # Print calibration info
        logger.info(f"\n{'='*60}")
        logger.info("UNCERTAINTY CALIBRATION")
        logger.info(f"{'='*60}")
        logger.info(f"Calibration factor (s): {self.calibration_factor:.4f}")
        logger.info(f"  - Future σ predictions will be multiplied by {self.calibration_factor:.4f}")
        logger.info(f"  - Note: Acquisition functions use uncalibrated uncertainties")
        
        if self.calibration_factor < 0.8:
            logger.info("  ⚠ Model appears under-confident (s < 1)")
            logger.info("     Predicted uncertainties will be DECREASED")
        elif self.calibration_factor > 1.2:
            logger.info("  ⚠ Model appears over-confident (s > 1)")
            logger.info("     Predicted uncertainties will be INCREASED")
        else:
            logger.info("  ✓ Uncertainty appears well-calibrated")
        
        logger.info(f"{'='*60}\n")

    def generate_contour_data(self, x_range, y_range, fixed_values, x_idx=0, y_idx=1):
        """
        Generate contour plot data for the sklearn model with proper scaling.
        
        Args:
            x_range: Tuple of (min, max) for x-axis values in original scale
            y_range: Tuple of (min, max) for y-axis values in original scale
            fixed_values: Dict mapping dimension indices to fixed values in original scale
            x_idx: Index of the x-axis dimension
            y_idx: Index of the y-axis dimension
            
        Returns:
            Tuple of (X, Y, Z) for contour plotting in original scale
        """
        if self.model is None:
            raise ValueError("Model is not trained yet")
            
        # Get the original variable names from training data
        if hasattr(self, 'original_feature_names') and self.original_feature_names:
            original_feature_names = self.original_feature_names
        elif hasattr(self, 'X_orig') and self.X_orig is not None:
            original_feature_names = self.X_orig.columns.tolist()
        else:
            # Fallback: use feature names if available, otherwise generic names
            if hasattr(self, 'feature_names') and self.feature_names:
                original_feature_names = self.feature_names
            else:
                original_feature_names = [f'dim_{i}' for i in range(2)]
        
        # Check if x or y axes correspond to categorical variables
        x_var_name = original_feature_names[x_idx] if x_idx < len(original_feature_names) else None
        y_var_name = original_feature_names[y_idx] if y_idx < len(original_feature_names) else None
        
        if (hasattr(self, 'categorical_variables') and 
            ((x_var_name and x_var_name in self.categorical_variables) or 
             (y_var_name and y_var_name in self.categorical_variables))):
            raise ValueError(f"Cannot create contour plot with categorical variables on axes. "
                           f"X-axis: {x_var_name}, Y-axis: {y_var_name}. "
                           f"Categorical variables: {self.categorical_variables}")
        
        # Create grid in original scale
        x_vals = np.linspace(x_range[0], x_range[1], 100)
        y_vals = np.linspace(y_range[0], y_range[1], 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        # Create DataFrame for predictions using original variable names
        grid_data = []
        for i in range(len(X.ravel())):
            row = {}
            for dim_idx in range(len(original_feature_names)):
                var_name = original_feature_names[dim_idx]
                
                if dim_idx == x_idx:
                    row[var_name] = X.ravel()[i]
                elif dim_idx == y_idx:
                    row[var_name] = Y.ravel()[i]
                elif dim_idx in fixed_values:
                    # Use the fixed value directly - it should already be in the correct format
                    row[var_name] = fixed_values[dim_idx]
                else:
                    # For variables not being plotted and not in fixed_values,
                    # we need to set appropriate default values
                    if hasattr(self, 'categorical_variables') and var_name in self.categorical_variables:
                        # For categorical variables, use the first category from training data
                        if hasattr(self, 'X_orig') and self.X_orig is not None and var_name in self.X_orig.columns:
                            unique_values = self.X_orig[var_name].unique()
                            row[var_name] = unique_values[0] if len(unique_values) > 0 else 'default'
                        else:
                            row[var_name] = 'default'
                    else:
                        # For numerical variables, use midpoint (0.5)
                        row[var_name] = 0.5
            grid_data.append(row)
        
        # Convert to DataFrame
        grid_df = pd.DataFrame(grid_data)
        
        # Use the model's predict method (which handles scaling internally)
        predictions = self.predict(grid_df)
        
        # Reshape predictions to match grid
        Z = predictions.reshape(X.shape)
        
        return X, Y, Z

    def evaluate_acquisition(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        acq_func: str = 'ucb', 
        acq_func_kwargs: Optional[dict] = None, 
        maximize: bool = True
    ) -> Tuple[np.ndarray, None]:
        """
        Evaluate acquisition function at given points using skopt functions.
        
        Args:
            X: Points to evaluate (DataFrame or array with shape (n, d))
            acq_func: Acquisition function name ('ei', 'pi', 'ucb/lcb')
            acq_func_kwargs: Additional parameters (e.g., {'xi': 0.01, 'kappa': 1.96})
            maximize: Whether we're maximizing (True) or minimizing (False)
            
        Returns:
            Tuple of (acq_values, None) - None because acq functions are deterministic
            
        Example:
            >>> points = pd.DataFrame({'temp': [300, 350, 400], 'pressure': [1, 2, 3]})\n            >>> acq_vals, _ = model.evaluate_acquisition(points, acq_func='ei', maximize=True)
        """
        from skopt.acquisition import gaussian_ei, gaussian_pi, gaussian_lcb
        
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluating acquisition functions.")
        
        # Convert input to expected format
        X_processed = self._preprocess_X(X)
        
        # Get y_opt from training data (in scaled space)
        if maximize:
            y_opt = np.max(self.y_train_)
        else:
            y_opt = np.min(self.y_train_)
        
        # Map acquisition function names
        acq_func_lower = acq_func.lower()
        
        # Parse kwargs with defaults
        if acq_func_kwargs is None:
            acq_func_kwargs = {}
        
        xi = acq_func_kwargs.get('xi', 0.01)
        kappa = acq_func_kwargs.get('kappa', 1.96)
        
        # Evaluate acquisition function
        # NOTE: skopt's gaussian_ei/pi/lcb are designed for MINIMIZATION
        # For maximization, we need to adapt the formulas
        if acq_func_lower in ['ei', 'expectedimprovement']:
            if maximize:
                # For maximization: EI = E[max(f(x) - f(x_best) - xi, 0)]
                # where f(x_best) = y_opt = max(y_train)
                mu, std = self.model.predict(X_processed, return_std=True)
                from scipy.stats import norm
                improve = mu - y_opt + xi  # Improvement over current max
                z = improve / (std + 1e-9)  # Avoid division by zero
                ei = improve * norm.cdf(z) + std * norm.pdf(z)
                acq_values = ei
            else:
                # For minimization: use gaussian_ei directly
                acq_values = gaussian_ei(
                    X_processed, 
                    self.model, 
                    y_opt=y_opt, 
                    xi=xi,
                    return_grad=False
                )
            
        elif acq_func_lower in ['pi', 'probabilityofimprovement']:
            if maximize:
                # For maximization: PI = P(f(x) > f(x_best) + xi)
                mu, std = self.model.predict(X_processed, return_std=True)
                from scipy.stats import norm
                improve = mu - y_opt + xi
                z = improve / (std + 1e-9)
                pi = norm.cdf(z)
                acq_values = pi
            else:
                # For minimization: use gaussian_pi directly
                acq_values = gaussian_pi(
                    X_processed, 
                    self.model, 
                    y_opt=y_opt, 
                    xi=xi,
                    return_grad=False
                )
                
        elif acq_func_lower in ['ucb', 'lcb', 'upperconfidencebound', 'lowerconfidencebound']:
            # For maximization: UCB = mean + kappa*std (higher is better)
            # For minimization: LCB = mean - kappa*std (lower is better)
            if maximize:
                # Calculate UCB directly from predictions
                mu, std = self.model.predict(X_processed, return_std=True)
                acq_values = mu + kappa * std
            else:
                # Use gaussian_lcb for minimization
                acq_values = gaussian_lcb(
                    X_processed, 
                    self.model, 
                    kappa=kappa,
                    return_grad=False
                )
        else:
            raise ValueError(
                f"Unknown acquisition function '{acq_func}' for sklearn backend. "
                f"Valid options are: 'ei', 'pi', 'ucb/lcb'"
            )
        
        # Ensure output is 1D array
        if acq_values.ndim > 1:
            acq_values = acq_values.ravel()
        
        return acq_values, None
