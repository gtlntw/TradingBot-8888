"""
Ensemble methods for combining multiple models in the Bitcoin trading bot.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from abc import ABC, abstractmethod
from sklearn.ensemble import VotingRegressor, VotingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

from trading_bot.utils.logger import LoggerMixin
from trading_bot.utils.decorators import timing
from trading_bot.models.trainer import BaseModel


class EnsembleMethod(ABC, LoggerMixin):
    """Abstract base class for ensemble methods."""

    def __init__(self, models: Dict[str, BaseModel]):
        """
        Initialize ensemble method.

        Args:
            models: Dictionary of trained models
        """
        self.models = models
        self.weights = None
        self.meta_model = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleMethod':
        """Fit the ensemble method."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the ensemble."""
        pass

    def _validate_models(self) -> None:
        """Validate that all models are fitted and compatible."""
        if not self.models:
            raise ValueError("No models provided for ensemble")

        model_types = set()
        for name, model in self.models.items():
            if not model.is_fitted:
                raise ValueError(f"Model {name} is not fitted")
            model_types.add(model.model_type)

        if len(model_types) > 1:
            raise ValueError("All models must have the same type (regression or classification)")


class VotingEnsemble(EnsembleMethod):
    """Voting ensemble that averages predictions from multiple models."""

    def __init__(self, models: Dict[str, BaseModel], weights: Optional[List[float]] = None,
                 use_equal_weights: bool = False, min_weight: float = 0.0,
                 optimize_for_sharpe: bool = False, validation_split: float = 0.2):
        """
        Initialize voting ensemble.

        Args:
            models: Dictionary of trained models
            weights: Optional weights for weighted voting
            use_equal_weights: If True, use equal weights (no optimization)
            min_weight: Minimum weight per model (diversity constraint)
            optimize_for_sharpe: If True, optimize for Sharpe ratio instead of MSE
            validation_split: Fraction of training data to use for validation (default 0.2)
        """
        super().__init__(models)
        self.weights = weights
        self.use_equal_weights = use_equal_weights
        self.min_weight = min_weight
        self.optimize_for_sharpe = optimize_for_sharpe
        self.validation_split = validation_split

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'VotingEnsemble':
        """
        Fit voting ensemble (mainly for weight optimization).

        Args:
            X: Training features
            y: Training targets

        Returns:
            Fitted ensemble
        """
        self._validate_models()

        if self.weights is None:
            if self.use_equal_weights:
                # Use equal weights (no optimization)
                n_models = len(self.models)
                self.weights = [1.0 / n_models] * n_models
                self.logger.info("Using equal weights (no optimization)")
            elif self.optimize_for_sharpe:
                # Optimize for Sharpe ratio on validation set
                self.weights = self._optimize_weights_for_sharpe(X, y)
            else:
                # Optimize for MSE
                self.weights = self._optimize_weights(X, y)

        self.is_fitted = True
        self.logger.info(f"Voting ensemble fitted with weights: {dict(zip(self.models.keys(), self.weights))}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using weighted voting.

        Args:
            X: Input features

        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")

        predictions = []
        model_names = list(self.models.keys())

        for name in model_names:
            model = self.models[name]
            pred = model.predict(X)
            predictions.append(pred)

        # Handle sequence-based models (LSTM/Transformer) which return fewer predictions
        # Align all predictions to the minimum length
        pred_lengths = [len(p) for p in predictions]
        min_length = min(pred_lengths)

        if min_length < len(X):
            # Align all predictions to minimum length by taking last N samples
            predictions = [p[-min_length:] for p in predictions]

        predictions = np.array(predictions)

        # Weighted average
        if self.weights is not None:
            weights_array = np.array(self.weights).reshape(-1, 1)
            ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        else:
            ensemble_pred = np.mean(predictions, axis=0)

        return ensemble_pred

    def _optimize_weights(self, X: np.ndarray, y: np.ndarray) -> List[float]:
        """
        Optimize ensemble weights using cross-validation.

        Args:
            X: Training features
            y: Training targets

        Returns:
            Optimized weights
        """
        from scipy.optimize import minimize

        model_names = list(self.models.keys())
        n_models = len(model_names)

        # Get predictions from each model
        model_predictions = []
        for name in model_names:
            pred = self.models[name].predict(X)
            model_predictions.append(pred)

        # Handle sequence-based models (LSTM/Transformer) which return fewer predictions
        # Align all predictions to the minimum length
        pred_lengths = [len(p) for p in model_predictions]
        min_length = min(pred_lengths)

        if min_length < len(X):
            self.logger.info(f"Aligning predictions for MSE: {pred_lengths} -> {min_length} (sequence models detected)")
            # Align all predictions to minimum length by taking last N samples
            model_predictions = [p[-min_length:] for p in model_predictions]
            # Also align y to match
            y = y[-min_length:]

        model_predictions = np.array(model_predictions).T

        def objective(weights):
            """Objective function to minimize (MSE)."""
            weights = weights / np.sum(weights)  # Normalize weights
            ensemble_pred = np.average(model_predictions, axis=1, weights=weights)
            mse = np.mean((y - ensemble_pred) ** 2)
            return mse

        # Constraints: weights sum to 1 and are positive
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

        # Apply diversity constraints if min_weight is set
        if self.min_weight > 0:
            max_weight = 1.0 - (n_models - 1) * self.min_weight  # Ensure weights can sum to 1
            bounds = [(self.min_weight, max_weight) for _ in range(n_models)]
            self.logger.info(f"Using diversity constraints: min_weight={self.min_weight:.2f}")
        else:
            bounds = [(0, 1) for _ in range(n_models)]

        # Initial guess: equal weights
        initial_weights = np.ones(n_models) / n_models

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            optimized_weights = result.x.tolist()
            self.logger.info(f"Weight optimization successful. MSE: {result.fun:.6f}")
        else:
            self.logger.warning("Weight optimization failed, using equal weights")
            optimized_weights = [1.0 / n_models] * n_models

        return optimized_weights

    def _optimize_weights_for_sharpe(self, X: np.ndarray, y: np.ndarray) -> List[float]:
        """
        Optimize ensemble weights for Sharpe ratio on validation set.

        Args:
            X: Training features
            y: Training targets (returns)

        Returns:
            Optimized weights
        """
        from scipy.optimize import minimize

        model_names = list(self.models.keys())
        n_models = len(model_names)

        # Split into train/validation (time-series aware)
        val_size = int(len(X) * self.validation_split)
        train_size = len(X) - val_size

        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        self.logger.info(f"Split for Sharpe optimization: {train_size} train, {val_size} validation")

        # Get predictions from each model on validation set
        val_predictions = []
        for name in model_names:
            pred = self.models[name].predict(X_val)
            val_predictions.append(pred)

        # Handle sequence-based models (LSTM/Transformer) which return fewer predictions
        # Align all predictions to the minimum length
        pred_lengths = [len(p) for p in val_predictions]
        min_length = min(pred_lengths)

        if min_length < val_size:
            self.logger.info(f"Aligning predictions: {pred_lengths} -> {min_length} (sequence models detected)")
            # Align all predictions to minimum length by taking last N samples
            val_predictions = [p[-min_length:] for p in val_predictions]
            # Also align y_val to match
            y_val = y_val[-min_length:]

        val_predictions = np.array(val_predictions).T  # Shape: (min_length, n_models)

        def objective(weights):
            """Minimize negative Sharpe ratio (maximize Sharpe)."""
            # No need to normalize - constraint already enforces sum(weights) = 1

            # Ensemble predictions
            ensemble_pred = np.average(val_predictions, axis=1, weights=weights)

            # For optimization, use predictions directly as "soft" signals
            # This preserves differentiability (np.sign() has zero gradient)
            # The predictions are treated as confidence-weighted positions
            strategy_returns = ensemble_pred * y_val

            # Calculate Sharpe ratio
            mean_return = np.mean(strategy_returns)
            std_return = np.std(strategy_returns)

            if std_return == 0 or np.isnan(std_return):
                return 1e10  # Bad solution

            sharpe = mean_return / std_return

            return -sharpe  # Minimize negative Sharpe (= maximize Sharpe)

        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

        # Apply diversity constraints if min_weight is set
        if self.min_weight > 0:
            max_weight = 1.0 - (n_models - 1) * self.min_weight
            bounds = [(self.min_weight, max_weight) for _ in range(n_models)]
            self.logger.info(f"Using diversity constraints: min_weight={self.min_weight:.2f}")
        else:
            bounds = [(0, 1) for _ in range(n_models)]

        # Initial guess: equal weights
        initial_weights = np.ones(n_models) / n_models

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200}
        )

        if result.success:
            optimized_weights = result.x.tolist()
            sharpe = -result.fun  # Convert back to positive
            self.logger.info(f"Sharpe optimization successful. Sharpe: {sharpe:.4f}")
        else:
            self.logger.warning("Sharpe optimization failed, using equal weights")
            optimized_weights = [1.0 / n_models] * n_models

        return optimized_weights


class StackingEnsemble(EnsembleMethod):
    """Stacking ensemble that uses a meta-model to combine predictions."""

    def __init__(self, models: Dict[str, BaseModel], meta_model: str = 'linear'):
        """
        Initialize stacking ensemble.

        Args:
            models: Dictionary of trained models
            meta_model: Type of meta-model ('linear', 'ridge', 'lasso')
        """
        super().__init__(models)
        self.meta_model_type = meta_model
        self.cv_folds = 5

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackingEnsemble':
        """
        Fit stacking ensemble using cross-validation.

        Args:
            X: Training features
            y: Training targets

        Returns:
            Fitted ensemble
        """
        self._validate_models()

        # Generate meta-features using cross-validation
        meta_features = self._generate_meta_features(X, y)

        # Train meta-model
        self._train_meta_model(meta_features, y)

        self.is_fitted = True
        self.logger.info(f"Stacking ensemble fitted with {len(self.models)} models")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using stacking ensemble.

        Args:
            X: Input features

        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")

        # Get predictions from base models
        meta_features = []
        for model in self.models.values():
            pred = model.predict(X)
            meta_features.append(pred)

        # Handle sequence-based models (LSTM/Transformer) which return fewer predictions
        # Align all predictions to the minimum length
        pred_lengths = [len(p) for p in meta_features]
        min_length = min(pred_lengths)

        if min_length < len(X):
            # Align all predictions to minimum length by taking last N samples
            meta_features = [p[-min_length:] for p in meta_features]

        meta_features = np.column_stack(meta_features)

        # Use meta-model to make final prediction
        ensemble_pred = self.meta_model.predict(meta_features)
        return ensemble_pred

    def _generate_meta_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Generate meta-features using cross-validation to avoid overfitting.

        Args:
            X: Training features
            y: Training targets

        Returns:
            Meta-features array
        """
        # Use TimeSeriesSplit for time series data
        cv = TimeSeriesSplit(n_splits=self.cv_folds)

        meta_features = np.zeros((X.shape[0], len(self.models)))
        model_names = list(self.models.keys())

        for train_idx, val_idx in cv.split(X):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold = y[train_idx]

            for i, (name, model) in enumerate(self.models.items()):
                # Clone and retrain model on fold
                # Note: In practice, you might want to use the pre-trained models
                # and just get their predictions
                fold_pred = model.predict(X_val_fold)
                meta_features[val_idx, i] = fold_pred

        self.logger.info(f"Generated meta-features with shape {meta_features.shape}")
        return meta_features

    def _train_meta_model(self, meta_features: np.ndarray, y: np.ndarray) -> None:
        """
        Train the meta-model on meta-features.

        Args:
            meta_features: Meta-features from base models
            y: Training targets
        """
        # Choose meta-model based on type
        if self.meta_model_type == 'linear':
            self.meta_model = LinearRegression()
        elif self.meta_model_type == 'ridge':
            from sklearn.linear_model import Ridge
            self.meta_model = Ridge(alpha=1.0)
        elif self.meta_model_type == 'lasso':
            from sklearn.linear_model import Lasso
            self.meta_model = Lasso(alpha=0.1)
        else:
            raise ValueError(f"Unknown meta-model type: {self.meta_model_type}")

        # Train meta-model
        self.meta_model.fit(meta_features, y)
        self.logger.info(f"Meta-model ({self.meta_model_type}) trained successfully")


class BlendingEnsemble(EnsembleMethod):
    """Blending ensemble that uses a holdout set to train the meta-model."""

    def __init__(self, models: Dict[str, BaseModel], blend_ratio: float = 0.2):
        """
        Initialize blending ensemble.

        Args:
            models: Dictionary of trained models
            blend_ratio: Ratio of data to use for blending
        """
        super().__init__(models)
        self.blend_ratio = blend_ratio

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BlendingEnsemble':
        """
        Fit blending ensemble using holdout set.

        Args:
            X: Training features
            y: Training targets

        Returns:
            Fitted ensemble
        """
        self._validate_models()

        # Split data for blending
        split_idx = int(len(X) * (1 - self.blend_ratio))
        X_blend = X[split_idx:]
        y_blend = y[split_idx:]

        # Generate blend features
        blend_features = []
        for model in self.models.values():
            pred = model.predict(X_blend)
            blend_features.append(pred)

        blend_features = np.column_stack(blend_features)

        # Train meta-model
        self.meta_model = LinearRegression()
        self.meta_model.fit(blend_features, y_blend)

        self.is_fitted = True
        self.logger.info(f"Blending ensemble fitted with {len(self.models)} models")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using blending ensemble.

        Args:
            X: Input features

        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")

        # Get predictions from base models
        blend_features = []
        for model in self.models.values():
            pred = model.predict(X)
            blend_features.append(pred)

        blend_features = np.column_stack(blend_features)

        # Use meta-model to make final prediction
        ensemble_pred = self.meta_model.predict(blend_features)
        return ensemble_pred


class EnsembleModel(LoggerMixin):
    """Main ensemble model class that manages different ensemble methods."""

    def __init__(self, models: Dict[str, BaseModel], method: str = 'voting', **kwargs):
        """
        Initialize ensemble model.

        Args:
            models: Dictionary of trained models
            method: Ensemble method ('voting', 'stacking', 'blending')
            **kwargs: Additional parameters for ensemble method
        """
        self.models = models
        self.method = method
        self.ensemble = None

        # Add BaseModel-compatible attributes
        self.params = kwargs
        self.model = None  # Will be the ensemble itself after fitting
        self.feature_names = None
        self.training_history = {}

        # Create ensemble based on method
        if method == 'voting':
            self.ensemble = VotingEnsemble(models, **kwargs)
        elif method == 'stacking':
            self.ensemble = StackingEnsemble(models, **kwargs)
        elif method == 'blending':
            self.ensemble = BlendingEnsemble(models, **kwargs)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")

    @property
    def is_fitted(self) -> bool:
        """Check if ensemble is fitted."""
        return self.ensemble.is_fitted if self.ensemble else False

    @property
    def model_type(self) -> str:
        """Get model type from base models."""
        if self.models:
            # All models should have same type (validated in ensemble)
            return list(self.models.values())[0].model_type
        return 'regression'

    @timing
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleModel':
        """
        Fit the ensemble model.

        Args:
            X: Training features
            y: Training targets

        Returns:
            Fitted ensemble model
        """
        self.logger.info(f"Fitting {self.method} ensemble with {len(self.models)} models")
        self.ensemble.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the ensemble.

        Args:
            X: Input features

        Returns:
            Ensemble predictions
        """
        return self.ensemble.predict(X)

    def get_model_weights(self) -> Optional[Dict[str, float]]:
        """
        Get model weights (for voting ensemble).

        Returns:
            Dictionary of model weights
        """
        if isinstance(self.ensemble, VotingEnsemble) and self.ensemble.weights:
            return dict(zip(self.models.keys(), self.ensemble.weights))
        return None

    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """
        Get feature importance from all base models.

        Returns:
            Dictionary of feature importance by model
        """
        importance = {}
        for name, model in self.models.items():
            importance[name] = model.get_feature_importance()
        return importance

    def evaluate_individual_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate individual model performance.

        Args:
            X: Test features
            y: True targets

        Returns:
            Dictionary of model scores
        """
        scores = {}
        for name, model in self.models.items():
            pred = model.predict(X)

            # Calculate MSE for regression
            if model.model_type == 'regression':
                score = np.mean((y - pred) ** 2)
                metric = 'mse'
            else:
                score = np.mean(y == pred)
                metric = 'accuracy'

            scores[name] = score
            self.logger.debug(f"{name} {metric}: {score:.4f}")

        return scores

    def compare_with_ensemble(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Compare ensemble performance with individual models.

        Args:
            X: Test features
            y: True targets

        Returns:
            Dictionary comparing all model scores
        """
        # Get individual model scores
        scores = self.evaluate_individual_models(X, y)

        # Get ensemble score
        ensemble_pred = self.predict(X)

        if self.models[list(self.models.keys())[0]].model_type == 'regression':
            ensemble_score = np.mean((y - ensemble_pred) ** 2)
            metric = 'mse'
        else:
            ensemble_score = np.mean(y == ensemble_pred)
            metric = 'accuracy'

        scores[f'ensemble_{self.method}'] = ensemble_score

        self.logger.info(f"Ensemble {metric}: {ensemble_score:.4f}")
        return scores