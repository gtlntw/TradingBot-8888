"""
Model training module for Bitcoin trading bot.
Handles ML model training, validation, and hyperparameter optimization.
"""

import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod

# Scikit-learn
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score

# XGBoost and LightGBM
import xgboost as xgb
import lightgbm as lgb

# TensorFlow/Keras for neural networks
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from trading_bot.utils.logger import LoggerMixin
from trading_bot.utils.decorators import timing
from trading_bot.utils.helpers import save_object, load_object, save_json
from trading_bot.config.settings import Settings


class BaseModel(ABC, LoggerMixin):
    """Abstract base class for ML models."""

    def __init__(self, model_type: str, params: Dict[str, Any]):
        """
        Initialize base model.

        Args:
            model_type: Type of model (regression or classification)
            params: Model parameters
        """
        self.model_type = model_type
        self.params = params
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.training_history = {}

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseModel':
        """Fit the model to training data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return prediction probabilities (for classification models)."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("Model does not support probability predictions")

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if hasattr(self.model, 'feature_importances_'):
            if self.feature_names:
                return dict(zip(self.feature_names, self.model.feature_importances_))
            else:
                return {f'feature_{i}': imp for i, imp in enumerate(self.model.feature_importances_)}
        else:
            return {}

    def save(self, filepath: Union[str, Path]) -> None:
        """Save model to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'params': self.params,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'training_history': self.training_history
        }

        save_object(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'BaseModel':
        """Load model from file."""
        model_data = load_object(filepath)

        # Create instance with saved parameters
        instance = cls(model_data['model_type'], model_data['params'])
        instance.model = model_data['model']
        instance.is_fitted = model_data['is_fitted']
        instance.feature_names = model_data['feature_names']
        instance.training_history = model_data.get('training_history', {})

        return instance


class RandomForestModel(BaseModel):
    """Random Forest model implementation."""

    def __init__(self, model_type: str = 'regression', params: Optional[Dict] = None):
        """Initialize Random Forest model."""
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }

        if params:
            default_params.update(params)

        super().__init__(model_type, default_params)

        if model_type == 'regression':
            self.model = RandomForestRegressor(**self.params)
        else:
            self.model = RandomForestClassifier(**self.params)

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> 'RandomForestModel':
        """Fit Random Forest model."""
        self.feature_names = feature_names

        self.logger.info(f"Training Random Forest {self.model_type} model with {X.shape[0]} samples")
        start_time = datetime.now()

        self.model.fit(X, y)
        self.is_fitted = True

        training_time = (datetime.now() - start_time).total_seconds()
        self.training_history['training_time'] = training_time
        self.training_history['n_samples'] = X.shape[0]
        self.training_history['n_features'] = X.shape[1]

        self.logger.info(f"Random Forest training completed in {training_time:.2f}s")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with Random Forest."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        return self.model.predict(X)


class XGBoostModel(BaseModel):
    """XGBoost model implementation."""

    def __init__(self, model_type: str = 'regression', params: Optional[Dict] = None):
        """Initialize XGBoost model."""
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }

        if params:
            default_params.update(params)

        super().__init__(model_type, default_params)

        if model_type == 'regression':
            self.model = xgb.XGBRegressor(**self.params)
        else:
            self.model = xgb.XGBClassifier(**self.params)

    def fit(self, X: np.ndarray, y: np.ndarray,
           X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
           feature_names: Optional[List[str]] = None) -> 'XGBoostModel':
        """Fit XGBoost model with optional validation set."""
        self.feature_names = feature_names

        self.logger.info(f"Training XGBoost {self.model_type} model with {X.shape[0]} samples")
        start_time = datetime.now()

        # Prepare evaluation set if validation data provided
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        self.model.fit(
            X, y,
            eval_set=eval_set,
            verbose=False
        )
        self.is_fitted = True

        training_time = (datetime.now() - start_time).total_seconds()
        self.training_history['training_time'] = training_time
        self.training_history['n_samples'] = X.shape[0]
        self.training_history['n_features'] = X.shape[1]

        # Store validation results if available
        if hasattr(self.model, 'evals_result_'):
            self.training_history['validation_results'] = self.model.evals_result_

        self.logger.info(f"XGBoost training completed in {training_time:.2f}s")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with XGBoost."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        return self.model.predict(X)


class LightGBMModel(BaseModel):
    """LightGBM model implementation."""

    def __init__(self, model_type: str = 'regression', params: Optional[Dict] = None):
        """Initialize LightGBM model."""
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }

        if params:
            default_params.update(params)

        super().__init__(model_type, default_params)

        if model_type == 'regression':
            self.model = lgb.LGBMRegressor(**self.params)
        else:
            self.model = lgb.LGBMClassifier(**self.params)

    def fit(self, X: np.ndarray, y: np.ndarray,
           X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
           feature_names: Optional[List[str]] = None) -> 'LightGBMModel':
        """Fit LightGBM model with optional validation set."""
        self.feature_names = feature_names

        self.logger.info(f"Training LightGBM {self.model_type} model with {X.shape[0]} samples")
        start_time = datetime.now()

        # Prepare evaluation set if validation data provided
        eval_set = None
        callbacks = [lgb.log_evaluation(0)]
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            callbacks.append(lgb.early_stopping(100))

        self.model.fit(
            X, y,
            eval_set=eval_set,
            callbacks=callbacks
        )
        self.is_fitted = True

        training_time = (datetime.now() - start_time).total_seconds()
        self.training_history['training_time'] = training_time
        self.training_history['n_samples'] = X.shape[0]
        self.training_history['n_features'] = X.shape[1]

        self.logger.info(f"LightGBM training completed in {training_time:.2f}s")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with LightGBM."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        return self.model.predict(X)


class LSTMModel(BaseModel):
    """LSTM neural network model implementation."""

    def __init__(self, model_type: str = 'regression', params: Optional[Dict] = None):
        """Initialize LSTM model."""
        default_params = {
            'units': 50,
            'dropout': 0.2,
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'patience': 10
        }

        if params:
            default_params.update(params)

        super().__init__(model_type, default_params)

    def fit(self, X: np.ndarray, y: np.ndarray,
           X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
           feature_names: Optional[List[str]] = None) -> 'LSTMModel':
        """Fit LSTM model."""
        self.feature_names = feature_names

        self.logger.info(f"Training LSTM {self.model_type} model with {X.shape[0]} samples")
        start_time = datetime.now()

        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)

        # Reshape 2D input to 3D for LSTM (samples, timesteps=1, features)
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
            if X_val is not None:
                X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])

        # Build model architecture
        self.model = Sequential([
            LSTM(self.params['units'], return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(self.params['dropout']),
            LSTM(self.params['units'] // 2, return_sequences=False),
            Dropout(self.params['dropout']),
            Dense(25),
            BatchNormalization(),
            Dense(1 if self.model_type == 'regression' else 2,
                  activation='linear' if self.model_type == 'regression' else 'softmax')
        ])

        # Compile model
        optimizer = Adam(learning_rate=self.params['learning_rate'])
        loss = 'mse' if self.model_type == 'regression' else 'sparse_categorical_crossentropy'
        metrics = ['mae'] if self.model_type == 'regression' else ['accuracy']

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # Prepare callbacks
        callbacks = [
            EarlyStopping(patience=self.params['patience'], restore_best_weights=True),
            ReduceLROnPlateau(patience=self.params['patience']//2, factor=0.5)
        ]

        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        # Train model
        history = self.model.fit(
            X, y,
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0
        )

        self.is_fitted = True

        training_time = (datetime.now() - start_time).total_seconds()
        self.training_history['training_time'] = training_time
        self.training_history['n_samples'] = X.shape[0]
        self.training_history['n_features'] = X.shape[1]
        self.training_history['keras_history'] = history.history

        self.logger.info(f"LSTM training completed in {training_time:.2f}s")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with LSTM."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Reshape 2D input to 3D for LSTM (samples, timesteps=1, features)
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])

        predictions = self.model.predict(X, verbose=0)

        if self.model_type == 'regression':
            return predictions.flatten()
        else:
            return np.argmax(predictions, axis=1)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save LSTM model to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save Keras model in new format with custom objects
        model_path = filepath.with_suffix('.keras')
        if self.model:
            self.model.save(model_path, save_format='keras')

        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'params': self.params,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'model_path': str(model_path)
        }

        save_object(metadata, filepath)
        self.logger.info(f"LSTM model saved to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'LSTMModel':
        """Load LSTM model from file."""
        metadata = load_object(filepath)

        # Create instance
        instance = cls(metadata['model_type'], metadata['params'])
        instance.is_fitted = metadata['is_fitted']
        instance.feature_names = metadata['feature_names']
        instance.training_history = metadata.get('training_history', {})

        # Load Keras model if it exists
        if instance.is_fitted and 'model_path' in metadata:
            model_path = Path(metadata['model_path'])
            if model_path.exists():
                try:
                    instance.model = tf.keras.models.load_model(model_path)
                except Exception as e:
                    # Try loading with custom objects if needed
                    custom_objects = {'mse': 'mse', 'mae': 'mae'}
                    instance.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

        return instance


class TransformerModel(BaseModel):
    """Transformer model implementation for time series prediction."""

    def __init__(self, model_type: str = 'regression', params: Optional[Dict] = None):
        """Initialize Transformer model."""
        default_params = {
            'num_heads': 8,
            'ff_dim': 32,
            'num_transformer_blocks': 2,
            'mlp_units': [128],
            'dropout': 0.1,
            'mlp_dropout': 0.1,
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'patience': 10
        }

        if params:
            default_params.update(params)

        super().__init__(model_type, default_params)

    def _transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        """Create a transformer encoder block."""
        # Attention and Normalization
        x = MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        x = Dropout(dropout)(x)
        x = LayerNormalization(epsilon=1e-6)(x + inputs)

        # Feed Forward Network
        ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dropout(dropout),
            Dense(inputs.shape[-1]),
        ])
        ffn_output = ffn(x)
        ffn_output = Dropout(dropout)(ffn_output)
        return LayerNormalization(epsilon=1e-6)(x + ffn_output)

    def fit(self, X: np.ndarray, y: np.ndarray,
           X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
           feature_names: Optional[List[str]] = None) -> 'TransformerModel':
        """Fit Transformer model."""
        self.feature_names = feature_names

        self.logger.info(f"Training Transformer {self.model_type} model with {X.shape[0]} samples")
        start_time = datetime.now()

        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)

        # Reshape 2D input to 3D for Transformer (samples, timesteps=1, features)
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
            if X_val is not None:
                X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])

        # Build model architecture
        inputs = Input(shape=(X.shape[1], X.shape[2]))
        x = inputs

        # Add transformer blocks
        head_size = X.shape[2] // self.params['num_heads']
        for _ in range(self.params['num_transformer_blocks']):
            x = self._transformer_encoder(
                x, head_size, self.params['num_heads'],
                self.params['ff_dim'], self.params['dropout']
            )

        # Global pooling and final layers
        x = GlobalAveragePooling1D(data_format="channels_first")(x)

        # Add MLP layers
        for dim in self.params['mlp_units']:
            x = Dense(dim, activation="relu")(x)
            x = Dropout(self.params['mlp_dropout'])(x)

        # Output layer
        outputs = Dense(
            1 if self.model_type == 'regression' else 2,
            activation='linear' if self.model_type == 'regression' else 'softmax'
        )(x)

        self.model = Model(inputs, outputs)

        # Compile model
        optimizer = Adam(learning_rate=self.params['learning_rate'])
        loss = 'mse' if self.model_type == 'regression' else 'sparse_categorical_crossentropy'
        metrics = ['mae'] if self.model_type == 'regression' else ['accuracy']
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # Prepare callbacks
        callbacks = [
            EarlyStopping(patience=self.params['patience'], restore_best_weights=True),
            ReduceLROnPlateau(patience=self.params['patience']//2, factor=0.5)
        ]

        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        # Train model
        history = self.model.fit(
            X, y,
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0
        )

        self.is_fitted = True

        training_time = (datetime.now() - start_time).total_seconds()
        self.training_history['training_time'] = training_time
        self.training_history['n_samples'] = X.shape[0]
        self.training_history['n_features'] = X.shape[2]
        self.training_history['keras_history'] = history.history

        self.logger.info(f"Transformer training completed in {training_time:.2f}s")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with Transformer."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Reshape 2D input to 3D for Transformer (samples, timesteps=1, features)
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])

        predictions = self.model.predict(X, verbose=0)

        if self.model_type == 'regression':
            return predictions.flatten()
        else:
            return np.argmax(predictions, axis=1)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save Transformer model to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save Keras model in new format
        model_path = filepath.with_suffix('.keras')
        if self.model:
            self.model.save(model_path, save_format='keras')

        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'params': self.params,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'model_path': str(model_path)
        }

        save_object(metadata, filepath)
        self.logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'TransformerModel':
        """Load Transformer model from file."""
        filepath = Path(filepath)

        # Load metadata
        metadata = load_object(filepath)

        # Create instance
        instance = cls(metadata['model_type'], metadata.get('params', {}))
        instance.is_fitted = metadata['is_fitted']
        instance.feature_names = metadata['feature_names']
        instance.training_history = metadata.get('training_history', {})

        # Load Keras model if it exists
        model_path = Path(metadata['model_path'])
        if model_path.exists():
            try:
                instance.model = tf.keras.models.load_model(model_path)
            except Exception as e:
                # Try loading with custom objects if needed
                custom_objects = {'mse': 'mse', 'mae': 'mae'}
                instance.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

        return instance


class ModelTrainer(LoggerMixin):
    """Main model trainer that handles training workflow."""

    def __init__(self, settings: Settings):
        """
        Initialize model trainer.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.models = {}
        self.model_registry = {
            'random_forest': RandomForestModel,
            'xgboost': XGBoostModel,
            'lightgbm': LightGBMModel,
            'lstm': LSTMModel,
            'transformer': TransformerModel
        }

    @timing
    def train_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        model_type: str = 'regression'
    ) -> Dict[str, BaseModel]:
        """
        Train multiple models based on configuration.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            feature_names: Feature names
            model_type: Type of models to train

        Returns:
            Dictionary of trained models
        """
        algorithms = self.settings.algorithms
        self.logger.info(f"Training {len(algorithms)} models: {algorithms}")

        trained_models = {}

        for algorithm in algorithms:
            if algorithm not in self.model_registry:
                self.logger.warning(f"Unknown algorithm: {algorithm}")
                continue

            try:
                # Get model configuration
                model_params = self.settings.get_model_config(algorithm)

                # Create and train model
                model_class = self.model_registry[algorithm]
                model = model_class(model_type=model_type, params=model_params)

                # Train with validation data if available
                if algorithm in ['xgboost', 'lightgbm', 'lstm'] and X_val is not None:
                    model.fit(X_train, y_train, X_val, y_val, feature_names)
                else:
                    model.fit(X_train, y_train, feature_names=feature_names)

                trained_models[algorithm] = model
                self.logger.info(f"Successfully trained {algorithm} model")

            except Exception as e:
                self.logger.error(f"Failed to train {algorithm} model: {e}")
                continue

        self.models = trained_models
        return trained_models

    def optimize_hyperparameters(
        self,
        algorithm: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: Dict[str, List],
        cv_folds: int = 3,
        scoring: str = 'neg_mean_squared_error',
        search_type: str = 'grid'
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a specific algorithm.

        Args:
            algorithm: Algorithm name
            X_train: Training features
            y_train: Training targets
            param_grid: Parameter search space
            cv_folds: Number of CV folds
            scoring: Scoring metric
            search_type: 'grid' or 'random'

        Returns:
            Best parameters and CV results
        """
        if algorithm not in self.model_registry:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        self.logger.info(f"Optimizing hyperparameters for {algorithm}")

        # Create base model
        model_class = self.model_registry[algorithm]
        base_model = model_class().model

        # Use TimeSeriesSplit for cross-validation
        cv = TimeSeriesSplit(n_splits=cv_folds)

        # Choose search strategy
        if search_type == 'grid':
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
        else:
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                n_iter=50,
                verbose=1,
                random_state=42
            )

        # Perform search
        search.fit(X_train, y_train)

        results = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_
        }

        self.logger.info(f"Best {algorithm} parameters: {search.best_params_}")
        self.logger.info(f"Best CV score: {search.best_score_:.4f}")

        return results

    def save_models(self, directory: Union[str, Path], timestamp: Optional[str] = None) -> None:
        """
        Save all trained models.

        Args:
            directory: Directory to save models
            timestamp: Optional timestamp for versioning
        """
        if not self.models:
            self.logger.warning("No models to save")
            return

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for name, model in self.models.items():
            model_path = directory / f"{name}_{timestamp}.pkl"
            model.save(model_path)

        # Save training metadata
        metadata = {
            'timestamp': timestamp,
            'models': list(self.models.keys()),
            'settings': self.settings.to_dict()
        }

        metadata_path = directory / f"training_metadata_{timestamp}.json"
        save_json(metadata, metadata_path)

        self.logger.info(f"Saved {len(self.models)} models to {directory}")

    def load_models(self, directory: Union[str, Path], timestamp: str) -> Dict[str, BaseModel]:
        """
        Load previously trained models.

        Args:
            directory: Directory containing models
            timestamp: Timestamp of model version

        Returns:
            Dictionary of loaded models
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Model directory not found: {directory}")

        loaded_models = {}

        for algorithm in self.model_registry.keys():
            model_path = directory / f"{algorithm}_{timestamp}.pkl"

            if model_path.exists():
                try:
                    model_class = self.model_registry[algorithm]
                    model = model_class.load(model_path)
                    loaded_models[algorithm] = model
                    self.logger.info(f"Loaded {algorithm} model")
                except Exception as e:
                    self.logger.error(f"Failed to load {algorithm} model: {e}")

        self.models = loaded_models
        return loaded_models

    def get_model_summary(self) -> Dict[str, Dict]:
        """
        Get summary of all trained models.

        Returns:
            Dictionary with model summaries
        """
        summary = {}

        for name, model in self.models.items():
            summary[name] = {
                'model_type': model.model_type,
                'is_fitted': model.is_fitted,
                'n_features': model.training_history.get('n_features'),
                'training_time': model.training_history.get('training_time'),
                'feature_importance': model.get_feature_importance()
            }

        return summary