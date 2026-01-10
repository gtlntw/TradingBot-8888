"""
Sequence-based models (LSTM, Transformer) with proper sequence handling.
Uses raw OHLCV data with 60-day lookback windows.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Input,
    Bidirectional, Attention, Concatenate, MultiHeadAttention,
    LayerNormalization, GlobalAveragePooling1D, Add
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import joblib

from trading_bot.models.trainer import BaseModel
from trading_bot.utils.logger import LoggerMixin


class SequenceLSTMModel(BaseModel):
    """LSTM model with bidirectional layers and attention for sequence data."""

    def __init__(self, model_type: str = 'classification', params: Optional[Dict] = None):
        """Initialize Sequence LSTM model."""
        default_params = {
            'units': 64,
            'layers': 2,
            'dropout': 0.3,
            'recurrent_dropout': 0.2,
            'use_bidirectional': True,
            'use_attention': True,
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'patience': 15
        }

        if params:
            default_params.update(params)

        super().__init__(model_type, default_params)

    def build_model(self, input_shape: tuple) -> Model:
        """
        Build LSTM model with bidirectional layers and attention.

        Args:
            input_shape: (sequence_length, num_features)

        Returns:
            Compiled Keras model
        """
        inputs = Input(shape=input_shape)
        x = inputs

        # Bidirectional LSTM layers
        for i in range(self.params['layers']):
            return_sequences = (i < self.params['layers'] - 1) or self.params['use_attention']

            lstm_layer = LSTM(
                self.params['units'] if i == 0 else self.params['units'] // (2 ** i),
                return_sequences=return_sequences,
                dropout=self.params['dropout'],
                recurrent_dropout=self.params['recurrent_dropout']
            )

            if self.params['use_bidirectional']:
                x = Bidirectional(lstm_layer)(x)
            else:
                x = lstm_layer(x)

            if return_sequences:
                x = Dropout(self.params['dropout'])(x)

        # Attention mechanism
        if self.params['use_attention']:
            # Self-attention
            attention_output = Attention()([x, x])
            # Combine attention with LSTM output
            x = Concatenate()([x, attention_output])
            # Global pooling
            x = GlobalAveragePooling1D()(x)
        else:
            # If no attention, x is already from last LSTM layer
            pass

        # Dense layers
        x = Dense(128, activation='relu')(x)
        x = Dropout(self.params['dropout'])(x)
        x = BatchNormalization()(x)

        x = Dense(64, activation='relu')(x)
        x = Dropout(self.params['dropout'])(x)

        # Output layer
        if self.model_type == 'classification':
            outputs = Dense(2, activation='softmax')(x)
        else:
            outputs = Dense(1, activation='linear')(x)

        model = Model(inputs=inputs, outputs=outputs)

        return model

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> 'SequenceLSTMModel':
        """
        Fit Sequence LSTM model.

        Args:
            X: Training sequences (samples, sequence_length, features)
            y: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            feature_names: Feature names (for logging only)

        Returns:
            Self
        """
        self.feature_names = feature_names

        if len(X.shape) != 3:
            raise ValueError(f"X must be 3D array (samples, sequence, features), got shape {X.shape}")

        self.logger.info(
            f"Training Sequence LSTM {self.model_type} model with {X.shape[0]} samples, "
            f"sequence_length={X.shape[1]}, features={X.shape[2]}"
        )
        start_time = datetime.now()

        # Set random seeds
        np.random.seed(42)
        tf.random.set_seed(42)

        # Build model
        input_shape = (X.shape[1], X.shape[2])
        self.model = self.build_model(input_shape)

        # Compile model
        optimizer = Adam(learning_rate=self.params['learning_rate'])

        if self.model_type == 'classification':
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        else:
            loss = 'mse'
            metrics = ['mae']

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # Log model architecture
        self.logger.info(f"Model architecture:")
        self.model.summary(print_fn=lambda x: self.logger.info(x))

        # Prepare callbacks
        callbacks = [
            EarlyStopping(
                patience=self.params['patience'],
                restore_best_weights=True,
                monitor='val_loss' if X_val is not None else 'loss'
            ),
            ReduceLROnPlateau(
                patience=self.params['patience'] // 2,
                factor=0.5,
                monitor='val_loss' if X_val is not None else 'loss'
            )
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
            verbose=1
        )

        self.is_fitted = True

        training_time = (datetime.now() - start_time).total_seconds()
        self.training_history['training_time'] = training_time
        self.training_history['n_samples'] = X.shape[0]
        self.training_history['n_features'] = X.shape[2]
        self.training_history['sequence_length'] = X.shape[1]
        self.training_history['keras_history'] = {k: [float(v) for v in values] for k, values in history.history.items()}

        best_epoch = len(history.history['loss']) - self.params['patience']
        self.logger.info(
            f"Sequence LSTM training completed in {training_time:.2f}s "
            f"(best epoch: {best_epoch})"
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with Sequence LSTM."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if len(X.shape) != 3:
            raise ValueError(f"X must be 3D array, got shape {X.shape}")

        predictions = self.model.predict(X, verbose=0)

        if self.model_type == 'regression':
            return predictions.flatten()
        else:
            return np.argmax(predictions, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return prediction probabilities (for classification)."""
        if self.model_type != 'classification':
            raise ValueError("predict_proba only available for classification models")

        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if len(X.shape) != 3:
            raise ValueError(f"X must be 3D array, got shape {X.shape}")

        return self.model.predict(X, verbose=0)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save Sequence LSTM model."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save Keras model
        model_path = filepath.with_suffix('.keras')
        if self.model:
            self.model.save(model_path, save_format='keras')

        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'params': self.params,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'training_history': self.training_history
        }

        metadata_path = filepath.with_suffix('.meta.pkl')
        joblib.dump(metadata, metadata_path)

        self.logger.info(f"Sequence LSTM model saved to {model_path}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'SequenceLSTMModel':
        """Load Sequence LSTM model."""
        filepath = Path(filepath)

        # Load Keras model
        model_path = filepath.with_suffix('.keras')
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load metadata
        metadata_path = filepath.with_suffix('.meta.pkl')
        metadata = joblib.load(metadata_path)

        # Create instance
        instance = cls(
            model_type=metadata['model_type'],
            params=metadata['params']
        )

        # Load Keras model
        instance.model = tf.keras.models.load_model(model_path)
        instance.is_fitted = metadata['is_fitted']
        instance.feature_names = metadata.get('feature_names')
        instance.training_history = metadata.get('training_history', {})

        return instance


class SequenceTransformerModel(BaseModel):
    """Transformer model with multi-head attention for sequence data."""

    def __init__(self, model_type: str = 'classification', params: Optional[Dict] = None):
        """Initialize Sequence Transformer model."""
        default_params = {
            'num_heads': 4,
            'ff_dim': 64,
            'num_transformer_blocks': 3,
            'mlp_units': [128, 64],
            'dropout': 0.2,
            'mlp_dropout': 0.3,
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.0005,
            'patience': 15
        }

        if params:
            default_params.update(params)

        super().__init__(model_type, default_params)

    def _transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        """Create a transformer encoder block."""
        # Multi-head self-attention
        x = MultiHeadAttention(
            key_dim=head_size,
            num_heads=num_heads,
            dropout=dropout
        )(inputs, inputs)
        x = Dropout(dropout)(x)
        x = LayerNormalization(epsilon=1e-6)(Add()([x, inputs]))

        # Feed-forward network
        ffn_output = Dense(ff_dim, activation="relu")(x)
        ffn_output = Dropout(dropout)(ffn_output)
        ffn_output = Dense(inputs.shape[-1])(ffn_output)
        ffn_output = Dropout(dropout)(ffn_output)

        # Residual connection and normalization
        return LayerNormalization(epsilon=1e-6)(Add()([ffn_output, x]))

    def build_model(self, input_shape: tuple) -> Model:
        """
        Build Transformer model.

        Args:
            input_shape: (sequence_length, num_features)

        Returns:
            Compiled Keras model
        """
        inputs = Input(shape=input_shape)
        x = inputs

        # Add transformer blocks
        head_size = input_shape[1] // self.params['num_heads']

        for _ in range(self.params['num_transformer_blocks']):
            x = self._transformer_encoder(
                x,
                head_size,
                self.params['num_heads'],
                self.params['ff_dim'],
                self.params['dropout']
            )

        # Global pooling
        x = GlobalAveragePooling1D()(x)

        # MLP layers
        for units in self.params['mlp_units']:
            x = Dense(units, activation="relu")(x)
            x = Dropout(self.params['mlp_dropout'])(x)
            x = BatchNormalization()(x)

        # Output layer
        if self.model_type == 'classification':
            outputs = Dense(2, activation='softmax')(x)
        else:
            outputs = Dense(1, activation='linear')(x)

        model = Model(inputs=inputs, outputs=outputs)

        return model

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> 'SequenceTransformerModel':
        """
        Fit Sequence Transformer model.

        Args:
            X: Training sequences (samples, sequence_length, features)
            y: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            feature_names: Feature names

        Returns:
            Self
        """
        self.feature_names = feature_names

        if len(X.shape) != 3:
            raise ValueError(f"X must be 3D array, got shape {X.shape}")

        self.logger.info(
            f"Training Sequence Transformer {self.model_type} model with {X.shape[0]} samples, "
            f"sequence_length={X.shape[1]}, features={X.shape[2]}"
        )
        start_time = datetime.now()

        # Set random seeds
        np.random.seed(42)
        tf.random.set_seed(42)

        # Build model
        input_shape = (X.shape[1], X.shape[2])
        self.model = self.build_model(input_shape)

        # Compile model
        optimizer = Adam(learning_rate=self.params['learning_rate'])

        if self.model_type == 'classification':
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        else:
            loss = 'mse'
            metrics = ['mae']

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # Callbacks
        callbacks = [
            EarlyStopping(
                patience=self.params['patience'],
                restore_best_weights=True,
                monitor='val_loss' if X_val is not None else 'loss'
            ),
            ReduceLROnPlateau(
                patience=self.params['patience'] // 2,
                factor=0.5,
                monitor='val_loss' if X_val is not None else 'loss'
            )
        ]

        # Validation data
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
            verbose=1
        )

        self.is_fitted = True

        training_time = (datetime.now() - start_time).total_seconds()
        self.training_history['training_time'] = training_time
        self.training_history['n_samples'] = X.shape[0]
        self.training_history['n_features'] = X.shape[2]
        self.training_history['sequence_length'] = X.shape[1]
        self.training_history['keras_history'] = {k: [float(v) for v in values] for k, values in history.history.items()}

        self.logger.info(f"Sequence Transformer training completed in {training_time:.2f}s")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with Sequence Transformer."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if len(X.shape) != 3:
            raise ValueError(f"X must be 3D array, got shape {X.shape}")

        predictions = self.model.predict(X, verbose=0)

        if self.model_type == 'regression':
            return predictions.flatten()
        else:
            return np.argmax(predictions, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return prediction probabilities."""
        if self.model_type != 'classification':
            raise ValueError("predict_proba only available for classification")

        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if len(X.shape) != 3:
            raise ValueError(f"X must be 3D array, got shape {X.shape}")

        return self.model.predict(X, verbose=0)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save Sequence Transformer model."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save Keras model
        model_path = filepath.with_suffix('.keras')
        if self.model:
            self.model.save(model_path, save_format='keras')

        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'params': self.params,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'training_history': self.training_history
        }

        metadata_path = filepath.with_suffix('.meta.pkl')
        joblib.dump(metadata, metadata_path)

        self.logger.info(f"Sequence Transformer model saved to {model_path}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'SequenceTransformerModel':
        """Load Sequence Transformer model."""
        filepath = Path(filepath)

        # Load Keras model
        model_path = filepath.with_suffix('.keras')
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load metadata
        metadata_path = filepath.with_suffix('.meta.pkl')
        metadata = joblib.load(metadata_path)

        # Create instance
        instance = cls(
            model_type=metadata['model_type'],
            params=metadata['params']
        )

        # Load Keras model
        instance.model = tf.keras.models.load_model(model_path)
        instance.is_fitted = metadata['is_fitted']
        instance.feature_names = metadata.get('feature_names')
        instance.training_history = metadata.get('training_history', {})

        return instance
