"""
Sequence generation utilities for LSTM and Transformer models.
Creates sliding window sequences from time series data.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from trading_bot.utils.logger import LoggerMixin


class SequenceGenerator(LoggerMixin):
    """Generate sequences for time series models."""

    def __init__(self, sequence_length: int = 60, target_horizon: int = 1):
        """
        Initialize sequence generator.

        Args:
            sequence_length: Number of timesteps to look back
            target_horizon: Number of timesteps ahead to predict
        """
        self.sequence_length = sequence_length
        self.target_horizon = target_horizon

    def create_sequences(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        target_column: Optional[str] = None,
        include_features: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Create sequences from time series data.

        Args:
            data: Input DataFrame with time series data
            feature_columns: List of columns to use for OHLCV sequences
            target_column: Target column name (optional)
            include_features: If True, also return engineered features

        Returns:
            Tuple of (X_sequences, y_targets, X_features)
            - X_sequences: 3D array (samples, sequence_length, features)
            - y_targets: 1D array (samples,) - optional
            - X_features: 2D array (samples, engineered_features) - optional
        """
        self.logger.info(
            f"Creating sequences: length={self.sequence_length}, "
            f"horizon={self.target_horizon}, features={len(feature_columns)}"
        )

        # Ensure data is sorted by date
        data = data.sort_index()

        # Extract OHLCV data for sequences
        ohlcv_data = data[feature_columns].values

        # Create sequences using sliding window
        sequences = []
        targets = []
        feature_rows = []

        for i in range(len(data) - self.sequence_length - self.target_horizon + 1):
            # Extract sequence
            seq = ohlcv_data[i:i + self.sequence_length]
            sequences.append(seq)

            # Extract target if provided
            if target_column is not None:
                target_idx = i + self.sequence_length + self.target_horizon - 1
                target = data[target_column].iloc[target_idx]
                targets.append(target)

            # Extract engineered features if requested
            if include_features:
                # Get features from the last timestep of the sequence
                feature_idx = i + self.sequence_length - 1
                feature_row = data.iloc[feature_idx]
                # Get all columns except OHLCV and target
                feature_cols = [c for c in data.columns
                               if c not in feature_columns and c != target_column]
                features = feature_row[feature_cols].values
                feature_rows.append(features)

        X_sequences = np.array(sequences)
        y_targets = np.array(targets) if targets else None
        X_features = np.array(feature_rows) if feature_rows else None

        self.logger.info(
            f"Created {len(sequences)} sequences with shape {X_sequences.shape}"
        )

        if y_targets is not None:
            self.logger.info(f"Target distribution: {np.bincount(y_targets.astype(int))}")

        return X_sequences, y_targets, X_features

    def create_raw_ohlcv_sequences(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences from raw OHLCV data only (no engineered features).

        Args:
            data: Input DataFrame with OHLCV columns
            target_column: Target column name (optional)

        Returns:
            Tuple of (X_sequences, y_targets)
            - X_sequences: 3D array (samples, sequence_length, 5) for OHLCV + volume
            - y_targets: 1D array (samples,) - optional
        """
        # Use OHLCV columns
        ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']

        # Check which columns exist
        available_columns = [c for c in ohlcv_columns if c in data.columns]

        if not available_columns:
            raise ValueError(f"Data must contain OHLCV columns, found: {data.columns.tolist()}")

        self.logger.info(f"Using OHLCV columns: {available_columns}")

        return self.create_sequences(
            data=data,
            feature_columns=available_columns,
            target_column=target_column,
            include_features=False
        )

    def split_sequences(
        self,
        X_sequences: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split sequences into train and test sets (time-series aware).

        Args:
            X_sequences: Sequence data
            y: Target data
            test_size: Fraction of data to use for testing

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        split_idx = int(len(X_sequences) * (1 - test_size))

        X_train = X_sequences[:split_idx]
        X_test = X_sequences[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]

        self.logger.info(
            f"Split sequences: train={len(X_train)}, test={len(X_test)}"
        )

        return X_train, X_test, y_train, y_test

    def normalize_sequences(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        method: str = 'minmax'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize sequence data.

        Args:
            X_train: Training sequences
            X_test: Test sequences
            method: Normalization method ('minmax' or 'standard')

        Returns:
            Tuple of (X_train_norm, X_test_norm, params)
            params contains normalization parameters for inverse transform
        """
        self.logger.info(f"Normalizing sequences using {method} method")

        if method == 'minmax':
            # Min-Max normalization per feature
            mins = X_train.min(axis=(0, 1))
            maxs = X_train.max(axis=(0, 1))

            # Avoid division by zero
            ranges = maxs - mins
            ranges[ranges == 0] = 1

            X_train_norm = (X_train - mins) / ranges
            X_test_norm = (X_test - mins) / ranges

            params = {'method': 'minmax', 'mins': mins, 'maxs': maxs, 'ranges': ranges}

        elif method == 'standard':
            # Z-score normalization per feature
            means = X_train.mean(axis=(0, 1))
            stds = X_train.std(axis=(0, 1))

            # Avoid division by zero
            stds[stds == 0] = 1

            X_train_norm = (X_train - means) / stds
            X_test_norm = (X_test - means) / stds

            params = {'method': 'standard', 'means': means, 'stds': stds}

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return X_train_norm, X_test_norm, params

    def create_target_from_returns(
        self,
        data: pd.DataFrame,
        close_col: str = 'close',
        transaction_cost: float = 0.002,
        target_name: str = 'profitable_trade'
    ) -> pd.DataFrame:
        """
        Create profitability target based on future returns.

        Args:
            data: Input DataFrame
            close_col: Name of close price column
            transaction_cost: Transaction cost threshold
            target_name: Name for target column

        Returns:
            DataFrame with target column added
        """
        # Calculate future return
        future_return = data[close_col].shift(-self.target_horizon) / data[close_col] - 1

        # Create binary target: 1 if profit > transaction cost
        data[target_name] = (future_return > transaction_cost).astype(float)

        # Drop rows with NaN targets
        data = data.dropna(subset=[target_name])

        self.logger.info(
            f"Created target '{target_name}': "
            f"profitable={data[target_name].sum():.0f} "
            f"({data[target_name].mean()*100:.1f}%), "
            f"unprofitable={(1-data[target_name]).sum():.0f} "
            f"({(1-data[target_name]).mean()*100:.1f}%)"
        )

        return data


class MultiScaleSequenceGenerator(SequenceGenerator):
    """Generate multi-scale sequences for advanced architectures."""

    def __init__(
        self,
        sequence_lengths: List[int] = [14, 60, 180],
        target_horizon: int = 1
    ):
        """
        Initialize multi-scale sequence generator.

        Args:
            sequence_lengths: List of sequence lengths for different scales
            target_horizon: Number of timesteps ahead to predict
        """
        # Use longest sequence as base
        super().__init__(sequence_length=max(sequence_lengths), target_horizon=target_horizon)
        self.sequence_lengths = sorted(sequence_lengths)
        self.logger.info(f"Multi-scale sequences: {self.sequence_lengths}")

    def create_multiscale_sequences(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        target_column: Optional[str] = None
    ) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        """
        Create sequences at multiple time scales.

        Args:
            data: Input DataFrame
            feature_columns: List of columns to use
            target_column: Target column name

        Returns:
            Tuple of (list of X_sequences at different scales, y_targets)
        """
        self.logger.info("Creating multi-scale sequences...")

        # Ensure data is sorted
        data = data.sort_index()
        ohlcv_data = data[feature_columns].values

        # Create sequences at each scale
        all_sequences = []
        targets = []

        # Use longest sequence length to determine valid samples
        max_len = max(self.sequence_lengths)

        for i in range(len(data) - max_len - self.target_horizon + 1):
            # For each scale, extract sequence of appropriate length
            scale_sequences = []

            for seq_len in self.sequence_lengths:
                # Extract last seq_len timesteps before prediction point
                start_idx = i + max_len - seq_len
                end_idx = i + max_len
                seq = ohlcv_data[start_idx:end_idx]
                scale_sequences.append(seq)

            all_sequences.append(scale_sequences)

            # Extract target
            if target_column is not None:
                target_idx = i + max_len + self.target_horizon - 1
                target = data[target_column].iloc[target_idx]
                targets.append(target)

        # Convert to arrays per scale
        sequences_by_scale = []
        for scale_idx in range(len(self.sequence_lengths)):
            scale_data = np.array([seq[scale_idx] for seq in all_sequences])
            sequences_by_scale.append(scale_data)
            self.logger.info(
                f"Scale {scale_idx+1} ({self.sequence_lengths[scale_idx]} steps): "
                f"shape={scale_data.shape}"
            )

        y_targets = np.array(targets) if targets else None

        return sequences_by_scale, y_targets
