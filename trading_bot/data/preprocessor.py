"""
Data preprocessing module for Bitcoin trading bot.
Handles data cleaning, normalization, and preparation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer

from trading_bot.utils.logger import LoggerMixin
from trading_bot.utils.helpers import ensure_dataframe_index


class DataPreprocessor(LoggerMixin):
    """Data preprocessing and cleaning utilities."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize data preprocessor.

        Args:
            config: Preprocessing configuration
        """
        self.config = config or {}
        self.scalers = {}
        self.imputers = {}

    def clean_data(self, df: pd.DataFrame, remove_outliers: bool = True) -> pd.DataFrame:
        """
        Clean and validate data.

        Args:
            df: Input DataFrame
            remove_outliers: Whether to remove outliers

        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()

        # Ensure datetime index
        df_clean = ensure_dataframe_index(df_clean)

        # Remove duplicates
        initial_len = len(df_clean)
        df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
        if len(df_clean) < initial_len:
            self.logger.info(f"Removed {initial_len - len(df_clean)} duplicate rows")

        # Sort by timestamp
        df_clean = df_clean.sort_index()

        # Validate OHLCV data consistency
        df_clean = self._validate_ohlcv(df_clean)

        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)

        # Remove outliers if requested
        if remove_outliers:
            df_clean = self._remove_outliers(df_clean)

        # Validate data ranges
        df_clean = self._validate_ranges(df_clean)

        self.logger.info(f"Data cleaning completed. Shape: {df_clean.shape}")
        return df_clean

    def _validate_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate OHLCV data consistency.

        Args:
            df: Input DataFrame

        Returns:
            Validated DataFrame
        """
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            self.logger.warning("Missing OHLC columns, skipping OHLCV validation")
            return df

        df_valid = df.copy()

        # Check that high >= max(open, close) and low <= min(open, close)
        invalid_high = df_valid['high'] < df_valid[['open', 'close']].max(axis=1)
        invalid_low = df_valid['low'] > df_valid[['open', 'close']].min(axis=1)

        if invalid_high.any():
            self.logger.warning(f"Found {invalid_high.sum()} rows with invalid high prices")
            # Correct invalid high values
            df_valid.loc[invalid_high, 'high'] = df_valid.loc[invalid_high, ['open', 'close']].max(axis=1)

        if invalid_low.any():
            self.logger.warning(f"Found {invalid_low.sum()} rows with invalid low prices")
            # Correct invalid low values
            df_valid.loc[invalid_low, 'low'] = df_valid.loc[invalid_low, ['open', 'close']].min(axis=1)

        # Check for negative values
        price_cols = ['open', 'high', 'low', 'close']
        negative_prices = (df_valid[price_cols] <= 0).any(axis=1)
        if negative_prices.any():
            self.logger.warning(f"Found {negative_prices.sum()} rows with non-positive prices")
            df_valid = df_valid[~negative_prices]

        # Check for negative volume
        if 'volume' in df_valid.columns:
            negative_volume = df_valid['volume'] < 0
            if negative_volume.any():
                self.logger.warning(f"Found {negative_volume.sum()} rows with negative volume")
                df_valid.loc[negative_volume, 'volume'] = 0

        return df_valid

    def _handle_missing_values(self, df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """
        Handle missing values in the data.

        Args:
            df: Input DataFrame
            method: Method to handle missing values

        Returns:
            DataFrame with missing values handled
        """
        df_filled = df.copy()
        missing_count = df_filled.isnull().sum().sum()

        if missing_count == 0:
            return df_filled

        self.logger.info(f"Handling {missing_count} missing values using {method}")

        if method == 'forward_fill':
            df_filled = df_filled.fillna(method='ffill')
            # If still missing values at the beginning, use backward fill
            df_filled = df_filled.fillna(method='bfill')

        elif method == 'interpolate':
            df_filled = df_filled.interpolate(method='linear')

        elif method == 'drop':
            df_filled = df_filled.dropna()

        elif method == 'knn':
            # Use KNN imputation for numerical columns
            numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                imputer = KNNImputer(n_neighbors=5)
                df_filled[numeric_cols] = imputer.fit_transform(df_filled[numeric_cols])

        elif method == 'mean':
            # Fill with mean for numerical columns
            numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
            df_filled[numeric_cols] = df_filled[numeric_cols].fillna(df_filled[numeric_cols].mean())

        remaining_missing = df_filled.isnull().sum().sum()
        if remaining_missing > 0:
            self.logger.warning(f"Still {remaining_missing} missing values after handling")

        return df_filled

    def _remove_outliers(self, df: pd.DataFrame, method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers from the data.

        Args:
            df: Input DataFrame
            method: Outlier detection method ('iqr', 'zscore', 'isolation_forest')
            threshold: Threshold for outlier detection

        Returns:
            DataFrame with outliers removed
        """
        df_clean = df.copy()
        initial_len = len(df_clean)

        # Apply outlier detection to price columns
        price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in df_clean.columns]

        if method == 'iqr':
            for col in price_cols:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                df_clean = df_clean[~outliers]

        elif method == 'zscore':
            for col in price_cols:
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                outliers = z_scores > threshold
                df_clean = df_clean[~outliers]

        elif method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest

            isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = isolation_forest.fit_predict(df_clean[price_cols]) == -1
            df_clean = df_clean[~outliers]

        outliers_removed = initial_len - len(df_clean)
        if outliers_removed > 0:
            self.logger.info(f"Removed {outliers_removed} outliers using {method} method")

        return df_clean

    def _validate_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate data ranges and remove extreme values.

        Args:
            df: Input DataFrame

        Returns:
            Validated DataFrame
        """
        df_valid = df.copy()

        # Define reasonable ranges for Bitcoin price (adjust as needed)
        price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in df_valid.columns]

        # Remove rows with extremely high prices (likely errors)
        for col in price_cols:
            extreme_high = df_valid[col] > 1000000  # $1M per BTC seems unreasonable for now
            extreme_low = df_valid[col] < 0.01      # Less than 1 cent

            if extreme_high.any():
                self.logger.warning(f"Removing {extreme_high.sum()} rows with extremely high {col} prices")
                df_valid = df_valid[~extreme_high]

            if extreme_low.any():
                self.logger.warning(f"Removing {extreme_low.sum()} rows with extremely low {col} prices")
                df_valid = df_valid[~extreme_low]

        return df_valid

    def normalize_data(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'standard'
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize/scale the data.

        Args:
            df: Input DataFrame
            columns: Columns to normalize (None for all numeric)
            method: Normalization method ('standard', 'minmax', 'robust')

        Returns:
            Tuple of (normalized DataFrame, scaler dictionary)
        """
        df_norm = df.copy()

        if columns is None:
            columns = df_norm.select_dtypes(include=[np.number]).columns.tolist()

        scalers = {}

        for col in columns:
            if col not in df_norm.columns:
                continue

            # Choose scaler based on method
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown normalization method: {method}")

            # Fit and transform the data
            df_norm[col] = scaler.fit_transform(df_norm[[col]])
            scalers[col] = scaler

        self.scalers.update(scalers)
        self.logger.info(f"Normalized {len(columns)} columns using {method} scaling")

        return df_norm, scalers

    def denormalize_data(self, df: pd.DataFrame, scalers: Dict) -> pd.DataFrame:
        """
        Denormalize data using provided scalers.

        Args:
            df: Normalized DataFrame
            scalers: Dictionary of column scalers

        Returns:
            Denormalized DataFrame
        """
        df_denorm = df.copy()

        for col, scaler in scalers.items():
            if col in df_denorm.columns:
                df_denorm[col] = scaler.inverse_transform(df_denorm[[col]])

        self.logger.debug(f"Denormalized {len(scalers)} columns")
        return df_denorm

    def create_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int,
        target_column: str = 'close',
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series modeling.

        Args:
            df: Input DataFrame
            sequence_length: Length of input sequences
            target_column: Target column name
            feature_columns: Feature columns (None for all except target)

        Returns:
            Tuple of (X, y) arrays
        """
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]

        # Ensure we have enough data
        if len(df) < sequence_length + 1:
            raise ValueError(f"Not enough data for sequence length {sequence_length}")

        X, y = [], []

        for i in range(len(df) - sequence_length):
            # Input sequence
            X.append(df[feature_columns].iloc[i:i + sequence_length].values)
            # Target value
            y.append(df[target_column].iloc[i + sequence_length])

        X = np.array(X)
        y = np.array(y)

        self.logger.info(f"Created {len(X)} sequences of length {sequence_length}")
        return X, y

    def split_time_series(
        self,
        df: pd.DataFrame,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split time series data maintaining temporal order.

        Args:
            df: Input DataFrame
            train_size: Training set proportion
            val_size: Validation set proportion
            test_size: Test set proportion

        Returns:
            Tuple of (train, validation, test) DataFrames
        """
        if abs(train_size + val_size + test_size - 1.0) > 1e-6:
            raise ValueError("Train, validation and test sizes must sum to 1.0")

        n = len(df)
        train_end = int(n * train_size)
        val_end = int(n * (train_size + val_size))

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        self.logger.info(
            f"Split data: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples"
        )

        return train_df, val_df, test_df

    def add_lag_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lags: List[int]
    ) -> pd.DataFrame:
        """
        Add lagged features to the DataFrame.

        Args:
            df: Input DataFrame
            columns: Columns to create lags for
            lags: List of lag periods

        Returns:
            DataFrame with lag features
        """
        df_lag = df.copy()

        for col in columns:
            if col not in df_lag.columns:
                continue

            for lag in lags:
                lag_col = f"{col}_lag_{lag}"
                df_lag[lag_col] = df_lag[col].shift(lag)

        # Drop rows with NaN values from lagging
        max_lag = max(lags)
        df_lag = df_lag.iloc[max_lag:]

        self.logger.info(f"Added {len(columns) * len(lags)} lag features")
        return df_lag

    def add_rolling_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        windows: List[int],
        functions: List[str] = ['mean', 'std']
    ) -> pd.DataFrame:
        """
        Add rolling window features.

        Args:
            df: Input DataFrame
            columns: Columns to create rolling features for
            windows: List of window sizes
            functions: List of functions to apply

        Returns:
            DataFrame with rolling features
        """
        df_roll = df.copy()

        for col in columns:
            if col not in df_roll.columns:
                continue

            for window in windows:
                for func in functions:
                    feature_name = f"{col}_rolling_{window}_{func}"

                    if func == 'mean':
                        df_roll[feature_name] = df_roll[col].rolling(window=window).mean()
                    elif func == 'std':
                        df_roll[feature_name] = df_roll[col].rolling(window=window).std()
                    elif func == 'min':
                        df_roll[feature_name] = df_roll[col].rolling(window=window).min()
                    elif func == 'max':
                        df_roll[feature_name] = df_roll[col].rolling(window=window).max()
                    elif func == 'median':
                        df_roll[feature_name] = df_roll[col].rolling(window=window).median()

        # Drop rows with NaN values from rolling calculations
        max_window = max(windows)
        df_roll = df_roll.iloc[max_window - 1:]

        added_features = len(columns) * len(windows) * len(functions)
        self.logger.info(f"Added {added_features} rolling features")

        return df_roll