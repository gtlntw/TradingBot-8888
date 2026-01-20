"""
Feature engineering module for Bitcoin trading bot.
Creates technical indicators and market features.
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional, Union
from abc import ABC, abstractmethod

from trading_bot.utils.logger import LoggerMixin
from trading_bot.utils.decorators import timing


class FeatureCalculator(ABC):
    """Abstract base class for feature calculators."""

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for the given DataFrame."""
        pass


class TechnicalIndicators(FeatureCalculator, LoggerMixin):
    """Calculate technical indicators using the ta library."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize technical indicators calculator.

        Args:
            config: Configuration for indicators
        """
        self.config = config or {}

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with technical indicators
        """
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            raise ValueError("DataFrame must contain OHLCV columns")

        df_features = df.copy()

        # Trend indicators
        df_features = self._add_trend_indicators(df_features)

        # Momentum indicators
        df_features = self._add_momentum_indicators(df_features)

        # Volume indicators
        df_features = self._add_volume_indicators(df_features)

        # Volatility indicators
        df_features = self._add_volatility_indicators(df_features)

        # Support/Resistance levels
        df_features = self._add_support_resistance(df_features)

        self.logger.info(f"Added technical indicators. Features: {df_features.shape[1]}")
        return df_features

    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-based technical indicators."""
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = ta.trend.sma_indicator(df['close'], window=period)

        # Exponential Moving Averages
        for period in [5, 10, 20, 50, 100]:
            df[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], window=period)

        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()

        # ADX (Average Directional Index)
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        df['adx_pos'] = ta.trend.adx_pos(df['high'], df['low'], df['close'], window=14)
        df['adx_neg'] = ta.trend.adx_neg(df['high'], df['low'], df['close'], window=14)

        # Aroon
        aroon = ta.trend.AroonIndicator(df['high'], df['low'], window=25)
        df['aroon_up'] = aroon.aroon_up()
        df['aroon_down'] = aroon.aroon_down()
        df['aroon_indicator'] = aroon.aroon_indicator()

        # Parabolic SAR
        psar_indicator = ta.trend.PSARIndicator(df['high'], df['low'], df['close'])
        df['psar'] = psar_indicator.psar()

        # Commodity Channel Index
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)

        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based technical indicators."""
        # RSI
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = ta.momentum.rsi(df['close'], window=period)

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        # Williams %R
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)

        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = ta.momentum.roc(df['close'], window=period)

        # Money Flow Index
        df['mfi'] = ta.volume.money_flow_index(
            df['high'], df['low'], df['close'], df['volume'], window=14
        )

        # Awesome Oscillator
        df['awesome_oscillator'] = ta.momentum.awesome_oscillator(df['high'], df['low'])

        # KAMA (Kaufman Adaptive Moving Average)
        df['kama'] = ta.momentum.kama(df['close'], window=10)

        return df

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based technical indicators."""
        # Volume SMA
        for period in [10, 20, 50]:
            df[f'volume_sma_{period}'] = ta.trend.sma_indicator(df['volume'], window=period)

        # On Balance Volume
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])

        # Accumulation/Distribution Line
        df['ad'] = ta.volume.acc_dist_index(df['high'], df['low'], df['close'], df['volume'])

        # Chaikin Money Flow
        df['cmf'] = ta.volume.chaikin_money_flow(
            df['high'], df['low'], df['close'], df['volume'], window=20
        )

        # Force Index
        df['force_index'] = ta.volume.force_index(df['close'], df['volume'], window=13)

        # Ease of Movement
        df['eom'] = ta.volume.ease_of_movement(
            df['high'], df['low'], df['volume'], window=14
        )

        # Volume Price Trend
        df['vpt'] = ta.volume.volume_price_trend(df['close'], df['volume'])

        # Negative Volume Index
        df['nvi'] = ta.volume.negative_volume_index(df['close'], df['volume'])

        return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based technical indicators."""
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        df['bb_percent'] = bb.bollinger_pband()

        # Average True Range
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

        # Donchian Channel
        dc = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'], window=20)
        df['dc_upper'] = dc.donchian_channel_hband()
        df['dc_middle'] = dc.donchian_channel_mband()
        df['dc_lower'] = dc.donchian_channel_lband()
        df['dc_width'] = dc.donchian_channel_wband()

        # Keltner Channel
        kc = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'], window=20)
        df['kc_upper'] = kc.keltner_channel_hband()
        df['kc_middle'] = kc.keltner_channel_mband()
        df['kc_lower'] = kc.keltner_channel_lband()
        df['kc_width'] = kc.keltner_channel_wband()

        # Ulcer Index
        df['ulcer_index'] = ta.volatility.ulcer_index(df['close'], window=14)

        return df

    def _add_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add support and resistance levels."""
        # Pivot Points
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['r1'] = 2 * df['pivot'] - df['low']
        df['s1'] = 2 * df['pivot'] - df['high']
        df['r2'] = df['pivot'] + (df['high'] - df['low'])
        df['s2'] = df['pivot'] - (df['high'] - df['low'])

        # Fibonacci retracement levels (based on recent high/low)
        window = 20
        df['high_20'] = df['high'].rolling(window=window).max()
        df['low_20'] = df['low'].rolling(window=window).min()
        df['fib_236'] = df['low_20'] + 0.236 * (df['high_20'] - df['low_20'])
        df['fib_382'] = df['low_20'] + 0.382 * (df['high_20'] - df['low_20'])
        df['fib_500'] = df['low_20'] + 0.500 * (df['high_20'] - df['low_20'])
        df['fib_618'] = df['low_20'] + 0.618 * (df['high_20'] - df['low_20'])

        return df


class MarketFeatures(FeatureCalculator, LoggerMixin):
    """Calculate market-based features."""

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market features.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with market features
        """
        df_features = df.copy()

        # Price-based features
        df_features = self._add_price_features(df_features)

        # Return-based features
        df_features = self._add_return_features(df_features)

        # Volatility features
        df_features = self._add_volatility_features(df_features)

        # Time-based features
        df_features = self._add_time_features(df_features)

        self.logger.info(f"Added market features. Features: {df_features.shape[1]}")
        return df_features

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        # Price ratios
        df['open_close_ratio'] = df['open'] / df['close']
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']

        # Price ranges
        df['daily_range'] = df['high'] - df['low']
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['body_size'] = np.abs(df['close'] - df['open'])

        # Price position within range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])

        # Distance from moving averages
        for period in [5, 10, 20, 50]:
            ma = df['close'].rolling(window=period).mean()
            df[f'price_ma_distance_{period}'] = (df['close'] - ma) / ma

        return df

    def _add_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return-based features."""
        # Simple returns
        for period in [1, 2, 3, 5, 10]:
            df[f'return_{period}d'] = df['close'].pct_change(periods=period)

        # Log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))

        # Cumulative returns
        for period in [5, 10, 20]:
            df[f'cum_return_{period}d'] = (1 + df['return_1d']).rolling(window=period).apply(
                lambda x: x.prod() - 1, raw=True
            )

        # Return statistics
        for window in [5, 10, 20]:
            returns = df['return_1d'].rolling(window=window)
            df[f'return_mean_{window}d'] = returns.mean()
            df[f'return_std_{window}d'] = returns.std()
            df[f'return_skew_{window}d'] = returns.skew()
            df[f'return_kurt_{window}d'] = returns.kurt()

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features."""
        # Realized volatility
        for window in [5, 10, 20, 30]:
            df[f'volatility_{window}d'] = df['return_1d'].rolling(window=window).std() * np.sqrt(252)

        # Parkinson volatility (using high-low)
        for window in [5, 10, 20]:
            hl_ratio = np.log(df['high'] / df['low'])
            df[f'parkinson_vol_{window}d'] = np.sqrt(
                hl_ratio.rolling(window=window).mean() / (4 * np.log(2))
            ) * np.sqrt(252)

        # Garman-Klass volatility
        for window in [5, 10, 20]:
            hl = np.log(df['high'] / df['low'])
            co = np.log(df['close'] / df['open'])
            gk_vol = hl - 2 * np.log(2) * co
            df[f'garman_klass_vol_{window}d'] = np.sqrt(
                gk_vol.rolling(window=window).mean()
            ) * np.sqrt(252)

        # Volume-weighted volatility
        for window in [5, 10, 20]:
            vol_weighted_returns = df['return_1d'] * df['volume']
            total_volume = df['volume'].rolling(window=window).sum()
            weighted_avg_return = vol_weighted_returns.rolling(window=window).sum() / total_volume

            vol_weighted_var = (
                ((df['return_1d'] - weighted_avg_return) ** 2 * df['volume'])
                .rolling(window=window).sum() / total_volume
            )
            df[f'vol_weighted_volatility_{window}d'] = np.sqrt(vol_weighted_var) * np.sqrt(252)

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            else:
                df.index = pd.to_datetime(df.index)

        # Day of week
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Month
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter

        # Hour (if intraday data)
        if df.index.freq and 'H' in str(df.index.freq):
            df['hour'] = df.index.hour
            df['is_market_hours'] = df['hour'].between(9, 16).astype(int)

        # Cyclical encoding for time features
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        return df


class SentimentFeatures(FeatureCalculator, LoggerMixin):
    """Calculate sentiment-based features."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize sentiment features calculator."""
        self.config = config or {}

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sentiment features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with sentiment features
        """
        df_features = df.copy()

        # Fear & Greed Index (simulated)
        df_features = self._add_fear_greed_index(df_features)

        # Market sentiment indicators
        df_features = self._add_market_sentiment(df_features)

        self.logger.info(f"Added sentiment features. Features: {df_features.shape[1]}")
        return df_features

    def _add_fear_greed_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Fear & Greed Index (simulated based on price action)."""
        # Simulate Fear & Greed based on multiple factors
        rsi_14 = ta.momentum.rsi(df['close'], window=14)
        volatility = df['close'].pct_change().rolling(20).std()
        volume_ratio = df['volume'] / df['volume'].rolling(50).mean()

        # Normalize components to 0-100 scale
        rsi_norm = rsi_14
        vol_norm = (volatility - volatility.rolling(100).min()) / (
            volatility.rolling(100).max() - volatility.rolling(100).min()
        ) * 100
        vol_norm = 100 - vol_norm  # Invert so low volatility = high score

        volume_norm = np.clip(volume_ratio * 50, 0, 100)

        # Combine factors
        df['fear_greed_index'] = (rsi_norm * 0.4 + vol_norm * 0.3 + volume_norm * 0.3)
        df['fear_greed_index'] = df['fear_greed_index'].fillna(50)  # Neutral default

        # Add numeric sentiment categories (0-4)
        df['market_sentiment'] = pd.cut(
            df['fear_greed_index'],
            bins=[0, 25, 45, 55, 75, 100],
            labels=[0, 1, 2, 3, 4]
        ).astype(float)

        return df

    def _add_market_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market sentiment indicators."""
        # Bull/Bear power
        ema_13 = ta.trend.ema_indicator(df['close'], window=13)
        df['bull_power'] = df['high'] - ema_13
        df['bear_power'] = df['low'] - ema_13

        # Market breadth (simulated)
        advances = (df['close'] > df['close'].shift(1)).astype(int)
        df['advance_decline_ratio'] = advances.rolling(20).mean()

        # Momentum sentiment
        momentum_5 = df['close'] / df['close'].shift(5) - 1
        momentum_20 = df['close'] / df['close'].shift(20) - 1
        df['momentum_sentiment'] = (momentum_5 + momentum_20) / 2

        return df


class FeatureEngineer(LoggerMixin):
    """Main feature engineering class that combines all feature calculators."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature engineer.

        Args:
            config: Feature engineering configuration
        """
        self.config = config or {}
        self.feature_calculators = {
            'technical': TechnicalIndicators(self.config.get('technical', {})),
            'market': MarketFeatures(),
            'sentiment': SentimentFeatures(self.config.get('sentiment', {})),
        }

    @timing
    def create_features(
        self,
        df: pd.DataFrame,
        feature_types: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create all features for the given DataFrame.

        Args:
            df: OHLCV DataFrame
            feature_types: List of feature types to calculate

        Returns:
            DataFrame with all features
        """
        if feature_types is None:
            feature_types = list(self.feature_calculators.keys())

        df_features = df.copy()

        for feature_type in feature_types:
            if feature_type in self.feature_calculators:
                try:
                    calculator = self.feature_calculators[feature_type]
                    df_features = calculator.calculate(df_features)
                    self.logger.debug(f"Added {feature_type} features")
                except Exception as e:
                    self.logger.error(f"Error calculating {feature_type} features: {e}")

        # Drop any infinite or extremely large values
        df_features = df_features.replace([np.inf, -np.inf], np.nan)

        # Remove features with too many NaN values (>50%)
        nan_threshold = len(df_features) * 0.5
        df_features = df_features.dropna(axis=1, thresh=nan_threshold)

        self.logger.info(
            f"Feature engineering completed. "
            f"Original: {df.shape[1]} features, Final: {df_features.shape[1]} features"
        )

        return df_features

    def get_feature_importance(
        self,
        df: pd.DataFrame,
        target_column: str = 'future_return',
        method: str = 'correlation'
    ) -> pd.Series:
        """
        Calculate feature importance.

        Args:
            df: DataFrame with features and target
            target_column: Target column name
            method: Importance calculation method

        Returns:
            Series with feature importance scores
        """
        if target_column not in df.columns:
            self.logger.error(f"Target column '{target_column}' not found")
            return pd.Series()

        numeric_features = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_features if col != target_column]

        if method == 'correlation':
            importance = df[feature_cols].corrwith(df[target_column]).abs()
        elif method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_regression
            importance = mutual_info_regression(
                df[feature_cols].fillna(0),
                df[target_column].fillna(0)
            )
            importance = pd.Series(importance, index=feature_cols)
        else:
            raise ValueError(f"Unknown importance method: {method}")

        importance = importance.sort_values(ascending=False)
        self.logger.info(f"Calculated feature importance using {method}")

        return importance

    def select_features(
        self,
        df: pd.DataFrame,
        target_column: str,
        n_features: int = 50,
        method: str = 'correlation'
    ) -> List[str]:
        """
        Select top features based on importance.

        Args:
            df: DataFrame with features and target
            target_column: Target column name
            n_features: Number of features to select
            method: Feature selection method

        Returns:
            List of selected feature names
        """
        importance = self.get_feature_importance(df, target_column, method)
        selected_features = importance.head(n_features).index.tolist()

        self.logger.info(f"Selected {len(selected_features)} features using {method}")
        return selected_features