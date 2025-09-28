"""
Signal generation module for Bitcoin trading bot.
Converts ML model predictions into trading signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from abc import ABC, abstractmethod
from enum import Enum

from trading_bot.utils.logger import LoggerMixin
from trading_bot.utils.decorators import timing


class SignalType(Enum):
    """Types of trading signals."""
    BUY = 1
    SELL = -1
    HOLD = 0


class SignalStrength(Enum):
    """Signal strength levels."""
    WEAK = 1
    MEDIUM = 2
    STRONG = 3


class TradingSignal:
    """Represents a trading signal."""

    def __init__(
        self,
        timestamp: pd.Timestamp,
        signal_type: SignalType,
        strength: SignalStrength,
        confidence: float,
        price: float,
        metadata: Optional[Dict] = None
    ):
        """
        Initialize trading signal.

        Args:
            timestamp: Signal timestamp
            signal_type: Type of signal (BUY/SELL/HOLD)
            strength: Signal strength
            confidence: Confidence level (0-1)
            price: Current price when signal generated
            metadata: Additional signal metadata
        """
        self.timestamp = timestamp
        self.signal_type = signal_type
        self.strength = strength
        self.confidence = confidence
        self.price = price
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return (f"TradingSignal({self.signal_type.name}, "
                f"strength={self.strength.name}, "
                f"confidence={self.confidence:.3f}, "
                f"price={self.price:.2f})")


class BaseSignalGenerator(ABC, LoggerMixin):
    """Abstract base class for signal generators."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize signal generator.

        Args:
            config: Configuration parameters
        """
        self.config = config or {}

    @abstractmethod
    def generate_signals(
        self,
        predictions: Union[np.ndarray, pd.Series],
        data: pd.DataFrame,
        **kwargs
    ) -> pd.Series:
        """
        Generate trading signals from predictions.

        Args:
            predictions: Model predictions
            data: Market data
            **kwargs: Additional parameters

        Returns:
            Series of trading signals
        """
        pass


class ThresholdSignalGenerator(BaseSignalGenerator):
    """Generate signals based on prediction thresholds."""

    def __init__(
        self,
        buy_threshold: float = 0.6,
        sell_threshold: float = -0.6,
        confidence_threshold: float = 0.5,
        config: Optional[Dict] = None
    ):
        """
        Initialize threshold-based signal generator.

        Args:
            buy_threshold: Threshold for buy signals
            sell_threshold: Threshold for sell signals
            confidence_threshold: Minimum confidence required
            config: Additional configuration
        """
        super().__init__(config)
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.confidence_threshold = confidence_threshold

    def generate_signals(
        self,
        predictions: Union[np.ndarray, pd.Series],
        data: pd.DataFrame,
        probabilities: Optional[np.ndarray] = None
    ) -> pd.Series:
        """
        Generate signals based on prediction thresholds.

        Args:
            predictions: Model predictions
            data: Market data
            probabilities: Prediction probabilities (optional)

        Returns:
            Series of trading signals
        """
        if isinstance(predictions, np.ndarray):
            predictions = pd.Series(predictions, index=data.index[-len(predictions):])

        signals = pd.Series(SignalType.HOLD.value, index=predictions.index)

        # Generate buy signals
        buy_mask = predictions >= self.buy_threshold
        signals[buy_mask] = SignalType.BUY.value

        # Generate sell signals
        sell_mask = predictions <= self.sell_threshold
        signals[sell_mask] = SignalType.SELL.value

        # Apply confidence filter if probabilities provided
        if probabilities is not None:
            if len(probabilities.shape) > 1:
                # For classification, use max probability
                confidence = np.max(probabilities, axis=1)
            else:
                # For regression, use absolute value as confidence
                confidence = np.abs(predictions)

            low_confidence = confidence < self.confidence_threshold
            signals[low_confidence] = SignalType.HOLD.value

        self.logger.info(
            f"Generated {(signals == SignalType.BUY.value).sum()} buy signals, "
            f"{(signals == SignalType.SELL.value).sum()} sell signals, "
            f"{(signals == SignalType.HOLD.value).sum()} hold signals"
        )

        return signals


class VolatilityAdjustedSignalGenerator(BaseSignalGenerator):
    """Generate signals adjusted for market volatility."""

    def __init__(
        self,
        base_threshold: float = 0.5,
        volatility_window: int = 20,
        volatility_factor: float = 2.0,
        config: Optional[Dict] = None
    ):
        """
        Initialize volatility-adjusted signal generator.

        Args:
            base_threshold: Base threshold for signals
            volatility_window: Window for volatility calculation
            volatility_factor: Factor to adjust thresholds based on volatility
            config: Additional configuration
        """
        super().__init__(config)
        self.base_threshold = base_threshold
        self.volatility_window = volatility_window
        self.volatility_factor = volatility_factor

    def generate_signals(
        self,
        predictions: Union[np.ndarray, pd.Series],
        data: pd.DataFrame,
        **kwargs
    ) -> pd.Series:
        """
        Generate volatility-adjusted signals.

        Args:
            predictions: Model predictions
            data: Market data

        Returns:
            Series of trading signals
        """
        if isinstance(predictions, np.ndarray):
            predictions = pd.Series(predictions, index=data.index[-len(predictions):])

        # Calculate rolling volatility
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=self.volatility_window).std()

        # Adjust thresholds based on volatility
        mean_vol = volatility.mean()
        vol_adjustment = volatility / mean_vol

        # Dynamic thresholds
        buy_thresholds = self.base_threshold * (1 + (vol_adjustment - 1) * self.volatility_factor)
        sell_thresholds = -self.base_threshold * (1 + (vol_adjustment - 1) * self.volatility_factor)

        # Align with predictions
        aligned_data = pd.concat([predictions, buy_thresholds, sell_thresholds], axis=1, join='inner')
        aligned_data.columns = ['predictions', 'buy_threshold', 'sell_threshold']

        signals = pd.Series(SignalType.HOLD.value, index=aligned_data.index)

        # Generate signals
        buy_mask = aligned_data['predictions'] >= aligned_data['buy_threshold']
        sell_mask = aligned_data['predictions'] <= aligned_data['sell_threshold']

        signals[buy_mask] = SignalType.BUY.value
        signals[sell_mask] = SignalType.SELL.value

        self.logger.info(f"Generated volatility-adjusted signals with dynamic thresholds")
        return signals


class TrendFollowingSignalGenerator(BaseSignalGenerator):
    """Generate signals that follow market trends."""

    def __init__(
        self,
        trend_window: int = 50,
        signal_threshold: float = 0.6,
        trend_strength_threshold: float = 0.02,
        config: Optional[Dict] = None
    ):
        """
        Initialize trend-following signal generator.

        Args:
            trend_window: Window for trend calculation
            signal_threshold: Threshold for base signals
            trend_strength_threshold: Minimum trend strength required
            config: Additional configuration
        """
        super().__init__(config)
        self.trend_window = trend_window
        self.signal_threshold = signal_threshold
        self.trend_strength_threshold = trend_strength_threshold

    def generate_signals(
        self,
        predictions: Union[np.ndarray, pd.Series],
        data: pd.DataFrame,
        **kwargs
    ) -> pd.Series:
        """
        Generate trend-following signals.

        Args:
            predictions: Model predictions
            data: Market data

        Returns:
            Series of trading signals
        """
        if isinstance(predictions, np.ndarray):
            predictions = pd.Series(predictions, index=data.index[-len(predictions):])

        # Calculate trend
        sma = data['close'].rolling(window=self.trend_window).mean()
        trend = (data['close'] - sma) / sma

        # Align data
        aligned_data = pd.concat([predictions, trend], axis=1, join='inner')
        aligned_data.columns = ['predictions', 'trend']

        signals = pd.Series(SignalType.HOLD.value, index=aligned_data.index)

        # Generate signals only when trend is strong enough
        strong_uptrend = aligned_data['trend'] >= self.trend_strength_threshold
        strong_downtrend = aligned_data['trend'] <= -self.trend_strength_threshold

        # Buy signals: positive prediction + uptrend
        buy_mask = (
            (aligned_data['predictions'] >= self.signal_threshold) &
            strong_uptrend
        )

        # Sell signals: negative prediction + downtrend
        sell_mask = (
            (aligned_data['predictions'] <= -self.signal_threshold) &
            strong_downtrend
        )

        signals[buy_mask] = SignalType.BUY.value
        signals[sell_mask] = SignalType.SELL.value

        self.logger.info(f"Generated trend-following signals")
        return signals


class MeanReversionSignalGenerator(BaseSignalGenerator):
    """Generate signals for mean reversion strategies."""

    def __init__(
        self,
        deviation_threshold: float = 2.0,
        lookback_window: int = 20,
        signal_threshold: float = 0.6,
        config: Optional[Dict] = None
    ):
        """
        Initialize mean reversion signal generator.

        Args:
            deviation_threshold: Z-score threshold for mean reversion
            lookback_window: Window for mean/std calculation
            signal_threshold: Threshold for base signals
            config: Additional configuration
        """
        super().__init__(config)
        self.deviation_threshold = deviation_threshold
        self.lookback_window = lookback_window
        self.signal_threshold = signal_threshold

    def generate_signals(
        self,
        predictions: Union[np.ndarray, pd.Series],
        data: pd.DataFrame,
        **kwargs
    ) -> pd.Series:
        """
        Generate mean reversion signals.

        Args:
            predictions: Model predictions
            data: Market data

        Returns:
            Series of trading signals
        """
        if isinstance(predictions, np.ndarray):
            predictions = pd.Series(predictions, index=data.index[-len(predictions):])

        # Calculate price deviation from mean
        rolling_mean = data['close'].rolling(window=self.lookback_window).mean()
        rolling_std = data['close'].rolling(window=self.lookback_window).std()
        z_score = (data['close'] - rolling_mean) / rolling_std

        # Align data
        aligned_data = pd.concat([predictions, z_score], axis=1, join='inner')
        aligned_data.columns = ['predictions', 'z_score']

        signals = pd.Series(SignalType.HOLD.value, index=aligned_data.index)

        # Mean reversion logic: buy when oversold, sell when overbought
        oversold = aligned_data['z_score'] <= -self.deviation_threshold
        overbought = aligned_data['z_score'] >= self.deviation_threshold

        # Buy signals: positive prediction + oversold
        buy_mask = (
            (aligned_data['predictions'] >= self.signal_threshold) &
            oversold
        )

        # Sell signals: negative prediction + overbought
        sell_mask = (
            (aligned_data['predictions'] <= -self.signal_threshold) &
            overbought
        )

        signals[buy_mask] = SignalType.BUY.value
        signals[sell_mask] = SignalType.SELL.value

        self.logger.info(f"Generated mean reversion signals")
        return signals


class EnsembleSignalGenerator(BaseSignalGenerator):
    """Combine signals from multiple generators."""

    def __init__(
        self,
        generators: List[BaseSignalGenerator],
        weights: Optional[List[float]] = None,
        consensus_threshold: float = 0.5,
        config: Optional[Dict] = None
    ):
        """
        Initialize ensemble signal generator.

        Args:
            generators: List of signal generators
            weights: Weights for each generator
            consensus_threshold: Threshold for consensus signals
            config: Additional configuration
        """
        super().__init__(config)
        self.generators = generators
        self.weights = weights or [1.0] * len(generators)
        self.consensus_threshold = consensus_threshold

        if len(self.weights) != len(self.generators):
            raise ValueError("Number of weights must match number of generators")

    def generate_signals(
        self,
        predictions: Union[np.ndarray, pd.Series],
        data: pd.DataFrame,
        **kwargs
    ) -> pd.Series:
        """
        Generate ensemble signals.

        Args:
            predictions: Model predictions
            data: Market data

        Returns:
            Series of trading signals
        """
        if isinstance(predictions, np.ndarray):
            predictions = pd.Series(predictions, index=data.index[-len(predictions):])

        # Generate signals from each generator
        individual_signals = []
        for generator in self.generators:
            signals = generator.generate_signals(predictions, data, **kwargs)
            individual_signals.append(signals)

        # Combine signals with weights
        combined_signals = pd.DataFrame(individual_signals).T
        weighted_signals = (combined_signals * self.weights).sum(axis=1) / sum(self.weights)

        # Convert to discrete signals based on consensus
        final_signals = pd.Series(SignalType.HOLD.value, index=weighted_signals.index)

        buy_mask = weighted_signals >= self.consensus_threshold
        sell_mask = weighted_signals <= -self.consensus_threshold

        final_signals[buy_mask] = SignalType.BUY.value
        final_signals[sell_mask] = SignalType.SELL.value

        self.logger.info(
            f"Generated ensemble signals from {len(self.generators)} generators "
            f"with consensus threshold {self.consensus_threshold}"
        )

        return final_signals


class SignalGenerator(LoggerMixin):
    """Main signal generator that orchestrates different signal generation strategies."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize signal generator.

        Args:
            config: Configuration for signal generation
        """
        self.config = config or {}
        self.generators = self._initialize_generators()

    def _initialize_generators(self) -> Dict[str, BaseSignalGenerator]:
        """Initialize available signal generators."""
        generators = {}

        # Threshold-based generator
        threshold_config = self.config.get('threshold', {})
        generators['threshold'] = ThresholdSignalGenerator(**threshold_config)

        # Volatility-adjusted generator
        volatility_config = self.config.get('volatility', {})
        generators['volatility'] = VolatilityAdjustedSignalGenerator(**volatility_config)

        # Trend-following generator
        trend_config = self.config.get('trend', {})
        generators['trend'] = TrendFollowingSignalGenerator(**trend_config)

        # Mean reversion generator
        mean_reversion_config = self.config.get('mean_reversion', {})
        generators['mean_reversion'] = MeanReversionSignalGenerator(**mean_reversion_config)

        return generators

    @timing
    def generate_signals(
        self,
        predictions: Union[np.ndarray, pd.Series],
        data: pd.DataFrame,
        strategy: str = 'threshold',
        **kwargs
    ) -> pd.Series:
        """
        Generate trading signals using specified strategy.

        Args:
            predictions: Model predictions
            data: Market data
            strategy: Signal generation strategy
            **kwargs: Additional parameters

        Returns:
            Series of trading signals
        """
        if strategy not in self.generators:
            raise ValueError(f"Unknown signal strategy: {strategy}")

        generator = self.generators[strategy]
        signals = generator.generate_signals(predictions, data, **kwargs)

        self.logger.info(f"Generated signals using {strategy} strategy")
        return signals

    def generate_ensemble_signals(
        self,
        predictions: Union[np.ndarray, pd.Series],
        data: pd.DataFrame,
        strategies: List[str],
        weights: Optional[List[float]] = None,
        **kwargs
    ) -> pd.Series:
        """
        Generate ensemble signals from multiple strategies.

        Args:
            predictions: Model predictions
            data: Market data
            strategies: List of strategies to combine
            weights: Weights for each strategy
            **kwargs: Additional parameters

        Returns:
            Series of ensemble trading signals
        """
        # Get generators for specified strategies
        selected_generators = []
        for strategy in strategies:
            if strategy in self.generators:
                selected_generators.append(self.generators[strategy])
            else:
                self.logger.warning(f"Unknown strategy {strategy}, skipping")

        if not selected_generators:
            raise ValueError("No valid strategies provided")

        # Create ensemble generator
        ensemble_config = self.config.get('ensemble', {})
        ensemble_generator = EnsembleSignalGenerator(
            generators=selected_generators,
            weights=weights,
            **ensemble_config
        )

        # Generate ensemble signals
        signals = ensemble_generator.generate_signals(predictions, data, **kwargs)

        self.logger.info(f"Generated ensemble signals from {len(strategies)} strategies")
        return signals

    def analyze_signal_quality(
        self,
        signals: pd.Series,
        data: pd.DataFrame,
        forward_returns: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Analyze the quality of generated signals.

        Args:
            signals: Generated signals
            data: Market data
            forward_returns: Forward returns for analysis

        Returns:
            Dictionary of signal quality metrics
        """
        analysis = {}

        # Basic signal statistics
        total_signals = len(signals[signals != SignalType.HOLD.value])
        buy_signals = (signals == SignalType.BUY.value).sum()
        sell_signals = (signals == SignalType.SELL.value).sum()

        analysis['total_signals'] = total_signals
        analysis['buy_signals'] = buy_signals
        analysis['sell_signals'] = sell_signals
        analysis['signal_frequency'] = total_signals / len(signals)

        # Signal distribution
        analysis['buy_ratio'] = buy_signals / total_signals if total_signals > 0 else 0
        analysis['sell_ratio'] = sell_signals / total_signals if total_signals > 0 else 0

        # Signal clustering (consecutive signals)
        signal_changes = signals.diff() != 0
        signal_periods = signal_changes.cumsum()
        analysis['avg_signal_duration'] = signals.groupby(signal_periods).size().mean()

        # Forward-looking analysis if returns provided
        if forward_returns is not None:
            aligned_data = pd.concat([signals, forward_returns], axis=1, join='inner')
            aligned_data.columns = ['signals', 'forward_returns']

            # Signal accuracy
            buy_returns = aligned_data[aligned_data['signals'] == SignalType.BUY.value]['forward_returns']
            sell_returns = aligned_data[aligned_data['signals'] == SignalType.SELL.value]['forward_returns']

            if len(buy_returns) > 0:
                analysis['buy_signal_accuracy'] = (buy_returns > 0).mean()
                analysis['avg_buy_return'] = buy_returns.mean()

            if len(sell_returns) > 0:
                analysis['sell_signal_accuracy'] = (sell_returns < 0).mean()
                analysis['avg_sell_return'] = sell_returns.mean()

        return analysis