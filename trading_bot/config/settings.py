"""
Settings and configuration management for the Bitcoin trading bot.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any

from trading_bot.config.base import YAMLConfig
from trading_bot.utils.logger import get_logger

logger = get_logger(__name__)


class Settings(YAMLConfig):
    """Main settings class for the trading bot."""

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize settings.

        Args:
            config_file: Path to configuration file
        """
        if config_file is None:
            config_file = self._find_config_file()

        super().__init__(config_file)
        self._validate_config()

    def _find_config_file(self) -> str:
        """Find the configuration file."""
        possible_paths = [
            "configs/default.yaml",
            "config.yaml",
            "trading_bot.yaml",
        ]

        for path in possible_paths:
            if Path(path).exists():
                return path

        # Return default path even if it doesn't exist
        return "configs/default.yaml"

    def _validate_config(self) -> None:
        """Validate configuration settings."""
        required_sections = ["data", "features", "models", "trading", "evaluation"]

        for section in required_sections:
            if not self.get(section):
                logger.warning(f"Missing configuration section: {section}")

        # Validate API keys if trading mode is live
        if self.trading_mode == "live":
            required_api_keys = ["binance.api_key", "binance.secret_key"]
            for key in required_api_keys:
                if not self.get(f"api_keys.{key}"):
                    logger.warning(f"Missing API key for live trading: {key}")

    # Data configuration properties
    @property
    def data_sources(self) -> List[str]:
        """Get list of data sources."""
        return self.get("data.sources", ["binance", "coingecko"])

    @property
    def symbols(self) -> List[str]:
        """Get list of trading symbols."""
        return self.get("data.symbols", ["BTC-USD"])

    @property
    def intervals(self) -> List[str]:
        """Get list of time intervals."""
        return self.get("data.intervals", ["1h", "1d"])

    @property
    def lookback_days(self) -> int:
        """Get number of days to look back for data."""
        return self.get("data.lookback_days", 365)

    @property
    def update_frequency(self) -> int:
        """Get data update frequency in seconds."""
        return self.get("data.update_frequency", 3600)

    # Feature configuration properties
    @property
    def technical_indicators(self) -> List[str]:
        """Get list of technical indicators."""
        return self.get("features.technical_indicators", ["sma", "ema", "rsi"])

    @property
    def feature_timeframes(self) -> List[int]:
        """Get feature calculation timeframes."""
        return self.get("features.timeframes", [5, 10, 20, 50])

    # Model configuration properties
    @property
    def algorithms(self) -> List[str]:
        """Get list of ML algorithms."""
        return self.get("models.algorithms", ["random_forest", "xgboost"])

    @property
    def ensemble_method(self) -> str:
        """Get ensemble method."""
        return self.get("models.ensemble.method", "voting")

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        """Get hyperparameters for models."""
        return self.get("models.hyperparameters", {})

    # Trading configuration properties
    @property
    def trading_mode(self) -> str:
        """Get trading mode (paper or live)."""
        return self.get("trading.mode", "paper")

    @property
    def signal_threshold(self) -> float:
        """Get signal threshold for trading decisions."""
        return self.get("trading.strategy.signal_threshold", 0.6)

    @property
    def position_sizing(self) -> str:
        """Get position sizing method."""
        return self.get("trading.strategy.position_sizing", "fixed")

    @property
    def stop_loss(self) -> float:
        """Get stop loss percentage."""
        return self.get("trading.strategy.stop_loss", 0.05)

    @property
    def take_profit(self) -> float:
        """Get take profit percentage."""
        return self.get("trading.strategy.take_profit", 0.10)

    @property
    def max_drawdown(self) -> float:
        """Get maximum drawdown limit."""
        return self.get("trading.risk_management.max_drawdown", 0.20)

    @property
    def position_limit(self) -> float:
        """Get position size limit."""
        return self.get("trading.risk_management.position_limit", 0.30)

    # Database configuration
    @property
    def database_url(self) -> str:
        """Get database URL."""
        return self.get("database.url", "sqlite:///data/trading_bot.db")

    @property
    def redis_url(self) -> str:
        """Get Redis URL."""
        return self.get("database.redis_url", "redis://localhost:6379")

    # API keys
    @property
    def binance_api_key(self) -> Optional[str]:
        """Get Binance API key."""
        return self.get("api_keys.binance.api_key")

    @property
    def binance_secret_key(self) -> Optional[str]:
        """Get Binance secret key."""
        return self.get("api_keys.binance.secret_key")

    @property
    def coingecko_api_key(self) -> Optional[str]:
        """Get CoinGecko API key."""
        return self.get("api_keys.coingecko.api_key")

    # Evaluation configuration
    @property
    def evaluation_metrics(self) -> List[str]:
        """Get evaluation metrics."""
        return self.get("evaluation.metrics", ["accuracy", "sharpe_ratio", "max_drawdown"])

    @property
    def baseline_strategies(self) -> List[str]:
        """Get baseline strategies for comparison."""
        return self.get("evaluation.baselines", ["buy_and_hold", "moving_average"])

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get configuration for specific model.

        Args:
            model_name: Name of the model

        Returns:
            Model configuration dictionary
        """
        return self.get(f"models.hyperparameters.{model_name}", {})