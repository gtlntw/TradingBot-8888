"""
Base configuration classes for the Bitcoin trading bot.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv


class BaseConfig(ABC):
    """Base configuration class."""

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self._config_data: Dict[str, Any] = {}
        self._load_config()

    @abstractmethod
    def _load_config(self) -> None:
        """Load configuration from file and environment."""
        pass

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config_data

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split(".")
        config = self._config_data

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config_data.copy()

    def update(self, config: Dict[str, Any]) -> None:
        """
        Update configuration with new values.

        Args:
            config: Dictionary of configuration values
        """
        self._deep_update(self._config_data, config)

    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """
        Recursively update nested dictionaries.

        Args:
            base_dict: Base dictionary to update
            update_dict: Dictionary with updates
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value


class YAMLConfig(BaseConfig):
    """Configuration loaded from YAML file."""

    def _load_config(self) -> None:
        """Load configuration from YAML file and environment variables."""
        # Load environment variables from .env file
        load_dotenv()

        # Load YAML configuration
        if self.config_file and Path(self.config_file).exists():
            with open(self.config_file, "r") as f:
                self._config_data = yaml.safe_load(f) or {}
        else:
            self._config_data = {}

        # Override with environment variables
        self._load_env_variables()

    def _load_env_variables(self) -> None:
        """Load configuration from environment variables."""
        # API keys
        env_mappings = {
            "BINANCE_API_KEY": "api_keys.binance.api_key",
            "BINANCE_SECRET_KEY": "api_keys.binance.secret_key",
            "COINGECKO_API_KEY": "api_keys.coingecko.api_key",
            "ALPHA_VANTAGE_API_KEY": "api_keys.alpha_vantage.api_key",

            # Database
            "DATABASE_URL": "database.url",
            "REDIS_URL": "database.redis_url",

            # Logging
            "LOG_LEVEL": "logging.level",
            "LOG_FILE": "logging.file_config.filename",

            # Trading
            "TRADING_MODE": "trading.mode",
            "INITIAL_CAPITAL": "trading.initial_capital",
            "MAX_POSITION_SIZE": "trading.risk_management.position_limit",
            "RISK_FREE_RATE": "trading.risk_free_rate",

            # Model
            "MODEL_RETRAIN_INTERVAL": "models.retrain_interval",
            "PREDICTION_HORIZON": "models.prediction_horizon",
            "FEATURE_WINDOW": "features.window_size",
        }

        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                if env_value.lower() in ("true", "false"):
                    env_value = env_value.lower() == "true"
                elif env_value.isdigit():
                    env_value = int(env_value)
                elif self._is_float(env_value):
                    env_value = float(env_value)

                self.set(config_key, env_value)

    @staticmethod
    def _is_float(value: str) -> bool:
        """Check if string represents a float."""
        try:
            float(value)
            return True
        except ValueError:
            return False