"""
Configuration module for Bitcoin trading bot.
Handles settings, configuration loading, and environment management.
"""

from trading_bot.config.settings import Settings
from trading_bot.config.base import BaseConfig

__all__ = ["Settings", "BaseConfig"]