"""
Bitcoin Trading Bot - ML-powered trading system for Bitcoin price prediction and automated trading.
"""

__version__ = "0.1.0"
__author__ = "Trading Bot Team"
__email__ = "team@tradingbot.com"

from trading_bot.utils.logger import get_logger
from trading_bot.config.settings import Settings

# Initialize package-level logger
logger = get_logger(__name__)

__all__ = ["logger", "Settings"]