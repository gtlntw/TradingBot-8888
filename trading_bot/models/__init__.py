"""
Models module for Bitcoin trading bot.
Handles ML model training, evaluation, and ensemble methods.
"""

from trading_bot.models.trainer import ModelTrainer
from trading_bot.models.ensemble import EnsembleModel

__all__ = ["ModelTrainer", "EnsembleModel"]