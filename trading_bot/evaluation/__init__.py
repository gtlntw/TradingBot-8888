"""
Evaluation module for Bitcoin trading bot.
Handles performance metrics, backtesting, and reporting.
"""

from trading_bot.evaluation.metrics import PerformanceMetrics
from trading_bot.evaluation.backtester import Backtester
from trading_bot.evaluation.reporter import ReportGenerator

__all__ = ["PerformanceMetrics", "Backtester", "ReportGenerator"]