"""
Performance metrics module for Bitcoin trading bot evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from scipy import stats
import warnings

from trading_bot.utils.logger import LoggerMixin


class PerformanceMetrics(LoggerMixin):
    """Calculate various performance metrics for trading strategies and ML models."""

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate for calculations
        """
        self.risk_free_rate = risk_free_rate

    def calculate_all_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        prices: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.

        Args:
            returns: Strategy returns series
            benchmark_returns: Benchmark returns for comparison
            prices: Price series for additional calculations

        Returns:
            Dictionary of performance metrics
        """
        metrics = {}

        # Return-based metrics
        metrics.update(self._calculate_return_metrics(returns))

        # Risk metrics
        metrics.update(self._calculate_risk_metrics(returns))

        # Risk-adjusted metrics
        metrics.update(self._calculate_risk_adjusted_metrics(returns))

        # Drawdown metrics
        if prices is not None:
            metrics.update(self._calculate_drawdown_metrics(prices))

        # Benchmark comparison metrics
        if benchmark_returns is not None:
            metrics.update(self._calculate_benchmark_metrics(returns, benchmark_returns))

        # Trading metrics
        metrics.update(self._calculate_trading_metrics(returns))

        self.logger.info(f"Calculated {len(metrics)} performance metrics")
        return metrics

    def _calculate_return_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate return-based metrics."""
        metrics = {}

        # Basic return statistics
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annualized_return'] = (1 + returns.mean()) ** 252 - 1
        metrics['mean_return'] = returns.mean()
        metrics['median_return'] = returns.median()

        # Cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        metrics['final_value'] = cumulative_returns.iloc[-1]

        # Compounding metrics
        metrics['geometric_mean'] = stats.gmean(1 + returns) - 1
        metrics['arithmetic_mean'] = returns.mean()

        return metrics

    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk-based metrics."""
        metrics = {}

        # Volatility metrics
        metrics['volatility'] = returns.std()
        metrics['annualized_volatility'] = returns.std() * np.sqrt(252)

        # Downside risk metrics
        negative_returns = returns[returns < 0]
        metrics['downside_deviation'] = negative_returns.std() * np.sqrt(252)
        metrics['downside_variance'] = negative_returns.var() * 252

        # Value at Risk (VaR)
        metrics['var_95'] = returns.quantile(0.05)
        metrics['var_99'] = returns.quantile(0.01)

        # Conditional Value at Risk (CVaR)
        metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
        metrics['cvar_99'] = returns[returns <= metrics['var_99']].mean()

        # Skewness and Kurtosis
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()

        # Semi-deviation
        mean_return = returns.mean()
        downside_returns = returns[returns < mean_return] - mean_return
        metrics['semi_deviation'] = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(252)

        return metrics

    def _calculate_risk_adjusted_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics."""
        metrics = {}

        daily_rf_rate = self.risk_free_rate / 252
        excess_returns = returns - daily_rf_rate

        # Sharpe Ratio
        if returns.std() != 0:
            metrics['sharpe_ratio'] = excess_returns.mean() / returns.std() * np.sqrt(252)
        else:
            metrics['sharpe_ratio'] = 0.0

        # Sortino Ratio
        downside_std = returns[returns < daily_rf_rate].std()
        if downside_std != 0:
            metrics['sortino_ratio'] = excess_returns.mean() / downside_std * np.sqrt(252)
        else:
            metrics['sortino_ratio'] = 0.0

        # Calmar Ratio (will be calculated in drawdown metrics if prices available)
        metrics['calmar_ratio'] = np.nan

        # Information Ratio (will be calculated in benchmark metrics if benchmark available)
        metrics['information_ratio'] = np.nan

        # Omega Ratio
        threshold = daily_rf_rate
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns < threshold]

        if losses.sum() != 0:
            metrics['omega_ratio'] = gains.sum() / losses.sum()
        else:
            metrics['omega_ratio'] = np.inf

        return metrics

    def _calculate_drawdown_metrics(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate drawdown-based metrics."""
        metrics = {}

        # Calculate drawdowns
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak

        metrics['max_drawdown'] = drawdown.min()
        metrics['avg_drawdown'] = drawdown[drawdown < 0].mean()

        # Drawdown duration
        is_drawdown = drawdown < 0
        drawdown_periods = []
        start = None

        for i, in_drawdown in enumerate(is_drawdown):
            if in_drawdown and start is None:
                start = i
            elif not in_drawdown and start is not None:
                drawdown_periods.append(i - start)
                start = None

        if start is not None:  # Still in drawdown at the end
            drawdown_periods.append(len(is_drawdown) - start)

        if drawdown_periods:
            metrics['max_drawdown_duration'] = max(drawdown_periods)
            metrics['avg_drawdown_duration'] = np.mean(drawdown_periods)
        else:
            metrics['max_drawdown_duration'] = 0
            metrics['avg_drawdown_duration'] = 0

        # Recovery factor
        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
        if metrics['max_drawdown'] != 0:
            metrics['recovery_factor'] = total_return / abs(metrics['max_drawdown'])
        else:
            metrics['recovery_factor'] = np.inf

        # Calmar Ratio
        annualized_return = (prices.iloc[-1] / prices.iloc[0]) ** (252 / len(prices)) - 1
        if metrics['max_drawdown'] != 0:
            metrics['calmar_ratio'] = annualized_return / abs(metrics['max_drawdown'])
        else:
            metrics['calmar_ratio'] = np.inf

        return metrics

    def _calculate_benchmark_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate metrics relative to benchmark."""
        metrics = {}

        # Align series
        aligned_data = pd.concat([returns, benchmark_returns], axis=1, join='inner')
        strategy_returns = aligned_data.iloc[:, 0]
        bench_returns = aligned_data.iloc[:, 1]

        # Tracking error
        active_returns = strategy_returns - bench_returns
        metrics['tracking_error'] = active_returns.std() * np.sqrt(252)

        # Information ratio
        if metrics['tracking_error'] != 0:
            metrics['information_ratio'] = active_returns.mean() / active_returns.std() * np.sqrt(252)
        else:
            metrics['information_ratio'] = 0.0

        # Beta
        covariance = np.cov(strategy_returns, bench_returns)[0, 1]
        benchmark_variance = bench_returns.var()
        if benchmark_variance != 0:
            metrics['beta'] = covariance / benchmark_variance
        else:
            metrics['beta'] = 0.0

        # Alpha
        daily_rf_rate = self.risk_free_rate / 252
        strategy_excess = strategy_returns.mean() - daily_rf_rate
        benchmark_excess = bench_returns.mean() - daily_rf_rate
        metrics['alpha'] = (strategy_excess - metrics['beta'] * benchmark_excess) * 252

        # Treynor ratio
        if metrics['beta'] != 0:
            metrics['treynor_ratio'] = strategy_excess * 252 / metrics['beta']
        else:
            metrics['treynor_ratio'] = np.inf

        # Up/Down capture ratios
        up_periods = bench_returns > 0
        down_periods = bench_returns < 0

        if up_periods.sum() > 0:
            up_capture = strategy_returns[up_periods].mean() / bench_returns[up_periods].mean()
            metrics['up_capture_ratio'] = up_capture
        else:
            metrics['up_capture_ratio'] = np.nan

        if down_periods.sum() > 0:
            down_capture = strategy_returns[down_periods].mean() / bench_returns[down_periods].mean()
            metrics['down_capture_ratio'] = down_capture
        else:
            metrics['down_capture_ratio'] = np.nan

        return metrics

    def _calculate_trading_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate trading-specific metrics."""
        metrics = {}

        # Win/Loss statistics
        winning_returns = returns[returns > 0]
        losing_returns = returns[returns < 0]
        zero_returns = returns[returns == 0]

        total_trades = len(returns)
        metrics['total_trades'] = total_trades
        metrics['winning_trades'] = len(winning_returns)
        metrics['losing_trades'] = len(losing_returns)
        metrics['zero_trades'] = len(zero_returns)

        # Win rate
        metrics['win_rate'] = len(winning_returns) / total_trades if total_trades > 0 else 0.0
        metrics['loss_rate'] = len(losing_returns) / total_trades if total_trades > 0 else 0.0

        # Average win/loss
        metrics['avg_win'] = winning_returns.mean() if len(winning_returns) > 0 else 0.0
        metrics['avg_loss'] = losing_returns.mean() if len(losing_returns) > 0 else 0.0

        # Profit factor
        gross_profit = winning_returns.sum()
        gross_loss = abs(losing_returns.sum())
        if gross_loss != 0:
            metrics['profit_factor'] = gross_profit / gross_loss
        else:
            metrics['profit_factor'] = np.inf

        # Expectancy
        if metrics['loss_rate'] != 0:
            metrics['expectancy'] = (metrics['win_rate'] * metrics['avg_win']) + \
                                  (metrics['loss_rate'] * metrics['avg_loss'])
        else:
            metrics['expectancy'] = metrics['avg_win']

        # Largest win/loss
        metrics['largest_win'] = winning_returns.max() if len(winning_returns) > 0 else 0.0
        metrics['largest_loss'] = losing_returns.min() if len(losing_returns) > 0 else 0.0

        # Consecutive wins/losses
        metrics.update(self._calculate_consecutive_metrics(returns))

        return metrics

    def _calculate_consecutive_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate consecutive win/loss statistics."""
        metrics = {}

        # Convert to win/loss/neutral signals
        signals = np.where(returns > 0, 1, np.where(returns < 0, -1, 0))

        # Find consecutive sequences
        consecutive_wins = []
        consecutive_losses = []
        current_streak = 0
        current_type = 0

        for signal in signals:
            if signal == current_type and signal != 0:
                current_streak += 1
            else:
                if current_type == 1 and current_streak > 0:
                    consecutive_wins.append(current_streak)
                elif current_type == -1 and current_streak > 0:
                    consecutive_losses.append(current_streak)

                current_streak = 1 if signal != 0 else 0
                current_type = signal

        # Handle final streak
        if current_type == 1 and current_streak > 0:
            consecutive_wins.append(current_streak)
        elif current_type == -1 and current_streak > 0:
            consecutive_losses.append(current_streak)

        # Calculate statistics
        metrics['max_consecutive_wins'] = max(consecutive_wins) if consecutive_wins else 0
        metrics['max_consecutive_losses'] = max(consecutive_losses) if consecutive_losses else 0
        metrics['avg_consecutive_wins'] = np.mean(consecutive_wins) if consecutive_wins else 0
        metrics['avg_consecutive_losses'] = np.mean(consecutive_losses) if consecutive_losses else 0

        return metrics

    def calculate_ml_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        model_type: str = 'regression'
    ) -> Dict[str, float]:
        """
        Calculate machine learning specific metrics.

        Args:
            y_true: True values
            y_pred: Predicted values
            y_proba: Prediction probabilities (for classification)
            model_type: Type of model ('regression' or 'classification')

        Returns:
            Dictionary of ML metrics
        """
        metrics = {}

        if model_type == 'regression':
            metrics.update(self._calculate_regression_metrics(y_true, y_pred))
        else:
            metrics.update(self._calculate_classification_metrics(y_true, y_pred, y_proba))

        return metrics

    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        metrics = {}

        # Basic metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2_score'] = r2_score(y_true, y_pred)

        # Additional metrics
        metrics['mean_error'] = np.mean(y_pred - y_true)
        metrics['std_error'] = np.std(y_pred - y_true)

        # Mean Absolute Percentage Error
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics['mape'] = mape if np.isfinite(mape) else np.nan

        # Direction accuracy (for financial predictions)
        if len(y_true) > 1:
            true_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            metrics['direction_accuracy'] = np.mean(true_direction == pred_direction)

        return metrics

    def _calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate classification metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, log_loss, confusion_matrix
        )

        metrics = {}

        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)

        # Handle multi-class case
        average_method = 'weighted' if len(np.unique(y_true)) > 2 else 'binary'

        metrics['precision'] = precision_score(y_true, y_pred, average=average_method, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average_method, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average=average_method, zero_division=0)

        # Confusion matrix statistics
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        # AUC and log loss (if probabilities available)
        if y_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
                else:
                    # Multi-class classification
                    metrics['auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')

                metrics['log_loss'] = log_loss(y_true, y_proba)
            except ValueError as e:
                self.logger.warning(f"Could not calculate AUC/log_loss: {e}")

        return metrics

    def create_performance_summary(self, metrics: Dict[str, float]) -> str:
        """
        Create a formatted performance summary.

        Args:
            metrics: Dictionary of calculated metrics

        Returns:
            Formatted summary string
        """
        summary = "\n" + "="*50 + "\n"
        summary += "PERFORMANCE SUMMARY\n"
        summary += "="*50 + "\n\n"

        # Return metrics
        if 'total_return' in metrics:
            summary += f"Total Return: {metrics['total_return']:.2%}\n"
        if 'annualized_return' in metrics:
            summary += f"Annualized Return: {metrics['annualized_return']:.2%}\n"

        # Risk metrics
        if 'annualized_volatility' in metrics:
            summary += f"Annualized Volatility: {metrics['annualized_volatility']:.2%}\n"
        if 'max_drawdown' in metrics:
            summary += f"Maximum Drawdown: {metrics['max_drawdown']:.2%}\n"

        # Risk-adjusted metrics
        if 'sharpe_ratio' in metrics:
            summary += f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}\n"
        if 'sortino_ratio' in metrics:
            summary += f"Sortino Ratio: {metrics['sortino_ratio']:.3f}\n"
        if 'calmar_ratio' in metrics and not np.isnan(metrics['calmar_ratio']):
            summary += f"Calmar Ratio: {metrics['calmar_ratio']:.3f}\n"

        # Trading metrics
        if 'win_rate' in metrics:
            summary += f"\nWin Rate: {metrics['win_rate']:.2%}\n"
        if 'profit_factor' in metrics and not np.isinf(metrics['profit_factor']):
            summary += f"Profit Factor: {metrics['profit_factor']:.3f}\n"
        if 'total_trades' in metrics:
            summary += f"Total Trades: {metrics['total_trades']:.0f}\n"

        summary += "\n" + "="*50 + "\n"

        return summary