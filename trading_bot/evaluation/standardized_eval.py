"""
Standardized evaluation framework for model comparison.
Ensures all models are evaluated on the same time period and calculates excess returns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from trading_bot.utils.logger import LoggerMixin
from trading_bot.utils.helpers import save_json
from trading_bot.evaluation.metrics import PerformanceMetrics


class StandardizedEvaluator(LoggerMixin):
    """Evaluate models with standardized dates and calculate excess returns."""

    def __init__(
        self,
        test_start_date: Optional[str] = None,
        test_end_date: Optional[str] = None,
        transaction_cost: float = 0.002
    ):
        """
        Initialize standardized evaluator.

        Args:
            test_start_date: Start date for test period (YYYY-MM-DD)
            test_end_date: End date for test period (YYYY-MM-DD)
            transaction_cost: Transaction cost per trade
        """
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        self.transaction_cost = transaction_cost
        self.results = {}
        self.buy_hold_return = None

    def calculate_buy_hold_return(
        self,
        prices: pd.Series,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> float:
        """
        Calculate buy & hold return for a given period.

        Args:
            prices: Price series (close prices)
            start_date: Start date (optional, uses all data if not specified)
            end_date: End date (optional, uses all data if not specified)

        Returns:
            Buy & hold return
        """
        if start_date:
            prices = prices[prices.index >= start_date]
        if end_date:
            prices = prices[prices.index <= end_date]

        if len(prices) < 2:
            return 0.0

        # Buy at start, hold until end
        buy_price = prices.iloc[0]
        sell_price = prices.iloc[-1]

        # Account for transaction costs (buy + sell)
        buy_hold_return = (sell_price / buy_price - 1) - (2 * self.transaction_cost)

        self.logger.info(
            f"Buy & Hold: {buy_hold_return*100:.2f}% "
            f"({prices.index[0]} → {prices.index[-1]})"
        )

        return buy_hold_return

    def calculate_strategy_return(
        self,
        signals: np.ndarray,
        returns: np.ndarray,
        probabilities: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate strategy returns from trading signals.

        Args:
            signals: Binary signals (1 = buy, 0 = no position)
            returns: Period returns
            probabilities: Prediction probabilities (optional)

        Returns:
            Dictionary with strategy metrics
        """
        # Calculate position changes (trades)
        position_changes = np.diff(signals, prepend=0)
        num_trades = np.sum(np.abs(position_changes))

        # Calculate returns
        strategy_returns = []
        cumulative_return = 0.0

        for i in range(len(signals)):
            if signals[i] == 1:
                # Long position: capture the return minus cost if entering
                if i > 0 and signals[i-1] == 0:
                    # Entering position: pay transaction cost
                    period_return = returns[i] - self.transaction_cost
                elif i < len(signals) - 1 and signals[i+1] == 0:
                    # Exiting position: pay transaction cost
                    period_return = returns[i] - self.transaction_cost
                else:
                    # Holding position: no cost
                    period_return = returns[i]

                cumulative_return += period_return
                strategy_returns.append(period_return)
            else:
                # No position: no return
                strategy_returns.append(0.0)

        strategy_returns = np.array(strategy_returns)

        # Calculate metrics
        total_return = cumulative_return
        avg_return = np.mean(strategy_returns[strategy_returns != 0]) if np.any(strategy_returns != 0) else 0.0

        # Sharpe ratio (assuming daily returns, annualized)
        if np.std(strategy_returns) > 0:
            sharpe = np.sqrt(252) * np.mean(strategy_returns) / np.std(strategy_returns)
        else:
            sharpe = 0.0

        # Win rate
        winning_trades = np.sum(strategy_returns > 0)
        total_trades = np.sum(strategy_returns != 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        return {
            'total_return': total_return,
            'avg_return_per_trade': avg_return,
            'sharpe_ratio': sharpe,
            'num_trades': int(num_trades),
            'win_rate': win_rate,
            'total_trades': int(total_trades)
        }

    def evaluate_model(
        self,
        model_name: str,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        dates: pd.DatetimeIndex,
        probabilities: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single model and calculate excess returns.

        Args:
            model_name: Name of the model
            predictions: Binary predictions (1 = buy, 0 = hold cash)
            actual_returns: Actual returns for the period
            dates: DatetimeIndex for the test period
            probabilities: Prediction probabilities (optional)

        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info(f"Evaluating {model_name}...")

        # Filter to test period if specified
        if self.test_start_date or self.test_end_date:
            mask = np.ones(len(dates), dtype=bool)
            if self.test_start_date:
                mask &= (dates >= self.test_start_date)
            if self.test_end_date:
                mask &= (dates <= self.test_end_date)

            predictions = predictions[mask]
            actual_returns = actual_returns[mask]
            dates = dates[mask]
            if probabilities is not None:
                probabilities = probabilities[mask]

        # Calculate strategy returns
        strategy_metrics = self.calculate_strategy_return(
            predictions,
            actual_returns,
            probabilities
        )

        # Calculate excess return over buy & hold
        if self.buy_hold_return is not None:
            excess_return = strategy_metrics['total_return'] - self.buy_hold_return
            strategy_metrics['excess_return'] = excess_return
            strategy_metrics['excess_return_pct'] = (excess_return / abs(self.buy_hold_return)) * 100 if self.buy_hold_return != 0 else 0

        # Store results
        self.results[model_name] = {
            **strategy_metrics,
            'test_period': f"{dates[0]} to {dates[-1]}",
            'num_days': len(dates)
        }

        self.logger.info(
            f"{model_name}: {strategy_metrics['total_return']*100:.2f}% "
            f"(Excess: {strategy_metrics.get('excess_return', 0)*100:+.2f}%)"
        )

        return self.results[model_name]

    def compare_models(
        self,
        results: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Create comparison DataFrame of model results.

        Args:
            results: Dictionary of model results

        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []

        for model_name, metrics in results.items():
            comparison_data.append({
                'Model': model_name,
                'Total Return (%)': metrics['total_return'] * 100,
                'Excess Return (%)': metrics.get('excess_return', 0) * 100,
                'Sharpe Ratio': metrics['sharpe_ratio'],
                'Win Rate (%)': metrics['win_rate'] * 100,
                'Num Trades': metrics['num_trades'],
                'Avg Return/Trade (%)': metrics['avg_return_per_trade'] * 100
            })

        # Add Buy & Hold as reference
        if self.buy_hold_return is not None:
            comparison_data.append({
                'Model': 'Buy & Hold',
                'Total Return (%)': self.buy_hold_return * 100,
                'Excess Return (%)': 0.0,
                'Sharpe Ratio': 0.0,
                'Win Rate (%)': 100.0,
                'Num Trades': 2,  # Buy + Sell
                'Avg Return/Trade (%)': self.buy_hold_return * 100 / 2
            })

        df = pd.DataFrame(comparison_data)

        # Sort by Total Return descending
        df = df.sort_values('Total Return (%)', ascending=False)

        return df

    def plot_comparison(
        self,
        comparison_df: pd.DataFrame,
        output_path: Path,
        metric: str = 'Total Return (%)'
    ):
        """Plot model comparison."""
        plt.figure(figsize=(12, 6))

        # Sort by metric
        df_sorted = comparison_df.sort_values(metric, ascending=True)

        # Color code: green for positive, red for negative
        colors = ['green' if x > 0 else 'red' for x in df_sorted[metric]]

        plt.barh(range(len(df_sorted)), df_sorted[metric], color=colors, alpha=0.7)
        plt.yticks(range(len(df_sorted)), df_sorted['Model'])
        plt.xlabel(metric)
        plt.title(f'Model Comparison: {metric}')
        plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved comparison plot to {output_path}")

    def plot_excess_returns(
        self,
        comparison_df: pd.DataFrame,
        output_path: Path
    ):
        """Plot excess returns over Buy & Hold."""
        plt.figure(figsize=(12, 6))

        # Filter out Buy & Hold
        df_models = comparison_df[comparison_df['Model'] != 'Buy & Hold'].copy()
        df_sorted = df_models.sort_values('Excess Return (%)', ascending=True)

        colors = ['green' if x > 0 else 'red' for x in df_sorted['Excess Return (%)']]

        plt.barh(range(len(df_sorted)), df_sorted['Excess Return (%)'], color=colors, alpha=0.7)
        plt.yticks(range(len(df_sorted)), df_sorted['Model'])
        plt.xlabel('Excess Return over Buy & Hold (%)')
        plt.title('Model Performance vs Buy & Hold Baseline')
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved excess returns plot to {output_path}")

    def save_results(self, output_dir: Path):
        """Save evaluation results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create comparison DataFrame
        comparison_df = self.compare_models(self.results)

        # Save CSV
        csv_path = output_dir / 'model_comparison.csv'
        comparison_df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved comparison to {csv_path}")

        # Save plots
        self.plot_comparison(
            comparison_df,
            output_dir / 'total_returns.png',
            metric='Total Return (%)'
        )

        if 'Excess Return (%)' in comparison_df.columns:
            self.plot_excess_returns(
                comparison_df,
                output_dir / 'excess_returns.png'
            )

        self.plot_comparison(
            comparison_df,
            output_dir / 'sharpe_ratio.png',
            metric='Sharpe Ratio'
        )

        # Save JSON
        save_json({
            'evaluation_config': {
                'test_start_date': self.test_start_date,
                'test_end_date': self.test_end_date,
                'transaction_cost': self.transaction_cost
            },
            'buy_hold_return': self.buy_hold_return,
            'model_results': self.results,
            'comparison_table': comparison_df.to_dict(orient='records')
        }, output_dir / 'evaluation_results.json')

        self.logger.info(f"Saved all results to {output_dir}")

        # Return comparison table for display
        return comparison_df
