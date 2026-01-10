"""
Transaction cost sensitivity analysis.
Tests model performance across different cost scenarios.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from trading_bot.utils.logger import LoggerMixin
from trading_bot.utils.helpers import save_json


class CostSensitivityAnalyzer(LoggerMixin):
    """Analyze model sensitivity to transaction costs."""

    def __init__(self, cost_levels: List[float] = None):
        """
        Initialize cost sensitivity analyzer.

        Args:
            cost_levels: List of cost levels to test (e.g., [0.0005, 0.001, 0.002, 0.005])
        """
        if cost_levels is None:
            # Default: 0.05%, 0.1%, 0.2%, 0.5%, 1.0%
            cost_levels = [0.0005, 0.001, 0.002, 0.005, 0.01]

        self.cost_levels = sorted(cost_levels)
        self.results = {}

    def calculate_returns_with_cost(
        self,
        signals: np.ndarray,
        returns: np.ndarray,
        transaction_cost: float
    ) -> Dict[str, float]:
        """
        Calculate strategy returns with specific transaction cost.

        Args:
            signals: Binary trading signals (1 = buy, 0 = no position)
            returns: Period returns
            transaction_cost: Transaction cost per trade

        Returns:
            Dictionary with performance metrics
        """
        # Calculate position changes (trades)
        position_changes = np.diff(signals, prepend=0)
        trades = np.sum(np.abs(position_changes))

        # Calculate strategy returns
        strategy_returns = []
        cumulative_return = 0.0
        total_costs = 0.0

        for i in range(len(signals)):
            if signals[i] == 1:
                # Long position
                if i > 0 and signals[i-1] == 0:
                    # Entering: pay cost
                    period_return = returns[i] - transaction_cost
                    total_costs += transaction_cost
                elif i < len(signals) - 1 and signals[i+1] == 0:
                    # Exiting: pay cost
                    period_return = returns[i] - transaction_cost
                    total_costs += transaction_cost
                else:
                    # Holding: no cost
                    period_return = returns[i]

                cumulative_return += period_return
                strategy_returns.append(period_return)
            else:
                strategy_returns.append(0.0)

        strategy_returns = np.array(strategy_returns)

        # Calculate metrics
        total_return = cumulative_return
        num_trades = int(trades)

        # Sharpe ratio
        if np.std(strategy_returns) > 0:
            sharpe = np.sqrt(252) * np.mean(strategy_returns) / np.std(strategy_returns)
        else:
            sharpe = 0.0

        # Win rate
        winning_trades = np.sum(strategy_returns > 0)
        total_trades_count = np.sum(strategy_returns != 0)
        win_rate = winning_trades / total_trades_count if total_trades_count > 0 else 0.0

        # Profitability threshold
        profitable = total_return > 0

        return {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'transaction_costs': total_costs,
            'cost_impact': total_costs * 100,
            'num_trades': num_trades,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'profitable': profitable,
            'return_per_trade': total_return / num_trades if num_trades > 0 else 0.0
        }

    def analyze_model(
        self,
        model_name: str,
        signals: np.ndarray,
        returns: np.ndarray
    ) -> pd.DataFrame:
        """
        Analyze model across all cost levels.

        Args:
            model_name: Name of the model
            signals: Trading signals
            returns: Actual returns

        Returns:
            DataFrame with results for each cost level
        """
        self.logger.info(f"Analyzing cost sensitivity for {model_name}...")

        results = []

        for cost in self.cost_levels:
            metrics = self.calculate_returns_with_cost(signals, returns, cost)

            results.append({
                'Model': model_name,
                'Transaction Cost (%)': cost * 100,
                'Total Return (%)': metrics['total_return_pct'],
                'Sharpe Ratio': metrics['sharpe_ratio'],
                'Win Rate (%)': metrics['win_rate'] * 100,
                'Num Trades': metrics['num_trades'],
                'Cost Impact (%)': metrics['cost_impact'],
                'Profitable': metrics['profitable'],
                'Return/Trade (%)': metrics['return_per_trade'] * 100
            })

        df = pd.DataFrame(results)
        self.results[model_name] = df

        # Log summary
        baseline_return = df.iloc[0]['Total Return (%)']
        highest_cost_return = df.iloc[-1]['Total Return (%)']
        degradation = baseline_return - highest_cost_return

        self.logger.info(
            f"{model_name}: {baseline_return:.2f}% @ {self.cost_levels[0]*100:.2f}% cost → "
            f"{highest_cost_return:.2f}% @ {self.cost_levels[-1]*100:.2f}% cost "
            f"(degradation: {degradation:.2f}%)"
        )

        return df

    def analyze_multiple_models(
        self,
        model_signals: Dict[str, np.ndarray],
        returns: np.ndarray
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyze multiple models.

        Args:
            model_signals: Dictionary of {model_name: signals}
            returns: Actual returns

        Returns:
            Dictionary of results DataFrames
        """
        self.logger.info(f"Analyzing {len(model_signals)} models across {len(self.cost_levels)} cost levels...")

        for model_name, signals in model_signals.items():
            self.analyze_model(model_name, signals, returns)

        return self.results

    def find_breakeven_cost(
        self,
        model_name: str,
        signals: np.ndarray,
        returns: np.ndarray,
        max_cost: float = 0.02  # 2%
    ) -> float:
        """
        Find the transaction cost level where strategy breaks even.

        Args:
            model_name: Model name
            signals: Trading signals
            returns: Actual returns
            max_cost: Maximum cost to search

        Returns:
            Breakeven cost level
        """
        self.logger.info(f"Finding breakeven cost for {model_name}...")

        # Binary search for breakeven
        low, high = 0.0, max_cost
        tolerance = 0.0001  # 0.01%

        while high - low > tolerance:
            mid = (low + high) / 2
            metrics = self.calculate_returns_with_cost(signals, returns, mid)

            if metrics['total_return'] > 0:
                low = mid  # Can afford higher cost
            else:
                high = mid  # Need lower cost

        breakeven = (low + high) / 2

        self.logger.info(f"{model_name} breakeven cost: {breakeven*100:.4f}%")

        return breakeven

    def plot_cost_sensitivity(
        self,
        output_path: Path,
        models: List[str] = None
    ):
        """Plot cost sensitivity curves."""
        if models is None:
            models = list(self.results.keys())

        plt.figure(figsize=(12, 7))

        for model_name in models:
            if model_name not in self.results:
                continue

            df = self.results[model_name]
            plt.plot(
                df['Transaction Cost (%)'],
                df['Total Return (%)'],
                marker='o',
                label=model_name,
                linewidth=2
            )

        plt.xlabel('Transaction Cost (%)')
        plt.ylabel('Total Return (%)')
        plt.title('Model Performance vs Transaction Costs')
        plt.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Breakeven')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved cost sensitivity plot to {output_path}")

    def plot_cost_impact(
        self,
        output_path: Path,
        models: List[str] = None
    ):
        """Plot cost impact analysis."""
        if models is None:
            models = list(self.results.keys())

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        for model_name in models:
            if model_name not in self.results:
                continue

            df = self.results[model_name]

            # Plot 1: Return vs Cost
            axes[0, 0].plot(df['Transaction Cost (%)'], df['Total Return (%)'],
                           marker='o', label=model_name)

            # Plot 2: Sharpe vs Cost
            axes[0, 1].plot(df['Transaction Cost (%)'], df['Sharpe Ratio'],
                           marker='o', label=model_name)

            # Plot 3: Win Rate vs Cost
            axes[1, 0].plot(df['Transaction Cost (%)'], df['Win Rate (%)'],
                           marker='o', label=model_name)

            # Plot 4: Return per Trade vs Cost
            axes[1, 1].plot(df['Transaction Cost (%)'], df['Return/Trade (%)'],
                           marker='o', label=model_name)

        axes[0, 0].set_xlabel('Transaction Cost (%)')
        axes[0, 0].set_ylabel('Total Return (%)')
        axes[0, 0].set_title('Total Return vs Cost')
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].set_xlabel('Transaction Cost (%)')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].set_title('Sharpe Ratio vs Cost')
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        axes[1, 0].set_xlabel('Transaction Cost (%)')
        axes[1, 0].set_ylabel('Win Rate (%)')
        axes[1, 0].set_title('Win Rate vs Cost')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        axes[1, 1].set_xlabel('Transaction Cost (%)')
        axes[1, 1].set_ylabel('Return per Trade (%)')
        axes[1, 1].set_title('Return per Trade vs Cost')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved cost impact analysis to {output_path}")

    def create_summary_table(self) -> pd.DataFrame:
        """Create summary table comparing all models."""
        summary_data = []

        for model_name, df in self.results.items():
            # Get metrics at lowest and highest cost
            lowest_cost = df.iloc[0]
            highest_cost = df.iloc[-1]

            degradation = lowest_cost['Total Return (%)'] - highest_cost['Total Return (%)']
            degradation_pct = (degradation / abs(lowest_cost['Total Return (%)']) * 100
                              if lowest_cost['Total Return (%)'] != 0 else 0)

            # Check if profitable at different cost levels
            profitable_at_low = lowest_cost['Profitable']
            profitable_at_high = highest_cost['Profitable']

            summary_data.append({
                'Model': model_name,
                f'Return @ {self.cost_levels[0]*100:.2f}% (%)': lowest_cost['Total Return (%)'],
                f'Return @ {self.cost_levels[-1]*100:.2f}% (%)': highest_cost['Total Return (%)'],
                'Degradation (%)': degradation,
                'Degradation (% of return)': degradation_pct,
                'Profitable at Low Cost': profitable_at_low,
                'Profitable at High Cost': profitable_at_high,
                'Num Trades': int(lowest_cost['Num Trades'])
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values(f'Return @ {self.cost_levels[0]*100:.2f}% (%)', ascending=False)

        return summary_df

    def save_results(self, output_dir: Path):
        """Save all results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save individual model results
        for model_name, df in self.results.items():
            csv_path = output_dir / f'cost_sensitivity_{model_name}.csv'
            df.to_csv(csv_path, index=False)

        # Save summary table
        summary_df = self.create_summary_table()
        summary_df.to_csv(output_dir / 'cost_sensitivity_summary.csv', index=False)

        # Save plots
        self.plot_cost_sensitivity(output_dir / 'cost_sensitivity_curves.png')
        self.plot_cost_impact(output_dir / 'cost_impact_analysis.png')

        # Save JSON
        save_json({
            'cost_levels': [c * 100 for c in self.cost_levels],
            'summary': summary_df.to_dict(orient='records'),
            'detailed_results': {
                model_name: df.to_dict(orient='records')
                for model_name, df in self.results.items()
            }
        }, output_dir / 'cost_sensitivity_results.json')

        self.logger.info(f"Saved all cost sensitivity results to {output_dir}")

        return summary_df
