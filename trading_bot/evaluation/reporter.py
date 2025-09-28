"""
Report generation module for Bitcoin trading bot evaluation.
Creates comprehensive performance reports with visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import warnings

from trading_bot.utils.logger import LoggerMixin
from trading_bot.utils.helpers import save_json, generate_timestamp
from trading_bot.evaluation.metrics import PerformanceMetrics

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')


class ReportGenerator(LoggerMixin):
    """Generate comprehensive performance reports with visualizations."""

    def __init__(self, output_dir: str = "reports"):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_full_report(
        self,
        backtest_results: Dict[str, Any],
        benchmark_data: Optional[pd.Series] = None,
        save_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.

        Args:
            backtest_results: Backtest results dictionary
            benchmark_data: Benchmark data for comparison
            save_plots: Whether to save plot files

        Returns:
            Report dictionary with all analyses
        """
        strategy_name = backtest_results.get('strategy_name', 'Strategy')
        timestamp = generate_timestamp()

        self.logger.info(f"Generating full report for {strategy_name}")

        report = {
            'strategy_name': strategy_name,
            'generated_at': datetime.now().isoformat(),
            'timestamp': timestamp,
            'summary': {},
            'analysis': {},
            'plots': {}
        }

        # Extract data
        equity_curve = backtest_results.get('equity_curve', pd.Series([]))
        returns = backtest_results.get('returns', pd.Series([]))
        metrics = backtest_results.get('metrics', {})
        trades = backtest_results.get('trades', pd.DataFrame())

        # Generate summary
        report['summary'] = self._generate_summary(backtest_results, benchmark_data)

        # Generate detailed analysis
        if not returns.empty:
            report['analysis']['returns_analysis'] = self._analyze_returns(returns)
            report['analysis']['risk_analysis'] = self._analyze_risk(returns, equity_curve)
            report['analysis']['drawdown_analysis'] = self._analyze_drawdowns(equity_curve)

        if not trades.empty:
            report['analysis']['trade_analysis'] = self._analyze_trades(trades)

        # Generate plots
        if save_plots:
            plots_dir = self.output_dir / f"{strategy_name}_{timestamp}"
            plots_dir.mkdir(parents=True, exist_ok=True)

            report['plots'] = self._generate_all_plots(
                backtest_results,
                benchmark_data,
                plots_dir
            )

        # Save report
        report_file = self.output_dir / f"{strategy_name}_report_{timestamp}.json"
        save_json(report, report_file)

        self.logger.info(f"Full report generated and saved to {report_file}")
        return report

    def _generate_summary(
        self,
        backtest_results: Dict[str, Any],
        benchmark_data: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Generate executive summary."""
        summary = {}

        # Basic information
        summary['period'] = {
            'start_date': backtest_results.get('start_date'),
            'end_date': backtest_results.get('end_date'),
            'duration_days': (backtest_results.get('end_date') - backtest_results.get('start_date')).days
        }

        summary['capital'] = {
            'initial': backtest_results.get('initial_capital'),
            'final': backtest_results.get('final_capital'),
            'total_return': (backtest_results.get('final_capital', 0) / backtest_results.get('initial_capital', 1)) - 1
        }

        # Key metrics
        metrics = backtest_results.get('metrics', {})
        summary['key_metrics'] = {
            'total_return': metrics.get('total_return', 0),
            'annualized_return': metrics.get('annualized_return', 0),
            'volatility': metrics.get('annualized_volatility', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'win_rate': metrics.get('win_rate', 0),
            'total_trades': metrics.get('total_trades', 0)
        }

        # Benchmark comparison
        if benchmark_data is not None:
            benchmark_returns = benchmark_data.pct_change().dropna()
            benchmark_total = (benchmark_data.iloc[-1] / benchmark_data.iloc[0]) - 1

            summary['benchmark_comparison'] = {
                'strategy_return': summary['key_metrics']['total_return'],
                'benchmark_return': benchmark_total,
                'excess_return': summary['key_metrics']['total_return'] - benchmark_total,
                'tracking_error': metrics.get('tracking_error', np.nan),
                'information_ratio': metrics.get('information_ratio', np.nan)
            }

        return summary

    def _analyze_returns(self, returns: pd.Series) -> Dict[str, Any]:
        """Analyze return characteristics."""
        analysis = {}

        # Basic statistics
        analysis['statistics'] = {
            'count': len(returns),
            'mean': returns.mean(),
            'std': returns.std(),
            'min': returns.min(),
            'max': returns.max(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        }

        # Distribution analysis
        analysis['distribution'] = {
            'positive_returns': (returns > 0).sum(),
            'negative_returns': (returns < 0).sum(),
            'zero_returns': (returns == 0).sum(),
            'positive_percentage': (returns > 0).mean() * 100,
            'best_day': returns.max(),
            'worst_day': returns.min()
        }

        # Rolling statistics
        analysis['rolling_stats'] = {
            'rolling_30d_volatility': returns.rolling(30).std().iloc[-1] * np.sqrt(252),
            'rolling_90d_volatility': returns.rolling(90).std().iloc[-1] * np.sqrt(252),
            'rolling_30d_sharpe': (returns.rolling(30).mean() / returns.rolling(30).std()).iloc[-1] * np.sqrt(252),
            'rolling_90d_sharpe': (returns.rolling(90).mean() / returns.rolling(90).std()).iloc[-1] * np.sqrt(252)
        }

        # Tail risk
        analysis['tail_risk'] = {
            'var_95': returns.quantile(0.05),
            'var_99': returns.quantile(0.01),
            'cvar_95': returns[returns <= returns.quantile(0.05)].mean(),
            'cvar_99': returns[returns <= returns.quantile(0.01)].mean(),
            'tail_ratio': returns.quantile(0.95) / abs(returns.quantile(0.05))
        }

        return analysis

    def _analyze_risk(self, returns: pd.Series, equity_curve: pd.Series) -> Dict[str, Any]:
        """Analyze risk characteristics."""
        analysis = {}

        # Volatility analysis
        analysis['volatility'] = {
            'daily_volatility': returns.std(),
            'annualized_volatility': returns.std() * np.sqrt(252),
            'volatility_of_volatility': returns.rolling(30).std().std(),
            'downside_volatility': returns[returns < 0].std() * np.sqrt(252)
        }

        # Drawdown analysis
        if not equity_curve.empty:
            peak = equity_curve.expanding().max()
            drawdown = (equity_curve - peak) / peak

            analysis['drawdown'] = {
                'current_drawdown': drawdown.iloc[-1],
                'max_drawdown': drawdown.min(),
                'avg_drawdown': drawdown[drawdown < 0].mean(),
                'drawdown_periods': self._count_drawdown_periods(drawdown),
                'time_underwater': (drawdown < 0).sum() / len(drawdown)
            }

        # Beta analysis (if benchmark available)
        analysis['systematic_risk'] = {
            'note': 'Requires benchmark data for beta calculation'
        }

        return analysis

    def _analyze_drawdowns(self, equity_curve: pd.Series) -> Dict[str, Any]:
        """Detailed drawdown analysis."""
        if equity_curve.empty:
            return {}

        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak

        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        start_idx = None
        start_value = None
        peak_value = None

        for i, (timestamp, dd_value) in enumerate(drawdown.items()):
            if dd_value < 0 and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                start_idx = i
                start_value = equity_curve.iloc[i]
                peak_value = peak.iloc[i]

            elif dd_value >= 0 and in_drawdown:
                # End of drawdown
                in_drawdown = False
                end_idx = i
                recovery_value = equity_curve.iloc[i]

                duration = end_idx - start_idx
                depth = (start_value - equity_curve.iloc[start_idx:end_idx].min()) / peak_value

                drawdown_periods.append({
                    'start_date': equity_curve.index[start_idx],
                    'end_date': equity_curve.index[end_idx],
                    'duration_days': duration,
                    'depth': depth,
                    'peak_value': peak_value,
                    'trough_value': equity_curve.iloc[start_idx:end_idx].min(),
                    'recovery_value': recovery_value
                })

        # Handle ongoing drawdown
        if in_drawdown:
            duration = len(drawdown) - start_idx
            depth = (start_value - equity_curve.iloc[start_idx:].min()) / peak_value

            drawdown_periods.append({
                'start_date': equity_curve.index[start_idx],
                'end_date': equity_curve.index[-1],
                'duration_days': duration,
                'depth': depth,
                'peak_value': peak_value,
                'trough_value': equity_curve.iloc[start_idx:].min(),
                'recovery_value': None,
                'ongoing': True
            })

        analysis = {
            'total_drawdown_periods': len(drawdown_periods),
            'drawdown_periods': drawdown_periods[:10],  # Top 10 worst drawdowns
            'average_drawdown_duration': np.mean([dd['duration_days'] for dd in drawdown_periods]),
            'average_drawdown_depth': np.mean([dd['depth'] for dd in drawdown_periods]),
            'longest_drawdown': max([dd['duration_days'] for dd in drawdown_periods]) if drawdown_periods else 0,
            'deepest_drawdown': max([dd['depth'] for dd in drawdown_periods]) if drawdown_periods else 0
        }

        return analysis

    def _analyze_trades(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Analyze individual trades."""
        if trades.empty:
            return {}

        analysis = {}

        # Basic trade statistics
        analysis['basic_stats'] = {
            'total_trades': len(trades),
            'buy_trades': len(trades[trades['side'] == 'buy']),
            'sell_trades': len(trades[trades['side'] == 'sell']),
            'average_trade_size': trades['quantity'].mean(),
            'total_volume': trades['value'].sum(),
            'total_commission': trades['commission'].sum()
        }

        # Trade timing analysis
        if len(trades) > 1:
            trade_intervals = trades['timestamp'].diff().dt.total_seconds() / 3600  # hours
            analysis['timing'] = {
                'average_time_between_trades': trade_intervals.mean(),
                'min_time_between_trades': trade_intervals.min(),
                'max_time_between_trades': trade_intervals.max(),
                'trades_per_day': len(trades) / ((trades['timestamp'].max() - trades['timestamp'].min()).days + 1)
            }

        # Price analysis
        analysis['price_analysis'] = {
            'average_execution_price': trades['price'].mean(),
            'price_range': {
                'min': trades['price'].min(),
                'max': trades['price'].max(),
                'std': trades['price'].std()
            }
        }

        return analysis

    def _count_drawdown_periods(self, drawdown: pd.Series) -> int:
        """Count the number of distinct drawdown periods."""
        is_drawdown = drawdown < 0
        periods = 0
        in_drawdown = False

        for dd in is_drawdown:
            if dd and not in_drawdown:
                periods += 1
                in_drawdown = True
            elif not dd:
                in_drawdown = False

        return periods

    def _generate_all_plots(
        self,
        backtest_results: Dict[str, Any],
        benchmark_data: Optional[pd.Series],
        plots_dir: Path
    ) -> Dict[str, str]:
        """Generate all visualization plots."""
        plot_files = {}

        equity_curve = backtest_results.get('equity_curve', pd.Series([]))
        returns = backtest_results.get('returns', pd.Series([]))
        trades = backtest_results.get('trades', pd.DataFrame())

        try:
            # 1. Equity curve
            if not equity_curve.empty:
                plot_files['equity_curve'] = self._plot_equity_curve(
                    equity_curve, benchmark_data, plots_dir / "equity_curve.html"
                )

            # 2. Returns distribution
            if not returns.empty:
                plot_files['returns_distribution'] = self._plot_returns_distribution(
                    returns, plots_dir / "returns_distribution.html"
                )

            # 3. Drawdown chart
            if not equity_curve.empty:
                plot_files['drawdown'] = self._plot_drawdown(
                    equity_curve, plots_dir / "drawdown.html"
                )

            # 4. Rolling metrics
            if not returns.empty:
                plot_files['rolling_metrics'] = self._plot_rolling_metrics(
                    returns, plots_dir / "rolling_metrics.html"
                )

            # 5. Trade analysis
            if not trades.empty:
                plot_files['trade_analysis'] = self._plot_trade_analysis(
                    trades, plots_dir / "trade_analysis.html"
                )

            # 6. Monthly returns heatmap
            if not returns.empty:
                plot_files['monthly_returns'] = self._plot_monthly_returns(
                    returns, plots_dir / "monthly_returns.html"
                )

        except Exception as e:
            self.logger.error(f"Error generating plots: {e}")

        return plot_files

    def _plot_equity_curve(
        self,
        equity_curve: pd.Series,
        benchmark_data: Optional[pd.Series],
        filepath: Path
    ) -> str:
        """Plot equity curve with optional benchmark."""
        fig = go.Figure()

        # Strategy equity curve
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode='lines',
            name='Strategy',
            line=dict(color='blue', width=2)
        ))

        # Benchmark if provided
        if benchmark_data is not None:
            # Normalize benchmark to same starting value
            benchmark_normalized = benchmark_data / benchmark_data.iloc[0] * equity_curve.iloc[0]
            fig.add_trace(go.Scatter(
                x=benchmark_normalized.index,
                y=benchmark_normalized.values,
                mode='lines',
                name='Benchmark',
                line=dict(color='red', width=2, dash='dash')
            ))

        fig.update_layout(
            title='Equity Curve',
            xaxis_title='Date',
            yaxis_title='Portfolio Value',
            hovermode='x unified',
            template='plotly_white'
        )

        fig.write_html(filepath)
        return str(filepath)

    def _plot_returns_distribution(self, returns: pd.Series, filepath: Path) -> str:
        """Plot returns distribution with statistics."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Returns Time Series', 'Distribution Histogram',
                           'Q-Q Plot vs Normal', 'Rolling Volatility'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Returns time series
        fig.add_trace(
            go.Scatter(x=returns.index, y=returns.values, mode='lines', name='Returns'),
            row=1, col=1
        )

        # Histogram
        fig.add_trace(
            go.Histogram(x=returns.values, nbinsx=50, name='Distribution'),
            row=1, col=2
        )

        # Q-Q plot (simplified)
        from scipy import stats
        qq_data = stats.probplot(returns.dropna(), dist="norm")
        fig.add_trace(
            go.Scatter(x=qq_data[0][0], y=qq_data[0][1], mode='markers', name='Q-Q Plot'),
            row=2, col=1
        )

        # Rolling volatility
        rolling_vol = returns.rolling(30).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(x=rolling_vol.index, y=rolling_vol.values, mode='lines', name='30D Volatility'),
            row=2, col=2
        )

        fig.update_layout(
            title='Returns Analysis',
            template='plotly_white',
            showlegend=False
        )

        fig.write_html(filepath)
        return str(filepath)

    def _plot_drawdown(self, equity_curve: pd.Series, filepath: Path) -> str:
        """Plot drawdown analysis."""
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Equity Curve with Peaks', 'Drawdown'],
            shared_xaxes=True
        )

        # Equity curve with peaks
        fig.add_trace(
            go.Scatter(x=equity_curve.index, y=equity_curve.values, mode='lines', name='Equity'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=peak.index, y=peak.values, mode='lines', name='Peak', line=dict(dash='dot')),
            row=1, col=1
        )

        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=drawdown.index, y=drawdown.values * 100, mode='lines',
                name='Drawdown %', fill='tonexty', fillcolor='rgba(255,0,0,0.3)'
            ),
            row=2, col=1
        )

        fig.update_layout(
            title='Drawdown Analysis',
            template='plotly_white'
        )

        fig.write_html(filepath)
        return str(filepath)

    def _plot_rolling_metrics(self, returns: pd.Series, filepath: Path) -> str:
        """Plot rolling performance metrics."""
        # Calculate rolling metrics
        rolling_sharpe = (returns.rolling(252).mean() / returns.rolling(252).std()) * np.sqrt(252)
        rolling_vol = returns.rolling(252).std() * np.sqrt(252)
        rolling_return = returns.rolling(252).mean() * 252

        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Rolling Annual Return', 'Rolling Volatility', 'Rolling Sharpe Ratio'],
            shared_xaxes=True
        )

        fig.add_trace(
            go.Scatter(x=rolling_return.index, y=rolling_return.values * 100, mode='lines', name='Return %'),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=rolling_vol.index, y=rolling_vol.values * 100, mode='lines', name='Volatility %'),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, mode='lines', name='Sharpe Ratio'),
            row=3, col=1
        )

        fig.update_layout(
            title='Rolling Performance Metrics (252-day window)',
            template='plotly_white',
            showlegend=False
        )

        fig.write_html(filepath)
        return str(filepath)

    def _plot_trade_analysis(self, trades: pd.DataFrame, filepath: Path) -> str:
        """Plot trade analysis."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Trade Sizes', 'Trade Prices', 'Trade Frequency', 'Cumulative Volume'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Trade sizes
        fig.add_trace(
            go.Scatter(x=trades['timestamp'], y=trades['quantity'], mode='markers', name='Trade Size'),
            row=1, col=1
        )

        # Trade prices
        fig.add_trace(
            go.Scatter(x=trades['timestamp'], y=trades['price'], mode='markers', name='Price'),
            row=1, col=2
        )

        # Trade frequency (daily)
        daily_trades = trades.groupby(trades['timestamp'].dt.date).size()
        fig.add_trace(
            go.Bar(x=daily_trades.index, y=daily_trades.values, name='Daily Trades'),
            row=2, col=1
        )

        # Cumulative volume
        cumulative_volume = trades['value'].cumsum()
        fig.add_trace(
            go.Scatter(x=trades['timestamp'], y=cumulative_volume, mode='lines', name='Cumulative Volume'),
            row=2, col=2
        )

        fig.update_layout(
            title='Trade Analysis',
            template='plotly_white',
            showlegend=False
        )

        fig.write_html(filepath)
        return str(filepath)

    def _plot_monthly_returns(self, returns: pd.Series, filepath: Path) -> str:
        """Plot monthly returns heatmap."""
        # Resample to monthly returns
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

        # Create year-month matrix
        monthly_returns_df = monthly_returns.to_frame('returns')
        monthly_returns_df['year'] = monthly_returns_df.index.year
        monthly_returns_df['month'] = monthly_returns_df.index.month

        heatmap_data = monthly_returns_df.pivot(index='year', columns='month', values='returns')

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values * 100,
            x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            y=heatmap_data.index,
            colorscale='RdYlGn',
            text=np.round(heatmap_data.values * 100, 2),
            texttemplate='%{text}%',
            textfont={"size": 10},
            hoverongaps=False
        ))

        fig.update_layout(
            title='Monthly Returns Heatmap (%)',
            xaxis_title='Month',
            yaxis_title='Year',
            template='plotly_white'
        )

        fig.write_html(filepath)
        return str(filepath)

    def generate_comparison_report(
        self,
        strategies_results: Dict[str, Dict],
        benchmark_data: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Generate comparison report for multiple strategies."""
        timestamp = generate_timestamp()

        self.logger.info(f"Generating comparison report for {len(strategies_results)} strategies")

        report = {
            'generated_at': datetime.now().isoformat(),
            'timestamp': timestamp,
            'strategies': list(strategies_results.keys()),
            'comparison': {},
            'plots': {}
        }

        # Create comparison table
        comparison_data = []
        for strategy_name, results in strategies_results.items():
            metrics = results.get('metrics', {})
            comparison_data.append({
                'Strategy': strategy_name,
                'Total Return': f"{metrics.get('total_return', 0):.2%}",
                'Annual Return': f"{metrics.get('annualized_return', 0):.2%}",
                'Volatility': f"{metrics.get('annualized_volatility', 0):.2%}",
                'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.3f}",
                'Max Drawdown': f"{metrics.get('max_drawdown', 0):.2%}",
                'Win Rate': f"{metrics.get('win_rate', 0):.2%}",
                'Total Trades': f"{metrics.get('total_trades', 0):.0f}"
            })

        report['comparison']['summary_table'] = comparison_data

        # Generate comparison plots
        plots_dir = self.output_dir / f"comparison_{timestamp}"
        plots_dir.mkdir(parents=True, exist_ok=True)

        try:
            report['plots'] = self._generate_comparison_plots(
                strategies_results, benchmark_data, plots_dir
            )
        except Exception as e:
            self.logger.error(f"Error generating comparison plots: {e}")

        # Save report
        report_file = self.output_dir / f"comparison_report_{timestamp}.json"
        save_json(report, report_file)

        self.logger.info(f"Comparison report generated and saved to {report_file}")
        return report

    def _generate_comparison_plots(
        self,
        strategies_results: Dict[str, Dict],
        benchmark_data: Optional[pd.Series],
        plots_dir: Path
    ) -> Dict[str, str]:
        """Generate comparison plots."""
        plot_files = {}

        # Equity curves comparison
        fig = go.Figure()

        for strategy_name, results in strategies_results.items():
            equity_curve = results.get('equity_curve', pd.Series([]))
            if not equity_curve.empty:
                # Normalize to 100 for comparison
                normalized_equity = equity_curve / equity_curve.iloc[0] * 100
                fig.add_trace(go.Scatter(
                    x=normalized_equity.index,
                    y=normalized_equity.values,
                    mode='lines',
                    name=strategy_name
                ))

        if benchmark_data is not None:
            benchmark_normalized = benchmark_data / benchmark_data.iloc[0] * 100
            fig.add_trace(go.Scatter(
                x=benchmark_normalized.index,
                y=benchmark_normalized.values,
                mode='lines',
                name='Benchmark',
                line=dict(dash='dash')
            ))

        fig.update_layout(
            title='Normalized Equity Curves Comparison',
            xaxis_title='Date',
            yaxis_title='Normalized Value (Base = 100)',
            template='plotly_white'
        )

        equity_file = plots_dir / "equity_comparison.html"
        fig.write_html(equity_file)
        plot_files['equity_comparison'] = str(equity_file)

        return plot_files