"""
Backtesting module for Bitcoin trading bot.
Simulates trading strategies on historical data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum

from trading_bot.utils.logger import LoggerMixin
from trading_bot.utils.decorators import timing
from trading_bot.evaluation.metrics import PerformanceMetrics


class OrderType(Enum):
    """Order types for backtesting."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Represents a trading order."""
    timestamp: pd.Timestamp
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    limit_price: Optional[float] = None
    order_id: Optional[str] = None


@dataclass
class Trade:
    """Represents an executed trade."""
    timestamp: pd.Timestamp
    side: OrderSide
    quantity: float
    price: float
    commission: float
    order_id: str
    trade_id: str


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float = 0.0


class Portfolio(LoggerMixin):
    """Portfolio management for backtesting."""

    def __init__(self, initial_capital: float, commission: float = 0.001):
        """
        Initialize portfolio.

        Args:
            initial_capital: Starting capital
            commission: Commission rate per trade
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.cash = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.returns = []

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value.

        Args:
            current_prices: Current market prices

        Returns:
            Total portfolio value
        """
        total_value = self.cash

        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position.current_price = current_prices[symbol]
                position.market_value = position.quantity * position.current_price
                position.unrealized_pnl = position.market_value - (position.quantity * position.avg_price)
                total_value += position.market_value

        return total_value

    def execute_order(self, order: Order, current_price: float) -> Optional[Trade]:
        """
        Execute a trading order.

        Args:
            order: Order to execute
            current_price: Current market price

        Returns:
            Executed trade or None if order cannot be filled
        """
        # For simplicity, assume all market orders are filled at current price
        if order.type == OrderType.MARKET:
            execution_price = current_price
        else:
            # Handle limit/stop orders
            execution_price = self._check_order_execution(order, current_price)
            if execution_price is None:
                return None

        # Calculate trade value and commission
        trade_value = order.quantity * execution_price
        commission = trade_value * self.commission

        # Check if we have enough cash for buy orders
        if order.side == OrderSide.BUY:
            total_cost = trade_value + commission
            if total_cost > self.cash:
                self.logger.warning(f"Insufficient cash for order. Required: {total_cost}, Available: {self.cash}")
                return None

        # Create trade
        trade = Trade(
            timestamp=order.timestamp,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            commission=commission,
            order_id=order.order_id or f"order_{datetime.now().timestamp()}",
            trade_id=f"trade_{datetime.now().timestamp()}"
        )

        # Update portfolio
        self._update_position(trade)
        self.trades.append(trade)

        return trade

    def _check_order_execution(self, order: Order, current_price: float) -> Optional[float]:
        """Check if limit/stop order should be executed."""
        if order.type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and current_price <= order.limit_price:
                return order.limit_price
            elif order.side == OrderSide.SELL and current_price >= order.limit_price:
                return order.limit_price
        elif order.type == OrderType.STOP:
            if order.side == OrderSide.BUY and current_price >= order.stop_price:
                return current_price
            elif order.side == OrderSide.SELL and current_price <= order.stop_price:
                return current_price

        return None

    def _update_position(self, trade: Trade):
        """Update position based on executed trade."""
        symbol = "BTC"  # Assuming single symbol for now

        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0.0,
                avg_price=0.0,
                current_price=trade.price,
                market_value=0.0,
                unrealized_pnl=0.0
            )

        position = self.positions[symbol]

        if trade.side == OrderSide.BUY:
            # Calculate new average price
            total_cost = (position.quantity * position.avg_price) + (trade.quantity * trade.price)
            total_quantity = position.quantity + trade.quantity

            if total_quantity > 0:
                position.avg_price = total_cost / total_quantity
            position.quantity = total_quantity

            # Update cash
            self.cash -= (trade.quantity * trade.price + trade.commission)

        else:  # SELL
            # Calculate realized PnL
            realized_pnl = (trade.price - position.avg_price) * trade.quantity - trade.commission
            position.realized_pnl += realized_pnl

            # Update position
            position.quantity -= trade.quantity

            # Update cash
            self.cash += (trade.quantity * trade.price - trade.commission)

            # Close position if quantity is zero or negative
            if position.quantity <= 0:
                if position.quantity < 0:
                    self.logger.warning(f"Position quantity became negative: {position.quantity}")
                del self.positions[symbol]

    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        trades_data = []
        for trade in self.trades:
            trades_data.append({
                'timestamp': trade.timestamp,
                'side': trade.side.value,
                'quantity': trade.quantity,
                'price': trade.price,
                'commission': trade.commission,
                'value': trade.quantity * trade.price,
                'order_id': trade.order_id,
                'trade_id': trade.trade_id
            })

        return pd.DataFrame(trades_data)


class Backtester(LoggerMixin):
    """Main backtesting engine."""

    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.0005
    ):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital
            commission: Commission rate per trade
            slippage: Slippage factor
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.portfolio = None
        self.results = {}

    @timing
    def run_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        strategy_name: str = "Strategy",
        position_size: float = 1.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data.

        Args:
            data: OHLCV data
            signals: Trading signals (-1: sell, 0: hold, 1: buy)
            strategy_name: Name of the strategy
            position_size: Position size as fraction of capital
            stop_loss: Stop loss percentage
            take_profit: Take profit percentage

        Returns:
            Backtest results dictionary
        """
        self.logger.info(f"Running backtest for {strategy_name}")

        # Initialize portfolio
        self.portfolio = Portfolio(self.initial_capital, self.commission)

        # Align data and signals
        aligned_data = pd.concat([data, signals], axis=1, join='inner')
        aligned_data.columns = list(data.columns) + ['signal']

        # Track portfolio metrics
        equity_curve = []
        returns = []
        positions_history = []

        current_position = 0  # 0: no position, 1: long, -1: short
        entry_price = 0.0
        stop_loss_price = None
        take_profit_price = None

        for i, (timestamp, row) in enumerate(aligned_data.iterrows()):
            current_price = row['close']
            signal = row['signal']

            # Apply slippage
            if self.slippage > 0:
                price_with_slippage = current_price * (1 + np.random.normal(0, self.slippage))
            else:
                price_with_slippage = current_price

            # Check for stop loss or take profit
            if current_position != 0:
                should_exit = False

                if stop_loss_price and current_price <= stop_loss_price:
                    should_exit = True
                    signal = -current_position  # Exit position
                    self.logger.debug(f"Stop loss triggered at {current_price}")

                elif take_profit_price and current_price >= take_profit_price:
                    should_exit = True
                    signal = -current_position  # Exit position
                    self.logger.debug(f"Take profit triggered at {current_price}")

            # Process trading signals
            if signal != 0 and signal != current_position:
                # Calculate position size
                portfolio_value = self.portfolio.get_portfolio_value({'BTC': current_price})
                trade_value = portfolio_value * position_size

                if signal > 0:  # Buy signal
                    if current_position <= 0:  # No position or short position
                        # Close short position first
                        if current_position < 0:
                            quantity = abs(current_position)
                            order = Order(
                                timestamp=timestamp,
                                side=OrderSide.BUY,
                                type=OrderType.MARKET,
                                quantity=quantity
                            )
                            self.portfolio.execute_order(order, price_with_slippage)

                        # Open long position
                        quantity = trade_value / price_with_slippage
                        order = Order(
                            timestamp=timestamp,
                            side=OrderSide.BUY,
                            type=OrderType.MARKET,
                            quantity=quantity
                        )

                        trade = self.portfolio.execute_order(order, price_with_slippage)
                        if trade:
                            current_position = 1
                            entry_price = price_with_slippage

                            # Set stop loss and take profit
                            if stop_loss:
                                stop_loss_price = entry_price * (1 - stop_loss)
                            if take_profit:
                                take_profit_price = entry_price * (1 + take_profit)

                elif signal < 0:  # Sell signal
                    if current_position >= 0:  # No position or long position
                        # Close long position first
                        if current_position > 0:
                            if 'BTC' in self.portfolio.positions:
                                quantity = self.portfolio.positions['BTC'].quantity
                                order = Order(
                                    timestamp=timestamp,
                                    side=OrderSide.SELL,
                                    type=OrderType.MARKET,
                                    quantity=quantity
                                )
                                self.portfolio.execute_order(order, price_with_slippage)

                        current_position = 0
                        stop_loss_price = None
                        take_profit_price = None

            # Record portfolio metrics
            portfolio_value = self.portfolio.get_portfolio_value({'BTC': current_price})
            equity_curve.append(portfolio_value)

            if i > 0:
                period_return = (portfolio_value - equity_curve[i-1]) / equity_curve[i-1]
                returns.append(period_return)

            positions_history.append(current_position)

        # Create results
        self.results = self._create_results(
            strategy_name,
            aligned_data,
            equity_curve,
            returns,
            positions_history
        )

        self.logger.info(f"Backtest completed for {strategy_name}")
        return self.results

    def _create_results(
        self,
        strategy_name: str,
        data: pd.DataFrame,
        equity_curve: List[float],
        returns: List[float],
        positions_history: List[int]
    ) -> Dict[str, Any]:
        """Create comprehensive backtest results."""
        results = {
            'strategy_name': strategy_name,
            'start_date': data.index[0],
            'end_date': data.index[-1],
            'initial_capital': self.initial_capital,
            'final_capital': equity_curve[-1] if equity_curve else self.initial_capital,
            'total_trades': len(self.portfolio.trades)
        }

        # Create time series
        equity_series = pd.Series(equity_curve, index=data.index)
        returns_series = pd.Series(returns, index=data.index[1:]) if returns else pd.Series([])

        results['equity_curve'] = equity_series
        results['returns'] = returns_series
        results['positions'] = pd.Series(positions_history, index=data.index)

        # Calculate performance metrics
        if len(returns) > 0:
            metrics_calculator = PerformanceMetrics()
            results['metrics'] = metrics_calculator.calculate_all_metrics(
                returns_series,
                prices=equity_series
            )

            # Add trading-specific metrics
            trades_df = self.portfolio.get_trades_df()
            if not trades_df.empty:
                results['trades'] = trades_df
                results['trade_analysis'] = self._analyze_trades(trades_df)

        return results

    def _analyze_trades(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze individual trades."""
        analysis = {}

        if trades_df.empty:
            return analysis

        # Group trades into round trips
        round_trips = []
        open_position = 0
        entry_price = 0
        entry_time = None

        for _, trade in trades_df.iterrows():
            if trade['side'] == 'buy':
                if open_position <= 0:
                    entry_price = trade['price']
                    entry_time = trade['timestamp']
                open_position += trade['quantity']
            else:  # sell
                if open_position > 0:
                    # Calculate round trip
                    pnl = (trade['price'] - entry_price) * trade['quantity'] - trade['commission']
                    duration = (trade['timestamp'] - entry_time).total_seconds() / 3600  # hours

                    round_trips.append({
                        'entry_time': entry_time,
                        'exit_time': trade['timestamp'],
                        'entry_price': entry_price,
                        'exit_price': trade['price'],
                        'quantity': trade['quantity'],
                        'pnl': pnl,
                        'duration_hours': duration,
                        'return_pct': (trade['price'] - entry_price) / entry_price
                    })

                open_position -= trade['quantity']

        if round_trips:
            round_trips_df = pd.DataFrame(round_trips)

            analysis['total_round_trips'] = len(round_trips)
            analysis['profitable_trades'] = len(round_trips_df[round_trips_df['pnl'] > 0])
            analysis['losing_trades'] = len(round_trips_df[round_trips_df['pnl'] < 0])
            analysis['avg_trade_pnl'] = round_trips_df['pnl'].mean()
            analysis['avg_trade_return'] = round_trips_df['return_pct'].mean()
            analysis['avg_trade_duration'] = round_trips_df['duration_hours'].mean()
            analysis['best_trade'] = round_trips_df['pnl'].max()
            analysis['worst_trade'] = round_trips_df['pnl'].min()

        return analysis

    def compare_strategies(
        self,
        strategies_results: Dict[str, Dict],
        benchmark_data: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Compare multiple strategy results.

        Args:
            strategies_results: Dictionary of strategy results
            benchmark_data: Benchmark data for comparison

        Returns:
            Comparison DataFrame
        """
        comparison_data = []

        for strategy_name, results in strategies_results.items():
            if 'metrics' in results:
                metrics = results['metrics']
                comparison_data.append({
                    'Strategy': strategy_name,
                    'Total Return': metrics.get('total_return', 0),
                    'Annualized Return': metrics.get('annualized_return', 0),
                    'Volatility': metrics.get('annualized_volatility', 0),
                    'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                    'Max Drawdown': metrics.get('max_drawdown', 0),
                    'Win Rate': metrics.get('win_rate', 0),
                    'Total Trades': metrics.get('total_trades', 0),
                    'Profit Factor': metrics.get('profit_factor', 0)
                })

        # Add benchmark if provided
        if benchmark_data is not None:
            benchmark_returns = benchmark_data.pct_change().dropna()
            metrics_calculator = PerformanceMetrics()
            benchmark_metrics = metrics_calculator.calculate_all_metrics(
                benchmark_returns,
                prices=benchmark_data
            )

            comparison_data.append({
                'Strategy': 'Benchmark',
                'Total Return': benchmark_metrics.get('total_return', 0),
                'Annualized Return': benchmark_metrics.get('annualized_return', 0),
                'Volatility': benchmark_metrics.get('annualized_volatility', 0),
                'Sharpe Ratio': benchmark_metrics.get('sharpe_ratio', 0),
                'Max Drawdown': benchmark_metrics.get('max_drawdown', 0),
                'Win Rate': np.nan,
                'Total Trades': np.nan,
                'Profit Factor': np.nan
            })

        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df.set_index('Strategy')

    def run_walk_forward_analysis(
        self,
        data: pd.DataFrame,
        signal_generator,
        train_window: int = 252,
        test_window: int = 63,
        step_size: int = 21
    ) -> Dict[str, Any]:
        """
        Run walk-forward analysis.

        Args:
            data: Historical data
            signal_generator: Function that generates signals from data
            train_window: Training window size in days
            test_window: Testing window size in days
            step_size: Step size for walk-forward

        Returns:
            Walk-forward analysis results
        """
        self.logger.info("Running walk-forward analysis")

        results = []
        start_idx = train_window

        while start_idx + test_window < len(data):
            # Define train and test periods
            train_start = start_idx - train_window
            train_end = start_idx
            test_start = start_idx
            test_end = start_idx + test_window

            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]

            # Generate signals for test period
            signals = signal_generator(train_data, test_data)

            # Run backtest on test period
            backtest_results = self.run_backtest(
                test_data,
                signals,
                strategy_name=f"WF_{test_data.index[0].strftime('%Y-%m-%d')}"
            )

            results.append({
                'period_start': test_data.index[0],
                'period_end': test_data.index[-1],
                'results': backtest_results
            })

            start_idx += step_size

        # Aggregate results
        aggregated_results = self._aggregate_walk_forward_results(results)

        self.logger.info(f"Walk-forward analysis completed with {len(results)} periods")
        return aggregated_results

    def _aggregate_walk_forward_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate walk-forward analysis results."""
        all_returns = []
        all_equity = []
        period_metrics = []

        for period_result in results:
            period_data = period_result['results']
            if 'returns' in period_data and len(period_data['returns']) > 0:
                all_returns.extend(period_data['returns'].tolist())
                all_equity.extend(period_data['equity_curve'].tolist())

                if 'metrics' in period_data:
                    period_metrics.append(period_data['metrics'])

        # Calculate overall metrics
        if all_returns:
            returns_series = pd.Series(all_returns)
            equity_series = pd.Series(all_equity)

            metrics_calculator = PerformanceMetrics()
            overall_metrics = metrics_calculator.calculate_all_metrics(
                returns_series,
                prices=equity_series
            )

            # Calculate stability metrics
            if period_metrics:
                metric_stability = {}
                for metric_name in period_metrics[0].keys():
                    values = [m.get(metric_name, np.nan) for m in period_metrics]
                    values = [v for v in values if not (np.isnan(v) or np.isinf(v))]
                    if values:
                        metric_stability[f"{metric_name}_mean"] = np.mean(values)
                        metric_stability[f"{metric_name}_std"] = np.std(values)

                overall_metrics.update(metric_stability)

        return {
            'periods': results,
            'overall_metrics': overall_metrics if all_returns else {},
            'combined_returns': pd.Series(all_returns) if all_returns else pd.Series([]),
            'combined_equity': pd.Series(all_equity) if all_equity else pd.Series([])
        }