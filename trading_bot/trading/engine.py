"""
Trading engine module for Bitcoin trading bot.
Orchestrates signal generation, risk management, and order execution.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import json

from trading_bot.utils.logger import LoggerMixin
from trading_bot.utils.decorators import timing, retry
from trading_bot.utils.helpers import save_json, generate_timestamp
from trading_bot.trading.signals import SignalGenerator, SignalType, TradingSignal
from trading_bot.trading.risk import RiskManager, PositionSizeMethod
from trading_bot.data.collector import DataCollector
from trading_bot.models.trainer import BaseModel
from trading_bot.config.settings import Settings


class TradingMode(Enum):
    """Trading modes."""
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"


class OrderStatus(Enum):
    """Order execution status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class TradingOrder:
    """Represents a trading order."""
    order_id: str
    timestamp: pd.Timestamp
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market', 'limit'
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    metadata: Optional[Dict] = None


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    entry_time: pd.Timestamp
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class TradingState:
    """Current state of the trading system."""
    mode: TradingMode
    is_active: bool
    portfolio_value: float
    cash_balance: float
    positions: Dict[str, Position]
    pending_orders: List[TradingOrder]
    last_update: pd.Timestamp
    daily_pnl: float
    total_pnl: float


class TradingEngine(LoggerMixin):
    """Main trading engine that orchestrates the entire trading system."""

    def __init__(
        self,
        settings: Settings,
        model: Optional[BaseModel] = None,
        mode: TradingMode = TradingMode.PAPER
    ):
        """
        Initialize trading engine.

        Args:
            settings: Application settings
            model: Trained ML model for predictions
            mode: Trading mode (paper/live/backtest)
        """
        self.settings = settings
        self.model = model
        self.mode = mode

        # Initialize components
        self.data_collector = DataCollector(settings)
        self.signal_generator = SignalGenerator(settings.get('trading.signals', {}))
        self.risk_manager = RiskManager(
            max_position_size=settings.position_limit,
            max_portfolio_var=settings.get('trading.risk_management.var_limit', 0.05),
            max_drawdown=settings.max_drawdown,
            stop_loss_pct=settings.stop_loss,
            take_profit_pct=settings.take_profit
        )

        # Trading state
        self.state = TradingState(
            mode=mode,
            is_active=False,
            portfolio_value=settings.get('trading.initial_capital', 100000),
            cash_balance=settings.get('trading.initial_capital', 100000),
            positions={},
            pending_orders=[],
            last_update=pd.Timestamp.now(),
            daily_pnl=0.0,
            total_pnl=0.0
        )

        # Trading history
        self.order_history = []
        self.trade_history = []
        self.performance_history = []

        # Configuration
        self.symbols = settings.symbols
        self.update_frequency = settings.get('data.update_frequency', 300)  # 5 minutes
        self.commission_rate = settings.get('trading.execution.commission', 0.001)
        self.slippage = settings.get('trading.execution.slippage', 0.001)

    async def start_trading(self) -> None:
        """Start the trading engine."""
        self.logger.info(f"Starting trading engine in {self.mode.value} mode")

        self.state.is_active = True
        self.state.last_update = pd.Timestamp.now()

        try:
            if self.mode == TradingMode.LIVE:
                await self._run_live_trading()
            elif self.mode == TradingMode.PAPER:
                await self._run_paper_trading()
            else:
                self.logger.error(f"Trading mode {self.mode.value} not supported in start_trading")

        except Exception as e:
            self.logger.error(f"Error in trading engine: {e}")
            self.state.is_active = False
            raise

    async def stop_trading(self) -> None:
        """Stop the trading engine."""
        self.logger.info("Stopping trading engine")
        self.state.is_active = False

        # Close all positions
        await self._close_all_positions()

        # Cancel pending orders
        await self._cancel_all_orders()

        # Save final state
        await self._save_trading_state()

    async def _run_live_trading(self) -> None:
        """Run live trading loop."""
        self.logger.info("Starting live trading loop")

        while self.state.is_active:
            try:
                # Fetch latest market data
                market_data = await self._fetch_market_data()

                # Generate predictions if model available
                if self.model:
                    predictions = await self._generate_predictions(market_data)
                else:
                    predictions = None

                # Update positions and portfolio
                await self._update_portfolio_state(market_data)

                # Check risk limits and emergency stops
                emergency_stop, reason = await self._check_emergency_conditions()
                if emergency_stop:
                    self.logger.critical(f"Emergency stop triggered: {reason}")
                    await self.stop_trading()
                    break

                # Generate and execute trading decisions
                if predictions is not None:
                    await self._execute_trading_cycle(market_data, predictions)

                # Process pending orders
                await self._process_pending_orders(market_data)

                # Update performance metrics
                await self._update_performance_metrics()

                # Save state periodically
                if len(self.performance_history) % 10 == 0:
                    await self._save_trading_state()

                # Wait before next iteration
                await asyncio.sleep(self.update_frequency)

            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    async def _run_paper_trading(self) -> None:
        """Run paper trading simulation."""
        self.logger.info("Starting paper trading simulation")

        while self.state.is_active:
            try:
                # Fetch latest market data
                market_data = await self._fetch_market_data()

                # Generate predictions
                if self.model:
                    predictions = await self._generate_predictions(market_data)

                    # Execute trading cycle
                    await self._execute_trading_cycle(market_data, predictions)

                # Update portfolio
                await self._update_portfolio_state(market_data)

                # Process orders (simulate execution)
                await self._simulate_order_execution(market_data)

                # Update metrics
                await self._update_performance_metrics()

                # Wait before next iteration
                await asyncio.sleep(self.update_frequency)

            except Exception as e:
                self.logger.error(f"Error in paper trading: {e}")
                await asyncio.sleep(60)

    @retry(max_attempts=3, delay=5.0)
    async def _fetch_market_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch latest market data for all symbols."""
        try:
            market_data = {}
            for symbol in self.symbols:
                data = await self.data_collector.fetch_data(
                    symbol=symbol,
                    timeframe='1h',
                    limit=100
                )
                if data:
                    # Get the most recent data source
                    latest_data = list(data.values())[0]
                    market_data[symbol] = latest_data

            return market_data

        except Exception as e:
            self.logger.error(f"Failed to fetch market data: {e}")
            raise

    async def _generate_predictions(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """Generate model predictions for all symbols."""
        predictions = {}

        for symbol, data in market_data.items():
            try:
                if len(data) < 50:  # Minimum data required
                    continue

                # Prepare features (simplified - should use FeatureEngineer)
                features = self._prepare_features(data)

                if features is not None and len(features) > 0:
                    # Generate prediction
                    pred = self.model.predict(features[-1].reshape(1, -1))
                    predictions[symbol] = pred

            except Exception as e:
                self.logger.error(f"Error generating prediction for {symbol}: {e}")

        return predictions

    def _prepare_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare features for model prediction (simplified version)."""
        try:
            # Simple feature preparation - in practice, use FeatureEngineer
            if len(data) < 20:
                return None

            features = []

            # Price features
            features.extend([
                data['close'].iloc[-1],
                data['close'].iloc[-1] / data['open'].iloc[-1] - 1,  # Daily return
                data['high'].iloc[-1] / data['low'].iloc[-1] - 1,    # Daily range
                data['volume'].iloc[-1] / data['volume'].rolling(20).mean().iloc[-1] - 1  # Volume ratio
            ])

            # Technical indicators (simplified)
            sma_20 = data['close'].rolling(20).mean().iloc[-1]
            features.append(data['close'].iloc[-1] / sma_20 - 1)  # Price vs SMA

            # Volatility
            returns = data['close'].pct_change()
            volatility = returns.rolling(20).std().iloc[-1]
            features.append(volatility)

            return np.array(features)

        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return None

    async def _execute_trading_cycle(
        self,
        market_data: Dict[str, pd.DataFrame],
        predictions: Dict[str, np.ndarray]
    ) -> None:
        """Execute complete trading cycle: signals -> risk check -> orders."""
        for symbol in self.symbols:
            if symbol not in market_data or symbol not in predictions:
                continue

            try:
                data = market_data[symbol]
                prediction = predictions[symbol][0] if len(predictions[symbol]) > 0 else 0

                # Generate trading signal
                signals = self.signal_generator.generate_signals(
                    predictions=pd.Series([prediction]),
                    data=data
                )

                if len(signals) == 0:
                    continue

                signal = signals.iloc[-1]
                current_price = data['close'].iloc[-1]

                # Check if we should trade
                if signal != SignalType.HOLD.value:
                    await self._process_trading_signal(
                        symbol=symbol,
                        signal=signal,
                        prediction=prediction,
                        current_price=current_price,
                        market_data=data
                    )

            except Exception as e:
                self.logger.error(f"Error in trading cycle for {symbol}: {e}")

    async def _process_trading_signal(
        self,
        symbol: str,
        signal: int,
        prediction: float,
        current_price: float,
        market_data: pd.DataFrame
    ) -> None:
        """Process a trading signal and potentially create orders."""
        # Calculate signal strength
        signal_strength = min(abs(prediction), 1.0)

        # Calculate volatility for risk management
        returns = market_data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.2

        # Get current position
        current_position = self.state.positions.get(symbol)

        # Determine action
        if signal == SignalType.BUY.value:
            if current_position is None or current_position.quantity <= 0:
                await self._create_buy_order(symbol, signal_strength, current_price, volatility)
        elif signal == SignalType.SELL.value:
            if current_position is not None and current_position.quantity > 0:
                await self._create_sell_order(symbol, current_position, current_price)

    async def _create_buy_order(
        self,
        symbol: str,
        signal_strength: float,
        current_price: float,
        volatility: float
    ) -> None:
        """Create a buy order."""
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            signal_strength=signal_strength,
            current_price=current_price,
            portfolio_value=self.state.portfolio_value,
            volatility=volatility,
            method=PositionSizeMethod.VOLATILITY
        )

        if position_size < 10:  # Minimum position size
            return

        # Calculate quantity
        quantity = position_size / current_price

        # Check risk limits
        current_positions = {s: {'weight': pos.market_value / self.state.portfolio_value}
                           for s, pos in self.state.positions.items()}

        is_allowed, violations = self.risk_manager.check_risk_limits(
            proposed_position=quantity,
            current_price=current_price,
            portfolio_value=self.state.portfolio_value,
            current_positions=current_positions,
            price_data=pd.DataFrame()  # Would need full price history
        )

        if not is_allowed:
            self.logger.warning(f"Buy order rejected for {symbol}: {violations}")
            return

        # Calculate stop loss and take profit
        stop_loss, take_profit = self.risk_manager.calculate_stop_loss_take_profit(
            entry_price=current_price,
            position_type='long',
            volatility=volatility
        )

        # Create order
        order = TradingOrder(
            order_id=f"buy_{symbol}_{generate_timestamp()}",
            timestamp=pd.Timestamp.now(),
            symbol=symbol,
            side='buy',
            order_type='market',
            quantity=quantity,
            price=current_price,
            metadata={
                'signal_strength': signal_strength,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
        )

        self.state.pending_orders.append(order)
        self.logger.info(f"Created buy order: {order.order_id} for {quantity:.6f} {symbol} at ${current_price:.2f}")

    async def _create_sell_order(
        self,
        symbol: str,
        position: Position,
        current_price: float
    ) -> None:
        """Create a sell order for existing position."""
        # Sell entire position
        order = TradingOrder(
            order_id=f"sell_{symbol}_{generate_timestamp()}",
            timestamp=pd.Timestamp.now(),
            symbol=symbol,
            side='sell',
            order_type='market',
            quantity=position.quantity,
            price=current_price,
            metadata={'position_close': True}
        )

        self.state.pending_orders.append(order)
        self.logger.info(f"Created sell order: {order.order_id} for {position.quantity:.6f} {symbol} at ${current_price:.2f}")

    async def _process_pending_orders(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """Process and execute pending orders."""
        executed_orders = []

        for order in self.state.pending_orders:
            if order.symbol in market_data:
                current_price = market_data[order.symbol]['close'].iloc[-1]

                # Simulate order execution
                if await self._execute_order(order, current_price):
                    executed_orders.append(order)

        # Remove executed orders
        for order in executed_orders:
            self.state.pending_orders.remove(order)

    async def _execute_order(self, order: TradingOrder, current_price: float) -> bool:
        """Execute a trading order."""
        try:
            # Apply slippage
            if order.side == 'buy':
                execution_price = current_price * (1 + self.slippage)
            else:
                execution_price = current_price * (1 - self.slippage)

            # Calculate commission
            trade_value = order.quantity * execution_price
            commission = trade_value * self.commission_rate

            # Check if we have enough cash for buy orders
            if order.side == 'buy':
                total_cost = trade_value + commission
                if total_cost > self.state.cash_balance:
                    self.logger.warning(f"Insufficient cash for order {order.order_id}")
                    order.status = OrderStatus.REJECTED
                    return True

            # Update order
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.avg_fill_price = execution_price
            order.commission = commission

            # Update position
            await self._update_position(order)

            # Add to history
            self.order_history.append(order)

            self.logger.info(
                f"Executed order {order.order_id}: {order.side} {order.quantity:.6f} {order.symbol} "
                f"at ${execution_price:.2f} (commission: ${commission:.2f})"
            )

            return True

        except Exception as e:
            self.logger.error(f"Error executing order {order.order_id}: {e}")
            order.status = OrderStatus.REJECTED
            return True

    async def _simulate_order_execution(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """Simulate order execution for paper trading."""
        await self._process_pending_orders(market_data)

    async def _update_position(self, order: TradingOrder) -> None:
        """Update position based on executed order."""
        symbol = order.symbol

        if symbol not in self.state.positions:
            # New position
            if order.side == 'buy':
                self.state.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=order.filled_quantity,
                    avg_price=order.avg_fill_price,
                    current_price=order.avg_fill_price,
                    market_value=order.filled_quantity * order.avg_fill_price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    entry_time=order.timestamp,
                    stop_loss=order.metadata.get('stop_loss'),
                    take_profit=order.metadata.get('take_profit')
                )

                # Update cash
                self.state.cash_balance -= (order.filled_quantity * order.avg_fill_price + order.commission)

        else:
            # Existing position
            position = self.state.positions[symbol]

            if order.side == 'buy':
                # Add to position
                total_cost = (position.quantity * position.avg_price) + (order.filled_quantity * order.avg_fill_price)
                total_quantity = position.quantity + order.filled_quantity

                position.avg_price = total_cost / total_quantity if total_quantity > 0 else 0
                position.quantity = total_quantity

                # Update cash
                self.state.cash_balance -= (order.filled_quantity * order.avg_fill_price + order.commission)

            else:  # sell
                # Calculate realized PnL
                realized_pnl = (order.avg_fill_price - position.avg_price) * order.filled_quantity - order.commission
                position.realized_pnl += realized_pnl
                self.state.total_pnl += realized_pnl

                # Update position
                position.quantity -= order.filled_quantity

                # Update cash
                self.state.cash_balance += (order.filled_quantity * order.avg_fill_price - order.commission)

                # Close position if fully sold
                if position.quantity <= 0:
                    del self.state.positions[symbol]

    async def _update_portfolio_state(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """Update portfolio value and position states."""
        total_value = self.state.cash_balance

        # Update positions
        for symbol, position in self.state.positions.items():
            if symbol in market_data:
                current_price = market_data[symbol]['close'].iloc[-1]
                position.current_price = current_price
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = (current_price - position.avg_price) * position.quantity

                total_value += position.market_value

        # Update portfolio value
        previous_value = self.state.portfolio_value
        self.state.portfolio_value = total_value

        # Calculate daily PnL
        if previous_value > 0:
            self.state.daily_pnl = (total_value - previous_value) / previous_value

        self.state.last_update = pd.Timestamp.now()

    async def _check_emergency_conditions(self) -> Tuple[bool, str]:
        """Check for emergency stop conditions."""
        # Calculate current drawdown
        if len(self.performance_history) > 0:
            peak_value = max(self.performance_history)
            current_drawdown = (self.state.portfolio_value - peak_value) / peak_value

            # Check emergency conditions
            return self.risk_manager.emergency_stop_check(
                current_drawdown=current_drawdown,
                portfolio_var=0.0  # Would need proper VaR calculation
            )

        return False, ""

    async def _update_performance_metrics(self) -> None:
        """Update performance tracking."""
        self.performance_history.append(self.state.portfolio_value)

        # Log performance periodically
        if len(self.performance_history) % 100 == 0:
            total_return = (self.state.portfolio_value / self.performance_history[0] - 1) * 100
            self.logger.info(f"Portfolio update: ${self.state.portfolio_value:.2f} ({total_return:+.2f}%)")

    async def _close_all_positions(self) -> None:
        """Close all open positions."""
        for symbol, position in list(self.state.positions.items()):
            try:
                # Create market sell order
                order = TradingOrder(
                    order_id=f"close_{symbol}_{generate_timestamp()}",
                    timestamp=pd.Timestamp.now(),
                    symbol=symbol,
                    side='sell',
                    order_type='market',
                    quantity=position.quantity,
                    price=position.current_price,
                    metadata={'force_close': True}
                )

                # Execute immediately
                await self._execute_order(order, position.current_price)

            except Exception as e:
                self.logger.error(f"Error closing position for {symbol}: {e}")

    async def _cancel_all_orders(self) -> None:
        """Cancel all pending orders."""
        for order in self.state.pending_orders:
            order.status = OrderStatus.CANCELLED

        self.state.pending_orders.clear()
        self.logger.info("Cancelled all pending orders")

    async def _save_trading_state(self) -> None:
        """Save current trading state to file."""
        try:
            state_data = {
                'timestamp': datetime.now().isoformat(),
                'mode': self.state.mode.value,
                'portfolio_value': self.state.portfolio_value,
                'cash_balance': self.state.cash_balance,
                'total_pnl': self.state.total_pnl,
                'positions': {symbol: asdict(pos) for symbol, pos in self.state.positions.items()},
                'pending_orders_count': len(self.state.pending_orders),
                'total_orders': len(self.order_history),
                'performance_history': self.performance_history[-100:]  # Last 100 values
            }

            filename = f"trading_state_{generate_timestamp()}.json"
            filepath = f"logs/{filename}"
            save_json(state_data, filepath)

        except Exception as e:
            self.logger.error(f"Error saving trading state: {e}")

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary."""
        summary = {
            'portfolio_value': self.state.portfolio_value,
            'cash_balance': self.state.cash_balance,
            'total_pnl': self.state.total_pnl,
            'daily_pnl': self.state.daily_pnl,
            'positions_count': len(self.state.positions),
            'pending_orders_count': len(self.state.pending_orders),
            'total_orders': len(self.order_history),
            'mode': self.state.mode.value,
            'is_active': self.state.is_active,
            'last_update': self.state.last_update.isoformat() if self.state.last_update else None
        }

        # Add position details
        if self.state.positions:
            summary['positions'] = []
            for symbol, position in self.state.positions.items():
                pos_summary = {
                    'symbol': symbol,
                    'quantity': position.quantity,
                    'avg_price': position.avg_price,
                    'current_price': position.current_price,
                    'market_value': position.market_value,
                    'unrealized_pnl': position.unrealized_pnl,
                    'pnl_percentage': (position.unrealized_pnl / (position.avg_price * position.quantity)) * 100
                }
                summary['positions'].append(pos_summary)

        return summary

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        if len(self.performance_history) < 2:
            return {'error': 'Insufficient data for metrics calculation'}

        values = pd.Series(self.performance_history)
        returns = values.pct_change().dropna()

        # Calculate metrics
        total_return = (values.iloc[-1] / values.iloc[0] - 1) * 100
        annualized_return = ((values.iloc[-1] / values.iloc[0]) ** (252 / len(values)) - 1) * 100
        volatility = returns.std() * np.sqrt(252) * 100

        # Drawdown
        peak = values.expanding().max()
        drawdown = (values - peak) / peak
        max_drawdown = drawdown.min() * 100

        # Sharpe ratio
        excess_returns = returns - (self.risk_manager.risk_free_rate / 252)
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        return {
            'total_return_pct': total_return,
            'annualized_return_pct': annualized_return,
            'volatility_pct': volatility,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'current_drawdown_pct': drawdown.iloc[-1] * 100,
            'win_rate': self._calculate_win_rate(),
            'total_trades': len(self.order_history)
        }

    def _calculate_win_rate(self) -> float:
        """Calculate win rate from closed positions."""
        realized_trades = [order for order in self.order_history if order.side == 'sell']

        if not realized_trades:
            return 0.0

        # This is simplified - would need proper trade pairing
        profitable_trades = sum(1 for order in realized_trades
                              if order.metadata and order.metadata.get('realized_pnl', 0) > 0)

        return (profitable_trades / len(realized_trades)) * 100