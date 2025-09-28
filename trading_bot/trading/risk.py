"""
Risk management module for Bitcoin trading bot.
Implements position sizing, stop-loss, and portfolio risk controls.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import warnings

from trading_bot.utils.logger import LoggerMixin
from trading_bot.utils.decorators import timing


class RiskLevel(Enum):
    """Risk levels for position sizing."""
    CONSERVATIVE = 0.01
    MODERATE = 0.02
    AGGRESSIVE = 0.05
    VERY_AGGRESSIVE = 0.10


class PositionSizeMethod(Enum):
    """Position sizing methods."""
    FIXED = "fixed"
    PERCENTAGE = "percentage"
    VOLATILITY = "volatility"
    KELLY = "kelly"
    RISK_PARITY = "risk_parity"


@dataclass
class RiskMetrics:
    """Container for risk metrics."""
    portfolio_value: float
    portfolio_var: float
    portfolio_cvar: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    volatility: float
    beta: float = None
    correlation: float = None


@dataclass
class PositionRisk:
    """Container for individual position risk metrics."""
    symbol: str
    position_size: float
    position_value: float
    position_weight: float
    var_contribution: float
    volatility: float
    beta: float = None


class RiskManager(LoggerMixin):
    """Comprehensive risk management system."""

    def __init__(
        self,
        max_position_size: float = 0.1,
        max_portfolio_var: float = 0.05,
        max_drawdown: float = 0.2,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.1,
        risk_free_rate: float = 0.02,
        lookback_window: int = 252
    ):
        """
        Initialize risk manager.

        Args:
            max_position_size: Maximum position size as fraction of portfolio
            max_portfolio_var: Maximum portfolio VaR
            max_drawdown: Maximum allowed drawdown
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            risk_free_rate: Risk-free rate for calculations
            lookback_window: Lookback window for risk calculations
        """
        self.max_position_size = max_position_size
        self.max_portfolio_var = max_portfolio_var
        self.max_drawdown = max_drawdown
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.risk_free_rate = risk_free_rate
        self.lookback_window = lookback_window

        # Risk state tracking
        self.portfolio_history = []
        self.risk_metrics_history = []
        self.alerts = []

    def calculate_position_size(
        self,
        signal_strength: float,
        current_price: float,
        portfolio_value: float,
        volatility: float,
        method: PositionSizeMethod = PositionSizeMethod.VOLATILITY,
        **kwargs
    ) -> float:
        """
        Calculate optimal position size based on risk management rules.

        Args:
            signal_strength: Strength of trading signal (0-1)
            current_price: Current asset price
            portfolio_value: Total portfolio value
            volatility: Asset volatility
            method: Position sizing method
            **kwargs: Additional parameters for specific methods

        Returns:
            Position size in currency units
        """
        if method == PositionSizeMethod.FIXED:
            base_size = kwargs.get('fixed_amount', 1000)
            position_size = base_size * signal_strength

        elif method == PositionSizeMethod.PERCENTAGE:
            percentage = kwargs.get('percentage', 0.05)
            position_size = portfolio_value * percentage * signal_strength

        elif method == PositionSizeMethod.VOLATILITY:
            target_volatility = kwargs.get('target_volatility', 0.02)
            # Inverse volatility scaling
            vol_scalar = min(target_volatility / max(volatility, 0.001), 10)  # Cap at 10x
            base_allocation = portfolio_value * self.max_position_size
            position_size = base_allocation * vol_scalar * signal_strength

        elif method == PositionSizeMethod.KELLY:
            position_size = self._kelly_position_size(
                signal_strength,
                portfolio_value,
                volatility,
                **kwargs
            )

        elif method == PositionSizeMethod.RISK_PARITY:
            position_size = self._risk_parity_position_size(
                signal_strength,
                portfolio_value,
                volatility,
                **kwargs
            )

        else:
            raise ValueError(f"Unknown position sizing method: {method}")

        # Apply risk limits
        max_position_value = portfolio_value * self.max_position_size
        position_size = min(position_size, max_position_value)

        # Ensure minimum position size
        min_position = kwargs.get('min_position', 10)
        if position_size < min_position:
            position_size = 0

        self.logger.debug(
            f"Calculated position size: ${position_size:.2f} "
            f"using {method.value} method (signal: {signal_strength:.3f})"
        )

        return position_size

    def _kelly_position_size(
        self,
        signal_strength: float,
        portfolio_value: float,
        volatility: float,
        win_rate: float = 0.55,
        avg_win: float = 0.02,
        avg_loss: float = 0.015,
        **kwargs
    ) -> float:
        """Calculate position size using Kelly criterion."""
        # Kelly formula: f = (bp - q) / b
        # where b = odds received (avg_win/avg_loss), p = win_rate, q = loss_rate

        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate

        kelly_fraction = (b * p - q) / b

        # Apply signal strength and safety margin
        safety_margin = kwargs.get('kelly_safety_margin', 0.25)
        adjusted_kelly = kelly_fraction * signal_strength * safety_margin

        # Cap Kelly fraction to prevent over-leveraging
        max_kelly = kwargs.get('max_kelly_fraction', 0.1)
        adjusted_kelly = min(adjusted_kelly, max_kelly)

        position_size = portfolio_value * max(adjusted_kelly, 0)
        return position_size

    def _risk_parity_position_size(
        self,
        signal_strength: float,
        portfolio_value: float,
        volatility: float,
        target_risk_contribution: float = 0.05,
        **kwargs
    ) -> float:
        """Calculate position size using risk parity approach."""
        # Risk parity: each position contributes equally to portfolio risk
        # Position size = Target Risk Contribution / (Weight * Volatility)

        target_weight = target_risk_contribution / volatility
        position_size = portfolio_value * target_weight * signal_strength

        return position_size

    def check_risk_limits(
        self,
        proposed_position: float,
        current_price: float,
        portfolio_value: float,
        current_positions: Dict[str, float],
        price_data: pd.DataFrame
    ) -> Tuple[bool, List[str]]:
        """
        Check if proposed position violates risk limits.

        Args:
            proposed_position: Proposed position size
            current_price: Current asset price
            portfolio_value: Current portfolio value
            current_positions: Current positions dict
            price_data: Historical price data for risk calculations

        Returns:
            Tuple of (is_allowed, list_of_violations)
        """
        violations = []

        # 1. Position size limit
        position_weight = (proposed_position * current_price) / portfolio_value
        if position_weight > self.max_position_size:
            violations.append(
                f"Position weight {position_weight:.3f} exceeds limit {self.max_position_size:.3f}"
            )

        # 2. Portfolio concentration
        total_position_value = sum(pos * current_price for pos in current_positions.values())
        new_total = total_position_value + (proposed_position * current_price)
        concentration = new_total / portfolio_value

        max_concentration = 0.5  # 50% max in risky assets
        if concentration > max_concentration:
            violations.append(
                f"Portfolio concentration {concentration:.3f} exceeds limit {max_concentration:.3f}"
            )

        # 3. VaR limit
        if len(price_data) > self.lookback_window:
            returns = price_data['close'].pct_change().dropna()
            portfolio_var = self._calculate_var(returns, portfolio_value)

            if portfolio_var > self.max_portfolio_var:
                violations.append(
                    f"Portfolio VaR {portfolio_var:.3f} exceeds limit {self.max_portfolio_var:.3f}"
                )

        # 4. Drawdown limit
        if len(self.portfolio_history) > 1:
            current_dd = self._calculate_current_drawdown()
            if current_dd < -self.max_drawdown:
                violations.append(
                    f"Current drawdown {current_dd:.3f} exceeds limit {self.max_drawdown:.3f}"
                )

        is_allowed = len(violations) == 0
        return is_allowed, violations

    def calculate_stop_loss_take_profit(
        self,
        entry_price: float,
        position_type: str,
        volatility: Optional[float] = None,
        atr: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels.

        Args:
            entry_price: Entry price for position
            position_type: 'long' or 'short'
            volatility: Price volatility for dynamic stops
            atr: Average True Range for dynamic stops

        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        if volatility is not None:
            # Dynamic stops based on volatility
            stop_distance = max(volatility * 2, self.stop_loss_pct)
            profit_distance = max(volatility * 3, self.take_profit_pct)
        elif atr is not None:
            # Dynamic stops based on ATR
            stop_distance = atr / entry_price * 2
            profit_distance = atr / entry_price * 3
        else:
            # Fixed percentage stops
            stop_distance = self.stop_loss_pct
            profit_distance = self.take_profit_pct

        if position_type.lower() == 'long':
            stop_loss = entry_price * (1 - stop_distance)
            take_profit = entry_price * (1 + profit_distance)
        else:  # short
            stop_loss = entry_price * (1 + stop_distance)
            take_profit = entry_price * (1 - profit_distance)

        self.logger.debug(
            f"Calculated stops for {position_type} position: "
            f"SL={stop_loss:.2f}, TP={take_profit:.2f}"
        )

        return stop_loss, take_profit

    def calculate_portfolio_risk_metrics(
        self,
        portfolio_value: float,
        positions: Dict[str, Dict],
        price_data: Dict[str, pd.DataFrame],
        benchmark_returns: Optional[pd.Series] = None
    ) -> RiskMetrics:
        """
        Calculate comprehensive portfolio risk metrics.

        Args:
            portfolio_value: Current portfolio value
            positions: Dictionary of positions with symbol and details
            price_data: Historical price data for each symbol
            benchmark_returns: Benchmark returns for beta calculation

        Returns:
            RiskMetrics object with calculated metrics
        """
        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(positions, price_data)

        if len(portfolio_returns) < 2:
            return RiskMetrics(
                portfolio_value=portfolio_value,
                portfolio_var=0,
                portfolio_cvar=0,
                max_drawdown=0,
                current_drawdown=0,
                sharpe_ratio=0,
                volatility=0
            )

        # Basic risk metrics
        portfolio_var = self._calculate_var(portfolio_returns, portfolio_value)
        portfolio_cvar = self._calculate_cvar(portfolio_returns, portfolio_value)
        volatility = portfolio_returns.std() * np.sqrt(252)

        # Drawdown metrics
        cumulative_returns = (1 + portfolio_returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        current_drawdown = drawdown.iloc[-1]

        # Risk-adjusted returns
        excess_returns = portfolio_returns - self.risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / portfolio_returns.std() * np.sqrt(252)

        # Market risk metrics
        beta = None
        correlation = None
        if benchmark_returns is not None:
            aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner')
            if len(aligned_data) > 1:
                correlation = aligned_data.corr().iloc[0, 1]
                beta = aligned_data.cov().iloc[0, 1] / aligned_data.iloc[:, 1].var()

        risk_metrics = RiskMetrics(
            portfolio_value=portfolio_value,
            portfolio_var=portfolio_var,
            portfolio_cvar=portfolio_cvar,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            sharpe_ratio=sharpe_ratio,
            volatility=volatility,
            beta=beta,
            correlation=correlation
        )

        # Store for tracking
        self.portfolio_history.append(portfolio_value)
        self.risk_metrics_history.append(risk_metrics)

        return risk_metrics

    def _calculate_var(
        self,
        returns: pd.Series,
        portfolio_value: float,
        confidence_level: float = 0.05
    ) -> float:
        """Calculate Value at Risk."""
        if len(returns) < 2:
            return 0

        var_return = returns.quantile(confidence_level)
        var_dollar = abs(var_return * portfolio_value)
        return var_dollar / portfolio_value  # Return as percentage

    def _calculate_cvar(
        self,
        returns: pd.Series,
        portfolio_value: float,
        confidence_level: float = 0.05
    ) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        if len(returns) < 2:
            return 0

        var_threshold = returns.quantile(confidence_level)
        tail_returns = returns[returns <= var_threshold]

        if len(tail_returns) == 0:
            return 0

        cvar_return = tail_returns.mean()
        cvar_dollar = abs(cvar_return * portfolio_value)
        return cvar_dollar / portfolio_value  # Return as percentage

    def _calculate_portfolio_returns(
        self,
        positions: Dict[str, Dict],
        price_data: Dict[str, pd.DataFrame]
    ) -> pd.Series:
        """Calculate portfolio returns from positions and price data."""
        if not positions or not price_data:
            return pd.Series([])

        portfolio_returns = []
        dates = None

        for symbol, position_info in positions.items():
            if symbol in price_data:
                prices = price_data[symbol]['close']
                returns = prices.pct_change().dropna()
                weight = position_info.get('weight', 0)

                weighted_returns = returns * weight

                if dates is None:
                    dates = returns.index
                    portfolio_returns = weighted_returns
                else:
                    # Align dates
                    aligned_data = pd.concat([pd.Series(portfolio_returns, index=dates), weighted_returns],
                                           axis=1, join='inner')
                    portfolio_returns = aligned_data.sum(axis=1)
                    dates = aligned_data.index

        return pd.Series(portfolio_returns, index=dates) if len(portfolio_returns) > 0 else pd.Series([])

    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from portfolio history."""
        if len(self.portfolio_history) < 2:
            return 0

        values = pd.Series(self.portfolio_history)
        peak = values.expanding().max()
        current_drawdown = (values.iloc[-1] - peak.iloc[-1]) / peak.iloc[-1]
        return current_drawdown

    def generate_risk_alerts(
        self,
        risk_metrics: RiskMetrics,
        positions: Dict[str, Dict]
    ) -> List[str]:
        """
        Generate risk alerts based on current portfolio state.

        Args:
            risk_metrics: Current risk metrics
            positions: Current positions

        Returns:
            List of risk alert messages
        """
        alerts = []

        # VaR alert
        if risk_metrics.portfolio_var > self.max_portfolio_var * 0.8:
            alerts.append(
                f"Portfolio VaR ({risk_metrics.portfolio_var:.3f}) approaching limit "
                f"({self.max_portfolio_var:.3f})"
            )

        # Drawdown alert
        if risk_metrics.current_drawdown < -self.max_drawdown * 0.8:
            alerts.append(
                f"Portfolio drawdown ({risk_metrics.current_drawdown:.3f}) approaching limit "
                f"({-self.max_drawdown:.3f})"
            )

        # Concentration alert
        total_risky_weight = sum(pos.get('weight', 0) for pos in positions.values())
        if total_risky_weight > 0.4:
            alerts.append(f"High portfolio concentration: {total_risky_weight:.2%}")

        # Volatility alert
        if risk_metrics.volatility > 0.4:  # 40% annual volatility
            alerts.append(f"High portfolio volatility: {risk_metrics.volatility:.2%}")

        # Sharpe ratio alert
        if risk_metrics.sharpe_ratio < 0:
            alerts.append(f"Negative Sharpe ratio: {risk_metrics.sharpe_ratio:.3f}")

        self.alerts.extend(alerts)
        return alerts

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get summary of current risk state."""
        if not self.risk_metrics_history:
            return {'status': 'No risk data available'}

        latest_metrics = self.risk_metrics_history[-1]

        summary = {
            'portfolio_value': latest_metrics.portfolio_value,
            'var_utilization': latest_metrics.portfolio_var / self.max_portfolio_var,
            'drawdown_utilization': abs(latest_metrics.current_drawdown) / self.max_drawdown,
            'volatility': latest_metrics.volatility,
            'sharpe_ratio': latest_metrics.sharpe_ratio,
            'recent_alerts': self.alerts[-5:] if self.alerts else [],
            'risk_level': self._assess_risk_level(latest_metrics)
        }

        return summary

    def _assess_risk_level(self, metrics: RiskMetrics) -> str:
        """Assess overall portfolio risk level."""
        risk_score = 0

        # VaR component
        risk_score += (metrics.portfolio_var / self.max_portfolio_var) * 0.3

        # Drawdown component
        risk_score += (abs(metrics.current_drawdown) / self.max_drawdown) * 0.3

        # Volatility component
        risk_score += min(metrics.volatility / 0.3, 1) * 0.2

        # Sharpe component (inverse)
        if metrics.sharpe_ratio > 0:
            risk_score += max(0, (2 - metrics.sharpe_ratio) / 2) * 0.2
        else:
            risk_score += 0.2

        if risk_score < 0.3:
            return "LOW"
        elif risk_score < 0.6:
            return "MEDIUM"
        elif risk_score < 0.8:
            return "HIGH"
        else:
            return "CRITICAL"

    def emergency_stop_check(
        self,
        current_drawdown: float,
        portfolio_var: float
    ) -> Tuple[bool, str]:
        """
        Check if emergency stop should be triggered.

        Args:
            current_drawdown: Current portfolio drawdown
            portfolio_var: Current portfolio VaR

        Returns:
            Tuple of (should_stop, reason)
        """
        # Emergency drawdown stop
        emergency_dd_limit = self.max_drawdown * 1.2  # 20% buffer beyond normal limit
        if current_drawdown < -emergency_dd_limit:
            return True, f"Emergency stop: Drawdown {current_drawdown:.3f} exceeds emergency limit"

        # Emergency VaR stop
        emergency_var_limit = self.max_portfolio_var * 2  # 100% above normal limit
        if portfolio_var > emergency_var_limit:
            return True, f"Emergency stop: VaR {portfolio_var:.3f} exceeds emergency limit"

        return False, ""