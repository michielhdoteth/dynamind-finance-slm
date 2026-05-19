"""
Risk Manager for RL Financial Markets Gym

Central risk management system that integrates CVaR, position limits,
and portfolio risk constraints for safe reinforcement learning.
"""

import numpy as np
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from environments.base_env import AssetConfig, TransactionCosts
from .cvar_reward_shaper import CVaRConfig, CVaRRewardShaper, RiskMeasure

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classifications"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PositionLimits:
    """Position limits for risk management"""

    max_position_size: float = 1.0  # Maximum position as percentage of portfolio
    max_sector_exposure: float = 0.3  # Maximum sector exposure
    max_single_asset: float = 0.2  # Maximum single asset position
    min_diversification: int = 3  # Minimum number of different assets


@dataclass
class PortfolioConstraints:
    """Portfolio-level risk constraints"""

    max_leverage: float = 2.0  # Maximum leverage
    max_drawdown_limit: float = 0.15  # Maximum allowed drawdown
    min_liquidity_ratio: float = 0.05  # Minimum cash/liquid assets ratio
    var_limit: float = 0.05  # Daily VaR limit (5%)
    concentration_limit: float = 0.4  # Maximum concentration in single position


@dataclass
class RiskAlert:
    """Risk alert for constraint violations"""

    alert_type: str
    severity: RiskLevel
    current_value: float
    limit_value: float
    description: str
    timestamp: int
    asset: str = ""


class RiskManager:
    """
    Central risk management system for the RL Financial Markets Gym.

    Features:
    - Position size and leverage limits
    - Portfolio risk constraint enforcement
    - CVaR and VaR monitoring
    - Risk alerts and early warning system
    - Dynamic risk limit adjustment
    - Comprehensive risk reporting
    """

    def __init__(
        self,
        assets: List[AssetConfig],
        position_limits: "PositionLimits" = None,
        portfolio_constraints: PortfolioConstraints = None,
        cvar_config: CVaRConfig = None,
        enable_risk_shaping: bool = True,
        enable_dynamic_limits: bool = True,
    ):
        """
        Initialize risk manager.

        Args:
            assets: List of available assets
            position_limits: Position limit configuration
            portfolio_constraints: Portfolio constraint configuration
            cvar_config: CVaR configuration
            enable_risk_shaping: Enable risk-based reward shaping
            enable_dynamic_limits: Enable dynamic risk limit adjustment
        """
        self.assets = assets
        self.position_limits = position_limits or risk.PositionLimits()
        self.portfolio_constraints = portfolio_constraints or PortfolioConstraints()
        self.enable_risk_shaping = enable_risk_shaping
        self.enable_dynamic_limits = enable_dynamic_limits

        # Initialize CVaR reward shaper
        self.cvar_shaper = (
            CVaRRewardShaper(config=cvar_config, enable_risk_budget=True)
            if enable_risk_shaping
            else None
        )

        # Risk monitoring
        self.risk_alerts = deque(maxlen=1000)
        self.constraint_violations = defaultdict(list)
        self.risk_history = deque(maxlen=1000)

        # Current state
        self.current_positions = {asset.symbol: 0.0 for asset in assets}
        self.current_portfolio_value = 0.0
        self.current_cash = 0.0
        self.step_count = 0

        # Dynamic risk limits
        self.base_limits = {
            "max_position_size": self.position_limits.max_position_size,
            "max_leverage": self.portfolio_constraints.max_leverage,
            "var_limit": self.portfolio_constraints.var_limit,
        }
        self.dynamic_limits = self.base_limits.copy()

        # Risk metrics tracking
        self.current_risk_level = RiskLevel.LOW
        self.risk_utilization = {
            "position_utilization": 0.0,
            "leverage_utilization": 0.0,
            "var_utilization": 0.0,
        }

        logger.info(f"Risk Manager initialized for {len(assets)} assets")

    def check_position_constraints(
        self, proposed_positions: Dict[str, float], current_prices: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Check if proposed positions violate position constraints.

        Args:
            proposed_positions: Proposed positions
            current_prices: Current asset prices

        Returns:
            Dictionary with constraint check results
        """
        results = {
            "is_violation": False,
            "violations": [],
            "utilization": {},
            "adjusted_positions": proposed_positions.copy(),
        }

        total_portfolio_value = self.current_portfolio_value
        if total_portfolio_value == 0:
            return results

        # Check individual position limits
        for symbol, position in proposed_positions.items():
            if symbol not in current_prices:
                continue

            position_value = abs(position) * current_prices[symbol]
            position_ratio = position_value / total_portfolio_value

            # Maximum position size
            if position_ratio > self.position_limits.max_position_size:
                results["is_violation"] = True
                results["violations"].append(
                    {
                        "type": "max_position_size",
                        "symbol": symbol,
                        "current": position_ratio,
                        "limit": self.position_limits.max_position_size,
                        "description": f"Position {symbol} exceeds maximum size limit",
                    }
                )
                # Adjust position
                max_position = (
                    self.position_limits.max_position_size
                    * total_portfolio_value
                    / current_prices[symbol]
                )
                results["adjusted_positions"][symbol] = np.sign(position) * min(
                    abs(position), max_position
                )

            # Single asset limit
            if position_ratio > self.position_limits.max_single_asset:
                results["is_violation"] = True
                results["violations"].append(
                    {
                        "type": "max_single_asset",
                        "symbol": symbol,
                        "current": position_ratio,
                        "limit": self.position_limits.max_single_asset,
                        "description": f"Position {symbol} exceeds single asset limit",
                    }
                )
                # Adjust position
                max_single_asset = (
                    self.position_limits.max_single_asset
                    * total_portfolio_value
                    / current_prices[symbol]
                )
                results["adjusted_positions"][symbol] = np.sign(position) * min(
                    abs(position), max_single_asset
                )

            results["utilization"][f"position_{symbol}"] = (
                position_ratio / self.position_limits.max_position_size
            )

        # Check sector concentration
        sector_exposure = defaultdict(float)
        for symbol, position in proposed_positions.items():
            if symbol in current_prices:
                # Find asset sector (assuming it's in AssetConfig)
                for asset in self.assets:
                    if asset.symbol == symbol:
                        position_value = abs(position) * current_prices[symbol]
                        sector_exposure[asset.sector] += (
                            position_value / total_portfolio_value
                        )

        for sector, exposure in sector_exposure.items():
            if exposure > self.position_limits.max_sector_exposure:
                results["is_violation"] = True
                results["violations"].append(
                    {
                        "type": "max_sector_exposure",
                        "sector": sector,
                        "current": exposure,
                        "limit": self.position_limits.max_sector_exposure,
                        "description": f"Sector {sector} exposure exceeds limit",
                    }
                )

            results["utilization"][f"sector_{sector}"] = (
                exposure / self.position_limits.max_sector_exposure
            )

        # Check minimum diversification
        non_zero_positions = sum(
            1 for pos in proposed_positions.values() if abs(pos) > 0
        )
        if non_zero_positions < self.position_limits.min_diversification:
            results["is_violation"] = True
            results["violations"].append(
                {
                    "type": "min_diversification",
                    "current": non_zero_positions,
                    "limit": self.position_limits.min_diversification,
                    "description": "Portfolio has insufficient diversification",
                }
            )

        results["utilization"]["diversification"] = non_zero_positions / max(
            self.position_limits.min_diversification, 1
        )

        return results

    def check_portfolio_constraints(
        self,
        portfolio_value: float,
        cash_balance: float,
        positions: Dict[str, float],
        current_prices: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Check portfolio-level constraints.

        Args:
            portfolio_value: Total portfolio value
            cash_balance: Current cash balance
            positions: Current positions
            current_prices: Current asset prices

        Returns:
            Dictionary with constraint check results
        """
        results = {"is_violation": False, "violations": [], "utilization": {}}

        if portfolio_value == 0:
            return results

        # Calculate leverage
        total_position_value = sum(
            abs(positions.get(symbol, 0)) * current_prices.get(symbol, 1.0)
            for symbol in positions
        )
        leverage = total_position_value / portfolio_value

        if leverage > self.portfolio_constraints.max_leverage:
            results["is_violation"] = True
            results["violations"].append(
                {
                    "type": "max_leverage",
                    "current": leverage,
                    "limit": self.portfolio_constraints.max_leverage,
                    "description": "Portfolio leverage exceeds limit",
                }
            )

        results["utilization"]["leverage"] = (
            leverage / self.portfolio_constraints.max_leverage
        )

        # Check cash/liquidity ratio
        liquidity_ratio = cash_balance / portfolio_value
        if liquidity_ratio < self.portfolio_constraints.min_liquidity_ratio:
            results["is_violation"] = True
            results["violations"].append(
                {
                    "type": "min_liquidity_ratio",
                    "current": liquidity_ratio,
                    "limit": self.portfolio_constraints.min_liquidity_ratio,
                    "description": "Cash ratio below minimum requirement",
                }
            )

        results["utilization"]["liquidity"] = max(
            0,
            (1.0 - liquidity_ratio)
            / (1.0 - self.portfolio_constraints.min_liquidity_ratio),
        )

        # Check concentration
        position_values = [
            abs(positions.get(symbol, 0)) * current_prices.get(symbol, 1.0)
            for symbol in positions
        ]
        if position_values and portfolio_value > 0:
            max_concentration = max(position_values) / portfolio_value
            if max_concentration > self.portfolio_constraints.concentration_limit:
                results["is_violation"] = True
                results["violations"].append(
                    {
                        "type": "concentration_limit",
                        "current": max_concentration,
                        "limit": self.portfolio_constraints.concentration_limit,
                        "description": "Portfolio concentration exceeds limit",
                    }
                )

            results["utilization"]["concentration"] = (
                max_concentration / self.portfolio_constraints.concentration_limit
            )

        return results

    def update_risk_state(
        self,
        portfolio_value: float,
        cash_balance: float,
        positions: Dict[str, float],
        current_prices: Dict[str, float],
        step: int,
    ):
        """
        Update current risk state.

        Args:
            portfolio_value: Total portfolio value
            cash_balance: Current cash balance
            positions: Current positions
            current_prices: Current asset prices
            step: Current step
        """
        self.current_portfolio_value = portfolio_value
        self.current_cash = cash_balance
        self.current_positions = positions.copy()
        self.step_count = step

        # Update CVaR reward shaper
        if self.cvar_shaper:
            additional_info = {
                "portfolio_value": portfolio_value,
                "positions": positions,
                "prices": current_prices,
            }
            self.cvar_shaper.shape_reward(0.0, step, additional_info)

        # Update risk utilization
        position_results = self.check_position_constraints(positions, current_prices)
        portfolio_results = self.check_portfolio_constraints(
            portfolio_value, cash_balance, positions, current_prices
        )

        self.risk_utilization.update(position_results["utilization"])
        self.risk_utilization.update(portfolio_results["utilization"])

        # Determine risk level
        max_utilization = (
            max(self.risk_utilization.values()) if self.risk_utilization else 0.0
        )
        if max_utilization > 1.0:
            self.current_risk_level = RiskLevel.CRITICAL
        elif max_utilization > 0.8:
            self.current_risk_level = RiskLevel.HIGH
        elif max_utilization > 0.5:
            self.current_risk_level = RiskLevel.MEDIUM
        else:
            self.current_risk_level = RiskLevel.LOW

        # Store risk history
        self.risk_history.append(
            {
                "step": step,
                "risk_level": self.current_risk_level.value,
                "utilization": self.risk_utilization.copy(),
                "portfolio_value": portfolio_value,
                "cash_ratio": cash_balance / portfolio_value
                if portfolio_value > 0
                else 0,
            }
        )

    def shape_reward(
        self, base_reward: float, step: int, additional_info: Dict[str, Any] = None
    ) -> float:
        """
        Shape reward based on risk metrics.

        Args:
            base_reward: Original reward
            step: Current step
            additional_info: Additional information

        Returns:
            Risk-adjusted reward
        """
        if self.cvar_shaper:
            return self.cvar_shaper.shape_reward(base_reward, step, additional_info)
        else:
            return base_reward

    def can_execute_trade(
        self,
        symbol: str,
        quantity: float,
        action_type: str,
        current_prices: Dict[str, float],
    ) -> Tuple[bool, str]:
        """
        Check if a trade can be executed based on risk constraints.

        Args:
            symbol: Asset symbol
            quantity: Trade quantity
            action_type: 'buy' or 'sell'
            current_prices: Current prices

        Returns:
            Tuple of (can_execute, reason)
        """
        if symbol not in current_prices:
            return False, f"Asset {symbol} not found in current prices"

        current_position = self.current_positions.get(symbol, 0.0)
        proposed_position = (
            current_position + quantity
            if action_type == "buy"
            else current_position - quantity
        )

        # Quick check for position limits
        position_value = abs(proposed_position) * current_prices[symbol]
        if self.current_portfolio_value > 0:
            position_ratio = position_value / self.current_portfolio_value

            if position_ratio > self.position_limits.max_single_asset:
                return False, "Position size exceeds single asset limit"

            if position_ratio > self.position_limits.max_position_size:
                return False, "Position size exceeds maximum position limit"

        # More detailed check using full constraint system
        proposed_positions = self.current_positions.copy()
        proposed_positions[symbol] = proposed_position

        position_results = self.check_position_constraints(
            proposed_positions, current_prices
        )
        if position_results["is_violation"]:
            return (
                False,
                f"Trade violates position constraints: {position_results['violations'][0]['description']}",
            )

        return True, "Trade approved"

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        summary = {
            "current_risk_level": self.current_risk_level.value,
            "risk_utilization": self.risk_utilization.copy(),
            "portfolio_value": self.current_portfolio_value,
            "cash_balance": self.current_cash,
            "positions": self.current_positions.copy(),
            "step_count": self.step_count,
            "recent_alerts": list(self.risk_alerts)[-10:],
            "total_alerts": len(self.risk_alerts),
            "constraint_violations": dict(self.constraint_violations),
            "dynamic_limits": self.dynamic_limits.copy(),
            "base_limits": self.base_limits.copy(),
        }

        # Add CVaR summary if available
        if self.cvar_shaper:
            cvar_summary = self.cvar_shaper.get_risk_summary()
            summary["cvar_metrics"] = cvar_summary

        return summary

    def adjust_dynamic_limits(self, market_volatility: float):
        """
        Adjust risk limits dynamically based on market conditions.

        Args:
            market_volatility: Current market volatility measure
        """
        if not self.enable_dynamic_limits:
            return

        # Increase risk limits in low volatility, decrease in high volatility
        volatility_factor = max(0.5, min(2.0, 1.0 / market_volatility))

        for limit_type in self.dynamic_limits:
            if limit_type in self.base_limits:
                self.dynamic_limits[limit_type] = (
                    self.base_limits[limit_type] * volatility_factor
                )

        logger.info(
            f"Dynamic limits adjusted by factor {volatility_factor:.2f} based on market volatility"
        )

    def reset(self):
        """Reset risk manager state."""
        self.current_positions = {asset.symbol: 0.0 for asset in self.assets}
        self.current_portfolio_value = 0.0
        self.current_cash = 0.0
        self.step_count = 0
        self.current_risk_level = RiskLevel.LOW
        self.risk_utilization = {key: 0.0 for key in self.risk_utilization}

        if self.cvar_shaper:
            self.cvar_shaper.reset()

        logger.info("Risk manager state reset")

    def get_risk_metrics_for_observation(self) -> np.ndarray:
        """Get risk metrics for inclusion in RL observation."""
        metrics = [
            float(self.current_risk_level.value == "critical"),
            float(self.current_risk_level.value == "high"),
            float(self.current_risk_level.value == "medium"),
            float(self.current_risk_level.value == "low"),
            self.risk_utilization.get("leverage", 0.0),
            self.risk_utilization.get("liquidity", 0.0),
            self.risk_utilization.get("concentration", 0.0),
            max(self.risk_utilization.values()) if self.risk_utilization else 0.0,
            len([v for v in self.constraint_violations.values() if v]),
            self.current_cash / max(self.current_portfolio_value, 1.0)
            if self.current_portfolio_value > 0
            else 0.0,
            sum(abs(p) for p in self.current_positions.values())
            / max(self.current_portfolio_value, 1.0)
            if self.current_portfolio_value > 0
            else 0.0,
        ]

        return np.array(metrics, dtype=np.float32)
