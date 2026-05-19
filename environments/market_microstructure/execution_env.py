"""
Execution Environment

Advanced execution environment for optimizing order splitting and minimizing slippage.
Simulates realistic order execution with market impact and timing constraints.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from gymnasium import spaces

from environments.base_env import AssetConfig, FinancialTradingBase
from .lob_simulator import LimitOrderBook, Order, OrderSide, OrderType, Trade

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Execution strategies"""

    IMMEDIATE = "immediate"
    TWAP = "twap"  # Time-weighted average price
    VWAP = "vwap"  # Volume-weighted average price
    ADAPTIVE = "adaptive"
    CUSTOM = "custom"


@dataclass
class ExecutionOrder:
    """Order to be executed"""

    symbol: str
    side: OrderSide
    total_quantity: int
    urgency: float  # 0 (low urgency) to 1 (high urgency)
    max_participation_rate: float  # Maximum percentage of market volume
    time_horizon: int  # Number of steps to execute over


class ExecutionEnvironment(FinancialTradingBase):
    """
    Advanced execution environment for optimizing trade execution.

    Features:
    - Order splitting strategies (TWAP, VWAP, adaptive)
    - Market impact modeling
    - Slippage estimation and minimization
    - Real-time execution risk management
    - Performance measurement (implementation shortfall)
    """

    def __init__(
        self,
        assets: List[AssetConfig],
        initial_cash: float = 1_000_000,
        max_episode_length: int = 100,
        lookback_window: int = 20,
        tick_size: float = 0.01,
        market_impact_factor: float = 0.001,
        volume_window: int = 10,
        participation_rate_limits: Tuple[float, float] = (0.01, 0.5),
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize execution environment.

        Args:
            assets: List of assets to trade
            initial_cash: Initial cash balance
            max_episode_length: Maximum episode length
            lookback_window: Lookback window for features
            tick_size: Minimum price increment
            market_impact_factor: Market impact coefficient
            volume_window: Window for volume averaging
            participation_rate_limits: Min/max participation rates
            seed: Random seed
            render_mode: Render mode
        """
        # Execution-specific parameters
        self.tick_size = tick_size
        self.market_impact_factor = market_impact_factor
        self.volume_window = volume_window
        self.participation_rate_limits = participation_rate_limits

        # Execution state
        self.pending_orders = []
        self.execution_history = []
        self.lob_simulators = {}

        # Initialize base environment
        super().__init__(
            assets=assets,
            initial_cash=initial_cash,
            max_episode_length=max_episode_length,
            lookback_window=lookback_window,
            seed=seed,
            render_mode=render_mode,
            data_source="synthetic",  # Use synthetic for LOB simulation
        )

    def _initialize_environment(self):
        """Initialize execution-specific components."""
        # Create LOB simulators for each asset
        for asset in self.assets:
            self.lob_simulators[asset.symbol] = LimitOrderBook(
                tick_size=self.tick_size, market_impact_factor=self.market_impact_factor
            )

        # Action space: [execution_rate, participation_rate, strategy_type]
        # execution_rate: 0-1 (fraction of remaining order to execute)
        # participation_rate: 0-1 (percentage of market volume)
        # strategy_type: 0-4 (immediate, twap, vwap, adaptive, custom)
        self.action_space = spaces.Box(
            low=np.array([0.0, self.participation_rate_limits[0], 0.0]),
            high=np.array([1.0, self.participation_rate_limits[1], 4.0]),
            dtype=np.float32,
        )

        # Observation space: market state + order state + execution metrics
        obs_size = self._calculate_observation_size()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Performance tracking
        self.execution_metrics = {
            "total_slippage": 0.0,
            "implementation_shortfall": 0.0,
            "execution_cost": 0.0,
            "timing_cost": 0.0,
            "market_impact": 0.0,
        }

    def _calculate_observation_size(self) -> int:
        """Calculate observation space size."""
        base_features = self.lookback_window * len(self.assets) * 5  # OHLCV
        lob_features = len(self.assets) * 10  # LOB depth features
        order_features = 20  # Pending order information
        execution_features = 10  # Execution metrics
        portfolio_features = 5  # Portfolio state

        return (
            base_features
            + lob_features
            + order_features
            + execution_features
            + portfolio_features
        )

    def add_execution_order(self, order: ExecutionOrder):
        """
        Add a new order to be executed.

        Args:
            order: Execution order to add
        """
        self.pending_orders.append(
            {
                "order": order,
                "remaining_quantity": order.total_quantity,
                "executed_quantity": 0,
                "average_price": 0.0,
                "start_step": self.current_step,
            }
        )

        logger.info(
            f"Added execution order: {order.side.value} {order.total_quantity} {order.symbol}"
        )

    def _get_observation(self) -> np.ndarray:
        """Get current observation including market state and execution information."""
        obs = []

        # Market data features
        for asset in self.assets:
            if hasattr(self, "market_data") and self.market_data is not None:
                # Get recent market data
                recent_data = self._get_recent_market_data(
                    asset.symbol, self.lookback_window
                )
                obs.extend(recent_data.flatten())

                # LOB features
                lob = self.lob_simulators[asset.symbol]
                lob_state = lob.get_order_book_state()
                lob_features = [
                    lob_state["best_bid"] or 0,
                    lob_state["best_ask"] or 0,
                    lob_state["mid_price"],
                    lob_state["spread"],
                    lob_state["bid_volume"],
                    lob_state["ask_volume"],
                    lob_state["total_orders"],
                    lob_state["last_trade_price"],
                    lob_state["total_volume"],
                    lob_state["total_trades"],
                ]
                obs.extend(lob_features)
            else:
                # Fallback features
                obs.extend([0.0] * (self.lookback_window * 5 + 10))

        # Order state features
        order_features = self._get_order_features()
        obs.extend(order_features)

        # Execution metrics
        exec_metrics = [
            self.execution_metrics["total_slippage"],
            self.execution_metrics["implementation_shortfall"],
            self.execution_metrics["execution_cost"],
            self.execution_metrics["timing_cost"],
            self.execution_metrics["market_impact"],
        ]
        obs.extend(exec_metrics)

        # Portfolio features
        portfolio_features = [
            self.portfolio_value,
            self.cash_balance,
            len(self.pending_orders),
            self.current_step,
            self.max_episode_length,
        ]
        obs.extend(portfolio_features)

        return np.array(obs, dtype=np.float32)

    def _get_recent_market_data(self, symbol: str, window: int) -> np.ndarray:
        """Get recent market data for a symbol."""
        if hasattr(self, "market_data") and self.market_data is not None:
            try:
                # Get last 'window' rows of market data for this symbol
                if symbol in self.market_data.columns.get_level_values(0):
                    symbol_data = self.market_data[symbol]
                    start_idx = max(0, self.current_step - window + 1)
                    end_idx = self.current_step + 1

                    if end_idx > start_idx:
                        recent_data = symbol_data.iloc[start_idx:end_idx]

                        # Pad if necessary
                        if len(recent_data) < window:
                            padding = np.zeros(
                                (window - len(recent_data), len(recent_data.columns))
                            )
                            recent_data = pd.concat(
                                [
                                    pd.DataFrame(padding, columns=recent_data.columns),
                                    recent_data,
                                ]
                            )

                        return recent_data[
                            ["Open", "High", "Low", "Close", "Volume"]
                        ].values.flatten()
            except Exception as e:
                logger.warning(f"Error getting market data for {symbol}: {e}")

        # Return zeros if no data available
        return np.zeros(window * 5)

    def _get_order_features(self) -> List[float]:
        """Get features for pending orders."""
        if not self.pending_orders:
            return [0.0] * 20

        features = []
        max_orders = 4  # Track up to 4 orders

        for i in range(max_orders):
            if i < len(self.pending_orders):
                order_info = self.pending_orders[i]
                order = order_info["order"]

                # Order features
                features.extend(
                    [
                        1.0 if order.side == OrderSide.BUY else -1.0,  # Side
                        order.total_quantity / 10000.0,  # Normalized quantity
                        order.urgency,  # Urgency
                        order.max_participation_rate,  # Max participation
                        order.time_horizon / 100.0,  # Normalized time horizon
                        order_info["remaining_quantity"]
                        / order.total_quantity,  # Progress
                        order_info["executed_quantity"] / order.total_quantity,
                        order_info["average_price"] / 100.0,  # Normalized avg price
                        (self.current_step - order_info["start_step"])
                        / order.time_horizon,  # Time progress
                        1.0,  # Order active
                    ]
                )
            else:
                features.extend([0.0] * 10)

        return features

    def _process_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Process execution action."""
        execution_rate = float(action[0])
        participation_rate = float(action[1])
        strategy_index = int(round(action[2])) % len(ExecutionStrategy)
        strategy = list(ExecutionStrategy)[strategy_index]

        execution_details = {
            "executed_orders": [],
            "total_cost": 0.0,
            "total_shares": 0,
            "strategy": strategy,
        }

        # Execute pending orders
        for order_info in self.pending_orders:
            if order_info["remaining_quantity"] > 0:
                result = self._execute_order_step(
                    order_info,
                    execution_rate,
                    participation_rate,
                    execution_details["strategy"],
                )
                execution_details["executed_orders"].append(result)

        return execution_details

    def _execute_order_step(
        self,
        order_info: Dict,
        execution_rate: float,
        participation_rate: float,
        strategy: ExecutionStrategy,
    ) -> Dict[str, Any]:
        """Execute one step of an order."""
        order = order_info["order"]
        remaining = order_info["remaining_quantity"]

        if remaining <= 0:
            return {"executed_quantity": 0, "execution_price": 0.0, "cost": 0.0}

        # Calculate execution quantity based on strategy
        execute_quantity = self._calculate_execution_quantity(
            order_info, execution_rate, participation_rate, strategy
        )

        execute_quantity = min(execute_quantity, remaining)

        if execute_quantity <= 0:
            return {"executed_quantity": 0, "execution_price": 0.0, "cost": 0.0}

        # Execute in LOB simulator
        lob = self.lob_simulators[order.symbol]
        execution_order = Order(
            order_id=f"exec_{self.current_step}_{order.symbol}",
            side=order.side,
            order_type=OrderType.MARKET,
            quantity=int(execute_quantity),
            trader_id="execution_agent",
        )

        trades = lob.place_order(execution_order)

        # Calculate execution metrics
        total_executed = 0
        total_cost = 0.0
        avg_price = 0.0

        for trade in trades:
            total_executed += trade.quantity
            if order.side == OrderSide.BUY:
                total_cost += trade.price * trade.quantity
            else:
                total_cost += trade.price * trade.quantity

        if total_executed > 0:
            avg_price = total_cost / total_executed

        # Update order info
        order_info["remaining_quantity"] -= total_executed
        order_info["executed_quantity"] += total_executed

        # Update average price
        if order_info["executed_quantity"] > 0:
            order_info["average_price"] = (
                order_info["average_price"]
                * (order_info["executed_quantity"] - total_executed)
                + avg_price * total_executed
            ) / order_info["executed_quantity"]

        # Update portfolio
        self._update_portfolio_after_execution(order, total_executed, avg_price)

        return {
            "executed_quantity": total_executed,
            "execution_price": avg_price,
            "cost": total_cost,
            "trades": trades,
        }

    def _calculate_execution_quantity(
        self,
        order_info: Dict,
        execution_rate: float,
        participation_rate: float,
        strategy: ExecutionStrategy,
    ) -> float:
        """Calculate quantity to execute based on strategy."""
        order = order_info["order"]
        remaining = order_info["remaining_quantity"]
        time_progress = (
            self.current_step - order_info["start_step"]
        ) / order.time_horizon

        if strategy == ExecutionStrategy.IMMEDIATE:
            return remaining
        elif strategy == ExecutionStrategy.TWAP:
            # Time-weighted average price
            target_progress = time_progress
            target_quantity = remaining * max(0, target_progress)
            return target_quantity - (
                order.total_quantity - order_info["remaining_quantity"]
            )
        elif strategy == ExecutionStrategy.VWAP:
            # Volume-weighted average price (simplified)
            # In reality, this would use actual volume forecasts
            return remaining * execution_rate
        elif strategy == ExecutionStrategy.ADAPTIVE:
            # Adaptive execution based on market conditions
            urgency_factor = order.urgency * execution_rate
            return remaining * urgency_factor
        else:  # CUSTOM
            return remaining * execution_rate

    def _update_portfolio_after_execution(
        self, order: ExecutionOrder, quantity: float, price: float
    ):
        """Update portfolio after order execution."""
        total_cost = quantity * price

        if order.side == OrderSide.BUY:
            self.cash_balance -= total_cost
            # Update position
            current_position = self.positions.get(order.symbol, 0)
            self.positions[order.symbol] = current_position + quantity
        else:  # SELL
            self.cash_balance += total_cost
            # Update position
            current_position = self.positions.get(order.symbol, 0)
            self.positions[order.symbol] = current_position - quantity

        # Update portfolio value
        self._update_portfolio_value()

    def _calculate_reward(self, execution_details: Dict[str, Any]) -> float:
        """Calculate reward based on execution performance."""
        if not execution_details["executed_orders"]:
            return -0.001  # Small penalty for inaction

        total_cost = execution_details["total_cost"]
        total_shares = execution_details["total_shares"]

        if total_shares == 0:
            return -0.001

        # Calculate execution quality metrics
        execution_quality = self._calculate_execution_quality(execution_details)

        # Reward components
        quality_reward = execution_quality
        completion_reward = min(
            total_shares / 1000.0, 1.0
        )  # Reward for completing orders
        cost_penalty = (
            -total_cost / self.portfolio_value if self.portfolio_value > 0 else 0
        )

        # Urgency-based reward
        urgency_bonus = 0
        for order_info in self.pending_orders:
            if order_info["remaining_quantity"] == 0:
                urgency_bonus += order_info["order"].urgency

        total_reward = (
            quality_reward + completion_reward + cost_penalty + urgency_bonus * 0.1
        )

        return total_reward

    def _calculate_execution_quality(self, execution_details: Dict[str, Any]) -> float:
        """Calculate execution quality score."""
        # Simplified execution quality metric
        # In reality, this would consider implementation shortfall, timing, etc.

        executed_quantity = sum(
            order["executed_quantity"] for order in execution_details["executed_orders"]
        )

        # Reward for executing without too much market impact
        market_impact_penalty = (
            sum(len(order["trades"]) for order in execution_details["executed_orders"])
            * 0.001
        )

        return executed_quantity / 1000.0 - market_impact_penalty

    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic market data for execution simulation."""
        dates = pd.date_range(
            start="2023-01-01", periods=self.max_episode_length + 100, freq="1min"
        )

        data = {}
        for asset in self.assets:
            # Generate realistic price series
            np.random.seed(hash(asset.symbol) % 2**32)
            returns = np.random.normal(0, 0.001, len(dates))
            prices = [asset.initial_price]

            for ret in returns:
                prices.append(prices[-1] * (1 + ret))

            prices = prices[1:]

            # Generate OHLCV data
            high_low_range = prices * np.random.uniform(0.001, 0.005, len(prices))
            opens = prices + np.random.normal(0, 0.0001, len(prices))
            highs = prices + high_low_range
            lows = prices - high_low_range
            closes = prices
            volumes = np.random.lognormal(10, 1, len(prices))

            # Create DataFrame for this asset
            asset_data = pd.DataFrame(
                {
                    "Open": opens,
                    "High": highs,
                    "Low": lows,
                    "Close": closes,
                    "Volume": volumes.astype(int),
                },
                index=dates,
            )

            # Add to multi-index DataFrame
            for col in asset_data.columns:
                data[(asset.symbol, col)] = asset_data[col]

        return pd.DataFrame(data)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        # Execute action
        execution_details = self._process_action(action)

        # Calculate reward
        reward = self._calculate_reward(execution_details)

        # Update market data
        self.current_step += 1
        self._update_market_data()

        # Update LOB simulators with new market data
        self._update_lob_simulators()

        # Add some liquidity to keep markets active
        if self.current_step % 10 == 0:
            for lob in self.lob_simulators.values():
                lob.add_liquidity(num_orders=5, price_range=0.2)

        # Check termination
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_episode_length

        # Get observation
        obs = self._get_observation()

        # Get info
        info = self._get_info()
        info.update(
            {
                "execution_details": execution_details,
                "pending_orders": len(self.pending_orders),
                "completed_orders": sum(
                    1 for o in self.pending_orders if o["remaining_quantity"] == 0
                ),
            }
        )

        # Remove completed orders
        self.pending_orders = [
            o for o in self.pending_orders if o["remaining_quantity"] > 0
        ]

        return obs, reward, terminated, truncated, info

    def _update_lob_simulators(self):
        """Update LOB simulators with current market data."""
        for asset in self.assets:
            if hasattr(self, "market_data") and self.market_data is not None:
                try:
                    # Get current price for this asset
                    if asset.symbol in self.market_data.columns.get_level_values(0):
                        current_data = self.market_data[asset.symbol]
                        if self.current_step < len(current_data):
                            current_price = current_data.iloc[self.current_step][
                                "Close"
                            ]
                            lob = self.lob_simulators[asset.symbol]

                            # Update mid price
                            lob.mid_price = current_price
                            lob.last_trade_price = current_price
                except Exception as e:
                    logger.warning(f"Error updating LOB for {asset.symbol}: {e}")

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of execution performance."""
        completed_orders = []
        pending_orders = []

        for order_info in self.pending_orders:
            order = order_info["order"]
            order_summary = {
                "symbol": order.symbol,
                "side": order.side.value,
                "total_quantity": order.total_quantity,
                "executed_quantity": order_info["executed_quantity"],
                "remaining_quantity": order_info["remaining_quantity"],
                "average_price": order_info["average_price"],
                "completion_rate": order_info["executed_quantity"]
                / order.total_quantity,
                "urgency": order.urgency,
            }

            if order_info["remaining_quantity"] == 0:
                completed_orders.append(order_summary)
            else:
                pending_orders.append(order_summary)

        return {
            "completed_orders": completed_orders,
            "pending_orders": pending_orders,
            "execution_metrics": self.execution_metrics.copy(),
            "portfolio_value": self.portfolio_value,
            "total_cash": self.cash_balance,
            "positions": self.positions.copy(),
        }
