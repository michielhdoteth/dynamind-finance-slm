"""
Market Making Environment

Advanced market-making environment with inventory management,
risk controls, and profitability optimization.
Simulates realistic market-making dynamics and constraints.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from gymnasium import spaces

from environments.base_env import AssetConfig, FinancialTradingBase, RiskConstraints
from .lob_simulator import LimitOrderBook, Order, OrderSide, OrderType, Trade

logger = logging.getLogger(__name__)


class MarketMakingStrategy(Enum):
    """Market-making strategies"""

    FIXED_SPREAD = "fixed_spread"
    ADAPTIVE_SPREAD = "adaptive_spread"
    INVENTORY_AWARE = "inventory_aware"
    VOLATILITY_AWARE = "volatility_aware"
    HYBRID = "hybrid"


@dataclass
class MarketMakingConfig:
    """Configuration for market-making parameters"""

    base_spread: float = 0.02  # Base bid-ask spread
    max_position: int = 1000  # Maximum inventory position
    inventory_penalty: float = 0.001  # Penalty for inventory imbalance
    target_inventory: int = 0  # Target inventory level
    min_spread: float = 0.01  # Minimum spread
    max_spread: float = 0.1  # Maximum spread
    quote_size: int = 100  # Size of quote orders
    max_quotes: int = 5  # Maximum number of quote levels
    volatility_window: int = 20  # Window for volatility calculation
    rebalance_frequency: int = 1  # Frequency of rebalancing quotes


class MarketMakingEnvironment(FinancialTradingBase):
    """
    Advanced market-making environment for RL training.

    Features:
    - Dynamic quote management with multiple levels
    - Inventory-aware risk management
    - Adaptive spread based on volatility and inventory
    - Real-time P&L tracking and risk metrics
    - Market impact and execution cost modeling
    """

    def __init__(
        self,
        assets: List[AssetConfig],
        initial_cash: float = 1_000_000,
        max_episode_length: int = 1000,
        lookback_window: int = 50,
        config: MarketMakingConfig = None,
        enable_risk_management: bool = True,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize market-making environment.

        Args:
            assets: List of assets to make markets in
            initial_cash: Initial cash balance
            max_episode_length: Maximum episode length
            lookback_window: Lookback window for features
            config: Market-making configuration
            enable_risk_management: Enable risk management constraints
            seed: Random seed
            render_mode: Render mode
        """
        # Market-making specific parameters
        self.config = config or MarketMakingConfig()
        self.enable_risk_management = enable_risk_management

        # Market-making state
        self.active_quotes = {}  # Active limit orders
        self.inventory = {asset.symbol: 0 for asset in assets}
        self.quote_history = []
        self.trade_history = []

        # Performance tracking
        self.total_pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.inventory_cost = 0.0
        self.execution_cost = 0.0

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
        """Initialize market-making specific components."""
        # Create LOB simulators for each asset
        self.lob_simulators = {}
        for asset in self.assets:
            self.lob_simulators[asset.symbol] = LimitOrderBook(
                tick_size=0.01,
                market_impact_factor=0.0001,
                base_spread=self.config.base_spread,
            )
            # Initialize active quotes for this asset
            self.active_quotes[asset.symbol] = {
                "bid_orders": [],  # List of (price, quantity, order_id)
                "ask_orders": [],  # List of (price, quantity, order_id)
            }

        # Action space: [spread_adjustment, inventory_adjustment, quote_size_adjustment, strategy_type]
        # spread_adjustment: -1 to 1 (adjust spread)
        # inventory_adjustment: -1 to 1 (adjust target inventory)
        # quote_size_adjustment: -1 to 1 (adjust quote size)
        # strategy_type: 0-4 (fixed, adaptive, inventory, volatility, hybrid)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 4.0]),
            dtype=np.float32,
        )

        # Observation space: market state + inventory + quotes + performance
        obs_size = self._calculate_observation_size()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        logger.info(
            f"Market Making Environment initialized with {len(self.assets)} assets"
        )

    def _calculate_observation_size(self) -> int:
        """Calculate observation space size."""
        market_features = self.lookback_window * len(self.assets) * 5  # OHLCV
        lob_features = len(self.assets) * 10  # LOB state
        inventory_features = len(self.assets) * 5  # Inventory levels and metrics
        quote_features = len(self.assets) * 10  # Active quotes
        performance_features = 10  # P&L and risk metrics
        risk_features = 10  # Risk metrics

        return (
            market_features
            + lob_features
            + inventory_features
            + quote_features
            + performance_features
            + risk_features
        )

    def _get_observation(self) -> np.ndarray:
        """Get current observation including market state and inventory."""
        obs = []

        # Market data features
        for asset in self.assets:
            if hasattr(self, "market_data") and self.market_data is not None:
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
                obs.extend([0.0] * (self.lookback_window * 5 + 10))

        # Inventory features
        for asset in self.assets:
            current_inventory = self.inventory[asset.symbol]
            mid_price = self.lob_simulators[asset.symbol].get_mid_price() or 100.0
            inventory_value = current_inventory * mid_price

            inventory_features = [
                current_inventory / self.config.max_position,  # Normalized inventory
                inventory_value / self.portfolio_value
                if self.portfolio_value > 0
                else 0,  # Inventory value ratio
                self._calculate_inventory_cost(asset.symbol),  # Inventory cost
                self._calculate_inventory_risk(asset.symbol),  # Inventory risk
                current_inventory / self.config.quote_size,  # Inventory in quote units
            ]
            obs.extend(inventory_features)

        # Quote features
        for asset in self.assets:
            quote_features = self._get_quote_features(asset.symbol)
            obs.extend(quote_features)

        # Performance features
        performance_metrics = [
            self.total_pnl / self.portfolio_value if self.portfolio_value > 0 else 0,
            self.realized_pnl / self.portfolio_value if self.portfolio_value > 0 else 0,
            self.unrealized_pnl / self.portfolio_value
            if self.portfolio_value > 0
            else 0,
            self.inventory_cost / self.portfolio_value
            if self.portfolio_value > 0
            else 0,
            self.execution_cost / self.portfolio_value
            if self.portfolio_value > 0
            else 0,
            self._calculate_sharpe_ratio(),
            self._calculate_max_drawdown(),
            self._calculate_order_flow_balance(),
            len(self.quote_history) / 100.0,  # Quote activity
            self.current_step / self.max_episode_length,  # Time progress
            self._calculate_risk_utilization(),
        ]
        obs.extend(performance_metrics)

        # Risk features
        risk_metrics = self._calculate_risk_metrics()
        obs.extend(risk_metrics)

        return np.array(obs, dtype=np.float32)

    def _get_recent_market_data(self, symbol: str, window: int) -> np.ndarray:
        """Get recent market data for a symbol."""
        if hasattr(self, "market_data") and self.market_data is not None:
            try:
                if symbol in self.market_data.columns.get_level_values(0):
                    symbol_data = self.market_data[symbol]
                    start_idx = max(0, self.current_step - window + 1)
                    end_idx = self.current_step + 1

                    if end_idx > start_idx:
                        recent_data = symbol_data.iloc[start_idx:end_idx]
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

        return np.zeros(window * 5)

    def _get_quote_features(self, symbol: str) -> List[float]:
        """Get features for active quotes."""
        quotes = self.active_quotes[symbol]
        features = []

        # Bid side features
        bid_levels = min(len(quotes["bid_orders"]), self.config.max_quotes)
        for i in range(self.config.max_quotes):
            if i < bid_levels:
                price, qty, order_id = quotes["bid_orders"][i]
                features.extend(
                    [
                        price / 100.0,  # Normalized price
                        qty / self.config.quote_size,  # Normalized quantity
                        1.0,  # Active
                    ]
                )
            else:
                features.extend([0.0, 0.0, 0.0])

        # Ask side features
        ask_levels = min(len(quotes["ask_orders"]), self.config.max_quotes)
        for i in range(self.config.max_quotes):
            if i < ask_levels:
                price, qty, order_id = quotes["ask_orders"][i]
                features.extend(
                    [
                        price / 100.0,  # Normalized price
                        qty / self.config.quote_size,  # Normalized quantity
                        1.0,  # Active
                    ]
                )
            else:
                features.extend([0.0, 0.0, 0.0])

        return features

    def _process_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Process market-making action."""
        spread_adjustment = float(action[0])
        inventory_adjustment = float(action[1])
        quote_size_adjustment = float(action[2])
        strategy_index = int(round(action[3])) % len(MarketMakingStrategy)
        strategy = list(MarketMakingStrategy)[strategy_index]

        execution_details = {
            "strategy": strategy,
            "spread_adjustment": spread_adjustment,
            "inventory_adjustment": inventory_adjustment,
            "quote_size_adjustment": quote_size_adjustment,
            "quotes_placed": 0,
            "quotes_cancelled": 0,
            "trades_executed": [],
        }

        # Update quotes based on action
        self._update_quotes(execution_details)

        return execution_details

    def _update_quotes(self, execution_details: Dict[str, Any]):
        """Update market-making quotes based on current strategy."""
        for asset in self.assets:
            lob = self.lob_simulators[asset.symbol]
            mid_price = lob.get_mid_price() or 100.0

            # Calculate optimal spread
            optimal_spread = self._calculate_optimal_spread(
                asset.symbol,
                execution_details["strategy"],
                execution_details["spread_adjustment"],
                execution_details["inventory_adjustment"],
            )

            # Calculate quote sizes
            quote_size = self._calculate_optimal_quote_size(
                asset.symbol, execution_details["quote_size_adjustment"]
            )

            # Calculate inventory-aware skew
            inventory_skew = self._calculate_inventory_skew(asset.symbol)

            # Cancel existing quotes
            self._cancel_all_quotes(asset.symbol)
            execution_details["quotes_cancelled"] += len(
                self.active_quotes[asset.symbol]["bid_orders"]
            ) + len(self.active_quotes[asset.symbol]["ask_orders"])

            # Place new quotes
            num_levels = min(3, self.config.max_quotes)  # Use up to 3 levels
            for level in range(num_levels):
                # Calculate bid and ask prices for this level
                level_offset = optimal_spread * (level + 1) / 2

                bid_price = mid_price - level_offset - inventory_skew
                ask_price = mid_price + level_offset - inventory_skew

                # Round to tick size
                bid_price = round(bid_price / 0.01) * 0.01
                ask_price = round(ask_price / 0.01) * 0.01

                # Place bid order
                if self._can_place_bid(asset.symbol, quote_size):
                    bid_order = Order(
                        order_id=f"bid_{self.current_step}_{asset.symbol}_{level}",
                        side=OrderSide.BUY,
                        order_type=OrderType.LIMIT,
                        quantity=quote_size,
                        price=bid_price,
                        trader_id="market_maker",
                    )
                    trades = lob.place_order(bid_order)
                    if trades:  # Order got filled immediately
                        self._handle_trades(trades, asset.symbol)
                    else:
                        self.active_quotes[asset.symbol]["bid_orders"].append(
                            (bid_price, quote_size, bid_order.order_id)
                        )
                    execution_details["quotes_placed"] += 1

                # Place ask order
                if self._can_place_ask(asset.symbol, quote_size):
                    ask_order = Order(
                        order_id=f"ask_{self.current_step}_{asset.symbol}_{level}",
                        side=OrderSide.SELL,
                        order_type=OrderType.LIMIT,
                        quantity=quote_size,
                        price=ask_price,
                        trader_id="market_maker",
                    )
                    trades = lob.place_order(ask_order)
                    if trades:  # Order got filled immediately
                        self._handle_trades(trades, asset.symbol)
                    else:
                        self.active_quotes[asset.symbol]["ask_orders"].append(
                            (ask_price, quote_size, ask_order.order_id)
                        )
                    execution_details["quotes_placed"] += 1

    def _calculate_optimal_spread(
        self,
        symbol: str,
        strategy: MarketMakingStrategy,
        spread_adjustment: float,
        inventory_adjustment: float,
    ) -> float:
        """Calculate optimal bid-ask spread."""
        base_spread = self.config.base_spread

        if strategy == MarketMakingStrategy.FIXED_SPREAD:
            return base_spread

        # Calculate volatility
        volatility = self._calculate_volatility(symbol)
        volatility_component = volatility * 2  # Scale volatility impact

        # Calculate inventory component
        inventory_ratio = self.inventory[symbol] / self.config.max_position
        inventory_component = abs(inventory_ratio) * self.config.inventory_penalty

        # Calculate order flow imbalance component
        flow_imbalance = self._calculate_order_flow_imbalance(symbol)
        flow_component = abs(flow_imbalance) * 0.01

        # Combine components
        optimal_spread = (
            base_spread + volatility_component + inventory_component + flow_component
        )

        # Apply adjustments
        optimal_spread *= 1 + spread_adjustment * 0.5  # Adjust spread by ±25%

        # Apply inventory adjustment
        if inventory_adjustment > 0:  # Want to reduce inventory
            optimal_spread *= 1 - inventory_adjustment * 0.2  # Tighten spread
        else:  # Want to maintain or increase inventory
            optimal_spread *= 1 + abs(inventory_adjustment) * 0.2  # Widen spread

        # Ensure spread is within bounds
        optimal_spread = max(
            self.config.min_spread, min(self.config.max_spread, optimal_spread)
        )

        return optimal_spread

    def _calculate_optimal_quote_size(self, symbol: str, size_adjustment: float) -> int:
        """Calculate optimal quote size."""
        base_size = self.config.quote_size

        # Adjust based on inventory
        inventory_ratio = self.inventory[symbol] / self.config.max_position
        if abs(inventory_ratio) > 0.5:  # High inventory
            size_adjustment *= 0.5  # Reduce size when inventory is high

        # Adjust based on volatility
        volatility = self._calculate_volatility(symbol)
        if volatility > 0.02:  # High volatility
            size_adjustment *= 0.7  # Reduce size in volatile markets

        optimal_size = int(base_size * (1 + size_adjustment * 0.5))
        optimal_size = max(10, min(self.config.quote_size * 2, optimal_size))

        return optimal_size

    def _calculate_inventory_skew(self, symbol: str) -> float:
        """Calculate inventory skew for quote pricing."""
        inventory_ratio = self.inventory[symbol] / self.config.max_position

        # Skew quotes to manage inventory
        # Positive inventory -> lower bid prices, higher ask prices
        # Negative inventory -> higher bid prices, lower ask prices
        skew = inventory_ratio * 0.005  # Scale skew effect

        return skew

    def _calculate_volatility(self, symbol: str) -> float:
        """Calculate recent volatility."""
        if hasattr(self, "market_data") and self.market_data is not None:
            try:
                if symbol in self.market_data.columns.get_level_values(0):
                    recent_data = self.market_data[symbol]
                    if self.current_step >= self.config.volatility_window:
                        window_data = recent_data.iloc[
                            self.current_step
                            - self.config.volatility_window : self.current_step
                        ]
                        returns = window_data["Close"].pct_change().dropna()
                        if len(returns) > 0:
                            return returns.std()
            except Exception:
                pass

        return 0.01  # Default volatility

    def _calculate_order_flow_imbalance(self, symbol: str) -> float:
        """Calculate recent order flow imbalance."""
        recent_trades = [t for t in self.trade_history if t.symbol == symbol][
            -20:
        ]  # Last 20 trades

        if not recent_trades:
            return 0.0

        buy_volume = sum(t.quantity for t in recent_trades if t.side == OrderSide.BUY)
        sell_volume = sum(t.quantity for t in recent_trades if t.side == OrderSide.SELL)
        total_volume = buy_volume + sell_volume

        if total_volume == 0:
            return 0.0

        return (buy_volume - sell_volume) / total_volume

    def _can_place_bid(self, symbol: str, quantity: int) -> bool:
        """Check if we can place a bid order."""
        if not self.enable_risk_management:
            return True

        # Check position limits
        new_inventory = self.inventory[symbol] + quantity
        if new_inventory > self.config.max_position:
            return False

        # Check cash constraints
        mid_price = self.lob_simulators[symbol].get_mid_price() or 100.0
        required_cash = quantity * mid_price
        if self.cash_balance < required_cash:
            return False

        return True

    def _can_place_ask(self, symbol: str, quantity: int) -> bool:
        """Check if we can place an ask order."""
        if not self.enable_risk_management:
            return True

        # Check position limits
        new_inventory = self.inventory[symbol] - quantity
        if new_inventory < -self.config.max_position:
            return False

        # Check if we have enough inventory
        if self.inventory[symbol] < quantity:
            return False

        return True

    def _cancel_all_quotes(self, symbol: str):
        """Cancel all active quotes for a symbol."""
        quotes = self.active_quotes[symbol]
        lob = self.lob_simulators[symbol]

        # Cancel bid orders
        for _, _, order_id in quotes["bid_orders"]:
            lob.cancel_order(order_id)

        # Cancel ask orders
        for _, _, order_id in quotes["ask_orders"]:
            lob.cancel_order(order_id)

        # Clear active quotes
        quotes["bid_orders"] = []
        quotes["ask_orders"] = []

    def _handle_trades(self, trades: List[Trade], symbol: str):
        """Handle executed trades."""
        for trade in trades:
            # Update inventory
            if trade.buyer_id == "market_maker":
                self.inventory[symbol] += trade.quantity
                self.cash_balance -= trade.price * trade.quantity
            elif trade.seller_id == "market_maker":
                self.inventory[symbol] -= trade.quantity
                self.cash_balance += trade.price * trade.quantity

            # Record trade
            self.trade_history.append(
                {
                    "symbol": symbol,
                    "side": OrderSide.BUY
                    if trade.buyer_id == "market_maker"
                    else OrderSide.SELL,
                    "quantity": trade.quantity,
                    "price": trade.price,
                    "timestamp": self.current_step,
                }
            )

            # Update quote history
            self.quote_history.append(
                {
                    "symbol": symbol,
                    "timestamp": self.current_step,
                    "mid_price": self.lob_simulators[symbol].get_mid_price(),
                    "inventory": self.inventory[symbol],
                    "action": "trade_executed",
                }
            )

        # Remove filled orders from active quotes
        self._remove_filled_quotes(symbol, trades)

    def _remove_filled_quotes(self, symbol: str, trades: List[Trade]):
        """Remove filled orders from active quotes."""
        quotes = self.active_quotes[symbol]
        filled_order_ids = {t.buy_order_id for t in trades} | {
            t.sell_order_id for t in trades
        }

        # Remove from bid orders
        quotes["bid_orders"] = [
            (price, qty, oid)
            for price, qty, oid in quotes["bid_orders"]
            if oid not in filled_order_ids
        ]

        # Remove from ask orders
        quotes["ask_orders"] = [
            (price, qty, oid)
            for price, qty, oid in quotes["ask_orders"]
            if oid not in filled_order_ids
        ]

    def _calculate_reward(self, execution_details: Dict[str, Any]) -> float:
        """Calculate reward based on market-making performance."""
        # Calculate P&L for this step
        step_pnl = self._calculate_step_pnl()

        # Calculate inventory cost
        inventory_cost = self._calculate_total_inventory_cost()

        # Calculate execution cost
        execution_cost = len(execution_details["trades_executed"]) * 0.001

        # Calculate spread capture
        spread_capture = self._calculate_spread_capture()

        # Risk penalty
        risk_penalty = self._calculate_risk_penalty()

        # Reward components
        pnl_reward = step_pnl / self.portfolio_value if self.portfolio_value > 0 else 0
        inventory_penalty = (
            -inventory_cost / self.portfolio_value if self.portfolio_value > 0 else 0
        )
        execution_penalty = -execution_cost
        spread_reward = spread_capture * 0.1
        risk_penalty_scaled = -risk_penalty

        total_reward = (
            pnl_reward
            + inventory_penalty
            + execution_penalty
            + spread_reward
            + risk_penalty_scaled
        )

        return total_reward

    def _calculate_step_pnl(self) -> float:
        """Calculate P&L for current step."""
        step_pnl = 0.0

        # P&L from trades in this step
        for trade in self.trade_history:
            if trade["timestamp"] == self.current_step:
                mid_price = (
                    self.lob_simulators[trade["symbol"]].get_mid_price() or 100.0
                )

                if trade["side"] == OrderSide.BUY:
                    # Bought at trade.price, could sell at mid_price
                    step_pnl += (mid_price - trade["price"]) * trade["quantity"]
                else:
                    # Sold at trade.price, could buy at mid_price
                    step_pnl += (trade["price"] - mid_price) * trade["quantity"]

        # Mark-to-market P&L from inventory
        for symbol, inventory in self.inventory.items():
            if inventory != 0:
                mid_price = self.lob_simulators[symbol].get_mid_price() or 100.0
                step_pnl += inventory * (
                    mid_price
                    - getattr(self, "_prev_mid_price", {}).get(symbol, mid_price)
                )

        # Store current mid prices for next step
        self._prev_mid_price = {
            symbol: self.lob_simulators[symbol].get_mid_price() or 100.0
            for symbol in self.assets
        }

        return step_pnl

    def _calculate_total_inventory_cost(self) -> float:
        """Calculate total inventory holding cost."""
        total_cost = 0.0
        for symbol, inventory in self.inventory.items():
            if inventory != 0:
                # Simple quadratic inventory cost
                inventory_ratio = inventory / self.config.max_position
                total_cost += abs(inventory_ratio) ** 2 * self.config.inventory_penalty

        return total_cost

    def _calculate_spread_capture(self) -> float:
        """Calculate spread capture from trades."""
        spread_capture = 0.0

        for trade in self.trade_history:
            if trade["timestamp"] == self.current_step:
                lob = self.lob_simulators[trade["symbol"]]
                current_spread = lob.get_spread()
                if current_spread > 0:
                    # Estimate spread capture (simplified)
                    spread_capture += (
                        current_spread * 0.5
                    )  # Assume we capture half the spread

        return spread_capture

    def _calculate_risk_penalty(self) -> float:
        """Calculate risk penalty for excessive risk taking."""
        penalty = 0.0

        # Inventory risk penalty
        for symbol, inventory in self.inventory.items():
            inventory_ratio = inventory / self.config.max_position
            if abs(inventory_ratio) > 0.8:  # High inventory
                penalty += abs(inventory_ratio) * 0.01

        # Concentration risk
        total_inventory = sum(abs(inv) for inv in self.inventory.values())
        if total_inventory > self.config.max_position * len(self.assets) * 0.5:
            penalty += 0.02

        return penalty

    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic market data for market-making simulation."""
        dates = pd.date_range(
            start="2023-01-01", periods=self.max_episode_length + 100, freq="1min"
        )

        data = {}
        for asset in self.assets:
            # Generate more volatile price series for market-making
            np.random.seed(hash(asset.symbol) % 2**32)

            # Base trend with mean reversion
            trend = np.random.normal(0, 0.0001, len(dates))
            mean_reversion = -0.1 * (np.random.randn(len(dates)).cumsum() * 0.01)
            noise = np.random.normal(0, 0.002, len(dates))

            returns = trend + mean_reversion + noise
            prices = [asset.initial_price]

            for ret in returns:
                prices.append(max(prices[-1] * (1 + ret), 1.0))

            prices = prices[1:]

            # Generate OHLCV data with realistic spreads
            spread_component = np.random.uniform(0.0001, 0.002, len(prices))
            volume_component = np.random.lognormal(10, 0.5, len(prices))

            opens = prices
            highs = prices + spread_component * prices
            lows = prices - spread_component * prices
            closes = prices + np.random.normal(0, 0.0001, len(prices))
            volumes = volume_component.astype(int)

            # Create DataFrame for this asset
            asset_data = pd.DataFrame(
                {
                    "Open": opens,
                    "High": highs,
                    "Low": lows,
                    "Close": closes,
                    "Volume": volumes,
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
        """Execute one step in the market-making environment."""
        # Process action (update quotes)
        execution_details = self._process_action(action)

        # Simulate some market activity
        self._simulate_market_activity()

        # Calculate reward
        reward = self._calculate_reward(execution_details)

        # Update state
        self.current_step += 1
        self._update_market_data()
        self._update_portfolio_value()

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
                "inventory": self.inventory.copy(),
                "active_quotes": {
                    symbol: len(quotes["bid_orders"]) + len(quotes["ask_orders"])
                    for symbol, quotes in self.active_quotes.items()
                },
                "total_trades": len(self.trade_history),
                "pnl_breakdown": {
                    "total": self.total_pnl,
                    "realized": self.realized_pnl,
                    "unrealized": self.unrealized_pnl,
                },
            }
        )

        return obs, reward, terminated, truncated, info

    def _simulate_market_activity(self):
        """Simulate external market activity."""
        for symbol, lob in self.lob_simulators.items():
            # Add some random liquidity
            if np.random.random() < 0.3:  # 30% chance each step
                lob.add_liquidity(num_orders=2, price_range=0.1)

            # Occasionally execute random market orders
            if np.random.random() < 0.1:  # 10% chance each step
                side = OrderSide.BUY if np.random.random() < 0.5 else OrderSide.SELL
                quantity = np.random.randint(50, 200)

                market_order = Order(
                    order_id=f"market_{self.current_step}_{symbol}",
                    side=side,
                    order_type=OrderType.MARKET,
                    quantity=quantity,
                    trader_id="external_trader",
                )

                trades = lob.place_order(market_order)
                if trades:
                    self._handle_trades(trades, symbol)

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio of P&L."""
        if len(self.trade_history) < 10:
            return 0.0

        # Calculate daily returns from trade history
        daily_pnl = []
        for step in range(max(0, self.current_step - 20), self.current_step + 1):
            step_pnl = sum(
                (t.price - (self.lob_simulators[t["symbol"]].get_mid_price() or 100.0))
                * t.quantity
                for t in self.trade_history
                if t["timestamp"] == step
            )
            daily_pnl.append(step_pnl)

        if len(daily_pnl) < 2:
            return 0.0

        returns = np.array(daily_pnl)
        if returns.std() == 0:
            return 0.0

        return returns.mean() / returns.std() * np.sqrt(252)  # Annualized

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if len(self.trade_history) < 2:
            return 0.0

        # Calculate cumulative P&L
        cumulative_pnl = []
        running_pnl = 0.0

        for trade in self.trade_history:
            if trade["side"] == OrderSide.BUY:
                running_pnl -= trade.price * trade.quantity
            else:
                running_pnl += trade.price * trade.quantity
            cumulative_pnl.append(running_pnl)

        if len(cumulative_pnl) < 2:
            return 0.0

        peak = np.maximum.accumulate(cumulative_pnl)
        drawdown = (peak - cumulative_pnl) / peak
        max_drawdown = np.max(drawdown)

        return max_drawdown if not np.isnan(max_drawdown) else 0.0

    def _calculate_order_flow_balance(self) -> float:
        """Calculate order flow balance."""
        recent_trades = self.trade_history[-50:] if len(self.trade_history) > 0 else []

        if not recent_trades:
            return 0.0

        buy_volume = sum(
            t.quantity for t in recent_trades if t["side"] == OrderSide.BUY
        )
        sell_volume = sum(
            t.quantity for t in recent_trades if t["side"] == OrderSide.SELL
        )
        total_volume = buy_volume + sell_volume

        if total_volume == 0:
            return 0.0

        return (buy_volume - sell_volume) / total_volume

    def _calculate_risk_utilization(self) -> float:
        """Calculate how much of risk limits are being used."""
        max_inventory = self.config.max_position
        total_inventory = sum(abs(inv) for inv in self.inventory.values())
        max_total_inventory = max_inventory * len(self.assets)

        return total_inventory / max_total_inventory if max_total_inventory > 0 else 0.0

    def _calculate_inventory_cost(self, symbol: str) -> float:
        """Calculate inventory cost for a specific symbol."""
        inventory = self.inventory[symbol]
        inventory_ratio = inventory / self.config.max_position
        return abs(inventory_ratio) ** 2 * self.config.inventory_penalty

    def _calculate_inventory_risk(self, symbol: str) -> float:
        """Calculate inventory risk metric."""
        inventory = self.inventory[symbol]
        volatility = self._calculate_volatility(symbol)
        return abs(inventory) * volatility / self.config.max_position

    def _calculate_risk_metrics(self) -> List[float]:
        """Calculate comprehensive risk metrics."""
        metrics = []

        # Value at Risk (simplified)
        total_inventory_value = sum(
            abs(inv) * (self.lob_simulators[symbol].get_mid_price() or 100.0)
            for symbol, inv in self.inventory.items()
        )
        var_5 = total_inventory_value * 0.05  # Simplified 5% VaR
        metrics.append(var_5 / self.portfolio_value if self.portfolio_value > 0 else 0)

        # Inventory turnover
        total_traded = sum(t.quantity for t in self.trade_history)
        avg_inventory = (
            np.mean([abs(inv) for inv in self.inventory.values()])
            if self.inventory
            else 1
        )
        turnover = total_traded / (avg_inventory * max(1, self.current_step))
        metrics.append(turnover)

        # Quote-to-trade ratio
        total_quotes = sum(
            len(quotes["bid_orders"]) + len(quotes["ask_orders"])
            for quotes in self.active_quotes.values()
        )
        quote_trade_ratio = total_quotes / max(1, len(self.trade_history))
        metrics.append(quote_trade_ratio)

        # Concentration risk
        inventory_values = [
            abs(inv) * (self.lob_simulators[symbol].get_mid_price() or 100.0)
            for symbol, inv in self.inventory.items()
        ]
        if sum(inventory_values) > 0:
            concentration = max(inventory_values) / sum(inventory_values)
        else:
            concentration = 0.0
        metrics.append(concentration)

        # Leverage
        leverage = (
            total_inventory_value / self.portfolio_value
            if self.portfolio_value > 0
            else 0
        )
        metrics.append(leverage)

        # Remaining metrics (padding to match expected size)
        metrics.extend([0.0] * 5)  # Add 5 more metrics as needed

        return metrics[:10]  # Return exactly 10 metrics

    def get_market_making_summary(self) -> Dict[str, Any]:
        """Get comprehensive market-making performance summary."""
        return {
            "performance": {
                "total_pnl": self.total_pnl,
                "realized_pnl": self.realized_pnl,
                "unrealized_pnl": self.unrealized_pnl,
                "sharpe_ratio": self._calculate_sharpe_ratio(),
                "max_drawdown": self._calculate_max_drawdown(),
            },
            "inventory": {
                "current_positions": self.inventory.copy(),
                "inventory_cost": self._calculate_total_inventory_cost(),
                "inventory_utilization": self._calculate_risk_utilization(),
            },
            "trading": {
                "total_trades": len(self.trade_history),
                "quotes_placed": len(self.quote_history),
                "average_spread": np.mean(
                    [lob.get_spread() for lob in self.lob_simulators.values()]
                )
                if self.lob_simulators
                else 0,
                "order_flow_balance": self._calculate_order_flow_balance(),
            },
            "risk": {
                "current_var": self._calculate_risk_metrics()[0],
                "leverage": self._calculate_risk_metrics()[4],
                "concentration": self._calculate_risk_metrics()[3],
            },
        }
