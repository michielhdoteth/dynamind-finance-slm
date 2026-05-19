"""
Market Making Environment

A financial market making environment where the agent provides liquidity by placing
bid and ask orders, managing inventory, and capturing the bid-ask spread.

Action Space: Box - [bid_price, ask_price, bid_size, ask_size]
Observation Space: Box - Order book data, inventory, market indicators
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from gymnasium import spaces

from .base_env import (
    AssetConfig,
    FinancialTradingBase,
    RiskConstraints,
    TransactionCosts,
)


@dataclass
class MarketMakingConfig:
    """Configuration specific to market making"""

    max_position: float = 1000  # Maximum inventory position
    inventory_penalty: float = 0.001  # Penalty for holding inventory
    spread_target: float = 0.002  # Target spread (2 bps)
    max_spread: float = 0.01  # Maximum allowed spread
    order_duration: int = 5  # Order lifetime in steps
    adverse_selection_factor: float = 0.3  # Probability of adverse selection
    market_impact_factor: float = 0.0001  # Market impact coefficient
    alpha_decay: float = 0.95  # Information decay factor


class MarketMakingEnv(FinancialTradingBase):
    """
    Market Making Environment

    A gymnasium environment for market making strategies.
    The agent learns to provide liquidity by placing bid and ask orders,
    managing inventory risk, and capturing the bid-ask spread.

    Features:
    - Realistic order book simulation
    - Inventory management constraints
    - Adverse selection modeling
    - Market impact calculations
    - Dynamic spread optimization
    - Information asymmetry challenges
    """

    def __init__(
        self,
        asset: AssetConfig,
        initial_cash: float = 100_000,
        max_episode_length: int = 1000,
        lookback_window: int = 50,
        transaction_costs: TransactionCosts = None,
        risk_constraints: RiskConstraints = None,
        config: MarketMakingConfig = None,
        order_book_depth: int = 10,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        # Store configuration
        self.config = config or MarketMakingConfig()
        self.order_book_depth = order_book_depth

        # Market making specific state
        self.current_orders = []  # Active limit orders
        self.order_history = []  # Completed orders
        self.inventory = 0  # Current inventory position
        self.mid_price = asset.initial_price
        self.true_alpha = 0  # Hidden information about true value
        self.market_orders = []  # Incoming market orders

        # Initialize with single asset for market making
        super().__init__(
            assets=[asset],
            initial_cash=initial_cash,
            max_episode_length=max_episode_length,
            lookback_window=lookback_window,
            transaction_costs=transaction_costs,
            risk_constraints=risk_constraints,
            seed=seed,
            render_mode=render_mode,
        )

        # Performance tracking
        self.pnl_components = {
            "spread_capture": 0,
            "inventory_pnl": 0,
            "adverse_selection": 0,
            "transaction_costs": 0,
        }

    def _initialize_environment(self):
        """Initialize environment-specific components"""
        # Action space: [bid_price_offset, ask_price_offset, bid_size, ask_size]
        # Price offsets are relative to mid price
        self.action_space = spaces.Box(
            low=np.array([-self.config.max_spread, -self.config.max_spread, 0, 0]),
            high=np.array(
                [
                    0,
                    self.config.max_spread,
                    self.config.max_position,
                    self.config.max_position,
                ]
            ),
            dtype=np.float32,
        )

        # Observation space components
        obs_size = (
            self.lookback_window * 5
            + self.order_book_depth * 4  # Price and volume history
            + 10  # Order book levels
            + 10  # Inventory and position metrics
            + 10  # Market microstructure indicators
            + 5  # Information and alpha indicators  # Order management metrics
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic market data with order book simulation"""
        n_steps = self.max_episode_length + self.lookback_window

        # Generate price process with information component
        prices, self.true_alpha_series = self._generate_price_with_alpha(n_steps)

        # Generate market order flow
        market_orders = self._generate_market_order_flow(n_steps)

        # Simulate order book dynamics
        order_book_data = self._simulate_order_book(prices, market_orders)

        # Create DataFrame
        data = pd.DataFrame(
            {
                "mid_price": prices,
                "true_alpha": self.true_alpha_series,
                "buy_pressure": market_orders["buy_flow"],
                "sell_pressure": market_orders["sell_flow"],
                "volume": market_orders["total_volume"],
            }
        )

        # Add order book levels
        for level in range(self.order_book_depth):
            data[f"bid_price_{level}"] = order_book_data["bids"][:, level]
            data[f"bid_size_{level}"] = order_book_data["bid_sizes"][:, level]
            data[f"ask_price_{level}"] = order_book_data["asks"][:, level]
            data[f"ask_size_{level}"] = order_book_data["ask_sizes"][:, level]

        # Calculate market microstructure indicators
        self._calculate_microstructure_indicators(data)

        return data

    def _generate_price_with_alpha(self, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate price process with hidden information component"""
        # Base price process (geometric Brownian motion)
        asset = self.assets[0]
        returns = np.random.normal(asset.drift, asset.volatility, n_steps)
        prices = asset.initial_price * np.cumprod(1 + returns)

        # Hidden alpha (information about true value)
        alpha = np.zeros(n_steps)
        current_alpha = 0
        alpha_revealed = False

        for t in range(n_steps):
            # Random information events
            if np.random.random() < 0.01:  # 1% chance of information event
                current_alpha = np.random.normal(0, 0.005)  # 50 bps information
                alpha_revealed = False

            # Gradual information revelation
            if not alpha_revealed and abs(current_alpha) > 0.001:
                current_alpha *= self.config.alpha_decay
                if abs(current_alpha) < 0.0001:  # Fully revealed
                    alpha_revealed = True

            alpha[t] = current_alpha
            # Apply alpha to price
            prices[t] *= 1 + alpha[t]

        return prices, alpha

    def _generate_market_order_flow(self, n_steps: int) -> Dict[str, np.ndarray]:
        """Generate market order flow"""
        buy_flow = np.zeros(n_steps)
        sell_flow = np.zeros(n_steps)
        total_volume = np.zeros(n_steps)

        for t in range(n_steps):
            # Base order flow intensity
            base_intensity = self.assets[0].avg_daily_volume / 252  # Daily average

            # Modulate by volatility and information
            intensity_multiplier = 1 + abs(self.true_alpha_series[t]) * 10

            # Generate Poisson-like order flow
            n_orders = np.random.poisson(base_intensity * intensity_multiplier / 1000)

            for _ in range(int(n_orders)):
                order_size = np.random.exponential(100)  # Average order size
                order_direction = np.random.choice([1, -1], p=[0.5, 0.5])

                # Information-driven order flow
                if self.true_alpha_series[t] > 0.001:  # Positive information
                    order_direction = 1 if np.random.random() < 0.7 else -1
                elif self.true_alpha_series[t] < -0.001:  # Negative information
                    order_direction = -1 if np.random.random() < 0.7 else 1

                if order_direction > 0:
                    buy_flow[t] += order_size
                else:
                    sell_flow[t] += order_size

                total_volume[t] += order_size

        return {
            "buy_flow": buy_flow,
            "sell_flow": sell_flow,
            "total_volume": total_volume,
        }

    def _simulate_order_book(
        self, prices: np.ndarray, market_orders: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Simulate order book dynamics"""
        n_steps = len(prices)
        bids = np.zeros((n_steps, self.order_book_depth))
        asks = np.zeros((n_steps, self.order_book_depth))
        bid_sizes = np.zeros((n_steps, self.order_book_depth))
        ask_sizes = np.zeros((n_steps, self.order_book_depth))

        for t in range(n_steps):
            mid_price = prices[t]
            base_spread = self.config.spread_target * mid_price

            # Generate order book levels
            for level in range(self.order_book_depth):
                # Bid prices (decreasing)
                price_offset = base_spread * (level + 1)
                bids[t, level] = mid_price - price_offset

                # Ask prices (increasing)
                asks[t, level] = mid_price + price_offset

                # Sizes (random with depth decay)
                size_multiplier = np.exp(-0.5 * level)
                bid_sizes[t, level] = np.random.exponential(500) * size_multiplier
                ask_sizes[t, level] = np.random.exponential(500) * size_multiplier

        return {
            "bids": bids,
            "asks": asks,
            "bid_sizes": bid_sizes,
            "ask_sizes": ask_sizes,
        }

    def _calculate_microstructure_indicators(self, data: pd.DataFrame):
        """Calculate market microstructure indicators"""
        # Spread measures
        data["quoted_spread"] = data["ask_price_0"] - data["bid_price_0"]
        data["quoted_spread_bps"] = data["quoted_spread"] / data["mid_price"] * 10000
        data["effective_spread"] = data["quoted_spread"] / 2

        # Order imbalance
        data["order_imbalance"] = (data["buy_flow"] - data["sell_flow"]) / (
            data["buy_flow"] + data["sell_flow"] + 1e-8
        )

        # Volume measures
        data["volume_ratio"] = data["volume"] / self.assets[0].avg_daily_volume

        # Price impact indicators
        data["price_volatility"] = (
            data["mid_price"].pct_change().rolling(window=20).std()
        )
        data["volume_volatility"] = data["volume"].rolling(window=20).std()

        # Market depth
        total_bid_depth = sum(
            data[f"bid_size_{i}"] for i in range(self.order_book_depth)
        )
        total_ask_depth = sum(
            data[f"ask_size_{i}"] for i in range(self.order_book_depth)
        )
        data["market_depth"] = (total_bid_depth + total_ask_depth) / 2

        # Fill NaN values
        data.fillna(method="bfill", inplace=True)
        data.fillna(0, inplace=True)

    def _get_observation(self) -> np.ndarray:
        """Get current market making observation"""
        obs_components = []

        # 1. Price and volume history
        for i in range(self.lookback_window):
            if self.current_step >= i:
                hist_step = self.current_step - i
                price_change = (
                    (
                        self.market_data["mid_price"].iloc[hist_step]
                        - self.market_data["mid_price"].iloc[hist_step - 1]
                    )
                    / self.market_data["mid_price"].iloc[hist_step - 1]
                    if hist_step > 0
                    else 0
                )
                obs_components.extend(
                    [
                        price_change,
                        self.market_data["volume"].iloc[hist_step]
                        / self.assets[0].avg_daily_volume,
                        self.market_data["order_imbalance"].iloc[hist_step],
                        self.market_data["quoted_spread_bps"].iloc[hist_step] / 100,
                        self.market_data["true_alpha"].iloc[hist_step]
                        * 100,  # Scaled alpha
                    ]
                )
            else:
                obs_components.extend([0] * 5)

        # 2. Current order book state
        current_step_data = self.market_data.iloc[self.current_step]
        for level in range(self.order_book_depth):
            bid_price = current_step_data[f"bid_price_{level}"]
            ask_price = current_step_data[f"ask_price_{level}"]
            mid_price = current_step_data["mid_price"]

            obs_components.extend(
                [
                    (mid_price - bid_price) / mid_price * 10000,  # Bid distance in bps
                    (ask_price - mid_price) / mid_price * 10000,  # Ask distance in bps
                    current_step_data[f"bid_size_{level}"] / 1000,  # Normalized size
                    current_step_data[f"ask_size_{level}"] / 1000,  # Normalized size
                ]
            )

        # 3. Inventory and position metrics
        inventory_value = self.inventory * self.mid_price
        portfolio_value = self.cash_balance + abs(inventory_value)

        obs_components.extend(
            [
                self.inventory / self.config.max_position,  # Normalized inventory
                inventory_value / portfolio_value,  # Inventory ratio
                self.cash_balance / portfolio_value,  # Cash ratio
                self.unrealized_pnl / portfolio_value,  # Unrealized P&L ratio
                len(self.current_orders) / 10,  # Number of active orders
                self._calculate_inventory_risk(),  # Inventory risk measure
                self._calculate_optimal_spread(),  # Theoretical optimal spread
                self._calculate_adverse_selection_risk(),  # Adverse selection risk
                self._calculate_Inventory_pressure(),  # Inventory pressure
                self._calculate_time_decay_factor(),  # Time decay for orders
            ]
        )

        # 4. Market microstructure indicators
        obs_components.extend(
            [
                current_step_data["quoted_spread_bps"] / 10,  # Normalized spread
                current_step_data["order_imbalance"],  # Order imbalance
                current_step_data["volume_ratio"],  # Volume ratio
                current_step_data["price_volatility"] / 0.02,  # Normalized volatility
                current_step_data["market_depth"] / 10000,  # Normalized depth
                self._calculate_price_pressure(),  # Net price pressure
                self._calculate_liquidity_score(),  # Liquidity score
                self._calculate_volatility_regime(),  # Volatility regime
                self._calculate_market_stress(),  # Market stress indicator
                self._calculate_competition_intensity(),  # Competition from other market makers
            ]
        )

        # 5. Information and alpha indicators
        obs_components.extend(
            [
                current_step_data["true_alpha"] * 1000,  # Scaled true alpha
                self._estimate_alpha_from_market(),  # Estimated alpha from market data
                self._calculate_alpha_uncertainty(),  # Uncertainty about alpha
                self._calculate_information_decay(),  # Information decay rate
                self._calculate_order_flow_toxicity(),  # Order flow toxicity
                self._detect_informed_trading(),  # Informed trading probability
                self._calculate_predictable_moves(),  # Predictable price movements
                self._estimate_market_impact(),  # Market impact estimate
                self._calculate_latency_advantage(),  # Latency/speed advantage
                self._calculate_crossing_probability(),  # Probability of crossing spread
            ]
        )

        # 6. Order management metrics
        obs_components.extend(
            [
                self._calculate_order_fill_rate(),  # Recent fill rate
                self._calculate_average_fill_size(),  # Average fill size
                self._calculate_cancellation_rate(),  # Cancellation rate
                self._calculate_order_age_distribution(),  # Order age distribution
                self._calculate_competitive_position(),  # Competitive position in order book
            ]
        )

        return np.array(obs_components, dtype=np.float32)

    def _calculate_inventory_risk(self) -> float:
        """Calculate inventory risk measure"""
        # Risk increases with position size and volatility
        current_vol = self.market_data["price_volatility"].iloc[self.current_step]
        inventory_ratio = abs(self.inventory) / self.config.max_position
        return inventory_ratio * (current_vol / 0.02 + 1)

    def _calculate_optimal_spread(self) -> float:
        """Calculate theoretical optimal spread"""
        # Avellaneda-Stoikov model approximation
        current_vol = self.market_data["price_volatility"].iloc[self.current_step]
        inventory_ratio = self.inventory / self.config.max_position

        # Base spread
        optimal_spread = 2 * current_vol + abs(inventory_ratio) * 0.002

        return optimal_spread

    def _calculate_adverse_selection_risk(self) -> float:
        """Calculate risk of adverse selection"""
        # Higher when true alpha is large and order flow is imbalanced
        current_alpha = abs(self.market_data["true_alpha"].iloc[self.current_step])
        order_imbalance = abs(
            self.market_data["order_imbalance"].iloc[self.current_step]
        )

        return min(1.0, current_alpha * 100 + order_imbalance * 0.5)

    def _calculate_Inventory_pressure(self) -> float:
        """Calculate pressure to reduce inventory"""
        # Pressure increases with inventory size and time
        inventory_ratio = self.inventory / self.config.max_position
        time_pressure = len(self.portfolio_history) / self.max_episode_length

        return inventory_ratio * (1 + time_pressure)

    def _calculate_time_decay_factor(self) -> float:
        """Calculate time decay factor for orders"""
        # Orders decay as they get older
        if not self.current_orders:
            return 0.0

        max_age = max(order["age"] for order in self.current_orders)
        return max_age / self.config.order_duration

    def _calculate_price_pressure(self) -> float:
        """Calculate net price pressure"""
        buy_pressure = self.market_data["buy_flow"].iloc[self.current_step]
        sell_pressure = self.market_data["sell_flow"].iloc[self.current_step]

        return (buy_pressure - sell_pressure) / (buy_pressure + sell_pressure + 1e-8)

    def _calculate_liquidity_score(self) -> float:
        """Calculate market liquidity score"""
        total_depth = 0
        for level in range(self.order_book_depth):
            total_depth += (
                self.market_data[f"bid_size_{level}"].iloc[self.current_step]
                + self.market_data[f"ask_size_{level}"].iloc[self.current_step]
            )

        return total_depth / 10000  # Normalized

    def _calculate_volatility_regime(self) -> float:
        """Calculate current volatility regime"""
        current_vol = self.market_data["price_volatility"].iloc[self.current_step]

        if current_vol < 0.01:
            return 0.0  # Low volatility
        elif current_vol < 0.02:
            return 0.5  # Normal volatility
        else:
            return 1.0  # High volatility

    def _calculate_market_stress(self) -> float:
        """Calculate market stress indicator"""
        stress_factors = [
            self.market_data["price_volatility"].iloc[self.current_step] / 0.03,
            abs(self.market_data["order_imbalance"].iloc[self.current_step]),
            1 - self._calculate_liquidity_score(),
        ]

        return np.mean(stress_factors)

    def _calculate_competition_intensity(self) -> float:
        """Estimate competition from other market makers"""
        # Proxy: tightness of quoted spread
        current_spread = self.market_data["quoted_spread_bps"].iloc[self.current_step]

        # Lower spread indicates more competition
        competition = max(0, 1 - current_spread / 20)  # 20 bps as reference
        return competition

    def _estimate_alpha_from_market(self) -> float:
        """Estimate alpha from observable market data"""
        # Use order flow imbalance and price pressure
        order_imbalance = self.market_data["order_imbalance"].iloc[self.current_step]
        price_pressure = self._calculate_price_pressure()

        estimated_alpha = (order_imbalance + price_pressure) * 0.0001
        return estimated_alpha

    def _calculate_alpha_uncertainty(self) -> float:
        """Calculate uncertainty about alpha estimation"""
        # Higher uncertainty in high volatility and low volume
        vol = self.market_data["price_volatility"].iloc[self.current_step]
        volume_ratio = self.market_data["volume_ratio"].iloc[self.current_step]

        uncertainty = vol / 0.02 + (1 / (volume_ratio + 0.1))
        return min(2.0, uncertainty)

    def _calculate_information_decay(self) -> float:
        """Calculate rate of information decay"""
        # Based on recent alpha changes
        if self.current_step < 5:
            return 0.5

        recent_alphas = self.market_data["true_alpha"].iloc[
            self.current_step - 5 : self.current_step + 1
        ]
        alpha_variance = np.var(recent_alphas)

        return min(1.0, alpha_variance * 1000)

    def _calculate_order_flow_toxicity(self) -> float:
        """Calculate order flow toxicity (probability of informed trading)"""
        # VPIN-like measure
        volume = self.market_data["volume"].iloc[self.current_step]
        buy_volume = self.market_data["buy_flow"].iloc[self.current_step]

        if volume > 0:
            toxicity = abs(buy_volume - volume / 2) / volume
        else:
            toxicity = 0.5

        return toxicity

    def _detect_informed_trading(self) -> float:
        """Detect probability of informed trading"""
        # Combination of factors
        toxicity = self._calculate_order_flow_toxicity()
        alpha_magnitude = abs(self.market_data["true_alpha"].iloc[self.current_step])
        price_impact = (
            abs(self.market_data["mid_price"].pct_change().iloc[self.current_step])
            if self.current_step > 0
            else 0
        )

        informed_trading_prob = (
            toxicity + alpha_magnitude * 100 + price_impact * 1000
        ) / 3
        return min(1.0, informed_trading_prob)

    def _calculate_predictable_moves(self) -> float:
        """Calculate predictable price movements"""
        # Use autocorrelation and momentum
        if self.current_step < 10:
            return 0.0

        recent_returns = (
            self.market_data["mid_price"]
            .pct_change()
            .iloc[self.current_step - 10 : self.current_step + 1]
        )
        autocorr = (
            np.corrcoef(recent_returns[:-1], recent_returns[1:])[0, 1]
            if len(recent_returns) > 1
            else 0
        )

        return abs(autocorr)

    def _estimate_market_impact(self) -> float:
        """Estimate market impact of trades"""
        # Square-root market impact model
        volume_ratio = self.market_data["volume_ratio"].iloc[self.current_step]

        # Impact increases with trade size relative to volume
        base_impact = self.config.market_impact_factor
        volume_adjustment = np.sqrt(1 / (volume_ratio + 0.1))

        return base_impact * volume_adjustment

    def _calculate_latency_advantage(self) -> float:
        """Calculate latency/speed advantage (simplified)"""
        # Assume we have average latency
        # Higher advantage when order flow is toxic
        toxicity = self._calculate_order_flow_toxicity()
        return min(1.0, toxicity * 2)

    def _calculate_crossing_probability(self) -> float:
        """Calculate probability of spread crossing"""
        # Higher in volatile markets with high inventory pressure
        vol = self.market_data["price_volatility"].iloc[self.current_step]
        inventory_pressure = self._calculate_Inventory_pressure()

        crossing_prob = (vol / 0.02 + inventory_pressure) / 2
        return min(1.0, crossing_prob)

    def _calculate_order_fill_rate(self) -> float:
        """Calculate recent order fill rate"""
        if not self.order_history:
            return 0.0

        recent_orders = self.order_history[-20:]  # Last 20 orders
        filled_orders = sum(1 for order in recent_orders if order.get("filled", False))

        return filled_orders / len(recent_orders)

    def _calculate_average_fill_size(self) -> float:
        """Calculate average fill size"""
        if not self.order_history:
            return 0.0

        filled_orders = [
            order for order in self.order_history[-20:] if order.get("filled", False)
        ]
        if not filled_orders:
            return 0.0

        avg_size = np.mean([order.get("fill_size", 0) for order in filled_orders])
        return avg_size / 100  # Normalized

    def _calculate_cancellation_rate(self) -> float:
        """Calculate order cancellation rate"""
        if not self.order_history:
            return 0.0

        recent_orders = self.order_history[-20:]
        cancelled_orders = sum(
            1 for order in recent_orders if order.get("cancelled", False)
        )

        return cancelled_orders / len(recent_orders)

    def _calculate_order_age_distribution(self) -> float:
        """Calculate average order age"""
        if not self.current_orders:
            return 0.0

        avg_age = np.mean([order["age"] for order in self.current_orders])
        return avg_age / self.config.order_duration

    def _calculate_competitive_position(self) -> float:
        """Calculate competitive position in order book"""
        if not self.current_orders:
            return 0.0

        # Check how competitive our prices are
        best_bid = self.market_data["bid_price_0"].iloc[self.current_step]
        best_ask = self.market_data["ask_price_0"].iloc[self.current_step]

        competitive_score = 0.0
        for order in self.current_orders:
            if order["side"] == "bid":
                if order["price"] >= best_bid:
                    competitive_score += 1
            else:  # ask
                if order["price"] <= best_ask:
                    competitive_score += 1

        return competitive_score / len(self.current_orders)

    def _process_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Process market making action"""
        bid_offset, ask_offset, bid_size, ask_size = action

        # Calculate actual prices
        bid_price = self.mid_price * (1 - bid_offset)
        ask_price = self.mid_price * (1 + ask_offset)

        # Ensure minimum spread
        if ask_price - bid_price < self.config.spread_target * self.mid_price:
            spread_adjustment = (
                self.config.spread_target * self.mid_price - (ask_price - bid_price)
            ) / 2
            bid_price -= spread_adjustment
            ask_price += spread_adjustment

        # Check inventory constraints
        total_position = self.inventory + bid_size - ask_size
        if abs(total_position) > self.config.max_position:
            # Adjust sizes to respect constraints
            if total_position > 0:
                ask_size = min(
                    ask_size, self.inventory + bid_size - self.config.max_position
                )
            else:
                bid_size = min(
                    bid_size, self.config.max_position - (self.inventory - ask_size)
                )

        # Place new orders (cancel existing ones)
        self._cancel_all_orders()

        execution_details = {
            "bid_placed": False,
            "ask_placed": False,
            "bid_price": bid_price,
            "ask_price": ask_price,
            "bid_size": bid_size,
            "ask_size": ask_size,
            "spread": ask_price - bid_price,
            "executed_trades": [],
        }

        # Place bid order
        if bid_size > 0:
            self._place_order("bid", bid_price, bid_size)
            execution_details["bid_placed"] = True

        # Place ask order
        if ask_size > 0:
            self._place_order("ask", ask_price, ask_size)
            execution_details["ask_placed"] = True

        # Simulate market order execution
        market_results = self._simulate_market_order_execution()
        execution_details["executed_trades"] = market_results

        # Update P&L components
        self._update_pnl_components(market_results)

        return execution_details

    def _cancel_all_orders(self):
        """Cancel all active orders"""
        for order in self.current_orders:
            order["cancelled"] = True
            self.order_history.append(order)

        self.current_orders = []

    def _place_order(self, side: str, price: float, size: int):
        """Place a limit order"""
        order = {
            "side": side,
            "price": price,
            "size": size,
            "filled_size": 0,
            "age": 0,
            "filled": False,
            "cancelled": False,
        }
        self.current_orders.append(order)

    def _simulate_market_order_execution(self) -> List[Dict[str, Any]]:
        """Simulate execution of limit orders against market orders"""
        executed_trades = []

        # Get current market orders
        buy_flow = self.market_data["buy_flow"].iloc[self.current_step]
        sell_flow = self.market_data["sell_flow"].iloc[self.current_step]

        # Process bid orders (executed against sell market orders)
        for order in self.current_orders[:]:  # Copy list to allow modification
            if order["side"] == "bid" and sell_flow > 0:
                # Check if our price is competitive
                best_bid = self.market_data["bid_price_0"].iloc[self.current_step]
                if order["price"] >= best_bid:
                    # Execute trade
                    fill_size = min(order["size"] - order["filled_size"], sell_flow)

                    if fill_size > 0:
                        # Account for adverse selection
                        if np.random.random() < self.config.adverse_selection_factor:
                            # Informed trader - lose money
                            true_alpha = self.market_data["true_alpha"].iloc[
                                self.current_step
                            ]
                            execution_price = order["price"] * (1 - true_alpha)
                        else:
                            execution_price = order["price"]

                        # Update inventory and cash
                        self.inventory += fill_size
                        self.cash_balance -= fill_size * execution_price
                        self.realized_pnl -= (
                            fill_size
                            * execution_price
                            * self.transaction_costs.commission_rate
                        )

                        # Update order
                        order["filled_size"] += fill_size
                        if order["filled_size"] >= order["size"]:
                            order["filled"] = True
                            self.current_orders.remove(order)
                            self.order_history.append(order)

                        # Record trade
                        executed_trades.append(
                            {
                                "side": "buy",
                                "price": execution_price,
                                "size": fill_size,
                                "pnl_impact": -true_alpha * fill_size * execution_price
                                if "true_alpha" in locals()
                                else 0,
                            }
                        )

                        sell_flow -= fill_size

        # Process ask orders (executed against buy market orders)
        for order in self.current_orders[:]:
            if order["side"] == "ask" and buy_flow > 0:
                # Check if our price is competitive
                best_ask = self.market_data["ask_price_0"].iloc[self.current_step]
                if order["price"] <= best_ask:
                    # Execute trade
                    fill_size = min(order["size"] - order["filled_size"], buy_flow)

                    if fill_size > 0 and self.inventory >= fill_size:
                        # Account for adverse selection
                        if np.random.random() < self.config.adverse_selection_factor:
                            # Informed trader - lose money
                            true_alpha = self.market_data["true_alpha"].iloc[
                                self.current_step
                            ]
                            execution_price = order["price"] * (1 + true_alpha)
                        else:
                            execution_price = order["price"]

                        # Update inventory and cash
                        self.inventory -= fill_size
                        self.cash_balance += fill_size * execution_price
                        self.realized_pnl -= (
                            fill_size
                            * execution_price
                            * self.transaction_costs.commission_rate
                        )

                        # Update order
                        order["filled_size"] += fill_size
                        if order["filled_size"] >= order["size"]:
                            order["filled"] = True
                            self.current_orders.remove(order)
                            self.order_history.append(order)

                        # Record trade
                        executed_trades.append(
                            {
                                "side": "sell",
                                "price": execution_price,
                                "size": fill_size,
                                "pnl_impact": true_alpha * fill_size * execution_price
                                if "true_alpha" in locals()
                                else 0,
                            }
                        )

                        buy_flow -= fill_size

        # Age remaining orders
        for order in self.current_orders:
            order["age"] += 1
            # Cancel old orders
            if order["age"] > self.config.order_duration:
                order["cancelled"] = True
                self.current_orders.remove(order)
                self.order_history.append(order)

        return executed_trades

    def _update_pnl_components(self, executed_trades: List[Dict[str, Any]]):
        """Update P&L component tracking"""
        for trade in executed_trades:
            if trade["side"] == "sell":
                # Selling inventory - capture spread
                self.pnl_components["spread_capture"] += (
                    trade["size"] * self.config.spread_target * self.mid_price
                )
            else:
                # Buying inventory - cost of spread
                self.pnl_components["spread_capture"] -= (
                    trade["size"] * self.config.spread_target * self.mid_price
                )

            # Adverse selection impact
            self.pnl_components["adverse_selection"] += trade.get("pnl_impact", 0)

        # Inventory P&L from mark-to-market
        if len(self.portfolio_history) > 1:
            prev_value = self.portfolio_history[-2]
            current_value = self.portfolio_value
            trading_pnl = current_value - prev_value

            # Remove spread capture to get inventory P&L
            self.pnl_components["inventory_pnl"] += trading_pnl

    def _calculate_reward(self, execution_details: Dict[str, Any]) -> float:
        """Calculate market making reward"""
        # Base reward: total P&L change
        if len(self.portfolio_history) > 1:
            portfolio_return = (
                self.portfolio_value - self.portfolio_history[-2]
            ) / self.portfolio_history[-2]
        else:
            portfolio_return = 0

        # Inventory penalty (discourage holding inventory)
        inventory_penalty = (
            -self.config.inventory_penalty
            * abs(self.inventory)
            / self.config.max_position
        )

        # Spread capture reward
        spread_reward = (
            execution_details["spread"]
            * len(execution_details["executed_trades"])
            * 0.0001
        )

        # Adverse selection penalty
        adverse_penalty = sum(
            trade.get("pnl_impact", 0) for trade in execution_details["executed_trades"]
        )

        # Order fill rate bonus
        fill_rate = len(execution_details["executed_trades"]) / max(
            1, len(execution_details["executed_trades"]) + len(self.current_orders)
        )
        fill_bonus = fill_rate * 0.001

        # Risk adjustment
        risk_penalty = -0.1 * self._calculate_inventory_risk()

        # Total reward
        total_reward = (
            portfolio_return
            + inventory_penalty
            + spread_reward
            + adverse_penalty
            + fill_bonus
            + risk_penalty
        )

        return float(total_reward)

    def get_market_making_stats(self) -> Dict[str, Any]:
        """Get comprehensive market making statistics"""
        stats = {
            "total_trades": len(self.order_history),
            "filled_orders": sum(
                1 for order in self.order_history if order.get("filled", False)
            ),
            "cancelled_orders": sum(
                1 for order in self.order_history if order.get("cancelled", False)
            ),
            "fill_rate": self._calculate_order_fill_rate(),
            "average_spread": np.mean(
                [
                    self.market_data["quoted_spread_bps"].iloc[i]
                    for i in range(len(self.market_data))
                ]
            )
            / 10000,
            "inventory_utilization": abs(self.inventory) / self.config.max_position,
            "pnl_breakdown": self.pnl_components.copy(),
            "current_inventory": self.inventory,
            "active_orders": len(self.current_orders),
        }

        # Add performance metrics
        if len(self.portfolio_history) > 30:
            returns = (
                np.diff(self.portfolio_history[-30:]) / self.portfolio_history[-30:-1]
            )
            stats.update(
                {
                    "sharpe_ratio": np.mean(returns)
                    / (np.std(returns) + 1e-8)
                    * np.sqrt(252),
                    "volatility": np.std(returns) * np.sqrt(252),
                    "max_drawdown": self._calculate_max_drawdown(),
                }
            )

        return stats

    def get_action_meanings(self) -> List[str]:
        """Get human-readable action meanings"""
        return [
            "Bid price offset (from mid price)",
            "Ask price offset (from mid price)",
            "Bid size (shares)",
            "Ask size (shares)",
        ]
