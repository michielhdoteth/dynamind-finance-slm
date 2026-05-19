"""
Single Asset Trading Environment

A financial trading environment for trading a single asset.
Perfect for learning basic trading strategies and as a research benchmark.

Action Space: Discrete(3) - [Hold, Buy, Sell]
Observation Space: Box of technical indicators and portfolio state
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
class SingleAssetConfig:
    """Configuration specific to single asset trading"""

    price_history_length: int = 252  # One year of daily data
    technical_indicators: List[str] = None
    position_normalization: bool = True
    allow_shorting: bool = True
    leverage_limit: float = 2.0


class SingleAssetTradingEnv(FinancialTradingBase):
    """
    Single Asset Trading Environment

    A gymnasium environment for trading a single financial asset.
    The agent can choose to hold, buy, or sell the asset at each time step.

    Features:
    - Realistic price dynamics with regime switching
    - Technical indicator observations
    - Transaction costs and slippage
    - Risk management constraints
    - Continuous and discrete action spaces
    """

    def __init__(
        self,
        asset: AssetConfig,
        initial_cash: float = 1_000_000,
        max_episode_length: int = 252,
        lookback_window: int = 30,
        transaction_costs: TransactionCosts = None,
        risk_constraints: RiskConstraints = None,
        config: SingleAssetConfig = None,
        action_space_type: str = "discrete",  # "discrete" or "continuous"
        data_source: str = "synthetic",  # "synthetic" or "real"
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        # Store asset and config
        self.asset = asset
        self.config = config or SingleAssetConfig()
        self.action_space_type = action_space_type

        # Initialize base environment
        super().__init__(
            assets=[asset],
            initial_cash=initial_cash,
            max_episode_length=max_episode_length,
            lookback_window=lookback_window,
            transaction_costs=transaction_costs,
            risk_constraints=risk_constraints,
            data_source=data_source,
            seed=seed,
            render_mode=render_mode,
        )

    def _initialize_environment(self):
        """Initialize environment-specific components"""
        # Default technical indicators
        if self.config.technical_indicators is None:
            self.config.technical_indicators = [
                "sma_10",
                "sma_30",
                "rsi_14",
                "macd",
                "bollinger_upper",
                "bollinger_lower",
                "atr_14",
                "volume_ratio",
                "price_momentum_5",
                "price_momentum_20",
            ]

        # Define action space
        if self.action_space_type == "discrete":
            self.action_space = spaces.Discrete(3)  # [Hold, Buy, Sell]
        else:  # continuous
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )  # Position weight [-1, 1]

        # Define observation space
        obs_size = (
            self.lookback_window
            + len(self.config.technical_indicators)  # Price history
            + 5  # Technical indicators
            + 4  # Portfolio state (cash, position, pnl, etc.)  # Market state (volatility, trend, etc.)
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic market data with realistic properties"""
        n_steps = self.max_episode_length + self.lookback_window

        # Create price series with regime switching
        returns = self._generate_regime_switching_returns(n_steps)
        prices = self.asset.initial_price * np.cumprod(1 + returns)

        # Generate volumes and other market data
        volumes = self._generate_volumes(n_steps, returns)
        high_low = self._generate_high_low(prices, returns)

        # Create DataFrame
        data = pd.DataFrame(
            {
                f"{self.asset.symbol}_open": prices
                * (1 + np.random.normal(0, 0.001, n_steps)),
                f"{self.asset.symbol}_high": high_low["high"],
                f"{self.asset.symbol}_low": high_low["low"],
                f"{self.asset.symbol}_close": prices,
                f"{self.asset.symbol}_volume": volumes,
                f"{self.asset.symbol}_returns": returns,
            }
        )

        # Calculate technical indicators
        self._calculate_technical_indicators(data)

        return data

    def _generate_regime_switching_returns(self, n_steps: int) -> np.ndarray:
        """Generate returns with regime switching dynamics"""
        # Define market regimes
        regimes = [
            {"mu": 0.0008, "sigma": 0.015, "persistence": 0.95},  # Bull market
            {"mu": -0.0003, "sigma": 0.012, "persistence": 0.90},  # Bear market
            {"mu": 0.0001, "sigma": 0.025, "persistence": 0.85},  # High volatility
            {"mu": 0.0, "sigma": 0.008, "persistence": 0.92},  # Low volatility
        ]

        returns = np.zeros(n_steps)
        current_regime = np.random.randint(0, len(regimes))

        for i in range(n_steps):
            regime = regimes[current_regime]
            returns[i] = np.random.normal(regime["mu"], regime["sigma"])

            # Regime switching
            if np.random.random() > regime["persistence"]:
                current_regime = np.random.randint(0, len(regimes))

        # Add some autocorrelation
        returns = self._add_autocorrelation(returns, rho=0.1)

        return returns

    def _add_autocorrelation(self, returns: np.ndarray, rho: float) -> np.ndarray:
        """Add autocorrelation to returns"""
        correlated_returns = returns.copy()
        for i in range(1, len(returns)):
            correlated_returns[i] = rho * returns[i - 1] + (1 - rho) * returns[i]
        return correlated_returns

    def _generate_volumes(self, n_steps: int, returns: np.ndarray) -> np.ndarray:
        """Generate realistic volume patterns"""
        # Base volume with day-of-week effects
        base_volume = self.asset.avg_daily_volume
        day_effect = 1 + 0.2 * np.sin(
            2 * np.pi * np.arange(n_steps) / 5
        )  # Weekly pattern

        # Volume correlates with absolute returns (higher volume on big moves)
        volume_effect = 1 + 0.5 * np.abs(returns) / np.std(returns)

        # Random component
        random_effect = np.random.lognormal(0, 0.3, n_steps)

        volumes = base_volume * day_effect * volume_effect * random_effect
        return volumes

    def _generate_high_low(
        self, prices: np.ndarray, returns: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Generate realistic high and low prices"""
        # Daily volatility as a proxy for intraday range
        daily_vol = (
            np.abs(returns) * 0.6
        )  # Typical intraday range is smaller than daily range

        high = prices * (1 + daily_vol * np.random.uniform(0.3, 0.7, len(prices)))
        low = prices * (1 - daily_vol * np.random.uniform(0.3, 0.7, len(prices)))

        # Ensure high >= close >= low
        high = np.maximum(high, prices)
        low = np.minimum(low, prices)

        return {"high": high, "low": low}

    def _calculate_technical_indicators(self, data: pd.DataFrame):
        """Calculate technical indicators"""
        close = data[f"{self.asset.symbol}_close"]
        volume = data[f"{self.asset.symbol}_volume"]

        # Simple Moving Averages
        data["sma_10"] = close.rolling(window=10).mean()
        data["sma_30"] = close.rolling(window=30).mean()

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        data["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = close.ewm(span=12).mean()
        exp2 = close.ewm(span=26).mean()
        data["macd"] = exp1 - exp2
        data["macd_signal"] = data["macd"].ewm(span=9).mean()

        # Bollinger Bands
        sma_20 = close.rolling(window=20).mean()
        std_20 = close.rolling(window=20).std()
        data["bollinger_upper"] = sma_20 + 2 * std_20
        data["bollinger_lower"] = sma_20 - 2 * std_20

        # ATR (Average True Range)
        high = data[f"{self.asset.symbol}_high"]
        low = data[f"{self.asset.symbol}_low"]
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        data["atr_14"] = true_range.rolling(window=14).mean()

        # Volume ratio (current volume vs average)
        data["volume_ratio"] = volume / volume.rolling(window=20).mean()

        # Price momentum
        data["price_momentum_5"] = close.pct_change(5)
        data["price_momentum_20"] = close.pct_change(20)

        # Fill NaN values
        data.fillna(method="bfill", inplace=True)
        data.fillna(0, inplace=True)

    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        if self.current_step < self.lookback_window:
            # Not enough history yet, pad with initial values
            price_history = np.concatenate(
                [
                    np.full(
                        self.lookback_window - self.current_step - 1,
                        self.asset.initial_price,
                    ),
                    self.market_data[f"{self.asset.symbol}_close"].values[
                        : self.current_step + 1
                    ],
                ]
            )
        else:
            price_history = self.market_data[f"{self.asset.symbol}_close"].values[
                self.current_step - self.lookback_window + 1 : self.current_step + 1
            ]

        # Normalize price history
        if len(price_history) > 0:
            price_history = (price_history - price_history[0]) / price_history[0]

        # Get current technical indicators
        current_data = self.market_data.iloc[self.current_step]
        technical_values = [
            current_data.get(indicator, 0)
            for indicator in self.config.technical_indicators
        ]

        # Normalize technical indicators
        technical_values = self._normalize_technical_indicators(technical_values)

        # Portfolio state
        portfolio_state = [
            self.cash_balance / self.portfolio_value,  # Cash ratio
            self.positions[self.asset.symbol]
            * self.current_prices[self.asset.symbol]
            / self.portfolio_value,  # Position value ratio
            self.unrealized_pnl / self.initial_cash,  # Unrealized P&L ratio
            self.realized_pnl / self.initial_cash,  # Realized P&L ratio
            len(self.portfolio_history) / self.max_episode_length,  # Time progress
        ]

        # Market state
        if self.current_step > 0:
            recent_returns = self.market_data[f"{self.asset.symbol}_returns"].values[
                max(0, self.current_step - 20) : self.current_step + 1
            ]
            market_state = [
                np.std(recent_returns)
                if len(recent_returns) > 1
                else 0,  # Recent volatility
                np.mean(recent_returns)
                if len(recent_returns) > 0
                else 0,  # Recent trend
                (self.current_prices[self.asset.symbol] - self.asset.initial_price)
                / self.asset.initial_price,  # Cumulative return
                len([r for r in recent_returns if r > 0]) / len(recent_returns)
                if len(recent_returns) > 0
                else 0.5,  # Win rate
            ]
        else:
            market_state = [0, 0, 0, 0.5]

        # Combine all observations
        obs = np.concatenate(
            [price_history, technical_values, portfolio_state, market_state]
        ).astype(np.float32)

        return obs

    def _normalize_technical_indicators(self, indicators: List[float]) -> List[float]:
        """Normalize technical indicators to reasonable ranges"""
        normalized = []

        for i, value in enumerate(indicators):
            indicator_name = self.config.technical_indicators[i]

            if "rsi" in indicator_name:
                # RSI is already 0-100, normalize to -1 to 1
                normalized.append((value - 50) / 50)
            elif "sma" in indicator_name:
                # SMAs: use percentage difference from current price
                current_price = self.current_prices[self.asset.symbol]
                normalized.append((value - current_price) / current_price)
            elif "macd" in indicator_name:
                # MACD: normalize by price
                current_price = self.current_prices[self.asset.symbol]
                normalized.append(value / current_price)
            elif "bollinger" in indicator_name:
                # Bollinger bands: use position relative to bands
                current_price = self.current_prices[self.asset.symbol]
                if "upper" in indicator_name:
                    normalized.append((current_price - value) / current_price)
                else:  # lower
                    normalized.append((value - current_price) / current_price)
            elif "atr" in indicator_name:
                # ATR: normalize by price
                current_price = self.current_prices[self.asset.symbol]
                normalized.append(value / current_price)
            elif "volume" in indicator_name:
                # Volume ratio: log transform
                normalized.append(np.log1p(value) / 10)
            elif "momentum" in indicator_name:
                # Momentum: already a return, just clip
                normalized.append(np.clip(value, -0.1, 0.1))
            else:
                # Generic normalization
                normalized.append(np.tanh(value / 100))

        return normalized

    def _process_action(self, action: Union[np.ndarray, int]) -> Dict[str, Any]:
        """Process action and return execution details"""
        if self.action_space_type == "discrete":
            return self._process_discrete_action(action)
        else:
            return self._process_continuous_action(action[0])

    def _process_discrete_action(self, action: int) -> Dict[str, Any]:
        """Process discrete action [Hold=0, Buy=1, Sell=2]"""
        current_position = self.positions[self.asset.symbol]
        current_price = self.current_prices[self.asset.symbol]
        max_position_value = self.portfolio_value * self.config.leverage_limit

        if action == 0:  # Hold
            return {
                "action": "hold",
                "executed": False,
                "quantity": 0,
                "price": current_price,
                "cost": 0,
                "new_position": current_position,
            }

        elif action == 1:  # Buy
            # Calculate target position (account for transaction costs)
            commission = getattr(self, 'transaction_costs', TransactionCosts()).commission_rate
            max_affordable = self.cash_balance / (1 + commission)
            target_position_value = min(max_position_value, self.portfolio_value, max_affordable)
            target_shares = max(0, int(target_position_value / current_price))
            shares_to_buy = target_shares - current_position

            if shares_to_buy <= 0:
                return {
                    "action": "buy",
                    "executed": False,
                    "quantity": 0,
                    "price": current_price,
                    "cost": 0,
                    "new_position": current_position,
                    "reason": "No shares to buy",
                }

            # Execute trade
            trade_value = shares_to_buy * current_price
            transaction_cost = self._calculate_transaction_costs(trade_value)

            if self.cash_balance >= trade_value + transaction_cost:
                self.positions[self.asset.symbol] += shares_to_buy
                self.cash_balance -= trade_value + transaction_cost
                self.realized_pnl -= transaction_cost

                return {
                    "action": "buy",
                    "executed": True,
                    "quantity": shares_to_buy,
                    "price": current_price,
                    "cost": transaction_cost,
                    "new_position": self.positions[self.asset.symbol],
                }
            else:
                # Try with reduced position
                affordable_value = self.cash_balance / (1 + commission)
                if affordable_value > current_price:
                    reduced_shares = max(0, int(affordable_value / current_price) - current_position)
                    if reduced_shares > 0:
                        trade_value = reduced_shares * current_price
                        transaction_cost = self._calculate_transaction_costs(trade_value)
                        self.positions[self.asset.symbol] += reduced_shares
                        self.cash_balance -= trade_value + transaction_cost
                        self.realized_pnl -= transaction_cost
                        return {
                            "action": "buy",
                            "executed": True,
                            "quantity": reduced_shares,
                            "price": current_price,
                            "cost": transaction_cost,
                            "new_position": self.positions[self.asset.symbol],
                        }
                return {
                    "action": "buy",
                    "executed": False,
                    "quantity": 0,
                    "price": current_price,
                    "cost": 0,
                    "new_position": current_position,
                    "reason": "Insufficient cash",
                }

        elif action == 2:  # Sell
            if current_position <= 0:
                return {
                    "action": "sell",
                    "executed": False,
                    "quantity": 0,
                    "price": current_price,
                    "cost": 0,
                    "new_position": current_position,
                    "reason": "No position to sell",
                }

            # Sell all shares
            shares_to_sell = current_position
            trade_value = shares_to_sell * current_price
            transaction_cost = self._calculate_transaction_costs(
                trade_value, is_short=not self.config.allow_shorting
            )

            self.positions[self.asset.symbol] = 0
            self.cash_balance += trade_value - transaction_cost
            self.realized_pnl -= transaction_cost

            return {
                "action": "sell",
                "executed": True,
                "quantity": shares_to_sell,
                "price": current_price,
                "cost": transaction_cost,
                "new_position": 0,
            }

    def _process_continuous_action(self, action: float) -> Dict[str, Any]:
        """Process continuous action representing target position weight [-1, 1]"""
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)

        # Calculate target position
        current_price = self.current_prices[self.asset.symbol]
        max_position_value = self.portfolio_value * self.config.leverage_limit
        target_position_value = action * max_position_value
        
        # Adjust for transaction costs: target should leave room for commission
        commission_rate = getattr(self, 'transaction_costs', TransactionCosts()).commission_rate
        target_position_value *= (1 - commission_rate)
        
        target_shares = int(target_position_value / current_price)
        shares_to_trade = target_shares - self.positions[self.asset.symbol]

        if abs(shares_to_trade) < 1:  # Minimum trade size
            return {
                "action": "hold",
                "executed": False,
                "quantity": 0,
                "price": current_price,
                "cost": 0,
                "new_position": self.positions[self.asset.symbol],
                "target_weight": action,
            }

        # Execute trade
        trade_value = abs(shares_to_trade) * current_price
        transaction_cost = self._calculate_transaction_costs(
            trade_value,
            is_short=(shares_to_trade < 0 and not self.config.allow_shorting),
        )

        if shares_to_trade > 0:  # Buying
            if self.cash_balance >= trade_value + transaction_cost:
                self.positions[self.asset.symbol] += shares_to_trade
                self.cash_balance -= trade_value + transaction_cost
                self.realized_pnl -= transaction_cost
                executed = True
            else:
                # Try with reduced position if cash is tight
                affordable_value = self.cash_balance - transaction_cost
                if affordable_value > current_price:
                    reduced_shares = int(affordable_value / current_price)
                    if reduced_shares > self.positions[self.asset.symbol] + 1:
                        shares_to_trade = reduced_shares - self.positions[self.asset.symbol]
                        trade_value = shares_to_trade * current_price
                        transaction_cost = self._calculate_transaction_costs(trade_value)
                        self.positions[self.asset.symbol] += shares_to_trade
                        self.cash_balance -= trade_value + transaction_cost
                        self.realized_pnl -= transaction_cost
                        executed = True
                    else:
                        executed = False
                        shares_to_trade = 0
                else:
                    executed = False
                    shares_to_trade = 0
        else:  # Selling
            if abs(shares_to_trade) <= abs(self.positions[self.asset.symbol]):
                self.positions[self.asset.symbol] += shares_to_trade
                self.cash_balance += trade_value - transaction_cost
                self.realized_pnl -= transaction_cost
                executed = True
            else:
                executed = False
                shares_to_trade = 0

        return {
            "action": "rebalance",
            "executed": executed,
            "quantity": shares_to_trade,
            "price": current_price,
            "cost": transaction_cost if executed else 0,
            "new_position": self.positions[self.asset.symbol],
            "target_weight": action,
        }

    def _calculate_reward(self, execution_details: Dict[str, Any]) -> float:
        """Calculate reward based on execution results and portfolio performance"""
        # Base reward: 1-step portfolio return (use last history entry, not second-to-last)
        if len(self.portfolio_history) >= 1:
            portfolio_return = (
                self.portfolio_value - self.portfolio_history[-1]
            ) / max(self.portfolio_history[-1], 1e-8)
        else:
            portfolio_return = 0

        # Transaction cost penalty
        cost_penalty = -execution_details["cost"] / self.portfolio_value

        # Risk-adjusted component - moderate penalty that allows positive returns
        if len(self.portfolio_history) > 10:
            recent_rets = (
                np.diff(self.portfolio_history[-20:]) / self.portfolio_history[-20:-1]
            )
            vol = np.std(recent_rets) if len(recent_rets) > 0 else 0
            volatility_penalty = -0.05 * vol  # Reduced from -0.5 to allow net positive rewards
        else:
            volatility_penalty = 0

        # Position management - penalize excess leverage, not holding positions
        current_position_ratio = (
            abs(
                self.positions[self.asset.symbol]
                * self.current_prices[self.asset.symbol]
            )
            / self.portfolio_value
        )
        # Penalize only positions exceeding available capital (leveraged positions)
        position_penalty = -0.02 * max(
            0, current_position_ratio - 1.0
        )  # Penalize only leverage > 1.0

        # Combine reward components
        total_reward = (
            portfolio_return
            + cost_penalty  # Primary return component
            + volatility_penalty  # Transaction cost penalty
            + position_penalty  # Risk penalty  # Position management penalty
        )

        return float(total_reward)

    def get_action_meanings(self) -> List[str]:
        """Get human-readable action meanings"""
        if self.action_space_type == "discrete":
            return ["Hold", "Buy", "Sell"]
        else:
            return ["Position weight: -1 (short) to +1 (long)"]

    def get_observation_meanings(self) -> List[str]:
        """Get human-readable observation meanings"""
        meanings = []

        # Price history
        for i in range(self.lookback_window):
            meanings.append(f"Price history t-{self.lookback_window-i-1}")

        # Technical indicators
        meanings.extend(self.config.technical_indicators)

        # Portfolio state
        meanings.extend(
            [
                "Cash ratio",
                "Position value ratio",
                "Unrealized P&L ratio",
                "Realized P&L ratio",
                "Time progress",
            ]
        )

        # Market state
        meanings.extend(
            [
                "Recent volatility",
                "Recent trend",
                "Cumulative return",
                "Recent win rate",
            ]
        )

        return meanings
