"""
Base Financial Trading Environment

Abstract base class for all financial trading environments.
Provides common functionality and ensures consistent interface.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

logger = logging.getLogger(__name__)


@dataclass
class AssetConfig:
    """Configuration for a single asset"""

    symbol: str
    name: str
    sector: str
    initial_price: float = 100.0
    volatility: float = 0.02
    drift: float = 0.0001
    market_cap: float = 1e9
    avg_daily_volume: float = 1e6


@dataclass
class TransactionCosts:
    """Transaction cost configuration"""

    commission_rate: float = 0.001  # 10 bps
    slippage_rate: float = 0.0005  # 5 bps
    short_fee_rate: float = 0.0003  # 3 bps per day


@dataclass
class RiskConstraints:
    """Risk management constraints"""

    max_leverage: float = 2.0
    max_position_size: float = 0.3  # 30% of portfolio
    max_sector_exposure: float = 0.4
    var_limit: float = 0.05  # 5% daily VaR
    max_drawdown: float = 0.15


@dataclass
class TradingState:
    """Current trading state information"""

    step: int
    portfolio_value: float
    cash_balance: float
    positions: Dict[str, float]
    unrealized_pnl: float
    realized_pnl: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    var: float


class FinancialTradingBase(gym.Env, ABC):
    """
    Abstract base class for financial trading environments.

    All financial trading environments should inherit from this class
    to ensure consistent interface and shared functionality.
    """

    def __init__(
        self,
        assets: List[AssetConfig],
        initial_cash: float = 1_000_000,
        max_episode_length: int = 252,
        lookback_window: int = 30,
        transaction_costs: TransactionCosts = None,
        risk_constraints: RiskConstraints = None,
        data_source: str = "synthetic",
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # Set seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            self.seed_value = seed
        else:
            self.seed_value = np.random.randint(0, 2**31 - 1)

        # Basic configuration
        self.assets = assets
        self.n_assets = len(assets)
        self.initial_cash = initial_cash
        self.max_episode_length = max_episode_length
        self.lookback_window = lookback_window
        self.data_source = data_source
        self.render_mode = render_mode

        # Use default configs if not provided
        self.transaction_costs = transaction_costs or TransactionCosts()
        self.risk_constraints = risk_constraints or RiskConstraints()

        # Trading state
        self.current_step = 0
        self.portfolio_value = initial_cash
        self.cash_balance = initial_cash
        self.positions = {asset.symbol: 0.0 for asset in assets}
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.portfolio_history = []

        # Market data
        self.market_data = None
        self.current_prices = {asset.symbol: asset.initial_price for asset in assets}

        # Performance tracking
        self.episode_returns = []
        self.risk_metrics_history = []

        # Abstract properties to be defined by subclasses
        self.action_space = None
        self.observation_space = None

        # Initialize environment
        self._initialize_environment()

    @abstractmethod
    def _initialize_environment(self):
        """Initialize environment-specific components"""
        pass

    @abstractmethod
    def _get_observation(self) -> Union[np.ndarray, Dict[str, Any]]:
        """Get current observation"""
        pass

    @abstractmethod
    def _process_action(self, action: Union[np.ndarray, int, Dict]) -> Dict[str, Any]:
        """Process action and return execution details"""
        pass

    @abstractmethod
    def _calculate_reward(self, execution_details: Dict[str, Any]) -> float:
        """Calculate reward based on execution results"""
        pass

    def reset(
        self, seed: Optional[int] = None, **kwargs
    ) -> Tuple[Union[np.ndarray, Dict[str, Any]], Dict[str, Any]]:
        """Reset environment to initial state"""
        if seed is not None:
            np.random.seed(seed)
            self.seed_value = seed

        # Reset state
        self.current_step = 0
        self.portfolio_value = self.initial_cash
        self.cash_balance = self.initial_cash
        self.positions = {asset.symbol: 0.0 for asset in self.assets}
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.portfolio_history = []
        self.episode_returns = []
        self.risk_metrics_history = []

        # Reset current prices
        self.current_prices = {
            asset.symbol: asset.initial_price for asset in self.assets
        }

        # Generate new market data
        self._generate_market_data()

        # Get initial observation
        obs = self._get_observation()

        # Initialize info dictionary
        info = self._get_info()

        return obs, info

    def step(
        self, action: Union[np.ndarray, int, Dict]
    ) -> Tuple[Union[np.ndarray, Dict[str, Any]], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment"""

        # Store previous state
        prev_portfolio_value = self.portfolio_value
        prev_step = self.current_step

        # Process action
        execution_details = self._process_action(action)

        # Advance time
        self.current_step += 1
        self._update_market_data()

        # Update portfolio value
        self._update_portfolio_value()

        # Calculate reward
        reward = self._calculate_reward(execution_details)

        # Check termination
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_episode_length

        # Get observation
        obs = self._get_observation()

        # Update info
        info = self._get_info()

        # Store history
        self.portfolio_history.append(self.portfolio_value)

        return obs, reward, terminated, truncated, info

    def _generate_market_data(self):
        """Load market data for the episode"""
        if self.data_source == "synthetic":
            self.market_data = self._generate_synthetic_data()
        else:
            self.market_data = self._load_real_data()

    def _load_real_data(self) -> pd.DataFrame:
        """Load real market data using the data manager"""
        try:
            # Import data manager
            from data import DataManager

            # Initialize data manager
            data_manager = DataManager(cache_dir="./market_data_cache")

            # Get symbol names from assets
            symbols = [asset.symbol for asset in self.assets]

            # Load data for the past 2 years
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.DateOffset(years=2)

            # Fetch real market data
            market_data = data_manager.get_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                source="yahoo",
                frequency="1d",
            )

            logger.info(f"Loaded real market data: {market_data.shape}")
            return market_data

        except Exception as e:
            logger.error(f"Failed to load real data, falling back to synthetic: {e}")
            return self._generate_synthetic_data()

    @abstractmethod
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic market data - implementation varies by environment"""
        pass

    def _update_market_data(self):
        """Update current prices and market data for next step"""
        if hasattr(self.market_data, "iloc") and self.current_step < len(
            self.market_data
        ):
            # Update current prices from market data
            for i, asset in enumerate(self.assets):
                if f"{asset.symbol}_close" in self.market_data.columns:
                    self.current_prices[asset.symbol] = self.market_data.iloc[
                        self.current_step
                    ][f"{asset.symbol}_close"]

    def _update_portfolio_value(self):
        """Update portfolio value based on current positions and prices"""
        position_value = sum(
            self.positions[asset.symbol] * self.current_prices[asset.symbol]
            for asset in self.assets
        )
        self.portfolio_value = self.cash_balance + position_value
        self.unrealized_pnl = (
            self.portfolio_value - self.initial_cash - self.realized_pnl
        )

    def _calculate_transaction_costs(
        self, trade_value: float, is_short: bool = False
    ) -> float:
        """Calculate transaction costs for a trade"""
        costs = trade_value * self.transaction_costs.commission_rate
        costs += trade_value * self.transaction_costs.slippage_rate

        if is_short:
            # Add short borrowing costs
            costs += (
                trade_value * self.transaction_costs.short_fee_rate / 252
            )  # Daily cost

        return costs

    def _check_risk_constraints(
        self, target_positions: Dict[str, float]
    ) -> Dict[str, bool]:
        """Check if target positions violate risk constraints"""
        constraints_met = {}

        # Calculate total position value
        total_position_value = sum(
            abs(target_positions[asset.symbol]) * self.current_prices[asset.symbol]
            for asset in self.assets
        )

        # Check leverage
        leverage = total_position_value / self.portfolio_value
        constraints_met["leverage"] = leverage <= self.risk_constraints.max_leverage

        # Check individual position sizes
        for asset in self.assets:
            position_value = (
                abs(target_positions[asset.symbol]) * self.current_prices[asset.symbol]
            )
            position_ratio = position_value / self.portfolio_value
            constraints_met[f"position_{asset.symbol}"] = (
                position_ratio <= self.risk_constraints.max_position_size
            )

        # Check portfolio-level risk metrics
        if len(self.portfolio_history) > 30:
            returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
            current_var = np.percentile(returns, 5)
            constraints_met["var"] = abs(current_var) <= self.risk_constraints.var_limit

        return constraints_met

    def _check_termination(self) -> bool:
        """Check if episode should terminate early"""
        # Check if portfolio value is too low
        if self.portfolio_value < self.initial_cash * 0.3:
            return True

        # Check maximum drawdown
        if len(self.portfolio_history) > 0:
            peak = max(self.portfolio_history)
            current_drawdown = (peak - self.portfolio_value) / peak
            if current_drawdown > self.risk_constraints.max_drawdown:
                return True

        return False

    def _get_info(self) -> Dict[str, Any]:
        """Get current environment information"""
        info = {
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "cash_balance": self.cash_balance,
            "total_return": (self.portfolio_value - self.initial_cash)
            / self.initial_cash,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "current_prices": self.current_prices.copy(),
            "positions": self.positions.copy(),
        }

        # Add risk metrics if we have enough history
        if len(self.portfolio_history) > 30:
            returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
            info.update(
                {
                    "volatility": np.std(returns) * np.sqrt(252),
                    "sharpe_ratio": np.mean(returns)
                    / (np.std(returns) + 1e-8)
                    * np.sqrt(252),
                    "max_drawdown": self._calculate_max_drawdown(),
                    "var_5": np.percentile(returns, 5),
                    "var_1": np.percentile(returns, 1),
                }
            )

        return info

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if len(self.portfolio_history) < 2:
            return 0.0

        portfolio_values = np.array(self.portfolio_history)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        return np.max(drawdown)

    def get_portfolio_stats(self) -> Dict[str, float]:
        """Get comprehensive portfolio statistics"""
        if len(self.portfolio_history) < 2:
            return {}

        returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]

        stats = {
            "total_return": (self.portfolio_value - self.initial_cash)
            / self.initial_cash,
            "annualized_return": np.mean(returns) * 252,
            "volatility": np.std(returns) * np.sqrt(252),
            "sharpe_ratio": np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252),
            "max_drawdown": self._calculate_max_drawdown(),
            "calmar_ratio": (np.mean(returns) * 252)
            / (self._calculate_max_drawdown() + 1e-8),
            "var_5": np.percentile(returns, 5),
            "var_1": np.percentile(returns, 1),
            "expected_shortfall_5": np.mean(
                returns[returns <= np.percentile(returns, 5)]
            ),
            "positive_days": np.mean(returns > 0),
            "average_win": np.mean(returns[returns > 0]) if np.any(returns > 0) else 0,
            "average_loss": np.mean(returns[returns < 0]) if np.any(returns < 0) else 0,
            "profit_factor": abs(
                np.mean(returns[returns > 0]) / np.mean(returns[returns < 0])
            )
            if np.any(returns < 0)
            else float("in"),
        }

        return stats

    def render(self):
        """Render the environment (optional)"""
        if self.render_mode == "human":
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(
                f"Total Return: {(self.portfolio_value - self.initial_cash) / self.initial_cash:.2%}"
            )
            print(f"Positions: {self.positions}")
            print(f"Current Prices: {self.current_prices}")
            print("-" * 50)

    def seed(self, seed: Optional[int] = None):
        """Set random seed for reproducibility"""
        if seed is not None:
            self.seed_value = seed
        else:
            self.seed_value = np.random.randint(0, 2**31 - 1)

        np.random.seed(self.seed_value)
        return [self.seed_value]
