"""
Portfolio Optimization Environment

A multi-asset portfolio optimization environment for reinforcement learning.
The agent learns to allocate capital across multiple assets to optimize risk-adjusted returns.

Action Space: Box - Portfolio weights for each asset
Observation Space: Box - Asset returns, correlations, technical indicators, portfolio state
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
class PortfolioConfig:
    """Configuration specific to portfolio optimization"""

    rebalance_frequency: int = 5  # Rebalance every N days
    lookback_returns: int = 60  # Use last N days for optimization
    correlation_window: int = 30  # Window for correlation calculation
    sector_constraints: Dict[str, float] = None  # Max sector exposure
    max_turnover: float = 0.5  # Maximum daily turnover
    benchmark_weights: Dict[str, float] = None  # Benchmark portfolio for tracking
    objective: str = "sharpe"  # "sharpe", "return", "risk_parity", "equal_weight"


class PortfolioOptimizationEnv(FinancialTradingBase):
    """
    Portfolio Optimization Environment

    A gymnasium environment for multi-asset portfolio optimization.
    The agent learns to allocate capital across multiple assets to achieve
    optimal risk-adjusted returns while respecting various constraints.

    Features:
    - Multi-asset allocation with realistic constraints
    - Risk management and sector exposure limits
    - Transaction cost and turnover management
    - Multiple objective functions
    - Benchmark tracking capabilities
    - Regime-aware optimization
    """

    def __init__(
        self,
        assets: List[AssetConfig],
        initial_cash: float = 1_000_000,
        max_episode_length: int = 252,
        lookback_window: int = 60,
        transaction_costs: TransactionCosts = None,
        risk_constraints: RiskConstraints = None,
        config: PortfolioConfig = None,
        rebalance_mode: str = "discrete",  # "discrete" or "continuous"
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        # Store configuration
        self.config = config or PortfolioConfig()
        self.rebalance_mode = rebalance_mode
        self.last_rebalance_step = -self.config.rebalance_frequency

        # Create sector mapping if not provided
        self.sector_mapping = {asset.symbol: asset.sector for asset in assets}
        self.unique_sectors = list(set(asset.sector for asset in assets))

        # Initialize benchmark weights
        if self.config.benchmark_weights is None:
            self.config.benchmark_weights = {
                asset.symbol: 1.0 / len(assets) for asset in assets
            }

        # Initialize base environment
        super().__init__(
            assets=assets,
            initial_cash=initial_cash,
            max_episode_length=max_episode_length,
            lookback_window=lookback_window,
            transaction_costs=transaction_costs,
            risk_constraints=risk_constraints,
            seed=seed,
            render_mode=render_mode,
        )

        # Portfolio tracking
        self.target_weights = {asset.symbol: 0.0 for asset in assets}
        self.current_weights = {asset.symbol: 0.0 for asset in assets}
        self.weight_history = []
        self.turnover_history = []

    def _initialize_environment(self):
        """Initialize environment-specific components"""
        # Action space: portfolio weights for each asset
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )

        # Calculate observation space size
        obs_size = (
            self.n_assets * self.lookback_window
            + self.n_assets * (self.n_assets - 1) // 2  # Return histories
            + self.n_assets * 5  # Correlation matrix (upper triangle)
            + self.n_assets  # Asset-specific indicators
            + len(self.unique_sectors)  # Current weights
            + 10  # Sector exposures  # Portfolio and market metrics
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic market data for multiple assets"""
        n_steps = self.max_episode_length + self.lookback_window

        # Generate correlated returns with regime switching
        returns_matrix = self._generate_correlated_returns(n_steps)

        # Create price series
        prices_matrix = np.zeros_like(returns_matrix)
        for i, asset in enumerate(self.assets):
            prices_matrix[:, i] = asset.initial_price * np.cumprod(
                1 + returns_matrix[:, i]
            )

        # Generate volumes
        volumes_matrix = self._generate_volumes_matrix(n_steps, returns_matrix)

        # Create DataFrame
        data = pd.DataFrame()
        for i, asset in enumerate(self.assets):
            prefix = f"{asset.symbol}_"
            data[f"{prefix}open"] = prices_matrix[:, i] * (
                1 + np.random.normal(0, 0.001, n_steps)
            )
            data[f"{prefix}high"] = prices_matrix[:, i] * (
                1 + np.abs(np.random.normal(0, 0.01, n_steps))
            )
            data[f"{prefix}low"] = prices_matrix[:, i] * (
                1 - np.abs(np.random.normal(0, 0.01, n_steps))
            )
            data[f"{prefix}close"] = prices_matrix[:, i]
            data[f"{prefix}volume"] = volumes_matrix[:, i]
            data[f"{prefix}returns"] = returns_matrix[:, i]

        # Calculate additional features
        self._calculate_portfolio_features(data)

        return data

    def _generate_correlated_returns(self, n_steps: int) -> np.ndarray:
        """Generate correlated returns with realistic properties"""
        # Define regimes with different correlation structures
        regimes = [
            {
                "name": "normal",
                "correlation_level": 0.3,
                "volatility_multiplier": 1.0,
                "persistence": 0.95,
            },
            {
                "name": "crisis",
                "correlation_level": 0.8,
                "volatility_multiplier": 2.5,
                "persistence": 0.90,
            },
            {
                "name": "recovery",
                "correlation_level": 0.2,
                "volatility_multiplier": 0.8,
                "persistence": 0.85,
            },
        ]

        returns = np.zeros((n_steps, self.n_assets))
        current_regime = 0

        for t in range(n_steps):
            regime = regimes[current_regime]

            # Create correlation matrix for current regime
            corr_matrix = self._create_correlation_matrix(regime["correlation_level"])

            # Generate base returns with regime-specific volatility
            base_vol = np.array([asset.volatility for asset in self.assets])
            regime_vol = base_vol * regime["volatility_multiplier"]
            regime_drift = np.array([asset.drift for asset in self.assets])

            # Generate correlated returns
            cov_matrix = np.diag(regime_vol) @ corr_matrix @ np.diag(regime_vol)
            returns[t] = np.random.multivariate_normal(regime_drift, cov_matrix)

            # Regime switching
            if np.random.random() > regime["persistence"]:
                current_regime = np.random.choice(len(regimes))

        # Add some momentum and mean reversion
        returns = self._add_momentum_effects(returns)
        returns = self._add_mean_reversion(returns)

        return returns

    def _create_correlation_matrix(self, base_correlation: float) -> np.ndarray:
        """Create realistic correlation matrix"""
        # Start with base correlation
        corr_matrix = np.full((self.n_assets, self.n_assets), base_correlation)
        np.fill_diagonal(corr_matrix, 1.0)

        # Add sector-based correlation adjustments
        for i, asset_i in enumerate(self.assets):
            for j, asset_j in enumerate(self.assets):
                if i != j:
                    if asset_i.sector == asset_j.sector:
                        # Same sector: higher correlation
                        corr_matrix[i, j] = min(0.9, base_correlation + 0.3)
                    else:
                        # Different sectors: lower correlation
                        corr_matrix[i, j] = max(0.1, base_correlation - 0.1)

        # Ensure positive semi-definite
        eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
        eigenvals = np.maximum(eigenvals, 0.01)
        corr_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        return corr_matrix

    def _add_momentum_effects(
        self, returns: np.ndarray, momentum_strength: float = 0.1
    ) -> np.ndarray:
        """Add momentum effects to returns"""
        momentum_returns = returns.copy()
        for t in range(1, len(returns)):
            momentum_returns[t] += momentum_strength * returns[t - 1]
        return momentum_returns

    def _add_mean_reversion(
        self, returns: np.ndarray, reversion_strength: float = 0.05
    ) -> np.ndarray:
        """Add mean reversion effects"""
        reversion_returns = returns.copy()
        for t in range(2, len(returns)):
            # Simple mean reversion: opposite sign of recent cumulative return
            recent_cumulative = np.sum(returns[t - 2 : t])
            reversion_returns[t] -= reversion_strength * recent_cumulative
        return reversion_returns

    def _generate_volumes_matrix(self, n_steps: int, returns: np.ndarray) -> np.ndarray:
        """Generate volume matrix for all assets"""
        volumes = np.zeros((n_steps, self.n_assets))

        for i, asset in enumerate(self.assets):
            base_volume = asset.avg_daily_volume

            # Volume correlates with absolute returns and has autocorrelation
            volume_effect = 1 + 0.5 * np.abs(returns[:, i]) / np.std(returns[:, i])
            autocorrelation = np.random.lognormal(0, 0.2, n_steps)

            # Apply exponential smoothing for autocorrelation
            for t in range(1, n_steps):
                autocorrelation[t] = (
                    0.7 * autocorrelation[t - 1] + 0.3 * autocorrelation[t]
                )

            volumes[:, i] = base_volume * volume_effect * autocorrelation

        return volumes

    def _calculate_portfolio_features(self, data: pd.DataFrame):
        """Calculate portfolio-level features"""
        # Calculate rolling correlations
        for window in [10, 30, 60]:
            if len(data) >= window:
                for i, asset_i in enumerate(self.assets):
                    for j, asset_j in enumerate(self.assets):
                        if i < j:
                            returns_i = data[f"{asset_i.symbol}_returns"]
                            returns_j = data[f"{asset_j.symbol}_returns"]
                            corr_name = (
                                f"corr_{asset_i.symbol}_{asset_j.symbol}_{window}d"
                            )
                            data[corr_name] = returns_i.rolling(window=window).corr(
                                returns_j
                            )

        # Calculate sector indices
        for sector in self.unique_sectors:
            sector_assets = [
                asset.symbol for asset in self.assets if asset.sector == sector
            ]
            if sector_assets:
                sector_returns = data[
                    [f"{symbol}_returns" for symbol in sector_assets]
                ].mean(axis=1)
                data[f"sector_{sector}_return"] = sector_returns

        # Calculate market factors
        all_returns = data[[f"{asset.symbol}_returns" for asset in self.assets]].mean(
            axis=1
        )
        data["market_return"] = all_returns
        data["market_volatility"] = all_returns.rolling(window=30).std()

    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        obs_components = []

        # 1. Return histories for each asset
        for asset in self.assets:
            if self.current_step >= self.lookback_window:
                returns_window = self.market_data[f"{asset.symbol}_returns"].values[
                    self.current_step - self.lookback_window : self.current_step
                ]
            else:
                # Pad with zeros if not enough history
                returns_window = np.zeros(self.lookback_window)
                available_returns = self.market_data[f"{asset.symbol}_returns"].values[
                    : self.current_step
                ]
                returns_window[-len(available_returns) :] = available_returns

            obs_components.extend(returns_window)

        # 2. Correlation matrix (upper triangle)
        correlation_matrix = self._calculate_current_correlations()
        for i in range(self.n_assets):
            for j in range(i + 1, self.n_assets):
                obs_components.append(correlation_matrix[i, j])

        # 3. Asset-specific indicators
        for asset in self.assets:
            current_price = self.current_prices[asset.symbol]

            # Recent returns
            recent_returns = self.market_data[f"{asset.symbol}_returns"].values[
                max(0, self.current_step - 20) : self.current_step + 1
            ]

            indicators = [
                np.mean(recent_returns)
                if len(recent_returns) > 0
                else 0,  # Mean return
                np.std(recent_returns) if len(recent_returns) > 1 else 0,  # Volatility
                (current_price - asset.initial_price)
                / asset.initial_price,  # Cumulative return
                self.market_data[f"{asset.symbol}_volume"].iloc[self.current_step]
                / asset.avg_daily_volume,  # Volume ratio
                self._calculate_asset_momentum(asset.symbol),  # Momentum indicator
            ]
            obs_components.extend(indicators)

        # 4. Current portfolio weights
        current_weights_array = np.array(
            [self.current_weights.get(asset.symbol, 0) for asset in self.assets]
        )
        obs_components.extend(current_weights_array)

        # 5. Sector exposures
        for sector in self.unique_sectors:
            sector_exposure = sum(
                self.current_weights.get(asset.symbol, 0)
                for asset in self.assets
                if asset.sector == sector
            )
            obs_components.append(sector_exposure)

        # 6. Portfolio and market metrics
        portfolio_metrics = [
            self._calculate_portfolio_return(),  # Portfolio return
            self._calculate_portfolio_volatility(),  # Portfolio volatility
            self._calculate_sharpe_ratio(),  # Sharpe ratio
            self._calculate_max_drawdown(),  # Max drawdown
            self._calculate_tracking_error(),  # Tracking error vs benchmark
            self._calculate_information_ratio(),  # Information ratio
            self._calculate_turnover(),  # Recent turnover
            len(self.portfolio_history) / self.max_episode_length,  # Time progress
            self._calculate_concentration(),  # Portfolio concentration
            self._calculate_liquidity_score(),  # Liquidity score
        ]
        obs_components.extend(portfolio_metrics)

        return np.array(obs_components, dtype=np.float32)

    def _calculate_current_correlations(self) -> np.ndarray:
        """Calculate current correlation matrix"""
        window = min(self.config.correlation_window, self.current_step + 1)

        if window < 2:
            return np.eye(self.n_assets)

        returns_matrix = np.zeros((window, self.n_assets))
        for i, asset in enumerate(self.assets):
            returns_data = self.market_data[f"{asset.symbol}_returns"].values[
                max(0, self.current_step - window + 1) : self.current_step + 1
            ]
            returns_matrix[:, i] = returns_data

        correlation_matrix = np.corrcoef(returns_matrix.T)

        # Handle NaN values
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        np.fill_diagonal(correlation_matrix, 1.0)

        return correlation_matrix

    def _calculate_asset_momentum(self, symbol: str) -> float:
        """Calculate momentum indicator for an asset"""
        returns = self.market_data[f"{symbol}_returns"].values[
            max(0, self.current_step - 20) : self.current_step + 1
        ]

        if len(returns) < 5:
            return 0.0

        # Simple momentum: average of recent returns
        return np.mean(returns)

    def _calculate_portfolio_return(self) -> float:
        """Calculate current portfolio return"""
        if len(self.portfolio_history) < 2:
            return 0.0
        return (
            self.portfolio_value - self.portfolio_history[-2]
        ) / self.portfolio_history[-2]

    def _calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio volatility"""
        if len(self.portfolio_history) < 30:
            return 0.0

        returns = np.diff(self.portfolio_history[-30:]) / self.portfolio_history[-30:-1]
        return np.std(returns)

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        if len(self.portfolio_history) < 30:
            return 0.0

        returns = np.diff(self.portfolio_history[-30:]) / self.portfolio_history[-30:-1]
        return np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)

    def _calculate_tracking_error(self) -> float:
        """Calculate tracking error relative to benchmark"""
        if len(self.portfolio_history) < 30:
            return 0.0

        portfolio_returns = (
            np.diff(self.portfolio_history[-30:]) / self.portfolio_history[-30:-1]
        )

        # Calculate benchmark returns
        benchmark_return = 0.0
        for asset in self.assets:
            asset_returns = self.market_data[f"{asset.symbol}_returns"].values[
                max(0, self.current_step - 30) : self.current_step + 1
            ]
            benchmark_weight = self.config.benchmark_weights.get(asset.symbol, 0)
            if len(asset_returns) == len(portfolio_returns):
                benchmark_return += (
                    benchmark_weight * asset_returns[-len(portfolio_returns) :]
                )

        tracking_error = np.std(portfolio_returns - benchmark_return)
        return tracking_error

    def _calculate_information_ratio(self) -> float:
        """Calculate information ratio"""
        tracking_error = self._calculate_tracking_error()
        if tracking_error < 1e-8:
            return 0.0

        portfolio_returns = (
            np.diff(self.portfolio_history[-30:]) / self.portfolio_history[-30:-1]
        )

        # Calculate benchmark returns (same as in tracking error)
        benchmark_return = 0.0
        for asset in self.assets:
            asset_returns = self.market_data[f"{asset.symbol}_returns"].values[
                max(0, self.current_step - 30) : self.current_step + 1
            ]
            benchmark_weight = self.config.benchmark_weights.get(asset.symbol, 0)
            if len(asset_returns) == len(portfolio_returns):
                benchmark_return += (
                    benchmark_weight * asset_returns[-len(portfolio_returns) :]
                )

        excess_return = np.mean(portfolio_returns - benchmark_return)
        return excess_return / tracking_error * np.sqrt(252)

    def _calculate_turnover(self) -> float:
        """Calculate recent portfolio turnover"""
        if len(self.weight_history) < 2:
            return 0.0

        recent_weights = self.weight_history[-10:]  # Last 10 rebalances
        turnover_sum = 0.0

        for i in range(1, len(recent_weights)):
            weight_change = sum(
                abs(
                    recent_weights[i].get(asset.symbol, 0)
                    - recent_weights[i - 1].get(asset.symbol, 0)
                )
                for asset in self.assets
            )
            turnover_sum += weight_change

        return turnover_sum / len(recent_weights) if len(recent_weights) > 1 else 0.0

    def _calculate_concentration(self) -> float:
        """Calculate portfolio concentration (Herfindahl index)"""
        weights = np.array(
            [self.current_weights.get(asset.symbol, 0) for asset in self.assets]
        )
        return np.sum(weights**2)

    def _calculate_liquidity_score(self) -> float:
        """Calculate portfolio liquidity score"""
        liquidity_score = 0.0
        total_weight = sum(self.current_weights.values())

        if total_weight > 0:
            for asset in self.assets:
                weight = self.current_weights.get(asset.symbol, 0)
                # Use average daily volume as liquidity proxy
                liquidity = asset.avg_daily_volume * self.current_prices[asset.symbol]
                asset_liquidity_score = weight * np.log1p(
                    liquidity / 1e6
                )  # Normalize by $1M
                liquidity_score += asset_liquidity_score

            liquidity_score /= total_weight

        return liquidity_score

    def _process_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Process portfolio rebalancing action"""
        # Check if rebalancing is allowed
        if (
            self.rebalance_mode == "discrete"
            and self.current_step - self.last_rebalance_step
            < self.config.rebalance_frequency
        ):
            return {
                "executed": False,
                "reason": "Rebalancing not allowed yet",
                "target_weights": self.current_weights.copy(),
                "execution_cost": 0.0,
            }

        # Normalize action to ensure weights sum to 1
        normalized_weights = np.abs(action)  # Use absolute values
        if np.sum(normalized_weights) > 0:
            normalized_weights = normalized_weights / np.sum(normalized_weights)
        else:
            normalized_weights = np.ones(self.n_assets) / self.n_assets

        # Apply constraints
        constrained_weights = self._apply_portfolio_constraints(normalized_weights)

        # Convert to dictionary
        target_weights_dict = {
            asset.symbol: constrained_weights[i] for i, asset in enumerate(self.assets)
        }

        # Check if rebalancing is worthwhile (minimum threshold)
        current_total_weight = sum(self.current_weights.values())
        weight_change = sum(
            abs(
                target_weights_dict[asset.symbol]
                - self.current_weights.get(asset.symbol, 0)
            )
            for asset in self.assets
        )

        min_rebalance_threshold = 0.1  # 10% minimum change
        if weight_change < min_rebalance_threshold and current_total_weight > 0:
            return {
                "executed": False,
                "reason": "Insufficient weight change",
                "target_weights": self.current_weights.copy(),
                "execution_cost": 0.0,
            }

        # Execute rebalancing
        execution_details = self._execute_rebalancing(target_weights_dict)
        self.last_rebalance_step = self.current_step

        return execution_details

    def _apply_portfolio_constraints(self, weights: np.ndarray) -> np.ndarray:
        """Apply portfolio constraints to weights"""
        constrained_weights = weights.copy()

        # Apply sector constraints
        if self.config.sector_constraints:
            for sector, max_exposure in self.config.sector_constraints.items():
                sector_indices = [
                    i for i, asset in enumerate(self.assets) if asset.sector == sector
                ]
                if sector_indices:
                    sector_total = sum(constrained_weights[i] for i in sector_indices)
                    if sector_total > max_exposure:
                        # Scale down sector weights proportionally
                        scale_factor = max_exposure / sector_total
                        for i in sector_indices:
                            constrained_weights[i] *= scale_factor

        # Apply position size constraints
        max_position = self.risk_constraints.max_position_size
        constrained_weights = np.clip(constrained_weights, 0, max_position)

        # Re-normalize weights
        total_weight = np.sum(constrained_weights)
        if total_weight > 0:
            constrained_weights = constrained_weights / total_weight

        return constrained_weights

    def _execute_rebalancing(self, target_weights: Dict[str, float]) -> Dict[str, Any]:
        """Execute portfolio rebalancing"""
        total_execution_cost = 0.0
        executed_trades = {}

        # Store current weights for history
        self.weight_history.append(self.current_weights.copy())

        # Execute trades for each asset
        for asset in self.assets:
            current_weight = self.current_weights.get(asset.symbol, 0)
            target_weight = target_weights[asset.symbol]
            current_price = self.current_prices[asset.symbol]

            # Calculate target position value
            target_value = target_weight * self.portfolio_value
            target_shares = int(target_value / current_price)

            # Calculate trade required
            current_shares = self.positions.get(asset.symbol, 0)
            shares_to_trade = target_shares - current_shares

            if abs(shares_to_trade) > 0:
                trade_value = abs(shares_to_trade) * current_price
                transaction_cost = self._calculate_transaction_costs(trade_value)

                # Execute trade if sufficient cash for buys
                if shares_to_trade > 0:  # Buy
                    required_cash = trade_value + transaction_cost
                    if self.cash_balance >= required_cash:
                        self.positions[asset.symbol] = target_shares
                        self.cash_balance -= required_cash
                        total_execution_cost += transaction_cost
                        executed_trades[asset.symbol] = shares_to_trade
                    else:
                        # Adjust target to available cash
                        affordable_shares = int(
                            (self.cash_balance - transaction_cost) / current_price
                        )
                        if affordable_shares > current_shares:
                            actual_trade = affordable_shares - current_shares
                            actual_value = actual_trade * current_price
                            actual_cost = self._calculate_transaction_costs(
                                actual_value
                            )
                            self.positions[asset.symbol] = affordable_shares
                            self.cash_balance -= actual_value + actual_cost
                            total_execution_cost += actual_cost
                            executed_trades[asset.symbol] = actual_trade
                            target_weight = (
                                affordable_shares * current_price
                            ) / self.portfolio_value

                else:  # Sell
                    self.positions[asset.symbol] = target_shares
                    self.cash_balance += trade_value - transaction_cost
                    total_execution_cost += transaction_cost
                    executed_trades[asset.symbol] = shares_to_trade

            # Update weight
            self.current_weights[asset.symbol] = target_weight

        # Update portfolio value
        self._update_portfolio_value()

        # Update turnover history
        turnover = sum(abs(trade) for trade in executed_trades.values())
        self.turnover_history.append(turnover)

        return {
            "executed": True,
            "target_weights": self.current_weights.copy(),
            "executed_trades": executed_trades,
            "execution_cost": total_execution_cost,
            "turnover": turnover,
        }

    def _calculate_reward(self, execution_details: Dict[str, Any]) -> float:
        """Calculate reward based on portfolio performance and execution"""
        if not execution_details.get("executed", False):
            return 0.0

        # Base reward: portfolio return
        portfolio_return = self._calculate_portfolio_return()

        # Risk-adjusted component
        portfolio_volatility = self._calculate_portfolio_volatility()
        risk_adjusted_return = portfolio_return - 0.5 * portfolio_volatility

        # Transaction cost penalty
        cost_penalty = -execution_details["execution_cost"] / self.portfolio_value

        # Turnover penalty
        turnover_penalty = -0.1 * execution_details.get("turnover", 0)

        # Tracking error component (if we want to track a benchmark)
        tracking_error_penalty = -0.2 * self._calculate_tracking_error()

        # Concentration penalty
        concentration_penalty = -0.1 * self._calculate_concentration()

        # Combine all components
        total_reward = (
            risk_adjusted_return
            + cost_penalty
            + turnover_penalty
            + tracking_error_penalty
            + concentration_penalty
        )

        return float(total_reward)

    def get_observation_meanings(self) -> List[str]:
        """Get human-readable observation meanings"""
        meanings = []

        # Return histories
        for asset in self.assets:
            for i in range(self.lookback_window):
                meanings.append(f"{asset.symbol} return t-{self.lookback_window-i}")

        # Correlations
        for i, asset_i in enumerate(self.assets):
            for j, asset_j in enumerate(self.assets):
                if i < j:
                    meanings.append(f"Correlation {asset_i.symbol}-{asset_j.symbol}")

        # Asset indicators
        for asset in self.assets:
            meanings.extend(
                [
                    f"{asset.symbol} mean return",
                    f"{asset.symbol} volatility",
                    f"{asset.symbol} cumulative return",
                    f"{asset.symbol} volume ratio",
                    f"{asset.symbol} momentum",
                ]
            )

        # Current weights
        for asset in self.assets:
            meanings.append(f"{asset.symbol} weight")

        # Sector exposures
        for sector in self.unique_sectors:
            meanings.append(f"{sector} sector exposure")

        # Portfolio metrics
        meanings.extend(
            [
                "Portfolio return",
                "Portfolio volatility",
                "Sharpe ratio",
                "Max drawdown",
                "Tracking error",
                "Information ratio",
                "Turnover",
                "Time progress",
                "Concentration",
                "Liquidity score",
            ]
        )

        return meanings
