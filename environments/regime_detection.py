"""
Regime Detection Environment

A financial environment where the agent must identify and adapt to different market regimes.
The agent learns to recognize market conditions and adjust trading strategies accordingly.

Action Space: Discrete - Regime prediction and trading action
Observation Space: Box - Market indicators, regime features, and portfolio state
"""

from dataclasses import dataclass
from enum import Enum
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


class MarketRegime(Enum):
    """Market regime types"""

    BULL = 0  # Rising prices, low volatility
    BEAR = 1  # Falling prices, high volatility
    SIDEWAYS = 2  # Range-bound prices
    VOLATILE = 3  # High volatility, direction unclear
    CRISIS = 4  # Extreme stress, correlations increase


@dataclass
class RegimeConfig:
    """Configuration for regime detection environment"""

    regime_types: List[MarketRegime] = None
    regime_persistence: float = 0.95  # Probability of staying in same regime
    prediction_horizon: int = 20  # Steps ahead to predict
    detection_delay: int = 5  # Delay in regime confirmation
    regime_reward_multiplier: float = 2.0  # Bonus for correct regime prediction
    adaptability_requirement: bool = True  # Must adapt strategy to regime
    hidden_regime_probability: float = 0.3  # Probability regime is hidden


class RegimeDetectionEnv(FinancialTradingBase):
    """
    Regime Detection Environment

    A gymnasium environment that challenges agents to identify and adapt to
    different market regimes. The agent must both predict the current regime
    and implement appropriate trading strategies.

    Features:
    - Hidden Markov Model for regime generation
    - Partial observability of true regime
    - Multi-step regime prediction challenges
    - Strategy adaptation requirements
    - Regime-specific reward structures
    """

    def __init__(
        self,
        assets: List[AssetConfig],
        initial_cash: float = 1_000_000,
        max_episode_length: int = 504,  # Longer for regime detection
        lookback_window: int = 60,
        transaction_costs: TransactionCosts = None,
        risk_constraints: RiskConstraints = None,
        config: RegimeConfig = None,
        action_mode: str = "combined",  # "combined", "separate", "prediction_only"
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        # Store configuration
        self.config = config or RegimeConfig()
        self.action_mode = action_mode

        if self.config.regime_types is None:
            self.config.regime_types = list(MarketRegime)

        self.n_regimes = len(self.config.regime_types)

        # Regime tracking
        self.true_regime_history = []
        self.predicted_regime_history = []
        self.regime_transition_matrix = self._create_regime_transition_matrix()
        self.current_regime = None
        self.regime_start_step = 0
        self.regime_confirmed = False
        self.regime_confirmation_delay = self.config.detection_delay

        # Performance tracking by regime
        self.regime_performance = {regime: [] for regime in self.config.regime_types}
        self.prediction_accuracy = []

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

    def _initialize_environment(self):
        """Initialize environment-specific components"""
        # Define action space based on mode
        if self.action_mode == "combined":
            # Combined action: [regime_prediction, trading_action]
            self.action_space = spaces.MultiDiscrete(
                [self.n_regimes, 3]
            )  # [regime, hold/buy/sell]
        elif self.action_mode == "separate":
            # Separate spaces (used in different ways)
            self.action_space = spaces.Tuple(
                (
                    spaces.Discrete(self.n_regimes),  # Regime prediction
                    spaces.Discrete(3),  # Trading action
                )
            )
        else:  # prediction_only
            self.action_space = spaces.Discrete(self.n_regimes)

        # Calculate observation space size
        obs_size = (
            self.n_assets * self.lookback_window
            + self.n_assets * 10  # Price histories
            + 20  # Technical indicators per asset
            + 10  # Regime-specific features
            + 5  # Market macro indicators
            + self.n_regimes  # Portfolio state
            + 10  # Regime prediction history  # Additional features
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

    def _create_regime_transition_matrix(self) -> np.ndarray:
        """Create regime transition probability matrix"""
        n = self.n_regimes
        transition_matrix = np.zeros((n, n))

        # Default: high diagonal (persistence), low off-diagonal transitions
        for i in range(n):
            transition_matrix[i, i] = self.config.regime_persistence
            remaining_prob = 1 - self.config.regime_persistence
            for j in range(n):
                if i != j:
                    transition_matrix[i, j] = remaining_prob / (n - 1)

        # Add some realistic regime transition patterns
        if (
            MarketRegime.BULL in self.config.regime_types
            and MarketRegime.BEAR in self.config.regime_types
        ):
            bull_idx = self.config.regime_types.index(MarketRegime.BULL)
            bear_idx = self.config.regime_types.index(MarketRegime.BEAR)
            # Slightly higher probability of bull -> bear transitions
            transition_matrix[bull_idx, bear_idx] *= 1.5
            transition_matrix[bull_idx, bull_idx] *= 0.9
            # Normalize row
            transition_matrix[bull_idx] /= transition_matrix[bull_idx].sum()

        return transition_matrix

    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic market data with regime-dependent characteristics"""
        n_steps = self.max_episode_length + self.lookback_window

        # Generate regime sequence
        regime_sequence = self._generate_regime_sequence(n_steps)

        # Generate returns for each regime
        returns_matrix = np.zeros((n_steps, self.n_assets))
        volatilities = np.zeros((n_steps, self.n_assets))

        for t in range(n_steps):
            regime = self.config.regime_types[regime_sequence[t]]
            asset_returns, asset_vols = self._generate_regime_returns(
                regime, self.n_assets
            )
            returns_matrix[t] = asset_returns
            volatilities[t] = asset_vols

        # Create price series
        prices_matrix = np.zeros_like(returns_matrix)
        for i, asset in enumerate(self.assets):
            prices_matrix[:, i] = asset.initial_price * np.cumprod(
                1 + returns_matrix[:, i]
            )

        # Generate additional market data
        volumes_matrix = self._generate_regime_volumes(n_steps, regime_sequence)
        correlations = self._generate_regime_correlations(n_steps, regime_sequence)

        # Create DataFrame
        data = pd.DataFrame()
        for i, asset in enumerate(self.assets):
            prefix = f"{asset.symbol}_"
            data[f"{prefix}close"] = prices_matrix[:, i]
            data[f"{prefix}returns"] = returns_matrix[:, i]
            data[f"{prefix}volume"] = volumes_matrix[:, i]
            data[f"{prefix}volatility"] = volatilities[:, i]

        # Add regime information (may be hidden from agent)
        data["true_regime"] = regime_sequence
        data["regime_volatility"] = np.mean(volatilities, axis=1)
        data["regime_correlation"] = correlations

        # Calculate regime-specific features
        self._calculate_regime_features(data)

        return data

    def _generate_regime_sequence(self, n_steps: int) -> np.ndarray:
        """Generate sequence of market regimes using Hidden Markov Model"""
        regime_sequence = np.zeros(n_steps, dtype=int)
        current_regime = np.random.randint(self.n_regimes)

        for t in range(n_steps):
            regime_sequence[t] = current_regime

            # Transition to next regime
            if np.random.random() > self.config.regime_persistence:
                probabilities = self.regime_transition_matrix[current_regime]
                current_regime = np.random.choice(self.n_regimes, p=probabilities)

        return regime_sequence

    def _generate_regime_returns(
        self, regime: MarketRegime, n_assets: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate asset returns and volatilities for a specific regime"""
        if regime == MarketRegime.BULL:
            # Positive returns, low volatility
            mean_return = np.random.normal(0.001, 0.0005, n_assets)
            volatility = np.random.uniform(0.01, 0.02, n_assets)
        elif regime == MarketRegime.BEAR:
            # Negative returns, high volatility
            mean_return = np.random.normal(-0.001, 0.0005, n_assets)
            volatility = np.random.uniform(0.02, 0.04, n_assets)
        elif regime == MarketRegime.SIDEWAYS:
            # Near-zero returns, low volatility
            mean_return = np.random.normal(0.0, 0.0002, n_assets)
            volatility = np.random.uniform(0.008, 0.015, n_assets)
        elif regime == MarketRegime.VOLATILE:
            # Mixed returns, very high volatility
            mean_return = np.random.normal(0.0, 0.002, n_assets)
            volatility = np.random.uniform(0.03, 0.06, n_assets)
        elif regime == MarketRegime.CRISIS:
            # Strong negative returns, extreme volatility
            mean_return = np.random.normal(-0.003, 0.001, n_assets)
            volatility = np.random.uniform(0.05, 0.10, n_assets)
        else:
            # Default regime
            mean_return = np.random.normal(0.0001, 0.0005, n_assets)
            volatility = np.random.uniform(0.015, 0.025, n_assets)

        # Generate correlated returns
        correlation_level = self._get_regime_correlation_level(regime)
        correlation_matrix = self._create_regime_correlation_matrix(
            correlation_level, n_assets
        )
        cov_matrix = np.diag(volatility) @ correlation_matrix @ np.diag(volatility)

        returns = np.random.multivariate_normal(mean_return, cov_matrix)

        return returns, volatility

    def _get_regime_correlation_level(self, regime: MarketRegime) -> float:
        """Get typical correlation level for a regime"""
        correlation_levels = {
            MarketRegime.BULL: 0.3,
            MarketRegime.BEAR: 0.6,
            MarketRegime.SIDEWAYS: 0.2,
            MarketRegime.VOLATILE: 0.7,
            MarketRegime.CRISIS: 0.9,
        }
        return correlation_levels.get(regime, 0.4)

    def _create_regime_correlation_matrix(
        self, base_correlation: float, n_assets: int
    ) -> np.ndarray:
        """Create correlation matrix for specific regime"""
        correlation_matrix = np.full((n_assets, n_assets), base_correlation)
        np.fill_diagonal(correlation_matrix, 1.0)

        # Add sector-based adjustments
        for i, asset_i in enumerate(self.assets):
            for j, asset_j in enumerate(self.assets):
                if i != j and asset_i.sector == asset_j.sector:
                    correlation_matrix[i, j] = min(0.95, base_correlation + 0.2)

        # Ensure positive semi-definite
        eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
        eigenvals = np.maximum(eigenvals, 0.01)
        correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        return correlation_matrix

    def _generate_regime_volumes(
        self, n_steps: int, regime_sequence: np.ndarray
    ) -> np.ndarray:
        """Generate volume patterns for different regimes"""
        volumes = np.zeros((n_steps, self.n_assets))

        for t in range(n_steps):
            regime = self.config.regime_types[regime_sequence[t]]

            # Base volume depends on regime
            if regime == MarketRegime.BULL:
                volume_multiplier = 1.2
            elif regime == MarketRegime.BEAR:
                volume_multiplier = 1.5
            elif regime == MarketRegime.VOLATILE:
                volume_multiplier = 2.0
            elif regime == MarketRegime.CRISIS:
                volume_multiplier = 3.0
            else:
                volume_multiplier = 1.0

            for i, asset in enumerate(self.assets):
                base_volume = asset.avg_daily_volume
                regime_volume = (
                    base_volume * volume_multiplier * np.random.lognormal(0, 0.3)
                )
                volumes[t, i] = regime_volume

        return volumes

    def _generate_regime_correlations(
        self, n_steps: int, regime_sequence: np.ndarray
    ) -> np.ndarray:
        """Generate average correlation time series"""
        correlations = np.zeros(n_steps)

        for t in range(n_steps):
            regime = self.config.regime_types[regime_sequence[t]]
            correlations[t] = self._get_regime_correlation_level(regime)

        # Add some smoothing
        for t in range(1, n_steps):
            correlations[t] = 0.8 * correlations[t] + 0.2 * correlations[t - 1]

        return correlations

    def _calculate_regime_features(self, data: pd.DataFrame):
        """Calculate features useful for regime detection"""
        # Rolling volatility
        for window in [10, 20, 60]:
            data[f"volatility_{window}d"] = (
                data["regime_volatility"].rolling(window=window).mean()
            )

        # Rolling correlation
        for window in [10, 20, 60]:
            data[f"correlation_{window}d"] = (
                data["regime_correlation"].rolling(window=window).mean()
            )

        # Cross-sectional features
        asset_return_cols = [f"{asset.symbol}_returns" for asset in self.assets]
        data["cross_sectional_dispersion"] = data[asset_return_cols].std(axis=1)
        data["cross_sectional_skew"] = data[asset_return_cols].skew(axis=1)
        data["market_return"] = data[asset_return_cols].mean(axis=1)

        # Trend features
        data["trend_20d"] = data["market_return"].rolling(window=20).mean()
        data["trend_60d"] = data["market_return"].rolling(window=60).mean()

        # Volatility clustering
        data["vol_clustering"] = data["regime_volatility"].rolling(window=10).std()

        # Fill NaN values
        data.fillna(method="bfill", inplace=True)
        data.fillna(0, inplace=True)

    def _get_observation(self) -> np.ndarray:
        """Get current observation with regime-relevant features"""
        obs_components = []

        # 1. Asset return histories
        for asset in self.assets:
            if self.current_step >= self.lookback_window:
                returns_window = self.market_data[f"{asset.symbol}_returns"].values[
                    self.current_step - self.lookback_window : self.current_step
                ]
            else:
                returns_window = np.zeros(self.lookback_window)
                available_returns = self.market_data[f"{asset.symbol}_returns"].values[
                    : self.current_step
                ]
                returns_window[-len(available_returns) :] = available_returns

            obs_components.extend(returns_window)

        # 2. Technical indicators for each asset
        for asset in self.assets:
            indicators = self._calculate_asset_regime_indicators(asset.symbol)
            obs_components.extend(indicators)

        # 3. Regime-specific features
        regime_features = self._calculate_regime_observation_features()
        obs_components.extend(regime_features)

        # 4. Market macro indicators
        macro_features = self._calculate_macro_features()
        obs_components.extend(macro_features)

        # 5. Portfolio state
        portfolio_features = [
            self.cash_balance / self.portfolio_value,
            self.unrealized_pnl / self.initial_cash,
            self.realized_pnl / self.initial_cash,
            len(self.portfolio_history) / self.max_episode_length,
            self._calculate_regime_adaptation_score(),
        ]
        obs_components.extend(portfolio_features)

        # 6. Regime prediction history
        prediction_history = self._get_prediction_history_features()
        obs_components.extend(prediction_history)

        # 7. Additional adaptive features
        adaptive_features = self._calculate_adaptive_features()
        obs_components.extend(adaptive_features)

        return np.array(obs_components, dtype=np.float32)

    def _calculate_asset_regime_indicators(self, symbol: str) -> List[float]:
        """Calculate regime-relevant indicators for an asset"""
        returns = self.market_data[f"{symbol}_returns"].values[
            max(0, self.current_step - 60) : self.current_step + 1
        ]

        if len(returns) < 10:
            return [0.0] * 10

        indicators = [
            np.mean(returns),  # Mean return
            np.std(returns),  # Volatility
            np.mean(np.abs(returns)),  # Average absolute return
            len([r for r in returns if r > 0]) / len(returns),  # Win rate
            np.mean(returns[-10:]) if len(returns) >= 10 else 0,  # Recent momentum
            np.std(returns[-10:]) if len(returns) >= 10 else 0,  # Recent volatility
            np.mean(returns[-20:]) if len(returns) >= 20 else 0,  # Medium momentum
            np.skew(returns) if len(returns) >= 10 else 0,  # Skewness
            np.kurtosis(returns) if len(returns) >= 10 else 0,  # Kurtosis
            self.market_data[f"{symbol}_volume"].iloc[self.current_step]
            / self.assets[0].avg_daily_volume,  # Volume ratio
        ]

        return indicators

    def _calculate_regime_observation_features(self) -> List[float]:
        """Calculate features specifically useful for regime detection"""
        if self.current_step < 20:
            return [0.0] * 20

        # Market-wide features
        market_returns = self.market_data["market_return"].values[
            max(0, self.current_step - 60) : self.current_step + 1
        ]

        features = [
            self.market_data["regime_volatility"].iloc[
                self.current_step
            ],  # Current volatility
            self.market_data["regime_correlation"].iloc[
                self.current_step
            ],  # Current correlation
            self.market_data["cross_sectional_dispersion"].iloc[
                self.current_step
            ],  # Dispersion
            self.market_data["cross_sectional_skew"].iloc[self.current_step],  # Skew
            self.market_data["volatility_20d"].iloc[
                self.current_step
            ],  # 20D volatility
            self.market_data["volatility_60d"].iloc[
                self.current_step
            ],  # 60D volatility
            self.market_data["correlation_20d"].iloc[
                self.current_step
            ],  # 20D correlation
            self.market_data["correlation_60d"].iloc[
                self.current_step
            ],  # 60D correlation
            self.market_data["trend_20d"].iloc[self.current_step],  # 20D trend
            self.market_data["trend_60d"].iloc[self.current_step],  # 60D trend
            self.market_data["vol_clustering"].iloc[
                self.current_step
            ],  # Vol clustering
        ]

        # Additional statistical features
        if len(market_returns) >= 30:
            recent_returns = market_returns[-30:]
            features.extend(
                [
                    np.mean(recent_returns),  # Recent mean
                    np.std(recent_returns),  # Recent std
                    np.skew(recent_returns),  # Recent skew
                    np.kurtosis(recent_returns),  # Recent kurtosis
                    np.max(np.abs(recent_returns)),  # Max move
                    len([r for r in recent_returns if abs(r) > 0.02])
                    / len(recent_returns),  # Extreme move frequency
                    np.sum(np.sign(recent_returns)),  # Net direction
                    np.mean(recent_returns**2),  # Second moment
                    np.mean(recent_returns**3),  # Third moment
                ]
            )
        else:
            features.extend([0.0] * 9)

        return features[:20]  # Ensure exactly 20 features

    def _calculate_macro_features(self) -> List[float]:
        """Calculate macro-level market features"""
        # Calculate market-wide metrics
        asset_returns = []
        asset_volumes = []

        for asset in self.assets:
            returns = self.market_data[f"{asset.symbol}_returns"].values[
                max(0, self.current_step - 20) : self.current_step + 1
            ]
            volume = self.market_data[f"{asset.symbol}_volume"].iloc[self.current_step]

            if len(returns) > 0:
                asset_returns.append(np.mean(returns))
            asset_volumes.append(volume / asset.avg_daily_volume)

        # Market features
        features = [
            np.mean(asset_returns) if asset_returns else 0,  # Average asset return
            np.std(asset_returns)
            if len(asset_returns) > 1
            else 0,  # Cross-asset volatility
            np.mean(asset_volumes),  # Average volume ratio
            len(self.portfolio_history) / self.max_episode_length,  # Time progress
            self._calculate_regime_duration(),  # Current regime duration
            self._calculate_regime_change_probability(),  # Probability of regime change
            self._calculate_market_sentiment(),  # Market sentiment indicator
            self._calculate_liquidity_indicator(),  # Liquidity indicator
            self._calculate_stress_indicator(),  # Market stress indicator
            self._calculate_momentum_indicator(),  # Market momentum
        ]

        return features

    def _calculate_regime_adaptation_score(self) -> float:
        """Calculate how well the agent is adapting to current regime"""
        if len(self.prediction_accuracy) < 5:
            return 0.0

        # Recent prediction accuracy
        recent_accuracy = np.mean(self.prediction_accuracy[-5:])

        # Performance in current regime
        current_regime = self.config.regime_types[
            self.market_data["true_regime"].iloc[self.current_step]
        ]
        if (
            current_regime in self.regime_performance
            and len(self.regime_performance[current_regime]) > 0
        ):
            recent_performance = np.mean(self.regime_performance[current_regime][-3:])
            return (recent_accuracy + recent_performance) / 2

        return recent_accuracy

    def _get_prediction_history_features(self) -> List[float]:
        """Get features from prediction history"""
        features = []

        # Recent predictions
        for i in range(min(self.n_regimes, 5)):
            if i < len(self.predicted_regime_history):
                # One-hot encoding of recent prediction
                pred_features = [
                    1.0 if pred == i else 0.0
                    for pred in self.predicted_regime_history[-5:]
                ]
                features.append(np.mean(pred_features))
            else:
                features.append(0.0)

        # Ensure we have exactly n_regimes features
        while len(features) < self.n_regimes:
            features.append(0.0)

        return features[: self.n_regimes]

    def _calculate_adaptive_features(self) -> List[float]:
        """Calculate features that help with adaptation"""
        features = [
            self._calculate_prediction_confidence(),  # Confidence in predictions
            self._calculate_regime_stability(),  # Regime stability measure
            self._calculate_strategy_effectiveness(),  # Current strategy effectiveness
            self._calculate_risk_adjustment_need(),  # Need for risk adjustment
            self._calculate_market_complexity(),  # Market complexity indicator
            self._calculate_information_efficiency(),  # Market information efficiency
            self._calculate_trend_strength(),  # Current trend strength
            self._calculate_volatility_regime(),  # Volatility regime indicator
            self._calculate_correlation_regime(),  # Correlation regime indicator
            self._calculate_liquidity_regime(),  # Liquidity regime indicator
        ]

        return features

    def _calculate_prediction_confidence(self) -> float:
        """Calculate confidence in regime predictions"""
        if len(self.prediction_accuracy) < 10:
            return 0.5

        recent_accuracy = np.mean(self.prediction_accuracy[-10:])
        return min(1.0, max(0.0, recent_accuracy))

    def _calculate_regime_duration(self) -> float:
        """Calculate duration of current regime"""
        if not hasattr(self, "current_regime_start"):
            return 0.0
        return (self.current_step - self.current_regime_start) / self.max_episode_length

    def _calculate_regime_change_probability(self) -> float:
        """Estimate probability of regime change"""
        # Use volatility and correlation as indicators
        current_vol = self.market_data["regime_volatility"].iloc[self.current_step]
        current_corr = self.market_data["regime_correlation"].iloc[self.current_step]

        # High volatility and changing correlation suggest regime change
        change_probability = (current_vol / 0.03 + abs(current_corr - 0.5) / 0.3) / 2
        return min(1.0, max(0.0, change_probability))

    def _calculate_market_sentiment(self) -> float:
        """Calculate market sentiment indicator"""
        market_return = self.market_data["market_return"].iloc[self.current_step]
        dispersion = self.market_data["cross_sectional_dispersion"].iloc[
            self.current_step
        ]

        # Positive return with low dispersion = bullish sentiment
        sentiment = (market_return / 0.01 - dispersion / 0.02) / 2
        return np.tanh(sentiment)

    def _calculate_liquidity_indicator(self) -> float:
        """Calculate market liquidity indicator"""
        volumes = []
        for asset in self.assets:
            vol = self.market_data[f"{asset.symbol}_volume"].iloc[self.current_step]
            volumes.append(vol / asset.avg_daily_volume)

        return np.mean(volumes)

    def _calculate_stress_indicator(self) -> float:
        """Calculate market stress indicator"""
        stress_factors = [
            self.market_data["regime_volatility"].iloc[self.current_step] / 0.03,
            abs(self.market_data["market_return"].iloc[self.current_step]) / 0.02,
            self.market_data["cross_sectional_dispersion"].iloc[self.current_step]
            / 0.03,
        ]

        return np.mean(stress_factors)

    def _calculate_momentum_indicator(self) -> float:
        """Calculate market momentum indicator"""
        short_trend = self.market_data["trend_20d"].iloc[self.current_step]
        long_trend = self.market_data["trend_60d"].iloc[self.current_step]

        return (short_trend + long_trend) / 2

    def _calculate_regime_stability(self) -> float:
        """Calculate current regime stability"""
        if len(self.true_regime_history) < 10:
            return 0.5

        recent_regimes = self.true_regime_history[-10:]
        stability = 1.0 - (len(set(recent_regimes)) - 1) / min(
            len(self.config.regime_types) - 1, 9
        )
        return stability

    def _calculate_strategy_effectiveness(self) -> float:
        """Calculate effectiveness of current trading strategy"""
        if len(self.portfolio_history) < 10:
            return 0.0

        recent_returns = (
            np.diff(self.portfolio_history[-10:]) / self.portfolio_history[-10:-1]
        )
        return np.mean(recent_returns) / (np.std(recent_returns) + 1e-8)

    def _calculate_risk_adjustment_need(self) -> float:
        """Calculate need for risk adjustment"""
        current_vol = self.market_data["regime_volatility"].iloc[self.current_step]
        normal_vol = 0.02

        return min(1.0, max(0.0, (current_vol - normal_vol) / normal_vol))

    def _calculate_market_complexity(self) -> float:
        """Calculate market complexity indicator"""
        complexity_factors = [
            self.market_data["cross_sectional_dispersion"].iloc[self.current_step]
            / 0.02,
            self.market_data["vol_clustering"].iloc[self.current_step] / 0.01,
            abs(self.market_data["cross_sectional_skew"].iloc[self.current_step]) / 0.5,
        ]

        return np.mean(complexity_factors)

    def _calculate_information_efficiency(self) -> float:
        """Calculate market information efficiency"""
        # Use autocorrelation as a proxy for inefficiency
        market_returns = self.market_data["market_return"].values[
            max(0, self.current_step - 20) : self.current_step + 1
        ]

        if len(market_returns) < 5:
            return 0.5

        autocorr = np.corrcoef(market_returns[:-1], market_returns[1:])[0, 1]
        efficiency = 1.0 - abs(autocorr)
        return max(0.0, min(1.0, efficiency))

    def _calculate_trend_strength(self) -> float:
        """Calculate trend strength"""
        if self.current_step < 20:
            return 0.0

        recent_returns = self.market_data["market_return"].values[
            self.current_step - 20 : self.current_step + 1
        ]
        trend_strength = abs(np.mean(recent_returns)) / (np.std(recent_returns) + 1e-8)

        return min(2.0, trend_strength)

    def _calculate_volatility_regime(self) -> float:
        """Calculate volatility regime indicator"""
        current_vol = self.market_data["regime_volatility"].iloc[self.current_step]

        if current_vol < 0.015:
            return 0.0  # Low volatility
        elif current_vol < 0.025:
            return 0.5  # Normal volatility
        else:
            return 1.0  # High volatility

    def _calculate_correlation_regime(self) -> float:
        """Calculate correlation regime indicator"""
        current_corr = self.market_data["regime_correlation"].iloc[self.current_step]

        return min(1.0, max(0.0, current_corr))

    def _calculate_liquidity_regime(self) -> float:
        """Calculate liquidity regime indicator"""
        liquidity = self._calculate_liquidity_indicator()

        if liquidity > 1.5:
            return 1.0  # High liquidity
        elif liquidity > 0.8:
            return 0.5  # Normal liquidity
        else:
            return 0.0  # Low liquidity

    def _process_action(self, action: Union[np.ndarray, int, Tuple]) -> Dict[str, Any]:
        """Process regime prediction and trading action"""
        if self.action_mode == "combined":
            # Combined action: [regime_prediction, trading_action]
            regime_prediction = (
                action[0] if hasattr(action, "__getitem__") else action // 3
            )
            trading_action = action[1] if hasattr(action, "__getitem__") else action % 3
        elif self.action_mode == "separate":
            # Separate tuple action
            regime_prediction, trading_action = action
        else:  # prediction_only
            regime_prediction = action
            trading_action = 0  # Hold

        # Store prediction
        predicted_regime = self.config.regime_types[
            min(regime_prediction, self.n_regimes - 1)
        ]
        self.predicted_regime_history.append(regime_prediction)

        # Evaluate prediction accuracy
        true_regime = self.config.regime_types[
            self.market_data["true_regime"].iloc[self.current_step]
        ]
        prediction_correct = int(predicted_regime == true_regime)
        self.prediction_accuracy.append(prediction_correct)

        # Execute trading action
        trading_result = self._execute_trading_action(trading_action)

        # Calculate regime-specific performance
        step_return = (
            self.portfolio_value / self.portfolio_history[-2] - 1
            if len(self.portfolio_history) > 1
            else 0
        )
        self.regime_performance[true_regime].append(step_return)

        return {
            "regime_prediction": regime_prediction,
            "true_regime": true_regime.value,
            "prediction_correct": prediction_correct,
            "trading_result": trading_result,
            "step_return": step_return,
        }

    def _execute_trading_action(self, trading_action: int) -> Dict[str, Any]:
        """Execute simple trading action"""
        if self.n_assets == 0:
            return {"executed": False, "reason": "No assets available"}

        # Simple trading: trade the first asset
        asset = self.assets[0]
        current_price = self.current_prices[asset.symbol]
        current_position = self.positions.get(asset.symbol, 0)

        if trading_action == 0:  # Hold
            return {"executed": False, "action": "hold"}

        elif trading_action == 1:  # Buy
            if self.cash_balance > current_price * 100:  # Buy 100 shares if possible
                shares = min(100, int(self.cash_balance / (current_price * 2)))
                trade_value = shares * current_price
                cost = self._calculate_transaction_costs(trade_value)

                if self.cash_balance >= trade_value + cost:
                    self.positions[asset.symbol] = current_position + shares
                    self.cash_balance -= trade_value + cost
                    self.realized_pnl -= cost

                    return {
                        "executed": True,
                        "action": "buy",
                        "shares": shares,
                        "price": current_price,
                        "cost": cost,
                    }

        elif trading_action == 2:  # Sell
            if current_position > 0:
                shares_to_sell = min(100, current_position)
                trade_value = shares_to_sell * current_price
                cost = self._calculate_transaction_costs(trade_value)

                self.positions[asset.symbol] = current_position - shares_to_sell
                self.cash_balance += trade_value - cost
                self.realized_pnl -= cost

                return {
                    "executed": True,
                    "action": "sell",
                    "shares": shares_to_sell,
                    "price": current_price,
                    "cost": cost,
                }

        return {"executed": False, "action": "hold", "reason": "Cannot execute trade"}

    def _calculate_reward(self, execution_details: Dict[str, Any]) -> float:
        """Calculate reward with regime detection bonuses"""
        # Base reward from trading
        portfolio_return = execution_details.get("step_return", 0)

        # Regime prediction bonus
        prediction_bonus = 0.0
        if execution_details["prediction_correct"]:
            prediction_bonus = self.config.regime_reward_multiplier * 0.01

        # Adaptation bonus (performing well in predicted regime)
        adaptation_bonus = 0.0
        predicted_regime = self.config.regime_types[
            execution_details["regime_prediction"]
        ]
        if (
            predicted_regime in self.regime_performance
            and len(self.regime_performance[predicted_regime]) > 0
        ):
            recent_performance = np.mean(self.regime_performance[predicted_regime][-5:])
            if recent_performance > 0:
                adaptation_bonus = recent_performance * 0.5

        # Learning bonus (improving prediction accuracy over time)
        learning_bonus = 0.0
        if len(self.prediction_accuracy) > 20:
            recent_accuracy = np.mean(self.prediction_accuracy[-10:])
            earlier_accuracy = np.mean(self.prediction_accuracy[-20:-10])
            if recent_accuracy > earlier_accuracy:
                learning_bonus = (recent_accuracy - earlier_accuracy) * 0.1

        # Total reward
        total_reward = (
            portfolio_return
            + prediction_bonus  # Trading performance
            + adaptation_bonus  # Correct prediction bonus
            + learning_bonus  # Adaptation bonus  # Learning improvement bonus
        )

        return float(total_reward)

    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get comprehensive regime detection statistics"""
        stats = {
            "overall_accuracy": np.mean(self.prediction_accuracy)
            if self.prediction_accuracy
            else 0,
            "total_predictions": len(self.prediction_accuracy),
            "regime_counts": {},
            "regime_accuracy": {},
            "performance_by_regime": {},
        }

        # Count predictions and accuracy by regime
        true_regimes = self.market_data["true_regime"].values[
            : len(self.prediction_accuracy)
        ]

        for regime in self.config.regime_types:
            regime_mask = true_regimes == regime.value
            regime_predictions = np.array(self.prediction_accuracy)[regime_mask]

            stats["regime_counts"][regime.name] = int(np.sum(regime_mask))
            stats["regime_accuracy"][regime.name] = (
                float(np.mean(regime_predictions)) if len(regime_predictions) > 0 else 0
            )

            if (
                regime in self.regime_performance
                and len(self.regime_performance[regime]) > 0
            ):
                performance = self.regime_performance[regime]
                stats["performance_by_regime"][regime.name] = {
                    "mean_return": float(np.mean(performance)),
                    "volatility": float(np.std(performance)),
                    "sharpe_ratio": float(
                        np.mean(performance) / (np.std(performance) + 1e-8)
                    ),
                    "samples": len(performance),
                }

        return stats

    def get_action_meanings(self) -> List[str]:
        """Get human-readable action meanings"""
        if self.action_mode == "combined":
            meanings = []
            for regime in self.config.regime_types:
                meanings.extend(
                    [
                        f"Predict {regime.name} + Hold",
                        f"Predict {regime.name} + Buy",
                        f"Predict {regime.name} + Sell",
                    ]
                )
            return meanings
        elif self.action_mode == "separate":
            return [f"Regime: {regime.name}" for regime in self.config.regime_types] + [
                "Hold",
                "Buy",
                "Sell",
            ]
        else:  # prediction_only
            return [f"Predict {regime.name}" for regime in self.config.regime_types]
