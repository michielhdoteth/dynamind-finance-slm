"""
Synthetic Market Data Generator

Advanced synthetic data generation for financial markets with realistic
properties, regime switching, and customizable characteristics.
"""

from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from environments.base_env import AssetConfig


class MarketRegime(Enum):
    """Market regime types for synthetic data generation"""

    BULL_NORMAL = "bull_normal"
    BULL_VOLATILE = "bull_volatile"
    BEAR_NORMAL = "bear_normal"
    BEAR_VOLATILE = "bear_volatile"
    SIDEWAYS_LOW_VOL = "sideways_low_vol"
    SIDEWAYS_HIGH_VOL = "sideways_high_vol"
    TRANSITION = "transition"
    CRISIS = "crisis"


@dataclass
class RegimeParameters:
    """Parameters for a specific market regime"""

    name: str
    drift_mean: float
    drift_std: float
    volatility_mean: float
    volatility_std: float
    correlation_base: float
    correlation_std: float
    jump_intensity: float
    jump_mean: float
    jump_std: float
    mean_reversion_strength: float
    momentum_strength: float
    volume_multiplier: float
    persistence: float  # Probability of staying in same regime


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation"""

    n_assets: int
    n_steps: int
    frequency: str = "daily"  # "daily", "hourly", "minute"
    start_date: datetime = field(default_factory=lambda: datetime(2020, 1, 1))
    regimes: List[RegimeParameters] = field(default_factory=list)
    sector_correlations: Dict[str, float] = field(default_factory=dict)
    event_frequency: float = 0.01  # Probability of market event per step
    noise_level: float = 0.1
    fat_tail_parameter: float = 3.0  # Degrees of freedom for t-distribution


class MarketDataGenerator:
    """
    Advanced synthetic market data generator with realistic properties.

    Features:
    - Multiple market regimes with smooth transitions
    - Realistic correlation structures
    - Jump diffusion and fat tails
    - Sector-based correlation patterns
    - Event-driven volatility spikes
    - Volume and liquidity modeling
    - Microstructure effects
    """

    def __init__(self, config: SyntheticDataConfig):
        self.config = config
        self.regime_sequence = []
        self.regime_transitions = []
        self.current_regime_idx = 0

        # Initialize default regimes if not provided
        if not self.config.regimes:
            self.config.regimes = self._create_default_regimes()

    def _create_default_regimes(self) -> List[RegimeParameters]:
        """Create default market regimes"""
        return [
            RegimeParameters(
                name="bull_normal",
                drift_mean=0.0008,
                drift_std=0.0003,
                volatility_mean=0.015,
                volatility_std=0.005,
                correlation_base=0.3,
                correlation_std=0.1,
                jump_intensity=0.01,
                jump_mean=0.002,
                jump_std=0.001,
                mean_reversion_strength=0.1,
                momentum_strength=0.05,
                volume_multiplier=1.2,
                persistence=0.95,
            ),
            RegimeParameters(
                name="bear_normal",
                drift_mean=-0.0003,
                drift_std=0.0002,
                volatility_mean=0.020,
                volatility_std=0.008,
                correlation_base=0.5,
                correlation_std=0.15,
                jump_intensity=0.02,
                jump_mean=-0.003,
                jump_std=0.002,
                mean_reversion_strength=0.15,
                momentum_strength=0.1,
                volume_multiplier=1.5,
                persistence=0.92,
            ),
            RegimeParameters(
                name="sideways_low_vol",
                drift_mean=0.0001,
                drift_std=0.0001,
                volatility_mean=0.010,
                volatility_std=0.003,
                correlation_base=0.2,
                correlation_std=0.05,
                jump_intensity=0.005,
                jump_mean=0.0,
                jump_std=0.0005,
                mean_reversion_strength=0.2,
                momentum_strength=0.02,
                volume_multiplier=0.8,
                persistence=0.90,
            ),
            RegimeParameters(
                name="crisis",
                drift_mean=-0.002,
                drift_std=0.001,
                volatility_mean=0.050,
                volatility_std=0.020,
                correlation_base=0.8,
                correlation_std=0.1,
                jump_intensity=0.1,
                jump_mean=-0.01,
                jump_std=0.005,
                mean_reversion_strength=0.05,
                momentum_strength=0.3,
                volume_multiplier=3.0,
                persistence=0.98,
            ),
        ]

    def generate_market_data(self, assets: List[AssetConfig]) -> pd.DataFrame:
        """Generate complete synthetic market data"""
        # Generate regime sequence
        self._generate_regime_sequence()

        # Generate price processes
        price_data = self._generate_price_processes(assets)

        # Generate volume data
        volume_data = self._generate_volume_data(assets)

        # Generate order book data
        order_book_data = self._generate_order_book_data(price_data)

        # Generate market events
        events = self._generate_market_events()

        # Combine all data
        complete_data = self._combine_data_sources(
            price_data, volume_data, order_book_data, events
        )

        # Add market microstructure effects
        complete_data = self._add_microstructure_effects(complete_data)

        return complete_data

    def _generate_regime_sequence(self):
        """Generate sequence of market regimes with smooth transitions"""
        n_steps = self.config.n_steps
        self.regime_sequence = np.zeros(n_steps, dtype=int)
        self.regime_transitions = []

        current_regime = np.random.randint(len(self.config.regimes))
        transition_periods = []

        for step in range(n_steps):
            self.regime_sequence[step] = current_regime

            # Check for regime transition
            regime_params = self.config.regimes[current_regime]
            if np.random.random() > regime_params.persistence:
                # Transition to new regime
                old_regime = current_regime
                current_regime = np.random.choice(len(self.config.regimes))

                # Generate transition period
                transition_length = np.random.randint(1, 5)  # 1-5 steps transition
                transition_periods.append(
                    {
                        "start_step": step,
                        "end_step": min(step + transition_length, n_steps),
                        "from_regime": old_regime,
                        "to_regime": current_regime,
                    }
                )

                self.regime_transitions.append(transition_periods[-1])

        # Smooth transitions
        self._smooth_regime_transitions()

    def _smooth_regime_transitions(self):
        """Apply smoothing to regime transitions"""
        for transition in self.regime_transitions:
            start = transition["start_step"]
            end = transition["end_step"]
            from_regime = transition["from_regime"]
            to_regime = transition["to_regime"]

            for step in range(start, end):
                # Linear interpolation between regimes
                alpha = (step - start) / (end - start)
                # This would be used to smoothly transition parameters
                # For now, we just mark the transition period

    def _generate_price_processes(self, assets: List[AssetConfig]) -> pd.DataFrame:
        """Generate realistic price processes with regime-dependent dynamics"""
        n_steps = self.config.n_steps
        n_assets = len(assets)

        # Initialize arrays
        returns = np.zeros((n_steps, n_assets))
        volatilities = np.zeros((n_steps, n_assets))
        correlations = np.zeros((n_steps, n_assets, n_assets))

        # Generate for each time step
        for step in range(n_steps):
            regime_idx = self.regime_sequence[step]
            regime = self.config.regimes[regime_idx]

            # Generate regime-dependent parameters
            asset_drifts = np.random.normal(
                regime.drift_mean, regime.drift_std, n_assets
            )
            asset_vols = np.abs(
                np.random.normal(
                    regime.volatility_mean, regime.volatility_std, n_assets
                )
            )

            # Generate correlation matrix
            corr_matrix = self._generate_correlation_matrix(
                regime.correlation_base, regime.correlation_std, assets
            )

            # Apply sector-based adjustments
            corr_matrix = self._apply_sector_correlations(corr_matrix, assets)

            # Generate returns with potential jumps
            step_returns = self._generate_returns_with_jumps(
                asset_drifts, asset_vols, corr_matrix, regime
            )

            # Apply momentum and mean reversion
            if step > 0:
                step_returns = self._apply_momentum_mean_reversion(
                    step_returns, returns[step - 1], regime
                )

            returns[step] = step_returns
            volatilities[step] = asset_vols
            correlations[step] = corr_matrix

        # Convert to prices
        prices = np.zeros_like(returns)
        for i, asset in enumerate(assets):
            prices[:, i] = asset.initial_price * np.cumprod(1 + returns[:, i])

        # Create DataFrame
        price_df = pd.DataFrame(index=range(n_steps))
        for i, asset in enumerate(assets):
            prefix = f"{asset.symbol}_"
            price_df[f"{prefix}close"] = prices[:, i]
            price_df[f"{prefix}returns"] = returns[:, i]
            price_df[f"{prefix}volatility"] = volatilities[:, i]

        return price_df

    def _generate_correlation_matrix(
        self, base_corr: float, corr_std: float, assets: List[AssetConfig]
    ) -> np.ndarray:
        """Generate realistic correlation matrix"""
        n_assets = len(assets)
        correlation = np.random.normal(base_corr, corr_std, (n_assets, n_assets))

        # Make symmetric and set diagonal
        correlation = (correlation + correlation.T) / 2
        np.fill_diagonal(correlation, 1.0)

        # Ensure positive semi-definite
        eigenvals, eigenvecs = np.linalg.eigh(correlation)
        eigenvals = np.maximum(eigenvals, 0.01)
        correlation = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        return correlation

    def _apply_sector_correlations(
        self, corr_matrix: np.ndarray, assets: List[AssetConfig]
    ) -> np.ndarray:
        """Apply sector-based correlation adjustments"""
        n_assets = len(assets)
        adjusted_corr = corr_matrix.copy()

        for i, asset_i in enumerate(assets):
            for j, asset_j in enumerate(assets):
                if i != j and asset_i.sector == asset_j.sector:
                    # Increase correlation within same sector
                    sector_boost = self.config.sector_correlations.get(
                        asset_i.sector, 0.1
                    )
                    adjusted_corr[i, j] = min(0.95, adjusted_corr[i, j] + sector_boost)

        # Re-normalize to ensure positive semi-definite
        eigenvals, eigenvecs = np.linalg.eigh(adjusted_corr)
        eigenvals = np.maximum(eigenvals, 0.01)
        adjusted_corr = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        return adjusted_corr

    def _generate_returns_with_jumps(
        self,
        drifts: np.ndarray,
        vols: np.ndarray,
        corr_matrix: np.ndarray,
        regime: RegimeParameters,
    ) -> np.ndarray:
        """Generate returns with jump diffusion"""
        n_assets = len(drifts)

        # Base returns from multivariate normal
        cov_matrix = np.diag(vols) @ corr_matrix @ np.diag(vols)
        base_returns = np.random.multivariate_normal(drifts, cov_matrix)

        # Add jumps
        for i in range(n_assets):
            if np.random.random() < regime.jump_intensity:
                jump_size = np.random.normal(regime.jump_mean, regime.jump_std)
                base_returns[i] += jump_size

        # Apply fat tails using t-distribution
        if self.config.fat_tail_parameter < 10:
            # Transform to t-distribution
            normal_samples = np.random.normal(0, 1, n_assets)
            chi_square = np.random.chisquare(self.config.fat_tail_parameter)
            t_samples = normal_samples / np.sqrt(
                chi_square / self.config.fat_tail_parameter
            )

            # Blend with normal returns
            fat_tail_factor = 0.3  # 30% fat tail influence
            base_returns = (
                1 - fat_tail_factor
            ) * base_returns + fat_tail_factor * base_returns * t_samples

        return base_returns

    def _apply_momentum_mean_reversion(
        self,
        current_returns: np.ndarray,
        previous_returns: np.ndarray,
        regime: RegimeParameters,
    ) -> np.ndarray:
        """Apply momentum and mean reversion effects"""
        # Momentum component
        momentum_component = regime.momentum_strength * previous_returns

        # Mean reversion component (pull towards zero)
        mean_reversion_component = -regime.mean_reversion_strength * previous_returns

        # Combined effect
        adjusted_returns = (
            current_returns + momentum_component + mean_reversion_component
        )

        return adjusted_returns

    def _generate_volume_data(self, assets: List[AssetConfig]) -> pd.DataFrame:
        """Generate realistic volume data"""
        n_steps = self.config.n_steps
        volume_df = pd.DataFrame(index=range(n_steps))

        for i, asset in enumerate(assets):
            volumes = np.zeros(n_steps)
            base_volume = asset.avg_daily_volume

            for step in range(n_steps):
                regime_idx = self.regime_sequence[step]
                regime = self.config.regimes[regime_idx]

                # Base volume with regime multiplier
                base_step_volume = base_volume * regime.volume_multiplier

                # Add daily/weekly patterns (if daily data)
                if self.config.frequency == "daily":
                    day_of_week = step % 5
                    if day_of_week == 0:  # Monday
                        volume_multiplier = 1.2
                    elif day_of_week == 4:  # Friday
                        volume_multiplier = 0.8
                    else:
                        volume_multiplier = 1.0
                else:
                    volume_multiplier = 1.0

                # Add random component
                random_component = np.random.lognormal(0, 0.3)

                volumes[step] = base_step_volume * volume_multiplier * random_component

            # Apply autocorrelation to volumes
            for t in range(1, n_steps):
                volumes[t] = 0.8 * volumes[t - 1] + 0.2 * volumes[t]

            volume_df[f"{asset.symbol}_volume"] = volumes

        return volume_df

    def _generate_order_book_data(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Generate simplified order book data"""
        n_steps = len(price_data)
        order_book_df = pd.DataFrame(index=range(n_steps))

        for i, asset in enumerate(self.assets):
            # Extract close prices
            close_prices = price_data[f"{asset.symbol}_close"].values
            returns = price_data[f"{asset.symbol}_returns"].values

            for level in range(5):  # 5 levels of order book
                # Bid prices
                bid_spreads = (
                    np.random.exponential(0.0001, n_steps) * close_prices * (level + 1)
                )
                bid_prices = close_prices - bid_spreads
                order_book_df[f"{asset.symbol}_bid_{level}"] = bid_prices

                # Ask prices
                ask_spreads = (
                    np.random.exponential(0.0001, n_steps) * close_prices * (level + 1)
                )
                ask_prices = close_prices + ask_spreads
                order_book_df[f"{asset.symbol}_ask_{level}"] = ask_prices

                # Bid sizes
                bid_sizes = np.random.exponential(1000, n_steps)
                order_book_df[f"{asset.symbol}_bid_size_{level}"] = bid_sizes

                # Ask sizes
                ask_sizes = np.random.exponential(1000, n_steps)
                order_book_df[f"{asset.symbol}_ask_size_{level}"] = ask_sizes

        return order_book_df

    def _generate_market_events(self) -> pd.DataFrame:
        """Generate market events (earnings, news, etc.)"""
        events_df = pd.DataFrame(index=range(self.config.n_steps))
        events_df["event_type"] = "none"
        events_df["event_magnitude"] = 0.0

        for step in range(self.config.n_steps):
            if np.random.random() < self.config.event_frequency:
                # Generate random event
                event_types = ["earnings", "news", "fed", "macro", "sector"]
                event_type = np.random.choice(event_types)
                event_magnitude = np.random.exponential(0.01)  # Average 1% impact

                events_df.loc[step, "event_type"] = event_type
                events_df.loc[step, "event_magnitude"] = event_magnitude

        return events_df

    def _combine_data_sources(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_book_data: pd.DataFrame,
        events: pd.DataFrame,
    ) -> pd.DataFrame:
        """Combine all data sources into a single DataFrame"""
        combined = pd.concat([price_data, volume_data, order_book_data, events], axis=1)

        # Add timestamp column
        if self.config.frequency == "daily":
            dates = pd.date_range(
                start=self.config.start_date, periods=self.config.n_steps, freq="D"
            )
        elif self.config.frequency == "hourly":
            dates = pd.date_range(
                start=self.config.start_date, periods=self.config.n_steps, freq="H"
            )
        else:  # minute
            dates = pd.date_range(
                start=self.config.start_date, periods=self.config.n_steps, freq="1min"
            )

        combined["timestamp"] = dates
        combined.set_index("timestamp", inplace=True)

        return combined

    def _add_microstructure_effects(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add realistic market microstructure effects"""
        # Add bid-ask bounce
        for asset in self.assets:
            close_col = f"{asset.symbol}_close"
            bid_0_col = f"{asset.symbol}_bid_0"
            ask_0_col = f"{asset.symbol}_ask_0"

            if all(col in data.columns for col in [close_col, bid_0_col, ask_0_col]):
                # Simulate trades executing at bid or ask
                mid_price = (data[bid_0_col] + data[ask_0_col]) / 2
                trade_direction = np.random.choice(
                    [-1, 0, 1], size=len(data), p=[0.3, 0.4, 0.3]
                )

                # Adjust close prices to simulate execution at bid/ask
                price_adjustment = np.where(
                    trade_direction == 1,  # Buy
                    data[ask_0_col] - mid_price,
                    np.where(
                        trade_direction == -1,  # Sell
                        data[bid_0_col] - mid_price,
                        0,  # No trade
                    ),
                )

                data[close_col] = (
                    mid_price + price_adjustment * 0.1
                )  # Partial adjustment

        # Add delayed price discovery
        for asset in self.assets:
            returns_col = f"{asset.symbol}_returns"
            if returns_col in data.columns:
                # Add small lag to price discovery
                data[returns_col] = (
                    data[returns_col].rolling(window=2).mean().fillna(data[returns_col])
                )

        return data

    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated regimes"""
        regime_counts = {}
        regime_durations = {}

        for i, regime in enumerate(self.config.regimes):
            regime_idx = i
            count = np.sum(self.regime_sequence == regime_idx)
            regime_counts[regime.name] = int(count)

            # Calculate average duration
            durations = []
            current_duration = 0
            for step in self.regime_sequence:
                if step == regime_idx:
                    current_duration += 1
                elif current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0

            if current_duration > 0:
                durations.append(current_duration)

            regime_durations[regime.name] = {
                "average_duration": np.mean(durations) if durations else 0,
                "max_duration": max(durations) if durations else 0,
                "min_duration": min(durations) if durations else 0,
                "num_periods": len(durations),
            }

        return {
            "total_steps": len(self.regime_sequence),
            "regime_counts": regime_counts,
            "regime_durations": regime_durations,
            "num_transitions": len(self.regime_transitions),
        }


# Factory function for easy usage
def create_synthetic_data(
    assets: List[AssetConfig],
    n_steps: int = 252,
    frequency: str = "daily",
    start_date: datetime = None,
    custom_regimes: List[RegimeParameters] = None,
) -> pd.DataFrame:
    """
    Factory function to create synthetic market data with sensible defaults.

    Args:
        assets: List of asset configurations
        n_steps: Number of time steps to generate
        frequency: Data frequency ("daily", "hourly", "minute")
        start_date: Start date for the data
        custom_regimes: Optional custom regime parameters

    Returns:
        DataFrame with synthetic market data
    """
    if start_date is None:
        start_date = datetime(2020, 1, 1)

    config = SyntheticDataConfig(
        n_assets=len(assets),
        n_steps=n_steps,
        frequency=frequency,
        start_date=start_date,
        regimes=custom_regimes,
    )

    generator = MarketDataGenerator(config)
    return generator.generate_market_data(assets)
