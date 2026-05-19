"""
CVaR Reward Shaper for RL Financial Markets Gym

Implements Conditional Value-at-Risk (CVaR) reward shaping for risk-aware
reinforcement learning. Provides multiple risk measures and reward adjustment
strategies for safe and robust trading strategies.
"""

import numpy as np
import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import scipy.stats as stats

logger = logging.getLogger(__name__)


class RiskMeasure(Enum):
    """Available risk measures for reward shaping"""

    CVAR = "cvar"  # Conditional Value-at-Risk
    VAR = "var"  # Value-at-Risk
    ES = "es"  # Expected Shortfall (same as CVaR)
    CDAR = "cdar"  # Conditional Drawdown at Risk
    MAX_DD = "max_dd"  # Maximum Drawdown
    SEMIVAR = "semivar"  # Semi-variance
    SKEWNESS = "skewness"  # Skewness preference
    KURTOSIS = "kurtosis"  # Kurtosis preference


@dataclass
class CVaRConfig:
    """Configuration for CVaR reward shaping"""

    confidence_level: float = 0.05  # CVaR confidence level (5% CVaR)
    window_size: int = 252  # Number of periods for risk calculation
    risk_aversion: float = 1.0  # Risk aversion coefficient
    reward_shaping_method: str = "cvar_adjustment"  # Method for reward adjustment
    benchmark_returns: Optional[np.ndarray] = None  # Benchmark returns for comparison
    min_samples: int = 30  # Minimum samples for risk calculation


class CVaRRewardShaper:
    """
    Advanced CVaR-based reward shaper for risk-aware reinforcement learning.

    Features:
    - CVaR and Expected Shortfall calculations
    - Multiple risk measures integration
    - Dynamic risk adjustment
    - Benchmark-relative performance
    - Risk budget management
    """

    def __init__(
        self,
        config: CVaRConfig = None,
        risk_measures: List[RiskMeasure] = None,
        enable_risk_budget: bool = True,
    ):
        """
        Initialize CVaR reward shaper.

        Args:
            config: CVaR configuration
            risk_measures: List of risk measures to use
            enable_risk_budget: Enable risk budget management
        """
        self.config = config or CVaRConfig()
        self.risk_measures = risk_measures or [
            RiskMeasure.CVAR,
            RiskMeasure.VAR,
            RiskMeasure.MAX_DD,
        ]
        self.enable_risk_budget = enable_risk_budget

        # Historical returns for risk calculation
        self.returns_history = deque(maxlen=self.config.window_size)
        self.portfolio_values = deque(maxlen=self.config.window_size * 2)

        # Risk budget
        self.risk_budget = 0.05  # 5% max CVaR
        self.risk_utilization = 0.0

        # Risk metrics cache
        self.risk_metrics_cache = {}
        self.last_calculation_step = -1

        # Performance tracking
        self.reward_adjustments = deque(maxlen=1000)
        self.violations = deque(maxlen=100)

        logger.info(
            f"CVaR Reward Shaper initialized with confidence={self.config.confidence_level}"
        )

    def calculate_cvar(
        self, returns: np.ndarray, confidence_level: float = None
    ) -> float:
        """
        Calculate Conditional Value-at-Risk.

        Args:
            returns: Array of returns
            confidence_level: Confidence level for CVaR

        Returns:
            CVaR value
        """
        if confidence_level is None:
            confidence_level = self.config.confidence_level

        if len(returns) < self.config.min_samples:
            return 0.0

        # Calculate VaR
        var = np.percentile(returns, confidence_level * 100)

        # Calculate CVaR (mean of returns below VaR)
        cvar_returns = returns[returns <= var]
        if len(cvar_returns) > 0:
            return np.mean(cvar_returns)
        else:
            return var

    def calculate_var(
        self, returns: np.ndarray, confidence_level: float = None
    ) -> float:
        """
        Calculate Value-at-Risk.

        Args:
            returns: Array of returns
            confidence_level: Confidence level for VaR

        Returns:
            VaR value
        """
        if confidence_level is None:
            confidence_level = self.config.confidence_level

        if len(returns) < self.config.min_samples:
            return 0.0

        return np.percentile(returns, confidence_level * 100)

    def calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """
        Calculate maximum drawdown.

        Args:
            portfolio_values: Array of portfolio values

        Returns:
            Maximum drawdown as a percentage
        """
        if len(portfolio_values) < 2:
            return 0.0

        # Calculate cumulative maximum
        cumulative_max = np.maximum.accumulate(portfolio_values)

        # Calculate drawdown
        drawdown = (cumulative_max - portfolio_values) / cumulative_max

        return np.max(drawdown)

    def calculate_conditional_drawdown_at_risk(
        self, portfolio_values: np.ndarray, confidence_level: float = None
    ) -> float:
        """
        Calculate Conditional Drawdown at Risk (CDaR).

        Args:
            portfolio_values: Array of portfolio values
            confidence_level: Confidence level for CDaR

        Returns:
            CDaR value
        """
        if confidence_level is None:
            confidence_level = self.config.confidence_level

        if len(portfolio_values) < self.config.min_samples:
            return 0.0

        # Calculate drawdown series
        cumulative_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (cumulative_max - portfolio_values) / cumulative_max

        # Calculate CDaR
        cvar_threshold = np.percentile(drawdowns, confidence_level * 100)
        cdar_drawdowns = drawdowns[drawdowns >= cvar_threshold]

        if len(cdar_drawdowns) > 0:
            return np.mean(cdar_drawdowns)
        else:
            return cvar_threshold

    def calculate_semivariance(self, returns: np.ndarray) -> float:
        """
        Calculate semi-variance (downside risk).

        Args:
            returns: Array of returns

        Returns:
            Semi-variance value
        """
        if len(returns) < self.config.min_samples:
            return 0.0

        mean_return = np.mean(returns)
        negative_returns = returns[returns < mean_return]

        if len(negative_returns) > 0:
            return np.mean((negative_returns - mean_return) ** 2)
        else:
            return 0.0

    def calculate_skewness_preference(self, returns: np.ndarray) -> float:
        """
        Calculate skewness preference reward (positive skewness is preferred).

        Args:
            returns: Array of returns

        Returns:
            Skewness preference reward
        """
        if len(returns) < self.config.min_samples:
            return 0.0

        skewness = stats.skew(returns)

        # Positive skewness is good, negative is bad
        # Apply a tanh function to bound the reward
        return np.tanh(skewness) * 0.1  # Scale down to keep it reasonable

    def calculate_kurtosis_preference(self, returns: np.ndarray) -> float:
        """
        Calculate kurtosis preference reward (lower kurtosis is preferred).

        Args:
            returns: Array of returns

        Returns:
            Kurtosis preference reward
        """
        if len(returns) < self.config.min_samples:
            return 0.0

        # Calculate excess kurtosis
        kurtosis = stats.kurtosis(returns, fisher=True)

        # Negative excess kurtosis is good (lighter tails), positive is bad (heavier tails)
        # Apply tanh and scale appropriately
        return -np.tanh(kurtosis / 5) * 0.05  # Scale down

    def calculate_risk_metrics(self, current_step: int) -> Dict[str, float]:
        """
        Calculate all configured risk metrics.

        Args:
            current_step: Current step number

        Returns:
            Dictionary of risk metrics
        """
        if (
            current_step == self.last_calculation_step
            and self.risk_metrics_cache
            and len(self.returns_history) > self.config.min_samples
        ):
            return self.risk_metrics_cache

        if len(self.returns_history) < self.config.min_samples:
            return {measure.value: 0.0 for measure in self.risk_measures}

        returns_array = np.array(self.returns_history)
        portfolio_values_array = np.array(self.portfolio_values)

        metrics = {}

        for risk_measure in self.risk_measures:
            if risk_measure == RiskMeasure.CVAR:
                metrics[risk_measure.value] = self.calculate_cvar(returns_array)
            elif risk_measure == RiskMeasure.VAR:
                metrics[risk_measure.value] = self.calculate_var(returns_array)
            elif risk_measure == RiskMeasure.ES:
                metrics[risk_measure.value] = self.calculate_cvar(
                    returns_array
                )  # Same as CVaR
            elif risk_measure == RiskMeasure.CDAR:
                metrics[
                    risk_measure.value
                ] = self.calculate_conditional_drawdown_at_risk(portfolio_values_array)
            elif risk_measure == RiskMeasure.MAX_DD:
                metrics[risk_measure.value] = self.calculate_max_drawdown(
                    portfolio_values_array
                )
            elif risk_measure == RiskMeasure.SEMIVAR:
                metrics[risk_measure.value] = self.calculate_semivariance(returns_array)
            elif risk_measure == RiskMeasure.SKEWNESS:
                metrics[risk_measure.value] = self.calculate_skewness_preference(
                    returns_array
                )
            elif risk_measure == RiskMeasure.KURTOSIS:
                metrics[risk_measure.value] = self.calculate_kurtosis_preference(
                    returns_array
                )

        # Cache results
        self.risk_metrics_cache = metrics
        self.last_calculation_step = current_step

        return metrics

    def shape_reward(
        self, base_reward: float, current_step: int, additional_info: Dict = None
    ) -> float:
        """
        Shape reward based on risk metrics.

        Args:
            base_reward: Original reward
            current_step: Current step number
            additional_info: Additional information for risk calculation

        Returns:
            Risk-adjusted reward
        """
        # Update returns and portfolio values
        if additional_info and "portfolio_value" in additional_info:
            self.portfolio_values.append(additional_info["portfolio_value"])

            # Calculate return if we have previous value
            if len(self.portfolio_values) > 1:
                prev_value = self.portfolio_values[-2]
                current_value = self.portfolio_values[-1]
                if prev_value > 0:
                    return_rate = (current_value - prev_value) / prev_value
                    self.returns_history.append(return_rate)

        # Calculate risk metrics
        risk_metrics = self.calculate_risk_metrics(current_step)

        # Apply reward shaping based on configuration
        if self.config.reward_shaping_method == "cvar_adjustment":
            adjusted_reward = self._cvar_adjustment(base_reward, risk_metrics)
        elif self.config.reward_shaping_method == "var_penalty":
            adjusted_reward = self._var_penalty(base_reward, risk_metrics)
        elif self.config.reward_shaping_method == "risk_budget":
            adjusted_reward = self._risk_budget_adjustment(base_reward, risk_metrics)
        elif self.config.reward_shaping_method == "multi_objective":
            adjusted_reward = self._multi_objective_adjustment(
                base_reward, risk_metrics
            )
        else:
            adjusted_reward = base_reward

        # Track reward adjustment
        adjustment = adjusted_reward - base_reward
        self.reward_adjustments.append(adjustment)

        # Check for risk violations
        self._check_risk_violations(risk_metrics)

        return adjusted_reward

    def _cvar_adjustment(
        self, base_reward: float, risk_metrics: Dict[str, float]
    ) -> float:
        """Apply CVaR-based reward adjustment."""
        cvar = risk_metrics.get("cvar", 0.0)

        # CVaR penalty: more negative CVaR = larger penalty
        cvar_penalty = -cvar * self.config.risk_aversion

        return base_reward + cvar_penalty

    def _var_penalty(self, base_reward: float, risk_metrics: Dict[str, float]) -> float:
        """Apply VaR-based penalty."""
        var = risk_metrics.get("var", 0.0)

        # VaR penalty
        var_penalty = (
            -var * self.config.risk_aversion * 0.5
        )  # Reduce penalty compared to CVaR

        return base_reward + var_penalty

    def _risk_budget_adjustment(
        self, base_reward: float, risk_metrics: Dict[str, float]
    ) -> float:
        """Apply risk budget adjustment."""
        cvar = risk_metrics.get("cvar", 0.0)

        # Calculate risk utilization
        self.risk_utilization = abs(cvar) / max(self.risk_budget, 0.001)

        # Heavy penalty for exceeding risk budget
        if self.risk_utilization > 1.0:
            risk_penalty = -(self.risk_utilization - 1.0) * 2.0 * abs(base_reward)
        else:
            # Small incentive for staying within budget
            risk_penalty = (1.0 - self.risk_utilization) * 0.01 * abs(base_reward)

        return base_reward + risk_penalty

    def _multi_objective_adjustment(
        self, base_reward: float, risk_metrics: Dict[str, float]
    ) -> float:
        """Apply multi-objective risk adjustment."""
        adjusted_reward = base_reward

        # CVaR adjustment (primary)
        cvar = risk_metrics.get("cvar", 0.0)
        adjusted_reward -= cvar * self.config.risk_aversion

        # Drawdown penalty
        max_dd = risk_metrics.get("max_dd", 0.0)
        adjusted_reward -= max_dd * self.config.risk_aversion * 0.5

        # Skewness preference
        skew_pref = risk_metrics.get("skewness", 0.0)
        adjusted_reward += skew_pref

        # Kurtosis preference
        kurt_pref = risk_metrics.get("kurtosis", 0.0)
        adjusted_reward += kurt_pref

        return adjusted_reward

    def _check_risk_violations(self, risk_metrics: Dict[str, float]):
        """Check for risk limit violations."""
        current_time = len(self.returns_history)

        # Check CVaR violation
        cvar = risk_metrics.get("cvar", 0.0)
        if cvar < -self.risk_budget:
            self.violations.append(
                {
                    "time": current_time,
                    "type": "cvar_violation",
                    "value": cvar,
                    "limit": -self.risk_budget,
                }
            )

        # Check maximum drawdown violation
        max_dd = risk_metrics.get("max_dd", 0.0)
        if max_dd > 0.2:  # 20% max drawdown limit
            self.violations.append(
                {
                    "time": current_time,
                    "type": "drawdown_violation",
                    "value": max_dd,
                    "limit": 0.2,
                }
            )

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        if not self.risk_metrics_cache:
            return {
                "message": "No risk metrics calculated yet",
                "returns_history_length": len(self.returns_history),
            }

        summary = {
            "risk_metrics": self.risk_metrics_cache.copy(),
            "risk_utilization": self.risk_utilization,
            "risk_budget": self.risk_budget,
            "total_violations": len(self.violations),
            "recent_violations": list(self.violations)[-10:],
            "returns_history_length": len(self.returns_history),
            "portfolio_values_length": len(self.portfolio_values),
            "config": {
                "confidence_level": self.config.confidence_level,
                "window_size": self.config.window_size,
                "risk_aversion": self.config.risk_aversion,
                "reward_shaping_method": self.config.reward_shaping_method,
            },
        }

        # Add reward adjustment statistics
        if self.reward_adjustments:
            adjustments_array = np.array(self.reward_adjustments)
            summary["reward_adjustment_stats"] = {
                "mean_adjustment": np.mean(adjustments_array),
                "std_adjustment": np.std(adjustments_array),
                "min_adjustment": np.min(adjustments_array),
                "max_adjustment": np.max(adjustments_array),
                "negative_adjustments": np.sum(adjustments_array < 0),
                "positive_adjustments": np.sum(adjustments_array > 0),
            }

        return summary

    def reset(self):
        """Reset internal state for new episode."""
        self.returns_history.clear()
        self.portfolio_values.clear()
        self.risk_metrics_cache.clear()
        self.last_calculation_step = -1
        self.risk_utilization = 0.0
        self.reward_adjustments.clear()
        self.violations.clear()

    def update_config(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated CVaR config: {key} = {value}")

    def set_risk_budget(self, budget: float):
        """Set risk budget for CVaR."""
        self.risk_budget = budget
        logger.info(f"Risk budget set to {budget:.2%}")

    def get_current_risk_level(self) -> str:
        """Get current risk level assessment."""
        if not self.risk_metrics_cache:
            return "Unknown"

        cvar = self.risk_metrics_cache.get("cvar", 0.0)
        max_dd = self.risk_metrics_cache.get("max_dd", 0.0)

        if abs(cvar) > self.risk_budget or max_dd > 0.15:
            return "High"
        elif abs(cvar) > self.risk_budget * 0.5 or max_dd > 0.1:
            return "Medium"
        else:
            return "Low"
