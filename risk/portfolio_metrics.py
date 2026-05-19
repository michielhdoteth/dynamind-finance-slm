"""
Advanced Portfolio Risk Metrics for RL Financial Markets Gym

Comprehensive portfolio risk measurement and analysis system for
reinforcement learning. Provides professional-grade risk metrics used in
institutional trading and portfolio management.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import scipy.stats as stats

logger = logging.getLogger(__name__)


@dataclass
class PortfolioMetricsConfig:
    """Configuration for portfolio risk metrics calculation"""

    lookback_period: int = 252  # Lookback period for metrics (1 year)
    confidence_levels: List[float] = None  # Confidence levels for VaR/CVaR
    benchmark_returns: Optional[np.ndarray] = None  # Benchmark returns
    risk_free_rate: float = 0.02  # Annual risk-free rate
    trading_days_per_year: int = 252  # Trading days per year


class PortfolioRiskMetrics:
    """
    Advanced portfolio risk metrics calculator for RL.

    Features:
    - VaR and CVaR at multiple confidence levels
    - Sharpe ratio and information ratio
    - Maximum drawdown and drawdown duration
    - Beta and alpha calculations
    - Portfolio diversification metrics
    - Sortino ratio and Calmar ratio
    - Upside/downside capture ratios
    """

    def __init__(
        self, config: PortfolioMetricsConfig = None, enable_benchmarking: bool = True
    ):
        """
        Initialize portfolio risk metrics calculator.

        Args:
            config: Configuration for metrics calculation
            enable_benchmarking: Enable benchmark-relative metrics
        """
        self.config = config or PortfolioMetricsConfig()
        self.enable_benchmarking = enable_benchmarking

        # Default confidence levels if not specified
        if self.config.confidence_levels is None:
            self.config.confidence_levels = [0.01, 0.025, 0.05, 0.1]

        # Data storage
        self.returns_history = deque(maxlen=self.config.lookback_period * 2)
        self.portfolio_values = deque(maxlen=self.config.lookback_period * 2)
        self.benchmark_history = deque(maxlen=self.config.lookback_period * 2)

        # Metrics cache
        self.metrics_cache = {}
        self.last_calculation_step = -1

        logger.info(
            f"Portfolio Risk Metrics initialized with lookback={self.config.lookback_period}"
        )

    def update_portfolio_data(
        self, portfolio_value: float, benchmark_value: float = None
    ):
        """
        Update portfolio data for metrics calculation.

        Args:
            portfolio_value: Current portfolio value
            benchmark_value: Current benchmark value (optional)
        """
        # Update portfolio values
        if len(self.portfolio_values) > 0:
            prev_value = self.portfolio_values[-1]
            if prev_value > 0:
                return_rate = (portfolio_value - prev_value) / prev_value
                self.returns_history.append(return_rate)

        self.portfolio_values.append(portfolio_value)

        # Update benchmark if provided
        if benchmark_value is not None and self.enable_benchmarking:
            if len(self.benchmark_history) > 0:
                prev_benchmark = self.benchmark_history[-1]
                if prev_benchmark > 0:
                    benchmark_return = (
                        benchmark_value - prev_benchmark
                    ) / prev_benchmark
                    self.benchmark_history.append(benchmark_return)

            self.benchmark_history.append(benchmark_value)

    def calculate_returns_metrics(self) -> Dict[str, float]:
        """Calculate basic return-based metrics."""
        if len(self.returns_history) < 2:
            return {}

        returns_array = np.array(self.returns_history)

        metrics = {}

        # Basic statistics
        metrics["mean_return"] = np.mean(returns_array)
        metrics["std_return"] = np.std(returns_array)
        metrics["min_return"] = np.min(returns_array)
        metrics["max_return"] = np.max(returns_array)
        metrics["median_return"] = np.median(returns_array)

        # Skewness and kurtosis
        metrics["skewness"] = stats.skew(returns_array)
        metrics["excess_kurtosis"] = stats.kurtosis(returns_array, fisher=True)

        # Percentile returns
        for percentile in [5, 10, 25, 75, 90, 95]:
            metrics[f"return_p{percentile}"] = np.percentile(returns_array, percentile)

        # Positive/negative return periods
        positive_periods = np.sum(returns_array > 0)
        total_periods = len(returns_array)
        metrics["win_rate"] = (
            positive_periods / total_periods if total_periods > 0 else 0
        )

        # Average win/loss
        wins = returns_array[returns_array > 0]
        losses = returns_array[returns_array < 0]

        metrics["avg_win"] = np.mean(wins) if len(wins) > 0 else 0
        metrics["avg_loss"] = np.mean(losses) if len(losses) > 0 else 0
        metrics["best_win"] = np.max(wins) if len(wins) > 0 else 0
        metrics["worst_loss"] = np.min(losses) if len(losses) > 0 else 0

        # Profit factor
        total_wins = np.sum(wins) if len(wins) > 0 else 0
        total_losses = abs(np.sum(losses)) if len(losses) > 0 else 0
        metrics["profit_factor"] = (
            total_wins / total_losses if total_losses > 0 else float("in")
        )

        return metrics

    def calculate_var_cvar(self) -> Dict[str, Dict[str, float]]:
        """Calculate VaR and CVaR at multiple confidence levels."""
        if len(self.returns_history) < 30:  # Need minimum samples
            return {}

        returns_array = np.array(self.returns_history)
        results = {}

        for confidence in self.config.confidence_levels:
            # VaR
            var = np.percentile(returns_array, confidence * 100)

            # CVaR (Expected Shortfall)
            cvar_returns = returns_array[returns_array <= var]
            cvar = np.mean(cvar_returns) if len(cvar_returns) > 0 else var

            results[f"var_{int(confidence*100)}"] = var
            results[f"cvar_{int(confidence*100)}"] = cvar
            results[f"expected_shortfall_{int(confidence*100)}"] = cvar  # Alias

        return results

    def calculate_sharpe_ratio(self) -> Dict[str, float]:
        """Calculate Sharpe ratio and related metrics."""
        if len(self.returns_history) < 30:
            return {}

        returns_array = np.array(self.returns_history)
        metrics = {}

        # Annualized return
        annual_return = np.mean(returns_array) * self.config.trading_days_per_year

        # Annualized volatility
        annual_volatility = np.std(returns_array) * np.sqrt(
            self.config.trading_days_per_year
        )

        # Sharpe ratio
        if annual_volatility > 0:
            metrics["sharpe_ratio"] = (
                annual_return - self.config.risk_free_rate
            ) / annual_volatility

        # Sortino ratio (using downside deviation)
        downside_returns = returns_array[returns_array < np.mean(returns_array)]
        if len(downside_returns) > 0:
            downside_deviation = np.std(downside_returns) * np.sqrt(
                self.config.trading_days_per_year
            )
            if downside_deviation > 0:
                metrics["sortino_ratio"] = (
                    annual_return - self.config.risk_free_rate
                ) / downside_deviation

        # Calmar ratio (using max drawdown)
        max_dd = self.calculate_max_drawdown()
        if max_dd > 0:
            metrics["calmar_ratio"] = annual_return / max_dd

        return metrics

    def calculate_drawdown_metrics(self) -> Dict[str, float]:
        """Calculate drawdown-related metrics."""
        if len(self.portfolio_values) < 2:
            return {}

        portfolio_values = np.array(self.portfolio_values)
        metrics = {}

        # Maximum drawdown
        cumulative_max = np.maximum.accumulate(portfolio_values)
        drawdown = (cumulative_max - portfolio_values) / cumulative_max

        metrics["max_drawdown"] = np.max(drawdown)
        metrics["avg_drawdown"] = np.mean(drawdown)
        metrics["current_drawdown"] = drawdown[-1]

        # Drawdown duration
        in_drawdown = drawdown > 0
        drawdown_periods = []
        current_duration = 0

        for dd in in_drawdown:
            if dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    drawdown_periods.append(current_duration)
                    current_duration = 0

        if current_duration > 0:
            drawdown_periods.append(current_duration)

        if drawdown_periods:
            metrics["max_drawdown_duration"] = max(drawdown_periods)
            metrics["avg_drawdown_duration"] = np.mean(drawdown_periods)
            metrics["current_drawdown_duration"] = current_duration

        # Recovery time (time to recover from drawdown)
        recovery_times = []
        for i in range(1, len(drawdown)):
            if drawdown[i - 1] > 0 and drawdown[i] == 0:
                recovery_times.append(i)

        if recovery_times:
            metrics["avg_recovery_time"] = np.mean(recovery_times)
            metrics["max_recovery_time"] = max(recovery_times)

        # Ulcer index (average squared drawdown)
        ulcer_index = np.mean(drawdown**2) if len(drawdown) > 0 else 0
        metrics["ulcer_index"] = ulcer_index * 100  # Usually expressed as percentage

        return metrics

    def calculate_maximum_drawdown(self) -> float:
        """Calculate maximum drawdown (shortcut method)."""
        if len(self.portfolio_values) < 2:
            return 0.0

        portfolio_values = np.array(self.portfolio_values)
        cumulative_max = np.maximum.accumulate(portfolio_values)
        drawdown = (cumulative_max - portfolio_values) / cumulative_max
        return np.max(drawdown)

    def calculate_beta_alpha(self) -> Dict[str, float]:
        """Calculate beta and alpha relative to benchmark."""
        if not self.enable_benchmarking or len(self.benchmark_history) < 30:
            return {}

        portfolio_returns = np.array(self.returns_history)
        benchmark_returns = np.array(self.benchmark_history)

        # Align arrays (take minimum length)
        min_length = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns[-min_length:]
        benchmark_returns = benchmark_returns[-min_length:]

        if len(portfolio_returns) < 2 or len(benchmark_returns) < 2:
            return {}

        metrics = {}

        # Calculate beta (covariance / variance)
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)

        if benchmark_variance > 0:
            metrics["beta"] = covariance / benchmark_variance

        # Calculate alpha (Jensen's alpha)
        risk_free_daily = self.config.risk_free_rate / self.config.trading_days_per_year
        excess_returns = portfolio_returns - risk_free_daily
        excess_benchmark_returns = benchmark_returns - risk_free_daily

        if len(excess_benchmark_returns) > 0 and np.var(excess_benchmark_returns) > 0:
            beta = metrics.get("beta", 1.0)
            expected_excess_return = beta * np.mean(excess_benchmark_returns)
            annual_alpha = (
                np.mean(excess_returns) - expected_excess_return
            ) * self.config.trading_days_per_year
            metrics["alpha"] = annual_alpha

        # Calculate information ratio (alpha / tracking error)
        tracking_error = portfolio_returns - benchmark_returns
        if len(tracking_error) > 0 and np.var(tracking_error) > 0:
            tracking_error_vol = np.std(tracking_error) * np.sqrt(
                self.config.trading_days_per_year
            )
            if tracking_error_vol > 0:
                metrics["information_ratio"] = metrics["alpha"] / tracking_error_vol

        # R-squared (explanatory power)
        if len(benchmark_returns) > 1 and np.var(benchmark_returns) > 0:
            r_squared = np.corr(portfolio_returns, benchmark_returns) ** 2
            metrics["r_squared"] = r_squared

        return metrics

    def calculate_diversification_metrics(self) -> Dict[str, float]:
        """Calculate portfolio diversification metrics."""
        # This is a simplified version - in practice you'd need individual asset returns
        metrics = {}

        # Herfindahl-Hirschman Index (HHI) - simplified
        # In practice, you'd calculate this from individual asset weights
        metrics["concentration_ratio"] = 0.0  # Placeholder

        # Effective number of bets
        if len(self.returns_history) > 30:
            # Very simplified approximation
            # In reality, this requires factor analysis
            metrics["effective_bets"] = 10.0  # Placeholder

        return metrics

    def calculate_capture_ratios(self) -> Dict[str, float]:
        """Calculate upside/downside capture ratios relative to benchmark."""
        if not self.enable_benchmarking or len(self.benchmark_history) < 30:
            return {}

        portfolio_returns = np.array(self.returns_history)
        benchmark_returns = np.array(self.benchmark_returns)

        # Align arrays
        min_length = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns[-min_length:]
        benchmark_returns = benchmark_returns[-min_length:]

        metrics = {}

        # Upside capture ratio
        up_periods = benchmark_returns > 0
        if np.any(up_periods) and len(portfolio_returns[up_periods]) > 0:
            portfolio_up = portfolio_returns[up_periods]
            benchmark_up = benchmark_returns[up_periods]

            portfolio_up_return = np.sum(portfolio_up)
            benchmark_up_return = np.sum(benchmark_up)

            if benchmark_up_return > 0:
                metrics["upside_capture"] = portfolio_up_return / benchmark_up_return

        # Downside capture ratio
        down_periods = benchmark_returns < 0
        if np.any(down_periods) and len(portfolio_returns[down_periods]) > 0:
            portfolio_down = portfolio_returns[down_periods]
            benchmark_down = benchmark_returns[down_periods]

            portfolio_down_return = np.sum(portfolio_down)
            benchmark_down_return = np.sum(benchmark_down)

            if benchmark_down_return != 0:
                metrics["downside_capture"] = (
                    portfolio_down_return / benchmark_down_return
                )

        # Overall capture ratio
        total_portfolio_return = np.sum(portfolio_returns)
        total_benchmark_return = np.sum(benchmark_returns)

        if total_benchmark_return != 0:
            metrics["overall_capture"] = total_portfolio_return / total_benchmark_return

        return metrics

    def calculate_all_metrics(self, current_step: int = None) -> Dict[str, Any]:
        """
        Calculate all portfolio risk metrics.

        Args:
            current_step: Current step for caching

        Returns:
            Dictionary containing all metrics
        """
        # Use cache if available and recent
        if (
            current_step is not None
            and current_step == self.last_calculation_step
            and self.metrics_cache
        ):
            return self.metrics_cache

        metrics = {}

        # Basic return metrics
        metrics.update(self.calculate_returns_metrics())

        # VaR and CVaR
        var_cvar_metrics = self.calculate_var_cvar()
        metrics["var_cvar"] = var_cvar_metrics

        # Sharpe ratio and related
        sharpe_metrics = self.calculate_sharpe_ratio()
        metrics["sharpe_metrics"] = sharpe_metrics

        # Drawdown metrics
        drawdown_metrics = self.calculate_drawdown_metrics()
        metrics["drawdown_metrics"] = drawdown_metrics

        # Beta and alpha
        beta_alpha_metrics = self.calculate_beta_alpha()
        metrics["beta_alpha_metrics"] = beta_alpha_metrics

        # Diversification metrics
        diversification_metrics = self.calculate_diversification_metrics()
        metrics["diversification_metrics"] = diversification_metrics

        # Capture ratios
        capture_metrics = self.calculate_capture_ratios()
        metrics["capture_metrics"] = capture_metrics

        # Additional derived metrics
        metrics["risk_adjusted_return"] = metrics.get("sharpe_metrics", {}).get(
            "sharpe_ratio", 0.0
        ) * metrics.get("sharpe_metrics", {}).get("std_return", 0.0)

        # Risk-adjusted metrics summary
        var_5 = var_cvar_metrics.get("var_5", 0.0)
        max_dd = drawdown_metrics.get("max_drawdown", 0.0)

        metrics["risk_summary"] = {
            "var_5_percent": var_5,
            "max_drawdown": max_dd,
            "sharpe_ratio": metrics.get("sharpe_metrics", {}).get("sharpe_ratio", 0.0),
            "alpha": metrics.get("beta_alpha_metrics", {}).get("alpha", 0.0),
            "beta": metrics.get("beta_alpha_metrics", {}).get("beta", 1.0),
        }

        # Cache results
        self.metrics_cache = metrics
        self.last_calculation_step = current_step if current_step is not None else -1

        return metrics

    def get_risk_metrics_for_observation(self) -> np.ndarray:
        """Get key risk metrics for inclusion in RL observation."""
        metrics = self.calculate_all_metrics()

        # Select most important metrics for observation
        obs_metrics = [
            metrics.get("sharpe_metrics", {}).get("sharpe_ratio", 0.0),
            metrics.get("drawdown_metrics", {}).get("max_drawdown", 0.0),
            metrics.get("drawdown_metrics", {}).get("current_drawdown", 0.0),
            metrics.get("var_cvar", {}).get("var_5", 0.0),
            metrics.get("var_cvar", {}).get("cvar_5", 0.0),
            metrics.get("beta_alpha_metrics", {}).get("beta", 1.0),
            metrics.get("beta_alpha_metrics", {}).get("alpha", 0.0),
            metrics.get("returns_metrics", {}).get("skewness", 0.0),
            metrics.get("returns_metrics", {}).get("profit_factor", 1.0),
            metrics.get("returns_metrics", {}).get("win_rate", 0.5),
        ]

        return np.array(obs_metrics, dtype=np.float32)

    def get_performance_attribution(self) -> Dict[str, Any]:
        """
        Get performance attribution analysis (simplified).

        Returns:
            Performance attribution breakdown
        """
        # This is a placeholder - in a real implementation you'd need
        # individual asset returns and weights to calculate attribution
        return {
            "note": "Performance attribution requires individual asset data",
            "total_return": np.mean(self.returns_history)
            if len(self.returns_history) > 0
            else 0.0,
            "benchmark_return": np.mean(self.benchmark_history)
            if len(self.benchmark_history) > 0
            else 0.0,
        }

    def reset(self):
        """Reset all stored data."""
        self.returns_history.clear()
        self.portfolio_values.clear()
        self.benchmark_history.clear()
        self.metrics_cache.clear()
        self.last_calculation_step = -1

        logger.info("Portfolio risk metrics reset")

    def export_metrics_to_dataframe(self) -> pd.DataFrame:
        """Export risk metrics to pandas DataFrame for analysis."""
        # Create time series of portfolio values and returns
        data = {
            "portfolio_value": list(self.portfolio_values),
            "return": list(self.returns_history),
        }

        if self.enable_benchmarking:
            data["benchmark_value"] = list(self.benchmark_history)
            data["benchmark_return"] = list(self.benchmark_history)

        df = pd.DataFrame(data)
        return df

    def generate_risk_report(self) -> str:
        """Generate a comprehensive risk report."""
        metrics = self.calculate_all_metrics()

        report = []
        report.append("PORTFOLLO RISK ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Data Period: {len(self.returns_history)} observations")
        report.append("")

        # Performance Summary
        report.append("PERFORMANCE SUMMARY")
        report.append("-" * 30)
        report.append(
            f"Sharpe Ratio: {metrics.get('sharpe_metrics', {}).get('sharpe_ratio', 0):.3f}"
        )
        report.append(
            f"Alpha: {metrics.get('beta_alpha_metrics', {}).get('alpha', 0):.3f}"
        )
        report.append(
            f"Beta: {metrics.get('beta_alpha_metrics', {}).get('beta', 0):.3f}"
        )
        report.append(
            f"R-squared: {metrics.get('beta_alpha_metrics', {}).get('r_squared', 0):.3f}"
        )
        report.append("")

        # Risk Metrics
        report.append("RISK METRICS")
        report.append("-" * 20)
        report.append(f"5% VaR: {metrics.get('var_cvar', {}).get('var_5', 0):.3%}")
        report.append(f"5% CVaR: {metrics.get('var_cvar', {}).get('cvar_5', 0):.3%}")
        report.append(
            f"Max Drawdown: {metrics.get('drawdown_metrics', {}).get('max_drawdown', 0):.3%}"
        )
        report.append(
            f"Current DD: {metrics.get('drawdown_metrics', {}).get('current_drawdown', 0):.3%}"
        )
        report.append("")

        # Return Statistics
        report.append("RETURN STATISTICS")
        report.append("-" * 25)
        returns_metrics = metrics.get("returns_metrics", {})
        report.append(f"Mean Return: {returns_metrics.get('mean_return', 0):.4f}")
        report.append(f"Std Deviation: {returns_metrics.get('std_return', 0):.4f}")
        report.append(f"Skewness: {returns_metrics.get('skewness', 0):.4f}")
        report.append(f"Win Rate: {returns_metrics.get('win_rate', 0):.2%}")
        report.append(f"Profit Factor: {returns_metrics.get('profit_factor', 0):.2f}")
        report.append("")

        return "\n".join(report)
