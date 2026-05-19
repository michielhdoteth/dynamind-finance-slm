"""
Model Evaluator for RL Financial Markets Gym

Comprehensive model evaluation system with multiple metrics, risk assessment,
performance attribution, and benchmarking capabilities.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""

    # Environment settings
    env_type: str = "single_asset"
    symbols: List[str] = None
    start_date: str = "2023-01-01"
    end_date: str = "2023-12-31"
    data_source: str = "yahoo"
    frequency: str = "1d"

    # Evaluation parameters
    n_episodes: int = 100
    max_steps_per_episode: int = 1000
    deterministic: bool = True
    render: bool = False

    # Risk settings
    enable_risk_analysis: bool = True
    confidence_levels: List[float] = None
    rolling_window: int = 30

    # Benchmark settings
    benchmarks: List[str] = None
    benchmark_data: Dict[str, pd.DataFrame] = None

    # Output settings
    save_results: bool = True
    output_dir: str = "./evaluation_results"
    create_plots: bool = True

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["AAPL", "MSFT", "JPM"]
        if self.confidence_levels is None:
            self.confidence_levels = [0.90, 0.95, 0.99]
        if self.benchmarks is None:
            self.benchmarks = ["buy_and_hold", "random_policy", "equal_weight"]


class PerformanceMetrics:
    """Calculate and analyze performance metrics."""

    @staticmethod
    def calculate_returns(returns: np.ndarray) -> Dict[str, float]:
        """Calculate basic return metrics."""
        if len(returns) == 0:
            return {}

        total_return = np.prod(1 + returns) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
        }

    @staticmethod
    def calculate_drawdown_metrics(cumulative_returns: np.ndarray) -> Dict[str, float]:
        """Calculate drawdown metrics."""
        if len(cumulative_returns) == 0:
            return {}

        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (running_max - cumulative_returns) / running_max

        max_drawdown = np.max(drawdown)
        max_drawdown_duration = self._calculate_max_dd_duration(drawdown)

        return {
            "max_drawdown": max_drawdown,
            "avg_drawdown": np.mean(drawdown[drawdown > 0])
            if np.any(drawdown > 0)
            else 0,
            "max_drawdown_duration": max_drawdown_duration,
        }

    @staticmethod
    def _calculate_max_dd_duration(drawdown: np.ndarray) -> int:
        """Calculate maximum drawdown duration in periods."""
        in_drawdown = drawdown > 0
        durations = []
        current_duration = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0

        if current_duration > 0:
            durations.append(current_duration)

        return max(durations) if durations else 0

    @staticmethod
    def calculate_var_cvar(
        returns: np.ndarray, confidence_levels: List[float]
    ) -> Dict[str, float]:
        """Calculate VaR and CVaR at multiple confidence levels."""
        metrics = {}

        for confidence in confidence_levels:
            var = np.percentile(returns, (1 - confidence) * 100)
            cvar_returns = returns[returns <= var]
            cvar = np.mean(cvar_returns) if len(cvar_returns) > 0 else var

            metrics[f"var_{int(confidence*100)}"] = var
            metrics[f"cvar_{int(confidence*100)}"] = cvar

        return metrics

    @staticmethod
    def calculate_calmar_ratio(returns: np.ndarray) -> float:
        """Calculate Calmar ratio."""
        if len(returns) == 0:
            return 0

        total_return = np.prod(1 + returns) - 1
        years = len(returns) / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (running_max - cumulative_returns) / running_max
        max_drawdown = np.max(drawdown)

        return annualized_return / max_drawdown if max_drawdown > 0 else 0

    @staticmethod
    def calculate_sortino_ratio(
        returns: np.ndarray, risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sortino ratio."""
        if len(returns) == 0:
            return 0

        mean_return = np.mean(returns) * 252
        downside_returns = returns[returns < 0]
        downside_deviation = (
            np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        )

        return (
            (mean_return - risk_free_rate) / downside_deviation
            if downside_deviation > 0
            else 0
        )


class RiskAnalyzer:
    """Analyze risk characteristics of trading strategies."""

    def __init__(self, confidence_levels: List[float] = None):
        self.confidence_levels = confidence_levels or [0.90, 0.95, 0.99]

    def analyze_portfolio_risk(
        self, returns: np.ndarray, positions: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """Comprehensive portfolio risk analysis."""
        risk_analysis = {}

        # Basic risk metrics
        risk_analysis.update(
            PerformanceMetrics.calculate_var_cvar(returns, self.confidence_levels)
        )

        # Volatility analysis
        risk_analysis["volatility"] = {
            "daily": np.std(returns),
            "annualized": np.std(returns) * np.sqrt(252),
            "rolling_30d": self._rolling_volatility(returns, 30),
        }

        # Tail risk
        risk_analysis["tail_risk"] = self._analyze_tail_risk(returns)

        # Position concentration (if positions provided)
        if positions is not None:
            risk_analysis["concentration"] = self._analyze_concentration(positions)

        # Correlation analysis (if multiple assets)
        if returns.ndim > 1:
            risk_analysis["correlation"] = self._analyze_correlations(returns)

        return risk_analysis

    def _rolling_volatility(self, returns: np.ndarray, window: int) -> Dict[str, float]:
        """Calculate rolling volatility statistics."""
        if len(returns) < window:
            return {
                "mean": np.std(returns),
                "std": 0,
                "max": np.std(returns),
                "min": np.std(returns),
            }

        rolling_vol = pd.Series(returns).rolling(window).std().dropna()
        return {
            "mean": rolling_vol.mean(),
            "std": rolling_vol.std(),
            "max": rolling_vol.max(),
            "min": rolling_vol.min(),
        }

    def _analyze_tail_risk(self, returns: np.ndarray) -> Dict[str, float]:
        """Analyze tail risk characteristics."""
        if len(returns) == 0:
            return {}

        # Skewness and kurtosis
        from scipy import stats

        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns, fisher=True)  # Excess kurtosis

        # Tail ratios
        lower_5 = np.percentile(returns, 5)
        upper_5 = np.percentile(returns, 95)
        tail_ratio = abs(lower_5 / upper_5) if upper_5 != 0 else float("in")

        # Expected shortfall
        var_95 = np.percentile(returns, 5)
        expected_shortfall = np.mean(returns[returns <= var_95])

        return {
            "skewness": skewness,
            "excess_kurtosis": kurtosis,
            "tail_ratio": tail_ratio,
            "expected_shortfall_95": expected_shortfall,
        }

    def _analyze_concentration(self, positions: pd.DataFrame) -> Dict[str, Any]:
        """Analyze position concentration risk."""
        if positions.empty:
            return {}

        # Calculate position weights
        total_value = positions.abs().sum(axis=1)
        weights = positions.abs().div(total_value, axis=0)

        # Concentration metrics
        max_concentration = weights.max(axis=1)
        herfindahl_index = (weights**2).sum(axis=1)

        return {
            "avg_max_concentration": max_concentration.mean(),
            "max_concentration": max_concentration.max(),
            "avg_herfindahl": herfindahl_index.mean(),
            "max_herfindahl": herfindahl_index.max(),
        }

    def _analyze_correlations(self, returns: np.ndarray) -> Dict[str, Any]:
        """Analyze correlation structure."""
        if returns.ndim == 1:
            return {}

        corr_matrix = np.corrcoef(returns.T)
        np.fill_diagonal(corr_matrix, np.nan)  # Remove diagonal

        # Remove NaN values for statistics
        corr_values = corr_matrix[~np.isnan(corr_matrix)]

        return {
            "mean_correlation": np.mean(corr_values),
            "max_correlation": np.max(corr_values),
            "min_correlation": np.min(corr_values),
            "std_correlation": np.std(corr_values),
        }


class BenchmarkEvaluator:
    """Evaluate models against benchmark strategies."""

    def __init__(self):
        self.benchmarks = {}

    def register_benchmark(self, name: str, strategy_func: callable):
        """Register a benchmark strategy."""
        self.benchmarks[name] = strategy_func

    def evaluate_benchmarks(
        self, env, n_episodes: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate all registered benchmarks."""
        results = {}

        for name, strategy_func in self.benchmarks.items():
            try:
                benchmark_returns = self._run_benchmark(env, strategy_func, n_episodes)
                results[name] = self._calculate_benchmark_metrics(benchmark_returns)
                logger.info(f"Evaluated benchmark: {name}")
            except Exception as e:
                logger.error(f"Failed to evaluate benchmark {name}: {e}")

        return results

    def _run_benchmark(
        self, env, strategy_func: callable, n_episodes: int
    ) -> List[np.ndarray]:
        """Run a benchmark strategy."""
        episode_returns = []

        for episode in range(n_episodes):
            obs = env.reset()
            done = False
            episode_return = 0

            while not done:
                action = strategy_func(obs)
                obs, reward, done, info = env.step(action)
                episode_return += reward

            episode_returns.append(episode_return)

        return episode_returns

    def _calculate_benchmark_metrics(self, returns: List[float]) -> Dict[str, float]:
        """Calculate metrics for benchmark returns."""
        if not returns:
            return {}

        returns_array = np.array(returns)
        return {
            "mean_return": np.mean(returns_array),
            "std_return": np.std(returns_array),
            "sharpe_ratio": np.mean(returns_array) / np.std(returns_array)
            if np.std(returns_array) > 0
            else 0,
            "max_return": np.max(returns_array),
            "min_return": np.min(returns_array),
        }


# Default benchmark strategies
def buy_and_hold_strategy(observation):
    """Buy and hold strategy."""
    return 1  # Always buy


def random_strategy(observation):
    """Random strategy."""

    return random.randint(0, 3)


class ModelEvaluator:
    """
    Comprehensive model evaluation system.

    Features:
    - Multiple performance metrics
    - Risk analysis and assessment
    - Benchmarking against standard strategies
    - Performance attribution
    - Visualization and reporting
    """

    def __init__(self, config: EvaluationConfig = None):
        """
        Initialize model evaluator.

        Args:
            config: Evaluation configuration
        """
        self.config = config or EvaluationConfig()
        self.performance_metrics = PerformanceMetrics()
        self.risk_analyzer = RiskAnalyzer(self.config.confidence_levels)
        self.benchmark_evaluator = BenchmarkEvaluator()

        # Register default benchmarks
        self.benchmark_evaluator.register_benchmark(
            "buy_and_hold", buy_and_hold_strategy
        )
        self.benchmark_evaluator.register_benchmark("random", random_strategy)

        # Create output directory
        if self.config.save_results:
            os.makedirs(self.config.output_dir, exist_ok=True)

        logger.info(
            f"ModelEvaluator initialized with {self.config.n_episodes} evaluation episodes"
        )

    def evaluate_model(self, model, env) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.

        Args:
            model: Trained RL model
            env: Environment for evaluation

        Returns:
            Comprehensive evaluation results
        """
        logger.info("Starting comprehensive model evaluation...")

        evaluation_results = {
            "config": asdict(self.config),
            "timestamp": datetime.now().isoformat(),
            "model_info": self._get_model_info(model),
        }

        # Run evaluation episodes
        episode_results = self._run_evaluation_episodes(model, env)
        evaluation_results["episode_results"] = episode_results

        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(episode_results)
        evaluation_results["performance_metrics"] = performance_metrics

        # Risk analysis
        if self.config.enable_risk_analysis:
            risk_analysis = self._analyze_risk(episode_results)
            evaluation_results["risk_analysis"] = risk_analysis

        # Benchmarking
        if self.config.benchmarks:
            benchmark_results = self._run_benchmarks(env)
            evaluation_results["benchmark_results"] = benchmark_results

        # Performance attribution
        attribution = self._performance_attribution(episode_results)
        evaluation_results["attribution"] = attribution

        # Generate visualizations
        if self.config.create_plots:
            plots = self._create_evaluation_plots(episode_results, evaluation_results)
            evaluation_results["plots"] = plots

        # Save results
        if self.config.save_results:
            self._save_evaluation_results(evaluation_results)

        logger.info("Model evaluation completed")
        return evaluation_results

    def _run_evaluation_episodes(self, model, env) -> List[Dict[str, Any]]:
        """Run evaluation episodes."""
        episode_results = []

        for episode in range(self.config.n_episodes):
            obs = env.reset()
            done = False
            step = 0
            episode_reward = 0
            episode_data = {
                "episode": episode,
                "steps": [],
                "rewards": [],
                "actions": [],
                "observations": [],
                "infos": [],
            }

            while not done and step < self.config.max_steps_per_episode:
                # Get action from model
                if hasattr(model, "predict"):
                    action, _ = model.predict(
                        obs, deterministic=self.config.deterministic
                    )
                else:
                    # Fallback for custom models
                    action = model.get_action(obs)

                # Take step
                obs, reward, done, info = env.step(action)

                # Store data
                episode_data["steps"].append(step)
                episode_data["rewards"].append(reward)
                episode_data["actions"].append(action)
                episode_data["observations"].append(obs.copy())
                episode_data["infos"].append(info)

                episode_reward += reward
                step += 1

            episode_data["total_reward"] = episode_reward
            episode_data["total_steps"] = step
            episode_results.append(episode_data)

        return episode_results

    def _calculate_performance_metrics(
        self, episode_results: List[Dict]
    ) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        all_returns = []
        all_rewards = [ep["total_reward"] for ep in episode_results]
        all_steps = [ep["total_steps"] for ep in episode_results]

        # Collect returns from all episodes
        for episode in episode_results:
            all_returns.extend(episode["rewards"])

        returns_array = np.array(all_returns)

        metrics = {
            "episode_metrics": {
                "mean_reward": np.mean(all_rewards),
                "std_reward": np.std(all_rewards),
                "max_reward": np.max(all_rewards),
                "min_reward": np.min(all_rewards),
                "mean_steps": np.mean(all_steps),
                "success_rate": len([r for r in all_rewards if r > 0])
                / len(all_rewards),
            },
            "return_metrics": self.performance_metrics.calculate_returns(returns_array),
        }

        # Add drawdown metrics
        cumulative_returns = np.cumprod(1 + returns_array)
        metrics[
            "drawdown_metrics"
        ] = self.performance_metrics.calculate_drawdown_metrics(cumulative_returns)

        # Add risk-adjusted metrics
        metrics["risk_adjusted_metrics"] = {
            "sharpe_ratio": metrics["return_metrics"]["sharpe_ratio"],
            "sortino_ratio": self.performance_metrics.calculate_sortino_ratio(
                returns_array
            ),
            "calmar_ratio": self.performance_metrics.calculate_calmar_ratio(
                returns_array
            ),
        }

        # Add VaR/CVaR metrics
        metrics["var_cvar_metrics"] = self.performance_metrics.calculate_var_cvar(
            returns_array, self.config.confidence_levels
        )

        return metrics

    def _analyze_risk(self, episode_results: List[Dict]) -> Dict[str, Any]:
        """Analyze risk characteristics."""
        all_returns = []
        positions_data = []

        for episode in episode_results:
            all_returns.extend(episode["rewards"])
            # Extract position data if available
            for info in episode["infos"]:
                if "positions" in info:
                    positions_data.append(info["positions"])

        returns_array = np.array(all_returns)

        risk_analysis = self.risk_analyzer.analyze_portfolio_risk(returns_array)

        # Add episode-level risk metrics
        episode_rewards = [ep["total_reward"] for ep in episode_results]
        risk_analysis["episode_risk"] = {
            "reward_volatility": np.std(episode_rewards),
            "reward_skewness": pd.Series(episode_rewards).skew(),
            "reward_kurtosis": pd.Series(episode_rewards).kurtosis(),
        }

        return risk_analysis

    def _run_benchmarks(self, env) -> Dict[str, Any]:
        """Run benchmark evaluations."""
        return self.benchmark_evaluator.evaluate_benchmarks(env, self.config.n_episodes)

    def _performance_attribution(self, episode_results: List[Dict]) -> Dict[str, Any]:
        """Analyze performance attribution."""
        attribution = {}

        # Action analysis
        all_actions = []
        for episode in episode_results:
            all_actions.extend(episode["actions"])

        action_counts = pd.Series(all_actions).value_counts(normalize=True)
        attribution["action_distribution"] = action_counts.to_dict()

        # Step analysis
        step_rewards = []
        for episode in episode_results:
            for i, reward in enumerate(episode["rewards"]):
                step_rewards.append({"step_in_episode": i, "reward": reward})

        step_df = pd.DataFrame(step_rewards)
        if not step_df.empty:
            step_analysis = step_df.groupby("step_in_episode")["reward"].agg(
                ["mean", "std", "count"]
            )
            attribution["step_analysis"] = step_analysis.to_dict()

        return attribution

    def _create_evaluation_plots(
        self, episode_results: List[Dict], evaluation_results: Dict
    ) -> Dict[str, str]:
        """Create evaluation plots."""
        plots = {}

        if not self.config.create_plots:
            return plots

        # Episode rewards plot
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 3, 1)
        episode_rewards = [ep["total_reward"] for ep in episode_results]
        plt.plot(episode_rewards)
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)

        # Reward distribution
        plt.subplot(2, 3, 2)
        plt.hist(episode_rewards, bins=30, alpha=0.7)
        plt.title("Reward Distribution")
        plt.xlabel("Reward")
        plt.ylabel("Frequency")
        plt.grid(True)

        # Cumulative rewards
        plt.subplot(2, 3, 3)
        cumulative_rewards = np.cumsum(episode_rewards)
        plt.plot(cumulative_rewards)
        plt.title("Cumulative Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Reward")
        plt.grid(True)

        # Action distribution
        plt.subplot(2, 3, 4)
        if (
            "attribution" in evaluation_results
            and "action_distribution" in evaluation_results["attribution"]
        ):
            actions = list(
                evaluation_results["attribution"]["action_distribution"].keys()
            )
            probs = list(
                evaluation_results["attribution"]["action_distribution"].values()
            )
            plt.bar(actions, probs)
            plt.title("Action Distribution")
            plt.xlabel("Action")
            plt.ylabel("Frequency")
            plt.xticks(rotation=45)

        # Rolling performance
        plt.subplot(2, 3, 5)
        window = min(20, len(episode_rewards) // 4)
        if window > 1:
            rolling_mean = pd.Series(episode_rewards).rolling(window).mean()
            plt.plot(rolling_mean)
            plt.title(f"Rolling Mean Reward (window={window})")
            plt.xlabel("Episode")
            plt.ylabel("Mean Reward")
            plt.grid(True)

        # Performance metrics comparison
        plt.subplot(2, 3, 6)
        if "benchmark_results" in evaluation_results:
            metrics = ["mean_return", "sharpe_ratio"]
            model_values = [
                evaluation_results["performance_metrics"]["episode_metrics"][
                    "mean_reward"
                ],
                evaluation_results["performance_metrics"]["risk_adjusted_metrics"][
                    "sharpe_ratio"
                ],
            ]

            x = np.arange(len(metrics))
            width = 0.35

            plt.bar(x - width / 2, model_values, width, label="Model")

            # Add benchmarks
            for i, (bench_name, bench_results) in enumerate(
                evaluation_results["benchmark_results"].items()
            ):
                if i < 2:  # Show only first 2 benchmarks
                    bench_values = [
                        bench_results.get("mean_return", 0),
                        bench_results.get("sharpe_ratio", 0),
                    ]
                    plt.bar(
                        x + (i + 1) * width / len(metrics),
                        bench_values,
                        width,
                        label=bench_name,
                    )

            plt.xlabel("Metrics")
            plt.ylabel("Value")
            plt.title("Performance Comparison")
            plt.xticks(x, metrics)
            plt.legend()
            plt.grid(True, axis="y")

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(
            self.config.output_dir,
            f'evaluation_plots_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        plots["main_plot"] = plot_path
        return plots

    def _get_model_info(self, model) -> Dict[str, Any]:
        """Get model information."""
        info = {"type": type(model).__name__, "parameters": "unknown"}

        # Try to get parameter count
        try:
            if hasattr(model, "policy"):
                # For stable-baselines3 models
                total_params = sum(p.numel() for p in model.policy.parameters())
                trainable_params = sum(
                    p.numel() for p in model.policy.parameters() if p.requires_grad
                )
                info["parameters"] = {
                    "total": total_params,
                    "trainable": trainable_params,
                }
        except Exception:
            pass

        return info

    def _save_evaluation_results(self, results: Dict[str, Any]):
        """Save evaluation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON results
        json_path = os.path.join(
            self.config.output_dir, f"evaluation_results_{timestamp}.json"
        )
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save summary CSV
        summary_path = os.path.join(
            self.config.output_dir, f"evaluation_summary_{timestamp}.csv"
        )
        summary_data = {
            "timestamp": [results["timestamp"]],
            "mean_reward": [
                results["performance_metrics"]["episode_metrics"]["mean_reward"]
            ],
            "std_reward": [
                results["performance_metrics"]["episode_metrics"]["std_reward"]
            ],
            "sharpe_ratio": [
                results["performance_metrics"]["risk_adjusted_metrics"]["sharpe_ratio"]
            ],
            "max_drawdown": [results["drawdown_metrics"].get("max_drawdown", 0)],
            "calmar_ratio": [
                results["performance_metrics"]["risk_adjusted_metrics"]["calmar_ratio"]
            ],
        }

        pd.DataFrame(summary_data).to_csv(summary_path, index=False)

        logger.info(f"Evaluation results saved to {self.config.output_dir}")


def main():
    """Example usage of the model evaluator."""
    # Create evaluation configuration
    config = EvaluationConfig(
        env_type="single_asset",
        symbols=["AAPL", "MSFT"],
        n_episodes=50,
        enable_risk_analysis=True,
        create_plots=True,
    )

    # Create evaluator
    evaluator = ModelEvaluator(config)

    # Mock model and environment for demonstration
    class MockModel:
        def predict(self, observation, deterministic=True):
            return 1, {}  # Always buy action

    class MockEnv:
        def __init__(self):
            self.action_space = type("obj", (object,), {"n": 4})()
            self.step_count = 0

        def reset(self):
            self.step_count = 0
            return np.random.randn(10)

        def step(self, action):
            reward = np.random.normal(0, 1)
            done = self.step_count > 100
            info = {"positions": {"AAPL": 100}}
            self.step_count += 1
            return np.random.randn(10), reward, done, info

    # Run evaluation
    model = MockModel()
    env = MockEnv()

    try:
        results = evaluator.evaluate_model(model, env)
        print("Evaluation completed successfully!")
        print(
            f"Mean reward: {results['performance_metrics']['episode_metrics']['mean_reward']:.2f}"
        )
        print(
            f"Sharpe ratio: {results['performance_metrics']['risk_adjusted_metrics']['sharpe_ratio']:.2f}"
        )

    except Exception as e:
        print(f"Evaluation failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
