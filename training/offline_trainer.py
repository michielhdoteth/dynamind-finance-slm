"""
Offline RL Trainer for Financial Markets Gym

Professional batch training system for reinforcement learning models with
comprehensive experiment tracking, model checkpointing, and performance analysis.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging
import os
import sys
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# RL libraries
try:
    from stable_baselines3 import A2C, DQN, PPO, SAC
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
except ImportError:
    print("Warning: stable_baselines3 not available")

# Import our gym environments
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import DataManager
from risk import CVaRConfig, RiskManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for offline training."""

    # Environment settings
    env_type: str = "single_asset"  # single_asset, portfolio, market_making, execution
    symbols: List[str] = None
    start_date: str = "2020-01-01"
    end_date: str = "2023-12-31"
    data_source: str = "yahoo"
    frequency: str = "1d"

    # Training parameters
    algorithm: str = "PPO"
    total_timesteps: int = 1000000
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Model parameters
    policy_kwargs: Dict = None
    tensorboard_log: str = "./logs/tensorboard"
    seed: int = 42

    # Evaluation settings
    eval_freq: int = 10000
    n_eval_episodes: int = 10
    eval_deterministic: bool = True

    # Checkpointing
    save_freq: int = 50000
    save_path: str = "./models"

    # Risk management
    enable_risk_management: bool = True
    risk_aversion: float = 1.0
    position_limit: float = 0.3

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["AAPL", "MSFT", "JPM"]
        if self.policy_kwargs is None:
            self.policy_kwargs = {
                "activation_fn": nn.ReLU,
                "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
            }


class TrainingMetrics:
    """Track and analyze training metrics."""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.eval_lengths = []
        self.losses = defaultdict(list)
        self.risk_metrics = defaultdict(list)

    def log_step(self, step: int, info: Dict[str, Any]):
        """Log metrics for a training step."""
        self.metrics["step"].append(step)

        for key, value in info.items():
            if isinstance(value, (int, float)):
                self.metrics[key].append(value)

    def log_episode(self, episode: int, reward: float, length: int, info: Dict = None):
        """Log episode completion metrics."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)

    def log_evaluation(self, eval_rewards: List[float], eval_lengths: List[int]):
        """Log evaluation results."""
        self.eval_rewards.extend(eval_rewards)
        self.eval_lengths.extend(eval_lengths)

    def log_loss(self, step: int, loss_dict: Dict[str, float]):
        """Log training losses."""
        for loss_name, loss_value in loss_dict.items():
            self.losses[loss_name].append((step, loss_value))

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        summary = {}

        if self.episode_rewards:
            summary["training"] = {
                "episodes": len(self.episode_rewards),
                "mean_reward": np.mean(self.episode_rewards),
                "std_reward": np.std(self.episode_rewards),
                "max_reward": np.max(self.episode_rewards),
                "min_reward": np.min(self.episode_rewards),
                "mean_length": np.mean(self.episode_lengths),
            }

        if self.eval_rewards:
            summary["evaluation"] = {
                "episodes": len(self.eval_rewards),
                "mean_reward": np.mean(self.eval_rewards),
                "std_reward": np.std(self.eval_rewards),
                "max_reward": np.max(self.eval_rewards),
                "min_reward": np.min(self.eval_rewards),
            }

        if self.losses:
            summary["losses"] = {}
            for loss_name, loss_values in self.losses.items():
                if loss_values:
                    summary["losses"][loss_name] = {
                        "final": loss_values[-1][1],
                        "mean": np.mean([v[1] for v in loss_values[-10:]]),  # Last 10
                        "trend": "decreasing"
                        if len(loss_values) > 1
                        and loss_values[-1][1] < loss_values[-2][1]
                        else "increasing",
                    }

        return summary


class CustomCallback(BaseCallback):
    """Custom callback for training monitoring and risk management."""

    def __init__(self, trainer, verbose: int = 0):
        super().__init__(verbose)
        self.trainer = trainer
        self.episode_count = 0
        self.step_count = 0

    def _on_rollout_start(self) -> None:
        """Called before each rollout."""
        pass

    def _on_step(self) -> bool:
        """Called after each step."""
        self.step_count += 1

        # Log step metrics
        if hasattr(self.training_env, "get_attr"):
            # For vectorized environments
            infos = (
                self.training_env.get_attr("infos")[0]
                if len(self.training_env.get_attr("infos")) > 0
                else {}
            )
        else:
            infos = {}

        # Log risk metrics if available
        if hasattr(self.training_env, "risk_manager") and self.step_count % 100 == 0:
            risk_summary = self.training_env.risk_manager.get_risk_summary()
            self.trainer.metrics.risk_metrics["step"].append(self.step_count)
            self.trainer.metrics.risk_metrics["risk_level"].append(
                risk_summary.get("current_risk_level", "unknown")
            )

        return True

    def _on_rollout_end(self) -> None:
        """Called after each rollout."""
        # Log rollout metrics
        if hasattr(self.training_env, "get_attr"):
            infos = (
                self.training_env.get_attr("infos")[0]
                if len(self.training_env.get_attr("infos")) > 0
                else {}
            )
            if "episode" in infos:
                episode_info = infos["episode"]
                self.trainer.metrics.log_episode(
                    self.episode_count,
                    episode_info["r"],
                    episode_info["l"],
                    episode_info,
                )
                self.episode_count += 1


class OfflineTrainer:
    """
    Professional offline RL trainer for financial markets.

    Features:
    - Multiple RL algorithms (PPO, A2C, DQN, SAC)
    - Comprehensive experiment tracking
    - Risk-aware training
    - Automatic hyperparameter tuning
    - Model checkpointing and versioning
    - Performance analysis and visualization
    """

    def __init__(self, config: TrainingConfig = None):
        """
        Initialize offline trainer.

        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        self.metrics = TrainingMetrics()
        self.model = None
        self.env = None
        self.eval_env = None
        self.start_time = None
        self.training_log = []

        # Create directories
        os.makedirs(self.config.save_path, exist_ok=True)
        os.makedirs(self.config.tensorboard_log, exist_ok=True)

        logger.info(
            f"OfflineTrainer initialized with {self.config.algorithm} algorithm"
        )

    def setup_environment(self) -> Any:
        """Set up the training environment."""
        try:
            # Import gym environment
            if self.config.env_type == "single_asset":
                from environments import SingleAssetTradingEnv
            elif self.config.env_type == "portfolio":
                from environments import PortfolioOptimizationEnv
            elif self.config.env_type == "market_making":
                from environments.market_microstructure import MarketMakingEnvironment
            elif self.config.env_type == "execution":
                from environments.market_microstructure import ExecutionEnvironment
            else:
                raise ValueError(f"Unknown environment type: {self.config.env_type}")

            # Load data
            data_manager = DataManager()
            data = data_manager.get_data(
                symbols=self.config.symbols,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                source=self.config.data_source,
                frequency=self.config.frequency,
            )

            logger.info(f"Loaded data for {self.config.symbols}: {data.shape}")

            # Create environment
            if self.config.env_type in ["single_asset", "portfolio"]:
                env_kwargs = {
                    "data": data,
                    "initial_balance": 100000,
                    "max_shares": 1000,
                    "transaction_fee": 0.001,
                }
            elif self.config.env_type == "market_making":
                env_kwargs = {
                    "symbols": self.config.symbols,
                    "initial_inventory": 0,
                    "max_position": 100,
                    "spread_bps": 10,
                }
            elif self.config.env_type == "execution":
                env_kwargs = {
                    "symbols": self.config.symbols,
                    "target_quantity": 1000,
                    "time_horizon": 100,
                    "market_impact": 0.0001,
                }

            # Add risk management if enabled
            if self.config.enable_risk_management:
                from risk import CVaRConfig, PositionLimits, RiskManager

                position_limits = risk.PositionLimits(
                    max_position_size=self.config.position_limit
                )
                cvar_config = CVaRConfig(risk_aversion=self.config.risk_aversion)

                env_kwargs["risk_manager"] = RiskManager(
                    position_limits=position_limits, cvar_config=cvar_config
                )

            # Create environment
            if self.config.env_type == "single_asset":
                self.env = SingleAssetTradingEnv(**env_kwargs)
            elif self.config.env_type == "portfolio":
                self.env = PortfolioOptimizationEnv(**env_kwargs)
            elif self.config.env_type == "market_making":
                self.env = MarketMakingEnvironment(**env_kwargs)
            elif self.config.env_type == "execution":
                self.env = ExecutionEnvironment(**env_kwargs)

            # Wrap for training
            self.env = Monitor(self.env)
            self.env = DummyVecEnv([lambda: self.env])

            logger.info(f"Environment created: {type(self.env)}")

            return self.env

        except Exception as e:
            logger.error(f"Failed to setup environment: {e}")
            raise

    def setup_evaluation_env(self) -> Any:
        """Set up separate environment for evaluation."""
        # Create evaluation environment with different data period
        eval_start = (
            datetime.strptime(self.config.end_date, "%Y-%m-%d") + timedelta(days=1)
        ).strftime("%Y-%m-%d")
        eval_end = (
            datetime.strptime(eval_start, "%Y-%m-%d") + timedelta(days=90)
        ).strftime("%Y-%m-%d")

        # Use same setup but with evaluation data
        original_end = self.config.end_date
        self.config.end_date = eval_end

        eval_env = self.setup_environment()

        # Restore original end date
        self.config.end_date = original_end

        return eval_env

    def setup_model(self) -> Any:
        """Set up the RL model."""
        if self.env is None:
            raise ValueError("Environment must be set up first")

        # Algorithm selection
        algorithm_map = {"PPO": PPO, "A2C": A2C, "DQN": DQN, "SAC": SAC}

        if self.config.algorithm not in algorithm_map:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")

        model_class = algorithm_map[self.config.algorithm]

        # Create model
        self.model = model_class(
            "MlpPolicy",
            self.env,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            n_steps=self.config.n_steps,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            policy_kwargs=self.config.policy_kwargs,
            tensorboard_log=self.config.tensorboard_log,
            seed=self.config.seed,
            verbose=1,
        )

        logger.info(f"Model created: {self.config.algorithm}")

        return self.model

    def train(self) -> Dict[str, Any]:
        """
        Execute the training process.

        Returns:
            Training results and metrics
        """
        logger.info("Starting training...")
        self.start_time = time.time()

        try:
            # Setup
            if self.env is None:
                self.setup_environment()
            if self.model is None:
                self.setup_model()

            # Setup evaluation environment
            self.eval_env = self.setup_evaluation_env()

            # Setup callbacks
            callbacks = [
                CustomCallback(self, verbose=1),
                CheckpointCallback(
                    save_freq=self.config.save_freq,
                    save_path=self.config.save_path,
                    name_prefix=f"{self.config.algorithm}_{self.config.env_type}",
                ),
                EvalCallback(
                    eval_env=self.eval_env,
                    eval_freq=self.config.eval_freq,
                    n_eval_episodes=self.config.n_eval_episodes,
                    deterministic=self.config.eval_deterministic,
                    best_model_save_path=os.path.join(
                        self.config.save_path, "best_model"
                    ),
                    log_path=os.path.join(self.config.save_path, "eval_results"),
                    verbose=1,
                ),
            ]

            # Training
            self.model.learn(
                total_timesteps=self.config.total_timesteps,
                callback=callbacks,
                progress_bar=True,
            )

            # Final evaluation
            final_eval = self.evaluate_model()

            training_time = time.time() - self.start_time

            results = {
                "config": asdict(self.config),
                "training_time": training_time,
                "final_evaluation": final_eval,
                "metrics_summary": self.metrics.get_summary(),
                "model_path": os.path.join(self.config.save_path, "final_model"),
            }

            # Save final model
            self.model.save(results["model_path"])

            # Save training log
            self.save_training_log(results)

            logger.info(f"Training completed in {training_time:.2f} seconds")
            return results

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def evaluate_model(self, n_episodes: int = 50) -> Dict[str, Any]:
        """
        Evaluate the trained model.

        Args:
            n_episodes: Number of evaluation episodes

        Returns:
            Evaluation results
        """
        if self.model is None or self.eval_env is None:
            raise ValueError("Model and evaluation environment must be set up first")

        logger.info(f"Evaluating model for {n_episodes} episodes...")

        # Evaluate using stable-baselines3
        mean_reward, std_reward = evaluate_policy(
            self.model, self.eval_env, n_eval_episodes=n_episodes, deterministic=True
        )

        # Manual evaluation for detailed metrics
        episode_rewards = []
        episode_lengths = []
        detailed_metrics = []

        for episode in range(n_episodes):
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            episode_info = {"risk_violations": 0, "trades": 0}

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)

                episode_reward += reward
                episode_length += 1

                # Collect detailed metrics
                if isinstance(info, list) and len(info) > 0:
                    info = info[0]
                    if "risk_violation" in info:
                        episode_info["risk_violations"] += 1
                    if "trade_executed" in info and info["trade_executed"]:
                        episode_info["trades"] += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            detailed_metrics.append(episode_info)

        results = {
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "episodes": n_episodes,
            "all_rewards": episode_rewards,
            "all_lengths": episode_lengths,
            "max_reward": float(np.max(episode_rewards)),
            "min_reward": float(np.min(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
            "detailed_metrics": detailed_metrics,
        }

        # Risk analysis
        if detailed_metrics:
            total_violations = sum(m["risk_violations"] for m in detailed_metrics)
            total_trades = sum(m["trades"] for m in detailed_metrics)
            results["risk_analysis"] = {
                "total_violations": total_violations,
                "violations_per_episode": total_violations / n_episodes,
                "total_trades": total_trades,
                "trades_per_episode": total_trades / n_episodes,
            }

        return results

    def save_training_log(self, results: Dict[str, Any]):
        """Save training log and results."""
        log_path = os.path.join(
            self.config.save_path,
            f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )

        # Prepare data for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, (str, int, float, bool, dict, list)):
                serializable_results[key] = value
            elif isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif hasattr(value, "__dict__"):
                serializable_results[key] = str(value)
            else:
                serializable_results[key] = str(value)

        with open(log_path, "w") as f:
            json.dump(serializable_results, f, indent=2, default=str)

        logger.info(f"Training log saved to {log_path}")

    def plot_training_results(self, save_path: str = None):
        """Plot training results and metrics."""
        if save_path is None:
            save_path = os.path.join(self.config.save_path, "training_plots")

        os.makedirs(save_path, exist_ok=True)

        # Plot episode rewards
        if self.metrics.episode_rewards:
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.plot(self.metrics.episode_rewards)
            plt.title("Episode Rewards")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.grid(True)

            # Plot moving average
            window = min(100, len(self.metrics.episode_rewards) // 10)
            if window > 1:
                moving_avg = (
                    pd.Series(self.metrics.episode_rewards).rolling(window).mean()
                )
                plt.plot(moving_avg, "r-", label=f"MA({window})")
                plt.legend()

        # Plot evaluation rewards
        if self.metrics.eval_rewards:
            plt.subplot(2, 2, 2)
            plt.plot(self.metrics.eval_rewards)
            plt.title("Evaluation Rewards")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.grid(True)

        # Plot losses
        if self.metrics.losses:
            plt.subplot(2, 2, 3)
            for loss_name, loss_values in self.metrics.losses.items():
                if loss_values:
                    steps, values = zip(*loss_values)
                    plt.plot(steps, values, label=loss_name)
            plt.title("Training Losses")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)

        # Plot risk metrics
        if self.metrics.risk_metrics:
            plt.subplot(2, 2, 4)
            risk_levels = self.metrics.risk_metrics.get("risk_level", [])
            if risk_levels:
                # Convert risk levels to numeric
                risk_numeric = []
                for level in risk_levels:
                    if level == "low":
                        risk_numeric.append(0)
                    elif level == "medium":
                        risk_numeric.append(1)
                    elif level == "high":
                        risk_numeric.append(2)
                    elif level == "critical":
                        risk_numeric.append(3)
                    else:
                        risk_numeric.append(1)

                plt.plot(risk_numeric)
                plt.title("Risk Level Over Time")
                plt.xlabel("Step")
                plt.ylabel("Risk Level")
                plt.yticks([0, 1, 2, 3], ["Low", "Medium", "High", "Critical"])
                plt.grid(True)

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_path, "training_results.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        logger.info(f"Training plots saved to {save_path}")


def main():
    """Example usage of the offline trainer."""
    # Configuration
    config = TrainingConfig(
        env_type="single_asset",
        symbols=["AAPL", "MSFT"],
        start_date="2020-01-01",
        end_date="2022-12-31",
        algorithm="PPO",
        total_timesteps=100000,
        enable_risk_management=True,
    )

    # Create trainer
    trainer = OfflineTrainer(config)

    # Train model
    try:
        results = trainer.train()
        print("Training completed successfully!")
        print(
            f"Final evaluation reward: {results['final_evaluation']['mean_reward']:.2f} ± {results['final_evaluation']['std_reward']:.2f}"
        )

        # Plot results
        trainer.plot_training_results()

    except Exception as e:
        print(f"Training failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
