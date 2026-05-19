"""
Online RL Trainer for Financial Markets Gym

Real-time training system for reinforcement learning models with live data updates,
continuous learning, and adaptive risk management.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging
import os
import queue
import sys
import threading
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

# Import our gym environments
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import DataManager
from risk import CVaRConfig, RiskManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OnlineTrainingConfig:
    """Configuration for online training."""

    # Environment settings
    env_type: str = "single_asset"
    symbols: List[str] = None
    data_source: str = "yahoo"
    update_frequency: str = "1m"  # 1m, 5m, 15m, 1h, 1d

    # Training parameters
    algorithm: str = "PPO"
    learning_rate: float = 1e-4
    batch_size: int = 32
    buffer_size: int = 10000
    target_update_freq: int = 1000

    # Online learning parameters
    experience_replay: bool = True
    replay_ratio: float = 0.25  # Fraction of training on replay buffer
    adaptation_rate: float = 0.01  # How quickly to adapt to new data
    drift_detection: bool = True

    # Model parameters
    policy_kwargs: Dict = None
    checkpoint_freq: int = 1000
    max_checkpoints: int = 10

    # Risk management
    enable_risk_management: bool = True
    risk_aversion: float = 1.5
    dynamic_risk_adjustment: bool = True
    risk_update_freq: int = 100

    # Monitoring
    log_frequency: int = 50
    evaluation_freq: int = 500
    performance_window: int = 1000

    # Data handling
    data_buffer_size: int = 5000
    max_episode_steps: int = 1000
    episode_timeout: int = 3600  # seconds

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["AAPL", "MSFT", "JPM"]
        if self.policy_kwargs is None:
            self.policy_kwargs = {
                "activation_fn": nn.ReLU,
                "net_arch": [dict(pi=[128, 128], vf=[128, 128])],
            }


class DataStreamer:
    """Live data streaming for online training."""

    def __init__(self, config: OnlineTrainingConfig):
        self.config = config
        self.data_manager = DataManager()
        self.data_queue = queue.Queue(maxsize=1000)
        self.is_running = False
        self.data_buffer = deque(maxlen=config.data_buffer_size)
        self.last_update = None

    def start_streaming(self):
        """Start streaming live data."""
        self.is_running = True
        self.thread = threading.Thread(target=self._stream_data_loop, daemon=True)
        self.thread.start()
        logger.info("Data streaming started")

    def stop_streaming(self):
        """Stop streaming data."""
        self.is_running = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=5)
        logger.info("Data streaming stopped")

    def _stream_data_loop(self):
        """Main data streaming loop."""
        while self.is_running:
            try:
                # Get latest data
                end_time = datetime.now()
                start_time = end_time - timedelta(days=1)  # Get last 24h

                data = self.data_manager.get_data(
                    symbols=self.config.symbols,
                    start_date=start_time.strftime("%Y-%m-%d"),
                    end_date=end_time.strftime("%Y-%m-%d"),
                    source=self.config.data_source,
                    frequency=self.config.update_frequency,
                )

                if not data.empty:
                    # Get the latest data point
                    latest_data = data.iloc[-1:].copy()
                    timestamp = latest_data.index[0]

                    # Add to buffer
                    self.data_buffer.append(
                        {"timestamp": timestamp, "data": latest_data}
                    )

                    # Put in queue
                    try:
                        self.data_queue.put_nowait(
                            {"timestamp": timestamp, "data": latest_data}
                        )
                    except queue.Full:
                        # Remove old data if queue is full
                        try:
                            self.data_queue.get_nowait()
                            self.data_queue.put_nowait(
                                {"timestamp": timestamp, "data": latest_data}
                            )
                        except queue.Empty:
                            pass

                    self.last_update = timestamp
                    logger.debug(f"Data updated: {timestamp}")

                # Sleep based on update frequency
                if self.config.update_frequency == "1m":
                    time.sleep(60)
                elif self.config.update_frequency == "5m":
                    time.sleep(300)
                elif self.config.update_frequency == "15m":
                    time.sleep(900)
                elif self.config.update_frequency == "1h":
                    time.sleep(3600)
                else:
                    time.sleep(60)

            except Exception as e:
                logger.error(f"Error in data streaming: {e}")
                time.sleep(60)  # Wait before retrying

    def get_latest_data(self) -> Optional[Dict[str, Any]]:
        """Get the latest data point."""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None

    def get_buffer_data(self) -> pd.DataFrame:
        """Get buffered data as DataFrame."""
        if not self.data_buffer:
            return pd.DataFrame()

        data_frames = []
        for entry in self.data_buffer:
            data_frames.append(entry["data"])

        if data_frames:
            return pd.concat(data_frames, ignore_index=False)
        else:
            return pd.DataFrame()


class ExperienceBuffer:
    """Experience replay buffer for online training."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)

    def add(self, experience: Tuple, priority: float = 1.0):
        """Add experience to buffer."""
        self.buffer.append(experience)
        self.priorities.append(priority)

    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample experiences from buffer."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)

        # Prioritized sampling
        if len(self.priorities) > 0:
            probs = np.array(self.priorities)
            probs = probs / probs.sum()
            indices = np.random.choice(
                len(self.buffer), size=batch_size, p=probs, replace=False
            )
            return [self.buffer[i] for i in indices]
        else:
            indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
            return [self.buffer[i] for i in indices]

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for experiences."""
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)


class PerformanceMonitor:
    """Monitor online training performance."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.rewards = deque(maxlen=window_size)
        self.actions = deque(maxlen=window_size)
        self.losses = deque(maxlen=window_size)
        self.risk_metrics = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)

        self.performance_history = []
        self.alerts = []

    def log_step(
        self, reward: float, action: int, loss: float = None, risk_metrics: Dict = None
    ):
        """Log a training step."""
        timestamp = datetime.now()
        self.rewards.append(reward)
        self.actions.append(action)
        self.timestamps.append(timestamp)

        if loss is not None:
            self.losses.append(loss)

        if risk_metrics:
            self.risk_metrics.append(risk_metrics)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        if not self.rewards:
            return {}

        recent_rewards = list(self.rewards)
        recent_actions = list(self.actions)

        stats = {
            "timestamp": self.timestamps[-1] if self.timestamps else datetime.now(),
            "steps": len(recent_rewards),
            "mean_reward": np.mean(recent_rewards),
            "std_reward": np.std(recent_rewards),
            "min_reward": np.min(recent_rewards),
            "max_reward": np.max(recent_rewards),
            "reward_trend": self._calculate_trend(recent_rewards),
            "action_distribution": self._get_action_distribution(recent_actions),
        }

        if self.losses:
            recent_losses = list(self.losses)
            stats.update(
                {
                    "mean_loss": np.mean(recent_losses),
                    "loss_trend": self._calculate_trend(recent_losses),
                }
            )

        if self.risk_metrics:
            recent_risk = list(self.risk_metrics)[-1]  # Latest risk metrics
            stats.update(
                {
                    "current_risk_level": recent_risk.get("risk_level", "unknown"),
                    "risk_utilization": recent_risk.get("utilization", {}),
                }
            )

        return stats

    def _calculate_trend(self, values: List[float], window: int = None) -> str:
        """Calculate trend direction."""
        if len(values) < 10:
            return "insufficient_data"

        window = window or min(50, len(values) // 4)
        recent = values[-window:]
        earlier = values[-(window * 2) : -window]

        recent_mean = np.mean(recent)
        earlier_mean = np.mean(earlier)

        diff = recent_mean - earlier_mean
        threshold = np.std(earlier) * 0.1

        if diff > threshold:
            return "improving"
        elif diff < -threshold:
            return "degrading"
        else:
            return "stable"

    def _get_action_distribution(self, actions: List[int]) -> Dict[int, float]:
        """Get action distribution."""
        if not actions:
            return {}

        unique, counts = np.unique(actions, return_counts=True)
        total = len(actions)
        return {int(action): count / total for action, count in zip(unique, counts)}

    def check_performance_alerts(self) -> List[str]:
        """Check for performance alerts."""
        alerts = []

        if len(self.rewards) < 100:
            return alerts

        recent_rewards = list(self.rewards)[-100:]
        mean_reward = np.mean(recent_rewards)

        # Check for poor performance
        if mean_reward < -10:  # Arbitrary threshold
            alerts.append("Poor performance detected")

        # Check for risk violations
        if self.risk_metrics:
            recent_risk = list(self.risk_metrics)[-1]
            if recent_risk.get("risk_level") in ["high", "critical"]:
                alerts.append(f"High risk level: {recent_risk['risk_level']}")

        # Check for action imbalance
        if self.actions:
            action_dist = self._get_action_distribution(list(self.actions)[-100:])
            max_action_prob = max(action_dist.values()) if action_dist else 0
            if max_action_prob > 0.8:
                alerts.append("Action imbalance detected")

        return alerts


class OnlineTrainer:
    """
    Online RL trainer for financial markets.

    Features:
    - Real-time data streaming
    - Continuous learning and adaptation
    - Dynamic risk management
    - Performance monitoring and alerting
    - Experience replay
    - Model checkpointing
    """

    def __init__(self, config: OnlineTrainingConfig = None):
        """
        Initialize online trainer.

        Args:
            config: Online training configuration
        """
        self.config = config or OnlineTrainingConfig()
        self.data_streamer = DataStreamer(self.config)
        self.experience_buffer = ExperienceBuffer(self.config.buffer_size)
        self.performance_monitor = PerformanceMonitor(self.config.performance_window)

        self.model = None
        self.env = None
        self.is_training = False
        self.training_thread = None
        self.step_count = 0
        self.episode_count = 0

        # Model checkpointing
        self.checkpoints = deque(maxlen=self.config.max_checkpoints)
        self.last_checkpoint_time = time.time()

        logger.info(f"OnlineTrainer initialized with {self.config.algorithm}")

    def setup_environment(self) -> Any:
        """Set up the training environment for online learning."""
        try:
            # Import gym environment
            if self.config.env_type == "single_asset":
                from environments import SingleAssetTradingEnv
            elif self.config.env_type == "portfolio":
                from environments import PortfolioOptimizationEnv
            else:
                raise ValueError(
                    f"Unsupported environment type for online training: {self.config.env_type}"
                )

            # Create environment with dynamic data
            env_kwargs = {
                "initial_balance": 100000,
                "max_shares": 1000,
                "transaction_fee": 0.001,
                "data_provider": self.data_streamer,  # Custom data provider
                "online_mode": True,
            }

            # Add risk management
            if self.config.enable_risk_management:
                position_limits = risk.PositionLimits(max_position_size=0.3)
                cvar_config = CVaRConfig(risk_aversion=self.config.risk_aversion)

                env_kwargs["risk_manager"] = RiskManager(
                    position_limits=position_limits, cvar_config=cvar_config
                )

            # Create environment
            if self.config.env_type == "single_asset":
                self.env = SingleAssetTradingEnv(**env_kwargs)
            elif self.config.env_type == "portfolio":
                self.env = PortfolioOptimizationEnv(**env_kwargs)

            logger.info(f"Online environment created: {type(self.env)}")
            return self.env

        except Exception as e:
            logger.error(f"Failed to setup online environment: {e}")
            raise

    def setup_model(self) -> Any:
        """Set up the RL model for online learning."""
        if self.env is None:
            raise ValueError("Environment must be set up first")

        # For online learning, we'll implement a simple PPO variant
        # In practice, you might want to use more sophisticated online algorithms
        self.model = self._create_online_model()

        logger.info(f"Online model created: {self.config.algorithm}")
        return self.model

    def _create_online_model(self) -> Any:
        """Create model suitable for online learning."""

        # This is a placeholder - in practice, you'd implement
        # a proper online RL algorithm or use existing ones
        class OnlinePPO:
            def __init__(self, env, learning_rate=1e-4):
                self.env = env
                self.learning_rate = learning_rate
                self.policy_network = None  # Implement your policy network
                self.value_network = None  # Implement your value network

            def predict(self, observation, deterministic=True):
                # Simple random policy for demonstration
                return self.env.action_space.sample(), {}

            def learn(self, experiences):
                # Online learning step
                pass

        return OnlinePPO(self.env, self.config.learning_rate)

    def start_training(self):
        """Start the online training process."""
        if self.is_training:
            logger.warning("Training is already running")
            return

        # Setup
        if self.env is None:
            self.setup_environment()
        if self.model is None:
            self.setup_model()

        # Start data streaming
        self.data_streamer.start_streaming()

        # Start training loop
        self.is_training = True
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()

        logger.info("Online training started")

    def stop_training(self):
        """Stop the online training process."""
        self.is_training = False

        if self.training_thread:
            self.training_thread.join(timeout=10)

        self.data_streamer.stop_streaming()

        logger.info("Online training stopped")

    def _training_loop(self):
        """Main online training loop."""
        logger.info("Starting online training loop")

        try:
            while self.is_training:
                # Get latest data
                latest_data = self.data_streamer.get_latest_data()
                if latest_data is None:
                    time.sleep(1)
                    continue

                # Run environment step
                observation = self.env.reset()
                episode_reward = 0
                episode_steps = 0
                episode_experiences = []

                for step in range(self.config.max_episode_steps):
                    if not self.is_training:
                        break

                    # Get action from model
                    action, _ = self.model.predict(observation, deterministic=True)

                    # Take environment step
                    next_observation, reward, done, info = self.env.step(action)

                    # Store experience
                    experience = (
                        observation,
                        action,
                        reward,
                        next_observation,
                        done,
                        info,
                    )
                    episode_experiences.append(experience)

                    # Add to replay buffer
                    if self.config.experience_replay:
                        self.experience_buffer.add(experience)

                    episode_reward += reward
                    episode_steps += 1
                    self.step_count += 1

                    # Log metrics
                    loss = None  # Get from model training if available
                    risk_metrics = info.get("risk_metrics", {})
                    self.performance_monitor.log_step(
                        reward, action, loss, risk_metrics
                    )

                    # Update model (online learning step)
                    if (
                        self.config.experience_replay
                        and len(self.experience_buffer) >= self.config.batch_size
                    ):
                        # Sample from replay buffer
                        batch_experiences = self.experience_buffer.sample(
                            self.config.batch_size
                        )
                        self.model.learn(batch_experiences)

                    # Check for performance alerts
                    alerts = self.performance_monitor.check_performance_alerts()
                    for alert in alerts:
                        logger.warning(f"Performance alert: {alert}")

                    # Checkpoint model
                    if self.step_count % self.config.checkpoint_freq == 0:
                        self._checkpoint_model()

                    # Dynamic risk adjustment
                    if (
                        self.config.dynamic_risk_adjustment
                        and self.step_count % self.config.risk_update_freq == 0
                    ):
                        self._adjust_risk_parameters()

                    observation = next_observation

                    if done:
                        break

                # Episode completed
                self.episode_count += 1
                logger.info(
                    f"Episode {self.episode_count}: Reward={episode_reward:.2f}, Steps={episode_steps}"
                )

                # Update model with episode experiences
                if episode_experiences:
                    self.model.learn(episode_experiences)

                # Log performance periodically
                if self.episode_count % self.config.log_frequency == 0:
                    stats = self.performance_monitor.get_performance_stats()
                    logger.info(f"Performance stats: {stats}")

                # Small delay to prevent excessive resource usage
                time.sleep(0.1)

        except Exception as e:
            logger.error(f"Error in training loop: {e}")
            self.is_training = False

    def _checkpoint_model(self):
        """Save model checkpoint."""
        checkpoint = {
            "step": self.step_count,
            "episode": self.episode_count,
            "timestamp": datetime.now().isoformat(),
            "model_state": None,  # Save actual model state
            "performance_stats": self.performance_monitor.get_performance_stats(),
        }

        self.checkpoints.append(checkpoint)
        self.last_checkpoint_time = time.time()

        logger.debug(f"Model checkpoint saved at step {self.step_count}")

    def _adjust_risk_parameters(self):
        """Dynamically adjust risk parameters based on performance."""
        stats = self.performance_monitor.get_performance_stats()

        # Adjust risk aversion based on performance
        if stats.get("reward_trend") == "degrading":
            # Increase risk aversion if performance is degrading
            new_risk_aversion = min(3.0, self.config.risk_aversion * 1.1)
            if hasattr(self.env, "risk_manager"):
                self.env.risk_manager.cvar_shaper.config.risk_aversion = (
                    new_risk_aversion
                )
            logger.info(f"Increased risk aversion to {new_risk_aversion}")

        elif stats.get("reward_trend") == "improving":
            # Decrease risk aversion if performance is improving
            new_risk_aversion = max(0.5, self.config.risk_aversion * 0.95)
            if hasattr(self.env, "risk_manager"):
                self.env.risk_manager.cvar_shaper.config.risk_aversion = (
                    new_risk_aversion
                )
            logger.info(f"Decreased risk aversion to {new_risk_aversion}")

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            "is_training": self.is_training,
            "step_count": self.step_count,
            "episode_count": self.episode_count,
            "experience_buffer_size": len(self.experience_buffer),
            "data_buffer_size": len(self.data_streamer.data_buffer),
            "last_data_update": self.data_streamer.last_update,
            "performance_stats": self.performance_monitor.get_performance_stats(),
            "checkpoints_saved": len(self.checkpoints),
            "last_checkpoint_time": self.last_checkpoint_time,
        }

    def save_model(self, path: str):
        """Save the current model state."""
        model_data = {
            "config": asdict(self.config),
            "step_count": self.step_count,
            "episode_count": self.episode_count,
            "model_state": None,  # Save actual model
            "performance_history": list(self.performance_monitor.performance_history),
            "timestamp": datetime.now().isoformat(),
        }

        with open(path, "w") as f:
            json.dump(model_data, f, indent=2, default=str)

        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load a saved model state."""
        with open(path, "r") as f:
            model_data = json.load(f)

        # Restore model state
        self.step_count = model_data["step_count"]
        self.episode_count = model_data["episode_count"]
        # Load actual model state

        logger.info(f"Model loaded from {path}")


def main():
    """Example usage of the online trainer."""
    # Configuration
    config = OnlineTrainingConfig(
        env_type="single_asset",
        symbols=["AAPL", "MSFT"],
        update_frequency="5m",
        algorithm="PPO",
        enable_risk_management=True,
        buffer_size=5000,
        checkpoint_freq=500,
    )

    # Create trainer
    trainer = OnlineTrainer(config)

    try:
        # Start training
        trainer.start_training()

        # Run for demonstration (in practice, you'd run indefinitely)
        time.sleep(300)  # 5 minutes

        # Get status
        status = trainer.get_training_status()
        print(f"Training status: {status}")

        # Stop training
        trainer.stop_training()

        # Save model
        trainer.save_model("online_model.json")

    except KeyboardInterrupt:
        print("Training interrupted by user")
        trainer.stop_training()
    except Exception as e:
        print(f"Training failed: {e}")
        trainer.stop_training()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
