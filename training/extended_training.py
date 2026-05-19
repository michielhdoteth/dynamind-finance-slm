#!/usr/bin/env python3
"""
Extended Training Pipeline for Financial Trading RL Gym.

Supports 500k+ timestep training with:
  - Automated checkpointing every 50k steps
  - Cosine annealing learning rate schedule
  - Page-Hinkley changepoint detection for performance monitoring
  - Real-time PPO metrics logging
  - Final model + intermediate checkpoint persistence

Usage:
    python training/extended_training.py --timesteps 500000 --seed 42
    python training/extended_training.py --timesteps 1000000 --seed 7 --resume checkpoints/step_200000.zip
"""

import os
import sys
import json
import time
import math
import logging
import warnings
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import torch
from torch import nn

warnings.filterwarnings("ignore")

# Ensure project root is on path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from environments import SingleAssetTradingEnv
from environments.base_env import AssetConfig
from config.loader import load_config

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("extended_training")

# ---------------------------------------------------------------------------
# Feature extractor (shared)
# ---------------------------------------------------------------------------

class QwenFeaturesExtractor(BaseFeaturesExtractor):
    """Qwen-style feature extractor for extended training."""

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        flat_dim = int(np.prod(observation_space.shape))
        self.net = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        return self.net(observations)


# ---------------------------------------------------------------------------
# Page-Hinkley changepoint detector (lightweight inline version)
# ---------------------------------------------------------------------------

class PageHinkleyDetector:
    """Detects changes in the mean of a streaming signal.

    Based on the Page-Hinkley test.  Used to identify potential
    meta-learning emergence thresholds during training.
    """

    def __init__(self, threshold: float = 10.0, delta: float = 0.01):
        self.threshold = threshold
        self.delta = delta
        self.cumulative_sum = 0.0
        self.mean_history: List[float] = []
        self.change_points: List[int] = []

    def update(self, value: float) -> bool:
        """Feed a new observation. Returns True if a changepoint is detected."""
        if not self.mean_history:
            self.mean_history.append(value)
            return False

        running_mean = self.mean_history[-1]
        new_mean = running_mean + self.delta * (value - running_mean)
        self.mean_history.append(new_mean)
        self.cumulative_sum += value - new_mean - self.delta

        if abs(self.cumulative_sum) > self.threshold:
            self.change_points.append(len(self.mean_history) - 1)
            self.cumulative_sum = 0.0
            return True
        return False


# ---------------------------------------------------------------------------
# Cosine annealing schedule helper
# ---------------------------------------------------------------------------

def cosine_annealing(step: int, total_steps: int, lr_min: float, lr_max: float) -> float:
    """Compute cosine-annealed learning rate at *step*."""
    progress = step / max(total_steps, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(
    max_steps: int = 252,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    volatility: float = 0.02,
    drift: float = 0.0001,
):
    """Return a callable that builds a SingleAssetTradingEnv."""
    asset = AssetConfig(
        symbol="AAPL",
        name="Apple Inc.",
        sector="Technology",
        initial_price=150.0,
        volatility=volatility,
        drift=drift,
    )

    def _init():
        return SingleAssetTradingEnv(
            asset=asset,
            max_steps=max_steps,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
        )
    return _init


# ---------------------------------------------------------------------------
# Custom logging callback
# ---------------------------------------------------------------------------

class ExtendedTrainingCallback(BaseCallback):
    """Callback that records PPO metrics, detects changepoints, and applies LR scheduling."""

    def __init__(
        self,
        total_timesteps: int,
        lr_initial: float = 3e-4,
        lr_final: float = 1e-5,
        checkpoint_dir: str = "checkpoints",
        checkpoint_freq: int = 50000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.lr_initial = lr_initial
        self.lr_final = lr_final
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        self.last_checkpoint_step = 0

        # Metrics history
        self.timestamps: List[float] = []
        self.steps: List[int] = []
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        self.entropy_values: List[float] = []
        self.approx_kls: List[float] = []
        self.clip_fractions: List[float] = []
        self.explained_variances: List[float] = []
        self.rewards: List[float] = []

        # Changepoint detector for explained_variance
        self.cp_detector = PageHinkleyDetector(threshold=10.0, delta=0.01)

        os.makedirs(checkpoint_dir, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:
        """Log metrics after each rollout."""

        try:
            logs = self.logger.get_current()
        except AttributeError:
            try:
                logs = self.logger
            except Exception:
                return True

        if logs is None:
            return True

        timestep = int(logs.get("time/total_timesteps", 0))
        self.timestamps.append(time.time())
        self.steps.append(timestep)

        # Extract PPO components
        policy_loss = logs.get("train/policy_gradient_loss", 0.0)
        value_loss = logs.get("train/value_loss", 0.0)
        entropy_loss = logs.get("train/entropy_loss", 0.0)
        approx_kl = logs.get("train/approx_kl", 0.0)
        clip_fraction = logs.get("train/clip_fraction", 0.0)
        explained_variance = logs.get("train/explained_variance", 0.0)

        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropy_values.append(abs(entropy_loss))
        self.approx_kls.append(approx_kl)
        self.clip_fractions.append(clip_fraction)
        self.explained_variances.append(explained_variance)

        # Changepoint detection
        if self.cp_detector.update(explained_variance):
            logger.info(
                f"Changepoint DETECTED at step {timestep} "
                f"(explained_variance={explained_variance:.4f})"
            )

        # Cosine annealing LR
        new_lr = cosine_annealing(
            timestep, self.total_timesteps, self.lr_final, self.lr_initial
        )
        for group in self.model.policy.optimizer.param_groups:
            group["lr"] = new_lr

        # Checkpoint if needed
        if timestep - self.last_checkpoint_step >= self.checkpoint_freq:
            self._save_checkpoint(timestep)

        return True

    def _save_checkpoint(self, timestep: int) -> None:
        """Save model checkpoint."""
        path = os.path.join(self.checkpoint_dir, f"model_step_{timestep}.zip")
        self.model.save(path)
        self.last_checkpoint_step = timestep
        logger.info(f"Checkpoint saved: {path} (step {timestep})")

    def get_metrics(self) -> Dict[str, Any]:
        """Return collected metrics as a dictionary."""
        return {
            "steps": self.steps,
            "policy_losses": self.policy_losses,
            "value_losses": self.value_losses,
            "entropy": self.entropy_values,
            "approx_kl": self.approx_kls,
            "clip_fractions": self.clip_fractions,
            "explained_variances": self.explained_variances,
            "changepoints": self.cp_detector.change_points,
        }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_extended(
    total_timesteps: int = 500000,
    seed: int = 42,
    resume_path: Optional[str] = None,
    checkpoint_dir: str = "checkpoints",
    checkpoint_freq: int = 50000,
    log_dir: str = "logs",
    n_eval_episodes: int = 20,
) -> Dict[str, Any]:
    """Run extended training with changepoint detection and cosine annealing.

    Args:
        total_timesteps: Total training timesteps (500k+).
        seed: Random seed.
        resume_path: Path to a saved model .zip to resume from.
        checkpoint_dir: Directory for intermediate checkpoints.
        checkpoint_freq: Save checkpoint every N steps.
        log_dir: Directory for logs/metrics.
        n_eval_episodes: Number of episodes for final evaluation.

    Returns:
        Dictionary with training summary and metrics.
    """
    cfg = load_config()
    ppo_cfg = cfg.get("ppo", {})
    env_cfg = cfg.get("environment", {})

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # -- Environments ---------------------------------------------------------
    train_env = DummyVecEnv([make_env()])
    eval_env = DummyVecEnv([make_env(max_steps=env_cfg.get("max_steps", 252))])

    # -- Model ----------------------------------------------------------------
    policy_kwargs = dict(
        features_extractor_class=QwenFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=ppo_cfg.get("features_dim", 256)),
        net_arch=[dict(pi=[256, 128], vf=[256, 128])],
    )

    if resume_path:
        logger.info(f"Resuming from {resume_path}")
        model = PPO.load(resume_path, env=train_env, device="auto")
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=ppo_cfg.get("learning_rate", 3e-4),
            n_steps=ppo_cfg.get("n_steps", 2048),
            batch_size=ppo_cfg.get("batch_size", 64),
            n_epochs=ppo_cfg.get("n_epochs", 10),
            gamma=ppo_cfg.get("gamma", 0.99),
            gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
            clip_range=ppo_cfg.get("clip_range", 0.2),
            ent_coef=ppo_cfg.get("ent_coef", 0.2),
            vf_coef=ppo_cfg.get("vf_coef", 0.5),
            max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
            device="auto",
        )

    # -- Callback -------------------------------------------------------------
    callback = ExtendedTrainingCallback(
        total_timesteps=total_timesteps,
        lr_initial=ppo_cfg.get("learning_rate", 3e-4),
        lr_final=1e-5,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=checkpoint_freq,
    )

    # -- Train ----------------------------------------------------------------
    logger.info(f"Starting extended training: {total_timesteps:,} timesteps, seed={seed}")
    start_time = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=False,
    )

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.1f}s")

    # -- Final checkpoint -----------------------------------------------------
    final_path = os.path.join(checkpoint_dir, "final_model.zip")
    model.save(final_path)
    logger.info(f"Final model saved: {final_path}")

    # -- Evaluation -----------------------------------------------------------
    eval_rewards = []
    for _ in range(n_eval_episodes):
        obs = eval_env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = eval_env.step(action)
            ep_return += float(reward[0])
        eval_rewards.append(ep_return)

    mean_return = float(np.mean(eval_rewards))
    sharpe = float(mean_return / np.std(eval_rewards)) if np.std(eval_rewards) > 0 else 0.0
    win_rate = float(np.mean([r > 0 for r in eval_rewards]))

    # -- Save metrics ---------------------------------------------------------
    metrics = callback.get_metrics()
    summary = {
        "seed": seed,
        "total_timesteps": total_timesteps,
        "training_time_seconds": training_time,
        "final_mean_return": mean_return,
        "final_sharpe": sharpe,
        "final_win_rate": win_rate,
        "changepoints": metrics["changepoints"],
        "checkpoints_saved": range(
            checkpoint_freq,
            total_timesteps + 1,
            checkpoint_freq,
        ),
        "timestamp": datetime.now().isoformat(),
    }

    summary_path = os.path.join(log_dir, "extended_training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved: {summary_path}")

    metrics_path = os.path.join(log_dir, "extended_training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved: {metrics_path}")

    train_env.close()
    eval_env.close()

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Extended training pipeline with 500k+ timesteps, "
                    "changepoint detection, and cosine annealing LR."
    )
    parser.add_argument(
        "--timesteps", type=int, default=500000,
        help="Total training timesteps (default: 500000).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to saved model .zip to resume training from.",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints",
        help="Checkpoint directory (default: checkpoints).",
    )
    parser.add_argument(
        "--checkpoint-freq", type=int, default=50000,
        help="Save checkpoint every N steps (default: 50000).",
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs",
        help="Log directory (default: logs).",
    )
    args = parser.parse_args()

    try:
        summary = train_extended(
            total_timesteps=args.timesteps,
            seed=args.seed,
            resume_path=args.resume,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_freq=args.checkpoint_freq,
            log_dir=args.log_dir,
        )
    except Exception as e:
        logger.exception("Extended training failed")
        print(f"[FATAL] {e}", file=sys.stderr)
        return 1

    print(f"\n{'=' * 60}")
    print("EXTENDED TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Timesteps:      {summary['total_timesteps']:,}")
    print(f"  Training time:  {summary['training_time_seconds']:.1f}s")
    print(f"  Mean return:    {summary['final_mean_return']:.4f}")
    print(f"  Sharpe ratio:   {summary['final_sharpe']:.4f}")
    print(f"  Win rate:       {summary['final_win_rate']:.2%}")
    print(f"  Changepoints:   {summary['changepoints']}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
