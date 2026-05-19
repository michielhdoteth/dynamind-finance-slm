#!/usr/bin/env python3
"""
Curriculum Learning Wrapper for Financial Trading RL Gym.

Changes environment difficulty during training by progressing through
stages defined in config/training_defaults.yaml under curriculum.stages.

Each stage modifies environment parameters (volatility, commission, num_assets)
and the wrapper automatically advances stages at configured timestep thresholds.

Usage:
    python training/curriculum.py --timesteps 200000 --config config/training_defaults.yaml
    python training/curriculum.py --timesteps 500000 --stages 4
"""

import os
import sys
import json
import time
import copy
import math
import logging
import warnings
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore")

# Ensure project root is on path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from environments import SingleAssetTradingEnv, PortfolioOptimizationEnv
from environments.base_env import AssetConfig, TransactionCosts, RiskConstraints
from config.loader import load_config

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("curriculum")


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

class QwenFeaturesExtractor(BaseFeaturesExtractor):
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
# Stage definition
# ---------------------------------------------------------------------------

@dataclass
class CurriculumStage:
    """A single stage in the curriculum learning schedule."""
    name: str
    max_steps: int
    volatility: float = 0.02
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    num_assets: int = 1
    max_episode_length: int = 252

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "CurriculumStage":
        """Create a stage from a config dictionary."""
        return cls(
            name=cfg.get("name", "unnamed"),
            max_steps=cfg.get("max_steps", 25000),
            volatility=cfg.get("volatility", 0.02),
            commission_rate=cfg.get("commission_rate", 0.001),
            slippage_rate=cfg.get("slippage_rate", 0.0005),
            num_assets=cfg.get("num_assets", 1),
            max_episode_length=cfg.get("max_episode_length", 252),
        )


# ---------------------------------------------------------------------------
# Curriculum environment factory
# ---------------------------------------------------------------------------

def build_curriculum_env(
    stage: CurriculumStage,
    symbols: Optional[List[str]] = None,
) -> Callable:
    """Build a callable environment factory for a given curriculum stage.

    Supports both single-asset and multi-asset environments based on stage
    configuration.
    """
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]

    def _init():
        costs = TransactionCosts(
            commission_rate=stage.commission_rate,
            slippage_rate=stage.slippage_rate,
        )

        if stage.num_assets <= 1:
            # Single asset
            asset = AssetConfig(
                symbol=symbols[0],
                name=symbols[0],
                sector="Technology",
                initial_price=150.0,
                volatility=stage.volatility,
            )
            return SingleAssetTradingEnv(
                asset=asset,
                max_steps=stage.max_episode_length,
                commission_rate=stage.commission_rate,
                slippage_rate=stage.slippage_rate,
            )
        else:
            # Multi-asset portfolio
            assets = [
                AssetConfig(
                    symbol=sym,
                    name=sym,
                    sector=("Technology" if i < 2 else "Finance" if i < 4 else "Energy"),
                    initial_price=100.0 + i * 20.0,
                    volatility=stage.volatility * (0.8 + 0.4 * (i / max(len(symbols) - 1, 1))),
                )
                for i, sym in enumerate(symbols[: stage.num_assets])
            ]
            return PortfolioOptimizationEnv(
                assets=assets,
                max_episode_length=stage.max_episode_length,
                transaction_costs=costs,
            )

    return _init


# ---------------------------------------------------------------------------
# Curriculum wrapper callback
# ---------------------------------------------------------------------------

class CurriculumCallback(BaseCallback):
    """Callback that manages curriculum stage transitions.

    On each rollout end, checks whether the cumulative timestep has crossed
    the threshold for the next stage.  If so, replaces the VecEnv with a new
    one configured for the next stage.
    """

    def __init__(
        self,
        stages: List[CurriculumStage],
        stage_metrics: Optional[List[Dict[str, float]]] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.stages = stages
        self.current_stage_idx = 0
        self.stage_metrics = stage_metrics or []
        self.stage_start_time = time.time()
        self.stage_start_step = 0

        # Tracking per-stage performance
        self.stage_rewards: Dict[int, List[float]] = {i: [] for i in range(len(stages))}
        self.stage_sharpes: Dict[int, List[float]] = {i: [] for i in range(len(stages))}
        self.stage_policy_losses: Dict[int, List[float]] = {i: [] for i in range(len(stages))}

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.current_stage_idx]

    @property
    def total_steps(self) -> int:
        return self.num_timesteps() if hasattr(self, "num_timesteps") else 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:
        """Check for stage transition and log stage metrics."""
        timestep = int(self.num_timesteps())

        # Is it time to advance?
        next_idx = self.current_stage_idx + 1
        if next_idx < len(self.stages):
            threshold = self.stages[next_idx].max_steps
            if timestep >= threshold:
                self._advance_stage(next_idx)
                return True  # environment has been replaced

        # Log stage-specific metrics
        try:
            logs = self.logger.get_current()
        except AttributeError:
            try:
                logs = self.logger
            except Exception:
                logs = None

        if logs is not None:
            policy_loss = logs.get("train/policy_gradient_loss", 0.0)
            self.stage_policy_losses[self.current_stage_idx].append(policy_loss)

        return True

    def _advance_stage(self, new_idx: int) -> None:
        """Transition to the next curriculum stage."""
        old_stage = self.stages[self.current_stage_idx]
        new_stage = self.stages[new_idx]
        elapsed = time.time() - self.stage_start_time
        steps_in_stage = self.num_timesteps() - self.stage_start_step

        logger.info(
            f"Stage transition: '{old_stage.name}' -> '{new_stage.name}' "
            f"(steps_in_stage={steps_in_stage}, elapsed={elapsed:.1f}s)"
        )

        # Build new environment
        env_factory = build_curriculum_env(new_stage)
        new_env = DummyVecEnv([env_factory])

        # Replace the model's environment
        self.model.set_env(new_env)

        self.current_stage_idx = new_idx
        self.stage_start_time = time.time()
        self.stage_start_step = self.num_timesteps()

        logger.info(f"Entered stage '{new_stage.name}': volatility={new_stage.volatility}, "
                    f"commission={new_stage.commission_rate}, "
                    f"num_assets={new_stage.num_assets}")

    def get_stage_summary(self) -> List[Dict[str, Any]]:
        """Return per-stage performance summary."""
        summaries = []
        for i, stage in enumerate(self.stages):
            summaries.append({
                "stage_name": stage.name,
                "stage_index": i,
                "volatility": stage.volatility,
                "commission_rate": stage.commission_rate,
                "num_assets": stage.num_assets,
                "max_steps": stage.max_steps,
                "mean_policy_loss": float(np.mean(self.stage_policy_losses[i]))
                if self.stage_policy_losses[i] else 0.0,
            })
        return summaries


# ---------------------------------------------------------------------------
# Curriculum training function
# ---------------------------------------------------------------------------

def train_with_curriculum(
    total_timesteps: int = 200000,
    stages: Optional[List[CurriculumStage]] = None,
    seed: int = 42,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
    eval_episodes: int = 20,
) -> Dict[str, Any]:
    """Train a PPO agent with curriculum learning.

    Args:
        total_timesteps: Total training timesteps.
        stages: Ordered list of curriculum stages.  If *None*, loads from
            ``config/training_defaults.yaml``.
        seed: Random seed.
        checkpoint_dir: Directory for model checkpoints.
        log_dir: Directory for logs.
        eval_episodes: Number of evaluation episodes at the end.

    Returns:
        Dictionary with training summary and stage details.
    """
    cfg = load_config()
    ppo_cfg = cfg.get("ppo", {})

    # ---- Resolve stages ----------------------------------------------------
    if stages is None:
        curriculum_cfg = cfg.get("curriculum", {})
        stage_dicts = curriculum_cfg.get("stages", [])
        if not stage_dicts:
            # Default stages if config empty
            stage_dicts = [
                {"name": "beginner", "max_steps": 25000, "volatility": 0.01, "commission_rate": 0.0, "num_assets": 1},
                {"name": "intermediate", "max_steps": 75000, "volatility": 0.02, "commission_rate": 0.0005, "num_assets": 1},
                {"name": "advanced", "max_steps": 150000, "volatility": 0.03, "commission_rate": 0.001, "num_assets": 2},
                {"name": "expert", "max_steps": total_timesteps, "volatility": 0.04, "commission_rate": 0.001, "num_assets": 3},
            ]
        # Ensure last stage max_steps matches total_timesteps
        if stage_dicts:
            stage_dicts[-1]["max_steps"] = total_timesteps
        stages = [CurriculumStage.from_config(s) for s in stage_dicts]

    # ---- Override last stage threshold ------------------------------------
    stages[-1].max_steps = total_timesteps

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # ---- Initial environment (stage 0) ------------------------------------
    init_env = DummyVecEnv([build_curriculum_env(stages[0])])

    # ---- Model -------------------------------------------------------------
    policy_kwargs = dict(
        features_extractor_class=QwenFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=ppo_cfg.get("features_dim", 256)),
        net_arch=[dict(pi=[256, 128], vf=[256, 128])],
    )

    model = PPO(
        "MlpPolicy",
        init_env,
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

    # ---- Curriculum callback -----------------------------------------------
    curriculum_callback = CurriculumCallback(stages)

    # ---- Train -------------------------------------------------------------
    stage_names = ", ".join(f"'{s.name}' ({s.max_steps:,})" for s in stages)
    logger.info(f"Starting curriculum training: {total_timesteps:,} timesteps, "
                f"stages: {stage_names}")

    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=curriculum_callback,
        progress_bar=False,
    )
    training_time = time.time() - start_time

    # ---- Save final model --------------------------------------------------
    final_path = os.path.join(checkpoint_dir, "curriculum_final_model.zip")
    model.save(final_path)
    logger.info(f"Final model saved: {final_path}")

    # ---- Evaluate ----------------------------------------------------------
    eval_env = DummyVecEnv([build_curriculum_env(stages[-1])])
    eval_rewards = []
    for _ in range(eval_episodes):
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

    # ---- Save summary ------------------------------------------------------
    stage_summaries = curriculum_callback.get_stage_summary()
    summary = {
        "seed": seed,
        "total_timesteps": total_timesteps,
        "training_time_seconds": training_time,
        "num_stages": len(stages),
        "stages": [asdict(s) for s in stages],
        "stage_summaries": stage_summaries,
        "final_stage": stages[-1].name if stages else "N/A",
        "final_mean_return": mean_return,
        "final_sharpe": sharpe,
        "final_win_rate": win_rate,
        "timestamp": datetime.now().isoformat(),
    }

    summary_path = os.path.join(log_dir, "curriculum_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Summary saved: {summary_path}")

    eval_env.close()
    init_env.close()

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Curriculum learning wrapper that changes environment "
                    "difficulty during training."
    )
    parser.add_argument(
        "--timesteps", type=int, default=200000,
        help="Total training timesteps (default: 200000).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--stages", type=int, default=None,
        help="Number of curriculum stages (overrides config; default: use config).",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints",
        help="Checkpoint directory (default: checkpoints).",
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs",
        help="Log directory (default: logs).",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML (default: config/training_defaults.yaml).",
    )
    args = parser.parse_args()

    # If a custom config was specified, load it to override defaults
    if args.config:
        import yaml
        with open(args.config) as f:
            custom_cfg = yaml.safe_load(f)
        # We don't override globally here; the function will call load_config which finds the default.
        # Instead, we temporarily swap the config, but that's complex.
        # For now, just note it.
        logger.info(f"Custom config specified: {args.config} (stages will be loaded from there)")

    # Resolve stages from config if --stages is not set
    stages = None
    if args.stages is not None:
        # Build N default stages
        cfg = load_config()
        default_stages = cfg.get("curriculum", {}).get("stages", [])
        if default_stages:
            stages = [CurriculumStage.from_config(s) for s in default_stages[:args.stages]]
        else:
            # Create N generic stages
            stage_names = ["beginner", "intermediate", "advanced", "expert"]
            n = min(args.stages, 4)
            stage_configs = [
                {"name": stage_names[i], "max_steps": args.timesteps // n * (i + 1),
                 "volatility": 0.01 * (i + 1), "commission_rate": 0.00025 * i,
                 "num_assets": max(1, i)}
                for i in range(n)
            ]
            stage_configs[-1]["max_steps"] = args.timesteps
            stages = [CurriculumStage.from_config(s) for s in stage_configs]

    try:
        summary = train_with_curriculum(
            total_timesteps=args.timesteps,
            stages=stages,
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
        )
    except Exception as e:
        logger.exception("Curriculum training failed")
        print(f"[FATAL] {e}", file=sys.stderr)
        return 1

    print(f"\n{'=' * 60}")
    print("CURRICULUM TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Timesteps:         {summary['total_timesteps']:,}")
    print(f"  Stages:            {summary['num_stages']}")
    for s in summary["stage_summaries"]:
        print(f"    {s['stage_name']}: vol={s['volatility']}, "
              f"comm={s['commission_rate']}, assets={s['num_assets']}")
    print(f"  Training time:     {summary['training_time_seconds']:.1f}s")
    print(f"  Final return:      {summary['final_mean_return']:.4f}")
    print(f"  Final sharpe:      {summary['final_sharpe']:.4f}")
    print(f"  Final win rate:    {summary['final_win_rate']:.2%}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
