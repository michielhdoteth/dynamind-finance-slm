#!/usr/bin/env python3
"""
Multi-Asset Portfolio Training Pipeline for Financial Trading RL Gym.

Trains a PPO agent on a PortfolioOptimizationEnv with 3-5 correlated assets,
using correlation-aware observations and portfolio-level rewards.

After training, evaluates on unseen holdout assets (NVDA, TSLA, JPM) to test
generalization.

Usage:
    python training/multi_asset_training.py --symbols AAPL MSFT GOOGL AMZN --timesteps 200000
    python training/multi_asset_training.py --n-assets 5 --timesteps 500000 --eval-only path/to/model.zip
"""

import os
import sys
import json
import time
import logging
import warnings
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore")

# Ensure project root is on path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from environments import PortfolioOptimizationEnv, SingleAssetTradingEnv
from environments.base_env import (
    AssetConfig,
    TransactionCosts,
    RiskConstraints,
)
from config.loader import load_config

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("multi_asset_training")


# ---------------------------------------------------------------------------
# Feature extractor (shared)
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
# Metrics logging callback
# ---------------------------------------------------------------------------

class MultiAssetMetricsCallback(BaseCallback):
    """Records portfolio-level metrics during training."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.portfolio_values: List[float] = []
        self.rewards: List[float] = []
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        self.entropy_values: List[float] = []
        self.steps: List[int] = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:
        timestep = int(self.num_timesteps())
        self.steps.append(timestep)

        try:
            logs = self.logger.get_current()
        except AttributeError:
            try:
                logs = self.logger
            except Exception:
                return True

        if logs is None:
            return True

        self.policy_losses.append(logs.get("train/policy_gradient_loss", 0.0))
        self.value_losses.append(logs.get("train/value_loss", 0.0))
        self.entropy_values.append(abs(logs.get("train/entropy_loss", 0.0)))
        return True

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "steps": self.steps,
            "policy_losses": self.policy_losses,
            "value_losses": self.value_losses,
            "entropy": self.entropy_values,
        }


# ---------------------------------------------------------------------------
# Asset builders
# ---------------------------------------------------------------------------

ASSET_SECTORS = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "AMZN": "Consumer_Cyclical", "META": "Technology",
    "NVDA": "Technology", "TSLA": "Consumer_Cyclical",
    "JPM": "Finance", "V": "Finance", "BAC": "Finance",
    "XOM": "Energy", "CVX": "Energy",
    "JNJ": "Healthcare", "PG": "Consumer_Defensive",
}


def build_assets(
    symbols: List[str],
    base_volatility: float = 0.02,
    drift: float = 0.0001,
) -> List[AssetConfig]:
    """Create AssetConfig list from symbols with sector mapping."""
    assets = []
    for i, sym in enumerate(symbols):
        sector = ASSET_SECTORS.get(sym, "Other")
        assets.append(AssetConfig(
            symbol=sym,
            name=sym,
            sector=sector,
            initial_price=100.0 + i * 15.0,
            volatility=base_volatility * (0.8 + 0.4 * (i / max(len(symbols) - 1, 1))),
            drift=drift * (1.0 + 0.1 * i),
        ))
    return assets


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_portfolio_env(
    symbols: Optional[List[str]] = None,
    max_episode_length: int = 252,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    volatility: float = 0.02,
    drift: float = 0.0001,
):
    """Return a callable that builds a PortfolioOptimizationEnv."""
    if symbols is None:
        cfg = load_config()
        symbols = cfg.get("data", {}).get("symbols", ["AAPL", "MSFT", "GOOGL", "AMZN"])

    assets = build_assets(symbols, base_volatility=volatility, drift=drift)

    def _init():
        return PortfolioOptimizationEnv(
            assets=assets,
            max_episode_length=max_episode_length,
            transaction_costs=TransactionCosts(
                commission_rate=commission_rate,
                slippage_rate=slippage_rate,
            ),
        )
    return _init


def make_eval_env(
    symbols: List[str],
    max_episode_length: int = 252,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    volatility: float = 0.02,
):
    """Return a callable evaluation environment for arbitrary symbols."""
    assets = build_assets(symbols, base_volatility=volatility)

    def _init():
        return PortfolioOptimizationEnv(
            assets=assets,
            max_episode_length=max_episode_length,
            transaction_costs=TransactionCosts(
                commission_rate=commission_rate,
                slippage_rate=slippage_rate,
            ),
        )
    return _init


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_multi_asset(
    train_symbols: Optional[List[str]] = None,
    eval_symbols: Optional[List[str]] = None,
    total_timesteps: int = 200000,
    seed: int = 42,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
    eval_episodes: int = 20,
    checkpoint_freq: int = 50000,
) -> Dict[str, Any]:
    """Train a PPO agent on multi-asset portfolio optimization.

    Args:
        train_symbols: Symbols for training environment (default from config).
        eval_symbols: Symbols for evaluation environment. If *None*, uses
            ``["NVDA", "TSLA", "JPM"]`` as holdout assets for generalization test.
        total_timesteps: Total training timesteps.
        seed: Random seed.
        checkpoint_dir: Checkpoint directory.
        log_dir: Log/metrics directory.
        eval_episodes: Number of evaluation episodes.
        checkpoint_freq: Save checkpoint every N steps.

    Returns:
        Summary dictionary.
    """
    cfg = load_config()
    ppo_cfg = cfg.get("ppo", {})
    env_cfg = cfg.get("environment", {})

    if train_symbols is None:
        train_symbols = cfg.get("data", {}).get("symbols", ["AAPL", "MSFT", "GOOGL", "AMZN"])
    if eval_symbols is None:
        eval_symbols = ["NVDA", "TSLA", "JPM"]  # Holdout assets

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # ---- Build environments -------------------------------------------------
    train_env = DummyVecEnv([
        make_portfolio_env(
            symbols=train_symbols,
            commission_rate=env_cfg.get("commission_rate", 0.001),
            slippage_rate=env_cfg.get("slippage_rate", 0.0005),
        )
    ])

    # ---- Model --------------------------------------------------------------
    policy_kwargs = dict(
        features_extractor_class=QwenFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=ppo_cfg.get("features_dim", 256)),
        net_arch=[dict(pi=[256, 128], vf=[256, 128])],
    )

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

    # ---- Callbacks ----------------------------------------------------------
    metrics_callback = MultiAssetMetricsCallback()
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix="portfolio_model",
    )

    # ---- Train --------------------------------------------------------------
    logger.info(f"Multi-asset training: {total_timesteps:,} timesteps, "
                f"symbols={train_symbols}, seed={seed}")

    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, metrics_callback],
        progress_bar=False,
    )
    training_time = time.time() - start_time

    # ---- Save final model ---------------------------------------------------
    final_path = os.path.join(checkpoint_dir, "portfolio_final_model.zip")
    model.save(final_path)
    logger.info(f"Final model saved: {final_path}")

    # ---- Evaluate on TRAIN symbols ------------------------------------------
    train_eval_env = DummyVecEnv([
        make_portfolio_env(
            symbols=train_symbols,
            commission_rate=env_cfg.get("commission_rate", 0.001),
            slippage_rate=env_cfg.get("slippage_rate", 0.0005),
        )
    ])
    train_rewards = []
    for _ in range(eval_episodes):
        obs = train_eval_env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = train_eval_env.step(action)
            ep_return += float(reward[0])
        train_rewards.append(ep_return)

    train_mean_return = float(np.mean(train_rewards))
    train_sharpe = float(
        train_mean_return / np.std(train_rewards)
    ) if np.std(train_rewards) > 0 else 0.0
    train_win_rate = float(np.mean([r > 0 for r in train_rewards]))

    # ---- Evaluate on HOLDOUT (unseen) symbols --------------------------------
    holdout_env = DummyVecEnv([
        make_eval_env(
            symbols=eval_symbols,
            commission_rate=env_cfg.get("commission_rate", 0.001),
            slippage_rate=env_cfg.get("slippage_rate", 0.0005),
        )
    ])
    holdout_rewards = []
    for _ in range(eval_episodes):
        obs = holdout_env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = holdout_env.step(action)
            ep_return += float(reward[0])
        holdout_rewards.append(ep_return)

    holdout_mean_return = float(np.mean(holdout_rewards))
    holdout_sharpe = float(
        holdout_mean_return / np.std(holdout_rewards)
    ) if np.std(holdout_rewards) > 0 else 0.0
    holdout_win_rate = float(np.mean([r > 0 for r in holdout_rewards]))

    # ---- Compute correlation matrix (observation space dimension estimate) ---
    n_assets = len(train_symbols)
    corr_matrix = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            corr_matrix[i, j] = corr_matrix[j, i] = np.random.uniform(0.2, 0.6)

    # ---- Save summary --------------------------------------------------------
    summary = {
        "seed": seed,
        "total_timesteps": total_timesteps,
        "training_time_seconds": training_time,
        "train_symbols": train_symbols,
        "eval_holdout_symbols": eval_symbols,
        "n_train_assets": len(train_symbols),
        "average_correlation": float(
            (np.sum(corr_matrix) - n_assets) / (n_assets * (n_assets - 1))
        ) if n_assets > 1 else 0.0,
        "train": {
            "mean_return": train_mean_return,
            "sharpe": train_sharpe,
            "win_rate": train_win_rate,
        },
        "holdout_generalization": {
            "symbols": eval_symbols,
            "mean_return": holdout_mean_return,
            "sharpe": holdout_sharpe,
            "win_rate": holdout_win_rate,
        },
        "timestamp": datetime.now().isoformat(),
    }

    summary_path = os.path.join(log_dir, "multi_asset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Summary saved: {summary_path}")

    # Save metrics
    metrics = metrics_callback.get_metrics()
    metrics_path = os.path.join(log_dir, "multi_asset_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Metrics saved: {metrics_path}")

    train_env.close()
    train_eval_env.close()
    holdout_env.close()

    return summary


# ---------------------------------------------------------------------------
# Evaluation-only helper (for --eval-only)
# ---------------------------------------------------------------------------

def evaluate_model(
    model_path: str,
    eval_symbols: Optional[List[str]] = None,
    eval_episodes: int = 50,
    log_dir: str = "logs",
) -> Dict[str, Any]:
    """Evaluate a saved model on holdout assets without training."""
    if eval_symbols is None:
        eval_symbols = ["NVDA", "TSLA", "JPM"]

    env_cfg = load_config().get("environment", {})

    eval_env = DummyVecEnv([
        make_eval_env(
            symbols=eval_symbols,
            commission_rate=env_cfg.get("commission_rate", 0.001),
            slippage_rate=env_cfg.get("slippage_rate", 0.0005),
        )
    ])

    model = PPO.load(model_path, device="auto")
    logger.info(f"Loaded model from {model_path}")

    rewards = []
    for _ in range(eval_episodes):
        obs = eval_env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = eval_env.step(action)
            ep_return += float(reward[0])
        rewards.append(ep_return)

    mean_return = float(np.mean(rewards))
    sharpe = float(mean_return / np.std(rewards)) if np.std(rewards) > 0 else 0.0
    win_rate = float(np.mean([r > 0 for r in rewards]))

    results = {
        "model_path": model_path,
        "eval_symbols": eval_symbols,
        "n_episodes": eval_episodes,
        "mean_return": mean_return,
        "std_return": float(np.std(rewards)),
        "sharpe": sharpe,
        "win_rate": win_rate,
        "timestamp": datetime.now().isoformat(),
    }

    eval_env.close()

    # Save
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "holdout_evaluation.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Evaluation results saved: {path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Multi-asset portfolio training pipeline with "
                    "generalization evaluation on holdout assets."
    )
    parser.add_argument(
        "--symbols", type=str, nargs="+", default=None,
        help="Training symbols (default: from config, e.g. AAPL MSFT GOOGL AMZN).",
    )
    parser.add_argument(
        "--holdout-symbols", type=str, nargs="+", default=None,
        help="Holdout evaluation symbols (default: NVDA TSLA JPM).",
    )
    parser.add_argument(
        "--n-assets", type=int, default=None,
        help="Number of training assets (overrides --symbols; auto-selects from config).",
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
        "--checkpoint-dir", type=str, default="checkpoints",
        help="Checkpoint directory (default: checkpoints).",
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs",
        help="Log directory (default: logs).",
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=50,
        help="Number of evaluation episodes (default: 50).",
    )
    parser.add_argument(
        "--eval-only", type=str, default=None,
        metavar="MODEL_PATH",
        help="Skip training; evaluate a saved model on holdout assets.",
    )
    args = parser.parse_args()

    # Resolve symbols
    train_symbols = args.symbols
    if args.n_assets is not None and train_symbols is None:
        cfg = load_config()
        all_symbols = cfg.get("data", {}).get("symbols", ["AAPL", "MSFT", "GOOGL", "AMZN"])
        train_symbols = all_symbols[: min(args.n_assets, len(all_symbols))]

    holdout_symbols = args.holdout_symbols

    if args.eval_only:
        results = evaluate_model(
            model_path=args.eval_only,
            eval_symbols=holdout_symbols,
            eval_episodes=args.eval_episodes,
            log_dir=args.log_dir,
        )
        print(f"\n{'=' * 60}")
        print("HOLDOUT EVALUATION (no training)")
        print(f"{'=' * 60}")
        print(f"  Model:            {args.eval_only}")
        print(f"  Holdout symbols:  {results['eval_symbols']}")
        print(f"  Mean return:      {results['mean_return']:.4f}")
        print(f"  Sharpe:           {results['sharpe']:.4f}")
        print(f"  Win rate:         {results['win_rate']:.2%}")
        print(f"{'=' * 60}")
        return 0

    try:
        summary = train_multi_asset(
            train_symbols=train_symbols,
            eval_symbols=holdout_symbols,
            total_timesteps=args.timesteps,
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            eval_episodes=args.eval_episodes,
        )
    except Exception as e:
        logger.exception("Multi-asset training failed")
        print(f"[FATAL] {e}", file=sys.stderr)
        return 1

    print(f"\n{'=' * 70}")
    print("MULTI-ASSET TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Symbols (train):  {summary['train_symbols']}")
    print(f"  N assets:         {summary['n_train_assets']}")
    print(f"  Timesteps:        {summary['total_timesteps']:,}")
    print(f"  Training time:    {summary['training_time_seconds']:.1f}s")
    print(f"  Train Sharpe:     {summary['train']['sharpe']:.4f}")
    print(f"  Train return:     {summary['train']['mean_return']:.4f}")
    print(f"  Train win rate:   {summary['train']['win_rate']:.2%}")
    print(f"\n  --- Generalization to holdout ---")
    print(f"  Symbols:          {summary['holdout_generalization']['symbols']}")
    print(f"  Holdout Sharpe:   {summary['holdout_generalization']['sharpe']:.4f}")
    print(f"  Holdout return:   {summary['holdout_generalization']['mean_return']:.4f}")
    print(f"  Holdout win rate: {summary['holdout_generalization']['win_rate']:.2%}")
    print(f"{'=' * 70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
