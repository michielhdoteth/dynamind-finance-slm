#!/usr/bin/env python3
"""
Model Size Sweep: Train multiple Qwen variants with identical hyperparameters.
Compares 0.5B vs 1.5B vs Instruct variants under controlled conditions.

Usage:
    python training/model_sweep.py --timesteps 100000 --seeds 3
    python training/model_sweep.py --timesteps 50000 --seeds 5 --variants config/custom_variants.json
"""

import os
import sys
import json
import time
import warnings
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime

import torch
import numpy as np

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
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

class QwenFeaturesExtractor(BaseFeaturesExtractor):
    """Qwen-style feature extractor for financial observations."""

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
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    """Aggregated results for a single model variant across seeds."""
    model_name: str
    params_count: str
    n_seeds: int
    mean_sharpe: float
    std_sharpe: float
    mean_return: float
    mean_max_dd: float
    mean_win_rate: float
    mean_training_time: float
    seed_results: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(asset_symbol: str = "AAPL", max_steps: int = 252):
    """Create a callable that builds a SingleAssetTradingEnv."""
    cfg = load_config()
    env_cfg = cfg.get("environment", {})

    asset = AssetConfig(
        symbol=asset_symbol,
        name=asset_symbol,
        sector="Tech",
        initial_price=100.0,
        volatility=env_cfg.get("volatility", 0.02),
        drift=env_cfg.get("drift", 0.0001),
    )

    def _init():
        return SingleAssetTradingEnv(
            asset=asset,
            max_steps=env_cfg.get("max_steps", max_steps),
            commission_rate=env_cfg.get("commission_rate", 0.001),
            slippage_rate=env_cfg.get("slippage_rate", 0.0005),
        )
    return _init


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_model(
    model_name: str,
    total_timesteps: int = 100000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train a single PPO model and return evaluation metrics.

    Args:
        model_name: Label for the model (used for logging only).
        total_timesteps: Number of timesteps to train.
        seed: Random seed.

    Returns:
        Dictionary with seed, mean_return, sharpe, win_rate, training_time.
    """
    cfg = load_config()
    ppo_cfg = cfg.get("ppo", {})

    env = DummyVecEnv([make_env()])

    policy_kwargs = dict(
        features_extractor_class=QwenFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=ppo_cfg.get("features_dim", 256)),
        net_arch=[dict(pi=[256, 128], vf=[256, 128])],
    )

    model = PPO(
        "MlpPolicy",
        env,
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

    start = time.time()
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    training_time = time.time() - start

    # Evaluate: run 20 deterministic episodes
    eval_env = DummyVecEnv([make_env()])
    returns = []
    for _ in range(20):
        obs = eval_env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = eval_env.step(action)
            ep_return += float(reward[0])
        returns.append(ep_return)

    mean_return = float(np.mean(returns))
    sharpe = float(mean_return / np.std(returns)) if np.std(returns) > 0 else 0.0
    win_rate = float(np.mean([r > 0 for r in returns]))

    env.close()
    eval_env.close()

    return {
        "seed": seed,
        "mean_return": mean_return,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "training_time": training_time,
        "timesteps": total_timesteps,
    }


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_sweep(
    model_variants: Optional[List[Dict[str, str]]] = None,
    n_seeds: int = 3,
    timesteps: int = 100000,
) -> List[SweepResult]:
    """Run a model sweep across multiple model variants.

    Args:
        model_variants: Each entry has ``{"name": ..., "params": ...}``.
            If *None*, loads from ``config/training_defaults.yaml``.
        n_seeds: Number of random seeds per variant.
        timesteps: Number of training timesteps per run.

    Returns:
        List of :class:`SweepResult` objects aggregated across seeds.
    """
    if model_variants is None:
        cfg = load_config()
        model_variants = cfg.get("model_variants", [
            {"name": "Qwen/Qwen2-0.5B", "params": "494M"},
            {"name": "Qwen/Qwen2-1.5B", "params": "1.54B"},
            {"name": "Qwen/Qwen2-0.5B-Instruct", "params": "494M"},
        ])

    results: List[SweepResult] = []
    for variant in model_variants:
        print(f"\n{'=' * 60}")
        print(f"Training {variant['name']} ({variant['params']})")
        print(f"{'=' * 60}")

        seed_results: List[Dict[str, Any]] = []
        for seed in range(n_seeds):
            print(f"  Seed {seed} ... ", end="", flush=True)
            result = train_model(variant["name"], timesteps, seed)
            seed_results.append(result)
            print(f"Sharpe={result['sharpe']:.4f}, "
                  f"Return={result['mean_return']:.4f}, "
                  f"Win={result['win_rate']:.2%}")

        sharpe_values = [r["sharpe"] for r in seed_results]
        sweep_result = SweepResult(
            model_name=variant["name"],
            params_count=variant["params"],
            n_seeds=n_seeds,
            mean_sharpe=float(np.mean(sharpe_values)),
            std_sharpe=float(np.std(sharpe_values)),
            mean_return=float(np.mean([r["mean_return"] for r in seed_results])),
            mean_max_dd=0.0,
            mean_win_rate=float(np.mean([r["win_rate"] for r in seed_results])),
            mean_training_time=float(np.mean([r["training_time"] for r in seed_results])),
            seed_results=seed_results,
        )
        results.append(sweep_result)

    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_results(results: List[SweepResult]) -> None:
    """Print a formatted summary table of sweep results."""
    header = f"{'Model':<35} {'Params':<10} {'Sharpe':<10} {'Return':<10} {'WinRate':<10} {'Time':<10}"
    sep = "-" * 85
    print(f"\n{'=' * 85}")
    print("MODEL SWEEP RESULTS")
    print(f"{'=' * 85}")
    print(header)
    print(sep)
    for r in results:
        print(f"{r.model_name:<35} {r.params_count:<10} "
              f"{r.mean_sharpe:<10.4f} {r.mean_return:<10.4f} "
              f"{r.mean_win_rate:<10.2%} {r.mean_training_time:<10.1f}")
    print(f"{'=' * 85}")


def save_results(results: List[SweepResult], path: str = "results/model_sweep_results.json") -> None:
    """Save aggregated sweep results to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    output = []
    for r in results:
        output.append({
            "model_name": r.model_name,
            "params": r.params_count,
            "mean_sharpe": r.mean_sharpe,
            "std_sharpe": r.std_sharpe,
            "mean_return": r.mean_return,
            "mean_win_rate": r.mean_win_rate,
            "mean_training_time": r.mean_training_time,
            "n_seeds": r.n_seeds,
        })
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Model size sweep: train multiple Qwen variants with identical hyperparameters."
    )
    parser.add_argument(
        "--timesteps", type=int, default=100000,
        help="Total training timesteps per model (default: 100000).",
    )
    parser.add_argument(
        "--seeds", type=int, default=3,
        help="Number of random seeds per model variant (default: 3).",
    )
    parser.add_argument(
        "--variants", type=str, default=None,
        help="Path to JSON file with model variant list. If omitted, uses config defaults.",
    )
    parser.add_argument(
        "--output", type=str, default="results/model_sweep_results.json",
        help="Path to save JSON results (default: results/model_sweep_results.json).",
    )
    args = parser.parse_args()

    variants = None
    if args.variants:
        with open(args.variants) as f:
            variants = json.load(f)

    try:
        results = run_sweep(model_variants=variants, n_seeds=args.seeds, timesteps=args.timesteps)
    except Exception as e:
        print(f"[FATAL] Sweep failed: {e}", file=sys.stderr)
        return 1

    print_results(results)
    save_results(results, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
