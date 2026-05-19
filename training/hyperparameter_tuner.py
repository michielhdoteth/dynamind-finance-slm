#!/usr/bin/env python3
"""
Hyperparameter Optimization for Financial Trading RL Gym.

Uses Optuna for Bayesian hyperparameter search (50 trials) with:
  - Pruning by early Sharpe ratio
  - Results logged to JSON
  - Top 3 configurations saved as YAML

Falls back to grid search if Optuna is not available.

Usage:
    python training/hyperparameter_tuner.py --n-trials 50
    python training/hyperparameter_tuner.py --method optuna --n-trials 30 --prune
    python training/hyperparameter_tuner.py --method grid
"""

import os
import sys
import json
import time
import itertools
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

from environments import SingleAssetTradingEnv
from environments.base_env import AssetConfig
from config.loader import load_config

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hyperparameter_tuner")

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
# Search space definition
# ---------------------------------------------------------------------------

SEARCH_SPACE: Dict[str, Dict[str, Any]] = {
    "learning_rate": {
        "type": "loguniform",
        "low": 1e-5,
        "high": 1e-3,
        "default": 3e-4,
    },
    "ent_coef": {
        "type": "loguniform",
        "low": 0.001,
        "high": 0.5,
        "default": 0.2,
    },
    "clip_range": {
        "type": "uniform",
        "low": 0.1,
        "high": 0.4,
        "default": 0.2,
    },
    "n_steps": {
        "type": "int_categorical",
        "values": [512, 1024, 2048, 4096],
        "default": 2048,
    },
    "batch_size": {
        "type": "int_categorical",
        "values": [32, 64, 128, 256],
        "default": 64,
    },
    "gamma": {
        "type": "uniform",
        "low": 0.9,
        "high": 0.999,
        "default": 0.99,
    },
    "kl_target": {
        "type": "uniform",
        "low": 0.005,
        "high": 0.05,
        "default": 0.015,
    },
}


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env():
    """Return a callable that builds a SingleAssetTradingEnv."""
    cfg = load_config()
    env_cfg = cfg.get("environment", {})
    asset = AssetConfig(
        symbol="AAPL",
        name="Apple Inc.",
        sector="Technology",
        initial_price=150.0,
        volatility=env_cfg.get("volatility", 0.02),
    )

    def _init():
        return SingleAssetTradingEnv(
            asset=asset,
            max_steps=env_cfg.get("max_steps", 252),
            commission_rate=env_cfg.get("commission_rate", 0.001),
            slippage_rate=env_cfg.get("slippage_rate", 0.0005),
        )
    return _init


# ---------------------------------------------------------------------------
# Objective function (train + evaluate)
# ---------------------------------------------------------------------------

def train_and_evaluate(
    hyperparams: Dict[str, Any],
    total_timesteps: int = 50000,
    seed: int = 42,
    eval_episodes: int = 10,
) -> Dict[str, float]:
    """Train a PPO model with given hyperparams and return performance metrics."""
    ppo_cfg = load_config().get("ppo", {})

    env = DummyVecEnv([make_env()])

    policy_kwargs = dict(
        features_extractor_class=QwenFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=ppo_cfg.get("features_dim", 256)),
        net_arch=[dict(pi=[256, 128], vf=[256, 128])],
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=hyperparams.get("learning_rate", 3e-4),
        n_steps=int(hyperparams.get("n_steps", 2048)),
        batch_size=int(hyperparams.get("batch_size", 64)),
        n_epochs=ppo_cfg.get("n_epochs", 10),
        gamma=hyperparams.get("gamma", 0.99),
        gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        clip_range=hyperparams.get("clip_range", 0.2),
        ent_coef=hyperparams.get("ent_coef", 0.2),
        vf_coef=ppo_cfg.get("vf_coef", 0.5),
        max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=seed,
        device="auto",
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    # Evaluate
    eval_env = DummyVecEnv([make_env()])
    returns = []
    for _ in range(eval_episodes):
        obs = eval_env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = eval_env.step(action)
            ep_return += float(reward[0])
        returns.append(ep_return)

    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns)) if len(returns) > 1 else 1.0
    sharpe = mean_return / std_return if std_return > 0 else 0.0
    win_rate = float(np.mean([r > 0 for r in returns]))

    env.close()
    eval_env.close()

    return {
        "mean_return": mean_return,
        "std_return": std_return,
        "sharpe": sharpe,
        "win_rate": win_rate,
    }


# ---------------------------------------------------------------------------
# Optuna search (primary)
# ---------------------------------------------------------------------------

def _suggest_params(trial, space: Dict[str, Any]) -> Dict[str, Any]:
    """Sample hyperparameters from the search space using Optuna."""
    params = {}
    for name, spec in space.items():
        if spec["type"] == "loguniform":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
        elif spec["type"] == "uniform":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"])
        elif spec["type"] == "int_categorical":
            params[name] = trial.suggest_categorical(name, spec["values"])
        else:
            params[name] = spec["default"]
    return params


def run_optuna_search(
    n_trials: int = 50,
    total_timesteps: int = 50000,
    prune: bool = True,
    seed: int = 42,
    n_startup_trials: int = 10,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run Optuna hyperparameter search.

    Returns:
        (all_trials_sorted, best_params)
    """
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed, n_startup_trials=n_startup_trials),
        pruner=MedianPruner(n_startup_trials=n_startup_trials) if prune else None,
        study_name="ppo_trading_tuning",
    )

    def objective(trial):
        params = _suggest_params(trial, SEARCH_SPACE)

        # --- Pruning: use intermediate reward at 50% progress ---
        if prune:
            # Quick 50% run for pruning check
            half_steps = total_timesteps // 2
            metrics = train_and_evaluate(params, total_timesteps=half_steps, seed=seed)
            trial.report(metrics["sharpe"], step=1)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # --- Full evaluation ---
        metrics = train_and_evaluate(params, total_timesteps=total_timesteps, seed=seed)
        return metrics["sharpe"]

    logger.info(f"Starting Optuna search: {n_trials} trials, timesteps={total_timesteps}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Collect results
    all_trials = []
    for t in study.trials:
        all_trials.append({
            "number": t.number,
            "params": t.params,
            "value": t.value,
            "state": str(t.state),
            "datetime_start": str(t.datetime_start),
            "datetime_complete": str(t.datetime_complete),
        })

    all_trials.sort(key=lambda x: x["value"] or -1e9, reverse=True)
    best_params = study.best_params
    best_params["_best_value"] = study.best_value

    logger.info(f"Optuna search complete. Best Sharpe: {study.best_value:.4f}")
    return all_trials, best_params


# ---------------------------------------------------------------------------
# Grid search fallback
# ---------------------------------------------------------------------------

def _grid_from_space(space: Dict[str, Any], max_combinations: int = 64) -> List[Dict[str, Any]]:
    """Build a discrete grid from the search space."""
    grid_values = {}
    for name, spec in space.items():
        if spec["type"] in ("loguniform", "uniform"):
            # Sample 3 points: low, mid, high
            if spec["type"] == "loguniform":
                grid_values[name] = [
                    spec["low"],
                    np.sqrt(spec["low"] * spec["high"]),
                    spec["high"],
                ]
            else:
                grid_values[name] = [spec["low"], (spec["low"] + spec["high"]) / 2, spec["high"]]
        elif spec["type"] == "int_categorical":
            grid_values[name] = spec["values"][:4]  # At most 4

    keys = list(grid_values.keys())
    value_lists = [grid_values[k] for k in keys]
    combinations = list(itertools.product(*value_lists))

    # Limit total combinations
    if len(combinations) > max_combinations:
        step = len(combinations) // max_combinations
        combinations = combinations[::step]

    return [dict(zip(keys, combo)) for combo in combinations]


def run_grid_search(
    total_timesteps: int = 50000,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run grid search over the search space."""
    grid_configs = _grid_from_space(SEARCH_SPACE)
    logger.info(f"Grid search: {len(grid_configs)} configurations")

    all_trials = []
    for i, params in enumerate(grid_configs):
        logger.info(f"Grid point {i + 1}/{len(grid_configs)}: {params}")
        try:
            metrics = train_and_evaluate(params, total_timesteps=total_timesteps, seed=seed)
        except Exception as e:
            logger.warning(f"Grid point {i} failed: {e}")
            continue

        all_trials.append({
            "number": i,
            "params": params,
            "value": metrics["sharpe"],
            "state": "COMPLETE",
            "mean_return": metrics["mean_return"],
            "win_rate": metrics["win_rate"],
        })

    all_trials.sort(key=lambda x: x["value"] or -1e9, reverse=True)
    best_params = dict(all_trials[0]["params"]) if all_trials else {}
    best_params["_best_value"] = all_trials[0]["value"] if all_trials else 0.0

    return all_trials, best_params


# ---------------------------------------------------------------------------
# Results output
# ---------------------------------------------------------------------------

def save_top_configs(
    all_trials: List[Dict[str, Any]],
    n_top: int = 3,
    output_dir: str = "results",
) -> List[Dict[str, Any]]:
    """Save top N hyperparameter configs as YAML (and JSON).

    Args:
        all_trials: Sorted trials (best first).
        n_top: Number of top configs to save.
        output_dir: Output directory.

    Returns:
        The top N configurations.
    """
    os.makedirs(output_dir, exist_ok=True)
    top = all_trials[:n_top]

    # Save as JSON (always)
    json_path = os.path.join(output_dir, "top_hp_configs.json")
    with open(json_path, "w") as f:
        json.dump(top, f, indent=2, default=str)
    logger.info(f"Top configs saved: {json_path}")

    # Save as YAML if PyYAML available
    try:
        import yaml
        for rank, trial in enumerate(top, 1):
            config = dict(trial["params"])
            config.pop("_best_value", None)
            config["hyperparameter_tuning_rank"] = rank
            config["hyperparameter_tuning_value"] = trial["value"]

            yaml_path = os.path.join(output_dir, f"top_hp_config_{rank}.yaml")
            with open(yaml_path, "w") as f:
                yaml.dump(
                    {"ppo": config},
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                )
            logger.info(f"Top config #{rank} saved: {yaml_path}")
    except ImportError:
        logger.info("PyYAML not available; skipping YAML output.")

    return top


def save_all_trials(all_trials: List[Dict[str, Any]], output_dir: str = "results") -> str:
    """Save all trial results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "hp_search_all_trials.json")
    with open(path, "w") as f:
        json.dump(all_trials, f, indent=2, default=str)
    logger.info(f"All trials saved: {path}")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for PPO trading agent. "
                    "Uses Optuna (Bayesian) with grid search fallback."
    )
    parser.add_argument(
        "--method", type=str, default="optuna", choices=["optuna", "grid"],
        help="Search method: optuna (Bayesian) or grid (default: optuna).",
    )
    parser.add_argument(
        "--n-trials", type=int, default=50,
        help="Number of Optuna trials (default: 50). Ignored for grid search.",
    )
    parser.add_argument(
        "--timesteps", type=int, default=50000,
        help="Training timesteps per trial (default: 50000).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed (default: 42).",
    )
    parser.add_argument(
        "--no-prune", action="store_false", dest="prune",
        help="Disable pruning for Optuna (default: enabled).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory for results (default: results).",
    )
    parser.add_argument(
        "--n-top", type=int, default=3,
        help="Number of top configs to save (default: 3).",
    )
    args = parser.parse_args()

    start = time.time()

    if args.method == "optuna":
        try:
            all_trials, best_params = run_optuna_search(
                n_trials=args.n_trials,
                total_timesteps=args.timesteps,
                prune=args.prune,
                seed=args.seed,
            )
        except ImportError:
            logger.warning("Optuna not installed. Falling back to grid search.")
            all_trials, best_params = run_grid_search(
                total_timesteps=args.timesteps,
                seed=args.seed,
            )
    else:
        all_trials, best_params = run_grid_search(
            total_timesteps=args.timesteps,
            seed=args.seed,
        )

    elapsed = time.time() - start

    # Save results
    save_all_trials(all_trials, args.output_dir)
    top = save_top_configs(all_trials, args.n_top, args.output_dir)

    # Print summary
    print(f"\n{'=' * 60}")
    print("HYPERPARAMETER TUNING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Method:        {args.method}")
    print(f"  Trials/configs: {len(all_trials)}")
    print(f"  Time:          {elapsed:.1f}s")
    print(f"\n  Best Sharpe:   {best_params.get('_best_value', 'N/A'):.4f}")
    print(f"  Best params:")
    for k, v in best_params.items():
        if k != "_best_value":
            print(f"    {k}: {v}")
    print(f"\n  Top {args.n_top} configs saved to {args.output_dir}/")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
