"""
Kronos-Enhanced PPO Training for Financial Trading RL Gym.

Combines Kronos (market structure understanding) with PPO (trading strategy discovery)
to train agents with innate market pattern recognition.

Architecture:
    OHLCV Data → Kronos Tokenizer → Kronos Transformer → Market Features ──┐
                                                                           ├──→ Concatenate → PPO Policy → Actions
    Portfolio State, Returns, Indicators → Standard Observations ──────────┘

Run:
    # With Kronos model (requires HuggingFace download):
    python training/kronos_training.py --kronos-size small --timesteps 200000
    
    # Fallback mode (technical indicators instead of Kronos):
    python training/kronos_training.py --kronos-size small --timesteps 200000
"""

import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from environments import SingleAssetTradingEnv, PortfolioOptimizationEnv
from environments.base_env import AssetConfig, TransactionCosts, RiskConstraints
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from config.loader import load_config


# Add Kronos to path
_KRONOS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "Kronos")
if os.path.exists(_KRONOS_PATH) and _KRONOS_PATH not in sys.path:
    sys.path.insert(0, _KRONOS_PATH)


@dataclass
class KronosTrainingConfig:
    """Configuration for Kronos-enhanced PPO training."""
    # Environment
    env_type: str = "single_asset"
    symbols: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL", "AMZN"])
    max_steps: int = 252
    
    # Kronos
    kronos_size: str = "small"  # mini, small, base
    kronos_feature_dim: int = 64
    kronos_lookback: int = 60
    
    # PPO
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_steps: int = 2048
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    kl_target: float = 0.015
    
    # Training
    total_timesteps: int = 200000
    n_seeds: int = 3
    eval_freq: int = 10000
    save_freq: int = 25000
    
    # Output
    output_dir: str = "checkpoints/kronos_trained"
    model_name: str = "kronos_ppo_finance"
    
    def to_dict(self) -> Dict:
        return {k: str(v) if isinstance(v, list) else v for k, v in self.__dict__.items()}


class KronosTrainingCallback(BaseCallback):
    """Callback that logs Kronos-enhanced training metrics."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.metrics = {
            "timesteps": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
            "explained_variance": [],
        }
    
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        if self.logger is None:
            return
        try:
            logs = self.logger.get_current()
            if logs:
                self.metrics["timesteps"].append(logs.get("time/total_timesteps", 0))
                self.metrics["policy_loss"].append(logs.get("train/policy_gradient_loss", 0))
                self.metrics["value_loss"].append(logs.get("train/value_loss", 0))
                self.metrics["entropy"].append(logs.get("train/entropy_loss", 0))
                self.metrics["approx_kl"].append(logs.get("train/approx_kl", 0))
        except Exception:
            pass


def make_env(config: KronosTrainingConfig, symbol: str = "AAPL", kronos_extractor=None):
    """Create a Kronos-enhanced environment factory."""
    from environments.kronos_wrapper import KronosObservationWrapper
    
    asset = AssetConfig(
        symbol=symbol,
        name=symbol,
        sector="Tech",
        initial_price=100.0,
        volatility=0.02,
        drift=0.0001,
    )
    
    def _init():
        env = SingleAssetTradingEnv(
            asset=asset,
            max_episode_length=config.max_steps,
            action_space_type="discrete",
        )
        wrapped = KronosObservationWrapper(
            env,
            kronos_extractor=kronos_extractor,
            ohlcv_history_len=config.kronos_lookback,
            feature_dim=config.kronos_feature_dim,
        )
        return wrapped
    
    return _init


def train_single(
    config: KronosTrainingConfig,
    seed: int = 42,
    progress_callback=None,
) -> Dict:
    """Train a single Kronos-enhanced PPO agent."""
    
    # Create single Kronos extractor (shared across env instances)
    from environments.kronos_wrapper import KronosFeatureExtractor
    kronos = KronosFeatureExtractor(
        model_size=config.kronos_size,
        feature_dim=config.kronos_feature_dim,
    )
    
    # Create environment
    env_fn = make_env(config, kronos_extractor=kronos)
    env = DummyVecEnv([env_fn])
    
    # Simple MLP policy (Kronos features are already in the observation)
    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 128], vf=[256, 128])],
    )
    
    print(f"  Creating PPO model (obs_dim={env.observation_space.shape[0]})...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        device="auto",
    )
    
    # Set up callbacks
    callback = KronosTrainingCallback()
    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_freq,
        save_path=os.path.join(config.output_dir, f"seed_{seed}"),
        name_prefix=config.model_name,
    )
    
    # Train with visible progress
    start = time.time()
    print(f"  Training {config.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=[callback, checkpoint_callback],
        progress_bar=True,
    )
    training_time = time.time() - start
    print(f"  Training complete: {training_time:.1f}s")
    
    # Save final model
    os.makedirs(config.output_dir, exist_ok=True)
    model.save(os.path.join(config.output_dir, f"{config.model_name}_s{seed}.zip"))
    
    # Evaluate
    eval_results = evaluate_model(model, config)
    
    env.close()
    
    return {
        "seed": seed,
        "training_time": training_time,
        "timesteps": config.total_timesteps,
        "metrics": callback.metrics,
        **eval_results,
    }


def evaluate_model(model, config: KronosTrainingConfig, n_episodes: int = 50) -> Dict:
    """Evaluate a trained model."""
    env = DummyVecEnv([make_env(config)])
    
    returns = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            ep_return += float(reward[0])
        returns.append(ep_return)
    
    env.close()
    
    returns = np.array(returns)
    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns))
    sharpe = float(mean_return / std_return) if std_return > 0 else 0.0
    win_rate = float(np.mean(returns > 0))
    max_dd = float(np.min(returns)) if len(returns) > 0 else 0.0
    
    return {
        "mean_return": mean_return,
        "std_return": std_return,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
    }


def train_multi_seed(config: KronosTrainingConfig):
    """Train multiple seeds and compile results."""
    
    print(f"\n{'='*60}")
    print(f"Kronos-Enhanced PPO Training")
    print(f"Model size: Kronos-{config.kronos_size}")
    print(f"Feature dim: {config.kronos_feature_dim}")
    print(f"Timesteps: {config.total_timesteps}")
    print(f"Seeds: {config.n_seeds}")
    print(f"{'='*60}\n")
    
    all_results = []
    for seed in range(config.n_seeds):
        print(f"\n--- Seed {seed} ---")
        result = train_single(config, seed=seed)
        all_results.append(result)
        print(f"  Sharpe: {result['sharpe']:.4f} +/- {result['std_return']:.4f}")
        print(f"  Return: {result['mean_return']:.4f}")
        print(f"  Win Rate: {result['win_rate']:.2%}")
        print(f"  Time: {result['training_time']:.1f}s")
    
    # Compile results
    sharpe_values = [r["sharpe"] for r in all_results]
    summary = {
        "config": config.to_dict(),
        "n_seeds": config.n_seeds,
        "mean_sharpe": float(np.mean(sharpe_values)),
        "std_sharpe": float(np.std(sharpe_values)),
        "mean_return": float(np.mean([r["mean_return"] for r in all_results])),
        "mean_win_rate": float(np.mean([r["win_rate"] for r in all_results])),
        "seed_results": all_results,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save summary
    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"RESULTS (across {config.n_seeds} seeds)")
    print(f"  Mean Sharpe: {summary['mean_sharpe']:.4f} +/- {summary['std_sharpe']:.4f}")
    print(f"  Mean Return: {summary['mean_return']:.4f}")
    print(f"  Mean Win Rate: {summary['mean_win_rate']:.2%}")
    print(f"{'='*60}")
    print(f"Results saved to {config.output_dir}/training_summary.json")
    
    return summary


def compare_without_kronos(config: KronosTrainingConfig, n_seeds: int = 2):
    """Run baseline (without Kronos) for comparison."""
    from training.multi_seed_training import train_multi_seed as baseline_train
    
    print(f"\nTraining BASELINE (without Kronos) for comparison...")
    
    # Create a config without Kronos features
    baseline_config = KronosTrainingConfig(
        total_timesteps=config.total_timesteps,
        n_seeds=n_seeds,
        output_dir=os.path.join(config.output_dir, "..", "baseline_comparison"),
        model_name="baseline_ppo",
    )
    
    # Run baseline using standard training
    from training.run_qwen_rl_training import train_single as baseline_single
    
    results = []
    for seed in range(n_seeds):
        asset = AssetConfig(symbol="AAPL", name="Apple", sector="Tech")
        env = DummyVecEnv([lambda: SingleAssetTradingEnv(asset=asset)])
        
        model = PPO("MlpPolicy", env, verbose=0, seed=seed)
        start = time.time()
        model.learn(total_timesteps=min(50000, config.total_timesteps))
        train_time = time.time() - start
        
        # Evaluate
        eval_returns = []
        for _ in range(30):
            obs = env.reset()
            done = False
            ep_ret = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                ep_ret += float(reward[0])
            eval_returns.append(ep_ret)
        
        env.close()
        results.append({
            "seed": seed,
            "sharpe": float(np.mean(eval_returns) / (np.std(eval_returns) + 1e-8)),
            "return": float(np.mean(eval_returns)),
        })
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kronos-enhanced PPO training")
    parser.add_argument("--kronos-size", choices=["mini", "small", "base"], default="small")
    parser.add_argument("--feature-dim", type=int, default=64, help="Kronos feature dimension")
    parser.add_argument("--timesteps", type=int, default=200000, help="Total training timesteps")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--output", default="checkpoints/kronos_trained", help="Output directory")
    parser.add_argument("--compare", action="store_true", help="Compare with baseline")
    parser.add_argument("--eval-only", type=str, default=None, help="Evaluate existing model")
    args = parser.parse_args()
    
    config = KronosTrainingConfig(
        kronos_size=args.kronos_size,
        kronos_feature_dim=args.feature_dim,
        total_timesteps=args.timesteps,
        n_seeds=args.seeds,
        learning_rate=args.lr,
        output_dir=args.output,
    )
    
    if args.eval_only:
        from stable_baselines3 import PPO
        model = PPO.load(args.eval_only)
        results = evaluate_model(model, config, n_episodes=100)
        print(f"Evaluation results:")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")
    else:
        summary = train_multi_seed(config)
        
        if args.compare:
            baseline = compare_without_kronos(config)
            print(f"\nComparison:")
            print(f"  With Kronos:   Sharpe={summary['mean_sharpe']:.4f}")
            if baseline:
                b_sharpes = [r["sharpe"] for r in baseline]
                print(f"  Without Kronos: Sharpe={np.mean(b_sharpes):.4f}")
