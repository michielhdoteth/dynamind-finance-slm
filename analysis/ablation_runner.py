"""
Full Ablation Study Framework.
Tests the contribution of each component to overall performance.
"""
import os, sys, json, time, itertools, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from environments import SingleAssetTradingEnv
from environments.base_env import AssetConfig
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


@dataclass
class AblationConfig:
    """Configuration for a single ablation run."""
    name: str
    model_size: str = "0.5B"        # 0.5B, 1.5B
    algorithm: str = "PPO"           # PPO, SAC, A2C
    reward_type: str = "cvar"        # simple, cvar, sharpe
    use_curriculum: bool = False
    feature_extractor: str = "mlp"   # mlp, qwen
    training_length: int = 100000
    seed: int = 42


# Define all ablation dimensions
MODEL_SIZES = ["0.5B"]
ALGORITHMS = ["PPO", "SAC", "A2C"]
REWARD_TYPES = ["simple", "cvar", "sharpe"]
FEATURE_EXTRACTORS = ["mlp", "qwen"]
CURRICULUM = [False, True]
TRAINING_LENGTHS = [50000, 100000, 200000]

# Specific ablation experiments
ABLATION_RUNS = [
    # Baseline
    AblationConfig(name="baseline"),
    # Algorithm comparison
    AblationConfig(name="alg_sac", algorithm="SAC"),
    AblationConfig(name="alg_a2c", algorithm="A2C"),
    # Reward ablation
    AblationConfig(name="reward_simple", reward_type="simple"),
    AblationConfig(name="reward_sharpe", reward_type="sharpe"),
    # Feature extractor
    AblationConfig(name="feat_qwen", feature_extractor="qwen"),
    # Curriculum
    AblationConfig(name="curriculum", use_curriculum=True),
    # Training length
    AblationConfig(name="len_50k", training_length=50000),
    AblationConfig(name="len_200k", training_length=200000),
]


class SimpleFeaturesExtractor(BaseFeaturesExtractor):
    """Simple MLP feature extractor for ablation testing."""

    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Linear(int(np.prod(observation_space.shape)), 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
        )

    def forward(self, observations):
        return self.net(observations)


class AblationRunner:
    """Run ablation studies and collect results."""

    def __init__(self, n_seeds: int = 3):
        self.n_seeds = n_seeds
        self.results = []

    def _make_env(self, seed: int = 42):
        """Create a single trading environment with deterministic seed."""
        np.random.seed(seed)
        n_days = 500
        prices = []
        current_price = 150.0
        for _ in range(n_days):
            daily_return = np.random.normal(0.0005, 0.02)
            current_price *= (1 + daily_return)
            prices.append(max(current_price, 1.0))

        asset = AssetConfig(
            symbol="AAPL",
            name="Apple Inc.",
            sector="Technology",
            initial_price=prices[0],
            volatility=np.std(np.diff(prices) / np.maximum(np.array(prices[:-1]), 1e-10)),
        )

        env = SingleAssetTradingEnv(
            asset=asset,
            initial_cash=100000,
            max_episode_length=252,
            lookback_window=30,
            render_mode=None,
        )
        return env

    def run_single(self, config: AblationConfig) -> Dict:
        """Run a single ablation experiment."""
        asset = AssetConfig(
            symbol="AAPL",
            name="Apple Inc.",
            sector="Tech",
        )

        # Use DummyVecEnv for stable-baselines3 compatibility
        base_env = self._make_env(seed=config.seed)
        env = DummyVecEnv([lambda: base_env])

        # Select algorithm
        alg_map = {
            "PPO": PPO,
            "SAC": SAC,
            "A2C": A2C,
        }
        alg_class = alg_map.get(config.algorithm, PPO)

        # Build policy kwargs based on feature extractor choice
        policy_kwargs = {}

        if config.feature_extractor == "qwen":
            policy_kwargs["features_extractor_class"] = SimpleFeaturesExtractor
            policy_kwargs["features_extractor_kwargs"] = {"features_dim": 256}
            policy_kwargs["net_arch"] = [dict(pi=[256, 128], vf=[256, 128])]
        elif config.feature_extractor == "mlp":
            policy_kwargs["net_arch"] = [dict(pi=[256, 128], vf=[256, 128])]

        # Handle different algorithm parameters
        common_params = dict(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            gamma=0.99,
            verbose=0,
            seed=config.seed,
            policy_kwargs=policy_kwargs,
        )

        if config.algorithm == "PPO":
            common_params["n_steps"] = 2048
            common_params["batch_size"] = 64
        elif config.algorithm == "SAC":
            common_params["batch_size"] = 64
        elif config.algorithm == "A2C":
            common_params["n_steps"] = 2048

        model = alg_class(**common_params)

        start = time.time()
        model.learn(total_timesteps=config.training_length, progress_bar=False)
        train_time = time.time() - start

        # Evaluate
        eval_env = self._make_env(seed=config.seed + 1000)
        eval_vec_env = DummyVecEnv([lambda: eval_env])
        returns = []
        for _ in range(50):
            obs = eval_vec_env.reset()
            done = False
            ep_ret = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done_arr, _ = eval_vec_env.step(action)
                ep_ret += float(reward[0])
                done = bool(done_arr[0])
            returns.append(ep_ret)

        mean_ret = float(np.mean(returns)) if returns else 0.0
        std_ret = float(np.std(returns)) if returns else 0.0
        sharpe = float(mean_ret / std_ret) if std_ret > 0 else 0.0
        win_rate = float(np.mean([r > 0 for r in returns])) if returns else 0.0

        env.close()
        eval_vec_env.close()

        return {
            "name": config.name,
            "model_size": config.model_size,
            "algorithm": config.algorithm,
            "reward_type": config.reward_type,
            "feature_extractor": config.feature_extractor,
            "use_curriculum": config.use_curriculum,
            "training_length": config.training_length,
            "seed": config.seed,
            "mean_return": mean_ret,
            "std_return": std_ret,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "training_time": train_time,
        }

    def run_all(self, parallel: bool = True) -> List[Dict]:
        """Run all ablation experiments."""
        all_configs = []
        for config in ABLATION_RUNS:
            for seed in range(self.n_seeds):
                c = AblationConfig(
                    name=f"{config.name}_s{seed}",
                    model_size=config.model_size,
                    algorithm=config.algorithm,
                    reward_type=config.reward_type,
                    feature_extractor=config.feature_extractor,
                    use_curriculum=config.use_curriculum,
                    training_length=config.training_length,
                    seed=seed,
                )
                all_configs.append(c)

        print(f"Running {len(all_configs)} ablation experiments...")

        if parallel:
            with ProcessPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(self.run_single, c): c for c in all_configs}
                for future in as_completed(futures):
                    config = futures[future]
                    try:
                        result = future.result()
                        self.results.append(result)
                        print(f"  Completed: {config.name} (Sharpe={result['sharpe']:.3f})")
                    except Exception as e:
                        print(f"  FAILED: {config.name}: {e}")
        else:
            for config in all_configs:
                try:
                    result = self.run_single(config)
                    self.results.append(result)
                    print(f"  Completed: {config.name} (Sharpe={result['sharpe']:.3f})")
                except Exception as e:
                    print(f"  FAILED: {config.name}: {e}")

        return self.results

    def analyze(self) -> pd.DataFrame:
        """Analyze results and determine contribution of each component."""
        if not self.results:
            return pd.DataFrame()

        df = pd.DataFrame(self.results)

        # Group by experiment name (across seeds)
        summary = df.groupby("name").agg({
            "sharpe": ["mean", "std"],
            "mean_return": "mean",
            "std_return": "mean",
            "win_rate": "mean",
            "training_time": "mean",
        }).round(4)

        return summary

    def compute_contributions(self) -> pd.DataFrame:
        """Compute the contribution of each component relative to baseline."""
        if not self.results:
            return pd.DataFrame()

        df = pd.DataFrame(self.results)
        baseline_sharpe = df[df["name"].str.startswith("baseline")]["sharpe"].mean()

        contributions = {}
        for config in ABLATION_RUNS:
            if config.name == "baseline":
                continue
            mask = df["name"].str.startswith(config.name)
            config_sharpe = df[mask]["sharpe"].mean()
            contributions[config.name] = {
                "sharpe": round(config_sharpe, 4),
                "delta": round(config_sharpe - baseline_sharpe, 4),
                "delta_pct": round((config_sharpe - baseline_sharpe) / baseline_sharpe * 100, 2) if baseline_sharpe != 0 else 0,
            }

        contrib_df = pd.DataFrame.from_dict(contributions, orient="index")
        contrib_df.index.name = "component"
        return contrib_df

    def plot_results(self, output_path: str = "results/ablation_results.png"):
        """Generate ablation study plots."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            df = pd.DataFrame(self.results)
            if df.empty:
                print("No results to plot.")
                return

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Clean up experiment names for display
            plot_df = df.copy()
            # Use the first 20 chars of name for display
            plot_df["display_name"] = plot_df["name"].apply(lambda x: x.rsplit("_s", 1)[0] if "_s" in x else x)

            # Sharpe by component
            ax = axes[0, 0]
            sns.barplot(data=plot_df, x="display_name", y="sharpe", ax=ax, ci="sd")
            ax.set_title("Sharpe Ratio by Ablation Component")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

            # Return by component
            ax = axes[0, 1]
            sns.barplot(data=plot_df, x="display_name", y="mean_return", ax=ax, ci="sd")
            ax.set_title("Mean Return by Ablation Component")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

            # Win rate by component
            ax = axes[1, 0]
            sns.barplot(data=plot_df, x="display_name", y="win_rate", ax=ax, ci="sd")
            ax.set_title("Win Rate by Ablation Component")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

            # Training time by component
            ax = axes[1, 1]
            sns.barplot(data=plot_df, x="display_name", y="training_time", ax=ax, ci="sd")
            ax.set_title("Training Time (seconds) by Ablation Component")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

            plt.tight_layout()
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {output_path}")
            plt.close()
        except ImportError:
            print("matplotlib/seaborn not available for plotting")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ablation study framework")
    parser.add_argument("--seeds", type=int, default=3, help="Seeds per config")
    parser.add_argument("--sequential", action="store_true", help="Run sequentially")
    parser.add_argument("--analyze-only", action="store_true", help="Analyze existing results")
    parser.add_argument("--output", default="results/ablation_results.json", help="Output path")
    args = parser.parse_args()

    runner = AblationRunner(n_seeds=args.seeds)

    if args.analyze_only:
        if os.path.exists(args.output):
            with open(args.output) as f:
                runner.results = json.load(f)
            summary = runner.analyze()
            print("\n=== ABLATION SUMMARY ===")
            print(summary)
            print("\n=== COMPONENT CONTRIBUTIONS ===")
            contribs = runner.compute_contributions()
            print(contribs)
            runner.plot_results()
        else:
            print(f"No existing results found at {args.output}. Run without --analyze-only first.")
    else:
        results = runner.run_all(parallel=not args.sequential)

        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {args.output}")

        summary = runner.analyze()
        print("\n=== ABLATION SUMMARY ===")
        print(summary)

        print("\n=== COMPONENT CONTRIBUTIONS ===")
        contribs = runner.compute_contributions()
        print(contribs)

        runner.plot_results()
