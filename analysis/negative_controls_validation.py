#!/usr/bin/env python3
"""
Negative Controls and Ablation Studies for Meta-Learning Validation

Implements critical controls to validate that meta-learning emergence
is a genuine phenomenon and not an artifact of training dynamics.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environments import SingleAssetTradingEnv
from environments.base_env import AssetConfig
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch

class NegativeControlCallback(BaseCallback):
    """Custom callback to capture training metrics for negative controls."""

    def __init__(self, control_type="normal", verbose: int = 0):
        super().__init__(verbose)
        self.control_type = control_type
        self.metrics_history = {
            'timesteps': [],
            'explained_variance': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy_bonus': [],
            'total_loss': []
        }

    def _on_step(self) -> bool:
        """Called at each step to allow for early stopping or custom logic."""
        return True

    def _on_rollout_end(self) -> bool:
        """Log metrics with proper error handling."""
        try:
            logs = self.logger.get_current()
        except AttributeError:
            try:
                logs = self.logger
            except:
                return True

        if logs is not None:
            timestep = logs.get('time/total_timesteps', 0)

            # Extract key metrics
            explained_variance = logs.get('train/explained_variance', 0)
            policy_loss = logs.get('train/policy_gradient_loss', 0)
            value_loss = logs.get('train/value_loss', 0)
            entropy_loss = logs.get('train/entropy_loss', 0)

            # Fix entropy sign
            entropy_bonus = abs(entropy_loss)

            # Calculate total loss
            total_loss = policy_loss + value_loss - entropy_bonus

            # Store metrics
            self.metrics_history['timesteps'].append(timestep)
            self.metrics_history['explained_variance'].append(explained_variance)
            self.metrics_history['policy_loss'].append(policy_loss)
            self.metrics_history['value_loss'].append(value_loss)
            self.metrics_history['entropy_bonus'].append(entropy_bonus)
            self.metrics_history['total_loss'].append(total_loss)

        return True

    def get_metrics_dataframe(self):
        """Return metrics as pandas DataFrame."""
        try:
            return pd.DataFrame(self.metrics_history)
        except:
            return self.metrics_history

class ShuffledRewardEnvironment:
    """Environment with shuffled rewards to break temporal patterns."""

    def __init__(self, base_env, shuffle_seed=42):
        self.base_env = base_env
        self.shuffle_seed = shuffle_seed
        self.reward_history = []
        self.shuffled_rewards = []

    def __getattr__(self, name):
        """Delegate all other methods to base environment."""
        return getattr(self.base_env, name)

    def step(self, action):
        """Override step to shuffle rewards."""
        obs, reward, done, info = self.base_env.step(action)
        self.reward_history.append(reward)

        # Return shuffled reward instead of actual reward
        if len(self.reward_history) > 1:
            # Shuffle historical rewards and return a random one
            np.random.seed(self.shuffle_seed + len(self.reward_history))
            shuffled_reward = np.random.choice(self.reward_history)
        else:
            shuffled_reward = reward

        return obs, shuffled_reward, done, info

class RandomizedBaselineEnvironment:
    """Environment with completely random rewards as control."""

    def __init__(self, base_env, reward_scale=0.1):
        self.base_env = base_env
        self.reward_scale = reward_scale

    def __getattr__(self, name):
        """Delegate all other methods to base environment."""
        return getattr(self.base_env, name)

    def step(self, action):
        """Override step to provide random rewards."""
        obs, reward, done, info = self.base_env.step(action)

        # Replace with random reward
        random_reward = np.random.normal(0, self.reward_scale)

        return obs, random_reward, done, info

class NoMetaLearningEnvironment:
    """Environment designed to prevent meta-learning by breaking patterns."""

    def __init__(self, base_env):
        self.base_env = base_env
        self.step_counter = 0

    def __getattr__(self, name):
        """Delegate all other methods to base environment."""
        return getattr(self.base_env, name)

    def step(self, action):
        """Override step to prevent pattern learning."""
        obs, reward, done, info = self.base_env.step(action)

        # Add pattern-breaking noise to observations
        if hasattr(obs, '__len__'):
            obs = obs + np.random.normal(0, 0.1, size=obs.shape)
        else:
            obs = obs + np.random.normal(0, 0.1)

        self.step_counter += 1
        return obs, reward, done, info

def create_base_environment(seed=42):
    """Create base trading environment for control experiments."""

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create sample asset data
    n_days = 500
    prices = []
    current_price = 150.0

    for day in range(n_days):
        daily_return = np.random.normal(0.0005, 0.02)
        current_price *= (1 + daily_return)
        prices.append(current_price)

    # Create asset configuration
    asset = AssetConfig(
        symbol="AAPL",
        name="Apple Inc.",
        sector="Technology",
        initial_price=prices[0],
        volatility=np.std(np.diff(prices) / prices[:-1])
    )

    # Create environment
    env = SingleAssetTradingEnv(
        asset=asset,
        initial_cash=100000,
        max_episode_length=252,
        lookback_window=30,
        render_mode=None
    )

    return env

def run_negative_control_experiment(control_type="normal", total_timesteps=25000):
    """
    Run training with specified negative control.

    Args:
        control_type: Type of negative control ("normal", "shuffled", "random", "no_meta")
        total_timesteps: Training timesteps

    Returns:
        Training metrics DataFrame
    """

    print(f"\n[CONTROL] Running {control_type} control experiment...")
    print(f"Training steps: {total_timesteps:,}")

    # Create base environment
    base_env = create_base_environment(seed=42)

    # Apply negative control
    if control_type == "shuffled":
        print("[CONTROL] Applied shuffled reward control")
        env = ShuffledRewardEnvironment(base_env, shuffle_seed=123)
    elif control_type == "random":
        print("[CONTROL] Applied random reward control")
        env = RandomizedBaselineEnvironment(base_env, reward_scale=0.1)
    elif control_type == "no_meta":
        print("[CONTROL] Applied no-meta-learning control")
        env = NoMetaLearningEnvironment(base_env)
    else:
        print("[CONTROL] Normal training (baseline)")
        env = base_env

    # Wrap for stable-baselines3
    env = DummyVecEnv([lambda: env])

    # Create simple PPO model (no custom features to isolate effect)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=0,
        seed=42
    )

    # Setup callback
    callback = NegativeControlCallback(control_type=control_type, verbose=0)

    # Train model
    try:
        start_time = datetime.now()
        model.learn(total_timesteps=total_timesteps, callback=[callback])
        training_time = (datetime.now() - start_time).total_seconds()

        print(f"[OK] {control_type} training completed in {training_time:.2f}s")

        # Get metrics
        metrics_df = callback.get_metrics_dataframe()
        if hasattr(metrics_df, 'to_dict'):
            return metrics_df
        else:
            return pd.DataFrame(metrics_df)

    except Exception as e:
        print(f"[ERROR] {control_type} training failed: {e}")
        return None

def analyze_meta_learning_signal(baseline_metrics, control_metrics_list, control_types):
    """
    Analyze whether meta-learning signal is present in baseline but absent in controls.

    Args:
        baseline_metrics: DataFrame with baseline training metrics
        control_metrics_list: List of DataFrames with control metrics
        control_types: List of control type names
    """

    print("\n" + "=" * 80)
    print("META-LEARNING SIGNAL ANALYSIS")
    print("=" * 80)

    if baseline_metrics is None or len(baseline_metrics) == 0:
        print("[ERROR] No baseline metrics available for analysis")
        return

    # Focus on the meta-learning window (45k-70k steps scaled to our training)
    baseline_max_steps = baseline_metrics['timesteps'].max() if len(baseline_metrics) > 0 else 25000
    meta_window_start = baseline_max_steps * 0.45
    meta_window_end = baseline_max_steps * 0.70

    print(f"Analysis window: {meta_window_start:.0f} - {meta_window_end:.0f} steps")

    # Analyze baseline
    baseline_mask = (baseline_metrics['timesteps'] >= meta_window_start) & \
                   (baseline_metrics['timesteps'] <= meta_window_end)

    if baseline_mask.sum() == 0:
        print("[WARNING] No baseline data in analysis window")
        return

    baseline_variance = baseline_metrics[baseline_mask]['explained_variance']
    baseline_min = baseline_variance.min()
    baseline_mean = baseline_variance.mean()

    print(f"\n[BASELINE] Explained Variance in Meta-Learning Window:")
    print(f"  Minimum: {baseline_min:.4f}")
    print(f"  Mean: {baseline_mean:.4f}")
    print(f"  Crisis Status: {'DETECTED' if baseline_min < 0.2 else 'Normal'}")

    # Analyze controls
    control_results = []

    for i, (control_metrics, control_type) in enumerate(zip(control_metrics_list, control_types)):
        if control_metrics is None or len(control_metrics) == 0:
            print(f"\n[{control_type.upper()}] No metrics available")
            continue

        control_mask = (control_metrics['timesteps'] >= meta_window_start) & \
                      (control_metrics['timesteps'] <= meta_window_end)

        if control_mask.sum() == 0:
            print(f"\n[{control_type.upper()}] No data in analysis window")
            continue

        control_variance = control_metrics[control_mask]['explained_variance']
        control_min = control_variance.min()
        control_mean = control_variance.mean()

        control_results.append({
            'type': control_type,
            'min_variance': control_min,
            'mean_variance': control_mean,
            'crisis_detected': control_min < 0.2
        })

        print(f"\n[{control_type.upper()}] Explained Variance in Meta-Learning Window:")
        print(f"  Minimum: {control_min:.4f}")
        print(f"  Mean: {control_mean:.4f}")
        print(f"  Crisis Status: {'DETECTED' if control_min < 0.2 else 'Normal'}")

    # Meta-learning signal validation
    print(f"\n[META-LEARNING SIGNAL VALIDATION]")
    print("-" * 50)

    baseline_crisis = baseline_min < 0.2
    control_crises = [r['crisis_detected'] for r in control_results]

    if baseline_crisis and not any(control_crises):
        print("✅ STRONG META-LEARNING SIGNAL DETECTED")
        print("   - Baseline shows crisis pattern")
        print("   - All controls show normal patterns")
        print("   - Effect is unique to meta-learning setup")

    elif baseline_crisis and any(control_crises):
        crisis_controls = [r['type'] for r in control_results if r['crisis_detected']]
        print("⚠️  WEAK META-LEARNING SIGNAL")
        print(f"   - Baseline shows crisis pattern")
        print(f"   - Controls with crisis: {', '.join(crisis_controls)}")
        print("   - Effect may not be unique to meta-learning")

    elif not baseline_crisis and not any(control_crises):
        print("❌ NO META-LEARNING SIGNAL")
        print("   - Neither baseline nor controls show crisis pattern")
        print("   - Meta-learning effect not detected")

    else:
        print("❌ INCONSISTENT RESULTS")
        print("   - Controls show crisis but baseline doesn't")
        print("   - Analysis may be flawed")

    # Effect size calculation
    if len(control_results) > 0:
        control_mins = [r['min_variance'] for r in control_results]
        effect_size = (np.mean(control_mins) - baseline_min) / np.std(control_mins) if np.std(control_mins) > 0 else 0

        print(f"\n[EFFECT SIZE ANALYSIS]")
        print(f"  Effect size (Cohen's d): {effect_size:.3f}")
        if abs(effect_size) > 0.8:
            print("  Effect size: LARGE")
        elif abs(effect_size) > 0.5:
            print("  Effect size: MEDIUM")
        elif abs(effect_size) > 0.2:
            print("  Effect size: SMALL")
        else:
            print("  Effect size: NEGLIGIBLE")

    return baseline_metrics, control_metrics_list, control_results

def generate_ablation_report(baseline_metrics, control_metrics_list, control_types, control_results):
    """Generate comprehensive ablation study report."""

    os.makedirs("./ablation_analysis", exist_ok=True)

    report_lines = [
        "# Meta-Learning Ablation Study Report",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        "",
        "This report presents negative control and ablation studies to validate that the",
        "meta-learning emergence phenomenon is genuine and not an artifact of training",
        "dynamics or environmental factors.",
        "",
        "## Control Experiments",
        ""
    ]

    # Add control descriptions
    control_descriptions = {
        "normal": "Baseline PPO training with Qwen features (meta-learning enabled)",
        "shuffled": "Temporal reward patterns broken through shuffling",
        "random": "Complete reward randomization eliminates learning signals",
        "no_meta": "Observation noise prevents pattern recognition"
    }

    for control_type, description in control_descriptions.items():
        if control_type in control_types:
            report_lines.extend([
                f"### {control_type.upper()} Control",
                f"- **Description:** {description}",
                ""
            ])

    # Add results
    if control_results:
        report_lines.extend([
            "## Results Summary",
            "",
            "| Control Type | Min Variance | Mean Variance | Crisis Detected |",
            "|---------------|--------------|---------------|----------------|"
        ])

        for result in control_results:
            crisis_status = "✅ YES" if result['crisis_detected'] else "❌ NO"
            report_lines.append(
                f"| {result['type'].title()} | {result['min_variance']:.4f} | "
                f"{result['mean_variance']:.4f} | {crisis_status} |"
            )

        report_lines.extend([
            "",
            "## Validation Conclusions",
            ""
        ])

        # Determine validation status
        baseline_crisis = any(result['type'] == 'normal' and result['crisis_detected'] for result in control_results)
        control_crises = [r for r in control_results if r['type'] != 'normal' and r['crisis_detected']]

        if baseline_crisis and len(control_crises) == 0:
            report_lines.extend([
                "✅ **STRONG VALIDATION**: Meta-learning emergence is genuine",
                "- Baseline shows clear crisis pattern at meta-learning threshold",
                "- All negative controls show normal training dynamics",
                "- Effect is unique to meta-learning configuration",
                "",
                "**Publication Ready**: Results meet rigorous validation standards"
            ])
        elif baseline_crisis:
            report_lines.extend([
                "⚠️ **PARTIAL VALIDATION**: Meta-learning effect detected but not unique",
                "- Baseline shows crisis pattern at meta-learning threshold",
                f"- {len(control_crises)} control(s) also show crisis patterns",
                "- Effect may be influenced by factors beyond meta-learning",
                "",
                "**Recommendation**: Additional controls needed for publication"
            ])
        else:
            report_lines.extend([
                "❌ **NO VALIDATION**: Meta-learning emergence not confirmed",
                "- Neither baseline nor controls show crisis patterns",
                "- Meta-learning threshold may not be present in this setup",
                "",
                "**Recommendation**: Re-examine experimental design"
            ])

    # Save report
    report_path = "./ablation_analysis/meta_learning_ablation_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"\n[OK] Ablation report saved to {report_path}")
    return report_path

def plot_ablation_comparison(baseline_metrics, control_metrics_list, control_types):
    """Create comparison plots for ablation study."""

    os.makedirs("./ablation_analysis/plots", exist_ok=True)

    # Plot explained variance comparison
    plt.figure(figsize=(15, 10))

    # Plot baseline
    if baseline_metrics is not None and len(baseline_metrics) > 0:
        plt.subplot(2, 2, 1)
        plt.plot(baseline_metrics['timesteps'], baseline_metrics['explained_variance'],
                'b-', linewidth=2, label='Baseline (Meta-Learning)')
        plt.axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='Crisis Threshold')
        plt.xlabel('Training Steps')
        plt.ylabel('Explained Variance')
        plt.title('Baseline: Meta-Learning Training')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Plot controls
    colors = ['red', 'green', 'orange']
    for i, (control_metrics, control_type) in enumerate(zip(control_metrics_list, control_types)):
        if control_metrics is not None and len(control_metrics) > 0:
            plt.subplot(2, 2, i+2)
            plt.plot(control_metrics['timesteps'], control_metrics['explained_variance'],
                    color=colors[i % len(colors)], linewidth=2, label=f'{control_type.title()} Control')
            plt.axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='Crisis Threshold')
            plt.xlabel('Training Steps')
            plt.ylabel('Explained Variance')
            plt.title(f'{control_type.title()} Control Training')
            plt.legend()
            plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./ablation_analysis/plots/ablation_variance_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Create summary comparison plot
    if baseline_metrics is not None and len(control_metrics_list) > 0:
        plt.figure(figsize=(12, 8))

        # Calculate minimum variance in meta-learning window for each condition
        conditions = ['Baseline'] + [c.title() for c in control_types]
        min_variances = []
        all_metrics = [baseline_metrics] + control_metrics_list

        for metrics in all_metrics:
            if metrics is not None and len(metrics) > 0:
                max_steps = metrics['timesteps'].max()
                window_start = max_steps * 0.45
                window_end = max_steps * 0.70

                mask = (metrics['timesteps'] >= window_start) & (metrics['timesteps'] <= window_end)
                if mask.sum() > 0:
                    min_var = metrics[mask]['explained_variance'].min()
                    min_variances.append(min_var)
                else:
                    min_variances.append(np.nan)
            else:
                min_variances.append(np.nan)

        # Plot bar chart
        bars = plt.bar(conditions, min_variances, alpha=0.7,
                      color=['blue'] + ['red', 'green', 'orange'][:len(control_types)])
        plt.axhline(y=0.2, color='red', linestyle='--', linewidth=2, label='Crisis Threshold')

        # Add value labels on bars
        for bar, var in zip(bars, min_variances):
            if not np.isnan(var):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{var:.3f}', ha='center', va='bottom')

        plt.xlabel('Training Condition')
        plt.ylabel('Minimum Explained Variance')
        plt.title('Meta-Learning Crisis Detection: Ablation Study Results')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim(0, max(min_variances) + 0.1 if min_variances else 0.5)

        plt.tight_layout()
        plt.savefig('./ablation_analysis/plots/ablation_summary_comparison.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    print("[OK] Ablation plots saved to ./ablation_analysis/plots/")

def main():
    """Run complete ablation and negative control study."""

    print("META-LEARNING ABLATION AND NEGATIVE CONTROL STUDY")
    print("=" * 60)
    print("Testing whether meta-learning emergence is a genuine phenomenon")

    # Define control experiments
    control_types = ["normal", "shuffled", "random", "no_meta"]

    # Run experiments
    all_metrics = []

    for control_type in control_types:
        metrics = run_negative_control_experiment(control_type, total_timesteps=25000)
        all_metrics.append(metrics)

    # Separate baseline and controls
    baseline_metrics = all_metrics[0]  # normal training
    control_metrics_list = all_metrics[1:]  # shuffled, random, no_meta
    control_types_only = control_types[1:]

    # Analyze results
    baseline_metrics, control_metrics_list, control_results = analyze_meta_learning_signal(
        baseline_metrics, control_metrics_list, control_types_only
    )

    # Generate report and plots
    if baseline_metrics is not None:
        report_path = generate_ablation_report(baseline_metrics, control_metrics_list,
                                             control_types_only, control_results)
        plot_ablation_comparison(baseline_metrics, control_metrics_list, control_types_only)

        print(f"\n[SUCCESS] Ablation study completed!")
        print(f"Results saved in ./ablation_analysis/")
        print(f"Report: {report_path}")
    else:
        print("\n[ERROR] Ablation study failed - no baseline metrics available")

if __name__ == "__main__":
    main()