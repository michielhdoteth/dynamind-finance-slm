#!/usr/bin/env python3
"""
Ablation Study for Qwen RL Model

Comprehensive ablation testing to understand which components are most important
for the trained Qwen RL model's performance.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our gym and training components
from environments import SingleAssetTradingEnv
from environments.base_env import AssetConfig
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import matplotlib.pyplot as plt

class ReducedFeaturesExtractor(BaseFeaturesExtractor):
    """Reduced feature extractor for ablation testing."""

    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        # Simplified neural network for feature extraction
        self.net = nn.Sequential(
            nn.Linear(np.prod(observation_space.shape), 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, features_dim)
        )

    def forward(self, observations):
        return self.net(observations)

class MinimalFeaturesExtractor(BaseFeaturesExtractor):
    """Minimal feature extractor for extreme ablation testing."""

    def __init__(self, observation_space, features_dim: int = 64):
        super().__init__(observation_space, features_dim)

        # Very simple neural network
        self.net = nn.Sequential(
            nn.Linear(np.prod(observation_space.shape), 128),
            nn.ReLU(),
            nn.Linear(128, features_dim)
        )

    def forward(self, observations):
        return self.net(observations)

def create_test_environment_with_noise(noise_level=0.0, volatility_scale=1.0, trend_strength=0.0):
    """Create test environment with different characteristics for ablation testing."""

    np.random.seed(456)  # Different seed for ablation tests
    n_days = 252
    prices = []
    current_price = 150.0

    for day in range(n_days):
        # Add noise, scaled volatility, and trend
        noise = np.random.normal(0, noise_level)
        scaled_vol = np.random.normal(0, 0.02 * volatility_scale)
        trend = trend_strength * 0.0001  # Small daily trend
        daily_return = np.random.normal(trend, 0.02 * volatility_scale) + noise

        current_price *= (1 + daily_return)
        prices.append(max(current_price, 1.0))  # Ensure positive prices

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

def test_model_configuration(model_path, config_name, env, num_episodes=3):
    """Test a specific model configuration."""

    print(f"\n[ABLATION] Testing {config_name}...")

    try:
        # Load model
        model = PPO.load(model_path)

        episode_rewards = []
        episode_returns = []

        for episode in range(num_episodes):
            obs = env.reset()
            total_reward = 0
            done = False

            while not done:
                # Handle observation format
                if isinstance(obs, tuple):
                    obs_array = obs[0]
                else:
                    obs_array = obs

                if len(obs_array.shape) == 1:
                    obs_array = obs_array.reshape(1, -1)

                action, _ = model.predict(obs_array, deterministic=True)
                action = action[0]

                step_result = env.step(action)

                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                elif len(step_result) == 5:
                    obs, reward, done, truncated, info = step_result
                    done = done or truncated
                else:
                    obs, reward, done = step_result[0], step_result[1], step_result[2]
                    info = {}

                total_reward += reward

            # Calculate portfolio return
            portfolio_return = (env.portfolio_value - 100000) / 100000
            episode_rewards.append(total_reward)
            episode_returns.append(portfolio_return)

            print(f"  Episode {episode + 1}: Reward={total_reward:.4f}, Return={portfolio_return:.2%}")

        avg_reward = np.mean(episode_rewards)
        avg_return = np.mean(episode_returns)

        results = {
            'config_name': config_name,
            'avg_reward': avg_reward,
            'avg_return': avg_return,
            'episode_rewards': episode_rewards,
            'episode_returns': episode_returns,
            'success': True
        }

        print(f"[OK] {config_name} - Avg Return: {avg_return:.2%}")
        return results

    except Exception as e:
        print(f"[ERROR] {config_name} failed: {e}")
        return {
            'config_name': config_name,
            'avg_reward': 0,
            'avg_return': 0,
            'episode_rewards': [],
            'episode_returns': [],
            'success': False,
            'error': str(e)
        }

def train_reduced_model(env, features_extractor_class, features_dim, config_name, timesteps=10000):
    """Train a model with reduced features for comparison."""

    print(f"\n[TRAIN] Training {config_name} for {timesteps} timesteps...")

    try:
        policy_kwargs = dict(
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=dict(features_dim=features_dim),
            net_arch=[dict(pi=[128, 64], vf=[128, 64])],
            activation_fn=torch.nn.ReLU
        )

        model = PPO(
            "MlpPolicy",
            DummyVecEnv([lambda: env]),
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=32,
            n_epochs=5,
            gamma=0.99,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=789
        )

        model.learn(total_timesteps=timesteps, progress_bar=False)

        print(f"[OK] {config_name} training completed")
        return model

    except Exception as e:
        print(f"[ERROR] {config_name} training failed: {e}")
        return None

def run_ablation_study():
    """Run comprehensive ablation study."""

    print("QWEN RL MODEL ABLATION STUDY")
    print("Testing model components and configurations to identify key performance factors")
    print("=" * 80)

    results = {}

    # Test 1: Original trained model (baseline)
    print("\n" + "=" * 50)
    print("ABLATION TEST 1: BASELINE MODEL")
    print("=" * 50)

    base_env = create_test_environment_with_noise()
    baseline_result = test_model_configuration(
        "./models/qwen_final_model",
        "Baseline Qwen Model",
        base_env
    )
    results['baseline'] = baseline_result

    # Test 2: Noisy environment
    print("\n" + "=" * 50)
    print("ABLATION TEST 2: NOISY ENVIRONMENT")
    print("=" * 50)

    noisy_env = create_test_environment_with_noise(noise_level=0.01)
    noisy_result = test_model_configuration(
        "./models/qwen_final_model",
        "Noisy Environment",
        noisy_env
    )
    results['noisy_env'] = noisy_result

    # Test 3: High volatility environment
    print("\n" + "=" * 50)
    print("ABLATION TEST 3: HIGH VOLATILITY")
    print("=" * 50)

    volatile_env = create_test_environment_with_noise(volatility_scale=2.0)
    volatile_result = test_model_configuration(
        "./models/qwen_final_model",
        "High Volatility",
        volatile_env
    )
    results['high_volatility'] = volatile_result

    # Test 4: Trending market
    print("\n" + "=" * 50)
    print("ABLATION TEST 4: TRENDING MARKET")
    print("=" * 50)

    trending_env = create_test_environment_with_noise(trend_strength=0.01)
    trending_result = test_model_configuration(
        "./models/qwen_final_model",
        "Trending Market",
        trending_env
    )
    results['trending_market'] = trending_result

    # Test 5: Train and test reduced features model
    print("\n" + "=" * 50)
    print("ABLATION TEST 5: REDUCED FEATURES")
    print("=" * 50)

    reduced_env = create_test_environment_with_noise()
    reduced_model = train_reduced_model(
        reduced_env,
        ReducedFeaturesExtractor,
        128,
        "Reduced Features Model",
        timesteps=10000
    )

    if reduced_model:
        reduced_result = test_model_configuration(
            reduced_model,  # Use the model object directly
            "Reduced Features Model",
            reduced_env
        )
        results['reduced_features'] = reduced_result
    else:
        results['reduced_features'] = {
            'config_name': 'Reduced Features Model',
            'avg_return': 0,
            'success': False,
            'error': 'Training failed'
        }

    # Test 6: Train and test minimal features model
    print("\n" + "=" * 50)
    print("ABLATION TEST 6: MINIMAL FEATURES")
    print("=" * 50)

    minimal_env = create_test_environment_with_noise()
    minimal_model = train_reduced_model(
        minimal_env,
        MinimalFeaturesExtractor,
        64,
        "Minimal Features Model",
        timesteps=10000
    )

    if minimal_model:
        minimal_result = test_model_configuration(
            minimal_model,
            "Minimal Features Model",
            minimal_env
        )
        results['minimal_features'] = minimal_result
    else:
        results['minimal_features'] = {
            'config_name': 'Minimal Features Model',
            'avg_return': 0,
            'success': False,
            'error': 'Training failed'
        }

    # Analyze results
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)

    print(f"\nConfiguration Performance Comparison:")
    print("-" * 60)

    successful_results = {k: v for k, v in results.items() if v.get('success', False)}

    if successful_results:
        # Sort by average return
        sorted_results = sorted(successful_results.items(),
                              key=lambda x: x[1]['avg_return'],
                              reverse=True)

        for i, (config_name, result) in enumerate(sorted_results, 1):
            print(f"{i:2d}. {config_name:20} | Return: {result['avg_return']:7.2%} | Reward: {result['avg_reward']:7.4f}")

        # Calculate performance drop from baseline
        baseline_return = results['baseline']['avg_return']
        print(f"\nPerformance Analysis vs Baseline:")
        print("-" * 40)

        for config_name, result in successful_results.items():
            if config_name != 'baseline':
                performance_diff = (result['avg_return'] - baseline_return) / baseline_return * 100
                print(f"{config_name:20}: {performance_diff:+7.1f}% from baseline")

    # Identify key insights
    print(f"\nKey Insights:")
    print("-" * 30)

    if 'noisy_env' in successful_results:
        noise_resilience = (successful_results['noisy_env']['avg_return'] / baseline_return) * 100
        print(f"• Model shows {noise_resilience:.1f}% performance in noisy markets")

    if 'high_volatility' in successful_results:
        vol_performance = (successful_results['high_volatility']['avg_return'] / baseline_return) * 100
        print(f"• Model achieves {vol_performance:.1f}% of baseline performance in high volatility")

    if 'trending_market' in successful_results:
        trend_performance = (successful_results['trending_market']['avg_return'] / baseline_return) * 100
        print(f"• Model performs {trend_performance:.1f}% of baseline in trending markets")

    if 'reduced_features' in successful_results and successful_results['reduced_features'].get('success'):
        feature_importance = (successful_results['reduced_features']['avg_return'] / baseline_return) * 100
        print(f"• Reduced features maintain {feature_importance:.1f}% of performance")

    print(f"\nAblation Study Completed Successfully!")
    return results

def main():
    """Main ablation study function."""

    try:
        results = run_ablation_study()

        # Save results summary
        print(f"\nAblation test completed with {len(results)} configurations tested.")
        return 0

    except Exception as e:
        print(f"[ERROR] Ablation study failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)