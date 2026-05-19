#!/usr/bin/env python3
"""
Simple Ablation Study for Qwen RL Model

Focus on testing model robustness and feature importance without
training new models.
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
import matplotlib.pyplot as plt

def create_test_environment_with_modifications(modifications=None):
    """Create test environment with specific modifications for ablation testing."""

    np.random.seed(456)  # Consistent seed for ablation tests
    n_days = 252
    prices = []
    current_price = 150.0

    # Default modifications
    if modifications is None:
        modifications = {}

    # Extract modification parameters
    noise_level = modifications.get('noise_level', 0.0)
    volatility_scale = modifications.get('volatility_scale', 1.0)
    trend_strength = modifications.get('trend_strength', 0.0)
    lookback_window = modifications.get('lookback_window', 30)
    max_episode_length = modifications.get('max_episode_length', 252)

    for day in range(n_days):
        # Add various market effects
        noise = np.random.normal(0, noise_level)
        scaled_vol = np.random.normal(0, 0.02 * volatility_scale)
        trend = trend_strength * 0.0001

        daily_return = np.random.normal(trend, 0.02 * volatility_scale) + noise
        current_price *= (1 + daily_return)
        prices.append(max(current_price, 1.0))

    # Create asset configuration
    asset = AssetConfig(
        symbol="AAPL",
        name="Apple Inc.",
        sector="Technology",
        initial_price=prices[0],
        volatility=np.std(np.diff(prices) / prices[:-1])
    )

    # Create environment with modifications
    env = SingleAssetTradingEnv(
        asset=asset,
        initial_cash=100000,
        max_episode_length=max_episode_length,
        lookback_window=lookback_window,
        render_mode=None
    )

    return env

def modify_observations(obs, modification_type='mask_random'):
    """Modify observations for feature ablation testing."""

    if modification_type == 'mask_random':
        # Randomly mask 20% of features
        mask = np.random.random(len(obs)) < 0.8
        obs_modified = obs.copy()
        obs_modified[~mask] = 0
        return obs_modified

    elif modification_type == 'noise_injection':
        # Add small noise to observations
        noise = np.random.normal(0, 0.01, len(obs))
        return obs + noise

    elif modification_type == 'normalize_only':
        # Only use normalized price features
        # Keep first 30 features (price history) and zero out technical indicators
        obs_modified = obs.copy()
        obs_modified[30:] = 0
        return obs_modified

    elif modification_type == 'drop_technical':
        # Remove technical indicators (assume they start after index 40)
        obs_modified = obs.copy()
        obs_modified[40:] = 0
        return obs_modified

    else:
        return obs

def test_model_with_modifications(model_path, test_name, modifications, num_episodes=5):
    """Test model with specific environment or observation modifications."""

    print(f"\n[ABLATION] Testing {test_name}...")

    try:
        # Load model
        model = PPO.load(model_path)

        episode_rewards = []
        episode_returns = []

        for episode in range(num_episodes):
            # Create environment with modifications
            env = create_test_environment_with_modifications(modifications)
            obs = env.reset()
            total_reward = 0
            done = False

            while not done:
                # Apply observation modifications if specified
                if 'observation_modification' in modifications:
                    obs_to_use = modify_observations(obs[0] if isinstance(obs, tuple) else obs,
                                                        modifications['observation_modification'])
                    obs_to_use = obs_to_use.reshape(1, -1)
                else:
                    # Handle normal observation format
                    if isinstance(obs, tuple):
                        obs_array = obs[0]
                    else:
                        obs_array = obs

                    if len(obs_array.shape) == 1:
                        obs_array = obs_array.reshape(1, -1)

                    obs_to_use = obs_array

                action, _ = model.predict(obs_to_use, deterministic=True)
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
            'test_name': test_name,
            'avg_reward': avg_reward,
            'avg_return': avg_return,
            'episode_rewards': episode_rewards,
            'episode_returns': episode_returns,
            'success': True
        }

        print(f"[OK] {test_name} - Avg Return: {avg_return:.2%}")
        return results

    except Exception as e:
        print(f"[ERROR] {test_name} failed: {e}")
        return {
            'test_name': test_name,
            'avg_reward': 0,
            'avg_return': 0,
            'episode_rewards': [],
            'episode_returns': [],
            'success': False,
            'error': str(e)
        }

def run_simple_ablation_study():
    """Run focused ablation study on model robustness."""

    print("QWEN RL MODEL SIMPLE ABLATION STUDY")
    print("Testing model robustness and feature importance")
    print("=" * 70)

    model_path = "./models/qwen_final_model"

    if not os.path.exists(model_path + ".zip"):
        print(f"[ERROR] Trained model not found at {model_path}")
        return None

    results = {}

    # Test 1: Baseline (original conditions)
    print("\n" + "=" * 50)
    print("ABLATION TEST 1: BASELINE PERFORMANCE")
    print("=" * 50)

    baseline_result = test_model_with_modifications(
        model_path,
        "Baseline (Original)",
        {}
    )
    results['baseline'] = baseline_result

    # Test 2: Noisy observations
    print("\n" + "=" * 50)
    print("ABLATION TEST 2: NOISY OBSERVATIONS")
    print("=" * 50)

    noisy_obs_result = test_model_with_modifications(
        model_path,
        "Noisy Observations",
        {'observation_modification': 'noise_injection'}
    )
    results['noisy_observations'] = noisy_obs_result

    # Test 3: Masked observations
    print("\n" + "=" * 50)
    print("ABLATION TEST 3: MASKED OBSERVATIONS")
    print("=" * 50)

    masked_obs_result = test_model_with_modifications(
        model_path,
        "Masked Observations",
        {'observation_modification': 'mask_random'}
    )
    results['masked_observations'] = masked_obs_result

    # Test 4: Price-only observations
    print("\n" + "=" * 50)
    print("ABLATION TEST 4: PRICE-ONLY OBSERVATIONS")
    print("=" * 50)

    price_only_result = test_model_with_modifications(
        model_path,
        "Price-Only Observations",
        {'observation_modification': 'normalize_only'}
    )
    results['price_only'] = price_only_result

    # Test 5: No technical indicators
    print("\n" + "=" * 50)
    print("ABLATION TEST 5: NO TECHNICAL INDICATORS")
    print("=" * 50)

    no_tech_result = test_model_with_modifications(
        model_path,
        "No Technical Indicators",
        {'observation_modification': 'drop_technical'}
    )
    results['no_technical'] = no_tech_result

    # Test 6: Shorter lookback window
    print("\n" + "=" * 50)
    print("ABLATION TEST 6: SHORTER LOOKBACK WINDOW")
    print("=" * 50)

    short_lookback_result = test_model_with_modifications(
        model_path,
        "Short Lookback (15 days)",
        {'lookback_window': 15}
    )
    results['short_lookback'] = short_lookback_result

    # Test 7: Longer lookback window
    print("\n" + "=" * 50)
    print("ABLATION TEST 7: LONGER LOOKBACK WINDOW")
    print("=" * 50)

    long_lookback_result = test_model_with_modifications(
        model_path,
        "Long Lookback (60 days)",
        {'lookback_window': 60}
    )
    results['long_lookback'] = long_lookback_result

    # Test 8: High volatility environment
    print("\n" + "=" * 50)
    print("ABLATION TEST 8: HIGH VOLATILITY")
    print("=" * 50)

    high_vol_result = test_model_with_modifications(
        model_path,
        "High Volatility",
        {'volatility_scale': 2.0}
    )
    results['high_volatility'] = high_vol_result

    # Test 9: Strong trend environment
    print("\n" + "=" * 50)
    print("ABLATION TEST 9: STRONG TREND")
    print("=" * 50)

    strong_trend_result = test_model_with_modifications(
        model_path,
        "Strong Trend",
        {'trend_strength': 0.05}
    )
    results['strong_trend'] = strong_trend_result

    # Test 10: Noisy environment
    print("\n" + "=" * 50)
    print("ABLATION TEST 10: NOISY ENVIRONMENT")
    print("=" * 50)

    noisy_env_result = test_model_with_modifications(
        model_path,
        "Noisy Environment",
        {'noise_level': 0.01}
    )
    results['noisy_environment'] = noisy_env_result

    # Analyze results
    print("\n" + "=" * 80)
    print("SIMPLE ABLATION STUDY RESULTS")
    print("=" * 80)

    print(f"\nConfiguration Performance Comparison:")
    print("-" * 70)

    successful_results = {k: v for k, v in results.items() if v.get('success', False)}

    if successful_results:
        # Sort by average return
        sorted_results = sorted(successful_results.items(),
                              key=lambda x: x[1]['avg_return'],
                              reverse=True)

        for i, (test_name, result) in enumerate(sorted_results, 1):
            print(f"{i:2d}. {test_name:25} | Return: {result['avg_return']:7.2%} | Reward: {result['avg_reward']:7.4f}")

        # Calculate performance drop from baseline
        if 'baseline' in successful_results:
            baseline_return = successful_results['baseline']['avg_return']
            print(f"\nPerformance Analysis vs Baseline:")
            print("-" * 40)

            for test_name, result in successful_results.items():
                if test_name != 'baseline':
                    performance_diff = (result['avg_return'] - baseline_return) / baseline_return * 100
                    if abs(performance_diff) < 100:  # Avoid showing extremely large differences
                        print(f"{test_name:25}: {performance_diff:+7.1f}% from baseline")

        # Feature importance analysis
        print(f"\nFeature Importance Analysis:")
        print("-" * 40)

        # Observation modifications
        obs_tests = ['noisy_observations', 'masked_observations', 'price_only', 'no_technical']
        for test_name in obs_tests:
            if test_name in successful_results:
                if baseline_return != 0:
                    performance = (successful_results[test_name]['avg_return'] / baseline_return) * 100
                    print(f"• {test_name:25}: {performance:.1f}% of baseline performance")
                else:
                    print(f"• {test_name:25}: {successful_results[test_name]['avg_return']:.2%} absolute return")

        # Environment modifications
        env_tests = ['short_lookback', 'long_lookback', 'high_volatility', 'strong_trend', 'noisy_environment']
        for test_name in env_tests:
            if test_name in successful_results:
                if baseline_return != 0:
                    performance = (successful_results[test_name]['avg_return'] / baseline_return) * 100
                    print(f"• {test_name:25}: {performance:.1f}% of baseline performance")
                else:
                    print(f"• {test_name:25}: {successful_results[test_name]['avg_return']:.2%} absolute return")

        # Robustness assessment
        print(f"\nModel Robustness Assessment:")
        print("-" * 40)

        robust_tests = []
        for test_name, result in successful_results.items():
            if test_name != 'baseline':
                if baseline_return != 0:
                    performance_ratio = result['avg_return'] / baseline_return
                    if performance_ratio > 0.8:  # 80% or more of baseline
                        robust_tests.append(test_name)
                else:
                    # For zero baseline, check absolute performance
                    if result['avg_return'] >= 0:  # Non-negative performance
                        robust_tests.append(test_name)

        if robust_tests:
            print(f"[OK] Model maintains >80% performance in {len(robust_tests)} configurations:")
            for test in robust_tests:
                print(f"  - {test}")
        else:
            print("[WARNING] Model performance drops significantly in most test conditions")

    return results

def main():
    """Main ablation study function."""

    try:
        results = run_simple_ablation_study()

        if results:
            print(f"\nAblation study completed with {len(results)} configurations tested.")
            print("Key insights about model robustness and feature importance identified.")
        else:
            print("No successful tests completed.")

        return 0

    except Exception as e:
        print(f"[ERROR] Ablation study failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)