#!/usr/bin/env python3
"""
Meta-Learning Validation Tests

Practical tests to validate meta-learning capabilities in the 100k model
compared to the 50k model through adaptation speed and learning efficiency.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import our gym and training components
from environments import SingleAssetTradingEnv
from environments.base_env import AssetConfig
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class MetaLearningValidator:
    """Validates meta-learning capabilities through practical tests."""

    def __init__(self):
        self.models = {
            '50k': './models/qwen_final_model',
            '100k': './models/qwen_final_model_100k'
        }

        self.results = {}

    def create_challenging_test_environments(self):
        """Create challenging test environments to validate meta-learning."""

        environments = {}

        # Environment 1: Different Market Regime (High Volatility)
        np.random.seed(999)
        n_days = 126  # Shorter, more challenging
        prices = []
        current_price = 200.0  # Different price level

        for day in range(n_days):
            # High volatility regime
            vol_scale = 3.0  # Triple normal volatility
            trend_shift = 0.001 if day < 60 else -0.002  # Regime shift
            daily_return = np.random.normal(trend_shift, 0.02 * vol_scale)
            current_price *= (1 + daily_return)
            prices.append(max(current_price, 1.0))

        high_vol_asset = AssetConfig(
            symbol="TSLA",
            name="Tesla Inc.",
            sector="Technology",
            initial_price=prices[0],
            volatility=np.std(np.diff(prices) / prices[:-1])
        )

        environments['high_volatility'] = SingleAssetTradingEnv(
            asset=high_vol_asset,
            initial_cash=100000,
            max_episode_length=126,
            lookback_window=30,
            render_mode=None
        )

        # Environment 2: Different Asset Characteristics
        np.random.seed(777)
        prices = []
        current_price = 50.0  # Lower price stock

        for day in range(n_days):
            # Different price dynamics
            mean_reversion = 0.0001 * (100 - current_price) / 100  # Mean reversion
            daily_return = np.random.normal(mean_reversion, 0.015)
            current_price *= (1 + daily_return)
            prices.append(max(current_price, 1.0))

        mean_rev_asset = AssetConfig(
            symbol="MSFT",
            name="Microsoft Corp.",
            sector="Technology",
            initial_price=prices[0],
            volatility=np.std(np.diff(prices) / prices[:-1])
        )

        environments['mean_reverting'] = SingleAssetTradingEnv(
            asset=mean_rev_asset,
            initial_cash=100000,
            max_episode_length=126,
            lookback_window=30,
            render_mode=None
        )

        # Environment 3: Bear Market Conditions
        np.random.seed(555)
        prices = []
        current_price = 150.0

        for day in range(n_days):
            # Bear market with negative drift
            bear_drift = -0.003  # Consistent downward trend
            daily_return = np.random.normal(bear_drift, 0.025)
            current_price *= (1 + daily_return)
            prices.append(max(current_price, 1.0))

        bear_asset = AssetConfig(
            symbol="AMZN",
            name="Amazon.com Inc.",
            sector="Consumer Discretionary",
            initial_price=prices[0],
            volatility=np.std(np.diff(prices) / prices[:-1])
        )

        environments['bear_market'] = SingleAssetTradingEnv(
            asset=bear_asset,
            initial_cash=100000,
            max_episode_length=126,
            lookback_window=30,
            render_mode=None
        )

        return environments

    def test_adaptation_speed(self, env, env_name, num_episodes=5):
        """Test how quickly models adapt to new environments."""

        print(f"\n[ADAPTATION TEST] {env_name.upper()} ENVIRONMENT")
        print("-" * 60)

        adaptation_results = {}

        for model_name, model_path in self.models.items():
            if not os.path.exists(model_path + ".zip"):
                print(f"[SKIP] {model_name} model not found")
                continue

            print(f"\nTesting {model_name} model adaptation...")

            try:
                # Load model
                model = PPO.load(model_path)

                episode_rewards = []
                adaptation_speeds = []
                final_returns = []

                for episode in range(num_episodes):
                    obs = env.reset()
                    total_reward = 0
                    done = False
                    step = 0

                    # Track early performance (first 25 steps)
                    early_reward = 0
                    early_steps = min(25, 126)  # 25 steps or episode length

                    while not done and step < 126:
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
                        step += 1

                        if step <= early_steps:
                            early_reward += reward

                    # Calculate portfolio return
                    portfolio_return = (env.portfolio_value - 100000) / 100000

                    episode_rewards.append(total_reward)
                    final_returns.append(portfolio_return)

                    # Calculate adaptation speed (early vs total performance)
                    adaptation_speed = early_reward / (total_reward + 1e-8)
                    adaptation_speeds.append(adaptation_speed)

                    print(f"  Episode {episode + 1}: Total={total_reward:+6.4f}, "
                          f"Portfolio={portfolio_return:+6.2%}, "
                          f"Early={early_reward:+6.4f}")

                # Calculate metrics
                avg_reward = np.mean(episode_rewards)
                avg_return = np.mean(final_returns)
                avg_adaptation = np.mean(adaptation_speeds)
                reward_std = np.std(episode_rewards)

                adaptation_results[model_name] = {
                    'avg_reward': avg_reward,
                    'avg_return': avg_return,
                    'adaptation_speed': avg_adaptation,
                    'reward_stability': 1 / (reward_std + 1e-8),
                    'episode_rewards': episode_rewards,
                    'final_returns': final_returns
                }

                print(f"  {model_name} Summary:")
                print(f"    Average Reward: {avg_reward:+.4f}")
                print(f"    Average Return: {avg_return:+.2%}")
                print(f"    Adaptation Speed: {avg_adaptation:+.4f}")
                print(f"    Stability Score: {adaptation_results[model_name]['reward_stability']:.3f}")

            except Exception as e:
                print(f"  [ERROR] {model_name} test failed: {e}")
                adaptation_results[model_name] = None

        return adaptation_results

    def test_few_shot_learning(self, env, env_name, adaptation_episodes=3):
        """Test few-shot learning capabilities with minimal exposure."""

        print(f"\n[FEW-SHOT TEST] {env_name.upper()} ENVIRONMENT")
        print("-" * 60)

        few_shot_results = {}

        for model_name, model_path in self.models.items():
            if not os.path.exists(model_path + ".zip"):
                continue

            print(f"\nTesting {model_name} few-shot learning...")

            try:
                # Load model
                model = PPO.load(model_path)

                # Test with minimal exposure
                pre_adaptation_rewards = []
                post_adaptation_rewards = []

                # Pre-adaptation test (cold start)
                for episode in range(2):
                    obs = env.reset()
                    total_reward = 0
                    done = False
                    step = 0

                    while not done and step < 63:  # Half episode
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
                        step += 1

                    pre_adaptation_rewards.append(total_reward)

                # Minimal adaptation exposure
                print(f"  Running {adaptation_episodes} adaptation episodes...")
                for episode in range(adaptation_episodes):
                    obs = env.reset()
                    done = False
                    step = 0

                    while not done and step < 63:
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

                        step += 1

                # Post-adaptation test
                for episode in range(2):
                    obs = env.reset()
                    total_reward = 0
                    done = False
                    step = 0

                    while not done and step < 63:
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
                        step += 1

                    post_adaptation_rewards.append(total_reward)

                # Calculate learning efficiency
                pre_avg = np.mean(pre_adaptation_rewards)
                post_avg = np.mean(post_adaptation_rewards)
                learning_efficiency = (post_avg - pre_avg) / (abs(pre_avg) + 1e-8)

                few_shot_results[model_name] = {
                    'pre_adaptation': pre_avg,
                    'post_adaptation': post_avg,
                    'learning_efficiency': learning_efficiency,
                    'improvement': post_avg - pre_avg
                }

                print(f"  {model_name} Few-Shot Results:")
                print(f"    Pre-Adaptation: {pre_avg:+.4f}")
                print(f"    Post-Adaptation: {post_avg:+.4f}")
                print(f"    Learning Efficiency: {learning_efficiency:+.2%}")

            except Exception as e:
                print(f"  [ERROR] {model_name} few-shot test failed: {e}")
                few_shot_results[model_name] = None

        return few_shot_results

    def analyze_meta_learning_evidence(self, all_results):
        """Analyze evidence for meta-learning across all tests."""

        print("\n" + "=" * 80)
        print("META-LEARNING VALIDATION ANALYSIS")
        print("=" * 80)

        meta_learning_score = {}
        evidence_summary = {}

        for env_name in all_results:
            if 'adaptation' in all_results[env_name]:
                adaptation_results = all_results[env_name]['adaptation']

                # Compare 50k vs 100k adaptation
                if '50k' in adaptation_results and '100k' in adaptation_results:
                    results_50k = adaptation_results['50k']
                    results_100k = adaptation_results['100k']

                    # Calculate meta-learning indicators
                    speed_improvement = (results_100k['adaptation_speed'] -
                                       results_50k['adaptation_speed']) / (abs(results_50k['adaptation_speed']) + 1e-8)

                    return_improvement = (results_100k['avg_return'] -
                                        results_50k['avg_return']) / (abs(results_50k['avg_return']) + 1e-8)

                    stability_improvement = (results_100k['reward_stability'] -
                                           results_50k['reward_stability']) / (abs(results_50k['reward_stability']) + 1e-8)

                    meta_learning_score[env_name] = (speed_improvement + return_improvement + stability_improvement) / 3

                    evidence_summary[env_name] = {
                        'speed_improvement': speed_improvement,
                        'return_improvement': return_improvement,
                        'stability_improvement': stability_improvement,
                        'meta_learning_score': meta_learning_score[env_name]
                    }

        # Print comprehensive analysis
        print("\nMETA-LEARNING EVIDENCE BY ENVIRONMENT:")
        for env_name, evidence in evidence_summary.items():
            print(f"\n{env_name.upper()}:")
            print(f"  Adaptation Speed Improvement: {evidence['speed_improvement']:+.2%}")
            print(f"  Return Improvement: {evidence['return_improvement']:+.2%}")
            print(f"  Stability Improvement: {evidence['stability_improvement']:+.2%}")
            print(f"  Overall Meta-Learning Score: {evidence['meta_learning_score']:+.2%}")

        # Overall conclusion
        if meta_learning_score:
            avg_score = np.mean(list(meta_learning_score.values()))
            print(f"\nOVERALL META-LEARNING ASSESSMENT:")
            print(f"Average Meta-Learning Score: {avg_score:+.2%}")

            if avg_score > 0.1:  # 10% improvement threshold
                print("CONCLUSION: STRONG EVIDENCE of meta-learning capabilities")
                print("The 100k model demonstrates significantly superior adaptation")
                print("and learning efficiency compared to the 50k model.")
            elif avg_score > 0.05:  # 5% improvement threshold
                print("CONCLUSION: MODERATE EVIDENCE of meta-learning capabilities")
                print("The 100k model shows improvement but evidence is not conclusive.")
            else:
                print("CONCLUSION: LIMITED EVIDENCE of meta-learning capabilities")
                print("Differences between models may be due to extended training.")

        return evidence_summary

    def run_comprehensive_validation(self):
        """Run all meta-learning validation tests."""

        print("META-LEARNING VALIDATION TEST SUITE")
        print("Comparing 50k vs 100k Model Adaptation Capabilities")
        print("=" * 80)

        # Create test environments
        print("\n[SETUP] Creating challenging test environments...")
        test_environments = self.create_challenging_test_environments()
        print(f"[OK] Created {len(test_environments)} test environments")

        all_results = {}

        # Run tests for each environment
        for env_name, env in test_environments.items():
            print(f"\n{'='*20} TESTING {env_name.upper()} {'='*20}")

            # Test 1: Adaptation Speed
            adaptation_results = self.test_adaptation_speed(env, env_name)

            # Test 2: Few-Shot Learning
            few_shot_results = self.test_few_shot_learning(env, env_name)

            all_results[env_name] = {
                'adaptation': adaptation_results,
                'few_shot': few_shot_results
            }

        # Analyze meta-learning evidence
        evidence = self.analyze_meta_learning_evidence(all_results)

        return all_results, evidence

def main():
    """Main validation function."""

    print("META-LEARNING VALIDATION")
    print("Practical tests to validate meta-learning emergence in 100k model")
    print()

    try:
        validator = MetaLearningValidator()
        results, evidence = validator.run_comprehensive_validation()

        print(f"\nValidation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return 0

    except Exception as e:
        print(f"[ERROR] Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)