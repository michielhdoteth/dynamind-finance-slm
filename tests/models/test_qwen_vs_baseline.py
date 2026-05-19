#!/usr/bin/env python3
"""
Test Trained Qwen RL Model vs Baseline

Compare the performance of our trained Qwen RL model against baseline strategies
on the financial trading environment.
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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import our gym and training components
from environments import SingleAssetTradingEnv
from environments.base_env import AssetConfig
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

class BaselineAgent:
    """Simple baseline trading strategies for comparison."""

    def __init__(self, strategy="buy_and_hold"):
        self.strategy = strategy
        self.position = 0  # -1: short, 0: neutral, 1: long

    def act(self, observation):
        """Return action based on strategy."""
        if self.strategy == "buy_and_hold":
            return 1  # Always buy/hold
        elif self.strategy == "random":
            return np.random.choice([0, 1, 2])  # Random action
        elif self.strategy == "momentum_simple":
            # Simple momentum based on recent price changes
            if len(observation) >= 10:
                price_trend = np.mean(observation[-5:]) - np.mean(observation[-10:-5])
                if price_trend > 0:
                    return 1  # Buy
                elif price_trend < 0:
                    return 0  # Sell
                else:
                    return 2  # Hold
            else:
                return 2  # Hold by default
        else:
            return 2  # Default to hold

def create_test_environment():
    """Create a test environment with different data than training."""

    # Create test asset data (different seed for variety)
    np.random.seed(123)  # Different seed from training
    n_days = 252  # One year of trading days
    prices = []
    current_price = 155.0  # Different starting price

    for day in range(n_days):
        # Simulate realistic price movements with trend
        trend = 0.0003  # Slight upward trend
        daily_return = np.random.normal(trend, 0.025)  # 2.5% volatility
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

    print(f"[TEST] Created test asset: {asset.symbol}")
    print(f"  - Price history: {len(prices)} days")
    print(f"  - Price range: ${min(prices):.2f} - ${max(prices):.2f}")
    print(f"  - Volatility: {asset.volatility:.4f}")

    # Create environment
    env = SingleAssetTradingEnv(
        asset=asset,
        initial_cash=100000,
        max_episode_length=252,
        lookback_window=30,
        render_mode=None
    )

    return env

def test_agent(env, agent, agent_name, num_episodes=5):
    """Test an agent and return performance metrics."""

    print(f"\n[TEST] Testing {agent_name}...")

    episode_rewards = []
    episode_returns = []
    episode_lengths = []
    final_portfolios = []

    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        step = 0

        while not done:
            if isinstance(agent, PPO):
                # For trained RL model - handle different environment API formats
                if isinstance(obs, tuple):
                    # New Gym API returns (obs, info)
                    obs_array = obs[0]
                else:
                    # Old API or vectorized
                    obs_array = obs

                # Ensure obs is properly shaped for the model
                if len(obs_array.shape) == 1:
                    obs_array = obs_array.reshape(1, -1)

                action, _ = agent.predict(obs_array, deterministic=True)
                action = action[0]  # Extract single action
            else:
                # For baseline agents
                obs_to_use = obs[0] if isinstance(obs, tuple) else obs
                action = agent.act(obs_to_use)

            step_result = env.step(action)

            # Handle different return formats
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

        # Calculate portfolio return
        initial_value = 100000
        final_value = env.portfolio_value
        portfolio_return = (final_value - initial_value) / initial_value

        episode_rewards.append(total_reward)
        episode_returns.append(portfolio_return)
        episode_lengths.append(step)
        final_portfolios.append(final_value)

        print(f"  Episode {episode + 1}: Reward={total_reward:.4f}, Return={portfolio_return:.2%}, Final Value=${final_value:.2f}")

    # Calculate statistics
    avg_reward = np.mean(episode_rewards)
    avg_return = np.mean(episode_returns)
    avg_length = np.mean(episode_lengths)
    avg_portfolio = np.mean(final_portfolios)

    # Calculate Sharpe ratio (simplified)
    returns_std = np.std(episode_returns)
    sharpe_ratio = avg_return / returns_std if returns_std > 0 else 0

    # Calculate win rate
    win_rate = sum(1 for r in episode_returns if r > 0) / len(episode_returns)

    results = {
        'agent_name': agent_name,
        'avg_reward': avg_reward,
        'avg_return': avg_return,
        'avg_portfolio_value': avg_portfolio,
        'avg_episode_length': avg_length,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'episode_rewards': episode_rewards,
        'episode_returns': episode_returns,
        'final_portfolios': final_portfolios
    }

    print(f"[OK] {agent_name} Results:")
    print(f"  - Average Reward: {avg_reward:.4f}")
    print(f"  - Average Return: {avg_return:.2%}")
    print(f"  - Average Portfolio Value: ${avg_portfolio:.2f}")
    print(f"  - Sharpe Ratio: {sharpe_ratio:.3f}")
    print(f"  - Win Rate: {win_rate:.2%}")

    return results

def compare_agents(qwen_results, baseline_results):
    """Compare and display results between agents."""

    print("\n" + "=" * 80)
    print("AGENT COMPARISON RESULTS")
    print("=" * 80)

    print(f"\nPerformance Metrics Comparison:")
    print("-" * 50)

    metrics = [
        ('Average Return', 'avg_return', '{:.2%}'),
        ('Average Portfolio Value', 'avg_portfolio_value', '${:,.2f}'),
        ('Sharpe Ratio', 'sharpe_ratio', '{:.3f}'),
        ('Win Rate', 'win_rate', '{:.2%}'),
        ('Average Episode Length', 'avg_episode_length', '{:.1f}')
    ]

    for metric_name, key, format_str in metrics:
        qwen_val = qwen_results[key]
        baseline_val = baseline_results[key]

        print(f"{metric_name:25} | Qwen RL: {format_str.format(qwen_val):10} | Baseline: {format_str.format(baseline_val):10}")

        # Calculate improvement
        if key == 'avg_episode_length':
            # Lower is better for episode length
            improvement = (baseline_val - qwen_val) / baseline_val * 100
            improvement_str = f"({improvement:+.1f}%)"
        else:
            # Higher is better for other metrics
            improvement = (qwen_val - baseline_val) / baseline_val * 100
            improvement_str = f"({improvement:+.1f}%)"

        print(f"{'':25} | Improvement: {improvement_str:>10}")
        print("-" * 50)

    # Determine winner
    qwen_score = 0
    baseline_score = 0

    # Simple scoring based on key metrics
    if qwen_results['avg_return'] > baseline_results['avg_return']:
        qwen_score += 1
    else:
        baseline_score += 1

    if qwen_results['sharpe_ratio'] > baseline_results['sharpe_ratio']:
        qwen_score += 1
    else:
        baseline_score += 1

    if qwen_results['win_rate'] > baseline_results['win_rate']:
        qwen_score += 1
    else:
        baseline_score += 1

    print(f"\nOverall Winner: {'Qwen RL Agent' if qwen_score > baseline_score else 'Baseline Agent'}")
    print(f"Score: Qwen RL {qwen_score} - {baseline_score} Baseline")

    return qwen_score > baseline_score

def main():
    """Main testing function."""

    print("QWEN RL MODEL VS BASELINE COMPARISON")
    print("Testing trained reinforcement learning model against baseline strategies")
    print()

    try:
        # Step 1: Create test environment
        print("[STEP 1] Creating test environment...")
        test_env = create_test_environment()

        # Step 2: Load trained Qwen model
        print("[STEP 2] Loading trained Qwen RL model...")

        model_path = "./models/qwen_final_model"
        if os.path.exists(model_path + ".zip"):
            qwen_model = PPO.load(model_path)
            print(f"[OK] Loaded Qwen model from {model_path}")
        else:
            print(f"[ERROR] Trained model not found at {model_path}")
            return 1

        # Step 3: Create baseline agent
        print("[STEP 3] Creating baseline agent...")
        baseline_agent = BaselineAgent(strategy="buy_and_hold")
        print(f"[OK] Created baseline agent: {baseline_agent.strategy}")

        # Step 4: Test Qwen model
        print("[STEP 4] Testing Qwen RL model...")
        qwen_results = test_agent(test_env, qwen_model, "Qwen RL Model", num_episodes=5)

        # Step 5: Test baseline
        print("[STEP 5] Testing baseline agent...")
        baseline_results = test_agent(test_env, baseline_agent, "Buy and Hold Baseline", num_episodes=5)

        # Step 6: Compare results
        print("[STEP 6] Comparing results...")
        qwen_wins = compare_agents(qwen_results, baseline_results)

        # Step 7: Summary
        print("\n" + "=" * 80)
        print("TESTING SUMMARY")
        print("=" * 80)

        if qwen_wins:
            print("[SUCCESS] Qwen RL model outperformed baseline!")
            print("The trained reinforcement learning agent showed superior performance")
            print("compared to the buy-and-hold strategy on the test environment.")
        else:
            print("[INFO] Baseline performed competitively")
            print("The buy-and-hold strategy showed comparable or better performance")
            print("in this particular test scenario.")

        print(f"\nKey Insights:")
        print(f"- Qwen RL achieved {qwen_results['avg_return']:.2%} average return")
        print(f"- Baseline achieved {baseline_results['avg_return']:.2%} average return")
        print(f"- Performance difference: {abs(qwen_results['avg_return'] - baseline_results['avg_return']):.2%}")

        return 0

    except Exception as e:
        print(f"[ERROR] Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)