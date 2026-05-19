"""
Quick Test Script - Test Gym Only

This script just tests the financial trading gym without loading Qwen.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 50)
    print("QUICK GYM TEST")
    print("=" * 50)

    try:
        # Import environments
        from environments import SingleAssetTradingEnv
        from environments.base_env import AssetConfig

        # Create test asset
        asset = AssetConfig(
            symbol="TEST",
            name="Test Asset",
            sector="Technology",
            initial_price=100.0
        )

        # Create environment
        env = SingleAssetTradingEnv(
            asset=asset,
            initial_cash=10000,
            max_episode_length=20,
            seed=42
        )

        print("Environment created successfully!")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space.shape}")

        # Test reset and few steps
        obs, info = env.reset(seed=42)
        print(f"Initial observation shape: {obs.shape}")
        print(f"Initial portfolio: ${info['portfolio_value']:,.2f}")

        total_reward = 0
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            print(f"Step {step+1}: Action={action}, Reward={reward:.4f}, "
                  f"Portfolio=${info['portfolio_value']:,.2f}")

            if terminated or truncated:
                break

        print(f"\nEpisode completed!")
        print(f"Total reward: {total_reward:.4f}")
        print(f"Final portfolio: ${info['portfolio_value']:,.2f}")

        # Test multiple environments
        print("\nTesting multiple random episodes...")
        rewards = []

        for episode in range(5):
            obs, info = env.reset(seed=episode)
            episode_reward = 0
            done = False
            steps = 0

            while not done and steps < 15:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1
                done = terminated or truncated

            rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.4f}, Steps = {steps}")

        print(f"\nResults over {len(rewards)} episodes:")
        print(f"Mean reward: {sum(rewards) / len(rewards):.4f}")
        print(f"Max reward: {max(rewards):.4f}")
        print(f"Min reward: {min(rewards):.4f}")

        env.close()

        print("\n" + "=" * 50)
        print("GYM TEST PASSED SUCCESSFULLY!")
        print("=" * 50)

        print("\nThe financial trading gym is working correctly!")
        print("You can now:")
        print("1. Use it with standard RL algorithms")
        print("2. Train Qwen on it when ready")
        print("3. Develop custom trading strategies")

        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)