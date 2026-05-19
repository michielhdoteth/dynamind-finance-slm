#!/usr/bin/env python3
"""
Simple test for basic package functionality.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

def test_basic_imports():
    """Test that core modules can be imported."""
    print("Testing basic imports...")

    try:
        # Test basic imports
        import environments
        import data
        import risk
        import training
        print("  - Core modules imported successfully")
        return True

    except ImportError as e:
        print(f"  - Import error: {e}")
        return False

def test_simple_environment():
    """Test creating a simple environment."""
    print("Testing environment creation...")

    try:
        from environments.single_asset import SingleAssetTradingEnv
        import numpy as np

        # Create simple price data
        prices = np.random.randn(100).cumsum() + 100
        volumes = np.random.randint(1000, 10000, 100)

        data = {
            'price': prices,
            'volume': volumes
        }

        env = SingleAssetTradingEnv(data=data)
        print(f"  - Environment created successfully")
        print(f"  - Observation space: {env.observation_space}")
        print(f"  - Action space: {env.action_space}")

        # Test one step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  - Environment step successful. Reward: {reward:.4f}")

        return True

    except Exception as e:
        print(f"  - Environment test failed: {e}")
        return False

def main():
    """Run simple tests."""
    print("SIMPLE PACKAGE TEST")
    print("=" * 40)

    tests_passed = 0
    total_tests = 2

    if test_basic_imports():
        tests_passed += 1

    if test_simple_environment():
        tests_passed += 1

    print("\n" + "=" * 40)
    print(f"RESULTS: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("SUCCESS: Package basic functionality working!")
        return 0
    else:
        print("FAILURE: Some tests failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)