#!/usr/bin/env python3
"""
Test basic package imports and functionality for OSS release.
"""

import sys
import traceback

def test_basic_imports():
    """Test that all core modules can be imported."""
    print("Testing basic package imports...")

    try:
        # Test core environments
        print("  - Testing environments...")
        from environments import SingleAssetTradingEnv
        from environments import PortfolioTradingEnv
        from environments import RegimeDetectionEnv
        from environments import MarketMakingEnv
        print("    ✓ Environments imported successfully")

        # Test data modules
        print("  - Testing data modules...")
        from data import DataManager
        from data import create_synthetic_data
        print("    ✓ Data modules imported successfully")

        # Test risk management
        print("  - Testing risk management...")
        from risk import RiskManager
        from risk import CVaRRewardShaper
        print("    ✓ Risk management imported successfully")

        # Test training modules
        print("  - Testing training modules...")
        from training import OnlineTrainer
        from training import OfflineTrainer
        print("    ✓ Training modules imported successfully")

        return True

    except ImportError as e:
        print(f"    ✗ Import error: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"    ✗ Unexpected error: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    print("\nTesting basic functionality...")

    try:
        # Test synthetic data generation
        print("  - Testing synthetic data generation...")
        data = create_synthetic_data(n_steps=100, n_assets=1)
        if data is not None and len(data) > 0:
            print(f"    ✓ Generated {len(data)} data points")
        else:
            print("    ✗ Failed to generate synthetic data")
            return False

        # Test environment creation
        print("  - Testing environment creation...")
        from environments import SingleAssetTradingEnv
        env = SingleAssetTradingEnv(data=data)
        print(f"    ✓ Created environment with observation space: {env.observation_space}")
        print(f"    ✓ Action space: {env.action_space}")

        # Test environment step
        print("  - Testing environment step...")
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        print(f"    ✓ Environment step successful. Reward: {reward:.4f}")

        return True

    except Exception as e:
        print(f"    ✗ Functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_package_structure():
    """Test that package structure is correct."""
    print("\nTesting package structure...")

    try:
        import financial_trading_gym
        print(f"  ✓ Package imported: {financial_trading_gym.__file__}")

        # Check version
        if hasattr(financial_trading_gym, '__version__'):
            print(f"  ✓ Version: {financial_trading_gym.__version__}")
        else:
            print("  - No version attribute found")

        return True

    except ImportError as e:
        print(f"  ✗ Package import failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Package structure test failed: {e}")
        return False

def main():
    """Run all package tests."""
    print("FINANCIAL TRADING GYM - PACKAGE TEST")
    print("=" * 50)

    tests_passed = 0
    total_tests = 3

    # Test 1: Basic imports
    if test_basic_imports():
        tests_passed += 1

    # Test 2: Basic functionality
    if test_basic_functionality():
        tests_passed += 1

    # Test 3: Package structure
    if test_package_structure():
        tests_passed += 1

    print("\n" + "=" * 50)
    print(f"TEST RESULTS: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("✓ All tests passed! Package is ready for OSS release.")
        return 0
    else:
        print("✗ Some tests failed. Please fix issues before release.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)