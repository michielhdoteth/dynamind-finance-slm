"""
Test Script for Financial Trading Gym

This script performs basic tests to verify that all environments
are working correctly and can be imported and used.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Add the package to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")

    try:
        from data.synthetic import MarketDataGenerator
        from environments import (
            MarketMakingEnv,
            PortfolioOptimizationEnv,
            RegimeDetectionEnv,
            SingleAssetTradingEnv,
        )
        from environments.base_env import AssetConfig, RiskConstraints, TransactionCosts

        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_single_asset_env():
    """Test single asset trading environment"""
    print("\nTesting Single Asset Trading Environment...")

    try:
        from environments import SingleAssetTradingEnv
        from environments.base_env import AssetConfig

        # Create test asset
        asset = AssetConfig(
            symbol="TEST",
            name="Test Asset",
            sector="Technology",
            initial_price=100.0,
            volatility=0.02,
            drift=0.0001,
        )

        # Create environment
        env = SingleAssetTradingEnv(
            asset=asset,
            initial_cash=100_000,
            max_episode_length=50,
            lookback_window=10,
            seed=42,
        )

        # Test reset
        obs, info = env.reset(seed=42)
        print(
            f"  Observation shape: {obs.shape if hasattr(obs, 'shape') else len(obs)}"
        )
        print(f"  Initial portfolio: ${info['portfolio_value']:,.2f}")

        # Test a few steps
        total_reward = 0
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        print(f"  Completed {step+1} steps")
        print(f"  Total reward: {total_reward:.4f}")
        print(f"  Final portfolio: ${info['portfolio_value']:,.2f}")

        env.close()
        print("✅ Single asset environment test passed")
        return True

    except Exception as e:
        print(f"❌ Single asset environment test failed: {e}")
        return False


def test_portfolio_env():
    """Test portfolio optimization environment"""
    print("\nTesting Portfolio Optimization Environment...")

    try:
        from environments import PortfolioOptimizationEnv
        from environments.base_env import AssetConfig

        # Create test assets
        assets = [
            AssetConfig(
                symbol="STOCK1", name="Stock 1", sector="Tech", initial_price=100.0
            ),
            AssetConfig(
                symbol="STOCK2", name="Stock 2", sector="Finance", initial_price=50.0
            ),
            AssetConfig(
                symbol="STOCK3", name="Stock 3", sector="Healthcare", initial_price=75.0
            ),
        ]

        # Create environment
        env = PortfolioOptimizationEnv(
            assets=assets,
            initial_cash=100_000,
            max_episode_length=50,
            lookback_window=20,
            seed=42,
        )

        # Test reset
        obs, info = env.reset(seed=42)
        print(f"  Observation shape: {obs.shape}")
        print(f"  Number of assets: {len(assets)}")

        # Test a few steps
        total_reward = 0
        for step in range(10):
            action = np.random.random(len(assets))
            action = action / np.sum(action)  # Normalize

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        print(f"  Completed {step+1} steps")
        print(f"  Total reward: {total_reward:.4f}")

        env.close()
        print("✅ Portfolio environment test passed")
        return True

    except Exception as e:
        print(f"❌ Portfolio environment test failed: {e}")
        return False


def test_regime_detection_env():
    """Test regime detection environment"""
    print("\nTesting Regime Detection Environment...")

    try:
        from environments import RegimeDetectionEnv
        from environments.base_env import AssetConfig

        # Create test assets
        assets = [
            AssetConfig(symbol="REGIME1", name="Regime Asset 1", sector="Tech"),
            AssetConfig(symbol="REGIME2", name="Regime Asset 2", sector="Finance"),
        ]

        # Create environment
        env = RegimeDetectionEnv(
            assets=assets,
            initial_cash=100_000,
            max_episode_length=50,
            lookback_window=20,
            seed=42,
        )

        # Test reset
        obs, info = env.reset(seed=42)
        print(f"  Observation shape: {obs.shape}")
        print(f"  Number of regimes: {len(env.config.regime_types)}")

        # Test a few steps
        total_reward = 0
        correct_predictions = 0

        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if step > 0:
                # Check if prediction was correct (simplified)
                correct_predictions += 1  # Placeholder

            if terminated or truncated:
                break

        print(f"  Completed {step+1} steps")
        print(f"  Total reward: {total_reward:.4f}")

        env.close()
        print("✅ Regime detection environment test passed")
        return True

    except Exception as e:
        print(f"❌ Regime detection environment test failed: {e}")
        return False


def test_market_making_env():
    """Test market making environment"""
    print("\nTesting Market Making Environment...")

    try:
        from environments import MarketMakingEnv
        from environments.base_env import AssetConfig

        # Create test asset
        asset = AssetConfig(
            symbol="MM_TEST",
            name="Market Making Test",
            sector="Technology",
            initial_price=100.0,
            volatility=0.02,
        )

        # Create environment
        env = MarketMakingEnv(
            asset=asset,
            initial_cash=50_000,
            max_episode_length=50,
            lookback_window=20,
            seed=42,
        )

        # Test reset
        obs, info = env.reset(seed=42)
        print(f"  Observation shape: {obs.shape}")
        print(f"  Initial cash: ${info['cash_balance']:,.2f}")

        # Test a few steps
        total_reward = 0
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        print(f"  Completed {step+1} steps")
        print(f"  Total reward: {total_reward:.4f}")
        print(f"  Final inventory: {env.inventory}")

        env.close()
        print("✅ Market making environment test passed")
        return True

    except Exception as e:
        print(f"❌ Market making environment test failed: {e}")
        return False


def test_data_generation():
    """Test synthetic data generation"""
    print("\nTesting Synthetic Data Generation...")

    try:
        from data.synthetic import (
            MarketDataGenerator,
            RegimeParameters,
            SyntheticDataConfig,
        )
        from environments.base_env import AssetConfig

        # Create test configuration
        config = SyntheticDataConfig(n_assets=3, n_steps=100, frequency="daily")

        # Create assets
        assets = [
            AssetConfig(symbol="DATA1", name="Data Test 1", sector="Tech"),
            AssetConfig(symbol="DATA2", name="Data Test 2", sector="Finance"),
            AssetConfig(symbol="DATA3", name="Data Test 3", sector="Healthcare"),
        ]

        # Generate data
        generator = MarketDataGenerator(config)
        data = generator.generate_market_data(assets)

        print(f"  Generated data shape: {data.shape}")
        print(f"  Data columns: {list(data.columns[:10])}...")  # Show first 10 columns

        # Check data integrity
        assert len(data) == config.n_steps, "Data length mismatch"
        assert all(
            f"{asset.symbol}_close" in data.columns for asset in assets
        ), "Missing price columns"

        print("✅ Synthetic data generation test passed")
        return True

    except Exception as e:
        print(f"❌ Synthetic data generation test failed: {e}")
        return False


def test_gym_registration():
    """Test gymnasium registration"""
    print("\nTesting Gymnasium Registration...")

    try:
        import financial_trading_gym

        # Test environment registration
        env_specs = [
            "FinancialTrading-SingleAsset-v0",
            "FinancialTrading-Portfolio-v0",
            "FinancialTrading-RegimeDetection-v0",
            "FinancialTrading-MarketMaking-v0",
        ]

        for spec in env_specs:
            try:
                env = gym.make(spec)
                print(f"  ✅ {spec} registered successfully")
                env.close()
            except Exception as e:
                print(f"  ❌ {spec} registration failed: {e}")
                return False

        print("✅ Gymnasium registration test passed")
        return True

    except Exception as e:
        print(f"❌ Gymnasium registration test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results"""
    print("=" * 60)
    print("FINANCIAL TRADING GYM - TEST SUITE")
    print("=" * 60)

    tests = [
        ("Import Test", test_imports),
        ("Single Asset Environment", test_single_asset_env),
        ("Portfolio Environment", test_portfolio_env),
        ("Regime Detection Environment", test_regime_detection_env),
        ("Market Making Environment", test_market_making_env),
        ("Data Generation", test_data_generation),
        ("Gym Registration", test_gym_registration),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! The gym is ready to use.")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
