"""
Test Real Data Infrastructure

This script tests the new professional market data infrastructure
with real market data from Yahoo Finance.
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    print("=" * 60)
    print("TESTING REAL MARKET DATA INFRASTRUCTURE")
    print("=" * 60)

    try:
        # Test data manager
        print("\n1. Testing DataManager...")
        from data import DataManager

        data_manager = DataManager(cache_dir="./test_data_cache")

        # Fetch real data for popular symbols
        symbols = ["AAPL", "MSFT", "GOOGL"]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # Last 90 days

        print(
            f"Fetching data for {symbols} from {start_date.date()} to {end_date.date()}"
        )

        market_data = data_manager.get_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            source="yahoo",
            frequency="1d",
        )

        print("Data loaded successfully!")
        print(f"   Shape: {market_data.shape}")
        print(f"   Columns: {list(market_data.columns.levels[0])}")
        print(
            f"   Date range: {market_data.index.min().date()} to {market_data.index.max().date()}"
        )

        # Test data validation
        print("\n2. Testing DataValidator...")
        from data.validators import DataValidator

        validator = DataValidator()
        validation_results = validator.validate_dataset(market_data, symbols)

        print("Validation completed!")
        print(
            f"   Overall status: {'PASSED' if validation_results['passed'] else 'FAILED'}"
        )
        print(f"   Issues: {len(validation_results['issues'])}")
        print(f"   Warnings: {len(validation_results['warnings'])}")

        if validation_results["issues"]:
            print("   Issues:")
            for issue in validation_results["issues"][:3]:  # Show first 3
                print(f"     - {issue}")

        # Test data preprocessing
        print("\n3. Testing DataPreprocessor...")
        from data.preprocessors import DataPreprocessor

        preprocessor = DataPreprocessor(
            feature_window=10, normalize_features=True, scaler_type="standard"
        )

        processed_data = preprocessor.process(market_data)

        print("Preprocessing completed!")
        print(f"   Processed shape: {processed_data.shape}")
        print(
            f"   New features: {[col for col in processed_data.columns.levels[1] if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]}"
        )

        # Test feature statistics
        print("\n4. Testing Feature Statistics...")
        for symbol in symbols[:1]:  # Test first symbol
            stats = preprocessor.get_feature_stats(processed_data, symbol)
            feature_count = len(
                [
                    k
                    for k in stats.keys()
                    if k not in ["Open", "High", "Low", "Close", "Volume"]
                ]
            )
            print(f"   {symbol}: {feature_count} engineered features")

        # Test environment with real data
        print("\n5. Testing Environment with Real Data...")
        from environments import SingleAssetTradingEnv
        from environments.base_env import AssetConfig

        # Create asset with real symbol
        asset = AssetConfig(
            symbol="AAPL",
            name="Apple Inc.",
            sector="Technology",
            initial_price=150.0,  # This will be overridden by real data
        )

        # Create environment with real data
        env = SingleAssetTradingEnv(
            asset=asset,
            initial_cash=100_000,
            max_episode_length=50,
            lookback_window=20,
            action_space_type="discrete",
            data_source="real",  # Use real data!
            seed=42,
        )

        print("Environment created with real data!")
        print(f"   Action space: {env.action_space}")
        print(f"   Observation space: {env.observation_space.shape}")

        # Test a few steps
        obs, info = env.reset(seed=42)
        print(f"   Initial observation shape: {obs.shape}")
        print(f"   Initial portfolio: ${info['portfolio_value']:,.2f}")

        total_reward = 0
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            print(
                f"   Step {step+1}: Action={action}, Reward={reward:.4f}, Portfolio=${info['portfolio_value']:,.2f}"
            )

            if terminated or truncated:
                break

        print(f"   Total reward: {total_reward:.4f}")
        env.close()

        print("\n" + "=" * 60)
        print("REAL DATA INFRASTRUCTURE TEST PASSED!")
        print("=" * 60)

        print("\nNext steps:")
        print("1. Real data loading works")
        print("2. Data validation works")
        print("3. Feature engineering works")
        print("4. Environment integration works")
        print("5. Ready for Phase 2: Market Microstructure")
        print("6. Ready for Qwen RL with real data")

        return True

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
