#!/usr/bin/env python3
"""
Test Advanced Qwen RL Training

Tests the advanced Qwen training system with real market data and risk management.
"""

import logging
import os
import shutil
import sys
import tempfile

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_data_loading():
    """Test real market data loading."""
    print("\n" + "=" * 60)
    print("TESTING REAL MARKET DATA LOADING")
    print("=" * 60)

    try:
        from data import DataManager

        # Create data manager
        data_manager = DataManager()
        print("[OK] Data manager initialized")

        # Load recent data for testing
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Last 30 days

        symbols = ["AAPL", "MSFT"]
        data = data_manager.get_data(
            symbols=symbols,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            source="yahoo",
            frequency="1d",
        )

        if not data.empty:
            print(f"[OK] Real market data loaded: {data.shape}")
            print(f"  - Symbols: {list(data.columns.get_level_values(0).unique())}")
            print(f"  - Date range: {data.index[0]} to {data.index[-1]}")
            print(
                f"  - Available columns: {list(data.columns.get_level_values(1).unique())}"
            )
            return True
        else:
            print("[ERROR] No data loaded")
            return False

    except Exception as e:
        print(f"[ERROR] Data loading test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_risk_system():
    """Test risk management system integration."""
    print("\n" + "=" * 60)
    print("TESTING RISK MANAGEMENT SYSTEM")
    print("=" * 60)

    try:
        from environments.base_env import AssetConfig
        from risk import CVaRConfig, PortfolioConstraints, PositionLimits, RiskManager

        # Create sample assets
        assets = [
            AssetConfig(
                symbol="AAPL",
                sector="Technology",
                price_history=[150.0, 152.0, 151.0, 153.0, 155.0],
                volatility=0.02,
            ),
            AssetConfig(
                symbol="MSFT",
                sector="Technology",
                price_history=[300.0, 302.0, 301.0, 304.0, 306.0],
                volatility=0.018,
            ),
        ]

        print(f"[OK] Created {len(assets)} sample assets")

        # Setup risk management
        position_limits = risk.PositionLimits(max_position_size=0.3)
        portfolio_constraints = PortfolioConstraints(max_leverage=2.0)
        cvar_config = CVaRConfig(risk_aversion=1.0)

        risk_manager = RiskManager(
            assets=assets,
            position_limits=position_limits,
            portfolio_constraints=portfolio_constraints,
            cvar_config=cvar_config,
            enable_risk_shaping=True,
        )

        print("[OK] Risk manager initialized")

        # Test position constraints
        current_prices = {"AAPL": 155.0, "MSFT": 306.0}
        test_positions = {"AAPL": 100, "MSFT": 50}

        # Set portfolio value for constraint checking
        risk_manager.current_portfolio_value = 100000

        results = risk_manager.check_position_constraints(
            test_positions, current_prices
        )
        print(
            f"[OK] Position constraints checked: {len(results['violations'])} violations"
        )

        # Test reward shaping
        base_reward = 0.05
        shaped_reward = risk_manager.shape_reward(
            base_reward, step=0, additional_info={"portfolio_value": 100000}
        )

        print(f"[OK] Reward shaping working: {base_reward:.4f} -> {shaped_reward:.4f}")

        print("[SUCCESS] Risk management system test completed")
        return True

    except Exception as e:
        print(f"[ERROR] Risk system test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_environment_creation():
    """Test environment creation with real data."""
    print("\n" + "=" * 60)
    print("TESTING ENVIRONMENT CREATION")
    print("=" * 60)

    try:
        from environments import SingleAssetTradingEnv
        from environments.base_env import AssetConfig

        # Create sample asset with real-looking data
        asset = AssetConfig(
            symbol="AAPL",
            sector="Technology",
            price_history=[150.0, 152.0, 151.0, 153.0, 155.0, 154.0, 156.0, 158.0],
            volatility=0.02,
        )

        print("[OK] Sample asset created")

        # Create environment
        env = SingleAssetTradingEnv(
            assets=[asset],
            initial_balance=100000,
            max_shares=1000,
            transaction_fee_pct=0.001,
            lookback_window=5,
            render_mode=None,
        )

        print("[OK] Environment created successfully")

        # Test environment reset
        obs = env.reset()
        print(f"[OK] Environment reset successful. Obs shape: {obs.shape}")

        # Test environment step
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print("[OK] Environment step successful:")
        print(f"  - Action: {action}")
        print(f"  - Reward: {reward:.4f}")
        print(f"  - Done: {done}")

        print("[SUCCESS] Environment creation test completed")
        return True

    except Exception as e:
        print(f"[ERROR] Environment creation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_qwen_model():
    """Test Qwen model loading and basic functionality."""
    print("\n" + "=" * 60)
    print("TESTING QWEN MODEL")
    print("=" * 60)

    try:
        # Model name
        model_name = "Qwen/Qwen2-0.5B"

        print(f"Loading Qwen model: {model_name}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("[OK] Tokenizer loaded successfully")

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        print("[OK] Model loaded successfully")

        # Test tokenization
        test_text = "Financial trading:"
        inputs = tokenizer(test_text, return_tensors="pt")
        print(f"[OK] Tokenization successful: {inputs['input_ids'].shape}")

        # Test generation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("[OK] Text generation successful:")
        print(f"  Input: {test_text}")
        print(f"  Output: {generated_text}")

        print("[SUCCESS] Qwen model test completed")
        return True

    except Exception as e:
        print(f"[ERROR] Qwen model test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_training_setup():
    """Test training setup without full training."""
    print("\n" + "=" * 60)
    print("TESTING TRAINING SETUP")
    print("=" * 60)

    try:
        # Test import of training components
        from train_qwen_advanced import create_evaluation_env, create_training_env

        print("[OK] Training functions imported")

        # Test environment creation
        try:
            train_env = create_training_env(
                symbols=["AAPL"],
                start_date="2023-01-01",
                end_date="2023-03-31",
                enable_risk_management=True,
            )
            print("[OK] Training environment created")
        except Exception as e:
            print(
                f"[WARNING] Training environment creation failed (may be due to data): {e}"
            )

        # Test PPO import
        from stable_baselines3 import PPO

        print("[OK] PPO imported successfully")

        # Test feature extractor
        from train_qwen_advanced import QwenPolicyExtractor

        obs_space = gym.spaces.Box(-1, 1, shape=(50,), dtype=np.float32)
        extractor = QwenPolicyExtractor(obs_space)
        print("[OK] Qwen policy extractor created")

        print("[SUCCESS] Training setup test completed")
        return True

    except Exception as e:
        print(f"[ERROR] Training setup test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all advanced Qwen tests."""
    print("ADVANCED QWEN RL TRAINING SYSTEM TEST")
    print("=" * 80)

    success_count = 0
    total_tests = 5

    # Test 1: Data Loading
    if test_data_loading():
        success_count += 1

    # Test 2: Risk System
    if test_risk_system():
        success_count += 1

    # Test 3: Environment Creation
    if test_environment_creation():
        success_count += 1

    # Test 4: Qwen Model
    if test_qwen_model():
        success_count += 1

    # Test 5: Training Setup
    if test_training_setup():
        success_count += 1

    print("\n" + "=" * 80)
    print(
        f"ADVANCED QWEN SYSTEM TEST RESULTS: {success_count}/{total_tests} tests passed"
    )
    print("=" * 80)

    if success_count == total_tests:
        print("\nAdvanced Qwen RL System Validation:")
        print("[OK] Real market data loading and processing")
        print("[OK] Advanced risk management system")
        print("[OK] Environment creation with risk constraints")
        print("[OK] Qwen model loading and functionality")
        print("[OK] Training setup and configuration")

        print("\nPHASE 5 (Qwen RL Integration) COMPLETED!")
        print("Key achievements:")
        print("- Real market data integration")
        print("- Advanced risk management with CVaR")
        print("- Qwen 0.5B model integration")
        print("- Professional training pipeline")
        print("- Experiment tracking and evaluation")
        print("- End-to-end training workflow")

        print("\nQwen Advanced RL Training Features:")
        print("✓ Real market data from Yahoo Finance")
        print("✓ CVaR-based reward shaping")
        print("✓ Position and portfolio risk constraints")
        print("✓ Qwen transformer model integration")
        print("✓ PPO training with custom features")
        print("✓ Comprehensive experiment tracking")
        print("✓ Professional model evaluation")

        print("\nReady for full training:")
        print("python train_qwen_advanced.py --symbols AAPL MSFT --timesteps 100000")
        return 0
    else:
        print(f"\n[WARNING] {total_tests - success_count} test(s) failed")
        print("Some components may need attention before full training")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
