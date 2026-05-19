"""
Setup script for Qwen RL Training

This script handles downloading Qwen 0.5B model and setting up dependencies
for RL training on the financial trading gym.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed."""
    logger.info("Checking dependencies...")

    required_packages = [
        "torch",
        "transformers",
        "gymnasium",
        "stable-baselines3",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn"
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            logger.info(f"{package} is installed")
        except ImportError:
            logger.warning(f"{package} is missing")
            missing_packages.append(package)

    if missing_packages:
        logger.info(f"Installing missing packages: {missing_packages}")
        return missing_packages
    else:
        logger.info("All dependencies are installed")
        return []

def install_dependencies(packages):
    """Install missing dependencies."""
    for package in packages:
        try:
            logger.info(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
            logger.info(f"{package} installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package}: {e}")
            return False

    return True

def check_torch_cuda():
    """Check if PyTorch CUDA support is available."""
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"Current CUDA device: {torch.cuda.get_device_name()}")
            return True
        else:
            logger.warning("CUDA is not available. Training will use CPU (slower).")
            return False

    except ImportError:
        logger.error("PyTorch is not installed")
        return False

def download_qwen_model():
    """Test downloading Qwen 0.5B model."""
    logger.info("Testing Qwen 0.5B model download...")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "Qwen/Qwen2-0.5B"
        logger.info(f"Downloading {model_name}...")

        # This will download the model if not already cached
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )

        logger.info("Qwen 0.5B model downloaded successfully!")
        logger.info(f"Model size: {sum(p.numel() for p in model.parameters()):,} parameters")
        logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")

        # Test basic functionality
        test_input = "Test financial trading:"
        tokens = tokenizer.encode(test_input, return_tensors="pt")
        logger.info(f"Test tokenization successful: {tokens.shape}")

        return True

    except Exception as e:
        logger.error(f"Failed to download Qwen model: {e}")
        return False

def create_directories():
    """Create necessary directories for training."""
    directories = [
        "logs",
        "checkpoints",
        "models",
        "results",
        "tensorboard_logs"
    ]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Directory '{directory}' created/verified")

def test_gym_functionality():
    """Test that our financial trading gym works."""
    logger.info("Testing financial trading gym functionality...")

    try:
        # Add current directory to path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

        # Test single asset environment
        from environments import SingleAssetTradingEnv
        from environments.base_env import AssetConfig

        asset = AssetConfig(
            symbol="TEST",
            name="Test Asset",
            sector="Technology",
            initial_price=100.0
        )

        env = SingleAssetTradingEnv(
            asset=asset,
            initial_cash=100_000,
            max_episode_length=10,  # Short test
            seed=42
        )

        # Test reset and step
        obs, info = env.reset(seed=42)
        logger.info(f"Environment reset successful. Obs shape: {obs.shape if hasattr(obs, 'shape') else len(obs)}")

        # Test few steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            logger.info(f"  Step {i+1}: Action={action}, Reward={reward:.4f}")

            if terminated or truncated:
                break

        logger.info("Gym functionality test passed")
        env.close()
        return True

    except Exception as e:
        logger.error(f"Gym functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main setup function."""
    print("=" * 60)
    print("QWEN RL FINANCIAL TRADING - SETUP")
    print("=" * 60)

    # Step 1: Check dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        if not install_dependencies(missing_deps):
            logger.error("Failed to install dependencies")
            return False

    # Step 2: Check CUDA
    has_cuda = check_torch_cuda()

    # Step 3: Create directories
    create_directories()

    # Step 4: Test gym functionality
    if not test_gym_functionality():
        logger.error("Gym functionality test failed")
        return False

    # Step 5: Download Qwen model
    if not download_qwen_model():
        logger.error("Qwen model download failed")
        return False

    # Success
    print("\n" + "=" * 60)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    print("\nNext steps:")
    print("1. Run training: python train_qwen_rl.py --env single_asset --timesteps 500000")
    print("2. Monitor progress in tensorboard_logs/")
    print("3. Test trained model: python train_qwen_rl.py --mode test --model-path qwen_trading_model")

    if has_cuda:
        print("CUDA is available - training will be fast!")
    else:
        print("Running on CPU - training will be slower but still functional")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)