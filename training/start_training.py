"""
Start Qwen RL Training

Simple script to start training Qwen on the financial trading gym.
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 60)
    print("STARTING QWEN RL TRAINING")
    print("=" * 60)

    # Import and start training
    try:
        from train_qwen_rl import train_qwen_on_financial_env

        print("Starting training on single asset environment...")
        print("This will take some time. Monitor progress in TensorBoard.")
        print("Press Ctrl+C to stop training early.")
        print()

        # Start training with smaller parameters for testing
        model, results = train_qwen_on_financial_env(
            env_type="single_asset",
            total_timesteps=100000,  # Reduced for testing
            learning_rate=3e-5,
            n_envs=2,  # Reduced for testing
            model_save_path="qwen_test_model"
        )

        print("\nTraining completed successfully!")
        print(f"Final results: {len(results)} evaluation points recorded")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Model should be saved as 'qwen_test_model_interrupted'")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()