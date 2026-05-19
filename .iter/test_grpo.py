"""Test GRPO trainer loads and runs correctly."""
import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, '.')
from stable_baselines3 import PPO
from training.dpo_trainer import GRPOTrainer
from training.kronos_training import make_env, KronosTrainingConfig
from environments.kronos_wrapper import KronosFeatureExtractor

print("Loading model...")
model = PPO.load("models/qwen_final_model_200k.zip")
print(f"Model loaded: {type(model).__name__}")

config = KronosTrainingConfig(kronos_size="small")
kronos = KronosFeatureExtractor(feature_dim=64)
env_fn = make_env(config, kronos_extractor=kronos)

trainer = GRPOTrainer(model.policy, beta=0.04, epsilon=0.2, group_size=4, learning_rate=1e-5)
print("GRPO trainer ready. Running 1 step...")
metrics = trainer.train_step(env_fn, n_envs=2)
print(f"GRPO step OK: {metrics}")

# Save checkpoint
model.save("checkpoints/grpo_test.zip")
print("Model saved to checkpoints/grpo_test.zip")
