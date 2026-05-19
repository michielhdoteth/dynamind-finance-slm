"""Run GRPO alignment on existing model properly.
This loads the original proven model and aligns it with GRPO.
"""
import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, '.')
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from training.dpo_trainer import GRPOTrainer
from environments import SingleAssetTradingEnv
from environments.base_env import AssetConfig

# Load the original model with its correct observation space
print("Loading model...")
model = PPO.load("models/qwen_final_model_200k.zip", device="auto")
print(f"Model loaded. Policy: {model.policy.__class__.__name__}")
print(f"Observation space: {model.observation_space}")

# Create env that matches the model's original space (no Kronos, discrete actions)
asset = AssetConfig(symbol="AAPL", name="Apple", sector="Tech")
def make_env():
    return SingleAssetTradingEnv(asset=asset, action_space_type="discrete")

# Quick test: evaluate current model on fixed env
print("Evaluating baseline model on fixed env...")
env = DummyVecEnv([make_env])
returns = []
for ep in range(20):
    obs = env.reset()
    done = [False]
    ep_ret = 0
    while not done[0]:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        ep_ret += float(reward[0])
    returns.append(ep_ret)
env.close()

rets = np.array(returns)
print(f"Baseline: sharpe={np.mean(rets)/(np.std(rets)+1e-8):.4f}, win_rate={np.mean(rets>0):.2%}, mean={np.mean(rets):.4f}")

# Run GRPO alignment
print("\nStarting GRPO alignment (100 iterations)...")
trainer = GRPOTrainer(
    model.policy, 
    beta=0.04, 
    epsilon=0.2, 
    group_size=8, 
    learning_rate=1e-5
)

for i in range(100):
    metrics = trainer.train_step(make_env, n_envs=4)
    if (i + 1) % 10 == 0:
        loss = metrics.get("grpo_loss", 0)
        kl = metrics.get("kl_divergence", 0)
        ret = metrics.get("mean_return", 0)
        shrp = metrics.get("mean_sharpe", 0)
        print(f"  Iter {i+1}/100: loss={loss:.4f} kl={kl:.4f} ret={ret:.4f} sharpe={shrp:.4f}")

# Save final model
model.save("checkpoints/grpo_aligned_final.zip")
print("\nModel saved to checkpoints/grpo_aligned_final.zip")

# Final evaluation
print("\nFinal evaluation...")
env = DummyVecEnv([make_env])
returns = []
for ep in range(50):
    obs = env.reset()
    done = [False]
    ep_ret = 0
    while not done[0]:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        ep_ret += float(reward[0])
    returns.append(ep_ret)
env.close()

rets = np.array(returns)
print(f"\n=== GRPO ALIGNED RESULTS ===")
print(f"Sharpe: {np.mean(rets)/(np.std(rets)+1e-8):.4f}")
print(f"Win Rate: {np.mean(rets>0):.2%}")
print(f"Mean Return: {np.mean(rets):.4f}")
