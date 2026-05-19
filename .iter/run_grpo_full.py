"""Overnight training: 1) Fresh PPO on fixed env, 2) GRPO alignment."""
import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, '.')
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from environments import SingleAssetTradingEnv
from environments.base_env import AssetConfig
from training.dpo_trainer import GRPOTrainer

asset = AssetConfig(symbol="AAPL", name="Apple", sector="Tech")
def make_env():
    return SingleAssetTradingEnv(asset=asset, action_space_type="discrete")

# === PHASE 1: Fresh PPO on FIXED environment ===
print("="*60)
print("PHASE 1: Fresh PPO training (fixed env)")
print("="*60)

env = DummyVecEnv([make_env])
model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
            n_epochs=10, seed=42, device="auto")
env.close()

model.learn(total_timesteps=200000, progress_bar=True)
model.save("checkpoints/ppo_fresh_fixed_env.zip")
print("\nPhase 1 complete: checkpoints/ppo_fresh_fixed_env.zip")

# Evaluate
env = DummyVecEnv([make_env])
returns = []
for ep in range(30):
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
print(f"Phase 1 result: sharpe={np.mean(rets)/(np.std(rets)+1e-8):.4f} win_rate={np.mean(rets>0):.2%} mean={np.mean(rets):.4f}")

# === PHASE 2: GRPO alignment ===
print("\n" + "="*60)
print("PHASE 2: GRPO alignment")
print("="*60)

trainer = GRPOTrainer(model.policy, beta=0.04, epsilon=0.2, group_size=8, learning_rate=1e-5)
for i in range(100):
    metrics = trainer.train_step(make_env, n_envs=4)
    if (i+1) % 10 == 0:
        print(f"  Iter {i+1}/100: loss={metrics.get('grpo_loss',0):.4f} kl={metrics.get('kl_divergence',0):.4f} ret={metrics.get('mean_return',0):.4f} sharpe={metrics.get('mean_sharpe',0):.4f}")

model.save("checkpoints/ppo_and_grpo_final.zip")
print("\nSaved to checkpoints/ppo_and_grpo_final.zip")

# Final evaluation
print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60)
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
print(f"Sharpe:    {np.mean(rets)/(np.std(rets)+1e-8):.4f}")
print(f"Win Rate:  {np.mean(rets>0):.2%}")
print(f"Mean:      {np.mean(rets):.4f}")
print(f"Std:       {np.std(rets):.4f}")
print(f"\nDone! Good night.")
