"""Run GRPO on the fresh PPO model from Phase 1."""
import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, '.')
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from environments import SingleAssetTradingEnv
from environments.base_env import AssetConfig
from training.dpo_trainer import GRPOTrainer

print("Loading Phase 1 model...")
asset = AssetConfig(symbol="AAPL", name="Apple", sector="Tech")
def make_env():
    return SingleAssetTradingEnv(asset=asset, action_space_type="discrete")

model = PPO.load("checkpoints/ppo_fresh_fixed_env.zip", device="auto")
print("Model loaded. Running GRPO...")

trainer = GRPOTrainer(model.policy, beta=0.04, epsilon=0.2, group_size=8, learning_rate=1e-5)
for i in range(100):
    metrics = trainer.train_step(make_env, n_envs=4)
    if (i+1) % 10 == 0:
        print(f"  Iter {i+1}/100: loss={metrics['grpo_loss']:.4f} kl={metrics['kl_divergence']:.4f} ret={metrics['mean_return']:.4f} sharpe={metrics['mean_sharpe']:.4f}")

model.save("checkpoints/ppo_and_grpo_final.zip")
print("\nSaved to checkpoints/ppo_and_grpo_final.zip")

# Final eval
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
print(f"\n=== FINAL RESULTS ===")
print(f"Sharpe:    {np.mean(rets)/(np.std(rets)+1e-8):.4f}")
print(f"Win Rate:  {np.mean(rets>0):.2%}")
print(f"Mean Ret:  {np.mean(rets):.4f}")
