"""Debug GRPO action shapes."""
import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, '.')
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from environments import SingleAssetTradingEnv
from environments.base_env import AssetConfig
from training.dpo_trainer import GRPOTrainer

# Create env and model
asset = AssetConfig(symbol="AAPL", name="Apple", sector="Tech")
def make_env():
    return SingleAssetTradingEnv(asset=asset, action_space_type="discrete")

model = PPO.load("checkpoints/ppo_fresh_fixed_env.zip", device="auto")

# Quick test: collect_one_step
env = DummyVecEnv([make_env])
obs = env.reset()

with torch.no_grad():
    obs_t = torch.FloatTensor(obs).to("cuda")
    dist = model.policy.get_distribution(obs_t)
    actions = dist.sample()
    log_probs = dist.log_prob(actions)

print(f"obs_t shape: {obs_t.shape}")
print(f"actions shape: {actions.shape}")
print(f"actions[0]: {actions[0]}, type: {type(actions[0])}")
print(f"actions[0].numpy(): {actions[0].cpu().numpy()}, shape: {actions[0].cpu().numpy().shape}")
print(f"log_probs shape: {log_probs.shape}")
print(f"log_probs[0]: {log_probs[0]}, shape: {log_probs[0].shape}")

# Test conversion
act_np = actions[0].cpu().numpy()
print(f"\nact_np: {act_np}, dtype: {act_np.dtype}, shape: {act_np.shape}")

act_t = torch.LongTensor(act_np)
print(f"LongTensor: {act_t}, shape: {act_t.shape}")

act_t_unsq = act_t.unsqueeze(0)
print(f"LongTensor unsq: {act_t_unsq}, shape: {act_t_unsq.shape}")

# Test log_prob
logp = dist.log_prob(act_t_unsq)
print(f"log_prob result: {logp}, shape: {logp.shape}")

env.close()
print("\nAction debug complete")
