"""Load model properly and run GRPO - handles architecture mismatch."""
import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, '.')
import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class QwenFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Linear(int(np.prod(observation_space.shape)), 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )
    def forward(self, observations):
        return self.net(observations)

from environments import SingleAssetTradingEnv
from environments.base_env import AssetConfig

# Create env that matches what the model expects
asset = AssetConfig(symbol="AAPL", name="Apple", sector="Tech")
env = SingleAssetTradingEnv(asset=asset, action_space_type="discrete")

print("Creating model with correct architecture...")
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=dict(
        features_extractor_class=QwenFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[{"pi": [256, 128], "vf": [256, 128]}],
        activation_fn=nn.ReLU,
    ),
    verbose=0,
    device="auto",
)
env.close()

# Now load the weights directly (exact_match=False to handle version differences)
print("Loading weights from checkpoint...")
checkpoint = PPO.load("models/qwen_final_model_200k.zip", device="cpu", print_system_info=False)
model.policy.load_state_dict(checkpoint.policy.state_dict(), strict=False)
model.policy.to("cuda" if __import__("torch").cuda.is_available() else "cpu")
print("Weights loaded!")

# Evaluate
def make_env():
    return SingleAssetTradingEnv(asset=asset, action_space_type="discrete")

print("\nEvaluating baseline...")
env = DummyVecEnv([make_env])
returns = []
for ep in range(10):
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
print(f"Baseline: sharpe={np.mean(rets)/(np.std(rets)+1e-8):.4f}")

# Run GRPO
print("\nRunning GRPO (100 iterations)...")
from training.dpo_trainer import GRPOTrainer

trainer = GRPOTrainer(model.policy, beta=0.04, epsilon=0.2, group_size=8, learning_rate=1e-5)

for i in range(100):
    metrics = trainer.train_step(make_env, n_envs=4)
    if (i + 1) % 10 == 0:
        print(f"  Iter {i+1}/100: loss={metrics.get('grpo_loss',0):.4f} sharpe={metrics.get('mean_sharpe',0):.4f}")

model.save("checkpoints/grpo_aligned_final.zip")
print("\nModel saved to checkpoints/grpo_aligned_final.zip")

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
print(f"\n=== FINAL ===")
print(f"Sharpe: {np.mean(rets)/(np.std(rets)+1e-8):.4f}")
print(f"Win Rate: {np.mean(rets>0):.2%}")
