"""Load model weights directly from zip and run GRPO."""
import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, '.')
import os, zipfile, io, pickle
import numpy as np
import torch
import torch.nn as nn

# Read the policy weights directly from the zip
print("Extracting policy weights from checkpoint...")
with zipfile.ZipFile("models/qwen_final_model_200k.zip") as z:
    # Load policy.pth
    policy_bytes = z.read("policy.pth")
    policy_state = torch.load(io.BytesIO(policy_bytes), map_location="cpu", weights_only=True)
    
    # Load data.json for architecture info
    import json
    data = json.loads(z.read("data"))
    
print(f"Policy keys: {list(policy_state.keys())[:5]}...")
print(f"Features dim: {data.get('policy_kwargs', {}).get('features_extractor_kwargs', {}).get('features_dim', '?')}")

# Create env
from environments import SingleAssetTradingEnv
from environments.base_env import AssetConfig
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

asset = AssetConfig(symbol="AAPL", name="Apple", sector="Tech")
env = SingleAssetTradingEnv(asset=asset, action_space_type="discrete")

print("Creating model with matching architecture...")
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=dict(
        features_extractor_class=QwenFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[{"pi": [256, 128], "vf": [256, 128]}],
        activation_fn=nn.ReLU,
        share_features_extractor=False,
    ),
    verbose=0,
    device="auto",
)
env.close()

# Manually load each weight with strict=False to handle key differences
print("Loading weights...")
missing, unexpected = model.policy.load_state_dict(policy_state, strict=False)
print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
if missing:
    print(f"  Missing sample: {missing[:3]}")
if unexpected:
    print(f"  Unexpected sample: {unexpected[:3]}")

# Quick eval
def make_env():
    return SingleAssetTradingEnv(asset=asset, action_space_type="discrete")

print("\nEvaluating...")
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
print(f"Baseline: sharpe={np.mean(rets)/(np.std(rets)+1e-8):.4f}, mean={np.mean(rets):.4f}")
