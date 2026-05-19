"""Run GRPO - loads model properly with custom feature extractor."""
import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, '.')
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

# Define the same QwenFeaturesExtractor class that was used during training
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

# Make it accessible at module level for SB3 deserialization
sys.modules[__name__].QwenFeaturesExtractor = QwenFeaturesExtractor

print("Loading model with custom feature extractor...")
model = PPO.load(
    "models/qwen_final_model_200k.zip",
    device="auto",
    custom_objects={
        "features_extractor_class": QwenFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": [{"pi": [256, 128], "vf": [256, 128]}],
        "activation_fn": nn.ReLU,
    }
)
print(f"Model loaded: {type(model).__name__}")

# Create env matching original setup (discrete actions, no Kronos wrapper)
from environments import SingleAssetTradingEnv
from environments.base_env import AssetConfig

asset = AssetConfig(symbol="AAPL", name="Apple", sector="Tech")
def make_env():
    return SingleAssetTradingEnv(asset=asset, action_space_type="discrete")

# Evaluate baseline
print("\nEvaluating baseline on fixed env...")
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
baseline_sharpe = np.mean(rets) / (np.std(rets) + 1e-8)
baseline_win = np.mean(rets > 0)
print(f"Baseline: sharpe={baseline_sharpe:.4f} win_rate={baseline_win:.2%} mean={np.mean(rets):.4f}")

# Run GRPO
print("\nRunning GRPO alignment (100 iterations)...")
from training.dpo_trainer import GRPOTrainer

trainer = GRPOTrainer(
    model.policy,
    beta=0.04,
    epsilon=0.2,
    group_size=8,
    learning_rate=1e-5,
)

for i in range(100):
    metrics = trainer.train_step(make_env, n_envs=4)
    if (i + 1) % 10 == 0:
        loss = metrics.get("grpo_loss", 0)
        kl = metrics.get("kl_divergence", 0)
        ret = metrics.get("mean_return", 0)
        shrp = metrics.get("mean_sharpe", 0)
        print(f"  Iter {i+1}/100: loss={loss:.4f} kl={kl:.4f} ret={ret:.4f} sharpe={shrp:.4f}")

model.save("checkpoints/grpo_aligned_final.zip")
print("\nSaved: checkpoints/grpo_aligned_final.zip")

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
print(f"\n=== GRPO RESULTS ===")
print(f"Sharpe:    {np.mean(rets)/(np.std(rets)+1e-8):.4f}")
print(f"Win Rate:  {np.mean(rets>0):.2%}")
print(f"Mean Ret:  {np.mean(rets):.4f}")
print(f"Std Ret:   {np.std(rets):.4f}")
