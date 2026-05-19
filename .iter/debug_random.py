"""Test random policy with position sizing fix."""
import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, '.')
import numpy as np
from environments import SingleAssetTradingEnv
from environments.base_env import AssetConfig

asset = AssetConfig(symbol='AAPL', name='Apple', sector='Tech')

# Test: continuous with random actions
env = SingleAssetTradingEnv(asset=asset, action_space_type='continuous')
returns = []
for ep in range(20):
    obs, _ = env.reset()
    done = False
    ep_ret = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, term, trunc, _ = env.step(action)
        ep_ret += reward
        done = term or trunc
    returns.append(ep_ret)

rets = np.array(returns)
print(f'Continuous random policy (with fix):')
print(f'  Returns: {rets}')
print(f'  Mean: {np.mean(returns):.4f}')
print(f'  Std: {np.std(returns):.4f}')
print(f'  Sharpe: {np.mean(returns) / (np.std(returns) + 1e-8):.4f}')
print(f'  Win rate: {np.mean(rets > 0):.2%}')
env.close()

# Test: always buy
env2 = SingleAssetTradingEnv(asset=asset, action_space_type='continuous')
returns2 = []
for ep in range(20):
    obs, _ = env2.reset()
    done = False
    ep_ret = 0
    while not done:
        obs, reward, term, trunc, _ = env2.step(np.array([1.0]))
        ep_ret += reward
        done = term or trunc
    returns2.append(ep_ret)
rets2 = np.array(returns2)
print(f'\nAlways buy (action=1.0):')
print(f'  Mean: {np.mean(rets2):.4f}')
print(f'  Win rate: {np.mean(rets2 > 0):.2%}')
env2.close()

# Test: always hold
env3 = SingleAssetTradingEnv(asset=asset, action_space_type='continuous')
returns3 = []
for ep in range(20):
    obs, _ = env3.reset()
    done = False
    ep_ret = 0
    while not done:
        obs, reward, term, trunc, _ = env3.step(np.array([0.0]))
        ep_ret += reward
        done = term or trunc
    returns3.append(ep_ret)
rets3 = np.array(returns3)
print(f'\nAlways neutral (action=0.0):')
print(f'  Mean: {np.mean(rets3):.4f}')
env3.close()
