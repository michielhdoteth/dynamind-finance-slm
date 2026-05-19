"""Test discrete action env returns."""
import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, '.')
import numpy as np
from environments import SingleAssetTradingEnv
from environments.base_env import AssetConfig

asset = AssetConfig(symbol='AAPL', name='Apple', sector='Tech')

# Test: random discrete actions [0=Hold, 1=Buy, 2=Sell]
env = SingleAssetTradingEnv(asset=asset, action_space_type='discrete')
returns = []
for ep in range(50):
    obs, _ = env.reset()
    done = False
    ep_ret = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, term, trunc, _ = env.step(int(action))
        ep_ret += reward
        done = term or trunc
    returns.append(ep_ret)

rets = np.array(returns)
print(f'Discrete random [Hold/Buy/Sell]:')
print(f'  Mean: {np.mean(rets):.4f}')
print(f'  Win rate: {np.mean(rets > 0):.2%}')
print(f'  Sharpe: {np.mean(rets) / (np.std(rets) + 1e-8):.4f}')
print(f'  Top 10%: {np.percentile(rets, 90):.4f}')
print(f'  Bottom 10%: {np.percentile(rets, 10):.4f}')
env.close()

# Test: always buy
env2 = SingleAssetTradingEnv(asset=asset, action_space_type='discrete')
returns2 = []
for ep in range(50):
    obs, _ = env2.reset()
    done = False
    ep_ret = 0
    while not done:
        obs, reward, term, trunc, _ = env2.step(1)  # Buy
        ep_ret += reward
        done = term or trunc
    returns2.append(ep_ret)
rets2 = np.array(returns2)
print(f'\nAlways Buy:')
print(f'  Mean: {np.mean(rets2):.4f}')
print(f'  Win rate: {np.mean(rets2 > 0):.2%}')
env2.close()

# Test: always hold
env3 = SingleAssetTradingEnv(asset=asset, action_space_type='discrete')
returns3 = []
for ep in range(50):
    obs, _ = env3.reset()
    done = False
    ep_ret = 0
    while not done:
        obs, reward, term, trunc, _ = env3.step(0)  # Hold
        ep_ret += reward
        done = term or trunc
    returns3.append(ep_ret)
rets3 = np.array(returns3)
print(f'\nAlways Hold:')
print(f'  Mean: {np.mean(rets3):.4f}')
env3.close()
