"""Multi-episode test of all strategies after fixes."""
import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, '.')
import numpy as np
from environments import SingleAssetTradingEnv
from environments.base_env import AssetConfig

asset = AssetConfig(symbol='AAPL', name='Apple', sector='Tech')
n_eps = 100

# Buy and hold
ret_bh = []
for ep in range(n_eps):
    env = SingleAssetTradingEnv(asset=asset, action_space_type='discrete', seed=ep*10)
    obs, _ = env.reset()
    done = False
    ep_ret = 0
    while not done:
        obs, reward, term, trunc, _ = env.step(1)
        ep_ret += reward
        done = term or trunc
    ret_bh.append(ep_ret)
    env.close()

rets_bh = np.array(ret_bh)

# Random
ret_rd = []
for ep in range(n_eps):
    env = SingleAssetTradingEnv(asset=asset, action_space_type='discrete', seed=ep*10)
    obs, _ = env.reset()
    done = False
    ep_ret = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, term, trunc, _ = env.step(int(action))
        ep_ret += reward
        done = term or trunc
    ret_rd.append(ep_ret)
    env.close()

rets_rd = np.array(ret_rd)

# Hold
ret_hl = []
for ep in range(n_eps):
    env = SingleAssetTradingEnv(asset=asset, action_space_type='discrete', seed=ep*10)
    obs, _ = env.reset()
    done = False
    ep_ret = 0
    while not done:
        obs, reward, term, trunc, _ = env.step(0)
        ep_ret += reward
        done = term or trunc
    ret_hl.append(ep_ret)
    env.close()

rets_hl = np.array(ret_hl)

print(f'Strategy    Mean    Std    Sharpe   Win%   Best   Worst')
print(f'Buy&Hold: {np.mean(rets_bh):6.2f} {np.std(rets_bh):6.2f} {np.mean(rets_bh)/(np.std(rets_bh)+1e-8):7.2f} {np.mean(rets_bh>0)*100:5.0f}% {np.max(rets_bh):6.2f} {np.min(rets_bh):6.2f}')
print(f'Random:   {np.mean(rets_rd):6.2f} {np.std(rets_rd):6.2f} {np.mean(rets_rd)/(np.std(rets_rd)+1e-8):7.2f} {np.mean(rets_rd>0)*100:5.0f}% {np.max(rets_rd):6.2f} {np.min(rets_rd):6.2f}')
print(f'Hold:     {np.mean(rets_hl):6.2f} {np.std(rets_hl):6.2f} {np.mean(rets_hl)/(np.std(rets_hl)+1e-8):7.2f} {np.mean(rets_hl>0)*100:5.0f}% {np.max(rets_hl):6.2f} {np.min(rets_hl):6.2f}')
