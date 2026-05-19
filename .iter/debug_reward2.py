"""Debug reward after fixing the 2-step bug."""
import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, '.')
import numpy as np
from environments import SingleAssetTradingEnv
from environments.base_env import AssetConfig

asset = AssetConfig(symbol='AAPL', name='Apple', sector='Tech')
env = SingleAssetTradingEnv(asset=asset, action_space_type='continuous')

# Track portfolio values and rewards
obs, _ = env.reset()
print(f'Init: val={env.portfolio_value:.0f}, hist={env.portfolio_history}')

for i in range(10):
    action = env.action_space.sample()
    obs, reward, term, trunc, _ = env.step(action)
    print(f'Step{i+1}: val={env.portfolio_value:.0f}, hist_len={len(env.portfolio_history)}, reward={reward:.6f}')
    if term or trunc:
        break
env.close()

print()

# Test: random policy aggregate
env2 = SingleAssetTradingEnv(asset=asset, action_space_type='continuous')
returns = []
for ep in range(20):
    obs, _ = env2.reset()
    done = False
    ep_ret = 0
    while not done:
        action = env2.action_space.sample()
        obs, reward, term, trunc, _ = env2.step(action)
        ep_ret += reward
        done = term or trunc
    returns.append(ep_ret)

rets = np.array(returns)
print(f'Random policy:')
print(f'  Returns: mean={np.mean(rets):.4f}, win_rate={np.mean(rets > 0):.2%}, sharpe={np.mean(rets)/(np.std(rets)+1e-8):.4f}')
env2.close()

# Test: buy and hold
env3 = SingleAssetTradingEnv(asset=asset, action_space_type='continuous')
returns3 = []
for ep in range(20):
    obs, _ = env3.reset()
    done = False
    ep_ret = 0
    while not done:
        obs, reward, term, trunc, _ = env3.step(np.array([1.0]))
        ep_ret += reward
        done = term or trunc
    returns3.append(ep_ret)
rets3 = np.array(returns3)
print(f'Buy & hold: mean={np.mean(rets3):.4f}, win_rate={np.mean(rets3 > 0):.2%}')
env3.close()
