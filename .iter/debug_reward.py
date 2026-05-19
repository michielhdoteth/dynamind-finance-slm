"""Debug reward computation in the trading env."""
import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, '.')
import numpy as np
from environments import SingleAssetTradingEnv
from environments.base_env import AssetConfig

asset = AssetConfig(symbol='AAPL', name='Apple', sector='Tech')
env = SingleAssetTradingEnv(asset=asset, action_space_type='continuous')

obs, _ = env.reset()
print(f'Initial: cash={env.cash_balance:.0f}, pos={env.positions[asset.symbol]}')

# Step 1 with buy
obs, reward, term, trunc, info = env.step(np.array([1.0]))
print(f'Step1 (buy=1.0): cash={env.cash_balance:.0f}, pos={env.positions[asset.symbol]}, val={env.portfolio_value:.0f}, reward={reward:.6f}')

# Step 2 with 0 (hold position)
obs, reward, term, trunc, info = env.step(np.array([0.0]))
print(f'Step2 (hold=0): cash={env.cash_balance:.0f}, pos={env.positions[asset.symbol]}, val={env.portfolio_value:.0f}, reward={reward:.6f}')

# Continue
for i in range(3, 15):
    obs, reward, term, trunc, info = env.step(np.array([1.0]))
    print(f'Step{i}: val={env.portfolio_value:.0f}, pos={env.positions[asset.symbol]}, reward={reward:.6f}')
    if term or trunc:
        print(f'  Episode ended at step {i}')
        break

env.close()
