"""Detailed trace of buy-and-hold strategy."""
import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, '.')
import numpy as np
from environments import SingleAssetTradingEnv
from environments.base_env import AssetConfig

asset = AssetConfig(symbol='AAPL', name='Apple', sector='Tech')
env = SingleAssetTradingEnv(asset=asset, action_space_type='discrete')

obs, _ = env.reset()
print(f'Step 0: cash={env.cash_balance:.0f} pos={env.positions[asset.symbol]} val={env.portfolio_value:.0f}')
print(f'  Market data rows: {len(env.market_data)}')
print(f'  Current price index: {env._current_row if hasattr(env, "_current_row") else "?"}')

for i in range(1, 255):
    obs, reward, term, trunc, _ = env.step(1)  # Always Buy
    if i <= 5 or i % 50 == 0:
        print(f'Step {i}: cash={env.cash_balance:.0f} pos={env.positions[asset.symbol]} val={env.portfolio_value:.0f} reward={reward:.6f}')
    if term or trunc:
        print(f'  Episode ended at step {i}')
        break

print(f'\nFinal: val={env.portfolio_value:.0f}, total_return={(env.portfolio_value - 1_000_000) / 1_000_000 * 100:.2f}%')
env.close()
