"""Debug each reward component."""
import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, '.')
import numpy as np
from environments import SingleAssetTradingEnv
from environments.base_env import AssetConfig

asset = AssetConfig(symbol='AAPL', name='Apple', sector='Tech')
env = SingleAssetTradingEnv(asset=asset, action_space_type='discrete')

# Monkey-patch to see reward components
orig_reward = env._calculate_reward
def debug_reward(self, details):
    # Calculate each component manually
    if len(self.portfolio_history) >= 1:
        port_ret = (self.portfolio_value - self.portfolio_history[-1]) / max(self.portfolio_history[-1], 1e-8)
    else:
        port_ret = 0
    
    cost_pen = -details["cost"] / self.portfolio_value if self.portfolio_value > 0 else 0
    
    if len(self.portfolio_history) > 30:
        rets = np.diff(self.portfolio_history[-30:]) / self.portfolio_history[-30:-1]
        vol_pen = -0.5 * np.std(rets)
    else:
        vol_pen = 0
    
    pos_ratio = abs(self.positions[self.asset.symbol] * self.current_prices[self.asset.symbol]) / self.portfolio_value if self.portfolio_value > 0 else 0
    pos_pen = -0.1 * max(0, pos_ratio - 0.8)
    
    total = port_ret + cost_pen + vol_pen + pos_pen
    
    return float(total)

env._calculate_reward = debug_reward.__get__(env, SingleAssetTradingEnv)

obs, _ = env.reset()
for i in range(1, 10):
    obs, reward, term, trunc, _ = env.step(1)  # Always Buy
    print(f'Step {i}: val={env.portfolio_value:.0f} hist_last={env.portfolio_history[-1]:.0f} reward={reward:.6f}')
    if term or trunc:
        break
env.close()
