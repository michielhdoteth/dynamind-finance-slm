"""Test config loader."""
import sys
sys.path.insert(0, '.')
from config.loader import load_config, get_config

cfg = load_config()
assert cfg, "Config should not be empty"
print(f'Model name: {get_config("model.name")}')
print(f'PPO LR: {get_config("ppo.learning_rate")}')
print(f'Total timesteps: {get_config("training.total_timesteps")}')
print(f'N seeds: {get_config("training.n_seeds")}')
print(f'Model variants: {len(get_config("model_variants", []))}')
print('Config loader OK')
