"""
Financial Trading Research Gym
=======================

A collection of financial trading environments for reinforcement learning research.

All environments, agents, training, and risk management components are
accessible via flat imports after importing this package:

    import financial_trading_gym
    from environments import SingleAssetTradingEnv
    from data import MarketDataGenerator, DataManager
    from risk import RiskManager
    from training import ModelEvaluator
"""

import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import gymnasium as gym

# Register environments with gymnasium
gym.register(
    id="FinancialTrading-SingleAsset-v0",
    entry_point="environments:SingleAssetTradingEnv",
    max_episode_steps=252,
)
gym.register(
    id="FinancialTrading-Portfolio-v0",
    entry_point="environments:PortfolioOptimizationEnv",
    max_episode_steps=252,
)
gym.register(
    id="FinancialTrading-RegimeDetection-v0",
    entry_point="environments:RegimeDetectionEnv",
    max_episode_steps=504,
)
gym.register(
    id="FinancialTrading-MarketMaking-v0",
    entry_point="environments:MarketMakingEnv",
    max_episode_steps=1000,
)

__version__ = "1.0.0"
__author__ = "Financial Trading RL Research Team"
