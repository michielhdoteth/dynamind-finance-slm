"""
Financial Trading Research Gym
A collection of financial trading environments for reinforcement learning research.

This package provides standardized gymnasium environments for various financial trading tasks:
- Single asset trading
- Portfolio optimization
- Regime detection
- Risk management
- Market making

Designed for research with proper reproducibility, benchmarking, and evaluation metrics.
"""

__version__ = "1.0.0"
__author__ = "Financial Trading RL Research Team"

from .environments import (
    SingleAssetTradingEnv,
    PortfolioOptimizationEnv,
    RegimeDetectionEnv,
    MarketMakingEnv
)

from .data import MarketDataGenerator
from .evaluation import BenchmarkSuite, FinancialMetrics, ModelEvaluator, RiskMetrics

# Environment registration for gymnasium
import gymnasium as gym

# Register environments
gym.register(
    id='FinancialTrading-SingleAsset-v0',
    entry_point='financial_trading_gym.environments:SingleAssetTradingEnv',
    max_episode_steps=252,
)

gym.register(
    id='FinancialTrading-Portfolio-v0',
    entry_point='financial_trading_gym.environments:PortfolioOptimizationEnv',
    max_episode_steps=252,
)

gym.register(
    id='FinancialTrading-RegimeDetection-v0',
    entry_point='financial_trading_gym.environments:RegimeDetectionEnv',
    max_episode_steps=504,
)

gym.register(
    id='FinancialTrading-MarketMaking-v0',
    entry_point='financial_trading_gym.environments:MarketMakingEnv',
    max_episode_steps=1000,
)

__all__ = [
    'SingleAssetTradingEnv',
    'PortfolioOptimizationEnv',
    'RegimeDetectionEnv',
    'MarketMakingEnv',
    'MarketDataGenerator',
    'FinancialMetrics',
    'RiskMetrics',
    'BenchmarkSuite',
    'ModelEvaluator',
]