# Financial Trading RL Gym

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive reinforcement learning environment specifically designed for financial trading applications, featuring realistic market dynamics, transaction costs, and multi-asset support.

## 🎯 Features

### 🏗️ Advanced Environment Design
- **Realistic Market Dynamics:** Regime switching, fat tails, jump diffusion, and microstructure effects
- **Sophisticated Risk Management:** VaR constraints, drawdown limits, position sizing rules
- **Multiple Market Frictions:** Transaction costs, slippage, market impact, and execution delays
- **Multi-Asset Support:** Portfolio optimization with correlated returns

### 🤖 Advanced RL Agents
- **Transformer-Based Agents:** Qwen transformer integration with custom feature extraction
- **Advanced PPO Training:** Trust-region control, entropy regularization, KL divergence constraints
- **Meta-Learning Capabilities:** Demonstrated learning-to-learn behaviors in financial contexts

### 📊 Research-Ready Features
- **Statistical Validation:** Bootstrap confidence intervals, effect size analysis, changepoint detection
- **Comprehensive Evaluation:** Financial metrics, RL metrics, risk-adjusted performance measures
- **Reproducible Experiments:** Multi-seed validation, proper seeding, uncertainty quantification

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/4mlabs/financial-trading-rl-gym.git
cd financial-trading-rl-gym

# Install dependencies
pip install -e .

# Install optional dependencies for examples
pip install -e ".[examples]"
```

### Basic Usage

```python
import gymnasium as gym
import financial_trading_gym
import numpy as np

# Create single asset trading environment
env = gym.make('FinancialTrading-SingleAsset-v0')

# Run an episode
obs, info = env.reset()
done = False
total_reward = 0

while not done:
    # Random action for demonstration
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

    print(f"Step: Portfolio=${info['portfolio_value']:,.0f}, Reward={reward:.4f}")

print(f"Episode complete! Total reward: {total_reward:.2f}")
```

### Portfolio Optimization Example

```python
from financial_trading_gym.environments import PortfolioOptimizationEnv
from financial_trading_gym.environments.base_env import AssetConfig

# Define assets
assets = [
    AssetConfig(symbol="AAPL", sector="Technology", volatility=0.025),
    AssetConfig(symbol="MSFT", sector="Technology", volatility=0.022),
    AssetConfig(symbol="JPM", sector="Financial", volatility=0.020),
    AssetConfig(symbol="JNJ", sector="Healthcare", volatility=0.015),
]

# Create environment
env = PortfolioOptimizationEnv(
    assets=assets,
    initial_cash=1_000_000,
    max_episode_length=252,
    rebalance_mode="discrete"
)

# Train your agent here...
```

### Transformer Agent Training

```python
from financial_trading_gym.agents import QwenTradingAgent
from stable_baselines3 import PPO

# Create transformer-based agent
agent = QwenTradingAgent(
    model_name="Qwen/Qwen-0.5B",  # Or use local model
    observation_space=env.observation_space,
    action_space=env.action_space
)

# Train with PPO
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=dict(
        features_extractor_class=agent.get_feature_extractor_class(),
        features_extractor_kwargs=dict(features_dim=256)
    ),
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    verbose=1
)

model.learn(total_timesteps=100_000)
```

## 📦 Available Environments

### Single Asset Trading (`FinancialTrading-SingleAsset-v0`)
- **Objective:** Maximize risk-adjusted returns through individual stock trading
- **Action Space:** `Discrete(3)` or `Box(1,)` - [Sell, Hold, Buy] or position weight
- **Features:** Technical indicators, price history, portfolio state
- **Applications:** Technical analysis, market timing, risk-adjusted optimization

### Portfolio Optimization (`FinancialTrading-Portfolio-v0`)
- **Objective:** Optimize multi-asset allocation for risk-adjusted returns
- **Action Space:** `Box(n_assets)` - Portfolio weights [0, 1]
- **Features:** Correlated returns, sector exposure, turnover penalties
- **Applications:** Modern portfolio theory, dynamic allocation, risk parity

### Regime Detection (`FinancialTrading-RegimeDetection-v0`)
- **Objective:** Identify market regimes and adapt trading strategies
- **Action Space:** `MultiDiscrete([n_regimes, 3])` - [Regime prediction, Trading action]
- **Features:** Hidden Markov Model states, partial observability, regime-specific rewards
- **Applications:** Market state classification, strategy adaptation, meta-learning

### Market Making (`FinancialTrading-MarketMaking-v0`)
- **Objective:** Provide liquidity while managing inventory risk
- **Action Space:** `Box(4)` - [Bid price, Ask price, Bid size, Ask size]
- **Features:** Order book simulation, inventory constraints, adverse selection
- **Applications:** Market making strategies, inventory risk management, HFT

## 📊 Evaluation Metrics

### Financial Metrics
- **Sharpe Ratio:** Risk-adjusted returns (`(mean_return - risk_free_rate) / volatility`)
- **Sortino Ratio:** Downside risk-adjusted returns
- **Maximum Drawdown:** Largest peak-to-trough decline
- **Calmar Ratio:** Annual return / maximum drawdown
- **Information Ratio:** Excess return per unit of tracking error

### Risk Metrics
- **Value-at-Risk (VaR):** Potential loss at confidence level
- **Expected Shortfall:** Average loss beyond VaR
- **Beta Exposure:** Market sensitivity
- **Concentration Risk:** Portfolio concentration metrics

### RL-Specific Metrics
- **Episode Returns:** Total reward per episode
- **Convergence Analysis:** Learning stability metrics
- **Policy Consistency:** Action distribution stability
- **Sample Efficiency:** Returns per training sample

## 🔧 Advanced Configuration

### Custom Asset Configurations

```python
from financial_trading_gym.environments.base_env import AssetConfig

# Create custom asset
custom_asset = AssetConfig(
    symbol="CRYPTO",
    name="Bitcoin",
    sector="Cryptocurrency",
    initial_price=50000.0,
    volatility=0.05,  # Higher volatility for crypto
    drift=0.001,      # Positive drift
    market_cap=1e12,
    avg_daily_volume=1e9
)
```

### Custom Risk Constraints

```python
from financial_trading_gym.environments.base_env import RiskConstraints

# Institutional-grade constraints
constraints = RiskConstraints(
    max_leverage=1.5,
    max_position_size=0.2,
    max_sector_exposure=0.3,
    var_limit=0.02,      # 2% daily VaR
    max_drawdown=0.10    # 10% max drawdown
)
```

### Market Regime Configuration

```python
from financial_trading_gym.data.synthetic import RegimeParameters

# Custom regime parameters
custom_regimes = [
    RegimeParameters(
        name="crypto_bull",
        drift_mean=0.002,
        volatility_mean=0.04,
        correlation_base=0.4,
        jump_intensity=0.05,
        persistence=0.93
    )
]
```

## 📚 Examples

See the `examples/` directory for comprehensive examples:

- `basic_usage.py` - Simple environment interaction
- `portfolio_optimization.py` - Multi-asset trading
- `transformer_agent.py` - Using Qwen transformer agents
- `risk_management.py` - Advanced risk constraint configuration
- `custom_environment.py` - Creating custom trading environments

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=financial_trading_gym --cov-report=html

# Run specific test categories
pytest tests/unit/  # Unit tests
pytest tests/integration/  # Integration tests
```

## 📈 Research Results

This environment has been validated through extensive research:

- **Meta-Learning Emergence:** Demonstrated phase transitions in RL agents at ~55k training steps
- **Statistical Validation:** Large effect sizes (Cohen's d > 0.8) with bootstrap confidence intervals
- **Real Market Performance:** Validated on 2024 market data with Sharpe ratios 0.58-0.89
- **Robustness:** Consistent performance across bull/bear/volatility regimes

See the `research/` directory for detailed papers and analysis.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/4mlabs/financial-trading-rl-gym.git
cd financial-trading-rl-gym

# Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run code formatting
black financial_trading_gym/
```

### Adding New Environments

1. Create new environment class inheriting from `FinancialTradingBase`
2. Implement required abstract methods
3. Add comprehensive documentation and examples
4. Include unit tests with 90%+ coverage
5. Update README with environment details

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not intended for live trading without proper risk management, regulatory compliance, and thorough testing. Past performance does not guarantee future results.

## 🙏 Acknowledgments

- OpenAI Gymnasium framework for environment standards
- Stable Baselines3 team for RL algorithms
- Qwen team for the transformer model
- The broader RL and quantitative finance communities

## 📞 Support

- **Issues:** Report bugs and request features on GitHub Issues
- **Discussions:** Join our community discussions
- **Documentation:** See `docs/` for detailed API documentation
- **Examples:** Check `examples/` for comprehensive usage examples

---

**Built with ❤️ for the RL and Quantitative Finance communities by [4M Labs](https://4mlabs.io)**