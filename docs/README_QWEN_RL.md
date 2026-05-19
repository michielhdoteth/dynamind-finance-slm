# Training Qwen 0.5B with RL on Financial Trading Gym

This guide shows how to use Qwen 0.5B as a transformer-based policy network trained with reinforcement learning on our financial trading environments.

## 🎯 Overview

We're using Qwen 0.5B as the base transformer model and fine-tuning it with PPO (Proximal Policy Optimization) to learn financial trading strategies. This approach leverages the transformer's sequence modeling capabilities for time-series financial data.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements_qwen.txt

# Run the setup script (downloads Qwen and tests everything)
python setup_qwen_rl.py
```

### 2. Train on Single Asset Trading

```bash
# Train Qwen on single asset trading (faster, good start)
python train_qwen_rl.py --env single_asset --timesteps 500000 --lr 3e-5

# Monitor training
tensorboard --logdir qwen_rl_tensorboard
```

### 3. Train on Portfolio Optimization

```bash
# Train Qwen on portfolio optimization (more complex)
python train_qwen_rl.py --env portfolio --timesteps 1000000 --lr 1e-5
```

### 4. Test Trained Model

```bash
# Test your trained model
python train_qwen_rl.py --mode test --model-path qwen_trading_model --test-episodes 20
```

## 🧠 Architecture

### Qwen Feature Extractor
- **Input**: Financial observations (technical indicators, portfolio state, market data)
- **Processing**: Discretize continuous values → Token embeddings → Qwen transformer
- **Output**: 512-dimensional feature representation

### Policy Network
- **Actor**: Maps Qwen features to action probabilities
- **Critic**: Maps Qwen features to state value estimates
- **Training**: PPO with clipped surrogate objective

### Key Features
- **Partial Fine-Tuning**: Only last 2 transformer layers trainable (faster, prevents overfitting)
- **Numerical Tokenization**: Converts continuous financial data to token-like format
- **Multi-Environment Training**: Parallel environments for sample efficiency
- **Adaptive Learning Rate**: Lower learning rates for stable transformer fine-tuning

## 📊 Training Configuration

### Recommended Hyperparameters

#### Single Asset Trading
```bash
python train_qwen_rl.py \
    --env single_asset \
    --timesteps 500000 \
    --lr 3e-5 \
    --n-envs 4
```

#### Portfolio Optimization
```bash
python train_qwen_rl.py \
    --env portfolio \
    --timesteps 1000000 \
    --lr 1e-5 \
    --n-envs 2
```

### PPO Parameters
- **Learning Rate**: 1e-5 to 3e-5 (much lower than typical RL)
- **Batch Size**: 64
- **N-Steps**: 2048
- **N-Epochs**: 10
- **Clip Range**: 0.2
- **Entropy Coef**: 0.01
- **Value Coef**: 0.5
- **Gamma**: 0.99
- **GAE Lambda**: 0.95

## 🎮 Environment Details

### Single Asset Trading (`single_asset`)
- **Action Space**: Discrete(3) = [Hold, Buy, Sell]
- **Observation Space**: ~512 dimensions (price history, technical indicators, portfolio state)
- **Episode Length**: 252 steps (1 trading year)
- **Reward**: Risk-adjusted returns with transaction costs

### Portfolio Optimization (`portfolio`)
- **Action Space**: Box(n_assets) = Portfolio weights [0, 1]
- **Observation Space**: ~1024 dimensions (multiple assets, correlations, risk metrics)
- **Episode Length**: 252 steps
- **Reward**: Sharpe ratio optimization with turnover penalties

## 📈 Training Results

### Expected Performance
- **Convergence**: 200K-500K timesteps for single asset
- **Final Performance**: Sharpe ratio > 1.0, positive returns
- **Training Time**: 2-6 hours on single GPU (RTX 3080+)

### Monitoring Metrics
- **Episode Rewards**: Learning progress
- **Portfolio Value**: Financial performance
- **Action Entropy**: Exploration vs exploitation
- **Value Loss**: Critic learning
- **Policy Loss**: Actor learning

## 🔧 Customization

### Different Environments
```python
# Add your own environment
env = YourCustomEnvironment()
model = PPO(QwenTradingPolicy, env, learning_rate=2e-5)
model.learn(total_timesteps=1000000)
```

### Custom Reward Functions
```python
# Modify reward in your environment
def _calculate_reward(self, execution_details):
    # Your custom reward logic
    return custom_reward
```

### Different Qwen Models
```python
# Use Qwen 1.5B instead (requires more memory)
model_name = "Qwen/Qwen2-1.5B"
self.qwen = AutoModelForCausalLM.from_pretrained(model_name)
```

## 📱 Hardware Requirements

### Minimum Requirements
- **CPU**: 4+ cores, 16GB RAM
- **GPU**: Not required but recommended
- **Storage**: 10GB free space

### Recommended Setup
- **GPU**: RTX 3080+ (8GB+ VRAM)
- **RAM**: 32GB
- **CPU**: 8+ cores
- **Storage**: SSD for faster I/O

### Performance by Hardware
| Setup | Single Asset (500K steps) | Portfolio (1M steps) |
|-------|-------------------------|----------------------|
| CPU Only | 8-12 hours | 16-24 hours |
| RTX 3060 | 3-4 hours | 6-8 hours |
| RTX 3080+ | 1-2 hours | 2-4 hours |

## 🐛 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size or use CPU
python train_qwen_rl.py --env single_asset --n-envs 1
```

#### Model Download Fails
```bash
# Set HuggingFace cache directory
export HF_HOME=./cache
python setup_qwen_rl.py
```

#### Training is Slow
```bash
# Reduce environment complexity
python train_qwen_rl.py --env single_asset --timesteps 100000 --n-envs 1
```

#### Poor Performance
```bash
# Increase learning rate or add more training steps
python train_qwen_rl.py --lr 5e-5 --timesteps 1000000
```

### Error Messages

**"Failed to load Qwen model"** → Check internet connection and HuggingFace access

**"Observation shape mismatch"** → Ensure environment output matches expected dimensions

**"CUDA out of memory"** → Reduce n_envs or use CPU training

**"Poor convergence"** → Adjust learning rate, increase timesteps, check reward function

## 📚 Advanced Topics

### Multi-Asset Trading
Extend to trade multiple assets simultaneously:
```python
# Multi-discrete action space for multiple assets
action_space = spaces.MultiDiscrete([3] * n_assets)
```

### Transformer Modifications
- Add positional encodings specific to financial time series
- Modify attention mechanisms for temporal patterns
- Add financial domain-specific tokenization

### Transfer Learning
```python
# Load pre-trained Qwen and fine-tune on financial data
model = QwenTradingPolicy.load_from_checkpoint("path/to/model")
```

### Ensemble Methods
```python
# Train multiple Qwen agents with different random seeds
agents = [train_agent(seed=i) for i in range(5)]
# Ensemble their predictions
```

## 🎓 Research Applications

This setup enables various research directions:
- **Transformer-based RL for Finance**: Novel architecture combinations
- **Transfer Learning**: Pre-train on historical data, fine-tune on specific assets
- **Multi-Task Learning**: Train on multiple environments simultaneously
- **Hierarchical RL**: Use Qwen for high-level strategy, traditional networks for execution
- **Interpretability**: Analyze attention patterns for trading insights

## 📄 License

This code follows the same license as the Financial Trading Research Gym. Qwen models are subject to their own license terms.

---

## 🚀 Ready to Start?

1. **Setup**: `python setup_qwen_rl.py`
2. **Train**: `python train_qwen_rl.py --env single_asset`
3. **Monitor**: Open TensorBoard
4. **Test**: `python train_qwen_rl.py --mode test`

Happy training! 🎯