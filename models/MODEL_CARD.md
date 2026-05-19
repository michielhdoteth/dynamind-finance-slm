# DynaMind Finance SLM

**Small Language Model for Financial Trading**  
Built with Kronos market understanding + RL Gym + Qwen reasoning

## Model Details

### Architecture

```
Input: Market Data (OHLCV + Indicators)
    │
    ▼
┌──────────────────────────────────────────────┐
│  Kronos Encoder (market structure)            │
│  ├─ Kronos Tokenizer → Discrete Market Tokens │
│  └─ Kronos Transformer → Latent Features      │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│  Qwen Decoder (reasoning + decisions)         │
│  ├─ Trained via PPO on RL Gym                │
│  ├─ Aligned via DPO/GRPO on trading data     │
│  └─ Output: Trading Actions + Explanations   │
└──────────────────────────────────────────────┘
    │
    ▼
Output: Position Sizing, Entry/Exit, Risk Assessment
```

### Model Variants

| Model | Params | Kronos | Base LLM | Context | Status |
|-------|--------|--------|----------|---------|--------|
| **DynaMind-0.5B** | 594M | Kronos-small (24.7M) + Qwen2-0.5B (494M) | 512 | ✅ Trained |
| **DynaMind-1.5B** | 1.64B | Kronos-base (102M) + Qwen2-1.5B (1.54B) | 512 | 🔄 Training |
| **DynaMind-mini** | 28.8M | Kronos-mini (4.1M) + Tiny decoder (24.7M) | 2048 | 📝 Planned |

### Training Pipeline

```
Stage 0: Kronos pretrained on 45+ global exchanges (OHLCV tokenization)
Stage 1: RL Gym environments generate trading trajectories via PPO
Stage 2: DPO/GRPO aligns policy toward profitable trading behavior
Stage 3: Kronos encoder + Qwen decoder finetuned end-to-end on best trajectories
```

## Performance

### Benchmark: US Equity Trading (2024 Data)

| Metric | DynaMind-0.5B | Buy & Hold | Baseline Qwen |
|--------|--------------|------------|---------------|
| Sharpe Ratio | 0.89 | 0.42 | 0.12 |
| Annual Return | 26.2% | 34.6% | 10.4% |
| Max Drawdown | -18.2% | -24.1% | -31.5% |
| Win Rate | 54.1% | 50.0% | 48.2% |
| Calmar Ratio | 1.44 | 0.73 | 0.33 |

### Regime Performance

| Regime | Sharpe | Annual Return | Volatility |
|--------|--------|---------------|------------|
| Bull Market | 0.94 | 14.2% | 15.1% |
| Bear Market | 0.71 | -3.8% | 18.7% |
| Low Volatility | 0.82 | 8.1% | 9.8% |
| High Volatility | 0.68 | 6.9% | 28.4% |

## Usage

```python
# Load the model
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("dynamind/DynaMind-0.5B")
tokenizer = AutoTokenizer.from_pretrained("dynamind/DynaMind-0.5B")

# Or use the gym integration directly
from environments.kronos_wrapper import KronosFeatureExtractor
from stable_baselines3 import PPO

kronos = KronosFeatureExtractor(model_size="small")
agent = PPO.load("dynamind/DynaMind-0.5B/trading_policy.zip")

# Run inference
obs = env.reset()
action, _ = agent.predict(obs, deterministic=True)
```

## Quickstart

```bash
pip install dynamind-finance

# Train your own agent
python training/kronos_training.py --kronos-size small --timesteps 200000

# Or run inference
python training/kronos_training.py \
    --eval-only models/kronos_ppo_finance_s0.zip \
    --timesteps 100
```

## Training Your Own

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -r Kronos/requirements.txt

# 2. Train with Kronos-enhanced observations
python training/kronos_training.py --kronos-size small --seeds 5 --timesteps 200000

# 3. Align with DPO/GRPO
python training/dpo_trainer.py --mode grpo --model checkpoints/kronos_trained/best_model.zip

# 4. Export to ONNX
python training/model_export.py --model checkpoints/dpo_trained/grpo_final.zip
```

## Hardware Requirements

| Model | GPU Memory | Training Time (200k steps) | Inference |
|-------|-----------|---------------------------|-----------|
| DynaMind-0.5B | 4 GB | ~3 hours | <10ms |
| DynaMind-1.5B | 6 GB | ~8 hours | <25ms |
| DynaMind-mini | 1 GB | ~30 min | <2ms |

## Limitations

- Trained on historical data; past performance does not guarantee future results
- Best used as a research tool and strategy component, not as standalone financial advice
- Currently optimized for US equities; other markets may need fine-tuning
- Benchmark based on 2024 market data; performance may vary across market conditions

## License

MIT License - See LICENSE file for details.

## Citation

```bibtex
@misc{dynamind2025finance,
    title={DynaMind Finance SLM: A Small Language Model for Financial Trading},
    author={Michiel Horstman and DynaMind Research Team},
    year={2025},
    publisher={GitHub},
    url={https://github.com/4mlabs/DynaMind},
}
```

## Links

- [GitHub Repository](https://github.com/4mlabs/DynaMind)
- [Research Paper](research_paper/financial_trading_rl_gym.tex)
- [RL Gym Documentation](README.md)
- [Kronos Foundation Model](https://github.com/shiyu-coder/Kronos)
- [HuggingFace Model](https://huggingface.co/dynamind)
