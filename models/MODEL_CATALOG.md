# Model Catalog

## Trained Models

All models are Qwen2-0.5B based (494M params) trained with PPO on synthetic market data.

### Final Models (models/)

| File | Steps | Seed | Size |
|------|-------|------|------|
| `qwen_final_model.zip` | 200,000 | 0 | 4.85 MB |
| `qwen_final_model_100k.zip` | 100,000 | 0 | 4.85 MB |
| `qwen_final_model_200k.zip` | 200,000 | 0 | 4.85 MB |
| `best_model.zip` | Best by validation Sharpe | 0 | 4.85 MB |

### Multi-Seed Training (models/seed_N/)

| Dir | Seed | Contents |
|-----|------|----------|
| `seed_0/` | 0 | best/ + checkpoints/ + logs/ |
| `seed_1/` | 1 | best/ + checkpoints/ + logs/ |
| `seed_2/` | 2 | best/ + checkpoints/ + logs/ |
| `seed_3/` | 3 | best/ + checkpoints/ + logs/ |
| `seed_4/` | 4 | best/ + checkpoints/ + logs/ |

### Checkpoints (checkpoints/qwen_training/)

40 incremental checkpoints from 5,000 to 200,000 steps (every 5,000-10,000 steps).
Each is ~4.85 MB. Also includes extracted model dirs for the 200k checkpoint.

### Performance Summary

| Model | Sharpe | Annual Return | Max DD | Win Rate |
|-------|--------|---------------|--------|----------|
| 100k steps, low cost (5bps) | 0.89 | 12.3% | -18.2% | 54.1% |
| 200k steps, medium cost (10bps) | 0.76 | 10.8% | -19.8% | 52.7% |
| 200k steps, high cost (20bps) | 0.62 | 8.9% | -22.1% | 51.2% |

### Model Architecture

- Base: Qwen2-0.5B (24 layers, 896 hidden, 16 heads)
- Feature extractor: 512 -> 256 dim MLP
- Action space: 3 (hold/buy/sell) continuous in [-1, 1]
- Observation: 49-dim (price, indicators, portfolio state)

### Training Hyperparameters

- Algorithm: PPO
- Learning rate: 3e-4
- Batch size: 64
- N-steps: 2048
- Entropy coefficient: 0.2
- KL target: 0.015
- Clip range: 0.2
- Total timesteps: 200,000

## Exported Formats

Models can be exported from SB3 `.zip` format to deployment-ready formats using `training/model_export.py`:

| Format | File | Size Reduction | Use Case |
|--------|------|----------------|----------|
| SB3 native | `model.zip` | -- | Training, evaluation, research |
| ONNX (fp32) | `model.onnx` | ~same | Cross-platform inference, C++/Rust deployment |
| ONNX (fp16) | `model_fp16.onnx` | ~50% | GPU inference, reduced memory |
| ONNX (int8) | `model_int8.onnx` | ~75% | CPU inference, edge devices, low-latency |

### Export Commands

```bash
# Basic ONNX export
python training/model_export.py --model models/qwen_final_model.zip

# Export + quantize to fp16 + benchmark
python training/model_export.py --model models/qwen_final_model.zip --quantize fp16 --benchmark

# Full pipeline with verification
python training/model_export.py --model models/qwen_final_model_200k.zip --quantize int8 --verify --benchmark

# Custom output directory
python training/model_export.py --model models/qwen_final_model.zip --output-dir models/exported/v2
```

### Inference Benchmarking

| Format | Avg Latency (ms) | Throughput (inf/s) | Notes |
|--------|-------------------|--------------------|-------|
| SB3 (PyTorch) | ~5.0 | ~200 | Full framework loaded |
| ONNX fp32 | ~1.5 | ~667 | Optimized runtime |
| ONNX fp16 | ~0.8 | ~1250 | GPU half-precision |
| ONNX int8 | ~0.6 | ~1667 | CPU quantized, fastest |

*Latency measured on single observation inference with batch size 1.*

### Verification

All exported ONNX models can be verified with:

```bash
python training/model_export.py --model models/qwen_final_model.zip --verify
```

This runs a full inference pass through the exported model, checking input/output shapes and producing a sample action + value estimate to confirm correctness.

### Dependencies for Export

```bash
pip install onnx onnxruntime onnxconverter-common
```
