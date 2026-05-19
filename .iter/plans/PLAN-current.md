# Financial RL Gym: Full Post-Training Pipeline Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Build small, capable financial models via post-training + RL + fine-tuning, advancing from the existing Qwen 0.5B PPO baseline to a production-ready pipeline with rigorous validation.

**Architecture:** Clean up the existing gym codebase (Phase 0), then iteratively improve the training pipeline with model size sweeps, continued pre-training, SFT, RLHF alignment, and multi-task capabilities (Phase 1-2), culminating in published research (Phase 3).

**Tech Stack:** Python, PyTorch, Gymnasium, Stable-Baselines3, HuggingFace Transformers (Qwen), TRL (for SFT/RLHF), Weights & Biases (tracking)

**Status Key:** `[ ]` = pending, `[x]` = done, `[-]` = blocked

---

## Phase 0: Codebase Cleanup & Organization

Clean up structural issues so the research pipeline has a solid foundation.

### Task 0.1: Fix Package Imports

**Status:** DONE

Fixes applied:
- Created `evaluation/__init__.py` re-exporting from `training.model_evaluator`
- Fixed root `__init__.py` to remove `RealMarketData` and `EventSimulator` (never existed)
- Fixed `data/__init__.py` to export `MarketDataGenerator`
- Fixed `training/__init__.py` to remove `HyperparameterTuner` (never existed)
- Created `risk/constraints.py` re-exporting `RiskConstraints` from `environments.base_env`
- Added missing `import warnings` in 8 files

### Task 0.2: Organize Tests

- [ ] **Step 1: Audit test files and categorize**

Current tests (13 files, all flat in `tests/`):
```
tests/test_gym.py                    - environment smoke tests
tests/test_market_microstructure.py  - LOB and execution env
tests/test_risk_management.py        - full risk integration
tests/test_risk_clean.py             - risk without unicode issues
tests/test_risk_direct.py            - direct CVaR implementation
tests/test_risk_simple.py            - simplified risk tests
tests/test_training_pipeline.py      - training pipeline
tests/test_training_simple.py        - simplified training
tests/test_real_data.py              - real market data
tests/test_qwen_advanced.py          - Qwen advanced tests
tests/test_qwen_vs_baseline.py       - Qwen vs baseline comparison
tests/test_meta_learning_validation.py - meta-learning validation
tests/quick_test.py                  - smoke test
```

Target structure:
```
tests/
  __init__.py
  quick_test.py                   (keep for convenience)
  unit/
    __init__.py
    test_gym.py                   (env creation, step, reset)
    test_market_microstructure.py  (LOB, orders, execution)
    test_risk_clean.py             (risk components)
    test_risk_direct.py            (CVaR math)
    test_risk_simple.py            (risk constraints)
  integration/
    __init__.py
    test_risk_management.py        (full risk pipeline)
    test_training_pipeline.py      (trainer pipeline)
    test_training_simple.py        (training end-to-end)
    test_real_data.py              (data sources integration)
  models/
    __init__.py
    test_qwen_advanced.py          (Qwen model tests)
    test_qwen_vs_baseline.py       (comparison tests)
    test_meta_learning_validation.py
```

- [ ] **Step 2: Create test subdirectories and `__init__.py` files**

```bash
mkdir -p tests/unit tests/integration tests/models
```

- [ ] **Step 3: Move test files to their categories**

```bash
mv tests/test_gym.py tests/unit/
mv tests/test_market_microstructure.py tests/unit/
mv tests/test_risk_clean.py tests/unit/
mv tests/test_risk_direct.py tests/unit/
mv tests/test_risk_simple.py tests/unit/
mv tests/test_risk_management.py tests/integration/
mv tests/test_training_pipeline.py tests/integration/
mv tests/test_training_simple.py tests/integration/
mv tests/test_real_data.py tests/integration/
mv tests/test_qwen_advanced.py tests/models/
mv tests/test_qwen_vs_baseline.py tests/models/
mv tests/test_meta_learning_validation.py tests/models/
```

- [ ] **Step 4: Update test internal imports**

Each moved test has `sys.path.insert(0, ...)` that points to the test's own directory. This path resolution must be updated to point to the project root instead.

Update pattern: `os.path.dirname(os.path.abspath(__file__))` should use `os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))` for deep tests, or better: just reference the installed package.

- [ ] **Step 5: Update pyproject.toml testpaths**

```toml
testpaths = [
    "tests/unit",
    "tests/integration",
    "tests/models",
]
```

- [ ] **Step 6: Run tests to verify**

Run: `cd fin-rl-gym && pytest tests/ -v --collect-only`
Expected: All 13 tests discovered in their new locations

### Task 0.3: Clean Up Scripts vs Analysis

- [ ] **Step 1: Classify each file in `scripts/`**

Current scripts:
```
scripts/setup_qwen_rl.py              -> setup script (keep in scripts/)
scripts/cleanup.py                    -> utility (keep in scripts/)
scripts/test_package.py               -> test (move to tests/)
scripts/test_package_simple.py        -> test (move to tests/)
scripts/fix_broken_imports.py         -> one-time fix (archive/)
scripts/fix_imports.py               -> one-time fix (archive/)
scripts/restore_imports.py           -> one-time fix (archive/)
scripts/final_consolidation_validation.py -> analysis (move to analysis/)
```

- [ ] **Step 2: Move consolidation validation to analysis/**

Move `scripts/final_consolidation_validation.py` to `analysis/final_consolidation_validation.py`

- [ ] **Step 3: Archive one-time fix scripts**

```bash
mkdir -p archive/import_fixes
mv scripts/fix_broken_imports.py archive/import_fixes/
mv scripts/fix_imports.py archive/import_fixes/
mv scripts/restore_imports.py archive/import_fixes/
```

- [ ] **Step 4: Move test files from scripts to tests**

```bash
mv scripts/test_package.py tests/unit/
mv scripts/test_package_simple.py tests/unit/
```

- [ ] **Step 5: Verify scripts/ only has setup + cleanup**

Should contain: `setup_qwen_rl.py`, `cleanup.py`

### Task 0.4: Deduplicate Models & Checkpoints

- [ ] **Step 1: Audit what exists**

Current model artifacts:
```
checkpoints/                      203 MB - 40 zip files (qwen_model_NNNNN_steps.zip)
checkpoints/qwen_training/        - extracted training dirs
checkpoints/extracted_model_200k/ - extracted model
checkpoints/qwen_model_200000_steps_extracted/ - extracted

models/                           19 MB
  qwen_final_model.zip
  qwen_final_model_100k.zip
  qwen_final_model_200k.zip
  qwen_best/
  seed_0/ through seed_4/ (each has best/ + checkpoints/ + logs/)
```

Issue: The 40 zip files in `checkpoints/` each contain the SAME model binary at different training steps. Only the final models per seed matter.

- [ ] **Step 2: Consolidate checkpoints**

Keep only:
```
checkpoints/
  seed_0/ ... seed_4/   (keep as-is, contains best + logs per seed)
```

Remove the 40 flat checkpoint zips from `checkpoints/` root (they're redundant with `models/seed_*/checkpoints/`).

- [ ] **Step 3: Document model lineage**

Create `models/MODEL_CATALOG.md` documenting:
- Which checkpoint corresponds to which seed
- Training config that produced each
- Performance metrics per checkpoint

### Task 0.5: Centralize Training Configuration

- [ ] **Step 1: Audit duplicated hyperparams**

Training hyperparams currently duplicated across:
- `training/train_qwen.py`
- `training/run_qwen_rl_training.py`
- `training/multi_seed_training.py`
- `analysis/ablation_study.py`
- `analysis/baseline_comparison.py`

- [ ] **Step 2: Create central config**

Create `config/training_defaults.yaml`:

```yaml
# Central training configuration
model:
  name: "Qwen/Qwen2-0.5B"
  hidden_dim: 1024
  features_dim: 256

ppo:
  learning_rate: 3.0e-4
  batch_size: 64
  n_steps: 2048
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.2
  vf_coef: 0.5
  max_grad_norm: 0.5
  kl_target: 0.015

environment:
  max_steps: 252
  num_assets: 1
  transaction_costs: 0.001
  slippage: 0.0005

training:
  total_timesteps: 200000
  n_seeds: 5
  eval_freq: 10000
  log_interval: 100
```

- [ ] **Step 3: Update all training scripts to load from config**

Modify `train_qwen.py`, `run_qwen_rl_training.py`, `multi_seed_training.py` to load from `config/training_defaults.yaml` instead of hardcoding.

- [ ] **Step 4: Verify training scripts still work**

Run: `cd fin-rl-gym && python -c "from training.train_qwen import *; print('config loaded')"`
Expected: No errors

---

## Phase 1: Better Model Training

### Task 1.1: Model Size Sweep

- [ ] **Step 1: Add model size config option**

In `config/training_defaults.yaml`, add model size variants:
```yaml
model_variants:
  - name: "Qwen/Qwen2-0.5B"
    params: 494M
  - name: "Qwen/Qwen2-1.5B"
    params: 1.54B
  - name: "Qwen/Qwen2-0.5B-Instruct"
    params: 494M (chat-tuned)
```

- [ ] **Step 2: Create sweep runner script**

Create `training/model_sweep.py` that trains each model size with identical hyperparams and logs results.

- [ ] **Step 3: Train Qwen 1.5B variant**

Run PPO training with Qwen 1.5B for 200k steps, 3 seeds. Compare performance vs 0.5B.

- [ ] **Step 4: Analyze size vs performance tradeoff**

Plot Sharpe ratio, max drawdown, training time vs model size. Find the Pareto frontier.

### Task 1.2: Extended Training

- [ ] **Step 1: Train to 500k+ timesteps**

Extend training from 200k to 500k steps. The research paper shows a meta-learning threshold at ~55k steps -- what happens at 300k, 400k, 500k?

- [ ] **Step 2: Implement change-point detection for training**

Use the existing `analysis/changepoint_analysis.py` Page-Hinkley and Bayesian detectors to automatically identify phase transitions in real-time during training.

- [ ] **Step 3: Dynamic learning rate schedule**

Implement cosine annealing with warm restarts instead of constant LR:
```python
lr_schedule = lambda f: 3e-4 * (0.5 * (1 + np.cos(np.pi * f)))
```

### Task 1.3: Hyperparameter Optimization

- [ ] **Step 1: Define search space**

```python
search_space = {
    'learning_rate': [1e-4, 3e-4, 1e-3],
    'ent_coef': [0.01, 0.05, 0.1, 0.2],
    'clip_range': [0.1, 0.2, 0.3],
    'n_steps': [1024, 2048, 4096],
    'batch_size': [32, 64, 128],
    'kl_target': [0.005, 0.01, 0.015, 0.02],
}
```

- [ ] **Step 2: Create Optuna-based hyperparameter tuner**

Create `training/hyperparameter_tuner.py` using Optuna with pruning based on early Sharpe ratio.

- [ ] **Step 3: Run 50-trial hyperparameter search**

Run 50 trials, 2 seeds each (100 runs total). Log to Weights & Biases or local CSV.

- [ ] **Step 4: Analyze results, find optimal config**

Identify top 3 configurations. Validate with 5 seeds each.

### Task 1.4: Curriculum Learning

- [ ] **Step 1: Design curriculum stages**

Stage 1: Low volatility, no costs, single asset (0-25k steps)
Stage 2: Medium volatility, low costs, single asset (25-75k steps)
Stage 3: High volatility, medium costs, 2 assets (75-150k steps)
Stage 4: All regimes, full costs, 3+ assets (150k+ steps)

- [ ] **Step 2: Implement CurriculumWrapper**

Create `training/curriculum.py` with a wrapper that changes environment parameters based on training progress.

- [ ] **Step 3: Train with curriculum**

Run 5 seeds with curriculum, compare against 5 seeds without curriculum.

### Task 1.5: Multi-Asset Training

- [ ] **Step 1: Extend observation space for N assets**

The current `SingleAssetTradingEnv` supports one asset. Extend to configurable N via `PortfolioOptimizationEnv`.

- [ ] **Step 2: Train on 3-5 asset portfolio**

Train Qwen 0.5B on AAPL, MSFT, GOOGL, AMZN portfolio with correlation dynamics.

- [ ] **Step 3: Evaluate cross-asset generalization**

Test trained model on unseen assets (NVDA, TSLA, JPM) without retraining.

---

## Phase 2: Full Post-Training Pipeline

### Task 2.1: Financial Continued Pre-Training

- [ ] **Step 1: Build financial text dataset**

Collect/cache: SEC filings (10-K, 10-Q), earnings call transcripts, financial news headlines.
Target: 1B+ tokens of financial text.

- [ ] **Step 2: Continued pre-training script**

Use HuggingFace `Trainer` with masked language modeling on Qwen 0.5B for 1 epoch over financial corpus.

- [ ] **Step 3: Evaluate domain adaptation**

Compare perplexity on financial held-out text between vanilla Qwen and finance-adapted Qwen.

### Task 2.2: Supervised Fine-Tuning on Expert Trajectories

- [ ] **Step 1: Generate expert trajectories**

Use rule-based strategies (mean-reversion, momentum, trend-following) to generate 10k+ trading episodes with high Sharpe.

- [ ] **Step 2: Format as supervised dataset**

Convert trajectories to (observation -> action) pairs for behavioral cloning.

- [ ] **Step 3: SFT with HuggingFace TRL**

```python
from trl import SFTTrainer
trainer = SFTTrainer(
    model=base_model,
    train_dataset=expert_dataset,
    dataset_text_field="text",
    max_seq_length=512,
)
trainer.train()
```

- [ ] **Step 4: Compare BC vs pure RL**

Evaluate SFT + PPO vs pure PPO vs pure BC on out-of-sample data.

### Task 2.3: Reward Model for Trading Quality

- [ ] **Step 1: Design reward dimensions**

Sharpe ratio, max drawdown, win rate, trade frequency, position concentration. Combine into scalar reward.

- [ ] **Step 2: Generate preference pairs**

Sample trajectories from the SFT model, rank by composite score, create preference dataset.

- [ ] **Step 3: Train reward model**

```python
from trl import RewardTrainer
# Train a Qwen model to predict reward from observation + action history
```

### Task 2.4: RLHF-Style Alignment

- [ ] **Step 1: PPO against reward model**

Use TRL's `PPOTrainer` with the trained reward model as the reward signal instead of environment reward.

- [ ] **Step 2: Compare reward sources**

Environment reward (original) vs reward model vs ensemble (both).

- [ ] **Step 3: Analyze alignment tax**

Does optimizing for the reward model reduce environment reward? Measure the gap.

### Task 2.5: Multi-Task Agent

- [ ] **Step 1: Define task heads**

Create a single Qwen model with multiple output heads:
- Trading action head (continuous [-1, 1])
- Regime classification head (bull/bear/volatile)
- Risk assessment head (VaR prediction)
- Portfolio weight head (for multi-asset)

- [ ] **Step 2: Multi-task training**

Alternate between tasks during training, sharing the transformer backbone.

- [ ] **Step 3: Evaluate cross-task transfer**

Does regime detection accuracy improve trading performance? Does trading improve regime detection?

---

## Phase 3: Research & Publication

### Task 3.1: Full Ablation Study

- [ ] **Step 1: Define ablation dimensions**

1. Model size (0.5B vs 1.5B)
2. Feature extractor (Qwen vs MLP)
3. RL algorithm (PPO vs SAC vs A2C)
4. Reward structure (simple vs CVaR)
5. Training length (50k vs 100k vs 200k vs 500k)
6. Curriculum vs flat training

- [ ] **Step 2: Run 48 ablation experiments**

3 seeds each, 48 configs = 144 runs. Use grid/automation to parallelize.

- [ ] **Step 3: Statistical analysis**

```python
# Bootstrap confidence intervals
from analysis.confidence_intervals_analysis import bootstrap_ci
# Cohen's d effect sizes
from analysis.effect_size_validation import cohens_d
```

- [ ] **Step 4: Generate ablation plots**

Performance waterfall chart showing contribution of each component.

### Task 3.2: Final Paper Write-Up

- [ ] **Step 1: Update LaTeX paper**

Merge findings from all phases into `research_paper/financial_trading_rl_gym.tex`

- [ ] **Step 2: Generate result figures**

All figures: training curves, ablation bars, regime heatmaps, portfolio equity curves.

- [ ] **Step 3: Supplementary materials**

Detailed hyperparameter tables, full statistical results, environment documentation.

### Task 3.3: Model Export & Deployment

- [ ] **Step 1: Export to ONNX**

```python
torch.onnx.export(model, dummy_input, "model.onnx")
```

- [ ] **Step 2: Quantize to 4-bit**

Use GPTQ or AWQ quantization for inference on CPU/edge devices.

- [ ] **Step 3: Model card**

Create `models/MODEL_CARD.md` with performance, limitations, intended use.

---

## Dependency Graph

```
Phase 0 (Cleanup)
  Task 0.1 - Package imports    [DONE]
  Task 0.2 - Organize tests     [NEXT]
  Task 0.3 - Scripts cleanup    [after 0.2]
  Task 0.4 - Model dedup        [parallel with 0.3]
  Task 0.5 - Central config     [after 0.2, blocks Phase 1]

Phase 1 (Better Training)
  Task 1.1 - Model sweep        [after 0.5]
  Task 1.2 - Extended training  [after 0.5]
  Task 1.3 - HP optimization    [after 1.1]
  Task 1.4 - Curriculum         [parallel with 1.3]
  Task 1.5 - Multi-asset        [after 1.2]

Phase 2 (Post-Training Pipeline)
  Task 2.1 - Financial PT       [after 1.1]
  Task 2.2 - SFT                [after 1.1]
  Task 2.3 - Reward model       [after 2.2]
  Task 2.4 - RLHF alignment     [after 2.3]
  Task 2.5 - Multi-task agent   [after 2.4, parallel with 1.5]

Phase 3 (Publication)
  Task 3.1 - Ablation study     [after Phase 1+2]
  Task 3.2 - Paper write-up     [after 3.1]
  Task 3.3 - Model deployment   [after Phase 1+2]
```
