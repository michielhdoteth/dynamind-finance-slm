"""
Multi-Task Agent: Shared Backbone for Trading + Regime Detection + Risk Assessment.

Architecture:
    Shared Qwen backbone -> 3 task-specific heads:
        - Trading head:     continuous action [-1, 1] (position weight)
        - Regime head:      classification (bull/bear/volatile/sideways)
        - Risk head:        VaR prediction (regression)

The multi-task training loop alternates between tasks with configurable
loss weights, enabling the shared backbone to learn representations useful
across all three financial tasks.

Usage:
    python training/multi_task_agent.py --mode train
    python training/multi_task_agent.py --mode evaluate
    python training/multi_task_agent.py --mode predict
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.loader import load_config, get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MultiTaskConfig:
    """Configuration for the multi-task trading agent."""

    model_name: str = "Qwen/Qwen2-0.5B"
    hidden_dim: int = 256
    max_length: int = 128
    batch_size: int = 16
    learning_rate: float = 3e-5
    num_epochs: int = 20
    output_dir: str = "./checkpoints/multi_task"

    # Task weights for multi-task loss
    task_weights: Dict[str, float] = field(default_factory=lambda: {
        "trading": 1.0,
        "regime": 0.5,
        "risk": 0.3,
    })

    # Per-task config
    num_regime_classes: int = 4  # bull, bear, volatile, sideways
    trading_action_dim: int = 1   # continuous action [-1, 1]

    # Training
    warmup_ratio: float = 0.05
    save_steps: int = 200
    logging_steps: int = 10
    save_total_limit: int = 3
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    eval_split_ratio: float = 0.15

    # Data
    num_samples: int = 1000
    seed: int = 42

    @classmethod
    def from_yaml(cls, overrides: Optional[Dict] = None) -> "MultiTaskConfig":
        cfg = load_config()
        mt_cfg = cfg.get("multi_task", {})
        kwargs = dict(
            model_name=mt_cfg.get("model_name", cls.model_name),
            hidden_dim=mt_cfg.get("hidden_dim", cls.hidden_dim),
            learning_rate=mt_cfg.get("learning_rate", cls.learning_rate),
            num_epochs=mt_cfg.get("num_epochs", cls.num_epochs),
            batch_size=mt_cfg.get("batch_size", cls.batch_size),
            output_dir=mt_cfg.get("output_dir", cls.output_dir),
        )
        if "task_weights" in mt_cfg and isinstance(mt_cfg["task_weights"], dict):
            kwargs["task_weights"] = {
                **cls.task_weights,
                **mt_cfg["task_weights"],
            }
        if overrides:
            kwargs.update(overrides)
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Multi-Task Model
# ---------------------------------------------------------------------------

class MultiTaskQwenAgent(nn.Module):
    """Shared Qwen backbone with three task-specific heads.

    Heads:
        - Trading:  continuous action prediction (tanh -> [-1, 1])
        - Regime:   market regime classification (softmax -> 4 classes)
        - Risk:     Value-at-Risk regression (positive output)
    """

    REGIME_LABELS = ["bull", "bear", "volatile", "sideways"]

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-0.5B",
        hidden_dim: int = 256,
        num_regime_classes: int = 4,
        trading_action_dim: int = 1,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_regime_classes = num_regime_classes
        self.trading_action_dim = trading_action_dim

        try:
            from transformers import AutoModel, AutoConfig
        except ImportError:
            raise ImportError("transformers package required.")

        logger.info("Loading Qwen backbone from %s", model_name)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen.")
        else:
            logger.info("Backbone will be fine-tuned.")

        encoder_dim = getattr(config, "hidden_size", 1024)
        logger.info("Encoder hidden size: %d", encoder_dim)

        # --- Task Heads ---
        # 1. Trading head: continuous action in [-1, 1]
        self.trading_head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, trading_action_dim),
            nn.Tanh(),
        )

        # 2. Regime head: classification
        self.regime_head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_regime_classes),
        )

        # 3. Risk head: VaR regression (positive via softplus)
        self.risk_head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

        self._init_heads()

    def _init_heads(self):
        """Initialize head weights."""
        for head in [self.trading_head, self.regime_head, self.risk_head]:
            for module in head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def _pool(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool encoder output to a single vector per sample.

        Uses mean pooling over non-padded tokens.
        """
        mask = attention_mask.unsqueeze(-1).float()  # [batch, seq, 1]
        masked = last_hidden * mask
        pooled = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return pooled

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        task: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: Tokenized inputs [batch, seq_len].
            attention_mask: Attention mask [batch, seq_len].
            task: If provided, only compute that head's output.
                  If None, compute all heads.

        Returns:
            dict with keys: 'trading_action', 'regime_logits', 'risk_var'.
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = self._pool(outputs.last_hidden_state, attention_mask)

        result = {}
        if task is None or task == "trading":
            result["trading_action"] = self.trading_head(pooled)
        if task is None or task == "regime":
            result["regime_logits"] = self.regime_head(pooled)
        if task is None or task == "risk":
            result["risk_var"] = self.risk_head(pooled)

        return result

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """Run full forward and return numpy predictions for all tasks."""
        self.eval()
        outputs = self.forward(input_ids, attention_mask)
        result = {}
        if "trading_action" in outputs:
            result["trading_action"] = outputs["trading_action"].cpu().numpy()
        if "regime_logits" in outputs:
            probs = F.softmax(outputs["regime_logits"], dim=-1)
            result["regime_probs"] = probs.cpu().numpy()
            result["regime_pred"] = probs.argmax(dim=-1).cpu().numpy()
        if "risk_var" in outputs:
            result["risk_var"] = outputs["risk_var"].cpu().numpy()
        return result


# ---------------------------------------------------------------------------
# Multi-Task Dataset
# ---------------------------------------------------------------------------

class MultiTaskDataset:
    """Generates synthetic multi-task financial data.

    Each sample includes:
        - market_state_text: description of current market conditions
        - trading_action: optimal position weight [-1, 1]
        - regime_label: market regime class index
        - risk_var: Value-at-Risk (positive float)
    """

    REGIME_LABELS = ["bull", "bear", "volatile", "sideways"]

    def __init__(self, num_samples: int = 1000, seed: int = 42):
        self.num_samples = num_samples
        self.rng = np.random.default_rng(seed)
        self._generate()

    def _generate(self):
        """Generate the multi-task dataset."""
        self.samples = []

        for _ in range(self.num_samples):
            # Random market parameters
            volatility = self.rng.uniform(0.005, 0.05)
            trend = self.rng.uniform(-0.003, 0.003)
            current_price = self.rng.uniform(50, 500)
            sma_10 = current_price * (1 + self.rng.uniform(-0.05, 0.05))
            sma_30 = current_price * (1 + self.rng.uniform(-0.08, 0.08))
            rsi = self.rng.uniform(20, 80)

            # Determine regime
            if trend > 0.001 and volatility < 0.025:
                regime = 0  # bull
            elif trend < -0.001 and volatility < 0.025:
                regime = 1  # bear
            elif volatility > 0.03:
                regime = 2  # volatile
            else:
                regime = 3  # sideways

            # Trading action: momentum-based
            if regime == 0:  # bull
                trading_action = self.rng.uniform(0.3, 1.0)
            elif regime == 1:  # bear
                trading_action = self.rng.uniform(-1.0, -0.3)
            elif regime == 2:  # volatile
                trading_action = self.rng.uniform(-0.3, 0.3)
            else:  # sideways
                trading_action = self.rng.uniform(-0.2, 0.2)

            # VaR: proportional to volatility
            var_95 = round(float(self.rng.uniform(0.01, 0.08) * (1 + volatility * 10)), 6)

            # Build market state text
            market_state_text = (
                f"Price=${current_price:.2f}, SMA_10=${sma_10:.2f}, SMA_30=${sma_30:.2f}, "
                f"RSI={rsi:.1f}, volatility={volatility:.3f}, trend={trend:+.5f}"
            )

            self.samples.append({
                "text": market_state_text,
                "trading_action": float(trading_action),
                "regime_label": int(regime),
                "risk_var": var_95,
            })

        logger.info("Generated %d multi-task samples", len(self.samples))
        logger.info("Regime distribution: %s", {
            label: sum(1 for s in self.samples if s["regime_label"] == i)
            for i, label in enumerate(self.REGIME_LABELS)
        })

    def to_dict(self) -> Dict[str, List]:
        """Convert to dictionary of lists for HuggingFace Dataset."""
        return {
            key: [s[key] for s in self.samples]
            for key in ["text", "trading_action", "regime_label", "risk_var"]
        }


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------

def prepare_multi_task_dataset(
    config: MultiTaskConfig,
) -> Tuple["Dataset", "Dataset", "AutoTokenizer"]:
    """Prepare tokenized multi-task datasets.

    Returns:
        tuple: (train_dataset, eval_dataset, tokenizer)
    """
    from transformers import AutoTokenizer
    from datasets import Dataset

    logger.info("Loading tokenizer: %s", config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Generate data
    generator = MultiTaskDataset(
        num_samples=config.num_samples,
        seed=config.seed,
    )
    raw_dataset = Dataset.from_dict(generator.to_dict())
    logger.info("Generated %d raw samples", len(raw_dataset))

    # Split
    split = raw_dataset.train_test_split(
        test_size=config.eval_split_ratio,
        seed=config.seed,
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=config.max_length,
        )

    train_dataset = split["train"].map(tokenize_function, batched=True)
    eval_dataset = split["test"].map(tokenize_function, batched=True)

    logger.info("Train: %d | Eval: %d samples", len(train_dataset), len(eval_dataset))
    return train_dataset, eval_dataset, tokenizer


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def compute_multi_task_loss(
    model: MultiTaskQwenAgent,
    batch: Dict[str, torch.Tensor],
    config: MultiTaskConfig,
) -> Dict[str, torch.Tensor]:
    """Compute losses for all tasks.

    Args:
        model: The multi-task agent.
        batch: Dict with 'input_ids', 'attention_mask', 'trading_action',
               'regime_label', 'risk_var'.
        config: Training configuration.

    Returns:
        dict with 'loss', 'trading_loss', 'regime_loss', 'risk_loss'.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    # Forward pass (all heads)
    outputs = model(input_ids, attention_mask)

    losses = {}

    # --- Trading Loss (MSE) ---
    if "trading_action" in outputs:
        target = batch["trading_action"].float().unsqueeze(-1)
        trading_loss = F.mse_loss(outputs["trading_action"], target)
        losses["trading_loss"] = trading_loss
    else:
        trading_loss = torch.tensor(0.0, device=input_ids.device)
        losses["trading_loss"] = trading_loss

    # --- Regime Loss (Cross-Entropy) ---
    if "regime_logits" in outputs:
        target = batch["regime_label"].long()
        regime_loss = F.cross_entropy(outputs["regime_logits"], target)
        losses["regime_loss"] = regime_loss
    else:
        regime_loss = torch.tensor(0.0, device=input_ids.device)
        losses["regime_loss"] = regime_loss

    # --- Risk Loss (MSE on log scale for positive targets) ---
    if "risk_var" in outputs:
        target = batch["risk_var"].float().unsqueeze(-1)
        risk_loss = F.mse_loss(outputs["risk_var"], target)
        losses["risk_loss"] = risk_loss
    else:
        risk_loss = torch.tensor(0.0, device=input_ids.device)
        losses["risk_loss"] = risk_loss

    # Weighted total loss
    total_loss = (
        config.task_weights.get("trading", 1.0) * trading_loss
        + config.task_weights.get("regime", 0.5) * regime_loss
        + config.task_weights.get("risk", 0.3) * risk_loss
    )
    losses["loss"] = total_loss

    return losses


def train_multi_task(config: MultiTaskConfig) -> MultiTaskQwenAgent:
    """Run multi-task training."""
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from torch.utils.data import DataLoader

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Prepare data
    train_dataset, eval_dataset, _ = prepare_multi_task_dataset(config)

    # Build model
    model = MultiTaskQwenAgent(
        model_name=config.model_name,
        hidden_dim=config.hidden_dim,
        num_regime_classes=config.num_regime_classes,
        trading_action_dim=config.trading_action_dim,
        freeze_backbone=False,
    )
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Scheduler
    total_steps = len(train_dataset) * config.num_epochs // config.batch_size
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Dataloader
    def collate_fn(batch):
        return {
            "input_ids": torch.tensor([b["input_ids"] for b in batch], dtype=torch.long),
            "attention_mask": torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long),
            "trading_action": torch.tensor([b["trading_action"] for b in batch], dtype=torch.float),
            "regime_label": torch.tensor([b["regime_label"] for b in batch], dtype=torch.long),
            "risk_var": torch.tensor([b["risk_var"] for b in batch], dtype=torch.float),
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Training loop
    global_step = 0
    best_eval_loss = float("inf")
    history = defaultdict(list)

    for epoch in range(config.num_epochs):
        model.train()
        epoch_losses = defaultdict(float)
        epoch_steps = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            losses = compute_multi_task_loss(model, batch, config)

            optimizer.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            for key, val in losses.items():
                epoch_losses[key] += val.item()
            epoch_steps += 1
            global_step += 1

            if global_step % config.logging_steps == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    "Epoch %d/%d | Step %d | loss=%.4f | trading=%.4f | "
                    "regime=%.4f | risk=%.4f | lr=%.2e",
                    epoch + 1, config.num_epochs, global_step,
                    losses["loss"].item(),
                    losses["trading_loss"].item(),
                    losses["regime_loss"].item(),
                    losses["risk_loss"].item(),
                    lr,
                )

            # Evaluation
            if global_step % config.save_steps == 0:
                eval_metrics = evaluate_on_loader(model, eval_loader, device, config)
                logger.info("Eval at step %d: %s", global_step, json.dumps(eval_metrics, indent=2))
                if eval_metrics["eval_loss"] < best_eval_loss:
                    best_eval_loss = eval_metrics["eval_loss"]
                    _save_checkpoint(model, config.output_dir, "best")
                    logger.info("New best model (loss: %.4f)", best_eval_loss)

                # Log to history
                for k, v in eval_metrics.items():
                    history[k].append(v)
                history["step"].append(global_step)

        # End of epoch
        avg = {k: v / max(epoch_steps, 1) for k, v in epoch_losses.items()}
        logger.info(
            "Epoch %d complete. Avg: %s",
            epoch + 1,
            json.dumps({k: round(v, 4) for k, v in avg.items()}),
        )

    # Save final
    _save_checkpoint(model, config.output_dir, "final")
    with open(os.path.join(config.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Training complete. Best eval loss: %.4f", best_eval_loss)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_on_loader(
    model: MultiTaskQwenAgent,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: MultiTaskConfig,
) -> Dict[str, float]:
    """Evaluate multi-task model on a dataloader.

    Returns dict with loss and per-task metrics.
    """
    model.eval()
    total_losses = defaultdict(float)
    total = 0

    # For accuracy metrics
    all_trading_true = []
    all_trading_pred = []
    all_regime_true = []
    all_regime_pred = []
    all_risk_true = []
    all_risk_pred = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        losses = compute_multi_task_loss(model, batch, config)

        for key, val in losses.items():
            total_losses[key] += val.item()
        total += 1

        # Collect predictions for metrics
        outputs = model(batch["input_ids"], batch["attention_mask"])
        all_trading_true.extend(batch["trading_action"].cpu().numpy().flatten())
        all_trading_pred.extend(outputs["trading_action"].cpu().numpy().flatten())
        all_regime_true.extend(batch["regime_label"].cpu().numpy())
        all_regime_pred.extend(outputs["regime_logits"].argmax(dim=-1).cpu().numpy())
        all_risk_true.extend(batch["risk_var"].cpu().numpy().flatten())
        all_risk_pred.extend(outputs["risk_var"].cpu().numpy().flatten())

    avg = {k: v / max(total, 1) for k, v in total_losses.items()}

    # Per-task metrics
    metrics = {
        "eval_loss": round(avg.get("loss", 0), 6),
        "trading_loss": round(avg.get("trading_loss", 0), 6),
        "regime_loss": round(avg.get("regime_loss", 0), 6),
        "risk_loss": round(avg.get("risk_loss", 0), 6),
    }

    # Trading: mean absolute error
    if all_trading_true and all_trading_pred:
        mae = np.mean(np.abs(np.array(all_trading_true) - np.array(all_trading_pred)))
        metrics["trading_mae"] = round(float(mae), 4)

    # Regime: accuracy
    if all_regime_true and all_regime_pred:
        acc = np.mean(np.array(all_regime_true) == np.array(all_regime_pred))
        metrics["regime_accuracy"] = round(float(acc), 4)

    # Risk: MAPE (error relative to true value)
    if all_risk_true and all_risk_pred:
        true_arr = np.array(all_risk_true)
        pred_arr = np.array(all_risk_pred)
        mape = np.mean(np.abs((true_arr - pred_arr) / (true_arr + 1e-8)))
        metrics["risk_mape"] = round(float(mape), 4)

    return metrics


def evaluate_multi_task(config: MultiTaskConfig) -> Dict[str, float]:
    """Load trained model and evaluate on held-out data."""
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    load_dir = os.path.join(config.output_dir, "best")
    logger.info("Loading model from %s", load_dir)
    model = MultiTaskQwenAgent(
        model_name=config.model_name,
        hidden_dim=config.hidden_dim,
        num_regime_classes=config.num_regime_classes,
        trading_action_dim=config.trading_action_dim,
        freeze_backbone=False,
    )
    # Load weights
    model_path = os.path.join(load_dir, "model.pt")
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        logger.info("Loaded weights from %s", model_path)
    else:
        logger.warning("No trained weights found at %s. Using untrained model.", model_path)

    model = model.to(device)
    model.eval()

    # Prepare eval dataset
    _, eval_dataset, _ = prepare_multi_task_dataset(config)

    def collate_fn(batch):
        return {
            "input_ids": torch.tensor([b["input_ids"] for b in batch], dtype=torch.long),
            "attention_mask": torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long),
            "trading_action": torch.tensor([b["trading_action"] for b in batch], dtype=torch.float),
            "regime_label": torch.tensor([b["regime_label"] for b in batch], dtype=torch.long),
            "risk_var": torch.tensor([b["risk_var"] for b in batch], dtype=torch.float),
        }

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    metrics = evaluate_on_loader(model, eval_loader, device, config)
    logger.info("Evaluation results: %s", json.dumps(metrics, indent=2))
    return metrics


# ---------------------------------------------------------------------------
# Prediction (Single Sample)
# ---------------------------------------------------------------------------

def predict_multi_task(config: MultiTaskConfig, text: str) -> Dict:
    """Predict trading action, regime, and VaR for a single market state text."""
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    load_dir = os.path.join(config.output_dir, "best")
    model = MultiTaskQwenAgent(
        model_name=config.model_name,
        hidden_dim=config.hidden_dim,
        num_regime_classes=config.num_regime_classes,
        trading_action_dim=config.trading_action_dim,
        freeze_backbone=False,
    )
    model_path = os.path.join(load_dir, "model.pt")
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        load_dir if os.path.exists(load_dir) else config.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=config.max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    outputs = model.predict(input_ids, attention_mask)

    result = {
        "input_text": text,
        "trading_action": float(outputs.get("trading_action", [0])[0][0]),
        "regime_prediction": MultiTaskQwenAgent.REGIME_LABELS[
            int(outputs.get("regime_pred", [0])[0])
        ],
        "regime_probs": {
            label: float(prob)
            for label, prob in zip(
                MultiTaskQwenAgent.REGIME_LABELS,
                outputs.get("regime_probs", [0])[0],
            )
        },
        "risk_var_95": float(outputs.get("risk_var", [0])[0][0]),
    }
    return result


# ---------------------------------------------------------------------------
# Saving / Loading Helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(model: MultiTaskQwenAgent, output_dir: str, suffix: str):
    """Save model checkpoint."""
    save_dir = os.path.join(output_dir, suffix)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    config_dict = {
        "model_name": model.model_name,
        "num_regime_classes": model.num_regime_classes,
        "trading_action_dim": model.trading_action_dim,
    }
    with open(os.path.join(save_dir, "multi_task_config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info("Checkpoint saved to %s", save_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-task agent: shared backbone for trading + regime + risk."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "predict"],
        required=True,
        help="Pipeline stage.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2-0.5B",
        help="HuggingFace model name.",
    )
    parser.add_argument(
        "--output-dir",
        default="./checkpoints/multi_task",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-5,
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of synthetic training samples.",
    )
    parser.add_argument(
        "--predict-text",
        type=str,
        default="",
        help="Market state text for prediction mode.",
    )
    args = parser.parse_args()

    overrides = {
        "model_name": args.model,
        "output_dir": args.output_dir,
        "hidden_dim": args.hidden_dim,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "num_samples": args.num_samples,
    }

    config = MultiTaskConfig.from_yaml(overrides)
    logger.info("Config: %s", json.dumps(vars(config), indent=2, default=str))

    if args.mode == "train":
        train_multi_task(config)
    elif args.mode == "evaluate":
        evaluate_multi_task(config)
    elif args.mode == "predict":
        if not args.predict_text:
            # Use a default sample
            args.predict_text = (
                "Price=$150.25, SMA_10=$148.10, SMA_30=$145.80, "
                "RSI=62.5, volatility=0.018, trend=+0.0002"
            )
        result = predict_multi_task(config, args.predict_text)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
