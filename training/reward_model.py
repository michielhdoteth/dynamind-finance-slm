"""
Reward Model for Trading Quality Prediction.

Learns to predict a scalar trading quality score from trajectory data
using a pairwise ranking loss (Bradley-Terry). The model uses a shared
Qwen encoder with a linear regression head.

This reward model can then be used in RLHF alignment to provide
dense reward signals for policy optimization.

Usage:
    python training/reward_model.py --mode prepare
    python training/reward_model.py --mode train
    python training/reward_model.py --mode evaluate
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import warnings
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
class RewardModelConfig:
    """Configuration for the reward model."""

    model_name: str = "Qwen/Qwen2-0.5B"
    hidden_dim: int = 256
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 1e-5
    num_epochs: int = 5
    output_dir: str = "./checkpoints/reward_model"
    num_trajectories: int = 200
    trajectory_length: int = 50
    pairs_per_trajectory: int = 5  # preference pairs per trajectory
    warmup_ratio: float = 0.1
    save_steps: int = 200
    logging_steps: int = 10
    save_total_limit: int = 3
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    eval_split_ratio: float = 0.15
    seed: int = 42

    @classmethod
    def from_yaml(cls, overrides: Optional[Dict] = None) -> "RewardModelConfig":
        cfg = load_config()
        rm_cfg = cfg.get("reward_model", {})
        kwargs = dict(
            model_name=rm_cfg.get("model_name", cls.model_name),
            hidden_dim=rm_cfg.get("hidden_dim", cls.hidden_dim),
            learning_rate=rm_cfg.get("learning_rate", cls.learning_rate),
            num_epochs=rm_cfg.get("num_epochs", cls.num_epochs),
            batch_size=rm_cfg.get("batch_size", cls.batch_size),
            output_dir=rm_cfg.get("output_dir", cls.output_dir),
        )
        if overrides:
            kwargs.update(overrides)
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Reward Model Architecture
# ---------------------------------------------------------------------------

class RewardModel(nn.Module):
    """Reward model: shared Qwen encoder + linear head -> scalar reward.

    Architecture:
        Qwen backbone (frozen or fine-tuned) -> pooled hidden state ->
        MLP projection -> scalar reward
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-0.5B",
        hidden_dim: int = 256,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.model_name = model_name

        try:
            from transformers import AutoModel, AutoConfig
        except ImportError:
            raise ImportError("transformers package required.")

        logger.info("Loading Qwen encoder from %s", model_name)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )

        # Freeze encoder if desired
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder frozen.")
        else:
            logger.info("Encoder will be fine-tuned.")

        # Determine hidden size from encoder config
        encoder_dim = getattr(config, "hidden_size", 1024)
        logger.info("Encoder hidden size: %d", encoder_dim)

        # Reward head: MLP projection to scalar
        self.reward_head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize the reward head weights."""
        for module in self.reward_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass: encode input and predict scalar reward.

        Args:
            input_ids: Tokenized input [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            rewards: Scalar rewards for each item in batch [batch, 1]
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Use the last hidden state of the last token as the pooled representation
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, dim]
        # Gather last non-padded token
        if attention_mask is not None:
            last_token_indices = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            pooled = hidden_states[batch_indices, last_token_indices]  # [batch, dim]
        else:
            pooled = hidden_states[:, -1, :]  # [batch, dim]

        rewards = self.reward_head(pooled)  # [batch, 1]
        return rewards

    @torch.no_grad()
    def predict_reward(
        self,
        texts: List[str],
        tokenizer: "AutoTokenizer",
        device: Optional[torch.device] = None,
        max_length: int = 512,
    ) -> np.ndarray:
        """Predict rewards for a list of text trajectories.

        Args:
            texts: List of trajectory text strings.
            tokenizer: HuggingFace tokenizer.
            device: Torch device.

        Returns:
            numpy array of predicted rewards.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.eval()
        self.to(device)

        rewards_list = []
        for text in texts:
            inputs = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            reward = self.forward(**inputs)
            rewards_list.append(reward.item())

        return np.array(rewards_list)


# ---------------------------------------------------------------------------
# Preference Pair Data Generation
# ---------------------------------------------------------------------------

class PreferencePairDataset:
    """Generates preference pairs from trajectory data for reward model training.

    A trajectory is "good" if it achieved a higher cumulative return.
    Pairs are sampled as (winner_trajectory, loser_trajectory) where
    winner has higher return.
    """

    def __init__(
        self,
        num_trajectories: int = 200,
        trajectory_length: int = 50,
        pairs_per_trajectory: int = 5,
        seed: int = 42,
    ):
        self.num_trajectories = num_trajectories
        self.trajectory_length = trajectory_length
        self.pairs_per_trajectory = pairs_per_trajectory
        self.rng = np.random.default_rng(seed)

    def _simulate_trajectory_return(self) -> Tuple[str, float]:
        """Simulate a single trajectory and compute its return."""
        # Generate price series
        prices = [100.0]
        for _ in range(self.trajectory_length):
            ret = self.rng.normal(0.0003, 0.02)
            prices.append(prices[-1] * (1 + ret))
        prices = np.array(prices)

        # Simulate a simple trading strategy to get action quality
        # We randomly pick a strategy type which correlates with return
        strategy_quality = self.rng.uniform(-0.5, 1.0)

        # Build trajectory text
        lines = []
        for step in range(len(prices) - 1):
            p = prices[step]
            ret_pct = (prices[step + 1] / p - 1) * 100
            sma = np.mean(prices[max(0, step - 9) : step + 1])
            lines.append(
                f"Observation: price=${p:.2f}, return={ret_pct:+.2f}%%, sma_10=${sma:.2f}"
            )
            # Decide action: buy (1) if strategy_quality > 0 and price < sma, etc.
            if strategy_quality > 0.3 and p < sma * 0.98:
                action = "Buy"
            elif strategy_quality < -0.2 and p > sma * 1.02:
                action = "Sell"
            else:
                action = "Hold"
            lines.append(f"Action: {action}")

        trajectory_text = "\n".join(lines)

        # Compute final return
        total_return = (prices[-1] / prices[0] - 1) * strategy_quality

        return trajectory_text, total_return

    def generate_pairs(self) -> List[Dict[str, any]]:
        """Generate preference pairs.

        Each pair: (chosen_text, rejected_text) where chosen has higher return.

        Returns:
            List of dicts with 'chosen', 'rejected', 'chosen_return', 'rejected_return'.
        """
        trajectories = []
        for _ in range(self.num_trajectories):
            text, ret = self._simulate_trajectory_return()
            trajectories.append({"text": text, "return": ret})

        pairs = []
        for _ in range(self.pairs_per_trajectory * self.num_trajectories // 2):
            # Sample two random trajectories
            i, j = self.rng.integers(0, len(trajectories), size=2)
            if i == j:
                continue
            t1, t2 = trajectories[i], trajectories[j]
            if t1["return"] == t2["return"]:
                continue

            if t1["return"] > t2["return"]:
                chosen, rejected = t1, t2
            else:
                chosen, rejected = t2, t1

            pairs.append({
                "chosen": chosen["text"],
                "rejected": rejected["text"],
                "chosen_return": chosen["return"],
                "rejected_return": rejected["return"],
            })

        logger.info("Generated %d preference pairs from %d trajectories", len(pairs), len(trajectories))
        return pairs

    def to_hf_dataset(self) -> "Dataset":
        """Convert pairs to a HuggingFace Dataset.

        Each sample has 'chosen' and 'rejected' text fields.
        """
        try:
            from datasets import Dataset
        except ImportError:
            raise ImportError("datasets package required. Install with: pip install datasets")

        pairs = self.generate_pairs()
        return Dataset.from_list(pairs)


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------

def prepare_reward_dataset(
    config: RewardModelConfig,
) -> Tuple["Dataset", "Dataset", "AutoTokenizer"]:
    """Prepare preference pair dataset for reward model training.

    Returns:
        tuple: (train_dataset, eval_dataset, tokenizer)
    """
    from transformers import AutoTokenizer

    logger.info("Loading tokenizer: %s", config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Generate preference pairs
    generator = PreferencePairDataset(
        num_trajectories=config.num_trajectories,
        trajectory_length=config.trajectory_length,
        pairs_per_trajectory=config.pairs_per_trajectory,
        seed=config.seed,
    )
    raw_dataset = generator.to_hf_dataset()
    logger.info("Generated %d raw preference pairs", len(raw_dataset))

    # Split
    split = raw_dataset.train_test_split(
        test_size=config.eval_split_ratio,
        seed=config.seed,
    )

    def tokenize_pair(examples):
        """Tokenize both chosen and rejected texts."""
        chosen_enc = tokenizer(
            examples["chosen"],
            truncation=True,
            padding="max_length",
            max_length=config.max_length,
        )
        rejected_enc = tokenizer(
            examples["rejected"],
            truncation=True,
            padding="max_length",
            max_length=config.max_length,
        )
        return {
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
        }

    train_dataset = split["train"].map(tokenize_pair, batched=True)
    eval_dataset = split["test"].map(tokenize_pair, batched=True)

    logger.info("Train: %d | Eval: %d pairs", len(train_dataset), len(eval_dataset))
    return train_dataset, eval_dataset, tokenizer


# ---------------------------------------------------------------------------
# Training (Pairwise Ranking Loss - Bradley-Terry)
# ---------------------------------------------------------------------------

def compute_ranking_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
) -> torch.Tensor:
    """Bradley-Terry pairwise ranking loss.

    Loss = -log(sigmoid(reward_chosen - reward_rejected))

    Args:
        chosen_rewards: [batch, 1] rewards for preferred trajectories
        rejected_rewards: [batch, 1] rewards for dispreferred trajectories

    Returns:
        Scalar loss.
    """
    logits = chosen_rewards - rejected_rewards  # [batch, 1]
    loss = -F.logsigmoid(logits).mean()
    return loss


def train_reward_model(config: RewardModelConfig) -> Tuple[RewardModel, "AutoTokenizer"]:
    """Train the reward model using pairwise ranking."""
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from torch.utils.data import DataLoader

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Prepare data
    train_dataset, eval_dataset, tokenizer = prepare_reward_dataset(config)

    # Build model
    model = RewardModel(
        model_name=config.model_name,
        hidden_dim=config.hidden_dim,
        freeze_encoder=True,
    )
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.reward_head.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Scheduler
    total_steps = len(train_dataset) * config.num_epochs // config.batch_size
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Dataloader helper
    def collate_fn(batch):
        return {
            "chosen_input_ids": torch.tensor([b["chosen_input_ids"] for b in batch], dtype=torch.long),
            "chosen_attention_mask": torch.tensor([b["chosen_attention_mask"] for b in batch], dtype=torch.long),
            "rejected_input_ids": torch.tensor([b["rejected_input_ids"] for b in batch], dtype=torch.long),
            "rejected_attention_mask": torch.tensor([b["rejected_attention_mask"] for b in batch], dtype=torch.long),
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

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in train_loader:
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            # Forward
            chosen_rewards = model(input_ids=chosen_ids, attention_mask=chosen_mask)
            rejected_rewards = model(input_ids=rejected_ids, attention_mask=rejected_mask)

            loss = compute_ranking_loss(chosen_rewards, rejected_rewards)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            if global_step % config.logging_steps == 0:
                logger.info(
                    "Epoch %d/%d | Step %d | Loss: %.4f | LR: %.2e",
                    epoch + 1, config.num_epochs, global_step, loss.item(),
                    scheduler.get_last_lr()[0],
                )

            # Evaluation
            if global_step % config.save_steps == 0:
                eval_loss = evaluate_on_loader(model, eval_loader, device)
                logger.info("Eval loss at step %d: %.4f", global_step, eval_loss)
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    _save_checkpoint(model, tokenizer, config.output_dir, "best")
                    logger.info("New best model saved (loss: %.4f)", eval_loss)

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        logger.info("Epoch %d complete. Avg loss: %.4f", epoch + 1, avg_epoch_loss)

    # Final save
    _save_checkpoint(model, tokenizer, config.output_dir, "final")
    logger.info("Training complete. Best eval loss: %.4f", best_eval_loss)

    return model, tokenizer


def evaluate_on_loader(
    model: RewardModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Evaluate reward model on a dataloader."""
    model.eval()
    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for batch in loader:
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            chosen_rewards = model(input_ids=chosen_ids, attention_mask=chosen_mask)
            rejected_rewards = model(input_ids=rejected_ids, attention_mask=rejected_mask)

            loss = compute_ranking_loss(chosen_rewards, rejected_rewards)
            total_loss += loss.item()
            total += 1

    avg_loss = total_loss / max(total, 1)
    model.train()
    return avg_loss


def _save_checkpoint(
    model: RewardModel,
    tokenizer: "AutoTokenizer",
    output_dir: str,
    suffix: str = "final",
) -> str:
    """Save model checkpoint."""
    save_dir = os.path.join(output_dir, suffix)
    os.makedirs(save_dir, exist_ok=True)

    # Save reward head weights
    torch.save(model.reward_head.state_dict(), os.path.join(save_dir, "reward_head.pt"))

    # Save model config
    config_dict = {
        "model_name": model.model_name,
        "reward_head_state": "reward_head.pt",
    }
    with open(os.path.join(save_dir, "reward_model_config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # Save tokenizer
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(save_dir)

    logger.info("Checkpoint saved to %s", save_dir)
    return save_dir


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_reward_model(config: RewardModelConfig) -> Dict[str, float]:
    """Evaluate reward model: Spearman correlation with actual returns."""
    from transformers import AutoTokenizer
    from scipy.stats import spearmanr

    logger.info("Loading reward model from %s", config.output_dir)
    load_dir = os.path.join(config.output_dir, "best")

    model = RewardModel(
        model_name=config.model_name,
        hidden_dim=config.hidden_dim,
        freeze_encoder=True,
    )

    # Load trained head
    head_path = os.path.join(load_dir, "reward_head.pt")
    if os.path.exists(head_path):
        model.reward_head.load_state_dict(torch.load(head_path, map_location="cpu"))
        logger.info("Loaded reward head from %s", head_path)
    else:
        logger.warning("No trained reward head found at %s. Using untrained model.", head_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(load_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Generate fresh trajectories with known returns
    generator = PreferencePairDataset(
        num_trajectories=50,
        trajectory_length=config.trajectory_length,
        pairs_per_trajectory=1,
        seed=config.seed + 10,
    )
    trajectories = []
    for _ in range(50):
        text, ret = generator._simulate_trajectory_return()
        trajectories.append({"text": text, "return": ret})

    texts = [t["text"] for t in trajectories]
    actual_returns = np.array([t["return"] for t in trajectories])

    # Predict rewards
    predicted_rewards = model.predict_reward(texts, tokenizer, device, config.max_length)
    predicted_rewards = predicted_rewards.flatten()

    # Spearman correlation
    corr, p_value = spearmanr(predicted_rewards, actual_returns)
    corr = float(corr) if not np.isnan(corr) else 0.0

    # Accuracy: fraction where predicted reward agrees with actual ranking
    correct_ranking = 0
    total_pairs = 0
    for i in range(len(trajectories)):
        for j in range(i + 1, len(trajectories)):
            if actual_returns[i] > actual_returns[j]:
                if predicted_rewards[i] > predicted_rewards[j]:
                    correct_ranking += 1
            elif actual_returns[i] < actual_returns[j]:
                if predicted_rewards[i] < predicted_rewards[j]:
                    correct_ranking += 1
            total_pairs += 1

    ranking_accuracy = correct_ranking / max(total_pairs, 1)

    results = {
        "spearman_correlation": round(corr, 4),
        "spearman_p_value": round(float(p_value), 6),
        "ranking_accuracy": round(ranking_accuracy, 4),
        "num_trajectories": len(trajectories),
    }
    logger.info("Evaluation results: %s", json.dumps(results, indent=2))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reward model for trading quality prediction."
    )
    parser.add_argument(
        "--mode",
        choices=["prepare", "train", "evaluate"],
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
        default="./checkpoints/reward_model",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension of reward head.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
    )
    args = parser.parse_args()

    overrides = {
        "model_name": args.model,
        "output_dir": args.output_dir,
        "hidden_dim": args.hidden_dim,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
    }

    config = RewardModelConfig.from_yaml(overrides)
    logger.info("Config: %s", json.dumps(vars(config), indent=2, default=str))

    if args.mode == "prepare":
        train_dataset, eval_dataset, tokenizer = prepare_reward_dataset(config)
        out_dir = os.path.join(config.output_dir, "data")
        os.makedirs(out_dir, exist_ok=True)
        train_dataset.save_to_disk(os.path.join(out_dir, "train"))
        eval_dataset.save_to_disk(os.path.join(out_dir, "eval"))
        tokenizer.save_pretrained(out_dir)
        logger.info("Data prepared and saved to %s", out_dir)

    elif args.mode == "train":
        train_reward_model(config)

    elif args.mode == "evaluate":
        evaluate_reward_model(config)


if __name__ == "__main__":
    main()
