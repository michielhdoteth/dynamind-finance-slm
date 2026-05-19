"""
Supervised Fine-Tuning on Expert Trading Trajectories.

Behavioral cloning from rule-based expert strategies (momentum, mean-reversion).
Converts (observation -> action) pairs to text format for causal LM fine-tuning,
enabling the Qwen model to learn trading behaviors from demonstration.

Usage:
    python training/supervised_finetuning.py --mode generate
    python training/supervised_finetuning.py --mode train
    python training/supervised_finetuning.py --mode evaluate
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

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
class SFTConfig:
    """Configuration for supervised fine-tuning on expert trajectories."""

    model_name: str = "Qwen/Qwen2-0.5B"
    max_length: int = 512
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 10
    output_dir: str = "./checkpoints/supervised_finetuned"
    trajectory_dir: str = "./data/expert_trajectories"
    num_trajectories: int = 100
    trajectory_length: int = 50
    warmup_ratio: float = 0.1
    save_steps: int = 200
    logging_steps: int = 10
    save_total_limit: int = 3
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    eval_split_ratio: float = 0.15
    seed: int = 42

    @classmethod
    def from_yaml(cls, overrides: Optional[Dict] = None) -> "SFTConfig":
        cfg = load_config()
        sft_cfg = cfg.get("supervised_finetuning", {})
        kwargs = dict(
            model_name=sft_cfg.get("model_name", cls.model_name),
            max_length=sft_cfg.get("max_length", cls.max_length),
            batch_size=sft_cfg.get("batch_size", cls.batch_size),
            learning_rate=sft_cfg.get("learning_rate", cls.learning_rate),
            num_epochs=sft_cfg.get("num_epochs", cls.num_epochs),
            output_dir=sft_cfg.get("output_dir", cls.output_dir),
            num_trajectories=sft_cfg.get("num_trajectories", cls.num_trajectories),
        )
        if overrides:
            kwargs.update(overrides)
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Expert Trajectory Dataset
# ---------------------------------------------------------------------------

class ExpertTrajectoryDataset:
    """Generates expert trading trajectories using rule-based strategies.

    Supports two strategies:
        - momentum:   Buy when short-term > long-term moving average
        - mean-reversion: Buy when price < lower Bollinger Band, sell when > upper
    """

    def __init__(
        self,
        num_trajectories: int = 100,
        trajectory_length: int = 50,
        seed: int = 42,
    ):
        self.num_trajectories = num_trajectories
        self.trajectory_length = trajectory_length
        self.rng = np.random.default_rng(seed)
        self._strategies = {
            "momentum": self._momentum_expert,
            "mean_reversion": self._mean_reversion_expert,
            "random": self._random_expert,
        }

    @staticmethod
    def _generate_price_series(length: int, rng: np.random.Generator) -> np.ndarray:
        """Generate a synthetic price series with regime-switching volatility."""
        returns = np.zeros(length)
        volatility = 0.015
        for i in range(1, length):
            # Regime switch every ~40 steps on average
            if rng.random() < 0.025:
                volatility = rng.uniform(0.005, 0.035)
            # Small drift + noise
            drift = rng.normal(0.0002, 0.0005)
            returns[i] = drift + rng.normal(0, volatility)
        prices = 100.0 * np.cumprod(1 + returns)
        return prices

    def _compute_indicators(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute common technical indicators."""
        sma_10 = pd.Series(prices).rolling(window=10, min_periods=1).mean().values
        sma_30 = pd.Series(prices).rolling(window=30, min_periods=1).mean().values

        # RSI
        delta = np.diff(prices, prepend=prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=14, min_periods=1).mean().values
        avg_loss = pd.Series(loss).rolling(window=14, min_periods=1).mean().values
        rs = avg_gain / (avg_loss + 1e-8)
        rsi_14 = 100 - (100 / (1 + rs))

        # Bollinger Bands
        sma_20 = pd.Series(prices).rolling(window=20, min_periods=1).mean().values
        std_20 = pd.Series(prices).rolling(window=20, min_periods=1).std().values
        bb_upper = sma_20 + 2 * std_20
        bb_lower = sma_20 - 2 * std_20

        return {
            "sma_10": sma_10,
            "sma_30": sma_30,
            "rsi_14": rsi_14,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
        }

    def _momentum_expert(
        self, prices: np.ndarray, indicators: Dict[str, np.ndarray], step: int
    ) -> int:
        """Momentum strategy: 1 (buy) if sma_10 > sma_30, 2 (sell) if sma_10 < sma_30, 0 (hold) otherwise."""
        if step < 30:
            return 0
        if indicators["sma_10"][step] > indicators["sma_30"][step] * 1.005:
            return 1  # Buy
        elif indicators["sma_10"][step] < indicators["sma_30"][step] * 0.995:
            return 2  # Sell
        return 0  # Hold

    def _mean_reversion_expert(
        self, prices: np.ndarray, indicators: Dict[str, np.ndarray], step: int
    ) -> int:
        """Mean-reversion: buy near lower band, sell near upper band."""
        if step < 20:
            return 0
        bb_lower = indicators["bb_lower"][step]
        bb_upper = indicators["bb_upper"][step]
        if prices[step] <= bb_lower * 1.01:
            return 1  # Buy (oversold)
        elif prices[step] >= bb_upper * 0.99:
            return 2  # Sell (overbought)
        return 0  # Hold

    def _random_expert(
        self, prices: np.ndarray, indicators: Dict[str, np.ndarray], step: int
    ) -> int:
        """Random baseline expert."""
        return self.rng.integers(0, 3)

    def _format_observation(self, prices: np.ndarray, step: int) -> str:
        """Format current market state as a text prompt."""
        price = prices[step]
        ret = (price / prices[max(0, step - 1)] - 1) * 100 if step > 0 else 0.0
        sma_10 = pd.Series(prices).rolling(10, min_periods=1).mean().values[step]
        return (
            f"Market state at step {step}: price=${price:.2f}, "
            f"return={ret:+.2f}%%, sma_10=${sma_10:.2f}"
        )

    def generate_trajectories(self) -> List[Dict[str, Any]]:
        """Generate expert trajectories as (prompt, action) pairs in text format.

        Returns:
            List of dicts with 'prompt' and 'completion' keys (text for causal LM).
        """
        trajectories = []
        strategy_names = list(self._strategies.keys())

        for traj_idx in range(self.num_trajectories):
            strategy = self.rng.choice(strategy_names)
            prices = self._generate_price_series(self.trajectory_length, self.rng)
            indicators = self._compute_indicators(prices)
            expert_fn = self._strategies[strategy]

            trajectory_text = ""
            action_count = {"Hold": 0, "Buy": 0, "Sell": 0}

            for step in range(self.trajectory_length):
                obs_text = self._format_observation(prices, step)
                action = expert_fn(prices, indicators, step)
                action_label = ["Hold", "Buy", "Sell"][action]
                action_count[action_label] += 1

                trajectory_text += f"Observation: {obs_text}\nAction: {action_label}\n"

            trajectories.append({
                "trajectory_id": traj_idx,
                "strategy": strategy,
                "text": trajectory_text.strip(),
                "action_counts": action_count,
            })

        logger.info(
            "Generated %d expert trajectories (strategy distribution: %s)",
            len(trajectories),
            {s: sum(1 for t in trajectories if t["strategy"] == s) for s in strategy_names},
        )
        return trajectories

    def to_hf_dataset(self) -> "Dataset":
        """Convert trajectories to a HuggingFace Dataset suitable for causal LM fine-tuning.

        Each sample is a text string containing the full trajectory,
        which the model learns to autoregressively predict.
        """
        try:
            from datasets import Dataset
        except ImportError:
            raise ImportError("datasets package required. Install with: pip install datasets")

        trajectories = self.generate_trajectories()
        return Dataset.from_list([{"text": t["text"]} for t in trajectories])


try:
    import pandas as pd
except ImportError:
    pd = None


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------

def prepare_dataset(config: SFTConfig) -> Tuple["Dataset", "Dataset", "AutoTokenizer"]:
    """Generate expert trajectories, tokenize, and split into train/eval.

    Returns:
        tuple: (train_dataset, eval_dataset, tokenizer)
    """
    from transformers import AutoTokenizer

    logger.info("Loading tokenizer: %s", config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Generate expert trajectories
    logger.info(
        "Generating %d trajectories of length %d...",
        config.num_trajectories,
        config.trajectory_length,
    )
    generator = ExpertTrajectoryDataset(
        num_trajectories=config.num_trajectories,
        trajectory_length=config.trajectory_length,
        seed=config.seed,
    )
    raw_dataset = generator.to_hf_dataset()
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
# Training
# ---------------------------------------------------------------------------

def train_sft(config: SFTConfig) -> Tuple[torch.nn.Module, "AutoTokenizer"]:
    """Run supervised fine-tuning on expert trading trajectories."""
    from transformers import (
        AutoModelForCausalLM,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # Prepare data
    train_dataset, eval_dataset, tokenizer = prepare_dataset(config)

    # Load model
    logger.info("Loading model: %s", config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    output_dir = os.path.abspath(config.output_dir)
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        evaluation_strategy="steps",
        eval_steps=config.save_steps,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="none",
        remove_unused_columns=True,
        fp16=torch.cuda.is_available(),
        seed=config.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting supervised fine-tuning...")
    trainer.train()

    # Final evaluation
    eval_metrics = trainer.evaluate()
    logger.info("Final eval metrics: %s", json.dumps(eval_metrics, indent=2))

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save metadata
    metadata = {
        "config": {
            "model_name": config.model_name,
            "max_length": config.max_length,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "num_trajectories": config.num_trajectories,
            "trajectory_length": config.trajectory_length,
        },
        "eval_metrics": eval_metrics,
        "device": device,
    }
    with open(os.path.join(output_dir, "sft_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("SFT model saved to %s", output_dir)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_sft(config: SFTConfig) -> Dict[str, float]:
    """Evaluate the SFT model on held-out expert trajectory data.

    Computes:
        - Perplexity on the eval set
        - Action prediction accuracy (whether the model's greedy completion
          matches the expert action at each step)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading SFT model from %s", config.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(config.output_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.output_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Get eval dataset
    _, eval_dataset, tokenizer_check = prepare_dataset(config)

    # --- Perplexity ---
    from transformers import DataCollatorForLanguageModeling

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    total_loss = 0.0
    total = 0

    indices = list(range(len(eval_dataset)))
    batch_size = min(config.batch_size, len(indices))
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i : i + batch_size]
        batch = data_collator([eval_dataset[idx] for idx in batch_indices])
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                outputs = model(input_ids=input_ids, labels=labels)

        total_loss += outputs.loss.item()
        total += 1

    avg_loss = total_loss / max(total, 1)
    perplexity = math.exp(avg_loss)

    # --- Action Prediction Accuracy ---
    logger.info("Measuring action prediction accuracy on eval set...")
    correct = 0
    total_steps = 0
    action_map = {"Hold": 0, "Buy": 1, "Sell": 2}

    for sample_idx in range(min(len(eval_dataset), 20)):
        text = eval_dataset[sample_idx]["text"] if isinstance(eval_dataset[sample_idx], dict) else ""
        # We need the original text — re-generate
        pass

    # Regenerate a small test set for accuracy measurement
    generator = ExpertTrajectoryDataset(
        num_trajectories=5,
        trajectory_length=config.trajectory_length,
        seed=config.seed + 1,
    )
    trajectories = generator.generate_trajectories()

    for traj in trajectories:
        lines = traj["text"].split("\n")
        for line in lines:
            if line.startswith("Action:"):
                true_action_str = line.replace("Action:", "").strip()
                true_action = action_map.get(true_action_str, -1)
                if true_action < 0:
                    continue
                total_steps += 1

        # Predict using model
        prompt_text = "\n".join(
            l for l in lines if l.startswith("Observation:")
        )
        if prompt_text:
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=config.max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            for predicted_action_str in ["Hold", "Buy", "Sell"]:
                if predicted_action_str in generated:
                    pred_action = action_map[predicted_action_str]
                    if pred_action == true_action:
                        correct += 1
                    break

    accuracy = correct / max(total_steps, 1)

    results = {
        "perplexity": round(perplexity, 4),
        "action_accuracy": round(accuracy, 4),
        "eval_loss": round(avg_loss, 6),
        "eval_samples": len(eval_dataset),
    }
    logger.info("Evaluation results: %s", json.dumps(results, indent=2))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Supervised fine-tuning on expert trading trajectories."
    )
    parser.add_argument(
        "--mode",
        choices=["generate", "train", "evaluate"],
        required=True,
        help="Pipeline stage.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2-0.5B",
        help="HuggingFace model name or path.",
    )
    parser.add_argument(
        "--output-dir",
        default="./checkpoints/supervised_finetuned",
        help="Output directory for model checkpoints.",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=100,
        help="Number of expert trajectories to generate.",
    )
    parser.add_argument(
        "--trajectory-length",
        type=int,
        default=50,
        help="Number of steps per trajectory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
    )
    args = parser.parse_args()

    overrides = {
        "model_name": args.model,
        "output_dir": args.output_dir,
        "num_trajectories": args.num_trajectories,
        "trajectory_length": args.trajectory_length,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
    }

    config = SFTConfig.from_yaml(overrides)
    logger.info("Config: %s", json.dumps(vars(config), indent=2, default=str))

    if args.mode == "generate":
        # Generate and save trajectories
        generator = ExpertTrajectoryDataset(
            num_trajectories=config.num_trajectories,
            trajectory_length=config.trajectory_length,
            seed=config.seed,
        )
        trajectories = generator.generate_trajectories()
        os.makedirs(config.trajectory_dir, exist_ok=True)
        out_path = os.path.join(config.trajectory_dir, "expert_trajectories.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for t in trajectories:
                f.write(json.dumps(t) + "\n")
        logger.info("Saved %d trajectories to %s", len(trajectories), out_path)

    elif args.mode == "train":
        train_sft(config)

    elif args.mode == "evaluate":
        evaluate_sft(config)


if __name__ == "__main__":
    main()
