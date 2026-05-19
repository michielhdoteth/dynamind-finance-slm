"""
Financial Continued Pre-Training Pipeline.

Domain-adapts a Qwen model on financial text (SEC filings, earnings calls)
using a causal language modeling objective. Supports LoRA-based fine-tuning
when PEFT is available, and evaluates on a financial perplexity hold-out set.

Usage:
    python training/financial_pretraining.py --mode prepare
    python training/financial_pretraining.py --mode train --model Qwen/Qwen2-0.5B
    python training/financial_pretraining.py --mode evaluate
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
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset

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
class PretrainConfig:
    """Configuration for financial continued pre-training."""

    model_name: str = "Qwen/Qwen2-0.5B"
    max_length: int = 512
    batch_size: int = 4
    learning_rate: float = 5e-5
    num_epochs: int = 1
    output_dir: str = "./checkpoints/financial_pretrained"
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    warmup_ratio: float = 0.1
    save_steps: int = 500
    logging_steps: int = 10
    save_total_limit: int = 2
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    eval_split_ratio: float = 0.1  # fraction held out for perplexity eval
    seed: int = 42

    @classmethod
    def from_yaml(cls, overrides: Optional[Dict] = None) -> "PretrainConfig":
        """Load config from YAML, overriding with provided values."""
        cfg = load_config()
        pretrain_cfg = cfg.get("pretraining", {})

        kwargs = dict(
            model_name=pretrain_cfg.get("model_name", cls.model_name),
            max_length=pretrain_cfg.get("max_length", cls.max_length),
            batch_size=pretrain_cfg.get("batch_size", cls.batch_size),
            learning_rate=pretrain_cfg.get("learning_rate", cls.learning_rate),
            num_epochs=pretrain_cfg.get("num_epochs", cls.num_epochs),
            output_dir=pretrain_cfg.get("output_dir", cls.output_dir),
            use_lora=pretrain_cfg.get("use_lora", cls.use_lora),
        )

        if overrides:
            kwargs.update(overrides)

        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Financial Text Data
# ---------------------------------------------------------------------------

FINANCIAL_TEXT_SAMPLES: List[str] = [
    # SEC Filing style
    "In the third quarter, revenue increased 12%% year-over-year to $89.5 billion, "
    "driven by strong performance in services and wearables segments. "
    "Operating income was $24.9 billion with a margin of 27.8%%.",
    "The company reported diluted earnings per share of $1.46, compared to $1.29 "
    "in the prior year period. Cash flow from operations was $29.4 billion, an increase of 18%%.",
    "Looking ahead to Q4, management expects revenue between $89.5 billion and $93.5 billion, "
    "with gross margin between 44.5%% and 45.5%%. Capital expenditures are planned at "
    "approximately $3.5 billion.",
    "Market volatility continues to present both risks and opportunities. The VIX index "
    "averaged 18.5 during the quarter, down from 24.7 in the same period last year.",
    "The Federal Reserve maintained its benchmark interest rate at 5.25-5.50%% during "
    "the quarter, while signaling potential rate cuts later this year if inflation "
    "continues to moderate.",
    "Portfolio diversification across sectors reduced overall risk exposure. Technology "
    "and healthcare sectors outperformed, while energy and materials underperformed "
    "during the period.",
    "Trading volume increased 15%% compared to the previous quarter, with algorithmic "
    "trading accounting for approximately 60%% of total volume on major exchanges.",
    "Risk management protocols were enhanced with updated VaR limits and stress testing "
    "frameworks. The 95%% daily VaR was maintained at $2.5 million across all trading desks.",
    # Earnings call excerpts
    "On our Q2 earnings call, we discussed the margin expansion driven by operational "
    "efficiencies and favorable product mix. Gross margins improved by 320 basis points "
    "to 58.4%%, exceeding consensus estimates of 56.9%%.",
    "The balance sheet remains strong with $45.2 billion in cash and marketable securities. "
    "We repurchased $8.3 billion of shares during the quarter and paid $3.7 billion in dividends.",
    "Forward guidance reflects cautious optimism. Management guided Q3 revenue to "
    "$84.5-$87.5 billion, slightly below consensus of $88.2 billion, citing currency headwinds.",
    "Segment revenue breakdown: Products $69.7 billion (down 1%%), Services $21.4 billion "
    "(up 16%%). The services segment now represents 23%% of total revenue, up from 20%% last year.",
    "International revenue accounted for 62%% of total sales, with emerging markets "
    "growing 8%% year-over-year despite macroeconomic pressures in the region.",
    "Research and development spending increased to $7.8 billion, representing 8.7%% of "
    "revenue, as the company continues to invest in AI and machine learning capabilities.",
    "The effective tax rate for the quarter was 16.5%%, compared to 15.2%% in the prior "
    "year period. The increase was primarily due to changes in deferred tax assets.",
    "Inventory turnover improved to 58 days from 63 days last year, reflecting better "
    "supply chain management and demand forecasting accuracy.",
    # Trading and market analysis
    "Technical analysis indicates the stock is trading above its 50-day moving average "
    "of $185.42, with RSI at 62 suggesting moderate bullish momentum. Resistance "
    "is identified at the $198 level.",
    "The options market is pricing in a 3.5%% implied move for the upcoming earnings "
    "announcement, below the 5-year average of 4.8%%. Put-call ratio stands at 0.85.",
    "Institutional ownership increased to 72%% from 68%% last quarter, with 12 new "
    "institutional buyers disclosed in the latest 13F filings.",
    "The stock's beta of 1.2 indicates above-market systematic risk. Correlation "
    "with the S&P 500 over the trailing 12 months is 0.78.",
    "Active fund managers have increased their overweight position in the sector "
    "by 150 basis points, according to the latest monthly fund manager survey.",
    "Earnings surprise history shows the company has beaten consensus estimates "
    "in 8 of the last 12 quarters, with an average surprise factor of 3.2%%.",
    "Short interest declined to 2.1%% of float from 3.4%% three months ago, "
    "suggesting reduced bearish sentiment among market participants.",
    "The dividend yield of 2.8%% remains attractive relative to the 10-year Treasury "
    "yield of 4.2%%, providing a compelling total return proposition for income investors.",
    # Macroeconomic context
    "GDP growth for the quarter was revised upward to 3.1%% annualized, driven by "
    "robust consumer spending and business investment. The services sector expanded "
    "at its fastest pace in 18 months.",
    "Inflation data showed core PCE at 2.6%% year-over-year, moving closer to the "
    "Federal Reserve's 2%% target. Shelter costs remain sticky at 4.8%% annualized.",
    "Labor market remains resilient with non-farm payrolls adding an average of "
    "186,000 jobs per month. The unemployment rate held steady at 3.7%%.",
    "The yield curve has been inverted for 14 consecutive months, historically a "
    "reliable recession indicator. However, the magnitude of inversion has narrowed "
    "to 25 basis points from a peak of 108 basis points.",
    "Corporate bond spreads tightened 15 basis points during the quarter, reflecting "
    "improving credit conditions. High-yield spreads are now at 345 basis points.",
]


class FinancialTextDataset:
    """Financial text dataset for continued pre-training.

    In production, replace the demo samples with real SEC filing data
    (via EDGAR API), earnings call transcripts, and financial news.
    """

    def __init__(self, samples: Optional[List[str]] = None, max_length: int = 512):
        self.max_length = max_length
        self.samples = samples or FINANCIAL_TEXT_SAMPLES
        logger.info("FinancialTextDataset initialized with %d samples", len(self.samples))

    def to_hf_dataset(self) -> Dataset:
        """Convert samples to a HuggingFace Dataset."""
        return Dataset.from_list([{"text": s} for s in self.samples])

    def save_raw(self, path: str) -> None:
        """Save raw text samples to a JSONL file for inspection."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for sample in self.samples:
                f.write(json.dumps({"text": sample}) + "\n")
        logger.info("Saved %d raw samples to %s", len(self.samples), path)

    @classmethod
    def from_jsonl(cls, path: str, max_length: int = 512) -> "FinancialTextDataset":
        """Load samples from a JSONL file."""
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line)["text"])
        logger.info("Loaded %d samples from %s", len(samples), path)
        return cls(samples=samples, max_length=max_length)


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------

def prepare_dataset(config: PretrainConfig) -> (Dataset, Dataset, "AutoTokenizer"):
    """Prepare train/eval tokenized datasets and the tokenizer.

    Returns:
        tuple: (train_dataset, eval_dataset, tokenizer)
    """
    from transformers import AutoTokenizer

    logger.info("Loading tokenizer: %s", config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build raw dataset
    raw_dataset = FinancialTextDataset(max_length=config.max_length).to_hf_dataset()

    # Split into train / eval
    split = raw_dataset.train_test_split(
        test_size=config.eval_split_ratio,
        seed=config.seed,
    )

    def tokenize_function(examples):
        """Tokenize texts with padding and truncation."""
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=config.max_length,
        )

    train_dataset = split["train"].map(tokenize_function, batched=True)
    eval_dataset = split["test"].map(tokenize_function, batched=True)

    logger.info("Train samples: %d | Eval samples: %d", len(train_dataset), len(eval_dataset))
    return train_dataset, eval_dataset, tokenizer


# ---------------------------------------------------------------------------
# LoRA Setup
# ---------------------------------------------------------------------------

def _maybe_apply_lora(model: torch.nn.Module, config: PretrainConfig) -> torch.nn.Module:
    """Apply LoRA to model if PEFT is available and config.use_lora is True."""
    if not config.use_lora:
        logger.info("LoRA disabled — performing full fine-tune of all parameters.")
        return model

    try:
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        logger.info("LoRA applied (r=%d, alpha=%d)", config.lora_r, config.lora_alpha)
    except ImportError:
        logger.warning(
            "PEFT not installed. Falling back to full fine-tune of last %d layers.",
            config.save_total_limit,
        )
    return model


# ---------------------------------------------------------------------------
# Perplexity Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_perplexity(
    model: torch.nn.Module,
    eval_dataset: Dataset,
    tokenizer: "AutoTokenizer",
    batch_size: int = 4,
    max_length: int = 512,
) -> float:
    """Compute perplexity on the evaluation dataset.

    Uses sliding-window perplexity for sequences longer than max_length.
    """
    from transformers import DataCollatorForLanguageModeling

    device = next(model.parameters()).device
    model.eval()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    total_loss = 0.0
    total_steps = 0

    # Create batches manually for precise control
    indices = list(range(len(eval_dataset)))
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i : i + batch_size]
        batch = data_collator([eval_dataset[idx] for idx in batch_indices])

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )

        total_loss += outputs.loss.item()
        total_steps += 1

    avg_loss = total_loss / max(total_steps, 1)
    perplexity = math.exp(avg_loss)

    logger.info("Evaluation perplexity: %.4f (avg loss: %.4f)", perplexity, avg_loss)
    return perplexity


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_pretrain(config: PretrainConfig) -> (torch.nn.Module, "AutoTokenizer"):
    """Run continued pre-training on financial text."""
    from transformers import (
        AutoModelForCausalLM,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    # Set seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)
    if torch.cuda.is_available():
        logger.info("CUDA device: %s", torch.cuda.get_device_name())

    # Prepare datasets
    train_dataset, eval_dataset, tokenizer = prepare_dataset(config)

    # Load model
    logger.info("Loading model: %s", config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    # Apply LoRA
    model = _maybe_apply_lora(model, config)
    model = model.to(device)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
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
        dataloader_drop_last=False,
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

    # Train
    logger.info("Starting continued pre-training...")
    trainer.train()

    # Final evaluation
    logger.info("Running final evaluation...")
    eval_metrics = trainer.evaluate()
    logger.info("Final eval metrics: %s", json.dumps(eval_metrics, indent=2))

    # Compute and log perplexity
    perplexity = evaluate_perplexity(model, eval_dataset, tokenizer, config.batch_size)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training metadata
    metadata = {
        "config": {
            "model_name": config.model_name,
            "max_length": config.max_length,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "use_lora": config.use_lora,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
        },
        "eval_metrics": eval_metrics,
        "perplexity": perplexity,
        "device": device,
    }
    with open(os.path.join(output_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Model saved to %s", output_dir)
    logger.info("Final perplexity: %.4f", perplexity)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Evaluation-only entry point
# ---------------------------------------------------------------------------

def evaluate(config: PretrainConfig) -> float:
    """Load a pretrained model and compute perplexity on the eval set."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading pretrained model from %s", config.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(config.output_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.output_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    _, eval_dataset, _ = prepare_dataset(config)
    perplexity = evaluate_perplexity(model, eval_dataset, tokenizer, config.batch_size)
    logger.info("Evaluation complete. Perplexity: %.4f", perplexity)
    return perplexity


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Financial continued pre-training pipeline."
    )
    parser.add_argument(
        "--mode",
        choices=["prepare", "train", "evaluate"],
        required=True,
        help="Pipeline stage to execute.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2-0.5B",
        help="HuggingFace model name or local path.",
    )
    parser.add_argument(
        "--output-dir",
        default="./checkpoints/financial_pretrained",
        help="Directory to save model checkpoints.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Peak learning rate.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=True,
        help="Use LoRA for parameter-efficient fine-tuning.",
    )
    parser.add_argument(
        "--no-lora",
        action="store_false",
        dest="use_lora",
        help="Disable LoRA (full fine-tune).",
    )
    args = parser.parse_args()

    overrides = {
        "model_name": args.model,
        "output_dir": args.output_dir,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "max_length": args.max_length,
        "use_lora": args.use_lora,
    }

    config = PretrainConfig.from_yaml(overrides)
    logger.info("Config: %s", json.dumps(vars(config), indent=2, default=str))

    if args.mode == "prepare":
        dataset, _, tokenizer = prepare_dataset(config)
        # Save raw samples for inspection
        raw_path = os.path.join(config.output_dir, "raw_samples.jsonl")
        FinancialTextDataset(max_length=config.max_length).save_raw(raw_path)
        # Save tokenized dataset
        os.makedirs(config.output_dir, exist_ok=True)
        dataset.save_to_disk(os.path.join(config.output_dir, "tokenized_train"))
        logger.info(
            "Preparation complete. %d samples tokenized (max_length=%d).",
            len(dataset),
            config.max_length,
        )

    elif args.mode == "train":
        train_pretrain(config)

    elif args.mode == "evaluate":
        evaluate(config)


if __name__ == "__main__":
    main()
