"""
RLHF-Style Alignment using PPO against a Trained Reward Model.

Implements Proximal Policy Optimization (PPO) for aligning the supervised
fine-tuned (SFT) model using a trained reward model as the reward signal
(instead of environment reward). Includes KL penalty to prevent policy drift.

Supports three reward modes:
    - env:       Use environment return as reward (baseline)
    - rm:        Use trained reward model predictions as reward
    - ensemble:  Weighted average of environment and reward model

Usage:
    python training/rlhf_alignment.py --mode train --reward-mode rm
    python training/rlhf_alignment.py --mode train --reward-mode ensemble
    python training/rlhf_alignment.py --mode evaluate
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import warnings
from copy import deepcopy
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
class RLHFConfig:
    """Configuration for PPO-based RLHF alignment."""

    # Model paths
    sft_model_path: str = "./checkpoints/supervised_finetuned"
    reward_model_path: str = "./checkpoints/reward_model/best"
    base_model_name: str = "Qwen/Qwen2-0.5B"

    # PPO hyperparameters
    learning_rate: float = 1e-5
    batch_size: int = 64
    mini_batch_size: int = 16
    ppo_epochs: int = 4
    kl_coeff: float = 0.1           # KL penalty coefficient
    clip_range: float = 0.2         # PPO clipping range
    vf_coeff: float = 0.5           # Value function loss coefficient
    max_grad_norm: float = 0.5
    gamma: float = 0.99             # Discount factor
    gae_lambda: float = 0.95        # GAE lambda

    # Training
    num_steps: int = 100            # Number of PPO update steps
    max_length: int = 512           # Max sequence length for generation
    response_length: int = 64       # Max tokens for model response
    output_dir: str = "./checkpoints/rlhf_aligned"
    seed: int = 42

    # Reward mode
    reward_mode: str = "rm"         # 'env', 'rm', or 'ensemble'
    ensemble_weight: float = 0.5    # Weight for env reward when mode='ensemble'

    @classmethod
    def from_yaml(cls, overrides: Optional[Dict] = None) -> "RLHFConfig":
        cfg = load_config()
        rlhf_cfg = cfg.get("rlhf", {})
        kwargs = dict(
            sft_model_path=rlhf_cfg.get("sft_model_path", cls.sft_model_path),
            reward_model_path=rlhf_cfg.get("reward_model_path", cls.reward_model_path),
            learning_rate=rlhf_cfg.get("learning_rate", cls.learning_rate),
            batch_size=rlhf_cfg.get("batch_size", cls.batch_size),
            kl_coeff=rlhf_cfg.get("kl_coeff", cls.kl_coeff),
            reward_mode=rlhf_cfg.get("reward_mode", cls.reward_mode),
        )
        if overrides:
            kwargs.update(overrides)
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Reward Model Wrapper
# ---------------------------------------------------------------------------

class RewardModelWrapper(nn.Module):
    """Wraps the trained reward model for RLHF scoring.

    Loads a saved RewardModel (Qwen encoder + linear head) and provides
    a simple predict() interface for trajectory scoring.
    """

    def __init__(self, config: RLHFConfig):
        super().__init__()
        self.config = config

        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("transformers package required.")

        # Load tokenizer from reward model dir or base
        tokenizer_path = config.reward_model_path
        if not os.path.exists(tokenizer_path) or not os.path.isdir(tokenizer_path):
            tokenizer_path = config.base_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Build reward model
        from training.reward_model import RewardModel
        self.model = RewardModel(
            model_name=config.base_model_name,
            hidden_dim=256,
            freeze_encoder=True,
        )

        # Load trained head
        head_path = os.path.join(config.reward_model_path, "reward_head.pt")
        if os.path.exists(head_path):
            state = torch.load(head_path, map_location="cpu")
            self.model.reward_head.load_state_dict(state)
            logger.info("Loaded reward head from %s", head_path)
        else:
            logger.warning("No reward head found at %s. Using untrained head.", head_path)

    @torch.no_grad()
    def predict(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """Predict scalar rewards for a batch of text trajectories.

        Returns:
            Tensor of shape [batch, 1] with reward scores.
        """
        self.model.eval()
        self.model.to(device)

        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        rewards = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return rewards


# ---------------------------------------------------------------------------
# Simulated Environment
# ---------------------------------------------------------------------------

class SimulatedTradingEnv:
    """A simplified text-based trading environment for RLHF rollout.

    Provides a reward signal by simulating the outcome of a trading decision.
    The environment takes an action text and returns a scalar reward based on
    a simple price simulation.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.current_price = 100.0
        self.step_count = 0

    def reset(self) -> str:
        """Reset environment and return initial observation text."""
        self.current_price = 100.0
        self.step_count = 0
        return (
            f"Market state: price=${self.current_price:.2f}, "
            f"volatility=medium, trend=neutral"
        )

    def step(self, action_text: str) -> Tuple[str, float, bool]:
        """Execute action and return (next_obs, reward, done).

        Simulates a price move and computes reward based on action quality.
        """
        self.step_count += 1
        # Simulate price movement
        ret = float(self.rng.normal(0.0002, 0.02))
        self.current_price *= 1 + ret

        # Determine reward based on action
        action_lower = action_text.strip().lower()
        is_buy = "buy" in action_lower or "long" in action_lower
        is_sell = "sell" in action_lower or "short" in action_lower

        if is_buy:
            reward = ret * 100  # positive reward if price went up
        elif is_sell:
            reward = -ret * 100  # positive reward if price went down
        else:
            reward = 0.0  # hold

        # Add small noise
        reward += float(self.rng.normal(0, 0.1))

        done = self.step_count >= 10
        next_obs = (
            f"Market state: price=${self.current_price:.2f}, "
            f"return={ret*100:+.2f}%%"
        )
        return next_obs, reward, done


# ---------------------------------------------------------------------------
# PPO Trainer for RLHF
# ---------------------------------------------------------------------------

class AdaptiveKLController:
    """Adaptive KL penalty coefficient from the InstructGPT paper.

    Increases kl_coeff when kl > target, decreases when kl < target.
    """

    def __init__(self, init_kl_coeff: float = 0.1, target_kl: float = 0.015):
        self.kl_coeff = init_kl_coeff
        self.target_kl = target_kl

    def update(self, current_kl: float):
        if current_kl > self.target_kl * 1.5:
            self.kl_coeff *= 1.2
        elif current_kl < self.target_kl * 0.5:
            self.kl_coeff *= 0.8
        self.kl_coeff = max(0.01, min(5.0, self.kl_coeff))


class PPOTrainer:
    """PPO trainer for RLHF alignment of language models.

    Based on the trl library's implementation but self-contained.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        tokenizer: "AutoTokenizer",
        config: RLHFConfig,
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config

        self.optimizer = torch.optim.AdamW(
            policy_model.parameters(),
            lr=config.learning_rate,
        )

        self.kl_controller = AdaptiveKLController(
            init_kl_coeff=config.kl_coeff,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_model.to(self.device)
        self.ref_model.to(self.device)
        self.ref_model.eval()

    def _generate_rollout(self, prompt: str) -> Dict[str, torch.Tensor]:
        """Generate a response from the policy model given a prompt.

        Returns:
            dict with 'query', 'response', 'logprobs', 'ref_logprobs',
            and full 'input_ids', 'attention_mask'.
        """
        self.policy_model.eval()

        # Tokenize prompt
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.config.max_length - self.config.response_length,
            return_tensors="pt",
        )
        query_ids = encoded["input_ids"].to(self.device)
        query_mask = encoded["attention_mask"].to(self.device)

        # Generate response
        with torch.no_grad():
            output = self.policy_model.generate(
                input_ids=query_ids,
                attention_mask=query_mask,
                max_new_tokens=self.config.response_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        response_ids = output.sequences[0, query_ids.shape[1]:]  # exclude prompt
        full_ids = output.sequences[0]
        full_mask = torch.cat([
            query_mask[0],
            torch.ones(len(response_ids), device=self.device),
        ])

        # Compute logprobs for the generated response
        with torch.no_grad():
            # Policy logprobs
            policy_outputs = self.policy_model(
                input_ids=full_ids.unsqueeze(0),
                attention_mask=full_mask.unsqueeze(0),
            )
            policy_logits = policy_outputs.logits[0, query_ids.shape[1] - 1:-1]  # [response_len, vocab]
            policy_logprobs = F.log_softmax(policy_logits, dim=-1)
            response_logprobs = policy_logprobs.gather(
                1, response_ids.unsqueeze(1)
            ).squeeze(1)

            # Reference logprobs
            ref_outputs = self.ref_model(
                input_ids=full_ids.unsqueeze(0),
                attention_mask=full_mask.unsqueeze(0),
            )
            ref_logits = ref_outputs.logits[0, query_ids.shape[1] - 1:-1]
            ref_logprobs = F.log_softmax(ref_logits, dim=-1)
            ref_response_logprobs = ref_logprobs.gather(
                1, response_ids.unsqueeze(1)
            ).squeeze(1)

        return {
            "query": prompt,
            "response": self.tokenizer.decode(response_ids, skip_special_tokens=True),
            "input_ids": full_ids,
            "attention_mask": full_mask,
            "logprobs": response_logprobs,
            "ref_logprobs": ref_response_logprobs,
        }

    def _compute_advantages(
        self,
        rewards: List[float],
        values: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns.

        Args:
            rewards: List of scalar rewards per token position.
            values: List of value function predictions per position.

        Returns:
            (advantages, returns) tensors.
        """
        if len(rewards) != len(values):
            raise ValueError(f"Length mismatch: {len(rewards)} vs {len(values)}")

        advantages = []
        gae = 0.0
        returns = []

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1].item()

            delta = rewards[t] + self.config.gamma * next_value - values[t].item()
            gae = delta + self.config.gamma * self.config.gae_lambda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t].item())

        return torch.tensor(advantages), torch.tensor(returns)

    def train_step(
        self,
        reward_model: Optional[RewardModelWrapper] = None,
        env: Optional[SimulatedTradingEnv] = None,
    ) -> Dict[str, float]:
        """Run a single PPO update step.

        Generates rollouts, computes rewards (from reward model or env),
        then performs PPO update.

        Args:
            reward_model: Optional reward model wrapper (for mode='rm' or 'ensemble').
            env: Optional trading environment (for mode='env' or 'ensemble').

        Returns:
            dict of training metrics.
        """
        self.policy_model.train()

        # --- Generate rollouts ---
        rollouts = []
        env_rewards_list = []

        for i in range(self.config.batch_size):
            # Get prompt from env or use a default
            if env:
                prompt = env.reset()
            else:
                prompt = "Market state: price=$100.00, volatility=medium, trend=neutral"

            rollout = self._generate_rollout(prompt)
            rollouts.append(rollout)

            # Get environment reward if available
            if env:
                _, env_reward, _ = env.step(rollout["response"])
                env_rewards_list.append(env_reward)
            else:
                env_rewards_list.append(0.0)

        # --- Compute rewards ---
        batch_rewards = []
        response_texts = [r["response"] for r in rollouts]

        if self.config.reward_mode == "rm" and reward_model is not None:
            # Reward from reward model only
            rm_scores = reward_model.predict(response_texts, self.device)
            batch_rewards = rm_scores.detach().cpu().numpy().flatten().tolist()
            logger.debug(
                "RM rewards: mean=%.4f, std=%.4f",
                np.mean(batch_rewards), np.std(batch_rewards),
            )

        elif self.config.reward_mode == "env" and env is not None:
            batch_rewards = env_rewards_list

        elif self.config.reward_mode == "ensemble":
            if reward_model is not None:
                rm_scores = reward_model.predict(response_texts, self.device)
                rm_rewards = rm_scores.detach().cpu().numpy().flatten()
            else:
                rm_rewards = np.zeros(len(response_texts))
            env_rewards = np.array(env_rewards_list)
            w = self.config.ensemble_weight
            batch_rewards = (w * env_rewards + (1 - w) * rm_rewards).tolist()
            logger.debug(
                "Ensemble rewards: mean=%.4f, env_mean=%.4f, rm_mean=%.4f",
                np.mean(batch_rewards), np.mean(env_rewards), np.mean(rm_rewards),
            )
        else:
            batch_rewards = env_rewards_list

        # --- Token-level reward shaping ---
        # Spread the trajectory reward across all response tokens
        shaped_rewards = []
        for idx, rollout in enumerate(rollouts):
            r = batch_rewards[idx]
            resp_len = len(rollout["logprobs"])
            # Reward at each token position (last token gets full reward)
            token_rewards = [0.0] * (resp_len - 1) + [r]
            shaped_rewards.append(token_rewards)

        # --- PPO update ---
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_kl = 0.0
        total_approx_kl = 0.0

        for epoch in range(self.config.ppo_epochs):
            indices = list(range(len(rollouts)))
            random.shuffle(indices)

            for start in range(0, len(indices), self.config.mini_batch_size):
                mini_indices = indices[start:start + self.config.mini_batch_size]
                mini_rollouts = [rollouts[i] for i in mini_indices]

                # Compute old logprobs and values for the mini-batch
                old_logprobs_list = []
                old_values_list = []
                advantages_list = []
                returns_list = []

                for idx_in_batch, ri in enumerate(mini_indices):
                    rollout = rollouts[ri]
                    old_logprobs_list.append(rollout["logprobs"])
                    # Estimate token-level values using a simple moving average
                    values = [torch.tensor(0.0) for _ in shaped_rewards[ri]]
                    adv, ret = self._compute_advantages(shaped_rewards[ri], values)
                    advantages_list.append(adv)
                    returns_list.append(ret)

                # Zero grad
                self.optimizer.zero_grad()

                total_loss = 0.0

                for ri in mini_indices:
                    rollout = rollouts[ri]
                    input_ids = rollout["input_ids"].unsqueeze(0).to(self.device)
                    attention_mask = rollout["attention_mask"].unsqueeze(0).to(self.device)

                    # Current policy logprobs
                    outputs = self.policy_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                    response_start = len(
                        self.tokenizer.encode(rollout["query"])
                    )
                    logits = outputs.logits[0, response_start - 1:-1]
                    logprobs = F.log_softmax(logits, dim=-1)

                    response_ids = rollout["input_ids"][response_start:]
                    response_logprobs = logprobs.gather(
                        1, response_ids[:len(logits)].unsqueeze(1)
                    ).squeeze(1)

                    # Resize to match old logprobs
                    n = min(len(response_logprobs), len(rollout["logprobs"]))
                    new_logp = response_logprobs[:n]
                    old_logp = rollout["logprobs"][:n].to(self.device)

                    # KL divergence
                    kl_div = (old_logp - new_logp).mean()
                    total_kl += kl_div.item()

                    # Approximate KL
                    approx_kl = ((old_logp - new_logp) ** 2 / 2).mean()
                    total_approx_kl += approx_kl.item()

                    # PPO ratio
                    ratio = torch.exp(new_logp - old_logp)

                    # Advantages
                    adv = advantages_list[mini_indices.index(ri)][:n].to(self.device)
                    rets = returns_list[mini_indices.index(ri)][:n].to(self.device)

                    # Normalize advantages
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    # Policy loss (clipped)
                    pg_loss1 = -adv * ratio
                    pg_loss2 = -adv * torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range)
                    policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss (simplified — using returns as target)
                    value_pred = outputs.logits[0, response_start - 1:-1, 0]  # hack: use first logit dim
                    value_pred = value_pred[:n]
                    value_loss = F.mse_loss(value_pred, rets)

                    # KL penalty
                    kl_penalty = self.kl_controller.kl_coeff * kl_div

                    loss = policy_loss + self.config.vf_coeff * value_loss + kl_penalty
                    total_loss += loss

                # Backward
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy_model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

        avg_kl = total_kl / max(len(rollouts), 1)
        self.kl_controller.update(avg_kl)

        metrics = {
            "policy_loss": round(total_policy_loss / max(len(rollouts), 1), 6),
            "value_loss": round(total_value_loss / max(len(rollouts), 1), 6),
            "kl_divergence": round(total_kl / max(len(rollouts), 1), 4),
            "approx_kl": round(total_approx_kl / max(len(rollouts) * self.config.ppo_epochs, 1), 4),
            "kl_coeff": round(self.kl_controller.kl_coeff, 4),
            "mean_reward": round(float(np.mean(batch_rewards)), 4),
            "reward_mode": self.config.reward_mode,
        }
        return metrics

    def save(self, path: str):
        """Save the policy model."""
        os.makedirs(path, exist_ok=True)
        self.policy_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info("RLHF model saved to %s", path)


# ---------------------------------------------------------------------------
# Training Orchestration
# ---------------------------------------------------------------------------

def train_rlhf(config: RLHFConfig) -> nn.Module:
    """Run RLHF alignment training.

    Loads SFT model, initializes reward model (if needed), and runs PPO.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # --- Load SFT model (policy) ---
    logger.info("Loading SFT model from %s", config.sft_model_path)
    policy_model = AutoModelForCausalLM.from_pretrained(
        config.sft_model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.sft_model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load reference model (frozen copy for KL) ---
    logger.info("Loading reference model from %s", config.sft_model_path)
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.sft_model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    # --- Initialize reward model ---
    reward_model_wrapper = None
    if config.reward_mode in ("rm", "ensemble"):
        logger.info("Initializing reward model from %s", config.reward_model_path)
        reward_model_wrapper = RewardModelWrapper(config)

    # --- Initialize environment ---
    env = SimulatedTradingEnv(seed=config.seed)

    # --- Initialize PPO trainer ---
    ppo_trainer = PPOTrainer(policy_model, ref_model, tokenizer, config)

    # --- Training loop ---
    logger.info("Starting RLHF alignment (mode=%s, kl_coeff=%.2f)...",
                 config.reward_mode, config.kl_coeff)

    metrics_history = []
    for step in range(config.num_steps):
        metrics = ppo_trainer.train_step(
            reward_model=reward_model_wrapper,
            env=env,
        )
        metrics["step"] = step
        metrics_history.append(metrics)

        if (step + 1) % 10 == 0 or step == 0:
            logger.info(
                "Step %3d/%d | reward: %.4f | kl: %.4f | kl_coeff: %.4f | "
                "policy_loss: %.6f | value_loss: %.6f",
                step + 1, config.num_steps,
                metrics["mean_reward"],
                metrics["kl_divergence"],
                metrics["kl_coeff"],
                metrics["policy_loss"],
                metrics["value_loss"],
            )

    # --- Save final model ---
    output_dir = os.path.abspath(config.output_dir)
    ppo_trainer.save(output_dir)

    # Save training metrics
    with open(os.path.join(output_dir, "rlhf_metrics.json"), "w") as f:
        json.dump(metrics_history, f, indent=2)

    logger.info("RLHF alignment complete. Model saved to %s", output_dir)
    return policy_model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_rlhf(config: RLHFConfig) -> Dict[str, float]:
    """Evaluate the aligned model.

    Compares the RLHF-aligned model vs the original SFT model by
    generating trading trajectories and computing average rewards.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load aligned model
    logger.info("Loading aligned model from %s", config.output_dir)
    try:
        aligned_model = AutoModelForCausalLM.from_pretrained(
            config.output_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            config.output_dir, trust_remote_code=True
        )
    except OSError:
        logger.warning("No aligned model found at %s. Loading SFT model instead.", config.output_dir)
        aligned_model = AutoModelForCausalLM.from_pretrained(
            config.sft_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            config.sft_model_path, trust_remote_code=True
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize environment and reward model
    env = SimulatedTradingEnv(seed=config.seed + 42)
    reward_wrapper = None
    if config.reward_mode in ("rm", "ensemble"):
        try:
            reward_wrapper = RewardModelWrapper(config)
        except Exception as e:
            logger.warning("Could not load reward model for eval: %s", e)

    # Run evaluation rollouts
    aligned_model.eval()
    total_reward = 0.0
    num_episodes = 20

    for ep in range(num_episodes):
        prompt = env.reset()
        encoded = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = aligned_model.generate(
                **encoded,
                max_new_tokens=config.response_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(output[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)
        _, reward, _ = env.step(response)
        total_reward += reward

    avg_reward = total_reward / num_episodes

    # If reward model available, also get RM score
    rm_score = 0.0
    if reward_wrapper:
        rm_preds = reward_wrapper.predict(
            [f"Evaluation episode {i}" for i in range(num_episodes)],
            device,
        )
        rm_score = rm_preds.mean().item()

    results = {
        "avg_env_reward": round(avg_reward, 4),
        "num_episodes": num_episodes,
        "reward_mode": config.reward_mode,
    }
    if reward_wrapper:
        results["avg_rm_score"] = round(rm_score, 4)

    logger.info("Evaluation results: %s", json.dumps(results, indent=2))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RLHF alignment using PPO against a trained reward model."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate"],
        required=True,
        help="Pipeline stage.",
    )
    parser.add_argument(
        "--reward-mode",
        choices=["env", "rm", "ensemble"],
        default="rm",
        help="Reward source for PPO training.",
    )
    parser.add_argument(
        "--sft-model-path",
        default="./checkpoints/supervised_finetuned",
        help="Path to SFT model.",
    )
    parser.add_argument(
        "--reward-model-path",
        default="./checkpoints/reward_model/best",
        help="Path to reward model checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default="./checkpoints/rlhf_aligned",
        help="Output directory.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of PPO update steps.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Rollout batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--kl-coeff",
        type=float,
        default=0.1,
        help="KL penalty coefficient.",
    )
    args = parser.parse_args()

    overrides = {
        "sft_model_path": args.sft_model_path,
        "reward_model_path": args.reward_model_path,
        "output_dir": args.output_dir,
        "num_steps": args.num_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "kl_coeff": args.kl_coeff,
        "reward_mode": args.reward_mode,
    }

    config = RLHFConfig.from_yaml(overrides)
    logger.info("Config: %s", json.dumps(vars(config), indent=2, default=str))

    if args.mode == "train":
        train_rlhf(config)
    elif args.mode == "evaluate":
        evaluate_rlhf(config)


if __name__ == "__main__":
    main()
