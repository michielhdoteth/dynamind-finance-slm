"""
DPO and GRPO Training for Financial Trading.
Direct Preference Optimization and Group Relative Policy Optimization
for aligning trading policies without a separate reward model.

Supports two modes:
  - DPO (Direct Preference Optimization): Uses preference pairs from trajectory
    rankings to directly optimize the policy via a binary cross-entropy-style loss.
  - GRPO (Group Relative Policy Optimization): Generates a group of trajectories,
    scores them relatively, and uses group-normalized advantages for policy updates.
    Inspired by DeepSeek-R1. No critic/value network needed.

Usage:
    # DPO mode
    python training/dpo_trainer.py --mode dpo --model models/qwen_final_model_200k.zip \\
        --iterations 50 --n-trajectories 100 --beta 0.1 --lr 1e-5

    # GRPO mode
    python training/dpo_trainer.py --mode grpo --model models/qwen_final_model_200k.zip \\
        --iterations 100 --group-size 8 --beta 0.04 --lr 1e-5

Programmatic:
    from training.dpo_trainer import DPOTrainer, GRPOTrainer, TrajectoryCollector
"""

import argparse
import json
import os
import sys
import time
import copy
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

# Ensure project root is on path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from config.loader import load_config
from environments import SingleAssetTradingEnv
from environments.base_env import AssetConfig, TransactionCosts, RiskConstraints


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    """A single trading episode trajectory with performance metrics."""

    observations: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[float]
    log_probs: List[float]
    total_return: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    length: int

    @property
    def score(self) -> float:
        """Composite score for preference ranking.

        Weighted combination designed for financial trading:
          - Sharpe ratio (50%): Risk-adjusted returns
          - Total return (30%): Raw profitability, clipped to [-1, 1]
          - Win rate (20%): Trading consistency, mapped from [0,1] to [-1,1]

        Returns:
            Float in approximately [-1.5, 1.5] for typical trading trajectories.
        """
        return (
            0.5 * max(-2.0, min(2.0, self.sharpe))
            + 0.3 * max(-1.0, min(1.0, self.total_return))
            + 0.2 * (self.win_rate * 2.0 - 1.0)
        )


@dataclass
class PreferencePair:
    """A pair of trajectories where one is preferred over the other.

    Used by DPOTrainer to compute the DPO loss. The 'chosen' trajectory
    has a higher composite score than the 'rejected' trajectory.
    """

    chosen_obs: List[np.ndarray]
    chosen_actions: List[np.ndarray]
    chosen_log_probs: List[float]
    rejected_obs: List[np.ndarray]
    rejected_actions: List[np.ndarray]
    rejected_log_probs: List[float]
    chosen_score: float
    rejected_score: float


# ---------------------------------------------------------------------------
# Trajectory Collection
# ---------------------------------------------------------------------------


class TrajectoryCollector:
    """Collect trajectories from a policy for preference learning.

    Wraps either a VecEnv (SB3) or a raw Gymnasium environment and uses
    the policy's distribution to sample actions and compute log probs.

    Args:
        env: A VecEnv or Gymnasium environment instance.
        policy: An SB3 policy object (model.policy) with get_distribution().
        device: Torch device ('cpu', 'cuda', or 'auto').
    """

    def __init__(
        self,
        env: Any,
        policy: nn.Module,
        device: str = "auto",
    ):
        self.env = env
        self.policy = policy
        self.device = (
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._is_vec_env = isinstance(env, VecEnv)

    def collect_trajectories(
        self,
        n_trajectories: int = 100,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Trajectory]:
        """Collect N trajectories and compute trading metrics for each.

        Args:
            n_trajectories: Number of complete episodes to collect.
            progress_callback: Optional callback fn(completed, total).

        Returns:
            List of Trajectory objects with performance metrics.
        """
        trajectories: List[Trajectory] = []

        for i in range(n_trajectories):
            traj = self._run_one_episode()
            trajectories.append(traj)

            if (i + 1) % 20 == 0:
                print(f"  Collected {i + 1}/{n_trajectories} trajectories...")

            if progress_callback is not None:
                progress_callback(i + 1, n_trajectories)

        return trajectories

    def _run_one_episode(self) -> Trajectory:
        """Run a single episode and return a Trajectory with metrics."""
        # Reset and handle VecEnv vs raw env
        obs = self.env.reset()
        if self._is_vec_env:
            # VecEnv.reset() returns only obs (numpy array)
            pass
        else:
            # Gymnasium reset returns (obs, info)
            if isinstance(obs, tuple):
                obs = obs[0]

        obs_list: List[np.ndarray] = []
        act_list: List[np.ndarray] = []
        rew_list: List[float] = []
        logp_list: List[float] = []

        # Collect step data
        done = False
        while not done:
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            with torch.no_grad():
                dist = self.policy.get_distribution(obs_tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            # Store step data (extract single env from batch if VecEnv)
            if self._is_vec_env:
                obs_list.append(obs[0].copy())
                action_np = action.cpu().numpy()
                act_list.append(action_np[0].copy() if action_np.ndim > 1 else action_np.copy().flatten())
                logp_list.append(log_prob.cpu().numpy()[0].copy())
            else:
                obs_list.append(obs.copy())
                act_list.append(action.cpu().numpy().flatten())
                logp_list.append(log_prob.cpu().numpy().flatten())

            # Step environment
            action_np = action.cpu().numpy()
            step_result = self.env.step(action_np)

            if self._is_vec_env:
                # VecEnv.step returns (obs, reward, done, info)
                obs, reward_arr, done_arr, info = step_result
                rew_list.append(float(reward_arr[0]))
                done = bool(done_arr[0])
            else:
                # Gymnasium.step returns (obs, reward, terminated, truncated, info)
                obs, reward, terminated, truncated, info = step_result
                rew_list.append(float(reward))
                done = terminated or truncated

        # Compute episode metrics
        returns = np.array(rew_list, dtype=np.float64)
        total_return = float(np.sum(returns))
        sharpe = float(np.mean(returns) / np.std(returns)) if np.std(returns) > 1e-10 else 0.0
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = float(np.max(running_max - cumulative)) if len(cumulative) > 0 else 0.0
        win_rate = float(np.mean([r > 0 for r in rew_list])) if rew_list else 0.0

        return Trajectory(
            observations=obs_list,
            actions=act_list,
            rewards=rew_list,
            log_probs=logp_list,
            total_return=total_return,
            sharpe=sharpe,
            max_drawdown=drawdown,
            win_rate=win_rate,
            length=len(rew_list),
        )


# ---------------------------------------------------------------------------
# Preference Pair Builder
# ---------------------------------------------------------------------------


class PreferenceBuilder:
    """Build preference pairs from collected trading trajectories.

    Ranks trajectories by composite score, takes the top and bottom fractions,
    and creates (chosen, rejected) pairs for DPO training.
    """

    @staticmethod
    def build_pairs(
        trajectories: List[Trajectory],
        top_pct: float = 0.3,
        bottom_pct: float = 0.3,
        max_pairs: int = 100,
        rng_seed: Optional[int] = None,
    ) -> List[PreferencePair]:
        """Build preference pairs from top and bottom trajectories.

        Args:
            trajectories: Full list of collected trajectories.
            top_pct: Fraction of top trajectories to use as 'chosen'.
            bottom_pct: Fraction of bottom trajectories to use as 'rejected'.
            max_pairs: Maximum number of pairs to generate.
            rng_seed: Random seed for reproducibility.

        Returns:
            List of PreferencePair objects.
        """
        if len(trajectories) < 4:
            return []

        # Sort by composite score descending
        sorted_trajs = sorted(trajectories, key=lambda t: t.score, reverse=True)

        n_top = max(1, int(len(sorted_trajs) * top_pct))
        n_bottom = max(1, int(len(sorted_trajs) * bottom_pct))

        top_trajs = sorted_trajs[:n_top]
        bottom_trajs = sorted_trajs[-n_bottom:]

        rng = np.random.RandomState(rng_seed)
        pairs: List[PreferencePair] = []

        for chosen in top_trajs:
            # Sample one rejected trajectory per chosen (without replacement per chosen)
            n_sample = min(1, len(bottom_trajs))
            if n_sample == 0:
                continue
            rejected = bottom_trajs[rng.randint(len(bottom_trajs))]

            # Ensure chosen has a higher score (redundant by construction, but safe)
            if chosen.score <= rejected.score:
                continue

            pairs.append(
                PreferencePair(
                    chosen_obs=chosen.observations,
                    chosen_actions=chosen.actions,
                    chosen_log_probs=chosen.log_probs,
                    rejected_obs=rejected.observations,
                    rejected_actions=rejected.actions,
                    rejected_log_probs=rejected.log_probs,
                    chosen_score=chosen.score,
                    rejected_score=rejected.score,
                )
            )

            if len(pairs) >= max_pairs:
                break

        return pairs


# ---------------------------------------------------------------------------
# DPO Trainer
# ---------------------------------------------------------------------------


class DPOTrainer:
    """Direct Preference Optimization for trading policies.

    Instead of training a separate reward model + PPO (RLHF), DPO directly
    optimizes the policy using preference pairs. Given (chosen, rejected)
    trajectory pairs, DPO updates the policy to increase likelihood of
    chosen actions and decrease likelihood of rejected actions.

    The DPO loss is:
        L = -E[ log sigma(beta * (log pi(y_w|x) - log pi_ref(y_w|x)
                                  - log pi(y_l|x) + log pi_ref(y_l|x))) ]

    where pi is the current policy, pi_ref is the frozen reference policy,
    y_w is the chosen trajectory, and y_l is the rejected trajectory.

    Args:
        policy: SB3 policy network (model.policy) to optimize.
        ref_policy: Reference policy (frozen). If None, a copy is made.
        beta: Temperature parameter controlling how much to prefer chosen.
              Higher values = stronger preference signal.
        learning_rate: AdamW learning rate.
        device: Torch device.
    """

    def __init__(
        self,
        policy: nn.Module,
        ref_policy: Optional[nn.Module] = None,
        beta: float = 0.1,
        learning_rate: float = 1e-5,
        device: str = "auto",
    ):
        self.policy = policy
        self.ref_policy = ref_policy if ref_policy is not None else copy.deepcopy(policy)
        self.beta = beta
        self.device = (
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Freeze reference policy
        self.ref_policy.eval()
        for param in self.ref_policy.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=learning_rate
        )
        self.policy.to(self.device)
        self.ref_policy.to(self.device)

    def compute_dpo_loss(
        self, pairs: List[PreferencePair]
    ) -> Tuple[torch.Tensor, float]:
        """Compute DPO loss from a batch of preference pairs.

        Args:
            pairs: List of PreferencePair objects.

        Returns:
            Tuple of (loss tensor, number of valid pairs processed).
        """
        total_loss = torch.tensor(0.0, device=self.device)
        n_valid = 0

        for pair in pairs:
            try:
                # Log probs of chosen trajectory under current and reference policy
                chosen_log_probs = self._get_trajectory_log_probs(
                    pair.chosen_obs, pair.chosen_actions
                )
                ref_chosen_log_probs = self._get_ref_log_probs(
                    pair.chosen_obs, pair.chosen_actions
                )

                # Log probs of rejected trajectory under current and reference policy
                rejected_log_probs = self._get_trajectory_log_probs(
                    pair.rejected_obs, pair.rejected_actions
                )
                ref_rejected_log_probs = self._get_ref_log_probs(
                    pair.rejected_obs, pair.rejected_actions
                )

                # DPO core equation
                # pi_ratio = (log pi(y_w|x) - log pi_ref(y_w|x)) - (log pi(y_l|x) - log pi_ref(y_l|x))
                pi_ratio = (
                    chosen_log_probs
                    - ref_chosen_log_probs
                    - (rejected_log_probs - ref_rejected_log_probs)
                )
                loss = -F.logsigmoid(self.beta * pi_ratio)
                total_loss = total_loss + loss
                n_valid += 1

            except Exception as exc:
                # Skip pairs that cause numerical issues (e.g. extreme log probs)
                continue

        if n_valid == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0.0

        return total_loss / n_valid, float(n_valid)

    def train_step(
        self, pairs: List[PreferencePair]
    ) -> Dict[str, Any]:
        """Single DPO training step.

        Args:
            pairs: List of PreferencePair objects.

        Returns:
            Dict with 'dpo_loss' and 'n_pairs' keys.
        """
        if not pairs:
            return {"dpo_loss": 0.0, "n_pairs": 0}

        self.policy.train()
        self.optimizer.zero_grad()

        loss, n_valid = self.compute_dpo_loss(pairs)
        if n_valid < 1:
            return {"dpo_loss": 0.0, "n_pairs": 0}

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        return {"dpo_loss": loss.item(), "n_pairs": int(n_valid)}

    def _get_trajectory_log_probs(
        self, observations: List[np.ndarray], actions: List[np.ndarray]
    ) -> torch.Tensor:
        """Mean log-probability of actions under the current policy.

        Args:
            observations: List of observation arrays.
            actions: List of action arrays.

        Returns:
            Scalar tensor: average log-probability across the trajectory.
        """
        total_log_prob = torch.tensor(0.0, device=self.device)
        n = max(1, len(observations))

        for obs, act in zip(observations, actions):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            act_t = torch.FloatTensor(act).unsqueeze(0).to(self.device)
            dist = self.policy.get_distribution(obs_t)
            log_prob = dist.log_prob(act_t)
            total_log_prob = total_log_prob + log_prob.sum()

        return total_log_prob / n

    def _get_ref_log_probs(
        self, observations: List[np.ndarray], actions: List[np.ndarray]
    ) -> torch.Tensor:
        """Mean log-probability of actions under the frozen reference policy.

        Args:
            observations: List of observation arrays.
            actions: List of action arrays.

        Returns:
            Scalar tensor: average log-probability across the trajectory.
        """
        total_log_prob = torch.tensor(0.0, device=self.device)
        n = max(1, len(observations))

        with torch.no_grad():
            for obs, act in zip(observations, actions):
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                act_t = torch.FloatTensor(act).unsqueeze(0).to(self.device)
                dist = self.ref_policy.get_distribution(obs_t)
                log_prob = dist.log_prob(act_t)
                total_log_prob = total_log_prob + log_prob.sum()

        return total_log_prob / n


# ---------------------------------------------------------------------------
# GRPO Trainer
# ---------------------------------------------------------------------------


class GRPOTrainer:
    """Group Relative Policy Optimization for trading policies.

    Inspired by DeepSeek-R1's GRPO algorithm. Instead of a critic/value
    network, GRPO generates a group of trajectories, scores them relatively
    using a composite financial metric, and uses group-normalized advantages
    for the policy gradient update.

    Key advantages over standard PPO:
      - No critic/value network needed (saves ~50% memory).
      - Relative scoring within a group naturally handles reward variance.
      - Works well with parallel environment setups.

    The GRPO loss consists of:
      1. A clipped surrogate objective (same as PPO)
      2. A KL divergence penalty against the reference policy

    Args:
        policy: SB3 policy network (model.policy) to optimize.
        beta: KL penalty coefficient (default 0.04, as used in DeepSeek-R1).
        epsilon: PPO clipping range (default 0.2).
        group_size: Number of trajectories to collect per update.
        learning_rate: AdamW learning rate.
        device: Torch device.
    """

    def __init__(
        self,
        policy: nn.Module,
        beta: float = 0.04,
        epsilon: float = 0.2,
        group_size: int = 8,
        learning_rate: float = 1e-5,
        device: str = "auto",
    ):
        self.policy = policy
        self.ref_policy = copy.deepcopy(policy)
        self.beta = beta
        self.epsilon = epsilon
        self.group_size = group_size
        self.device = (
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Freeze reference policy
        self.ref_policy.eval()
        for param in self.ref_policy.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=learning_rate
        )
        self.policy.to(self.device)
        self.ref_policy.to(self.device)

    def collect_group(
        self,
        env_fn: Callable[[], Any],
        n_envs: Optional[int] = None,
    ) -> Tuple[List[Trajectory], Dict[str, float]]:
        """Collect a group of trajectories in parallel for GRPO update.

        Args:
            env_fn: Zero-argument callable that returns a Gymnasium environment.
            n_envs: Number of parallel environments (defaults to group_size // 2).

        Returns:
            Tuple of (list of Trajectory objects, dict of aggregate metrics).
        """
        n_envs = n_envs or max(1, self.group_size // 2)
        if n_envs > self.group_size:
            n_envs = self.group_size

        envs = DummyVecEnv([env_fn for _ in range(n_envs)])
        obs = envs.reset()

        # Per-environment storage
        trajs: List[Dict[str, Any]] = [
            {"obs": [], "acts": [], "rews": [], "logps": []} for _ in range(n_envs)
        ]
        dones = [False] * n_envs
        active_mask = np.ones(n_envs, dtype=bool)

        step_count = 0
        while active_mask.any():
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            with torch.no_grad():
                dist = self.policy.get_distribution(obs_tensor)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            next_obs, rewards, new_dones, _ = envs.step(actions.cpu().numpy())

            for i in range(n_envs):
                if not dones[i]:
                    trajs[i]["obs"].append(obs[i].copy())
                    act_np = actions[i].cpu().numpy()
                    trajs[i]["acts"].append(act_np.copy())
                    trajs[i]["rews"].append(float(rewards[i]))
                    trajs[i]["logps"].append(log_probs[i].cpu().numpy().copy())
                    if new_dones[i]:
                        dones[i] = True

            obs = next_obs
            active_mask = np.array([not d for d in dones], dtype=bool)
            step_count += 1

            # Safety: prevent infinite loops
            if step_count > 10_000:
                print("  [WARN] collect_group exceeded 10k steps, forcing termination")
                break

        envs.close()

        # Convert to Trajectory objects
        trajectories: List[Trajectory] = []
        for t in trajs:
            rews = t["rews"]
            if not rews:
                continue

            total_ret = float(np.sum(rews))
            sharpe = (
                float(np.mean(rews) / np.std(rews))
                if np.std(rews) > 1e-10
                else 0.0
            )
            cumulative = np.cumsum(rews)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = float(np.max(running_max - cumulative)) if len(cumulative) > 0 else 0.0

            trajectories.append(
                Trajectory(
                    observations=t["obs"],
                    actions=t["acts"],
                    rewards=rews,
                    log_probs=t["logps"],
                    total_return=total_ret,
                    sharpe=sharpe,
                    max_drawdown=drawdown,
                    win_rate=float(np.mean([r > 0 for r in rews])),
                    length=len(rews),
                )
            )

        metrics = {
            "mean_return": float(np.mean([t.total_return for t in trajectories])),
            "mean_sharpe": float(np.mean([t.sharpe for t in trajectories])),
            "n_trajectories": len(trajectories),
        }

        return trajectories, metrics

    def compute_grpo_loss(
        self, trajectories: List[Trajectory]
    ) -> Tuple[torch.Tensor, float]:
        """Compute GRPO loss from a group of trajectories.

        The loss has two components:
          1. Clipped surrogate objective using group-normalized advantages.
          2. KL divergence penalty against the reference policy.

        Args:
            trajectories: List of Trajectory objects from collect_group().

        Returns:
            Tuple of (loss tensor, scalar KL divergence value).
        """
        if len(trajectories) < 2:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            return loss, 0.0

        # Group-normalized advantages based on composite score
        scores = np.array([t.score for t in trajectories], dtype=np.float64)
        adv_mean = np.mean(scores)
        adv_std = max(np.std(scores), 1e-8)
        advantages = (scores - adv_mean) / adv_std

        total_loss = torch.tensor(0.0, device=self.device)
        total_kl = 0.0
        n_steps = 0

        for traj, adv in zip(trajectories, advantages):
            for obs, act, old_logp_raw in zip(
                traj.observations, traj.actions, traj.log_probs
            ):
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                act_t = torch.FloatTensor(act).unsqueeze(0).to(self.device)
                old_logp_t = torch.FloatTensor(old_logp_raw).to(self.device)

                # Current policy log prob
                dist = self.policy.get_distribution(obs_t)
                new_logp = dist.log_prob(act_t)

                # Importance sampling ratio
                ratio = torch.exp(new_logp.sum() - old_logp_t.sum())

                # Clipped surrogate objective
                adv_t = torch.tensor(adv, device=self.device, dtype=torch.float32)
                surr1 = ratio * adv_t
                surr2 = torch.clamp(
                    ratio, 1.0 - self.epsilon, 1.0 + self.epsilon
                ) * adv_t
                policy_loss = -torch.min(surr1, surr2)

                # KL penalty: exp(x) - 1 - x  where x = log(pi/pi_ref)
                with torch.no_grad():
                    ref_dist = self.ref_policy.get_distribution(obs_t)
                    ref_logp = ref_dist.log_prob(act_t)

                # Compute per-step KL using the approximation from TRPO/PPO
                log_ratio = old_logp_t - ref_logp
                kl_div = torch.exp(log_ratio) - 1.0 - log_ratio
                kl_loss = self.beta * kl_div.sum()

                total_loss = total_loss + policy_loss + kl_loss
                total_kl += float(kl_div.sum().item())
                n_steps += 1

        if n_steps < 1:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0.0

        return total_loss / n_steps, total_kl / n_steps

    def train_step(
        self, env_fn: Callable[[], Any], n_envs: Optional[int] = None
    ) -> Dict[str, Any]:
        """Single GRPO training step: collect group + update policy.

        Args:
            env_fn: Zero-argument callable returning a Gymnasium environment.
            n_envs: Number of parallel environments.

        Returns:
            Dict with training metrics.
        """
        trajectories, metrics = self.collect_group(env_fn, n_envs)

        if len(trajectories) < 2:
            return {
                "error": "not enough trajectories for GRPO update",
                "n_trajectories": len(trajectories),
            }

        self.policy.train()
        self.optimizer.zero_grad()

        loss, kl = self.compute_grpo_loss(trajectories)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        return {
            "grpo_loss": float(loss.item()),
            "kl_divergence": float(kl),
            "mean_return": metrics["mean_return"],
            "mean_sharpe": metrics["mean_sharpe"],
            "n_trajectories": metrics["n_trajectories"],
        }


# ---------------------------------------------------------------------------
# Environment Factory
# ---------------------------------------------------------------------------


def make_env_fn(
    asset_symbol: str = "AAPL",
    initial_cash: float = 1_000_000,
    max_episode_length: int = 252,
    lookback_window: int = 30,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    data_source: str = "synthetic",
    seed: Optional[int] = None,
) -> Callable[[], SingleAssetTradingEnv]:
    """Create a factory function for the SingleAssetTradingEnv.

    Args:
        asset_symbol: Ticker symbol for the asset.
        initial_cash: Starting portfolio cash.
        max_episode_length: Maximum steps per episode.
        lookback_window: Number of past observations to include.
        commission_rate: Trading commission as a fraction.
        slippage_rate: Slippage as a fraction.
        data_source: 'synthetic' or 'real'.
        seed: Random seed for reproducibility.

    Returns:
        Zero-argument callable that creates a SingleAssetTradingEnv.
    """
    asset = AssetConfig(
        symbol=asset_symbol,
        name=asset_symbol,
        sector="Technology",
    )
    txn_costs = TransactionCosts(
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
    )
    risk_constraints = RiskConstraints(
        max_leverage=2.0,
        max_position_size=0.3,
        max_drawdown=0.15,
    )

    def _make_env() -> SingleAssetTradingEnv:
        return SingleAssetTradingEnv(
            asset=asset,
            initial_cash=initial_cash,
            max_episode_length=max_episode_length,
            lookback_window=lookback_window,
            transaction_costs=txn_costs,
            risk_constraints=risk_constraints,
            data_source=data_source,
            seed=seed,
        )

    return _make_env


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for DPO/GRPO training."""
    parser = argparse.ArgumentParser(
        description="DPO/GRPO training for trading policies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    parser.add_argument(
        "--mode",
        choices=["dpo", "grpo"],
        default="grpo",
        help="Training mode: Direct Preference Optimization or Group Relative Policy Optimization",
    )

    # Model
    parser.add_argument(
        "--model",
        default="models/qwen_final_model_200k.zip",
        help="Path to pretrained SB3 PPO model (.zip)",
    )
    parser.add_argument(
        "--output",
        default="checkpoints/dpo_trained",
        help="Output directory for checkpoints and final model",
    )

    # Training hyperparameters
    parser.add_argument(
        "--iterations", type=int, default=100, help="Number of training iterations"
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="Number of trajectories per group (GRPO)",
    )
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=100,
        help="Number of trajectories to collect per iteration (DPO)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments for trajectory collection",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="KL penalty coefficient (GRPO) or DPO beta temperature",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.2,
        help="PPO clipping range for GRPO",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate for AdamW optimizer",
    )
    parser.add_argument(
        "--top-pct",
        type=float,
        default=0.3,
        help="Fraction of top trajectories to use as 'chosen' (DPO)",
    )
    parser.add_argument(
        "--bottom-pct",
        type=float,
        default=0.3,
        help="Fraction of bottom trajectories to use as 'rejected' (DPO)",
    )

    # Checkpointing
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N iterations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args(argv)


def train_dpo(args: argparse.Namespace) -> None:
    """Run DPO training loop.

    At each iteration:
      1. Collect N trajectories from a parallel environment
      2. Build preference pairs from top/bottom ranked trajectories
      3. Update policy with DPO loss
      4. Save checkpoint
    """
    print("=" * 60)
    print("  DPO TRAINING")
    print("=" * 60)

    # Load model
    print(f"Loading model from {args.model} ...")
    model = PPO.load(args.model, device="auto")
    policy = model.policy
    policy.eval()
    print(f"  Policy device: {next(policy.parameters()).device}")

    # Setup
    env_fn = make_env_fn(seed=args.seed)
    collector = TrajectoryCollector(env=None, policy=policy)
    trainer = DPOTrainer(
        policy=policy,
        beta=args.beta,
        learning_rate=args.lr,
    )

    os.makedirs(args.output, exist_ok=True)

    # Metrics tracking
    history: List[Dict[str, Any]] = []

    for iteration in range(args.iterations):
        iter_start = time.time()
        print(f"\n--- Iteration {iteration + 1}/{args.iterations} ---")

        # 1. Collect trajectories
        env = DummyVecEnv([env_fn])
        collector.env = env
        trajectories = collector.collect_trajectories(
            n_trajectories=args.n_trajectories,
        )
        env.close()

        scores = [t.score for t in trajectories]
        print(
            f"  Trajectories: {len(trajectories)} | "
            f"Score: mean={np.mean(scores):.4f} "
            f"best={max(scores):.4f} worst={min(scores):.4f}"
        )

        # 2. Build preference pairs
        pairs = PreferenceBuilder.build_pairs(
            trajectories,
            top_pct=args.top_pct,
            bottom_pct=args.bottom_pct,
            rng_seed=args.seed + iteration,
        )
        print(f"  Preference pairs: {len(pairs)}")

        # 3. Train
        if pairs:
            metrics = trainer.train_step(pairs)
            print(
                f"  DPO loss: {metrics['dpo_loss']:.6f}  "
                f"(valid pairs: {metrics['n_pairs']})"
            )
        else:
            metrics = {"dpo_loss": 0.0, "n_pairs": 0}
            print("  SKIP: no valid preference pairs")

        elapsed = time.time() - iter_start
        metrics["iteration"] = iteration + 1
        metrics["mean_score"] = float(np.mean(scores))
        metrics["best_score"] = float(max(scores))
        metrics["time_seconds"] = round(elapsed, 2)
        history.append(metrics)

        # 4. Save checkpoint
        if (iteration + 1) % args.save_every == 0:
            cp_path = os.path.join(args.output, f"dpo_iter_{iteration + 1}.zip")
            model.save(cp_path)
            print(f"  Checkpoint saved: {cp_path}")

    # Save final model
    final_path = os.path.join(args.output, "dpo_final.zip")
    model.save(final_path)

    # Save training history
    history_path = os.path.join(args.output, "dpo_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)

    print(f"\nTraining complete!")
    print(f"  Final model: {final_path}")
    print(f"  History:     {history_path}")
    print(f"  Iterations:  {args.iterations}")


def train_grpo(args: argparse.Namespace) -> None:
    """Run GRPO training loop.

    At each iteration:
      1. Collect a group of trajectories in parallel
      2. Score them and compute group-normalized advantages
      3. Update policy with clipped surrogate + KL penalty
      4. Save checkpoint
    """
    print("=" * 60)
    print("  GRPO TRAINING")
    print("=" * 60)

    # Load model
    print(f"Loading model from {args.model} ...")
    model = PPO.load(args.model, device="auto")
    policy = model.policy
    policy.eval()
    print(f"  Policy device: {next(policy.parameters()).device}")

    # Setup
    env_fn = make_env_fn(seed=args.seed)
    trainer = GRPOTrainer(
        policy=policy,
        beta=args.beta,
        epsilon=args.epsilon,
        group_size=args.group_size,
        learning_rate=args.lr,
    )

    os.makedirs(args.output, exist_ok=True)

    # Metrics tracking
    history: List[Dict[str, Any]] = []

    for iteration in range(args.iterations):
        iter_start = time.time()
        print(f"\n--- Iteration {iteration + 1}/{args.iterations} ---")

        # 1. Collect group + train
        metrics = trainer.train_step(
            env_fn,
            n_envs=min(args.n_envs, args.group_size),
        )

        if "error" in metrics:
            print(f"  SKIP: {metrics['error']}")
            continue

        elapsed = time.time() - iter_start
        print(
            f"  Loss: {metrics['grpo_loss']:.6f} | "
            f"KL: {metrics['kl_divergence']:.6f} | "
            f"Return: {metrics['mean_return']:.4f} | "
            f"Sharpe: {metrics['mean_sharpe']:.4f} | "
            f"({elapsed:.1f}s)"
        )

        metrics["iteration"] = iteration + 1
        metrics["time_seconds"] = round(elapsed, 2)
        history.append(metrics)

        # 2. Save checkpoint
        if (iteration + 1) % args.save_every == 0:
            cp_path = os.path.join(args.output, f"grpo_iter_{iteration + 1}.zip")
            model.save(cp_path)
            print(f"  Checkpoint saved: {cp_path}")

    # Save final model
    final_path = os.path.join(args.output, "grpo_final.zip")
    model.save(final_path)

    # Save training history
    history_path = os.path.join(args.output, "grpo_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)

    print(f"\nTraining complete!")
    print(f"  Final model: {final_path}")
    print(f"  History:     {history_path}")
    print(f"  Iterations:  {args.iterations}")


def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point for DPO/GRPO training.

    Parses command-line arguments and runs the appropriate training mode.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).
    """
    args = parse_args(argv)

    print(f"Starting {args.mode.upper()} training...")
    print(f"  Output: {args.output}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Beta: {args.beta}")
    print(f"  LR: {args.lr}")
    print()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == "dpo":
        train_dpo(args)
    elif args.mode == "grpo":
        train_grpo(args)


# ---------------------------------------------------------------------------
# Script Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
