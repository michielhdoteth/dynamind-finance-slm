"""
Professional Training Pipeline for RL Financial Markets Gym

Advanced training infrastructure for reinforcement learning models with
offline batch training, online learning, comprehensive experiment tracking,
model sweeping, extended training, hyperparameter tuning, curriculum learning,
multi-asset portfolio training, and Phase 2 post-training pipeline (financial
pretraining, supervised fine-tuning, reward modeling, RLHF alignment, multi-task agent).
"""

from .experiment_tracker import ExperimentTracker
from .model_evaluator import ModelEvaluator
from .offline_trainer import OfflineTrainer
from .online_trainer import OnlineTrainer
from .model_sweep import SweepResult, QwenFeaturesExtractor as SweepFeaturesExtractor
from .extended_training import ExtendedTrainingCallback, PageHinkleyDetector, cosine_annealing
from .curriculum import CurriculumStage, CurriculumCallback

# Phase 2 Post-Training Pipeline
from .financial_pretraining import PretrainConfig, FinancialTextDataset, prepare_dataset as prepare_pretrain_dataset, train_pretrain, evaluate as evaluate_pretrain
from .supervised_finetuning import SFTConfig, ExpertTrajectoryDataset, prepare_dataset as prepare_sft_dataset, train_sft, evaluate_sft
from .reward_model import RewardModelConfig, RewardModel, PreferencePairDataset, compute_ranking_loss, train_reward_model, evaluate_reward_model
from .rlhf_alignment import RLHFConfig, RewardModelWrapper, SimulatedTradingEnv, PPOTrainer, AdaptiveKLController, train_rlhf, evaluate_rlhf
from .multi_task_agent import MultiTaskConfig, MultiTaskQwenAgent, MultiTaskDataset, compute_multi_task_loss, train_multi_task, evaluate_multi_task, predict_multi_task

# Kronos-Enhanced Training
from .kronos_training import KronosTrainingConfig, KronosTrainingCallback

# Phase 3: DPO / GRPO Alignment (Preference Optimization)
from .dpo_trainer import (
    DPOTrainer,
    GRPOTrainer,
    TrajectoryCollector,
    PreferenceBuilder,
    Trajectory,
    PreferencePair,
)

__all__ = [
    # Phase 1
    "OfflineTrainer",
    "OnlineTrainer",
    "ExperimentTracker",
    "ModelEvaluator",
    "SweepResult",
    "SweepFeaturesExtractor",
    "ExtendedTrainingCallback",
    "PageHinkleyDetector",
    "cosine_annealing",
    "CurriculumStage",
    "CurriculumCallback",
    # Phase 2: Financial Pretraining
    "PretrainConfig",
    "FinancialTextDataset",
    "prepare_pretrain_dataset",
    "train_pretrain",
    "evaluate_pretrain",
    # Phase 2: Supervised Fine-Tuning
    "SFTConfig",
    "ExpertTrajectoryDataset",
    "prepare_sft_dataset",
    "train_sft",
    "evaluate_sft",
    # Phase 2: Reward Model
    "RewardModelConfig",
    "RewardModel",
    "PreferencePairDataset",
    "compute_ranking_loss",
    "train_reward_model",
    "evaluate_reward_model",
    # Phase 2: RLHF Alignment
    "RLHFConfig",
    "RewardModelWrapper",
    "SimulatedTradingEnv",
    "PPOTrainer",
    "AdaptiveKLController",
    "train_rlhf",
    "evaluate_rlhf",
    # Phase 2: Multi-Task Agent
    "MultiTaskConfig",
    "MultiTaskQwenAgent",
    "MultiTaskDataset",
    "compute_multi_task_loss",
    "train_multi_task",
    "evaluate_multi_task",
    "predict_multi_task",
    # Kronos-Enhanced Training
    "KronosTrainingConfig",
    "KronosTrainingCallback",
    # Phase 3: DPO / GRPO Alignment
    "DPOTrainer",
    "GRPOTrainer",
    "TrajectoryCollector",
    "PreferenceBuilder",
    "Trajectory",
    "PreferencePair",
]
