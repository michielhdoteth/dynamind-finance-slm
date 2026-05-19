#!/usr/bin/env python3
"""
Advanced Qwen RL Training with Risk Management

Integrates Qwen 0.5B model with our professional training pipeline,
real market data, and advanced risk management systems.
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# RL and ML libraries
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Our gym components
from environments import SingleAssetTradingEnv
from environments.base_env import AssetConfig
from data import DataManager
from risk import RiskManager, PositionLimits, PortfolioConstraints, CVaRConfig, RiskMeasure
from training import ExperimentTracker, ExperimentConfig, ModelEvaluator, EvaluationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenPolicyExtractor(BaseFeaturesExtractor):
    """Custom feature extractor for Qwen-based policy."""

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # Simple neural network for feature extraction
        import torch.nn as nn
        self.net = nn.Sequential(
            nn.Linear(np.prod(observation_space.shape), 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim)
        )

    def forward(self, observations):
        return self.net(observations)

class TrainingCallback(BaseCallback):
    """Custom callback for training with risk management."""

    def __init__(self, experiment_tracker: ExperimentTracker, verbose: int = 0):
        super().__init__(verbose)
        self.experiment_tracker = experiment_tracker
        self.episode_count = 0
        self.total_reward = 0

    def _on_step(self) -> bool:
        # Log training step
        if len(self.training_env.envs) > 0:
            env = self.training_env.envs[0]
            if hasattr(env, 'get_current_step'):
                step_info = env.get_current_step()
                if step_info:
                    reward = step_info.get('reward', 0)
                    episode = step_info.get('episode', 0)

                    self.experiment_tracker.log_training_step(
                        step=self.num_timesteps,
                        episode=episode,
                        reward=reward
                    )

        return True

    def _on_rollout_end(self) -> None:
        # Log episode completion
        self.episode_count += 1
        logger.info(f"Episode {self.episode_count} completed")

def create_training_env(
    symbols: List[str],
    start_date: str,
    end_date: str,
    data_source: str = "yahoo",
    enable_risk_management: bool = True,
    risk_aversion: float = 1.0,
    max_position_size: float = 0.3
) -> Any:
    """Create training environment with real data and risk management."""

    # Load real market data
    data_manager = DataManager()
    data = data_manager.get_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        source=data_source,
        frequency="1d"
    )

    logger.info(f"Loaded real market data: {data.shape}")

    # Create asset configurations
    assets = []
    for symbol in symbols:
        if symbol in data.columns.get_level_values(0):
            symbol_data = data[symbol]
            if 'close' in symbol_data.columns:
                price_data = symbol_data['close'].dropna()
                if len(price_data) > 0:
                    asset = AssetConfig(
                        symbol=symbol,
                        sector="Technology" if symbol in ["AAPL", "MSFT", "GOOGL"] else "Finance",
                        price_history=price_data.tolist(),
                        volatility=price_data.pct_change().std()
                    )
                    assets.append(asset)

    if not assets:
        raise ValueError("No valid assets found in data")

    logger.info(f"Created {len(assets)} asset configurations")

    # Setup risk management if enabled
    risk_manager = None
    if enable_risk_management:
        position_limits = PositionLimits(
            max_position_size=max_position_size,
            max_sector_exposure=0.5,
            min_diversification=1
        )

        portfolio_constraints = PortfolioConstraints(
            max_leverage=2.0,
            max_drawdown_limit=0.15,
            var_limit=0.04
        )

        cvar_config = CVaRConfig(
            confidence_level=0.05,
            window_size=30,
            risk_aversion=risk_aversion,
            reward_shaping_method="cvar_adjustment"
        )

        risk_manager = RiskManager(
            assets=assets,
            position_limits=position_limits,
            portfolio_constraints=portfolio_constraints,
            cvar_config=cvar_config,
            enable_risk_shaping=True
        )

        logger.info("Risk management system initialized")

    # Create environment
    env = SingleAssetTradingEnv(
        assets=assets[:1],  # Use first asset for single asset trading
        initial_balance=100000,
        max_shares=1000,
        transaction_fee_pct=0.001,
        lookback_window=30,
        risk_manager=risk_manager,
        render_mode=None
    )

    return env

def create_evaluation_env(
    symbols: List[str],
    start_date: str,
    end_date: str,
    data_source: str = "yahoo"
) -> Any:
    """Create separate evaluation environment with different data period."""

    # Shift evaluation period forward
    train_end = datetime.strptime(end_date, "%Y-%m-%d")
    eval_start = (train_end + timedelta(days=1)).strftime("%Y-%m-%d")
    eval_end = (train_end + timedelta(days=90)).strftime("%Y-%m-%d")

    return create_training_env(
        symbols=symbols,
        start_date=eval_start,
        end_date=eval_end,
        data_source=data_source,
        enable_risk_management=True
    )

def train_qwen_with_risk_management(
    symbols: List[str] = ["AAPL", "MSFT"],
    start_date: str = "2020-01-01",
    end_date: str = "2022-12-31",
    total_timesteps: int = 100000,
    learning_rate: float = 3e-5,
    risk_aversion: float = 1.0,
    enable_risk_management: bool = True,
    model_save_path: str = "qwen_advanced_model",
    experiment_name: str = "Qwen_Advanced_Trading"
) -> tuple:
    """
    Train Qwen model with advanced risk management.

    Returns:
        tuple: (trained_model, training_results)
    """

    logger.info("Starting advanced Qwen RL training with risk management...")

    # Initialize experiment tracking
    experiment_tracker = ExperimentTracker(project_name="Qwen_Advanced_Trading")

    # Create experiment configuration
    exp_config = ExperimentConfig(
        name=experiment_name,
        description=f"Qwen RL training with risk management on {symbols}",
        tags=["Qwen", "PPO", "risk_management", "real_data"],
        env_type="single_asset",
        symbols=symbols,
        algorithm="PPO",
        total_timesteps=total_timesteps,
        learning_rate=learning_rate,
        enable_risk_management=enable_risk_management,
        risk_aversion=risk_aversion
    )

    # Start experiment
    experiment_id = experiment_tracker.start_experiment(exp_config)
    logger.info(f"Started experiment: {experiment_id}")

    try:
        # Create training environment
        train_env = create_training_env(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            enable_risk_management=enable_risk_management,
            risk_aversion=risk_aversion
        )

        # Wrap for stable-baselines3
        train_env = DummyVecEnv([lambda: train_env])

        # Create evaluation environment
        eval_env = create_evaluation_env(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        eval_env = DummyVecEnv([lambda: eval_env])

        # Setup PPO model with custom policy
        policy_kwargs = dict(
            features_extractor_class=QwenPolicyExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=[dict(pi=[256, 128], vf=[256, 128])],
            activation_fn=torch.nn.ReLU
        )

        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./logs/qwen_advanced",
            verbose=1,
            seed=42
        )

        logger.info("PPO model with Qwen features initialized")

        # Setup callbacks
        callbacks = [
            TrainingCallback(experiment_tracker),
            CheckpointCallback(
                save_freq=10000,
                save_path=f"./checkpoints/{experiment_name}",
                name_prefix="qwen_model"
            ),
            EvalCallback(
                eval_env=eval_env,
                eval_freq=5000,
                n_eval_episodes=10,
                deterministic=True,
                best_model_save_path=f"./models/{experiment_name}/best",
                log_path=f"./logs/{experiment_name}/eval",
                verbose=1
            )
        ]

        # Train model
        logger.info(f"Starting training for {total_timesteps:,} timesteps...")
        start_time = datetime.now()

        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )

        training_time = (datetime.now() - start_time).total_seconds()

        # Save final model
        model.save(f"./models/{experiment_name}/final")
        logger.info(f"Model saved to ./models/{experiment_name}/final")

        # Evaluate trained model
        logger.info("Evaluating trained model...")
        evaluator = ModelEvaluator(EvaluationConfig(n_episodes=50))
        eval_results = evaluator.evaluate_model(model, eval_env.envs[0])

        # Complete experiment
        final_metrics = {
            'training_time': training_time,
            'final_performance': eval_results.get('performance_metrics', {}).get('episode_metrics', {}).get('mean_reward', 0),
            'sharpe_ratio': eval_results.get('performance_metrics', {}).get('risk_adjusted_metrics', {}).get('sharpe_ratio', 0),
            'max_drawdown': eval_results.get('performance_metrics', {}).get('drawdown_metrics', {}).get('max_drawdown', 0)
        }

        experiment_tracker.complete_experiment(final_metrics)

        # Generate report
        report_path = experiment_tracker.generate_experiment_report(experiment_id)
        logger.info(f"Experiment report generated: {report_path}")

        training_results = {
            'experiment_id': experiment_id,
            'training_time': training_time,
            'evaluation_results': eval_results,
            'final_metrics': final_metrics,
            'model_path': f"./models/{experiment_name}/final"
        }

        logger.info("Advanced Qwen RL training completed successfully!")
        return model, training_results

    except Exception as e:
        logger.error(f"Training failed: {e}")
        experiment_tracker.fail_experiment(str(e))
        raise

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Advanced Qwen RL Training with Risk Management")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT"], help="Trading symbols")
    parser.add_argument("--start-date", default="2020-01-01", help="Training start date")
    parser.add_argument("--end-date", default="2022-12-31", help="Training end date")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--learning-rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--risk-aversion", type=float, default=1.0, help="Risk aversion coefficient")
    parser.add_argument("--no-risk", action="store_true", help="Disable risk management")
    parser.add_argument("--model-name", default="qwen_advanced_trading", help="Model name")

    args = parser.parse_args()

    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    if device == "cuda":
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")

    try:
        # Run training
        model, results = train_qwen_with_risk_management(
            symbols=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            total_timesteps=args.timesteps,
            learning_rate=args.learning_rate,
            risk_aversion=args.risk_aversion,
            enable_risk_management=not args.no_risk,
            model_save_path=args.model_name,
            experiment_name=args.model_name
        )

        # Print results
        print("\n" + "="*80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Experiment ID: {results['experiment_id']}")
        print(f"Training Time: {results['training_time']:.2f} seconds")
        print(f"Final Performance: {results['final_metrics']['final_performance']:.4f}")
        print(f"Sharpe Ratio: {results['final_metrics']['sharpe_ratio']:.4f}")
        print(f"Max Drawdown: {results['final_metrics']['max_drawdown']:.4f}")
        print(f"Model saved to: {results['model_path']}")

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)