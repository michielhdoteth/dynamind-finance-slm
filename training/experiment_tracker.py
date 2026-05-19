"""
Experiment Tracker for RL Financial Markets Gym

Comprehensive experiment tracking system for managing RL experiments,
logging results, comparing models, and analyzing performance.
"""

from datetime import datetime
import hashlib
import logging
import os
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""

    # Experiment metadata
    name: str
    description: str = ""
    tags: List[str] = None
    researcher: str = "Unknown"
    project: str = "Default"

    # Environment settings
    env_type: str = ""
    symbols: List[str] = None
    data_source: str = ""
    start_date: str = ""
    end_date: str = ""

    # Model settings
    algorithm: str = ""
    model_params: Dict = None
    policy_architecture: str = ""

    # Training settings
    total_timesteps: int = 0
    learning_rate: float = 0.0
    batch_size: int = 0
    training_time: float = 0.0

    # Evaluation results
    final_performance: float = 0.0
    eval_episodes: int = 0
    std_performance: float = 0.0

    # Risk metrics
    final_sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0

    # Additional metrics
    convergence_step: int = 0
    stability_score: float = 0.0
    computational_cost: float = 0.0

    # Status
    status: str = "running"  # running, completed, failed, stopped
    created_at: str = ""
    completed_at: str = ""

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.symbols is None:
            self.symbols = []
        if self.model_params is None:
            self.model_params = {}
        if self.created_at == "":
            self.created_at = datetime.now().isoformat()

    def get_id(self) -> str:
        """Generate unique experiment ID."""
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]


class DatabaseManager:
    """Manage experiment database."""

    def __init__(self, db_path: str = "./experiments/experiments.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()

    def init_database(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    researcher TEXT,
                    project TEXT,
                    tags TEXT,
                    config TEXT NOT NULL,
                    metrics TEXT,
                    status TEXT,
                    created_at TEXT,
                    completed_at TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS training_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    step INTEGER,
                    episode INTEGER,
                    reward REAL,
                    loss REAL,
                    metrics TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    episode INTEGER,
                    reward REAL,
                    length INTEGER,
                    metrics TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    step INTEGER,
                    checkpoint_path TEXT,
                    metrics TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """
            )

            conn.commit()

    def save_experiment(self, config: ExperimentConfig, metrics: Dict = None) -> str:
        """Save experiment to database."""
        experiment_id = config.get_id()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiments
                (id, name, description, researcher, project, tags, config, metrics, status, created_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    experiment_id,
                    config.name,
                    config.description,
                    config.researcher,
                    config.project,
                    json.dumps(config.tags),
                    json.dumps(asdict(config)),
                    json.dumps(metrics or {}),
                    config.status,
                    config.created_at,
                    config.completed_at,
                ),
            )
            conn.commit()

        return experiment_id

    def update_experiment_status(
        self, experiment_id: str, status: str, metrics: Dict = None
    ):
        """Update experiment status and metrics."""
        with sqlite3.connect(self.db_path) as conn:
            updates = ["status = ?"]
            values = [status]

            if metrics:
                updates.append("metrics = ?")
                values.append(json.dumps(metrics))

            if status == "completed":
                updates.append("completed_at = ?")
                values.append(datetime.now().isoformat())

            values.append(experiment_id)

            conn.execute(
                """
                UPDATE experiments
                SET {', '.join(updates)}
                WHERE id = ?
            """,
                values,
            )
            conn.commit()

    def log_training_step(
        self,
        experiment_id: str,
        step: int,
        episode: int,
        reward: float,
        loss: float,
        metrics: Dict = None,
    ):
        """Log training step."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO training_logs
                (experiment_id, step, episode, reward, loss, metrics)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (experiment_id, step, episode, reward, loss, json.dumps(metrics or {})),
            )
            conn.commit()

    def log_evaluation(
        self,
        experiment_id: str,
        episode: int,
        reward: float,
        length: int,
        metrics: Dict = None,
    ):
        """Log evaluation episode."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO evaluations
                (experiment_id, episode, reward, length, metrics)
                VALUES (?, ?, ?, ?, ?)
            """,
                (experiment_id, episode, reward, length, json.dumps(metrics or {})),
            )
            conn.commit()

    def save_checkpoint(
        self, experiment_id: str, step: int, checkpoint_path: str, metrics: Dict = None
    ):
        """Save model checkpoint info."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO checkpoints
                (experiment_id, step, checkpoint_path, metrics)
                VALUES (?, ?, ?, ?)
            """,
                (experiment_id, step, checkpoint_path, json.dumps(metrics or {})),
            )
            conn.commit()

    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Get experiment by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM experiments WHERE id = ?
            """,
                (experiment_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_experiments(self, filters: Dict = None, limit: int = 100) -> List[Dict]:
        """List experiments with optional filters."""
        query = "SELECT * FROM experiments"
        params = []

        if filters:
            conditions = []
            for key, value in filters.items():
                if key == "tags":
                    conditions.append("tags LIKE ?")
                    params.append(f"%{value}%")
                elif key == "status":
                    conditions.append("status = ?")
                    params.append(value)
                elif key == "researcher":
                    conditions.append("researcher = ?")
                    params.append(value)
                elif key == "project":
                    conditions.append("project = ?")
                    params.append(value)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_training_history(self, experiment_id: str) -> pd.DataFrame:
        """Get training history for an experiment."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                """
                SELECT * FROM training_logs
                WHERE experiment_id = ?
                ORDER BY step
            """,
                conn,
                params=(experiment_id,),
            )
            return df

    def get_evaluation_history(self, experiment_id: str) -> pd.DataFrame:
        """Get evaluation history for an experiment."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                """
                SELECT * FROM evaluations
                WHERE experiment_id = ?
                ORDER BY episode
            """,
                conn,
                params=(experiment_id,),
            )
            return df

    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """Compare multiple experiments."""
        placeholders = ",".join(["?"] * len(experiment_ids))
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM experiments
                WHERE id IN ({placeholders})
                ORDER BY final_performance DESC
            """,
                experiment_ids,
            )
            return pd.DataFrame([dict(row) for row in cursor.fetchall()])


class ExperimentTracker:
    """
    Comprehensive experiment tracking system.

    Features:
    - Experiment configuration and metadata management
    - Training progress logging
    - Performance metrics tracking
    - Model checkpointing
    - Experiment comparison and analysis
    - Visualization and reporting
    """

    def __init__(self, project_name: str = "Default", db_path: str = "./experiments"):
        """
        Initialize experiment tracker.

        Args:
            project_name: Name of the project
            db_path: Path to experiment database
        """
        self.project_name = project_name
        self.db_manager = DatabaseManager(os.path.join(db_path, "experiments.db"))
        self.current_experiment_id = None
        self.current_config = None

        # Create directories
        self.base_path = Path(db_path)
        self.models_path = self.base_path / "models"
        self.plots_path = self.base_path / "plots"
        self.logs_path = self.base_path / "logs"

        for path in [self.models_path, self.plots_path, self.logs_path]:
            path.mkdir(parents=True, exist_ok=True)

        logger.info(f"ExperimentTracker initialized for project: {project_name}")

    def create_experiment(self, config: ExperimentConfig) -> str:
        """
        Create a new experiment.

        Args:
            config: Experiment configuration

        Returns:
            Experiment ID
        """
        config.project = self.project_name
        experiment_id = self.db_manager.save_experiment(config)
        self.current_experiment_id = experiment_id
        self.current_config = config

        logger.info(f"Created experiment: {config.name} (ID: {experiment_id})")
        return experiment_id

    def start_experiment(self, config: ExperimentConfig) -> str:
        """
        Start a new experiment.

        Args:
            config: Experiment configuration

        Returns:
            Experiment ID
        """
        config.status = "running"
        return self.create_experiment(config)

    def log_training_step(
        self,
        step: int,
        episode: int,
        reward: float,
        loss: float = None,
        metrics: Dict = None,
    ):
        """Log training step for current experiment."""
        if self.current_experiment_id is None:
            logger.warning("No active experiment to log to")
            return

        self.db_manager.log_training_step(
            self.current_experiment_id, step, episode, reward, loss, metrics
        )

    def log_evaluation(
        self, episode: int, reward: float, length: int, metrics: Dict = None
    ):
        """Log evaluation episode for current experiment."""
        if self.current_experiment_id is None:
            logger.warning("No active experiment to log to")
            return

        self.db_manager.log_evaluation(
            self.current_experiment_id, episode, reward, length, metrics
        )

    def save_checkpoint(self, step: int, checkpoint_path: str, metrics: Dict = None):
        """Save model checkpoint for current experiment."""
        if self.current_experiment_id is None:
            logger.warning("No active experiment to save checkpoint for")
            return

        self.db_manager.save_checkpoint(
            self.current_experiment_id, step, checkpoint_path, metrics
        )

    def complete_experiment(self, final_metrics: Dict = None):
        """Mark current experiment as completed."""
        if self.current_experiment_id is None:
            logger.warning("No active experiment to complete")
            return

        if self.current_config:
            self.current_config.status = "completed"
            self.current_config.completed_at = datetime.now().isoformat()

            # Update config with final metrics
            if final_metrics:
                for key, value in final_metrics.items():
                    if hasattr(self.current_config, key):
                        setattr(self.current_config, key, value)

            self.db_manager.save_experiment(self.current_config, final_metrics)
            self.db_manager.update_experiment_status(
                self.current_experiment_id, "completed", final_metrics
            )

        logger.info(f"Completed experiment: {self.current_experiment_id}")

    def fail_experiment(self, error_message: str = ""):
        """Mark current experiment as failed."""
        if self.current_experiment_id is None:
            logger.warning("No active experiment to fail")
            return

        if self.current_config:
            self.current_config.status = "failed"
            self.db_manager.save_experiment(
                self.current_config, {"error": error_message}
            )

        self.db_manager.update_experiment_status(
            self.current_experiment_id, "failed", {"error": error_message}
        )

        logger.error(
            f"Failed experiment: {self.current_experiment_id} - {error_message}"
        )

    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Get experiment details."""
        return self.db_manager.get_experiment(experiment_id)

    def list_experiments(self, filters: Dict = None, limit: int = 50) -> List[Dict]:
        """List experiments."""
        filters = filters or {}
        filters["project"] = self.project_name
        return self.db_manager.list_experiments(filters, limit)

    def get_training_curve(self, experiment_id: str) -> pd.DataFrame:
        """Get training curve for an experiment."""
        return self.db_manager.get_training_history(experiment_id)

    def get_evaluation_curve(self, experiment_id: str) -> pd.DataFrame:
        """Get evaluation curve for an experiment."""
        return self.db_manager.get_evaluation_history(experiment_id)

    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """Compare multiple experiments."""
        return self.db_manager.compare_experiments(experiment_ids)

    def plot_training_curves(self, experiment_ids: List[str], save_path: str = None):
        """Plot training curves for multiple experiments."""
        if save_path is None:
            save_path = (
                self.plots_path
                / f"training_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )

        plt.figure(figsize=(15, 10))

        # Rewards over time
        plt.subplot(2, 2, 1)
        for exp_id in experiment_ids:
            df = self.get_training_curve(exp_id)
            if not df.empty:
                exp_config = self.get_experiment(exp_id)
                label = exp_config["name"] if exp_config else exp_id[:8]
                plt.plot(df["step"], df["reward"], label=label, alpha=0.7)

        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.title("Training Rewards")
        plt.legend()
        plt.grid(True)

        # Loss over time
        plt.subplot(2, 2, 2)
        for exp_id in experiment_ids:
            df = self.get_training_curve(exp_id)
            if not df.empty and "loss" in df.columns:
                exp_config = self.get_experiment(exp_id)
                label = exp_config["name"] if exp_config else exp_id[:8]
                plt.plot(df["step"], df["loss"], label=label, alpha=0.7)

        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.grid(True)

        # Evaluation rewards
        plt.subplot(2, 2, 3)
        for exp_id in experiment_ids:
            df = self.get_evaluation_curve(exp_id)
            if not df.empty:
                exp_config = self.get_experiment(exp_id)
                label = exp_config["name"] if exp_config else exp_id[:8]
                plt.plot(df["episode"], df["reward"], label=label, alpha=0.7)

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Evaluation Rewards")
        plt.legend()
        plt.grid(True)

        # Performance comparison
        plt.subplot(2, 2, 4)
        comparison = self.compare_experiments(experiment_ids)
        if not comparison.empty:
            performance = comparison[["name", "final_performance"]].dropna()
            if not performance.empty:
                plt.bar(performance["name"], performance["final_performance"])
                plt.xticks(rotation=45)
                plt.ylabel("Final Performance")
                plt.title("Performance Comparison")
                plt.grid(True, axis="y")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Training curves plot saved to {save_path}")

    def generate_experiment_report(self, experiment_id: str) -> str:
        """Generate comprehensive experiment report."""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return "Experiment not found"

        training_df = self.get_training_curve(experiment_id)
        evaluation_df = self.get_evaluation_curve(experiment_id)

        report = """
# Experiment Report: {experiment['name']}

## Configuration
- **ID**: {experiment_id}
- **Algorithm**: {experiment['config'].get('algorithm', 'N/A')}
- **Environment**: {experiment['config'].get('env_type', 'N/A')}
- **Symbols**: {experiment['config'].get('symbols', [])}
- **Total Timesteps**: {experiment['config'].get('total_timesteps', 0):,}
- **Learning Rate**: {experiment['config'].get('learning_rate', 0)}
- **Status**: {experiment['status']}
- **Created**: {experiment['created_at']}
- **Completed**: {experiment.get('completed_at', 'N/A')}

## Performance Metrics
"""

        # Parse config for performance metrics
        try:
            config = json.loads(experiment["config"])
            performance_metrics = [
                ("Final Performance", config.get("final_performance", 0)),
                ("Sharpe Ratio", config.get("final_sharpe_ratio", 0)),
                ("Max Drawdown", config.get("max_drawdown", 0)),
                ("VaR (95%)", config.get("var_95", 0)),
                ("CVaR (95%)", config.get("cvar_95", 0)),
                ("Convergence Step", config.get("convergence_step", 0)),
                ("Stability Score", config.get("stability_score", 0)),
            ]

            for metric, value in performance_metrics:
                if isinstance(value, float):
                    report += f"- **{metric}**: {value:.4f}\n"
                else:
                    report += f"- **{metric}**: {value}\n"
        except Exception:
            pass

        if not training_df.empty:
            report += """
## Training Statistics
- **Total Steps**: {len(training_df)}
- **Total Episodes**: {training_df['episode'].max() if 'episode' in training_df.columns else 'N/A'}
- **Mean Training Reward**: {training_df['reward'].mean():.4f}
- **Std Training Reward**: {training_df['reward'].std():.4f}
- **Max Training Reward**: {training_df['reward'].max():.4f}
- **Min Training Reward**: {training_df['reward'].min():.4f}
"""

        if not evaluation_df.empty:
            report += """
## Evaluation Statistics
- **Evaluation Episodes**: {len(evaluation_df)}
- **Mean Evaluation Reward**: {evaluation_df['reward'].mean():.4f}
- **Std Evaluation Reward**: {evaluation_df['reward'].std():.4f}
- **Max Evaluation Reward**: {evaluation_df['reward'].max():.4f}
- **Min Evaluation Reward**: {evaluation_df['reward'].min():.4f}
"""

        report += """
## Training Plots
Training curves and performance visualizations are available in the plots directory.

## Model Checkpoints
Model checkpoints are saved in the models directory.

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        # Save report
        report_path = self.logs_path / f"experiment_report_{experiment_id}.md"
        with open(report_path, "w") as f:
            f.write(report)

        logger.info(f"Experiment report saved to {report_path}")
        return str(report_path)


def main():
    """Example usage of the experiment tracker."""
    # Create experiment tracker
    tracker = ExperimentTracker(project_name="Demo Project")

    # Create experiment configuration
    config = ExperimentConfig(
        name="PPO Trading Agent",
        description="Testing PPO algorithm on single asset trading",
        tags=["PPO", "single_asset", "risk_management"],
        researcher="Demo User",
        env_type="single_asset",
        symbols=["AAPL", "MSFT"],
        algorithm="PPO",
        total_timesteps=100000,
        learning_rate=3e-4,
    )

    # Start experiment
    exp_id = tracker.start_experiment(config)
    print(f"Started experiment: {exp_id}")

    # Simulate training
    for step in range(0, 1000, 100):
        reward = (
            np.random.normal(0, 1) + step * 0.001
        )  # Simulated improving performance
        episode = step // 100
        loss = max(0.1, 1.0 - step * 0.0005)  # Simulated decreasing loss

        tracker.log_training_step(step, episode, reward, loss)

        if step % 200 == 0:
            eval_reward = np.random.normal(reward, 0.5)
            tracker.log_evaluation(episode, eval_reward, 100)

    # Complete experiment
    final_metrics = {
        "final_performance": 5.2,
        "final_sharpe_ratio": 1.8,
        "max_drawdown": 0.12,
        "convergence_step": 800,
    }

    tracker.complete_experiment(final_metrics)

    # Generate report
    report_path = tracker.generate_experiment_report(exp_id)
    print(f"Experiment report saved to: {report_path}")

    # List experiments
    experiments = tracker.list_experiments()
    print(f"Total experiments in project: {len(experiments)}")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
