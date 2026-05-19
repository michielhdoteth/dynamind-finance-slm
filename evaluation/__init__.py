"""
Evaluation Module for RL Financial Markets Gym

Provides financial metrics, risk analysis, and benchmark evaluation
for reinforcement learning trading agents. Backed by the training
package's model evaluation system.
"""

from training.model_evaluator import (
    BenchmarkEvaluator as BenchmarkSuite,
    ModelEvaluator,
    PerformanceMetrics as FinancialMetrics,
    RiskAnalyzer as RiskMetrics,
)

__all__ = [
    "FinancialMetrics",
    "RiskMetrics",
    "BenchmarkSuite",
    "ModelEvaluator",
]
