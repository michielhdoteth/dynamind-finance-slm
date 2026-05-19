"""Verify that all Phase 1 training modules import correctly."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 1. Config
from config.loader import load_config, get_config
cfg = load_config()
print(f"[OK] config.loader - keys: {list(cfg.keys())}, PPO lr: {cfg['ppo']['learning_rate']}")
print(f"     Curriculum stages: {len(cfg['curriculum']['stages'])}")
print(f"     Model variants: {len(cfg['model_variants'])}")

# 2. training __init__
from training import (
    ExperimentTracker, ModelEvaluator, OfflineTrainer, OnlineTrainer,
    SweepResult, PageHinkleyDetector, CurriculumStage
)
print("[OK] training.__init__ - all symbols exported")

# 3. model_sweep
from training.model_sweep import SweepResult, QwenFeaturesExtractor, run_sweep, train_model
print("[OK] training.model_sweep")

# 4. extended_training
from training.extended_training import (
    ExtendedTrainingCallback, PageHinkleyDetector, cosine_annealing, train_extended
)
lr = cosine_annealing(250000, 500000, 1e-5, 3e-4)
print(f"[OK] training.extended_training - cosine LR at midpoint: {lr:.6f}")

# 5. curriculum
from training.curriculum import CurriculumStage, CurriculumCallback, train_with_curriculum
print("[OK] training.curriculum")

# 6. multi_asset_training
from training.multi_asset_training import (
    build_assets, train_multi_asset, evaluate_model, MultiAssetMetricsCallback
)
print("[OK] training.multi_asset_training")

print("\n[PASS] All Phase 1 training modules imported successfully.")
