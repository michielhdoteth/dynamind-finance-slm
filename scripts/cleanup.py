#!/usr/bin/env python3
"""
Cleanup script for Financial Trading RL Gym Open Source Release.
Removes development artifacts, large files, and temporary data.
"""

import glob
import os
import shutil
from pathlib import Path


def remove_pycache():
    """Remove all __pycache__ directories and .pyc files."""
    print("Removing __pycache__ directories...")

    # Find and remove __pycache__ directories
    for pycache_dir in Path(".").rglob("__pycache__"):
        if pycache_dir.is_dir():
            print(f"  Removing: {pycache_dir}")
            shutil.rmtree(pycache_dir)

    # Find and remove .pyc files
    for pyc_file in Path(".").rglob("*.pyc"):
        print(f"  Removing: {pyc_file}")
        pyc_file.unlink()

    # Find and remove .pyo files
    for pyo_file in Path(".").rglob("*.pyo"):
        print(f"  Removing: {pyo_file}")
        pyo_file.unlink()


def remove_logs_and_cache():
    """Remove log files and cache directories."""
    print("Removing logs and cache directories...")

    # Remove log directories
    log_dirs = ["logs", "tensorboard_logs", "data_cache", "test_data_cache"]
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            print(f"  Removing: {log_dir}/")
            shutil.rmtree(log_dir)

    # Remove large log files in root
    for log_file in Path(".").glob("*.log"):
        print(f"  Removing: {log_file}")
        log_file.unlink()


def remove_large_artifacts():
    """Remove large model artifacts and temporary files."""
    print("Removing large artifacts...")

    # Remove model directories (keep for now, user can manually delete)
    model_dirs = ["models", "models_backup"]
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            print(f"  Found large directory: {model_dir}/ (consider manual removal)")

    # Remove temporary files
    temp_patterns = ["*.tmp", "*.temp", "*.swp", "*.swo", ".DS_Store", "Thumbs.db"]

    for pattern in temp_patterns:
        for temp_file in Path(".").rglob(pattern):
            print(f"  Removing: {temp_file}")
            temp_file.unlink()


def remove_research_artifacts():
    """Remove research-specific artifacts that shouldn't be in OSS."""
    print("Cleaning up research artifacts...")

    research_dirs = [
        "research_results",
        "validation_results",
        "changepoint_analysis",
        "confidence_analysis",
        "consolidation_dashboard",
        "pipeline_results",
        "publication_package",
        "effect_validation",
    ]

    for research_dir in research_dirs:
        if os.path.exists(research_dir):
            print(f"  Moving to research_backup: {research_dir}/")
            backup_dir = f"research_backup_{research_dir}"
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
            shutil.move(research_dir, backup_dir)


def create_clean_structure():
    """Create proper directory structure for OSS release."""
    print("Creating clean directory structure...")

    dirs_to_create = [
        "examples",
        "docs",
        "scripts",
        "research/papers",
        "research/results",
        "tests/integration",
        "tests/unit",
    ]

    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"  Created: {dir_path}/")


def main():
    """Main cleanup function."""
    print("Starting Financial Trading RL Gym cleanup for OSS release...")
    print("=" * 60)

    # Change to repository root
    script_dir = Path(__file__).parent
    os.chdir(script_dir.parent)

    # Run cleanup steps
    remove_pycache()
    remove_logs_and_cache()
    remove_large_artifacts()
    remove_research_artifacts()
    create_clean_structure()

    print("=" * 60)
    print("Cleanup completed!")
    print("\nNext steps:")
    print("1. Review large directories (models/) and remove manually if needed")
    print("2. Update setup.py and README.md")
    print("3. Add LICENSE file")
    print("4. Run tests to ensure everything still works")


if __name__ == "__main__":
    main()
