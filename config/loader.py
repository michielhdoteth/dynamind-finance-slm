"""
Configuration Loader for Financial Trading RL Gym.

Loads training configuration from YAML files. Provides a single source
of truth for all hyperparameters used across the training pipeline.

Usage:
    from config.loader import load_config, get_config
    
    cfg = load_config()  # Loads config/training_defaults.yaml
    cfg = load_config("config/custom_config.yaml")
    
    lr = cfg["ppo"]["learning_rate"]
    model_name = cfg["model"]["name"]
"""

import os
from pathlib import Path
from typing import Any, Dict


def _find_project_root() -> Path:
    """Find the project root by looking for the config directory."""
    # Try common locations relative to this file
    candidates = [
        Path(__file__).parent.parent,  # config/../  (project root)
        Path.cwd(),                     # current working directory
    ]
    for c in candidates:
        config_dir = c / "config"
        if config_dir.exists() and (config_dir / "training_defaults.yaml").exists():
            return c
    return candidates[0]


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load training configuration from a YAML file.
    
    Args:
        config_path: Path to YAML config file. If None, loads default config.
        
    Returns:
        Dictionary with configuration values.
    """
    try:
        import yaml
    except ImportError:
        return _load_config_fallback(config_path)
    
    if config_path is None:
        root = _find_project_root()
        config_path = str(root / "config" / "training_defaults.yaml")
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    return cfg if cfg else {}


def _load_config_fallback(config_path: str = None) -> Dict[str, Any]:
    """Fallback config loader using basic Python parsing (no PyYAML dependency)."""
    if config_path is None:
        root = _find_project_root()
        config_path = str(root / "config" / "training_defaults.yaml")
    
    cfg: Dict[str, Any] = {}
    current_key = None
    
    with open(config_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            
            # Check for top-level key
            if not stripped.startswith(" ") and not stripped.startswith("\t"):
                if stripped.endswith(":"):
                    current_key = stripped.rstrip(":")
                    cfg[current_key] = {}
                    continue
            
            # Check for nested values
            if ":" in stripped and current_key:
                key, _, value = stripped.partition(":")
                key = key.strip()
                value = value.strip()
                
                # Skip nested dicts
                if not value:
                    continue
                
                # Parse value types
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                elif value.lower() == "null" or value.lower() == "none":
                    value = None
                else:
                    try:
                        if "." in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass  # Keep as string
                
                cfg[current_key][key] = value
    
    return cfg


def get_config(key: str = None, default: Any = None, config_path: str = None) -> Any:
    """Get a specific config value using dot notation.
    
    Args:
        key: Dot-separated key path (e.g., "ppo.learning_rate")
        default: Default value if key not found
        config_path: Path to config file
        
    Returns:
        Config value at the specified key path.
    """
    cfg = load_config(config_path)
    
    if key is None:
        return cfg
    
    parts = key.split(".")
    value = cfg
    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return default
    
    return value
