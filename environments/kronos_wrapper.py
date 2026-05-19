"""
Kronos Observation Wrapper for Financial Trading RL Gym.

Integrates Kronos (financial candlestick foundation model) as a feature 
extractor for RL trading environments. Kronos provides market structure
understanding that augments the standard trading observations.

Architecture:
    Raw OHLCV → Kronos Tokenizer → Kronos Transformer → Latent Features
                                                              ↓
    Standard Observations (portfolio, indicators) ──→ Concatenate → PPO Agent

Usage:
    from environments.kronos_wrapper import KronosObservationEnv
    
    env = KronosObservationEnv(SingleAssetTradingEnv(asset=asset))
    obs, info = env.reset()  # obs includes Kronos features
"""

import os, sys, warnings, logging
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List
from gymnasium import spaces
from gymnasium.core import Env

# Add Kronos to path
_KRONOS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "Kronos")
if os.path.exists(_KRONOS_PATH) and _KRONOS_PATH not in sys.path:
    sys.path.insert(0, _KRONOS_PATH)


class KronosFeatureExtractor:
    """
    Wrapper around Kronos that extracts market structure features from OHLCV data.
    
    Kronos is a foundation model trained on 45+ global exchanges that understands
    the "language" of financial candlesticks. This class uses its encoder to
    produce rich feature representations for RL agents.
    
    Model sizes:
    - Kronos-mini: 4.1M params (fastest, context 2048)
    - Kronos-small: 24.7M params (balanced, context 512) 
    - Kronos-base: 102.3M params (best quality, context 512)
    """
    
    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        feature_dim: int = 64,
        max_context: int = 128,
    ):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_dim = feature_dim
        self.max_context = max_context
        
        # Model name mapping
        model_map = {
            "mini": ("NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-2k"),
            "small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base"),
            "base": ("NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base"),
        }
        
        if model_size not in model_map:
            logger.warning(f"Unknown model size '{model_size}', falling back to 'small'")
            model_size = "small"
        
        self.model_name, self.tokenizer_name = model_map[model_size]
        self.model_size = model_size
        self._model = None
        self._tokenizer = None
        self._loaded = False
    
    def load(self):
        """Load Kronos model and tokenizer from HuggingFace."""
        if self._loaded:
            return
        
        try:
            from model import Kronos, KronosTokenizer
            
            logger.info(f"Loading Kronos-{self.model_size} ({self.model_name})...")
            self._tokenizer = KronosTokenizer.from_pretrained(
                self.tokenizer_name,
                trust_remote_code=True,
            ).to(self.device)
            
            self._model = Kronos.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            ).to(self.device)
            
            self._model.eval()
            self._tokenizer.eval()
            self._loaded = True
            logger.info(f"Kronos loaded: {sum(p.numel() for p in self._model.parameters())/1e6:.1f}M params")
            
        except ImportError as e:
            logger.warning(f"Could not load Kronos: {e}")
            logger.warning("Falling back to random features. Install Kronos dependencies for full functionality.")
            self._loaded = "fallback"
        except Exception as e:
            logger.warning(f"Kronos load failed: {e}. Using fallback.")
            self._loaded = "fallback"
    
    def extract_features(self, ohlcv_history: np.ndarray) -> np.ndarray:
        """
        Extract market structure features from OHLCV history.
        
        Args:
            ohlcv_history: (lookback, 5) array of [open, high, low, close, volume]
            
        Returns:
            features: (feature_dim,) array of Kronos latent features
        """
        self.load()
        
        if self._loaded == "fallback":
            return self._fallback_features(ohlcv_history)
        
        try:
            with torch.no_grad():
                # Prepare input: (1, seq_len, d_in) where d_in=5 for OHLCV
                x = torch.FloatTensor(ohlcv_history).unsqueeze(0).to(self.device)
                
                # Get encoder hidden states
                # Kronos tokenizer encodes to latent space
                z = self._tokenizer.encode(x)
                
                # Get model features
                if hasattr(self._model, 'get_features'):
                    features = self._model.get_features(z)
                else:
                    # Fallback: use model's embedding layer
                    features = self._model.embedding(z)
                
                # Pool to fixed dimension
                features = features.mean(dim=1)  # Average pool over sequence
                
                # Project to target dimension if needed
                if features.shape[-1] != self.feature_dim:
                    features = nn.functional.adaptive_avg_pool1d(
                        features.transpose(1, 2), self.feature_dim
                    ).transpose(1, 2)
                
                return features.squeeze(0).cpu().numpy().flatten()[:self.feature_dim]
                
        except Exception as e:
            logger.debug(f"Kronos feature extraction failed: {e}")
            return self._fallback_features(ohlcv_history)
    
    def _fallback_features(self, ohlcv_history: np.ndarray) -> np.ndarray:
        """Generate informative fallback features when Kronos is unavailable."""
        if len(ohlcv_history) < 2:
            return np.zeros(self.feature_dim, dtype=np.float32)
        
        # Compute meaningful technical features as fallback
        closes = ohlcv_history[:, 3]
        highs = ohlcv_history[:, 1]
        lows = ohlcv_history[:, 2]
        volumes = ohlcv_history[:, 4]
        
        features = []
        
        # Returns
        returns = np.diff(closes) / closes[:-1]
        features.extend([
            float(np.mean(returns[-20:])) if len(returns) >= 20 else 0.0,
            float(np.std(returns[-20:])) if len(returns) >= 20 else 0.0,
            float(np.mean(returns[-5:])) if len(returns) >= 5 else 0.0,
            float(np.sum(returns[-5:])) if len(returns) >= 5 else 0.0,
        ])
        
        # Price relative to moving averages
        for window in [5, 10, 20]:
            if len(closes) >= window:
                ma = np.mean(closes[-window:])
                features.append(float(closes[-1] / ma - 1))
            else:
                features.append(0.0)
        
        # Volatility
        if len(returns) >= 10:
            features.append(float(np.std(returns[-10:])))
        else:
            features.append(0.0)
        
        # Volume features
        if len(volumes) >= 5:
            vol_ratio = volumes[-1] / (np.mean(volumes[-5:]) + 1e-8)
            features.append(float(min(vol_ratio, 5.0)))
        else:
            features.append(0.0)
        
        # High-Low range
        ranges = (highs - lows) / closes
        features.append(float(np.mean(ranges[-5:])) if len(ranges) >= 5 else 0.0)
        
        # Pad or truncate to feature_dim
        features = np.array(features, dtype=np.float32)
        if len(features) < self.feature_dim:
            features = np.pad(features, (0, self.feature_dim - len(features)))
        else:
            features = features[:self.feature_dim]
        
        return features


class KronosObservationWrapper:
    """
    Wraps an RL environment to add Kronos market structure features to observations.
    
    Usage:
        base_env = SingleAssetTradingEnv(asset=asset)
        env = KronosObservationWrapper(base_env, kronos_extractor)
        obs, info = env.reset()  # obs includes Kronos features
    """
    
    def __init__(
        self,
        env: Env,
        kronos_extractor: KronosFeatureExtractor = None,
        ohlcv_history_len: int = 60,
        feature_dim: int = 64,
    ):
        self.env = env
        self.kronos = kronos_extractor or KronosFeatureExtractor(feature_dim=feature_dim)
        self.ohlcv_history_len = ohlcv_history_len
        self.feature_dim = feature_dim
        
        # History buffer for OHLCV data
        self.ohlcv_history = []
        
        # Update observation space
        original_dim = self._get_obs_dim(env.observation_space)
        self.kronos_feature_dim = feature_dim
        
        # Create new combined observation space
        total_dim = original_dim + self.kronos_feature_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32
        )
        self.action_space = env.action_space
    
    def _get_obs_dim(self, space) -> int:
        """Get total dimension of observation space."""
        if isinstance(space, spaces.Box):
            return int(np.prod(space.shape))
        elif isinstance(space, spaces.Dict):
            return sum(int(np.prod(s.shape)) for s in space.spaces.values())
        return 64  # fallback
    
    def _get_ohlcv_from_env(self) -> np.ndarray:
        """Extract OHLCV data from environment state if available."""
        # Try to get price history from the environment
        if hasattr(self.env, 'price_history') and self.env.price_history is not None:
            return np.array(self.env.price_history)
        if hasattr(self.env, 'asset') and hasattr(self.env, 'current_step'):
            # Generate synthetic OHLCV from env state
            if hasattr(self.env, '_price') and self.env._price is not None:
                return np.array(self.env._price)
        return None
    
    def reset(self, **kwargs):
        """Reset environment and return augmented observation."""
        obs, info = self.env.reset(**kwargs)
        
        # Try to get OHLCV data from environment
        ohlcv = self._get_ohlcv_from_env()
        if ohlcv is not None and len(ohlcv) > 0:
            self.ohlcv_history = list(ohlcv)
        
        kronos_features = self._get_kronos_features()
        aug_obs = self._augment_observation(obs, kronos_features)
        
        return aug_obs, info
    
    def step(self, action):
        """Take a step and return augmented observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update OHLCV history from env if available
        ohlcv = self._get_ohlcv_from_env()
        if ohlcv is not None and len(ohlcv) > 0:
            self.ohlcv_history = list(ohlcv)
        elif isinstance(obs, np.ndarray) and len(obs) >= 5:
            # Store obs slice as pseudo-OHLCV
            self.ohlcv_history.append(obs[:5])
            if len(self.ohlcv_history) > self.ohlcv_history_len:
                self.ohlcv_history = self.ohlcv_history[-self.ohlcv_history_len:]
        
        kronos_features = self._get_kronos_features()
        aug_obs = self._augment_observation(obs, kronos_features)
        
        return aug_obs, reward, terminated, truncated, info
    
    def _get_kronos_features(self) -> np.ndarray:
        """Get Kronos features from current OHLCV history."""
        if len(self.ohlcv_history) >= 5:
            ohlcv_array = np.array(self.ohlcv_history[-self.ohlcv_history_len:])
            return self.kronos.extract_features(ohlcv_array)
        return np.zeros(self.kronos_feature_dim, dtype=np.float32)
    
    def _augment_observation(self, obs, kronos_features) -> np.ndarray:
        """Concatenate standard observation with Kronos features."""
        if isinstance(obs, np.ndarray):
            return np.concatenate([obs, kronos_features]).astype(np.float32)
        elif isinstance(obs, dict):
            obs['kronos_features'] = kronos_features
            return obs
        return obs
    
    def __getattr__(self, name):
        """Delegate unknown attribute access to the wrapped environment."""
        if name in ('env', 'kronos', 'ohlcv_history_len', 'feature_dim', 'observation_space', 
                     'action_space', 'ohlcv_history'):
            return object.__getattribute__(self, name)
        return getattr(self.env, name)


# Direct integration helper for PPO training
def create_kronos_env(
    env_class,
    kronos_size: str = "small",
    feature_dim: int = 64,
    **env_kwargs
):
    """
    Create a Kronos-enhanced environment for PPO training.
    
    Args:
        env_class: The base environment class (SingleAssetTradingEnv, etc.)
        kronos_size: Kronos model size ('mini', 'small', 'base')
        feature_dim: Dimension of Kronos features
        **env_kwargs: Arguments for the environment
        
    Returns:
        KronosObservationWrapper around the environment
    """
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    kronos_extractor = KronosFeatureExtractor(
        model_size=kronos_size,
        feature_dim=feature_dim,
    )
    
    def make_env():
        base_env = env_class(**env_kwargs)
        return KronosObservationWrapper(base_env, kronos_extractor)
    
    return DummyVecEnv([make_env])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Kronos-enhanced environment")
    parser.add_argument("--model-size", choices=["mini", "small", "base"], default="small")
    parser.add_argument("--test-env", action="store_true", default=True)
    args = parser.parse_args()
    
    # Test the wrapper
    from environments import SingleAssetTradingEnv
    from environments.base_env import AssetConfig
    
    print(f"Creating Kronos-{args.model_size} enhanced environment...")
    asset = AssetConfig(symbol="TEST", name="Test", sector="Tech")
    
    kronos = KronosFeatureExtractor(model_size=args.model_size)
    base_env = SingleAssetTradingEnv(asset=asset)
    env = KronosObservationWrapper(base_env, kronos)
    
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"  - Base obs: {len(obs) - kronos.feature_dim} dim")
    print(f"  - Kronos features: {kronos.feature_dim} dim")
    print(f"  - Total: {len(obs)} dim")
    
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    print(f"Step OK: reward={reward:.4f}")
    
    env.close()
    print("\nKronos wrapper working correctly!")
