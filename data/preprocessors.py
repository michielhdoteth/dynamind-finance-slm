"""
Data Preprocessing for RL Financial Markets Gym

Professional data preprocessing system for reinforcement learning.
Creates RL-appropriate features and normalization for financial market data.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Professional data preprocessor for RL financial environments.

    Creates features, normalizes data, and prepares market data for
    reinforcement learning training with proper scaling and feature engineering.
    """

    def __init__(
        self,
        feature_window: int = 20,
        normalize_features: bool = True,
        scaler_type: str = "standard",
        target_returns: bool = False,
    ):
        """
        Initialize the data preprocessor.

        Args:
            feature_window: Lookback window for feature calculation
            normalize_features: Whether to normalize features
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
            target_returns: Whether to target specific return distribution
        """
        self.feature_window = feature_window
        self.normalize_features = normalize_features
        self.target_returns = target_returns

        # Initialize scalers
        self.scalers = {}
        self.scaler_type = scaler_type

        # Feature configurations
        self.feature_configs = {
            "basic_returns": True,
            "log_returns": True,
            "volatility": True,
            "technical_indicators": True,
            "volume_features": True,
            "price_position": True,
            "momentum": True,
            "regime_features": True,
        }

        logger.info(
            f"DataPreprocessor initialized with window={feature_window}, scaler={scaler_type}"
        )

    def process(
        self, data: pd.DataFrame, requested_features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Process market data for RL training.

        Args:
            data: MultiIndex DataFrame with OHLCV data
            requested_features: List of specific features to compute

        Returns:
            Processed DataFrame with RL-appropriate features
        """
        if requested_features is None:
            requested_features = [
                "returns",
                "log_returns",
                "volatility",
                "rsi",
                "macd",
                "bollinger_position",
                "volume_ratio",
                "price_momentum",
                "high_low_ratio",
                "volume_price_trend",
            ]

        processed_data = data.copy()

        # Process each symbol
        symbols = processed_data.columns.get_level_values(0).unique()
        for symbol in symbols:
            if symbol in processed_data.columns:
                symbol_data = processed_data[symbol].copy()
                symbol_features = self._compute_features(
                    symbol_data, requested_features
                )

                # Merge features back
                for feature_name, feature_data in symbol_features.items():
                    processed_data[(symbol, feature_name)] = feature_data

        # Normalize features if requested
        if self.normalize_features:
            processed_data = self._normalize_features(processed_data, symbols)

        # Remove rows with NaN values (from feature computation)
        processed_data = processed_data.dropna()

        logger.info(f"Processed data shape: {processed_data.shape}")
        return processed_data

    def _compute_features(
        self, data: pd.DataFrame, requested_features: List[str]
    ) -> Dict[str, pd.Series]:
        """Compute features for a single symbol."""
        features = {}

        # Basic price returns
        if "returns" in requested_features and "Close" in data.columns:
            features["returns"] = data["Close"].pct_change()

        # Log returns
        if "log_returns" in requested_features and "Close" in data.columns:
            features["log_returns"] = np.log(data["Close"] / data["Close"].shift(1))

        # Volatility measures
        if "volatility" in requested_features and "returns" in features:
            # Rolling standard deviation of returns
            features["volatility"] = (
                features["returns"].rolling(self.feature_window).std()
            )
            # Parkinson volatility (using high-low range)
            if "High" in data.columns and "Low" in data.columns:
                hl_ratio = np.log(data["High"] / data["Low"])
                features["parkinson_volatility"] = np.sqrt(
                    0.361 * (hl_ratio**2).rolling(self.feature_window).sum()
                )

        # Technical indicators
        if any(
            indicator in requested_features
            for indicator in ["rsi", "macd", "bollinger_position"]
        ):
            technical_features = self._compute_technical_indicators(
                data, requested_features
            )
            features.update(technical_features)

        # Volume features
        if any(
            volume_feat in requested_features
            for volume_feat in ["volume_ratio", "volume_price_trend"]
        ):
            volume_features = self._compute_volume_features(data, requested_features)
            features.update(volume_features)

        # Price position features
        if any(
            pos_feat in requested_features
            for pos_feat in ["price_position", "high_low_ratio"]
        ):
            position_features = self._compute_position_features(
                data, requested_features
            )
            features.update(position_features)

        # Momentum features
        if "price_momentum" in requested_features:
            momentum_features = self._compute_momentum_features(
                data, requested_features
            )
            features.update(momentum_features)

        # Regime features
        if "regime_features" in requested_features:
            regime_features = self._compute_regime_features(data, requested_features)
            features.update(regime_features)

        return features

    def _compute_technical_indicators(
        self, data: pd.DataFrame, requested_features: List[str]
    ) -> Dict[str, pd.Series]:
        """Compute technical indicators."""
        features = {}

        if "Close" not in data.columns:
            return features

        close_prices = data["Close"]

        # RSI (Relative Strength Index)
        if "rsi" in requested_features:
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features["rsi"] = 100 - (100 / (1 + rs))

        # MACD (Moving Average Convergence Divergence)
        if "macd" in requested_features:
            ema_12 = close_prices.ewm(span=12).mean()
            ema_26 = close_prices.ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            features["macd"] = macd_line
            features["macd_signal"] = signal_line
            features["macd_histogram"] = macd_line - signal_line

        # Bollinger Bands
        if "bollinger_position" in requested_features:
            sma = close_prices.rolling(window=20).mean()
            std = close_prices.rolling(window=20).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            features["bollinger_position"] = (close_prices - lower_band) / (
                upper_band - lower_band
            )
            features["bollinger_width"] = (upper_band - lower_band) / sma

        # Moving averages
        if "sma_cross" in requested_features:
            sma_short = close_prices.rolling(window=10).mean()
            sma_long = close_prices.rolling(window=30).mean()
            features["sma_cross"] = (sma_short - sma_long) / sma_long

        return features

    def _compute_volume_features(
        self, data: pd.DataFrame, requested_features: List[str]
    ) -> Dict[str, pd.Series]:
        """Compute volume-based features."""
        features = {}

        if "Volume" not in data.columns:
            return features

        volume = data["Volume"]

        # Volume ratio (current volume to moving average)
        if "volume_ratio" in requested_features:
            volume_ma = volume.rolling(window=self.feature_window).mean()
            features["volume_ratio"] = volume / volume_ma

        # Volume-price trend
        if "volume_price_trend" in requested_features and "Close" in data.columns:
            price_change = data["Close"].pct_change()
            features["volume_price_trend"] = volume * price_change

        # On-balance volume
        if "obv" in requested_features and "Close" in data.columns:
            price_change = data["Close"].diff()
            obv = (volume * np.sign(price_change)).cumsum()
            features["obv"] = obv

        # Volume weighted average price (VWAP)
        if "vwap" in requested_features and all(
            col in data.columns for col in ["High", "Low", "Close"]
        ):
            typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
            vwap = (typical_price * volume).rolling(
                window=self.feature_window
            ).sum() / volume.rolling(window=self.feature_window).sum()
            features["vwap"] = vwap

        return features

    def _compute_position_features(
        self, data: pd.DataFrame, requested_features: List[str]
    ) -> Dict[str, pd.Series]:
        """Compute price position features."""
        features = {}

        if "Close" not in data.columns:
            return features

        close_prices = data["Close"]

        # High-Low ratio
        if "high_low_ratio" in requested_features:
            if "High" in data.columns and "Low" in data.columns:
                features["high_low_ratio"] = (close_prices - data["Low"]) / (
                    data["High"] - data["Low"]
                )

        # Position in rolling range
        if "price_position" in requested_features:
            rolling_max = close_prices.rolling(window=self.feature_window).max()
            rolling_min = close_prices.rolling(window=self.feature_window).min()
            features["price_position"] = (close_prices - rolling_min) / (
                rolling_max - rolling_min
            )

        # Distance from moving average
        if "ma_distance" in requested_features:
            ma = close_prices.rolling(window=self.feature_window).mean()
            features["ma_distance"] = (close_prices - ma) / ma

        return features

    def _compute_momentum_features(
        self, data: pd.DataFrame, requested_features: List[str]
    ) -> Dict[str, pd.Series]:
        """Compute momentum features."""
        features = {}

        if "Close" not in data.columns:
            return features

        close_prices = data["Close"]

        # Price momentum (different timeframes)
        if "price_momentum" in requested_features:
            for period in [1, 5, 10, 20]:
                features[f"momentum_{period}d"] = close_prices.pct_change(period)

        # Acceleration (change in momentum)
        if "acceleration" in requested_features:
            momentum_5 = close_prices.pct_change(5)
            momentum_10 = close_prices.pct_change(10)
            features["acceleration"] = momentum_5 - momentum_10

        # Rate of change
        if "rate_of_change" in requested_features:
            features["rate_of_change"] = (
                (close_prices - close_prices.shift(10)) / close_prices.shift(10)
            ) * 100

        return features

    def _compute_regime_features(
        self, data: pd.DataFrame, requested_features: List[str]
    ) -> Dict[str, pd.Series]:
        """Compute market regime features."""
        features = {}

        if "returns" not in data.columns and "Close" in data.columns:
            returns = data["Close"].pct_change()
        else:
            returns = data.get("returns", pd.Series(index=data.index))

        if returns.empty:
            return features

        # Trend strength (using rolling correlation with time)
        if "trend_strength" in requested_features:
            time_index = np.arange(len(returns))
            trend_correlation = returns.rolling(self.feature_window).apply(
                lambda x: np.corrcoef(x, time_index[-len(x) :])[0, 1]
                if len(x) > 1
                else 0
            )
            features["trend_strength"] = trend_correlation

        # Volatility regime
        if "volatility_regime" in requested_features:
            rolling_vol = returns.rolling(self.feature_window).std()
            vol_ma = rolling_vol.rolling(window=50).mean()
            features["volatility_regime"] = rolling_vol / vol_ma

        # Return regime (positive vs negative periods)
        if "return_regime" in requested_features:
            positive_returns = (returns > 0).rolling(self.feature_window).mean()
            features["return_regime"] = positive_returns

        return features

    def _normalize_features(
        self, data: pd.DataFrame, symbols: List[str]
    ) -> pd.DataFrame:
        """Normalize features for RL training."""
        normalized_data = data.copy()

        # Get feature columns (exclude OHLCV)
        ohlcv_columns = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]

        for symbol in symbols:
            if symbol in normalized_data.columns:
                symbol_columns = [
                    col
                    for col in normalized_data[symbol].columns
                    if col not in ohlcv_columns
                ]

                if symbol_columns:
                    symbol_features = normalized_data[symbol][symbol_columns]

                    # Create or get scaler for this symbol
                    scaler_key = f"{symbol}_features"
                    if scaler_key not in self.scalers:
                        self.scalers[scaler_key] = self._create_scaler()

                    # Fit scaler and transform
                    if len(symbol_features.dropna()) > 0:
                        # Fit on available data
                        clean_features = symbol_features.dropna()
                        if len(clean_features) > 0:
                            self.scalers[scaler_key].fit(clean_features)

                        # Transform all features
                        normalized_features = self.scalers[scaler_key].transform(
                            symbol_features.fillna(0)
                        )
                        normalized_data[symbol][symbol_columns] = normalized_features

        return normalized_data

    def _create_scaler(self):
        """Create appropriate scaler based on scaler_type."""
        if self.scaler_type == "standard":
            return StandardScaler()
        elif self.scaler_type == "minmax":
            return MinMaxScaler()
        elif self.scaler_type == "robust":
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")

    def inverse_transform_features(
        self, features: np.ndarray, symbol: str
    ) -> np.ndarray:
        """Inverse transform normalized features."""
        scaler_key = f"{symbol}_features"
        if scaler_key in self.scalers:
            return self.scalers[scaler_key].inverse_transform(features)
        return features

    def get_feature_stats(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Get statistics for features of a specific symbol."""
        if symbol not in data.columns:
            return {}

        symbol_data = data[symbol]
        stats = {}

        for column in symbol_data.columns:
            series = symbol_data[column].dropna()
            if len(series) > 0:
                stats[column] = {
                    "mean": series.mean(),
                    "std": series.std(),
                    "min": series.min(),
                    "max": series.max(),
                    "median": series.median(),
                    "skewness": series.skew(),
                    "kurtosis": series.kurtosis(),
                    "missing_pct": series.isnull().sum() / len(symbol_data[column]),
                }

        return stats

    def create_sequences(
        self,
        data: pd.DataFrame,
        sequence_length: int = 30,
        target_col: str = "Close",
        prediction_horizon: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for sequential RL models.

        Args:
            data: Processed market data
            sequence_length: Length of input sequences
            target_col: Column to predict
            prediction_horizon: Steps ahead to predict

        Returns:
            Tuple of (sequences, targets)
        """
        sequences = []
        targets = []

        # Process each symbol
        symbols = data.columns.get_level_values(0).unique()
        for symbol in symbols:
            if symbol in data.columns:
                symbol_data = data[symbol].dropna()

                if (
                    target_col in symbol_data.columns
                    and len(symbol_data) > sequence_length + prediction_horizon
                ):
                    # Create sequences
                    feature_cols = [
                        col for col in symbol_data.columns if col != target_col
                    ]
                    feature_data = symbol_data[feature_cols].values
                    target_data = symbol_data[target_col].values

                    for i in range(
                        len(feature_data) - sequence_length - prediction_horizon + 1
                    ):
                        seq = feature_data[i : i + sequence_length]
                        target = target_data[
                            i + sequence_length + prediction_horizon - 1
                        ]
                        sequences.append(seq)
                        targets.append(target)

        return np.array(sequences), np.array(targets)
