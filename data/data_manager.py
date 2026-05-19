"""
Professional Data Manager for RL Financial Markets Gym

Central orchestration of multiple market data sources with validation,
cleaning, and caching for reinforcement learning research.
"""

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from .preprocessors import DataPreprocessor
from .sources import AlphaVantageSource, BaseDataSource, CSVSource, YahooFinanceSource
from .validators import DataValidator

logger = logging.getLogger(__name__)


class DataManager:
    """
    Professional data management system for financial market data.

    Provides unified access to multiple data sources with validation,
    cleaning, and preprocessing capabilities specifically designed for RL.
    """

    def __init__(
        self,
        cache_dir: str = "./data_cache",
        enable_validation: bool = True,
        enable_preprocessing: bool = True,
        cache_ttl_hours: int = 24,
    ):
        """
        Initialize the Data Manager.

        Args:
            cache_dir: Directory for data caching
            enable_validation: Whether to validate data quality
            enable_preprocessing: Whether to preprocess data
            cache_ttl_hours: Cache time-to-live in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.enable_validation = enable_validation
        self.enable_preprocessing = enable_preprocessing
        self.cache_ttl = timedelta(hours=cache_ttl_hours)

        # Initialize data sources
        self.sources = {"yahoo": YahooFinanceSource(), "csv": CSVSource()}

        # Alpha Vantage source (optional, requires API key)
        try:
            self.sources["alphavantage"] = AlphaVantageSource()
        except ValueError:
            logger.info("Alpha Vantage source not available (no API key)")

        # Initialize validator and preprocessor
        self.validator = DataValidator()
        self.preprocessor = DataPreprocessor()

        # Cache management
        self._cache_index = self._load_cache_index()

        logger.info(f"DataManager initialized with cache directory: {self.cache_dir}")

    def get_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        source: str = "yahoo",
        frequency: str = "1d",
        features: Optional[List[str]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Get market data with validation and preprocessing.

        Args:
            symbols: Symbol(s) to fetch data for
            start_date: Start date for data
            end_date: End date for data
            source: Data source to use ('yahoo', 'alphavantage', 'csv')
            frequency: Data frequency ('1m', '5m', '1h', '1d', etc.)
            features: List of features to include
            **kwargs: Additional source-specific parameters

        Returns:
            Validated and preprocessed market data DataFrame
        """
        # Normalize inputs
        if isinstance(symbols, str):
            symbols = [symbols]

        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        # Check cache first
        cache_key = self._generate_cache_key(
            symbols, start_date, end_date, source, frequency
        )
        cached_data = self._load_from_cache(cache_key)

        if cached_data is not None:
            logger.info(f"Loaded {len(symbols)} symbols from cache")
            return cached_data

        # Fetch from source
        logger.info(f"Fetching {len(symbols)} symbols from {source}")

        if source not in self.sources:
            raise ValueError(f"Unknown data source: {source}")

        data_source = self.sources[source]
        raw_data = data_source.fetch_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            **kwargs,
        )

        # Validate data
        if self.enable_validation:
            raw_data = self._validate_data(raw_data, symbols)

        # Preprocess data
        if self.enable_preprocessing:
            raw_data = self._preprocess_data(raw_data, features)

        # Cache the processed data
        self._save_to_cache(cache_key, raw_data)

        return raw_data

    def get_available_symbols(self, source: str = "yahoo") -> List[str]:
        """Get list of available symbols from a data source."""
        if source not in self.sources:
            raise ValueError(f"Unknown data source: {source}")

        return self.sources[source].get_available_symbols()

    def get_market_calendar(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        market: str = "NYSE",
    ) -> pd.DatetimeIndex:
        """
        Get market calendar (trading days) for specified period.

        Args:
            start_date: Start date
            end_date: End date
            market: Market calendar to use

        Returns:
            DatetimeIndex of trading days
        """
        # Implement market calendar logic
        # For now, return pandas business days
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        return pd.bdate_range(start=start_date, end=end_date)

    def _validate_data(self, data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """Validate data quality and fix common issues."""
        logger.info("Validating data quality")

        validated_data = data.copy()

        # Basic validation checks
        for symbol in symbols:
            if symbol in validated_data.columns.get_level_values(0):
                symbol_data = validated_data[symbol]

                # Check for missing values
                missing_pct = symbol_data.isnull().sum() / len(symbol_data)
                if missing_pct.max() > 0.1:  # More than 10% missing
                    logger.warning(
                        f"Symbol {symbol} has high missing data: {missing_pct.max():.2%}"
                    )

                # Check for outliers (basic price validation)
                if "Close" in symbol_data.columns:
                    close_prices = symbol_data["Close"].dropna()
                    if len(close_prices) > 1:
                        price_change = close_prices.pct_change().abs()
                        extreme_changes = price_change > 0.5  # >50% daily change
                        if extreme_changes.any():
                            logger.warning(f"Symbol {symbol} has extreme price changes")

                # Forward fill missing values (common in financial data)
                symbol_data = symbol_data.ffill().bfill()
                validated_data[symbol] = symbol_data

        return validated_data

    def _preprocess_data(
        self, data: pd.DataFrame, features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Preprocess data for RL training."""
        logger.info("Preprocessing data for RL")

        if features is None:
            features = [
                "returns",
                "volatility",
                "rsi",
                "macd",
                "bollinger_high",
                "bollinger_low",
                "volume_ratio",
                "price_momentum",
            ]

        return self.preprocessor.process(data, features)

    def _generate_cache_key(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        source: str,
        frequency: str,
    ) -> str:
        """Generate cache key for data request."""
        symbols_str = "_".join(sorted(symbols))
        return f"{source}_{symbols_str}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{frequency}"

    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache if valid."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if not cache_file.exists():
            return None

        # Check cache age
        cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if cache_age > self.cache_ttl:
            cache_file.unlink()  # Remove expired cache
            return None

        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache file {cache_key}: {e}")
            cache_file.unlink()  # Remove corrupted cache
            return None

    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """Save data to cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache file {cache_key}: {e}")

    def _load_cache_index(self) -> Dict[str, Any]:
        """Load cache index for management."""
        index_file = self.cache_dir / "cache_index.pkl"

        if index_file.exists():
            try:
                with open(index_file, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass

        return {}

    def clear_cache(self, older_than_hours: Optional[int] = None):
        """Clear cached data."""
        if older_than_hours is None:
            # Clear all cache
            for cache_file in self.cache_dir.glob("*.pkl"):
                if cache_file.name != "cache_index.pkl":
                    cache_file.unlink()
            logger.info("Cleared all cached data")
        else:
            # Clear old cache
            cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
            cleared_count = 0

            for cache_file in self.cache_dir.glob("*.pkl"):
                if cache_file.name != "cache_index.pkl":
                    file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if file_time < cutoff_time:
                        cache_file.unlink()
                        cleared_count += 1

            logger.info(f"Cleared {cleared_count} expired cache files")

    def get_data_info(
        self, symbols: List[str], source: str = "yahoo"
    ) -> Dict[str, Any]:
        """Get metadata about available data for symbols."""
        info = {}

        for symbol in symbols:
            try:
                symbol_info = self.sources[source].get_symbol_info(symbol)
                info[symbol] = symbol_info
            except Exception as e:
                logger.warning(f"Failed to get info for {symbol}: {e}")
                info[symbol] = {"error": str(e)}

        return info
