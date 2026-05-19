"""
Data Validation for RL Financial Markets Gym

Professional data validation system for market data quality assurance.
Detects and handles common data issues to ensure reliable RL training.
"""

from datetime import datetime
import warnings
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Professional data validator for financial market data.

    Provides comprehensive validation checks for market data quality,
    consistency, and reliability for RL training.
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize the data validator.

        Args:
            strict_mode: Whether to be strict about data quality issues
        """
        self.strict_mode = strict_mode
        self.validation_results = {}

        # Validation thresholds
        self.thresholds = {
            "max_missing_pct": 0.1,  # 10% missing data tolerance
            "max_price_jump": 0.5,  # 50% price jump tolerance
            "min_volume": 0,  # Minimum volume
            "max_volume_change": 10.0,  # 10x volume spike tolerance
            "min_trading_days": 10,  # Minimum trading days
            "price_consistency": 0.01,  # 1% price consistency tolerance
        }

    def validate_dataset(
        self, data: pd.DataFrame, symbols: List[str]
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of market data dataset.

        Args:
            data: MultiIndex DataFrame with market data
            symbols: List of symbols to validate

        Returns:
            Dictionary with validation results and recommendations
        """
        validation_results = {
            "timestamp": datetime.now(),
            "symbols_validated": symbols,
            "total_records": len(data),
            "issues": [],
            "warnings": [],
            "symbol_details": {},
            "passed": True,
        }

        for symbol in symbols:
            if symbol in data.columns.get_level_values(0):
                symbol_data = data[symbol]
                symbol_validation = self.validate_symbol(symbol_data, symbol)
                validation_results["symbol_details"][symbol] = symbol_validation

                # Collect issues and warnings
                if symbol_validation["issues"]:
                    validation_results["issues"].extend(
                        [f"{symbol}: {issue}" for issue in symbol_validation["issues"]]
                    )
                    validation_results["passed"] = False

                if symbol_validation["warnings"]:
                    validation_results["warnings"].extend(
                        [
                            f"{symbol}: {warning}"
                            for warning in symbol_validation["warnings"]
                        ]
                    )

        self.validation_results = validation_results
        return validation_results

    def validate_symbol(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Validate data for a single symbol.

        Args:
            data: DataFrame with OHLCV data for single symbol
            symbol: Symbol name

        Returns:
            Dictionary with validation results for the symbol
        """
        result = {
            "symbol": symbol,
            "issues": [],
            "warnings": [],
            "metrics": {},
            "passed": True,
        }

        # Basic structure validation
        structure_issues = self._validate_structure(data)
        result["issues"].extend(structure_issues)

        if structure_issues:
            result["passed"] = False
            return result

        # Data quality checks
        quality_checks = self._check_data_quality(data)
        result["issues"].extend(quality_checks["issues"])
        result["warnings"].extend(quality_checks["warnings"])
        result["metrics"].update(quality_checks["metrics"])

        # Price consistency checks
        price_checks = self._check_price_consistency(data)
        result["issues"].extend(price_checks["issues"])
        result["warnings"].extend(price_checks["warnings"])
        result["metrics"].update(price_checks["metrics"])

        # Volume checks
        volume_checks = self._check_volume(data)
        result["issues"].extend(volume_checks["issues"])
        result["warnings"].extend(volume_checks["warnings"])
        result["metrics"].update(volume_checks["metrics"])

        # Temporal checks
        temporal_checks = self._check_temporal_consistency(data)
        result["issues"].extend(temporal_checks["issues"])
        result["warnings"].extend(temporal_checks["warnings"])
        result["metrics"].update(temporal_checks["metrics"])

        if result["issues"]:
            result["passed"] = False

        return result

    def _validate_structure(self, data: pd.DataFrame) -> List[str]:
        """Validate basic DataFrame structure."""
        issues = []

        # Check required columns
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")

        # Check data types
        for col in ["Open", "High", "Low", "Close"]:
            if col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    issues.append(f"Column {col} should be numeric")

        if "Volume" in data.columns:
            if not pd.api.types.is_numeric_dtype(data["Volume"]):
                issues.append("Volume column should be numeric")

        # Check minimum records
        if len(data) < self.thresholds["min_trading_days"]:
            issues.append(
                f"Insufficient data: {len(data)} records (minimum: {self.thresholds['min_trading_days']})"
            )

        return issues

    def _check_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data quality metrics."""
        issues = []
        warnings = []
        metrics = {}

        # Missing data analysis
        missing_data = data.isnull().sum()
        total_records = len(data)

        for column, missing_count in missing_data.items():
            missing_pct = missing_count / total_records
            metrics[f"{column}_missing_pct"] = missing_pct

            if missing_pct > self.thresholds["max_missing_pct"]:
                issues.append(f"High missing data in {column}: {missing_pct:.2%}")
            elif missing_pct > 0:
                warnings.append(f"Missing data in {column}: {missing_pct:.2%}")

        # Duplicate dates
        if isinstance(data.index, pd.DatetimeIndex):
            duplicate_dates = data.index.duplicated().sum()
            if duplicate_dates > 0:
                issues.append(f"Found {duplicate_dates} duplicate dates")
                metrics["duplicate_dates"] = duplicate_dates

        # Zero or negative values where inappropriate
        for col in ["Open", "High", "Low", "Close"]:
            if col in data.columns:
                zero_count = (data[col] <= 0).sum()
                if zero_count > 0:
                    warnings.append(f"Found {zero_count} non-positive values in {col}")
                    metrics[f"{col}_non_positive"] = zero_count

        # Negative volume
        if "Volume" in data.columns:
            negative_volume = (data["Volume"] < 0).sum()
            if negative_volume > 0:
                issues.append(f"Found {negative_volume} negative volume values")
                metrics["negative_volume"] = negative_volume

        return {"issues": issues, "warnings": warnings, "metrics": metrics}

    def _check_price_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check price consistency and relationships."""
        issues = []
        warnings = []
        metrics = {}

        # High >= Low consistency
        if "High" in data.columns and "Low" in data.columns:
            inconsistent_prices = (data["High"] < data["Low"]).sum()
            if inconsistent_prices > 0:
                issues.append(f"Found {inconsistent_prices} records where High < Low")
                metrics["high_low_inconsistency"] = inconsistent_prices

        # Close within High-Low range
        if "Close" in data.columns and "High" in data.columns and "Low" in data.columns:
            close_outside_range = (
                (data["Close"] > data["High"]) | (data["Close"] < data["Low"])
            ).sum()
            if close_outside_range > 0:
                warnings.append(
                    f"Found {close_outside_range} records where Close is outside High-Low range"
                )
                metrics["close_outside_range"] = close_outside_range

        # Open within reasonable range of previous close
        if all(col in data.columns for col in ["Open", "Close"]):
            if len(data) > 1:
                prev_close = data["Close"].shift(1)
                price_change_pct = abs((data["Open"] - prev_close) / prev_close)

                extreme_gaps = (
                    price_change_pct > self.thresholds["max_price_jump"]
                ).sum()
                if extreme_gaps > 0:
                    warnings.append(f"Found {extreme_gaps} extreme price gaps (>50%)")
                    metrics["extreme_price_gaps"] = extreme_gaps

                metrics["max_price_gap"] = price_change_pct.max()
                metrics["avg_price_gap"] = price_change_pct.mean()

        # Price continuity (no impossible jumps)
        if "Close" in data.columns and len(data) > 1:
            price_changes = data["Close"].pct_change().abs()
            extreme_changes = (price_changes > self.thresholds["max_price_jump"]).sum()
            if extreme_changes > 0:
                warnings.append(f"Found {extreme_changes} extreme price changes")
                metrics["extreme_price_changes"] = extreme_changes

        return {"issues": issues, "warnings": warnings, "metrics": metrics}

    def _check_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check volume data quality."""
        issues = []
        warnings = []
        metrics = {}

        if "Volume" not in data.columns:
            return {"issues": issues, "warnings": warnings, "metrics": metrics}

        # Zero volume days
        zero_volume_days = (data["Volume"] == 0).sum()
        if zero_volume_days > 0:
            warnings.append(f"Found {zero_volume_days} zero volume days")
            metrics["zero_volume_days"] = zero_volume_days

        # Volume spikes
        if len(data) > 1:
            volume_change = data["Volume"].pct_change()
            extreme_volume_spikes = (
                volume_change.abs() > self.thresholds["max_volume_change"]
            ).sum()
            if extreme_volume_spikes > 0:
                warnings.append(f"Found {extreme_volume_spikes} extreme volume spikes")
                metrics["extreme_volume_spikes"] = extreme_volume_spikes

            metrics["max_volume_change"] = volume_change.abs().max()
            metrics["avg_volume"] = data["Volume"].mean()
            metrics["volume_std"] = data["Volume"].std()

        # Volume-price relationship
        if "Close" in data.columns and len(data) > 10:
            # High volume should correlate with significant price moves
            price_volatility = data["Close"].pct_change().abs().rolling(5).mean()
            volume_ma = data["Volume"].rolling(5).mean()

            correlation = price_volatility.corr(volume_ma)
            if not np.isnan(correlation):
                metrics["volume_price_correlation"] = correlation

                if correlation < 0.1:
                    warnings.append(
                        "Low volume-price correlation may indicate data quality issues"
                    )

        return {"issues": issues, "warnings": warnings, "metrics": metrics}

    def _check_temporal_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check temporal consistency of data."""
        issues = []
        warnings = []
        metrics = {}

        if not isinstance(data.index, pd.DatetimeIndex):
            warnings.append("Data index is not DatetimeIndex")
            return {"issues": issues, "warnings": warnings, "metrics": metrics}

        # Check for gaps in trading days
        if len(data) > 1:
            date_gaps = data.index.to_series().diff().dt.days
            # Weekends are normal gaps (>2 days might indicate missing data)
            large_gaps = (date_gaps > 7).sum()
            if large_gaps > 0:
                warnings.append(f"Found {large_gaps} gaps larger than 7 days")
                metrics["large_date_gaps"] = large_gaps

            # Check weekend data (shouldn't exist for daily data)
            weekend_data = data.index.dayofweek.isin([5, 6]).sum()
            if weekend_data > 0:
                warnings.append(f"Found {weekend_data} weekend data points")
                metrics["weekend_data"] = weekend_data

        # Check data frequency consistency
        if len(data) > 1:
            time_diffs = data.index.to_series().diff().dropna()
            unique_diffs = time_diffs.nunique()
            if unique_diffs > 3:  # Allow for holidays/weekends
                warnings.append(
                    f"Inconsistent data frequency: {unique_diffs} unique time differences"
                )
                metrics["frequency_inconsistency"] = unique_diffs

        # Time range
        if len(data) > 0:
            metrics["date_range_start"] = data.index.min()
            metrics["date_range_end"] = data.index.max()
            metrics["total_days"] = (data.index.max() - data.index.min()).days
            metrics["trading_days"] = len(data)

        return {"issues": issues, "warnings": warnings, "metrics": metrics}

    def clean_data(self, data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """
        Clean data based on validation results.

        Args:
            data: Raw market data
            symbols: List of symbols

        Returns:
            Cleaned market data
        """
        cleaned_data = data.copy()

        for symbol in symbols:
            if symbol in cleaned_data.columns.get_level_values(0):
                symbol_data = cleaned_data[symbol]

                # Forward fill missing values
                symbol_data = symbol_data.ffill().bfill()

                # Remove obvious outliers (price jumps > 90%)
                if "Close" in symbol_data.columns and len(symbol_data) > 1:
                    price_changes = symbol_data["Close"].pct_change().abs()
                    outliers = price_changes > 0.9
                    if outliers.any():
                        logger.warning(
                            f"Removing {outliers.sum()} extreme price outliers for {symbol}"
                        )
                        # Use forward fill for outliers
                        symbol_data.loc[outliers] = symbol_data.shift(1).loc[outliers]

                # Fix High/Low consistency
                if "High" in symbol_data.columns and "Low" in symbol_data.columns:
                    # Ensure High >= Low
                    max_prices = symbol_data[["High", "Low"]].max(axis=1)
                    min_prices = symbol_data[["High", "Low"]].min(axis=1)
                    symbol_data["High"] = max_prices
                    symbol_data["Low"] = min_prices

                # Ensure Close is within High-Low range
                if all(col in symbol_data.columns for col in ["Close", "High", "Low"]):
                    symbol_data["Close"] = symbol_data["Close"].clip(
                        lower=symbol_data["Low"], upper=symbol_data["High"]
                    )

                # Update the DataFrame
                cleaned_data[symbol] = symbol_data

        return cleaned_data

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of the latest validation results."""
        if not self.validation_results:
            return {"message": "No validation results available"}

        return {
            "validation_timestamp": self.validation_results["timestamp"],
            "total_symbols": len(self.validation_results["symbols_validated"]),
            "total_issues": len(self.validation_results["issues"]),
            "total_warnings": len(self.validation_results["warnings"]),
            "overall_status": "PASSED"
            if self.validation_results["passed"]
            else "FAILED",
            "symbols_with_issues": [
                symbol
                for symbol, details in self.validation_results["symbol_details"].items()
                if not details["passed"]
            ],
        }
