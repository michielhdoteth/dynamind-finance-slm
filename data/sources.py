"""
Market Data Sources for RL Financial Markets Gym

Professional data sources providing real market data from multiple providers.
Supports Yahoo Finance, Alpha Vantage, and custom CSV files.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


class BaseDataSource(ABC):
    """Abstract base class for market data sources."""

    @abstractmethod
    def fetch_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        frequency: str = "1d",
        **kwargs,
    ) -> pd.DataFrame:
        """Fetch market data for given symbols and date range."""
        pass

    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        pass

    @abstractmethod
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get detailed information about a symbol."""
        pass


class YahooFinanceSource(BaseDataSource):
    """Yahoo Finance data source using yfinance library."""

    def __init__(self):
        """Initialize Yahoo Finance source."""
        try:
            import yfinance as yf

            self.yf = yf
            logger.info("Yahoo Finance source initialized")
        except ImportError:
            raise ImportError(
                "yfinance package required. Install with: pip install yfinance"
            )

    def fetch_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        frequency: str = "1d",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance.

        Args:
            symbols: List of ticker symbols
            start_date: Start date
            end_date: End date
            frequency: Data frequency ('1m', '5m', '15m', '30m', '1h', '1d', '5d', '1wk', '1mo')
            **kwargs: Additional parameters

        Returns:
            MultiIndex DataFrame with OHLCV data
        """
        # Map frequency to yfinance intervals
        interval_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "1d": "1d",
            "5d": "5d",
            "1w": "1wk",
            "1wk": "1wk",
            "1mo": "1mo",
        }

        interval = interval_map.get(frequency, "1d")

        # Download data
        try:
            data = self.yf.download(
                tickers=symbols,
                start=start_date,
                end=end_date,
                interval=interval,
                group_by="ticker",
                auto_adjust=True,
                prepost=False,
                threads=True,
                **kwargs,
            )

            # Standardize column structure
            if len(symbols) == 1:
                # Single symbol - add symbol level to columns
                data.columns = pd.MultiIndex.from_product([[symbols[0]], data.columns])

            return data

        except Exception as e:
            logger.error(f"Failed to fetch data from Yahoo Finance: {e}")
            raise

    def get_available_symbols(self) -> List[str]:
        """Get popular symbols for Yahoo Finance."""
        # Return a curated list of popular symbols
        return [
            # Tech stocks
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "META",
            "NVDA",
            "TSLA",
            # Finance
            "JPM",
            "BAC",
            "WFC",
            "GS",
            "MS",
            # Healthcare
            "JNJ",
            "PFE",
            "UNH",
            "ABT",
            # Consumer
            "WMT",
            "PG",
            "KO",
            "PEP",
            # Energy
            "XOM",
            "CVX",
            "COP",
            # Indices
            "^GSPC",
            "^DJI",
            "^IXIC",
            "^VIX",
        ]

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get detailed information about a symbol."""
        try:
            ticker = self.yf.Ticker(symbol)
            info = ticker.info

            return {
                "symbol": symbol,
                "shortName": info.get("shortName", ""),
                "longName": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "marketCap": info.get("marketCap", 0),
                "currency": info.get("currency", "USD"),
                "exchange": info.get("exchange", ""),
                "country": info.get("country", ""),
                "timezone": info.get("timeZoneShortName", ""),
                "dividendYield": info.get("dividendYield", 0),
                "beta": info.get("beta", None),
            }
        except Exception as e:
            logger.warning(f"Failed to get info for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}


class AlphaVantageSource(BaseDataSource):
    """Alpha Vantage data source using their API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Alpha Vantage source.

        Args:
            api_key: Alpha Vantage API key (can be set via ALPHA_VANTAGE_API_KEY env var)
        """
        import os

        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Alpha Vantage API key required. Set ALPHA_VANTAGE_API_KEY environment variable or pass api_key parameter"
            )

        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_delay = 12  # Alpha Vantage free tier: 5 calls per minute
        self.last_call_time = 0

        logger.info("Alpha Vantage source initialized")

    def fetch_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        frequency: str = "1d",
        **kwargs,
    ) -> pd.DataFrame:
        """Fetch data from Alpha Vantage API."""
        all_data = {}

        for symbol in symbols:
            try:
                # Rate limiting
                self._rate_limit()

                symbol_data = self._fetch_symbol_data(symbol, frequency)
                if symbol_data is not None:
                    # Filter by date range
                    symbol_data = symbol_data[
                        (symbol_data.index >= start_date)
                        & (symbol_data.index <= end_date)
                    ]
                    all_data[symbol] = symbol_data

            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                continue

        if not all_data:
            raise ValueError("No data successfully fetched")

        # Combine into MultiIndex DataFrame
        combined_data = pd.concat(all_data, axis=1, keys=list(all_data.keys()))
        return combined_data

    def _fetch_symbol_data(self, symbol: str, frequency: str) -> Optional[pd.DataFrame]:
        """Fetch data for a single symbol."""
        # Determine function based on frequency
        if frequency in ["1d", "5d", "1w", "1mo"]:
            function = "TIME_SERIES_DAILY_ADJUSTED"
        elif frequency in ["1m", "5m", "15m", "30m", "1h"]:
            function = "TIME_SERIES_INTRADAY"
            # For intraday, we need to specify interval
            interval_map = {
                "1m": "1min",
                "5m": "5min",
                "15m": "15min",
                "30m": "30min",
                "1h": "60min",
            }
            interval = interval_map.get(frequency, "60min")
        else:
            function = "TIME_SERIES_DAILY_ADJUSTED"

        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": "full",
        }

        if function == "TIME_SERIES_INTRADAY":
            params["interval"] = interval

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Check for API errors
            if "Error Message" in data:
                raise ValueError(f"API Error: {data['Error Message']}")

            if "Note" in data:
                raise ValueError(f"API Rate Limit: {data['Note']}")

            # Parse response based on function
            if function == "TIME_SERIES_DAILY_ADJUSTED":
                return self._parse_daily_data(data)
            else:
                return self._parse_intraday_data(data)

        except Exception as e:
            logger.error(f"Failed to fetch {symbol} data: {e}")
            return None

    def _parse_daily_data(self, data: Dict) -> pd.DataFrame:
        """Parse daily adjusted time series data."""
        time_series = data.get("Time Series (Daily)", {})
        if not time_series:
            return None

        df_data = []
        for date_str, values in time_series.items():
            df_data.append(
                {
                    "Open": float(values["1. open"]),
                    "High": float(values["2. high"]),
                    "Low": float(values["3. low"]),
                    "Close": float(values["4. close"]),
                    "Adj Close": float(values["5. adjusted close"]),
                    "Volume": int(values["6. volume"]),
                }
            )

        df = pd.DataFrame(df_data)
        df.index = pd.to_datetime(list(time_series.keys()))
        df = df.sort_index()
        return df

    def _parse_intraday_data(self, data: Dict) -> pd.DataFrame:
        """Parse intraday time series data."""
        time_series_key = next((k for k in data.keys() if "Time Series" in k), None)
        if not time_series_key:
            return None

        time_series = data[time_series_key]
        if not time_series:
            return None

        df_data = []
        for date_str, values in time_series.items():
            df_data.append(
                {
                    "Open": float(values["1. open"]),
                    "High": float(values["2. high"]),
                    "Low": float(values["3. low"]),
                    "Close": float(values["4. close"]),
                    "Volume": int(values["5. volume"]),
                }
            )

        df = pd.DataFrame(df_data)
        df.index = pd.to_datetime(list(time_series.keys()))
        df = df.sort_index()
        return df

    def _rate_limit(self):
        """Implement rate limiting for API calls."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time

        if time_since_last_call < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_call
            time.sleep(sleep_time)

        self.last_call_time = time.time()

    def get_available_symbols(self) -> List[str]:
        """Get available symbols (same as Yahoo Finance for compatibility)."""
        # Alpha Vantage supports most stock symbols
        return YahooFinanceSource().get_available_symbols()

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get symbol information from Alpha Vantage."""
        try:
            self._rate_limit()

            params = {"function": "OVERVIEW", "symbol": symbol, "apikey": self.api_key}

            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "Symbol" not in data:
                raise ValueError("Symbol not found")

            return {
                "symbol": symbol,
                "shortName": data.get("Name", ""),
                "sector": data.get("Sector", ""),
                "industry": data.get("Industry", ""),
                "marketCap": int(data.get("MarketCapitalization", 0)),
                "currency": data.get("Currency", "USD"),
                "country": data.get("Country", ""),
                "dividendYield": float(data.get("DividendYield", 0)) / 100,
                "beta": float(data.get("Beta", 0)),
                "pe_ratio": float(data.get("PERatio", 0)),
                "eps": float(data.get("EPS", 0)),
            }
        except Exception as e:
            logger.warning(f"Failed to get info for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}


class CSVSource(BaseDataSource):
    """CSV file data source for custom market data."""

    def __init__(self, data_dir: str = "./data/csv"):
        """
        Initialize CSV source.

        Args:
            data_dir: Directory containing CSV files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Expected CSV columns
        self.required_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        self.optional_columns = ["Adj Close"]

        logger.info(f"CSV source initialized with data directory: {self.data_dir}")

    def fetch_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        frequency: str = "1d",
        **kwargs,
    ) -> pd.DataFrame:
        """Load data from CSV files."""
        all_data = {}

        for symbol in symbols:
            try:
                csv_file = self.data_dir / f"{symbol}.csv"
                if not csv_file.exists():
                    logger.warning(f"CSV file not found for {symbol}: {csv_file}")
                    continue

                # Load CSV data
                df = pd.read_csv(csv_file)

                # Standardize columns
                df = self._standardize_columns(df)

                # Convert date column
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"])
                    df.set_index("Date", inplace=True)

                # Filter by date range
                df = df[(df.index >= start_date) & (df.index <= end_date)]

                all_data[symbol] = df

            except Exception as e:
                logger.error(f"Failed to load CSV data for {symbol}: {e}")
                continue

        if not all_data:
            raise ValueError("No data successfully loaded from CSV files")

        # Combine into MultiIndex DataFrame
        combined_data = pd.concat(all_data, axis=1, keys=list(all_data.keys()))
        return combined_data

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize CSV column names."""
        # Common column name variations
        column_mapping = {
            "date": "Date",
            "DATE": "Date",
            "open": "Open",
            "OPEN": "Open",
            "high": "High",
            "HIGH": "High",
            "low": "Low",
            "LOW": "Low",
            "close": "Close",
            "CLOSE": "Close",
            "adj close": "Adj Close",
            "ADJ_CLOSE": "Adj Close",
            "volume": "Volume",
            "VOLUME": "Volume",
        }

        df = df.rename(columns=column_mapping)

        # Check required columns
        missing_columns = [
            col for col in self.required_columns if col not in df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        return df

    def get_available_symbols(self) -> List[str]:
        """Get list of symbols from available CSV files."""
        symbols = []

        for csv_file in self.data_dir.glob("*.csv"):
            symbol = csv_file.stem
            symbols.append(symbol)

        return sorted(symbols)

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get basic symbol info from CSV file."""
        try:
            csv_file = self.data_dir / f"{symbol}.csv"
            if not csv_file.exists():
                return {"symbol": symbol, "error": "CSV file not found"}

            # Load first few rows to get basic info
            df = pd.read_csv(csv_file, nrows=5)

            return {
                "symbol": symbol,
                "source": "CSV",
                "file_path": str(csv_file),
                "columns": list(df.columns),
                "data_points": len(pd.read_csv(csv_file)),
            }
        except Exception as e:
            logger.warning(f"Failed to get info for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}
