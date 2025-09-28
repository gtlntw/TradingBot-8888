"""
Data collection module for Bitcoin trading bot.
Handles data acquisition from multiple sources.
"""

import asyncio
import aiohttp
import ccxt
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from abc import ABC, abstractmethod

from trading_bot.utils.logger import LoggerMixin
from trading_bot.utils.decorators import retry, rate_limit, error_handler
from trading_bot.config.settings import Settings


class DataSource(ABC):
    """Abstract base class for data sources."""

    @abstractmethod
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data."""
        pass

    @abstractmethod
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch current ticker data."""
        pass


class BinanceDataSource(DataSource, LoggerMixin):
    """Binance data source using CCXT."""

    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        """Initialize Binance data source."""
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret_key,
            'sandbox': False,
            'enableRateLimit': True,
        })

    @retry(max_attempts=3, delay=1.0)
    @rate_limit(calls_per_second=10)
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch OHLCV data from Binance.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '1d')
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            self.logger.debug(f"Fetched {len(df)} candles for {symbol} from Binance")
            return df
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV from Binance: {e}")
            raise

    @retry(max_attempts=3, delay=1.0)
    @rate_limit(calls_per_second=10)
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current ticker data from Binance.

        Args:
            symbol: Trading symbol

        Returns:
            Ticker data dictionary
        """
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            self.logger.debug(f"Fetched ticker for {symbol} from Binance")
            return ticker
        except Exception as e:
            self.logger.error(f"Error fetching ticker from Binance: {e}")
            raise

    async def close(self):
        """Close the exchange connection."""
        await self.exchange.close()


class YFinanceDataSource(DataSource, LoggerMixin):
    """Yahoo Finance data source."""

    @retry(max_attempts=3, delay=1.0)
    @rate_limit(calls_per_second=5)
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance.

        Args:
            symbol: Trading symbol (e.g., 'BTC-USD')
            timeframe: Timeframe (e.g., '1h', '1d')
            limit: Number of periods to fetch

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert timeframe to period for yfinance
            period_map = {
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '30m': '30m',
                '1h': '1h',
                '1d': '1d'
            }

            period = period_map.get(timeframe, '1d')

            # Calculate the start date based on limit and timeframe
            if timeframe == '1d':
                start_date = datetime.now() - timedelta(days=limit)
            else:
                # For intraday data, use the maximum allowed period
                start_date = datetime.now() - timedelta(days=30)

            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                interval=period,
                auto_adjust=True,
                prepost=True
            )

            if data.empty:
                raise ValueError(f"No data returned for {symbol}")

            # Standardize column names
            df = data.copy()
            df.columns = df.columns.str.lower()
            df = df.rename(columns={'adj close': 'close'})

            # Ensure we have the required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    self.logger.warning(f"Missing column {col} in Yahoo Finance data")

            # Take only the requested number of rows
            df = df.tail(limit)

            self.logger.debug(f"Fetched {len(df)} candles for {symbol} from Yahoo Finance")
            return df

        except Exception as e:
            self.logger.error(f"Error fetching OHLCV from Yahoo Finance: {e}")
            raise

    @retry(max_attempts=3, delay=1.0)
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current ticker data from Yahoo Finance.

        Args:
            symbol: Trading symbol

        Returns:
            Ticker data dictionary
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Get current price from fast_info if available
            try:
                current_price = ticker.fast_info['lastPrice']
            except:
                current_price = info.get('currentPrice', info.get('regularMarketPrice'))

            ticker_data = {
                'symbol': symbol,
                'last': current_price,
                'bid': info.get('bid'),
                'ask': info.get('ask'),
                'high': info.get('dayHigh'),
                'low': info.get('dayLow'),
                'volume': info.get('volume'),
                'timestamp': datetime.now()
            }

            self.logger.debug(f"Fetched ticker for {symbol} from Yahoo Finance")
            return ticker_data

        except Exception as e:
            self.logger.error(f"Error fetching ticker from Yahoo Finance: {e}")
            raise


class CoinGeckoDataSource(DataSource, LoggerMixin):
    """CoinGecko data source."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize CoinGecko data source."""
        self.api_key = api_key
        self.base_url = "https://api.coingecko.com/api/v3"

    @retry(max_attempts=3, delay=2.0)
    @rate_limit(calls_per_second=1)  # CoinGecko has strict rate limits
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch OHLCV data from CoinGecko.

        Args:
            symbol: Coin ID (e.g., 'bitcoin')
            timeframe: Timeframe (only daily supported by free API)
            limit: Number of days to fetch

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # CoinGecko uses coin IDs, not symbols
            coin_id = self._symbol_to_coin_id(symbol)

            url = f"{self.base_url}/coins/{coin_id}/ohlc"
            params = {
                'vs_currency': 'usd',
                'days': min(limit, 365)  # Max 365 days for free API
            }

            if self.api_key:
                params['x_cg_demo_api_key'] = self.api_key

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Convert to DataFrame
                        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.set_index('timestamp', inplace=True)
                        df['volume'] = 0  # Volume not provided in OHLC endpoint

                        self.logger.debug(f"Fetched {len(df)} candles for {symbol} from CoinGecko")
                        return df
                    else:
                        raise Exception(f"CoinGecko API error: {response.status}")

        except Exception as e:
            self.logger.error(f"Error fetching OHLCV from CoinGecko: {e}")
            raise

    @retry(max_attempts=3, delay=2.0)
    @rate_limit(calls_per_second=1)
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current ticker data from CoinGecko.

        Args:
            symbol: Coin ID

        Returns:
            Ticker data dictionary
        """
        try:
            coin_id = self._symbol_to_coin_id(symbol)

            url = f"{self.base_url}/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_24hr_vol': 'true',
                'include_last_updated_at': 'true'
            }

            if self.api_key:
                params['x_cg_demo_api_key'] = self.api_key

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        coin_data = data[coin_id]

                        ticker_data = {
                            'symbol': symbol,
                            'last': coin_data['usd'],
                            'change_24h': coin_data.get('usd_24h_change', 0),
                            'volume_24h': coin_data.get('usd_24h_vol', 0),
                            'timestamp': datetime.fromtimestamp(coin_data.get('last_updated_at', 0))
                        }

                        self.logger.debug(f"Fetched ticker for {symbol} from CoinGecko")
                        return ticker_data
                    else:
                        raise Exception(f"CoinGecko API error: {response.status}")

        except Exception as e:
            self.logger.error(f"Error fetching ticker from CoinGecko: {e}")
            raise

    def _symbol_to_coin_id(self, symbol: str) -> str:
        """Convert symbol to CoinGecko coin ID."""
        symbol_map = {
            'BTC': 'bitcoin',
            'BTC-USD': 'bitcoin',
            'BTC/USD': 'bitcoin',
            'BTC-USDT': 'bitcoin',
            'BTC/USDT': 'bitcoin'
        }
        return symbol_map.get(symbol.upper(), 'bitcoin')


class DataCollector(LoggerMixin):
    """Main data collector that aggregates data from multiple sources."""

    def __init__(self, settings: Settings):
        """
        Initialize data collector.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.sources = self._initialize_sources()

    def _initialize_sources(self) -> Dict[str, DataSource]:
        """Initialize data sources based on configuration."""
        sources = {}

        if 'binance' in self.settings.data_sources:
            sources['binance'] = BinanceDataSource(
                api_key=self.settings.binance_api_key,
                secret_key=self.settings.binance_secret_key
            )

        if 'yfinance' in self.settings.data_sources:
            sources['yfinance'] = YFinanceDataSource()

        if 'coingecko' in self.settings.data_sources:
            sources['coingecko'] = CoinGeckoDataSource(
                api_key=self.settings.coingecko_api_key
            )

        self.logger.info(f"Initialized {len(sources)} data sources: {list(sources.keys())}")
        return sources

    async def fetch_data(
        self,
        symbol: str,
        timeframe: str = '1d',
        limit: int = 100,
        sources: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data from multiple sources.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            limit: Number of periods to fetch
            sources: List of sources to use (None for all)

        Returns:
            Dictionary mapping source names to DataFrames
        """
        if sources is None:
            sources = list(self.sources.keys())

        data = {}
        tasks = []

        for source_name in sources:
            if source_name in self.sources:
                source = self.sources[source_name]
                task = asyncio.create_task(
                    source.fetch_ohlcv(symbol, timeframe, limit),
                    name=f"{source_name}_{symbol}"
                )
                tasks.append((source_name, task))

        # Wait for all tasks to complete
        for source_name, task in tasks:
            try:
                result = await task
                data[source_name] = result
                self.logger.debug(f"Successfully fetched data from {source_name}")
            except Exception as e:
                self.logger.error(f"Failed to fetch data from {source_name}: {e}")

        self.logger.info(f"Fetched data from {len(data)} sources for {symbol}")
        return data

    async def fetch_latest_prices(
        self,
        symbols: List[str],
        sources: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch latest prices for multiple symbols.

        Args:
            symbols: List of trading symbols
            sources: List of sources to use

        Returns:
            Dictionary mapping symbols to price data
        """
        if sources is None:
            sources = list(self.sources.keys())

        prices = {}
        tasks = []

        for symbol in symbols:
            for source_name in sources:
                if source_name in self.sources:
                    source = self.sources[source_name]
                    task = asyncio.create_task(
                        source.fetch_ticker(symbol),
                        name=f"{source_name}_{symbol}_ticker"
                    )
                    tasks.append((symbol, source_name, task))

        # Wait for all tasks to complete
        for symbol, source_name, task in tasks:
            try:
                result = await task
                if symbol not in prices:
                    prices[symbol] = {}
                prices[symbol][source_name] = result
            except Exception as e:
                self.logger.error(f"Failed to fetch ticker from {source_name} for {symbol}: {e}")

        return prices

    def combine_data(self, data: Dict[str, pd.DataFrame], method: str = 'average') -> pd.DataFrame:
        """
        Combine data from multiple sources.

        Args:
            data: Dictionary of DataFrames from different sources
            method: Combination method ('average', 'first', 'binance_priority')

        Returns:
            Combined DataFrame
        """
        if not data:
            return pd.DataFrame()

        if len(data) == 1:
            return list(data.values())[0]

        if method == 'first':
            return list(data.values())[0]

        elif method == 'binance_priority':
            if 'binance' in data:
                return data['binance']
            else:
                return list(data.values())[0]

        elif method == 'average':
            # Align all DataFrames on timestamp
            aligned_dfs = []
            for source_name, df in data.items():
                df_copy = df.copy()
                df_copy.columns = [f"{col}_{source_name}" for col in df_copy.columns]
                aligned_dfs.append(df_copy)

            # Combine all DataFrames
            combined = pd.concat(aligned_dfs, axis=1, join='inner')

            # Calculate averages
            result = pd.DataFrame()
            for col in ['open', 'high', 'low', 'close', 'volume']:
                col_data = [c for c in combined.columns if c.startswith(col)]
                if col_data:
                    result[col] = combined[col_data].mean(axis=1)

            self.logger.info(f"Combined data from {len(data)} sources using {method} method")
            return result

        else:
            raise ValueError(f"Unknown combination method: {method}")

    async def close(self):
        """Close all data source connections."""
        for source in self.sources.values():
            if hasattr(source, 'close'):
                await source.close()
        self.logger.info("Closed all data source connections")