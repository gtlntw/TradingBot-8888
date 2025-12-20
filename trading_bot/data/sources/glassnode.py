"""
Glassnode on-chain data source.
Provides Bitcoin blockchain metrics via Glassnode API.
"""

import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum

from trading_bot.utils.logger import LoggerMixin
from trading_bot.utils.decorators import retry, rate_limit


class OnChainMetric(Enum):
    """Available on-chain metrics from Glassnode."""

    # Market indicators
    MVRV = "market/mvrv"
    MVRV_Z_SCORE = "market/mvrv_z_score"
    NVT = "indicators/nvt"
    SOPR = "indicators/sopr"

    # Network fundamentals
    ACTIVE_ADDRESSES = "addresses/active_count"
    NEW_ADDRESSES = "addresses/new_non_zero_count"
    HASH_RATE = "mining/hash_rate_mean"
    DIFFICULTY = "mining/difficulty_latest"

    # Supply metrics
    SUPPLY_HELD_1Y = "supply/hodl_waves/1y_2y"
    SUPPLY_HELD_2Y = "supply/hodl_waves/2y_3y"
    ILLIQUID_SUPPLY = "supply/illiquid_sum"
    LIQUID_SUPPLY = "supply/liquid_sum"

    # Exchange flows
    EXCHANGE_BALANCE = "distribution/balance_exchanges"
    EXCHANGE_INFLOW = "transactions/transfers_volume_exchanges_in_sum"
    EXCHANGE_OUTFLOW = "transactions/transfers_volume_exchanges_out_sum"
    EXCHANGE_NET_FLOW = "transactions/transfers_volume_exchanges_net"

    # Whale metrics
    SUPPLY_TOP_1PCT = "distribution/balance_1pct_holders"
    WHALE_COUNT = "distribution/balance_addresses_count_10k_100k"

    # Miner metrics
    MINER_BALANCE = "mining/miner_balances"
    MINER_OUTFLOW = "mining/miner_outflow_multiple"

    # Derivatives (if available)
    FUTURES_OPEN_INTEREST = "derivatives/futures_open_interest_sum"
    FUTURES_VOLUME = "derivatives/futures_volume_daily_sum"


class GlassnodeDataSource(LoggerMixin):
    """
    Glassnode on-chain data source.

    API Documentation: https://docs.glassnode.com/api/
    Free tier: 20 requests/day, 1 year historical data
    """

    BASE_URL = "https://api.glassnode.com/v1/metrics"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Glassnode data source.

        Args:
            api_key: Glassnode API key (get from https://glassnode.com)
        """
        self.api_key = api_key
        if not self.api_key:
            self.logger.warning(
                "No Glassnode API key provided. "
                "Set GLASSNODE_API_KEY in environment or .env file. "
                "Get your key at: https://glassnode.com"
            )

    @retry(max_attempts=3, delay=2.0)
    @rate_limit(calls_per_second=0.5)  # Conservative rate limit for free tier
    async def fetch_metric(
        self,
        metric: OnChainMetric,
        asset: str = "BTC",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "24h"
    ) -> pd.DataFrame:
        """
        Fetch a specific on-chain metric from Glassnode.

        Args:
            metric: OnChainMetric to fetch
            asset: Asset symbol (default: BTC)
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval (1h, 24h, 1w, 1month)

        Returns:
            DataFrame with timestamp and metric value
        """
        if not self.api_key:
            raise ValueError(
                "Glassnode API key required. "
                "Set GLASSNODE_API_KEY environment variable."
            )

        # Default to last 365 days if not specified
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)

        # Build API endpoint
        endpoint = f"{self.BASE_URL}/{metric.value}"

        # Build parameters
        params = {
            'a': asset,
            'i': interval,
            's': int(start_date.timestamp()),
            'u': int(end_date.timestamp()),
            'api_key': self.api_key
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Convert to DataFrame
                        df = pd.DataFrame(data)

                        if df.empty:
                            self.logger.warning(f"No data returned for {metric.name}")
                            return pd.DataFrame()

                        # Process timestamps
                        df['timestamp'] = pd.to_datetime(df['t'], unit='s')
                        df['value'] = df['v'].astype(float)
                        df = df[['timestamp', 'value']]
                        df.set_index('timestamp', inplace=True)

                        self.logger.info(
                            f"Fetched {len(df)} data points for {metric.name} "
                            f"from {start_date.date()} to {end_date.date()}"
                        )

                        return df

                    elif response.status == 401:
                        raise ValueError("Invalid Glassnode API key")
                    elif response.status == 429:
                        raise ValueError("Glassnode rate limit exceeded. Upgrade plan or wait.")
                    else:
                        error_text = await response.text()
                        raise ValueError(
                            f"Glassnode API error {response.status}: {error_text}"
                        )

        except Exception as e:
            self.logger.error(f"Error fetching {metric.name} from Glassnode: {e}")
            raise

    async def fetch_multiple_metrics(
        self,
        metrics: List[OnChainMetric],
        asset: str = "BTC",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "24h"
    ) -> pd.DataFrame:
        """
        Fetch multiple on-chain metrics and combine into single DataFrame.

        Args:
            metrics: List of OnChainMetrics to fetch
            asset: Asset symbol
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval

        Returns:
            DataFrame with all metrics as columns
        """
        self.logger.info(f"Fetching {len(metrics)} on-chain metrics from Glassnode")

        # Fetch all metrics concurrently
        tasks = [
            self.fetch_metric(metric, asset, start_date, end_date, interval)
            for metric in metrics
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine into single DataFrame
        combined_df = pd.DataFrame()

        for metric, result in zip(metrics, results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to fetch {metric.name}: {result}")
                continue

            if not result.empty:
                # Rename column to metric name
                result_renamed = result.rename(columns={'value': metric.name.lower()})

                if combined_df.empty:
                    combined_df = result_renamed
                else:
                    combined_df = combined_df.join(result_renamed, how='outer')

        if not combined_df.empty:
            # Forward fill missing values (some metrics update less frequently)
            combined_df = combined_df.fillna(method='ffill')

            self.logger.info(
                f"Successfully fetched {len(combined_df.columns)} on-chain metrics "
                f"with {len(combined_df)} data points"
            )

        return combined_df

    async def get_whale_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get comprehensive whale activity metrics.

        Returns:
            DataFrame with whale-related metrics
        """
        whale_metrics = [
            OnChainMetric.SUPPLY_TOP_1PCT,
            OnChainMetric.WHALE_COUNT,
            OnChainMetric.EXCHANGE_BALANCE,
            OnChainMetric.EXCHANGE_NET_FLOW,
        ]

        return await self.fetch_multiple_metrics(
            whale_metrics,
            start_date=start_date,
            end_date=end_date
        )

    async def get_market_health_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get market health indicators (MVRV, SOPR, NVT).

        Returns:
            DataFrame with market health metrics
        """
        health_metrics = [
            OnChainMetric.MVRV,
            OnChainMetric.MVRV_Z_SCORE,
            OnChainMetric.SOPR,
            OnChainMetric.NVT,
        ]

        return await self.fetch_multiple_metrics(
            health_metrics,
            start_date=start_date,
            end_date=end_date
        )

    async def get_supply_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get supply distribution and holder behavior metrics.

        Returns:
            DataFrame with supply metrics
        """
        supply_metrics = [
            OnChainMetric.SUPPLY_HELD_1Y,
            OnChainMetric.SUPPLY_HELD_2Y,
            OnChainMetric.ILLIQUID_SUPPLY,
            OnChainMetric.LIQUID_SUPPLY,
        ]

        return await self.fetch_multiple_metrics(
            supply_metrics,
            start_date=start_date,
            end_date=end_date
        )

    async def get_all_essential_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get all essential on-chain metrics for swing trading.

        Focuses on metrics most relevant for 2-8 week swing trades:
        - Whale behavior (smart money)
        - Market valuation (MVRV, SOPR)
        - Supply dynamics (liquid vs illiquid)
        - Exchange flows (buying vs selling pressure)

        Returns:
            DataFrame with all essential metrics
        """
        essential_metrics = [
            # Market valuation
            OnChainMetric.MVRV,
            OnChainMetric.SOPR,

            # Whale activity
            OnChainMetric.SUPPLY_TOP_1PCT,
            OnChainMetric.EXCHANGE_BALANCE,
            OnChainMetric.EXCHANGE_NET_FLOW,

            # Supply dynamics
            OnChainMetric.ILLIQUID_SUPPLY,
            OnChainMetric.LIQUID_SUPPLY,

            # Network health
            OnChainMetric.ACTIVE_ADDRESSES,
            OnChainMetric.HASH_RATE,
        ]

        return await self.fetch_multiple_metrics(
            essential_metrics,
            start_date=start_date,
            end_date=end_date
        )


# Example usage and testing
async def main():
    """Example usage of GlassnodeDataSource."""
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv('GLASSNODE_API_KEY')

    source = GlassnodeDataSource(api_key=api_key)

    # Fetch last 90 days of essential metrics
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    print("Fetching on-chain metrics from Glassnode...")
    df = await source.get_all_essential_metrics(start_date, end_date)

    print(f"\nFetched {len(df)} days of data")
    print(f"Columns: {list(df.columns)}")
    print(f"\nLatest data:")
    print(df.tail())


if __name__ == "__main__":
    asyncio.run(main())
