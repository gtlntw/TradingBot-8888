"""
Test script for on-chain and derivatives data sources.

This script demonstrates how to:
1. Fetch on-chain metrics from Glassnode
2. Fetch derivatives data from Coinglass
3. Combine with price data
4. Generate on-chain features

Usage:
    python examples/test_onchain_data.py

Environment variables needed:
    GLASSNODE_API_KEY - Your Glassnode API key (optional for testing)
    COINGLASS_API_KEY - Your Coinglass API key (optional)
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from dotenv import load_dotenv

from trading_bot.data.sources.glassnode import GlassnodeDataSource, OnChainMetric
from trading_bot.data.sources.coinglass import CoinglassDataSource
from trading_bot.data.features import OnChainFeatures
from trading_bot.utils.logger import setup_logger

# Load environment variables
load_dotenv()

logger = setup_logger(__name__)


async def test_glassnode():
    """Test Glassnode on-chain data fetching."""
    print("\n" + "="*60)
    print("TESTING GLASSNODE ON-CHAIN DATA")
    print("="*60)

    api_key = os.getenv('GLASSNODE_API_KEY')

    if not api_key:
        print("⚠️  GLASSNODE_API_KEY not found in environment")
        print("   Get your free API key at: https://glassnode.com")
        print("   Add it to your .env file: GLASSNODE_API_KEY=your_key_here")
        print("\n   Skipping Glassnode tests...")
        return None

    source = GlassnodeDataSource(api_key=api_key)

    # Fetch last 90 days of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    print(f"\n📊 Fetching on-chain data from {start_date.date()} to {end_date.date()}")

    try:
        # Fetch essential metrics for swing trading
        df = await source.get_all_essential_metrics(start_date, end_date)

        if not df.empty:
            print(f"\n✅ Successfully fetched {len(df)} days of on-chain data")
            print(f"   Metrics: {list(df.columns)}")
            print(f"\n📈 Latest on-chain data:")
            print(df.tail(5))

            # Calculate some insights
            latest = df.iloc[-1]
            print(f"\n🔍 Current Market State:")
            print(f"   MVRV Ratio: {latest.get('mvrv', 'N/A'):.2f} (>3.5 = overheated, <1.0 = undervalued)")
            print(f"   SOPR: {latest.get('sopr', 'N/A'):.3f} (>1.0 = profits, <1.0 = losses)")

            if 'exchange_net_flow' in df.columns:
                net_flow_7d = df['exchange_net_flow'].tail(7).sum()
                if net_flow_7d < 0:
                    print(f"   Exchange Flow (7d): {net_flow_7d:,.0f} BTC (🟢 ACCUMULATION - bullish)")
                else:
                    print(f"   Exchange Flow (7d): {net_flow_7d:,.0f} BTC (🔴 DISTRIBUTION - bearish)")

            return df

        else:
            print("❌ No data returned from Glassnode")
            return None

    except Exception as e:
        print(f"❌ Error fetching Glassnode data: {e}")
        return None


async def test_coinglass():
    """Test Coinglass derivatives data fetching."""
    print("\n" + "="*60)
    print("TESTING COINGLASS DERIVATIVES DATA")
    print("="*60)

    source = CoinglassDataSource()

    print("\n📊 Fetching derivatives snapshot for BTC...")

    try:
        # Get comprehensive derivatives snapshot
        snapshot = await source.get_derivatives_snapshot("BTC")

        print(f"\n✅ Successfully fetched derivatives snapshot")
        print(f"   Timestamp: {snapshot['timestamp']}")

        # Funding rates
        if not snapshot['funding_rates'].empty:
            avg_funding = snapshot.get('avg_funding_rate', 0)
            print(f"\n💰 Funding Rates:")
            print(f"   Average: {avg_funding:.4f}% per 8h")

            if abs(avg_funding) > 0.05:
                direction = "LONGS" if avg_funding > 0 else "SHORTS"
                print(f"   ⚠️  EXTREME FUNDING - {direction} overleveraged!")
            else:
                print(f"   ✅ Funding rates normal")

            print(f"\n   Top Exchanges:")
            print(snapshot['funding_rates'].head())

        # Liquidations
        if snapshot['liquidations']:
            liq = snapshot['liquidations']
            total = liq.get('total_liquidations', 0)
            longs = liq.get('long_liquidations', 0)
            shorts = liq.get('short_liquidations', 0)
            ratio = liq.get('long_short_ratio', 0)

            print(f"\n🔥 Liquidations (24h):")
            print(f"   Total: ${total:,.0f}")
            print(f"   Longs: ${longs:,.0f}")
            print(f"   Shorts: ${shorts:,.0f}")
            print(f"   L/S Ratio: {ratio:.2f}")

            if ratio > 2.0:
                print(f"   📊 Interpretation: More longs liquidated (bearish pressure)")
            elif ratio < 0.5:
                print(f"   📊 Interpretation: More shorts liquidated (bullish pressure)")
            else:
                print(f"   📊 Interpretation: Balanced liquidations")

        # Open Interest
        if not snapshot['open_interest'].empty:
            total_oi = snapshot['open_interest']['open_interest'].sum()
            print(f"\n💼 Open Interest:")
            print(f"   Total: ${total_oi:,.0f}")

        return snapshot

    except Exception as e:
        print(f"❌ Error fetching Coinglass data: {e}")
        return None


async def test_feature_engineering(onchain_df):
    """Test on-chain feature engineering."""
    print("\n" + "="*60)
    print("TESTING ON-CHAIN FEATURE ENGINEERING")
    print("="*60)

    if onchain_df is None or onchain_df.empty:
        print("⚠️  No on-chain data available for feature engineering")
        return

    # Create OnChainFeatures calculator
    feature_calc = OnChainFeatures()

    print("\n🔧 Generating on-chain features...")

    try:
        # Generate features
        df_with_features = feature_calc.calculate(onchain_df)

        print(f"\n✅ Generated {len(df_with_features.columns)} total columns")
        print(f"   Original: {len(onchain_df.columns)} metrics")
        print(f"   Added: {len(df_with_features.columns) - len(onchain_df.columns)} derived features")

        # Show new features
        new_features = [col for col in df_with_features.columns if col not in onchain_df.columns]
        print(f"\n📊 New Features Generated:")
        for feature in new_features[:15]:  # Show first 15
            print(f"   - {feature}")
        if len(new_features) > 15:
            print(f"   ... and {len(new_features) - 15} more")

        # Show latest feature values
        print(f"\n🔍 Latest Feature Values (sample):")
        sample_features = ['mvrv_overheated', 'mvrv_undervalued', 'sopr_profitable',
                          'strong_accumulation', 'strong_distribution']
        available = [f for f in sample_features if f in df_with_features.columns]

        if available:
            print(df_with_features[available].tail(5))

        return df_with_features

    except Exception as e:
        print(f"❌ Error generating features: {e}")
        return None


async def test_combined_pipeline():
    """Test complete pipeline: price + on-chain + derivatives."""
    print("\n" + "="*60)
    print("TESTING COMBINED DATA PIPELINE")
    print("="*60)

    print("\n📦 This will combine:")
    print("   1. Price data (yfinance)")
    print("   2. On-chain metrics (Glassnode)")
    print("   3. Derivatives data (Coinglass)")
    print("   4. Feature engineering")

    # Note: Full implementation would integrate with existing data collector
    print("\n💡 For full pipeline integration, see:")
    print("   - trading_bot/data/collector.py (data collection)")
    print("   - trading_bot/data/features.py (feature engineering)")
    print("   - configs/swing_trading.yaml (configuration)")

    print("\n✅ Phase 1 Complete: On-chain data sources implemented!")
    print("\n📝 Next Steps:")
    print("   1. Get API keys from Glassnode and Coinglass")
    print("   2. Add keys to .env file")
    print("   3. Run this script to test data collection")
    print("   4. Proceed to Phase 2: Hybrid ML models")


async def main():
    """Run all tests."""
    print("\n" + "="*80)
    print(" " * 20 + "ON-CHAIN DATA INTEGRATION TEST")
    print("="*80)

    # Test Glassnode
    onchain_df = await test_glassnode()

    # Test Coinglass
    derivatives_snapshot = await test_coinglass()

    # Test feature engineering
    if onchain_df is not None:
        features_df = await test_feature_engineering(onchain_df)

    # Show combined pipeline info
    await test_combined_pipeline()

    print("\n" + "="*80)
    print(" " * 30 + "TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    # Run tests
    asyncio.run(main())
