#!/usr/bin/env python3
"""
Verify that the cross-horizon fix works correctly.

This script tests that all prediction horizons now test the same calendar period,
which should result in identical buy-and-hold returns across all horizons.

Usage:
    python scripts/verify_cross_horizon_fix.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.walk_forward_test_enhanced import EnhancedWalkForwardTester

async def verify_fix():
    """Run quick test across multiple horizons to verify they test same period."""
    print(f"\n{'='*80}")
    print(f"VERIFICATION: Cross-Horizon Fix")
    print(f"{'='*80}\n")
    print("Testing that all horizons now use the same calendar period...")
    print("Expected: Buy-and-hold returns should be identical across all horizons\n")

    horizons = [1, 7, 14, 28, 60]
    results = {}

    # Quick test: small dataset, 3 windows
    base_days = 500  # Small dataset for quick verification

    for horizon in horizons:
        print(f"\n{'='*80}")
        print(f"Testing {horizon}-day horizon")
        print(f"{'='*80}")

        tester = EnhancedWalkForwardTester(
            prediction_horizon=horizon,
            mode='expanding',
            train_window=200,
            test_window=60,
            step_size=60,
            use_sequences=False,  # Disable for speed
            transaction_cost=0.002,
            max_features=15  # Fewer features for speed
        )

        # Run test
        result = await tester.run_walk_forward_test(days=base_days, interval='1d')

        # Extract buy-and-hold metrics
        bh_stats = result['aggregated'].get('buy_and_hold', {})

        results[horizon] = {
            'total_return': bh_stats.get('total_compounded_return', 0),
            'mean_return': bh_stats.get('mean_return', 0),
            'num_windows': bh_stats.get('num_windows', 0),
            'test_start': result['windows'][0]['test_start'] if result['windows'] else None,
            'test_end': result['windows'][-1]['test_end'] if result['windows'] else None
        }

    # Print verification results
    print(f"\n{'='*80}")
    print(f"VERIFICATION RESULTS")
    print(f"{'='*80}\n")

    print(f"Buy-and-Hold Returns by Horizon:")
    print(f"{'-'*80}")
    print(f"{'Horizon':<10} {'Total Return':>15} {'Mean Return':>15} {'Windows':>10} {'Test Period':<30}")
    print(f"{'-'*80}")

    for horizon in horizons:
        r = results[horizon]
        period = f"{r['test_start']} to {r['test_end']}" if r['test_start'] else "N/A"
        print(f"{horizon:>3}-day   {r['total_return']*100:>13.2f}%  {r['mean_return']*100:>13.2f}%  {r['num_windows']:>8}   {period}")

    # Check if returns are identical (within 0.1% tolerance)
    returns = [r['total_return'] for r in results.values()]
    max_return = max(returns)
    min_return = min(returns)
    difference = abs(max_return - min_return)

    print(f"\n{'='*80}")
    if difference < 0.001:  # Within 0.1%
        print(f"✅ SUCCESS: All horizons show identical buy-and-hold returns!")
        print(f"   Max difference: {difference*100:.3f}%")
        print(f"   Cross-horizon comparison is now FAIR")
    else:
        print(f"❌ ISSUE: Buy-and-hold returns differ across horizons")
        print(f"   Max difference: {difference*100:.2f}%")
        print(f"   Range: {min_return*100:.2f}% to {max_return*100:.2f}%")
        print(f"   This suggests horizons are testing different periods")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    asyncio.run(verify_fix())
