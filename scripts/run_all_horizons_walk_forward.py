#!/usr/bin/env python3
"""
Run ENHANCED walk-forward testing across all prediction horizons.

NOW USES ALL NEW FEATURES (2026-01-12):
- Feature Selection (60+ → 30 features)
- 60-Day Sequence Models (LSTM/Transformer)
- Data Quality Validation
- Profitability Target (not raw returns)
- Cost Sensitivity Analysis
- Standardized Evaluation

This script runs comprehensive walk-forward tests for all 5 horizons
(1, 7, 14, 28, 60 days) and generates a comparison report.

Usage:
    python scripts/run_all_horizons_walk_forward.py
    python scripts/run_all_horizons_walk_forward.py --mode rolling
    python scripts/run_all_horizons_walk_forward.py --days 1095 --quick
    python scripts/run_all_horizons_walk_forward.py --no-sequences  # Disable LSTM/Transformer for speed
"""

import asyncio
import argparse
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd


def find_existing_result(horizon: int, max_age_hours: int = 24) -> dict:
    """Find existing result file for a horizon if it exists and is recent."""
    output_dir = Path('experiments/walk_forward_enhanced')
    if not output_dir.exists():
        return None

    pattern = f'enhanced_wf_{horizon}day_*.json'
    files = sorted(output_dir.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)

    if not files:
        return None

    # Check if most recent file is fresh enough
    latest_file = files[0]
    file_age_hours = (datetime.now().timestamp() - latest_file.stat().st_mtime) / 3600

    if file_age_hours > max_age_hours:
        return None

    try:
        with open(latest_file, 'r') as f:
            result = json.load(f)

        # Validate it has the expected structure
        if 'aggregated' in result and 'windows' in result:
            print(f"  Found existing result: {latest_file.name} ({file_age_hours:.1f}h old)")
            return result
    except Exception as e:
        print(f"  Warning: Could not load {latest_file.name}: {e}")
        return None

    return None


def run_walk_forward(horizon: int, args: argparse.Namespace) -> dict:
    """Run walk-forward test for a single horizon."""
    print(f"\n{'='*80}")
    print(f"RUNNING WALK-FORWARD TEST: {horizon}-DAY HORIZON")
    print(f"{'='*80}\n")

    # Check for existing result if resume mode
    if args.resume and not args.force:
        existing = find_existing_result(horizon, max_age_hours=args.max_age)
        if existing:
            print(f"✓ Resuming: Using existing result for {horizon}-day horizon\n")
            return existing

    # Calculate minimum required test window for sequence generation
    min_required = args.sequence_length + horizon
    test_window = max(args.test_window, min_required + 30)

    if test_window != args.test_window:
        print(f"⚠️  Adjusted test window: {args.test_window} → {test_window} days")
        print(f"   (Horizon {horizon} + sequence length {args.sequence_length} = {min_required} days minimum)\n")

    cmd = [
        'python', 'scripts/walk_forward_test_enhanced.py',
        '--horizon', str(horizon),
        '--days', str(args.days),
        '--mode', args.mode,
        '--train-window', str(args.train_window),
        '--test-window', str(test_window),
        '--step-size', str(args.step_size),
        '--transaction-cost', str(args.transaction_cost),
        '--max-features', str(args.max_features),
        '--sequence-length', str(args.sequence_length)
    ]

    # Add optional flags
    if args.no_sequences:
        cmd.append('--no-sequences')
    if args.quick:
        cmd.append('--quick')

    # CRITICAL FIX: Don't capture output - stream to console/log to avoid OOM
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\n✗ Walk-forward test failed for {horizon}-day horizon")
        print(f"   (Check output above for errors)")
        return None

    # Find the output file
    output_dir = Path('experiments/walk_forward_enhanced')
    pattern = f'enhanced_wf_{horizon}day_*.json'
    files = sorted(output_dir.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)

    if files:
        with open(files[0], 'r') as f:
            return json.load(f)

    return None


def create_comparison_report(all_results: dict, output_file: Path):
    """Create comparison report across all horizons."""
    rows = []

    for horizon, results in all_results.items():
        if not results or 'aggregated' not in results:
            continue

        for model_name, stats in results['aggregated'].items():
            rows.append({
                'Horizon': f'{horizon}day',
                'Model': model_name,
                'Mean Return (%)': stats['mean_return'] * 100,
                'Total Return (%)': stats['total_compounded_return'] * 100,
                'Std Return (%)': stats['std_return'] * 100,
                'Median Return (%)': stats['median_return'] * 100,
                'Mean Sharpe': stats['mean_sharpe'],
                'Mean Accuracy (%)': stats['mean_accuracy'] * 100,
                'Mean F1 Score': stats['mean_f1_score'],
                'Positive Windows': stats['positive_windows'],
                'Negative Windows': stats['negative_windows'],
                'Total Windows': stats['num_windows']
            })

    df = pd.DataFrame(rows)

    # Sort by horizon then by total return
    df = df.sort_values(['Horizon', 'Total Return (%)'], ascending=[True, False])

    # Save to CSV
    df.to_csv(output_file, index=False)

    print(f"\n✓ Comparison report saved to: {output_file}")

    return df


def print_summary(df: pd.DataFrame):
    """Print summary of results."""
    print(f"\n{'='*80}")
    print(f"WALK-FORWARD RESULTS SUMMARY")
    print(f"{'='*80}\n")

    for horizon in ['1day', '7day', '14day', '28day', '60day']:
        horizon_df = df[df['Horizon'] == horizon]
        if len(horizon_df) == 0:
            continue

        print(f"\n{horizon.upper()} HORIZON:")
        print(f"{'-'*80}")

        # Best by total return
        best_model = horizon_df.iloc[0]
        print(f"Best Model: {best_model['Model']}")
        print(f"  Total Return: {best_model['Total Return (%)']:+.2f}%")
        print(f"  Mean Return: {best_model['Mean Return (%)']:+.2f}%")
        print(f"  Mean Sharpe: {best_model['Mean Sharpe']:.2f}")
        print(f"  Mean Accuracy: {best_model['Mean Accuracy (%)']:.1f}%")
        print(f"  Mean F1 Score: {best_model['Mean F1 Score']:.3f}")
        print(f"  Win Rate: {best_model['Positive Windows']}/{best_model['Total Windows']} windows")

        # Buy-and-hold comparison
        bh_row = horizon_df[horizon_df['Model'] == 'buy_and_hold']
        if len(bh_row) > 0:
            bh_return = bh_row.iloc[0]['Total Return (%)']
            outperformance = best_model['Total Return (%)'] - bh_return
            print(f"  vs Buy-Hold: {outperformance:+.2f}pp ({bh_return:+.2f}% BH)")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run walk-forward tests across all horizons')

    parser.add_argument('--days', type=int, default=2190,
                       help='Total days of historical data (default: 2190)')
    parser.add_argument('--mode', type=str, default='expanding', choices=['expanding', 'rolling'],
                       help='Window mode (default: expanding)')
    parser.add_argument('--train-window', type=int, default=730,
                       help='Training window size in days (default: 730)')
    parser.add_argument('--test-window', type=int, default=90,
                       help='Test window size in days (default: 90)')
    parser.add_argument('--step-size', type=int, default=90,
                       help='Step size in days (default: 90)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with smaller dataset')
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 7, 14, 28, 60],
                       help='Horizons to test (default: 1 7 14 28 60)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing results (skip already-completed horizons)')
    parser.add_argument('--force', action='store_true',
                       help='Force re-run even if results exist (overrides --resume)')
    parser.add_argument('--max-age', type=int, default=24,
                       help='Maximum age of existing results in hours for resume (default: 24)')
    parser.add_argument('--no-sequences', action='store_true',
                       help='Disable sequence models (LSTM/Transformer) for faster execution')
    parser.add_argument('--transaction-cost', type=float, default=0.000,
                       help='Transaction cost threshold (default: 0.000 = 0.0%%)')
    parser.add_argument('--max-features', type=int, default=30,
                       help='Maximum features to select (default: 30)')
    parser.add_argument('--sequence-length', type=int, default=60,
                       help='Sequence lookback window for LSTM/Transformer (default: 60)')

    args = parser.parse_args()

    # Quick mode: smaller dataset
    if args.quick:
        args.days = 1095  # 3 years
        args.train_window = 365
        args.test_window = 60
        args.step_size = 60

    print(f"\n{'='*80}")
    print(f"ENHANCED MULTI-HORIZON WALK-FORWARD TESTING")
    print(f"{'='*80}")
    print(f"🆕 NEW FEATURES ENABLED:")
    print(f"  ✓ Feature Selection (max {args.max_features} features)")
    print(f"  ✓ Profitability Target (cost={args.transaction_cost:.2%})")
    print(f"  ✓ Sequence Models (lookback={args.sequence_length})" if not args.no_sequences else "  - Sequence Models: DISABLED")
    print(f"  ✓ Data Quality Validation")
    print(f"  ✓ Cost Sensitivity Analysis")
    print(f"  ✓ Standardized Evaluation")
    print(f"\nConfiguration:")
    print(f"  Horizons: {args.horizons}")
    print(f"  Mode: {args.mode}")
    print(f"  Data: {args.days} days")
    print(f"  Train Window: {args.train_window} days")
    print(f"  Test Window: {args.test_window} days")
    print(f"  Step Size: {args.step_size} days")
    if args.resume:
        print(f"  Resume: Enabled (max age: {args.max_age}h)")
    if args.force:
        print(f"  Force: Re-run all horizons")

    # Validate test window requirements for each horizon
    print(f"\n📊 Test Window Validation:")
    adjustments_needed = False
    for horizon in args.horizons:
        min_required = args.sequence_length + horizon
        if args.test_window < min_required:
            adjusted = min_required + 30
            print(f"  ⚠️  {horizon}-day: {args.test_window} days → {adjusted} days (need {min_required} min)")
            adjustments_needed = True
        else:
            print(f"  ✓ {horizon}-day: {args.test_window} days (need {min_required} min)")

    if adjustments_needed:
        print(f"\n  Note: Test windows will be auto-adjusted per horizon to ensure sufficient data")

    print(f"{'='*80}\n")

    # Run walk-forward for each horizon
    all_results = {}

    for horizon in args.horizons:
        result = run_walk_forward(horizon, args)
        if result:
            all_results[horizon] = result
            print(f"✓ Completed {horizon}-day horizon")
        else:
            print(f"✗ Failed {horizon}-day horizon")

    if not all_results:
        print("\n✗ No results to analyze")
        return 1

    # Create comparison report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('experiments/walk_forward_enhanced')
    output_dir.mkdir(parents=True, exist_ok=True)

    report_file = output_dir / f'enhanced_multi_horizon_comparison_{timestamp}.csv'
    df = create_comparison_report(all_results, report_file)

    # Print summary
    print_summary(df)

    print(f"\n{'='*80}")
    print(f"ENHANCED MULTI-HORIZON TESTING COMPLETE")
    print(f"{'='*80}")
    print(f"Tested {len(all_results)} horizons with ALL new features")
    print(f"Results: {report_file}")
    print(f"Features: Selection, Sequences, Quality, Cost Sensitivity, Standardized Eval")
    print(f"{'='*80}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
