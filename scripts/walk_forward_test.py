#!/usr/bin/env python3
"""
Walk-Forward Testing for ML Trading Models

This script implements walk-forward analysis to validate model performance
in a more realistic way than simple train/test split. It simulates how models
would perform if retrained periodically on expanding historical data.

Usage:
    python scripts/walk_forward_test.py --horizon 14 --days 2190
    python scripts/walk_forward_test.py --horizon 14 --mode expanding --train-window 730 --test-window 90
    python scripts/walk_forward_test.py --horizon 14 --mode rolling --train-window 365 --test-window 60
"""

import asyncio
import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_bot.data.collector import DataCollector
from trading_bot.data.preprocessor import DataPreprocessor
from trading_bot.data.features import FeatureEngineer
from trading_bot.models.trainer import ModelTrainer
from trading_bot.models.ensemble import EnsembleModel
from trading_bot.evaluation.backtester import Backtester
from trading_bot.evaluation.metrics import PerformanceMetrics
from trading_bot.config.settings import Settings


class WalkForwardTester:
    """Walk-forward testing for ML trading models."""

    def __init__(
        self,
        prediction_horizon: int = 14,
        mode: str = 'expanding',  # 'expanding' or 'rolling'
        train_window: int = 730,  # days
        test_window: int = 90,     # days
        step_size: int = 30,       # days
        min_train_size: int = 365  # minimum training data
    ):
        """
        Initialize walk-forward tester.

        Args:
            prediction_horizon: Days ahead to predict
            mode: 'expanding' (growing train) or 'rolling' (fixed train window)
            train_window: Initial/fixed training window size (days)
            test_window: Test window size (days)
            step_size: Days to step forward each iteration
            min_train_size: Minimum training data required
        """
        self.prediction_horizon = prediction_horizon
        self.mode = mode
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.min_train_size = min_train_size

        self.settings = Settings()
        self.collector = DataCollector(self.settings)
        self.preprocessor = DataPreprocessor(self.settings._config_data)
        self.feature_engineer = FeatureEngineer(self.settings._config_data)
        self.backtester = Backtester(self.settings._config_data)

        print(f"\n{'='*80}")
        print(f"WALK-FORWARD TESTING: {prediction_horizon}-DAY PREDICTION")
        print(f"{'='*80}")
        print(f"Mode: {mode.upper()}")
        print(f"Train Window: {train_window} days ({'fixed' if mode == 'rolling' else 'expanding'})")
        print(f"Test Window: {test_window} days")
        print(f"Step Size: {step_size} days")
        print(f"{'='*80}\n")

    async def collect_data(self, days: int, interval: str = '1d') -> pd.DataFrame:
        """Collect historical data."""
        print(f"\n{'='*80}")
        print(f"DATA COLLECTION")
        print(f"{'='*80}")

        try:
            data = await self.collector.fetch_data(
                symbol='BTC-USD',
                timeframe=interval,
                limit=days
            )

            df = self.collector.combine_data(data, method='average')

            print(f"✓ Collected {len(df)} records")
            print(f"  Date range: {df.index[0]} to {df.index[-1]}")

            return df

        finally:
            await self.collector.close()

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess and engineer features."""
        print(f"\n{'='*80}")
        print(f"DATA PREPARATION")
        print(f"{'='*80}")

        # Preprocess
        cleaned_data = self.preprocessor.clean_data(data)
        print(f"✓ Preprocessed: {len(cleaned_data)} records")

        # Feature engineering
        features_df = self.feature_engineer.create_features(cleaned_data)
        features_df = features_df.dropna()
        print(f"✓ Features: {len(features_df.columns)} features, {len(features_df)} records")

        # Add target
        target_col = f'future_return_{self.prediction_horizon}d'
        features_df[target_col] = features_df['close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
        features_df = features_df.dropna(subset=[target_col])

        print(f"✓ Target: {target_col} ({len(features_df)} records after target creation)")

        return features_df

    def split_windows(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Split data into walk-forward windows.

        Returns:
            List of dicts with 'train_start', 'train_end', 'test_start', 'test_end' indices
        """
        windows = []
        data_len = len(data)

        # Start with initial train window
        current_train_end = self.train_window

        while current_train_end + self.test_window <= data_len:
            if self.mode == 'expanding':
                # Expanding window: train on all data from start
                train_start = 0
                train_end = current_train_end
            else:  # rolling
                # Rolling window: fixed-size train window
                train_start = max(0, current_train_end - self.train_window)
                train_end = current_train_end

            # Ensure minimum training size
            if train_end - train_start < self.min_train_size:
                current_train_end += self.step_size
                continue

            test_start = train_end
            test_end = min(test_start + self.test_window, data_len)

            windows.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_dates': (data.index[train_start], data.index[train_end - 1]),
                'test_dates': (data.index[test_start], data.index[test_end - 1])
            })

            current_train_end += self.step_size

        return windows

    def train_models_for_window(
        self,
        train_data: pd.DataFrame,
        target_col: str,
        window_idx: int
    ) -> Dict[str, Any]:
        """Train models on a specific window."""
        print(f"\n  Training models on {len(train_data)} samples...")

        # Prepare features and target
        feature_cols = [col for col in train_data.columns
                       if col not in ['close', 'open', 'high', 'low', 'volume', 'ticker', target_col]]

        X_train = train_data[feature_cols].values
        y_train = train_data[target_col].values

        # Train individual models
        models = {}
        algorithms = ['random_forest', 'xgboost', 'lightgbm', 'lstm', 'transformer']

        for algo in algorithms:
            try:
                trainer = ModelTrainer(self.settings, model_types=[algo])
                trained_models = trainer.train_models(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=None,
                    y_val=None,
                    feature_names=feature_cols
                )
                models[algo] = trained_models[algo]
                print(f"    ✓ {algo}")
            except Exception as e:
                print(f"    ✗ {algo}: {str(e)}")

        # Train ensemble (only if we have at least 2 models)
        if len(models) >= 2:
            try:
                ensemble = EnsembleModel(
                    models=models,
                    method='voting',
                    optimize_for_sharpe=True,
                    validation_split=0.2
                )
                ensemble.fit(X_train, y_train)
                models['ensemble_sharpe'] = ensemble
                print(f"    ✓ ensemble_sharpe")
            except Exception as e:
                print(f"    ✗ ensemble_sharpe: {str(e)}")

        return models

    def test_models_for_window(
        self,
        models: Dict[str, Any],
        test_data: pd.DataFrame,
        target_col: str,
        window_idx: int
    ) -> Dict[str, Any]:
        """Test models on a specific window."""
        print(f"\n  Testing models on {len(test_data)} samples...")

        feature_cols = [col for col in test_data.columns
                       if col not in ['close', 'open', 'high', 'low', 'volume', 'ticker', target_col]]

        X_test = test_data[feature_cols].values
        y_test = test_data[target_col].values

        window_results = {}

        for name, model in models.items():
            try:
                # Get predictions
                if name == 'ensemble_sharpe':
                    predictions = model.predict(X_test)
                else:
                    predictions = model.predict(X_test)

                # Create signals
                signals = np.sign(predictions)

                # Backtest
                test_df = test_data.copy()
                test_df['signal'] = signals
                test_df['actual_return'] = y_test

                # Calculate returns
                test_df['strategy_return'] = test_df['signal'] * test_df['actual_return']

                # Performance metrics
                total_return = (1 + test_df['strategy_return']).prod() - 1
                sharpe = (test_df['strategy_return'].mean() / test_df['strategy_return'].std()) * np.sqrt(252 / self.prediction_horizon)
                win_rate = (test_df['strategy_return'] > 0).sum() / len(test_df)

                window_results[name] = {
                    'total_return': total_return,
                    'sharpe_ratio': sharpe,
                    'win_rate': win_rate,
                    'num_trades': len(test_df)
                }

                print(f"    ✓ {name}: {total_return*100:+.2f}%")

            except Exception as e:
                print(f"    ✗ {name}: {str(e)}")
                window_results[name] = None

        return window_results

    async def run_walk_forward_test(
        self,
        days: int = 2190,
        interval: str = '1d'
    ) -> Dict[str, Any]:
        """
        Run complete walk-forward test.

        Args:
            days: Total historical data to collect
            interval: Data interval

        Returns:
            Aggregated results across all windows
        """
        # Collect and prepare data
        raw_data = await self.collect_data(days, interval)
        prepared_data = self.prepare_data(raw_data)

        # Split into windows
        windows = self.split_windows(prepared_data)
        print(f"\n✓ Created {len(windows)} walk-forward windows")

        if len(windows) == 0:
            raise ValueError("No windows created! Try reducing test_window or increasing days")

        target_col = f'future_return_{self.prediction_horizon}d'

        # Process each window
        all_results = []

        for i, window in enumerate(windows, 1):
            print(f"\n{'='*80}")
            print(f"WINDOW {i}/{len(windows)}")
            print(f"{'='*80}")
            print(f"Train: {window['train_dates'][0].date()} to {window['train_dates'][1].date()} ({window['train_end'] - window['train_start']} days)")
            print(f"Test:  {window['test_dates'][0].date()} to {window['test_dates'][1].date()} ({window['test_end'] - window['test_start']} days)")

            # Get train and test data
            train_data = prepared_data.iloc[window['train_start']:window['train_end']]
            test_data = prepared_data.iloc[window['test_start']:window['test_end']]

            # Train models
            models = self.train_models_for_window(train_data, target_col, i)

            # Test models
            results = self.test_models_for_window(models, test_data, target_col, i)

            all_results.append({
                'window': i,
                'train_start': window['train_dates'][0],
                'train_end': window['train_dates'][1],
                'test_start': window['test_dates'][0],
                'test_end': window['test_dates'][1],
                'results': results
            })

        # Aggregate results
        aggregated = self.aggregate_results(all_results)

        return {
            'config': {
                'prediction_horizon': self.prediction_horizon,
                'mode': self.mode,
                'train_window': self.train_window,
                'test_window': self.test_window,
                'step_size': self.step_size,
                'num_windows': len(windows)
            },
            'windows': all_results,
            'aggregated': aggregated
        }

    def aggregate_results(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results across all windows."""
        print(f"\n{'='*80}")
        print(f"AGGREGATED RESULTS")
        print(f"{'='*80}")

        # Collect all model names
        model_names = set()
        for window_result in all_results:
            model_names.update(window_result['results'].keys())

        aggregated = {}

        for model_name in sorted(model_names):
            returns = []
            sharpes = []
            win_rates = []

            for window_result in all_results:
                if model_name in window_result['results'] and window_result['results'][model_name]:
                    result = window_result['results'][model_name]
                    returns.append(result['total_return'])
                    sharpes.append(result['sharpe_ratio'])
                    win_rates.append(result['win_rate'])

            if returns:
                aggregated[model_name] = {
                    'mean_return': np.mean(returns),
                    'std_return': np.std(returns),
                    'median_return': np.median(returns),
                    'total_compounded_return': np.prod([1 + r for r in returns]) - 1,
                    'mean_sharpe': np.mean(sharpes),
                    'mean_win_rate': np.mean(win_rates),
                    'num_windows': len(returns),
                    'positive_windows': sum(1 for r in returns if r > 0),
                    'negative_windows': sum(1 for r in returns if r < 0)
                }

        # Print summary
        print(f"\nModel Performance Summary (across {len(all_results)} windows):")
        print(f"{'-'*80}")
        print(f"{'Model':<25} {'Mean Return':>12} {'Total Return':>12} {'Mean Sharpe':>12} {'Win %':>8}")
        print(f"{'-'*80}")

        for model_name in sorted(aggregated.keys()):
            stats = aggregated[model_name]
            print(f"{model_name:<25} "
                  f"{stats['mean_return']*100:>11.2f}% "
                  f"{stats['total_compounded_return']*100:>11.2f}% "
                  f"{stats['mean_sharpe']:>12.2f} "
                  f"{stats['positive_windows']/stats['num_windows']*100:>7.1f}%")

        return aggregated


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Walk-forward testing for ML trading models')

    parser.add_argument('--horizon', type=int, default=14,
                       help='Prediction horizon in days (default: 14)')
    parser.add_argument('--days', type=int, default=2190,
                       help='Total days of historical data (default: 2190 = ~6 years)')
    parser.add_argument('--mode', type=str, default='expanding', choices=['expanding', 'rolling'],
                       help='Window mode: expanding (growing) or rolling (fixed)')
    parser.add_argument('--train-window', type=int, default=730,
                       help='Training window size in days (default: 730 = 2 years)')
    parser.add_argument('--test-window', type=int, default=90,
                       help='Test window size in days (default: 90 = ~3 months)')
    parser.add_argument('--step-size', type=int, default=30,
                       help='Step size in days (default: 30 = 1 month)')
    parser.add_argument('--output', type=str, default='experiments/walk_forward',
                       help='Output directory')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize tester
    tester = WalkForwardTester(
        prediction_horizon=args.horizon,
        mode=args.mode,
        train_window=args.train_window,
        test_window=args.test_window,
        step_size=args.step_size
    )

    # Run walk-forward test
    results = await tester.run_walk_forward_test(
        days=args.days,
        interval='1d'
    )

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'walk_forward_{args.horizon}day_{timestamp}.json'

    # Convert datetime objects to strings for JSON serialization
    def convert_datetime(obj):
        if isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=convert_datetime)

    print(f"\n✓ Results saved to: {output_file}")

    # Print final summary
    print(f"\n{'='*80}")
    print(f"WALK-FORWARD TEST COMPLETE")
    print(f"{'='*80}")
    print(f"Mode: {args.mode}")
    print(f"Horizon: {args.horizon} days")
    print(f"Windows tested: {results['config']['num_windows']}")

    # Find best model
    best_model = None
    best_return = float('-inf')

    for model_name, stats in results['aggregated'].items():
        if stats['total_compounded_return'] > best_return:
            best_return = stats['total_compounded_return']
            best_model = model_name

    if best_model:
        print(f"\n🏆 Best Model: {best_model}")
        print(f"   Total Return: {best_return*100:+.2f}%")
        print(f"   Mean Return: {results['aggregated'][best_model]['mean_return']*100:+.2f}%")
        print(f"   Mean Sharpe: {results['aggregated'][best_model]['mean_sharpe']:.2f}")
        print(f"   Win Rate: {results['aggregated'][best_model]['positive_windows']}/{results['aggregated'][best_model]['num_windows']}")

    print(f"{'='*80}\n")


if __name__ == '__main__':
    asyncio.run(main())
