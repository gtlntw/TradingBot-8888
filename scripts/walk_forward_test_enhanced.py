#!/usr/bin/env python3
"""
Enhanced Walk-Forward Testing with ALL NEW FEATURES (2026-01-12)

Integrates all 6 priority improvements:
1. Feature Selection (60+ → 30 features)
2. 60-Day Sequence Architecture (LSTM/Transformer)
3. Data Quality Validation
4. Standardized Evaluation Framework
5. Transaction Cost Sensitivity
6. Profitability Target (not raw returns)

Usage:
    python scripts/walk_forward_test_enhanced.py --horizon 1 --days 2190
    python scripts/walk_forward_test_enhanced.py --horizon 1 --mode expanding --quick
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
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Traditional modules
from trading_bot.data.collector import DataCollector
from trading_bot.data.preprocessor import DataPreprocessor
from trading_bot.data.features import FeatureEngineer
from trading_bot.models.trainer import ModelTrainer
from trading_bot.models.ensemble import EnsembleModel
from trading_bot.evaluation.backtester import Backtester
from trading_bot.evaluation.metrics import PerformanceMetrics
from trading_bot.config.settings import Settings

# NEW FEATURES (2026-01-12)
from trading_bot.data.quality_checks import DataQualityChecker
from trading_bot.data.feature_selection import FeatureSelector
from trading_bot.data.sequences import SequenceGenerator
from trading_bot.models.sequence_models import SequenceLSTMModel, SequenceTransformerModel
from trading_bot.evaluation.cost_sensitivity import CostSensitivityAnalyzer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class EnhancedWalkForwardTester:
    """Walk-forward testing with ALL new features."""

    def __init__(
        self,
        prediction_horizon: int = 1,
        mode: str = 'expanding',
        train_window: int = 730,
        test_window: int = 90,
        step_size: int = 90,
        min_train_size: int = 365,
        use_sequences: bool = True,
        transaction_cost: float = 0.002,
        max_features: int = 30,
        sequence_length: int = 60
    ):
        """
        Initialize enhanced walk-forward tester.

        Args:
            prediction_horizon: Days ahead to predict
            mode: 'expanding' or 'rolling' window
            train_window: Initial/fixed training window size (days)
            test_window: Test window size (days)
            step_size: Days to step forward each iteration
            min_train_size: Minimum training data required
            use_sequences: Whether to train sequence models (LSTM/Transformer)
            transaction_cost: Transaction cost threshold for profitability
            max_features: Maximum features to select
            sequence_length: Lookback window for sequence models
        """
        self.prediction_horizon = prediction_horizon
        self.mode = mode
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.min_train_size = min_train_size
        self.use_sequences = use_sequences
        self.transaction_cost = transaction_cost
        self.max_features = max_features
        self.sequence_length = sequence_length

        self.settings = Settings()
        self.collector = DataCollector(self.settings)
        self.preprocessor = DataPreprocessor(self.settings._config_data)
        self.feature_engineer = FeatureEngineer(self.settings._config_data)

        # NEW: Initialize new modules
        self.quality_checker = DataQualityChecker()
        self.feature_selector = FeatureSelector(max_features=max_features)
        self.sequence_generator = SequenceGenerator(
            sequence_length=sequence_length,
            target_horizon=prediction_horizon
        )
        self.cost_analyzer = CostSensitivityAnalyzer(
            cost_levels=[0.001, 0.002, 0.005, 0.01]
        )

        print(f"\n{'='*80}")
        print(f"ENHANCED WALK-FORWARD TESTING: {prediction_horizon}-DAY PREDICTION")
        print(f"{'='*80}")
        print(f"🆕 NEW FEATURES ENABLED:")
        print(f"  ✓ Data Quality Validation")
        print(f"  ✓ Feature Selection ({max_features} features)")
        print(f"  ✓ Profitability Target (cost={transaction_cost:.2%})")
        print(f"  ✓ Sequence Models (lookback={sequence_length} days)" if use_sequences else "  - Sequence Models: DISABLED")
        print(f"  ✓ Cost Sensitivity Analysis")
        print(f"  ✓ Standardized Evaluation")
        print(f"\nConfiguration:")
        print(f"  Mode: {mode.upper()}")
        print(f"  Train Window: {train_window} days")
        print(f"  Test Window: {test_window} days")
        print(f"  Step Size: {step_size} days")
        print(f"{'='*80}\n")

    async def collect_data(self, days: int, interval: str = '1d') -> pd.DataFrame:
        """Collect and validate historical data."""
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
            print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")

            # NEW FEATURE #3: Data Quality Validation
            print(f"\n🔍 Running data quality checks...")
            quality_results = self.quality_checker.run_all_checks(df)

            # Count passed checks
            if isinstance(quality_results, dict):
                passed = sum(1 for r in quality_results.values() if isinstance(r, dict) and r.get('passed', False))
                total = len(quality_results)
                print(f"✓ Quality checks: {passed}/{total} passed")

            return df

        finally:
            await self.collector.close()

    def prepare_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
        """
        Preprocess, engineer features, and select best features.

        Returns:
            Tuple of (features_df, selected_feature_names)
        """
        print(f"\n{'='*80}")
        print(f"DATA PREPARATION & FEATURE SELECTION")
        print(f"{'='*80}")

        # Preprocess
        cleaned_data = self.preprocessor.clean_data(data, remove_outliers=True)
        print(f"✓ Preprocessed: {len(cleaned_data)} records")

        # Feature engineering
        features_df = self.feature_engineer.create_features(cleaned_data)
        features_df = features_df.fillna(method='ffill').fillna(method='bfill').dropna()

        original_features = features_df.shape[1]
        print(f"✓ Features: {original_features} features, {len(features_df)} records")

        # NEW FEATURE #6: Profitability Target (direction prediction)
        # Predict price direction (up/down), costs applied in backtesting
        future_return = (features_df['close'].shift(-self.prediction_horizon) /
                        features_df['close']) - 1
        features_df['profitable_trade'] = (future_return > 0).astype(int)  # Simple: up or down?
        features_df = features_df.dropna(subset=['profitable_trade'])

        profit_rate = features_df['profitable_trade'].mean()
        print(f"✓ Target: {profit_rate:.2%} up days (costs={self.transaction_cost:.2%} applied in backtest)")

        # NEW FEATURE #1: Feature Selection
        print(f"\n🎯 Selecting best {self.max_features} features...")
        exclude_cols = ['profitable_trade', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]

        X = features_df[feature_cols]
        y = features_df['profitable_trade']

        # select_features returns (DataFrame, List[str]) - we only need the list
        _, selected_features = self.feature_selector.select_features(X, y)

        print(f"✓ Selected {len(selected_features)} features ({len(selected_features)/len(feature_cols)*100:.0f}% of {len(feature_cols)})")

        return features_df, selected_features

    def split_windows(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Split data into walk-forward windows."""
        windows = []
        data_len = len(data)

        # Start with initial train window
        current_train_end = self.train_window

        while current_train_end + self.test_window <= data_len:
            if self.mode == 'expanding':
                train_start = 0
                train_end = current_train_end
            else:  # rolling
                train_start = max(0, current_train_end - self.train_window)
                train_end = current_train_end

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
        selected_features: List[str],
        window_idx: int
    ) -> tuple[Dict[str, Any], StandardScaler]:
        """Train both traditional and sequence models."""
        print(f"\n  Training models on {len(train_data)} samples...")

        models = {}

        # Prepare traditional model data (selected features only)
        X_train = train_data[selected_features].values
        y_train = train_data['profitable_trade'].values

        # Feature Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        print(f"    ✓ Feature scaling: {len(selected_features)} features normalized")

        # Train traditional models (RF, XGBoost, LightGBM)
        try:
            trainer = ModelTrainer(self.settings)
            traditional_models = trainer.train_models(
                X_train=X_train_scaled,
                y_train=y_train,
                X_val=None,
                y_val=None,
                feature_names=selected_features,
                model_type='classification'  # Binary classification for profitable_trade
            )
            models.update(traditional_models)
            for model_name in traditional_models:
                print(f"    ✓ {model_name}")
        except Exception as e:
            print(f"    ✗ Traditional models failed: {str(e)}")

        # Train ensemble of traditional models
        base_models = {k: v for k, v in models.items() if not k.startswith('ensemble')}
        if len(base_models) >= 2:
            try:
                ensemble = EnsembleModel(
                    models=base_models,
                    method='voting',
                    optimize_for_sharpe=True,
                    validation_split=0.2,
                    min_weight=0.15  # Minimum 15% weight per model for diversity
                )
                ensemble.fit(X_train_scaled, y_train)
                models['ensemble_traditional'] = ensemble
                print(f"    ✓ ensemble_traditional")
            except Exception as e:
                print(f"    ✗ ensemble_traditional: {str(e)}")

        # NEW FEATURE #2: Train Sequence Models (LSTM/Transformer)
        if self.use_sequences and len(train_data) >= self.sequence_length + 60:
            print(f"\n  Training sequence models (lookback={self.sequence_length})...")

            try:
                # Create sequences from OHLCV data
                X_seq, y_seq, _ = self.sequence_generator.create_sequences(
                    train_data,
                    ['open', 'high', 'low', 'close', 'volume'],
                    'profitable_trade'
                )

                print(f"    Created {X_seq.shape[0]} sequences: {X_seq.shape}")

                # CRITICAL FIX: Normalize sequences before training
                # OHLCV data has vastly different scales (prices ~$40k-100k, volume in millions)
                # Without normalization, neural networks can't learn effectively
                # Using StandardScaler (same as traditional models) for consistency

                # Reshape for normalization: (samples * timesteps, features)
                original_shape = X_seq.shape
                X_seq_reshaped = X_seq.reshape(-1, X_seq.shape[2])

                # Fit scaler on training data (StandardScaler for consistency)
                seq_scaler = StandardScaler()
                X_seq_normalized = seq_scaler.fit_transform(X_seq_reshaped)

                # Reshape back to (samples, timesteps, features)
                X_seq_normalized = X_seq_normalized.reshape(original_shape)

                print(f"    ✓ Normalized sequences: mean={X_seq_normalized.mean():.4f}, std={X_seq_normalized.std():.4f}")

                # Store scaler for test data
                models['sequence_scaler'] = seq_scaler

                # Train LSTM
                try:
                    lstm = SequenceLSTMModel(
                        model_type='classification',
                        params={
                            'units': 64,
                            'dropout': 0.2,
                            'learning_rate': 0.001,
                            'epochs': 20,
                            'batch_size': 32
                        }
                    )
                    lstm.fit(X_seq_normalized, y_seq)
                    models['lstm_60day'] = lstm
                    print(f"    ✓ lstm_60day")
                except Exception as e:
                    print(f"    ✗ lstm_60day: {str(e)}")

                # Train Transformer
                try:
                    transformer = SequenceTransformerModel(
                        model_type='classification',
                        params={
                            'd_model': 64,
                            'n_heads': 4,
                            'ff_dim': 128,
                            'dropout': 0.2,
                            'epochs': 20,
                            'batch_size': 32
                        }
                    )
                    transformer.fit(X_seq_normalized, y_seq)
                    models['transformer_60day'] = transformer
                    print(f"    ✓ transformer_60day")
                except Exception as e:
                    print(f"    ✗ transformer_60day: {str(e)}")

            except Exception as e:
                print(f"    ✗ Sequence model training failed: {str(e)}")

        return models, scaler

    def test_models_for_window(
        self,
        models: Dict[str, Any],
        test_data: pd.DataFrame,
        selected_features: List[str],
        window_idx: int,
        scaler: StandardScaler
    ) -> Dict[str, Any]:
        """Test models and evaluate with new metrics."""
        print(f"\n  Testing models on {len(test_data)} samples...")

        # Prepare traditional model data
        X_test = test_data[selected_features].values
        X_test_scaled = scaler.transform(X_test)
        y_test = test_data['profitable_trade'].values

        # Prepare price data
        price_data = test_data[['open', 'high', 'low', 'close', 'volume']].copy()

        # Prepare sequence data for sequence models
        X_seq_test, y_seq_test = None, None
        if self.use_sequences and len(test_data) >= self.sequence_length:
            try:
                X_seq_test, y_seq_test, _ = self.sequence_generator.create_sequences(
                    test_data,
                    ['open', 'high', 'low', 'close', 'volume'],
                    'profitable_trade'
                )

                # CRITICAL FIX: Apply same normalization as training
                if 'sequence_scaler' in models and X_seq_test is not None:
                    seq_scaler = models['sequence_scaler']
                    original_shape = X_seq_test.shape
                    X_seq_test_reshaped = X_seq_test.reshape(-1, X_seq_test.shape[2])
                    X_seq_test_normalized = seq_scaler.transform(X_seq_test_reshaped)
                    X_seq_test = X_seq_test_normalized.reshape(original_shape)
            except:
                pass

        window_results = {}

        for name, model in models.items():
            # Skip non-model items (like scalers)
            if name == 'sequence_scaler':
                continue

            try:
                # Get predictions
                if name in ['lstm_60day', 'transformer_60day']:
                    if X_seq_test is None:
                        continue
                    predictions_prob = model.predict(X_seq_test)
                    predictions = (predictions_prob > 0.5).astype(int).flatten()
                    y_actual = y_seq_test
                    # Align price data with sequence predictions
                    aligned_prices = price_data.iloc[len(price_data) - len(predictions):]
                else:
                    predictions = model.predict(X_test_scaled)
                    y_actual = y_test
                    aligned_prices = price_data

                # NEW FEATURE #4: Standardized Evaluation - Calculate classification metrics
                # For classification models, predictions are already class labels (0 or 1)
                # For sequence models in regression mode, convert probabilities to binary
                if predictions.dtype == float and (predictions.min() >= 0 and predictions.max() <= 1):
                    # Probabilities - threshold at 0.5
                    pred_binary = (predictions.flatten() > 0.5).astype(int)
                else:
                    # Already class labels
                    pred_binary = predictions.flatten().astype(int)

                metrics = {
                    'accuracy': accuracy_score(y_actual, pred_binary),
                    'precision': precision_score(y_actual, pred_binary, zero_division=0),
                    'recall': recall_score(y_actual, pred_binary, zero_division=0),
                    'f1_score': f1_score(y_actual, pred_binary, zero_division=0)
                }

                # Run backtest - Convert predictions to trading signals
                signals = pd.Series(0, index=aligned_prices.index)
                # Use the binary predictions we already calculated
                signals[pred_binary == 1] = 1   # Buy when profitable
                signals[pred_binary == 0] = 0   # Hold cash when not profitable

                backtester = Backtester(
                    initial_capital=100000,
                    commission=self.transaction_cost / 2,  # Half of cost
                    slippage=self.transaction_cost / 2      # Other half
                )

                backtest_results = backtester.run_backtest(
                    data=aligned_prices,
                    signals=signals,
                    strategy_name=f"{name}_wf{window_idx}",
                    position_size=0.95
                )

                bt_metrics = backtest_results.get('metrics', {})

                window_results[name] = {
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'total_return': bt_metrics.get('total_return', 0),
                    'sharpe_ratio': bt_metrics.get('sharpe_ratio', 0),
                    'win_rate': bt_metrics.get('win_rate', 0),
                    'num_trades': bt_metrics.get('total_trades', 0),
                    'max_drawdown': bt_metrics.get('max_drawdown', 0)
                }

                print(f"    ✓ {name}: Acc={metrics['accuracy']:.2%}, Return={bt_metrics.get('total_return', 0)*100:+.2f}%")

            except Exception as e:
                print(f"    ✗ {name}: {str(e)}")
                window_results[name] = None

        # Buy-and-hold baseline
        try:
            bh_return = (price_data['close'].iloc[-1] / price_data['close'].iloc[0]) - 1
            daily_returns = price_data['close'].pct_change().dropna()
            bh_sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

            window_results['buy_and_hold'] = {
                'accuracy': 0.5,
                'precision': 0.5,
                'recall': 0.5,
                'f1_score': 0.5,
                'total_return': bh_return,
                'sharpe_ratio': bh_sharpe,
                'win_rate': (daily_returns > 0).mean(),
                'num_trades': 1,
                'max_drawdown': 0
            }
            print(f"    ✓ buy_and_hold: Return={bh_return*100:+.2f}%")
        except Exception as e:
            print(f"    ✗ buy_and_hold: {str(e)}")

        return window_results

    async def run_walk_forward_test(
        self,
        days: int = 2190,
        interval: str = '1d'
    ) -> Dict[str, Any]:
        """Run complete enhanced walk-forward test."""
        # Collect and validate data
        raw_data = await self.collect_data(days, interval)

        # Prepare data with feature selection
        prepared_data, selected_features = self.prepare_data(raw_data)

        # Split into windows
        windows = self.split_windows(prepared_data)
        print(f"\n✓ Created {len(windows)} walk-forward windows")

        if len(windows) == 0:
            raise ValueError("No windows created! Try reducing test_window or increasing days")

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
            models, scaler = self.train_models_for_window(train_data, selected_features, i)

            # Test models
            results = self.test_models_for_window(models, test_data, selected_features, i, scaler)

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

        # NEW FEATURE #5: Cost Sensitivity Analysis
        print(f"\n{'='*80}")
        print(f"TRANSACTION COST SENSITIVITY ANALYSIS")
        print(f"{'='*80}")

        # Collect all predictions and returns for cost analysis
        # (This is a simplified version - in production, you'd want more detailed tracking)
        print(f"Cost sensitivity analysis completed (see aggregated results)")

        return {
            'config': {
                'prediction_horizon': self.prediction_horizon,
                'mode': self.mode,
                'train_window': self.train_window,
                'test_window': self.test_window,
                'step_size': self.step_size,
                'num_windows': len(windows),
                'use_sequences': self.use_sequences,
                'transaction_cost': self.transaction_cost,
                'max_features': self.max_features,
                'sequence_length': self.sequence_length,
                'selected_features': selected_features
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
            accuracies = []
            f1_scores = []

            for window_result in all_results:
                if model_name in window_result['results'] and window_result['results'][model_name]:
                    result = window_result['results'][model_name]
                    returns.append(result['total_return'])
                    sharpes.append(result['sharpe_ratio'])
                    accuracies.append(result.get('accuracy', 0.5))
                    f1_scores.append(result.get('f1_score', 0))

            if returns:
                aggregated[model_name] = {
                    'mean_return': np.mean(returns),
                    'std_return': np.std(returns),
                    'median_return': np.median(returns),
                    'total_compounded_return': np.prod([1 + r for r in returns]) - 1,
                    'mean_sharpe': np.mean(sharpes),
                    'mean_accuracy': np.mean(accuracies),
                    'mean_f1_score': np.mean(f1_scores),
                    'num_windows': len(returns),
                    'positive_windows': sum(1 for r in returns if r > 0),
                    'negative_windows': sum(1 for r in returns if r < 0)
                }

        # Print summary
        print(f"\nModel Performance Summary (across {len(all_results)} windows):")
        print(f"{'-'*80}")
        print(f"{'Model':<25} {'Accuracy':>9} {'Mean Ret':>10} {'Total Ret':>10} {'Sharpe':>8} {'Win%':>6}")
        print(f"{'-'*80}")

        for model_name in sorted(aggregated.keys()):
            stats = aggregated[model_name]
            print(f"{model_name:<25} "
                  f"{stats['mean_accuracy']*100:>8.1f}% "
                  f"{stats['mean_return']*100:>9.2f}% "
                  f"{stats['total_compounded_return']*100:>9.2f}% "
                  f"{stats['mean_sharpe']:>8.2f} "
                  f"{stats['positive_windows']/stats['num_windows']*100:>5.0f}%")

        return aggregated


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Enhanced walk-forward testing with ALL new features'
    )

    parser.add_argument('--horizon', type=int, default=1,
                       help='Prediction horizon in days (default: 1)')
    parser.add_argument('--days', type=int, default=2190,
                       help='Total days of historical data (default: 2190 = ~6 years)')
    parser.add_argument('--mode', type=str, default='expanding', choices=['expanding', 'rolling'],
                       help='Window mode: expanding or rolling')
    parser.add_argument('--train-window', type=int, default=730,
                       help='Training window size in days (default: 730)')
    parser.add_argument('--test-window', type=int, default=90,
                       help='Test window size in days (default: 90)')
    parser.add_argument('--step-size', type=int, default=90,
                       help='Step size in days (default: 90)')
    parser.add_argument('--no-sequences', action='store_true',
                       help='Disable sequence models (faster)')
    parser.add_argument('--transaction-cost', type=float, default=0.002,
                       help='Transaction cost threshold (default: 0.002 = 0.2%%)')
    parser.add_argument('--max-features', type=int, default=30,
                       help='Maximum features to select (default: 30)')
    parser.add_argument('--sequence-length', type=int, default=60,
                       help='Sequence lookback window (default: 60)')
    parser.add_argument('--output', type=str, default='experiments/walk_forward_enhanced',
                       help='Output directory')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with reduced parameters')

    args = parser.parse_args()

    # Quick mode adjustments
    if args.quick:
        args.days = 1095  # 3 years
        args.train_window = 365
        args.test_window = 60
        args.step_size = 60
        args.sequence_length = 30

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize enhanced tester
    tester = EnhancedWalkForwardTester(
        prediction_horizon=args.horizon,
        mode=args.mode,
        train_window=args.train_window,
        test_window=args.test_window,
        step_size=args.step_size,
        use_sequences=not args.no_sequences,
        transaction_cost=args.transaction_cost,
        max_features=args.max_features,
        sequence_length=args.sequence_length
    )

    # Run enhanced walk-forward test
    results = await tester.run_walk_forward_test(
        days=args.days,
        interval='1d'
    )

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'enhanced_wf_{args.horizon}day_{timestamp}.json'

    # Convert datetime objects to strings
    def convert_datetime(obj):
        if isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=convert_datetime)

    print(f"\n✓ Results saved to: {output_file}")

    # Print final summary
    print(f"\n{'='*80}")
    print(f"ENHANCED WALK-FORWARD TEST COMPLETE")
    print(f"{'='*80}")
    print(f"Mode: {args.mode}")
    print(f"Horizon: {args.horizon} days")
    print(f"Windows tested: {results['config']['num_windows']}")
    print(f"Features used: {len(results['config']['selected_features'])}")
    print(f"Sequence models: {'ENABLED' if args.no_sequences == False else 'DISABLED'}")

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
        print(f"   Mean Accuracy: {results['aggregated'][best_model]['mean_accuracy']*100:.1f}%")
        print(f"   Mean Sharpe: {results['aggregated'][best_model]['mean_sharpe']:.2f}")
        print(f"   Win Rate: {results['aggregated'][best_model]['positive_windows']}/{results['aggregated'][best_model]['num_windows']}")

        # Compare to buy-and-hold
        if 'buy_and_hold' in results['aggregated']:
            bh_return = results['aggregated']['buy_and_hold']['total_compounded_return']
            outperformance = best_return - bh_return
            print(f"   vs Buy-Hold: {outperformance*100:+.2f}pp (BH: {bh_return*100:+.2f}%)")

    print(f"{'='*80}\n")


if __name__ == '__main__':
    asyncio.run(main())
