#!/usr/bin/env python3
"""
Comprehensive test script for NEW features added on 2026-01-12.

Tests all 6 priority improvements:
1. Feature Selection (60+ → 30 features)
2. 60-Day Sequence Architecture (LSTM/Transformer)
3. Data Quality Validation
4. Standardized Evaluation Framework
5. Transaction Cost Sensitivity
6. Profitability Target (not raw returns)

Usage:
    python scripts/test_new_features.py --days 730  # 2 years for testing
    python scripts/test_new_features.py --days 3650  # 10 years for full analysis
"""
import sys
import asyncio
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_bot.config.settings import Settings
from trading_bot.data.collector import DataCollector
from trading_bot.data.preprocessor import DataPreprocessor
from trading_bot.data.features import FeatureEngineer
from trading_bot.data.feature_selection import FeatureSelector
from trading_bot.data.sequences import SequenceGenerator
from trading_bot.data.quality_checks import DataQualityChecker
from trading_bot.models.trainer import ModelTrainer
from trading_bot.models.sequence_models import SequenceLSTMModel, SequenceTransformerModel
from trading_bot.evaluation.standardized_eval import StandardizedEvaluator
from trading_bot.evaluation.cost_sensitivity import CostSensitivityAnalyzer
from trading_bot.utils.logger import setup_logging, get_logger
from trading_bot.utils.helpers import generate_timestamp


class NewFeaturesTester:
    """Tests all new features added on 2026-01-12."""

    def __init__(self, days: int = 730):
        """
        Initialize tester.

        Args:
            days: Number of days of historical data (default: 730 = 2 years)
        """
        self.days = days
        self.timestamp = generate_timestamp()

        # Setup paths
        self.output_dir = Path(f'experiments/new_features_test_{self.timestamp}')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize settings
        self.settings = Settings()
        setup_logging(self.settings.get('logging', {}))
        self.logger = get_logger(__name__)

        # Results storage
        self.results = {
            'config': {
                'days': days,
                'timestamp': self.timestamp,
                'test_date': str(datetime.now())
            },
            'quality_checks': {},
            'feature_selection': {},
            'traditional_models': {},
            'sequence_models': {},
            'cost_sensitivity': {},
            'standardized_eval': {}
        }

        print("="*80)
        print("🚀 TESTING ALL NEW FEATURES (2026-01-12)")
        print("="*80)
        print(f"Configuration:")
        print(f"  Historical Data: {days} days ({days/365:.1f} years)")
        print(f"  Output Directory: {self.output_dir}")
        print("="*80)

    async def run(self):
        """Execute all tests."""
        try:
            # Step 1: Data Collection
            await self._collect_data()

            # Step 2: Data Quality Checks (NEW FEATURE #3)
            self._check_data_quality()

            # Step 3: Preprocessing & Feature Engineering
            self._preprocess_and_engineer()

            # Step 4: Feature Selection (NEW FEATURE #1)
            self._select_features()

            # Step 5: Train Traditional Models
            self._train_traditional_models()

            # Step 6: Train Sequence Models (NEW FEATURE #2)
            self._train_sequence_models()

            # Step 7: Standardized Evaluation (NEW FEATURE #4)
            self._standardized_evaluation()

            # Step 8: Cost Sensitivity Analysis (NEW FEATURE #5)
            self._cost_sensitivity()

            # Save final results
            self._save_results()

            print("\n" + "="*80)
            print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"\nResults saved to: {self.output_dir}")

            return self.results

        except Exception as e:
            self.logger.error(f"Test failed: {e}", exc_info=True)
            print(f"\n❌ TEST FAILED: {e}")
            raise

    async def _collect_data(self):
        """Collect BTC data."""
        print("\n" + "="*80)
        print("📊 STEP 1: DATA COLLECTION")
        print("="*80)

        collector = DataCollector(self.settings)

        try:
            data = await collector.fetch_data(
                symbol='BTC-USD',
                timeframe='1d',
                limit=self.days,
                sources=['yfinance']
            )

            if data:
                self.raw_data = collector.combine_data(data, method='average')

                # Save raw data
                raw_file = self.output_dir / 'raw_data.csv'
                self.raw_data.to_csv(raw_file)

                print(f"✓ Collected {len(self.raw_data)} records")
                print(f"  Date range: {self.raw_data.index[0]} to {self.raw_data.index[-1]}")
                print(f"  Columns: {list(self.raw_data.columns)}")

                self.results['data_collection'] = {
                    'records': len(self.raw_data),
                    'date_range': [str(self.raw_data.index[0]), str(self.raw_data.index[-1])],
                    'file': str(raw_file)
                }
            else:
                raise ValueError("No data collected")
        finally:
            await collector.close()

    def _check_data_quality(self):
        """NEW FEATURE #3: Data Quality Validation."""
        print("\n" + "="*80)
        print("🔍 STEP 2: DATA QUALITY CHECKS (NEW FEATURE #3)")
        print("="*80)

        checker = DataQualityChecker()
        quality_results = checker.run_all_checks(self.raw_data)

        print("\n Quality Check Results:")
        print("-" * 80)
        for check_name, result in quality_results.items():
            status = "✓ PASS" if result['passed'] else "❌ FAIL"
            print(f"{status} - {check_name}")
            if not result['passed']:
                print(f"       Issues: {result.get('issues', 'Unknown')}")

        # Save quality report
        report_file = self.output_dir / 'quality_report.json'
        checker.save_report(report_file)
        print(f"\n✓ Quality report saved to: {report_file}")

        self.results['quality_checks'] = quality_results

    def _preprocess_and_engineer(self):
        """Preprocess and engineer features."""
        print("\n" + "="*80)
        print("⚙️  STEP 3: PREPROCESSING & FEATURE ENGINEERING")
        print("="*80)

        # Preprocess
        preprocessor = DataPreprocessor(self.settings._config_data)
        df_clean = preprocessor.clean_data(self.raw_data, remove_outliers=True)

        print(f"✓ Cleaned data: {len(df_clean)} records ({len(self.raw_data) - len(df_clean)} outliers removed)")

        # Engineer features
        feature_engineer = FeatureEngineer(self.settings._config_data)
        self.features = feature_engineer.create_features(df_clean)
        self.features = self.features.dropna()

        print(f"✓ Generated {self.features.shape[1]} features")
        print(f"✓ After dropping NaN: {len(self.features)} records")

        # Add profitability target (NEW FEATURE #6)
        transaction_cost = 0.002  # 0.2%
        future_return = (self.features['close'].shift(-1) / self.features['close']) - 1
        self.features['profitable_trade'] = (future_return > transaction_cost).astype(int)
        self.features = self.features.dropna()

        print(f"✓ Created profitability target (cost threshold: {transaction_cost:.2%})")
        print(f"  Profitability rate: {self.features['profitable_trade'].mean():.2%}")

    def _select_features(self):
        """NEW FEATURE #1: Feature Selection."""
        print("\n" + "="*80)
        print("🎯 STEP 4: FEATURE SELECTION (NEW FEATURE #1)")
        print("="*80)

        selector = FeatureSelector(max_features=30, correlation_threshold=0.95)

        # Prepare features
        exclude_cols = ['profitable_trade', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in self.features.columns if col not in exclude_cols]

        X = self.features[feature_cols]
        y = self.features['profitable_trade']

        print(f"Starting with {len(feature_cols)} features")

        # Select features
        selected_features = selector.select_features(X, y)

        print(f"\n✓ Selected {len(selected_features)} features")
        print(f"  Reduction: {len(feature_cols)} → {len(selected_features)} ({len(selected_features)/len(feature_cols)*100:.1f}% retained)")

        # Save visualization
        viz_file = self.output_dir / 'feature_selection.png'
        selector.plot_feature_importance(save_path=viz_file)

        self.selected_features = selected_features
        self.results['feature_selection'] = {
            'original_count': len(feature_cols),
            'selected_count': len(selected_features),
            'selected_features': selected_features
        }

    def _train_traditional_models(self):
        """Train traditional ML models on selected features."""
        print("\n" + "="*80)
        print("🤖 STEP 5: TRAIN TRADITIONAL MODELS (with feature selection)")
        print("="*80)

        # Prepare data with selected features
        X = self.features[self.selected_features]
        y = self.features['profitable_trade']

        # Time-series split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        print(f"Features: {len(self.selected_features)}")

        # Train models
        trainer = ModelTrainer(self.settings)
        models = trainer.train_models(
            X_train=X_train.values,
            y_train=y_train.values,
            X_val=X_test.values,
            y_val=y_test.values,
            feature_names=self.selected_features
        )

        print(f"\n✓ Trained {len(models)} models: {list(models.keys())}")

        # Store for later use
        self.traditional_models = models
        self.X_train_trad = X_train
        self.X_test_trad = X_test
        self.y_train_trad = y_train
        self.y_test_trad = y_test

    def _train_sequence_models(self):
        """NEW FEATURE #2: Train 60-day sequence models."""
        print("\n" + "="*80)
        print("🔮 STEP 6: TRAIN SEQUENCE MODELS (NEW FEATURE #2)")
        print("="*80)

        # Create sequences
        seq_gen = SequenceGenerator(sequence_length=60, target_horizon=1)

        # Use only OHLCV for sequences
        ohlcv_features = ['open', 'high', 'low', 'close', 'volume']
        X_seq, y_seq, metadata = seq_gen.create_sequences(
            self.features,
            ohlcv_features,
            'profitable_trade'
        )

        print(f"✓ Created sequences: {X_seq.shape}")
        print(f"  Sequence length: 60 days")
        print(f"  Features per timestep: 5 (OHLCV)")
        print(f"  Total samples: {len(X_seq)}")

        # Time-series split
        split_idx = int(len(X_seq) * 0.8)
        X_train_seq = X_seq[:split_idx]
        X_test_seq = X_seq[split_idx:]
        y_train_seq = y_seq[:split_idx]
        y_test_seq = y_seq[split_idx:]

        print(f"\nTrain: {len(X_train_seq)} sequences, Test: {len(X_test_seq)} sequences")

        # Train LSTM
        print("\nTraining LSTM...")
        lstm_model = SequenceLSTMModel(
            sequence_length=60,
            n_features=5,
            lstm_units=64,
            dropout=0.2,
            learning_rate=0.001
        )
        lstm_model.fit(X_train_seq, y_train_seq, epochs=20, batch_size=32, verbose=0)
        print("✓ LSTM trained")

        # Train Transformer
        print("\nTraining Transformer...")
        transformer_model = SequenceTransformerModel(
            sequence_length=60,
            n_features=5,
            d_model=64,
            n_heads=4,
            ff_dim=128,
            dropout=0.2
        )
        transformer_model.fit(X_train_seq, y_train_seq, epochs=20, batch_size=32, verbose=0)
        print("✓ Transformer trained")

        # Store for evaluation
        self.sequence_models = {
            'lstm': lstm_model,
            'transformer': transformer_model
        }
        self.X_test_seq = X_test_seq
        self.y_test_seq = y_test_seq

    def _standardized_evaluation(self):
        """NEW FEATURE #4: Standardized Evaluation Framework."""
        print("\n" + "="*80)
        print("📊 STEP 7: STANDARDIZED EVALUATION (NEW FEATURE #4)")
        print("="*80)

        evaluator = StandardizedEvaluator(test_split=0.2, validation_split=0.1)

        # Evaluate traditional models
        print("\n Traditional Models:")
        print("-" * 80)
        for name, model in self.traditional_models.items():
            y_pred = model.predict(self.X_test_trad.values)
            metrics = evaluator.evaluate_model(
                y_true=self.y_test_trad.values,
                y_pred=y_pred,
                model_name=name,
                prices=self.features.loc[self.y_test_trad.index, 'close'].values
            )

            print(f"\n{name.upper()}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1: {metrics['f1_score']:.4f}")

            self.results['traditional_models'][name] = metrics

        # Evaluate sequence models
        print("\n\n Sequence Models:")
        print("-" * 80)
        for name, model in self.sequence_models.items():
            y_pred_prob = model.predict(self.X_test_seq)
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()

            # Get corresponding prices
            test_prices = self.features.iloc[len(self.features) - len(y_pred):]['close'].values

            metrics = evaluator.evaluate_model(
                y_true=self.y_test_seq,
                y_pred=y_pred,
                model_name=name,
                prices=test_prices
            )

            print(f"\n{name.upper()}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1: {metrics['f1_score']:.4f}")

            self.results['sequence_models'][name] = metrics

        # Buy-and-hold baseline
        baseline_metrics = evaluator.compare_to_baseline(
            y_true=self.y_test_trad.values,
            y_pred=self.traditional_models['random_forest'].predict(self.X_test_trad.values),
            prices=self.features.loc[self.y_test_trad.index, 'close'].values
        )

        print("\n\n📈 Buy-and-Hold Baseline Comparison:")
        print("-" * 80)
        print(f"Buy-and-Hold Return: {baseline_metrics['buy_and_hold_return']:.2%}")
        print(f"Best Model Excess Return: {baseline_metrics['excess_return']:.2%}")

        self.results['standardized_eval'] = {
            'baseline': baseline_metrics
        }

    def _cost_sensitivity(self):
        """NEW FEATURE #5: Transaction Cost Sensitivity Analysis."""
        print("\n" + "="*80)
        print("💰 STEP 8: COST SENSITIVITY ANALYSIS (NEW FEATURE #5)")
        print("="*80)

        analyzer = CostSensitivityAnalyzer()

        # Test cost levels
        cost_levels = [0.0005, 0.001, 0.002, 0.005, 0.01]

        # Analyze best traditional model
        best_model = self.traditional_models['random_forest']
        y_pred = best_model.predict(self.X_test_trad.values)
        test_returns = self.features.loc[self.y_test_trad.index, 'close'].pct_change().shift(-1).dropna()

        # Align predictions with returns
        y_pred_aligned = y_pred[:len(test_returns)]

        print("\nCost Sensitivity for Random Forest:")
        print("-" * 80)

        cost_results = {}
        for cost in cost_levels:
            result = analyzer.analyze_single_cost(
                returns=test_returns.values,
                predictions=y_pred_aligned,
                transaction_cost=cost
            )

            cost_results[cost] = result
            print(f"\nCost = {cost:.2%}:")
            print(f"  Net Return: {result['net_return']:.2%}")
            print(f"  Sharpe Ratio: {result['sharpe_ratio']:.3f}")
            print(f"  Total Trades: {result['num_trades']}")

        # Find breakeven cost
        breakeven = analyzer.find_breakeven_cost(test_returns.values, y_pred_aligned)
        print(f"\n✓ Breakeven Cost: {breakeven:.4%}")

        self.results['cost_sensitivity'] = {
            'cost_levels': cost_results,
            'breakeven_cost': breakeven
        }

    def _save_results(self):
        """Save all results to JSON."""
        results_file = self.output_dir / 'test_results.json'

        # Convert any numpy types to Python types
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj

        results_converted = convert_types(self.results)

        with open(results_file, 'w') as f:
            json.dump(results_converted, f, indent=2)

        print(f"\n✅ Results saved to: {results_file}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Test all new features added on 2026-01-12'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=730,
        help='Number of days of historical data (default: 730 = 2 years)'
    )

    args = parser.parse_args()

    print(f"\n🚀 Testing with {args.days} days ({args.days/365:.1f} years) of data\n")

    # Run tests
    tester = NewFeaturesTester(days=args.days)
    await tester.run()


if __name__ == '__main__':
    asyncio.run(main())
