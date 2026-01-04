"""
Pipeline testing script for experimenting with different configurations.

Tests the complete ML pipeline (steps 1-5) with various prediction horizons
and saves results for comparison.

Usage:
    python scripts/test_pipeline.py --horizons 1 7 14 28
    python scripts/test_pipeline.py --horizon 7 --days 365
    python scripts/test_pipeline.py --all  # Run all default horizons
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
from trading_bot.models.trainer import ModelTrainer
from trading_bot.models.ensemble import EnsembleModel
from trading_bot.evaluation.metrics import PerformanceMetrics
from trading_bot.utils.logger import setup_logging, get_logger
from trading_bot.utils.helpers import setup_directories, generate_timestamp


class PipelineExperiment:
    """Run pipeline experiments with different configurations."""

    def __init__(self, prediction_horizon: int, days: int = 365, interval: str = '1d'):
        """
        Initialize experiment.

        Args:
            prediction_horizon: Number of periods ahead to predict
            days: Number of days of historical data
            interval: Data interval (1d, 1h, etc.)
        """
        self.prediction_horizon = prediction_horizon
        self.days = days
        self.interval = interval
        self.timestamp = generate_timestamp()

        # Setup paths
        self.experiment_dir = Path(f'experiments/results/{prediction_horizon}day')
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Initialize
        self.settings = Settings()
        setup_logging(self.settings.get('logging', {}))
        self.logger = get_logger(__name__)

        # Results storage
        self.results = {
            'config': {
                'prediction_horizon': prediction_horizon,
                'days': days,
                'interval': interval,
                'timestamp': self.timestamp
            },
            'steps': {},
            'metrics': {},
            'models': {}
        }

    async def run(self):
        """Execute complete pipeline experiment."""
        print("="*80)
        print(f"PIPELINE EXPERIMENT: {self.prediction_horizon}-DAY PREDICTION HORIZON")
        print("="*80)
        print(f"Configuration:")
        print(f"  Prediction Horizon: {self.prediction_horizon} days")
        print(f"  Historical Data: {self.days} days")
        print(f"  Interval: {self.interval}")
        print(f"  Output: {self.experiment_dir}")
        print("="*80)

        try:
            # Step 1: Data Collection
            await self._step1_collect_data()

            # Step 2: Data Preprocessing
            self._step2_preprocess()

            # Step 3: Feature Engineering
            self._step3_engineer_features()

            # Step 4: Model Training (with custom horizon)
            self._step4_train_models()

            # Step 5: Model Evaluation
            self._step5_evaluate()

            # Save results
            self._save_results()

            print("\n" + "="*80)
            print("✓ EXPERIMENT COMPLETED SUCCESSFULLY")
            print("="*80)

            return self.results

        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            print(f"\n✗ EXPERIMENT FAILED: {e}")
            raise

    async def _step1_collect_data(self):
        """Step 1: Collect data."""
        print("\n" + "="*80)
        print("STEP 1: DATA COLLECTION")
        print("="*80)

        collector = DataCollector(self.settings)

        try:
            data = await collector.fetch_data(
                symbol='BTC-USD',
                timeframe=self.interval,
                limit=self.days,
                sources=['yfinance']
            )

            if data:
                combined_data = collector.combine_data(data, method='average')

                # Save raw data
                raw_file = self.experiment_dir / f'raw_data_{self.timestamp}.csv'
                combined_data.to_csv(raw_file)

                print(f"✓ Collected {len(combined_data)} records")
                print(f"  Date range: {combined_data.index[0]} to {combined_data.index[-1]}")
                print(f"  Saved to: {raw_file}")

                self.results['steps']['data_collection'] = {
                    'records': len(combined_data),
                    'date_range': [str(combined_data.index[0]), str(combined_data.index[-1])],
                    'file': str(raw_file)
                }
                self.raw_data = combined_data
            else:
                raise ValueError("No data collected")

        finally:
            await collector.close()

    def _step2_preprocess(self):
        """Step 2: Preprocess data."""
        print("\n" + "="*80)
        print("STEP 2: DATA PREPROCESSING")
        print("="*80)

        preprocessor = DataPreprocessor(self.settings._config_data)
        df_clean = preprocessor.clean_data(self.raw_data, remove_outliers=True)

        print(f"✓ Cleaned {len(df_clean)} records")
        print(f"  Removed {len(self.raw_data) - len(df_clean)} outliers")

        self.results['steps']['preprocessing'] = {
            'records_before': len(self.raw_data),
            'records_after': len(df_clean),
            'outliers_removed': len(self.raw_data) - len(df_clean)
        }
        self.clean_data = df_clean

    def _step3_engineer_features(self):
        """Step 3: Engineer features."""
        print("\n" + "="*80)
        print("STEP 3: FEATURE ENGINEERING")
        print("="*80)

        feature_engineer = FeatureEngineer(self.settings._config_data)
        df_features = feature_engineer.create_features(self.clean_data)
        df_features = df_features.dropna()

        print(f"✓ Generated {df_features.shape[1]} features")
        print(f"✓ After dropping NaN: {len(df_features)} records")

        # Save processed data
        processed_file = self.experiment_dir / f'processed_data_{self.timestamp}.csv'
        df_features.to_csv(processed_file)
        print(f"  Saved to: {processed_file}")

        self.results['steps']['feature_engineering'] = {
            'total_features': df_features.shape[1],
            'records': len(df_features),
            'file': str(processed_file)
        }
        self.features = df_features

    def _step4_train_models(self):
        """Step 4: Train models with custom prediction horizon."""
        print("\n" + "="*80)
        print(f"STEP 4: MODEL TRAINING (Horizon: {self.prediction_horizon} days)")
        print("="*80)

        # Create target with custom horizon
        target_col = f'future_return_{self.prediction_horizon}d'
        self.features[target_col] = (
            self.features['close'].shift(-self.prediction_horizon) /
            self.features['close'] - 1
        )
        self.features = self.features.dropna()

        print(f"✓ Created target: {target_col}")
        print(f"  Formula: (close[t+{self.prediction_horizon}] / close[t]) - 1")

        # Select features
        exclude_cols = [target_col, 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in self.features.columns if col not in exclude_cols]

        X = self.features[feature_cols].fillna(0)
        y = self.features[target_col]

        # Time-series split
        test_size = 0.2
        split_idx = int(len(self.features) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"✓ Train: {len(X_train)} samples ({X_train.index[0]} to {X_train.index[-1]})")
        print(f"✓ Test:  {len(X_test)} samples ({X_test.index[0]} to {X_test.index[-1]})")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Data loss from horizon: {len(self.clean_data) - len(self.features)} days")

        # Train models
        trainer = ModelTrainer(self.settings)
        models = trainer.train_models(
            X_train=X_train.values,
            y_train=y_train.values,
            X_val=X_test.values,
            y_val=y_test.values,
            feature_names=feature_cols
        )

        print(f"✓ Trained {len(models)} models")

        # Save models
        models_dir = self.experiment_dir / 'models'
        models_dir.mkdir(exist_ok=True)
        trainer.save_models(models_dir)
        print(f"  Models saved to: {models_dir}")

        self.results['steps']['model_training'] = {
            'target_column': target_col,
            'num_features': len(feature_cols),
            'train_samples': len(X_train),
            'train_date_range': [str(X_train.index[0]), str(X_train.index[-1])],
            'test_samples': len(X_test),
            'test_date_range': [str(X_test.index[0]), str(X_test.index[-1])],
            'data_loss_from_horizon': len(self.clean_data) - len(self.features),
            'models_trained': list(models.keys()),
            'models_dir': str(models_dir)
        }

        # Store for evaluation
        self.models = models
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.target_col = target_col

    def _step5_evaluate(self):
        """Step 5: Evaluate models."""
        print("\n" + "="*80)
        print("STEP 5: MODEL EVALUATION")
        print("="*80)

        metrics_calc = PerformanceMetrics()

        print("\nIndividual Model Performance:")
        print("-" * 80)

        for name, model in self.models.items():
            predictions = model.predict(self.X_test.values)

            # Calculate metrics
            ml_metrics = metrics_calc.calculate_ml_metrics(
                self.y_test.values,
                predictions,
                model_type='regression'
            )

            # Directional accuracy
            actual_direction = np.sign(self.y_test.values)
            pred_direction = np.sign(predictions)
            directional_accuracy = np.mean(actual_direction == pred_direction)

            print(f"\n{name.upper()}:")
            print(f"  MSE:  {ml_metrics['mse']:.8f}")
            print(f"  RMSE: {ml_metrics['rmse']:.8f}")
            print(f"  MAE:  {ml_metrics['mae']:.8f}")
            print(f"  R²:   {ml_metrics['r2_score']:.6f}")
            print(f"  Directional Accuracy: {directional_accuracy:.2%}")

            # Store results
            self.results['models'][name] = {
                'mse': float(ml_metrics['mse']),
                'rmse': float(ml_metrics['rmse']),
                'mae': float(ml_metrics['mae']),
                'r2_score': float(ml_metrics['r2_score']),
                'directional_accuracy': float(directional_accuracy)
            }

        # Ensemble
        if len(self.models) > 1:
            print("\n" + "="*80)
            print("ENSEMBLE MODEL (Voting)")
            print("="*80)

            ensemble = EnsembleModel(self.models, method='voting')
            ensemble.fit(self.X_train.values, self.y_train.values)

            ensemble_pred = ensemble.predict(self.X_test.values)
            ensemble_metrics = metrics_calc.calculate_ml_metrics(
                self.y_test.values,
                ensemble_pred,
                model_type='regression'
            )

            ensemble_directional = np.mean(
                np.sign(self.y_test.values) == np.sign(ensemble_pred)
            )

            print(f"  MSE:  {ensemble_metrics['mse']:.8f}")
            print(f"  RMSE: {ensemble_metrics['rmse']:.8f}")
            print(f"  MAE:  {ensemble_metrics['mae']:.8f}")
            print(f"  R²:   {ensemble_metrics['r2_score']:.6f}")
            print(f"  Directional Accuracy: {ensemble_directional:.2%}")

            self.results['models']['ensemble'] = {
                'mse': float(ensemble_metrics['mse']),
                'rmse': float(ensemble_metrics['rmse']),
                'mae': float(ensemble_metrics['mae']),
                'r2_score': float(ensemble_metrics['r2_score']),
                'directional_accuracy': float(ensemble_directional)
            }

    def _save_results(self):
        """Save experiment results."""
        results_file = self.experiment_dir / f'results_{self.timestamp}.json'

        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✓ Results saved to: {results_file}")


async def run_experiments(horizons: list, days: int = 365, interval: str = '1d'):
    """Run multiple experiments with different prediction horizons."""

    all_results = {}

    for horizon in horizons:
        print("\n\n")
        experiment = PipelineExperiment(
            prediction_horizon=horizon,
            days=days,
            interval=interval
        )

        try:
            results = await experiment.run()
            all_results[f'{horizon}day'] = results
        except Exception as e:
            print(f"Experiment failed for {horizon}-day horizon: {e}")
            continue

    # Generate comparison report
    _generate_comparison_report(all_results)

    return all_results


def _generate_comparison_report(all_results: dict):
    """Generate comparison report across all experiments."""

    print("\n\n" + "="*80)
    print("EXPERIMENT COMPARISON REPORT")
    print("="*80)

    # Create comparison table
    comparison_data = []

    for horizon_name, results in all_results.items():
        for model_name, metrics in results['models'].items():
            comparison_data.append({
                'Horizon': horizon_name,
                'Model': model_name,
                'R²': metrics['r2_score'],
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'Directional Acc': metrics['directional_accuracy']
            })

    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)

        print("\nBest Models by Directional Accuracy:")
        print("-" * 80)
        best_by_horizon = df_comparison.loc[
            df_comparison.groupby('Horizon')['Directional Acc'].idxmax()
        ]
        print(best_by_horizon.to_string(index=False))

        print("\nBest Models by R²:")
        print("-" * 80)
        best_by_r2 = df_comparison.loc[
            df_comparison.groupby('Horizon')['R²'].idxmax()
        ]
        print(best_by_r2.to_string(index=False))

        # Save comparison
        reports_dir = Path('experiments/reports')
        reports_dir.mkdir(parents=True, exist_ok=True)

        timestamp = generate_timestamp()
        comparison_file = reports_dir / f'comparison_{timestamp}.csv'
        df_comparison.to_csv(comparison_file, index=False)

        print(f"\n✓ Comparison report saved to: {comparison_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Test ML pipeline with different prediction horizons'
    )
    parser.add_argument(
        '--horizons',
        type=int,
        nargs='+',
        help='Prediction horizons to test (e.g., 1 7 14 28)'
    )
    parser.add_argument(
        '--horizon',
        type=int,
        help='Single prediction horizon to test'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Number of days of historical data (default: 365)'
    )
    parser.add_argument(
        '--interval',
        type=str,
        default='1d',
        help='Data interval (default: 1d)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all default horizons (1, 7, 14, 28 days)'
    )

    args = parser.parse_args()

    # Determine horizons to test
    if args.all:
        horizons = [1, 7, 14, 28]
    elif args.horizon:
        horizons = [args.horizon]
    elif args.horizons:
        horizons = args.horizons
    else:
        # Default: just 1 day
        horizons = [1]

    print(f"Testing prediction horizons: {horizons}")
    print(f"Historical data: {args.days} days")
    print(f"Interval: {args.interval}")

    # Run experiments
    asyncio.run(run_experiments(horizons, args.days, args.interval))


if __name__ == '__main__':
    main()
