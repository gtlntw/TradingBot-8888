"""
Command-line interface for Bitcoin trading bot.
Provides commands for data collection, training, backtesting, and live trading.
"""

import click
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import sys
import os

from trading_bot.config.settings import Settings
from trading_bot.data.collector import DataCollector
from trading_bot.data.preprocessor import DataPreprocessor
from trading_bot.data.features import FeatureEngineer
from trading_bot.models.trainer import ModelTrainer
from trading_bot.models.ensemble import EnsembleModel
from trading_bot.evaluation.backtester import Backtester
from trading_bot.evaluation.metrics import PerformanceMetrics
from trading_bot.evaluation.reporter import ReportGenerator
from trading_bot.trading.engine import TradingEngine, TradingMode
from trading_bot.trading.signals import SignalGenerator
from trading_bot.utils.logger import get_logger, setup_logging
from trading_bot.utils.helpers import setup_directories, save_json, load_json, generate_timestamp


# Global settings
settings = None
logger = None


def initialize_app(config_file: str = None, log_level: str = "INFO"):
    """Initialize application with settings and logging."""
    global settings, logger

    # Initialize settings
    settings = Settings(config_file)

    # Setup logging
    setup_logging(settings.get('logging', {}))
    logger = get_logger(__name__)

    # Setup directories
    setup_directories([
        'data/raw', 'data/processed', 'data/models',
        'logs', 'reports', 'notebooks'
    ])

    logger.info("Bitcoin Trading Bot initialized")
    return settings, logger


@click.group()
@click.option('--config', '-c', default=None, help='Configuration file path')
@click.option('--log-level', default='INFO', help='Logging level')
@click.pass_context
def cli(ctx, config, log_level):
    """Bitcoin Trading Bot - ML-powered cryptocurrency trading system."""
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['log_level'] = log_level

    # Initialize app
    try:
        ctx.obj['settings'], ctx.obj['logger'] = initialize_app(config, log_level)
    except Exception as e:
        click.echo(f"Error initializing application: {e}")
        sys.exit(1)


@cli.group()
def data():
    """Data collection and preprocessing commands."""
    pass


@data.command()
@click.option('--symbols', '-s', multiple=True, help='Symbols to collect (e.g., BTC-USD)')
@click.option('--days', '-d', default=365, help='Number of days to collect')
@click.option('--interval', '-i', default='1d', help='Data interval (1h, 4h, 1d)')
@click.option('--sources', multiple=True, help='Data sources (binance, yfinance, coingecko)')
@click.option('--output', '-o', default='data/raw', help='Output directory')
@click.pass_context
def collect(ctx, symbols, days, interval, sources, output):
    """Collect historical market data."""
    settings = ctx.obj['settings']
    logger = ctx.obj['logger']

    # Use config defaults if not specified
    symbols = symbols or settings.symbols
    sources = sources or settings.data_sources

    click.echo(f"Collecting data for symbols: {list(symbols)}")
    click.echo(f"Sources: {list(sources)}, Interval: {interval}, Days: {days}")

    async def collect_data():
        collector = DataCollector(settings)

        try:
            for symbol in symbols:
                click.echo(f"\nCollecting data for {symbol}...")

                data = await collector.fetch_data(
                    symbol=symbol,
                    timeframe=interval,
                    limit=days,
                    sources=list(sources)
                )

                if data:
                    # Combine data from sources
                    combined_data = collector.combine_data(data, method='average')

                    # Save to file
                    output_path = Path(output)
                    output_path.mkdir(parents=True, exist_ok=True)

                    filename = f"{symbol.replace('/', '-')}_{interval}_{days}d_{generate_timestamp()}.csv"
                    filepath = output_path / filename

                    combined_data.to_csv(filepath)
                    click.echo(f"Saved {len(combined_data)} records to {filepath}")
                else:
                    click.echo(f"No data collected for {symbol}")

        except Exception as e:
            logger.error(f"Error collecting data: {e}")
            click.echo(f"Error: {e}")
        finally:
            await collector.close()

    asyncio.run(collect_data())


@data.command()
@click.option('--input', '-i', required=True, help='Input data file or directory')
@click.option('--output', '-o', default='data/processed', help='Output directory')
@click.option('--clean', is_flag=True, help='Apply data cleaning')
@click.option('--features', is_flag=True, help='Generate features')
@click.pass_context
def preprocess(ctx, input, output, clean, features):
    """Preprocess and clean market data."""
    settings = ctx.obj['settings']
    logger = ctx.obj['logger']

    input_path = Path(input)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        if input_path.is_file():
            files = [input_path]
        else:
            files = list(input_path.glob('*.csv'))

        if not files:
            click.echo("No CSV files found in input path")
            return

        preprocessor = DataPreprocessor(settings._config_data)
        feature_engineer = FeatureEngineer(settings._config_data) if features else None

        for file_path in files:
            click.echo(f"Processing {file_path.name}...")

            # Load data
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)

            # Clean data
            if clean:
                df = preprocessor.clean_data(df, remove_outliers=True)
                click.echo(f"  Cleaned data: {len(df)} records")

            # Generate features
            if features and feature_engineer:
                df = feature_engineer.create_features(df)
                click.echo(f"  Generated features: {df.shape[1]} columns")

            # Save processed data
            output_file = output_path / f"processed_{file_path.name}"
            df.to_csv(output_file)
            click.echo(f"  Saved to {output_file}")

    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        click.echo(f"Error: {e}")


@cli.group()
def model():
    """Model training and evaluation commands."""
    pass


@model.command()
@click.option('--data', '-d', required=True, help='Training data file')
@click.option('--target', '-t', default='future_return', help='Target column name')
@click.option('--algorithms', '-a', multiple=True, help='Algorithms to train')
@click.option('--test-size', default=0.2, help='Test set size')
@click.option('--output', '-o', default='data/models', help='Output directory')
@click.pass_context
def train(ctx, data, target, algorithms, test_size, output):
    """Train ML models on market data."""
    settings = ctx.obj['settings']
    logger = ctx.obj['logger']

    # Use config defaults if not specified
    algorithms = algorithms or settings.algorithms

    click.echo(f"Training models: {list(algorithms)}")
    click.echo(f"Data: {data}, Target: {target}")

    try:
        # Load data
        df = pd.read_csv(data, index_col=0, parse_dates=True)
        click.echo(f"Loaded data: {df.shape}")

        # Create target if it doesn't exist
        if target not in df.columns:
            # Create future return target
            df[target] = df['close'].shift(-1) / df['close'] - 1
            df = df.dropna()
            click.echo(f"Created target column: {target}")

        # Prepare features and target
        feature_cols = [col for col in df.columns if col != target and not col.startswith('close')]
        X = df[feature_cols].fillna(0)
        y = df[target]

        # Split data
        split_idx = int(len(df) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        click.echo(f"Train: {len(X_train)}, Test: {len(X_test)}")

        # Train models
        trainer = ModelTrainer(settings)
        models = trainer.train_models(
            X_train=X_train.values,
            y_train=y_train.values,
            X_val=X_test.values,
            y_val=y_test.values,
            feature_names=feature_cols
        )

        # Save models
        output_path = Path(output)
        trainer.save_models(output_path)

        # Print results
        click.echo("\nTraining Results:")
        for name, model in models.items():
            predictions = model.predict(X_test.values)
            mse = np.mean((predictions - y_test.values) ** 2)
            click.echo(f"  {name}: MSE = {mse:.6f}")

    except Exception as e:
        logger.error(f"Error training models: {e}")
        click.echo(f"Error: {e}")


@model.command()
@click.option('--data', '-d', required=True, help='Test data file')
@click.option('--models', '-m', required=True, help='Models directory')
@click.option('--timestamp', '-t', help='Model timestamp to load')
@click.option('--ensemble', is_flag=True, help='Create ensemble model')
@click.option('--output', '-o', default='reports', help='Output directory')
@click.pass_context
def evaluate(ctx, data, models, timestamp, ensemble, output):
    """Evaluate trained models."""
    settings = ctx.obj['settings']
    logger = ctx.obj['logger']

    try:
        # Load test data
        df = pd.read_csv(data, index_col=0, parse_dates=True)
        click.echo(f"Loaded test data: {df.shape}")

        # Load models
        models_path = Path(models)
        if not timestamp:
            # Find latest timestamp
            model_files = list(models_path.glob('*_*.pkl'))
            if not model_files:
                click.echo("No model files found")
                return
            timestamp = model_files[0].stem.split('_')[-1]

        trainer = ModelTrainer(settings)
        loaded_models = trainer.load_models(models_path, timestamp)

        if not loaded_models:
            click.echo("No models loaded")
            return

        click.echo(f"Loaded {len(loaded_models)} models")

        # Prepare data
        target_col = 'future_return'
        if target_col not in df.columns:
            df[target_col] = df['close'].shift(-1) / df['close'] - 1
            df = df.dropna()

        feature_cols = [col for col in df.columns if col != target_col and not col.startswith('close')]
        X = df[feature_cols].fillna(0).values
        y = df[target_col].values

        # Evaluate individual models
        click.echo("\nModel Evaluation:")
        metrics_calc = PerformanceMetrics()

        for name, model in loaded_models.items():
            predictions = model.predict(X)
            metrics = metrics_calc.calculate_ml_metrics(y, predictions, model_type='regression')

            click.echo(f"\n{name}:")
            click.echo(f"  MSE: {metrics['mse']:.6f}")
            click.echo(f"  MAE: {metrics['mae']:.6f}")
            click.echo(f"  R²: {metrics['r2_score']:.4f}")

        # Create ensemble if requested
        if ensemble and len(loaded_models) > 1:
            click.echo("\nCreating ensemble model...")
            ensemble_model = EnsembleModel(loaded_models, method='voting')
            ensemble_model.fit(X, y)

            ensemble_pred = ensemble_model.predict(X)
            ensemble_metrics = metrics_calc.calculate_ml_metrics(y, ensemble_pred, model_type='regression')

            click.echo(f"\nEnsemble Model:")
            click.echo(f"  MSE: {ensemble_metrics['mse']:.6f}")
            click.echo(f"  MAE: {ensemble_metrics['mae']:.6f}")
            click.echo(f"  R²: {ensemble_metrics['r2_score']:.4f}")

    except Exception as e:
        logger.error(f"Error evaluating models: {e}")
        click.echo(f"Error: {e}")


@cli.group()
def backtest():
    """Backtesting commands."""
    pass


@backtest.command()
@click.option('--data', '-d', required=True, help='Historical data file')
@click.option('--model', '-m', help='Model file to use for predictions')
@click.option('--strategy', '-s', default='threshold', help='Signal generation strategy')
@click.option('--capital', '-c', default=100000, help='Initial capital')
@click.option('--output', '-o', default='reports', help='Output directory')
@click.pass_context
def run(ctx, data, model, strategy, capital, output):
    """Run backtest on historical data."""
    settings = ctx.obj['settings']
    logger = ctx.obj['logger']

    try:
        # Load data
        df = pd.read_csv(data, index_col=0, parse_dates=True)
        click.echo(f"Loaded data: {df.shape} from {df.index[0]} to {df.index[-1]}")

        # Generate signals
        signal_generator = SignalGenerator()

        if model:
            # Load model and generate predictions
            click.echo("Loading model for predictions...")
            # This would need proper model loading logic
            predictions = np.random.randn(len(df)) * 0.1  # Placeholder
        else:
            # Use simple strategy
            returns = df['close'].pct_change()
            predictions = returns.rolling(5).mean()  # Simple momentum

        # Generate trading signals
        signals = signal_generator.generate_signals(
            predictions=predictions,
            data=df,
            strategy=strategy
        )

        click.echo(f"Generated {(signals != 0).sum()} trading signals")

        # Run backtest
        backtester = Backtester(
            initial_capital=capital,
            commission=settings.get('trading.execution.commission', 0.001),
            slippage=settings.get('trading.execution.slippage', 0.001)
        )

        results = backtester.run_backtest(
            data=df,
            signals=signals,
            strategy_name=f"{strategy}_strategy",
            position_size=0.1,
            stop_loss=settings.stop_loss,
            take_profit=settings.take_profit
        )

        # Display results
        metrics = results.get('metrics', {})
        click.echo(f"\nBacktest Results:")
        click.echo(f"  Total Return: {metrics.get('total_return', 0):.2%}")
        click.echo(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        click.echo(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        click.echo(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
        click.echo(f"  Total Trades: {metrics.get('total_trades', 0)}")

        # Generate report
        output_path = Path(output)
        reporter = ReportGenerator(output_path)
        report = reporter.generate_full_report(results)

        click.echo(f"\nDetailed report saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        click.echo(f"Error: {e}")


@cli.group()
def trade():
    """Live and paper trading commands."""
    pass


@trade.command()
@click.option('--model', '-m', help='Model file for predictions')
@click.option('--capital', '-c', default=100000, help='Initial capital')
@click.option('--mode', default='paper', type=click.Choice(['paper', 'live']), help='Trading mode')
@click.pass_context
def start(ctx, model, capital, mode):
    """Start trading engine."""
    settings = ctx.obj['settings']
    logger = ctx.obj['logger']

    click.echo(f"Starting trading engine in {mode} mode")
    click.echo(f"Initial capital: ${capital:,.2f}")

    try:
        # Load model if provided
        trained_model = None
        if model:
            click.echo(f"Loading model: {model}")
            # Would need proper model loading
            # trained_model = load_model(model)

        # Create trading engine
        trading_mode = TradingMode.PAPER if mode == 'paper' else TradingMode.LIVE
        engine = TradingEngine(
            settings=settings,
            model=trained_model,
            mode=trading_mode
        )

        # Set initial capital
        engine.state.portfolio_value = capital
        engine.state.cash_balance = capital

        async def run_trading():
            try:
                await engine.start_trading()
            except KeyboardInterrupt:
                click.echo("\nShutting down trading engine...")
                await engine.stop_trading()
            except Exception as e:
                logger.error(f"Trading engine error: {e}")
                click.echo(f"Error: {e}")

        # Run trading engine
        asyncio.run(run_trading())

    except Exception as e:
        logger.error(f"Error starting trading: {e}")
        click.echo(f"Error: {e}")


@trade.command()
@click.pass_context
def status(ctx):
    """Show trading status and portfolio."""
    # This would need to connect to running trading engine
    click.echo("Trading status: Not implemented yet")
    click.echo("This would show current portfolio status, positions, and performance")


@cli.command()
@click.option('--data-dir', default='data', help='Data directory to check')
@click.option('--models-dir', default='data/models', help='Models directory to check')
@click.pass_context
def status(ctx, data_dir, models_dir):
    """Show system status and available data/models."""
    settings = ctx.obj['settings']

    click.echo("Bitcoin Trading Bot - System Status")
    click.echo("=" * 50)

    # Configuration status
    click.echo(f"Configuration: {settings.config_file or 'default'}")
    click.echo(f"Data sources: {', '.join(settings.data_sources)}")
    click.echo(f"Symbols: {', '.join(settings.symbols)}")
    click.echo(f"Algorithms: {', '.join(settings.algorithms)}")

    # Data status
    data_path = Path(data_dir)
    if data_path.exists():
        raw_files = list((data_path / 'raw').glob('*.csv')) if (data_path / 'raw').exists() else []
        processed_files = list((data_path / 'processed').glob('*.csv')) if (data_path / 'processed').exists() else []

        click.echo(f"\nData Status:")
        click.echo(f"  Raw files: {len(raw_files)}")
        click.echo(f"  Processed files: {len(processed_files)}")

    # Models status
    models_path = Path(models_dir)
    if models_path.exists():
        model_files = list(models_path.glob('*.pkl'))
        click.echo(f"\nModels: {len(model_files)} saved models")

    # Recent activity
    logs_path = Path('logs')
    if logs_path.exists():
        log_files = list(logs_path.glob('*.log'))
        if log_files:
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            click.echo(f"\nLatest log: {latest_log.name}")


@cli.command()
def version():
    """Show version information."""
    import trading_bot
    click.echo(f"Bitcoin Trading Bot v{trading_bot.__version__}")
    click.echo("ML-powered cryptocurrency trading system")


def main():
    """Main entry point for CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()