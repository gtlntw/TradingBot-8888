# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Run with Docker
docker-compose up

# Run tests
pytest tests/

# Run specific test
pytest tests/unit/test_data_collector.py -v

# Lint code
flake8 trading_bot/
black trading_bot/

# Type checking
mypy trading_bot/
```

## CLI Command Structure

The system provides a comprehensive CLI via `trading-bot` command with these groups:

```bash
# Data operations
trading-bot data collect --symbols BTC-USD --days 365 --interval 1d
trading-bot data preprocess --input data/raw --clean --features

# Model operations
trading-bot model train --data processed_data.csv --algorithms xgboost lightgbm
trading-bot model evaluate --data test_data.csv --models data/models --ensemble

# Backtesting
trading-bot backtest run --data historical_data.csv --strategy threshold --capital 100000

# Trading
trading-bot trade start --mode paper --capital 100000
trading-bot trade status

# System management
trading-bot status
trading-bot version
```

## Architecture Overview

This is a machine learning-powered Bitcoin trading bot with a modular, pipeline-based architecture:

### Core Pipeline Flow
1. **Data Collection** (`trading_bot/data/collector.py`) - Multi-source async data fetching from Binance, Yahoo Finance, CoinGecko
2. **Data Preprocessing** (`trading_bot/data/preprocessor.py`) - Cleaning, validation, normalization, outlier removal
3. **Feature Engineering** (`trading_bot/data/features.py`) - Technical indicators, market features, sentiment analysis
4. **Model Training** (`trading_bot/models/trainer.py`) - ML pipeline with multiple algorithms and ensembles
5. **Evaluation** (`trading_bot/evaluation/`) - Backtesting, performance metrics, baseline comparisons
6. **Trading Engine** (`trading_bot/trading/`) - Signal generation, risk management, execution

### Key Abstract Base Classes & Interfaces

**Data Layer**:
- `DataSource` (ABC): Interface for data providers (Binance, YFinance, CoinGecko)
- `FeatureCalculator` (ABC): Base for feature engineering components

**Models Layer**:
- `BaseModel` (ABC): Interface for all ML models with fit/predict/save/load
- `EnsembleMethod` (ABC): Base for ensemble strategies (voting, stacking, blending)

**Trading Layer**:
- `BaseSignalGenerator` (ABC): Interface for signal generation strategies
- Signal generators: Threshold, VolatilityAdjusted, TrendFollowing, MeanReversion, Ensemble

**Configuration System**:
- `BaseConfig` (ABC): Base configuration management
- `YAMLConfig`: YAML file + environment variable configuration
- `Settings`: Main settings class with typed property access

### Key Design Patterns

**Configuration-Driven**: All behavior controlled via `configs/default.yaml` and environment variables (`.env`)

**Async Data Pipeline**:
- `DataCollector` handles multiple sources concurrently with retry logic and rate limiting
- Each data source implements the `DataSource` interface
- Data combination supports multiple merge strategies (average, first, binance_priority)

**Feature Engineering Pipeline**:
- `FeatureEngineer` orchestrates multiple `FeatureCalculator` implementations
- `TechnicalIndicators`: 50+ technical indicators using the `ta` library
- `MarketFeatures`: Price ratios, returns, volatility measures, time-based features
- `SentimentFeatures`: Fear/Greed index, market sentiment indicators

**Model Training & Ensemble Framework**:
- `ModelTrainer` supports multiple algorithms: RandomForest, XGBoost, LightGBM, LSTM
- `EnsembleModel` combines models using voting, stacking, or blending
- All models inherit from `BaseModel` with standardized fit/predict/save/load interface
- Hyperparameter optimization with time-series cross-validation

**Trading Engine Architecture**:
- `TradingEngine` orchestrates the complete trading workflow
- `SignalGenerator` converts ML predictions to trading signals using multiple strategies
- `RiskManager` handles position sizing, stop-loss, portfolio risk controls
- Support for paper trading, live trading, and backtesting modes

**Settings Management**:
- `Settings` class (inherits from `YAMLConfig`) provides typed access to all configuration
- Environment variables override YAML config using structured key mapping
- Validation ensures required sections exist

**Logging & Error Handling**:
- `LoggerMixin` provides consistent logging across all classes
- Decorators for retry logic (`@retry`), timing (`@timing`), rate limiting (`@rate_limit`)
- Comprehensive error handling with graceful degradation

### Data Flow Architecture

```
Raw Data → DataCollector → DataPreprocessor → FeatureEngineer → ModelTrainer → TradingEngine
    ↓            ↓              ↓               ↓              ↓            ↓
External APIs → Clean OHLCV → Normalized → Feature Matrix → Predictions → Trades
```

### Risk Management Framework

The `RiskManager` class provides comprehensive risk controls:
- **Position Sizing**: Fixed, percentage, volatility-based, Kelly criterion, risk parity
- **Risk Limits**: VaR constraints, drawdown monitoring, position limits
- **Stop Management**: Fixed, volatility-based, ATR-based stops
- **Emergency Controls**: Automatic shutdown on extreme conditions

### Evaluation & Reporting System

- `PerformanceMetrics`: 40+ metrics including Sharpe ratio, drawdown, win rate
- `Backtester`: Realistic trading simulation with slippage and commissions
- `ReportGenerator`: Interactive Plotly visualizations and comprehensive reports
- Walk-forward analysis and strategy comparison capabilities

### Development Patterns

**Time Series Aware**: All data splitting and validation respects temporal order using `split_time_series()` and time-series cross-validation.

**Plugin Architecture**:
- New data sources implement the `DataSource` interface
- New models extend `BaseModel` class
- New signal strategies extend `BaseSignalGenerator`
- New feature calculators extend `FeatureCalculator`

**Async Design**: Data collection and trading execution use async/await patterns for concurrent operations

**Production Ready**:
- Docker containerization with multi-service setup
- Comprehensive logging with rotation
- Environment-specific configuration
- API key management via environment variables
