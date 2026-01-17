# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Setup

### Quick Installation

```bash
# Install dependencies with workaround for build issues
pip install multitasking --use-pep517  # Fixes common build error
pip install ta --use-pep517            # Fixes ta library build error
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

**Note**: `pandas-ta` is commented out in requirements.txt due to Python 3.11 compatibility issues. The `ta` library provides sufficient technical indicators.

### Alternative: Single Command Installation

```bash
# Install all core dependencies directly
pip install multitasking --use-pep517 && \
pip install numpy pandas scipy scikit-learn xgboost lightgbm yfinance ta \
PyYAML python-dotenv aiohttp requests tqdm ccxt matplotlib seaborn plotly tensorflow
```

### Docker Setup

```bash
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

### Pipeline Defaults

**Prediction Horizon**: 1 period ahead (hardcoded via `shift(-1)`)
- For daily data (1d): Predicts next day's return
- For hourly data (1h): Predicts next hour's return
- Target formula: `(next_close / current_close) - 1`
- Location: `trading_bot/cli.py` lines 225, 306

**Models Trained by Default**:
1. Random Forest (tree-based ensemble)
2. XGBoost (gradient boosting)
3. LightGBM (fast gradient boosting)
4. LSTM (deep learning, sequence model)
5. Transformer (advanced deep learning)

**Data Split**: 80% train / 20% test (time-series split, respects temporal order)

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
- `ModelTrainer` supports multiple algorithms: RandomForest, XGBoost, LightGBM, LSTM, Transformer
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

---

## Testing Scripts

### Current Testing Framework (2026-01-15)

**Recommended:** Use `scripts/walk_forward_test_enhanced.py` for all testing - it includes all 6 new features and proper fixes.

### Testing New Features (2026-01-12 - UPDATED 2026-01-15)

The project now includes **6 major enhancements** integrated into an enhanced walk-forward testing framework:

#### New Features Overview

1. **Feature Selection** - Automatic selection of best features (default: 30)
2. **Sequence Models** - 30-day LSTM/Transformer models on raw OHLCV
3. **Data Quality Validation** - Comprehensive data quality checks
4. **Profitability Target** - Binary classification (up/down vs raw returns)
5. **Transaction Cost Sensitivity** - Realistic cost modeling (0.2% default)
6. **Standardized Evaluation** - Consistent metrics across all models

#### Enhanced Walk-Forward Testing

**Primary Script:** `scripts/walk_forward_test_enhanced.py`

This script integrates all 6 new features into a comprehensive testing framework with proper time-series validation.

**Basic Usage:**
```bash
# Quick test (1-day horizon, 12 windows)
python scripts/walk_forward_test_enhanced.py --horizon 1 --quick

# Full test (1-day horizon, all data)
python scripts/walk_forward_test_enhanced.py --horizon 1 --days 2190

# Multiple horizons
python scripts/walk_forward_test_enhanced.py --horizon 7 --days 2190
```

**Command-Line Options:**
```bash
--horizon N         # Prediction horizon in days (1, 7, 14, 28, 60)
--days N           # Total days of historical data to use (default: 2190)
--mode MODE        # 'expanding' (default) or 'rolling' windows
--quick            # Quick test mode (12 windows instead of full split)
--no-sequences     # Skip sequence model training (faster)
```

**Example Outputs:**
```
Configuration:
  Mode: EXPANDING
  Train Window: 365 days
  Test Window: 60 days
  Step Size: 60 days

Models Trained Per Window:
  - 5 Traditional: Random Forest, XGBoost, LightGBM, LSTM, Transformer
  - 1 Ensemble: Sharpe-optimized voting (min_weight=0.15)
  - 2 Sequence: lstm_60day, transformer_60day (30-day lookback)

Results:
  ✓ Model Performance Summary (per window)
  ✓ Aggregated Results (across all windows)
  ✓ Buy & Hold Baseline Comparison
```

#### Test Configuration Details

**Data Split:**
- **Expanding windows** (default): Train size grows each window
- **Rolling windows**: Fixed train size, sliding forward
- **Time-series safe**: No future data leakage

**Feature Selection:**
- Starts with 60+ engineered features
- Removes correlated features (>0.95 correlation)
- Selects top 30 by Random Forest importance
- Applied per-window to prevent leakage

**Normalization:**
- **Traditional models**: StandardScaler on 30 engineered features
- **Sequence models**: StandardScaler on 5 raw OHLCV features
- **Per-window fitting**: Each window fits its own scaler
- See `NORMALIZATION_ARCHITECTURE.md` for rationale

**Target Creation:**
- **Formula**: `profitable_trade = (future_return > 0)`
- **Binary classification**: 1 if price goes up, 0 if down
- **Transaction costs**: Applied only in backtesting (0.1% commission + 0.1% slippage)
- **NO cost in target**: Prevents double-counting issue

**Ensemble Method:**
- **Type**: Sharpe-optimized voting ensemble
- **Optimization**: Maximizes Sharpe ratio on 20% validation split
- **Diversity constraint**: `min_weight=0.15` (prevents model exclusion)
- **Models combined**: 5 traditional models (RF, XGB, LGB, LSTM, Transformer)

#### Important Fixes Applied

**1. Sequence Model Normalization Fix** (2026-01-15)
- **Issue**: Sequence models produced degenerate predictions (all constant)
- **Root cause**: Missing normalization of OHLCV data
- **Fix**: Added StandardScaler normalization (same as traditional models)
- **Impact**: Sequence models now produce diverse predictions
- **Details**: See `SEQUENCE_MODEL_FIX.md`

**2. Transaction Cost Accounting**
- **Issue**: Costs were applied twice (in target AND backtesting)
- **Fix**: Costs only in backtesting, target is `return > 0`
- **Impact**: More realistic returns, better class balance (50% vs 45%)

**3. Ensemble Minimum Weight**
- **Issue**: Optimizer could exclude models (0.001 weight)
- **Fix**: Added `min_weight=0.15` constraint
- **Impact**: All models contribute meaningfully to ensemble

#### Sequence Models Architecture

**Traditional Models:**
- Input: 30 engineered features (technical indicators, market features)
- Architecture: Single timestep prediction
- Examples: Random Forest, XGBoost, LightGBM

**Sequence Models:**
- Input: 30-day × 5 OHLCV sequences (raw data, no engineering)
- Architecture: Bidirectional LSTM/Transformer with attention
- Models: `lstm_60day`, `transformer_60day`
- Purpose: Compare raw sequences vs engineered features

**Key Difference:**
```
Traditional: [RSI, MACD, SMA, ...] (30 features, 1 timestep) → Prediction
Sequence:    [[O,H,L,C,V], [O,H,L,C,V], ...] (5 features, 30 timesteps) → Prediction
```

#### Testing Multiple Horizons

**Recommended:** Use `scripts/run_all_horizons_walk_forward.py` to test all horizons automatically.

This script runs the enhanced walk-forward test across all 5 horizons (1, 7, 14, 28, 60 days) with all 6 new features included.

```bash
# Run all 5 horizons with new features
python scripts/run_all_horizons_walk_forward.py

# Quick mode (faster, 12 windows per horizon)
python scripts/run_all_horizons_walk_forward.py --quick

# Skip sequence models for speed
python scripts/run_all_horizons_walk_forward.py --quick --no-sequences

# Use rolling windows instead of expanding
python scripts/run_all_horizons_walk_forward.py --mode rolling
```

**What it does:**
- Runs `walk_forward_test_enhanced.py` for each horizon (1, 7, 14, 28, 60 days)
- Includes ALL 6 new features (feature selection, sequence models, data quality, etc.)
- Includes ALL fixes (normalization, transaction costs, ensemble min_weight)
- Generates comparison report across all horizons
- Saves results to `experiments/walk_forward_enhanced/`

**Expected Runtime:**
- Quick mode (12 windows × 5 horizons): ~2.5-4 hours total
- Full mode (all windows × 5 horizons): ~10-20 hours total
- Sequence models add ~50% overhead per horizon

#### Output Files & Logs

All tests create comprehensive logs and can save results:

```python
# Results are printed to stdout and can be redirected
python scripts/walk_forward_test_enhanced.py --horizon 1 --quick > results_1day.log 2>&1

# Key sections in output:
# - Data quality checks
# - Feature selection summary
# - Per-window results (8 models × 12 windows)
# - Aggregated performance summary
# - Buy & Hold baseline comparison
```

#### Debugging & Validation Scripts

Several debugging scripts are available:

- `scripts/debug_simple.py` - Quick LSTM prediction test
- `scripts/test_normalization_fix.py` - Validates normalization fix
- `scripts/test_standardscaler.py` - Tests StandardScaler consistency

#### Architecture Documentation

For deeper understanding, see the `docs/` directory:
- **`docs/NORMALIZATION_ARCHITECTURE.md`** - Why normalization is in test script
- **`docs/SEQUENCE_MODEL_FIX.md`** - Degenerate model investigation & fix
- **`docs/COST_ACCOUNTING_ISSUE.md`** - Transaction cost double-counting issue
- **`docs/COST_FIX_COMPARISON.md`** - Before/after comparison of cost fix
- **`docs/ENSEMBLE_AND_SEQUENCE_ANALYSIS.md`** - Analysis of ensemble and sequence models
- **`docs/BOTH_FIXES_VERIFIED.md`** - Verification of normalization and cost fixes
- **`docs/TRANSFORMER_FIX_SUMMARY.md`** - Transformer model improvements
- **`docs/EXPERIMENT_STATUS.md`** - Current experiment status and results
- **`CLAUDE.md`** - This file (you're reading it!)

---

### Legacy Testing Scripts (Pre-2026 Features)

The following scripts predate the 6 new features and use older architectures. They are kept for backward compatibility but are **NOT recommended** for new work:

**`scripts/test_pipeline.py`** - Traditional pipeline testing
- Tests different prediction horizons (1, 7, 14, 28 days)
- Full pipeline: data collection → preprocessing → feature engineering → training → evaluation → backtesting
- ⚠️ Does NOT include: feature selection, sequence models, data quality checks, or fixes
- Usage: `python scripts/test_pipeline.py --horizon 7 --days 730`

**`scripts/walk_forward_test.py`** - Basic walk-forward validation (OLD)
- Tests model robustness with expanding/rolling windows
- ⚠️ Does NOT include new features or normalization fixes
- ⚠️ REPLACED by `walk_forward_test_enhanced.py`
- Usage: `python scripts/walk_forward_test.py` (not recommended)

**Migration Note:** If you have existing results from these legacy scripts, they are NOT directly comparable to results from the enhanced testing framework due to:
- Different normalization approach (no StandardScaler for sequences)
- Transaction cost accounting changes (double-counting fixed)
- Addition of sequence models (lstm_60day, transformer_60day)
- Feature selection differences (30 features vs all features)
- Ensemble min_weight constraint (not present in old version)
