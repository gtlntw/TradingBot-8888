# Bitcoin Trading Bot

A sophisticated machine learning-powered Bitcoin trading bot with comprehensive backtesting, risk management, and automated trading capabilities.

## 🚀 Features

### Core Capabilities
- **Multi-source Data Collection**: Binance, Yahoo Finance, CoinGecko APIs with async fetching
- **Advanced Feature Engineering**: 50+ technical indicators, market features, sentiment analysis
- **Machine Learning Pipeline**: Random Forest, XGBoost, LightGBM, LSTM, and Transformer models
- **Ensemble Methods**: Sharpe-optimized voting, stacking, and blending with diversity constraints
- **Sequence Models**: LSTM/Transformer models trained on raw 30-day OHLCV sequences
- **Feature Selection**: Automatic selection of top 30 features using importance-based filtering
- **Data Quality Validation**: Comprehensive data quality checks and validation
- **Comprehensive Backtesting**: Realistic trading simulation with slippage and commissions
- **Risk Management**: Position sizing, stop-loss, portfolio VaR, drawdown controls
- **Live Trading**: Paper and live trading modes with real-time execution
- **Performance Analytics**: 40+ metrics with detailed reporting and visualizations

### Architecture Highlights
- **Modular Design**: Clean separation of data, models, trading, and evaluation
- **Async Processing**: Concurrent data fetching and order execution
- **Production Ready**: Docker support, comprehensive logging, error handling
- **Time Series Aware**: Proper temporal handling throughout the pipeline
- **Extensible**: Plugin architecture for new models and data sources

## 📋 Requirements

- Python 3.8+
- 4GB+ RAM recommended
- API keys for live data (optional)

## 🛠️ Installation

### Local Installation
```bash
# Clone repository
git clone <repository-url>
cd bitcoin-trading-bot

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

### Docker Installation
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t bitcoin-trading-bot .
docker run -it bitcoin-trading-bot
```

## 🎯 Recent Enhancements (2026-01)

The project includes **6 major enhancements** for improved prediction accuracy:

1. **Feature Selection** - Automatic selection of top 30 features from 60+ engineered features
2. **Sequence Models** - LSTM/Transformer models trained on 30-day raw OHLCV sequences
3. **Data Quality Validation** - Comprehensive checks for missing data, outliers, and quality issues
4. **Profitability Target** - Binary classification (up/down) optimized for trading decisions
5. **Transaction Cost Sensitivity** - Realistic cost modeling (0.2% default) in backtesting
6. **Standardized Evaluation** - Consistent metrics and normalization across all models

See `docs/` directory for detailed documentation on architecture and experiments.

## 🚀 Quick Start

### 1. Collect Data
```bash
# Collect Bitcoin data from multiple sources
trading-bot data collect --symbols BTC-USD --days 365 --interval 1d

# Preprocess and generate features
trading-bot data preprocess --input data/raw --clean --features
```

### 2. Train Models
```bash
# Train multiple ML models
trading-bot model train --data data/processed/processed_BTC-USD_1d_365d.csv

# Evaluate model performance
trading-bot model evaluate --data data/processed/processed_BTC-USD_1d_365d.csv --models data/models
```

### 3. Backtest Strategy
```bash
# Run backtest with trained model
trading-bot backtest run --data data/processed/processed_BTC-USD_1d_365d.csv --capital 100000

# Advanced backtest with custom strategy
trading-bot backtest run --data data/processed/processed_BTC-USD_1d_365d.csv --strategy volatility --capital 50000
```

### 4. Start Trading
```bash
# Paper trading (recommended for testing)
trading-bot trade start --mode paper --capital 100000

# Live trading (requires API keys)
trading-bot trade start --mode live --capital 10000
```

## 📊 Configuration

### Main Configuration (`configs/default.yaml`)
```yaml
# Data sources and symbols
data:
  sources: [binance, yfinance, coingecko]
  symbols: [BTC-USD, BTC-USDT]
  intervals: [1h, 4h, 1d]

# ML models and training
models:
  algorithms: [random_forest, xgboost, lightgbm, lstm]
  ensemble:
    method: voting
    weights: auto

# Trading and risk management
trading:
  strategy:
    signal_threshold: 0.6
    position_sizing: volatility
    stop_loss: 0.05
    take_profit: 0.10
  risk_management:
    max_drawdown: 0.20
    position_limit: 0.30
    var_limit: 0.05
```

### Environment Variables (`.env`)
```bash
# API Keys
BINANCE_API_KEY=your_key_here
BINANCE_SECRET_KEY=your_secret_here
COINGECKO_API_KEY=your_key_here

# Database
DATABASE_URL=sqlite:///data/trading_bot.db

# Trading
TRADING_MODE=paper
INITIAL_CAPITAL=100000
```

## 🔧 CLI Commands

### Data Management
```bash
# Collect historical data
trading-bot data collect --symbols BTC-USD ETH-USD --days 365

# Preprocess data with cleaning and feature generation
trading-bot data preprocess --input data/raw --clean --features
```

### Model Training
```bash
# Train models with cross-validation
trading-bot model train --data processed_data.csv --algorithms xgboost lightgbm

# Evaluate model performance
trading-bot model evaluate --data test_data.csv --models data/models --ensemble
```

### Backtesting
```bash
# Run comprehensive backtest
trading-bot backtest run --data historical_data.csv --strategy threshold

# Advanced backtesting with custom parameters
trading-bot backtest run --data data.csv --strategy ensemble --capital 50000
```

### Trading
```bash
# Start paper trading
trading-bot trade start --mode paper

# Check trading status
trading-bot trade status

# Start live trading (production)
trading-bot trade start --mode live --model best_model.pkl
```

### System Management
```bash
# Check system status
trading-bot status

# Show version information
trading-bot version
```

## 📈 Performance Metrics

The system tracks comprehensive performance metrics:

### Return Metrics
- Total Return, Annualized Return, Sharpe Ratio
- Sortino Ratio, Calmar Ratio, Omega Ratio

### Risk Metrics
- Maximum Drawdown, Value at Risk (VaR)
- Conditional VaR, Volatility, Beta

### Trading Metrics
- Win Rate, Profit Factor, Average Trade
- Maximum Consecutive Wins/Losses

## 🔄 Architecture Overview

```
Data Sources → Data Collector → Preprocessor → Feature Engineer
     ↓              ↓              ↓              ↓
External APIs → Raw OHLCV → Clean Data → Feature Matrix
                                              ↓
Risk Manager ← Trading Engine ← Signal Generator ← Model Trainer
     ↓              ↓              ↓              ↓
Risk Controls → Trade Execution → Trading Signals → ML Predictions
```

### Key Components

- **Data Pipeline**: Multi-source async data collection with validation
- **Feature Engineering**: Technical indicators, market features, sentiment
- **ML Framework**: Multiple algorithms with ensemble methods
- **Trading Engine**: Signal generation, risk management, execution
- **Evaluation Suite**: Backtesting, performance metrics, reporting

## 🛡️ Risk Management

### Position Sizing Methods
- Fixed percentage allocation
- Volatility-based sizing
- Kelly criterion optimization
- Risk parity allocation

### Risk Controls
- Maximum position size limits
- Portfolio VaR constraints
- Drawdown monitoring
- Emergency stop mechanisms

### Stop Loss & Take Profit
- Fixed percentage stops
- Volatility-based dynamic stops
- ATR-based stops
- Trailing stops

## 📊 Reporting & Visualization

### Automated Reports
- Comprehensive performance analysis
- Interactive Plotly charts
- Monthly returns heatmaps
- Drawdown analysis
- Trade analysis

### Key Visualizations
- Equity curve with benchmark comparison
- Rolling performance metrics
- Returns distribution analysis
- Risk-return scatter plots

## 🐳 Docker Deployment

### Development
```bash
# Start development environment
docker-compose up

# Access Jupyter notebooks
open http://localhost:8888
```

### Production
```bash
# Build production image
docker build -t trading-bot:prod .

# Run with custom configuration
docker run -v $(pwd)/configs:/app/configs trading-bot:prod
```

## 📝 Development

### Project Structure
```
trading_bot/
├── data/           # Data collection and preprocessing
├── models/         # ML training and ensemble methods
├── trading/        # Signal generation and execution
├── evaluation/     # Backtesting and performance analysis
├── utils/          # Utilities and helpers
└── config/         # Configuration management
```

### Adding New Features

1. **New Data Source**: Implement `DataSource` interface in `data/collector.py`
2. **New Model**: Extend `BaseModel` class in `models/trainer.py`
3. **New Strategy**: Add signal generator in `trading/signals.py`
4. **New Metric**: Extend `PerformanceMetrics` in `evaluation/metrics.py`

### Testing

#### Enhanced Walk-Forward Testing (Recommended)

**Cross-Horizon Fairness (2026-01-25):** All multi-horizon tests now ensure fair comparison:
- ✅ All horizons test the SAME calendar period
- ✅ All horizons use the SAME test window size
- ✅ Buy-and-hold returns are IDENTICAL across all horizons
- ✅ Cross-horizon comparisons are VALID

**Production Testing Commands:**
```bash
# Full 10-year test with 10 windows (RECOMMENDED)
python scripts/run_all_horizons_walk_forward.py \
    --days 3650 --train-window 730 --test-window 150 --step-size 300 \
    2>&1 | tee results_full_10windows.log

# Quick verification without sequence models (~45-60 min)
python scripts/run_all_horizons_walk_forward.py \
    --days 3650 --train-window 730 --test-window 150 --step-size 300 \
    --no-sequences \
    2>&1 | tee results_quick_verification.log

# Single horizon test (14-day example)
python scripts/walk_forward_test_enhanced.py \
    --horizon 14 --days 3650 --train-window 730 --test-window 150 --step-size 300 \
    2>&1 | tee results_14day.log

# Quick development test (3 years)
python scripts/run_all_horizons_walk_forward.py --quick
```

**What Gets Tested:**
- 5 traditional models: Random Forest, XGBoost, LightGBM, LSTM, Transformer
- 2 sequence models: lstm_60day, transformer_60day (with --sequences)
- 1 ensemble: Sharpe-optimized voting with diversity constraints
- Multiple prediction horizons: 1, 7, 14, 28, 60 days
- 10 walk-forward windows with proper time-series validation

**Expected Runtime:**
- Quick verification (10 windows, no sequences): ~45-60 minutes
- Full test (10 windows, with sequences): ~10-15 hours

#### Unit & Integration Tests
```bash
# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run with coverage
pytest --cov=trading_bot tests/
```

**Note**: The enhanced walk-forward testing framework includes all 6 new features, cross-horizon fairness fixes, normalization fixes, and proper time-series validation. See `CLAUDE.md` and `scripts/verify_cross_horizon_fix.py` for detailed testing documentation and verification.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk and never trade with money you cannot afford to lose.

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📞 Support

- **Documentation**: See `docs/` directory for architecture details, experiment results, and implementation guides
  - `docs/NORMALIZATION_ARCHITECTURE.md` - Normalization approach and rationale
  - `docs/SEQUENCE_MODEL_FIX.md` - Sequence model improvements
  - `docs/ENSEMBLE_AND_SEQUENCE_ANALYSIS.md` - Model performance analysis
  - `docs/EXPERIMENT_STATUS.md` - Current experiment results
- **Development Guide**: See `CLAUDE.md` for detailed development instructions
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

---

**Built with ❤️ for the crypto trading community**