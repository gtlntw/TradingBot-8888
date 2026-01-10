# Pipeline Experiments

This directory contains development experiments for testing the ML pipeline with different configurations.

## Directory Structure

```
experiments/
├── results/          # Experiment results (gitignored)
│   ├── 1day/        # 1-day prediction horizon experiments
│   ├── 7day/        # 7-day prediction horizon experiments
│   ├── 14day/       # 14-day prediction horizon experiments
│   └── 28day/       # 28-day prediction horizon experiments
├── configs/         # Experiment configurations
└── reports/         # Comparison reports
```

## Usage

### Run Single Experiment

```bash
# Test 7-day prediction horizon
python scripts/test_pipeline.py --horizon 7

# Test with more historical data
python scripts/test_pipeline.py --horizon 14 --days 730
```

### Run Multiple Experiments

```bash
# Test multiple horizons
python scripts/test_pipeline.py --horizons 1 7 14 28

# Test all default horizons
python scripts/test_pipeline.py --all
```

### Advanced Usage

```bash
# Test with hourly data
python scripts/test_pipeline.py --horizon 24 --interval 1h --days 90

# Custom horizon combination
python scripts/test_pipeline.py --horizons 3 5 10 21 --days 500
```

## Experiment Output

Each experiment creates:
- `raw_data_*.csv` - Collected market data
- `processed_data_*.csv` - Preprocessed data with features
- `models/` - Trained model files
- `results_*.json` - Complete experiment results and metrics

## Comparison Reports

After running multiple experiments, a comparison report is generated in `experiments/reports/` showing:
- Best model for each prediction horizon
- Performance metrics comparison
- Directional accuracy trends

## Understanding Results

### Metrics

- **R²**: Coefficient of determination (closer to 1 is better, negative means worse than mean)
- **RMSE**: Root mean squared error (lower is better)
- **MAE**: Mean absolute error (lower is better)
- **Directional Accuracy**: % of times the model correctly predicts up/down (>50% is better than random)

### Interpretation

- Longer horizons typically have lower directional accuracy but may capture trends better
- Shorter horizons are noisier but more predictable with technical indicators
- Directional accuracy is most important for trading profitability
- R² can be negative for financial time series - focus on directional accuracy

## Best Practices

1. **Start with default horizons** (1, 7, 14, 28 days)
2. **Use sufficient historical data** (at least 365 days, prefer 730+)
3. **Compare across horizons** to find the sweet spot for your strategy
4. **Consider your trading style**:
   - Day trading: 1-3 day horizons
   - Swing trading: 7-14 day horizons
   - Position trading: 21-60 day horizons
