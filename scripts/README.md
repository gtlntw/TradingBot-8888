# Development Scripts

Development and testing scripts for the trading bot.

## Available Scripts

### `test_pipeline.py`

Pipeline testing script for experimenting with different prediction horizons.

**Features**:
- Test multiple prediction horizons in parallel
- Automated data collection, preprocessing, feature engineering, model training, and evaluation
- Generates comparison reports across experiments
- Organized output structure

**Usage**:

```bash
# Test single horizon
python scripts/test_pipeline.py --horizon 7

# Test multiple horizons
python scripts/test_pipeline.py --horizons 1 7 14 28

# Test all default horizons (1, 7, 14, 28 days)
python scripts/test_pipeline.py --all

# Test with custom data period
python scripts/test_pipeline.py --horizon 14 --days 730

# Test with hourly data
python scripts/test_pipeline.py --horizon 24 --interval 1h --days 90
```

**Output**:
- Results saved to `experiments/results/{horizon}day/`
- Comparison reports in `experiments/reports/`
- Trained models in each experiment directory

See `experiments/README.md` for detailed documentation on interpreting results.
