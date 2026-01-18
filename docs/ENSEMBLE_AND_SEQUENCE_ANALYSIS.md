# Analysis: Ensemble Method & Sequence Model Issues

## Question 1: Which Ensemble Method Are We Using?

### Answer: Sharpe-Optimized Voting Ensemble

**Code Location:** `scripts/walk_forward_test_enhanced.py` lines 285-290

```python
ensemble = EnsembleModel(
    models=base_models,
    method='voting',
    optimize_for_sharpe=True,     # ← Sharpe optimization enabled
    validation_split=0.2           # ← Uses 20% for validation
)
```

### How It Works:

1. **Base Models:** Combines RF, XGBoost, LightGBM, LSTM, Transformer
2. **Method:** Voting (weighted average of predictions)
3. **Optimization:** Finds optimal weights to maximize Sharpe ratio
4. **Validation:** Uses 20% of training data to find best weights

### Example from Logs:

```
Window 1:
Split for Sharpe optimization: 292 train, 73 validation
Sharpe optimization successful. Sharpe: 1.0420
Voting ensemble fitted with weights: {
    'random_forest': 0.250,
    'xgboost': 0.250,
    'lightgbm': 0.250,
    'lstm': 0.250,
    'transformer': 0.001    # ← Transformer gets very low weight!
}
```

**Key Insight:** The optimizer assigns nearly equal weights to RF/XGBoost/LightGBM (0.25 each) but almost zero to Transformer (0.001). This suggests Transformer wasn't performing well during validation.

### Why Sharpe Optimization?

- **Goal:** Maximize risk-adjusted returns (not just raw returns)
- **Formula:** Sharpe = (Return - RiskFree) / Volatility
- **Benefit:** Ensemble prefers consistent models over volatile ones

## Question 2: Why Are Models Trained Twice?

### Answer: Two Different Architectures

Looking at the logs, you see:

**First Training (Traditional):**
```
Training models on 365 samples...
  ✓ random_forest
  ✓ xgboost
  ✓ lightgbm
  ✓ lstm          ← Traditional LSTM (30 features, 1 timestep)
  ✓ transformer   ← Traditional Transformer (30 features, 1 timestep)
  ✓ ensemble_traditional
```

**Second Training (Sequence):**
```
Training sequence models (lookback=30)...
Created 335 sequences: (335, 30, 5)
  ✗ lstm_60day: SequenceLSTMModel.__init__() got an unexpected keyword argument 'sequence_length'
  ✗ transformer_60day: SequenceTransformerModel.__init__() got an unexpected keyword argument 'sequence_length'
```

### The Two Architectures:

| Aspect | Traditional | Sequence |
|--------|-------------|----------|
| **Model Name** | `lstm`, `transformer` | `lstm_60day`, `transformer_60day` |
| **Input** | 30 selected features | 5 OHLCV features |
| **Timesteps** | 1 (single point) | 30 (sequence) |
| **Shape** | (samples, 30) | (samples, 30, 5) |
| **Purpose** | Learn from engineered features | Learn temporal patterns |
| **Implementation** | `LSTMModel` (trainer.py) | `SequenceLSTMModel` (sequence_models.py) |

### Why Both?

**Traditional LSTM/Transformer:**
- Uses engineered features (RSI, MACD, volatility, etc.)
- Single timestep per prediction
- Faster to train
- ✅ Working correctly

**Sequence LSTM/Transformer:**
- Uses raw OHLCV data in sequences
- Learns temporal patterns directly
- Slower to train
- ❌ **FAILING** - Parameter error

## The Bug: Sequence Models Are Failing

### Root Cause:

**Code (lines 313-319):**
```python
lstm = SequenceLSTMModel(
    sequence_length=self.sequence_length,  # ← ERROR: Not a valid parameter
    n_features=5,
    lstm_units=64,
    dropout=0.2,
    learning_rate=0.001
)
```

**Expected Signature:**
```python
class SequenceLSTMModel(BaseModel):
    def __init__(self, model_type: str = 'classification', params: Optional[Dict] = None):
        ...
```

### The Problem:

`SequenceLSTMModel` follows the `BaseModel` pattern and expects:
1. `model_type` (string: 'classification' or 'regression')
2. `params` (dictionary of hyperparameters)

But we're passing kwargs directly like `sequence_length=...`, which causes:
```
✗ lstm_60day: SequenceLSTMModel.__init__() got an unexpected keyword argument 'sequence_length'
```

### The Fix:

**Current (Broken):**
```python
lstm = SequenceLSTMModel(
    sequence_length=self.sequence_length,
    n_features=5,
    lstm_units=64,
    dropout=0.2,
    learning_rate=0.001
)
```

**Should Be:**
```python
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
```

**Note:** `SequenceLSTMModel` infers sequence shape from the data during `.fit()`, so we don't need to pass `sequence_length` or `n_features` explicitly.

## Impact on Results

### Current Situation:

**Working Models:**
- ✅ Traditional: RF, XGBoost, LightGBM, LSTM, Transformer (6 models)
- ✅ Ensemble: Combines the 5 traditional models
- ❌ Sequence: lstm_60day, transformer_60day (both failing)

**Results Reported:**
- Only traditional models + ensemble
- Missing potential benefits of sequence learning
- Ensemble weights show Transformer underperforming

### If Sequence Models Worked:

**Potential Benefits:**
- Learn temporal patterns (trends, momentum, reversals)
- No need for feature engineering on sequences
- May capture patterns traditional models miss

**Potential Issues:**
- More data hungry (need longer sequences)
- Slower to train
- May overfit to recent patterns

## Recommendations

1. **Fix Sequence Model Initialization:**
   - Update lines 313-340 in walk_forward_test_enhanced.py
   - Use proper `params` dict instead of kwargs

2. **Consider If Sequence Models Are Needed:**
   - Traditional LSTM already performs well (+122% return)
   - Sequence models add ~2x training time
   - May not improve results significantly

3. **Investigate Low Transformer Weights:**
   - Why does ensemble give Transformer weight of 0.001?
   - Is Transformer overfitting during validation?
   - Could benefit from different hyperparameters

4. **Test Ensemble Methods:**
   - Current: Sharpe-optimized voting
   - Alternative: Stacking (meta-learner on top)
   - Alternative: Equal weights (simple average)

## Summary

**Question 1:** We're using **Sharpe-optimized voting ensemble** that finds optimal weights to maximize risk-adjusted returns on a validation set.

**Question 2:** Models are trained **twice with different architectures**:
- **Traditional:** 30 features, 1 timestep (WORKING)
- **Sequence:** 5 OHLCV, 30 timesteps (FAILING due to parameter bug)

The sequence models could potentially improve results but are currently broken due to incorrect parameter passing.

