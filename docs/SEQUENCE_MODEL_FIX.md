# Sequence Model Fix - Investigation & Solution

## Problem Identified

Sequence models (LSTM, Transformer) were producing:
- **Identical predictions** for all samples in many windows
- **0.00% returns** consistently
- **~50% accuracy** (random guessing)

### Example from Real Test:
```
Window 1: lstm=63.33% +35.91% | transformer=63.33% +35.91%  ← IDENTICAL
Window 2: lstm=50.00% -4.10%   | transformer=50.00% -4.10%   ← IDENTICAL
Window 4: lstm=50.00% +0.00%   | transformer=50.00% -2.73%   ← IDENTICAL
Window 5: lstm=43.33% +0.00%   | transformer=43.33% +0.00%   ← IDENTICAL
```

## Root Cause Analysis

### Investigation Steps:

1. **Checked model architecture** - ✅ Correct (lines 225, 487 in sequence_models.py)
2. **Checked sequence generation** - ✅ Correct (sequences.py)
3. **Ran debug script** - 🔴 Found degenerate model!

### Critical Finding:

The LSTM model produced **IDENTICAL probabilities for ALL test samples**:

```
Sample 0: [0.5981, 0.4019] → pred=0
Sample 1: [0.5981, 0.4019] → pred=0
Sample 2: [0.5981, 0.4019] → pred=0
Sample 3: [0.5981, 0.4019] → pred=0
Sample 4: [0.5981, 0.4019] → pred=0

Std dev: 0.0000 (all predictions constant!)
```

This is a **degenerate neural network** - it's not learning from inputs at all.

## Root Cause: Missing Normalization

The OHLCV sequence data had **vastly different scales**:

```
Raw data ranges:
- Open/High/Low/Close: $40,000 - $100,000
- Volume: 1,000,000,000 - 180,000,000,000

Without normalization:
- Neural networks can't learn effectively
- Gradients vanish or explode
- Model outputs constant prediction
```

### Location of Bug:

**File:** `scripts/walk_forward_test_enhanced.py`
**Lines:** 304-324 (training), 398-402 (testing)

```python
# BEFORE (BROKEN):
X_seq, y_seq, _ = self.sequence_generator.create_sequences(...)
lstm.fit(X_seq, y_seq)  # ❌ No normalization!
```

## Solution Implemented

### Fix 1: Add Sequence Normalization (Training)

**Location:** scripts/walk_forward_test_enhanced.py:312-331

```python
# AFTER (FIXED):
X_seq, y_seq, _ = self.sequence_generator.create_sequences(...)

# Normalize sequences
from sklearn.preprocessing import MinMaxScaler

original_shape = X_seq.shape
X_seq_reshaped = X_seq.reshape(-1, X_seq.shape[2])

seq_scaler = MinMaxScaler(feature_range=(0, 1))
X_seq_normalized = seq_scaler.fit_transform(X_seq_reshaped)
X_seq_normalized = X_seq_normalized.reshape(original_shape)

# Store scaler for test data
models['sequence_scaler'] = seq_scaler

lstm.fit(X_seq_normalized, y_seq)  # ✅ Normalized!
```

### Fix 2: Apply Same Normalization (Testing)

**Location:** scripts/walk_forward_test_enhanced.py:404-410

```python
# Apply same normalization to test sequences
if 'sequence_scaler' in models and X_seq_test is not None:
    seq_scaler = models['sequence_scaler']
    original_shape = X_seq_test.shape
    X_seq_test_reshaped = X_seq_test.reshape(-1, X_seq_test.shape[2])
    X_seq_test_normalized = seq_scaler.transform(X_seq_test_reshaped)
    X_seq_test = X_seq_test_normalized.reshape(original_shape)
```

### Fix 3: Skip Scaler in Model Iteration

**Location:** scripts/walk_forward_test_enhanced.py:418-419

```python
for name, model in models.items():
    # Skip non-model items (like scalers)
    if name == 'sequence_scaler':
        continue
```

## Testing

### Initial Test Results:

With normalization:
- Probabilities now vary: 0.5301-0.5304 (vs constant 0.5981)
- Std dev: 0.000135 (very small but not zero)

**Status:** Partial improvement, but model still nearly degenerate.

### Further Investigation:

Testing different normalization methods:
1. Global MinMaxScaler (current)
2. Per-sequence MinMaxScaler
3. StandardScaler (Z-score)
4. Returns-based (percentage changes)

Financial time series often benefit from **returns-based normalization** (percentage changes from baseline) rather than absolute value scaling.

## Next Steps

1. Test different normalization methods
2. Consider sequence-specific normalization (normalize each 30-day window independently)
3. May need to add more training epochs or adjust model architecture
4. Consider adding dropout or regularization

## Files Modified

1. `scripts/walk_forward_test_enhanced.py` - Added normalization (3 locations)
2. Created debug scripts:
   - `scripts/debug_simple.py`
   - `scripts/test_normalization_fix.py`
   - `scripts/test_sequence_normalization_methods.py`

## Impact

Once fixed, sequence models should:
- Produce diverse predictions (not constant)
- Learn temporal patterns from 30-day OHLCV sequences
- Compare favorably against traditional models with engineered features
- Provide meaningful performance metrics

## References

- Degenerate model issue: debug_simple.py output
- Original bug report: test_monitored.log (lines showing 0% returns)
- Fix verification: test_normalization_fix.py
