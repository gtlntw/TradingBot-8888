# Architecture Decision: Normalization Location

## Question
Why is data normalization in `walk_forward_test_enhanced.py` instead of the feature engineering pipeline?

## Answer: Time-Series Data Leakage Prevention

### The Problem
In time-series forecasting, **future information must never leak into training data**.

If we normalized data in the feature engineering step (before splitting into train/test windows), we would:
1. Calculate mean/std from ALL data (including future)
2. Apply that global scaler to all windows
3. **Leak future information** into training → overly optimistic results

### The Correct Approach (Current Implementation)

**Walk-forward testing requires per-window normalization:**

```python
Window 1 (2023-01-01 to 2023-03-31):
  ├─ Fit scaler on Window 1 train data
  └─ Apply to Window 1 test data

Window 2 (2023-02-01 to 2023-04-30):
  ├─ Fit NEW scaler on Window 2 train data  # Different from Window 1!
  └─ Apply to Window 2 test data

Window 3 ...
```

Each window has its own scaler fitted ONLY on that window's training data.

### Implementation

We created a `DataNormalizer` helper class in `scripts/walk_forward_test_enhanced.py`:

```python
class DataNormalizer:
    """
    Helper class for data normalization in walk-forward testing.

    NOTE: Normalization is done HERE (not in feature engineering) because:
    - Walk-forward testing requires per-window normalization
    - Each window must fit its own scaler on training data
    - Prevents future data leakage
    """

    @staticmethod
    def normalize_features(X_train, X_test):
        """Normalize 2D feature data."""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler

    @staticmethod
    def normalize_sequences(X_train_seq, X_test_seq):
        """Normalize 3D sequence data."""
        # Implementation...
```

### Why Not in Feature Engineering?

The `FeatureEngineer` class is designed to:
- Calculate technical indicators (RSI, MACD, etc.)
- Create market features (returns, volatility, etc.)
- Add sentiment features (fear/greed index)

These are **feature creation** tasks, not **preprocessing** tasks.

Normalization is a **preprocessing step that depends on train/test split**, so it belongs with the train/test logic.

### Why Not in the Model Classes?

While models COULD normalize internally, this would:
- Duplicate scaler logic across 7 different model classes
- Make it harder to ensure consistent normalization
- Complicate model serialization (need to save scalers)

The current approach centralizes normalization logic while respecting time-series constraints.

### StandardScaler vs MinMaxScaler

We use **StandardScaler (Z-score normalization)** for both:
- **Traditional models**: 30 engineered features
- **Sequence models**: 5 raw OHLCV features

**Why StandardScaler?**
- More robust to outliers (financial data has many)
- Preserves the distribution shape
- Industry standard for neural networks
- Consistent across all models

**Why NOT MinMaxScaler?**
- Sensitive to outliers (one extreme value affects all scaling)
- Can cause issues if test data has values outside training range
- Initial bug: we used MinMaxScaler for sequences, StandardScaler for traditional

### Files Modified

1. **scripts/walk_forward_test_enhanced.py**
   - Lines 53-110: `DataNormalizer` helper class
   - Line 378: Training - normalize sequences
   - Line 457: Testing - apply same scaler to test sequences

2. **SEQUENCE_MODEL_FIX.md**
   - Documents the degenerate model bug caused by missing normalization

3. **This file (NORMALIZATION_ARCHITECTURE.md)**
   - Explains architectural decision

### References

- [Scikit-learn: StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [Time Series Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)
- Walk-forward analysis methodology

### Summary

**Normalization is correctly placed in the walk-forward test script** to prevent data leakage in time-series forecasting. This is the standard approach for time-series cross-validation.
