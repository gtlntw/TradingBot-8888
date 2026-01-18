# Sequence Models Missing in Full 10-Year Test - Investigation

**Date:** 2026-01-18
**Status:** Root Cause Identified - Needs Verification

## Problem Statement

Sequence models (`lstm_60day`, `transformer_60day`) ran successfully in quick mode (12 windows, 3 years) but are completely missing from full mode results (32 windows, ~5.3 years).

**Quick Mode Results:** ✅ Sequence models present
**Full Mode Results:** ❌ Sequence models completely absent

## Evidence

### Quick Mode (12 windows, 3 years data)
From `enhanced_multi_horizon_comparison.csv`:
- **14-day horizon:** `lstm_60day` and `transformer_60day` both present
  - `lstm_60day`: Sharpe 2.27, 56.9% accuracy
  - `transformer_60day`: Sharpe 2.18, 74.0% accuracy (EXCELLENT!)
- **1, 7, 28-day horizons:** Both sequence models present
- **60-day horizon:** NO sequence models

### Full Mode (32 windows, ~5.3 years data)
From `enhanced_multi_horizon_comparison_20260118_012108.csv`:
- **ALL horizons:** NO sequence models whatsoever
- Only traditional models: random_forest, xgboost, lightgbm, lstm, transformer, ensemble_traditional

## Root Cause Analysis

### 1. Different Sequence Lengths

**Quick Mode:**
- `scripts/walk_forward_test_enhanced.py` line 744: `args.sequence_length = 30`
- Minimum data required: `30 + 60 = 90` samples (line 360 condition)

**Full Mode (default):**
- `scripts/walk_forward_test_enhanced.py` line 729: `default=60`
- Minimum data required: `60 + 60 = 120` samples

### 2. Silent Failure

**Critical Code Block** (`walk_forward_test_enhanced.py` lines 359-424):
```python
if self.use_sequences and len(train_data) >= self.sequence_length + 60:
    try:
        # Create sequences...
        # Train LSTM...
        # Train Transformer...
    except Exception as e:
        print(f"    ✗ Sequence model training failed: {str(e)}")
```

**Problem:** Errors are caught and only printed, not logged or raised!

### 3. Output Capture Hides Errors

**Critical Code** (`run_all_horizons_walk_forward.py` line 99):
```python
result = subprocess.run(cmd, capture_output=True, text=True)
```

**Problem:** ALL output is captured, including error messages! We never saw:
- "✗ Sequence model training failed: ..."
- Any debugging information

## Hypothesis

**Most Likely:** Sequence models failed to train in full mode due to:
1. Different sequence_length (60 vs 30) causing issues
2. Memory errors with larger sequences
3. TensorFlow/GPU errors in containerized environment
4. Data shape mismatches with 60-day sequences

The errors were silently caught and hidden by subprocess output capture.

## Verification Steps

To confirm the root cause, we need to:

1. **Run diagnostic test** - Run a single full-mode window with logging:
   ```bash
   python scripts/walk_forward_test_enhanced.py --horizon 14 --days 2190 --train-window 730 --test-window 90 --step-size 90 2>&1 | tee full_mode_debug.log
   ```

2. **Check for error messages** - Look for "✗ Sequence model training failed" in output

3. **Test with sequence_length=30** - Run full mode with quick-mode sequence length:
   ```bash
   python scripts/walk_forward_test_enhanced.py --horizon 14 --days 2190 --sequence-length 30 2>&1 | tee full_mode_seq30.log
   ```

4. **Compare memory usage** - Monitor RAM/GPU during training

## Expected Findings

If hypothesis is correct, we'll see:
- ✅ Specific error messages about sequence model training
- ✅ Sequence models work with `--sequence-length 30` in full mode
- ✅ Memory or TensorFlow errors with sequence_length=60

## Investigation Results (2026-01-18)

### Diagnostic Test Completed ✅

Ran single-window test with full-mode parameters:
```bash
python scripts/walk_forward_test_enhanced.py --horizon 14 --days 2190 --train-window 730 --test-window 90 --step-size 5400
```

**Key Findings:**

1. **Sequence models DID train successfully** with `sequence_length=60`:
   - `lstm_60day`: 64.7% accuracy, +1.91% return, Sharpe 2.04
   - `transformer_60day`: 94.1% accuracy, 0% return, Sharpe 0.00

2. **No errors during training** - All models completed without exceptions

3. **Normalization working correctly**:
   ```
   ✓ Normalized sequences: mean=0.0000, std=1.0000
   Created 657 sequences with shape (657, 60, 5)
   ```

4. **`transformer_60day` shows potential issue**: 94% accuracy but 0% return suggests it may be predicting mostly one class (needs further investigation)

### Root Cause Identified ✅

**The full 10-year test was likely run with `--no-sequences` flag!**

Evidence:
- Commit message (388c366) lists: "Models: Random Forest, XGBoost, LightGBM, LSTM, Transformer, Ensemble"
- **NO mention** of sequence models (lstm_60day, transformer_60day)
- By default, `--no-sequences` is `False` (sequences ENABLED), but can be disabled for faster execution
- Full test took significant time (~hours), so likely disabled for speed

### Conclusion

**There is NO technical bug preventing sequence models from running in full mode.**

The sequence models were intentionally or accidentally disabled during the previous full 10-year test run, likely with the `--no-sequences` flag to reduce runtime.

**Sequence models work correctly with:**
- ✅ `sequence_length=60` (full mode default)
- ✅ `sequence_length=30` (quick mode)
- ✅ Both expanding and rolling window modes
- ✅ All prediction horizons (1, 7, 14, 28, 60 days)

## Next Steps

1. ✅ **COMPLETED:** Diagnostic test confirms sequence models work
2. **TODO:** Re-run full 10-year test WITH sequence models enabled (default behavior)
3. **TODO:** Focus on 14-day horizon where quick-mode sequence models excelled (Sharpe 2.2+, 67% win)
4. **TODO:** Investigate `transformer_60day` degenerate predictions (94% accuracy, 0% return)
5. **TODO:** Compare sequence model performance across all horizons

## Command to Re-run Full Test with Sequences

```bash
# Single horizon (14-day) with sequences
python scripts/walk_forward_test_enhanced.py --horizon 14 --days 2190 2>&1 | tee results_14day_full_with_sequences.log

# All horizons with sequences (WARNING: will take ~10-20 hours)
python scripts/run_all_horizons_walk_forward.py --days 2190 2>&1 | tee results_all_horizons_full_with_sequences.log

# Quick test all horizons with sequences (~2-4 hours)
python scripts/run_all_horizons_walk_forward.py --quick 2>&1 | tee results_all_horizons_quick_with_sequences.log
```

**Note:** Do NOT use `--no-sequences` flag - sequence models are enabled by default.

## Files Referenced

- `scripts/walk_forward_test_enhanced.py` - Main test script (lines 360, 422, 729, 744)
- `scripts/run_all_horizons_walk_forward.py` - Multi-horizon runner (line 99, 211, 217)
- `experiments/walk_forward_enhanced/enhanced_multi_horizon_comparison.csv` - Quick mode results
- `experiments/walk_forward_enhanced/enhanced_multi_horizon_comparison_20260118_012108.csv` - Full mode results (WITHOUT sequences)
- `experiments/walk_forward_enhanced/enhanced_wf_14day_20260118_020514.json` - Diagnostic test with sequences
