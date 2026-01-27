# CRITICAL: Signal Semantics Bug - All Previous Results Invalid

**Date:** 2026-01-27
**Severity:** CRITICAL
**Status:** FIXED

## Summary

All walk-forward test results from 2026-01-26 and earlier are **INVALID** due to a signal semantics bug. Models were not trading on their predictions - they were buying once and holding forever.

## The Bug

### What Happened

**Location:** `scripts/walk_forward_test_enhanced.py` line 538

**Buggy Code:**
```python
signals = pd.Series(0, index=aligned_prices.index)
signals[pred_binary == 1] = 1   # Buy when profitable
signals[pred_binary == 0] = 0   # Hold cash when not profitable  ← BUG!
```

**The Problem:**
- Models predict: 0 = "not profitable", 1 = "profitable"
- Script mapped to signals: 0 → signal=0, 1 → signal=1
- **But backtester interprets:**
  - `signal=1`: Go long (buy if not already long)
  - `signal=-1`: Go to cash (sell if long, stay cash if in cash)
  - `signal=0`: **Hold current position** (do nothing!)

### Actual Behavior (Bug)

For a model predicting `[1, 0, 1, 0, 1, 0, ...]`:

```
Day 1: pred=1 → signal=1, position=0 → BUY BTC
Day 2: pred=0 → signal=0, position=1 → Hold BTC (signal=0 means hold!)
Day 3: pred=1 → signal=1, position=1 → Hold BTC (already long!)
Day 4: pred=0 → signal=0, position=1 → Hold BTC
...
Result: Buy once, hold forever = accidental buy-and-hold strategy
```

**Backtester condition** (line 329):
```python
if signal != 0 and signal != current_position:
    # Execute trade
```

- `signal=0` fails the first check → No trade
- `signal=1` when already `position=1` fails second check → No trade

### Why Results Looked Good (Misleading)

Previous results showed:
- Traditional models: +20-140% returns
- Sequence models: +77-225% returns
- Low "trade counts" (which we now know was actually zero trading!)

These were actually **buy-and-hold returns during a bull market**, not ML model alpha!

## The Fix

**Fixed Code:**
```python
signals = pd.Series(0, index=aligned_prices.index)
signals[pred_binary == 1] = 1    # Be long when profitable
signals[pred_binary == 0] = -1   # Be in cash when not profitable  ← FIXED!
```

**Now:**
- `pred=1` → `signal=1` → "Be long" (buy if not already long)
- `pred=0` → `signal=-1` → "Be in cash" (sell if long)

### Expected Behavior (Fixed)

For a model predicting `[1, 0, 1, 0, 1, 0, ...]`:

```
Day 1: pred=1 → signal=1, position=0 → BUY BTC
Day 2: pred=0 → signal=-1, position=1 → SELL BTC (go to cash)
Day 3: pred=1 → signal=1, position=0 → BUY BTC
Day 4: pred=0 → signal=-1, position=1 → SELL BTC
...
Result: Trade on every prediction change = actual trading strategy
```

## Impact on Results

### Before Fix (Invalid Results)

```
1-day transformer_60day: +77% return
7-day transformer_60day: +226% return
60-day transformer_60day: +157% return
Traditional models: 1 execution per window
Sequence models: 1 execution per window
```

**These were buy-and-hold returns, NOT model performance!**

### After Fix (Expected Real Results)

```
Execution frequency: 30-150 trades per 150-day window
Transaction costs: 0.6-3.0% per window (6-30% across 10 windows)
Returns: MUCH lower, possibly negative for high-frequency models
Only low-turnover, high-accuracy models will be profitable
```

## Verification Test

Created test with alternating signals:

```python
# Alternating predictions: [1, -1, 1, -1, ...]
signals = [1 if i % 2 == 0 else -1 for i in range(150)]
```

**Results:**
- **OLD (buggy) [1, 0, 1, 0, ...]**: 1 execution, +14.26% (buy-and-hold)
- **NEW (fixed) [1, -1, 1, -1, ...]**: 150 executions, -6.56% (actual trading)

The negative return with high-frequency trading makes sense due to transaction costs!

## What Was Actually Tested?

### Previous Tests (Invalid)

**Not tested:**
- Model prediction accuracy translated to trading performance
- ML model alpha generation
- Transaction cost impact

**Actually tested:**
- Buy-and-hold returns during 2016-2026 period
- Market beta, not model alpha
- First prediction only (models bought once then ignored all future predictions)

### What We Learned (Still Valid)

✅ **Cross-horizon fairness works** - All horizons tested same period
✅ **Data quality validation works** - No data issues detected
✅ **Feature selection works** - Models trained successfully
✅ **Sequence models work** - LSTM/Transformer made predictions

❌ **Trading performance is unknown** - Never actually traded on predictions!

## Action Required

### Immediate: Re-run All Tests

All previous test results must be discarded and re-run with the fix:

```bash
# Full 10-year test (REQUIRED)
python scripts/run_all_horizons_walk_forward.py \
    --days 3650 --train-window 730 --test-window 150 --step-size 300 \
    2>&1 | tee results_FIXED_full_10years.log

# Quick verification (~1 hour)
python scripts/run_all_horizons_walk_forward.py \
    --days 3650 --train-window 730 --test-window 150 --step-size 300 \
    --no-sequences \
    2>&1 | tee results_FIXED_quick.log
```

### Expected New Results

**Likely outcomes:**
- Traditional models: Lower returns, possibly negative due to high turnover
- Sequence models: Lower returns but may stay positive with lower turnover
- Only high-accuracy, low-turnover models will be profitable
- Transaction costs will dominate performance for frequent traders

**What to look for:**
- `actual_trades` in JSON (now tracks real executions)
- Models with <50 trades per 150-day window (low turnover)
- High accuracy (>55%) + low turnover = profitable
- Low accuracy or high turnover = unprofitable after costs

## Files Changed

1. `scripts/walk_forward_test_enhanced.py` - Fixed signal mapping (line 538)
2. `trading_bot/evaluation/backtester.py` - Added actual_trades tracking
3. `docs/CRITICAL_SIGNAL_BUG_FIX.md` - This document
4. `docs/TRADE_COUNT_FIX.md` - Related documentation

## Lessons Learned

### Testing Blind Spots

**What we missed:**
- Never verified that models were actually trading
- Assumed "149 trades" meant 149 executions (it meant 149 days)
- Didn't inspect individual trade executions
- Didn't question why returns were so high

**How to prevent:**
- Always inspect `len(portfolio.trades)` directly
- Check trade logs for actual buy/sell executions
- Verify signal generation logic with unit tests
- Question results that seem too good to be true

### Architecture Issues

**Problem:** Test scripts bypass trading bot's signal generation
- `walk_forward_test_enhanced.py` creates signals directly
- Doesn't use `trading_bot/trading/signals.py` SignalGenerator classes
- Duplicates logic that should be in one place

**Better approach:**
- Move signal generation to `trading_bot/trading/signals.py`
- Create `BinaryPredictionSignalGenerator` class
- Test scripts should use this class, not create signals ad-hoc
- One source of truth for signal semantics

## Conclusion

This was a **critical bug** that invalidated all previous test results. The bug was subtle:
- Code looked correct at first glance
- Results looked plausible (positive returns)
- "Trade counts" seemed reasonable (149 trades)
- But models weren't actually trading!

The fix is simple but requires **re-running all tests** to get valid results. Previous results measured buy-and-hold performance during a bull market, not ML model trading performance.

---

**Status:** Fixed in commit `ccaaa1e`
**Branch:** `claude/fix-sequence-model-predictions-8XcJ9`
**Next Step:** Re-run all tests and document real trading performance
