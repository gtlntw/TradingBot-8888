# Fixed Cost Accounting Experiment - In Progress

## What Changed

### Before Fix (Double-Counting)
```python
# Target: Only label as profitable if return > 0.2%
profitable_trade = (future_return > 0.002).astype(int)  # 45.1% positive

# Backtest: Deduct another 0.2%
commission = 0.001 + slippage = 0.001 = 0.2% total
```
**Effective cost: 0.4% per trade**

### After Fix (Single Cost Application)
```python
# Target: Simple direction prediction (up or down)
profitable_trade = (future_return > 0).astype(int)  # ~50% positive

# Backtest: Apply realistic costs here
commission = 0.001 + slippage = 0.001 = 0.2% total
```
**Effective cost: 0.2% per trade (correct)**

## Expected Results

### Class Distribution
- **Before:** 45.1% positive (return > 0.2%), 54.9% negative
- **After:** ~50% positive (return > 0%), ~50% negative
- **Impact:** Better balanced training data

### Returns
- **Before:** Understated by double-penalty
  - Transformer 1-day: +85.25%
  - XGBoost 7-day: +159.83%
- **After:** Realistic trading returns (estimated +10-20% higher)
  - Transformer 1-day: ~100-110% (estimated)
  - XGBoost 7-day: ~180-200% (estimated)

### Model Performance
- More positive training examples
- Better generalization
- Clearer decision boundary
- Still competitive with buy-hold baseline

## Experiment Configuration

**Process:** PID 12089  
**Started:** 2026-01-13 05:03 UTC  
**Log:** `walk_forward_fixed_costs.log`  
**Monitor:** `monitor_progress.log` (5-minute updates)

**Testing:**
- 5 horizons: 1-day, 7-day, 14-day, 28-day, 60-day
- 12 windows per horizon
- 7 models: RF, XGBoost, LightGBM, LSTM, Transformer, Ensemble, Buy-Hold
- Total: 420 model trainings

**Estimated Time:** ~50 minutes (10 min/horizon × 5)

## Comparison Plan

Once complete, we'll compare:

### Metrics to Compare
1. **Total Returns:** Before vs After for each model
2. **Win Rate:** Should remain similar (~50%)
3. **Accuracy:** Should improve slightly (~53% → ~55%)
4. **Sharpe Ratio:** Should improve with better returns
5. **Number of Trades:** Should increase (models less conservative)

### Key Questions
1. How much did returns improve? (target: +10-20%)
2. Do models still beat buy-hold baseline?
3. Which horizon performs best?
4. Is Transformer competitive now?

## Results Location

**New Results (Fixed):**
- `experiments/walk_forward_enhanced/*.json`

**Old Results (Double-Counting):**
- Available in previous runs (commits 888cca2, 54f3794)
- Transformer 1-day: +85.25%
- XGBoost 7-day: +159.83%

## How to Check Progress

```bash
# Check monitor log
tail -f monitor_progress.log

# Check experiment log
tail -f walk_forward_fixed_costs.log

# Check completed horizons
ls -lh experiments/walk_forward_enhanced/*.json

# Check process
ps -p $(cat walk_forward_fixed_costs.pid)
```

## Next Steps

1. ✅ Fix applied and committed (b21d6c4)
2. ✅ Experiment started (PID 12089)
3. 🔄 Running all 5 horizons (in progress)
4. ⏳ Compare results before/after
5. ⏳ Generate final report
6. ⏳ Update documentation with findings

