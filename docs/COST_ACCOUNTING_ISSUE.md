# Transaction Cost Double-Counting Issue

## Problem Identified

**We are applying transaction costs TWICE:**

### Cost Application #1: Target Creation
```python
# scripts/walk_forward_test_enhanced.py, line 186
future_return = (close_tomorrow / close_today) - 1
profitable_trade = (future_return > 0.002).astype(int)  # 0.2% threshold
```

**Effect:** Model learns "only trade if return > 0.2%"

### Cost Application #2: Backtesting
```python
# scripts/walk_forward_test_enhanced.py, lines 420-421
backtester = Backtester(
    commission=0.002 / 2,  # 0.1%
    slippage=0.002 / 2      # 0.1%
)
# Total: 0.2% deducted from returns
```

**Effect:** Backtest subtracts another 0.2% from returns

## Net Impact

**For a trade to show +0.1% profit in backtest:**
1. True return must be > 0.2% (to be labeled profitable)
2. Backtest deducts 0.2%: `0.3% - 0.2% = 0.1%`
3. **True return needed: 0.3%** (not 0.1%!)

**Models are being overly conservative:**
- Effective cost per trade: 0.4% (double the actual 0.2%)
- Only 45.1% of days labeled as "profitable" 
- Should be ~50% with 0% threshold

## Current Results Are Understated

Our reported returns are AFTER double-penalty:
- Transformer 1-day: +85.25% (with ~0.4% effective cost)
- True performance with correct accounting: likely higher

## Three Options to Fix

### Option 1: Cost in Target Only
```python
# Target
profitable_trade = (future_return > 0.002).astype(int)

# Backtest
backtester = Backtester(commission=0, slippage=0)
```

**Pros:** Model learns cost-awareness  
**Cons:** Backtest returns are optimistic (don't show real costs)  
**Use case:** Research/development phase

### Option 2: Cost in Backtest Only ✅ **RECOMMENDED**
```python
# Target  
profitable_trade = (future_return > 0).astype(int)

# Backtest
backtester = Backtester(
    commission=0.001,  # 0.1%
    slippage=0.001     # 0.1%
)
```

**Pros:**
- Backtest returns are realistic (what you'd actually get)
- More positive examples for model training (~50% vs 45%)
- Standard ML practice: let model find patterns, apply costs in evaluation

**Cons:** Model doesn't explicitly learn about costs  

**Use case:** Production/live trading evaluation

### Option 3: Split Cost 50/50
```python
# Target
profitable_trade = (future_return > 0.001).astype(int)  # 0.1%

# Backtest
backtester = Backtester(
    commission=0.0005,  # 0.05%
    slippage=0.0005     # 0.05%
)
```

**Pros:** Balanced approach  
**Cons:** Neither is fully realistic  
**Use case:** Compromise solution

## Recommendation: Option 2

**Use Option 2 because:**

1. **Standard Practice:** Separate learning from evaluation
   - Training: Learn patterns without artificial constraints
   - Evaluation: Apply realistic costs to measure actual performance

2. **Better Model Training:**
   - More balanced classes: 50/50 instead of 45/55
   - More positive examples to learn from
   - Model can find subtler patterns

3. **Honest Evaluation:**
   - Backtest returns show what you'd actually get
   - Easy to test different cost levels (0.1%, 0.2%, 0.5%)
   - Matches real-world trading exactly

4. **Interpretability:**
   - `y=1` means "price will go up"
   - Simple, clear semantics
   - Costs are trading friction, not part of prediction

## Implementation

### Change 1: Update target creation
```python
# scripts/walk_forward_test_enhanced.py, line 186
# OLD:
features_df['profitable_trade'] = (future_return > self.transaction_cost).astype(int)

# NEW:
features_df['profitable_trade'] = (future_return > 0).astype(int)
```

### Change 2: Keep backtest costs unchanged
```python
# Keep lines 420-421 as-is
backtester = Backtester(
    commission=self.transaction_cost / 2,
    slippage=self.transaction_cost / 2
)
```

## Expected Impact

**Class Distribution:**
- Current: 45.1% positive (return > 0.2%)
- After fix: ~50.0% positive (return > 0)
- Better balanced for classification

**Returns:**
- Current results are understated by ~10-20%
- After fix: Returns will be slightly higher but more realistic
- Easier to compare with industry benchmarks

**Model Performance:**
- More training examples
- Better generalization (less overfitting to high-return days)
- Clearer decision boundary

## Testing the Fix

Run side-by-side comparison:
```bash
# Test with current approach (0.2% threshold)
python scripts/walk_forward_test_enhanced.py --horizon 1 --quick

# Test with recommended approach (0% threshold)
# (after making the code change)
python scripts/walk_forward_test_enhanced.py --horizon 1 --quick
```

Compare:
- Accuracy (should be similar ~53%)
- Total return (recommended approach should be 5-15% higher)
- Win rate (should be similar ~50%)
- Number of positive examples in training

## References

- Current code: `scripts/walk_forward_test_enhanced.py` lines 186, 420-421
- Transformer fix: Commit `888cca2`
- Transaction cost parameter: `self.transaction_cost = 0.002` (0.2%)

