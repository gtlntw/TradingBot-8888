# Before vs After: Transaction Cost Fix Comparison

## 1-Day Horizon Results

### Before Fix (Double-Counting - 45.1% positive examples)
Target: `profitable_trade = (return > 0.002)` - Only label as profitable if return > 0.2%  
Backtest: Deduct another 0.2% in commission + slippage  
**Effective cost: 0.4% per trade**

| Model | Accuracy | Total Return | Mean Ret | Sharpe | Win Rate |
|-------|----------|--------------|----------|--------|----------|
| Buy & Hold | 50.0% | **+127.91%** | 9.46% | 0.95 | 50% |
| LightGBM | 52.9% | +91.30% | 8.13% | 0.73 | 42% |
| Ensemble | 54.3% | +89.89% | 8.07% | 0.71 | 42% |
| **Transformer** | 53.5% | +85.25% | 7.46% | 0.72 | 50% |
| LSTM | 53.9% | +83.45% | 7.61% | 0.74 | 42% |
| XGBoost | 52.8% | +78.57% | 7.34% | 0.71 | 42% |
| Random Forest | 54.4% | +56.69% | 6.42% | 0.46 | 42% |

### After Fix (Single Cost - 50.23% positive examples)
Target: `profitable_trade = (return > 0)` - Simple direction prediction (up or down)  
Backtest: Deduct 0.2% in commission + slippage  
**Effective cost: 0.2% per trade (correct)**

| Model | Accuracy | Total Return | Mean Ret | Sharpe | Win Rate |
|-------|----------|--------------|----------|--------|----------|
| Buy & Hold | 50.0% | **+127.91%** | 9.46% | 0.95 | 50% |
| LSTM | 50.4% | **+121.88%** ⬆️ | 9.23% | 0.91 | 42% |
| **Transformer** | 51.4% | **+114.20%** ⬆️ | 8.59% | 0.84 | 50% |
| XGBoost | 52.1% | +81.43% ⬆️ | 7.14% | 0.72 | 50% |
| Random Forest | 51.8% | +79.51% ⬆️ | 7.22% | 0.67 | 42% |
| LightGBM | 53.1% | +69.88% ⬇️ | 6.59% | 0.61 | 42% |
| Ensemble | 51.7% | +64.49% ⬇️ | 6.34% | 0.58 | 42% |

## Key Changes

### Winners (Improved Performance)

**🏆 LSTM: +38.4pp improvement**
- Before: +83.45%
- After: +121.88%
- **Change: +38.43%**
- Now nearly matches buy-hold baseline!

**🏆 Transformer: +29.0pp improvement**
- Before: +85.25%
- After: +114.20%
- **Change: +28.95%**
- Excellent improvement, benefits from more training data

**🏆 Random Forest: +22.8pp improvement**
- Before: +56.69%
- After: +79.51%
- **Change: +22.82%**
- Tree-based models benefit from balanced data

**🏆 XGBoost: +2.9pp improvement**
- Before: +78.57%
- After: +81.43%
- **Change: +2.86%**
- Modest but positive improvement

### Losers (Declined Performance)

**⚠️ LightGBM: -21.4pp decline**
- Before: +91.30% (was best ML model)
- After: +69.88%
- **Change: -21.42%**
- May have been overfitting to conservative labels

**⚠️ Ensemble: -25.4pp decline**
- Before: +89.89%
- After: +64.49%
- **Change: -25.40%**
- Ensemble combines models, inherits LightGBM's decline

## Analysis

### Why Did Some Models Get Worse?

**LightGBM and Ensemble performed better with double-counting because:**

1. **More Conservative Labels:** Only 45% positive examples
2. **High-Confidence Trades:** Models learned to only trade on very strong signals (>0.4% expected return)
3. **Lower Trade Frequency:** Fewer trades = less exposure to market noise

**With the fix (50% positive):**
- More balanced data
- Models trade more frequently
- Some models (LSTM, Transformer) handle this better
- Others (LightGBM) may be making more marginal trades

### Why Did Others Improve?

**Deep Learning Models (LSTM, Transformer) benefit from:**
1. **More Training Data:** 50% positive vs 45% (5pp more examples)
2. **Clearer Patterns:** Learn price direction, not arbitrary 0.2% threshold
3. **Better Generalization:** More examples = less overfitting

**Random Forest improved because:**
1. **Balanced Data:** Tree algorithms prefer balanced classes
2. **Simpler Decision Boundary:** Up/down is clearer than >0.2%/>0.4%

### Which Approach Is Better?

**After Fix (Recommended):**
- ✅ LSTM now competitive: +121.88% (vs buy-hold +127.91%)
- ✅ Transformer strong: +114.20%
- ✅ Realistic evaluation (single cost application)
- ✅ Standard ML practice
- ✅ Easier to tune transaction cost parameter

**Before Fix (Conservative):**
- ✅ LightGBM was best: +91.30%
- ✅ Fewer, higher-confidence trades
- ❌ Double-counting costs (0.4% effective)
- ❌ Understated returns
- ❌ Artificial constraint on model learning

## Recommendations

1. **Use the fixed approach** (return > 0) for:
   - Production trading systems
   - Comparing with industry benchmarks
   - Testing different transaction cost levels

2. **Consider the old approach** (return > 0.2%) if:
   - You want ultra-conservative trading
   - You prefer fewer, higher-confidence signals
   - You're trading with LightGBM specifically

3. **Best Overall Model:** 
   - **LSTM** (+121.88%) - Best balance of return and consistency
   - Nearly matches buy-hold with active trading
   - High Sharpe ratio (0.91)

4. **Best for Risk-Adjusted Returns:**
   - **Buy & Hold** (Sharpe 0.95) - Still the baseline to beat
   - **LSTM** (Sharpe 0.91) - Very close second

## Conclusion

The cost fix reveals that **LSTM and Transformer perform much better** than previously measured:
- LSTM: 84% → 122% (+38pp)
- Transformer: 85% → 114% (+29pp)

However, LightGBM and Ensemble perform worse with balanced data.

**The fix is correct** - it provides realistic returns and standard ML practice. The old results were understated due to double-counting.

For production use: **LSTM with fixed cost accounting** achieves near-baseline returns (+122% vs +128% buy-hold) with active trading strategy.

