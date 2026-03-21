# Intra-Step Look-Ahead Bias in OOS-Weighted Voting: Case Study

## Context

This document records the discovery and fix of a look-ahead bias bug in the OOS-Weighted Voting combiner method of a 9-dimension gold (XAU/USD) trading signal dashboard. The bug inflated the strategy's total return from +177% to +30,074% — a 170x exaggeration.

## Strategy Configuration

| Parameter | Value |
|---|---|
| Asset | XAU/USD (Gold) |
| Period | 2006-12-12 to 2026-01-30 |
| Dimensions | 9 independent signal dimensions |
| IS Window | 504 days (2 years) |
| OOS Window | 63 days (3 months) |
| Step Size | 63 days |
| Total Steps | 79 |
| Combination Methods | Equal-Weight, OOS-Weighted, Gradient Boost |

## The Bug

The OOS-Weighted Voting method assigns dimension weights based on an exponential moving average (EMA, alpha=0.3) of each dimension's OOS Sharpe ratio. The bug was in the execution order within each walk-forward step.

### Buggy Code (Before Fix)

```python
# Step N of walk-forward loop:

# 1. Update dim_oos_sharpes with CURRENT OOS window data
for dim_idx in range(NUM_DIMS):
    dim_sig = signals[oos_start:oos_end-1, dim_idx]
    dim_strat = dim_sig * y_oos_rets
    dim_sharpe = compute_sharpe(dim_strat)
    dim_oos_sharpes[dim_idx] = 0.3 * dim_sharpe + 0.7 * dim_oos_sharpes[dim_idx]

# 2. Compute weights from the JUST-UPDATED dim_oos_sharpes
weights = np.maximum(dim_oos_sharpes, 0)
weights_norm = weights / np.sum(weights)

# 3. Generate signals using weights that contain current OOS info
ow_preds = [oos_weighted_signal(X_oos[i], weights_norm) for i in range(len(X_oos))]
```

The problem: Step 1 incorporates the current OOS window's performance into `dim_oos_sharpes` before Step 2 uses those values to set weights. The combiner effectively "knows" which dimensions will perform well in the current window.

### Fixed Code (After Fix)

```python
# Step N of walk-forward loop:

# 1. Compute weights from dim_oos_sharpes (ONLY prior steps' data)
weights = np.maximum(dim_oos_sharpes, 0)
weights_norm = weights / np.sum(weights)

# 2. Generate signals using weights from prior steps only
ow_preds = [oos_weighted_signal(X_oos[i], weights_norm) for i in range(len(X_oos))]

# 3. Record equity and performance...

# 4. THEN update dim_oos_sharpes for use in the NEXT step
for dim_idx in range(NUM_DIMS):
    dim_sig = signals[oos_start:oos_end-1, dim_idx]
    dim_strat = dim_sig * y_oos_rets
    dim_sharpe = compute_sharpe(dim_strat)
    dim_oos_sharpes[dim_idx] = 0.3 * dim_sharpe + 0.7 * dim_oos_sharpes[dim_idx]
```

## Impact

### All 9 Dimensions Signal Set

| Metric | Before Fix | After Fix | Change |
|---|---|---|---|
| Total Return | +30,074% | +177% | -170x |
| Annual Return | +34.2% | +5.4% | -28.8pp |
| Sharpe Ratio | 1.83 | 0.39 | -1.44 |
| OOS Avg Sharpe | 2.01 | 0.41 | -1.60 |
| IS Avg Sharpe | 0.70 | 0.87 | +0.17 |
| Overfit Ratio | -1.88 | +0.53 | Negative to moderate |
| Max Drawdown | -16.6% | -42.8% | -26.2pp |
| Win Rate | 56.0% | 51.9% | -4.1pp |

### 8 Dimensions Signal Set (excl. Institutional Flows)

| Metric | Before Fix | After Fix | Change |
|---|---|---|---|
| Total Return | +35,544% | +336% | -106x |
| Sharpe Ratio | 1.88 | 0.53 | -1.35 |
| OOS Avg Sharpe | 2.05 | 0.56 | -1.49 |
| Overfit Ratio | -1.69 | +0.40 | Negative to moderate |

### Unaffected Methods (Confirming Isolation)

Equal-Weight Voting and Gradient Boosting were completely unaffected by the fix, confirming the bug was isolated to the OOS-weighted weight computation.

## Detection Heuristics

This bug was detected through the following observations:

1. **Extraordinary total return**: +30,074% total return with "only" a 2.0 Sharpe seemed inconsistent at first glance, but the math actually checks out (34% CAGR over 19 years). The real red flag was the comparison to Equal-Weight (+168%) — a 170x difference between two voting methods is implausible.

2. **Negative overfit ratio**: An overfit ratio of -1.88 means OOS performance was nearly 3x better than IS performance. While the skill documents this as a red flag (Q3: Negative Overfit Ratio Mechanism), the specific mechanism was the intra-step look-ahead.

3. **Rolling Sharpe consistency**: The buggy OOS-weighted method had a rolling 1-year Sharpe that almost never went below 1.0 over 19 years. No legitimate strategy maintains this consistency.

4. **Daily return distribution**: Despite having 170x the total return of Buy & Hold, the daily return standard deviation was nearly identical (1.06% vs 1.08%). This means the strategy was achieving its returns through near-perfect directional accuracy, not through leverage or volatility.

## Lessons Learned

1. **Intra-step ordering matters**: Walk-forward validation protects against cross-step look-ahead, but cannot detect look-ahead within a single step. Code review is the only defense.

2. **Compare methods**: If an adaptive method outperforms a non-adaptive baseline by >10x, the adaptation mechanism itself should be audited for look-ahead.

3. **Negative overfit ratios demand explanation**: A negative overfit ratio is extraordinary. The Q3 skeptical question ("What specific mechanism explains OOS > IS?") would have caught this if applied rigorously.

4. **EMA weights are particularly vulnerable**: Any weight scheme that uses recent performance to set current weights must ensure strict temporal ordering. The EMA's exponential decay does not protect against same-step contamination.
