# Statistical Checks for Backtest Validation

## 1. Deflated Sharpe Ratio (DSR)

The DSR corrects observed Sharpe ratios for the inflationary effect of multiple testing, non-normal returns, and finite sample length. Based on Bailey & Lopez de Prado (2014).

**Formula:**
```
DSR = Φ[(SR_hat - SR_benchmark) / SE(SR)]
```

Where:
- `SR_hat` = mean of per-step OOS Sharpe ratios
- `SR_benchmark` = expected max SR from N independent trials under null (all strategies are noise)
- `SE(SR)` = standard error adjusted for skewness and kurtosis
- `Φ` = standard normal CDF

**Expected Max SR from N trials:**
```
SR_benchmark = (1 - γ) × Φ⁻¹(1 - 1/N) + γ × Φ⁻¹(1 - 1/(N·e))
```
Where γ = 0.5772 (Euler-Mascheroni constant).

**Counting N (independent trials):**
Count every strategy variation tested, including:
- Each individual signal dimension
- Each combination method (equal-weight, weighted, ML)
- Each signal set variant (with/without certain dimensions)
- Any hyperparameter sweeps (each parameter set = 1 trial)

**Interpretation:**
| DSR Value | Meaning |
|---|---|
| > 0.95 | Statistically significant at 95% confidence |
| 0.80 - 0.95 | Suggestive but not conclusive |
| < 0.80 | Not significant — could be explained by chance |

**Script:** `scripts/deflated_sharpe_ratio.py`

## 2. IS-OOS Correlation

Compute Pearson correlation between per-step IS Sharpe and per-step OOS Sharpe.

| Correlation | Interpretation |
|---|---|
| > +0.3 | IS has predictive power for OOS (good sign) |
| −0.1 to +0.3 | IS does not predict OOS (neutral) |
| < −0.1 | IS inversely predicts OOS (red flag: curve-fitting) |

Negative correlation means the strategy finds patterns in-sample that reverse out-of-sample — a hallmark of overfitting.

## 3. Directional Bias Detection

Count the percentage of bullish, bearish, and neutral signals. Flag if:
- Any direction > 60% of total signals (moderate bias)
- Any direction > 75% of total signals (severe bias)
- Neutral signals = 0% (tie-breaking rule is forcing a direction)

Common cause: tie-breaking rules that default tied votes to bullish (+1). Fix: tied vote → neutral (0, no position).

## 4. Regime-Conditional Analysis

Break performance into distinct market regimes and check if the strategy only works in one:

1. Define regimes by rolling 63-day cumulative return (positive = bull, negative = bear)
2. Compute Sharpe ratio separately for bull and bear days
3. Flag if the difference exceeds 1.0 Sharpe (regime-dependent)

A strategy that only works in bull markets may simply be capturing the secular trend rather than generating alpha.

## 5. Transaction Cost Sensitivity

Estimate round-trip cost (spread + commission) and compute:
```
Net Annual Return = Gross Annual Return - (Annual Trades × Cost Per Trade)
```

Conservative estimates by asset class:
| Asset | Round-Trip Cost |
|---|---|
| XAU/USD spot | 5-10 bps |
| Gold futures | 2-5 bps |
| Gold ETF (GLD) | 3-8 bps |
| Equity indices | 5-15 bps |
| Individual stocks | 10-30 bps |

Flag strategies where net return turns negative after costs.

## 6. Buy-and-Hold Benchmark Comparison

Always compute a buy-and-hold benchmark from the same start date as the strategy's first OOS window. Compare:
- Total return (strategy vs benchmark)
- Sharpe ratio (strategy vs benchmark)
- Max drawdown (strategy vs benchmark)

If the strategy underperforms buy-and-hold on all three metrics, it adds no value beyond being long the asset.

## 7. Overfit Ratio Validation

Formula: `overfit_ratio = 1 - (OOS_Sharpe / IS_Sharpe)`

| Range | Classification | Action |
|---|---|---|
| < 0 | Negative (OOS > IS) | Investigate — could be genuine or adaptive momentum |
| 0 - 0.30 | Low Overfitting | Acceptable |
| 0.30 - 0.70 | Moderate Overfitting | Caution — may not persist |
| > 0.70 | Severe Overfitting | Exclude from live trading |

**Blind spots of overfit ratio:**
- Does not correct for multiple testing (use DSR instead)
- Does not detect directional bias
- Does not account for regime dependence
- Treats all walk-forward steps equally (extreme regimes may skew)

## 8. Signal Distribution Stationarity

Check if the signal distribution changes over time by comparing the first half vs second half:
- If bullish % shifts by > 15 percentage points, the signal is non-stationary
- Non-stationary signals may work in one regime and fail in another

## References

- Bailey, D.H. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio." *Journal of Portfolio Management*.
- Bailey, D.H. et al. (2015). "The Probability of Backtest Overfitting." *Journal of Computational Finance*.
- Harvey, C.R. et al. (2016). "...and the Cross-Section of Expected Returns." *Review of Financial Studies*.
