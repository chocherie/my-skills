# Walk-Forward Backtesting Methodology

## Why Walk-Forward (Not Simple Train/Test Split)

A single 70/30 split is regime-dependent — the split point determines which regime is IS vs OOS. Walk-forward rolls through ALL regimes, producing 30-70+ OOS windows for statistical confidence.

## Configuration

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| IS Window | 504 days (~2 years) | Enough data for stable parameter estimation |
| OOS Window | 63 days (~3 months) | Long enough for meaningful returns, short enough for many steps |
| Step Size | 63 days | Non-overlapping OOS windows |
| Min History | 20 years recommended | Captures bull, bear, sideways regimes |

## Metrics Computed Per Window

- Total Return (%)
- Sharpe Ratio (annualized, 252 trading days)
- Max Drawdown (%)
- Win Rate (% of days with positive return when positioned correctly)
- Hit Rate (% of correct direction predictions)

## Overfit Ratio

```
overfit_ratio = 1 - (OOS_avg_sharpe / IS_avg_sharpe)
```

Interpretation:
- < 0.3: Low overfitting (signal is robust)
- 0.3-0.7: Moderate overfitting (use with caution)
- > 0.7: Severe overfitting (signal is unreliable OOS)
- Negative: OOS outperforms IS (rare, possibly lucky)

## OOS-Only Equity Curve

Stitch together ONLY the OOS periods to build a "pure" out-of-sample equity curve. This represents what a trader would have actually experienced using the signal in real-time.

## Anti-Overfitting Checklist

1. Never optimize parameters on the full dataset
2. Report OOS Sharpe alongside IS Sharpe — always
3. Flag any dimension with overfit ratio > 0.7
4. Use at least 20+ walk-forward steps for statistical significance
5. Test across multiple market regimes (bull, bear, sideways)
6. For signal combination: use non-linear methods (RF, GBM) over fixed-weight linear
