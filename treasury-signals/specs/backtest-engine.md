# Spec: backtest_runner.py

## Walk-Forward Configuration

- **IS window**: 504 trading days (~2 years)
- **OOS window**: 63 trading days (~3 months)
- **Step**: 63 days
- **Min steps**: 10 (signals with <10 WF steps are flagged)
- **Common start**: 2003-01-01 (for consistency across all signals)

## compute_metrics(returns, signal, prefix="")

Inherited from `trading-signal-dashboard/scripts/backtest_engine.py`.
Returns dict with:
- `total_return` — cumulative return %
- `annual_return` — CAGR %
- `sharpe_ratio` — annualized Sharpe (excess return / std)
- `max_drawdown` — max peak-to-trough %
- `win_rate` — % of active days with positive return
- `profit_factor` — gross gains / gross losses
- `num_trades` — number of signal direction changes
- `hit_rate` — % of days signal correctly predicts direction

## Buy-and-Hold Benchmark

```python
bah_signal = pd.Series(1.0, index=returns.index)
bah_metrics = compute_metrics(returns, bah_signal, prefix="bah_")
```

The B&H benchmark is long IEF every day from the signal's first valid date.

## Rolling Sharpe

```python
strat_returns = signal.shift(1) * returns
rolling_sharpe = (strat_returns.rolling(252).mean() /
                  strat_returns.rolling(252).std() * np.sqrt(252))
```

## Output per Signal

```python
{
    "id": "S01",
    "name": "3M Momentum",
    "category": "A",
    "start_date": "2003-01-01",
    "full": {metrics dict},
    "bah": {B&H metrics dict},
    "is_avg": {avg IS metrics},
    "oos_avg": {avg OOS metrics},
    "overfit_ratio": float,  # 1 - (oos_sharpe / is_sharpe)
    "n_wf_steps": int,
    "equity_curve": pd.DataFrame,  # date, strategy, buyandhold
    "rolling_sharpe": pd.Series,
}
```

## Master Summary Table Columns

```
ID | Name | Category | Start | Total_Return% | Ann_Return% | Sharpe | MaxDD% |
WinRate% | HitRate% | NumTrades | IS_Sharpe | OOS_Sharpe | OverfitRatio |
BAH_Return% | BAH_Sharpe | BAH_MaxDD% | Excess_Sharpe
```

Where `Excess_Sharpe = signal Sharpe − B&H Sharpe`

## Printed Summary Block (grep-able for autoresearch loop)

```
---
composite_sharpe: 0.523
n_signals: 25
n_positive_oos: 18
best_signal: S20_HMM (OOS Sharpe 0.71)
worst_signal: S17_RRP (OOS Sharpe -0.12)
```
