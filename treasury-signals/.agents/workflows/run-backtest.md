# Workflow: Run Full Backtest + Interpret Results

## Running

```bash
# Full run (all 25 signals) — redirect for autoresearch loop
python backtest_runner.py > run.log 2>&1

# Extract key metric
grep "^composite_sharpe:" run.log

# Check for crashes
grep -i "error\|traceback" run.log | head -20
```

## Interpreting Results

After `backtest_runner.py` completes, check `results/backtest_summary.csv`:

| Column | Good | Needs Work |
|:---|:---|:---|
| OOS Sharpe | > 0.3 | < 0.1 |
| Overfit Ratio | < 0.5 | > 0.7 |
| vs B&H Sharpe | Positive excess | Negative |
| NumTrades | 20–200/yr | > 500/yr |

## Generating Report

```bash
python report.py
# Opens: results/report.html
```

## Common Issues

**Signal has Sharpe 0.0**: Check if signal is all-flat (never trades). Print `sig.value_counts()`.

**OOS Sharpe negative but IS positive**: Classic overfit. Reduce parameters.

**TypeError in statsmodels**: HMM/VECM may fail on short series. Check `df.dropna()` length.

**COT data not loading**: Check `~/.cache/treasury_signals/cot_lev_net.parquet` exists.
  Re-run `data_layer.py` if missing.
