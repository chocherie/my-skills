# Workflow: Add or Tune a Signal (autoresearch loop)

Use this during the autonomous experiment loop (program.md).

## Adding a New Signal

1. Add spec entry to `specs/signals.md` (ID, logic, data, params, expected Sharpe)
2. Implement `compute_SXX(df)` in `signals.py` following the output contract
3. Register in `ALL_SIGNALS` list in `backtest_runner.py`
4. Run `python backtest_runner.py --signals SXX`
5. Log to `results.tsv`

## Tuning an Existing Signal

1. Read the signal's spec in `specs/signals.md`
2. Identify the parameter to change (lookback, threshold, z_window)
3. Edit only the PARAMS block at the top of the signal function
4. Run full backtest, check OOS Sharpe vs previous commit
5. If improved: `git commit -m "tune(S01): lookback 63d→90d, OOS Sharpe 0.31→0.38"`
6. If not improved: `git reset HEAD~1` and log as "discard"

## Output Contract Reminder

Every signal function returns:
- `pd.Series` aligned to `df.index`
- Values in `{-1, 0, 1}`
- NaN for warmup period
- Named: `f"S{id:02d}_{short_name}"`
