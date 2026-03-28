# Treasury Signals — Autoresearch Agent Loop
*Adapted from karpathy/autoresearch*

## The Loop

You are an overnight research agent. Your goal is to improve the composite OOS Sharpe of a 10-year US Treasury signal dashboard.

You have one file you are allowed to edit: **`signals.py`**

Everything else is fixed infrastructure:
- `data_layer.py` — data download + immutable evaluators (never touch)
- `backtest_runner.py` — walk-forward engine (never touch)
- `report.py` — report generation (never touch)

---

## Iteration Protocol

Each iteration you must:

1. **Read** `results/backtest_summary.csv` and the current `composite_sharpe:` line in the last run output
2. **Read** `signals.py` and reason about which signals underperform
3. **Edit** exactly one thing in `signals.py` (parameter tweak, signal logic fix, or new signal)
4. **Run** the backtest:
   ```
   python backtest_runner.py 2>&1 | tee results/last_run.txt
   ```
5. **Check** the result:
   ```
   grep -E "composite_sharpe:|n_positive_oos:|best_signal:" results/last_run.txt
   ```
6. **Decide**: keep or revert
   - Keep if: new `composite_sharpe` ≥ old `composite_sharpe` OR `n_positive_oos` increased by ≥ 2
   - Revert if worse (use `git checkout signals.py`)
7. **Log** the result to `results.tsv`:
   ```
   echo "<commit>\t<composite_sharpe>\t<n_positive_oos>\t<keep|revert>\t<description>" >> results.tsv
   ```
8. **Commit** if keeping:
   ```
   git add signals.py results.tsv
   git commit -m "signals: <description> — sharpe <X.XXX>"
   ```

---

## What You May Edit in signals.py

### Tier 1 — Parameter tuning (fastest, try first)
Change the `P_` parameter constants at the top of signals.py:
```python
P_S01_LOOKBACK = 63    # try 42, 63, 126, 252
P_S04_SLOW = 200       # try 150, 200, 252
P_S07_VOL_WINDOW = 63  # try 42, 63, 126
P_S20_N_STATES = 2     # try 2 or 3
P_S22_COINT_WINDOW = 504  # try 252, 504, 756
```

### Tier 2 — Signal logic improvements
- Tighten entry/exit thresholds (e.g. z-score ±1.0 → ±1.5)
- Add filters (e.g. only trade momentum when VIX < 25)
- Improve regime conditioning (only hold long in HMM bull-bond state)
- Combine two weak signals into one composite signal

### Tier 3 — New signals (riskiest, highest potential)
- Add signals S26+ following the same contract: `pd.Series` in {-1, 0, 1}, shifted by 1 day
- Register in `ALL_SIGNALS` list
- Must have a warmup period (NaN for first N rows)

---

## Non-Negotiables (Hard Rules)

```
NEVER introduce look-ahead bias:
  - No .shift(-1) on signals
  - No using tomorrow's data to compute today's signal
  - signal.shift(1) in compute_metrics() is the 1-day execution lag (correct)

NEVER modify:
  - evaluate_sharpe() in data_layer.py
  - compute_buy_and_hold() in data_layer.py
  - walk_forward_backtest() in backtest_runner.py
  - IS_WINDOW, OOS_WINDOW, STEP_SIZE in backtest_runner.py

NEVER use data that wasn't available at signal time:
  - COT report has 3-day publication lag (already modeled)
  - FOMC minutes are released ~3 weeks after the meeting (already modeled)
  - CPI/GDP data have 1–2 month lags (use reindex+ffill)
```

---

## Key Metrics to Watch

| Metric | Target | Meaning |
|--------|--------|---------|
| `composite_sharpe` | > 0.50 | Equal-weighted OOS Sharpe across all signals |
| `n_positive_oos` | ≥ 15/25 | Signals beating B&H in OOS |
| `best_signal` | any | Best individual OOS Sharpe |
| `overfit_ratio` | < 2.0 | IS/OOS Sharpe ratio (lower = less overfitting) |

---

## Grep Targets

After each run, check:
```bash
grep "composite_sharpe:" results/last_run.txt
grep "n_positive_oos:" results/last_run.txt
grep "best_signal:" results/last_run.txt
```

These lines are also written to `results/backtest_summary.csv` for history.

---

## Branch Naming

Each experiment branch (optional): `signals/exp-<description>-<date>`

Main development branch: `claude/treasury-signals-research-FOHvb`

---

## Full Pipeline Commands

```bash
# 1. Download all data (cached 24h)
python data_layer.py

# 2. Run all 25 signals + walk-forward backtest
python backtest_runner.py

# 3. Generate HTML report
python report.py

# 4. View report
open results/report.html
```

---

## Stopping Criterion

Stop iterating when any of:
- `composite_sharpe` > 0.70 (excellent)
- `n_positive_oos` = 25/25 (all signals beat B&H)
- 50 iterations completed without improvement (local optimum)
- `git log --oneline | wc -l` > 100 (sufficient search history)

---

## results.tsv Format

Columns (tab-separated):
```
commit  composite_sharpe  n_positive_oos  status  description
```

Status: `keep` | `revert` | `neutral`

Example:
```
abc1234  0.523  18  keep    S20 HMM: added yield_change^2 feature
def5678  0.501  16  revert  S15 COT: tightened percentile threshold — worse
```
