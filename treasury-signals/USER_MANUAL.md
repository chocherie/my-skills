# 10Y US Treasury Signal Dashboard — User Manual

**Branch**: `claude/treasury-signals-research-FOHvb`
**Directory**: `my-skills/treasury-signals/`

---

## Table of Contents

1. [What This Is](#1-what-this-is)
2. [Architecture Overview](#2-architecture-overview)
3. [File Reference](#3-file-reference)
4. [First-Time Setup](#4-first-time-setup)
5. [Running the Pipeline](#5-running-the-pipeline)
6. [Understanding the Report](#6-understanding-the-report)
7. [The 25 Signals](#7-the-25-signals)
8. [How Returns Are Calculated](#8-how-returns-are-calculated)
9. [How the Backtest Works](#9-how-the-backtest-works)
10. [The Autoresearch Loop](#10-the-autoresearch-loop)
11. [Interpreting Results](#11-interpreting-results)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. What This Is

A systematic research platform for 10-year US Treasury trading signals. It:

- Downloads ~36 years of market data (1990–present) from FRED, Yahoo Finance, NY Fed, and CFTC
- Computes **25 trading signals** across 6 categories (momentum, macro, cross-asset, positioning, statistical, global/NLP)
- Backtests every signal individually using **rolling walk-forward validation** (~130 out-of-sample windows)
- Compares every signal vs a buy-and-hold benchmark over the same period
- Generates an HTML report with equity curves, rolling Sharpe ratios, and a master comparison table
- Supports an **overnight autoresearch loop** where an agent continuously improves signals

The underlying instrument is the **IEF ETF** (iShares 7-10 Year Treasury Bond ETF). Pre-2002 history uses a duration-approximated return series.

---

## 2. Architecture Overview

```
data_layer.py          ← downloads & caches all data (IMMUTABLE evaluators)
    ↓
signals.py             ← computes all 25 signals   ← ONLY FILE YOU EDIT
    ↓
backtest_runner.py     ← rolling walk-forward backtest (IMMUTABLE engine)
    ↓
report.py              ← generates results/report.html
    ↓
results/
  backtest_summary.csv
  equity_S01.csv ... equity_S25.csv
  rolling_sharpe.csv
  report.html
```

**The autoresearch pattern** (from karpathy/autoresearch):
- `signals.py` is the only editable file
- `evaluate_sharpe()` in `data_layer.py` is the immutable ground truth
- The loop: edit → run → check Sharpe → keep or revert → repeat

---

## 3. File Reference

| File | Purpose | Edit? |
|------|---------|-------|
| `data_layer.py` | Downloads all data, defines immutable `evaluate_sharpe()` and `compute_buy_and_hold()` | **Never** |
| `signals.py` | All 25 signal implementations + parameter block | **Yes — this is the research file** |
| `backtest_runner.py` | Walk-forward engine, metrics, output CSV | **Never** |
| `report.py` | HTML report generator | No |
| `program.md` | Autoresearch loop instructions for agents | Reference |
| `results.tsv` | Log of every autoresearch iteration | Auto-updated |
| `make_synthetic_data.py` | Generates fake data for offline testing | Dev only |
| `AGENTS.md` | <100-line orientation for AI agents | Reference |
| `specs/` | Detailed specifications for signals, data, backtest | Reference |
| `docs/` | Architecture, core beliefs, quality grades | Reference |

---

## 4. First-Time Setup

### Prerequisites
- Python 3.9+
- A free FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html

### Install dependencies

```bash
pip3 install fredapi yfinance pandas numpy statsmodels scikit-learn \
  hmmlearn arch pykalman nelson-siegel-svensson scipy vaderSentiment \
  "pdfminer.six" matplotlib seaborn jinja2 beautifulsoup4 pytz \
  protobuf platformdirs patsy scikit-base pillow pyarrow openpyxl
```

### Clone and navigate

```bash
git clone https://github.com/chocherie/my-skills.git
cd my-skills
git checkout claude/treasury-signals-research-FOHvb
cd treasury-signals
```

### Set your FRED API key

```bash
export FRED_API_KEY=your_key_here
# To make it permanent:
echo 'export FRED_API_KEY=your_key_here' >> ~/.zshrc
```

---

## 5. Running the Pipeline

### Step 1 — Download data (~5 minutes, cached for 24h)

```bash
python3 data_layer.py
```

Downloads ~36 years of daily data from:
- **Yahoo Finance**: IEF (ETF), VIX, Oil (WTI)
- **FRED**: 13 yield series, 9 macro series (CPI, Fed Funds, SOFR, HY spread, DXY, EUR/JPY yields, RRP, foreign custody)
- **NY Fed**: ACM term premium, primary dealer positions
- **CFTC**: Commitment of Traders (leveraged funds, 10Y)
- **Fed website**: FOMC minutes (VADER sentiment)

Data is cached at `~/.cache/treasury_signals/master_df.parquet`. Re-run to refresh.

Expected output at the end:
```
IEF 2022 total return: -15.9%  [OK]
Buy-and-Hold (IEF, 1990-01-01 to 2026-03-27):
  Sharpe:       0.51
  Total Return: 320.4%
  Annual Return: 4.1%
  Max Drawdown: -23.9%
```

If any series fails (proxy/network issue), it shows `WARNING:` and continues — most signals will still run using the available data.

### Step 2 — Run all 25 backtests (~5–30 minutes)

```bash
python3 backtest_runner.py
```

For each signal it:
1. Computes the signal over the full history
2. Runs ~130 rolling walk-forward windows (IS=504d, OOS=63d, step=63d)
3. Stitches all OOS windows into one continuous equity curve
4. Computes metrics on the full stitched OOS period

Watch the terminal — it prints one line per signal:
```
[S01] 3M_Momentum        Sharpe= 0.241  MaxDD= -8.43%  OOS=0.187  ExcessSharpe=-0.323
[S05] RSI14_Yield        Sharpe= 0.612  MaxDD= -5.21%  OOS=0.743  ExcessSharpe=+0.232
```

At the end, look for the **grep-able summary**:
```
composite_sharpe: 0.312
n_signals: 25
n_positive_oos: 14
best_signal: S05_RSI14_Yield (OOS Sharpe 0.743)
worst_signal: S04_SMA_Cross (OOS Sharpe -0.091)
bah_sharpe: 0.519
```

Outputs saved to `results/`:
- `backtest_summary.csv` — one row per signal, all metrics
- `equity_S01.csv` … `equity_S25.csv` — daily stitched OOS equity curves
- `rolling_sharpe.csv` — 252-day rolling Sharpe for all signals

### Step 3 — Generate HTML report

```bash
python3 report.py
open results/report.html        # Mac
xdg-open results/report.html   # Linux
```

---

## 6. Understanding the Report

The HTML report has four sections:

### Master Comparison Table

| Column | Meaning |
|--------|---------|
| IS Sharpe | Average in-sample Sharpe across all IS windows |
| OOS Sharpe | Sharpe ratio of the full **stitched** out-of-sample equity curve |
| B&H Sharpe | Buy-and-hold Sharpe over the same OOS period |
| Overfit Ratio | `1 − (OOS_Sharpe / IS_Sharpe)`. Near 0 = well-generalised. >1 = overfit |
| OOS Ann. Ret | Annualised return of stitched OOS equity curve |
| OOS Total Ret | Cumulative return of stitched OOS equity curve |
| OOS Max DD | Maximum drawdown of stitched OOS equity curve |
| Win Rate | % of active trading days with positive P&L (OOS) |
| OOS Wins | Number of 63-day OOS windows where strategy Sharpe > 0 |
| DSR | Deflated Sharpe Ratio (multiple-testing correction) |
| DSR Pass | PASS if DSR > 0.5 — statistically significant after testing 25 strategies |

**Colour coding**: green = Sharpe ≥ 0.5, yellow = 0–0.5, red = negative.

### DSR Table

The Deflated Sharpe Ratio corrects for the fact that we tested 25 strategies. A raw Sharpe of 0.5 is much less impressive after 25 tries than after 1 try. DSR > 0.5 means the signal is statistically significant even after accounting for multiple testing.

### Correlation Heatmap

Shows the pairwise correlation of daily OOS strategy returns across all signals. Signals clustered together (high correlation) provide less diversification — prefer uncorrelated signals when combining.

### Per-Signal Charts

Each signal gets a two-panel chart:
- **Top**: Stitched OOS equity curve (blue = strategy, orange dashed = buy-and-hold)
- **Bottom**: 252-day rolling Sharpe ratio

The chart title shows how many OOS windows contributed (e.g., "130 OOS windows (rolling WF)").

---

## 7. The 25 Signals

### Category A — Momentum (S01–S05)

| ID | Name | Logic |
|----|------|-------|
| S01 | 3M_Momentum | If IEF 63-day cumulative return > 0 → long, else short |
| S02 | 6M_Momentum | Same with 126-day window |
| S03 | 12_1_Momentum | 12-month return minus last 1 month (skip-1-month momentum) |
| S04 | SMA_Cross | IEF price above 200-day SMA → long; below → flat |
| S05 | RSI14_Yield | RSI-14 on DGS10 yield. High RSI = yield overbought → buy bonds |

### Category B — Macro (S06–S10)

| ID | Name | Logic |
|----|------|-------|
| S06 | Carry_Slope | T10Y2Y z-score. Steep curve → positive carry → long |
| S07 | Breakeven_Mom | Rising breakeven inflation → short bonds |
| S08 | ACM_TermPrem | Low ACM term premium → bonds cheap → long |
| S09 | Real_Yield | Negative TIPS real yield → financial repression → long |
| S10 | CPI_Mom | Rising CPI momentum → short bonds |

### Category C — Cross-Asset (S11–S14)

| ID | Name | Logic |
|----|------|-------|
| S11 | VIX_Safety | VIX spike (>1σ) → risk-off → flight to quality → long |
| S12 | HY_Spread | HY credit spread widening → risk-off → long bonds |
| S13 | DXY_Mom | USD momentum → mild bond correlation signal |
| S14 | Oil_Mom | Rising oil → inflation → short bonds |

### Category D — Positioning (S15–S17)

| ID | Name | Logic |
|----|------|-------|
| S15 | COT_Lev | CFTC leveraged fund net position percentile. Crowded short → contrarian long |
| S16 | PrimDealer | NY Fed primary dealer accumulation → bullish |
| S17 | RRP_Filter | Fed reverse repo usage trend → liquidity signal |

### Category E — Statistical (S18–S22)

| ID | Name | Logic |
|----|------|-------|
| S18 | PCA_Slope | Rolling PCA on yield curve. PC2 (slope factor) positive → long |
| S19 | Kalman_Spread | Kalman filter on T10Y2Y. Mean-revert residual from filtered trend |
| S20 | HMM_Regime | 2-state Gaussian HMM on yield change + VIX. Identifies bull-bond regime |
| S21 | NSS_Dev | Nelson-Siegel-Svensson model. 10Y deviation above fitted curve → long |
| S22 | VECM_EC | OLS cointegration DGS10 vs log(CPI). Error-correction term signal |

### Category F — Global / NLP (S23–S25)

| ID | Name | Logic |
|----|------|-------|
| S23 | ForeignCustody | NY Fed foreign custody holdings trend → foreign CB demand signal |
| S24 | GlobalFactor | Global bond factor: average of DE10Y + JP10Y vs DGS10 |
| S25 | FOMC_Sentiment | VADER sentiment of Fed minutes. Hawkish → short, dovish → long |

---

## 8. How Returns Are Calculated

The backtest uses a **spliced return series**:

```
1990 ──────────────── 2002-07-22 ──────────────── Present
  Duration-approx                  IEF ETF actual
```

**Duration-approximated return (pre-2002)**:
```
R_t = -8.5 × Δyield_t + (yield_{t-1} / 252) - (fed_funds_{t-1} / 252)
```
- `-8.5 × Δyield`: price change from yield movement (modified duration ≈ 8.5 years)
- `yield / 252`: daily accrual of the coupon (carry)
- `fed_funds / 252`: funding cost (financing a leveraged position)

**IEF ETF total return (2002–present)**:
- Direct daily price change including reinvested dividends
- More accurate than the approximation — captures roll, coupons, bid/ask

The 2022 bear market check: IEF returned **−15.9%** in 2022 (actual: −16.0%). The approximation is accurate enough for signal testing.

---

## 9. How the Backtest Works

### Walk-Forward Methodology

```
Time ──────────────────────────────────────────────────────────▶
       [====IS====][OOS]
            [====IS====][OOS]
                 [====IS====][OOS]
                      ...          (~130 windows over 36 years)
```

- **IS (In-Sample)**: 504 trading days (~2 years). The signal is "trained" — parameters are fixed per `signals.py`, but the signal uses IS data to build its rolling statistics.
- **OOS (Out-of-Sample)**: 63 trading days (~3 months). The signal generates positions using only data available up to that point. No look-ahead.
- **Step**: 63 days. Each window moves forward one quarter.
- **Total windows**: ~130

### Signal Execution

Each position is shifted forward by 1 day before multiplying returns:
```python
strategy_return_t = signal_{t-1} × return_t
```

This models **next-day execution** — a signal generated at close on day t is executed at open on day t+1.

### Metrics Computed

All metrics are computed on the **stitched OOS equity curve** (the concatenation of all 130 OOS windows):

- **Sharpe Ratio**: `mean(daily_return) / std(daily_return) × √252`
- **Annual Return**: `(1 + total_return)^(1/n_years) − 1`
- **Max Drawdown**: Maximum peak-to-trough decline of the equity curve
- **Win Rate**: % of active days (signal ≠ 0) with positive P&L
- **Overfit Ratio**: `1 − (OOS_Sharpe / IS_avg_Sharpe)`. Should be < 1.0.

### Buy-and-Hold Benchmark

The same return series, always long, over the same period as each signal's stitched OOS. This is the correct benchmark — not a fixed calendar period.

---

## 10. The Autoresearch Loop

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch). One agent runs in a loop overnight improving signals.

### The Loop

```
1. Read composite_sharpe from results/last_run.txt
2. Read signals.py — identify underperforming signals
3. Edit ONE thing in signals.py (parameter or logic)
4. python3 backtest_runner.py 2>&1 | tee results/last_run.txt
5. grep "composite_sharpe:" results/last_run.txt
6. If better: keep + commit. If worse: git checkout signals.py
7. Append to results.tsv
8. Go to step 1
```

### What You Can Change

**Tier 1 — Parameters** (fastest, lowest risk):
```python
# At the top of signals.py
P_S01_LOOKBACK = 63    # try 42, 63, 126, 252
P_S04_SLOW = 200       # try 150, 200, 252
P_S20_N_STATES = 2     # try 2 or 3
```

**Tier 2 — Signal logic** (medium risk):
- Change entry/exit thresholds
- Add regime filters (e.g., only trade S01 when VIX < 25)
- Change long/flat to long/short where justified

**Tier 3 — New signals** (highest risk, highest upside):
- Add S26, S27… following the same contract
- Register in `ALL_SIGNALS` list at the bottom of `signals.py`

### Hard Rules

```
NEVER:
  - Use .shift(-1) on a signal (look-ahead bias)
  - Modify evaluate_sharpe() or compute_buy_and_hold() in data_layer.py
  - Change IS_WINDOW, OOS_WINDOW, STEP_SIZE in backtest_runner.py
  - Use data that wasn't available at signal time
```

### Stopping Criterion

Stop when any of:
- `composite_sharpe > 0.70` (excellent — beats B&H with margin)
- `n_positive_oos = 25/25` (all signals beat flat in OOS)
- 50 consecutive iterations without improvement

### Tracking Progress

```bash
cat results.tsv           # full history
tail -5 results.tsv       # last 5 experiments
```

---

## 11. Interpreting Results

### What "Good" Looks Like

| Metric | Weak | Acceptable | Strong |
|--------|------|------------|--------|
| OOS Sharpe | < 0.3 | 0.3–0.5 | > 0.5 |
| vs B&H Sharpe | Below | Within 0.1 | Above |
| Overfit Ratio | > 2.0 | 0.5–2.0 | < 0.5 |
| DSR | FAIL | — | PASS |
| OOS Wins | < 60/130 | 70–90/130 | > 100/130 |

### Context: Why Beating B&H Is Hard

The IEF buy-and-hold had a Sharpe of ~0.52 from 1990–2026 — this is unusually high for a single asset class because:
- The 35-year bond bull market (falling rates 1981–2021) consistently rewarded holding
- Bonds have positive carry (you earn the coupon just for holding)

Any signal that goes short bonds fights against both the secular trend and the carry. Most signals will underperform B&H in raw Sharpe. The useful question is whether a signal:
1. Reduces max drawdown (especially 2022: −16% for IEF)
2. Has lower volatility than B&H
3. Has uncorrelated returns (useful for portfolio construction)

### Red Flags

| Pattern | Likely Cause |
|---------|-------------|
| IS Sharpe >> OOS Sharpe (overfit > 2) | Parameter overfitting — too many parameters, too few OOS samples |
| OOS Sharpe = 0.000 | Signal always returns 0 (data unavailable or logic bug) |
| OOS Sharpe negative but large | Signal direction inverted — check bull/bear logic |
| Equity curve jumps at splice point (2002) | Regime difference between approx and IEF return series |

---

## 12. Troubleshooting

### `zsh: command not found: python`
Use `python3` — macOS uses `python3`, not `python`.

### `ModuleNotFoundError: No module named 'X'`
```bash
pip3 install X
```
Common ones: `pyarrow`, `openpyxl`, `protobuf`, `platformdirs`, `patsy`, `scikit-base`.

### `ImportError: Unable to find a usable engine` (parquet)
```bash
pip3 install pyarrow
```

### FRED download fails (`ProxyError` / `403`)
Your network blocks FRED (common in corporate environments). Two options:
1. Use a different network
2. Run `python3 make_synthetic_data.py` to generate realistic fake data for testing — all code works with it

### `KeyError: 'signal_id'` in report.py
The `backtest_summary.csv` column names don't match. Delete `results/backtest_summary.csv` and re-run `backtest_runner.py`, then `report.py`.

### Signal shows 0.000 Sharpe for both IS and OOS

The signal's required data column is missing. Check which data failed in `data_layer.py` output:

| Signal | Required column | Common failure reason |
|--------|-----------------|----------------------|
| S08 | `acm_tp` | NY Fed Excel URL changed |
| S15 | `cot_lev_net` | `cot_reports` module not available |
| S16 | `pd_net_6_11y` | NY Fed primary dealer Excel |
| S23 | `farbast` | FRED FARBAST series unavailable |
| S25 | `fomc_hawk_score` | Fed website scrape blocked |

### Backtest finishes too quickly
If `backtest_runner.py` finishes in under 2 minutes, S20 (HMM) may have silently failed. Check for `WARNING: S20` in the output. The HMM signal is the most computationally expensive.

### Report shows `—` for OOS metrics
Re-run `backtest_runner.py` first — the CSV may be from a previous version that didn't include OOS detail columns.

---

*Last updated: 2026-03-28*
