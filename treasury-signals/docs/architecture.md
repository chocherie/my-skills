# Architecture: 10Y UST Signal Dashboard

## System Overview

```
[Data Sources]          [data_layer.py]         [signals.py]
FRED API          →     download_all_data()  →   25 signal functions
yfinance (IEF)    →     build_master_df()    →   each returns pd.Series [-1,0,1]
CFTC COT          →     evaluate_sharpe()    ↓
NY Fed CSV        →     (immutable metric)   [backtest_runner.py]
TreasuryDirect    →                          →   walk_forward_backtest() per signal
                                             →   buy_and_hold comparison
                                             →   rolling_sharpe()
                                             ↓
                                         [report.py]
                                         →   equity curves per signal
                                         →   master comparison table
                                         →   rolling Sharpe heatmap
                                         →   DSR table
```

## Directory Structure

```
treasury-signals/
├── data_layer.py              # Fixed: download + evaluation functions
├── signals.py                 # Editable: all 25 signals + params
├── backtest_runner.py         # Wires signals → backtest engine → results/
├── report.py                  # Generates HTML report + charts
├── program.md                 # autoresearch agent loop instructions
├── results.tsv                # Experiment log (gitignored)
├── results/
│   ├── backtest_summary.csv   # Master table: all signals vs B&H
│   ├── equity_{ID}.csv        # Per-signal equity curve (strategy + B&H)
│   └── rolling_sharpe.csv     # 252d rolling Sharpe per signal over time
├── signals/                   # (reserved for modular signal files if refactored)
├── docs/
│   ├── architecture.md        # This file
│   ├── core-beliefs.md        # Research principles
│   ├── quality.md             # Per-signal quality scorecard
│   └── plans/active/          # Active ExecPlans
├── specs/
│   ├── README.md              # Spec index
│   ├── data-layer.md          # Data sources spec
│   ├── signals.md             # All 25 signals spec
│   ├── backtest-engine.md     # Backtest config spec
│   └── report.md              # Report output spec
├── .cursor/rules/             # Auto-injected behavioral rules
├── .agents/workflows/         # Task-specific agent instructions
└── pyproject.toml             # Dependencies
```

## Data Flow

### Step 1: Data Download (`data_layer.py`)
- Downloads to `~/.cache/treasury_signals/` (persistent cache, re-downloads if stale)
- Returns a single `master_df` DataFrame with DatetimeIndex
- Key columns:
  - `ief_return` — IEF daily total return (primary backtest instrument)
  - `ust_return_approx` — duration-approx return from DGS10 (extended history)
  - `dgs10` — 10Y yield (%)
  - `dgs2`, `dgs5`, `dgs30` — other maturities
  - `t10y2y` — 10Y-2Y spread
  - `t10yie` — breakeven inflation
  - `dfii10` — TIPS real yield
  - `cpiaucsl` — CPI index
  - `fedfunds` — Fed funds rate
  - `sofr` — SOFR overnight rate
  - `vix` — VIX index
  - `hy_spread` — HY OAS (BAMLH0A0HYM2)
  - `dxy` — US Dollar index
  - `oil` — WTI crude price
  - `cot_lev_net` — COT Leveraged Fund net position
  - `acm_tp` — ACM term premium (NY Fed)
  - `farbast` — Foreign custody holdings
  - `rrp` — Fed RRP usage
  - `de10y`, `jp10y` — German and Japanese 10Y yields

### Step 2: Signal Computation (`signals.py`)
- Each function: `compute_SXX(df: pd.DataFrame) -> pd.Series`
- Returns daily position series, DatetimeIndex aligned, values in {-1, 0, 1}
- All signals z-score normalized before thresholding where applicable
- Continuous signals discretized via `signal_from_score(score, bull=0.3, bear=-0.3)`

### Step 3: Backtest (`backtest_runner.py`)
- `compute_metrics(returns, signal)` — from existing backtest_engine.py
- `walk_forward_backtest(returns, signal)` — 504d IS / 63d OOS / 63d step
- Buy-and-hold: `pd.Series(1, index=returns.index)` as signal
- Rolling Sharpe: 252d rolling window on strategy daily returns

### Step 4: Report (`report.py`)
- Per-signal: dual equity curve chart + rolling Sharpe subplot
- Master table CSV + HTML
- Correlation heatmap of signal returns
- DSR table using `four-eye-backtest-audit/scripts/deflated_sharpe_ratio.py`

## Package Layering

```
report.py
    └── backtest_runner.py
            ├── signals.py
            │       └── (scipy, sklearn, statsmodels, hmmlearn, etc.)
            └── data_layer.py
                    └── (fredapi, yfinance, cot_reports, requests)
```

## Key Functions

| Function | File | Purpose |
|:---|:---|:---|
| `download_all_data()` | data_layer.py | Downloads all series, returns master_df |
| `evaluate_sharpe(signal, returns)` | data_layer.py | IMMUTABLE metric for autoresearch loop |
| `compute_buy_and_hold(returns)` | data_layer.py | IMMUTABLE B&H benchmark |
| `compute_metrics(returns, signal)` | backtest_runner.py | Full performance metrics dict |
| `walk_forward_backtest(returns, signal)` | backtest_runner.py | IS/OOS walk-forward |
| `signal_from_score(score, bull, bear)` | signals.py | Continuous → discrete signal |
| `zscore(series, window)` | signals.py | Rolling z-score helper |

## Conventions

- All yields in percent (%) not decimal
- All returns in decimal (0.01 = 1%), not percent
- Signal applied with `.shift(1)` — never use same-day signal on same-day return
- Forward-fill weekly/monthly data to daily frequency before use
- Z-score window: 252d default (1 year of trading days)
- Walk-forward: IS=504d, OOS=63d, step=63d (inherited from backtest_engine.py)
