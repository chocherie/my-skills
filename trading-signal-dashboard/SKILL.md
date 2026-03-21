---
name: trading-signal-dashboard
description: Build multi-dimensional trading signal dashboards with walk-forward backtesting. Use when creating trading signal systems, backtesting strategies with in-sample/out-of-sample validation, integrating Bloomberg or Yahoo Finance data, or building financial dashboards. Supports gold and adaptable to other assets. Includes execution timing audit to detect look-ahead bias from asynchronous data settlement times.
---

# Trading Signal Dashboard

Build a multi-dimensional trading signal system with walk-forward backtesting and an interactive web dashboard. Originally designed for gold (XAU/USD), adaptable to any asset.

## Process Overview

1. **Data Acquisition** — Gather price + multi-dimensional data (Bloomberg or Yahoo Finance)
2. **Signal Generation** — Produce independent signals per dimension
3. **Walk-Forward Backtest** — IS/OOS validation with overfit detection
4. **Website Dashboard** — Interactive display of signals and backtest results
5. **Data Quality Audit** — Assess data completeness and signal timeframes
6. **Execution Timing Audit** — Verify no look-ahead bias from asynchronous settlement times

## Step 1: Data Acquisition

### With Bloomberg Terminal

Ask user to export data per `references/bloomberg-data-guide.md`. The confirmed COT tickers are `CFFDUMML/S/N Index` (Managed Money) and `CFFDUPML/S/N Index` (Producer). The GPR Index is free from matteoiacoviello.com/gpr.htm — NOT on Bloomberg.

Provide user with a copy-pasteable Excel table of required tickers (see `references/Bloomberg_Data_Export_List_CORRECTED.xlsx`).

### Without Bloomberg (Yahoo Finance Only)

Run `scripts/integrate_bloomberg.py` with `BLOOMBERG_AVAILABLE=False`. It pulls all data from Yahoo Finance with proxy indicators:
- Gold: `GC=F`, DXY: `DX-Y.NYB`, VIX: `^VIX`, Crude: `CL=F`
- Silver: `SI=F`, JPY: `JPY=X`, BTC: `BTC-USD`
- Treasury: `^TNX`, `^FVX`, TIPS: `TIP`
- ETFs: `GLD`, `IAU`

### Data Integration

```bash
pip install yfinance fredapi scikit-learn openpyxl xlrd
python scripts/integrate_bloomberg.py
```

The script merges Bloomberg exports (if available) with Yahoo Finance data, forward-fills gaps, and outputs unified CSVs. Target: 20+ years (2005-present) for regime diversity.

## Step 2: Signal Generation

Run `scripts/signal_engine_v2.py` to generate signals for all 8 dimensions.

```bash
python scripts/signal_engine_v2.py
```

Each dimension produces: `score` (float), `signal` (+1/0/-1), `confidence` (0-100%). See `references/signal-dimensions.md` for sub-indicator details and weights.

**Key design principles:**
- Z-score normalize all indicators over 252-day rolling window
- Composite = weighted sum of sub-indicator z-scores
- Discretize: >0.5 → Bullish, <-0.5 → Bearish, else Neutral
- Forward return = next-day close-to-close (for backtest alignment)

## Step 3: Walk-Forward Backtest

Run `scripts/backtest_engine.py` for walk-forward validation.

```bash
python scripts/backtest_engine.py
```

Default config: 504-day IS / 63-day OOS / 63-day step. See `references/backtesting-methodology.md` for methodology, overfit ratio formula, and anti-overfitting checklist.

**Critical: always report OOS Sharpe alongside IS Sharpe.** Flag dimensions with overfit ratio > 0.7 as unreliable.

## Step 4: Website Dashboard

Generate JSON data for the website:

```bash
python scripts/generate_charts.py
```

Copy output JSONs to `client/public/data/`. **Important:** sanitize NaN values before writing JSON — use `content.replace('NaN', '0')` or a custom JSON encoder.

Build the dashboard as a React + Tailwind static site. See `references/dashboard-design.md` for the "Obsidian Command Center" design system, component list, and JSON data format.

Required pages:
- **Home**: Gold price chart, consensus bar, 8 dimension cards (expandable), backtest comparison table
- **Methodology**: Data quality audit, signal timeframe analysis, data source assessment

## Step 5: Data Quality Audit

Run `scripts/data_quality_audit.py` to assess data completeness, frequency alignment, and signal holding periods.

```bash
python scripts/data_quality_audit.py
```

## Step 6: Execution Timing Audit

**This step is critical.** Multi-dimensional strategies combine data from instruments that settle at different times of day. If the backtest assumes entry at the asset's close but signal inputs settle hours later, the backtest contains look-ahead bias.

See `references/execution-timing-audit.md` for the full fixing times table (in SGT with DST adjustments), dimension feasibility analysis, and corrected backtest approach.

**Audit checklist (mandatory before reporting results):**

1. List every ticker used across all dimensions with its official settlement time
2. Identify the latest-settling input across all dimensions
3. Compare that time to the assumed entry price timestamp
4. If entry < latest input → **look-ahead bias confirmed** — re-run backtest with corrected entry prices
5. For gold: only Dim 1 (Technical) and Dim 6 (Seasonality) are feasible at the 1:30 PM ET gold close; the remaining 6 dimensions require data settling up to 3.5 hours later
6. Corrected entry: use Globex session reopen (6:00 PM ET) instead of settlement close (1:30 PM ET)

**Negative overfit ratios** (OOS Sharpe > IS Sharpe) should be treated with extreme skepticism. Common causes: look-ahead bias, OOS window selection bias, or momentum-following weight allocation (e.g., EMA-weighted combiner that chases recent winners).

## Adapting to Other Assets

To adapt from gold to another asset:

1. **Data**: Change `GC=F` to the target asset ticker. Replace gold-specific intermarket pairs (e.g., gold-silver ratio → relevant cross-asset pair).
2. **Signals**: Dimension 6 (Seasonality) needs asset-specific cultural events. Dimension 5 (COT) needs the correct CFTC contract code. Other dimensions are generic.
3. **Parameters**: RSI/MACD/BB parameters may need re-optimization via the walk-forward framework.
4. **Timing**: Re-run the execution timing audit for the new asset's settlement time. Different assets settle at different times (e.g., crude oil at 2:30 PM ET vs gold at 1:30 PM ET).

## Signal Combination (Phase 3-4, Advanced)

When combining dimensions into a single signal:
- **Avoid fixed-weight linear combination** — it overfits and ignores regime changes
- **Preferred: walk-forward ML ensemble** (Random Forest or Gradient Boosting) trained on IS, evaluated on OOS
- **Baseline: majority vote** — simple, robust, hard to overfit
- **Always compare combined signal OOS performance against individual best dimension**
- **Watch for multi-collinearity**: if multiple dimensions use the same underlying tickers (e.g., Real Yields in Dim 2 and Dim 4, VIX in Dim 4 and Dim 7), the combiner effectively over-weights that single market signal

## Common Pitfalls

1. **NaN in JSON**: Python `json.dump` writes `NaN` which breaks browser JSON.parse. Always sanitize.
2. **COT data is weekly**: Forward-fill to daily before merging with daily signals.
3. **GPR Index not on Bloomberg**: Free download from matteoiacoviello.com. Updated weekly on Mondays, not daily.
4. **Short backtest period**: <10 years captures only 1-2 regimes. Target 20+ years.
5. **News & Sentiment outliers**: Extremely high returns (>1000%) warrant stress-testing with transaction costs.
6. **Overfit ratio > 1.0**: Means OOS is negative while IS is positive — the signal is harmful in practice.
7. **Look-ahead bias from asynchronous settlements**: The gold close (1:30 PM ET) precedes ETF close (4:00 PM), VIX close (4:15 PM), and forex close (5:00 PM) by up to 3.5 hours. Using these as signal inputs while assuming entry at the gold close is look-ahead bias. Always use the corrected entry price (Globex reopen at 6:00 PM ET or later).
8. **Zero transaction costs**: Always stress-test with realistic costs. At ~74 trades/year and $0.60/trade, annual drag is ~2.2%. Calculate the "Terminal Spread" — the maximum cost at which the strategy breaks even.
9. **No circuit breaker for extreme events**: Z-score normalization over rolling windows (e.g., 120-day) will spike during Black Swans, then suppress subsequent readings for months as the inflated standard deviation persists. Consider capping Z-scores or using adaptive windows.
10. **Dimension 5 (Institutional Flows) uses T+1 data**: ETF shares outstanding and AUM are published the morning after the trading day. The backtest must account for this additional 1-day lag.
