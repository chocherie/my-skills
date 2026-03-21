# Skeptical Questions Framework

## Table of Contents

1. [Overview](#overview)
2. [The Five Core Questions](#the-five-core-questions)
3. [Execution Timing Audit Methodology](#execution-timing-audit-methodology)
4. [Data Source Timestamp Integrity](#data-source-timestamp-integrity)
5. [Candidate Signal Evaluation Framework](#candidate-signal-evaluation-framework)

## Overview

When auditing a multi-dimensional trading signal strategy, apply these five structured questions before accepting any backtest result. Each question targets a specific class of bias that standard statistical tests (DSR, overfit ratio) cannot detect.

## The Five Core Questions

### Q1: Entry Price Feasibility

> "At the exact moment the backtest records an entry, do all signal inputs already exist?"

Trace the clock: list every instrument's official settlement time, find the latest one, and compare it to the assumed entry price timestamp. If the entry precedes any input, the backtest uses future data. Run `scripts/execution_timing_audit.py` with a config JSON listing all instruments and their settlement times.

**Red flag:** Entry at asset close (e.g., gold 1:30 PM ET) while using inputs that settle later (e.g., VIX at 4:15 PM ET, forex at 5:00 PM ET).

### Q2: Transaction Cost Sensitivity

> "What is the maximum per-trade cost at which this strategy breaks even against buy-and-hold?"

Compute the "Terminal Spread": the cost per trade at which net annual return equals buy-and-hold annual return.

```
Terminal Spread = (Strategy Annual Return - BnH Annual Return) / Annual Trades
```

If the Terminal Spread is below the realistic round-trip cost for the asset class (see `references/statistical-checks.md` Section 5), the strategy is unprofitable in practice.

### Q3: Negative Overfit Ratio Mechanism

> "If OOS performance exceeds IS performance, what specific mechanism explains this?"

A negative overfit ratio is extraordinary. Legitimate causes are rare (e.g., OOS period happened to be a perfect trend). Common illegitimate causes:

- **Look-ahead bias** — future data leaking into signal computation
- **Adaptive momentum allocation** — EMA-weighted combiners that chase recent winners (not genuine generalization)
- **Intra-step look-ahead in adaptive weights** — EMA weight update uses current OOS window data before generating signals for that same window (see `references/oos-weighted-lookahead-fix.md`). Confirmed to inflate returns by 170x in a 9-dimension gold strategy.
- **OOS window selection bias** — cherry-picked OOS period that happens to be favorable

Demand a specific, falsifiable explanation. If the answer is "the model generalizes well," that is not sufficient.

### Q4: Multi-Collinearity Across Dimensions

> "How many dimensions share the same underlying ticker, and does the combiner account for this?"

List every ticker and which dimensions use it. If the same ticker appears in 3+ dimensions, the combiner is effectively triple-counting one market signal. Common offenders: Real Yields (macro + intermarket), VIX (volatility + intermarket + geopolitical), DXY (macro + intermarket).

**Fix:** Compute the pairwise correlation matrix of dimension signals. If any pair exceeds 0.6 correlation, consider merging them or capping combined weight.

### Q5: Extreme Event Handling

> "What happens to the signal when volatility exceeds the historical bounds of the training data?"

Z-score normalization over rolling windows (e.g., 60-120 days) has a known failure mode during Black Swans: the Z-score spikes to an extreme level, then the inflated standard deviation suppresses all subsequent readings for months.

Check: Does the strategy have a circuit breaker, Z-score cap, or adaptive window? If not, the signal is unreliable during the most important market events.

## Execution Timing Audit Methodology

For any multi-source strategy, build a fixing times table:

1. **List every data source** with its official settlement/close time in UTC
2. **Convert to the trader's timezone** (e.g., SGT = UTC+8) with DST adjustments
3. **Identify the latest-settling input** across all dimensions
4. **Compare to the assumed entry time** in the backtest
5. **If entry < latest input:** look-ahead bias is confirmed

Use `scripts/execution_timing_audit.py` to automate this. Input is a JSON config with instrument names, tickers, settlement times (UTC decimal hours), and which dimensions use each instrument.

**Key settlement times for gold strategies (UTC):**

| Instrument | Winter UTC | Summer UTC | Notes |
|---|---|---|---|
| COMEX Gold | 18:30 | 17:30 | CME official settlement |
| COMEX Silver | 18:25 | 17:25 | CME official settlement |
| Crude Oil | 19:30 | 18:30 | NYMEX settlement |
| DXY, TNX | 20:00 | 19:00 | ICE/CBOT close |
| GLD, IAU, TIP | 21:00 | 20:00 | NYSE/Arca close |
| VIX, GVZ | 21:15 | 20:15 | CBOE close |
| USD/JPY | 22:00 | 21:00 | Forex daily close (5 PM ET) |

## Data Source Timestamp Integrity

Yahoo Finance daily "Close" for futures contracts (e.g., `GC=F`) is **not** the official settlement price. It is a floating snapshot taken at an inconsistent time (observed range: 3:30 PM to 9:00 PM ET). This makes backtests non-reproducible and introduces random timing noise.

**Preferred data sources for precise timestamps:**
- **CME DataMine** — Official settlement prices with exact timestamps
- **Bloomberg** — `PX_SETTLE` for futures, `PX_LAST` for spot with known close times
- **LBMA Gold Price** — AM (10:30 London) and PM (3:00 PM London) fixings
- **Globex session data** — 5:00 PM ET close available via most data vendors

When replacing an ambiguous source with a precise one, always re-run the full signal engine and backtest to measure the impact.

## Candidate Signal Evaluation Framework

When testing new data (e.g., options implied volatility, risk reversals, sentiment indices) as potential strategy improvements:

1. **Construct multiple signal variants** from the raw data (level, Z-score, momentum, regime-filtered)
2. **Run full-period backtest** for each variant to identify promising candidates
3. **Run walk-forward validation** with the same IS/OOS config as the existing strategy
4. **Apply classification thresholds:**

| OOS Sharpe | Overfit Ratio | Classification |
|---|---|---|
| > 0.3 | < 0.5 | STRONG — Add to strategy |
| > 0.1 | < 0.7 | MODERATE — Consider with lower weight |
| > 0.0 | < 1.0 | WEAK — Monitor before adding |
| ≤ 0.0 | any | REJECT — No OOS value |

5. **Check orthogonality** — compute correlation between the candidate signal and all existing dimension signals. If correlation > 0.6 with any existing dimension, the new signal adds redundant information.

Use `scripts/new_signal_tester.py` to automate steps 2-4.
