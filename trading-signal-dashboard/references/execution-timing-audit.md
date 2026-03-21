# Execution Timing Audit

## Purpose

Validate that all signal inputs are available before the strategy's assumed entry price is determined. This prevents look-ahead bias where the backtest uses data that would not exist at execution time.

## The Core Problem

Multi-dimensional strategies combine data from instruments that settle at different times. If the backtest assumes entry at the gold close (1:30 PM ET), but some signal inputs don't settle until 5:00 PM ET, the backtest implicitly uses future information.

## Gold Futures Fixing Times (SGT = UTC+8)

| Instrument | Exchange | Dimensions | ET Settlement | SGT (Summer/DST) | SGT (Winter/Std) |
|---|---|---|---|---|---|
| Gold Futures (GC) | COMEX | 1,4,8 | 1:30 PM | 1:30 AM +1d | 2:30 AM +1d |
| Silver Futures (SI) | COMEX | 8 | 1:25 PM | 1:25 AM +1d | 2:25 AM +1d |
| Crude Oil (CL) | NYMEX | 4 | 2:30 PM | 2:30 AM +1d | 3:30 AM +1d |
| DXY | ICE | 4 | 3:00 PM | 3:00 AM +1d | 4:00 AM +1d |
| GLD/IAU/TIP ETFs | NYSE Arca | 2,3,5 | 4:00 PM | 4:00 AM +1d | 5:00 AM +1d |
| VIX / GVZ | CBOE | 3,4,7 | 4:15 PM | 4:15 AM +1d | 5:15 AM +1d |
| TNX (10Y Yield) | CBOE | 2,4 | 3:00 PM | 3:00 AM +1d | 4:00 AM +1d |
| 10Y Real Yield | US Treasury | 2,4 | 3:30 PM | 3:30 AM +1d | 4:30 AM +1d |
| USD/JPY | OTC Forex | 4 | 5:00 PM | 5:00 AM +1d | 6:00 AM +1d |
| Econ Surprise | Citi/BBG | 2,3 | 5:00 PM | 5:00 AM +1d | 6:00 AM +1d |
| Fed Funds Rate | NY Fed | 2 | 9:00 AM | 9:00 PM same | 10:00 PM same |
| ETF Shares Out. | SSGA | 5 | 8:00 AM | 8:00 PM same | 9:00 PM same |
| CFTC COT | CFTC | 5 | 3:30 PM Fri | 3:30 AM Sat | 4:30 AM Sat |
| GPR Index | Fed Board | 7 | Weekly Mon | N/A | N/A |

US DST: 2nd Sunday Mar → 1st Sunday Nov. During DST: SGT = ET + 12h. Standard: SGT = ET + 13h.

## Dimension Feasibility at Gold Close (1:30 PM ET)

| Dimension | Latest Input | Settles | Feasible? |
|---|---|---|---|
| 1: Technical | Gold Close | 1:30 PM | Yes |
| 2: Macro | CESIUSD / TIP | 5:00 PM / 4:00 PM | **No** (+3.5h) |
| 3: Sentiment | GVZ / CESIUSD | 4:15 PM / 5:00 PM | **No** (+3.5h) |
| 4: Intermarket | USD/JPY | 5:00 PM | **No** (+3.5h) |
| 5: Inst. Flows | Shares Out. | Next AM 8:00 | **No** (+18.5h) |
| 6: Seasonality | Calendar | N/A | Yes |
| 7: Geopolitical | GVZ / VIX | 4:15 PM | **No** (+2.75h) |
| 8: Mkt Structure | OI / Volume | ~midnight CT | **No** (+11.5h) |

Only 2 of 8 dimensions are feasible at gold close. The combined signal requires waiting until at least 5:00 PM ET.

## Corrected Backtest Approach

Replace close-to-close returns with next-session-open returns:

```python
# WRONG: assumes entry at gold close (1:30 PM) with signals using 5:00 PM data
strat_returns = signals.shift(1) * gold_close_returns

# CORRECT: entry at Globex reopen (6:00 PM ET) after all data available
strat_returns = signals.shift(1) * globex_open_returns
```

Where `globex_open_returns` = return from Day N+1 Globex open (6 PM ET) to Day N+2 Globex open.

## Audit Checklist

1. List every ticker/data source used across all dimensions
2. Record each source's official settlement/publication time
3. Identify the latest-settling input across all dimensions
4. Compare that time to the assumed entry price timestamp
5. If entry price timestamp < latest input timestamp → **look-ahead bias confirmed**
6. Quantify the gap in hours and assess correlation impact
7. Re-run backtest with corrected entry prices if bias found
