# Testing Strategy

## Validation Layers

### Layer 1: Data Integrity
Run `data_layer.py` standalone and verify:
- [ ] All 30 series downloaded without error
- [ ] No NaN gaps except expected (pre-series-start dates)
- [ ] IEF returns have correct sign (yield up → IEF down)
- [ ] COT data forward-filled correctly (weekly → daily)

### Layer 2: Signal Sanity Checks
For each signal:
- [ ] No future data used (apply `bias_detector.py` from four-eye-backtest-audit)
- [ ] Warmup period respected (first N values are NaN)
- [ ] Values strictly in {-1, 0, 1} after discretization
- [ ] Directional bias: long% and short% both > 10% (not all-long or all-short)

### Layer 3: Backtest Validation
- [ ] OOS equity curve stitches correctly (no jumps at window boundaries)
- [ ] B&H benchmark matches IEF total return over same period
- [ ] Signals with expected Sharpe < 0.1 flagged as weak candidates
- [ ] Overfit ratio > 0.7 → signal rejected

### Layer 4: Statistical Significance
- [ ] DSR > 0.5 for individual signals (using `deflated_sharpe_ratio.py`)
- [ ] DSR > 0.95 for composite signal
- [ ] No signal with OOS Sharpe < 0 included in composite

## Execution Timing Audit
Run `execution_timing_audit.py` from four-eye-backtest-audit on each signal:
- FRED data available same day (pub at 8:30am ET) → can use next-day open (shift(1) is correct)
- COT data available Friday 3:30pm ET → use Monday close at earliest (2 extra days of shift)
- FOMC minutes available 2pm ET day of release → use next day (shift(1) correct)
- IEF return computed from previous close → no look-ahead
