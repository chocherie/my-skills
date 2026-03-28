# Core Beliefs: 10Y UST Signal Research

Six principles that govern every decision in this project.

## 1. On Look-Ahead Bias

**Belief:** Look-ahead bias is the cardinal sin of backtesting. A strategy that looks even one
day into the future is worthless in production.

**Practice:**
- Always `signal.shift(1)` before multiplying by returns
- For economic data releases: use the release date + 1 business day as the signal date
- Use ALFRED (vintage data) not FRED (revised data) for macro series that are heavily revised
- Never compute a rolling mean or std that includes the current observation in the lookahead sense

## 2. On Simplicity

**Belief:** A 0.01 Sharpe improvement that adds 20 lines of hacky code is not worth it.
Removing complexity and getting equal or better results is a great outcome.

**Practice:** Prefer fewer parameters. If a signal works with a 252d window, don't grid-search
for the "optimal" window. A 3-parameter signal that backtests well is more credible than a
10-parameter signal with the same result.

## 3. On Statistical Significance

**Belief:** Most backtested strategies are overfitting. The Deflated Sharpe Ratio (DSR) must
exceed 0.95 for a signal to be considered statistically significant after multiple testing.

**Practice:**
- Count every parameter combination tested as a "trial" for DSR
- Single signal DSR threshold: >0.5 (we tested ~25 signals total)
- Composite signal DSR threshold: >0.95
- Reject signals with OOS Sharpe < 0 or overfit ratio > 0.7

## 4. On Data Quality

**Belief:** Free data is good enough for strategy development. Paid data provides incremental
improvements but the core signals should work with FRED + yfinance + CFTC.

**Practice:**
- FRED API for all macro series (free, 20+ years daily)
- IEF via yfinance for returns (free, 2002–present)
- CFTC.gov for COT (free, 1995–present)
- NY Fed website for ACM term premium (free, 1961–present)
- Never assume a signal is broken just because free data is imperfect

## 5. On Transaction Costs

**Belief:** Ignoring transaction costs turns 70% of "alpha" into illusion.

**Practice:**
- Budget 2bps per trade (IEF bid-ask + market impact)
- For ZN futures: ~1 tick ($15.63) per side = ~0.3bps
- High-turnover signals (daily RSI) must clear a higher Sharpe bar than low-turnover signals
- A signal that trades 200x/year needs >0.3 Sharpe gross to be worthwhile net

## 6. On Regime Awareness

**Belief:** A signal that worked from 1990–2020 in a falling-rate environment is not validated
until it also worked in 2022 (rising rates) and 2008 (flight to quality).

**Practice:**
- Always segment backtest by regime: falling rates (1990–2020), rising rates (2021–2023),
  crisis/flight-to-quality (2008, 2020)
- A signal that only works in one regime is a regime bet, not a signal
- The HMM regime classifier (S20) exists precisely to identify when other signals are valid
