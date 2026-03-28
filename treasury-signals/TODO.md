# TODO: 10Y UST Signal Dashboard

**Current Phase:** Phase 1 — Foundation
**Last Updated:** 2026-03-28
**Overall Progress:** 5%

---

## Quick Status

| Phase | Status | Progress |
|:---|:---|:---|
| Phase 0: Project Scaffold | COMPLETE | 100% |
| Phase 1: Foundation | IN PROGRESS | 0% |
| Phase 2: Signals A–C | Pending | 0% |
| Phase 3: Signals D–F | Pending | 0% |
| Phase 4: Backtest + Report | Pending | 0% |
| Phase 5: autoresearch Loop | Pending | 0% |

---

## Phase 0: Project Scaffold — COMPLETE

- [x] Create directory structure
- [x] Write AGENTS.md
- [x] Write docs/architecture.md
- [x] Write docs/core-beliefs.md
- [x] Write docs/quality.md
- [x] Write specs/README.md
- [x] Write specs/signals.md (all 25 signals)
- [x] Write specs/data-layer.md
- [x] Write specs/backtest-engine.md
- [x] Write specs/testing-strategy.md
- [x] Write .cursor/rules/ (3 files)
- [x] Write .agents/workflows/ (3 files)
- [x] Write TODO.md

---

## Phase 1: Foundation — IN PROGRESS

**Relevant specs:** `specs/data-layer.md`

### 1.1 pyproject.toml
- [ ] Create pyproject.toml with all dependencies
- [ ] Verify install: `pip install -e .`

### 1.2 data_layer.py
- [ ] FRED API download for all 30 series
- [ ] yfinance download (IEF, VIX, oil)
- [ ] COT download via `cot_reports`
- [ ] NY Fed ACM term premium download
- [ ] FOMC minutes scrape + VADER scoring
- [ ] Primary dealer data download (NY Fed)
- [ ] Build master_df with aligned daily index
- [ ] Implement `evaluate_sharpe()` (IMMUTABLE)
- [ ] Implement `compute_buy_and_hold()` (IMMUTABLE)
- [ ] Implement `ust_return_approx` (duration-based)
- [ ] Test: `python data_layer.py` prints all series OK

### 1.3 Verification
- [ ] All 30 series present in master_df
- [ ] Date range 2003-01-01 to today
- [ ] IEF return sign check (2022 should be negative)
- [ ] No NaN gaps in core series

---

## Phase 2: Signals A–C — Pending

**Relevant specs:** `specs/signals.md` Category A, B, C

### 2.1 Category A: Momentum (S01–S05)
- [ ] S01 — 3M Momentum
- [ ] S02 — 6M Momentum
- [ ] S03 — 12-1 Month Momentum
- [ ] S04 — SMA Crossover (50d/200d)
- [ ] S05 — RSI-14 on Yield

### 2.2 Category B: Macro (S06–S10)
- [ ] S06 — Carry (Yield Curve Slope)
- [ ] S07 — Breakeven Inflation Momentum
- [ ] S08 — ACM Term Premium Deviation
- [ ] S09 — TIPS Real Yield
- [ ] S10 — CPI Momentum Proxy

### 2.3 Category C: Cross-Asset (S11–S14)
- [ ] S11 — VIX Flight-to-Safety
- [ ] S12 — HY Credit Spread
- [ ] S13 — DXY Momentum
- [ ] S14 — Oil Momentum

### 2.4 Verification (via `.agents/workflows/implement-and-verify.md`)
- [ ] Sanity checks pass for all S01–S14
- [ ] No directional bias (each signal >10% long and >10% short)
- [ ] Initial backtest run for S01–S14
- [ ] Update docs/quality.md grades

---

## Phase 3: Signals D–F — Pending

**Relevant specs:** `specs/signals.md` Category D, E, F

### 3.1 Category D: Positioning (S15–S17)
- [ ] S15 — COT Leveraged Fund Percentile
- [ ] S16 — Primary Dealer Net Positioning
- [ ] S17 — Fed RRP Usage Filter

### 3.2 Category E: Statistical (S18–S22)
- [ ] S18 — PCA Slope Factor
- [ ] S19 — Kalman Dynamic Carry Spread
- [ ] S20 — HMM Regime Signal
- [ ] S21 — NSS Curve Deviation
- [ ] S22 — VECM Error Correction

### 3.3 Category F: Global/NLP (S23–S25)
- [ ] S23 — Foreign Custody Holdings
- [ ] S24 — Global Bond Factor Residual
- [ ] S25 — FOMC Minutes Sentiment

### 3.4 Verification
- [ ] All statistical signals converge without errors
- [ ] HMM identifies recognizable regimes (2008, 2022)
- [ ] FOMC signal updates correctly on 8 annual release dates
- [ ] Update docs/quality.md grades

---

## Phase 4: Backtest + Report — Pending

### 4.1 backtest_runner.py
- [ ] Wire all 25 signals into unified runner
- [ ] compute_metrics() for each signal
- [ ] walk_forward_backtest() for each signal
- [ ] Buy-and-hold comparison
- [ ] Rolling 252d Sharpe
- [ ] Printed summary block with grep-able metrics
- [ ] Save results/ CSV files

### 4.2 report.py
- [ ] Per-signal equity curve chart (strategy vs B&H)
- [ ] Rolling Sharpe subplot
- [ ] Master comparison table (HTML + CSV)
- [ ] Signal correlation heatmap
- [ ] DSR table (using four-eye-backtest-audit)
- [ ] Regime-segmented performance (2003-2020 vs 2021-2023)

### 4.3 Verification
- [ ] `python backtest_runner.py` completes without error
- [ ] `python report.py` generates results/report.html
- [ ] At least 15/25 signals have positive OOS Sharpe
- [ ] DSR passes for composite signal

---

## Phase 5: autoresearch Loop — Pending

### 5.1 program.md
- [ ] Write autoresearch agent loop instructions (treasury-adapted)
- [ ] Define composite metric (equal-weight OOS Sharpe across all signals)
- [ ] Define keep/discard rules
- [ ] Test one loop iteration manually

### 5.2 results.tsv
- [ ] Initialize with header row
- [ ] Record baseline composite Sharpe

---

## Blocked / On Hold

| Task | Blocker | Waiting On |
|:---|:---|:---|
| S25 FinBERT upgrade | transformers install size | Optional enhancement |
| S16 Primary Dealer | Manual NY Fed download | FR 2004 form scraper needed |
| Intraday signals | No intraday data source | Out of scope for now |
