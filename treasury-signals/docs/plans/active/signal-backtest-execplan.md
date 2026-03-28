# ExecPlan: 10Y UST Signal Backtest Implementation

**Purpose:** Implement all 25 signals, backtest engine, and report generator.
**Started:** 2026-03-28
**Status:** In Progress

## Current State
- Project scaffold complete (AGENTS.md, specs/, docs/, workflows/, rules/)
- No Python code yet

## Work Plan

| Milestone | Files | Status |
|:---|:---|:---|
| 1. pyproject.toml + data_layer.py | pyproject.toml, data_layer.py | Pending |
| 2. Signals A–C (S01–S14) | signals.py | Pending |
| 3. Signals D–F (S15–S25) | signals.py | Pending |
| 4. Backtest runner | backtest_runner.py | Pending |
| 5. Report generator | report.py | Pending |
| 6. autoresearch loop | program.md, results.tsv | Pending |

## Validation Criteria

- `python data_layer.py` → prints 30 series, no errors
- `python backtest_runner.py` → prints 25-signal summary table
- `python report.py` → generates results/report.html
- At least 15/25 signals with positive OOS Sharpe
- Buy-and-hold Sharpe over 2003–2026 ≈ 0.4–0.6 (sanity check)

## Surprises / Decisions

| Date | Item | Resolution |
|:---|:---|:---|
| 2026-03-28 | IEF return methodology | Using IEF for primary, DGS10-duration for extended history |
| 2026-03-28 | COT data lag | 3-day lag modeled with extra shift in S15 |
| 2026-03-28 | Primary dealer data | NY Fed CSV scraper to be built in data_layer.py |

## Progress Notes

_(Update as implementation proceeds)_
