# Quality Scorecard: 10Y UST Signal Dashboard

Grades: A (excellent) | B (good) | C (acceptable) | D (needs work) | F (broken)

| Domain | Spec | Code | Backtest | Review | Overall | Notes |
|:---|:---|:---|:---|:---|:---|:---|
| Data Layer | — | — | — | — | — | Not yet implemented |
| Category A: Momentum (S01–S05) | — | — | — | — | — | Not yet implemented |
| Category B: Macro (S06–S10) | — | — | — | — | — | Not yet implemented |
| Category C: Cross-Asset (S11–S14) | — | — | — | — | — | Not yet implemented |
| Category D: Positioning (S15–S17) | — | — | — | — | — | Not yet implemented |
| Category E: Statistical (S18–S22) | — | — | — | — | — | Not yet implemented |
| Category F: Global/NLP (S23–S25) | — | — | — | — | — | Not yet implemented |
| Backtest Engine | — | — | — | — | — | Not yet implemented |
| Report | — | — | — | — | — | Not yet implemented |

## Score History

| Date | Event | Change |
|:---|:---|:---|
| 2026-03-28 | Project initialized | Scaffold created |

## Known Gaps

- [ ] All signals need look-ahead bias audit (execution_timing_audit.py)
- [ ] DSR calculation pending signal implementation
- [ ] COT data lag (3 days) not yet modeled in signal timing
- [ ] FOMC minutes NLP: only lexical (VADER), not FinBERT yet
- [ ] No intraday data — all signals daily frequency only
