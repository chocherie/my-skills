# 10Y US Treasury Signal Dashboard

Autonomous signal research system for the 10-year US Treasury bond. Implements 25 individual
signals across 6 categories, backtests each vs buy-and-hold, and uses an autoresearch loop
(program.md) to autonomously optimize signal parameters overnight.

## Non-Negotiable Rules

These apply to every task, every time. Also enforced by `.cursor/rules/`.

1. **Read the spec before coding.** Before editing any signal, read `specs/signals.md`. Before
   changing data logic, read `specs/data-layer.md`. Specs are the source of truth.
2. **No look-ahead bias.** Always apply `signal.shift(1)` before multiplying by returns. Using
   same-day data to generate same-day signal is a critical error.
3. **Never modify `data_layer.py` evaluation functions.** `evaluate_sharpe()` and
   `compute_buy_and_hold()` are immutable ground-truth metrics (like `evaluate_bpb` in autoresearch).
4. **Update docs after every change.** After adding/tuning a signal: update `TODO.md`,
   `docs/quality.md`, and `specs/README.md` verification status.
5. **Write all artifacts to disk.** Results go in `results/`. Plans go in `docs/plans/active/`.
   Never leave important output only in conversation.

## Repository Map

| Document | Purpose | Read when... |
|:---|:---|:---|
| `docs/architecture.md` | Data flow, file layout, key functions | Understanding project structure |
| `docs/core-beliefs.md` | Signal research principles | Making design decisions |
| `docs/quality.md` | Per-signal quality scorecard | Knowing what's solid vs. gaps |
| `specs/README.md` | Index of all specs with status | Before implementing/modifying |
| `specs/signals.md` | All 25 signals: params, logic, output contract | Before editing any signal |
| `specs/data-layer.md` | Data sources, series, frequencies | Before changing data pipeline |
| `specs/backtest-engine.md` | Walk-forward config, metrics, B&H comparison | Before changing backtest logic |
| `TODO.md` | Phased checklist | What to work on next |

## Execution Plans

| Location | Purpose |
|:---|:---|
| `docs/plans/active/` | Living plans for work touching 2+ signals or files |
| `docs/plans/completed/` | Finished plans for retrospective |

## Workflows

| Workflow | When to use |
|:---|:---|
| `.agents/workflows/implement-and-verify.md` | After implementing each signal batch |
| `.agents/workflows/add-new-signal.md` | autoresearch loop: adding/tuning a signal |
| `.agents/workflows/run-backtest.md` | Running full backtest + interpreting results |

## Commands

| Command | Purpose |
|:---|:---|
| `python data_layer.py` | Download / refresh all data |
| `python backtest_runner.py` | Run all 25 signal backtests, save results/ |
| `python report.py` | Generate HTML report + charts |
| `python backtest_runner.py > run.log 2>&1` | autoresearch loop run (redirect output) |
| `grep "^composite_sharpe:" run.log` | Extract key metric from log |

## Key Files

| File | Role | Editable by agent? |
|:---|:---|:---|
| `data_layer.py` | Data download + immutable evaluator | NO |
| `signals.py` | All 25 signal implementations + params | YES |
| `backtest_runner.py` | Wires signals → backtest → results | YES (carefully) |
| `report.py` | Generates charts + comparison table | YES |
| `program.md` | autoresearch loop instructions | Human edits to steer |
| `results.tsv` | Experiment log (untracked by git) | Auto-updated by loop |

## Environment Variables

| Variable | Purpose | Required |
|:---|:---|:---|
| `FRED_API_KEY` | FRED API access for yield/macro data | YES |
