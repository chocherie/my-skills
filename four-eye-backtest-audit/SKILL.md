---
name: four-eye-backtest-audit
description: Independent four-eye validation of trading signal backtests. Use when auditing walk-forward backtested strategies, detecting overfitting or systematic bias in signal systems, computing Deflated Sharpe Ratios for multiple testing correction, reviewing composite signal combiners, auditing execution timing for look-ahead bias from asynchronous data settlements, correcting ambiguous price sources, or evaluating candidate signals for strategy improvement. Covers data integrity, statistical significance, directional bias, regime dependence, execution timing feasibility, data source integrity, and actionable improvement recommendations.
---

# Four-Eye Backtest Audit

Independent validation workflow for trading signal backtests. Produces a structured assessment report with statistical tests, bias detection, execution timing audit, and actionable recommendations including candidate signal evaluation.

## When to Use

- User asks to "validate", "audit", "review", or "four-eye check" a trading strategy or backtest
- User wants to know if backtest results are statistically significant
- User suspects overfitting or wants to test for it rigorously
- User has a composite signal combiner and wants to compare methods honestly
- User asks about look-ahead bias, execution timing, or settlement time feasibility
- User provides new data (e.g., options, sentiment) and asks if it improves the strategy
- User questions whether the price data source is reliable or precisely timestamped

## Workflow

The audit follows 11 sequential steps. Complete each before moving to the next. Steps 1-8 are the core statistical audit. Steps 9-11 extend into execution feasibility, data correction, and improvement experiments.

### Step 1: Inventory the Strategy

Collect these facts before any analysis:
- Asset(s) traded, date range, number of data points
- Number of independent signals/dimensions
- Walk-forward configuration (IS window, OOS window, step size, total steps)
- Number of combination methods tested
- Total independent trials N (signals + methods + variants)

### Step 2: Data Integrity Check

Verify:
- Date ranges match across all data sources (no misaligned series)
- No gaps in price data that could inflate returns
- Data sources are documented with update frequency and quality risk
- Survivorship bias is not present (e.g., only using assets that still exist)

### Step 3: Walk-Forward Configuration Audit

Validate against these thresholds:

| Parameter | Minimum | Recommended | Red Flag |
|---|---|---|---|
| IS/OOS ratio | 3x | 5-8x | > 15x (IS dominates) |
| Total OOS steps | 20 | 40-80 | < 15 (insufficient) |
| OOS window | 21 days | 63 days | < 10 days |
| Signal lag | 1 day | 1 day | 0 days (look-ahead) |

Check: OOS windows must be non-overlapping. Step size should equal OOS window size.

### Step 4: Deflated Sharpe Ratio Test

Run `scripts/deflated_sharpe_ratio.py` or compute manually. See `references/statistical-checks.md` Section 1 for the full formula.

Count N_trials honestly:
```
N = (individual signals) + (combination methods) + (signal set variants) + (hyperparam sweeps)
```

Example: 8 dimensions + 3 methods x 2 signal sets = 14 trials.

**Decision rule:** If no method passes DSR at 95%, state clearly: "The observed performance could be explained by chance given the number of strategies tested."

### Step 5: Bias Detection

Check three types of bias:

**5a. Directional bias** — Count signal distribution (% bullish / bearish / neutral).
- Flag if any direction > 60%
- Flag if neutral = 0% (tie-breaking rule forcing direction)
- In a secular bull market, any long-biased strategy looks good

**5b. IS-OOS correlation** — Compute Pearson correlation between per-step IS and OOS Sharpe.
- Negative correlation = red flag (IS patterns reverse OOS)
- Near-zero = IS has no predictive power for OOS

**5c. Regime dependence** — Split performance by market regime.
- If strategy only works in bull markets, it may just be riding the trend
- Compare strategy Sharpe in bull vs bear regimes

Run `scripts/bias_detector.py` for automated checks, or compute manually.

### Step 6: Transaction Cost Sensitivity

Estimate round-trip cost for the asset class (see `references/statistical-checks.md` Section 5). Compute:
```
Net Annual Return = Gross Annual Return - (Annual Trades x Cost Per Trade)
```

Flag any method where net return turns negative. Also compute the "Terminal Spread" — the maximum per-trade cost at which the strategy still beats buy-and-hold.

### Step 7: Buy-and-Hold Benchmark

Compute buy-and-hold from the same start date as the strategy's first OOS window. Compare total return, Sharpe, and max drawdown. If the strategy underperforms buy-and-hold on all three, it adds no value.

### Step 8: Skeptical Questions

Apply the five core questions from `references/skeptical-questions.md`:
1. **Entry Price Feasibility** — Can the trade be executed at the backtested price?
2. **Transaction Cost Sensitivity** — What is the Terminal Spread?
3. **Negative Overfit Ratio Mechanism** — If OOS > IS, what specific mechanism explains it?
4. **Multi-Collinearity** — How many dimensions share the same underlying tickers?
5. **Extreme Event Handling** — Does the strategy have a circuit breaker?

Each question must have a specific, evidence-based answer. "The model generalizes well" is not sufficient.

### Step 9: Execution Timing Audit

**This step detects look-ahead bias that no statistical test can find.**

1. List every instrument used across all dimensions with its official settlement time
2. Convert all times to the trader's timezone (with DST adjustments)
3. Identify the latest-settling instrument
4. Compare to the assumed entry time in the backtest
5. If entry time < latest settlement: **look-ahead bias confirmed**

Run `scripts/execution_timing_audit.py` with a config JSON. See `references/skeptical-questions.md` Section 3 for the fixing times reference table.

**Common finding:** Multi-source strategies that use gold futures (settle 1:30 PM ET) alongside VIX (4:15 PM ET) and forex (5:00 PM ET) have 3.5 hours of look-ahead bias if the backtest assumes entry at the gold close.

**Fix:** Replace the entry price with a price at or after the latest settlement time (e.g., 5:00 PM ET Globex close for gold), then re-run the full backtest.

### Step 10: Data Source Integrity

Verify the exact timestamp of the target asset's daily price series:

1. **Identify the data source** — Is it Yahoo Finance, Bloomberg, CME DataMine, LBMA, or another vendor?
2. **Verify timestamp consistency** — Compare daily "Close" against intraday bars at known settlement times
3. **Flag ambiguous sources** — Yahoo Finance futures "Close" (e.g., `GC=F`) is NOT the official settlement; it is a floating snapshot at an inconsistent time (observed range: 3:30 PM to 9:00 PM ET)

**Preferred sources for precise timestamps:**
- CME DataMine: official settlement prices
- Bloomberg: `PX_SETTLE` for futures, `PX_LAST` for spot
- LBMA Gold Price: AM (10:30 London) and PM (3:00 PM London) fixings

If the price source is ambiguous, replace it with a precise source and re-run the full signal engine and backtest. Document the before/after impact.

### Step 11: Candidate Signal Evaluation

When new data is available (e.g., options implied volatility, risk reversals, sentiment indices):

1. **Construct multiple signal variants** from the raw data:
   - Level-based (e.g., buy when risk reversal > 0)
   - Z-score-based (e.g., buy when ATM vol Z-score < -1)
   - Momentum-based (e.g., buy when vol is declining)
   - Regime-filtered (e.g., only trade when vol is above median)

2. **Run full-period backtest** for each variant to identify promising candidates

3. **Run walk-forward validation** with the same IS/OOS config as the existing strategy. Use `scripts/new_signal_tester.py`

4. **Apply classification thresholds:**

| OOS Sharpe | Overfit Ratio | Classification |
|---|---|---|
| > 0.3 | < 0.5 | STRONG — Add to strategy |
| > 0.1 | < 0.7 | MODERATE — Consider with lower weight |
| > 0.0 | < 1.0 | WEAK — Monitor before adding |
| ≤ 0.0 | any | REJECT — No OOS value |

5. **Check orthogonality** — Compute correlation between the candidate signal and all existing dimension signals. If correlation > 0.6 with any existing dimension, the new signal adds redundant information.

### Step 12: Write the Assessment Report

Follow the template in `references/report-structure.md`. The template now includes sections for:
- Execution Timing Feasibility (Section 6)
- Data Source Integrity (Section 7)
- Improvement Experiments (Section 8)

Key principles:
- Be honest, not diplomatic — if it fails, say so
- Separate "useful for analysis" from "useful for trading"
- Quantify every claim (numbers, not adjectives)
- Every concern gets a specific, actionable fix
- Verify the clock before the math — timing audit precedes statistical tests
- Test improvements rigorously — new signals must pass the same walk-forward framework
- Always include the standard disclaimer about past performance

## Risk Severity Classification

| Severity | Criteria |
|---|---|
| **High** | Fails DSR, directional bias > 75%, net return negative after costs, or look-ahead bias confirmed |
| **Medium** | DSR 0.80-0.95, moderate bias, regime-dependent, or ambiguous price source |
| **Low** | Passes DSR, balanced signals, robust across regimes, clean timing, precise data sources |

## Common Findings and Fixes

| Finding | Fix |
|---|---|
| Tie-breaking long bias | Change tied vote from +1 to 0 (neutral) |
| All methods fail DSR | Report honestly; recommend as analysis tool, not signal generator |
| Severe overfit on one dimension | Exclude from composite or cap its weight |
| Negative IS-OOS correlation | Strategy is curve-fitted; simplify or reduce parameters |
| Extraordinary returns (>1000%) | Check for adaptive momentum, compounding artifacts, or leverage |
| Only works in bull regime | Add regime filter or compare to buy-and-hold |
| Look-ahead bias from async settlements | Replace entry price with post-settlement time; re-run full backtest |
| Yahoo Finance futures "Close" used | Replace with CME settlement, LBMA fix, or Globex close; re-run |
| Same ticker in 3+ dimensions | Compute correlation matrix; merge or cap combined weight |
| No circuit breaker for extreme events | Add Z-score cap or adaptive window for volatility normalization |
| Negative overfit ratio (OOS > IS) | Investigate: adaptive momentum allocation, not genuine generalization |
| OOS-weighted combiner uses current OOS window to set weights | Move weight computation before OOS Sharpe update; weights must use only prior steps' OOS data |
| Adaptive EMA weights with extraordinary returns (>1000%) | Check if weight update precedes or follows signal generation within each walk-forward step |

## Intra-Step Look-Ahead Bias in Adaptive Weight Combiners

This is a subtle but critical class of look-ahead bias that occurs **within** a single walk-forward step, not across steps. It is invisible to standard overfit ratio and DSR tests.

### Pattern

Adaptive weight combiners (e.g., OOS-weighted voting) update dimension weights using an EMA of per-dimension OOS Sharpe ratios. The bug occurs when:

1. The current OOS window's per-dimension Sharpe is computed
2. The EMA weights are updated with this information
3. The updated weights are used to generate signals for the **same** OOS window

This creates a feedback loop: the combiner "knows" which dimensions performed well in the current OOS window before generating signals for that window.

### Detection

- **Symptom:** OOS-weighted method has Sharpe 3-5x higher than equal-weight baseline, with negative overfit ratio
- **Symptom:** Total return >10,000% while equal-weight shows <300%
- **Symptom:** Rolling 1-year Sharpe consistently 2-3 with no negative periods
- **Diagnostic:** Compare OOS-weighted equity curve to equal-weight on log scale; if they diverge by >10x, investigate the weight update order

### Fix

In the walk-forward loop, ensure this execution order:

```
for each step:
  1. Compute weights from dim_oos_sharpes (contains ONLY prior steps' data)
  2. Generate signals using those weights
  3. Record equity and performance
  4. THEN update dim_oos_sharpes with current step's OOS performance
```

The key principle: **weights must be computed before the current step's OOS data is incorporated into the EMA.**

### Verified Impact

Gold 9-dimension strategy (2006-2026, 79 walk-forward steps):

| Metric | Before Fix | After Fix | Change |
|---|---|---|---|
| Total Return | +30,074% | +177% | -170x |
| Sharpe Ratio | 1.83 | 0.39 | -1.44 |
| OOS Avg Sharpe | 2.01 | 0.41 | -1.60 |
| Overfit Ratio | -1.88 | +0.53 | Negative to moderate |
| Max Drawdown | -16.6% | -42.8% | -26.2pp |

Equal-weight and gradient boost methods were unaffected (they do not use adaptive weights).

## Resources

- `scripts/deflated_sharpe_ratio.py` — Standalone DSR calculator (input: JSON of step Sharpes)
- `scripts/bias_detector.py` — Directional bias and regime dependence detector
- `scripts/execution_timing_audit.py` — Fixing time analysis and look-ahead bias detection (input: JSON config of instruments)
- `scripts/new_signal_tester.py` — Walk-forward evaluation of candidate signals (input: CSV of signals + returns)
- `references/statistical-checks.md` — Full formulas and interpretation tables for all statistical checks
- `references/report-structure.md` — Assessment report template with 10 sections including timing and improvement experiments
- `references/skeptical-questions.md` — The five core skeptical questions, execution timing methodology, data source integrity guide, and candidate signal evaluation framework
- `references/oos-weighted-lookahead-fix.md` — Detailed case study of the intra-step look-ahead bias fix with before/after comparison
