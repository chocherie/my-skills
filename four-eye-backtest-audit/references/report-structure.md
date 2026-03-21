# Four-Eye Assessment Report Structure

Use this template for the final assessment report. Adapt sections to the specific strategy being audited.

## Report Template

```markdown
# Four-Eye Validation and Assessment Report

## [Strategy Name] — Trading Strategy Methodology Review

**Prepared by:** [Reviewer]
**Date:** [Date]
**Subject:** [Strategy description]
**Scope:** [What is being reviewed]

---

## Executive Summary

[One paragraph: what the strategy does, key strengths, key concerns, overall recommendation.
Be direct — state whether the strategy is suitable for live trading or analysis only.]

---

## 1. Data Integrity Assessment

### 1.1 Coverage and Completeness
[Date range, number of data points, market regimes covered, gaps]

### 1.2 Data Source Assessment
[Table of data sources with update frequency and quality risk]

### 1.3 Data Quality Issues
[Any inconsistencies found]

---

## 2. Walk-Forward Framework Assessment

### 2.1 Configuration
[Table: IS window, OOS window, step size, total steps, IS/OOS ratio, signal lag]

### 2.2 Strengths
[What the framework does correctly]

### 2.3 Limitations
[Structural limitations: cross-step parameter selection, missing transaction costs, etc.]

---

## 3. Individual Signal Assessment

### 3.1 Signal Quality Overview
[Table: each signal/dimension with OOS Sharpe, overfit ratio, classification, assessment]

### 3.2 Concerns
[Flag any signals with OOS Sharpe > 1.5 (unusually high), severe overfitting, or negative OOS]

---

## 4. Composite/Combined Signal Assessment

### 4.1 Method Comparison
[Table: each combination method with OOS Sharpe, overfit, total return, max drawdown, trades]

### 4.2 Critical Findings
[Deep analysis of the best-performing method — is the performance genuine or an artifact?]

### 4.3 Deflated Sharpe Ratio Test
[Table: each method with observed SR, DSR, pass/fail. State N_trials used.]

### 4.4 Bias Detection
[Directional bias, tie-breaking rules, regime dependence]

### 4.5 Transaction Cost Impact
[Table: each method with annual trades, annual cost, gross return, net return]

---

## 5. Overfitting Detection Assessment

### 5.1 Overfit Ratio
[Evaluate the overfit metric used — is it correctly computed? Are thresholds reasonable?]

### 5.2 Blind Spots
[What the overfit metric does NOT capture]

### 5.3 Anti-Overfitting Safeguards
[Evaluate regularization, complexity constraints, validation methodology]

---

## 6. Execution Timing Feasibility

### 6.1 Fixing Times Table
[Table: every instrument used, official settlement time in UTC and trader's timezone, DST adjustment, which dimensions use it]

### 6.2 Entry Price Feasibility
[Compare assumed entry time to latest-settling instrument. State whether look-ahead bias exists.]

### 6.3 Corrected Entry Time
[If bias exists: state the corrected entry time, re-run results, and impact on performance]

---

## 7. Data Source Integrity

### 7.1 Price Source Audit
[Document the exact data source for the target asset's daily price. Is it an official settlement, a floating snapshot, or a benchmark fixing?]

### 7.2 Correction Impact
[If the price source was replaced with a precise fixing: before/after comparison table of key metrics]

---

## 8. Improvement Experiments

### 8.1 Candidate Signals Tested
[Table: each candidate signal, data source, signal logic, full-period Sharpe, OOS Sharpe, overfit ratio, classification]

### 8.2 Orthogonality Check
[Correlation matrix between candidate signals and existing dimensions]

### 8.3 Recommendation
[Which candidates to add, at what weight, and expected impact on composite performance]

---

## 9. Overall Risk Assessment

### 9.1 Risk Matrix
[Table: risk category, severity (High/Medium/Low), description]

### 9.2 Recommendations
[Separate assessment for: (a) as analytical tool, (b) as trading signal.
Numbered list of specific improvements.]

---

## 10. Conclusion

[Final paragraph: balanced assessment. Acknowledge strengths, state limitations clearly.
Include the standard disclaimer about past performance.]

---

## References

[Academic citations supporting the methodology]
```

## Key Principles

1. **Be honest, not diplomatic.** If a strategy fails statistical tests, say so clearly.
2. **Separate analysis value from trading value.** A dashboard can be excellent for understanding markets while its signals are not tradeable.
3. **Always quantify.** Replace "the strategy may be overfit" with "the overfit ratio is 0.43 (moderate), and the DSR is 0.32 (fails 95% test)."
4. **Provide actionable recommendations.** Each concern should have a specific fix.
5. **Include the buy-and-hold benchmark.** Every strategy must beat doing nothing.
6. **Verify the clock before the math.** No statistical test can detect look-ahead bias from asynchronous settlements. The timing audit must come first.
7. **Test improvements rigorously.** New signals must pass the same walk-forward framework as existing signals before being added.
