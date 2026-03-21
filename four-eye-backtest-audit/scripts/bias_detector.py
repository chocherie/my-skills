#!/usr/bin/env python3
"""
Systematic Bias Detector for Trading Signal Backtests

Usage:
  python bias_detector.py <signal_json> <benchmark_json>

signal_json: array of { "date": "YYYY-MM-DD", "signal": 1|-1|0 }
benchmark_json: array of { "date": "YYYY-MM-DD", "return": 0.005 }

Checks for:
  1. Directional bias (asymmetric long/short distribution)
  2. Tie-breaking bias (neutral signal handling)
  3. Regime dependence (performance only in one market direction)
  4. IS-OOS correlation (does IS predict OOS?)
"""

import json
import sys
import numpy as np


def detect_directional_bias(signals):
    """Check if signals are asymmetrically distributed."""
    sigs = [s["signal"] for s in signals]
    n = len(sigs)
    n_long = sum(1 for s in sigs if s > 0)
    n_short = sum(1 for s in sigs if s < 0)
    n_neutral = sum(1 for s in sigs if s == 0)

    long_pct = n_long / n * 100
    short_pct = n_short / n * 100
    neutral_pct = n_neutral / n * 100

    # Bias severity: >60% in one direction is moderate, >75% is severe
    max_dir = max(long_pct, short_pct)
    if max_dir > 75:
        severity = "SEVERE"
    elif max_dir > 60:
        severity = "MODERATE"
    elif max_dir > 55:
        severity = "MILD"
    else:
        severity = "NONE"

    return {
        "long_pct": round(long_pct, 1),
        "short_pct": round(short_pct, 1),
        "neutral_pct": round(neutral_pct, 1),
        "bias_direction": "LONG" if long_pct > short_pct else "SHORT",
        "severity": severity,
        "recommendation": "Remove tie-breaking rule; tied votes should be neutral (0)"
        if severity != "NONE" else "No action needed",
    }


def detect_regime_dependence(signals, returns):
    """Check if strategy only works in bull or bear regimes."""
    date_returns = {r["date"]: r["return"] for r in returns}

    bull_strat_rets = []
    bear_strat_rets = []

    # Use rolling 63-day return to classify regime
    ret_dates = sorted(date_returns.keys())
    regime_map = {}
    for i in range(63, len(ret_dates)):
        window = [date_returns[ret_dates[j]] for j in range(i - 63, i)]
        cum = np.prod([1 + r for r in window]) - 1
        regime_map[ret_dates[i]] = "bull" if cum > 0 else "bear"

    for s in signals:
        date = s["date"]
        if date in date_returns and date in regime_map:
            strat_ret = s["signal"] * date_returns[date]
            if regime_map[date] == "bull":
                bull_strat_rets.append(strat_ret)
            else:
                bear_strat_rets.append(strat_ret)

    bull_sharpe = _sharpe(bull_strat_rets)
    bear_sharpe = _sharpe(bear_strat_rets)

    if bull_sharpe > 0.5 and bear_sharpe < -0.5:
        dependence = "BULL-ONLY"
    elif bear_sharpe > 0.5 and bull_sharpe < -0.5:
        dependence = "BEAR-ONLY"
    elif abs(bull_sharpe - bear_sharpe) > 1.0:
        dependence = "MODERATE"
    else:
        dependence = "BALANCED"

    return {
        "bull_regime_sharpe": round(bull_sharpe, 4),
        "bear_regime_sharpe": round(bear_sharpe, 4),
        "dependence": dependence,
        "recommendation": "Add regime-conditional analysis to the dashboard"
        if dependence != "BALANCED" else "No action needed",
    }


def detect_is_oos_correlation(is_sharpes, oos_sharpes):
    """Check if IS performance predicts OOS performance."""
    if len(is_sharpes) < 10 or len(oos_sharpes) < 10:
        return {"correlation": 0.0, "interpretation": "insufficient data"}

    n = min(len(is_sharpes), len(oos_sharpes))
    corr = float(np.corrcoef(is_sharpes[:n], oos_sharpes[:n])[0, 1])

    if corr > 0.3:
        interp = "POSITIVE — IS has some predictive power for OOS (good)"
    elif corr > -0.1:
        interp = "NEAR-ZERO — IS does not predict OOS (neutral)"
    else:
        interp = "NEGATIVE — IS inversely predicts OOS (red flag: possible overfitting)"

    return {
        "correlation": round(corr, 4),
        "interpretation": interp,
        "recommendation": "If negative, the strategy may be curve-fitted to IS patterns that reverse OOS",
    }


def _sharpe(returns):
    if len(returns) == 0 or np.std(returns) < 1e-10:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(252))


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        signals = json.load(f)
    with open(sys.argv[2]) as f:
        returns = json.load(f)

    print("Systematic Bias Detection Report")
    print("=" * 50)

    bias = detect_directional_bias(signals)
    print(f"\n1. Directional Bias: {bias['severity']}")
    print(f"   Long: {bias['long_pct']}%  Short: {bias['short_pct']}%  Neutral: {bias['neutral_pct']}%")
    print(f"   -> {bias['recommendation']}")

    regime = detect_regime_dependence(signals, returns)
    print(f"\n2. Regime Dependence: {regime['dependence']}")
    print(f"   Bull Sharpe: {regime['bull_regime_sharpe']}  Bear Sharpe: {regime['bear_regime_sharpe']}")
    print(f"   -> {regime['recommendation']}")


if __name__ == "__main__":
    main()
