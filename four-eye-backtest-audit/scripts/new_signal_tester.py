#!/usr/bin/env python3
"""
New Signal Walk-Forward Tester — Evaluate Candidate Signals for Strategy Improvement

Usage:
  python new_signal_tester.py <signals_csv> <returns_csv> [--is_window 504] [--oos_window 63]

signals_csv: CSV with date index and one or more signal columns (values: +1, 0, -1)
returns_csv: CSV with date index and a 'return' column (daily returns of the target asset)

Tests each signal column in walk-forward and compares to buy-and-hold.
Output: per-signal metrics (full period + walk-forward), comparison table, recommendation.
"""

import sys
import argparse
import numpy as np
import pandas as pd
import json


def compute_metrics(returns, label=""):
    """Compute standard performance metrics for a return series."""
    returns = returns.dropna()
    if len(returns) == 0 or returns.std() == 0:
        return None
    equity = (1 + returns).cumprod()
    total_ret = float(equity.iloc[-1] - 1) * 100
    n_years = len(returns) / 252
    annual_ret = float((1 + total_ret / 100) ** (1 / max(n_years, 0.01)) - 1) * 100
    sharpe = float(returns.mean() / returns.std() * np.sqrt(252))
    peak = equity.cummax()
    max_dd = float(((equity - peak) / peak).min() * 100)
    return {
        "label": label,
        "total_return_pct": round(total_ret, 2),
        "annual_return_pct": round(annual_ret, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "n_days": len(returns),
    }


def walk_forward_test(signal, returns, is_window, oos_window, step_size):
    """Run walk-forward test on a signal series."""
    n = len(signal)
    oos_sharpes = []
    is_sharpes = []
    oos_equity = []
    eq_val = 1.0

    start = 0
    while start + is_window + oos_window <= n:
        is_end = start + is_window
        oos_end = min(is_end + oos_window, n)

        # IS
        is_strat = signal.iloc[start:is_end].shift(1) * returns.iloc[start:is_end]
        is_strat = is_strat.dropna()
        is_sharpe = float(is_strat.mean() / is_strat.std() * np.sqrt(252)) if is_strat.std() > 0 else 0
        is_sharpes.append(is_sharpe)

        # OOS
        oos_strat = signal.iloc[is_end:oos_end].shift(1) * returns.iloc[is_end:oos_end]
        oos_strat = oos_strat.dropna()
        oos_sharpe = float(oos_strat.mean() / oos_strat.std() * np.sqrt(252)) if oos_strat.std() > 0 else 0
        oos_sharpes.append(oos_sharpe)

        for ret in oos_strat:
            eq_val *= (1 + ret)
            oos_equity.append(eq_val)

        start += step_size

    avg_is = np.mean(is_sharpes) if is_sharpes else 0
    avg_oos = np.mean(oos_sharpes) if oos_sharpes else 0
    overfit = round(1 - (avg_oos / avg_is), 3) if abs(avg_is) > 1e-6 else 0
    total_ret = (eq_val - 1) * 100

    return {
        "is_sharpe": round(avg_is, 3),
        "oos_sharpe": round(avg_oos, 3),
        "overfit_ratio": overfit,
        "oos_total_return_pct": round(total_ret, 2),
        "wf_steps": len(oos_sharpes),
        "is_sharpes": [round(s, 4) for s in is_sharpes],
        "oos_sharpes": [round(s, 4) for s in oos_sharpes],
    }


def classify_signal(wf_result):
    """Classify a signal based on walk-forward results."""
    oos = wf_result["oos_sharpe"]
    overfit = wf_result["overfit_ratio"]

    if oos > 0.3 and overfit < 0.5:
        return "STRONG — Add to strategy"
    elif oos > 0.1 and overfit < 0.7:
        return "MODERATE — Consider adding with lower weight"
    elif oos > 0 and overfit < 1.0:
        return "WEAK — Marginal value, monitor before adding"
    else:
        return "REJECT — No OOS value"


def main():
    parser = argparse.ArgumentParser(description="Walk-forward test for candidate signals")
    parser.add_argument("signals_csv", help="CSV with signal columns (+1/0/-1)")
    parser.add_argument("returns_csv", help="CSV with 'return' column")
    parser.add_argument("--is_window", type=int, default=504, help="In-sample window (days)")
    parser.add_argument("--oos_window", type=int, default=63, help="Out-of-sample window (days)")
    parser.add_argument("--step_size", type=int, default=63, help="Step size (days)")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    args = parser.parse_args()

    signals = pd.read_csv(args.signals_csv, index_col=0, parse_dates=True)
    returns_df = pd.read_csv(args.returns_csv, index_col=0, parse_dates=True)
    returns = returns_df["return"] if "return" in returns_df.columns else returns_df.iloc[:, 0]

    # Align
    common = signals.index.intersection(returns.index)
    signals = signals.loc[common]
    returns = returns.loc[common]

    print(f"New Signal Walk-Forward Tester")
    print(f"Data: {len(common)} days, {len(signals.columns)} candidate signals")
    print(f"Config: IS={args.is_window}d, OOS={args.oos_window}d, Step={args.step_size}d")
    print("=" * 80)

    # Buy-and-hold benchmark
    bnh = compute_metrics(returns, "Buy-and-Hold")
    print(f"\nBenchmark (Buy-and-Hold): Sharpe={bnh['sharpe']}, Return={bnh['total_return_pct']:.1f}%")

    # Test each signal
    all_results = []
    for col in signals.columns:
        sig = signals[col]
        full = compute_metrics(sig.shift(1) * returns, col)
        wf = walk_forward_test(sig, returns, args.is_window, args.oos_window, args.step_size)
        classification = classify_signal(wf)

        result = {
            "signal_name": col,
            "full_period": full,
            "walk_forward": wf,
            "classification": classification,
        }
        all_results.append(result)

    # Print summary table
    print(f"\n{'Signal':<35} {'Full Sharpe':<12} {'OOS Sharpe':<11} {'Overfit':<9} {'OOS Ret%':<10} {'Classification'}")
    print("-" * 100)
    for r in sorted(all_results, key=lambda x: x["walk_forward"]["oos_sharpe"], reverse=True):
        fp = r["full_period"]
        wf = r["walk_forward"]
        print(f"{r['signal_name']:<35} {fp['sharpe'] if fp else 0:<12} {wf['oos_sharpe']:<11} {wf['overfit_ratio']:<9} {wf['oos_total_return_pct']:<10} {r['classification']}")

    # Recommendation
    strong = [r for r in all_results if r["classification"].startswith("STRONG")]
    moderate = [r for r in all_results if r["classification"].startswith("MODERATE")]

    print(f"\n{'='*80}")
    print(f"RECOMMENDATION:")
    if strong:
        print(f"  ADD these signals: {', '.join(r['signal_name'] for r in strong)}")
    if moderate:
        print(f"  CONSIDER these signals: {', '.join(r['signal_name'] for r in moderate)}")
    if not strong and not moderate:
        print(f"  No candidate signals passed the walk-forward threshold.")

    # Save output
    if args.output:
        output = {"benchmark": bnh, "signals": all_results}
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
