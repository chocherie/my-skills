#!/usr/bin/env python3
"""
Deflated Sharpe Ratio (DSR) — Multiple Testing Correction
Based on Bailey & Lopez de Prado (2014)

Usage:
  python deflated_sharpe_ratio.py <json_file> [--n_trials N]

Input JSON: array of objects, each with:
  { "name": "Strategy A", "oos_step_sharpes": [0.3, 0.1, -0.2, ...] }

Output: per-strategy DSR, pass/fail at 95%, and expected max SR from chance.
"""

import json
import sys
import numpy as np
from scipy import stats as scipy_stats


def compute_dsr(step_sharpes, n_trials):
    """
    Compute the Deflated Sharpe Ratio.

    Parameters:
        step_sharpes: list of per-step OOS Sharpe ratios
        n_trials: total number of independent strategies tested

    Returns:
        dict with dsr, passes_95, sr_benchmark, sr_hat, sr_se, skew, kurtosis
    """
    arr = np.array(step_sharpes, dtype=float)
    if len(arr) < 5 or n_trials < 2:
        return {"dsr": 0.0, "passes_95": False, "error": "insufficient data"}

    sr_hat = float(np.mean(arr))
    n = len(arr)
    skew = float(scipy_stats.skew(arr))
    kurt = float(scipy_stats.kurtosis(arr, fisher=True)) + 3  # raw kurtosis

    # Expected max SR from N independent trials (E[SR]=0, V[SR]=1)
    gamma_em = 0.5772  # Euler-Mascheroni constant
    z1 = scipy_stats.norm.ppf(1 - 1 / n_trials)
    z2 = scipy_stats.norm.ppf(1 - 1 / (n_trials * np.e))
    sr_benchmark = (1 - gamma_em) * z1 + gamma_em * z2

    # SR standard error (accounting for skewness and kurtosis)
    sr_se = np.sqrt(
        (1 + 0.5 * sr_hat**2 - skew * sr_hat + (kurt - 3) / 4 * sr_hat**2)
        / (n - 1)
    )

    if sr_se < 1e-10:
        return {"dsr": 0.0, "passes_95": False, "error": "zero standard error"}

    dsr = float(scipy_stats.norm.cdf((sr_hat - sr_benchmark) / sr_se))

    return {
        "dsr": round(dsr, 4),
        "passes_95": dsr > 0.95,
        "sr_hat": round(sr_hat, 4),
        "sr_benchmark": round(float(sr_benchmark), 4),
        "sr_se": round(float(sr_se), 4),
        "skewness": round(skew, 4),
        "kurtosis": round(float(kurt), 4),
        "n_steps": n,
        "n_trials": n_trials,
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    json_file = sys.argv[1]
    n_trials = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == "--n_trials" else None

    with open(json_file) as f:
        strategies = json.load(f)

    if n_trials is None:
        n_trials = len(strategies)
        print(f"Auto-detected {n_trials} independent trials from input.")

    print(f"\nDeflated Sharpe Ratio Analysis (N_trials={n_trials})")
    print(f"{'='*65}")

    for s in strategies:
        name = s.get("name", "Unknown")
        sharpes = s.get("oos_step_sharpes", [])
        result = compute_dsr(sharpes, n_trials)

        status = "PASS" if result.get("passes_95") else "FAIL"
        print(f"\n  {name}:")
        print(f"    Observed SR:  {result.get('sr_hat', 0):+.4f}")
        print(f"    Benchmark SR: {result.get('sr_benchmark', 0):+.4f} (expected max from {n_trials} trials)")
        print(f"    DSR:          {result.get('dsr', 0):.4f}  [{status} at 95%]")
        if result.get("passes_95"):
            print(f"    -> Statistically significant after multiple testing correction")
        else:
            print(f"    -> NOT significant — could be explained by chance")

    print()


if __name__ == "__main__":
    main()
