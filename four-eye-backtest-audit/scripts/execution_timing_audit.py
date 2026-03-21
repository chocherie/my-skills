#!/usr/bin/env python3
"""
Execution Timing Audit — Detect Look-Ahead Bias from Asynchronous Settlement

Usage:
  python execution_timing_audit.py <config_json>

config_json: {
  "asset_name": "Gold (XAU/USD)",
  "entry_time_utc": 17.5,           # 5:30 PM UTC = assumed entry time
  "instruments": [
    {"ticker": "GC=F", "name": "Gold Futures", "settlement_utc": 17.5, "dimensions": [1,2]},
    {"ticker": "^VIX", "name": "VIX Index", "settlement_utc": 20.25, "dimensions": [3,4,7]},
    ...
  ]
}

Output: feasibility matrix, latest-settling instrument, look-ahead bias verdict.
"""

import json
import sys


def utc_to_display(utc_hour):
    """Convert UTC decimal hour to HH:MM string."""
    h = int(utc_hour)
    m = int((utc_hour - h) * 60)
    return f"{h:02d}:{m:02d} UTC"


def utc_to_sgt(utc_hour):
    """Convert UTC decimal hour to SGT (UTC+8) HH:MM string."""
    sgt = (utc_hour + 8) % 24
    h = int(sgt)
    m = int((sgt - h) * 60)
    next_day = " (+1d)" if utc_hour + 8 >= 24 else ""
    return f"{h:02d}:{m:02d} SGT{next_day}"


def audit(config):
    """Run the execution timing audit."""
    asset = config["asset_name"]
    entry_utc = config["entry_time_utc"]
    instruments = config["instruments"]

    print(f"Execution Timing Audit: {asset}")
    print(f"Assumed Entry Time: {utc_to_display(entry_utc)} / {utc_to_sgt(entry_utc)}")
    print("=" * 75)

    # Find latest-settling instrument
    latest = max(instruments, key=lambda x: x["settlement_utc"])
    latest_utc = latest["settlement_utc"]

    # Classify each instrument
    results = []
    for inst in sorted(instruments, key=lambda x: x["settlement_utc"]):
        settle = inst["settlement_utc"]
        lag_hours = settle - entry_utc
        if lag_hours < 0:
            lag_hours += 24  # crosses midnight

        if settle <= entry_utc:
            status = "OK"
        elif lag_hours <= 0.5:
            status = "MARGINAL"
        else:
            status = "LOOK-AHEAD"

        results.append({
            "ticker": inst["ticker"],
            "name": inst["name"],
            "settlement": utc_to_display(settle),
            "settlement_sgt": utc_to_sgt(settle),
            "lag_hours": round(lag_hours, 2),
            "dimensions": inst.get("dimensions", []),
            "status": status,
        })

    # Print results table
    print(f"\n{'Ticker':<12} {'Name':<25} {'Settlement':<12} {'SGT':<16} {'Lag (h)':<8} {'Dims':<12} {'Status'}")
    print("-" * 105)
    for r in results:
        dims_str = ",".join(str(d) for d in r["dimensions"])
        print(f"{r['ticker']:<12} {r['name']:<25} {r['settlement']:<12} {r['settlement_sgt']:<16} {r['lag_hours']:<8} {dims_str:<12} {r['status']}")

    # Summary
    n_look_ahead = sum(1 for r in results if r["status"] == "LOOK-AHEAD")
    n_marginal = sum(1 for r in results if r["status"] == "MARGINAL")
    n_ok = sum(1 for r in results if r["status"] == "OK")

    affected_dims = set()
    for r in results:
        if r["status"] == "LOOK-AHEAD":
            affected_dims.update(r["dimensions"])

    print(f"\n{'='*75}")
    print(f"VERDICT:")
    print(f"  OK: {n_ok}  |  Marginal: {n_marginal}  |  Look-Ahead: {n_look_ahead}")
    print(f"  Latest instrument: {latest['name']} at {utc_to_display(latest_utc)} / {utc_to_sgt(latest_utc)}")
    print(f"  Max look-ahead: {round(latest_utc - entry_utc, 2)} hours")

    if n_look_ahead > 0:
        print(f"\n  ** LOOK-AHEAD BIAS DETECTED **")
        print(f"  Affected dimensions: {sorted(affected_dims)}")
        corrected_entry = latest_utc + 0.5  # 30 min buffer after latest settlement
        print(f"  Corrected entry time: {utc_to_display(corrected_entry)} / {utc_to_sgt(corrected_entry)}")
        print(f"  Action: Re-run backtest with entry at {utc_to_display(corrected_entry)} or later")
    else:
        print(f"\n  No look-ahead bias detected. Entry time is after all data settlements.")

    return {
        "verdict": "LOOK_AHEAD_BIAS" if n_look_ahead > 0 else "CLEAN",
        "n_look_ahead": n_look_ahead,
        "affected_dimensions": sorted(affected_dims),
        "latest_instrument": latest["name"],
        "latest_settlement_utc": latest_utc,
        "max_lag_hours": round(latest_utc - entry_utc, 2),
        "instruments": results,
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        config = json.load(f)

    result = audit(config)

    # Save result
    out_file = sys.argv[1].replace(".json", "_result.json")
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
