"""
backtest_runner.py — 10Y UST Signal Dashboard
===============================================
Runs walk-forward backtests for all 25 signals and compares each vs buy-and-hold.
Adapted from trading-signal-dashboard/scripts/backtest_engine.py.

Usage:
    python backtest_runner.py              # run all signals
    python backtest_runner.py > run.log 2>&1   # autoresearch loop mode

grep-able output (for autoresearch loop):
    grep "^composite_sharpe:" run.log
    grep "^n_positive_oos:" run.log
"""

import os
import sys
import math
import json
import numpy as np
import pandas as pd
from pathlib import Path

from data_layer import load_data, evaluate_sharpe, compute_buy_and_hold
from signals import ALL_SIGNALS, compute_all_signals

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

IS_WINDOW = 504      # ~2 years in-sample
OOS_WINDOW = 63      # ~3 months out-of-sample
STEP_SIZE = 63       # roll forward
MIN_WF_STEPS = 5     # minimum walk-forward steps to report
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Core metrics (adapted from backtest_engine.py)
# ---------------------------------------------------------------------------

def compute_metrics(returns: pd.Series, signal: pd.Series, prefix: str = "") -> dict:
    """Compute trading performance metrics. Signal shifted 1d before use."""
    strat = signal.shift(1) * returns
    strat = strat.dropna()

    if len(strat) == 0 or strat.std() == 0:
        return {
            f"{prefix}total_return": 0.0,
            f"{prefix}annual_return": 0.0,
            f"{prefix}sharpe_ratio": 0.0,
            f"{prefix}max_drawdown": 0.0,
            f"{prefix}win_rate": 0.0,
            f"{prefix}profit_factor": 0.0,
            f"{prefix}num_trades": 0,
            f"{prefix}hit_rate": 0.0,
        }

    equity = (1 + strat).cumprod()
    total_ret = float(equity.iloc[-1] - 1)
    n_years = len(strat) / 252
    annual_ret = float((1 + total_ret) ** (1 / max(n_years, 0.01)) - 1)
    sharpe = float(strat.mean() / strat.std() * math.sqrt(252))

    peak = equity.cummax()
    max_dd = float(((equity - peak) / peak).min())

    active = strat[signal.shift(1).reindex(strat.index) != 0]
    win_rate = float((active > 0).mean()) if len(active) > 0 else 0.0

    gross_profit = float(active[active > 0].sum()) if (active > 0).any() else 0.0
    gross_loss = float(abs(active[active < 0].sum())) if (active < 0).any() else 1e-6
    profit_factor = gross_profit / max(gross_loss, 1e-6)

    num_trades = int((signal.diff().abs() > 0).sum())

    ret_aligned = returns.reindex(strat.index)
    sig_aligned = signal.shift(1).reindex(strat.index)
    correct = ((sig_aligned > 0) & (ret_aligned > 0)) | ((sig_aligned < 0) & (ret_aligned < 0))
    active_days = sig_aligned != 0
    hit_rate = float(correct[active_days].mean()) if active_days.sum() > 0 else 0.0

    return {
        f"{prefix}total_return": round(total_ret * 100, 2),
        f"{prefix}annual_return": round(annual_ret * 100, 2),
        f"{prefix}sharpe_ratio": round(sharpe, 3),
        f"{prefix}max_drawdown": round(max_dd * 100, 2),
        f"{prefix}win_rate": round(win_rate * 100, 2),
        f"{prefix}profit_factor": round(profit_factor, 3),
        f"{prefix}num_trades": num_trades,
        f"{prefix}hit_rate": round(hit_rate * 100, 2),
    }


def walk_forward_backtest(returns: pd.Series, signal: pd.Series):
    """Walk-forward IS/OOS backtest. Returns summary dict."""
    aligned = pd.concat([returns, signal], axis=1).dropna()
    if len(aligned) < IS_WINDOW + OOS_WINDOW:
        return None

    ret = aligned.iloc[:, 0]
    sig = aligned.iloc[:, 1]
    n = len(aligned)

    is_results, oos_results = [], []
    oos_returns = []
    oos_bah_returns = []
    step = 0
    start = 0

    while start + IS_WINDOW + OOS_WINDOW <= n:
        is_end = start + IS_WINDOW
        oos_end = min(is_end + OOS_WINDOW, n)

        is_metrics = compute_metrics(ret.iloc[start:is_end], sig.iloc[start:is_end], "is_")
        oos_metrics = compute_metrics(ret.iloc[is_end:oos_end], sig.iloc[is_end:oos_end], "oos_")

        is_results.append(is_metrics)
        oos_results.append(oos_metrics)

        # Collect OOS returns for stitched equity curve
        oos_strat = sig.iloc[is_end:oos_end].shift(1) * ret.iloc[is_end:oos_end]
        oos_bah = ret.iloc[is_end:oos_end]
        oos_returns.append(oos_strat.dropna())
        oos_bah_returns.append(oos_bah.dropna())

        step += 1
        start += STEP_SIZE

    if step < MIN_WF_STEPS:
        return None

    is_avg = {k: round(np.mean([r[k] for r in is_results]), 3) for k in is_results[0]}
    oos_avg = {k: round(np.mean([r[k] for r in oos_results]), 3) for k in oos_results[0]}

    is_sharpe = is_avg.get("is_sharpe_ratio", 0)
    oos_sharpe = oos_avg.get("oos_sharpe_ratio", 0)
    overfit_ratio = round(1 - (oos_sharpe / is_sharpe), 3) if is_sharpe != 0 else 0.0

    # Count OOS windows where strategy Sharpe > 0 (beats flat)
    n_oos_positive = sum(1 for r in oos_results if r.get("oos_sharpe_ratio", 0) > 0)

    # Stitched OOS equity curve
    if oos_returns:
        all_oos = pd.concat(oos_returns).sort_index()
        all_bah = pd.concat(oos_bah_returns).sort_index()
        equity_strat = (1 + all_oos).cumprod()
        equity_bah = (1 + all_bah).cumprod()
    else:
        equity_strat = pd.Series(dtype=float)
        equity_bah = pd.Series(dtype=float)

    return {
        "is_avg": is_avg,
        "oos_avg": oos_avg,
        "overfit_ratio": overfit_ratio,
        "n_wf_steps": step,
        "n_oos_positive": n_oos_positive,
        "equity_strat": equity_strat,
        "equity_bah": equity_bah,
    }


def rolling_sharpe(returns: pd.Series, signal: pd.Series, window: int = 252) -> pd.Series:
    """252-day rolling Sharpe ratio of the strategy."""
    strat = signal.shift(1) * returns
    roll = strat.rolling(window)
    rs = roll.mean() / roll.std() * math.sqrt(252)
    return rs.fillna(np.nan)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all_backtests():
    print("=" * 70)
    print("10Y UST SIGNAL BACKTEST ENGINE")
    print(f"IS={IS_WINDOW}d | OOS={OOS_WINDOW}d | Step={STEP_SIZE}d")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df = load_data()
    returns = df["ief_return"].dropna()
    print(f"Returns: {len(returns)} days, {returns.index[0].date()} to {returns.index[-1].date()}")

    # Buy-and-hold benchmark
    bah = compute_buy_and_hold(returns)
    print(f"\nBuy-and-Hold: Sharpe={bah['sharpe']}, Return={bah['total_return']}%, "
          f"MaxDD={bah['max_drawdown']}%")

    # Compute all signals
    print("\nComputing 25 signals...")
    signals_df = compute_all_signals(df)

    # Run backtests
    print("\nRunning walk-forward backtests...")
    summary_rows = []
    equity_curves = {}
    rolling_sharpes = {}

    for sig_id, sig_name, category, _ in ALL_SIGNALS:
        col = f"{sig_id}_{sig_name}"
        if col not in signals_df.columns:
            continue
        signal = signals_df[col]

        # Align with returns
        common = returns.index.intersection(signal.dropna().index)
        if len(common) < IS_WINDOW + OOS_WINDOW:
            print(f"  [{sig_id}] SKIP — insufficient data ({len(common)} rows)")
            continue

        ret_aligned = returns.reindex(common)
        sig_aligned = signal.reindex(common)

        # Full-period metrics
        full = compute_metrics(ret_aligned, sig_aligned, "")
        bah_full = compute_metrics(ret_aligned,
                                   pd.Series(1.0, index=ret_aligned.index), "bah_")

        # Walk-forward
        wf = walk_forward_backtest(ret_aligned, sig_aligned)

        # Rolling Sharpe
        rs = rolling_sharpe(ret_aligned, sig_aligned)

        excess_sharpe = round(full["sharpe_ratio"] - bah_full["bah_sharpe_ratio"], 3)

        row = {
            "ID": sig_id,
            "Name": sig_name,
            "Category": category,
            "Start": str(common[0].date()),
            "N_Days": len(common),
            **full,
            "BAH_Return%": bah_full["bah_total_return"],
            "BAH_Sharpe": bah_full["bah_sharpe_ratio"],
            "BAH_MaxDD%": bah_full["bah_max_drawdown"],
            "Excess_Sharpe": excess_sharpe,
        }

        if wf:
            row["IS_Sharpe"]      = wf["is_avg"].get("is_sharpe_ratio", 0)
            row["OOS_Sharpe"]     = wf["oos_avg"].get("oos_sharpe_ratio", 0)
            row["OOS_AnnReturn"]  = wf["oos_avg"].get("oos_annual_return", 0)
            row["OOS_MaxDD"]      = wf["oos_avg"].get("oos_max_drawdown", 0)
            row["OOS_WinRate"]    = wf["oos_avg"].get("oos_win_rate", 0)
            row["OOS_HitRate"]    = wf["oos_avg"].get("oos_hit_rate", 0)
            row["Overfit_Ratio"]  = wf["overfit_ratio"]
            row["WF_Steps"]       = wf["n_wf_steps"]
            row["N_OOS_Positive"] = wf["n_oos_positive"]

            # Save equity curves
            if not wf["equity_strat"].empty:
                eq_df = pd.DataFrame({
                    "date": wf["equity_strat"].index,
                    "strategy": wf["equity_strat"].values,
                    "buy_and_hold": wf["equity_bah"].reindex(
                        wf["equity_strat"].index).values,
                })
                eq_df.to_csv(RESULTS_DIR / f"equity_{sig_id}.csv", index=False)
                equity_curves[sig_id] = eq_df
        else:
            row["IS_Sharpe"] = row["OOS_Sharpe"] = row["OOS_AnnReturn"] = None
            row["OOS_MaxDD"] = row["OOS_WinRate"] = row["OOS_HitRate"] = None
            row["Overfit_Ratio"] = row["WF_Steps"] = row["N_OOS_Positive"] = None

        # Save rolling Sharpe
        rolling_sharpes[sig_id] = rs

        summary_rows.append(row)
        oos_str = f"OOS={row.get('OOS_Sharpe', 'N/A')}" if row.get("OOS_Sharpe") else "OOS=N/A"
        print(f"  [{sig_id}] {sig_name:20s}  "
              f"Sharpe={full['sharpe_ratio']:6.3f}  "
              f"MaxDD={full['max_drawdown']:7.2f}%  "
              f"{oos_str}  ExcessSharpe={excess_sharpe:+.3f}")

    # Save summary table
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "backtest_summary.csv", index=False)

    # Save rolling Sharpe matrix
    rs_df = pd.DataFrame(rolling_sharpes, index=returns.index)
    rs_df.to_csv(RESULTS_DIR / "rolling_sharpe.csv")

    # Save B&H benchmark
    with open(RESULTS_DIR / "buy_and_hold.json", "w") as f:
        json.dump(bah, f, indent=2)

    # ---------------------------------------------------------------------------
    # Print summary table
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BACKTEST SUMMARY — ALL SIGNALS VS BUY-AND-HOLD")
    print("=" * 70)
    display_cols = ["ID", "Name", "Category", "sharpe_ratio", "max_drawdown",
                    "hit_rate", "OOS_Sharpe", "OOS_AnnReturn", "OOS_MaxDD",
                    "N_OOS_Positive", "Overfit_Ratio", "BAH_Sharpe", "Excess_Sharpe"]
    display_cols = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[display_cols].to_string(index=False))

    # ---------------------------------------------------------------------------
    # grep-able summary block for autoresearch loop
    # ---------------------------------------------------------------------------
    oos_sharpes = [r.get("OOS_Sharpe") for r in summary_rows if r.get("OOS_Sharpe") is not None]
    composite = round(float(np.mean(oos_sharpes)), 4) if oos_sharpes else 0.0
    n_positive = sum(1 for s in oos_sharpes if s > 0) if oos_sharpes else 0
    best_row = max(summary_rows, key=lambda r: r.get("OOS_Sharpe") or -99)
    worst_row = min(summary_rows, key=lambda r: r.get("OOS_Sharpe") or 99)

    print("\n---")
    print(f"composite_sharpe: {composite}")
    print(f"n_signals: {len(summary_rows)}")
    print(f"n_positive_oos: {n_positive}")
    print(f"best_signal: {best_row['ID']}_{best_row['Name']} (OOS Sharpe {best_row.get('OOS_Sharpe', 'N/A')})")
    print(f"worst_signal: {worst_row['ID']}_{worst_row['Name']} (OOS Sharpe {worst_row.get('OOS_Sharpe', 'N/A')})")
    print(f"bah_sharpe: {bah['sharpe']}")
    print(f"results_dir: {RESULTS_DIR}")

    return summary_df


if __name__ == "__main__":
    run_all_backtests()
