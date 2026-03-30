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
from signals import ALL_SIGNALS, compute_all_signals, SIGNAL_PARAM_GRIDS, SKIP_IS_OPTIMIZATION

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

IS_WINDOW  = 504      # ~2 years in-sample
OOS_WINDOW = 63       # ~3 months out-of-sample
STEP_SIZE  = 63       # roll forward
MIN_WF_STEPS = 5      # minimum walk-forward steps to report
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Set to False to restore one-pass mode (no per-window parameter search)
ENABLE_IS_OPTIMIZATION = True


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


# ---------------------------------------------------------------------------
# Shared aggregation helper
# ---------------------------------------------------------------------------

def _aggregate_wf_results(is_results, oos_results, oos_returns, oos_bah_returns, step):
    """Aggregate per-window IS/OOS results into summary dict."""
    is_avg  = {k: round(np.mean([r[k] for r in is_results]),  3) for k in is_results[0]}
    oos_avg = {k: round(np.mean([r[k] for r in oos_results]), 3) for k in oos_results[0]}

    is_sharpe  = is_avg.get("is_sharpe_ratio",  0)
    oos_sharpe = oos_avg.get("oos_sharpe_ratio", 0)
    overfit_ratio = round(1 - (oos_sharpe / is_sharpe), 3) if is_sharpe != 0 else 0.0

    n_oos_positive = sum(1 for r in oos_results if r.get("oos_sharpe_ratio", 0) > 0)

    if oos_returns:
        all_oos = pd.concat(oos_returns).sort_index()
        all_bah = pd.concat(oos_bah_returns).sort_index()
        equity_strat = (1 + all_oos).cumprod()
        equity_bah   = (1 + all_bah).cumprod()

        n_oos_years   = len(all_oos) / 252
        oos_total_ret = float(equity_strat.iloc[-1] - 1)
        oos_ann_ret   = float((1 + oos_total_ret) ** (1 / max(n_oos_years, 0.01)) - 1)
        oos_sharpe_stitched = float(
            all_oos.mean() / all_oos.std() * math.sqrt(252)
        ) if all_oos.std() > 0 else 0.0
        peak       = equity_strat.cummax()
        oos_max_dd = float(((equity_strat - peak) / peak).min())
        active     = all_oos[all_oos != 0]
        oos_win_rate = float((active > 0).mean() * 100) if len(active) > 0 else 0.0

        oos_stitched = {
            "oos_ann_return_stitched": round(oos_ann_ret * 100, 2),
            "oos_sharpe_stitched":     round(oos_sharpe_stitched, 3),
            "oos_max_dd_stitched":     round(oos_max_dd * 100, 2),
            "oos_win_rate_stitched":   round(oos_win_rate, 2),
            "oos_total_ret_stitched":  round(oos_total_ret * 100, 2),
        }
    else:
        equity_strat = pd.Series(dtype=float)
        equity_bah   = pd.Series(dtype=float)
        oos_stitched = {}

    return {
        "is_avg":        is_avg,
        "oos_avg":       oos_avg,
        "oos_stitched":  oos_stitched,
        "overfit_ratio": overfit_ratio,
        "n_wf_steps":    step,
        "n_oos_positive": n_oos_positive,
        "equity_strat":  equity_strat,
        "equity_bah":    equity_bah,
    }


# ---------------------------------------------------------------------------
# Walk-forward (one-pass: pre-computed signal)
# ---------------------------------------------------------------------------

def walk_forward_backtest(returns: pd.Series, signal: pd.Series):
    """Walk-forward IS/OOS backtest with a pre-computed signal."""
    aligned = pd.concat([returns, signal], axis=1).dropna()
    if len(aligned) < IS_WINDOW + OOS_WINDOW:
        return None

    ret = aligned.iloc[:, 0]
    sig = aligned.iloc[:, 1]
    n   = len(aligned)

    is_results, oos_results = [], []
    oos_returns, oos_bah_returns = [], []
    step  = 0
    start = 0

    while start + IS_WINDOW + OOS_WINDOW <= n:
        is_end  = start + IS_WINDOW
        oos_end = min(is_end + OOS_WINDOW, n)

        is_m  = compute_metrics(ret.iloc[start:is_end],  sig.iloc[start:is_end],  "is_")
        oos_m = compute_metrics(ret.iloc[is_end:oos_end], sig.iloc[is_end:oos_end], "oos_")

        is_results.append(is_m)
        oos_results.append(oos_m)

        oos_strat = sig.iloc[is_end:oos_end].shift(1) * ret.iloc[is_end:oos_end]
        oos_returns.append(oos_strat.dropna())
        oos_bah_returns.append(ret.iloc[is_end:oos_end].dropna())

        step  += 1
        start += STEP_SIZE

    if step < MIN_WF_STEPS:
        return None

    return _aggregate_wf_results(is_results, oos_results, oos_returns, oos_bah_returns, step)


# ---------------------------------------------------------------------------
# Walk-forward (optimised: per-IS-window parameter grid search)
# ---------------------------------------------------------------------------

def walk_forward_backtest_optimized(
    returns: pd.Series,
    df: pd.DataFrame,
    sig_id: str,
    compute_fn,
    param_grid: dict,
):
    """
    Walk-forward with per-IS-window parameter optimisation.

    For each IS window, grid-search over the candidate parameter values,
    pick the one with highest IS Sharpe, then apply it to the OOS window.
    No look-ahead: OOS signal is computed from df[is_start : oos_end].
    """
    n = len(returns)
    if n < IS_WINDOW + OOS_WINDOW:
        return None

    param_name, param_values_all = next(iter(param_grid.items()))
    # Drop integer window params that exceed the IS window (can't compute meaningful signal)
    param_values = [
        v for v in param_values_all
        if not isinstance(v, int) or v <= IS_WINDOW
    ]
    if not param_values:
        return None

    default_param = param_values[len(param_values) // 2]   # middle value as fallback

    is_results, oos_results = [], []
    oos_returns, oos_bah_returns = [], []
    best_params_log = []
    step  = 0
    start = 0

    while start + IS_WINDOW + OOS_WINDOW <= n:
        is_end  = start + IS_WINDOW
        oos_end = min(is_end + OOS_WINDOW, n)

        is_idx  = returns.index[start:is_end]
        oos_idx = returns.index[is_end:oos_end]

        ret_is  = returns.iloc[start:is_end]
        ret_oos = returns.iloc[is_end:oos_end]

        # ── Grid search on IS data only ───────────────────────────────────
        best_sharpe = -np.inf
        best_param  = default_param
        df_is = df.loc[is_idx[0]:is_idx[-1]]

        for pval in param_values:
            try:
                sig_cand = compute_fn(df_is, **{param_name: pval})
                sig_cand_aligned = sig_cand.reindex(ret_is.index)
                if sig_cand_aligned.dropna().empty:
                    continue
                m = compute_metrics(ret_is, sig_cand_aligned, "is_")
                s = m.get("is_sharpe_ratio", -np.inf)
                if not (np.isnan(s) or np.isinf(s)) and s > best_sharpe:
                    best_sharpe = s
                    best_param  = pval
            except Exception:
                continue

        best_params_log.append({
            "step":          step,
            "is_start":      str(is_idx[0].date()),
            "is_end":        str(is_idx[-1].date()),
            "best_param":    best_param,
            "best_is_sharpe": round(best_sharpe, 3) if best_sharpe != -np.inf else None,
        })

        # ── IS metrics with best param ────────────────────────────────────
        try:
            sig_is_best = compute_fn(df_is, **{param_name: best_param})
            sig_is_best = sig_is_best.reindex(ret_is.index)
        except Exception:
            sig_is_best = pd.Series(0, index=ret_is.index)
        is_m = compute_metrics(ret_is, sig_is_best, "is_")

        # ── OOS signal: df from is_start to oos_end (no look-ahead) ──────
        df_is_to_oos = df.loc[is_idx[0]:oos_idx[-1]]
        try:
            sig_oos_full = compute_fn(df_is_to_oos, **{param_name: best_param})
            sig_oos = sig_oos_full.reindex(ret_oos.index)
        except Exception:
            sig_oos = pd.Series(0, index=ret_oos.index)
        oos_m = compute_metrics(ret_oos, sig_oos, "oos_")

        is_results.append(is_m)
        oos_results.append(oos_m)

        oos_strat = sig_oos.shift(1) * ret_oos
        oos_returns.append(oos_strat.dropna())
        oos_bah_returns.append(ret_oos.dropna())

        step  += 1
        start += STEP_SIZE

    if step < MIN_WF_STEPS:
        return None

    # Save per-window best-param log
    try:
        with open(RESULTS_DIR / f"best_params_{sig_id}.json", "w") as f:
            json.dump(best_params_log, f, indent=2)
    except Exception:
        pass

    return _aggregate_wf_results(is_results, oos_results, oos_returns, oos_bah_returns, step)


def rolling_sharpe(returns: pd.Series, signal: pd.Series, window: int = 252) -> pd.Series:
    """252-day rolling Sharpe ratio of the strategy."""
    strat = signal.shift(1) * returns
    roll  = strat.rolling(window)
    rs    = roll.mean() / roll.std() * math.sqrt(252)
    return rs.fillna(np.nan)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all_backtests():
    print("=" * 70)
    print("10Y UST SIGNAL BACKTEST ENGINE")
    print(f"IS={IS_WINDOW}d | OOS={OOS_WINDOW}d | Step={STEP_SIZE}d  "
          f"| IS_OPT={'ON' if ENABLE_IS_OPTIMIZATION else 'OFF'}")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df = load_data()

    # Combined return series: IEF (2002+) spliced with duration-approx (pre-2002)
    ief    = df["ief_return"]
    approx = df.get("ust_return_approx", pd.Series(dtype=float))
    returns = ief.combine_first(approx).dropna()
    ief_start    = ief.first_valid_index()
    approx_start = approx.first_valid_index() if not approx.empty else None
    print(f"Combined return series: {len(returns)} days, "
          f"{returns.index[0].date()} → {returns.index[-1].date()}")
    print(f"  IEF (real):   {ief_start.date() if ief_start else 'N/A'} → {returns.index[-1].date()}")
    print(f"  Approx:       {approx_start.date() if approx_start else 'N/A'} → "
          f"{(ief_start - pd.Timedelta(days=1)).date() if ief_start else 'N/A'}")

    n_total = len(returns)
    expected_steps = max(0, (n_total - IS_WINDOW - OOS_WINDOW) // STEP_SIZE)
    print(f"  Expected WF steps: ~{expected_steps} "
          f"(IS={IS_WINDOW}d, OOS={OOS_WINDOW}d, step={STEP_SIZE}d)")

    # Buy-and-hold benchmark
    bah = compute_buy_and_hold(returns)
    print(f"\nBuy-and-Hold: Sharpe={bah['sharpe']}, Return={bah['total_return']}%, "
          f"MaxDD={bah['max_drawdown']}%")

    # Compute all signals once (used for full-period metrics + one-pass WF fallback)
    print("\nComputing 25 signals...")
    signals_df = compute_all_signals(df)

    # Run backtests
    print("\nRunning walk-forward backtests...")
    summary_rows  = []
    equity_curves = {}
    rolling_sharpes = {}

    for sig_id, sig_name, category, fn in ALL_SIGNALS:
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

        # Full-period metrics (fixed params, whole history)
        full     = compute_metrics(ret_aligned, sig_aligned, "")
        bah_full = compute_metrics(ret_aligned,
                                   pd.Series(1.0, index=ret_aligned.index), "bah_")

        # Walk-forward — optimised or one-pass
        param_grid = SIGNAL_PARAM_GRIDS.get(sig_id)
        if ENABLE_IS_OPTIMIZATION and param_grid and sig_id not in SKIP_IS_OPTIMIZATION:
            wf = walk_forward_backtest_optimized(ret_aligned, df, sig_id, fn, param_grid)
            opt_label = "[OPT]"
        else:
            wf = walk_forward_backtest(ret_aligned, sig_aligned)
            opt_label = ""

        # Rolling Sharpe
        rs = rolling_sharpe(ret_aligned, sig_aligned)

        excess_sharpe = round(full["sharpe_ratio"] - bah_full["bah_sharpe_ratio"], 3)

        row = {
            "ID":       sig_id,
            "Name":     sig_name,
            "Category": category,
            "Start":    str(common[0].date()),
            "N_Days":   len(common),
            **full,
            "BAH_Return%":   bah_full["bah_total_return"],
            "BAH_Sharpe":    bah_full["bah_sharpe_ratio"],
            "BAH_MaxDD%":    bah_full["bah_max_drawdown"],
            "Excess_Sharpe": excess_sharpe,
        }

        if wf:
            row["IS_Sharpe"]      = wf["is_avg"].get("is_sharpe_ratio", 0)
            row["OOS_Sharpe"]     = wf["oos_stitched"].get("oos_sharpe_stitched",
                                     wf["oos_avg"].get("oos_sharpe_ratio", 0))
            row["OOS_AnnReturn"]  = wf["oos_stitched"].get("oos_ann_return_stitched", None)
            row["OOS_TotalRet"]   = wf["oos_stitched"].get("oos_total_ret_stitched", None)
            row["OOS_MaxDD"]      = wf["oos_stitched"].get("oos_max_dd_stitched", None)
            row["OOS_WinRate"]    = wf["oos_stitched"].get("oos_win_rate_stitched", None)
            row["Overfit_Ratio"]  = wf["overfit_ratio"]
            row["WF_Steps"]       = wf["n_wf_steps"]
            row["N_OOS_Positive"] = wf["n_oos_positive"]

            # Save equity curves
            if not wf["equity_strat"].empty:
                eq_df = pd.DataFrame({
                    "date":         wf["equity_strat"].index,
                    "strategy":     wf["equity_strat"].values,
                    "buy_and_hold": wf["equity_bah"].reindex(
                                        wf["equity_strat"].index).values,
                })
                eq_df.to_csv(RESULTS_DIR / f"equity_{sig_id}.csv", index=False)
                equity_curves[sig_id] = eq_df
        else:
            row["IS_Sharpe"] = row["OOS_Sharpe"] = row["OOS_AnnReturn"] = None
            row["OOS_TotalRet"] = row["OOS_MaxDD"] = row["OOS_WinRate"] = None
            row["Overfit_Ratio"] = row["WF_Steps"] = row["N_OOS_Positive"] = None

        rolling_sharpes[sig_id] = rs
        summary_rows.append(row)

        oos_str = f"OOS={row.get('OOS_Sharpe', 'N/A')}" if row.get("OOS_Sharpe") is not None else "OOS=N/A"
        print(f"  [{sig_id}] {sig_name:20s}  "
              f"Sharpe={full['sharpe_ratio']:6.3f}  "
              f"MaxDD={full['max_drawdown']:7.2f}%  "
              f"{oos_str}  ExcessSharpe={excess_sharpe:+.3f}  {opt_label}")

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
                    "hit_rate", "IS_Sharpe", "OOS_Sharpe", "OOS_AnnReturn", "OOS_MaxDD",
                    "N_OOS_Positive", "Overfit_Ratio", "BAH_Sharpe", "Excess_Sharpe"]
    display_cols = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[display_cols].to_string(index=False))

    # ---------------------------------------------------------------------------
    # grep-able summary block for autoresearch loop
    # ---------------------------------------------------------------------------
    oos_sharpes = [r.get("OOS_Sharpe") for r in summary_rows if r.get("OOS_Sharpe") is not None]
    composite   = round(float(np.mean(oos_sharpes)), 4) if oos_sharpes else 0.0
    n_positive  = sum(1 for s in oos_sharpes if s > 0) if oos_sharpes else 0
    best_row  = max(summary_rows, key=lambda r: r.get("OOS_Sharpe") or -99)
    worst_row = min(summary_rows, key=lambda r: r.get("OOS_Sharpe") or  99)

    print("\n---")
    print(f"composite_sharpe: {composite}")
    print(f"n_signals: {len(summary_rows)}")
    print(f"n_positive_oos: {n_positive}")
    print(f"best_signal:  {best_row['ID']}_{best_row['Name']} "
          f"(OOS Sharpe {best_row.get('OOS_Sharpe', 'N/A')})")
    print(f"worst_signal: {worst_row['ID']}_{worst_row['Name']} "
          f"(OOS Sharpe {worst_row.get('OOS_Sharpe', 'N/A')})")
    print(f"bah_sharpe: {bah['sharpe']}")
    print(f"results_dir: {RESULTS_DIR}")

    return summary_df


if __name__ == "__main__":
    run_all_backtests()
