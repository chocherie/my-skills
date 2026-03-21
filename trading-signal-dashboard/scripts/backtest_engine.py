"""
Walk-Forward Backtesting Engine for Gold Trading Signals
========================================================
- Uses rolling walk-forward optimization (not a single train/test split)
- In-sample window: 504 trading days (~2 years)
- Out-of-sample window: 63 trading days (~3 months)
- Rolls forward by 63 days each step
- Evaluates each dimension independently
- Tracks IS vs OOS performance to detect overfitting
"""
import pandas as pd
import numpy as np
import os
import json

SIGNALS_DIR = "/home/ubuntu/gold_signals/signals"
BACKTEST_DIR = "/home/ubuntu/gold_signals/backtest"

IS_WINDOW = 504    # ~2 years in-sample
OOS_WINDOW = 63    # ~3 months out-of-sample
STEP_SIZE = 63     # roll forward by 3 months

DIM_NAMES = {
    "dim1": "Technical Indicators",
    "dim2": "Macroeconomic Data",
    "dim3": "News & Sentiment",
    "dim4": "Intermarket Analysis",
    "dim5": "Institutional Flows",
    "dim6": "Seasonality",
    "dim7": "Geopolitical Risk",
    "dim8": "Market Structure"
}

def load_signals():
    path = os.path.join(SIGNALS_DIR, "all_signals.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df

def compute_metrics(returns, signals, prefix=""):
    """Compute trading performance metrics for a signal series."""
    # Strategy returns: signal * next-day return (signal is applied at close, return realized next day)
    strat_returns = signals.shift(1) * returns  # shift signal by 1 to avoid look-ahead
    strat_returns = strat_returns.dropna()

    if len(strat_returns) == 0 or strat_returns.std() == 0:
        return {
            f"{prefix}total_return": 0,
            f"{prefix}annual_return": 0,
            f"{prefix}sharpe_ratio": 0,
            f"{prefix}max_drawdown": 0,
            f"{prefix}win_rate": 0,
            f"{prefix}profit_factor": 0,
            f"{prefix}num_trades": 0,
            f"{prefix}avg_trade": 0,
            f"{prefix}hit_rate": 0,
        }

    # Equity curve
    equity = (1 + strat_returns).cumprod()
    total_return = float(equity.iloc[-1] - 1) if len(equity) > 0 else 0

    # Annualized return
    n_years = len(strat_returns) / 252
    annual_return = float((1 + total_return) ** (1 / max(n_years, 0.01)) - 1)

    # Sharpe ratio (annualized)
    sharpe = float(strat_returns.mean() / strat_returns.std() * np.sqrt(252)) if strat_returns.std() > 0 else 0

    # Max drawdown
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = float(drawdown.min())

    # Win rate (days with positive return when signal != 0)
    active = strat_returns[signals.shift(1) != 0]
    win_rate = float((active > 0).mean()) if len(active) > 0 else 0

    # Profit factor
    gross_profit = float(active[active > 0].sum()) if len(active[active > 0]) > 0 else 0
    gross_loss = float(abs(active[active < 0].sum())) if len(active[active < 0]) > 0 else 0.0001
    profit_factor = gross_profit / max(gross_loss, 0.0001)

    # Number of signal changes (trades)
    signal_changes = (signals.diff().abs() > 0).sum()

    # Hit rate: % of days signal correctly predicts direction
    correct = ((signals.shift(1) > 0) & (returns > 0)) | ((signals.shift(1) < 0) & (returns < 0))
    active_days = signals.shift(1) != 0
    hit_rate = float(correct[active_days].mean()) if active_days.sum() > 0 else 0

    return {
        f"{prefix}total_return": round(total_return * 100, 2),
        f"{prefix}annual_return": round(annual_return * 100, 2),
        f"{prefix}sharpe_ratio": round(sharpe, 3),
        f"{prefix}max_drawdown": round(max_dd * 100, 2),
        f"{prefix}win_rate": round(win_rate * 100, 2),
        f"{prefix}profit_factor": round(profit_factor, 3),
        f"{prefix}num_trades": int(signal_changes),
        f"{prefix}avg_trade": round(float(active.mean()) * 100, 4) if len(active) > 0 else 0,
        f"{prefix}hit_rate": round(hit_rate * 100, 2),
    }

def walk_forward_backtest(df, dim_key):
    """Run walk-forward backtest for a single dimension."""
    signal_col = f"{dim_key}_signal"
    score_col = f"{dim_key}_score"

    if signal_col not in df.columns:
        return None

    returns = df["gold_return"].copy()
    signals = df[signal_col].copy()

    n = len(df)
    oos_results = []
    is_results = []
    equity_points = []

    step = 0
    start = 0

    while start + IS_WINDOW + OOS_WINDOW <= n:
        is_start = start
        is_end = start + IS_WINDOW
        oos_start = is_end
        oos_end = min(is_end + OOS_WINDOW, n)

        # In-sample metrics
        is_ret = returns.iloc[is_start:is_end]
        is_sig = signals.iloc[is_start:is_end]
        is_metrics = compute_metrics(is_ret, is_sig, "is_")

        # Out-of-sample metrics
        oos_ret = returns.iloc[oos_start:oos_end]
        oos_sig = signals.iloc[oos_start:oos_end]
        oos_metrics = compute_metrics(oos_ret, oos_sig, "oos_")

        # Record
        window_info = {
            "step": step,
            "is_start": str(df.index[is_start].date()),
            "is_end": str(df.index[is_end - 1].date()),
            "oos_start": str(df.index[oos_start].date()),
            "oos_end": str(df.index[oos_end - 1].date()),
        }
        window_info.update(is_metrics)
        window_info.update(oos_metrics)

        is_results.append(is_metrics)
        oos_results.append(oos_metrics)

        # Equity tracking for OOS periods
        oos_strat = (oos_sig.shift(1) * oos_ret).dropna()
        for date, ret in oos_strat.items():
            equity_points.append({"date": str(date.date()), "return": float(ret), "type": "oos"})

        step += 1
        start += STEP_SIZE

    # Aggregate IS vs OOS comparison
    if not oos_results:
        return None

    is_avg = {k: round(np.mean([r[k] for r in is_results]), 3) for k in is_results[0]}
    oos_avg = {k: round(np.mean([r[k] for r in oos_results]), 3) for k in oos_results[0]}

    # Full-period metrics
    full_metrics = compute_metrics(returns, signals, "full_")

    # Build equity curve from OOS periods only
    if equity_points:
        eq_df = pd.DataFrame(equity_points)
        eq_df["date"] = pd.to_datetime(eq_df["date"])
        eq_df = eq_df.sort_values("date")
        eq_df["cumulative"] = (1 + eq_df["return"]).cumprod()
        equity_curve = eq_df[["date", "cumulative"]].copy()
        equity_curve["date"] = equity_curve["date"].dt.strftime("%Y-%m-%d")
    else:
        equity_curve = pd.DataFrame()

    # Overfitting ratio: how much IS performance degrades in OOS
    is_sharpe = is_avg.get("is_sharpe_ratio", 0)
    oos_sharpe = oos_avg.get("oos_sharpe_ratio", 0)
    overfit_ratio = round(1 - (oos_sharpe / is_sharpe), 3) if is_sharpe != 0 else 0

    result = {
        "dimension": dim_key,
        "name": DIM_NAMES[dim_key],
        "full_period": full_metrics,
        "is_average": is_avg,
        "oos_average": oos_avg,
        "overfit_ratio": overfit_ratio,
        "num_walk_forward_steps": step,
        "equity_curve": equity_curve.to_dict(orient="records") if not equity_curve.empty else [],
    }

    return result

def run_all_backtests():
    print("=" * 60)
    print("WALK-FORWARD BACKTESTING ENGINE")
    print(f"In-Sample: {IS_WINDOW} days | Out-of-Sample: {OOS_WINDOW} days | Step: {STEP_SIZE} days")
    print("=" * 60)

    df = load_signals()
    df = df.dropna(subset=["gold_return"])

    print(f"Data: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")

    all_results = {}
    summary_table = []

    for dim_key in sorted(DIM_NAMES.keys()):
        print(f"\n  Backtesting {DIM_NAMES[dim_key]}...")
        result = walk_forward_backtest(df, dim_key)

        if result:
            all_results[dim_key] = result

            # Print summary
            fp = result["full_period"]
            is_avg = result["is_average"]
            oos_avg = result["oos_average"]

            print(f"    Full Period:  Return={fp['full_total_return']}%, Sharpe={fp['full_sharpe_ratio']}, "
                  f"MaxDD={fp['full_max_drawdown']}%, WinRate={fp['full_win_rate']}%")
            print(f"    IS Average:   Sharpe={is_avg['is_sharpe_ratio']}")
            print(f"    OOS Average:  Sharpe={oos_avg['oos_sharpe_ratio']}")
            print(f"    Overfit Ratio: {result['overfit_ratio']} (0=no overfit, 1=total overfit)")

            summary_table.append({
                "Dimension": DIM_NAMES[dim_key],
                "Full Return %": fp["full_total_return"],
                "Full Sharpe": fp["full_sharpe_ratio"],
                "Full MaxDD %": fp["full_max_drawdown"],
                "Full Win Rate %": fp["full_win_rate"],
                "Full Hit Rate %": fp["full_hit_rate"],
                "IS Avg Sharpe": is_avg["is_sharpe_ratio"],
                "OOS Avg Sharpe": oos_avg["oos_sharpe_ratio"],
                "Overfit Ratio": result["overfit_ratio"],
                "WF Steps": result["num_walk_forward_steps"],
            })

    # Save results
    with open(os.path.join(BACKTEST_DIR, "backtest_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save summary table
    summary_df = pd.DataFrame(summary_table)
    summary_df.to_csv(os.path.join(BACKTEST_DIR, "backtest_summary.csv"), index=False)

    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    # Also generate the full equity curve for each dimension
    for dim_key, result in all_results.items():
        if result["equity_curve"]:
            eq_df = pd.DataFrame(result["equity_curve"])
            eq_df.to_csv(os.path.join(BACKTEST_DIR, f"equity_{dim_key}.csv"), index=False)

    print(f"\nResults saved to {BACKTEST_DIR}/")
    return all_results

if __name__ == "__main__":
    run_all_backtests()
