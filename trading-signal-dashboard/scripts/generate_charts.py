"""
Generate chart data and comprehensive metrics for the website.
Produces JSON files that the frontend will consume.
"""
import pandas as pd
import numpy as np
import os
import json

SIGNALS_DIR = "/home/ubuntu/gold_signals/signals"
BACKTEST_DIR = "/home/ubuntu/gold_signals/backtest"
OUTPUT_DIR = "/home/ubuntu/gold_signals/website_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

DIM_ICONS = {
    "dim1": "chart-line",
    "dim2": "landmark",
    "dim3": "newspaper",
    "dim4": "arrows-left-right",
    "dim5": "building-columns",
    "dim6": "calendar",
    "dim7": "globe",
    "dim8": "layer-group"
}

DIM_DESCRIPTIONS = {
    "dim1": "RSI (9), MACD (8,17,6), Bollinger Bands, EMA 50/200 crossover",
    "dim2": "BBG Real Yields (USGGT10Y), TIPS, Fed Funds Rate, Economic Surprise Index",
    "dim3": "ETF volume anomalies, GVZ fear gauge, Economic Surprise sentiment, premium/discount",
    "dim4": "DXY, VIX, crude oil, BBG Real Yields, GC1-GC2 curve spread, JPY",
    "dim5": "CFTC COT (Managed Money + Producer), BBG GLD/IAU shares outstanding & AUM",
    "dim6": "Monthly bias, day-of-week, Indian wedding season, Chinese NY, September effect",
    "dim7": "Caldara-Iacoviello GPR Index (Threats + Acts), GVZ, VIX fear gauge",
    "dim8": "Gold-silver ratio, price structure, BBG futures OI & volume confirmation"
}

def generate_website_data():
    print("Generating website data...")

    # Load signals
    signals = pd.read_csv(os.path.join(SIGNALS_DIR, "all_signals.csv"), index_col=0, parse_dates=True)

    # Load backtest results
    with open(os.path.join(BACKTEST_DIR, "backtest_results.json")) as f:
        backtest_results = json.load(f)

    # 1. Gold price chart data (sampled for performance)
    gold_chart = signals[["gold_close"]].dropna().copy()
    gold_chart.index = gold_chart.index.strftime("%Y-%m-%d")
    gold_chart_data = [{"date": d, "price": round(float(p), 2)} for d, p in gold_chart["gold_close"].items()]

    # 2. Per-dimension data
    dimensions_data = {}
    for dim_key in sorted(DIM_NAMES.keys()):
        sig_col = f"{dim_key}_signal"
        score_col = f"{dim_key}_score"
        conf_col = f"{dim_key}_confidence"

        if sig_col not in signals.columns:
            continue

        # Latest values
        latest = signals.dropna(subset=[sig_col]).iloc[-1]
        latest_signal = int(latest[sig_col])
        latest_confidence = int(latest[conf_col]) if conf_col in signals.columns else 0
        latest_score = round(float(latest[score_col]), 4) if score_col in signals.columns else 0

        # Signal history (last 252 trading days = ~1 year)
        recent = signals[[sig_col, score_col, conf_col]].dropna().tail(252)
        signal_history = []
        for date, row in recent.iterrows():
            signal_history.append({
                "date": date.strftime("%Y-%m-%d"),
                "signal": int(row[sig_col]),
                "score": round(float(row[score_col]), 4),
                "confidence": int(row[conf_col])
            })

        # Equity curve from backtest
        equity_file = os.path.join(BACKTEST_DIR, f"equity_{dim_key}.csv")
        equity_data = []
        if os.path.exists(equity_file):
            eq_df = pd.read_csv(equity_file)
            for _, row in eq_df.iterrows():
                equity_data.append({
                    "date": row["date"],
                    "value": round(float(row["cumulative"]), 4)
                })

        # Backtest metrics
        bt = backtest_results.get(dim_key, {})

        dimensions_data[dim_key] = {
            "key": dim_key,
            "name": DIM_NAMES[dim_key],
            "icon": DIM_ICONS[dim_key],
            "description": DIM_DESCRIPTIONS[dim_key],
            "latest_signal": latest_signal,
            "latest_signal_label": "BULLISH" if latest_signal > 0 else ("BEARISH" if latest_signal < 0 else "NEUTRAL"),
            "latest_confidence": latest_confidence,
            "latest_score": latest_score,
            "latest_date": signals.dropna(subset=[sig_col]).index[-1].strftime("%Y-%m-%d"),
            "signal_history": signal_history,
            "equity_curve": equity_data,
            "backtest": {
                "full_period": bt.get("full_period", {}),
                "is_average": bt.get("is_average", {}),
                "oos_average": bt.get("oos_average", {}),
                "overfit_ratio": bt.get("overfit_ratio", 0),
                "num_steps": bt.get("num_walk_forward_steps", 0),
            }
        }

    # 3. Summary dashboard data
    dashboard = {
        "last_updated": signals.index[-1].strftime("%Y-%m-%d"),
        "gold_price": round(float(signals["gold_close"].dropna().iloc[-1]), 2),
        "total_dimensions": len(dimensions_data),
        "bullish_count": sum(1 for d in dimensions_data.values() if d["latest_signal"] > 0),
        "bearish_count": sum(1 for d in dimensions_data.values() if d["latest_signal"] < 0),
        "neutral_count": sum(1 for d in dimensions_data.values() if d["latest_signal"] == 0),
        "consensus_signal": "BULLISH" if sum(d["latest_signal"] for d in dimensions_data.values()) > 1
                           else ("BEARISH" if sum(d["latest_signal"] for d in dimensions_data.values()) < -1 else "MIXED"),
        "consensus_score": sum(d["latest_signal"] for d in dimensions_data.values()),
    }

    # 4. Backtest comparison table
    comparison = []
    for dim_key in sorted(DIM_NAMES.keys()):
        if dim_key in dimensions_data:
            d = dimensions_data[dim_key]
            bt = d["backtest"]
            fp = bt.get("full_period", {})
            oos = bt.get("oos_average", {})
            comparison.append({
                "dimension": DIM_NAMES[dim_key],
                "dim_key": dim_key,
                "full_return": fp.get("full_total_return", 0),
                "full_sharpe": fp.get("full_sharpe_ratio", 0),
                "full_max_dd": fp.get("full_max_drawdown", 0),
                "full_win_rate": fp.get("full_win_rate", 0),
                "full_hit_rate": fp.get("full_hit_rate", 0),
                "oos_sharpe": oos.get("oos_sharpe_ratio", 0),
                "overfit_ratio": bt.get("overfit_ratio", 0),
            })

    # Save all data
    with open(os.path.join(OUTPUT_DIR, "gold_chart.json"), "w") as f:
        json.dump(gold_chart_data, f)

    with open(os.path.join(OUTPUT_DIR, "dimensions.json"), "w") as f:
        json.dump(dimensions_data, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "dashboard.json"), "w") as f:
        json.dump(dashboard, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"Dashboard: {dashboard}")
    print(f"Saved {len(dimensions_data)} dimension datasets to {OUTPUT_DIR}/")
    print("Website data generation complete!")

if __name__ == "__main__":
    generate_website_data()
