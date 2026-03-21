"""
Data Quality Audit & Signal Timeframe Analysis
================================================
1. Assess data quality for every feed (gaps, staleness, frequency, coverage)
2. Compute signal holding periods and turnover for each dimension
3. Analyze effective trading timeframe (intraday, swing, position, long-term)
4. Output JSON for the website
"""
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

DATA_DIR = "/home/ubuntu/gold_signals/data"
SIGNALS_DIR = "/home/ubuntu/gold_signals/signals"
OUTPUT_DIR = "/home/ubuntu/gold_signals/website_data"

# ============================================================
# 1. DATA QUALITY AUDIT
# ============================================================

def audit_csv(filepath, name):
    """Audit a single CSV data file."""
    if not os.path.exists(filepath):
        return {"name": name, "status": "MISSING", "quality_score": 0}
    
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    if df.empty:
        return {"name": name, "status": "EMPTY", "quality_score": 0}
    
    # Basic stats
    total_rows = len(df)
    date_range_start = df.index.min().strftime("%Y-%m-%d")
    date_range_end = df.index.max().strftime("%Y-%m-%d")
    total_calendar_days = (df.index.max() - df.index.min()).days
    
    # Expected trading days (~252/year)
    years = total_calendar_days / 365.25
    expected_trading_days = int(years * 252)
    coverage_ratio = round(total_rows / max(expected_trading_days, 1), 4)
    
    # Gap analysis
    date_diffs = df.index.to_series().diff().dt.days.dropna()
    # Normal gaps: weekends (2-3 days). Abnormal: > 4 days
    abnormal_gaps = date_diffs[date_diffs > 4]
    num_abnormal_gaps = len(abnormal_gaps)
    max_gap_days = int(date_diffs.max()) if len(date_diffs) > 0 else 0
    avg_gap_days = round(float(date_diffs.mean()), 2) if len(date_diffs) > 0 else 0
    
    # Missing values per column
    missing_pct = {}
    for col in df.columns:
        pct = round(df[col].isna().mean() * 100, 2)
        missing_pct[col] = pct
    total_missing_pct = round(df.isna().mean().mean() * 100, 2)
    
    # Staleness: days since last data point
    last_date = df.index.max()
    staleness_days = (pd.Timestamp("2026-03-19") - last_date).days
    
    # Data frequency
    if avg_gap_days < 1.5:
        frequency = "Daily (calendar)"
    elif avg_gap_days < 2.0:
        frequency = "Daily (trading days)"
    elif avg_gap_days < 6:
        frequency = "Weekly"
    else:
        frequency = "Irregular"
    
    # Quality score (0-100)
    score = 100
    score -= min(30, num_abnormal_gaps * 3)  # Penalize gaps
    score -= min(20, total_missing_pct * 2)   # Penalize missing data
    score -= min(15, max(0, staleness_days - 3) * 3)  # Penalize staleness
    score -= min(15, max(0, (1 - coverage_ratio) * 30))  # Penalize low coverage
    score = max(0, round(score))
    
    # Quality label
    if score >= 85:
        quality_label = "Excellent"
    elif score >= 70:
        quality_label = "Good"
    elif score >= 50:
        quality_label = "Fair"
    else:
        quality_label = "Poor"
    
    # Top gaps (for display)
    top_gaps = []
    if num_abnormal_gaps > 0:
        for idx in abnormal_gaps.nlargest(5).index:
            gap_start = df.index[df.index.get_loc(idx) - 1].strftime("%Y-%m-%d")
            gap_end = idx.strftime("%Y-%m-%d")
            top_gaps.append({
                "from": gap_start,
                "to": gap_end,
                "days": int(abnormal_gaps[idx])
            })
    
    return {
        "name": name,
        "status": "OK",
        "total_rows": total_rows,
        "date_range": f"{date_range_start} to {date_range_end}",
        "date_start": date_range_start,
        "date_end": date_range_end,
        "total_calendar_days": total_calendar_days,
        "expected_trading_days": expected_trading_days,
        "coverage_ratio": coverage_ratio,
        "frequency": frequency,
        "avg_gap_days": avg_gap_days,
        "max_gap_days": max_gap_days,
        "num_abnormal_gaps": num_abnormal_gaps,
        "top_gaps": top_gaps,
        "missing_pct_by_column": missing_pct,
        "total_missing_pct": total_missing_pct,
        "staleness_days": staleness_days,
        "quality_score": score,
        "quality_label": quality_label,
        "columns": list(df.columns),
    }

def run_data_audit():
    """Audit all data files."""
    files_to_audit = [
        ("gold_price.csv", "Gold Futures (GC=F)"),
        ("intermarket.csv", "Intermarket Assets"),
        ("gld_etf.csv", "SPDR Gold Shares (GLD)"),
        ("iau_etf.csv", "iShares Gold Trust (IAU)"),
        ("silver_price.csv", "Silver Futures (SI=F)"),
        ("seasonality.csv", "Seasonality Flags"),
        ("geopolitical_proxy.csv", "Geopolitical Risk Proxy"),
        ("usdchf.csv", "USD/CHF"),
    ]
    
    audits = []
    for filename, name in files_to_audit:
        filepath = os.path.join(DATA_DIR, filename)
        audit = audit_csv(filepath, name)
        audits.append(audit)
        print(f"  {name}: Score={audit['quality_score']} ({audit.get('quality_label', 'N/A')}), "
              f"Rows={audit.get('total_rows', 0)}, Gaps={audit.get('num_abnormal_gaps', 0)}")
    
    # Also audit intermarket sub-columns
    im_path = os.path.join(DATA_DIR, "intermarket.csv")
    if os.path.exists(im_path):
        im_df = pd.read_csv(im_path, index_col=0, parse_dates=True)
        im_sub = []
        for col in im_df.columns:
            valid = im_df[col].dropna()
            if len(valid) > 0:
                im_sub.append({
                    "ticker": col,
                    "valid_rows": len(valid),
                    "missing_pct": round(im_df[col].isna().mean() * 100, 2),
                    "first_date": valid.index.min().strftime("%Y-%m-%d"),
                    "last_date": valid.index.max().strftime("%Y-%m-%d"),
                })
        # Attach to intermarket audit
        for a in audits:
            if a["name"] == "Intermarket Assets":
                a["sub_feeds"] = im_sub
    
    return audits

# ============================================================
# 2. SIGNAL TIMEFRAME ANALYSIS
# ============================================================

def analyze_signal_timeframes():
    """Compute holding periods, turnover, and effective timeframe for each dimension."""
    signals_path = os.path.join(SIGNALS_DIR, "all_signals.csv")
    df = pd.read_csv(signals_path, index_col=0, parse_dates=True)
    
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
    
    timeframe_results = {}
    
    for dim_key, dim_name in DIM_NAMES.items():
        sig_col = f"{dim_key}_signal"
        if sig_col not in df.columns:
            continue
        
        signals = df[sig_col].dropna()
        
        # Compute holding periods (consecutive days with same signal)
        signal_changes = signals.diff().abs() > 0
        # Group consecutive same-signal periods
        groups = signal_changes.cumsum()
        holding_periods = []
        
        for group_id, group_data in signals.groupby(groups):
            if len(group_data) > 0:
                holding_periods.append({
                    "signal": int(group_data.iloc[0]),
                    "duration_days": len(group_data),
                    "start": group_data.index[0].strftime("%Y-%m-%d"),
                    "end": group_data.index[-1].strftime("%Y-%m-%d"),
                })
        
        durations = [h["duration_days"] for h in holding_periods]
        active_durations = [h["duration_days"] for h in holding_periods if h["signal"] != 0]
        
        # Statistics
        avg_holding = round(np.mean(durations), 1) if durations else 0
        median_holding = round(np.median(durations), 1) if durations else 0
        avg_active_holding = round(np.mean(active_durations), 1) if active_durations else 0
        median_active_holding = round(np.median(active_durations), 1) if active_durations else 0
        max_holding = max(durations) if durations else 0
        min_holding = min(durations) if durations else 0
        
        # Turnover: number of signal changes per year
        total_days = len(signals)
        total_changes = int(signal_changes.sum())
        years = total_days / 252
        annual_turnover = round(total_changes / max(years, 0.01), 1)
        
        # Signal distribution
        bullish_pct = round((signals == 1).mean() * 100, 1)
        neutral_pct = round((signals == 0).mean() * 100, 1)
        bearish_pct = round((signals == -1).mean() * 100, 1)
        
        # Time in market (non-neutral)
        time_in_market = round((signals != 0).mean() * 100, 1)
        
        # Classify timeframe
        if avg_active_holding <= 3:
            timeframe = "Very Short-Term (1-3 days)"
            timeframe_category = "scalp"
        elif avg_active_holding <= 7:
            timeframe = "Short-Term / Swing (3-7 days)"
            timeframe_category = "swing_short"
        elif avg_active_holding <= 20:
            timeframe = "Swing Trade (1-4 weeks)"
            timeframe_category = "swing"
        elif avg_active_holding <= 60:
            timeframe = "Position Trade (1-3 months)"
            timeframe_category = "position"
        else:
            timeframe = "Long-Term (3+ months)"
            timeframe_category = "long_term"
        
        # Holding period distribution buckets
        hp_dist = {
            "1-3 days": len([d for d in active_durations if d <= 3]),
            "4-7 days": len([d for d in active_durations if 4 <= d <= 7]),
            "1-2 weeks": len([d for d in active_durations if 8 <= d <= 14]),
            "2-4 weeks": len([d for d in active_durations if 15 <= d <= 28]),
            "1-3 months": len([d for d in active_durations if 29 <= d <= 90]),
            "3+ months": len([d for d in active_durations if d > 90]),
        }
        
        timeframe_results[dim_key] = {
            "name": dim_name,
            "avg_holding_days": avg_holding,
            "median_holding_days": median_holding,
            "avg_active_holding_days": avg_active_holding,
            "median_active_holding_days": median_active_holding,
            "max_holding_days": max_holding,
            "min_holding_days": min_holding,
            "total_signal_changes": total_changes,
            "annual_turnover": annual_turnover,
            "bullish_pct": bullish_pct,
            "neutral_pct": neutral_pct,
            "bearish_pct": bearish_pct,
            "time_in_market_pct": time_in_market,
            "timeframe": timeframe,
            "timeframe_category": timeframe_category,
            "holding_period_distribution": hp_dist,
            "num_trades": len(holding_periods),
        }
        
        print(f"  {dim_name}: Avg Hold={avg_active_holding}d, Turnover={annual_turnover}/yr, "
              f"Timeframe={timeframe}, In Market={time_in_market}%")
    
    return timeframe_results

# ============================================================
# 3. BLOOMBERG DATA VALUE ASSESSMENT
# ============================================================

def assess_bloomberg_value():
    """Assess what Bloomberg Terminal data would add vs current Yahoo Finance data."""
    
    assessment = {
        "current_source": {
            "provider": "Yahoo Finance (yfinance)",
            "cost": "Free",
            "frequency": "Daily (end-of-day)",
            "delay": "15-20 minutes (real-time for some tickers)",
            "coverage": "US equities, ETFs, futures, FX, crypto",
            "limitations": [
                "No intraday data below daily bars",
                "No real COT (Commitment of Traders) positioning data",
                "No real-time news sentiment scoring",
                "No ETF fund flow data (shares outstanding / AUM changes)",
                "No Geopolitical Risk Index (GPR) — using volatility proxy instead",
                "No central bank reserve data",
                "No real-time options flow / put-call ratios for gold",
                "Occasional data gaps and delayed updates",
                "No tick-level or order book data",
                "Limited historical depth for some tickers"
            ]
        },
        "bloomberg_advantages": [
            {
                "category": "Real-Time & Intraday Data",
                "impact": "HIGH",
                "description": "Bloomberg provides tick-level and intraday bars (1min, 5min, 15min, 1hr) for gold futures. This would enable intraday signal generation and significantly reduce signal lag from daily close to real-time.",
                "dimensions_improved": ["dim1", "dim4", "dim8"],
                "new_capability": "Could add intraday technical signals, session-based breakout detection, and real-time intermarket divergence alerts."
            },
            {
                "category": "COT Positioning Data",
                "impact": "HIGH",
                "description": "Bloomberg provides parsed CFTC Commitment of Traders data with managed money, commercial, and non-commercial positions. This is the gold standard for institutional positioning — far superior to our ETF volume proxy.",
                "dimensions_improved": ["dim5"],
                "new_capability": "Direct measurement of speculative vs commercial positioning, net long/short extremes, and position change momentum."
            },
            {
                "category": "ETF Fund Flows (AUM & Shares Outstanding)",
                "impact": "HIGH",
                "description": "Bloomberg tracks daily changes in GLD/IAU shares outstanding and AUM, which directly measure institutional inflows and outflows. Our current proxy uses volume patterns, which is a much noisier signal.",
                "dimensions_improved": ["dim5", "dim3"],
                "new_capability": "Direct ETF flow measurement, creation/redemption tracking, and flow momentum indicators."
            },
            {
                "category": "News Sentiment & Event Data",
                "impact": "MEDIUM-HIGH",
                "description": "Bloomberg News Sentiment (BNSE) provides machine-scored sentiment on gold-related news articles. Bloomberg also has event calendars with economic release timestamps and consensus estimates.",
                "dimensions_improved": ["dim3", "dim2"],
                "new_capability": "Real-time news sentiment scoring, economic surprise indices (actual vs consensus), and event-driven signal triggers."
            },
            {
                "category": "Central Bank Reserve Data",
                "impact": "MEDIUM",
                "description": "Bloomberg aggregates IMF and central bank gold reserve reports (monthly). This data is critical for understanding long-term demand trends, especially from China, India, and emerging market central banks.",
                "dimensions_improved": ["dim5"],
                "new_capability": "Central bank buying/selling trends, reserve diversification signals."
            },
            {
                "category": "Options Flow & Volatility Surface",
                "impact": "MEDIUM",
                "description": "Bloomberg provides gold options open interest, put-call ratios, implied volatility surface, and skew data. This would significantly improve the geopolitical risk dimension.",
                "dimensions_improved": ["dim7", "dim8"],
                "new_capability": "Implied vol term structure, skew-based fear indicators, options positioning extremes."
            },
            {
                "category": "Geopolitical Risk Index (GPR)",
                "impact": "MEDIUM",
                "description": "The Caldara-Iacoviello GPR Index is available on Bloomberg. Currently we use a volatility-based proxy, which captures some geopolitical effects but misses events that haven't yet impacted gold vol.",
                "dimensions_improved": ["dim7"],
                "new_capability": "Direct GPR measurement, GPR sub-indices (threats vs acts), country-specific risk scores."
            },
            {
                "category": "Cross-Asset Correlation Analytics",
                "impact": "LOW-MEDIUM",
                "description": "Bloomberg's PORT and correlation tools provide rolling cross-asset correlations, beta estimates, and factor exposures. We compute these manually, but Bloomberg's are more robust with corporate action adjustments.",
                "dimensions_improved": ["dim4"],
                "new_capability": "More accurate correlation estimates, regime-adjusted betas, factor decomposition."
            }
        ],
        "bloomberg_cost": {
            "terminal": "$24,000/year per seat",
            "b_pipe": "$1,500-3,000/month for data feed",
            "api": "Bloomberg B-QUANT or BLPAPI — requires terminal license",
            "alternatives": [
                {"name": "Refinitiv Eikon", "cost": "$3,600-22,000/year", "coverage": "Similar to Bloomberg, slightly less depth in commodities"},
                {"name": "Quandl (Nasdaq Data Link)", "cost": "$50-500/month", "coverage": "COT data, economic indicators, some ETF flows"},
                {"name": "FRED API", "cost": "Free", "coverage": "Macro data (CPI, NFP, Fed Funds) — already partially used"},
                {"name": "GoldAPI.io", "cost": "$10-100/month", "coverage": "Real-time gold prices, historical data"},
            ]
        },
        "recommendation": {
            "verdict": "Bloomberg would meaningfully improve 5 of 8 dimensions, with the biggest gains in Institutional Flows (COT + ETF flows), Sentiment (news scoring), and Geopolitical Risk (GPR Index). However, the $24K/year cost is significant. A more cost-effective approach would be to add Quandl for COT data ($50-100/month) and FRED for macro releases (free), which would improve 3 dimensions at <1% of Bloomberg's cost.",
            "priority_upgrades": [
                {"source": "Quandl / CFTC", "cost": "$50-100/month", "improves": "Dim 5 (Institutional Flows)", "impact": "HIGH"},
                {"source": "FRED API", "cost": "Free", "improves": "Dim 2 (Macro)", "impact": "MEDIUM-HIGH"},
                {"source": "GoldAPI.io", "cost": "$10-50/month", "improves": "Dim 1, 8 (Technical, Structure)", "impact": "MEDIUM"},
                {"source": "Bloomberg Terminal", "cost": "$24,000/year", "improves": "All dimensions", "impact": "HIGHEST (but expensive)"},
            ]
        }
    }
    
    return assessment

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("DATA QUALITY AUDIT & SIGNAL TIMEFRAME ANALYSIS")
    print("=" * 60)
    
    # 1. Data Quality Audit
    print("\n--- DATA QUALITY AUDIT ---")
    audits = run_data_audit()
    
    # Overall quality
    scores = [a["quality_score"] for a in audits if a.get("quality_score", 0) > 0]
    avg_score = round(np.mean(scores), 1) if scores else 0
    print(f"\n  Overall Average Quality Score: {avg_score}/100")
    
    # 2. Signal Timeframe Analysis
    print("\n--- SIGNAL TIMEFRAME ANALYSIS ---")
    timeframes = analyze_signal_timeframes()
    
    # 3. Bloomberg Assessment
    print("\n--- BLOOMBERG DATA ASSESSMENT ---")
    bloomberg = assess_bloomberg_value()
    print(f"  Recommendation: {bloomberg['recommendation']['verdict'][:100]}...")
    
    # 4. Save all results
    output = {
        "data_quality": {
            "overall_score": avg_score,
            "overall_label": "Good" if avg_score >= 70 else ("Fair" if avg_score >= 50 else "Poor"),
            "audit_date": "2026-03-19",
            "data_source": "Yahoo Finance (yfinance)",
            "data_frequency": "Daily (end-of-day close)",
            "data_period": "2020-01-01 to 2026-03-18",
            "feeds": audits,
        },
        "signal_timeframes": timeframes,
        "bloomberg_assessment": bloomberg,
    }
    
    output_path = os.path.join(OUTPUT_DIR, "data_quality.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nSaved to {output_path}")
    print("Done!")

if __name__ == "__main__":
    main()
