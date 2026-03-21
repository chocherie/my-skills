"""
Gold Trading Signal Engine V2 — 8 Dimensions (Bloomberg + GPR Enhanced)
=======================================================================
Upgraded from V1 to use:
  - Real CFTC COT data (Managed Money, Producer/Merchant, Swap Dealers)
  - Real Caldara-Iacoviello GPR Index (daily, with Threats + Acts sub-indices)
  - Bloomberg macro data (CPI surprise, real yields, fed funds, economic surprise)
  - Bloomberg market structure (futures OI, volume, GC1-GC2 spread)
  - Bloomberg ETF fundamentals (GLD/IAU shares outstanding, AUM)
"""
import pandas as pd
import numpy as np
import os
import json

DATA_DIR = "/home/ubuntu/gold_signals/data"
SIGNALS_DIR = "/home/ubuntu/gold_signals/signals"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df

def zscore(series, window=60):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)

def signal_from_score(score, bull_thresh=0.5, bear_thresh=-0.5):
    if pd.isna(score):
        return 0
    if score > bull_thresh:
        return 1
    elif score < bear_thresh:
        return -1
    return 0

def confidence_from_score(score, scale=2.0):
    if pd.isna(score):
        return 0
    return int(min(100, abs(score) / scale * 100))

# ============================================================
# DIMENSION 1: TECHNICAL INDICATORS (unchanged)
# ============================================================

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def dim1_technical(gold):
    close = gold["Close"]
    results = pd.DataFrame(index=gold.index)

    rsi = compute_rsi(close, 9)
    rsi_score = np.where(rsi < 30, 1.0, np.where(rsi > 70, -1.0, (50 - rsi) / 50))

    macd_line, signal_line, histogram = compute_macd(close, 8, 17, 6)
    macd_score = zscore(histogram, 60)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_pct = (close - bb_lower) / (bb_upper - bb_lower)
    bb_score = np.where(bb_pct < 0.2, 1.0, np.where(bb_pct > 0.8, -1.0, (0.5 - bb_pct) * 2))

    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    ma_score = zscore(ema50 - ema200, 60)

    composite = (pd.Series(rsi_score, index=gold.index) * 0.25 +
                 macd_score.fillna(0) * 0.30 +
                 pd.Series(bb_score, index=gold.index) * 0.20 +
                 ma_score.fillna(0) * 0.25)

    results["dim1_score"] = composite
    results["dim1_signal"] = composite.apply(lambda x: signal_from_score(x, 0.3, -0.3))
    results["dim1_confidence"] = composite.apply(lambda x: confidence_from_score(x, 1.5))
    results["rsi_9"] = rsi
    results["macd_hist"] = histogram
    results["bb_pct"] = bb_pct
    results["ema50"] = ema50
    results["ema200"] = ema200

    return results

# ============================================================
# DIMENSION 2: MACROECONOMIC DATA (Bloomberg Enhanced)
# ============================================================

def dim2_macro(gold, intermarket):
    """Enhanced with Bloomberg: real yields, CPI surprise, fed funds, economic surprise."""
    results = pd.DataFrame(index=gold.index)
    scores = []

    # Real Yield 10Y (Bloomberg USGGT10Y) — rising real yields = bearish for gold
    if "REAL_YIELD_10Y" in intermarket.columns:
        ry = intermarket["REAL_YIELD_10Y"].reindex(gold.index).ffill()
        ry_mom = -zscore(ry.diff(20), 60)  # negative: rising real yields = bearish
        scores.append(("real_yield", ry_mom, 0.30))
        results["real_yield_mom"] = ry_mom
    elif "TNX" in intermarket.columns:
        tnx = intermarket["TNX"].reindex(gold.index).ffill()
        tnx_mom = -zscore(tnx.diff(20), 60)
        scores.append(("tnx", tnx_mom, 0.30))

    # Economic Surprise Index (Bloomberg CESIUSD) — positive surprise = hawkish = bearish gold
    if "ECON_SURPRISE" in intermarket.columns:
        esi = intermarket["ECON_SURPRISE"].reindex(gold.index).ffill()
        esi_score = -zscore(esi, 60)  # positive surprise = bearish for gold
        scores.append(("econ_surprise", esi_score, 0.20))
        results["econ_surprise_score"] = esi_score

    # Fed Funds Rate momentum (Bloomberg FDTR)
    if "FED_FUNDS" in intermarket.columns:
        ff = intermarket["FED_FUNDS"].reindex(gold.index).ffill()
        ff_mom = -zscore(ff.diff(60), 120)  # rising rates = bearish
        scores.append(("fed_funds", ff_mom, 0.20))
    elif "TWO" in intermarket.columns:
        two = intermarket["TWO"].reindex(gold.index).ffill()
        two_mom = -zscore(two.diff(10), 60)
        scores.append(("two", two_mom, 0.20))

    # TIP (TIPS ETF) momentum — rising TIPS = falling real yields = bullish
    if "TIP" in intermarket.columns:
        tip = intermarket["TIP"].reindex(gold.index).ffill()
        tip_mom = zscore(tip.pct_change(20), 60)
        scores.append(("tip", tip_mom, 0.30))
        results["tip_mom"] = tip_mom

    if scores:
        total_weight = sum(w for _, _, w in scores)
        composite = sum(s.fillna(0) * (w / total_weight) for _, s, w in scores)
    else:
        composite = pd.Series(0, index=gold.index)

    results["dim2_score"] = composite
    results["dim2_signal"] = composite.apply(lambda x: signal_from_score(x, 0.3, -0.3))
    results["dim2_confidence"] = composite.apply(lambda x: confidence_from_score(x, 1.5))

    return results

# ============================================================
# DIMENSION 3: NEWS & SENTIMENT (Bloomberg Enhanced)
# ============================================================

def dim3_sentiment(gold, gld, iau, intermarket):
    """Enhanced with Bloomberg Economic Surprise Index as sentiment proxy."""
    results = pd.DataFrame(index=gold.index)
    close = gold["Close"]
    scores = []

    # GLD volume anomaly (contrarian)
    if gld is not None and "Volume" in gld.columns:
        gld_vol = gld["Volume"].reindex(gold.index).ffill()
        vol_zscore = zscore(gld_vol, 40)
        gld_close = gld["Close"].reindex(gold.index).ffill()
        gld_ret = gld_close.pct_change(5)
        sentiment_score = -vol_zscore * np.sign(gld_ret)
        scores.append(("etf_sentiment", sentiment_score, 0.35))

    # Gold premium/discount
    if gld is not None and "Close" in gld.columns:
        gld_close = gld["Close"].reindex(gold.index).ffill()
        ratio = close / gld_close
        ratio_z = zscore(ratio, 40)
        premium_score = -ratio_z
        scores.append(("premium", premium_score, 0.25))

    # Economic Surprise as sentiment gauge
    if intermarket is not None and "ECON_SURPRISE" in intermarket.columns:
        esi = intermarket["ECON_SURPRISE"].reindex(gold.index).ffill()
        # Extreme positive surprise = market euphoria = contrarian bearish for gold
        esi_extreme = -zscore(esi, 120)
        scores.append(("econ_surprise_sentiment", esi_extreme, 0.20))

    # GVZ (Gold Volatility Index) — high fear = bullish gold
    if intermarket is not None and "GVZ" in intermarket.columns:
        gvz = intermarket["GVZ"].reindex(gold.index).ffill()
        gvz_score = zscore(gvz, 60)
        scores.append(("gvz_fear", gvz_score, 0.20))

    if scores:
        total_weight = sum(w for _, _, w in scores)
        composite = sum(s.fillna(0) * (w / total_weight) for _, s, w in scores)
    else:
        composite = pd.Series(0, index=gold.index)

    results["dim3_score"] = composite
    results["dim3_signal"] = composite.apply(lambda x: signal_from_score(x, 0.4, -0.4))
    results["dim3_confidence"] = composite.apply(lambda x: confidence_from_score(x, 1.5))

    return results

# ============================================================
# DIMENSION 4: INTERMARKET ANALYSIS (Bloomberg Enhanced)
# ============================================================

def dim4_intermarket(gold, intermarket, mkt_struct):
    """Enhanced with Bloomberg real yields and GC1-GC2 contango/backwardation."""
    results = pd.DataFrame(index=gold.index)
    scores = []

    # DXY (inverse correlation)
    if "DXY" in intermarket.columns:
        dxy = intermarket["DXY"].reindex(gold.index).ffill()
        dxy_score = -zscore(dxy.pct_change(10), 60)
        scores.append(("dxy", dxy_score, 0.25))
        results["dxy_score"] = dxy_score

    # VIX (positive — fear = bullish gold)
    if "VIX" in intermarket.columns:
        vix = intermarket["VIX"].reindex(gold.index).ffill()
        vix_score = zscore(vix, 60)
        scores.append(("vix", vix_score, 0.15))
        results["vix_score"] = vix_score

    # Oil (positive correlation)
    if "OIL" in intermarket.columns:
        oil = intermarket["OIL"].reindex(gold.index).ffill()
        oil_score = zscore(oil.pct_change(10), 60)
        scores.append(("oil", oil_score, 0.10))

    # Real Yield 10Y (Bloomberg) — inverse: rising real yields = bearish gold
    if "REAL_YIELD_10Y" in intermarket.columns:
        ry = intermarket["REAL_YIELD_10Y"].reindex(gold.index).ffill()
        ry_score = -zscore(ry, 120)
        scores.append(("real_yield", ry_score, 0.20))
        results["real_yield_score"] = ry_score

    # GC1-GC2 spread (contango/backwardation)
    if mkt_struct is not None and "gc2_price" in mkt_struct.columns:
        gc2 = mkt_struct["gc2_price"].reindex(gold.index).ffill()
        gc1 = gold["Close"]
        spread = gc1 - gc2
        spread_z = zscore(spread, 120)
        # Backwardation (GC1 > GC2) = strong demand = bullish
        scores.append(("curve_spread", spread_z, 0.15))

    # JPY (safe haven)
    if "USDJPY" in intermarket.columns:
        usdjpy = intermarket["USDJPY"].reindex(gold.index).ffill()
        jpy_score = -zscore(usdjpy.pct_change(10), 60)
        scores.append(("jpy", jpy_score, 0.15))

    if scores:
        total_weight = sum(w for _, _, w in scores)
        composite = sum(s.fillna(0) * (w / total_weight) for _, s, w in scores)
    else:
        composite = pd.Series(0, index=gold.index)

    results["dim4_score"] = composite
    results["dim4_signal"] = composite.apply(lambda x: signal_from_score(x, 0.3, -0.3))
    results["dim4_confidence"] = composite.apply(lambda x: confidence_from_score(x, 1.5))

    return results

# ============================================================
# DIMENSION 5: INSTITUTIONAL FLOWS (Bloomberg COT + ETF Fundamentals)
# ============================================================

def dim5_institutional(gold, gld, iau, cot_df, etf_fund):
    """MAJOR UPGRADE: Real CFTC COT data + Bloomberg ETF shares outstanding/AUM."""
    results = pd.DataFrame(index=gold.index)
    scores = []

    # COT Managed Money Net Position (weekly, forward-filled to daily)
    if cot_df is not None and "managed_money_net" in cot_df.columns:
        mm_net = cot_df["managed_money_net"].reindex(gold.index).ffill()
        mm_z = zscore(mm_net, 52 * 5)  # ~1 year of weekly data in daily terms
        # Extreme long = crowded trade = contrarian bearish
        # But moderate long = trend following = bullish
        # Use non-linear: moderate positive = bullish, extreme = bearish
        mm_score = mm_z.copy()
        # Clip extremes for contrarian
        mm_score = mm_score.clip(-3, 3) / 3
        scores.append(("cot_managed_money", pd.Series(mm_score, index=gold.index), 0.30))
        results["cot_mm_net"] = mm_net
        results["cot_mm_zscore"] = mm_z

    # COT Producer/Merchant Net (commercial hedgers — contrarian indicator)
    if cot_df is not None and "producer_net" in cot_df.columns:
        prod_net = cot_df["producer_net"].reindex(gold.index).ffill()
        prod_z = zscore(prod_net, 52 * 5)
        # Producers are contrarian: extreme short = they're hedging = bullish
        prod_score = -prod_z.clip(-3, 3) / 3
        scores.append(("cot_producer", pd.Series(prod_score, index=gold.index), 0.20))

    # GLD Shares Outstanding (Bloomberg) — rising = inflows = bullish
    if etf_fund is not None and "gld_shares" in etf_fund.columns:
        gld_shares = etf_fund["gld_shares"].reindex(gold.index).ffill()
        gld_shares_mom = zscore(gld_shares.pct_change(20), 60)
        scores.append(("gld_shares_flow", gld_shares_mom, 0.20))
        results["gld_shares"] = gld_shares

    # IAU Shares Outstanding (Bloomberg)
    if etf_fund is not None and "iau_shares" in etf_fund.columns:
        iau_shares = etf_fund["iau_shares"].reindex(gold.index).ffill()
        iau_shares_mom = zscore(iau_shares.pct_change(20), 60)
        scores.append(("iau_shares_flow", iau_shares_mom, 0.15))

    # GLD/IAU combined AUM momentum
    if etf_fund is not None and "gld_aum" in etf_fund.columns:
        gld_aum = etf_fund["gld_aum"].reindex(gold.index).ffill()
        aum_mom = zscore(gld_aum.pct_change(20), 60)
        scores.append(("etf_aum_flow", aum_mom, 0.15))

    if scores:
        total_weight = sum(w for _, _, w in scores)
        composite = sum(s.fillna(0) * (w / total_weight) for _, s, w in scores)
    else:
        composite = pd.Series(0, index=gold.index)

    results["dim5_score"] = composite
    results["dim5_signal"] = composite.apply(lambda x: signal_from_score(x, 0.3, -0.3))
    results["dim5_confidence"] = composite.apply(lambda x: confidence_from_score(x, 1.5))

    return results

# ============================================================
# DIMENSION 6: SEASONALITY (unchanged)
# ============================================================

def dim6_seasonality(gold, seasonality):
    results = pd.DataFrame(index=gold.index)
    close = gold["Close"]

    # Reindex seasonality to gold's index
    seasonality = seasonality.reindex(gold.index)
    seasonality["month"] = gold.index.month
    seasonality["day_of_week"] = gold.index.dayofweek
    seasonality["is_indian_wedding_season"] = seasonality["month"].isin([11, 12, 1, 2, 3]).astype(int)
    seasonality["is_chinese_ny_period"] = ((seasonality["month"] == 1) | (seasonality["month"] == 2)).astype(int)
    seasonality["is_september"] = (seasonality["month"] == 9).astype(int)

    monthly_ret = close.resample("ME").last().pct_change()
    monthly_avg = monthly_ret.groupby(monthly_ret.index.month).mean()
    month_bias = seasonality["month"].map(monthly_avg)

    daily_ret = close.pct_change()
    dow_avg = daily_ret.groupby(daily_ret.index.dayofweek).mean()
    dow_bias = pd.Series(gold.index.dayofweek, index=gold.index).map(dow_avg)

    wedding_boost = seasonality["is_indian_wedding_season"].values * 0.2
    cny_boost = seasonality["is_chinese_ny_period"].values * 0.15
    sept_boost = seasonality["is_september"].values * 0.25

    month_z = zscore(month_bias, 252).fillna(0)
    dow_z = zscore(dow_bias, 252).fillna(0)
    cultural = pd.Series(wedding_boost + cny_boost + sept_boost, index=gold.index)

    composite = month_z * 0.35 + dow_z * 0.25 + cultural * 0.40

    results["dim6_score"] = composite
    results["dim6_signal"] = composite.apply(lambda x: signal_from_score(x, 0.15, -0.15))
    results["dim6_confidence"] = composite.apply(lambda x: confidence_from_score(x, 1.0))
    results["month_bias"] = month_bias
    results["cultural_boost"] = cultural

    return results

# ============================================================
# DIMENSION 7: GEOPOLITICAL RISK (Real GPR Index!)
# ============================================================

def dim7_geopolitical(gold, gpr_daily, intermarket):
    """MAJOR UPGRADE: Real Caldara-Iacoviello GPR Index with Threats + Acts sub-indices."""
    results = pd.DataFrame(index=gold.index)
    scores = []

    # GPR Index (daily)
    if gpr_daily is not None and "gpr_index" in gpr_daily.columns:
        gpr = gpr_daily["gpr_index"].reindex(gold.index).ffill()
        gpr_z = zscore(gpr, 120)
        # Rising GPR = rising geopolitical risk = bullish for gold (safe haven)
        scores.append(("gpr_level", gpr_z, 0.30))
        results["gpr_index"] = gpr
        results["gpr_zscore"] = gpr_z

        # GPR momentum (spike detection)
        gpr_mom = zscore(gpr.pct_change(5), 60)
        scores.append(("gpr_momentum", gpr_mom, 0.15))

    # GPR Threats sub-index
    if gpr_daily is not None and "gpr_threats" in gpr_daily.columns:
        threats = gpr_daily["gpr_threats"].reindex(gold.index).ffill()
        threats_z = zscore(threats, 120)
        scores.append(("gpr_threats", threats_z, 0.15))

    # GPR Acts sub-index
    if gpr_daily is not None and "gpr_acts" in gpr_daily.columns:
        acts = gpr_daily["gpr_acts"].reindex(gold.index).ffill()
        acts_z = zscore(acts, 120)
        scores.append(("gpr_acts", acts_z, 0.10))

    # GVZ (Gold Volatility Index from Bloomberg)
    if intermarket is not None and "GVZ" in intermarket.columns:
        gvz = intermarket["GVZ"].reindex(gold.index).ffill()
        gvz_z = zscore(gvz, 120)
        scores.append(("gvz", gvz_z, 0.15))
        results["gvz"] = gvz

    # VIX as backup fear gauge
    if intermarket is not None and "VIX" in intermarket.columns:
        vix = intermarket["VIX"].reindex(gold.index).ffill()
        vix_z = zscore(vix, 120)
        scores.append(("vix_fear", vix_z, 0.15))

    if scores:
        total_weight = sum(w for _, _, w in scores)
        composite = sum(s.fillna(0) * (w / total_weight) for _, s, w in scores)
    else:
        composite = pd.Series(0, index=gold.index)

    results["dim7_score"] = composite
    results["dim7_signal"] = composite.apply(lambda x: signal_from_score(x, 0.4, -0.4))
    results["dim7_confidence"] = composite.apply(lambda x: confidence_from_score(x, 1.5))

    return results

# ============================================================
# DIMENSION 8: MARKET STRUCTURE (Bloomberg Enhanced)
# ============================================================

def dim8_market_structure(gold, silver, mkt_struct):
    """Enhanced with Bloomberg: futures OI, volume, GC1-GC2 spread."""
    results = pd.DataFrame(index=gold.index)
    close = gold["Close"]
    scores = []

    # Gold-Silver Ratio
    if silver is not None and "Close" in silver.columns:
        silver_close = silver["Close"].reindex(gold.index).ffill()
        gs_ratio = close / silver_close.replace(0, np.nan)
        gs_z = zscore(gs_ratio, 120)
        gs_score = -gs_z  # extreme high ratio = gold overvalued = bearish
        scores.append(("gold_silver", gs_score, 0.25))
        results["gold_silver_ratio"] = gs_ratio

    # Price structure: higher highs / lower lows
    high = gold["High"] if "High" in gold.columns else close
    low = gold["Low"] if "Low" in gold.columns else close
    hh = (high.rolling(20).max() > high.shift(20).rolling(20).max()).astype(int)
    ll = (low.rolling(20).min() < low.shift(20).rolling(20).min()).astype(int)
    structure_score = pd.Series(np.where(hh & ~ll, 1.0, np.where(ll & ~hh, -1.0, 0.0)), index=gold.index)
    scores.append(("structure", structure_score, 0.25))
    results["structure_score"] = structure_score

    # Bloomberg Futures Open Interest momentum
    if mkt_struct is not None and "gc1_open_interest" in mkt_struct.columns:
        oi = mkt_struct["gc1_open_interest"].reindex(gold.index).ffill()
        oi_mom = zscore(oi.pct_change(20), 60)
        # Rising OI + rising price = strong trend = bullish
        price_dir = np.sign(close.pct_change(20))
        oi_confirm = oi_mom * price_dir
        scores.append(("oi_confirm", oi_confirm, 0.20))
        results["futures_oi"] = oi

    # Bloomberg Futures Volume trend
    if mkt_struct is not None and "gc1_volume" in mkt_struct.columns:
        vol = mkt_struct["gc1_volume"].reindex(gold.index).ffill()
        vol_trend = zscore(vol.rolling(10).mean(), 60)
        price_dir = np.sign(close.pct_change(10))
        vol_confirm = vol_trend * price_dir
        scores.append(("vol_confirm", vol_confirm, 0.15))

    # BCOMGC (Bloomberg Commodity Gold Sub-index) — trend confirmation
    if "BCOMGC" in (mkt_struct.columns if mkt_struct is not None else []):
        pass  # Already captured via intermarket
    
    # Fallback: YF volume if no Bloomberg data
    if not any(n in ["oi_confirm", "vol_confirm"] for n, _, _ in scores):
        if "Volume" in gold.columns:
            vol = gold["Volume"]
            vol_trend = zscore(vol.rolling(10).mean(), 60)
            price_dir = np.sign(close.pct_change(10))
            vol_confirm = vol_trend * price_dir
            scores.append(("vol_confirm_yf", vol_confirm, 0.15))

    if scores:
        total_weight = sum(w for _, _, w in scores)
        composite = sum(s.fillna(0) * (w / total_weight) for _, s, w in scores)
    else:
        composite = pd.Series(0, index=gold.index)

    results["dim8_score"] = composite
    results["dim8_signal"] = composite.apply(lambda x: signal_from_score(x, 0.3, -0.3))
    results["dim8_confidence"] = composite.apply(lambda x: confidence_from_score(x, 1.5))

    return results

# ============================================================
# MAIN: Generate all signals
# ============================================================

def generate_all_signals():
    print("=" * 70)
    print("GOLD SIGNAL ENGINE V2 — Bloomberg + GPR Enhanced")
    print("=" * 70)

    print("\nLoading data...")
    gold = load_csv("gold_price.csv")
    intermarket = load_csv("intermarket.csv")
    gld = load_csv("gld_etf.csv")
    iau = load_csv("iau_etf.csv")
    silver = load_csv("silver_price.csv")
    seasonality = load_csv("seasonality.csv")
    cot_df = load_csv("cot_data.csv")
    gpr_daily = load_csv("gpr_daily.csv")
    etf_fund = load_csv("etf_fundamentals.csv")
    mkt_struct = load_csv("market_structure_bbg.csv")

    if gold is None:
        raise ValueError("Gold price data not found!")

    print(f"Gold: {len(gold)} rows, {gold.index.min().date()} to {gold.index.max().date()}")
    print(f"COT: {len(cot_df) if cot_df is not None else 0} rows")
    print(f"GPR: {len(gpr_daily) if gpr_daily is not None else 0} rows")
    print(f"ETF Fund: {len(etf_fund) if etf_fund is not None else 0} rows")
    print(f"Mkt Struct: {len(mkt_struct) if mkt_struct is not None else 0} rows")

    print("\nGenerating signals for 8 dimensions...")

    print("  [1/8] Technical Indicators...")
    d1 = dim1_technical(gold)

    print("  [2/8] Macroeconomic Data (Bloomberg Enhanced)...")
    d2 = dim2_macro(gold, intermarket)

    print("  [3/8] News & Sentiment (Bloomberg Enhanced)...")
    d3 = dim3_sentiment(gold, gld, iau, intermarket)

    print("  [4/8] Intermarket Analysis (Bloomberg Enhanced)...")
    d4 = dim4_intermarket(gold, intermarket, mkt_struct)

    print("  [5/8] Institutional Flows (REAL COT + Bloomberg ETF)...")
    d5 = dim5_institutional(gold, gld, iau, cot_df, etf_fund)

    print("  [6/8] Seasonality...")
    d6 = dim6_seasonality(gold, seasonality)

    print("  [7/8] Geopolitical Risk (REAL GPR Index)...")
    d7 = dim7_geopolitical(gold, gpr_daily, intermarket)

    print("  [8/8] Market Structure (Bloomberg Enhanced)...")
    d8 = dim8_market_structure(gold, silver, mkt_struct)

    # Combine all
    master = gold[["Close"]].copy()
    master.columns = ["gold_close"]
    master["gold_return"] = master["gold_close"].pct_change()
    master["gold_fwd_return"] = master["gold_close"].pct_change().shift(-1)

    for dim_df in [d1, d2, d3, d4, d5, d6, d7, d8]:
        master = master.join(dim_df, how="left")

    # Save
    master.to_csv(os.path.join(SIGNALS_DIR, "all_signals.csv"))
    print(f"\nSaved master signal file: {master.shape}")
    print(f"Date range: {master.index.min().date()} to {master.index.max().date()}")

    # Signal summary
    dim_names = {
        "dim1": "Technical Indicators",
        "dim2": "Macroeconomic Data",
        "dim3": "News & Sentiment",
        "dim4": "Intermarket Analysis",
        "dim5": "Institutional Flows",
        "dim6": "Seasonality",
        "dim7": "Geopolitical Risk",
        "dim8": "Market Structure"
    }

    summary = {}
    for dim_key, dim_name in dim_names.items():
        sig_col = f"{dim_key}_signal"
        if sig_col in master.columns:
            counts = master[sig_col].value_counts().to_dict()
            latest_signal = int(master[sig_col].dropna().iloc[-1]) if not master[sig_col].dropna().empty else 0
            latest_conf = int(master[f"{dim_key}_confidence"].dropna().iloc[-1]) if f"{dim_key}_confidence" in master.columns else 0
            summary[dim_key] = {
                "name": dim_name,
                "latest_signal": latest_signal,
                "latest_confidence": latest_conf,
                "bullish_days": int(counts.get(1, 0)),
                "neutral_days": int(counts.get(0, 0)),
                "bearish_days": int(counts.get(-1, 0))
            }
            print(f"  {dim_name}: Latest={latest_signal} (conf={latest_conf}%), "
                  f"Bull={counts.get(1,0)}, Neutral={counts.get(0,0)}, Bear={counts.get(-1,0)}")

    with open(os.path.join(SIGNALS_DIR, "signal_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return master

if __name__ == "__main__":
    generate_all_signals()
