"""
Gold Trading Signal — Full Data Integration Pipeline
=====================================================
Merges:
  1. Yahoo Finance data (re-pulled back to 2005)
  2. Bloomberg COT data (grid1.xlsx — 2006-2026 weekly)
  3. Bloomberg macro/intermarket data (Book1.xlsx Sheet2 — 2005-2026 daily)
  4. GPR Index (daily recent + monthly historical)
Outputs unified CSVs for the signal engine.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import os
import json
from datetime import datetime

DATA_DIR = "/home/ubuntu/gold_signals/data"
UPLOAD_DIR = "/home/ubuntu/upload"
START = "2005-01-01"
END = "2026-03-20"

def download_yf(ticker, name):
    """Download data from Yahoo Finance."""
    print(f"  Downloading {name} ({ticker})...")
    try:
        df = yf.download(ticker, start=START, end=END, progress=False)
        if df.empty:
            print(f"  WARNING: No data for {ticker}")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None) if df.index.tz else df.index
        print(f"  Got {len(df)} rows for {name} ({df.index.min().date()} to {df.index.max().date()})")
        return df
    except Exception as e:
        print(f"  ERROR downloading {ticker}: {e}")
        return None


def parse_bloomberg_book1():
    """Parse Bloomberg data from Book1.xlsx Sheet2 (BDH format with date-value pairs)."""
    print("\n[BLOOMBERG] Parsing Book1.xlsx Sheet2...")
    import openpyxl
    wb = openpyxl.load_workbook(os.path.join(UPLOAD_DIR, "Book1.xlsx"), data_only=True)
    ws = wb["Sheet2"]

    # Map: column index -> (ticker, field)
    # Row 1 has tickers in even columns (C2=col2, C4=col4, etc.)
    # Row 3 has field names
    # Data starts at row 4: odd col = date, even col = value
    ticker_map = {}
    for col in range(2, ws.max_column + 1, 2):
        ticker = ws.cell(row=1, column=col + 1).value  # tickers in odd columns after first pair
        field = ws.cell(row=3, column=col + 1).value
        if ticker and not str(ticker).startswith("#"):
            ticker_map[col] = ticker

    # Actually the structure is: pairs of (date_col, value_col) starting from col 2
    # Let me re-parse based on what we know from the analysis
    series_data = {}

    # Column mapping from our analysis:
    # C6,C7 = GLD EQY_SH_OUT (date, value)
    # C8,C9 = GLD FUND_TOTAL_ASSETS
    # C10,C11 = IAU EQY_SH_OUT
    # C12,C13 = IAU FUND_TOTAL_ASSETS
    # C20,C21 = GVZ PX_LAST
    # C22,C23 = CESIUSD PX_LAST
    # C24,C25 = BCOMGC PX_LAST
    # C26,C27 = CPI YOY PX_LAST
    # C28,C29 = CPUPAXFE PX_LAST
    # C30,C31 = NFP TCH PX_LAST
    # C32,C33 = FDTR PX_LAST
    # C34,C35 = USGGT10Y PX_LAST
    # C36,C37 = GC1 FUT_AGGTE_OPEN_INT
    # C38,C39 = GC1 PX_VOLUME
    # C40,C41 = GC2 PX_LAST

    col_map = {
        "gld_shares": (6, 7),
        "gld_aum": (8, 9),
        "iau_shares": (10, 11),
        "iau_aum": (12, 13),
        "gvz": (20, 21),
        "cesiusd": (22, 23),
        "bcomgc": (24, 25),
        "cpi_yoy": (26, 27),
        "core_cpi_yoy": (28, 29),
        "nfp_change": (30, 31),
        "fed_funds": (32, 33),
        "real_yield_10y": (34, 35),
        "gc1_open_interest": (36, 37),
        "gc1_volume": (38, 39),
        "gc2_price": (40, 41),
    }

    for name, (date_col, val_col) in col_map.items():
        dates = []
        values = []
        for row in range(4, ws.max_row + 1):
            d = ws.cell(row=row, column=date_col).value
            v = ws.cell(row=row, column=val_col).value
            if d is not None and v is not None:
                if isinstance(v, str) and v.startswith("#"):
                    continue
                try:
                    dates.append(pd.to_datetime(d))
                    values.append(float(v))
                except (ValueError, TypeError):
                    continue
        if dates:
            s = pd.Series(values, index=pd.DatetimeIndex(dates), name=name)
            s = s[~s.index.duplicated(keep='first')].sort_index()
            series_data[name] = s
            print(f"  {name}: {len(s)} rows, {s.index.min().date()} to {s.index.max().date()}")
        else:
            print(f"  {name}: NO DATA")

    return series_data


def parse_cot_data():
    """Parse Bloomberg COT data from grid1.xlsx."""
    print("\n[BLOOMBERG] Parsing COT data from grid1.xlsx...")
    import openpyxl
    wb = openpyxl.load_workbook(os.path.join(UPLOAD_DIR, "grid1.xlsx"), data_only=True)
    ws = wb.active

    # Tickers in row 1: col 13=CFFDUMML, 15=CFFDUMMS, 17=CFFDUMMN, 19=CFFDUPML, 21=CFFDUPMS, 23=CFFDUPMN, 25=CFFDUSWN
    # Dates in col 12, values in col 13,15,17,19,21,23,25
    cot_cols = {
        "managed_money_long": 13,
        "managed_money_short": 15,
        "managed_money_net": 17,
        "producer_long": 19,
        "producer_short": 21,
        "producer_net": 23,
        "swap_dealers_net": 25,
    }

    cot_data = {}
    for name, val_col in cot_cols.items():
        dates = []
        values = []
        for row in range(4, ws.max_row + 1):
            d = ws.cell(row=row, column=12).value  # date column
            v = ws.cell(row=row, column=val_col).value
            if d is not None and v is not None:
                try:
                    dates.append(pd.to_datetime(d))
                    values.append(float(v))
                except (ValueError, TypeError):
                    continue
        if dates:
            s = pd.Series(values, index=pd.DatetimeIndex(dates), name=name)
            s = s[~s.index.duplicated(keep='first')].sort_index()
            cot_data[name] = s
            print(f"  {name}: {len(s)} rows, {s.index.min().date()} to {s.index.max().date()}")

    cot_df = pd.DataFrame(cot_data)
    return cot_df


def parse_gpr_data():
    """Parse GPR Index from daily and monthly files."""
    print("\n[GPR] Parsing GPR Index data...")

    # Daily recent data
    gpr_daily = pd.read_excel(os.path.join(UPLOAD_DIR, "data_gpr_daily_recent.xls"))
    gpr_daily["date"] = pd.to_datetime(gpr_daily["date"])
    gpr_daily = gpr_daily.set_index("date")
    gpr_daily = gpr_daily[["GPRD", "GPRD_ACT", "GPRD_THREAT"]].copy()
    gpr_daily.columns = ["gpr_index", "gpr_acts", "gpr_threats"]
    gpr_daily = gpr_daily[~gpr_daily.index.duplicated(keep='first')].sort_index()
    print(f"  GPR Daily: {len(gpr_daily)} rows, {gpr_daily.index.min().date()} to {gpr_daily.index.max().date()}")

    # Monthly historical (for reference)
    gpr_monthly = pd.read_excel(os.path.join(UPLOAD_DIR, "data_gpr_export.xls"))
    gpr_monthly["month"] = pd.to_datetime(gpr_monthly["month"])
    gpr_monthly = gpr_monthly.set_index("month")
    gpr_monthly = gpr_monthly[["GPR", "GPRT", "GPRA"]].copy()
    gpr_monthly.columns = ["gpr_monthly", "gpr_threats_monthly", "gpr_acts_monthly"]
    print(f"  GPR Monthly: {len(gpr_monthly)} rows, {gpr_monthly.index.min().date()} to {gpr_monthly.index.max().date()}")

    return gpr_daily, gpr_monthly


def main():
    print("=" * 70)
    print("GOLD SIGNAL — FULL DATA INTEGRATION PIPELINE")
    print(f"Target period: {START} to {END}")
    print("=" * 70)

    # ============================================================
    # STEP 1: Re-pull Yahoo Finance data back to 2005
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 1: Yahoo Finance data (2005-2026)")
    print("=" * 70)

    # Gold futures
    print("\n[YF] Gold Price (GC=F)")
    gold = download_yf("GC=F", "Gold Futures")
    if gold is not None:
        gold.to_csv(os.path.join(DATA_DIR, "gold_price.csv"))

    # Silver
    print("\n[YF] Silver Price (SI=F)")
    silver = download_yf("SI=F", "Silver Futures")
    if silver is not None:
        silver.to_csv(os.path.join(DATA_DIR, "silver_price.csv"))

    # Intermarket assets
    print("\n[YF] Intermarket Assets")
    intermarket_tickers = {
        "DXY": "DX-Y.NYB",
        "TIP": "TIP",
        "TNX": "^TNX",
        "TWO": "^IRX",
        "VIX": "^VIX",
        "SPX": "^GSPC",
        "OIL": "CL=F",
        "BTC": "BTC-USD",
        "USDJPY": "JPY=X",
    }
    intermarket_data = {}
    for name, ticker in intermarket_tickers.items():
        df = download_yf(ticker, name)
        if df is not None:
            intermarket_data[name] = df["Close"]

    # GLD and IAU (for volume data — shares outstanding comes from Bloomberg)
    print("\n[YF] GLD and IAU ETFs")
    gld_yf = download_yf("GLD", "SPDR Gold Shares")
    if gld_yf is not None:
        gld_yf.to_csv(os.path.join(DATA_DIR, "gld_etf.csv"))

    iau_yf = download_yf("IAU", "iShares Gold Trust")
    if iau_yf is not None:
        iau_yf.to_csv(os.path.join(DATA_DIR, "iau_etf.csv"))

    # ============================================================
    # STEP 2: Parse Bloomberg data
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 2: Bloomberg data integration")
    print("=" * 70)

    bbg = parse_bloomberg_book1()
    cot_df = parse_cot_data()
    gpr_daily, gpr_monthly = parse_gpr_data()

    # ============================================================
    # STEP 3: Merge Bloomberg data into intermarket
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 3: Merging all data sources")
    print("=" * 70)

    # Add Bloomberg macro data to intermarket
    if "real_yield_10y" in bbg:
        intermarket_data["REAL_YIELD_10Y"] = bbg["real_yield_10y"]
    if "fed_funds" in bbg:
        intermarket_data["FED_FUNDS"] = bbg["fed_funds"]
    if "cesiusd" in bbg:
        intermarket_data["ECON_SURPRISE"] = bbg["cesiusd"]
    if "gvz" in bbg:
        intermarket_data["GVZ"] = bbg["gvz"]
    if "cpi_yoy" in bbg:
        intermarket_data["CPI_YOY"] = bbg["cpi_yoy"]
    if "core_cpi_yoy" in bbg:
        intermarket_data["CORE_CPI_YOY"] = bbg["core_cpi_yoy"]
    if "nfp_change" in bbg:
        intermarket_data["NFP_CHANGE"] = bbg["nfp_change"]
    if "bcomgc" in bbg:
        intermarket_data["BCOMGC"] = bbg["bcomgc"]

    # Save enriched intermarket
    if intermarket_data:
        im_df = pd.DataFrame(intermarket_data)
        im_df.to_csv(os.path.join(DATA_DIR, "intermarket.csv"))
        print(f"  Saved enriched intermarket: {im_df.shape}, {im_df.index.min()} to {im_df.index.max()}")

    # Save COT data
    cot_df.to_csv(os.path.join(DATA_DIR, "cot_data.csv"))
    print(f"  Saved COT data: {cot_df.shape}")

    # Save GPR data
    gpr_daily.to_csv(os.path.join(DATA_DIR, "gpr_daily.csv"))
    gpr_monthly.to_csv(os.path.join(DATA_DIR, "gpr_monthly.csv"))
    print(f"  Saved GPR daily: {gpr_daily.shape}")

    # Save Bloomberg ETF fundamentals
    etf_fund = pd.DataFrame()
    for name in ["gld_shares", "gld_aum", "iau_shares", "iau_aum"]:
        if name in bbg:
            etf_fund[name] = bbg[name]
    if not etf_fund.empty:
        etf_fund.to_csv(os.path.join(DATA_DIR, "etf_fundamentals.csv"))
        print(f"  Saved ETF fundamentals: {etf_fund.shape}")

    # Save Bloomberg market structure data
    mkt_struct = pd.DataFrame()
    for name in ["gc1_open_interest", "gc1_volume", "gc2_price"]:
        if name in bbg:
            mkt_struct[name] = bbg[name]
    if not mkt_struct.empty:
        mkt_struct.to_csv(os.path.join(DATA_DIR, "market_structure_bbg.csv"))
        print(f"  Saved market structure (BBG): {mkt_struct.shape}")

    # Generate seasonality for extended period
    if gold is not None:
        gold_close = gold["Close"].dropna()
        season_df = pd.DataFrame(index=gold_close.index)
        season_df["month"] = season_df.index.month
        season_df["day_of_week"] = season_df.index.dayofweek
        season_df["is_indian_wedding_season"] = season_df["month"].isin([11, 12, 1, 2, 3]).astype(int)
        season_df["is_chinese_ny_period"] = ((season_df["month"] == 1) | (season_df["month"] == 2)).astype(int)
        season_df["is_september"] = (season_df["month"] == 9).astype(int)
        season_df["quarter"] = season_df.index.quarter
        season_df.to_csv(os.path.join(DATA_DIR, "seasonality.csv"))
        print(f"  Saved seasonality: {season_df.shape}")

    # Generate geopolitical proxy (now enhanced with real GPR)
    if gold is not None:
        gold_close = gold["Close"].dropna()
        geo_df = pd.DataFrame(index=gold_close.index)
        geo_df["gold_realized_vol_20d"] = gold_close.pct_change().rolling(20).std() * np.sqrt(252)
        geo_df["gold_realized_vol_5d"] = gold_close.pct_change().rolling(5).std() * np.sqrt(252)
        geo_df["vol_of_vol"] = geo_df["gold_realized_vol_20d"].rolling(20).std()
        geo_df.to_csv(os.path.join(DATA_DIR, "geopolitical_proxy.csv"))
        print(f"  Saved geopolitical proxy: {geo_df.shape}")

    # ============================================================
    # STEP 4: Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("DATA INTEGRATION COMPLETE — SUMMARY")
    print("=" * 70)

    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
    summary = {}
    for f in files:
        df = pd.read_csv(os.path.join(DATA_DIR, f), index_col=0, parse_dates=True)
        info = {
            "rows": len(df),
            "columns": list(df.columns)[:8],
            "date_range": f"{df.index.min().date()} to {df.index.max().date()}" if len(df) > 0 else "empty"
        }
        summary[f] = info
        print(f"  {f}: {info['rows']} rows, {info['date_range']}")

    with open(os.path.join(DATA_DIR, "data_summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2, default=str)

    print("\nAll data ready for signal generation!")


if __name__ == "__main__":
    main()
