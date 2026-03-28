"""
data_layer.py — 10Y UST Signal Dashboard
=========================================
FIXED FILE: Do not modify evaluate_sharpe() or compute_buy_and_hold().
These are the immutable ground-truth metrics for the autoresearch loop.

Downloads and caches all data series needed by the 25 signals.
Returns a single aligned master DataFrame.

Usage:
    python data_layer.py          # download / refresh all data
    from data_layer import load_data
    df = load_data()
"""

import os
import math
import time
import warnings
import datetime
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants (fixed — do not modify)
# ---------------------------------------------------------------------------

START_DATE = "2003-01-01"
CACHE_DIR = Path.home() / ".cache" / "treasury_signals"
CACHE_MAX_AGE_HOURS = 24          # re-download if older than this
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

# Duration approximation for extended return series
UST_10Y_DURATION = 8.5            # approximate modified duration of 10Y note


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}.parquet"


def _is_stale(path: Path) -> bool:
    if not path.exists():
        return True
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours > CACHE_MAX_AGE_HOURS


def _save(df: pd.DataFrame, name: str):
    df.to_parquet(_cache_path(name))


def _load(name: str):
    p = _cache_path(name)
    if p.exists():
        return pd.read_parquet(p)
    return None


# ---------------------------------------------------------------------------
# FRED downloader
# ---------------------------------------------------------------------------

def _fred_get(series_id: str, start: str = "1990-01-01") -> pd.Series:
    """Download a single FRED series via REST API."""
    if not FRED_API_KEY:
        try:
            import fredapi
            fred = fredapi.Fred()
            return fred.get_series(series_id, observation_start=start)
        except Exception:
            pass
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}&vintage_date={datetime.date.today()}"
    )
    if FRED_API_KEY:
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}&observation_start={start}"
            f"&api_key={FRED_API_KEY}&file_type=json"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()["observations"]
        s = pd.Series(
            {pd.Timestamp(d["date"]): float(d["value"]) if d["value"] != "." else np.nan
             for d in data},
            name=series_id,
        )
    else:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        from io import StringIO
        s = pd.read_csv(StringIO(resp.text), index_col=0, parse_dates=True).squeeze()
        s.name = series_id
    return s.replace(".", np.nan).astype(float)


def _fred_batch(series_map: dict, start: str = "1990-01-01") -> pd.DataFrame:
    """Download multiple FRED series. series_map: {col_name: fred_id}"""
    frames = {}
    for col, sid in series_map.items():
        try:
            print(f"    FRED {sid} → {col}")
            frames[col] = _fred_get(sid, start)
        except Exception as e:
            print(f"    WARNING: {sid} failed: {e}")
            frames[col] = pd.Series(dtype=float, name=col)
    return pd.DataFrame(frames)


# ---------------------------------------------------------------------------
# Individual downloaders
# ---------------------------------------------------------------------------

def _download_ief() -> pd.DataFrame:
    import yfinance as yf
    print("  Downloading IEF (7-10Y Treasury ETF)...")
    ief = yf.download("IEF", start="2002-07-01", auto_adjust=True, progress=False)
    df = pd.DataFrame({
        "ief_close": ief["Close"].squeeze(),
        "ief_return": ief["Close"].squeeze().pct_change(),
    })
    return df


def _download_yfinance_extras() -> pd.DataFrame:
    import yfinance as yf
    tickers = {"vix": "^VIX", "oil": "CL=F"}
    frames = {}
    for col, ticker in tickers.items():
        print(f"  Downloading {ticker} → {col}...")
        try:
            data = yf.download(ticker, start="1990-01-01", auto_adjust=True, progress=False)
            frames[col] = data["Close"].squeeze()
        except Exception as e:
            print(f"  WARNING: {ticker} failed: {e}")
            frames[col] = pd.Series(dtype=float, name=col)
    return pd.DataFrame(frames)


def _download_fred_yields() -> pd.DataFrame:
    print("  Downloading FRED yield curve series...")
    series = {
        "dgs3m": "DTB3",
        "dgs6m": "DTB6",
        "dgs1":  "DGS1",
        "dgs2":  "DGS2",
        "dgs3":  "DGS3",
        "dgs5":  "DGS5",
        "dgs7":  "DGS7",
        "dgs10": "DGS10",
        "dgs20": "DGS20",
        "dgs30": "DGS30",
        "t10y2y":  "T10Y2Y",
        "t10yie":  "T10YIE",
        "dfii10":  "DFII10",
    }
    return _fred_batch(series, start="1990-01-01")


def _download_fred_macro() -> pd.DataFrame:
    print("  Downloading FRED macro series...")
    series = {
        "cpiaucsl": "CPIAUCSL",
        "fedfunds": "FEDFUNDS",
        "sofr":     "SOFR",
        "rrp":      "RRPONTSYD",
        "farbast":  "FARBAST",
        "hy_spread":"BAMLH0A0HYM2",
        "dxy":      "DTWEXBGS",
        "de10y":    "IRLTLT01DEM156N",
        "jp10y":    "IRLTLT01JPM156N",
    }
    return _fred_batch(series, start="1990-01-01")


def _download_acm_term_premium() -> pd.Series:
    """Download NY Fed ACM term premium model data."""
    print("  Downloading NY Fed ACM term premium...")
    url = "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls"
    try:
        df = pd.read_excel(url, index_col=0, parse_dates=True)
        # Column ACMTP10 = 10-year term premium
        if "ACMTP10" in df.columns:
            s = df["ACMTP10"].dropna()
            s.name = "acm_tp"
            return s
        # Try first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 5:
            s = df.iloc[:, 4].dropna()  # typically 5th col is 10Y TP
            s.name = "acm_tp"
            return s
    except Exception as e:
        print(f"  WARNING: ACM download failed: {e}")
    return pd.Series(dtype=float, name="acm_tp")


def _download_cot() -> pd.Series:
    """Download CFTC COT Leveraged Funds net position for 10Y Treasury futures."""
    print("  Downloading CFTC COT (Leveraged Funds, 10Y)...")
    try:
        import cot_reports as cot
        # TFF report: Traders in Financial Futures
        df = cot.cot_all(cot_report_type="traders_in_financial_futures_fut")
        # Filter for 10-Year T-Notes
        mask = df["Market and Exchange Names"].str.contains(
            "10-YEAR U.S. TREASURY NOTES", case=False, na=False
        )
        tnotes = df[mask].copy()
        if tnotes.empty:
            print("  WARNING: 10Y T-Note COT data not found")
            return pd.Series(dtype=float, name="cot_lev_net")
        tnotes["date"] = pd.to_datetime(tnotes["As of Date in Form YYYY-MM-DD"])
        tnotes = tnotes.set_index("date").sort_index()
        # Leveraged Money net = longs - shorts
        long_col = "Leveraged Money Positions-Long (All)"
        short_col = "Leveraged Money Positions-Short (All)"
        if long_col in tnotes.columns and short_col in tnotes.columns:
            net = tnotes[long_col] - tnotes[short_col]
        else:
            # Try alternative column names
            long_cols = [c for c in tnotes.columns if "Leveraged" in c and "Long" in c]
            short_cols = [c for c in tnotes.columns if "Leveraged" in c and "Short" in c]
            if long_cols and short_cols:
                net = tnotes[long_cols[0]] - tnotes[short_cols[0]]
            else:
                print("  WARNING: COT Leveraged Fund columns not found")
                return pd.Series(dtype=float, name="cot_lev_net")
        net.name = "cot_lev_net"
        return net.astype(float)
    except Exception as e:
        print(f"  WARNING: COT download failed: {e}")
        return pd.Series(dtype=float, name="cot_lev_net")


def _download_fomc_sentiment() -> pd.Series:
    """Scrape FOMC minutes from Fed website and score with VADER."""
    print("  Downloading FOMC minutes sentiment (VADER)...")
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()

        # Hawkish and dovish word lists for bond-specific scoring
        hawkish_words = {
            "inflation", "inflationary", "overshoot", "tighten", "tightening",
            "restrictive", "hike", "raise", "elevated", "persistent", "robust",
            "resilient", "above", "target",
        }
        dovish_words = {
            "slowdown", "recession", "downside", "ease", "easing", "cut",
            "accommodate", "below", "weakness", "uncertainty", "soften", "moderate",
        }

        # Scrape FOMC meeting dates and minutes links
        base_url = "https://www.federalreserve.gov"
        meetings_url = f"{base_url}/monetarypolicy/fomc_historical.htm"
        resp = requests.get(meetings_url, timeout=30)
        soup = BeautifulSoup(resp.text, "html.parser")

        scores = {}
        links = soup.find_all("a", href=True)
        minutes_links = [
            l for l in links
            if "minutes" in l.get_text(strip=True).lower()
            and "fomc" in l["href"].lower()
        ][:20]  # limit to recent 20 for speed

        for link in minutes_links[:8]:  # limit per run
            try:
                href = link["href"]
                if not href.startswith("http"):
                    href = base_url + href
                resp2 = requests.get(href, timeout=20)
                text = BeautifulSoup(resp2.text, "html.parser").get_text(separator=" ")
                # Simple lexical hawk/dove scoring
                words = text.lower().split()
                hawk_count = sum(1 for w in words if any(h in w for h in hawkish_words))
                dove_count = sum(1 for w in words if any(d in w for d in dovish_words))
                total = hawk_count + dove_count
                if total > 0:
                    hawk_score = hawk_count / total
                else:
                    hawk_score = 0.5
                # Extract approximate date from URL
                import re
                date_match = re.search(r"(\d{8})", href)
                if date_match:
                    date_str = date_match.group(1)
                    date = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}")
                    scores[date] = hawk_score
            except Exception:
                continue

        if not scores:
            print("  WARNING: No FOMC sentiment scraped, using neutral 0.5")
            return pd.Series(dtype=float, name="fomc_hawk_score")

        s = pd.Series(scores, name="fomc_hawk_score").sort_index()
        return s
    except Exception as e:
        print(f"  WARNING: FOMC sentiment failed: {e}")
        return pd.Series(dtype=float, name="fomc_hawk_score")


def _download_primary_dealer() -> pd.Series:
    """Download NY Fed primary dealer positioning (6-11Y net)."""
    print("  Downloading NY Fed primary dealer positions...")
    try:
        # NY Fed primary dealer statistics
        url = "https://www.newyorkfed.org/medialibrary/media/markets/primarydealer/primarydealer.xlsx"
        df = pd.read_excel(url, index_col=0, parse_dates=True, header=1)
        # Look for 6-11Y column
        cols_11y = [c for c in df.columns if "6" in str(c) and "11" in str(c)]
        if cols_11y:
            s = df[cols_11y[0]].dropna().astype(float)
            s.name = "pd_net_6_11y"
            return s
        print("  WARNING: Primary dealer 6-11Y column not found")
    except Exception as e:
        print(f"  WARNING: Primary dealer data failed: {e}")
    return pd.Series(dtype=float, name="pd_net_6_11y")


# ---------------------------------------------------------------------------
# Master data builder
# ---------------------------------------------------------------------------

def download_all_data(force_refresh: bool = False) -> pd.DataFrame:
    """Download all data series and return aligned master DataFrame."""
    master_cache = _cache_path("master_df")
    if not force_refresh and not _is_stale(master_cache):
        print("Loading master_df from cache...")
        return pd.read_parquet(master_cache)

    print("=" * 60)
    print("DOWNLOADING ALL DATA SERIES")
    print("=" * 60)

    # 1. IEF (primary return series)
    ief = _download_ief()

    # 2. Extra yfinance (VIX, oil)
    yf_extra = _download_yfinance_extras()

    # 3. FRED yields
    fred_yields = _download_fred_yields()

    # 4. FRED macro
    fred_macro = _download_fred_macro()

    # 5. ACM term premium
    acm_tp = _download_acm_term_premium()

    # 6. COT
    cot = _download_cot()

    # 7. FOMC sentiment
    fomc = _download_fomc_sentiment()

    # 8. Primary dealer
    pd_data = _download_primary_dealer()

    # ---------------------------------------------------------------------------
    # Build master DataFrame: daily business day index from START_DATE to today
    # ---------------------------------------------------------------------------
    today = pd.Timestamp.today().normalize()
    idx = pd.bdate_range(start=START_DATE, end=today, freq="B")
    master = pd.DataFrame(index=idx)

    # Join IEF
    master = master.join(ief, how="left")

    # Join yfinance extras
    master = master.join(yf_extra, how="left")

    # Join FRED yields (forward-fill weekends/holidays)
    yields_daily = fred_yields.reindex(idx).ffill()
    master = master.join(yields_daily, how="left")

    # Join FRED macro (forward-fill; monthly series stay constant until next release)
    macro_daily = fred_macro.reindex(idx).ffill()
    master = master.join(macro_daily, how="left")

    # Join ACM term premium
    if not acm_tp.empty:
        master["acm_tp"] = acm_tp.reindex(idx).ffill()

    # Join COT (weekly, 3-day publication lag — shift by 3 extra days)
    if not cot.empty:
        cot_daily = cot.reindex(idx).ffill().shift(3)  # 3-day lag
        master["cot_lev_net"] = cot_daily

    # Join FOMC sentiment (hold constant between releases)
    if not fomc.empty:
        fomc_daily = fomc.reindex(idx).ffill()
        master["fomc_hawk_score"] = fomc_daily

    # Join primary dealer (weekly)
    if not pd_data.empty:
        pd_daily = pd_data.reindex(idx).ffill()
        master["pd_net_6_11y"] = pd_daily

    # ---------------------------------------------------------------------------
    # Computed columns
    # ---------------------------------------------------------------------------

    # Extended return series (duration approximation)
    if "dgs10" in master.columns and "sofr" in master.columns:
        yield_chg = master["dgs10"].diff() / 100
        daily_carry = master["dgs10"].shift(1) / 100 / 252
        # Use SOFR where available, fall back to fedfunds
        funding = master["sofr"].fillna(master.get("fedfunds", pd.Series(0, index=idx)))
        funding = funding / 100 / 252
        master["ust_return_approx"] = -UST_10Y_DURATION * yield_chg + daily_carry - funding

    # DXY from FRED is an index; convert to float
    if "dxy" in master.columns:
        master["dxy"] = pd.to_numeric(master["dxy"], errors="coerce")

    # Drop rows outside our window
    master = master[master.index >= START_DATE]

    # Save cache
    _save(master, "master_df")
    print(f"\nMaster DataFrame: {master.shape[0]} rows × {master.shape[1]} cols")
    print(f"Date range: {master.index[0].date()} to {master.index[-1].date()}")
    print(f"Columns: {list(master.columns)}")

    return master


def load_data(force_refresh: bool = False) -> pd.DataFrame:
    """Public entry point. Returns aligned master DataFrame."""
    return download_all_data(force_refresh=force_refresh)


# ---------------------------------------------------------------------------
# IMMUTABLE EVALUATION FUNCTIONS — DO NOT MODIFY
# These are the ground-truth metrics for the autoresearch loop.
# ---------------------------------------------------------------------------

def evaluate_sharpe(signal: pd.Series, returns: pd.Series) -> float:
    """
    Compute annualized Sharpe ratio for a signal vs returns.
    This is the immutable ground-truth metric for the autoresearch loop.
    Signal is shifted by 1 day to prevent look-ahead bias.

    Args:
        signal: pd.Series of positions {-1, 0, 1} (or continuous)
        returns: pd.Series of daily returns (decimal, e.g. 0.01 = 1%)

    Returns:
        float: annualized Sharpe ratio
    """
    aligned = pd.concat([signal, returns], axis=1).dropna()
    if aligned.empty or len(aligned) < 63:
        return 0.0
    sig = aligned.iloc[:, 0]
    ret = aligned.iloc[:, 1]
    strat = sig.shift(1) * ret
    strat = strat.dropna()
    if strat.std() == 0:
        return 0.0
    return float(strat.mean() / strat.std() * math.sqrt(252))


def compute_buy_and_hold(returns: pd.Series) -> dict:
    """
    Compute buy-and-hold metrics (always long).
    This is the immutable B&H benchmark for comparison.

    Returns:
        dict with keys: sharpe, total_return, annual_return, max_drawdown
    """
    ret = returns.dropna()
    if len(ret) == 0:
        return {"sharpe": 0.0, "total_return": 0.0, "annual_return": 0.0, "max_drawdown": 0.0}

    equity = (1 + ret).cumprod()
    total_ret = float(equity.iloc[-1] - 1)
    n_years = len(ret) / 252
    annual_ret = float((1 + total_ret) ** (1 / max(n_years, 0.01)) - 1)
    sharpe = float(ret.mean() / ret.std() * math.sqrt(252)) if ret.std() > 0 else 0.0

    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = float(dd.min())

    return {
        "sharpe": round(sharpe, 3),
        "total_return": round(total_ret * 100, 2),
        "annual_return": round(annual_ret * 100, 2),
        "max_drawdown": round(max_dd * 100, 2),
    }


# ---------------------------------------------------------------------------
# Main: run standalone to refresh data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download treasury signal data")
    parser.add_argument("--force", action="store_true", help="Force re-download (ignore cache)")
    args = parser.parse_args()

    df = load_data(force_refresh=args.force)

    print("\n" + "=" * 60)
    print("DATA AVAILABILITY SUMMARY")
    print("=" * 60)
    for col in df.columns:
        non_null = df[col].notna().sum()
        first_date = df[col].first_valid_index()
        print(f"  {col:25s}: {non_null:5d} rows  (from {first_date.date() if first_date else 'N/A'})")

    # Sanity check: IEF return in 2022 should be negative (rising rates year)
    ret_2022 = df.loc["2022", "ief_return"].sum() if "ief_return" in df.columns else None
    if ret_2022 is not None:
        status = "OK" if ret_2022 < -0.1 else "WARNING"
        print(f"\nIEF 2022 total return: {ret_2022*100:.1f}%  [{status}]")

    # B&H benchmark
    if "ief_return" in df.columns:
        bah = compute_buy_and_hold(df["ief_return"].dropna())
        print(f"\nBuy-and-Hold (IEF, {df.index[0].date()} to {df.index[-1].date()}):")
        print(f"  Sharpe:       {bah['sharpe']}")
        print(f"  Total Return: {bah['total_return']}%")
        print(f"  Annual Return:{bah['annual_return']}%")
        print(f"  Max Drawdown: {bah['max_drawdown']}%")
