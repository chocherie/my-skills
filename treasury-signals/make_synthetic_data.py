"""
make_synthetic_data.py
======================
Generates realistic synthetic Treasury market data and saves it directly to
the data_layer cache (~/.cache/treasury_signals/master_df.parquet).

Use this when the environment has no internet access. The synthetic data
preserves the statistical properties of real Treasury markets:
  - Yield curve mean-reversion, realistic volatility
  - IEF return correlated with DGS10 changes (duration ~8.5yr)
  - VIX mean-reverting with vol clusters
  - CPI trending with occasional regime breaks
  - All 27 columns expected by signals.py

Run once before backtest_runner.py:
    python make_synthetic_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# ─── date range ───────────────────────────────────────────────────────────────
START = "2003-01-01"
END   = "2026-03-27"
idx   = pd.bdate_range(start=START, end=END, freq="B")
N     = len(idx)
T     = np.arange(N) / 252          # time in years

# ─── 10Y yield (Ornstein-Uhlenbeck mean-reverting process) ───────────────────
# Parameters tuned to match ~2003-2026 history: high in 2007, low in 2020,
# rising in 2022-23
theta = 0.10          # mean reversion speed
sigma_y = 0.0050      # daily yield vol ≈ 50bp annualised
mu_path = (
    5.0 - 1.5 * np.clip(T / 5, 0, 1)          # secular decline 2003-2007
    + 2.5 * np.clip((T - 5) / 3, 0, 1)         # rise 2007-2010
    - 2.0 * np.clip((T - 8) / 10, 0, 1)        # long decline 2010-2020
    - 0.5 * np.exp(-((T - 17) ** 2) / 0.5)     # COVID trough 2020
    + 2.5 * np.clip((T - 18) / 2, 0, 1)        # 2022 rate shock
    - 0.5 * np.clip((T - 21) / 2, 0, 1)        # 2024 stabilisation
)
dgs10 = np.zeros(N)
dgs10[0] = 4.0
for i in range(1, N):
    dgs10[i] = dgs10[i - 1] + theta * (mu_path[i] - dgs10[i - 1]) + sigma_y * np.random.randn()
dgs10 = np.clip(dgs10, 0.01, 18.0)

# ─── Yield curve (spread each maturity off 10Y) ──────────────────────────────
spreads = {
    "dgs3m": -1.5 + 0.5 * np.random.randn(N).cumsum() / np.sqrt(N),
    "dgs6m": -1.2 + 0.4 * np.random.randn(N).cumsum() / np.sqrt(N),
    "dgs1":  -0.8 + 0.3 * np.random.randn(N).cumsum() / np.sqrt(N),
    "dgs2":  -0.5 + 0.2 * np.random.randn(N).cumsum() / np.sqrt(N),
    "dgs3":  -0.3 + 0.15 * np.random.randn(N).cumsum() / np.sqrt(N),
    "dgs5":   0.0 + 0.10 * np.random.randn(N).cumsum() / np.sqrt(N),
    "dgs7":   0.2 + 0.08 * np.random.randn(N).cumsum() / np.sqrt(N),
    "dgs20":  0.4 + 0.10 * np.random.randn(N).cumsum() / np.sqrt(N),
    "dgs30":  0.6 + 0.12 * np.random.randn(N).cumsum() / np.sqrt(N),
}
yields = {k: np.clip(dgs10 + v, 0.01, 20.0) for k, v in spreads.items()}
yields["dgs10"] = dgs10
t10y2y = dgs10 - yields["dgs2"]      # 10Y-2Y slope
yields["t10y2y"] = t10y2y

# ─── TIPS / breakeven ────────────────────────────────────────────────────────
dfii10 = np.clip(dgs10 - 2.0 + 0.5 * np.random.randn(N).cumsum() / np.sqrt(N), -1.0, 5.0)
t10yie = dgs10 - dfii10               # breakeven inflation
yields["dfii10"] = dfii10
yields["t10yie"] = t10yie

# ─── IEF total return (7-10Y ETF, duration ≈ 8.5yr) ─────────────────────────
yield_chg = np.diff(dgs10 / 100, prepend=dgs10[0] / 100)
carry_daily = np.roll(dgs10, 1) / 100 / 252
fedfunds_level = np.clip(
    mu_path * 0.6 - 0.5 + 0.3 * np.random.randn(N).cumsum() / np.sqrt(N), 0, 6
)
funding_daily = fedfunds_level / 100 / 252
ief_return = -8.5 * yield_chg + carry_daily - funding_daily
# Add small idiosyncratic noise (ETF tracking error)
ief_return += 0.0002 * np.random.randn(N)
ief_close = 100.0 * np.cumprod(1 + ief_return)

# ─── VIX (mean-reverting, log-normal) ────────────────────────────────────────
log_vix = np.zeros(N)
log_vix[0] = np.log(20)
for i in range(1, N):
    log_vix[i] = log_vix[i - 1] + 0.05 * (np.log(18) - log_vix[i - 1]) + 0.04 * np.random.randn()
vix = np.exp(log_vix)
# Add crisis spikes
crisis_dates = [int(N * 0.15), int(N * 0.40), int(N * 0.72)]  # GFC, COVID, SVB
for d in crisis_dates:
    width = 40
    spike = 30 * np.exp(-((np.arange(N) - d) ** 2) / (2 * width ** 2))
    vix += spike

# ─── Oil (WTI, geometric Brownian motion with trend) ─────────────────────────
log_oil = np.log(35)
oil = np.zeros(N)
for i in range(N):
    log_oil += 0.0003 + 0.015 * np.random.randn()
    oil[i] = np.exp(log_oil)
oil = np.clip(oil, 10, 150)

# ─── CPI (monthly, forward-filled, trending up then moderated) ───────────────
monthly_idx = pd.date_range(start=START, end=END, freq="MS")
cpi = np.zeros(len(monthly_idx))
cpi[0] = 181.7
inflation_path = (
    2.5 + 1.5 * np.clip((np.arange(len(monthly_idx)) - 200) / 30, 0, 1)
    - 2.5 * np.clip((np.arange(len(monthly_idx)) - 230) / 20, 0, 1)
)
for i in range(1, len(cpi)):
    cpi[i] = cpi[i - 1] * (1 + inflation_path[i] / 100 / 12 + 0.001 * np.random.randn())
cpi_series = pd.Series(cpi, index=monthly_idx)
cpi_daily = cpi_series.reindex(idx).ffill()

# ─── Fed Funds rate ──────────────────────────────────────────────────────────
ff = pd.Series(np.clip(fedfunds_level, 0, 6), index=idx)

# ─── SOFR (≈ fed funds with basis) ──────────────────────────────────────────
sofr = ff + 0.05 * np.random.randn(N) * 0.1
sofr = np.clip(sofr, 0, 6)

# ─── RRP (reverse repo, zero in low-rate, rises 2021-23) ─────────────────────
rrp = np.where(T < 17, 0.0, np.clip((T - 17) * 500 - 300, 0, 2500)) + 10 * np.random.randn(N)
rrp = np.clip(rrp, 0, 3000)

# ─── Foreign custody holdings (NY Fed FARBAST) ──────────────────────────────
farbast = 2000 + 800 * np.sin(T * 0.5) + 100 * np.random.randn(N).cumsum() / np.sqrt(N)
farbast = np.clip(farbast, 1000, 4000)

# ─── HY spread ───────────────────────────────────────────────────────────────
hy_spread = 4.0 + 2.0 * (vix / 20 - 1) + 0.5 * np.random.randn(N).cumsum() / np.sqrt(N)
hy_spread = np.clip(hy_spread, 1.5, 20.0)

# ─── DXY (broad dollar index) ────────────────────────────────────────────────
dxy = 100 + 10 * np.sin(T * 0.8) + 5 * np.random.randn(N).cumsum() / np.sqrt(N)
dxy = np.clip(dxy, 70, 130)

# ─── German / Japan 10Y (global factor) ─────────────────────────────────────
de10y = np.clip(dgs10 * 0.7 - 1.0 + 0.3 * np.random.randn(N).cumsum() / np.sqrt(N), -1.0, 5.0)
jp10y = np.clip(dgs10 * 0.2 - 0.5 + 0.1 * np.random.randn(N).cumsum() / np.sqrt(N), -0.5, 2.0)

# ─── ACM term premium ────────────────────────────────────────────────────────
acm_tp = 1.5 - 0.5 * (dgs10 - 4) + 0.2 * np.random.randn(N).cumsum() / np.sqrt(N)
acm_tp = np.clip(acm_tp, -1.5, 4.0)

# ─── COT leveraged fund net (weekly, mimicking CFTC) ─────────────────────────
weekly_idx = pd.date_range(start=START, end=END, freq="W-TUE")
cot_net = np.zeros(len(weekly_idx))
cot_net[0] = -50000
for i in range(1, len(cot_net)):
    cot_net[i] = cot_net[i - 1] * 0.95 + 5000 * np.random.randn()
cot_series = pd.Series(np.clip(cot_net, -300000, 300000), index=weekly_idx)
cot_daily = cot_series.reindex(idx).ffill().shift(3)  # 3-day publication lag

# ─── FOMC hawk score (8× per year) ──────────────────────────────────────────
fomc_dates = pd.date_range(start=START, end=END, freq="45D")  # ~8/year approx
fomc_hawk = pd.Series(
    np.clip(0.0 + 0.3 * np.random.randn(len(fomc_dates)), -1, 1),
    index=fomc_dates,
)
fomc_daily = fomc_hawk.reindex(idx).ffill()

# ─── Primary dealer net (6-11Y) ──────────────────────────────────────────────
pd_weekly = pd.date_range(start=START, end=END, freq="W-WED")
pd_net = np.zeros(len(pd_weekly))
pd_net[0] = 50000
for i in range(1, len(pd_net)):
    pd_net[i] = pd_net[i - 1] * 0.90 + 10000 * np.random.randn()
pd_series = pd.Series(np.clip(pd_net, -500000, 500000), index=pd_weekly)
pd_daily = pd_series.reindex(idx).ffill()

# ─── Duration-approx return (ust_return_approx) ──────────────────────────────
funding_daily_series = pd.Series(fedfunds_level / 100 / 252, index=idx)
ust_return_approx = (
    -8.5 * pd.Series(yield_chg, index=idx)
    + pd.Series(carry_daily, index=idx)
    - funding_daily_series
)

# ─── Assemble master DataFrame ───────────────────────────────────────────────
master = pd.DataFrame(index=idx)
master["ief_close"]         = ief_close
master["ief_return"]        = ief_return
master["vix"]               = vix
master["oil"]               = oil
for col, arr in yields.items():
    master[col] = arr
master["cpiaucsl"]          = cpi_daily.values
master["fedfunds"]          = ff.values
master["sofr"]              = sofr
master["rrp"]               = rrp
master["farbast"]           = farbast
master["hy_spread"]         = hy_spread
master["dxy"]               = dxy
master["de10y"]             = de10y
master["jp10y"]             = jp10y
master["acm_tp"]            = acm_tp
master["cot_lev_net"]       = cot_daily.values
master["fomc_hawk_score"]   = fomc_daily.values
master["pd_net_6_11y"]      = pd_daily.values
master["ust_return_approx"] = ust_return_approx.values

# ─── Save to cache ─────────────────────────────────────────────────────────
CACHE_DIR = Path.home() / ".cache" / "treasury_signals"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
out_path = CACHE_DIR / "master_df.parquet"
master.to_parquet(out_path)

print(f"Synthetic data written to: {out_path}")
print(f"  Shape: {master.shape[0]} rows × {master.shape[1]} cols")
print(f"  Date range: {master.index[0].date()} → {master.index[-1].date()}")
print(f"  IEF 2022 est. return: {master.loc['2022', 'ief_return'].add(1).prod() - 1:.1%}")
print(f"  DGS10 range: {master['dgs10'].min():.2f}% – {master['dgs10'].max():.2f}%")
print(f"  Columns ({master.shape[1]}): {list(master.columns)}")
print("\nReady — run: python backtest_runner.py")
