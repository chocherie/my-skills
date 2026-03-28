# Spec: data_layer.py

## Purpose
Download, cache, and align all data series needed by the 25 signals.
Provide the IMMUTABLE evaluation functions used by the autoresearch loop.

## Immutable Functions (NEVER modify these)

```python
def evaluate_sharpe(signal: pd.Series, returns: pd.Series) -> float:
    """Ground-truth metric for autoresearch loop. Returns annualized Sharpe."""

def compute_buy_and_hold(returns: pd.Series) -> dict:
    """Ground-truth B&H benchmark. Returns metrics dict."""
```

## Data Series

| Column | Source | FRED Series | Frequency | Available From |
|:---|:---|:---|:---|:---|
| `ief_return` | yfinance IEF | — | Daily | 2002-07-30 |
| `ief_close` | yfinance IEF | — | Daily | 2002-07-30 |
| `dgs10` | FRED | DGS10 | Daily | 1962-01-02 |
| `dgs2` | FRED | DGS2 | Daily | 1976-06-01 |
| `dgs3` | FRED | DGS3 | Daily | 1962-01-02 |
| `dgs5` | FRED | DGS5 | Daily | 1962-01-02 |
| `dgs7` | FRED | DGS7 | Daily | 1969-07-01 |
| `dgs30` | FRED | DGS30 | Daily | 1977-02-15 |
| `dgs3m` | FRED | DTB3 | Daily | 1954-01-04 |
| `dgs6m` | FRED | DTB6 | Daily | 1958-12-09 |
| `dgs1` | FRED | DGS1 | Daily | 1962-01-02 |
| `dgs20` | FRED | DGS20 | Daily | 1993-10-01 |
| `t10y2y` | FRED | T10Y2Y | Daily | 1976-06-01 |
| `t10yie` | FRED | T10YIE | Daily | 2003-01-02 |
| `dfii10` | FRED | DFII10 | Daily | 2003-01-02 |
| `cpiaucsl` | FRED | CPIAUCSL | Monthly→Daily | 1947-01-01 |
| `fedfunds` | FRED | FEDFUNDS | Monthly→Daily | 1954-07-01 |
| `sofr` | FRED | SOFR | Daily | 2018-04-02 |
| `rrp` | FRED | RRPONTSYD | Daily | 2003-01-14 |
| `farbast` | FRED | FARBAST | Weekly→Daily | 1978-01-04 |
| `hy_spread` | FRED | BAMLH0A0HYM2 | Daily | 1996-12-31 |
| `vix` | yfinance ^VIX | — | Daily | 1990-01-02 |
| `dxy` | FRED | DTWEXBGS | Daily | 2006-01-02 |
| `oil` | yfinance CL=F | — | Daily | 1983-03-30 |
| `de10y` | FRED | IRLTLT01DEM156N | Monthly→Daily | 1960-01-01 |
| `jp10y` | FRED | IRLTLT01JPM156N | Monthly→Daily | 1960-01-01 |
| `acm_tp` | NY Fed CSV | — | Daily | 1961-06-14 |
| `cot_lev_net` | CFTC.gov | — | Weekly→Daily | 2006-06-13 |
| `ust_return_approx` | Computed | DGS10+SOFR | Daily | 1990-01-01 |
| `fomc_hawk_score` | Fed website | — | 8x/year→Daily | 1993-01-01 |

## Caching

- Cache directory: `~/.cache/treasury_signals/`
- Cache file: `master_df.parquet` (re-downloaded if >24h stale)
- Individual series cached as `{series_name}.parquet`

## Date Alignment

- Master index: all business days from `START_DATE = "2003-01-01"` to today
- Forward-fill all weekly/monthly series to daily
- Drop rows where `ief_return` is NaN (non-trading days)

## Extended Return Series

```python
# For signals requiring pre-2002 history
duration = 8.5  # approx modified duration of 10Y note
yield_chg = dgs10.diff() / 100
daily_carry = dgs10.shift(1) / 100 / 252
funding = sofr.fillna(fedfunds) / 100 / 252  # SOFR from 2018, fed funds before
ust_return_approx = -duration * yield_chg + daily_carry - funding
```
