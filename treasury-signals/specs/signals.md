# Spec: signals.py — All 25 Signals

## Output Contract

Every signal function MUST:
1. Accept `df: pd.DataFrame` (the master_df from data_layer.py)
2. Return `pd.Series` with DatetimeIndex aligned to df.index
3. Return values in {-1, 0, 1} (long, flat, short)
4. Return NaN for warmup periods (no signal during insufficient history)
5. Never access future data — use only lagged or same-day available data

## Helper Functions (top of signals.py)

```python
def zscore(series, window=252):
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

def signal_from_score(score, bull=0.3, bear=-0.3):
    return score.apply(lambda x: 1 if x > bull else (-1 if x < bear else 0))
```

---

## Category A: Trend / Momentum

### S01 — 3-Month Momentum
- **Logic**: Sign of IEF excess return over trailing 63 trading days
- **Data**: `ief_return` (daily)
- **Warmup**: 63 days
- **Params**: lookback=63
- **Expected Sharpe**: 0.3–0.5

### S02 — 6-Month Momentum
- **Logic**: Sign of IEF excess return over trailing 126 trading days
- **Data**: `ief_return`
- **Warmup**: 126 days
- **Params**: lookback=126
- **Expected Sharpe**: 0.3–0.5

### S03 — 12-1 Month Momentum (skip last month)
- **Logic**: Sign of 252-day return minus last 21-day return (standard "12-1" momentum)
- **Data**: `ief_return`
- **Warmup**: 252 days
- **Params**: lookback=252, skip=21
- **Expected Sharpe**: 0.3–0.6

### S04 — SMA Crossover (50d vs 200d)
- **Logic**: IEF price above 200d SMA → long; below → short
- **Data**: `ief_close` (IEF price level)
- **Warmup**: 200 days
- **Params**: fast=50, slow=200
- **Expected Sharpe**: 0.2–0.4

### S05 — RSI-14 on Yield
- **Logic**: DGS10 RSI(14) < 30 → yield oversold → bond bearish (short); > 70 → yield overbought → bond bullish (long)
- **Note**: RSI on yield, direction inverted (high yield RSI = bond price oversold = buy)
- **Data**: `dgs10`
- **Warmup**: 14 days
- **Params**: period=14, oversold=30, overbought=70
- **Expected Sharpe**: 0.1–0.3

---

## Category B: Macro / Fundamental

### S06 — Carry (Yield Curve Slope)
- **Logic**: T10Y2Y spread z-scored vs 252d; high spread → positive carry → long
- **Data**: `t10y2y`
- **Warmup**: 252 days
- **Params**: window=252, bull_thresh=0.5, bear_thresh=-0.5
- **Expected Sharpe**: 0.3–0.6

### S07 — Breakeven Inflation Momentum
- **Logic**: 3-month change in breakeven inflation (T10YIE), z-scored; rising breakeven → short bonds
- **Data**: `t10yie`
- **Warmup**: 252 days
- **Params**: momentum_window=63, z_window=252
- **Direction**: inverted (rising breakeven → short)
- **Expected Sharpe**: 0.2–0.4

### S08 — ACM Term Premium Deviation
- **Logic**: ACM term premium z-scored vs 252d MA; below median → cheap bonds → long
- **Data**: `acm_tp`
- **Warmup**: 252 days
- **Params**: window=252
- **Direction**: inverted (low TP = expensive bonds = short)
- **Expected Sharpe**: 0.3–0.6

### S09 — TIPS Real Yield
- **Logic**: DFII10 real yield level; deeply negative → bonds expensive but supported → flat/long; positive and rising → short
- **Data**: `dfii10`
- **Warmup**: 63 days
- **Params**: negative_thresh=-1.0 (below = bullish), rising_window=63
- **Expected Sharpe**: 0.2–0.4

### S10 — CPI Momentum Proxy
- **Logic**: 3-month annualized CPI change, z-scored; above threshold → inflation rising → short bonds
- **Data**: `cpiaucsl`
- **Warmup**: 252 days
- **Params**: window=63, z_window=252
- **Direction**: inverted (rising CPI → short)
- **Expected Sharpe**: 0.2–0.4

---

## Category C: Cross-Asset

### S11 — VIX Flight-to-Safety
- **Logic**: VIX z-score > 1.5σ → risk-off → long bonds; extreme VIX > 2σ = strong long signal
- **Data**: `vix`
- **Warmup**: 252 days
- **Params**: window=252, mild_thresh=1.5, strong_thresh=2.0
- **Expected Sharpe**: 0.2–0.4

### S12 — HY Credit Spread
- **Logic**: HY OAS spread z-score; widening → flight to safety → long bonds
- **Data**: `hy_spread`
- **Warmup**: 252 days
- **Params**: window=252, bull_thresh=0.5
- **Expected Sharpe**: 0.3–0.5

### S13 — DXY Momentum
- **Logic**: USD 10-day momentum z-scored; USD strengthening → capital inflows → mild bullish bonds
- **Data**: `dxy`
- **Warmup**: 252 days
- **Params**: momentum_window=10, z_window=252
- **Expected Sharpe**: 0.1–0.3

### S14 — Oil Momentum (Inflation Proxy)
- **Logic**: WTI 10-day momentum z-scored; rising oil → inflationary → short bonds
- **Data**: `oil`
- **Warmup**: 252 days
- **Params**: momentum_window=10, z_window=252
- **Direction**: inverted
- **Expected Sharpe**: 0.1–0.3

---

## Category D: Positioning / Sentiment

### S15 — COT Leveraged Fund Percentile
- **Logic**: CFTC TFF Leveraged Fund net position, ranked as percentile over trailing 3 years;
  <20th pctl = crowded short → contrarian long; >80th pctl = crowded long → contrarian short
- **Data**: `cot_lev_net` (weekly, forward-filled to daily)
- **Warmup**: 756 days (3 years)
- **Params**: window=756, short_thresh=0.2, long_thresh=0.8
- **Note**: 3-day publication lag — use data available as of Friday release
- **Expected Sharpe**: 0.3–0.5

### S16 — Primary Dealer Net Positioning
- **Logic**: Week-over-week change in NY Fed primary dealer 6–11Y net position;
  accumulation (positive change) → bullish; distribution → bearish
- **Data**: `pd_net_6_11y` (weekly NY Fed FR 2004, forward-filled)
- **Warmup**: 252 days
- **Params**: window=252
- **Expected Sharpe**: 0.2–0.4

### S17 — Fed RRP Usage Filter
- **Logic**: 4-week change in RRP usage (RRPONTSYD); large decline → liquidity tightening → bearish filter;
  used primarily as a position-size scaler, not standalone
- **Data**: `rrp`
- **Warmup**: 63 days
- **Params**: window=20
- **Expected Sharpe**: 0.1–0.2 (best as filter/overlay)

---

## Category E: Statistical

### S18 — PCA Slope Factor
- **Logic**: PCA on 6-maturity yield changes (2Y,3Y,5Y,7Y,10Y,30Y); PC2 = slope factor;
  steep curve (positive PC2 z-score) → positive carry → long
- **Data**: `dgs2`, `dgs3`, `dgs5`, `dgs7`, `dgs10`, `dgs30`
- **Warmup**: 252 days
- **Params**: n_components=3, z_window=252
- **Expected Sharpe**: 0.3–0.5

### S19 — Kalman Dynamic Carry Spread
- **Logic**: Kalman filter applied to 10Y-2Y spread to extract smooth trend;
  signal = z-score of (actual spread − Kalman-filtered spread); mean-reversion
- **Data**: `t10y2y`
- **Warmup**: 126 days
- **Params**: transition_covariance=0.01, observation_covariance=1.0
- **Expected Sharpe**: 0.3–0.5

### S20 — HMM Regime Signal
- **Logic**: 2-state Gaussian HMM on [dgs10 daily change, vix change];
  P(bull_bond_regime) → long when > 0.65, short when < 0.35
- **Data**: `dgs10`, `vix`
- **Warmup**: 252 days
- **Params**: n_components=2, n_iter=100
- **Expected Sharpe**: 0.4–0.7

### S21 — NSS Curve Deviation
- **Logic**: Fit Nelson-Siegel-Svensson to daily yield curve; residual = actual 10Y − fitted 10Y;
  positive residual (10Y cheap vs model) → long; negative → short
- **Data**: `dgs3m`, `dgs6m`, `dgs1`, `dgs2`, `dgs5`, `dgs10`, `dgs20`, `dgs30`
- **Warmup**: 63 days
- **Params**: z_window=252, bull_thresh=0.5
- **Expected Sharpe**: 0.2–0.4

### S22 — VECM Error Correction (Cointegration)
- **Logic**: Engle-Granger cointegration between DGS10 and CPIAUCSL;
  error correction term (EC) = DGS10 − (α + β×CPI); EC < −1.5σ → yield too low relative to CPI → short;
  EC > +1.5σ → yield too high → long
- **Data**: `dgs10`, `cpiaucsl`
- **Warmup**: 504 days
- **Params**: window=504, ec_thresh=1.5
- **Expected Sharpe**: 0.2–0.4

---

## Category F: Global Macro / Alt Data

### S23 — Foreign Custody Holdings
- **Logic**: FARBAST (foreign official UST custody at NY Fed), 4-week rolling change;
  large decline (>$30B over 4 weeks) → foreign CB selling → bearish bonds
- **Data**: `farbast` (weekly FRED, forward-filled)
- **Warmup**: 63 days
- **Params**: roll_window=20, thresh_z=1.0
- **Expected Sharpe**: 0.2–0.4

### S24 — Global Bond Factor Residual
- **Logic**: PCA on [dgs10, de10y, jp10y] daily changes; PC1 = global factor;
  US residual from global factor → when positive (US yields elevated vs global) → buy
- **Data**: `dgs10`, `de10y`, `jp10y`
- **Warmup**: 252 days
- **Params**: z_window=252, bull_thresh=0.5
- **Expected Sharpe**: 0.2–0.4

### S25 — FOMC Minutes Sentiment (VADER Lexical)
- **Logic**: VADER sentiment on Fed FOMC minutes text; hawk-score = (negative_sentiment / total);
  8 releases/year, held constant between releases;
  high hawkish score → short bonds; dovish → long
- **Data**: `fomc_hawk_score` (pre-computed from Fed website scrape)
- **Warmup**: 2 FOMC cycles (≈6 months)
- **Params**: hawk_threshold=0.6, dove_threshold=0.4
- **Note**: Only 8 updates/year — low frequency signal, use as multi-month overlay
- **Expected Sharpe**: 0.3–0.5
