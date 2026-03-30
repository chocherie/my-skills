"""
signals.py — 10Y UST Signal Dashboard
=======================================
THE EDITABLE FILE for the autoresearch loop.

Contains all 25 signal implementations. Each function returns a pd.Series
of daily positions {-1, 0, 1} aligned to the master DataFrame index.

Rules (enforced by .cursor/rules/):
- Read specs/signals.md before editing any signal
- Never access future data
- Always return values in {-1, 0, 1} after discretization
- Return NaN for warmup periods
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# PARAMETER BLOCK — agent edits these to tune signals
# ---------------------------------------------------------------------------

# Category A: Momentum
P_S01_LOOKBACK = 63        # 3M momentum lookback (days)
P_S02_LOOKBACK = 126       # 6M momentum lookback
P_S03_LOOKBACK = 252       # 12M momentum lookback
P_S03_SKIP = 21            # Skip last month for 12-1 momentum
P_S04_FAST = 50            # SMA fast window
P_S04_SLOW = 200           # SMA slow window
P_S05_PERIOD = 14          # RSI period
P_S05_OVERSOLD = 30        # RSI oversold threshold (yield)
P_S05_OVERBOUGHT = 70      # RSI overbought threshold (yield)

# Category B: Macro
P_S06_Z_WINDOW = 252       # Carry z-score window
P_S07_MOM_WINDOW = 63      # Breakeven momentum window
P_S07_Z_WINDOW = 252
P_S08_Z_WINDOW = 252       # ACM term premium z-score window
P_S09_NEG_THRESH = -1.0    # Real yield negative threshold (%)
P_S10_MOM_WINDOW = 63      # CPI momentum window
P_S10_Z_WINDOW = 252

# Category C: Cross-Asset
P_S11_Z_WINDOW = 252       # VIX z-score window
P_S11_MILD_THRESH = 1.5
P_S12_Z_WINDOW = 252       # HY spread z-score window
P_S13_MOM_WINDOW = 10      # DXY momentum window
P_S13_Z_WINDOW = 252
P_S14_MOM_WINDOW = 10      # Oil momentum window
P_S14_Z_WINDOW = 252

# Category D: Positioning
P_S15_WINDOW = 756         # COT percentile window (3 years)
P_S15_SHORT_THRESH = 0.20  # Crowded short → contrarian long
P_S15_LONG_THRESH = 0.80   # Crowded long → contrarian short
P_S16_Z_WINDOW = 252       # Primary dealer z-score window
P_S17_ROLL_WINDOW = 20     # RRP rolling window

# Category E: Statistical
P_S18_Z_WINDOW = 252       # PCA slope z-score window
P_S19_OBS_NOISE = 1.0      # Kalman observation noise
P_S19_TRANS_NOISE = 0.01   # Kalman transition noise
P_S20_N_STATES = 2         # HMM number of states
P_S20_N_ITER = 100         # HMM EM iterations
P_S21_Z_WINDOW = 252       # NSS deviation z-score window
P_S22_WINDOW = 504         # VECM cointegration window
P_S22_EC_THRESH = 1.5      # VECM error correction threshold (σ)

# Category F: Global / NLP
P_S23_ROLL_WINDOW = 20     # Foreign custody rolling window (4 weeks)
P_S23_Z_WINDOW = 252
P_S24_Z_WINDOW = 252       # Global bond factor z-score window
P_S25_HAWK_THRESH = 0.60   # FOMC hawk threshold → short bonds
P_S25_DOVE_THRESH = 0.40   # FOMC dove threshold → long bonds


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """Rolling z-score."""
    mean = series.rolling(window, min_periods=window // 2).mean()
    std = series.rolling(window, min_periods=window // 2).std()
    return (series - mean) / std.replace(0, np.nan)


def signal_from_score(score: pd.Series, bull: float = 0.3, bear: float = -0.3) -> pd.Series:
    """Convert continuous score to {-1, 0, 1} signal."""
    return score.apply(lambda x: 1 if (not np.isnan(x) and x > bull)
                       else (-1 if (not np.isnan(x) and x < bear) else 0))


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# CATEGORY A: Trend / Momentum
# ---------------------------------------------------------------------------

def compute_S01(df: pd.DataFrame, lookback: int = None) -> pd.Series:
    """S01 — 3-Month Momentum. Sign of IEF trailing 63d excess return."""
    _lb = lookback if lookback is not None else P_S01_LOOKBACK
    if "ief_return" not in df.columns:
        return pd.Series(0, index=df.index, name="S01_3M_Momentum")
    cum_ret = (1 + df["ief_return"].fillna(0)).rolling(_lb).apply(
        lambda x: x.prod() - 1, raw=True
    )
    sig = cum_ret.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    sig[:_lb] = np.nan
    sig.name = "S01_3M_Momentum"
    return sig


def compute_S02(df: pd.DataFrame, lookback: int = None) -> pd.Series:
    """S02 — 6-Month Momentum. Sign of IEF trailing 126d return."""
    _lb = lookback if lookback is not None else P_S02_LOOKBACK
    if "ief_return" not in df.columns:
        return pd.Series(0, index=df.index, name="S02_6M_Momentum")
    cum_ret = (1 + df["ief_return"].fillna(0)).rolling(_lb).apply(
        lambda x: x.prod() - 1, raw=True
    )
    sig = cum_ret.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    sig[:_lb] = np.nan
    sig.name = "S02_6M_Momentum"
    return sig


def compute_S03(df: pd.DataFrame, lookback: int = None) -> pd.Series:
    """S03 — 12-1 Month Momentum (skip last month)."""
    _lb = lookback if lookback is not None else P_S03_LOOKBACK
    if "ief_return" not in df.columns:
        return pd.Series(0, index=df.index, name="S03_12_1_Momentum")
    ret = df["ief_return"].fillna(0)
    cum_12 = (1 + ret).rolling(_lb).apply(lambda x: x.prod() - 1, raw=True)
    cum_skip = (1 + ret).rolling(P_S03_SKIP).apply(lambda x: x.prod() - 1, raw=True)
    mom = cum_12 - cum_skip
    sig = mom.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    sig[:_lb] = np.nan
    sig.name = "S03_12_1_Momentum"
    return sig


def compute_S04(df: pd.DataFrame, slow: int = None) -> pd.Series:
    """S04 — SMA Crossover. IEF price above SMA → long, below → flat."""
    _slow = slow if slow is not None else P_S04_SLOW
    if "ief_close" not in df.columns:
        return pd.Series(0, index=df.index, name="S04_SMA_Cross")
    price = df["ief_close"]
    sma_slow = price.rolling(_slow).mean()
    sig = pd.Series(np.where(price > sma_slow, 1, 0), index=df.index, dtype=float)
    sig[:_slow] = np.nan
    sig.name = "S04_SMA_Cross"
    return sig


def compute_S05(df: pd.DataFrame, period: int = None) -> pd.Series:
    """S05 — RSI on DGS10 yield. High RSI on yield = overbought yield = buy bonds."""
    _period = period if period is not None else P_S05_PERIOD
    if "dgs10" not in df.columns:
        return pd.Series(0, index=df.index, name="S05_RSI14_Yield")
    rsi = compute_rsi(df["dgs10"].ffill(), _period)
    sig = pd.Series(
        np.where(rsi > P_S05_OVERBOUGHT, 1,
                 np.where(rsi < P_S05_OVERSOLD, -1, 0)),
        index=df.index, dtype=float
    )
    sig[:_period * 3] = np.nan
    sig.name = "S05_RSI14_Yield"
    return sig


# ---------------------------------------------------------------------------
# CATEGORY B: Macro / Fundamental
# ---------------------------------------------------------------------------

def compute_S06(df: pd.DataFrame, z_window: int = None) -> pd.Series:
    """S06 — Carry (Yield Curve Slope). T10Y2Y z-score."""
    _z = z_window if z_window is not None else P_S06_Z_WINDOW
    if "t10y2y" not in df.columns:
        return pd.Series(0, index=df.index, name="S06_Carry_Slope")
    score = zscore(df["t10y2y"].ffill(), _z)
    sig = signal_from_score(score, bull=0.3, bear=-0.3)
    sig[:_z] = np.nan
    sig.name = "S06_Carry_Slope"
    return sig


def compute_S07(df: pd.DataFrame, mom_window: int = None) -> pd.Series:
    """S07 — Breakeven Inflation Momentum. Rising → short bonds."""
    _mw = mom_window if mom_window is not None else P_S07_MOM_WINDOW
    if "t10yie" not in df.columns:
        return pd.Series(0, index=df.index, name="S07_Breakeven_Mom")
    be = df["t10yie"].ffill()
    mom = be.diff(_mw)
    score = -zscore(mom, P_S07_Z_WINDOW)
    sig = signal_from_score(score, bull=0.3, bear=-0.3)
    sig[:P_S07_Z_WINDOW + _mw] = np.nan
    sig.name = "S07_Breakeven_Mom"
    return sig


def compute_S08(df: pd.DataFrame, z_window: int = None) -> pd.Series:
    """S08 — ACM Term Premium Deviation. Below median → cheap bonds → long."""
    _z = z_window if z_window is not None else P_S08_Z_WINDOW
    if "acm_tp" not in df.columns:
        return pd.Series(0, index=df.index, name="S08_ACM_TermPrem")
    tp = df["acm_tp"].ffill()
    score = -zscore(tp, _z)
    sig = signal_from_score(score, bull=0.3, bear=-0.3)
    sig[:_z] = np.nan
    sig.name = "S08_ACM_TermPrem"
    return sig


def compute_S09(df: pd.DataFrame, z_window: int = None) -> pd.Series:
    """S09 — TIPS Real Yield. Deeply negative real yields → long bonds (long/flat)."""
    _z = z_window if z_window is not None else 252
    if "dfii10" not in df.columns:
        return pd.Series(0, index=df.index, name="S09_Real_Yield")
    ry = df["dfii10"].ffill()
    score = -zscore(ry, _z)
    sig = pd.Series(np.where(score > 0.5, 1, 0), index=df.index, dtype=float)
    sig[:_z + 63] = np.nan
    sig.name = "S09_Real_Yield"
    return sig


def compute_S10(df: pd.DataFrame, z_window: int = None) -> pd.Series:
    """S10 — CPI Momentum Proxy. Rising CPI → short bonds."""
    _z = z_window if z_window is not None else P_S10_Z_WINDOW
    if "cpiaucsl" not in df.columns:
        return pd.Series(0, index=df.index, name="S10_CPI_Mom")
    cpi = df["cpiaucsl"].ffill()
    cpi_mom = (cpi / cpi.shift(63)) ** 4 - 1
    score = -zscore(cpi_mom, _z)
    sig = signal_from_score(score, bull=0.3, bear=-0.3)
    sig[:_z + 63] = np.nan
    sig.name = "S10_CPI_Mom"
    return sig


# ---------------------------------------------------------------------------
# CATEGORY C: Cross-Asset
# ---------------------------------------------------------------------------

def compute_S11(df: pd.DataFrame, z_window: int = None) -> pd.Series:
    """S11 — VIX Flight-to-Safety. High VIX → risk-off → long bonds, flat otherwise.
    Long/flat only: we want to capture flight-to-quality without shorting at low-vol."""
    _z = z_window if z_window is not None else P_S11_Z_WINDOW
    if "vix" not in df.columns:
        return pd.Series(0, index=df.index, name="S11_VIX_Safety")
    vix = df["vix"].ffill()
    score = zscore(vix, _z)
    # Long only when VIX is elevated (>1σ), flat otherwise
    sig = pd.Series(np.where(score > 1.0, 1, 0), index=df.index, dtype=float)
    sig[:_z] = np.nan
    sig.name = "S11_VIX_Safety"
    return sig


def compute_S12(df: pd.DataFrame, z_window: int = None) -> pd.Series:
    """S12 — HY Credit Spread. Widening → flight to safety → long bonds."""
    _z = z_window if z_window is not None else P_S12_Z_WINDOW
    if "hy_spread" not in df.columns:
        return pd.Series(0, index=df.index, name="S12_HY_Spread")
    spread = df["hy_spread"].ffill()
    score = zscore(spread.diff(10), _z)  # spread widening = bullish bonds
    sig = signal_from_score(score, bull=0.5, bear=-0.5)
    sig[:_z + 10] = np.nan
    sig.name = "S12_HY_Spread"
    return sig


def compute_S13(df: pd.DataFrame, mom_window: int = None) -> pd.Series:
    """S13 — DXY Momentum. USD strength → mild bullish for bonds."""
    _mw = mom_window if mom_window is not None else P_S13_MOM_WINDOW
    if "dxy" not in df.columns:
        return pd.Series(0, index=df.index, name="S13_DXY_Mom")
    dxy = df["dxy"].ffill()
    mom = dxy.pct_change(_mw)
    score = zscore(mom, P_S13_Z_WINDOW)
    sig = signal_from_score(score, bull=0.5, bear=-0.5)
    sig[:P_S13_Z_WINDOW + _mw] = np.nan
    sig.name = "S13_DXY_Mom"
    return sig


def compute_S14(df: pd.DataFrame, mom_window: int = None) -> pd.Series:
    """S14 — Oil Momentum. Rising oil → inflation → short bonds."""
    _mw = mom_window if mom_window is not None else P_S14_MOM_WINDOW
    if "oil" not in df.columns:
        return pd.Series(0, index=df.index, name="S14_Oil_Mom")
    oil = df["oil"].ffill()
    mom = oil.pct_change(_mw)
    score = -zscore(mom, P_S14_Z_WINDOW)  # inverted: rising oil → short
    sig = signal_from_score(score, bull=0.5, bear=-0.5)
    sig[:P_S14_Z_WINDOW + _mw] = np.nan
    sig.name = "S14_Oil_Mom"
    return sig


# ---------------------------------------------------------------------------
# CATEGORY D: Positioning / Sentiment
# ---------------------------------------------------------------------------

def compute_S15(df: pd.DataFrame, window: int = None) -> pd.Series:
    """S15 — COT Leveraged Fund Percentile. Crowded short → contrarian long."""
    _w = window if window is not None else P_S15_WINDOW
    if "cot_lev_net" not in df.columns:
        return pd.Series(0, index=df.index, name="S15_COT_Lev")
    cot = df["cot_lev_net"].ffill()
    # Rolling percentile rank over trailing window
    rank = cot.rolling(_w, min_periods=_w // 2).apply(
        lambda x: (x[-1] > x[:-1]).sum() / max(len(x) - 1, 1), raw=True
    )
    sig = pd.Series(
        np.where(rank < P_S15_SHORT_THRESH, 1,      # crowded short → contrarian long
                 np.where(rank > P_S15_LONG_THRESH, -1, 0)),  # crowded long → contrarian short
        index=df.index, dtype=float
    )
    sig[:_w] = np.nan
    sig.name = "S15_COT_Lev"
    return sig


def compute_S16(df: pd.DataFrame, z_window: int = None) -> pd.Series:
    """S16 — Primary Dealer Net Positioning. Accumulation → bullish."""
    _z = z_window if z_window is not None else P_S16_Z_WINDOW
    if "pd_net_6_11y" not in df.columns:
        return pd.Series(0, index=df.index, name="S16_PrimDealer")
    pd_pos = df["pd_net_6_11y"].ffill()
    # Week-over-week change, z-scored
    wow_change = pd_pos.diff(5)
    score = zscore(wow_change, _z)
    sig = signal_from_score(score, bull=0.5, bear=-0.5)
    sig[:_z + 5] = np.nan
    sig.name = "S16_PrimDealer"
    return sig


def compute_S17(df: pd.DataFrame, roll_window: int = None) -> pd.Series:
    """S17 — Fed RRP Usage. Large decline → liquidity tightening → bearish filter."""
    _rw = roll_window if roll_window is not None else P_S17_ROLL_WINDOW
    if "rrp" not in df.columns:
        return pd.Series(0, index=df.index, name="S17_RRP_Filter")
    rrp = df["rrp"].ffill().fillna(0)
    # 4-week rolling change, z-scored
    change = rrp.diff(_rw)
    score = zscore(change, 252)
    sig = signal_from_score(score, bull=0.5, bear=-0.5)
    sig[:252 + _rw] = np.nan
    sig.name = "S17_RRP_Filter"
    return sig


# ---------------------------------------------------------------------------
# CATEGORY E: Statistical
# ---------------------------------------------------------------------------

def compute_S18(df: pd.DataFrame, z_window: int = None) -> pd.Series:
    """S18 — PCA Slope Factor. PC2 of yield curve changes → steep curve → long."""
    _z = z_window if z_window is not None else P_S18_Z_WINDOW
    yield_cols = ["dgs2", "dgs3", "dgs5", "dgs7", "dgs10", "dgs30"]
    available = [c for c in yield_cols if c in df.columns]
    if len(available) < 4:
        return pd.Series(0, index=df.index, name="S18_PCA_Slope")

    yields = df[available].ffill()
    changes = yields.diff()

    # Rolling PCA: fit on trailing window, extract PC2 (slope factor)
    pca_scores = pd.Series(index=df.index, dtype=float, name="S18_PCA_Slope")
    for i in range(_z, len(df)):
        window = changes.iloc[i - _z:i].dropna()
        if len(window) < 100:
            continue
        try:
            pca = PCA(n_components=min(3, len(available)))
            factors = pca.fit_transform(window)
            # PC2 = slope factor; higher = steeper curve = positive carry
            pca_scores.iloc[i] = factors[-1, 1] if factors.shape[1] > 1 else 0.0
        except Exception:
            continue

    score = zscore(pca_scores, _z)
    sig = signal_from_score(score, bull=0.3, bear=-0.3)
    sig[:_z * 2] = np.nan
    sig.name = "S18_PCA_Slope"
    return sig


def compute_S19(df: pd.DataFrame, obs_noise: float = None) -> pd.Series:
    """S19 — Kalman Dynamic Carry Spread. Mean-reversion of spread vs Kalman trend."""
    _on = obs_noise if obs_noise is not None else P_S19_OBS_NOISE
    if "t10y2y" not in df.columns:
        return pd.Series(0, index=df.index, name="S19_Kalman_Spread")
    try:
        from pykalman import KalmanFilter
        spread = df["t10y2y"].ffill().dropna()

        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=spread.iloc[0],
            initial_state_covariance=1,
            observation_covariance=_on,
            transition_covariance=P_S19_TRANS_NOISE,
        )
        state_means, _ = kf.filter(spread.values)
        filtered = pd.Series(state_means.flatten(), index=spread.index)

        residual = (spread - filtered).reindex(df.index)
        # Kalman filter tracks closely → residuals small → use tighter z-score window
        score = zscore(residual.ffill(), 126)
        sig = signal_from_score(score, bull=0.3, bear=-0.3)
        sig[:252 + 126] = np.nan
        sig.name = "S19_Kalman_Spread"
        return sig
    except ImportError:
        print("WARNING: pykalman not installed, S19 disabled")
        return pd.Series(0, index=df.index, name="S19_Kalman_Spread")
    except Exception as e:
        print(f"WARNING: S19 Kalman failed: {e}")
        return pd.Series(0, index=df.index, name="S19_Kalman_Spread")


def compute_S20(df: pd.DataFrame) -> pd.Series:
    """S20 — HMM Regime Signal. P(bull_bond_regime) from 2-state HMM."""
    if "dgs10" not in df.columns:
        return pd.Series(0, index=df.index, name="S20_HMM_Regime")
    try:
        from hmmlearn.hmm import GaussianHMM

        # Features: daily yield change + VIX change
        yield_chg = df["dgs10"].ffill().diff().fillna(0)
        vix_chg = df["vix"].ffill().diff().fillna(0) if "vix" in df.columns else pd.Series(0, index=df.index)

        X = np.column_stack([yield_chg.values, vix_chg.values])
        X = np.nan_to_num(X, nan=0.0)

        scores = pd.Series(index=df.index, dtype=float)
        warmup = P_S18_Z_WINDOW  # use 252 warmup

        for i in range(warmup, len(df)):
            X_window = X[max(0, i - warmup):i]
            try:
                model = GaussianHMM(
                    n_components=P_S20_N_STATES,
                    covariance_type="diag",
                    n_iter=P_S20_N_ITER,
                    random_state=42,
                )
                model.fit(X_window)
                # Predict state at current point
                state_seq = model.predict(X_window)
                last_state = state_seq[-1]
                # Identify which state is "bull bond" (lower mean yield change)
                state_means = [X_window[state_seq == s, 0].mean() for s in range(P_S20_N_STATES)]
                bull_state = int(np.argmin(state_means))  # lower yield change = bull bond
                scores.iloc[i] = 1.0 if last_state == bull_state else -1.0
            except Exception:
                pass

        # Smooth with 5-day MA to avoid daily flipping
        scores_smooth = scores.rolling(5, min_periods=1).mean()
        sig = signal_from_score(scores_smooth, bull=0.3, bear=-0.3)
        sig[:warmup + 5] = np.nan
        sig.name = "S20_HMM_Regime"
        return sig

    except ImportError:
        print("WARNING: hmmlearn not installed, S20 disabled")
        return pd.Series(0, index=df.index, name="S20_HMM_Regime")
    except Exception as e:
        print(f"WARNING: S20 HMM failed: {e}")
        return pd.Series(0, index=df.index, name="S20_HMM_Regime")


def compute_S21(df: pd.DataFrame) -> pd.Series:
    """S21 — NSS Curve Deviation. 10Y residual vs fitted NSS curve."""
    yield_cols = {"dgs3m": 0.25, "dgs6m": 0.5, "dgs1": 1, "dgs2": 2,
                  "dgs5": 5, "dgs10": 10, "dgs20": 20, "dgs30": 30}
    available = {mat: col for col, mat in yield_cols.items() if col in df.columns}
    if len(available) < 5 or "dgs10" not in df.columns:
        return pd.Series(0, index=df.index, name="S21_NSS_Dev")

    try:
        from nelson_siegel_svensson.calibrate import calibrate_nss_ols
        import numpy as _np

        residuals = pd.Series(index=df.index, dtype=float)
        maturities = _np.array(sorted(available.keys()))

        for i in range(0, len(df), 5):  # compute every 5 days for speed
            row = df.iloc[i]
            yields_vals = _np.array([
                row.get(col, _np.nan)
                for mat, col in sorted((mat, col) for col, mat in yield_cols.items()
                                       if col in df.columns)
            ])
            if _np.isnan(yields_vals).sum() > 2:
                continue
            valid = ~_np.isnan(yields_vals)
            if valid.sum() < 5:
                continue
            try:
                curve, status = calibrate_nss_ols(maturities[valid], yields_vals[valid] / 100)
                if status:
                    fitted_10y = curve(10.0) * 100
                    actual_10y = row.get("dgs10", _np.nan)
                    if not _np.isnan(actual_10y):
                        residuals.iloc[i] = actual_10y - fitted_10y
            except Exception:
                continue

        # Forward-fill the every-5-day residuals
        residuals = residuals.ffill()
        score = zscore(residuals, P_S21_Z_WINDOW)
        # Positive residual = 10Y elevated vs model = buy (mean reversion to cheaper)
        sig = signal_from_score(score, bull=0.5, bear=-0.5)
        sig[:P_S21_Z_WINDOW + 63] = np.nan
        sig.name = "S21_NSS_Dev"
        return sig

    except ImportError:
        print("WARNING: nelson_siegel_svensson not installed, S21 disabled")
        return pd.Series(0, index=df.index, name="S21_NSS_Dev")
    except Exception as e:
        print(f"WARNING: S21 NSS failed: {e}")
        return pd.Series(0, index=df.index, name="S21_NSS_Dev")


def compute_S22(df: pd.DataFrame) -> pd.Series:
    """S22 — VECM Error Correction. Cointegration DGS10 vs CPI."""
    if "dgs10" not in df.columns or "cpiaucsl" not in df.columns:
        return pd.Series(0, index=df.index, name="S22_VECM_EC")
    try:
        import statsmodels.api as sm

        y = df["dgs10"].ffill()
        x = df["cpiaucsl"].ffill()
        x_log = np.log(x)

        ec_scores = pd.Series(index=df.index, dtype=float)
        step = P_S22_WINDOW

        for i in range(step, len(df), 21):  # refit monthly
            y_w = y.iloc[i - step:i].dropna()
            x_w = x_log.iloc[i - step:i].dropna()
            common = y_w.index.intersection(x_w.index)
            if len(common) < 200:
                continue
            try:
                ols = sm.OLS(y_w[common].values, sm.add_constant(x_w[common].values)).fit()
                alpha, beta = ols.params
                ec = y.iloc[i] - (alpha + beta * x_log.iloc[i])
                ec_scores.iloc[i] = ec
            except Exception:
                continue

        ec_scores = ec_scores.ffill()
        score = zscore(ec_scores, 252)
        # Positive EC = yield too high relative to CPI → expect mean reversion → long bonds
        sig = signal_from_score(score, bull=P_S22_EC_THRESH, bear=-P_S22_EC_THRESH)
        sig[:step + 252] = np.nan
        sig.name = "S22_VECM_EC"
        return sig
    except Exception as e:
        print(f"WARNING: S22 VECM failed: {e}")
        return pd.Series(0, index=df.index, name="S22_VECM_EC")


# ---------------------------------------------------------------------------
# CATEGORY F: Global Macro / Alt Data
# ---------------------------------------------------------------------------

def compute_S23(df: pd.DataFrame, roll_window: int = None) -> pd.Series:
    """S23 — Foreign Custody Holdings. Declining → foreign CB selling → bearish."""
    _rw = roll_window if roll_window is not None else P_S23_ROLL_WINDOW
    if "farbast" not in df.columns:
        return pd.Series(0, index=df.index, name="S23_ForeignCustody")
    custody = df["farbast"].ffill()
    # 4-week rolling change, z-scored
    change = custody.diff(_rw)
    score = zscore(change, P_S23_Z_WINDOW)
    # Positive change = accumulation = bullish; declining = bearish
    sig = signal_from_score(score, bull=0.5, bear=-0.5)
    sig[:P_S23_Z_WINDOW + _rw] = np.nan
    sig.name = "S23_ForeignCustody"
    return sig


def compute_S24(df: pd.DataFrame, z_window: int = None) -> pd.Series:
    """S24 — Global Bond Factor Residual. US yield elevated vs global → buy."""
    _z = z_window if z_window is not None else P_S24_Z_WINDOW
    cols = ["dgs10", "de10y", "jp10y"]
    available = [c for c in cols if c in df.columns]
    if len(available) < 2:
        return pd.Series(0, index=df.index, name="S24_GlobalFactor")

    yields = df[available].ffill()
    changes = yields.diff()

    scores = pd.Series(index=df.index, dtype=float)

    for i in range(_z, len(df)):
        window = changes.iloc[i - _z:i].dropna()
        if len(window) < 100 or window.shape[1] < 2:
            continue
        try:
            pca = PCA(n_components=1)
            global_factor = pca.fit_transform(window)[:, 0]
            # Loading of US yield on global factor
            us_loading = pca.components_[0, 0]
            # US residual from global factor
            us_reconstructed = global_factor * us_loading
            us_actual = window.iloc[:, 0].values  # dgs10 changes
            residual = us_actual[-1] - us_reconstructed[-1]
            scores.iloc[i] = residual
        except Exception:
            continue

    score = zscore(scores, _z)
    # Positive residual = US yield elevated vs global = mean reversion = buy
    sig = signal_from_score(score, bull=0.5, bear=-0.5)
    sig[:_z * 2] = np.nan
    sig.name = "S24_GlobalFactor"
    return sig


def compute_S25(df: pd.DataFrame, hawk_thresh: float = None) -> pd.Series:
    """S25 — FOMC Minutes Sentiment. Hawkish → short; dovish → long bonds."""
    _ht = hawk_thresh if hawk_thresh is not None else P_S25_HAWK_THRESH
    _dt = _ht - 0.20  # dove threshold stays 0.20 below hawk threshold
    if "fomc_hawk_score" not in df.columns:
        return pd.Series(0, index=df.index, name="S25_FOMC_Sentiment")
    hawk = df["fomc_hawk_score"].ffill()
    sig = pd.Series(
        np.where(hawk > _ht, -1,
                 np.where(hawk < _dt, 1, 0)),
        index=df.index, dtype=float
    )
    # Need at least 2 FOMC cycles before trusting
    first_valid = hawk.first_valid_index()
    if first_valid:
        sig[:df.index.get_loc(first_valid) + 120] = np.nan
    sig.name = "S25_FOMC_Sentiment"
    return sig


# ---------------------------------------------------------------------------
# Signal registry — all 25 signals with metadata
# ---------------------------------------------------------------------------

ALL_SIGNALS = [
    # (id, short_name, category, compute_fn)
    ("S01", "3M_Momentum",       "A", compute_S01),
    ("S02", "6M_Momentum",       "A", compute_S02),
    ("S03", "12_1_Momentum",     "A", compute_S03),
    ("S04", "SMA_Cross",         "A", compute_S04),
    ("S05", "RSI14_Yield",       "A", compute_S05),
    ("S06", "Carry_Slope",       "B", compute_S06),
    ("S07", "Breakeven_Mom",     "B", compute_S07),
    ("S08", "ACM_TermPrem",      "B", compute_S08),
    ("S09", "Real_Yield",        "B", compute_S09),
    ("S10", "CPI_Mom",           "B", compute_S10),
    ("S11", "VIX_Safety",        "C", compute_S11),
    ("S12", "HY_Spread",         "C", compute_S12),
    ("S13", "DXY_Mom",           "C", compute_S13),
    ("S14", "Oil_Mom",           "C", compute_S14),
    ("S15", "COT_Lev",           "D", compute_S15),
    ("S16", "PrimDealer",        "D", compute_S16),
    ("S17", "RRP_Filter",        "D", compute_S17),
    ("S18", "PCA_Slope",         "E", compute_S18),
    ("S19", "Kalman_Spread",     "E", compute_S19),
    ("S20", "HMM_Regime",        "E", compute_S20),
    ("S21", "NSS_Dev",           "E", compute_S21),
    ("S22", "VECM_EC",           "E", compute_S22),
    ("S23", "ForeignCustody",    "F", compute_S23),
    ("S24", "GlobalFactor",      "F", compute_S24),
    ("S25", "FOMC_Sentiment",    "F", compute_S25),
]


# ---------------------------------------------------------------------------
# IS optimisation registries
# ---------------------------------------------------------------------------

# Signals excluded from per-window grid search (too slow: HMM, NSS, VECM)
SKIP_IS_OPTIMIZATION = frozenset({"S20", "S21", "S22"})

# For each optimizable signal: {kwarg_name: [candidate_values]}
SIGNAL_PARAM_GRIDS = {
    "S01": {"lookback":     [21,  42,  63,  126]},
    "S02": {"lookback":     [63, 126, 189,  252]},
    "S03": {"lookback":    [126, 189, 252,  315]},
    "S04": {"slow":        [100, 150, 200,  252]},
    "S05": {"period":      [  7,  10,  14,   21]},
    "S06": {"z_window":   [126, 189, 252,  315]},
    "S07": {"mom_window":  [ 21,  42,  63,  126]},
    "S08": {"z_window":   [126, 189, 252,  315]},
    "S09": {"z_window":   [126, 189, 252,  315]},
    "S10": {"z_window":   [126, 189, 252,  315]},
    "S11": {"z_window":   [126, 189, 252,  315]},
    "S12": {"z_window":   [126, 189, 252,  315]},
    "S13": {"mom_window":  [  5,  10,  21,   42]},
    "S14": {"mom_window":  [  5,  10,  21,   42]},
    "S15": {"window":     [252, 378, 504,  630]},   # NOTE: 630 > IS_WINDOW=504, will be skipped
    "S16": {"z_window":   [126, 189, 252,  315]},
    "S17": {"roll_window": [ 10,  15,  20,   40]},
    "S18": {"z_window":   [126, 189, 252]},         # 3 values (each IS window is slow)
    "S19": {"obs_noise":   [0.1, 0.5, 1.0,  2.0]},
    "S23": {"roll_window": [ 10,  15,  20,   40]},
    "S24": {"z_window":   [126, 189, 252,  315]},
    "S25": {"hawk_thresh": [0.50, 0.55, 0.60, 0.65]},
}


def compute_all_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 25 signals. Returns DataFrame with one column per signal."""
    results = {}
    for sig_id, sig_name, category, fn in ALL_SIGNALS:
        col = f"{sig_id}_{sig_name}"
        print(f"  [{sig_id}] {sig_name}...")
        try:
            results[col] = fn(df)
        except Exception as e:
            print(f"  ERROR in {sig_id}: {e}")
            results[col] = pd.Series(0, index=df.index, name=col)
    return pd.DataFrame(results)
