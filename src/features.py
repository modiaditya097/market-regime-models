"""17-feature engineering per factor for the Sparse Jump Model."""

import numpy as np
import pandas as pd


def compute_ewma_return(ret: pd.Series, span: int) -> pd.Series:
    return ret.ewm(span=span, adjust=False).mean()


def compute_rsi(ret: pd.Series, span: int) -> pd.Series:
    """RSI using EWM smoothing with given span."""
    gain = ret.clip(lower=0)
    loss = (-ret).clip(lower=0)
    avg_gain = gain.ewm(span=span, adjust=False).mean()
    avg_loss = loss.ewm(span=span, adjust=False).mean()
    # When avg_loss == 0 all gains, no losses → RSI = 100
    rsi = pd.Series(np.where(
        avg_loss == 0.0,
        100.0,
        100.0 - 100.0 / (1.0 + avg_gain / avg_loss),
    ), index=ret.index, name=f"RSI_{span}")
    return rsi


def compute_stochastic_k(ret: pd.Series, window: int) -> pd.Series:
    """Stochastic %K using rolling window on cumulative return level."""
    price = (1.0 + ret).cumprod()
    lo  = price.rolling(window, min_periods=window).min()
    hi  = price.rolling(window, min_periods=window).max()
    denom = (hi - lo).replace(0.0, np.nan)
    k = (100.0 * (price - lo) / denom).clip(0.0, 100.0)
    return k.rename(f"K_{window}")


def compute_macd(ret: pd.Series, fast: int, slow: int) -> pd.Series:
    fast_ewm = ret.ewm(span=fast, adjust=False).mean()
    slow_ewm = ret.ewm(span=slow, adjust=False).mean()
    return (fast_ewm - slow_ewm).rename(f"MACD_{fast}_{slow}")


def compute_log_dd(ret: pd.Series, span: int) -> pd.Series:
    """Log of EWM downside deviation (using negative returns only)."""
    neg = ret.clip(upper=0.0)
    sq_mean = neg.pow(2).ewm(span=span, adjust=False).mean()
    dd = np.sqrt(sq_mean).replace(0.0, np.nan)
    return np.log(dd).rename(f"logDD_{span}")


def compute_active_beta(factor_ret: pd.Series, mkt_ret: pd.Series, span: int) -> pd.Series:
    """EWM active market beta: cov(factor, mkt) / var(mkt)."""
    f_dm = factor_ret - factor_ret.ewm(span=span, adjust=False).mean()
    m_dm = mkt_ret   - mkt_ret.ewm(span=span,   adjust=False).mean()
    cov  = (f_dm * m_dm).ewm(span=span, adjust=False).mean()
    var  = m_dm.pow(2).ewm(span=span, adjust=False).mean().replace(0.0, np.nan)
    return (cov / var).rename(f"beta_{span}")


def compute_factor_features(active_ret: pd.Series, mkt_ret: pd.Series) -> pd.DataFrame:
    """13 factor-specific features for one factor's active return series."""
    parts = []
    for span in [8, 21, 63]:
        parts.append(compute_ewma_return(active_ret, span).rename(f"ret_{span}"))
    for span in [8, 21, 63]:
        parts.append(compute_rsi(active_ret, span))
    for w in [8, 21, 63]:
        parts.append(compute_stochastic_k(active_ret, w))
    parts.append(compute_macd(active_ret, 8, 21))
    parts.append(compute_macd(active_ret, 21, 63))
    parts.append(compute_log_dd(active_ret, 21))
    parts.append(compute_active_beta(active_ret, mkt_ret, 21))
    return pd.concat(parts, axis=1)


def compute_market_features(
    mkt_ret: pd.Series,
    vix: pd.Series,
    y2: pd.Series,
    y10: pd.Series,
) -> pd.DataFrame:
    """4 market-environment features shared across all factors."""
    mkt_feat    = mkt_ret.ewm(span=21, adjust=False).mean().rename("mkt_ret_21")
    vix_logret  = np.log(vix / vix.shift(1))
    vix_feat    = vix_logret.ewm(span=21, adjust=False).mean().rename("vix_21")
    y2_diff     = y2.diff().ewm(span=21, adjust=False).mean().rename("y2_diff_21")
    slope_diff  = (y10 - y2).diff().ewm(span=21, adjust=False).mean().rename("slope_diff_21")
    return pd.concat([mkt_feat, vix_feat, y2_diff, slope_diff], axis=1)


def compute_all_features(
    active_rets: dict,
    mkt_ret: pd.Series,
    vix: pd.Series,
    y2: pd.Series,
    y10: pd.Series,
) -> dict:
    """
    Compute 17-feature matrix for each factor.
    Returns dict[factor_name -> pd.DataFrame with 17 columns].
    """
    mkt_feats = compute_market_features(mkt_ret, vix, y2, y10)
    result = {}
    for factor, ret in active_rets.items():
        factor_feats = compute_factor_features(ret, mkt_ret)
        X = pd.concat([factor_feats, mkt_feats], axis=1)
        X = X.reindex(ret.index)
        result[factor] = X
    return result
