import numpy as np
import pandas as pd

FACTORS = ['value', 'size', 'quality', 'growth', 'momentum']
ASSETS  = ['market', 'value', 'size', 'quality', 'growth', 'momentum']
N_ASSETS = len(ASSETS)
TRADING_DAYS = 252


def annualize_return(ret_ser: pd.Series) -> float:
    """Compound annualized return from daily return series."""
    cum = (1 + ret_ser).prod()
    n_years = len(ret_ser) / TRADING_DAYS
    return float(cum ** (1 / n_years) - 1) if n_years > 0 else 0.0


def annualize_vol(ret_ser: pd.Series) -> float:
    return float(ret_ser.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(ret_ser: pd.Series, rf_ser: pd.Series) -> float:
    excess = ret_ser.values - rf_ser.values
    std = np.std(excess, ddof=1)
    if std < 1e-10:
        # zero volatility: return sign of mean * large number, or 0 if flat
        mean = np.mean(excess)
        if abs(mean) < 1e-10:
            return 0.0
        return float(np.sign(mean) * np.inf)
    return float(np.mean(excess) / std * np.sqrt(TRADING_DAYS))


def info_ratio(active_ret: pd.Series) -> float:
    std = active_ret.std(ddof=1)
    if std < 1e-10:
        mean = active_ret.mean()
        if abs(mean) < 1e-10:
            return 0.0
        return float(np.sign(mean) * np.inf)
    return float(active_ret.mean() / std * np.sqrt(TRADING_DAYS))


def max_drawdown(ret_ser: pd.Series) -> float:
    cum = (1 + ret_ser).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    return float(drawdown.min())


def load_config(path: str = "config.yaml") -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)
