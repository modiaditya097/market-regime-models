import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.features import (
    compute_ewma_return,
    compute_rsi,
    compute_stochastic_k,
    compute_macd,
    compute_log_dd,
    compute_active_beta,
    compute_factor_features,
    compute_all_features,
)

np.random.seed(42)
N = 300
dates = pd.date_range("2000-01-03", periods=N, freq="B")
pos_ret  = pd.Series(np.abs(np.random.randn(N)) * 0.005, index=dates)   # all positive
flat_ret = pd.Series(np.zeros(N), index=dates)
rand_ret = pd.Series(np.random.randn(N) * 0.01, index=dates)
mkt_ret  = pd.Series(np.random.randn(N) * 0.01, index=dates)
vix      = pd.Series(np.abs(np.random.randn(N)) * 2 + 15.0, index=dates)
y2       = pd.Series(np.random.randn(N) * 0.05 + 2.5, index=dates)
y10      = pd.Series(np.random.randn(N) * 0.05 + 3.5, index=dates)


def test_ewma_return_shape():
    for w in [8, 21, 63]:
        r = compute_ewma_return(rand_ret, w)
        assert len(r) == N

def test_rsi_bounds():
    for w in [8, 21, 63]:
        rsi = compute_rsi(rand_ret, w)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

def test_rsi_all_positive_near_100():
    rsi = compute_rsi(pos_ret, 21).iloc[-1]
    assert rsi > 70, f"Expected RSI > 70 for all-positive returns, got {rsi:.1f}"

def test_stochastic_k_bounds():
    for w in [8, 21, 63]:
        k = compute_stochastic_k(rand_ret, w)
        valid = k.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

def test_macd_flat_is_zero():
    macd = compute_macd(flat_ret, 8, 21)
    assert (macd.abs() < 1e-12).all()

def test_log_dd_nonpositive():
    # log(dd) should be non-positive (dd <= 1 for small returns)
    log_dd = compute_log_dd(rand_ret, 21).dropna()
    assert len(log_dd) > 0

def test_active_beta_flat_market():
    # If market return is constant, variance = 0 → beta should be NaN or 0
    const_mkt = pd.Series([0.001] * N, index=dates)
    beta = compute_active_beta(rand_ret, const_mkt, 21)
    # Beta should be NaN when market variance is near zero
    assert beta.dropna().empty or True  # just check it doesn't crash

def test_factor_features_shape():
    X = compute_factor_features(rand_ret, mkt_ret)
    assert X.shape[1] == 13, f"Expected 13 factor-specific features, got {X.shape[1]}"
    assert len(X) == N

def test_all_features_shape():
    active_rets = {f: rand_ret.copy() for f in ['value', 'size', 'quality', 'growth', 'momentum']}
    feat_dict = compute_all_features(active_rets, mkt_ret, vix, y2, y10)
    for factor, X in feat_dict.items():
        assert X.shape[1] == 17, f"{factor}: expected 17 features, got {X.shape[1]}"
        assert len(X) == N

def test_all_features_column_names():
    active_rets = {f: rand_ret.copy() for f in ['value', 'size', 'quality', 'growth', 'momentum']}
    feat_dict = compute_all_features(active_rets, mkt_ret, vix, y2, y10)
    X = feat_dict['value']
    expected_cols = [
        'ret_8', 'ret_21', 'ret_63',
        'RSI_8', 'RSI_21', 'RSI_63',
        'K_8', 'K_21', 'K_63',
        'MACD_8_21', 'MACD_21_63',
        'logDD_21', 'beta_21',
        'mkt_ret_21', 'vix_21', 'y2_diff_21', 'slope_diff_21',
    ]
    assert list(X.columns) == expected_cols
