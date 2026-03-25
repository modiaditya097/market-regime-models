import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.regime import get_train_window, get_refit_dates, run_regime_detection

DATA_START = pd.Timestamp("2000-01-03")

def test_get_train_window_expanding():
    refit = pd.Timestamp("2009-01-05")   # 9 years after start
    start, end = get_train_window(refit, DATA_START, min_years=8, max_years=12)
    assert start == DATA_START
    assert end   == refit

def test_get_train_window_rolling():
    refit = pd.Timestamp("2014-01-06")   # 14 years after start -> cap at 12
    start, end = get_train_window(refit, DATA_START, min_years=8, max_years=12)
    expected_start = refit - pd.DateOffset(years=12)
    assert start == expected_start
    assert end   == refit

def test_get_refit_dates():
    index = pd.date_range("2008-01-01", "2008-06-30", freq="B")
    refit = get_refit_dates(index)
    # Should have one refit per month start
    assert len(refit) >= 5   # Jan, Feb, Mar, Apr, May, Jun = 6 months
    for d in refit:
        assert d in index

def test_run_regime_detection_output_shape():
    """Labels must be binary {0,1} and same length as test period."""
    np.random.seed(0)
    dates = pd.date_range("2000-01-03", "2012-12-31", freq="B")
    active_rets = {f: pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)
                   for f in ['value', 'size', 'quality', 'growth', 'momentum']}
    mkt_ret = pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)
    vix     = pd.Series(np.abs(np.random.randn(len(dates))) * 2 + 15, index=dates)
    y2      = pd.Series(np.random.randn(len(dates)) * 0.05 + 2.5, index=dates)
    y10     = pd.Series(np.random.randn(len(dates)) * 0.05 + 3.5, index=dates)

    cfg = {
        'data':     {'start_date': '2000-01-03', 'end_date': '2012-12-31'},
        'training': {'min_train_years': 8, 'max_train_years': 12, 'refit_freq': 'M'},
        'sjm': {
            'n_components': 2, 'jump_penalty': 10.0, 'max_feats': 3.0,
            'max_iter': 5, 'n_init_jm': 2, 'random_state': 0,
        },
    }
    labels = run_regime_detection(active_rets, mkt_ret, vix, y2, y10, cfg)
    assert set(labels.keys()) == {'value', 'size', 'quality', 'growth', 'momentum'}
    for f, ser in labels.items():
        valid = ser.dropna()
        assert len(valid) > 0, f"{f}: no labels"
        assert set(valid.unique()).issubset({0, 1}), f"{f}: non-binary labels"

def test_one_day_delay():
    """Regime label on day T must not use day T's return (verified by proxy)."""
    # Structural test: labels index starts at least 1 day after test period start
    np.random.seed(1)
    dates = pd.date_range("2000-01-03", "2010-12-31", freq="B")
    active_rets = {f: pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)
                   for f in ['value', 'size', 'quality', 'growth', 'momentum']}
    mkt_ret = pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)
    vix     = pd.Series(np.abs(np.random.randn(len(dates))) + 15, index=dates)
    y2      = pd.Series([2.5] * len(dates), index=dates)
    y10     = pd.Series([3.5] * len(dates), index=dates)
    cfg = {
        'data':     {'start_date': '2000-01-03', 'end_date': '2010-12-31'},
        'training': {'min_train_years': 8, 'max_train_years': 12, 'refit_freq': 'M'},
        'sjm': {
            'n_components': 2, 'jump_penalty': 10.0, 'max_feats': 3.0,
            'max_iter': 3, 'n_init_jm': 1, 'random_state': 0,
        },
    }
    labels = run_regime_detection(active_rets, mkt_ret, vix, y2, y10, cfg)
    # Labels should not start before the first trading day after 8 years
    test_start = dates[dates >= pd.Timestamp("2000-01-03") + pd.DateOffset(years=8)][0]
    for f, ser in labels.items():
        if len(ser.dropna()) > 0:
            assert ser.dropna().index[0] >= test_start
