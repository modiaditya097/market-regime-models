import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils import annualize_return, annualize_vol, sharpe_ratio, info_ratio, max_drawdown, FACTORS, ASSETS

def make_ret(val, n=252):
    return pd.Series([val] * n)

def test_constants():
    assert len(FACTORS) == 5
    assert len(ASSETS) == 6
    assert 'market' in ASSETS
    assert 'market' not in FACTORS

def test_annualize_return_zero():
    assert annualize_return(make_ret(0.0)) == pytest.approx(0.0, abs=1e-6)

def test_annualize_return_positive():
    # 1% daily for 252 days → (1.01)^252 - 1 ≈ 11.6x
    r = annualize_return(make_ret(0.01))
    assert r > 10.0

def test_annualize_vol():
    ret = make_ret(0.01)
    # constant series → vol = 0
    assert annualize_vol(ret) == pytest.approx(0.0, abs=1e-6)

def test_sharpe_ratio():
    ret = make_ret(0.001)
    rf  = make_ret(0.0)
    # positive return, zero vol → very large Sharpe
    assert sharpe_ratio(ret, rf) > 5.0

def test_max_drawdown_flat():
    ret = make_ret(0.0)
    assert max_drawdown(ret) == pytest.approx(0.0, abs=1e-6)

def test_max_drawdown_negative():
    # Falling 1% every day for 10 days
    ret = make_ret(-0.01, n=10)
    mdd = max_drawdown(ret)
    assert mdd < 0

def test_info_ratio():
    active = make_ret(0.001)
    ir = info_ratio(active)
    assert ir > 5.0
