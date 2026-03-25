import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.backtest import (
    compute_transaction_costs,
    compute_portfolio_returns,
    compute_ew_returns,
    compute_performance_table,
)
from src.utils import N_ASSETS, ASSETS

np.random.seed(42)
N = 504
dates = pd.date_range("2008-01-02", periods=N, freq="B")

# Constant weights: no turnover
const_weights = pd.DataFrame(
    np.tile(np.ones(N_ASSETS) / N_ASSETS, (N, 1)), index=dates, columns=ASSETS
)
asset_rets = pd.DataFrame(
    np.random.randn(N, N_ASSETS) * 0.01, index=dates, columns=ASSETS
)
rf = pd.Series(np.full(N, 0.00005), index=dates)  # ~1.3% annual


def test_transaction_costs_zero_turnover():
    # Constant weights → zero turnover → zero costs
    costs = compute_transaction_costs(const_weights, cost_bps=5)
    assert (costs.abs() < 1e-12).all()

def test_transaction_costs_full_turnover():
    # Flip from all-market to all-momentum: 100% one-way turnover each day
    w = const_weights.copy()
    w.iloc[1::2, :] = 0.0
    w.iloc[1::2, ASSETS.index('momentum')] = 1.0
    costs = compute_transaction_costs(w, cost_bps=5)
    # On flip days: one-way turnover ≈ 1.0 → cost = 2 * 0.0005 * 1.0 = 0.001
    flip_costs = costs.iloc[1::2]
    assert (flip_costs > 0).all()

def test_portfolio_returns_shape():
    port_ret = compute_portfolio_returns(const_weights, asset_rets, cost_bps=5)
    assert len(port_ret) == N
    assert isinstance(port_ret, pd.Series)

def test_portfolio_returns_no_cost_is_weighted_mean():
    port_ret = compute_portfolio_returns(const_weights, asset_rets, cost_bps=0)
    expected = (const_weights.shift(1) * asset_rets).sum(axis=1)
    # First row is NaN due to shift
    pd.testing.assert_series_equal(
        port_ret.iloc[1:].reset_index(drop=True),
        expected.iloc[1:].reset_index(drop=True),
        atol=1e-10,
    )

def test_ew_returns():
    ew = compute_ew_returns(asset_rets)
    assert len(ew) == N
    assert isinstance(ew, pd.Series)

def test_performance_table_keys():
    port_ret = compute_portfolio_returns(const_weights, asset_rets, cost_bps=5)
    ew_ret   = compute_ew_returns(asset_rets)
    mkt_ret  = asset_rets['market']
    metrics  = compute_performance_table(port_ret, mkt_ret, ew_ret, rf, const_weights)
    required = ['sharpe', 'max_drawdown', 'excess_return', 'ir_vs_market',
                'ir_vs_ew', 'active_ret_vs_market', 'active_ret_vs_ew', 'turnover']
    for k in required:
        assert k in metrics, f"Missing metric: {k}"
