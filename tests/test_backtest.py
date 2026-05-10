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
    save_weights_csv,
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


def test_save_weights_csv(tmp_path):
    dates = pd.bdate_range("2020-01-01", periods=10)
    weights = pd.DataFrame(
        {a: [1 / 6] * 10 for a in ["market", "value", "size", "quality", "growth", "momentum"]},
        index=dates,
    )
    save_weights_csv(weights, str(tmp_path), te_suffix="_te3")
    path = tmp_path / "weights_te3.csv"
    assert path.exists()
    loaded = pd.read_csv(path, index_col=0, parse_dates=True)
    assert list(loaded.columns) == ["market", "value", "size", "quality", "growth", "momentum"]
    assert len(loaded) == 10


from unittest.mock import patch

def test_run_walk_forward_returns_dataframe_with_expected_columns():
    from src.backtest import run_walk_forward
    import numpy as np

    np.random.seed(0)
    N = 252 * 5
    d = pd.date_range("2000-01-03", periods=N, freq="B")
    total = pd.DataFrame(np.random.randn(N, 6) * 0.01, index=d,
                         columns=["market","value","size","quality","growth","momentum"])
    active = pd.DataFrame(np.random.randn(N, 5) * 0.005, index=d,
                          columns=["value","size","quality","growth","momentum"])
    rf_ser = pd.Series(np.full(N, 0.0001), index=d)
    macro  = {
        "vix": pd.Series(np.abs(np.random.randn(N)) + 15, index=d),
        "y2":  pd.Series(np.full(N, 2.5), index=d),
        "y10": pd.Series(np.full(N, 3.5), index=d),
    }
    data = {"total_returns": total, "active_returns": active, "rf": rf_ser, "macro": macro}
    cfg = {
        "data":     {"start_date": "2000-01-03", "end_date": "2004-12-31"},
        "training": {"min_train_years": 1, "max_train_years": 3,
                     "refit_freq": "M", "test_start": "2001-01-01"},
        "sjm":      {"n_components": 2, "jump_penalty": 50.0, "max_feats": 9.5,
                     "max_iter": 5, "n_init_jm": 2, "random_state": 42},
        "black_litterman": {"risk_aversion": 2.5, "cov_halflife": 63, "tau": 0.05,
                            "target_tracking_error": 0.03, "transaction_cost_bps": 5,
                            "benchmark_rebalance_freq": "Q"},
        "macro_features": {"enabled": []},
        "walk_forward":   {"enabled": True, "n_folds": 2, "fold_test_months": 12},
    }

    # Mock the expensive pipeline steps
    test_start = pd.Timestamp("2001-01-01")
    test_dates = d[d >= test_start]
    mock_labels = {f: pd.Series(np.zeros(len(test_dates), dtype=int), index=test_dates)
                   for f in ["value","size","quality","growth","momentum"]}
    mock_weights = pd.DataFrame(
        np.ones((len(test_dates), 6)) / 6, index=test_dates,
        columns=["market","value","size","quality","growth","momentum"]
    )
    mock_port_ret = pd.Series(np.random.randn(len(test_dates)) * 0.01, index=test_dates)

    with patch("src.backtest.run_regime_detection", return_value=mock_labels), \
         patch("src.backtest.run_portfolio_construction", return_value=mock_weights), \
         patch("src.backtest.compute_portfolio_returns", return_value=mock_port_ret):
        result = run_walk_forward(data, cfg)

    assert isinstance(result, pd.DataFrame)
    required_cols = {"fold", "period_start", "period_end", "sharpe",
                     "ir_vs_market", "max_drawdown", "active_ret_vs_market",
                     "volatility", "turnover"}
    assert required_cols.issubset(set(result.columns))
    # Last row is the average
    assert result.iloc[-1]["fold"] == "avg"
    # Non-avg rows count <= n_folds
    data_rows = result[result["fold"] != "avg"]
    assert len(data_rows) <= cfg["walk_forward"]["n_folds"]
