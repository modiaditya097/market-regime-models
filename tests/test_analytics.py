"""Tests for shiny_app/components/analytics.py"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from shiny_app.components.analytics import (
    load_weights_df,
    load_regimes_df,
    load_all_metrics,
    cumulative_returns_plot,
    rolling_sharpe_plot,
    drawdown_plot,
    realized_te_plot,
    portfolio_weights_plot,
    regime_plot,
    _metrics_comparison_html,
)


def _returns_df(n=300):
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "portfolio": rng.normal(0.0005, 0.01, n),
            "market":    rng.normal(0.0004, 0.01, n),
            "ew":        rng.normal(0.0003, 0.01, n),
        },
        index=dates,
    )


def test_load_weights_df_missing_returns_none(tmp_path):
    assert load_weights_df(tmp_path, 3) is None


def test_load_weights_df_reads_csv(tmp_path):
    dates = pd.bdate_range("2020-01-01", periods=10)
    df = pd.DataFrame({"market": [0.1] * 10, "value": [0.2] * 10}, index=dates)
    df.to_csv(tmp_path / "weights_te3.csv")
    result = load_weights_df(tmp_path, 3)
    assert result is not None
    assert "market" in result.columns
    assert len(result) == 10


def test_load_regimes_df_missing_returns_none(tmp_path):
    assert load_regimes_df(tmp_path) is None


def test_load_regimes_df_reads_csv(tmp_path):
    dates = pd.bdate_range("2020-01-01", periods=5)
    df = pd.DataFrame({"value": [0, 1, 0, 0, 1]}, index=dates)
    df.to_csv(tmp_path / "regimes.csv")
    result = load_regimes_df(tmp_path)
    assert result is not None
    assert "value" in result.columns


def test_load_all_metrics_missing_returns_empty(tmp_path):
    assert load_all_metrics(tmp_path).empty


def test_load_all_metrics_reads_csv(tmp_path):
    df = pd.DataFrame([{"strategy": "EW", "target_te": None, "sharpe": 0.5}])
    df.to_csv(tmp_path / "results.csv", index=False)
    result = load_all_metrics(tmp_path)
    assert "sharpe" in result.columns


def test_cumulative_returns_plot_returns_figure():
    fig = cumulative_returns_plot(_returns_df())
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_cumulative_returns_plot_none_input():
    fig = cumulative_returns_plot(None)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_rolling_sharpe_plot_returns_figure():
    fig = rolling_sharpe_plot(_returns_df())
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_drawdown_plot_returns_figure():
    fig = drawdown_plot(_returns_df())
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_realized_te_plot_returns_figure():
    fig = realized_te_plot(_returns_df(), target_te=0.03)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_portfolio_weights_plot_returns_figure():
    dates = pd.bdate_range("2020-01-01", periods=50)
    df = pd.DataFrame(
        {a: [1 / 6] * 50 for a in ["market", "value", "size", "quality", "growth", "momentum"]},
        index=dates,
    )
    fig = portfolio_weights_plot(df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_portfolio_weights_plot_none_input():
    fig = portfolio_weights_plot(None)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_regime_plot_returns_figure():
    n = 100
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(0)
    regimes = pd.DataFrame({"value": rng.integers(0, 2, n)}, index=dates)
    active  = pd.DataFrame({"value": rng.normal(0, 0.01, n)}, index=dates)
    fig = regime_plot("value", regimes, active)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_regime_plot_missing_factor_returns_blank():
    dates = pd.bdate_range("2020-01-01", periods=10)
    regimes = pd.DataFrame({"value": [0] * 10}, index=dates)
    active  = pd.DataFrame({"value": [0.0] * 10}, index=dates)
    fig = regime_plot("momentum", regimes, active)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_metrics_comparison_html_contains_sharpe_row():
    rows = [
        {
            "strategy": "Dynamic (TE=3%)", "target_te": 0.03,
            "sharpe": 0.61, "max_drawdown": -0.18, "ir_vs_market": 0.42,
            "volatility": 0.11, "active_ret_vs_market": 0.018, "turnover": 0.031,
        },
        {
            "strategy": "EW Benchmark", "target_te": None,
            "sharpe": 0.52, "max_drawdown": -0.19, "ir_vs_market": 0.28,
            "volatility": 0.10, "active_ret_vs_market": 0.0, "turnover": 0.000,
        },
    ]
    df = pd.DataFrame(rows)
    html = _metrics_comparison_html(df, selected_te=3)
    assert "Sharpe" in html
    assert "TE=3%" in html
    assert "★" in html


def test_metrics_comparison_html_empty_df():
    html = _metrics_comparison_html(pd.DataFrame(), selected_te=3)
    assert isinstance(html, str)


# ── Walk-Forward Evaluation tests ────────────────────────────────────────────

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shiny_app.components.analytics import (
    load_wfe_results,
    wfe_metrics_html,
    wfe_folds_plot,
)
from pathlib import Path
import tempfile

_WFE_DATA = pd.DataFrame([
    {"fold": 1, "period_start": "2008-01-01", "period_end": "2011-01-01",
     "sharpe": 0.21, "ir_vs_market": -0.05, "max_drawdown": -0.58,
     "active_ret_vs_market": -0.012, "volatility": 0.18, "turnover": 0.4},
    {"fold": 2, "period_start": "2011-01-01", "period_end": "2014-01-01",
     "sharpe": 0.74, "ir_vs_market": 0.18, "max_drawdown": -0.22,
     "active_ret_vs_market": 0.021, "volatility": 0.14, "turnover": 0.3},
    {"fold": "avg", "period_start": "", "period_end": "",
     "sharpe": 0.475, "ir_vs_market": 0.065, "max_drawdown": -0.40,
     "active_ret_vs_market": 0.0045, "volatility": 0.16, "turnover": 0.35},
])


def test_load_wfe_results_returns_none_when_missing():
    result = load_wfe_results(Path("/nonexistent/path"))
    assert result is None


def test_load_wfe_results_returns_dataframe(tmp_path):
    _WFE_DATA.to_csv(tmp_path / "wfe_results.csv", index=False)
    result = load_wfe_results(tmp_path)
    assert isinstance(result, pd.DataFrame)
    assert "sharpe" in result.columns


def test_wfe_metrics_html_returns_string():
    html = wfe_metrics_html(_WFE_DATA)
    assert isinstance(html, str)
    assert "<table" in html
    assert "avg" in html.lower() or "Avg" in html


def test_wfe_metrics_html_empty_df():
    html = wfe_metrics_html(pd.DataFrame())
    assert "No walk-forward" in html


def test_wfe_folds_plot_returns_figure():
    fig = wfe_folds_plot(_WFE_DATA, metric="sharpe")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_wfe_folds_plot_ir_metric():
    fig = wfe_folds_plot(_WFE_DATA, metric="ir_vs_market")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
