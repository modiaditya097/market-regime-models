import pytest
import pandas as pd
from pathlib import Path
from shiny_app.components.charts import load_metrics_row, load_returns_df

def make_results_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame([
        {"strategy": "EW Benchmark", "target_te": None,
         "sharpe": 0.55, "ir_vs_market": 0.17, "max_drawdown": -0.53,
         "volatility": 0.20, "active_ret_vs_market": 0.005,
         "active_ret_vs_ew": 0.0, "ir_vs_ew": 0.0, "turnover": 0.0},
        {"strategy": "Dynamic (TE=3%)", "target_te": 0.03,
         "sharpe": 0.562, "ir_vs_market": 0.093, "max_drawdown": -0.524,
         "volatility": 0.197, "active_ret_vs_market": 0.003,
         "active_ret_vs_ew": -0.001, "ir_vs_ew": -0.02, "turnover": 9.16},
    ])
    path = tmp_path / "results.csv"
    df.to_csv(path, index=False)
    return tmp_path

def make_returns_csv(tmp_path: Path, te_pct: int) -> None:
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=5),
        "portfolio": [0.01, -0.005, 0.002, 0.003, 0.001],
        "market": [0.008, -0.006, 0.001, 0.002, 0.0],
        "ew": [0.009, -0.005, 0.001, 0.002, 0.001],
    })
    df.to_csv(tmp_path / f"returns_te{te_pct}.csv", index=False)

def test_load_metrics_row_returns_dataframe(tmp_path):
    output_dir = make_results_csv(tmp_path)
    df = load_metrics_row(output_dir, te_pct=3)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert abs(df.iloc[0]["sharpe"] - 0.562) < 0.001

def test_load_metrics_row_missing_file_returns_empty(tmp_path):
    df = load_metrics_row(tmp_path / "nonexistent", te_pct=3)
    assert df.empty

def test_load_returns_df_parses_dates(tmp_path):
    make_returns_csv(tmp_path, te_pct=3)
    df = load_returns_df(tmp_path, te_pct=3)
    assert df is not None
    assert pd.api.types.is_datetime64_any_dtype(df.index)
    assert list(df.columns) == ["portfolio", "market", "ew"]

def test_load_returns_df_missing_file_returns_none(tmp_path):
    result = load_returns_df(tmp_path, te_pct=3)
    assert result is None
