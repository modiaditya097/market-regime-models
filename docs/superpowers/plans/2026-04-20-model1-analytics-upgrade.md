# Model 1 Analytics Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the Dynamic Factor Allocation tab with interactive matplotlib charts (replacing static PNGs), 3 new analytical charts, and a multi-TE comparison metrics table.

**Architecture:** Three sequential tasks — (1) add two CSV export functions to the pipeline so the dashboard has raw data, (2) create a new `analytics.py` component with all chart functions and loaders, (3) update `model1.py` to use the new analytics module and render everything as live plots. All new charts react to the TE selector in the sidebar.

**Tech Stack:** Python 3.10+, Posit Shiny for Python ≥0.9, matplotlib, pandas, numpy, pytest

---

## File Map

| File | Action |
|---|---|
| `src/backtest.py` | Modify — add `save_weights_csv` function |
| `main.py` | Modify — import `save_weights_csv`, save regimes CSV after Step 3, call `save_weights_csv` in TE loop |
| `shiny_app/components/analytics.py` | **Create** — 3 loaders, 6 chart functions, `_metrics_comparison_html` |
| `shiny_app/modules/model1.py` | Modify — new imports, new sidebar anchors, new UI sections, replace 4 renders, add 4 new renders, add dynamic regime render loop |
| `tests/test_backtest.py` | Modify — add `test_save_weights_csv` |
| `tests/test_analytics.py` | **Create** — tests for all loaders and chart functions |

---

## Task 1: Pipeline CSV exports

**Files:**
- Modify: `src/backtest.py`
- Modify: `main.py`
- Modify: `tests/test_backtest.py`

### Context

`src/backtest.py` already has `save_returns_csv` which saves daily return series. We need the same pattern for weights. `main.py` also needs to save `regimes.csv` once after Step 3 (regime detection), and call `save_weights_csv` inside the per-TE loop.

The existing `save_returns_csv` signature for reference:
```python
def save_returns_csv(port_ret, mkt_ret, ew_ret, output_dir: str, te_suffix: str) -> None:
```

- [ ] **Step 1: Write the failing test for `save_weights_csv`**

Open `tests/test_backtest.py` and add this test (add the `save_weights_csv` import alongside existing backtest imports at the top of the file):

```python
from src.backtest import save_weights_csv
```

Add the test function at the bottom of the file:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "C:\Users\adity\Desktop\Final Project"
python -m pytest tests/test_backtest.py::test_save_weights_csv -v
```

Expected: FAIL — `ImportError: cannot import name 'save_weights_csv'`

- [ ] **Step 3: Implement `save_weights_csv` in `src/backtest.py`**

Add this function directly after `save_returns_csv` (around line 104):

```python
def save_weights_csv(weights: pd.DataFrame, output_dir: str, te_suffix: str = "") -> None:
    """Save daily portfolio weights to CSV for the Shiny dashboard."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"weights{te_suffix}.csv")
    weights.to_csv(path)
    print(f"Saved {path}")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd "C:\Users\adity\Desktop\Final Project"
python -m pytest tests/test_backtest.py::test_save_weights_csv -v
```

Expected: PASS

- [ ] **Step 5: Update `main.py` — import and call `save_weights_csv`**

In `main.py`, find the import block from `src.backtest` (lines 23–32). Add `save_weights_csv` to it:

```python
from src.backtest import (
    compute_portfolio_returns,
    compute_ew_returns,
    compute_performance_table,
    save_results,
    save_weights_csv,
    plot_cumulative_returns,
    plot_regime,
    plot_portfolio_weights,
    save_returns_csv,
)
```

- [ ] **Step 6: Save `regimes.csv` in `main.py` after Step 3**

After the loop `for f, labels in regime_labels.items(): ...` (around line 80), add these lines:

```python
    # Save regime labels for dashboard (TE-independent — save once)
    _out_dir = os.path.dirname(cfg["output"]["results_path"])
    os.makedirs(_out_dir, exist_ok=True)
    pd.DataFrame(regime_labels).to_csv(os.path.join(_out_dir, "regimes.csv"))
    print("      Saved regimes.csv")
```

- [ ] **Step 7: Call `save_weights_csv` in the per-TE loop in `main.py`**

In the second for-loop `for te in te_targets:` (around lines 158–180), add the call immediately after `plot_portfolio_weights(...)`:

```python
        plot_portfolio_weights(weights_te.reindex(weights.index), plots_dir, te_suffix=suffix)
        save_weights_csv(weights_te.reindex(weights.index), output_dir, te_suffix=suffix)
        save_returns_csv(port_ret_te, mkt_ret, ew_ret, output_dir, te_suffix=suffix)
```

- [ ] **Step 8: Verify `main.py` still imports cleanly**

```bash
cd "C:\Users\adity\Desktop\Final Project"
python -c "import main; print('OK')"
```

Expected: `OK` (no import errors)

- [ ] **Step 9: Run full test suite to check nothing broke**

```bash
cd "C:\Users\adity\Desktop\Final Project"
python -m pytest tests/test_backtest.py -v
```

Expected: all tests PASS

- [ ] **Step 10: Commit**

```bash
cd "C:\Users\adity\Desktop\Final Project"
git add src/backtest.py main.py tests/test_backtest.py
git commit -m "feat(pipeline): export weights_te{N}.csv and regimes.csv for dashboard"
```

---

## Task 2: Create analytics module

**Files:**
- Create: `shiny_app/components/analytics.py`
- Create: `tests/test_analytics.py`

### Context

This is a pure Python module — no Shiny imports. It has:
- 3 data loaders that read CSVs/parquet from `outputs/`
- 6 chart functions that accept DataFrames and return `matplotlib.figure.Figure`
- 1 HTML table builder for the multi-TE metrics comparison

The `_format_value` and `_METRIC_COLORS` helpers are copied from `shiny_app/modules/generic_model_tab.py` (already working in production).

- [ ] **Step 1: Write failing tests for loaders**

Create `tests/test_analytics.py`:

```python
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


# ── Loaders ────────────────────────────────────────────────────────────────


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


# ── Chart functions ────────────────────────────────────────────────────────


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
    fig = regime_plot("momentum", regimes, active)  # "momentum" not in df
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ── HTML table ─────────────────────────────────────────────────────────────


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "C:\Users\adity\Desktop\Final Project"
python -m pytest tests/test_analytics.py -v 2>&1 | head -20
```

Expected: FAIL — `ModuleNotFoundError: No module named 'shiny_app.components.analytics'`

- [ ] **Step 3: Create `shiny_app/components/analytics.py`**

Create the file with this exact content:

```python
"""Analytics charts and data loaders for the Dynamic Factor Allocation tab."""

from math import sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_ASSETS   = ["market", "value", "size", "quality", "growth", "momentum"]
_FACTORS  = ["value", "size", "quality", "growth", "momentum"]
_COLORS   = ["#888888", "#7c3aed", "#10b981", "#f97316", "#0ea5e9", "#eab308"]

_METRIC_COLORS = {
    "Sharpe":     lambda v: "#198754" if v >= 0.5 else ("#dc3545" if v < 0 else "inherit"),
    "IR vs Mkt":  lambda v: "#198754" if v > 0 else "#dc3545",
    "Max DD":     lambda v: "#dc3545",
    "Volatility": lambda v: "inherit",
    "Active Ret": lambda v: "#198754" if v > 0 else "#dc3545",
    "Turnover":   lambda v: "inherit",
}

_DISPLAY = {
    "sharpe":               "Sharpe",
    "ir_vs_market":         "IR vs Mkt",
    "max_drawdown":         "Max DD",
    "volatility":           "Volatility",
    "active_ret_vs_market": "Active Ret",
    "turnover":             "Turnover",
}


def _format_value(col_display: str, raw: float) -> str:
    if col_display in ("Sharpe", "IR vs Mkt"):
        return f"{raw:.3f}"
    if col_display in ("Max DD", "Volatility", "Active Ret"):
        return f"{raw:.2f}%"
    if col_display == "Turnover":
        return f"{raw:.4f}"
    return str(raw)


def _blank_fig(msg: str = "No data available") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            color="grey", fontsize=12, transform=ax.transAxes)
    ax.set_axis_off()
    plt.tight_layout()
    return fig


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_weights_df(output_dir: Path, te_pct: int) -> pd.DataFrame | None:
    path = Path(output_dir) / f"weights_te{te_pct}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_regimes_df(output_dir: Path) -> pd.DataFrame | None:
    path = Path(output_dir) / "regimes.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_all_metrics(output_dir: Path) -> pd.DataFrame:
    path = Path(output_dir) / "results.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df.rename(columns={"active_return": "active_ret_vs_market"})


# ── Chart functions ───────────────────────────────────────────────────────────

def cumulative_returns_plot(returns_df: pd.DataFrame | None) -> plt.Figure:
    if returns_df is None or returns_df.empty:
        return _blank_fig()
    fig, ax = plt.subplots(figsize=(10, 4))
    _styles = [
        ("portfolio", "Dynamic Allocation", "#7c3aed", "-",  1.5),
        ("market",    "Market (S&P 500)",   "#888888", "--", 1.2),
        ("ew",        "Equal Weight",       "#10b981", "--", 1.2),
    ]
    for col, label, color, ls, lw in _styles:
        if col not in returns_df.columns:
            continue
        cum = (1 + returns_df[col]).cumprod() - 1
        ax.plot(cum.index, cum * 100, label=label, color=color, linestyle=ls, linewidth=lw)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def rolling_sharpe_plot(returns_df: pd.DataFrame | None, window: int = 252) -> plt.Figure:
    if returns_df is None or returns_df.empty:
        return _blank_fig()
    fig, ax = plt.subplots(figsize=(10, 4))
    _styles = [
        ("portfolio", "Dynamic Allocation", "#7c3aed", "-",  1.5),
        ("market",    "Market (S&P 500)",   "#888888", "--", 1.2),
        ("ew",        "Equal Weight",       "#10b981", "--", 1.2),
    ]
    for col, label, color, ls, lw in _styles:
        if col not in returns_df.columns:
            continue
        r = returns_df[col]
        rs = r.rolling(window).mean() / r.rolling(window).std() * sqrt(252)
        ax.plot(rs.index, rs, label=label, color=color, linestyle=ls, linewidth=lw)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylabel(f"Rolling Sharpe Ratio ({window}-day)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def drawdown_plot(returns_df: pd.DataFrame | None) -> plt.Figure:
    if returns_df is None or returns_df.empty:
        return _blank_fig()
    fig, ax = plt.subplots(figsize=(10, 4))
    _styles = [
        ("portfolio", "Dynamic Allocation", "#7c3aed", "-",  1.5),
        ("market",    "Market (S&P 500)",   "#888888", "--", 1.2),
        ("ew",        "Equal Weight",       "#10b981", "--", 1.2),
    ]
    for col, label, color, ls, lw in _styles:
        if col not in returns_df.columns:
            continue
        cum = (1 + returns_df[col]).cumprod()
        dd  = (cum / cum.cummax() - 1) * 100
        ax.plot(dd.index, dd, label=label, color=color, linestyle=ls, linewidth=lw)
        if col == "portfolio":
            ax.fill_between(dd.index, dd, 0, alpha=0.12, color=color)
    ax.set_ylabel("Drawdown (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def realized_te_plot(
    returns_df: pd.DataFrame | None,
    target_te: float,
    window: int = 63,
) -> plt.Figure:
    if (
        returns_df is None
        or returns_df.empty
        or "portfolio" not in returns_df.columns
        or "market" not in returns_df.columns
    ):
        return _blank_fig()
    active   = returns_df["portfolio"] - returns_df["market"]
    realized = active.rolling(window).std() * sqrt(252) * 100
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(realized.index, realized, color="#7c3aed", linewidth=1.5, label="Realized TE")
    ax.axhline(
        target_te * 100,
        color="#dc3545", linewidth=1.2, linestyle="--",
        label=f"Target TE ({target_te * 100:.0f}%)",
    )
    ax.set_ylabel("Annualized Tracking Error (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def portfolio_weights_plot(weights_df: pd.DataFrame | None) -> plt.Figure:
    if weights_df is None or weights_df.empty:
        return _blank_fig()
    cols   = [c for c in _ASSETS if c in weights_df.columns]
    colors = [_COLORS[_ASSETS.index(c)] for c in cols]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stackplot(
        weights_df.index,
        [weights_df[c].values * 100 for c in cols],
        labels=cols,
        colors=colors,
        alpha=0.85,
    )
    ax.set_ylabel("Weight (%)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def regime_plot(
    factor: str,
    regimes_df: pd.DataFrame | None,
    active_returns_df: pd.DataFrame | None,
) -> plt.Figure:
    if regimes_df is None or active_returns_df is None:
        return _blank_fig(f"No regime data for {factor}")
    if factor not in regimes_df.columns or factor not in active_returns_df.columns:
        return _blank_fig(f"No regime data for {factor}")
    active     = active_returns_df[factor]
    cum_active = (1 + active).cumprod() - 1
    idx        = cum_active.index
    labels     = regimes_df[factor].reindex(idx)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(idx, cum_active * 100, color="#7c3aed", linewidth=1.0)
    prev_date, prev_label = idx[0], labels.iloc[0]
    for i in range(1, len(idx)):
        if labels.iloc[i] != prev_label or i == len(idx) - 1:
            color = "green" if prev_label == 0 else "red"
            ax.axvspan(prev_date, idx[i], alpha=0.2, color=color)
            prev_date, prev_label = idx[i], labels.iloc[i]
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"{factor.capitalize()} — Regime Detection")
    ax.set_ylabel("Cumulative Active Return (%)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ── Metrics comparison HTML ───────────────────────────────────────────────────

def _metrics_comparison_html(df: pd.DataFrame, selected_te: int) -> str:
    if df.empty:
        return "<p class='text-muted'>No results — run the model first.</p>"

    te_vals = [1, 2, 3, 4]
    te_rows: dict[int, pd.Series | None] = {}
    for te in te_vals:
        target = te / 100
        mask   = (df["target_te"] - target).abs() < 1e-6
        te_rows[te] = df[mask].iloc[0] if mask.any() else None

    ew_mask = df["strategy"].str.contains("EW", na=False)
    ew_row  = df[ew_mask].iloc[0] if ew_mask.any() else None

    def _th(label: str, selected: bool, ew: bool = False) -> str:
        bg = "#d4edda" if selected else ("#fff3cd" if ew else "#e8f4ff")
        fw = "700"    if selected else "normal"
        star = " ★"   if selected else ""
        return (
            f"<th style='padding:5px 8px;text-align:right;"
            f"background:{bg};font-weight:{fw}'>TE={label}%{star}</th>"
        )

    header = (
        "<tr style='border-bottom:2px solid #dee2e6'>"
        "<th style='padding:5px 8px;text-align:left'>Metric</th>"
    )
    for te in te_vals:
        header += _th(str(te), te == selected_te)
    header += "<th style='padding:5px 8px;text-align:right;background:#fff3cd'>EW Bench</th></tr>"

    body = ""
    for raw_col, display_col in _DISPLAY.items():
        row_html = f"<tr><td style='padding:5px 8px;font-weight:600'>{display_col}</td>"
        for te in te_vals:
            bg  = "#d4edda" if te == selected_te else "#e8f4ff"
            row = te_rows[te]
            if row is None or raw_col not in row.index:
                row_html += f"<td style='padding:5px 8px;text-align:right;background:{bg}'>—</td>"
            else:
                val = row[raw_col]
                try:
                    color = _METRIC_COLORS[display_col](float(val))
                    fmt   = _format_value(display_col, float(val))
                    fw    = "700" if te == selected_te else "normal"
                    row_html += (
                        f"<td style='padding:5px 8px;text-align:right;"
                        f"background:{bg};color:{color};font-weight:{fw}'>{fmt}</td>"
                    )
                except (TypeError, ValueError):
                    row_html += f"<td style='padding:5px 8px;text-align:right;background:{bg}'>—</td>"
        if ew_row is not None and raw_col in ew_row.index:
            val = ew_row[raw_col]
            try:
                color = _METRIC_COLORS[display_col](float(val))
                fmt   = _format_value(display_col, float(val))
                row_html += (
                    f"<td style='padding:5px 8px;text-align:right;"
                    f"background:#fff3cd;color:{color}'>{fmt}</td>"
                )
            except (TypeError, ValueError):
                row_html += "<td style='padding:5px 8px;text-align:right;background:#fff3cd'>—</td>"
        else:
            row_html += "<td style='padding:5px 8px;text-align:right;background:#fff3cd'>—</td>"
        row_html += "</tr>"
        body += row_html

    return (
        "<table class='table table-sm' style='max-width:700px'>"
        f"<thead>{header}</thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
        "<p style='font-size:10px;color:#6c757d;margin-top:4px'>★ = currently selected TE</p>"
    )
```

- [ ] **Step 4: Run all analytics tests to verify they pass**

```bash
cd "C:\Users\adity\Desktop\Final Project"
python -m pytest tests/test_analytics.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Verify the module imports cleanly**

```bash
cd "C:\Users\adity\Desktop\Final Project"
python -c "from shiny_app.components.analytics import load_weights_df, cumulative_returns_plot, _metrics_comparison_html; print('OK')"
```

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
cd "C:\Users\adity\Desktop\Final Project"
git add shiny_app/components/analytics.py tests/test_analytics.py
git commit -m "feat(analytics): add analytics module with 6 chart functions and metrics table"
```

---

## Task 3: Update model1.py

**Files:**
- Modify: `shiny_app/modules/model1.py`

### Context

`model1.py` currently uses `@render.table` for the metrics table and `@render.ui` (returning `img_tag(...)`) for all three plot sections. We're replacing all of these with `@render.plot` (returning matplotlib `Figure`) and `@render.ui` (returning `ui.HTML(...)`) for the metrics table.

The current file structure:
- Lines 1–13: imports
- Lines 14–23: module constants and CSS
- Lines 26–37: `_load_param_defaults` function
- Lines 40–90: `model_tab_ui` (sidebar + main UI layout)
- Lines 93–236: `model_tab_server`

Key changes in `model_tab_server`:
1. Replace `@render.table metrics_tbl` (lines 200–206) with `@render.ui metrics_tbl`
2. Replace `@render.ui returns_img` (lines 208–214) with `@render.plot returns_plot`
3. Replace `@render.ui weights_img` (lines 216–222) with `@render.plot weights_plot`
4. Replace `@render.ui regime_imgs` (lines 224–236) with a dynamic loop of 5 `@render.plot` outputs
5. Add new `@render.plot` for `rolling_sharpe_plot`, `drawdown_plot`, `realized_te_plot`

- [ ] **Step 1: Update the imports at the top of `model1.py`**

Replace the existing import block (lines 1–12) with:

```python
"""SJM + Black-Litterman model tab (Layout C: sticky sidebar + scrollable sections)."""

import asyncio
import re
from pathlib import Path

import pandas as pd
import yaml as _yaml

from shiny import module, output, ui, render, reactive

from shiny_app.components.analytics import (
    load_weights_df,
    load_regimes_df,
    load_all_metrics,
    cumulative_returns_plot  as _cum_returns_chart,
    rolling_sharpe_plot      as _rolling_sharpe_chart,
    drawdown_plot            as _drawdown_chart,
    realized_te_plot         as _realized_te_chart,
    portfolio_weights_plot   as _weights_chart,
    regime_plot              as _regime_chart,
    _metrics_comparison_html,
)
from shiny_app.components.charts import load_returns_df
from shiny_app.components.layout import placeholder_card, section
```

Also find and remove the inline `import pandas as pd` line inside the `model_tab_server` function body (around line 95 in the original file) — `pd` is now imported at module level and the inline import is redundant.

- [ ] **Step 2: Update the sidebar anchor links in `model_tab_ui`**

Find the `ui.div(... class_="anchor-links" ...)` block inside `model_tab_ui`. Replace it with:

```python
        ui.div(
            ui.tags.a("Metrics",              href="#metrics"),
            ui.tags.a("Cumulative Returns",   href="#returns"),
            ui.tags.a("Rolling Sharpe",       href="#rolling-sharpe"),
            ui.tags.a("Drawdown",             href="#drawdown"),
            ui.tags.a("Realized TE",          href="#realized-te"),
            ui.tags.a("Portfolio Weights",    href="#weights"),
            ui.tags.a("Regime Plots",         href="#regimes"),
            class_="anchor-links",
        ),
```

- [ ] **Step 3: Update the `main` content area in `model_tab_ui`**

Replace the existing `main = ui.div(...)` block with:

```python
    main = ui.div(
        section("Performance Metrics",     "metrics",
                ui.output_ui("metrics_tbl")),
        section("Cumulative Returns",      "returns",
                ui.output_plot("returns_plot", height="350px")),
        section("Rolling Sharpe",          "rolling-sharpe",
                ui.output_plot("rolling_sharpe_plot", height="300px")),
        section("Drawdown",                "drawdown",
                ui.output_plot("drawdown_plot", height="300px")),
        section("Realized Tracking Error", "realized-te",
                ui.output_plot("realized_te_plot", height="300px")),
        section("Portfolio Weights",       "weights",
                ui.output_plot("weights_plot", height="350px")),
        section("Regime Plots",            "regimes",
                ui.div(*[
                    ui.div(
                        ui.h6(f.capitalize(), class_="text-muted"),
                        ui.output_plot(f"regime_{f}_plot", height="250px"),
                        class_="mb-3",
                    )
                    for f in _FACTORS
                ])),
        style="padding:1rem",
    )
```

- [ ] **Step 4: Replace `metrics_tbl` renderer in `model_tab_server`**

Find and delete the `@render.table` block (currently lines 200–206):
```python
    @render.table
    def metrics_tbl():
        te = int(input.te())
        df = load_metrics_row(output_dir, te_pct=te)
        if df.empty:
            return pd.DataFrame({"Status": ["No results — run the model first"]})
        return df.rename(columns=_DISPLAY_COLS)
```

Replace with:
```python
    @render.ui
    def metrics_tbl():
        df = load_all_metrics(output_dir)
        return ui.HTML(_metrics_comparison_html(df, selected_te=int(input.te())))
```

- [ ] **Step 5: Replace static image renders with `@render.plot`**

Delete the three `@render.ui` blocks for `returns_img`, `weights_img`, and `regime_imgs` (currently lines 208–236).

Replace with:

```python
    @render.plot(alt="Cumulative returns")
    def returns_plot():
        df = load_returns_df(output_dir, te_pct=int(input.te()))
        return _cum_returns_chart(df)

    @render.plot(alt="Portfolio weights")
    def weights_plot():
        wdf = load_weights_df(output_dir, te_pct=int(input.te()))
        return _weights_chart(wdf)
```

- [ ] **Step 6: Add three new analytical chart renders**

Add these after `weights_plot`:

```python
    @render.plot(alt="Rolling Sharpe")
    def rolling_sharpe_plot():
        df = load_returns_df(output_dir, te_pct=int(input.te()))
        return _rolling_sharpe_chart(df)

    @render.plot(alt="Drawdown")
    def drawdown_plot():
        df = load_returns_df(output_dir, te_pct=int(input.te()))
        return _drawdown_chart(df)

    @render.plot(alt="Realized TE")
    def realized_te_plot():
        df = load_returns_df(output_dir, te_pct=int(input.te()))
        return _realized_te_chart(df, target_te=int(input.te()) / 100)
```

- [ ] **Step 7: Add dynamic regime plot renders**

Add this after `realized_te_plot`:

```python
    def _make_regime_renderer(f: str):
        @output(id=f"regime_{f}_plot")
        @render.plot(alt=f"{f} regime")
        def _regime():
            reg    = load_regimes_df(output_dir)
            active = pd.read_parquet(output_dir / "cache" / "active_returns.parquet")
            return _regime_chart(f, reg, active)
        return _regime

    for _f in _FACTORS:
        _make_regime_renderer(_f)
```

- [ ] **Step 8: Verify model1 imports cleanly**

```bash
cd "C:\Users\adity\Desktop\Final Project"
python -c "from shiny_app.modules.model1 import model_tab_ui, model_tab_server; print('OK')"
```

Expected: `OK`

- [ ] **Step 9: Verify full app builds with no errors**

```bash
cd "C:\Users\adity\Desktop\Final Project"
python -c "from shiny_app.app import build_app; build_app(); print('OK')" 2>&1
```

Expected: `OK` with no tracebacks.

- [ ] **Step 10: Run full test suite**

```bash
cd "C:\Users\adity\Desktop\Final Project"
python -m pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected: all tests PASS (the analytics tests cover the new code; model1.py has no unit tests but the import and build checks verify correctness)

- [ ] **Step 11: Commit**

```bash
cd "C:\Users\adity\Desktop\Final Project"
git add shiny_app/modules/model1.py
git commit -m "feat(model1): replace static PNGs with interactive charts, add rolling Sharpe/drawdown/realized TE"
```
