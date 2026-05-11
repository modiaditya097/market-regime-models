# Model 1 Analytics Upgrade Design
**Date:** 2026-04-20
**Context:** QWIM dashboard — Dynamic Factor Allocation tab analytics depth + interactive charts

---

## Background

The Model 1 tab currently shows static pre-generated PNGs (cumulative returns, portfolio weights, 5 regime plots) and a single-row metrics table for the selected TE target. This upgrade replaces all static PNGs with computed matplotlib figures that react to the TE selector, adds 3 new analytical charts, and replaces the metrics table with a full multi-TE comparison table.

---

## Section 1 — Pipeline CSV exports

**Goal:** Save two new CSVs so the dashboard has the raw data needed for interactive charts.

### New function: `save_weights_csv` in `src/backtest.py`

```python
def save_weights_csv(weights: pd.DataFrame, output_dir: str, te_suffix: str = "") -> None:
    path = os.path.join(output_dir, f"weights{te_suffix}.csv")
    weights.to_csv(path)
```

Called in the TE loop in `main.py` after `plot_portfolio_weights(...)`:
```python
save_weights_csv(weights_te.reindex(weights.index), output_dir, te_suffix=suffix)
```

Outputs: `outputs/weights_te1.csv`, `outputs/weights_te2.csv`, `outputs/weights_te3.csv`, `outputs/weights_te4.csv`
Columns: `market, value, size, quality, growth, momentum` (daily weights, decimal fractions)

### New CSV export in `main.py` after Step 3

After the `for f, labels in regime_labels.items()` loop (line 78), add:
```python
regimes_df = pd.DataFrame(regime_labels)
regimes_df.to_csv(os.path.join(os.path.dirname(cfg["output"]["results_path"]), "regimes.csv"))
```

Output: `outputs/regimes.csv`
Columns: `value, size, quality, growth, momentum`
Values: 0 (bull) or 1 (bear) per trading date

---

## Section 2 — New analytics module

**File:** `shiny_app/components/analytics.py` (create new)

### Data loaders

```python
def load_weights_df(output_dir: Path, te_pct: int) -> pd.DataFrame | None:
    path = output_dir / f"weights_te{te_pct}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)

def load_regimes_df(output_dir: Path) -> pd.DataFrame | None:
    path = output_dir / "regimes.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)

def load_all_metrics(output_dir: Path) -> pd.DataFrame:
    """Return full results.csv (all TE targets + EW benchmark)."""
    path = output_dir / "results.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
```

### Chart functions

All functions:
- Accept DataFrames as input (no file I/O)
- Return `matplotlib.figure.Figure`
- Use `fig, ax = plt.subplots(figsize=(10, 4))`
- Return a blank figure with grey "No data available" text if input is `None` or empty
- Do NOT call `plt.show()` or `plt.close()`

**1. `cumulative_returns_plot(returns_df)`**
- Input columns: `portfolio`, `market`, `ew`
- Compute: `(1 + series).cumprod() - 1` × 100 for each
- Colors: portfolio=`#7c3aed` solid, market=`#888888` dashed, ew=`#10b981` dashed
- Labels: "Dynamic Allocation", "Market (S&P 500)", "Equal Weight"
- `ax.axhline(0, color="black", lw=0.5)`, grid, legend, y-label "Cumulative Return (%)"

**2. `rolling_sharpe_plot(returns_df, window=252)`**
- Same input columns
- Compute per series: `series.rolling(window).mean() / series.rolling(window).std() * sqrt(252)`
- `ax.axhline(0, color="black", lw=0.5, linestyle="--")`
- Same colors/labels as above; y-label "Rolling Sharpe Ratio (252-day)"

**3. `drawdown_plot(returns_df)`**
- Same input columns
- Per series: `cum = (1+r).cumprod(); dd = (cum / cum.cummax() - 1) * 100`
- Fill between 0 and portfolio drawdown with `alpha=0.12`, color `#7c3aed`
- Same colors/labels; y-label "Drawdown (%)"

**4. `realized_te_plot(returns_df, target_te: float)`**
- Compute: `active = returns_df["portfolio"] - returns_df["market"]`
- `realized = active.rolling(63).std() * sqrt(252) * 100`
- Plot realized TE as solid line (`#7c3aed`), target TE × 100 as horizontal dashed line (`#dc3545`)
- Label the dashed line `f"Target TE ({target_te*100:.0f}%)"`
- y-label "Annualized Tracking Error (%)"

**5. `portfolio_weights_plot(weights_df)`**
- Input columns: `market, value, size, quality, growth, momentum`
- Stacked area chart: `ax.stackplot(dates, [weights_df[c]*100 for c in ASSETS], labels=ASSETS, alpha=0.85)`
- Colors: fixed 6-color palette `["#888888","#7c3aed","#10b981","#f97316","#0ea5e9","#eab308"]`
- y-label "Weight (%)", legend below chart (`loc="upper left"`)

**6. `regime_plot(factor, regimes_df, active_returns_df)`**
- Compute cumulative active return from `active_returns_df[factor]`
- Get regime labels from `regimes_df[factor]` (0=bull, 1=bear), reindexed to active_ret dates
- Shade contiguous bull spans green (`alpha=0.2`), bear spans red (`alpha=0.2`) using `ax.axvspan`
- Plot cumulative active return as `#7c3aed` line
- `ax.axhline(0, color="black", lw=0.5)`; y-label "Cumulative Active Return (%)"

---

## Section 3 — Dashboard changes (`shiny_app/modules/model1.py`)

### Import changes

Add:
```python
import matplotlib
matplotlib.use("Agg")
from shiny_app.components.analytics import (
    load_weights_df, load_regimes_df, load_all_metrics,
    cumulative_returns_plot as _cum_returns_chart,
    rolling_sharpe_plot    as _rolling_sharpe_chart,
    drawdown_plot          as _drawdown_chart,
    realized_te_plot       as _realized_te_chart,
    portfolio_weights_plot as _weights_chart,
    regime_plot            as _regime_chart,
)
```

Remove: `img_tag`, `load_metrics_row`, `_DISPLAY_COLS` from the `charts` import line.
Keep: `load_returns_df`, `placeholder_card`, `section`.

Note: aliases are required because the render output function names (`rolling_sharpe_plot`, etc.) would shadow the module-level imports if they shared the same name.

### Sidebar anchor links (updated)

```python
ui.tags.a("Metrics",             href="#metrics"),
ui.tags.a("Cumulative Returns",  href="#returns"),
ui.tags.a("Rolling Sharpe",      href="#rolling-sharpe"),
ui.tags.a("Drawdown",            href="#drawdown"),
ui.tags.a("Realized TE",         href="#realized-te"),
ui.tags.a("Portfolio Weights",   href="#weights"),
ui.tags.a("Regime Plots",        href="#regimes"),
```

### UI main sections (updated)

```python
main = ui.div(
    section("Performance Metrics",    "metrics",        ui.output_ui("metrics_tbl")),
    section("Cumulative Returns",     "returns",        ui.output_plot("returns_plot", height="350px")),
    section("Rolling Sharpe",         "rolling-sharpe", ui.output_plot("rolling_sharpe_plot", height="300px")),
    section("Drawdown",               "drawdown",       ui.output_plot("drawdown_plot", height="300px")),
    section("Realized Tracking Error","realized-te",    ui.output_plot("realized_te_plot", height="300px")),
    section("Portfolio Weights",      "weights",        ui.output_plot("weights_plot", height="350px")),
    section("Regime Plots",           "regimes",        ui.div(*[
        ui.div(
            ui.h6(f.capitalize(), class_="text-muted"),
            ui.output_plot(f"regime_{f}_plot", height="250px"),
            class_="mb-3",
        ) for f in _FACTORS
    ])),
    style="padding:1rem",
)
```

### Server: metrics table (`@render.ui`)

Replace the existing `@render.table metrics_tbl` with `@render.ui`:

```python
@render.ui
def metrics_tbl():
    df = load_all_metrics(output_dir)
    if df.empty:
        return placeholder_card("No results — run the model first.")
    te = int(input.te())
    return ui.HTML(_metrics_comparison_html(df, selected_te=te))
```

`_metrics_comparison_html(df, selected_te)` is a module-level helper that:
- Filters for rows where `target_te` is in `[0.01, 0.02, 0.03, 0.04]` or `strategy == "EW Benchmark"`
- Renders the table from Section 3 of the design conversation (TE targets as columns, metrics as rows)
- Highlights the column matching `selected_te` with a green `#d4edda` background
- Color-codes: Sharpe green ≥0.5 / red <0; Max DD always red; IR vs Mkt green >0 / red <0

Metrics rows to display: `sharpe`, `max_drawdown`, `ir_vs_market`, `volatility`, `active_ret_vs_market`, `turnover`

Display names: `Sharpe`, `Max DD`, `IR vs Mkt`, `Volatility`, `Active Ret`, `Turnover`

Format: use the same `_format_value` and `_METRIC_COLORS` logic already proven in `shiny_app/modules/generic_model_tab.py`. Copy those two helpers verbatim into `analytics.py` so `_metrics_comparison_html` can reuse them.

### Server: chart renders

```python
@render.plot(alt="Cumulative returns")
def returns_plot():
    df = load_returns_df(output_dir, te_pct=int(input.te()))
    return _cum_returns_chart(df)

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

@render.plot(alt="Portfolio weights")
def weights_plot():
    wdf = load_weights_df(output_dir, te_pct=int(input.te()))
    return _weights_chart(wdf)
```

**Regime plots — dynamic loop:**
```python
def _make_regime_renderer(f: str):
    @output(id=f"regime_{f}_plot")
    @render.plot(alt=f"{f} regime")
    def _regime():
        reg = load_regimes_df(output_dir)
        active = pd.read_parquet(output_dir / "cache/active_returns.parquet")
        return _regime_chart(f, reg, active)
    return _regime

for _f in _FACTORS:
    _make_regime_renderer(_f)
```

---

## Files Changed

| File | Change |
|---|---|
| `src/backtest.py` | Add `save_weights_csv(weights, output_dir, te_suffix)` function |
| `main.py` | Call `save_regimes_csv` after Step 3; call `save_weights_csv` in TE loop; import `save_weights_csv` |
| `shiny_app/components/analytics.py` | **Create** — 3 loaders + 6 chart functions + `_metrics_comparison_html` helper |
| `shiny_app/modules/model1.py` | Update imports; add `_metrics_comparison_html`; update sidebar anchors; update UI sections; replace 4 renders with `@render.plot`; add 4 new `@render.plot`; add dynamic regime loop |

---

## Out of Scope

- Changes to SC-HMM / HMM / HSMM / MS-GARCH tabs
- Modifying SJM or Black-Litterman algorithm logic
- New data sources
- Exporting the new CSVs from the SC-HMM pipeline
