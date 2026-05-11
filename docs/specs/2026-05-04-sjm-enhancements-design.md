# SJM Model 1 Enhancements — Design Spec
**Date:** 2026-05-04  
**Scope:** Three enhancements to the Sparse Jump Model (Model 1) pipeline and dashboard:
1. Extended macro feature set with frontend feature selection
2. Configurable 2-state / 3-state regime detection
3. Walk-forward evaluation with Results / Validation sub-tabs

---

## 1. Overview

The SJM pipeline currently has 4 fixed macro features, a hardcoded 2-state regime model, and a single train→test split with no cross-validation. This spec adds:

- **7 new optional macro features** selectable per run from the dashboard sidebar
- **3-state regime support** (bull / neutral / bear) toggled alongside the existing 2-state mode
- **Walk-forward evaluation** (expanding-window, n folds) running as an optional second phase after the standard backtest, with results shown in a new Validation sub-tab

All new parameters are exposed in the frontend and written to `tmp_config.yaml` before each run — no hardcoding.

---

## 2. Config Changes (`config.yaml`)

```yaml
data:
  start_date: "2000-01-01"
  end_date: "2026-01-30"       # was implicit; now frontend-configurable
  cache_dir: "outputs/cache"
  refresh: false

training:
  min_train_years: 8
  max_train_years: 12
  refit_freq: "M"
  test_start: "2008-01-01"     # NEW — was implicit as data_start + min_train_years

sjm:
  n_components: 2              # existing — now also 3 via frontend toggle
  jump_penalty: 50.0
  max_feats: 9.5
  max_iter: 30
  n_init_jm: 10
  random_state: 42

black_litterman:
  risk_aversion: 2.5
  cov_halflife: 126
  tau: 0.05
  target_tracking_error: 0.03
  transaction_cost_bps: 5
  benchmark_rebalance_freq: "Q"

macro_features:                # NEW
  enabled:                     # subset of the 7 available; all on by default
    - vix_level
    - dxy
    - oil_ret
    - gold_ret
    - real_yield
    - unemployment
    - consumer_sent

walk_forward:                  # NEW
  enabled: false
  n_folds: 6
  fold_test_months: 36

output:
  results_path: "outputs/results.csv"
  plots_dir: "outputs/plots"
```

The existing 4 macro features (`mkt_ret_21`, `vix_21`, `y2_diff_21`, `slope_diff_21`) are **always-on** and not in the selectable list.

---

## 3. Backend Changes

### 3.1 `src/data.py`

**New fetchers** (each returns a daily `pd.Series` ffill'd to the trading calendar):

| Key | Source | Frequency |
|---|---|---|
| `vix_level` | `yf.download("^VIX")` — Close | Daily |
| `dxy` | `yf.download("DX-Y.NYB")` — log return | Daily |
| `oil_ret` | `yf.download("CL=F")` — log return | Daily |
| `gold_ret` | `yf.download("GC=F")` — log return | Daily |
| `real_yield` | `FRED REAINTRATREARAT10Y` | Daily, ffill |
| `unemployment` | `FRED UNRATE` | Monthly, ffill to daily |
| `consumer_sent` | `FRED UMCSENT` | Monthly, ffill to daily |

**`load_macro(start, end, enabled: list[str])`** — updated signature. Only fetches and caches the series in `enabled`. Cache files are named per feature (e.g., `macro_vix_level.parquet`) so adding/removing features doesn't invalidate the full cache.

`load_all_data()` passes `cfg["macro_features"]["enabled"]` to `load_macro()`.

### 3.2 `src/features.py`

**`compute_market_features(mkt_ret, vix, y2, y10, macro_extras: dict, enabled: list[str]) -> pd.DataFrame`**

- Always computes the existing 4 features
- For each key in `enabled`, builds the corresponding feature from `macro_extras` and appends it
- Feature builders per key:

| Key | Transform |
|---|---|
| `vix_level` | log(VIX) — absolute level |
| `dxy` | EWMA(log return, span=21) |
| `oil_ret` | EWMA(log return, span=21) |
| `gold_ret` | EWMA(log return, span=21) |
| `real_yield` | diff().ewm(span=21) |
| `unemployment` | diff().ewm(span=21) |
| `consumer_sent` | diff().ewm(span=21) |

`compute_all_features()` passes `macro_extras` and `enabled` through from the caller.

`run_regime_detection()` in `regime.py` gains two new parameters: `macro_extras: dict` and `enabled: list[str]`, which it forwards to `compute_all_features()`. All callers (`main.py`, `run_walk_forward()`) pass these from config.

### 3.3 `src/regime.py`

No interface change — `n_components` already comes from `cfg["sjm"]["n_components"]`.

**3-state label convention** (when `n_components=3`, `sort_by="cumret"`):
- Label `0` = bull (highest cumulative return)
- Label `1` = neutral (middle)
- Label `2` = bear (lowest cumulative return)

The 1-day delay shift is unchanged.

### 3.4 `src/portfolio.py`

**`compute_view_returns()`** extended for 3-state:
- Label `0` (bull): `q[k]` = mean active return during bull periods in training data
- Label `1` (neutral): `q[k] = 0.0` — no view; BL posterior falls back to equal-weight prior
- Label `2` (bear): `q[k]` = mean active return during bear periods in training data (will be negative)

2-state behaviour is unchanged (labels 0/1).

### 3.5 `src/backtest.py`

**New function `run_walk_forward(data: dict, cfg: dict) -> pd.DataFrame`**

Calls `run_regime_detection()`, `run_portfolio_construction()`, and `compute_portfolio_returns()` directly (same functions as `main.py` Phase 1) on each fold slice.

```
test_start       = pd.Timestamp(cfg["training"]["test_start"])
fold_test_months = cfg["walk_forward"]["fold_test_months"]
n_folds          = cfg["walk_forward"]["n_folds"]

for fold in 1..n_folds:
    fold_test_start = test_start + (fold-1) * fold_test_months months
    fold_test_end   = fold_test_start + fold_test_months months
    if fold_test_end > data end: break

    train data: data_start → fold_test_start   (expanding window)
    test  data: fold_test_start → fold_test_end

    regime_labels   = run_regime_detection(train+test slice, cfg, macro_extras, enabled)
    weights         = run_portfolio_construction(regime_labels, ..., cfg)
    port_ret        = compute_portfolio_returns(weights, asset_returns, cost_bps)
    metrics         = compute_performance_table(port_ret, mkt_ret, ew_ret, rf, weights)

    print(f"[{fold}/{n_folds}] walk-forward fold {fold} ({fold_test_start.date()} – {fold_test_end.date()})")
    record: fold, period_label, sharpe, ir_vs_market, max_drawdown,
            active_ret_vs_market, volatility, turnover
```

Output: `outputs/wfe_results.csv` with one row per fold plus an average row.  
Progress output format: `[fold/n_folds] walk-forward fold {fold} (YYYY-MM-DD – YYYY-MM-DD)` (reuses existing `[step/total]` pattern the dashboard already parses).

### 3.6 `main.py`

```
Phase 1 (always):   standard backtest → outputs/results.csv, returns_te*.csv, weights_te*.csv, regimes.csv
Phase 2 (optional): if cfg["walk_forward"]["enabled"]: run_walk_forward() → outputs/wfe_results.csv
```

Two-phase progress: Phase 1 steps are `[1..N]` out of total, Phase 2 steps continue the count.

---

## 4. Frontend Changes

### 4.1 Sidebar — Accordion Structure (`shiny_app/modules/model1.py`)

Four collapsible sections. **Run Configuration** is open by default; others are collapsed.

**Section 1 — Run Configuration**
- `input_date("data_start")` — Data Start (default: `2000-01-01`)
- `input_date("data_end")` — Data End (default: `2026-01-30`)
- `input_date("test_start")` — Test Period Start (default: `2008-01-01`)
- `input_select("te")` — TE Target (existing)
- `input_radio_buttons("n_components", choices={"2": "2-state", "3": "3-state"})` — Regime States

**Section 2 — Model Parameters** (existing controls, unchanged)
- `input_numeric("jump_penalty")`, `input_numeric("max_feats")`, `input_numeric("risk_aversion")`, `input_numeric("txn_cost")`

**Section 3 — Macro Features**
- One `input_checkbox` per feature key, labelled with the feature's display name
- All 7 checked by default
- Summary line when collapsed: "N / 7 active"

**Section 4 — Walk-Forward**
- `input_checkbox("wfe_enabled", "Enable Walk-Forward")` — default unchecked
- `input_slider("n_folds", min=3, max=8, value=6)` — greyed out when disabled
- `input_select("fold_test_months", choices=["12","24","36"], selected="36")` — greyed out when disabled

**Progress indicator** — two progress bars shown during run:
- Phase 1 bar (green, labelled "Backtest")
- Phase 2 bar (blue, labelled "Walk-Forward") — hidden when WFE disabled

### 4.2 Main Content — Sub-tabs

`ui.navset_tab()` with two panels:

**Tab 1 — Results** (all existing sections, unchanged):
Metrics, Cumulative Returns, Rolling Sharpe, Drawdown, Realized TE, Portfolio Weights, Regime Plots

**Tab 2 — Validation**

When WFE not yet run:
> "Enable Walk-Forward in the sidebar and run the model to see validation results."

When `wfe_results.csv` exists:
1. **Fold Metrics Table** — `output_ui("wfe_metrics_tbl")` — HTML table with columns: Fold, Period, Sharpe, IR vs Mkt, Max DD, Active Ret, Turnover. Average row at bottom. Values colour-coded green/red.
2. **Sharpe by Fold** — `output_ui("wfe_sharpe_plot")` — horizontal bar chart, one bar per fold, coloured by positive/negative, dashed line at average Sharpe.
3. **IR by Fold** — `output_ui("wfe_ir_plot")` — same layout as Sharpe chart.

### 4.3 New Analytics Functions (`shiny_app/components/analytics.py`)

**`load_wfe_results(output_dir: Path) -> pd.DataFrame | None`**  
Reads `outputs/wfe_results.csv`. Returns `None` if file absent.

**`wfe_metrics_html(df: pd.DataFrame) -> str`**  
HTML table matching the styling of `_metrics_comparison_html()`. Colour-codes Sharpe (green ≥ 0.5), IR (green > 0), Max DD (always red).

**`wfe_folds_plot(df: pd.DataFrame, metric: str) -> plt.Figure`**  
Horizontal bar chart for the given metric (e.g., `"sharpe"`, `"ir_vs_market"`) across folds. Bars coloured `#198754` (positive) / `#dc3545` (negative). Dashed vertical line at fold average. Returns `plt.Figure`.

### 4.4 `tmp_config.yaml` — New Fields Written on Run

The `_launch_pipeline` server function writes all new sidebar values to `tmp_config.yaml` before spawning the subprocess:

```python
cfg_data["data"]["start_date"]              = str(input.data_start())
cfg_data["data"]["end_date"]                = str(input.data_end())
cfg_data["training"]["test_start"]          = str(input.test_start())
cfg_data["sjm"]["n_components"]             = int(input.n_components())
cfg_data["macro_features"]["enabled"]       = [k for k in MACRO_KEYS if getattr(input, k)()]
cfg_data["walk_forward"]["enabled"]         = bool(input.wfe_enabled())
cfg_data["walk_forward"]["n_folds"]         = int(input.n_folds())
cfg_data["walk_forward"]["fold_test_months"]= int(input.fold_test_months())
```

`MACRO_KEYS` is the list of 7 feature keys defined at module level.

---

## 5. Data Flow Summary

```
Sidebar inputs
    → tmp_config.yaml
    → main.py Phase 1 (always)
        → data.py (fetch/cache enabled macro series)
        → features.py (build 4 + N enabled extra features)
        → regime.py (SJM with n_components=2 or 3)
        → portfolio.py (BL with 2/3-state view logic)
        → backtest.py (returns, metrics, weights, regimes)
        → outputs/results.csv, returns_te*.csv, weights_te*.csv, regimes.csv
    → main.py Phase 2 (if walk_forward.enabled)
        → backtest.run_walk_forward()
        → outputs/wfe_results.csv
    → Dashboard reads outputs/
        → Results tab: existing charts (unchanged)
        → Validation tab: wfe_metrics_html(), wfe_folds_plot()
```

---

## 6. Files Changed

| File | Change type |
|---|---|
| `config.yaml` | Add `training.test_start`, `macro_features`, `walk_forward` blocks |
| `src/data.py` | Add 7 fetchers; refactor `load_macro()` signature |
| `src/features.py` | Extend `compute_market_features()` with `macro_extras` + `enabled` |
| `src/backtest.py` | Add `run_walk_forward()` |
| `src/portfolio.py` | Extend `compute_view_returns()` for 3-state |
| `main.py` | Add Phase 2 conditional block |
| `shiny_app/modules/model1.py` | Accordion sidebar; inner sub-tabs; new `tmp_config` fields |
| `shiny_app/components/analytics.py` | Add `load_wfe_results()`, `wfe_metrics_html()`, `wfe_folds_plot()` |

No new files required. All changes are additive or extend existing functions with backward-compatible signatures.

---

## 7. Out of Scope

- Changes to Model 2 or Model 3
- Parallel fold execution (folds run sequentially)
- Caching of intermediate regime labels across folds
- Export / download of walk-forward results as CSV from the dashboard
