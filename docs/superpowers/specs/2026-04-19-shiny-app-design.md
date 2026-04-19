# Shiny App Design — Dynamic Factor Allocation Dashboard

**Date:** 2026-04-19  
**Status:** Approved

---

## 1. Overview

A Python Shiny app that presents 3 quantitative portfolio models, each in its own tab, with a first tab comparing all three. The app is config-driven so teammates can register their models without editing the core app code.

**Tech stack:** Python Shiny (`shiny` package), `app_config.yaml` for model registry, `subprocess` for pipeline re-runs.

---

## 2. File Structure

```
shiny_app/
├── app.py                  # Entry point: reads config, builds tab bar, mounts modules
├── app_config.yaml         # Model registry
├── modules/
│   ├── __init__.py
│   ├── model1.py           # SJM+BL model (fully implemented)
│   ├── model2.py           # Teammate stub
│   └── model3.py           # Teammate stub
├── components/
│   ├── comparison.py       # Comparison tab module
│   ├── charts.py           # Shared renderers: PNG loader, metrics table builder
│   └── layout.py           # Shared UI blocks: placeholder card, section wrapper
└── utils/
    ├── config.py           # Loads and validates app_config.yaml
    └── runner.py           # Spawns main.py subprocess, streams progress
```

---

## 3. Configuration Schema

`app_config.yaml`:

```yaml
models:
  - id: model1
    name: "Dynamic Factor Allocation (SJM+BL)"
    output_dir: "outputs/"
    module: "modules.model1"
    te_targets: [0.01, 0.02, 0.03, 0.04]
    run_command: ["python", "main.py"]
  - id: model2
    name: "Model 2"
    output_dir: "outputs/model2/"
    module: "modules.model2"
    te_targets: [0.03]
    run_command: null
  - id: model3
    name: "Model 3"
    output_dir: "outputs/model3/"
    module: "modules.model3"
    te_targets: [0.03]
    run_command: null
```

`app.py` reads this at startup and dynamically assembles the tab bar and mounts each module.

---

## 4. Teammate Module Contract

Each model module must export exactly two functions:

```python
from shiny import ui, module

def model_ui(id: str) -> ui.Tag:
    """Returns the tab's complete UI."""
    ...

def model_server(id: str, cfg: dict) -> None:
    """Handles server-side logic for the tab."""
    ...
```

`cfg` is the model's entry from `app_config.yaml` (dict with `id`, `name`, `output_dir`, `te_targets`, `run_command`).

Teammates add one file in `modules/` and one entry in `app_config.yaml`. No other files need to be touched.

---

## 5. Tab Structure

The app has 4 tabs:

| Tab | Content |
|-----|---------|
| **Comparison** | Overlaid cumulative returns chart + side-by-side metrics table for all 3 models |
| **SJM+BL** | Full model results for Model 1 |
| **Model 2** | Full model results (or placeholder) |
| **Model 3** | Full model results (or placeholder) |

---

## 6. Individual Model Tab Layout (Layout C)

Each model tab uses a **sticky sidebar + scrollable main area** layout:

**Left sidebar (fixed, ~180px wide):**
- Section anchor links: Metrics, Cumulative Returns, Portfolio Weights, Regime Plots
- TE target dropdown (values from `te_targets` in config, default 3%)
- **Re-run** button (disabled if `run_command` is null)

**Main area (scrollable):**

1. **Metrics section** — table from `results.csv`, filtered to selected TE row. Columns: Sharpe, Information Ratio, Max Drawdown, Volatility, Active Return vs Market, Active Return vs EW, Turnover.
2. **Cumulative Returns section** — displays `plots/cumulative_returns_te{N}.png` for selected TE.
3. **Portfolio Weights section** — displays `plots/portfolio_weights_te{N}.png`.
4. **Regime Plots section** — displays 5 PNGs: `plots/regime_{factor}_te{N}.png` for value, size, quality, growth, momentum.

Switching the TE dropdown swaps all PNGs and re-filters the metrics row — no re-computation.

---

## 7. Comparison Tab Layout (Layout A)

**Full-width overlaid cumulative returns chart** on top: `components/comparison.py` reads each model's `returns_te3.csv` (default TE=3%) and overlays cumulative return curves on a single dynamically-generated Matplotlib figure. A TE dropdown at the top reloads all three curves for the chosen TE. Models with missing `returns_te{N}.csv` are omitted from the chart with a legend note.

**Full-width metrics table** below: one row per model, same columns as the individual model metrics section. Missing models show `—` placeholders.

---

## 8. Data Flow & Output File Naming

### Pre-computed outputs (default)

The pipeline (`main.py`) must produce **per-TE plots** using suffixed filenames:

```
outputs/
├── results.csv                         # all TE rows in one file (already the case)
├── returns_te1.csv                     # daily return series for TE=1%
├── returns_te2.csv                     # daily return series for TE=2%
├── returns_te3.csv                     # daily return series for TE=3% ← default
├── returns_te4.csv                     # daily return series for TE=4%
└── plots/
    ├── cumulative_returns_te1.png
    ├── cumulative_returns_te2.png
    ├── cumulative_returns_te3.png       ← default shown
    ├── cumulative_returns_te4.png
    ├── portfolio_weights_te1.png
    ├── ...
    ├── regime_value_te3.png
    ├── regime_size_te3.png
    ├── regime_quality_te3.png
    ├── regime_growth_te3.png
    └── regime_momentum_te3.png
```

Each `returns_te{N}.csv` has columns: `date, portfolio, market, ew` (daily returns as decimals).

`backtest.py` will be updated to: (1) save `returns_te{N}.csv` per TE target, and (2) accept a `te_suffix` parameter in plot functions and append it to output filenames. `main.py` loops over all TE targets and calls both functions once per target.

### Teammate output contract

Each teammate's model must produce at minimum:

```
outputs/<model_id>/
├── results.csv               # same column schema: strategy, target_te, sharpe, ir, max_drawdown, ...
├── returns_te3.csv           # daily returns (date, portfolio, market, ew) — minimum TE=3%
└── plots/
    └── cumulative_returns_te3.png   # at minimum TE=3%; additional TEs optional
```

### Re-run flow

1. User clicks **Re-run** in the model tab.
2. `runner.py` spawns `run_command` as a subprocess.
3. A progress spinner replaces the content area; stdout is streamed to a log panel.
4. On exit code 0, the app reloads all plots and metrics from disk.
5. On non-zero exit, an error message is shown with the last 20 lines of stderr.

---

## 9. Placeholder State

When `output_dir` exists but required plot/CSV files are missing:

- The tab is **visible** in the nav bar.
- Content area shows a card: _"Results not yet available. Run the model to generate outputs."_
- A **Run** button appears (disabled if `run_command` is null, with tooltip _"No run command configured — ask your teammate to generate outputs manually"_).

---

## 10. Key Constraints

- App reads all data from disk at startup (or on reactive TE change). No in-memory pipeline state.
- Comparison tab's overlaid chart is generated dynamically by `components/comparison.py` using Matplotlib — it reads each model's `returns_te{N}.csv` and overlays cumulative return series.
- The app does not require the pipeline to be run before launch — placeholder state handles missing outputs gracefully.
- `run_command: null` in config means the Re-run button is disabled for that model.
