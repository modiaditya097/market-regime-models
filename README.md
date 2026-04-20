# Dynamic Factor Allocation with Regime-Switching Signals

Research implementation of Shu & Mulvey (2025). Detects bull/bear regimes for individual equity factors using Sparse Jump Models (SJM), then constructs optimal portfolios via Black-Litterman optimization with configurable tracking error targets.

---

## Setup

**Requirements:** Python 3.10+

```bash
pip install -r requirements.txt
pip install jump-models-author/   # install local SJM package
```

---

## Running the Pipeline

```bash
# First run (downloads and caches data)
python main.py --refresh

# Subsequent runs (uses cached data)
python main.py
```

This runs 5 steps in sequence:

| Step | Module | What it does |
|------|--------|-------------|
| 1 | `src/data.py` | Downloads Ken French factors + FRED macro data, caches as Parquet |
| 2 | `src/features.py` | Computes 17 technical features per factor (EWMAs, RSI, MACD, beta) |
| 3 | `src/regime.py` | Fits SJM per factor using expanding monthly windows, labels bull/bear |
| 4 | `src/portfolio.py` | Runs Black-Litterman optimization targeting each TE level |
| 5 | `src/backtest.py` | Computes metrics, generates plots, saves CSVs |

Outputs are written to `outputs/` (plots, `results.csv`, `returns_te*.csv`).

---

## Viewing the Dashboard

```bash
python -m shiny run shiny_app/app.py --reload
```

Open `http://localhost:8000`.

### Tabs

| Tab | Module | Description |
|-----|--------|-------------|
| Comparison | `components/comparison.py` | Overlaid cumulative returns + side-by-side metrics for all models |
| Dynamic Factor Allocation (SJM+BL) | `modules/model1.py` | Metrics, returns, portfolio weights, per-factor regime plots; live Run button with progress |
| SC-HMM | `modules/model3.py` | Full/in-sample/OOS metrics, drawdown, rolling Sharpe, regime timeline, transition matrix, stress/bull tables, macro signal panel |
| Simple-HMM (S&P 500) | `modules/simple_hmm.py` | 3-state HMM on S&P 500 returns |
| HMM (S&P 500) | `modules/hmm.py` | 5-state multi-feature HMM |
| MS-GARCH (S&P 500) | `modules/msgarch.py` | Markov-Switching GARCH volatility regimes |
| HSMM (S&P 500) | `modules/hsmm.py` | 5-state Hidden Semi-Markov Model with duration modeling |

### Dashboard Architecture

**Framework:** Posit Shiny for Python (v0.9+) — reactive UI, Bootstrap dark theme, no JS required.

**Model registration** is config-driven. `app.py` reads `shiny_app/app_config.yaml` and loads each module via `importlib` at startup. Adding a model requires only a YAML entry and a Python module — no changes to `app.py`.

**Data flow:**

```
Raw data (FRED / Ken French / yfinance)
    ↓  src/ pipeline (main.py)
outputs/<model>/   ← CSVs + PNGs written here
    ↓  shiny_app/components/charts.py
Dashboard tab      ← reads files on demand, re-renders on TE selector change
```

**Sidebar controls** (per model tab): anchor navigation, TE-target dropdown, optional Run button, status indicator.

---

## Adding a New Model

### 1. Create your module

```python
# shiny_app/modules/my_model.py
from shiny import module, ui, render
from pathlib import Path
import base64, pandas as pd

@module.ui
def model_tab_ui(cfg: dict):
    return ui.page_fluid(
        ui.output_table("metrics"),
        ui.output_ui("cumulative_plot"),
    )

@module.server
def model_tab_server(input, output, session, cfg: dict, project_root):
    output_dir = Path(cfg["output_dir"])

    @render.table
    def metrics():
        df = pd.read_csv(output_dir / "results.csv")
        return df

    @render.ui
    def cumulative_plot():
        path = output_dir / "plots/cumulative_returns.png"
        encoded = base64.b64encode(path.read_bytes()).decode()
        return ui.img(src=f"data:image/png;base64,{encoded}", style="width:100%")
```

### 2. Register in `app_config.yaml`

```yaml
models:
  - id: my_model
    name: "My Model Name"
    output_dir: "outputs/my_model"
    module: "shiny_app.modules.my_model"
    te_targets: [0.03]
    run_command: null        # or ["python", "run_my_model.py"]
```

### 3. Write outputs to `outputs/my_model/`

Expected files (adjust to what your tab renders):
- `results.csv` — performance metrics
- `returns.csv` — columns: `date`, `portfolio`, `market`
- `plots/cumulative_returns.png`
- `plots/regime_timeline.png`

---

## Configuration

**`config.yaml`** — controls the full pipeline:

```yaml
data:
  start_date: "2000-01-01"
  end_date: "2026-01-30"
  refresh: false              # set true to re-download

training:
  min_train_years: 8
  max_train_years: 12
  refit_freq: "M"             # monthly refit

sjm:
  n_components: 2             # binary regime (bull/bear)
  jump_penalty: 50.0          # penalizes regime switches
  max_feats: 9.5              # feature sparsity (lasso)

black_litterman:
  risk_aversion: 2.5
  target_tracking_error: 0.03
  transaction_cost_bps: 5
```

**`shiny_app/app_config.yaml`** — controls which models appear in the dashboard and where their outputs are.

---

## Project Structure

```
├── main.py                      # Pipeline entry point
├── config.yaml                  # Pipeline configuration
├── requirements.txt
│
├── src/
│   ├── data.py                  # Data download + caching
│   ├── features.py              # Feature engineering
│   ├── regime.py                # SJM regime detection
│   ├── portfolio.py             # Black-Litterman optimization
│   ├── backtest.py              # Performance evaluation
│   └── utils.py                 # Constants + metrics
│
├── shiny_app/
│   ├── app.py                   # Dashboard entry point
│   ├── app_config.yaml          # Model registry (config-driven tab loading)
│   ├── modules/
│   │   ├── model1.py            # Dynamic Factor Allocation (SJM+BL)
│   │   ├── model2.py            # Placeholder
│   │   ├── model3.py            # SC-HMM
│   │   ├── simple_hmm.py        # Simple 3-state HMM (S&P 500)
│   │   ├── hmm.py               # 5-state multi-feature HMM (S&P 500)
│   │   ├── msgarch.py           # MS-GARCH (S&P 500)
│   │   └── hsmm.py              # Hidden Semi-Markov Model (S&P 500)
│   ├── components/
│   │   ├── comparison.py        # Cross-model comparison tab
│   │   ├── charts.py            # Chart rendering utilities
│   │   └── layout.py            # Reusable UI helpers
│   └── utils/
│       ├── config.py            # YAML loader + path resolution
│       └── runner.py            # Subprocess pipeline runner
│
├── tests/                       # pytest test suite
└── outputs/                     # Generated results (gitignored)
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Factors & Assets

The universe consists of **6 assets**: `market`, `value`, `size`, `quality`, `growth`, `momentum` (Ken French 5-factor + momentum).

Each factor gets its own regime signal. The portfolio allocates across all 6 based on current regime forecasts.
