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
shiny run shiny_app/app.py --reload
```

Open `http://localhost:8000`. The dashboard has:
- **Model tabs** — metrics table, cumulative returns, portfolio weights, regime plots
- **Comparison tab** — overlaid returns and side-by-side metrics across all models
- **TE target selector** — switch between 1%, 2%, 3%, 4% tracking error targets
- **Run Model button** — executes the pipeline directly from the browser

---

## Adding Your Model (Teammates)

Each team member contributes a model tab. Here's how:

### 1. Create your module

Create `shiny_app/modules/model2.py` (or `model3.py`) with these two functions:

```python
from shiny import ui, module, render

@module.ui
def model_tab_ui():
    return ui.div(
        # your tab layout here
    )

@module.server
def model_tab_server(input, output, session, model_cfg):
    # model_cfg contains: id, name, output_dir, te_targets, run_command
    # load your outputs from model_cfg["output_dir"]
    pass
```

### 2. Register in `app_config.yaml`

```yaml
models:
  - id: model2
    name: "Your Model Name"
    output_dir: "outputs/model2"     # where your outputs live
    module: "shiny_app.modules.model2"
    te_targets: [0.03]
    run_command: ["python", "your_main.py"]  # or null
```

### 3. Save your outputs

Your pipeline should write to `outputs/model2/`:
- `results.csv` — one row per TE target with columns: `te_target`, `sharpe`, `ir_vs_market`, `max_drawdown`, `volatility`, `active_return`, `turnover`
- `returns_te0.03.csv` — columns: `date`, `portfolio`, `market`, `ew`
- `plots/cumulative_returns_te0.03.png`
- `plots/portfolio_weights_te0.03.png`

The dashboard auto-loads anything matching those filenames.

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
│   ├── app_config.yaml          # Model registry
│   ├── modules/
│   │   ├── model1.py            # SJM+BL model tab
│   │   ├── model2.py            # Teammate model (add yours here)
│   │   └── model3.py            # Teammate model (add yours here)
│   └── components/
│       ├── comparison.py        # Cross-model comparison tab
│       └── charts.py            # Chart rendering utilities
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
