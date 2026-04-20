# App Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Polish the dashboard for a professor presentation — cleanup dead code, unify UI across model tabs, fix the Comparison tab, and add a live config panel to Model 1.

**Architecture:** Four independent changes applied in sequence: (1) delete orphaned files + fix deprecation, (2) create a shared `generic_model_tab` module replacing three near-identical modules, (3) patch the Comparison tab, (4) extend Model 1's sidebar with parameter inputs that write a temp config on run.

**Tech Stack:** Python 3.10+, Posit Shiny for Python ≥0.9, PyYAML, pandas, matplotlib

---

## File Map

| File | Action |
|---|---|
| `shiny_app/app.py` | Modify — fix `page_navbar` deprecation |
| `shiny_app/app_config.yaml` | Modify — remove `simple_hmm`, redirect `hmm`/`hsmm`/`msgarch` to generic module, add `description` field |
| `shiny_app/modules/generic_model_tab.py` | Create — shared module for HMM / HSMM / MS-GARCH |
| `shiny_app/modules/model1.py` | Modify — add config inputs + temp-config run logic |
| `shiny_app/components/comparison.py` | Modify — add TE note, add shared Market benchmark line |
| `shiny_app/modules/simple_hmm.py` | Delete |
| `shiny_app/modules/model2.py` | Delete |
| `shiny_app/modules/hmm.py` | Delete |
| `shiny_app/modules/hsmm.py` | Delete |
| `shiny_app/modules/msgarch.py` | Delete |

---

## Task 1: Cleanup — delete dead files and fix deprecation warning

**Files:**
- Delete: `shiny_app/modules/simple_hmm.py`, `shiny_app/modules/model2.py`, `shiny_app/modules/hmm.py`, `shiny_app/modules/hsmm.py`, `shiny_app/modules/msgarch.py`
- Modify: `shiny_app/app.py`

- [ ] **Step 1: Delete the five dead module files**

```bash
cd "C:\Users\adity\Desktop\Final Project"
git rm shiny_app/modules/simple_hmm.py shiny_app/modules/model2.py shiny_app/modules/hmm.py shiny_app/modules/hsmm.py shiny_app/modules/msgarch.py
```

- [ ] **Step 2: Fix page_navbar deprecation in `shiny_app/app.py`**

Current code at lines 32–38:
```python
    app_ui = ui.page_navbar(
        *nav_panels,
        title="Dynamic Factor Allocation Dashboard",
        id="main_nav",
        bg="#1a1a2e",
        inverse=True,
    )
```

Replace with:
```python
    app_ui = ui.page_navbar(
        *nav_panels,
        title="Dynamic Factor Allocation Dashboard",
        id="main_nav",
        navbar_options=ui.navbar_options(bg="#1a1a2e", inverse=True),
    )
```

- [ ] **Step 3: Verify the app builds without the deprecation warning**

```bash
cd "C:\Users\adity\Desktop\Final Project"
python -c "from shiny_app.app import build_app; build_app(); print('OK')" 2>&1 | grep -v "^$"
```

Expected: output contains `OK` and does NOT contain `ShinyDeprecationWarning`.

Note: if `ui.navbar_options` is not available in the installed Shiny version, try `ui.NavbarOptions` instead. Check available names with: `python -c "from shiny import ui; print([x for x in dir(ui) if 'navbar' in x.lower()])`

- [ ] **Step 4: Commit**

```bash
cd "C:\Users\adity\Desktop\Final Project"
git add shiny_app/app.py
git commit -m "chore: delete dead modules, fix page_navbar deprecation"
```

---

## Task 2: Create generic_model_tab module

**Files:**
- Create: `shiny_app/modules/generic_model_tab.py`

- [ ] **Step 1: Create `shiny_app/modules/generic_model_tab.py`**

```python
"""Shared tab module for HMM, HSMM, and MS-GARCH models.

All three have the same output structure:
  outputs/<model>/results.csv
  outputs/<model>/returns.csv
  outputs/<model>/plots/cumulative_returns.png
  outputs/<model>/plots/regime_timeline.png
  outputs/<model>/plots/regime_characteristics.png
  outputs/<model>/plots/transition_matrix.png
"""

from pathlib import Path

import pandas as pd
from shiny import module, render, ui

from shiny_app.components.charts import img_tag, load_metrics_row
from shiny_app.components.layout import placeholder_card, section

_SIDEBAR_CSS = """
<style>
.model-sidebar { position: sticky; top: 1rem; }
.anchor-links a { display: block; padding: 4px 0; font-size: .875rem; color: #6c757d; text-decoration: none; }
.anchor-links a:hover { color: #0d6efd; }
</style>
"""

_METRIC_COLORS = {
    "Sharpe":        lambda v: "#198754" if v >= 0.5 else ("#dc3545" if v < 0 else "inherit"),
    "IR vs Mkt":     lambda v: "#198754" if v > 0 else "#dc3545",
    "Max DD":        lambda v: "#dc3545",
    "Volatility":    lambda v: "inherit",
    "Active Ret":    lambda v: "#198754" if v > 0 else "#dc3545",
    "Turnover":      lambda v: "inherit",
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


def _metrics_html(output_dir: Path) -> str:
    df = load_metrics_row(output_dir, te_pct=3)
    if df.empty:
        return "<p class='text-muted'>No results available.</p>"

    row = df.iloc[0]
    rows_html = ""
    for raw_col, display_col in _DISPLAY.items():
        if raw_col not in row.index:
            continue
        val = row[raw_col]
        try:
            color = _METRIC_COLORS[display_col](float(val))
            formatted = _format_value(display_col, float(val))
        except (TypeError, ValueError):
            color = "inherit"
            formatted = str(val)
        rows_html += (
            f"<tr>"
            f"<td style='padding:6px 12px;font-weight:600'>{display_col}</td>"
            f"<td style='padding:6px 12px;color:{color};font-weight:600'>{formatted}</td>"
            f"</tr>"
        )

    return (
        "<table class='table table-sm' style='max-width:400px'>"
        "<thead><tr><th>Metric</th><th>Value</th></tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table>"
    )


@module.ui
def model_tab_ui(cfg: dict):
    output_dir = Path(cfg["output_dir"])
    if not output_dir.exists():
        return ui.page_fluid(
            ui.h2(cfg["name"], class_="text-primary"),
            ui.hr(),
            placeholder_card("Outputs for this model have not been generated yet."),
        )

    description = cfg.get("description", "S&P 500 regime detection model.")

    sidebar = ui.sidebar(
        ui.HTML(_SIDEBAR_CSS),
        ui.div(
            ui.tags.b("Sections"),
            ui.div(
                ui.tags.a("Metrics",              href="#metrics"),
                ui.tags.a("Cumulative Returns",   href="#returns"),
                ui.tags.a("Regime Timeline",      href="#timeline"),
                ui.tags.a("Characteristics",      href="#chars"),
                ui.tags.a("Transition Matrix",    href="#transition"),
                class_="anchor-links",
            ),
            class_="mb-3",
        ),
        ui.hr(),
        ui.div(
            ui.tags.b(cfg["name"]),
            ui.p(description, style="font-size:.8rem;color:#6c757d;margin-top:4px"),
        ),
        width=200,
        class_="model-sidebar",
    )

    main = ui.div(
        section("Performance Metrics",     "metrics",    ui.output_ui("metrics_tbl")),
        section("Cumulative Returns",      "returns",    ui.output_ui("returns_img")),
        section("Regime Timeline",         "timeline",   ui.output_ui("timeline_img")),
        section("Regime Characteristics",  "chars",      ui.output_ui("chars_img")),
        section("Transition Matrix",       "transition", ui.output_ui("transition_img")),
        style="padding:1rem",
    )

    return ui.layout_sidebar(sidebar, main)


@module.server
def model_tab_server(input, output, session, cfg: dict, project_root):
    output_dir = Path(cfg["output_dir"])

    @render.ui
    def metrics_tbl():
        return ui.HTML(_metrics_html(output_dir))

    @render.ui
    def returns_img():
        return img_tag(output_dir / "plots/cumulative_returns.png", alt="Cumulative returns")

    @render.ui
    def timeline_img():
        return img_tag(output_dir / "plots/regime_timeline.png", alt="Regime timeline")

    @render.ui
    def chars_img():
        return img_tag(output_dir / "plots/regime_characteristics.png", alt="Regime characteristics")

    @render.ui
    def transition_img():
        return img_tag(output_dir / "plots/transition_matrix.png", alt="Transition matrix")
```

- [ ] **Step 2: Verify the module imports cleanly**

```bash
cd "C:\Users\adity\Desktop\Final Project"
python -c "from shiny_app.modules.generic_model_tab import model_tab_ui, model_tab_server; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Update `shiny_app/app_config.yaml`**

Replace the entire file content with:

```yaml
models:
  - id: model1
    name: "Dynamic Factor Allocation (SJM+BL)"
    output_dir: "outputs"
    module: "shiny_app.modules.model1"
    te_targets: [0.01, 0.02, 0.03, 0.04]
    run_command: ["python", "main.py"]

  - id: model3
    name: "SC-HMM"
    output_dir: "outputs/model3"
    module: "shiny_app.modules.model3"
    te_targets: [0.03]
    run_command: null

  - id: hmm
    name: "HMM (S&P 500)"
    description: "5-state multi-feature Hidden Markov Model. Training: 2013–2022. Test: 2023–2024."
    output_dir: "outputs/hmm"
    module: "shiny_app.modules.generic_model_tab"
    te_targets: [0.03]
    run_command: null

  - id: hsmm
    name: "HSMM (S&P 500)"
    description: "5-state Hidden Semi-Markov Model with explicit duration modeling. Training: 2013–2022. Test: 2023–2024."
    output_dir: "outputs/hsmm"
    module: "shiny_app.modules.generic_model_tab"
    te_targets: [0.03]
    run_command: null

  - id: msgarch
    name: "MS-GARCH (S&P 500)"
    description: "Markov-Switching GARCH model with volatility-based regime switching. Training: 2013–2022. Test: 2023–2024."
    output_dir: "outputs/msgarch"
    module: "shiny_app.modules.generic_model_tab"
    te_targets: [0.03]
    run_command: null
```

- [ ] **Step 4: Verify the full app builds with all 5 tabs**

```bash
cd "C:\Users\adity\Desktop\Final Project"
python -c "
from shiny_app.app import build_app
import re, urllib.request
app = build_app()
html = app.ui['html']
tabs = list(dict.fromkeys(re.findall(r'data-value=\"([^\"<]+)\"', html)))
top = [t for t in tabs if 'Period' not in t and 'Sample' not in t and 'Stress' not in t and 'Bull' not in t]
print('Tabs:', top)
assert len(top) == 6, f'Expected 6 tabs (Comparison + 5 models), got {len(top)}'
print('OK')
" 2>&1 | grep -v ShinyDeprecation | grep -v navbar_options
```

Expected output:
```
Tabs: ['Comparison', 'Dynamic Factor Allocation (SJM+BL)', 'SC-HMM', 'HMM (S&P 500)', 'HSMM (S&P 500)', 'MS-GARCH (S&P 500)']
OK
```

- [ ] **Step 5: Commit**

```bash
cd "C:\Users\adity\Desktop\Final Project"
git add shiny_app/modules/generic_model_tab.py shiny_app/app_config.yaml
git commit -m "feat: add generic_model_tab shared module for HMM/HSMM/MS-GARCH"
```

---

## Task 3: Fix Comparison tab

**Files:**
- Modify: `shiny_app/components/comparison.py`

- [ ] **Step 1: Add TE note and shared Market line to `shiny_app/components/comparison.py`**

Replace the entire file with:

```python
"""Comparison tab — overlaid cumulative returns chart + side-by-side metrics table."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from shiny import module, ui, render

from shiny_app.components.charts import load_returns_df, load_metrics_row, _DISPLAY_COLS

_COLORS = ["#7c3aed", "#10b981", "#f97316", "#e11d48", "#0ea5e9", "#eab308", "#8b5cf6"]


@module.ui
def comparison_ui(all_cfg: dict):
    model1_cfg = all_cfg["models"][0]
    te_choices = {
        str(int(t * 100)): f"{int(t * 100)}%"
        for t in model1_cfg.get("te_targets", [0.03])
    }
    default_te = "3" if "3" in te_choices else list(te_choices.keys())[0]

    return ui.div(
        ui.div(
            ui.input_select("te", "TE Target", choices=te_choices, selected=default_te),
            ui.p(
                ui.tags.em("TE target applies to Model 1 only. "
                           "Other models display their single pre-computed series."),
                style="font-size:.8rem;color:#6c757d;margin-top:4px",
            ),
            style="max-width:400px;margin-bottom:1rem",
        ),
        ui.h5("Cumulative Returns — All Models"),
        ui.output_plot("overlay_chart", height="400px"),
        ui.hr(),
        ui.h5("Performance Metrics"),
        ui.output_table("metrics_tbl"),
        style="padding:1.5rem",
    )


@module.server
def comparison_server(input, output, session, all_cfg: dict):
    @render.plot(alt="Cumulative returns comparison")
    def overlay_chart():
        te = int(input.te())
        fig, ax = plt.subplots(figsize=(12, 5))
        plotted = False
        market_series = None

        for i, model_cfg in enumerate(all_cfg["models"]):
            df = load_returns_df(Path(model_cfg["output_dir"]), te_pct=te)
            if df is None:
                continue
            cum = (1 + df["portfolio"]).cumprod() - 1
            ax.plot(cum.index, cum * 100, label=model_cfg["name"],
                    color=_COLORS[i % len(_COLORS)], linewidth=1.5)
            plotted = True
            if market_series is None and "market" in df.columns:
                market_series = df["market"]

        if not plotted:
            ax.text(0.5, 0.5, "No model outputs available",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=12, color="grey")
        else:
            if market_series is not None:
                cum_mkt = (1 + market_series).cumprod() - 1
                ax.plot(cum_mkt.index, cum_mkt * 100,
                        label="Market (S&P 500)", color="#888888",
                        linewidth=1.2, linestyle="--")
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_ylabel("Cumulative Excess Return (%)")
            ax.legend()
            ax.grid(True, alpha=0.3)

        ax.set_title(f"Cumulative Returns Comparison (TE={te}%)")
        plt.tight_layout()
        return fig

    @render.table
    def metrics_tbl():
        te = int(input.te())
        _PLACEHOLDER = {k: "—" for k in _DISPLAY_COLS.keys()}
        rows = []
        for model_cfg in all_cfg["models"]:
            df = load_metrics_row(Path(model_cfg["output_dir"]), te_pct=te)
            if df.empty:
                row = dict(_PLACEHOLDER)
                row["strategy"] = model_cfg["name"]
            else:
                row = df.iloc[0].to_dict()
                row["strategy"] = model_cfg["name"]
            rows.append(row)
        return pd.DataFrame(rows).rename(columns=_DISPLAY_COLS)
```

- [ ] **Step 2: Verify comparison module imports cleanly**

```bash
cd "C:\Users\adity\Desktop\Final Project"
python -c "from shiny_app.components.comparison import comparison_ui, comparison_server; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
cd "C:\Users\adity\Desktop\Final Project"
git add shiny_app/components/comparison.py
git commit -m "fix(comparison): add TE note and shared market benchmark line"
```

---

## Task 4: Model 1 config panel

**Files:**
- Modify: `shiny_app/modules/model1.py`

- [ ] **Step 1: Add a helper to load defaults from config.yaml**

At the top of `shiny_app/modules/model1.py`, after the existing imports, add:

```python
import yaml as _yaml

def _load_param_defaults(project_root: Path) -> dict:
    cfg_path = project_root / "config.yaml"
    if not cfg_path.exists():
        return {"jump_penalty": 50.0, "max_feats": 9.5, "risk_aversion": 2.5, "txn_cost": 5}
    with open(cfg_path) as f:
        cfg = _yaml.safe_load(f)
    return {
        "jump_penalty":   cfg["sjm"]["jump_penalty"],
        "max_feats":      cfg["sjm"]["max_feats"],
        "risk_aversion":  cfg["black_litterman"]["risk_aversion"],
        "txn_cost":       cfg["black_litterman"]["transaction_cost_bps"],
    }
```

- [ ] **Step 2: Add config inputs to the sidebar in `model_tab_ui`**

In `model_tab_ui`, replace the existing `sidebar = ui.sidebar(...)` block (lines 29–53) with:

```python
    defaults = _load_param_defaults(Path(__file__).parent.parent.parent)

    sidebar = ui.sidebar(
        ui.HTML(_SIDEBAR_CSS),
        ui.div(
            ui.tags.b("Sections"),
            ui.div(
                ui.tags.a("Metrics",            href="#metrics"),
                ui.tags.a("Cumulative Returns", href="#returns"),
                ui.tags.a("Portfolio Weights",  href="#weights"),
                ui.tags.a("Regime Plots",       href="#regimes"),
                class_="anchor-links",
            ),
            class_="mb-3",
        ),
        ui.hr(),
        ui.input_select("te", "TE Target", choices=te_choices, selected=default_te),
        ui.hr(),
        ui.div(ui.tags.b("⚙ Parameters"), class_="mb-2"),
        ui.input_numeric("jump_penalty", "Jump Penalty (λ)",
                         value=defaults["jump_penalty"], step=1, min=1),
        ui.input_numeric("max_feats", "Feature Sparsity (κ²)",
                         value=defaults["max_feats"], step=0.5, min=0.5),
        ui.input_numeric("risk_aversion", "Risk Aversion (δ)",
                         value=defaults["risk_aversion"], step=0.1, min=0.1),
        ui.input_numeric("txn_cost", "Txn Cost (bps)",
                         value=defaults["txn_cost"], step=1, min=0),
        ui.hr(),
        ui.input_action_button(
            "rerun", "▶ Run Model",
            class_="btn-primary btn-sm w-100 mt-2",
            disabled=cfg.get("run_command") is None,
        ),
        ui.output_ui("run_progress"),
        ui.output_ui("run_status"),
        width=220,
        class_="model-sidebar",
    )
```

- [ ] **Step 3: Update `_launch_pipeline` to write a temp config and pass --config**

In `model_tab_server`, replace the `_launch_pipeline` async function with:

```python
    @reactive.effect
    @reactive.event(input.rerun)
    async def _launch_pipeline():
        if run_cmd is None or running():
            return
        running.set(True)
        run_log.set("")
        progress_pct.set(0)
        step_msg.set("Starting...")

        tmp_cfg_path = project_root / "outputs" / "tmp_config.yaml"
        try:
            # Build temp config with user's parameter overrides
            base_cfg_path = project_root / "config.yaml"
            with open(base_cfg_path) as f:
                cfg_data = _yaml.safe_load(f)
            cfg_data["sjm"]["jump_penalty"]                        = float(input.jump_penalty())
            cfg_data["sjm"]["max_feats"]                           = float(input.max_feats())
            cfg_data["black_litterman"]["risk_aversion"]           = float(input.risk_aversion())
            cfg_data["black_litterman"]["transaction_cost_bps"]    = int(input.txn_cost())
            with open(tmp_cfg_path, "w") as f:
                _yaml.dump(cfg_data, f, default_flow_style=False)

            actual_cmd = run_cmd + ["--config", str(tmp_cfg_path)]
            proc = await asyncio.create_subprocess_exec(
                *actual_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(project_root),
            )
            lines = []
            async for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace").rstrip()
                lines.append(line)
                m = _STEP_RE.search(line)
                if m:
                    current, total, desc = int(m.group(1)), int(m.group(2)), m.group(3).strip()
                    progress_pct.set(int(current / total * 100))
                    step_msg.set(f"[{current}/{total}] {desc}")
            await proc.wait()
            if proc.returncode == 0:
                progress_pct.set(100)
                step_msg.set("Complete")
                run_log.set("Pipeline finished successfully.")
            else:
                step_msg.set("Failed")
                run_log.set("\n".join(lines[-20:]))
        except Exception as exc:
            step_msg.set("Error")
            run_log.set(str(exc))
        finally:
            running.set(False)
            if tmp_cfg_path.exists():
                tmp_cfg_path.unlink()
```

- [ ] **Step 4: Verify model1 module imports cleanly**

```bash
cd "C:\Users\adity\Desktop\Final Project"
python -c "from shiny_app.modules.model1 import model_tab_ui, model_tab_server; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Verify full app builds with all changes**

```bash
cd "C:\Users\adity\Desktop\Final Project"
python -c "from shiny_app.app import build_app; build_app(); print('OK')" 2>&1 | grep -v ShinyDeprecation | grep -v navbar_options
```

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
cd "C:\Users\adity\Desktop\Final Project"
git add shiny_app/modules/model1.py
git commit -m "feat(model1): add parameter config panel with temp-config run logic"
```

---

## Task 5: Push and restart

- [ ] **Step 1: Push to remote**

```bash
cd "C:\Users\adity\Desktop\Final Project"
git push origin master
```

- [ ] **Step 2: Kill existing app processes**

```bash
tasklist | grep python
```

Then kill all listed PIDs:
```bash
taskkill //F //PID <pid1> //PID <pid2>
```

- [ ] **Step 3: Start fresh**

```bash
cd "C:\Users\adity\Desktop\Final Project"
python -m shiny run shiny_app/app.py --reload
```

- [ ] **Step 4: Verify in browser**

Open http://127.0.0.1:8000 and confirm:
- 6 tabs visible: Comparison, Dynamic Factor Allocation (SJM+BL), SC-HMM, HMM, HSMM, MS-GARCH
- HMM/HSMM/MS-GARCH tabs have a sidebar with anchor links and a model overview card
- Metrics tables are color-coded (Sharpe in green/red, Max DD in red)
- Comparison tab shows the TE note below the dropdown
- Comparison chart shows a dashed grey Market line
- Model 1 sidebar has 4 parameter inputs above the Run button
- No deprecation warning in the server startup log
