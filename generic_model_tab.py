import asyncio
import re
from pathlib import Path

import pandas as pd
import yaml as _yaml
from shiny import module, render, ui, reactive

from shiny_app.components.charts import img_tag, load_metrics_row
from shiny_app.components.layout import placeholder_card, section
from shiny_app.utils.runner import run_pipeline

_SIDEBAR_CSS = """
<style>
.model-sidebar { position: sticky; top: 1rem; }
.anchor-links a { display: block; padding: 4px 0; font-size: .875rem; color: #6c757d; text-decoration: none; }
.anchor-links a:hover { color: #0d6efd; }
</style>
"""

_STEP_RE = re.compile(r"\[(\d+)/(\d+)\](.*)$")

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

# Default parameters per model type
_MODEL_DEFAULTS = {
    "simple_hmm": {"n_states": 3, "features": "Returns_lag1", "random_state": 44},
    "hmm":        {"n_states": 5, "features": "Returns_lag1,Volatility_lag1,Volume_Change_lag1,HL_Range_lag1", "random_state": 42},
    "hsmm":       {"n_states": 5, "features": "Returns_lag1,Volatility_lag1,Volume_Change_lag1,HL_Range_lag1", "random_state": 43},
    "msgarch":    {"k_regimes": 3, "features": "", "random_state": 42},
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
    model_id = cfg["id"]
    defaults = _MODEL_DEFAULTS.get(model_id, {"n_states": 3, "features": "", "random_state": 42})
    description = cfg.get("description", "S&P 500 regime detection model.")
    has_run_cmd = cfg.get("run_command") is not None

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
        ui.div(ui.tags.b("\u2699 Parameters"), class_="mb-2"),
        ui.input_select("ticker", "Ticker", choices={
            "SPY": "S&P 500 (SPY)",
            "QQQ": "NASDAQ-100 (QQQ)",
            "IWM": "Russell 2000 (IWM)",
            "DIA": "Dow Jones (DIA)",
            "EFA": "Intl Developed (EFA)",
            "EEM": "Emerging Markets (EEM)",
            "TLT": "US Treasury 20Y (TLT)",
            "GLD": "Gold (GLD)",
            "VTI": "Total US Market (VTI)",
            "XLF": "Financials (XLF)",
            "XLK": "Technology (XLK)",
            "XLE": "Energy (XLE)",
            "XLV": "Healthcare (XLV)",
            "ARKK": "ARK Innovation (ARKK)",
        }, selected="SPY"),
        ui.input_text("train_start", "Train Start", value="2000-01-01"),
        ui.input_text("train_end", "Train End", value="2014-12-31"),
        ui.input_text("test_start", "Test Start", value="2015-01-01"),
        ui.input_text("test_end", "Test End", value="2026-03-25"),
        ui.input_numeric("n_states", "Num States/Regimes",
                         value=defaults.get("n_states", defaults.get("k_regimes", 3)),
                         min=2, max=10, step=1),
        ui.input_text("features", "Features (comma-sep)",
                      value=defaults.get("features", "")),
        ui.input_numeric("txn_cost", "Txn Cost (bps)", value=10, min=0, max=100, step=1),
        ui.hr(),
        ui.input_action_button(
            "rerun", "Run Model",
            class_="btn-primary btn-sm w-100 mt-2",
            disabled=not has_run_cmd,
        ),
        ui.output_ui("run_progress"),
        ui.output_ui("run_status"),
        ui.hr(),
        ui.div(
            ui.tags.b(cfg["name"]),
            ui.p(description, style="font-size:.8rem;color:#6c757d;margin-top:4px"),
        ),
        width=220,
        class_="model-sidebar",
    )

    if not output_dir.exists():
        main = ui.div(
            placeholder_card("Outputs not generated yet. Click 'Run Model' to generate."),
            style="padding:1rem",
        )
    else:
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
    run_cmd = cfg.get("run_command")
    model_id = cfg["id"]

    running = reactive.value(False)
    run_log = reactive.value("")
    progress_pct = reactive.value(0)
    step_msg = reactive.value("")
    refresh_trigger = reactive.value(0)  # incremented AFTER pipeline completes

    @reactive.effect
    @reactive.event(input.rerun)
    async def _launch_pipeline():
        if run_cmd is None or running():
            return
        running.set(True)
        run_log.set("")
        progress_pct.set(0)
        step_msg.set("Starting...")

        # Build temp config from UI inputs
        tmp_cfg_path = project_root / "outputs" / "tmp_user_config.yaml"
        user_cfg = {
            "data": {
                "ticker": input.ticker(),
                "train_start": input.train_start(),
                "train_end": input.train_end(),
                "test_start": input.test_start(),
                "test_end": input.test_end(),
                "transaction_cost": input.txn_cost() / 10000,
            },
        }
        # Add model-specific params
        if model_id == "msgarch":
            user_cfg["msgarch"] = {"k_regimes": int(input.n_states())}
        else:
            features_str = input.features()
            features_list = [f.strip() for f in features_str.split(",") if f.strip()]
            user_cfg[model_id] = {
                "n_states": int(input.n_states()),
                "features": features_list,
            }

        tmp_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_cfg_path, "w") as f:
            _yaml.dump(user_cfg, f)

        # Build command with config
        cmd = run_cmd + ["--config", str(tmp_cfg_path)]
        proc = run_pipeline(cmd, project_root)

        log_lines = []
        try:
            while True:
                line = proc.stdout.readline()
                if not line and proc.poll() is not None:
                    break
                if line:
                    log_lines.append(line.rstrip())
                    run_log.set("\n".join(log_lines[-15:]))
                    m = _STEP_RE.search(line)
                    if m:
                        cur, total = int(m.group(1)), int(m.group(2))
                        progress_pct.set(int(cur / total * 100))
                        step_msg.set(m.group(3).strip())
                await asyncio.sleep(0.05)
        finally:
            rc = proc.wait()
            if rc == 0:
                step_msg.set("Done!")
                progress_pct.set(100)
                refresh_trigger.set(refresh_trigger() + 1)  # trigger plot/metrics refresh
            else:
                step_msg.set(f"Failed (exit {rc})")
            running.set(False)

    @render.ui
    def run_progress():
        if not running() and progress_pct() == 0:
            return ui.div()
        pct = progress_pct()
        return ui.div(
            ui.p(step_msg(), style="font-size:.75rem;margin:4px 0"),
            ui.HTML(
                f"<div class='progress' style='height:6px'>"
                f"<div class='progress-bar' style='width:{pct}%'></div></div>"
            ),
        )

    @render.ui
    def run_status():
        log = run_log()
        if not log:
            return ui.div()
        return ui.pre(log, style="font-size:.65rem;max-height:200px;overflow-y:auto;background:#f8f9fa;padding:6px;border-radius:4px")

    @render.ui
    def metrics_tbl():
        refresh_trigger()  # re-render when pipeline completes
        return ui.HTML(_metrics_html(output_dir))

    @render.ui
    def returns_img():
        refresh_trigger()
        return img_tag(output_dir / "plots/cumulative_returns.png", alt="Cumulative returns")

    @render.ui
    def timeline_img():
        refresh_trigger()
        return img_tag(output_dir / "plots/regime_timeline.png", alt="Regime timeline")

    @render.ui
    def chars_img():
        refresh_trigger()
        return img_tag(output_dir / "plots/regime_characteristics.png", alt="Regime characteristics")

    @render.ui
    def transition_img():
        refresh_trigger()
        return img_tag(output_dir / "plots/transition_matrix.png", alt="Transition matrix")
