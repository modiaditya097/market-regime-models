"""SJM + Black-Litterman model tab (Layout C: sticky sidebar + scrollable sections)."""

import asyncio
import re
from pathlib import Path

import pandas as pd
import yaml as _yaml

from shiny import module, ui, render, reactive

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

_FACTORS = ["value", "size", "quality", "growth", "momentum"]
_STEP_RE = re.compile(r"\[(\d+)/(\d+)\](.*)$")

_SIDEBAR_CSS = """
<style>
.model-sidebar { position: sticky; top: 1rem; }
.anchor-links a { display: block; padding: 4px 0; font-size: .875rem; color: #6c757d; text-decoration: none; }
.anchor-links a:hover { color: #0d6efd; }
</style>
"""


def _load_param_defaults(project_root: Path) -> dict:
    cfg_path = project_root / "config.yaml"
    if not cfg_path.exists():
        return {"jump_penalty": 50.0, "max_feats": 9.5, "risk_aversion": 2.5, "txn_cost": 5}
    with open(cfg_path) as f:
        cfg = _yaml.safe_load(f)
    return {
        "jump_penalty":  cfg["sjm"]["jump_penalty"],
        "max_feats":     cfg["sjm"]["max_feats"],
        "risk_aversion": cfg["black_litterman"]["risk_aversion"],
        "txn_cost":      cfg["black_litterman"]["transaction_cost_bps"],
    }


@module.ui
def model_tab_ui(cfg: dict):
    defaults = _load_param_defaults(Path(__file__).parent.parent.parent)
    te_choices = {str(int(t * 100)): f"{int(t * 100)}%" for t in cfg.get("te_targets", [0.03])}
    default_te = "3" if "3" in te_choices else list(te_choices.keys())[0]

    sidebar = ui.sidebar(
        ui.HTML(_SIDEBAR_CSS),
        ui.div(
            ui.tags.b("Sections"),
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
            class_="mb-3",
        ),
        ui.hr(),
        ui.input_select("te", "TE Target", choices=te_choices, selected=default_te),
        ui.hr(),
        ui.div(ui.tags.b("\u2699 Parameters"), class_="mb-2"),
        ui.input_numeric("jump_penalty", "Jump Penalty (\u03bb)",
                         value=defaults["jump_penalty"], step=1, min=1),
        ui.input_numeric("max_feats", "Feature Sparsity (\u03ba\u00b2)",
                         value=defaults["max_feats"], step=0.5, min=0.5),
        ui.input_numeric("risk_aversion", "Risk Aversion (\u03b4)",
                         value=defaults["risk_aversion"], step=0.1, min=0.1),
        ui.input_numeric("txn_cost", "Txn Cost (bps)",
                         value=defaults["txn_cost"], step=1, min=0),
        ui.input_action_button(
            "rerun", "Run Model",
            class_="btn-primary btn-sm w-100 mt-2",
            disabled=cfg.get("run_command") is None,
        ),
        ui.output_ui("run_progress"),
        ui.output_ui("run_status"),
        width=200,
        class_="model-sidebar",
    )

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

    return ui.layout_sidebar(sidebar, main)


@module.server
def model_tab_server(input, output, session, cfg: dict, project_root: Path):
    output_dir = Path(cfg["output_dir"])
    run_cmd = cfg.get("run_command")

    running = reactive.value(False)
    run_log = reactive.value("")
    progress_pct = reactive.value(0)
    step_msg = reactive.value("")

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
        base_cfg_path = project_root / "config.yaml"
        with open(base_cfg_path) as f:
            cfg_data = _yaml.safe_load(f)
        cfg_data["sjm"]["jump_penalty"]                     = float(input.jump_penalty())
        cfg_data["sjm"]["max_feats"]                        = float(input.max_feats())
        cfg_data["black_litterman"]["risk_aversion"]        = float(input.risk_aversion())
        cfg_data["black_litterman"]["transaction_cost_bps"] = int(input.txn_cost())
        with open(tmp_cfg_path, "w") as f:
            _yaml.dump(cfg_data, f, default_flow_style=False)
        actual_cmd = run_cmd + ["--config", str(tmp_cfg_path)]

        try:
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

    @render.ui
    def run_progress():
        if not running() and progress_pct() == 0:
            return ui.div()
        pct = progress_pct()
        msg = step_msg()
        is_running = running()
        bar_class = "progress-bar progress-bar-striped progress-bar-animated" if is_running else "progress-bar"
        status_color = "#198754" if pct == 100 else ("#dc3545" if msg == "Failed" else "#0d6efd")
        indicator = ui.div(
            ui.tags.span(
                "● Running" if is_running else ("✓ Done" if pct == 100 else "✗ Failed"),
                style=f"font-size:.75rem;color:{status_color};font-weight:600",
            ),
            class_="mt-2",
        )
        return ui.div(
            indicator,
            ui.div(
                ui.div(
                    class_=bar_class,
                    role="progressbar",
                    style=f"width:{pct}%",
                    **{"aria-valuenow": str(pct), "aria-valuemin": "0", "aria-valuemax": "100"},
                ),
                class_="progress mt-1",
                style="height:8px",
            ),
            ui.div(msg, style="font-size:.7rem;color:#6c757d;margin-top:2px") if msg else ui.div(),
            class_="mt-2",
        )

    @render.ui
    def run_status():
        log = run_log()
        if not log:
            return ui.div()
        return ui.div(
            ui.tags.pre(log, style="font-size:.7rem;max-height:120px;overflow-y:auto"),
            class_="mt-2",
        )

    @render.ui
    def metrics_tbl():
        df = load_all_metrics(output_dir)
        return ui.HTML(_metrics_comparison_html(df, selected_te=int(input.te())))

    @render.plot(alt="Cumulative returns")
    def returns_plot():
        df = load_returns_df(output_dir, te_pct=int(input.te()))
        return _cum_returns_chart(df)

    @render.plot(alt="Portfolio weights")
    def weights_plot():
        wdf = load_weights_df(output_dir, te_pct=int(input.te()))
        return _weights_chart(wdf)

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

    def _make_regime_renderer(f: str):
        @output(id=f"regime_{f}_plot")
        @render.plot(alt=f"{f} regime")
        def _regime():
            reg          = load_regimes_df(output_dir)
            parquet_path = output_dir / "cache" / "active_returns.parquet"
            if not parquet_path.exists():
                from shiny_app.components.analytics import _blank_fig
                return _blank_fig("Run the model to generate regime data")
            active = pd.read_parquet(parquet_path)
            return _regime_chart(f, reg, active)
        return _regime

    for _f in _FACTORS:
        _make_regime_renderer(_f)
