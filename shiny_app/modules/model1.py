"""SJM + Black-Litterman model tab (Layout C: sticky sidebar + scrollable sections)."""

import asyncio
import re
from pathlib import Path

import yaml as _yaml

from shiny import module, ui, render, reactive

from shiny_app.components.charts import img_tag, load_metrics_row, load_returns_df, _DISPLAY_COLS
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
                ui.tags.a("Metrics", href="#metrics"),
                ui.tags.a("Cumulative Returns", href="#returns"),
                ui.tags.a("Portfolio Weights", href="#weights"),
                ui.tags.a("Regime Plots", href="#regimes"),
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
        section("Performance Metrics", "metrics", ui.output_table("metrics_tbl")),
        section("Cumulative Returns", "returns", ui.output_ui("returns_img")),
        section("Portfolio Weights", "weights", ui.output_ui("weights_img")),
        section("Regime Plots", "regimes", ui.output_ui("regime_imgs")),
        style="padding:1rem",
    )

    return ui.layout_sidebar(sidebar, main)


@module.server
def model_tab_server(input, output, session, cfg: dict, project_root: Path):
    import pandas as pd
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

    @render.table
    def metrics_tbl():
        te = int(input.te())
        df = load_metrics_row(output_dir, te_pct=te)
        if df.empty:
            return pd.DataFrame({"Status": ["No results — run the model first"]})
        return df.rename(columns=_DISPLAY_COLS)

    @render.ui
    def returns_img():
        te = int(input.te())
        path = output_dir / f"plots/cumulative_returns_te{te}.png"
        if not path.exists():
            return placeholder_card(f"Cumulative returns plot not found (TE={te}%).")
        return img_tag(path, alt=f"Cumulative returns TE={te}%")

    @render.ui
    def weights_img():
        te = int(input.te())
        path = output_dir / f"plots/portfolio_weights_te{te}.png"
        if not path.exists():
            return placeholder_card(f"Portfolio weights plot not found (TE={te}%).")
        return img_tag(path, alt=f"Portfolio weights TE={te}%")

    @render.ui
    def regime_imgs():
        imgs = []
        for factor in _FACTORS:
            path = output_dir / f"plots/regime_{factor}.png"
            imgs.append(
                ui.div(
                    ui.h6(factor.capitalize(), class_="text-muted"),
                    img_tag(path, alt=f"{factor} regime"),
                    class_="mb-3",
                )
            )
        return ui.div(*imgs)
