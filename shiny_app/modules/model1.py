"""SJM + Black-Litterman model tab (Layout C: sticky sidebar + scrollable sections)."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shiny import module, ui, render, reactive

from shiny_app.components.charts import img_tag, load_metrics_row, load_returns_df, _DISPLAY_COLS
from shiny_app.components.layout import placeholder_card, section
from shiny_app.utils.runner import run_pipeline

_FACTORS = ["value", "size", "quality", "growth", "momentum"]

_SIDEBAR_CSS = """
<style>
.model-sidebar { position: sticky; top: 1rem; }
.anchor-links a { display: block; padding: 4px 0; font-size: .875rem; color: #6c757d; text-decoration: none; }
.anchor-links a:hover { color: #0d6efd; }
</style>
"""


@module.ui
def model_tab_ui(cfg: dict):
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
        ui.input_action_button(
            "rerun", "Run Model",
            class_="btn-primary btn-sm w-100 mt-2",
            disabled=cfg.get("run_command") is None,
        ),
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

    @reactive.effect
    @reactive.event(input.rerun)
    def _launch_pipeline():
        if run_cmd is None or running():
            return
        running.set(True)
        run_log.set("Running pipeline...")
        proc = run_pipeline(run_cmd, project_root)
        stdout, _ = proc.communicate()
        run_log.set(stdout[-2000:] if stdout else "Done.")
        running.set(False)

    @render.ui
    def run_status():
        log = run_log()
        if not log:
            return ui.div()
        return ui.div(
            ui.tags.pre(log, style="font-size:.7rem;max-height:150px;overflow-y:auto"),
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
