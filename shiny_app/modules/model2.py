"""Model 2 tab — placeholder until teammate provides outputs."""

from pathlib import Path
from shiny import module, ui, reactive

from shiny_app.components.layout import placeholder_card
from shiny_app.utils.runner import run_pipeline


@module.ui
def model_tab_ui(cfg: dict):
    run_btn_id = "rerun" if cfg.get("run_command") else None
    return ui.div(
        placeholder_card(
            "This model's outputs have not been generated yet.",
            run_btn_id=run_btn_id,
        ),
        style="padding:2rem",
    )


@module.server
def model_tab_server(input, output, session, cfg: dict, project_root: Path):
    run_cmd = cfg.get("run_command")

    if run_cmd:
        @reactive.effect
        @reactive.event(input.rerun)
        def _launch():
            run_pipeline(run_cmd, project_root)
