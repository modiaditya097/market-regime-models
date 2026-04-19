"""Entry point for the Dynamic Factor Allocation Shiny dashboard.

Run from the project root:
    shiny run shiny_app/app.py --reload
"""

import importlib
from pathlib import Path

from shiny import App, ui

from shiny_app.utils.config import load_config
from shiny_app.components.comparison import comparison_ui, comparison_server

_CONFIG_PATH = Path(__file__).parent / "app_config.yaml"
_PROJECT_ROOT = Path(__file__).parent.parent


def build_app() -> App:
    cfg = load_config(_CONFIG_PATH, project_root=_PROJECT_ROOT)

    nav_panels = [
        ui.nav_panel("Comparison", comparison_ui("comp", all_cfg=cfg)),
    ]

    for model_cfg in cfg["models"]:
        mod = importlib.import_module(model_cfg["module"])
        nav_panels.append(
            ui.nav_panel(model_cfg["name"], mod.model_tab_ui(model_cfg["id"], cfg=model_cfg))
        )

    app_ui = ui.page_navbar(
        *nav_panels,
        title="Dynamic Factor Allocation Dashboard",
        id="main_nav",
        bg="#1a1a2e",
        inverse=True,
    )

    def server(input, output, session):
        comparison_server("comp", all_cfg=cfg)
        for model_cfg in cfg["models"]:
            mod = importlib.import_module(model_cfg["module"])
            mod.model_tab_server(
                model_cfg["id"],
                cfg=model_cfg,
                project_root=_PROJECT_ROOT,
            )

    return App(app_ui, server)


app = build_app()
