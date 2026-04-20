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
