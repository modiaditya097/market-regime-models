"""Comparison tab — overlaid cumulative returns chart + side-by-side metrics table."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from shiny import module, ui, render

from shiny_app.components.charts import load_returns_df, load_metrics_row, _DISPLAY_COLS

_COLORS = ["#7c3aed", "#10b981", "#f97316"]  # model1, model2, model3


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
            style="max-width:200px;margin-bottom:1rem",
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

        for i, model_cfg in enumerate(all_cfg["models"]):
            df = load_returns_df(Path(model_cfg["output_dir"]), te_pct=te)
            if df is None:
                continue
            cum = (1 + df["portfolio"]).cumprod() - 1
            ax.plot(cum.index, cum * 100, label=model_cfg["name"],
                    color=_COLORS[i % len(_COLORS)], linewidth=1.5)
            plotted = True

        if not plotted:
            ax.text(0.5, 0.5, "No model outputs available",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=12, color="grey")
        else:
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
