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
