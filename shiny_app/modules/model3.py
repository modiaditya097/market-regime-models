"""SC-HMM (Spectral Clustering + HMM) model tab.

Layout: sticky sidebar with section anchors + scrollable main area.
Displays regime timeline, cumulative returns, drawdown, rolling Sharpe,
portfolio weights, transition matrix heatmap, stress/bull period tables,
and macro signal panel — all driven by pre-computed CSVs and PNGs.

Output contract (files expected in cfg['output_dir'] = outputs/model3/):
    returns_te3.csv          — date, portfolio, market, ew
    plots/
        cumulative_returns.png
        drawdown.png
        rolling_sharpe.png
        portfolio_weights.png
        regime_timeline.png
        transition_matrix.png
    metrics_full.csv         — full-period performance table
    metrics_train.csv        — in-sample performance table
    metrics_test.csv         — out-of-sample performance table
    stress_table.csv         — stress period returns
    bull_table.csv           — bull period returns
    macro_latest.csv         — latest macro snapshot (indicator, value, signal)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from shiny import module, ui, render, reactive
from shiny_app.components.charts import img_tag
from shiny_app.components.layout import placeholder_card, section
from shiny_app.utils.runner import run_pipeline


# ── colour palette ─────────────────────────────────────────────────────────────
_C = {
    "hard":    "#2c3e50",
    "soft":    "#8e44ad",
    "macro":   "#16a085",
    "spy":     "#2980b9",
    "rp":      "#e67e22",
    "ew":      "#27ae60",
    "bull":    "#2ecc71",
    "neutral": "#f39c12",
    "bear":    "#e74c3c",
}

_SIDEBAR_CSS = """
<style>
  .sc-hmm-sidebar .anchor-links a {
    display:block; padding:4px 0;
    font-size:.82rem; color:#6c757d; text-decoration:none;
  }
  .sc-hmm-sidebar .anchor-links a:hover { color:#0d6efd; }
</style>
"""


# ── internal helpers ───────────────────────────────────────────────────────────

def _load_csv(output_dir: Path, name: str):
    """Load CSV → DataFrame, return empty DataFrame if file missing."""
    import pandas as pd
    p = output_dir / name
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _fmt_val(val: str, col: str) -> str:
    """Return colour-coded HTML for a metric cell value."""
    try:
        fv = float(str(val).replace("%", "").replace(",", ""))
        if col in ("Sharpe", "Sortino", "Calmar", "CAGR", "Cumul.Ret", "Hit Rate"):
            colour = "#198754" if fv > 0 else "#dc3545"
        elif col == "Max DD":
            colour = "#dc3545" if fv < -15 else "#fd7e14" if fv < -5 else "#198754"
        else:
            colour = "#212529"
        return f'<span style="color:{colour};font-weight:600">{val}</span>'
    except Exception:
        return str(val)


def _metrics_html(df) -> "ui.Tag":
    """Render a performance DataFrame as a Bootstrap HTML table."""
    import pandas as pd
    if df.empty:
        return ui.p("No data — run the model first.", class_="text-muted")
    strat_col = df.columns[0]
    metric_cols = [c for c in df.columns if c != strat_col]
    header = [ui.tags.th(strat_col, style="min-width:200px;font-size:.78rem")] + \
             [ui.tags.th(c, style="text-align:right;font-size:.78rem") for c in metric_cols]
    rows = []
    for _, row in df.iterrows():
        cells = [ui.tags.td(str(row[strat_col]),
                            style="font-size:.79rem;white-space:nowrap;font-weight:500")]
        for col in metric_cols:
            cells.append(ui.tags.td(
                ui.HTML(_fmt_val(str(row[col]), col)),
                style="text-align:right;font-size:.79rem"))
        rows.append(ui.tags.tr(*cells))
    return ui.div(
        ui.tags.table(
            ui.tags.thead(ui.tags.tr(*header, style="background:#f8f9fa")),
            ui.tags.tbody(*rows),
            class_="table table-sm table-hover",
            style="width:100%;border-collapse:collapse",
        ),
        style="overflow-x:auto",
    )


def _period_html(df) -> "ui.Tag":
    """Render a stress/bull period DataFrame as colour-coded HTML table."""
    if df.empty:
        return ui.p("No data — run the model first.", class_="text-muted")
    period_col  = df.columns[0]
    strat_cols  = [c for c in df.columns if c != period_col]
    header = [ui.tags.th(period_col, style="min-width:200px;font-size:.78rem")] + \
             [ui.tags.th(c, style="text-align:right;font-size:.78rem") for c in strat_cols]
    rows = []
    for _, row in df.iterrows():
        cells = [ui.tags.td(str(row[period_col]),
                            style="font-size:.79rem;white-space:nowrap;font-weight:500")]
        for c in strat_cols:
            val = str(row[c])
            try:
                fv  = float(val.replace("%","").replace(",",""))
                col = "#198754" if fv >= 0 else "#dc3545"
                html = f'<span style="color:{col};font-weight:600">{val}</span>'
            except Exception:
                html = val
            cells.append(ui.tags.td(ui.HTML(html), style="text-align:right;font-size:.79rem"))
        rows.append(ui.tags.tr(*cells))
    return ui.div(
        ui.tags.table(
            ui.tags.thead(ui.tags.tr(*header, style="background:#f8f9fa")),
            ui.tags.tbody(*rows),
            class_="table table-sm table-hover",
            style="width:100%;border-collapse:collapse",
        ),
        style="overflow-x:auto",
    )


# ── UI definition ──────────────────────────────────────────────────────────────

@module.ui
def model_tab_ui(cfg: dict):
    has_run = bool(cfg.get("run_command"))

    sidebar = ui.sidebar(
        ui.HTML(_SIDEBAR_CSS),
        ui.div(
            ui.tags.b("Sections", style="font-size:.85rem"),
            ui.div(
                ui.tags.a("📊 Performance",          href="#perf"),
                ui.tags.a("📈 Cumulative Returns",   href="#cumret"),
                ui.tags.a("📉 Drawdown",             href="#dd"),
                ui.tags.a("⚡ Rolling Sharpe",        href="#sharpe"),
                ui.tags.a("⚖️ Portfolio Weights",    href="#weights"),
                ui.tags.a("🔴 Regime Timeline",      href="#timeline"),
                ui.tags.a("🔁 Transition Matrix",    href="#trans"),
                ui.tags.a("📋 Stress / Bull Tables", href="#tables"),
                ui.tags.a("🌡️ Macro Signals",        href="#macro"),
                class_="anchor-links",
            ),
            class_="mb-3",
        ),
        ui.hr(),
        ui.input_action_button(
            "rerun", "▶ Re-run Model",
            class_="btn-primary btn-sm w-100",
            disabled=not has_run,
        ),
        *([] if has_run else [
            ui.p("No run command configured.",
                 class_="text-muted mt-1", style="font-size:.75rem")
        ]),
        ui.output_ui("run_status"),
        width=190,
        class_="sc-hmm-sidebar",
    )

    main = ui.div(
        section("📊 Performance Summary", "perf",
            ui.navset_tab(
                ui.nav_panel("Full Period",       ui.output_ui("tbl_full")),
                ui.nav_panel("In-Sample",         ui.output_ui("tbl_train")),
                ui.nav_panel("Out-of-Sample",     ui.output_ui("tbl_test")),
            ),
        ),
        section("📈 Cumulative Returns",   "cumret",   ui.output_ui("img_cumret")),
        section("📉 Drawdown",             "dd",       ui.output_ui("img_dd")),
        section("⚡ Rolling 52-Week Sharpe","sharpe",  ui.output_ui("img_sharpe")),
        section("⚖️ Portfolio Weights (Hard)", "weights", ui.output_ui("img_weights")),
        section("🔴 Regime Timeline",      "timeline", ui.output_ui("img_timeline")),
        section("🔁 Transition Matrix",    "trans",    ui.output_ui("img_trans")),
        section("📋 Stress & Bull Periods","tables",
            ui.navset_tab(
                ui.nav_panel("Stress Periods", ui.output_ui("tbl_stress")),
                ui.nav_panel("Bull Periods",   ui.output_ui("tbl_bull")),
            ),
        ),
        section("🌡️ Macro Signal Overview","macro",   ui.output_ui("macro_panel")),
        style="padding:1rem 1.5rem",
    )

    return ui.layout_sidebar(sidebar, main)


# ── Server definition ──────────────────────────────────────────────────────────

@module.server
def model_tab_server(input, output, session, cfg: dict, project_root: Path):
    output_dir = Path(cfg["output_dir"])
    plots_dir  = output_dir / "plots"
    run_cmd    = cfg.get("run_command")

    running = reactive.value(False)
    run_log = reactive.value("")

    # ── Re-run ───────────────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.rerun)
    def _launch():
        if run_cmd is None or running():
            return
        running.set(True)
        run_log.set("⏳ Pipeline running…")
        proc = run_pipeline(run_cmd, project_root)
        out, _ = proc.communicate()
        if proc.returncode == 0:
            run_log.set("✅ Done.\n" + (out or "")[-1500:])
        else:
            run_log.set(f"❌ Error (exit {proc.returncode})\n" + (out or "")[-2000:])
        running.set(False)

    @render.ui
    def run_status():
        log = run_log()
        if not log:
            return ui.div()
        c = "#198754" if log.startswith("✅") else "#dc3545" if log.startswith("❌") else "#6c757d"
        return ui.div(
            ui.tags.pre(log, style=f"font-size:.68rem;max-height:130px;overflow-y:auto;color:{c}"),
            class_="mt-2",
        )

    # ── Performance tables ────────────────────────────────────────────────────
    @render.ui
    def tbl_full():
        return _metrics_html(_load_csv(output_dir, "metrics_full.csv"))

    @render.ui
    def tbl_train():
        return _metrics_html(_load_csv(output_dir, "metrics_train.csv"))

    @render.ui
    def tbl_test():
        return _metrics_html(_load_csv(output_dir, "metrics_test.csv"))

    # ── Plot images ───────────────────────────────────────────────────────────
    def _img_or_placeholder(fname: str, alt: str, msg: str):
        p = plots_dir / fname
        return img_tag(p, alt) if p.exists() else placeholder_card(msg)

    @render.ui
    def img_cumret():
        return _img_or_placeholder(
            "cumulative_returns.png", "Cumulative Returns",
            "Cumulative returns plot not found. Run the model to generate outputs.")

    @render.ui
    def img_dd():
        return _img_or_placeholder(
            "drawdown.png", "Drawdown",
            "Drawdown plot not found. Run the model to generate outputs.")

    @render.ui
    def img_sharpe():
        return _img_or_placeholder(
            "rolling_sharpe.png", "Rolling Sharpe",
            "Rolling Sharpe plot not found. Run the model to generate outputs.")

    @render.ui
    def img_weights():
        return _img_or_placeholder(
            "portfolio_weights.png", "Portfolio Weights",
            "Portfolio weights plot not found. Run the model to generate outputs.")

    @render.ui
    def img_timeline():
        return _img_or_placeholder(
            "regime_timeline.png", "Regime Timeline",
            "Regime timeline plot not found. Run the model to generate outputs.")

    @render.ui
    def img_trans():
        return _img_or_placeholder(
            "transition_matrix.png", "Transition Matrix",
            "Transition matrix not found. Run the model to generate outputs.")

    # ── Stress / bull tables ──────────────────────────────────────────────────
    @render.ui
    def tbl_stress():
        return _period_html(_load_csv(output_dir, "stress_table.csv"))

    @render.ui
    def tbl_bull():
        return _period_html(_load_csv(output_dir, "bull_table.csv"))

    # ── Macro panel ───────────────────────────────────────────────────────────
    @render.ui
    def macro_panel():
        df = _load_csv(output_dir, "macro_latest.csv")
        if df.empty:
            return ui.p("Macro snapshot not available — run the model first.",
                        class_="text-muted")

        colour_map = {"bull": "#198754", "neutral": "#fd7e14", "bear": "#dc3545"}
        cards = []
        for _, row in df.iterrows():
            label  = str(row.get("indicator", ""))
            value  = str(row.get("value", ""))
            signal = str(row.get("signal", "neutral")).lower()
            colour = colour_map.get(signal, "#6c757d")
            cards.append(ui.div(
                ui.div(
                    ui.p(label, style="font-size:.74rem;color:#6c757d;margin:0"),
                    ui.p(value, style="font-size:1.25rem;font-weight:700;margin:0"),
                    ui.tags.span(signal.upper(),
                                 style=f"font-size:.7rem;font-weight:700;color:{colour}"),
                    class_="card-body py-2 px-3",
                ),
                class_="card",
                style=f"border-top:3px solid {colour};flex:1;min-width:140px",
            ))

        return ui.div(
            ui.div(*cards, style="display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1rem"),
            ui.p(
                "Thresholds — VIX: Bull < 16, Bear > 25. "
                "Yield Curve (10Y-2Y): Inverted < 0, Bull > 1.5. "
                "HY Spread: Bull < 3.0, Bear > 4.5. "
                "Signal = weighted majority vote (VIX ×2, Curve ×1, HY ×1).",
                style="font-size:.74rem;color:#6c757d",
            ),
        )
