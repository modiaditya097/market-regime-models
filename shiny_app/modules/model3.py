"""SC-HMM Phase 3 — Spectral Clustering + Macro Overlay dashboard tab.

Professor requirements:
  - Configurable input parameters from the frontend
  - Pipeline runnable from the Re-run button
  - All sections visible and populated correctly

Output contract  (outputs/model3/):
    returns_te3.csv        — date, portfolio, market, ew
    metrics_full.csv       — full-period performance table
    metrics_train.csv      — in-sample performance table
    metrics_test.csv       — out-of-sample performance table
    stress_table.csv       — stress period returns
    bull_table.csv         — bull period returns
    macro_latest.csv       — VIX / yield curve / HY snapshot
    k_analysis.csv         — K=2 vs K=3 frequency table
    plots/
        cumulative_returns.png
        drawdown.png
        rolling_sharpe.png
        portfolio_weights.png
        regime_timeline.png
        transition_matrix.png
        annual_heatmap.png
        risk_return_scatter.png
"""

from pathlib import Path
import pandas as pd
import numpy as np

from shiny import module, ui, render, reactive
from shiny_app.components.layout import placeholder_card, section
from shiny_app.utils.runner import run_pipeline


# ── helpers ────────────────────────────────────────────────────────────────────

def _load(output_dir: Path, name: str) -> pd.DataFrame:
    p = output_dir / name
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _colour(val: str, col: str) -> str:
    try:
        fv = float(str(val).replace("%", "").replace(",", ""))
        if col in ("Sharpe", "Sortino", "Calmar", "CAGR", "Cumul.Ret"):
            c = "#198754" if fv > 0 else "#dc3545"
        elif col == "Max DD":
            c = "#dc3545" if fv < -15 else "#fd7e14" if fv < -5 else "#198754"
        else:
            c = "#212529"
        return f'<span style="color:{c};font-weight:600">{val}</span>'
    except Exception:
        return str(val)


def _metrics_table(df) -> "ui.Tag":
    if df.empty:
        return ui.p("No data — run the model first.", class_="text-muted fst-italic")
    strat_col  = df.columns[0]
    metric_cols = [c for c in df.columns if c != strat_col]
    hdr = [ui.tags.th(strat_col, style="min-width:220px;font-size:.78rem")] + \
          [ui.tags.th(c, style="text-align:right;font-size:.78rem") for c in metric_cols]
    rows = []
    for _, row in df.iterrows():
        star = str(row[strat_col]).startswith("*")
        style = "font-size:.79rem;white-space:nowrap;font-weight:" + ("700" if star else "400")
        cells = [ui.tags.td(str(row[strat_col]), style=style)]
        for col in metric_cols:
            cells.append(ui.tags.td(ui.HTML(_colour(str(row[col]), col)),
                                    style="text-align:right;font-size:.79rem"))
        rows.append(ui.tags.tr(*cells, style="background:#fffde7" if star else ""))
    return ui.div(
        ui.tags.table(
            ui.tags.thead(ui.tags.tr(*hdr, style="background:#f8f9fa")),
            ui.tags.tbody(*rows),
            class_="table table-sm table-hover",
            style="width:100%;border-collapse:collapse",
        ),
        style="overflow-x:auto",
    )


def _period_table(df) -> "ui.Tag":
    if df.empty:
        return ui.p("No data — run the model first.", class_="text-muted fst-italic")
    period_col = df.columns[0]
    strat_cols = [c for c in df.columns if c != period_col]
    hdr = [ui.tags.th(period_col, style="min-width:200px;font-size:.78rem")] + \
          [ui.tags.th(c, style="text-align:right;font-size:.78rem") for c in strat_cols]
    rows = []
    for _, row in df.iterrows():
        cells = [ui.tags.td(str(row[period_col]),
                            style="font-size:.79rem;white-space:nowrap;font-weight:500")]
        for c in strat_cols:
            val = str(row[c])
            try:
                fv  = float(val.replace("%", "").replace(",", ""))
                col = "#198754" if fv >= 0 else "#dc3545"
                html = f'<span style="color:{col};font-weight:600">{val}</span>'
            except Exception:
                html = val
            cells.append(ui.tags.td(ui.HTML(html), style="text-align:right;font-size:.79rem"))
        rows.append(ui.tags.tr(*cells))
    return ui.div(
        ui.tags.table(
            ui.tags.thead(ui.tags.tr(*hdr, style="background:#f8f9fa")),
            ui.tags.tbody(*rows),
            class_="table table-sm table-hover",
            style="width:100%;border-collapse:collapse",
        ),
        style="overflow-x:auto",
    )


def _img(plots_dir: Path, fname: str, alt: str, msg: str) -> "ui.Tag":
    p = plots_dir / fname
    if not p.exists():
        return placeholder_card(msg)
    import base64
    data = base64.b64encode(p.read_bytes()).decode()
    return ui.tags.img(
        src=f"data:image/png;base64,{data}",
        alt=alt,
        style="width:100%;border-radius:6px",
    )


# ── UI ─────────────────────────────────────────────────────────────────────────

@module.ui
def model_tab_ui(cfg: dict):
    has_run = bool(cfg.get("run_command"))

    sidebar = ui.sidebar(
        ui.tags.style("""
            .sc3-sidebar .anchor-links a {
                display:block;padding:3px 0;font-size:.82rem;
                color:#6c757d;text-decoration:none;
            }
            .sc3-sidebar .anchor-links a:hover{color:#0d6efd}
            .sc3-param-label{font-size:.74rem;color:#6c757d;margin-bottom:2px}
        """),
        # ── Section navigation ──────────────────────────────────────────────
        ui.div(
            ui.tags.b("Sections", style="font-size:.85rem"),
            ui.div(
                ui.tags.a("📊 Performance",         href="#perf"),
                ui.tags.a("📈 Cumulative Returns",  href="#cumret"),
                ui.tags.a("📉 Drawdown",            href="#dd"),
                ui.tags.a("⚡ Rolling Sharpe",       href="#sharpe"),
                ui.tags.a("⚖️ Portfolio Weights",   href="#weights"),
                ui.tags.a("🔴 Regime Timeline",     href="#timeline"),
                ui.tags.a("🔁 Transition Matrix",   href="#trans"),
                ui.tags.a("📅 Annual Heatmap",      href="#heatmap"),
                ui.tags.a("🎯 Risk-Return",         href="#scatter"),
                ui.tags.a("📋 Stress / Bull",       href="#tables"),
                ui.tags.a("🌡️ Macro Signals",       href="#macro"),
                class_="anchor-links",
            ),
            class_="mb-3",
        ),
        ui.hr(),
        # ── Configurable parameters ─────────────────────────────────────────
        ui.tags.b("Model Parameters", style="font-size:.85rem"),
        ui.p("Changes take effect on Re-run.", style="font-size:.72rem;color:#6c757d;margin:2px 0 8px"),

        ui.p("Asset Universe", class_="sc3-param-label mt-2"),
        ui.input_checkbox_group(
            "assets", None,
            choices={"SPY": "SPY", "IWM": "IWM", "IEF": "IEF",
                     "TIP": "TIP", "GLD": "GLD"},
            selected=["SPY", "IWM", "IEF", "TIP", "GLD"],
            inline=True,
        ),

        ui.p("K Range (# Regimes)", class_="sc3-param-label mt-1"),
        ui.layout_columns(
            ui.div(
                ui.p("Min", style="font-size:.72rem;text-align:center;margin:0"),
                ui.input_numeric("k_min", None, value=2, min=2, max=3, step=1),
            ),
            ui.div(
                ui.p("Max", style="font-size:.72rem;text-align:center;margin:0"),
                ui.input_numeric("k_max", None, value=3, min=2, max=5, step=1),
            ),
            col_widths=[6, 6],
        ),

        ui.p("Initial Window (weeks)", class_="sc3-param-label mt-1"),
        ui.input_numeric("initial_window", None, value=104, min=52, max=260, step=26),

        ui.p("Refit Cadence (weeks)", class_="sc3-param-label mt-1"),
        ui.input_numeric("refit_cadence", None, value=4, min=1, max=52, step=1),

        ui.p("Smoothing Window (weeks)", class_="sc3-param-label mt-1"),
        ui.input_numeric("smooth_window", None, value=3, min=1, max=8, step=1),

        ui.p("Macro Neutral Threshold", class_="sc3-param-label mt-1"),
        ui.input_numeric("macro_neutral_thresh", None, value=0.5,
                         min=0.0, max=2.0, step=0.1),

        ui.p("Macro Extreme Threshold", class_="sc3-param-label mt-1"),
        ui.input_numeric("macro_extreme_thresh", None, value=1.5,
                         min=0.5, max=3.0, step=0.1),

        ui.p("Transaction Cost (bps)", class_="sc3-param-label mt-1"),
        ui.input_numeric("tc_bps", None, value=10, min=0, max=50, step=5),

        ui.hr(),
        ui.input_action_button(
            "rerun", "▶ Re-run Model",
            class_="btn-primary btn-sm w-100",
            disabled=not has_run,
        ),
        *([] if has_run else [
            ui.p("Configure run_command in app_config.yaml to enable.",
                 class_="text-muted mt-1", style="font-size:.72rem")
        ]),
        ui.output_ui("run_status"),
        width=210,
        class_="sc3-sidebar",
    )

    main = ui.div(
        section("📊 Performance Summary", "perf",
            ui.navset_tab(
                ui.nav_panel("Full Period",     ui.output_ui("tbl_full")),
                ui.nav_panel("In-Sample",       ui.output_ui("tbl_train")),
                ui.nav_panel("Out-of-Sample",   ui.output_ui("tbl_test")),
            ),
        ),
        section("📈 Cumulative Returns",  "cumret",  ui.output_ui("img_cumret")),
        section("📉 Drawdown",            "dd",      ui.output_ui("img_dd")),
        section("⚡ Rolling 52-Week Sharpe","sharpe", ui.output_ui("img_sharpe")),
        section("⚖️ Portfolio Weights",   "weights", ui.output_ui("img_weights")),
        section("🔴 Regime Timeline",     "timeline",ui.output_ui("img_timeline")),
        section("🔁 Transition Matrix",   "trans",   ui.output_ui("img_trans")),
        section("📅 Annual Returns Heatmap","heatmap",ui.output_ui("img_heatmap")),
        section("🎯 Risk-Return Scatter", "scatter", ui.output_ui("img_scatter")),
        section("📋 Stress & Bull Periods","tables",
            ui.navset_tab(
                ui.nav_panel("Stress Periods", ui.output_ui("tbl_stress")),
                ui.nav_panel("Bull Periods",   ui.output_ui("tbl_bull")),
            ),
        ),
        section("🌡️ Macro Signal Overview","macro",  ui.output_ui("macro_panel")),
        style="padding:1rem 1.5rem",
    )

    return ui.layout_sidebar(sidebar, main)


# ── Server ──────────────────────────────────────────────────────────────────────

@module.server
def model_tab_server(input, output, session, cfg: dict, project_root: Path):
    output_dir = Path(cfg["output_dir"])
    plots_dir  = output_dir / "plots"
    run_cmd    = cfg.get("run_command")

    running  = reactive.value(False)
    run_log  = reactive.value("")
    _trigger = reactive.value(0)   # incremented after each run to refresh outputs

    # ── Re-run with parameters ────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.rerun)
    def _launch():
        if run_cmd is None or running():
            return
        running.set(True)
        run_log.set("⏳ Pipeline running…")

        # Build parameter env vars so the script can read them
        import os, subprocess
        env = os.environ.copy()
        env["SCHMM_ASSETS"]               = ",".join(list(input.assets()))
        env["SCHMM_K_MIN"]                = str(int(input.k_min()))
        env["SCHMM_K_MAX"]                = str(int(input.k_max()))
        env["SCHMM_INITIAL_WINDOW"]       = str(int(input.initial_window()))
        env["SCHMM_REFIT_CADENCE"]        = str(int(input.refit_cadence()))
        env["SCHMM_SMOOTH_WINDOW"]        = str(int(input.smooth_window()))
        env["SCHMM_MACRO_NEUTRAL_THRESH"] = str(float(input.macro_neutral_thresh()))
        env["SCHMM_MACRO_EXTREME_THRESH"] = str(float(input.macro_extreme_thresh()))
        env["SCHMM_TC_BPS"]              = str(int(input.tc_bps()))

        proc = subprocess.Popen(
            run_cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        out, _ = proc.communicate()
        if proc.returncode == 0:
            run_log.set("✅ Done.\n" + (out or "")[-1500:])
            _trigger.set(_trigger() + 1)
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
            ui.tags.pre(log, style=f"font-size:.68rem;max-height:140px;overflow-y:auto;color:{c}"),
            class_="mt-2",
        )

    # ── Performance tables ────────────────────────────────────────────────────
    @render.ui
    def tbl_full():
        _trigger()
        return _metrics_table(_load(output_dir, "metrics_full.csv"))

    @render.ui
    def tbl_train():
        _trigger()
        return _metrics_table(_load(output_dir, "metrics_train.csv"))

    @render.ui
    def tbl_test():
        _trigger()
        return _metrics_table(_load(output_dir, "metrics_test.csv"))

    # ── Plot images ───────────────────────────────────────────────────────────
    def _i(fname, alt, msg):
        _trigger()
        return _img(plots_dir, fname, alt, msg)

    @render.ui
    def img_cumret():
        return _i("cumulative_returns.png", "Cumulative Returns",
                  "Cumulative returns plot not found. Run the model to generate outputs.")

    @render.ui
    def img_dd():
        return _i("drawdown.png", "Drawdown",
                  "Drawdown plot not found. Run the model to generate outputs.")

    @render.ui
    def img_sharpe():
        return _i("rolling_sharpe.png", "Rolling Sharpe",
                  "Rolling Sharpe plot not found. Run the model to generate outputs.")

    @render.ui
    def img_weights():
        return _i("portfolio_weights.png", "Portfolio Weights",
                  "Portfolio weights plot not found. Run the model to generate outputs.")

    @render.ui
    def img_timeline():
        return _i("regime_timeline.png", "Regime Timeline",
                  "Regime timeline plot not found. Run the model to generate outputs.")

    @render.ui
    def img_trans():
        return _i("transition_matrix.png", "Transition Matrix",
                  "Transition matrix not found. Run the model to generate outputs.")

    @render.ui
    def img_heatmap():
        return _i("annual_heatmap.png", "Annual Returns Heatmap",
                  "Annual heatmap not found. Run the model to generate outputs.")

    @render.ui
    def img_scatter():
        return _i("risk_return_scatter.png", "Risk-Return Scatter",
                  "Risk-return scatter not found. Run the model to generate outputs.")

    # ── Stress / bull tables ──────────────────────────────────────────────────
    @render.ui
    def tbl_stress():
        _trigger()
        return _period_table(_load(output_dir, "stress_table.csv"))

    @render.ui
    def tbl_bull():
        _trigger()
        return _period_table(_load(output_dir, "bull_table.csv"))

    # ── Macro panel ───────────────────────────────────────────────────────────
    @render.ui
    def macro_panel():
        _trigger()
        df = _load(output_dir, "macro_latest.csv")
        if df.empty:
            return ui.p("Macro snapshot not available — run the model first.",
                        class_="text-muted fst-italic")
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
                "Composite z-score = (z(VIX) + z(HY OAS) + z(−Yield Curve)) / 3. "
                "Bear confirmed if composite > 0; downgraded to neutral if composite < −threshold.",
                style="font-size:.74rem;color:#6c757d",
            ),
        )
