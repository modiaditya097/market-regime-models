"""Analytics charts and data loaders for the Dynamic Factor Allocation tab."""

from math import sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_ASSETS   = ["market", "value", "size", "quality", "growth", "momentum"]
_FACTORS  = ["value", "size", "quality", "growth", "momentum"]
_COLORS   = ["#888888", "#7c3aed", "#10b981", "#f97316", "#0ea5e9", "#eab308"]

_METRIC_COLORS = {
    "Sharpe":     lambda v: "#198754" if v >= 0.5 else ("#dc3545" if v < 0 else "inherit"),
    "IR vs Mkt":  lambda v: "#198754" if v > 0 else "#dc3545",
    "Max DD":     lambda v: "#dc3545",
    "Volatility": lambda v: "inherit",
    "Active Ret": lambda v: "#198754" if v > 0 else "#dc3545",
    "Turnover":   lambda v: "inherit",
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
        return f"{raw * 100:.2f}%"
    if col_display == "Turnover":
        return f"{raw:.4f}"
    return str(raw)


def _blank_fig(msg: str = "No data available") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            color="grey", fontsize=12, transform=ax.transAxes)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_weights_df(output_dir: Path, te_pct: int) -> pd.DataFrame | None:
    path = Path(output_dir) / f"weights_te{te_pct}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_regimes_df(output_dir: Path) -> pd.DataFrame | None:
    path = Path(output_dir) / "regimes.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_all_metrics(output_dir: Path) -> pd.DataFrame:
    path = Path(output_dir) / "results.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# ── Chart functions ───────────────────────────────────────────────────────────

def cumulative_returns_plot(returns_df: pd.DataFrame | None) -> plt.Figure:
    if returns_df is None or returns_df.empty:
        return _blank_fig()
    fig, ax = plt.subplots(figsize=(10, 4))
    _styles = [
        ("portfolio", "Dynamic Allocation", "#7c3aed", "-",  1.5),
        ("market",    "Market (S&P 500)",   "#888888", "--", 1.2),
        ("ew",        "Equal Weight",       "#10b981", "--", 1.2),
    ]
    for col, label, color, ls, lw in _styles:
        if col not in returns_df.columns:
            continue
        cum = (1 + returns_df[col]).cumprod() - 1
        ax.plot(cum.index, cum * 100, label=label, color=color, linestyle=ls, linewidth=lw)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def rolling_sharpe_plot(returns_df: pd.DataFrame | None, window: int = 252) -> plt.Figure:
    if returns_df is None or returns_df.empty:
        return _blank_fig()
    fig, ax = plt.subplots(figsize=(10, 4))
    _styles = [
        ("portfolio", "Dynamic Allocation", "#7c3aed", "-",  1.5),
        ("market",    "Market (S&P 500)",   "#888888", "--", 1.2),
        ("ew",        "Equal Weight",       "#10b981", "--", 1.2),
    ]
    for col, label, color, ls, lw in _styles:
        if col not in returns_df.columns:
            continue
        r = returns_df[col]
        roll_std = r.rolling(window).std()
        rs = (r.rolling(window).mean() / roll_std.replace(0, pd.NA)) * sqrt(252)
        ax.plot(rs.index, rs, label=label, color=color, linestyle=ls, linewidth=lw)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylabel(f"Rolling Sharpe Ratio ({window}-day)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def drawdown_plot(returns_df: pd.DataFrame | None) -> plt.Figure:
    if returns_df is None or returns_df.empty:
        return _blank_fig()
    fig, ax = plt.subplots(figsize=(10, 4))
    _styles = [
        ("portfolio", "Dynamic Allocation", "#7c3aed", "-",  1.5),
        ("market",    "Market (S&P 500)",   "#888888", "--", 1.2),
        ("ew",        "Equal Weight",       "#10b981", "--", 1.2),
    ]
    for col, label, color, ls, lw in _styles:
        if col not in returns_df.columns:
            continue
        cum = (1 + returns_df[col]).cumprod()
        dd  = (cum / cum.cummax() - 1) * 100
        ax.plot(dd.index, dd, label=label, color=color, linestyle=ls, linewidth=lw)
        if col == "portfolio":
            ax.fill_between(dd.index, dd, 0, alpha=0.12, color=color)
    ax.set_ylabel("Drawdown (%)")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def realized_te_plot(
    returns_df: pd.DataFrame | None,
    target_te: float,
    window: int = 63,
) -> plt.Figure:
    if (
        returns_df is None
        or returns_df.empty
        or "portfolio" not in returns_df.columns
        or "market" not in returns_df.columns
    ):
        return _blank_fig()
    active   = returns_df["portfolio"] - returns_df["market"]
    realized = active.rolling(window).std() * sqrt(252) * 100
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(realized.index, realized, color="#7c3aed", linewidth=1.5, label="Realized TE")
    ax.axhline(
        target_te * 100,
        color="#dc3545", linewidth=1.2, linestyle="--",
        label=f"Target TE ({target_te * 100:.0f}%)",
    )
    ax.set_ylabel("Annualized Tracking Error (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def portfolio_weights_plot(weights_df: pd.DataFrame | None) -> plt.Figure:
    if weights_df is None or weights_df.empty:
        return _blank_fig()
    cols   = [c for c in _ASSETS if c in weights_df.columns]
    colors = [_COLORS[_ASSETS.index(c)] for c in cols]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stackplot(
        weights_df.index,
        [weights_df[c].values * 100 for c in cols],
        labels=cols,
        colors=colors,
        alpha=0.85,
    )
    ax.set_ylabel("Weight (%)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def regime_plot(
    factor: str,
    regimes_df: pd.DataFrame | None,
    active_returns_df: pd.DataFrame | None,
) -> plt.Figure:
    if regimes_df is None or active_returns_df is None:
        return _blank_fig(f"No regime data for {factor}")
    if factor not in regimes_df.columns or factor not in active_returns_df.columns:
        return _blank_fig(f"No regime data for {factor}")
    active     = active_returns_df[factor]
    cum_active = (1 + active).cumprod() - 1
    idx        = cum_active.index
    labels     = regimes_df[factor].reindex(idx)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(idx, cum_active * 100, color="#7c3aed", linewidth=1.0)
    prev_date, prev_label = idx[0], labels.iloc[0]
    for i in range(1, len(idx)):
        if labels.iloc[i] != prev_label:
            color = "green" if prev_label == 0 else "red"
            ax.axvspan(prev_date, idx[i], alpha=0.2, color=color)
            prev_date, prev_label = idx[i], labels.iloc[i]
    # flush the last segment
    color = "green" if prev_label == 0 else "red"
    ax.axvspan(prev_date, idx[-1], alpha=0.2, color=color)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"{factor.capitalize()} — Regime Detection")
    ax.set_ylabel("Cumulative Active Return (%)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── Metrics comparison HTML ───────────────────────────────────────────────────

def _metrics_comparison_html(df: pd.DataFrame, selected_te: int) -> str:
    if df.empty:
        return "<p class='text-muted'>No results — run the model first.</p>"

    te_vals = [1, 2, 3, 4]
    te_rows: dict[int, pd.Series | None] = {}
    for te in te_vals:
        target = te / 100
        mask   = (df["target_te"] - target).abs() < 1e-6
        te_rows[te] = df[mask].iloc[0] if mask.any() else None

    ew_mask = df["strategy"].str.contains("EW", na=False)
    ew_row  = df[ew_mask].iloc[0] if ew_mask.any() else None

    def _th(label: str, selected: bool, ew: bool = False) -> str:
        bg = "#d4edda" if selected else ("#fff3cd" if ew else "#e8f4ff")
        fw = "700"    if selected else "normal"
        star = " ★"   if selected else ""
        return (
            f"<th style='padding:5px 8px;text-align:right;"
            f"background:{bg};font-weight:{fw}'>TE={label}%{star}</th>"
        )

    header = (
        "<tr style='border-bottom:2px solid #dee2e6'>"
        "<th style='padding:5px 8px;text-align:left'>Metric</th>"
    )
    for te in te_vals:
        header += _th(str(te), te == selected_te)
    header += "<th style='padding:5px 8px;text-align:right;background:#fff3cd'>EW Bench</th></tr>"

    body = ""
    for raw_col, display_col in _DISPLAY.items():
        row_html = f"<tr><td style='padding:5px 8px;font-weight:600'>{display_col}</td>"
        for te in te_vals:
            bg  = "#d4edda" if te == selected_te else "#e8f4ff"
            row = te_rows[te]
            if row is None or raw_col not in row.index:
                row_html += f"<td style='padding:5px 8px;text-align:right;background:{bg}'>—</td>"
            else:
                val = row[raw_col]
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    row_html += f"<td style='padding:5px 8px;text-align:right;background:{bg}'>—</td>"
                    continue
                color = _METRIC_COLORS[display_col](fval)
                fmt   = _format_value(display_col, fval)
                fw    = "700" if te == selected_te else "normal"
                row_html += (
                    f"<td style='padding:5px 8px;text-align:right;"
                    f"background:{bg};color:{color};font-weight:{fw}'>{fmt}</td>"
                )
        if ew_row is not None and raw_col in ew_row.index:
            val = ew_row[raw_col]
            try:
                fval = float(val)
            except (TypeError, ValueError):
                row_html += "<td style='padding:5px 8px;text-align:right;background:#fff3cd'>—</td>"
            else:
                color = _METRIC_COLORS[display_col](fval)
                fmt   = _format_value(display_col, fval)
                row_html += (
                    f"<td style='padding:5px 8px;text-align:right;"
                    f"background:#fff3cd;color:{color}'>{fmt}</td>"
                )
        else:
            row_html += "<td style='padding:5px 8px;text-align:right;background:#fff3cd'>—</td>"
        row_html += "</tr>"
        body += row_html

    return (
        "<table class='table table-sm' style='max-width:700px'>"
        f"<thead>{header}</thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
        "<p style='font-size:10px;color:#6c757d;margin-top:4px'>★ = currently selected TE</p>"
    )


# ── Walk-Forward Evaluation ──────────────────────────────────────────────────

def load_wfe_results(output_dir: Path) -> pd.DataFrame | None:
    path = Path(output_dir) / "wfe_results.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def wfe_metrics_html(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<p class='text-muted'>No walk-forward results — enable Walk-Forward and run the model.</p>"

    numeric_cols = ["sharpe", "ir_vs_market", "max_drawdown", "active_ret_vs_market", "turnover"]
    display_cols = {
        "sharpe":               "Sharpe",
        "ir_vs_market":         "IR vs Mkt",
        "max_drawdown":         "Max DD",
        "active_ret_vs_market": "Active Ret",
        "turnover":             "Turnover",
    }
    color_fn = {
        "Sharpe":     lambda v: "#198754" if v >= 0.5 else ("#dc3545" if v < 0 else "inherit"),
        "IR vs Mkt":  lambda v: "#198754" if v > 0 else "#dc3545",
        "Max DD":     lambda v: "#dc3545",
        "Active Ret": lambda v: "#198754" if v > 0 else "#dc3545",
        "Turnover":   lambda v: "inherit",
    }
    fmt_fn = {
        "Sharpe":     lambda v: f"{v:.3f}",
        "IR vs Mkt":  lambda v: f"{v:.3f}",
        "Max DD":     lambda v: f"{v * 100:.1f}%",
        "Active Ret": lambda v: f"{v * 100:.2f}%",
        "Turnover":   lambda v: f"{v:.4f}",
    }

    fold_rows = df[df["fold"] != "avg"]
    avg_row   = df[df["fold"] == "avg"]

    header = ("<tr style='border-bottom:2px solid #dee2e6'>"
              "<th style='padding:5px 8px'>Fold</th>"
              "<th style='padding:5px 8px'>Period</th>")
    for col in numeric_cols:
        header += f"<th style='padding:5px 8px;text-align:right'>{display_cols[col]}</th>"
    header += "</tr>"

    body = ""
    for _, row in pd.concat([fold_rows, avg_row]).iterrows():
        is_avg = row["fold"] == "avg"
        bg     = "#fff8e1" if is_avg else "#f8f9fa"
        fw     = "700"     if is_avg else "normal"
        try:
            fold_label = "Avg" if is_avg else int(float(row["fold"]))
        except (TypeError, ValueError):
            fold_label = str(row["fold"])
        period_label = "" if is_avg else (
            f"{str(row['period_start'])[:7]} – {str(row['period_end'])[:7]}"
        )
        row_html = (f"<tr style='background:{bg}'>"
                    f"<td style='padding:5px 8px;font-weight:{fw}'>{fold_label}</td>"
                    f"<td style='padding:5px 8px;color:#888;font-size:11px'>{period_label}</td>")
        for col in numeric_cols:
            disp  = display_cols[col]
            val   = row.get(col, float("nan"))
            try:
                fval  = float(val)
                color = color_fn[disp](fval)
                txt   = fmt_fn[disp](fval)
            except (TypeError, ValueError):
                color, txt = "inherit", "—"
            row_html += (f"<td style='padding:5px 8px;text-align:right;"
                         f"color:{color};font-weight:{fw}'>{txt}</td>")
        row_html += "</tr>"
        body += row_html

    return (f"<table class='table table-sm' style='max-width:750px'>"
            f"<thead>{header}</thead><tbody>{body}</tbody></table>")


def wfe_folds_plot(df: pd.DataFrame, metric: str = "sharpe") -> plt.Figure:
    """Bar chart of a metric across walk-forward folds. Avg shown as dashed line."""
    fold_rows = df[df["fold"] != "avg"].copy()
    avg_rows  = df[df["fold"] == "avg"]

    if fold_rows.empty:
        return _blank_fig("No fold data")

    fold_rows["fold"] = fold_rows["fold"].astype(str)
    if metric not in fold_rows.columns:
        return _blank_fig(f"Metric '{metric}' not in data")
    values = fold_rows[metric].astype(float).values
    labels = [f"F{r}" for r in fold_rows["fold"]]
    colors = ["#198754" if v >= 0 else "#dc3545" for v in values]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.5)

    if not avg_rows.empty:
        try:
            avg_val = float(avg_rows.iloc[0][metric])
            ax.axhline(avg_val, color="#7c3aed", linewidth=1.5, linestyle="--",
                       label=f"Avg {avg_val:.3f}")
            ax.legend(fontsize=9)
        except (KeyError, TypeError, ValueError):
            pass

    display = {"sharpe": "Sharpe Ratio", "ir_vs_market": "IR vs Market",
               "max_drawdown": "Max Drawdown", "active_ret_vs_market": "Active Return",
               "volatility": "Volatility", "turnover": "Turnover"}
    ax.set_ylabel(display.get(metric, metric))
    ax.set_title(f"{display.get(metric, metric)} by Walk-Forward Fold")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig
