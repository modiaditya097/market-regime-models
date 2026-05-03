"""
SC-HMM Phase 3 — Dashboard Output Exporter
===========================================
Run this script AFTER executing the SC-HMM notebook (all cells).

It reads configurable parameters from environment variables so the
Shiny dashboard's Re-run button can pass user-selected values:

    SCHMM_ASSETS               comma-separated, e.g. "SPY,IWM,IEF,TIP,GLD"
    SCHMM_K_MIN                int (default 2)
    SCHMM_K_MAX                int (default 3)
    SCHMM_INITIAL_WINDOW       int weeks (default 104)
    SCHMM_REFIT_CADENCE        int weeks (default 4)
    SCHMM_SMOOTH_WINDOW        int weeks (default 3)
    SCHMM_MACRO_NEUTRAL_THRESH float (default 0.5)
    SCHMM_MACRO_EXTREME_THRESH float (default 1.5)
    SCHMM_TC_BPS               int basis points (default 10)

Usage from notebook (last cell):
    exec(open('export_sc_hmm_outputs.py').read())

Usage from terminal (with custom params):
    SCHMM_K_MIN=2 SCHMM_K_MAX=3 python export_sc_hmm_outputs.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from collections import Counter

# ── Read parameters from env (with defaults matching notebook) ─────────────────
_ASSETS_STR  = os.environ.get("SCHMM_ASSETS", "SPY,IWM,IEF,TIP,GLD")
_ASSETS      = [a.strip() for a in _ASSETS_STR.split(",")]
_K_MIN       = int(os.environ.get("SCHMM_K_MIN", 2))
_K_MAX       = int(os.environ.get("SCHMM_K_MAX", 3))
_INI_WIN     = int(os.environ.get("SCHMM_INITIAL_WINDOW", 104))
_REFIT       = int(os.environ.get("SCHMM_REFIT_CADENCE", 4))
_SMOOTH      = int(os.environ.get("SCHMM_SMOOTH_WINDOW", 3))
_MAC_NEU     = float(os.environ.get("SCHMM_MACRO_NEUTRAL_THRESH", 0.5))
_MAC_EXT     = float(os.environ.get("SCHMM_MACRO_EXTREME_THRESH", 1.5))
_TC_BPS      = int(os.environ.get("SCHMM_TC_BPS", 10))

print("SC-HMM Phase 3 — Dashboard Output Exporter")
print(f"  Assets              : {_ASSETS}")
print(f"  K range             : [{_K_MIN}, {_K_MAX}]")
print(f"  Initial window      : {_INI_WIN} weeks")
print(f"  Refit cadence       : {_REFIT} weeks")
print(f"  Smooth window       : {_SMOOTH} weeks")
print(f"  Macro neutral thresh: {_MAC_NEU}")
print(f"  Macro extreme thresh: {_MAC_EXT}")
print(f"  Transaction cost    : {_TC_BPS} bps")
print()

# ── Output directories ─────────────────────────────────────────────────────────
OUTPUT_DIR = Path("outputs/model3")
PLOTS_DIR  = OUTPUT_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR.resolve()}")

# ── Colour palette ─────────────────────────────────────────────────────────────
C = {
    "hard":    "#2c3e50",
    "overlay": "#8e44ad",
    "spy":     "#2980b9",
    "rp":      "#e67e22",
    "s6040":   "#16a085",
    "s5a":     "#d35400",
    "bull":    "#2ecc71",
    "neutral": "#f39c12",
    "bear":    "#e74c3c",
}

# ── 1. returns_te3.csv  (Comparison tab) ───────────────────────────────────────
returns_df = pd.DataFrame({
    "date":      dates_bt,
    "portfolio": ret_sm_hard,
    "market":    bh_spy,
    "ew":        ew_5,
})
returns_df.to_csv(OUTPUT_DIR / "returns_te3.csv", index=False)
print("✓ returns_te3.csv")

# ── 2. Performance tables ──────────────────────────────────────────────────────
STRAT_LIST = [
    ("* SC-HMM",              ret_sm_hard),
    ("* SC-HMM + Macro Overlay", ret_sm_overlay),
    ("  Buy-Hold SPY",        bh_spy),
    ("  Risk Parity (inv-vol)", rp_ret),
    ("  Static 60/40",        static_6040),
    ("  Static 5-Asset Blend",static_5a),
]

def _make_metrics(mask):
    rows = []
    for name, series in STRAT_LIST:
        row = {"Strategy": name}
        row.update(compute_metrics(series[mask]))
        rows.append(row)
    return pd.DataFrame(rows)

full_m  = np.ones(len(dates_bt), dtype=bool)
train_m = dates_bt <= pd.Timestamp(TRAIN_END)
test_m  = dates_bt >= pd.Timestamp(TEST_START)

_make_metrics(full_m).to_csv(OUTPUT_DIR / "metrics_full.csv",  index=False)
_make_metrics(train_m).to_csv(OUTPUT_DIR / "metrics_train.csv", index=False)
_make_metrics(test_m).to_csv(OUTPUT_DIR / "metrics_test.csv",  index=False)
print("✓ metrics_full/train/test.csv")

# ── 3. Stress & Bull period tables ─────────────────────────────────────────────
STRAT_SUB = {
    "SC-HMM Hard":    ret_sm_hard,
    "Macro Overlay":  ret_sm_overlay,
    "SPY B&H":        bh_spy,
    "Risk Parity":    rp_ret,
    "Static 60/40":   static_6040,
}

def _period_df(periods):
    rows = []
    for pname, (s, e) in periods.items():
        mask = (dates_bt >= s) & (dates_bt <= e)
        if mask.sum() < 3:
            continue
        row = {"Period": pname}
        for sname, series in STRAT_SUB.items():
            tot = float((np.cumprod(1 + series[mask]) - 1)[-1])
            row[sname] = f"{tot:.1%}"
        rows.append(row)
    return pd.DataFrame(rows)

_period_df(STRESS_PERIODS).to_csv(OUTPUT_DIR / "stress_table.csv", index=False)
_period_df(BULL_PERIODS).to_csv(OUTPUT_DIR / "bull_table.csv",   index=False)
print("✓ stress_table.csv, bull_table.csv")

# ── 4. Macro latest snapshot ───────────────────────────────────────────────────
latest = macro.iloc[-1]
vix    = float(latest["VIX"])
curve  = float(latest["T10Y2Y"])
hy     = float(latest["BAMLH0A0HYM2"])

def _sig_vix(v):   return "bear" if v > 25 else "bull" if v < 16 else "neutral"
def _sig_curve(c): return "bear" if c < 0 else "bull" if c > 1.5 else "neutral"
def _sig_hy(h):    return "bear" if h > 4.5 else "bull" if h < 3.0 else "neutral"

pd.DataFrame([
    {"indicator": "VIX",              "value": f"{vix:.1f}",    "signal": _sig_vix(vix)},
    {"indicator": "Yield Curve 10Y-2Y","value": f"{curve:+.2f}","signal": _sig_curve(curve)},
    {"indicator": "HY Spread",         "value": f"{hy:.2f}%",   "signal": _sig_hy(hy)},
]).to_csv(OUTPUT_DIR / "macro_latest.csv", index=False)
print("✓ macro_latest.csv")

# ── 5. K analysis ──────────────────────────────────────────────────────────────
K_arr = np.array(store.get("K_at_t", []))
if len(K_arr) > 0:
    pd.DataFrame([
        {"K": 2, "Frequency": f"{(K_arr==2).mean():.1%}", "Weeks": int((K_arr==2).sum())},
        {"K": 3, "Frequency": f"{(K_arr==3).mean():.1%}", "Weeks": int((K_arr==3).sum())},
    ]).to_csv(OUTPUT_DIR / "k_analysis.csv", index=False)
    print("✓ k_analysis.csv")

# ── 6. Plots ───────────────────────────────────────────────────────────────────

# 6a. Cumulative Returns
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(dates_bt, cum_ret(ret_sm_hard),    color=C["hard"],    lw=2.2, label="SC-HMM")
ax.plot(dates_bt, cum_ret(ret_sm_overlay), color=C["overlay"], lw=2.0, ls="--", label="Macro Overlay")
ax.plot(dates_bt, cum_ret(bh_spy),         color=C["spy"],     lw=1.8, alpha=0.8, label="Buy-Hold SPY")
ax.plot(dates_bt, cum_ret(rp_ret),         color=C["rp"],      lw=1.5, label="Risk Parity")
ax.plot(dates_bt, cum_ret(static_6040),    color=C["s6040"],   lw=1.3, label="Static 60/40")
ax.plot(dates_bt, cum_ret(static_5a),      color=C["s5a"],     lw=1.3, label="Static 5-Asset")
ax.axvline(pd.Timestamp(TEST_START), color="black", ls=":", lw=1.5, label=f"OOS ({TEST_START})")
for _, (s, e) in STRESS_PERIODS.items():
    ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.07, color="red")
ax.set_title("SC-HMM Phase 3: Cumulative Return (red = stress periods)", fontsize=12, fontweight="bold")
ax.set_ylabel("Cumulative Return")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.legend(fontsize=9, ncol=3); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "cumulative_returns.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ plots/cumulative_returns.png")

# 6b. Drawdown
fig, ax = plt.subplots(figsize=(14, 4))
ax.fill_between(dates_bt, drawdown_series(ret_sm_hard),    0, alpha=0.5, color=C["hard"],    label="SC-HMM")
ax.fill_between(dates_bt, drawdown_series(ret_sm_overlay), 0, alpha=0.3, color=C["overlay"], label="Macro Overlay")
ax.plot(dates_bt, drawdown_series(bh_spy), color=C["spy"], lw=1.2, label="SPY")
ax.plot(dates_bt, drawdown_series(rp_ret), color=C["rp"],  lw=1.2, label="Risk Parity")
ax.axvline(pd.Timestamp(TEST_START), color="black", ls=":", lw=1.5)
ax.set_title("SC-HMM Phase 3: Drawdown", fontsize=12, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "drawdown.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ plots/drawdown.png")

# 6c. Rolling Sharpe
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(dates_bt, rolling_sharpe(ret_sm_hard),    color=C["hard"],    lw=1.8, label="SC-HMM")
ax.plot(dates_bt, rolling_sharpe(ret_sm_overlay), color=C["overlay"], lw=1.8, ls="--", label="Macro Overlay")
ax.plot(dates_bt, rolling_sharpe(bh_spy),         color=C["spy"],     lw=1.3, alpha=0.8, label="SPY")
ax.plot(dates_bt, rolling_sharpe(rp_ret),         color=C["rp"],      lw=1.3, alpha=0.8, label="Risk Parity")
ax.plot(dates_bt, rolling_sharpe(static_6040),    color=C["s6040"],   lw=1.2, alpha=0.7, label="60/40")
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.axhline(1, color="green", lw=0.8, ls=":", alpha=0.6)
ax.axvline(pd.Timestamp(TEST_START), color="black", ls=":", lw=1.5)
ax.set_title("SC-HMM Phase 3: Rolling 52-Week Sharpe", fontsize=12, fontweight="bold")
ax.legend(fontsize=9, ncol=3); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "rolling_sharpe.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ plots/rolling_sharpe.png")

# 6d. Portfolio Weights (both strategies stacked)
w_hard_arr    = np.array(store["w_hard"])
w_overlay_arr = np.array(store["w_overlay"])
colors5 = ["#2980b9","#27ae60","#e74c3c","#8e44ad","#f1c40f"]

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle("SC-HMM Phase 3: Portfolio Weights Over Time", fontsize=12, fontweight="bold")
for ax, w_arr, title in zip(axes,
    [w_hard_arr, w_overlay_arr],
    ["SC-HMM (Hard)", "SC-HMM + Macro Overlay"]):
    ax.stackplot(dates_bt, *[w_arr[:, i] for i in range(len(_ASSETS))],
                 labels=_ASSETS, colors=colors5[:len(_ASSETS)], alpha=0.85)
    ax.axvline(pd.Timestamp(TEST_START), color="black", ls=":", lw=1.5, label="OOS")
    ax.set_title(title, fontsize=10); ax.set_ylim(0, 1)
    ax.legend(ncol=6, fontsize=8, loc="upper left"); ax.grid(alpha=0.2)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "portfolio_weights.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ plots/portfolio_weights.png")

# 6e. Regime Timeline
fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
fig.suptitle("SC-HMM Phase 3: Regime Timeline", fontsize=12, fontweight="bold")
for ax, reg_arr, title in zip(axes,
    [reg_smooth, np.array(store["overlay_reg"])],
    ["SC-HMM (Hard)", "Macro Overlay"]):
    for reg, col in REGIME_COLORS.items():
        ax.fill_between(dates_bt, 0, 1, where=(reg_arr == reg),
                        color=col, alpha=0.7,
                        transform=ax.get_xaxis_transform(), label=reg)
    spy_sc = cum_ret(bh_spy)
    spy_sc = (spy_sc - spy_sc.min()) / (spy_sc.max() - spy_sc.min() + 1e-10)
    ax.plot(dates_bt, spy_sc, "k-", lw=0.7, alpha=0.5, label="SPY (scaled)")
    ax.axvline(pd.Timestamp(TEST_START), color="black", ls=":", lw=1.5)
    ax.set_title(title, fontsize=9); ax.set_yticks([])
    ax.legend(loc="upper left", ncol=5, fontsize=8)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "regime_timeline.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ plots/regime_timeline.png")

# 6f. Transition Matrix
fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(P_final, cmap="Blues", vmin=0, vmax=1)
for i in range(P_final.shape[0]):
    for j in range(P_final.shape[1]):
        v = P_final[i, j]
        ax.text(j, i, f"{v:.1%}", ha="center", va="center",
                fontsize=12, fontweight="bold" if i == j else "normal",
                color="white" if v > 0.55 else "#1a1a2e")
lbs = list(REGIME_COLORS.keys())
ax.set_xticks(range(len(lbs))); ax.set_yticks(range(len(lbs)))
ax.set_xticklabels([f"→{l.capitalize()}" for l in lbs], fontsize=9)
ax.set_yticklabels([l.capitalize() for l in lbs], fontsize=9)
ax.set_title("Transition Matrix P[i→j]", fontsize=11, fontweight="bold")
for i in range(len(lbs)):
    ax.add_patch(plt.Rectangle((i-.5, i-.5), 1, 1, fill=False, edgecolor="#e74c3c", lw=2))
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "transition_matrix.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ plots/transition_matrix.png")

# 6g. Annual Returns Heatmap
import matplotlib.colors as mcolors
strats_ann = {
    "SC-HMM":       ret_sm_hard,
    "Macro Overlay":ret_sm_overlay,
    "Buy-Hold SPY": bh_spy,
    "Risk Parity":  rp_ret,
    "Static 60/40": static_6040,
    "5-Asset Blend":static_5a,
}
years = sorted(dates_bt.year.unique())
ann_data = {}
for name, series in strats_ann.items():
    ann_data[name] = {}
    for yr in years:
        mask = dates_bt.year == yr
        ann_data[name][yr] = (np.nan if mask.sum() < 4
                              else np.cumprod(1 + series[mask])[-1] - 1)
df_ann = pd.DataFrame(ann_data).T
vmax = max(abs(df_ann.values[~np.isnan(df_ann.values)]).max(), 0.01)
cmap = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
fig, ax = plt.subplots(figsize=(max(14, len(years) * 0.85), len(strats_ann) * 0.85 + 1.5))
fig.suptitle("Annual Returns by Strategy", fontsize=12, fontweight="bold")
im2 = ax.imshow(df_ann.values, cmap="RdYlGn", norm=cmap, aspect="auto")
ax.set_xticks(range(len(years))); ax.set_xticklabels(years, rotation=45, fontsize=8)
ax.set_yticks(range(len(strats_ann))); ax.set_yticklabels(list(strats_ann.keys()), fontsize=9)
for i in range(len(strats_ann)):
    for j, yr in enumerate(years):
        v = df_ann.iloc[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                    fontsize=7, color="white" if abs(v) > vmax * 0.5 else "black")
plt.colorbar(im2, ax=ax, fraction=0.02, pad=0.02,
             format=mticker.PercentFormatter(1.0))
plt.tight_layout()
fig.savefig(PLOTS_DIR / "annual_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ plots/annual_heatmap.png")

# 6h. Risk-Return Scatter
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Risk-Return Profile — Full Period & OOS", fontsize=12, fontweight="bold")
sc_colors_map = {
    "SC-HMM":       C["hard"],
    "Macro Overlay":C["overlay"],
    "Buy-Hold SPY": C["spy"],
    "Risk Parity":  C["rp"],
    "Static 60/40": C["s6040"],
    "5-Asset Blend":C["s5a"],
}
for ax, mask, period_label in zip(
        axes,
        [np.ones(len(dates_bt), dtype=bool), test_m],
        [f"Full Period ({dates_bt[0].year}–{dates_bt[-1].year})",
         f"OOS Only (≥ {TEST_START})"]):
    for name, series in strats_ann.items():
        m = compute_metrics(series[mask])
        vol  = float(str(m.get("Ann.Vol", "0")).replace("%", "")) / 100
        cagr = float(str(m.get("CAGR", "0")).replace("%", "")) / 100
        col  = sc_colors_map.get(name, "#888888")
        star = name in ("SC-HMM", "Macro Overlay")
        ax.scatter(vol, cagr, color=col, s=140 if star else 80,
                   zorder=5, marker="*" if star else "o")
        ax.annotate(name, (vol, cagr), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color=col)
    ax.axhline(0, color="black", lw=0.7, ls="--")
    ax.set_xlabel("Annualised Volatility"); ax.set_ylabel("CAGR")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title(period_label, fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "risk_return_scatter.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ plots/risk_return_scatter.png")

print(f"\n✅ All dashboard outputs saved to: {OUTPUT_DIR.resolve()}")
print("   Launch dashboard: PYTHONPATH=. shiny run shiny_app/app.py --reload")

# ── 7. results.csv  (Comparison tab metrics table) ─────────────────────────────
# The comparison tab reads results.csv with these EXACT columns:
#   target_te, sharpe, ir_vs_market, max_drawdown, volatility,
#   active_ret_vs_market, turnover
#
# We produce one row for TE=3% (our only target), using SC-HMM Hard as
# the representative strategy.

def _scalar(series, mask=None):
    s = series[mask] if mask is not None else series
    ann    = float(np.mean(s) * ANNUALISE)
    vol    = float(np.std(s, ddof=1) * np.sqrt(ANNUALISE))
    sharpe = ann / vol if vol > 1e-10 else 0.0
    cum    = np.cumprod(1 + s.values)
    roll_max = np.maximum.accumulate(cum)
    dd_series = cum / roll_max - 1
    max_dd = float(dd_series.min()) * 100   # in %

    # Active return vs SPY
    mkt   = bh_spy[mask] if mask is not None else bh_spy
    act   = float((np.mean(s) - np.mean(mkt)) * ANNUALISE)

    # IR = active_return / tracking_error
    te    = float(np.std((s - mkt).values, ddof=1) * np.sqrt(ANNUALISE))
    ir    = act / te if te > 1e-10 else 0.0

    # Turnover — average absolute weekly weight change
    w_arr = np.array(store["w_hard"])
    turnover = float(np.mean(np.sum(np.abs(np.diff(w_arr, axis=0)), axis=1)) * ANNUALISE)

    return {
        "target_te":           0.03,
        "sharpe":              round(sharpe, 3),
        "ir_vs_market":        round(ir, 3),
        "max_drawdown":        round(max_dd, 2),
        "volatility":          round(vol * 100, 2),
        "active_ret_vs_market":round(act * 100, 2),
        "turnover":            round(turnover, 3),
    }

full_mask_arr = np.ones(len(ret_sm_hard), dtype=bool)
results_row = _scalar(ret_sm_hard, full_mask_arr)
pd.DataFrame([results_row]).to_csv(OUTPUT_DIR / "results.csv", index=False)
print("✓ results.csv  (Comparison tab metrics)")
