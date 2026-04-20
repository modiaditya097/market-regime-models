"""
SC-HMM Dashboard Output Exporter
=================================
Run this script AFTER executing your SC-HMM notebook (all cells).
It saves all plots, CSVs, and tables that the Shiny dashboard (model3) needs.

Usage (from your project directory):
    python export_sc_hmm_outputs.py

The script expects these variables to already exist in the Python session
(i.e. the notebook must have been run first):
    dates_bt, ret_sm_hard, ret_sm_soft, ret_sm_macro,
    bh_spy, bh_shy, bh_gld, ew_4, rp_ret, static,
    reg_smooth, mac_sig_r, P_final, macro,
    bt_ret, w_sm_hard, store,
    STRESS_PERIODS, BULL_PERIODS, REGIME_COLORS, REGIME_ORDER,
    ANNUALISE, TEST_START, TRAIN_END, assets,
    compute_metrics, cum_ret, drawdown_series, rolling_sharpe

HOW TO ADD THIS TO YOUR NOTEBOOK:
  1. Open SC_HMM_Phase2_Final.ipynb
  2. Add a new cell at the very end (after Cell 16)
  3. Paste the code below (or use: exec(open('export_sc_hmm_outputs.py').read()))
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── Output directories ─────────────────────────────────────────────────────────
OUTPUT_DIR = Path("outputs/model3")
PLOTS_DIR  = OUTPUT_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR.resolve()}")


# ── 1. returns_te3.csv  (for the Comparison tab) ──────────────────────────────
returns_df = pd.DataFrame({
    "date":      dates_bt,
    "portfolio": ret_sm_hard,   # SC-HMM Hard is the representative strategy
    "market":    bh_spy,
    "ew":        ew_4,
})
returns_df.to_csv(OUTPUT_DIR / "returns_te3.csv", index=False)
print("✓ returns_te3.csv")


# ── 2. Performance tables ──────────────────────────────────────────────────────
STRATEGIES = [
    ("SC-HMM Hard (net)",    ret_sm_hard),
    ("SC-HMM Soft (net)",    ret_sm_soft),
    ("SC-HMM Macro (net)",   ret_sm_macro),
    ("Buy-Hold SPY",         bh_spy),
    ("Buy-Hold SHY",         bh_shy),
    ("Buy-Hold GLD",         bh_gld),
    ("Equal-Weight 4A",      ew_4),
    ("Risk Parity",          rp_ret),
    ("Static 30/10/30/30",   static),
]

def make_metrics_df(mask):
    rows = []
    for name, series in STRATEGIES:
        row = {"Strategy": name}
        row.update(compute_metrics(series[mask]))
        rows.append(row)
    return pd.DataFrame(rows)

full_m  = np.ones(len(dates_bt), dtype=bool)
train_m = dates_bt <= pd.Timestamp(TRAIN_END)
test_m  = dates_bt >= pd.Timestamp(TEST_START)

make_metrics_df(full_m).to_csv(OUTPUT_DIR / "metrics_full.csv",  index=False)
make_metrics_df(train_m).to_csv(OUTPUT_DIR / "metrics_train.csv", index=False)
make_metrics_df(test_m).to_csv(OUTPUT_DIR / "metrics_test.csv",  index=False)
print("✓ metrics_full/train/test.csv")


# ── 3. Stress & Bull period tables ─────────────────────────────────────────────
STRAT_SUBSET = {
    "SC-HMM Hard": ret_sm_hard,
    "SC-HMM Soft": ret_sm_soft,
    "Macro-Filt":  ret_sm_macro,
    "SPY B&H":     bh_spy,
    "Risk Parity": rp_ret,
}

def make_period_df(periods):
    rows = []
    for pname, (s, e) in periods.items():
        mask = (dates_bt >= s) & (dates_bt <= e)
        if mask.sum() < 3:
            continue
        row = {"Period": pname}
        for sname, series in STRAT_SUBSET.items():
            tot = float((np.cumprod(1 + series[mask]) - 1)[-1])
            row[sname] = f"{tot:.1%}"
        rows.append(row)
    return pd.DataFrame(rows)

make_period_df(STRESS_PERIODS).to_csv(OUTPUT_DIR / "stress_table.csv", index=False)
make_period_df(BULL_PERIODS).to_csv(OUTPUT_DIR / "bull_table.csv",   index=False)
print("✓ stress_table.csv, bull_table.csv")


# ── 4. Macro latest snapshot ───────────────────────────────────────────────────
latest_macro = macro.iloc[-1]
vix   = float(latest_macro["VIX"])
curve = float(latest_macro["T10Y2Y"])
hy    = float(latest_macro["HY_SPREAD"])

def _sig(val, bear, bull, higher_is_bad=True):
    if higher_is_bad:
        if val > bear: return "bear"
        if val < bull: return "bull"
    else:
        if val < bear: return "bear"
        if val > bull: return "bull"
    return "neutral"

macro_df = pd.DataFrame([
    {"indicator": "VIX",             "value": f"{vix:.1f}",   "signal": _sig(vix,  25.0, 16.0, True)},
    {"indicator": "Yield Curve 10Y-2Y","value": f"{curve:+.2f}", "signal": _sig(curve, 0.0, 1.5,  False)},
    {"indicator": "HY Spread",        "value": f"{hy:.2f}%",  "signal": _sig(hy,   4.5, 3.0,  True)},
])
macro_df.to_csv(OUTPUT_DIR / "macro_latest.csv", index=False)
print("✓ macro_latest.csv")


# ── 5. Plots ───────────────────────────────────────────────────────────────────

# 5a. Cumulative Returns
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(dates_bt, cum_ret(ret_sm_hard),  color="#2c3e50", lw=2.2, label="SC-HMM Hard (net)")
ax.plot(dates_bt, cum_ret(ret_sm_soft),  color="#8e44ad", lw=1.8, ls="-.", label="SC-HMM Soft (net)")
ax.plot(dates_bt, cum_ret(ret_sm_macro), color="#16a085", lw=2.0, ls="--", label="SC-HMM Macro (net)")
ax.plot(dates_bt, cum_ret(bh_spy),       color="#2980b9", lw=1.8, alpha=0.8, label="Buy-Hold SPY")
ax.plot(dates_bt, cum_ret(rp_ret),       color="#e67e22", lw=1.5, label="Risk Parity")
ax.plot(dates_bt, cum_ret(ew_4),         color="#27ae60", lw=1.2, alpha=0.7, label="Equal-Weight 4A")
ax.axvline(pd.Timestamp(TEST_START), color="black", ls=":", lw=1.5, label=f"OOS start ({TEST_START})")
for _, (s, e) in STRESS_PERIODS.items():
    ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.07, color="red")
ax.set_title("SC-HMM: Cumulative Return (red = stress periods)", fontsize=12, fontweight="bold")
ax.set_ylabel("Cumulative Return")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.legend(fontsize=9, ncol=3); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "cumulative_returns.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ plots/cumulative_returns.png")

# 5b. Drawdown
fig, ax = plt.subplots(figsize=(14, 4))
ax.fill_between(dates_bt, drawdown_series(ret_sm_hard), 0, alpha=0.55, color="#2c3e50", label="SC-HMM Hard")
ax.fill_between(dates_bt, drawdown_series(ret_sm_soft), 0, alpha=0.25, color="#8e44ad", label="SC-HMM Soft")
ax.plot(dates_bt, drawdown_series(bh_spy), color="#2980b9", lw=1.2, label="SPY")
ax.plot(dates_bt, drawdown_series(rp_ret), color="#e67e22", lw=1.2, label="Risk Parity")
ax.axvline(pd.Timestamp(TEST_START), color="black", ls=":", lw=1.5)
ax.set_title("SC-HMM: Drawdown", fontsize=12, fontweight="bold")
ax.set_ylabel("Drawdown")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "drawdown.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ plots/drawdown.png")

# 5c. Rolling Sharpe
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(dates_bt, rolling_sharpe(ret_sm_hard),  color="#2c3e50", lw=1.5, label="SC-HMM Hard")
ax.plot(dates_bt, rolling_sharpe(ret_sm_soft),  color="#8e44ad", lw=1.5, ls="-.", label="SC-HMM Soft")
ax.plot(dates_bt, rolling_sharpe(ret_sm_macro), color="#16a085", lw=1.5, ls="--", label="SC-HMM Macro")
ax.plot(dates_bt, rolling_sharpe(bh_spy),       color="#2980b9", lw=1.5, alpha=0.8, label="SPY")
ax.plot(dates_bt, rolling_sharpe(rp_ret),       color="#e67e22", lw=1.2, alpha=0.8, label="Risk Parity")
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.axhline(1, color="green", lw=0.8, ls=":", alpha=0.6)
ax.axvline(pd.Timestamp(TEST_START), color="black", ls=":", lw=1.5)
ax.set_title("SC-HMM: Rolling 52-Week Sharpe Ratio", fontsize=12, fontweight="bold")
ax.set_ylabel("Sharpe"); ax.legend(fontsize=9, ncol=3); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "rolling_sharpe.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ plots/rolling_sharpe.png")

# 5d. Portfolio Weights (Hard)
w_hard_arr = np.array(store["w_hard"])
fig, ax = plt.subplots(figsize=(14, 4))
ax.stackplot(dates_bt,
             w_hard_arr[:,0], w_hard_arr[:,1], w_hard_arr[:,2], w_hard_arr[:,3],
             labels=["SPY", "IWM", "SHY", "GLD"],
             colors=["#2980b9", "#27ae60", "#e74c3c", "#f1c40f"], alpha=0.85)
ax.axvline(pd.Timestamp(TEST_START), color="black", ls=":", lw=1.5, label="OOS")
ax.set_title("SC-HMM: Portfolio Weights — Hard Strategy", fontsize=12, fontweight="bold")
ax.set_ylabel("Weight"); ax.set_ylim(0, 1)
ax.legend(ncol=5, fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "portfolio_weights.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ plots/portfolio_weights.png")

# 5e. Regime Timeline
fig, ax = plt.subplots(figsize=(14, 3))
spy_cs = cum_ret(bh_spy)
spy_sc = (spy_cs - spy_cs.min()) / (spy_cs.max() - spy_cs.min() + 1e-10)
for reg, col in REGIME_COLORS.items():
    ax.fill_between(dates_bt, 0, 1,
                    where=(reg_smooth == reg),
                    color=col, alpha=0.65,
                    transform=ax.get_xaxis_transform(),
                    label=f"{reg}")
ax.plot(dates_bt, spy_sc, "k-", lw=0.8, alpha=0.6, label="SPY (scaled)")
ax.axvline(pd.Timestamp(TEST_START), color="black", ls=":", lw=1.5, label="OOS start")
ax.set_title("SC-HMM: Regime Timeline (smoothed)", fontsize=12, fontweight="bold")
ax.set_yticks([])
ax.legend(loc="upper left", ncol=5, fontsize=9)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "regime_timeline.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ plots/regime_timeline.png")

# 5f. Transition Matrix
fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(P_final, cmap="Blues", vmin=0, vmax=1)
regime_labels = ["Bull", "Neutral", "Bear"]
for i in range(3):
    for j in range(3):
        v = P_final[i, j]
        ax.text(j, i, f"{v:.1%}", ha="center", va="center", fontsize=13,
                color="white" if v > 0.55 else "#1a1a2e",
                fontweight="bold" if i == j else "normal")
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(["→ Bull", "→ Neutral", "→ Bear"], fontsize=10)
ax.set_yticklabels(["Bull", "Neutral", "Bear"], fontsize=10)
ax.set_title("Transition Matrix P[i→j]", fontsize=11, fontweight="bold")
for i in range(3):
    ax.add_patch(plt.Rectangle((i-.5, i-.5), 1, 1, fill=False, edgecolor="#e74c3c", lw=2))
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "transition_matrix.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ plots/transition_matrix.png")

print("\n✅ All dashboard outputs saved to:", OUTPUT_DIR.resolve())
print("   You can now run the Shiny dashboard:")
print("   shiny run shiny_app/app.py --reload")
