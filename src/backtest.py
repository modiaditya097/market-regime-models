"""Backtest engine: daily returns, transaction costs, metrics, and plots."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import (
    ASSETS, FACTORS, TRADING_DAYS,
    annualize_return, annualize_vol, sharpe_ratio, info_ratio, max_drawdown,
)


def compute_transaction_costs(weights: pd.DataFrame, cost_bps: float) -> pd.Series:
    """
    Transaction cost = cost_bps/10000 per side × one-way turnover.
    Turnover on day t = sum(|w[t] - w[t-1]|) / 2  (one-way).
    """
    cost_rate = cost_bps / 10_000
    turnover = weights.diff().abs().sum(axis=1) / 2.0
    return turnover * cost_rate * 2  # buy + sell


def compute_portfolio_returns(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    cost_bps: float,
) -> pd.Series:
    """
    Daily portfolio return = dot(prev_weights, today_returns) - transaction_cost.
    First row is NaN (no previous weight).
    """
    w_prev   = weights.shift(1)
    gross    = (w_prev * asset_returns).sum(axis=1)
    costs    = compute_transaction_costs(weights, cost_bps)
    return (gross - costs)


def compute_ew_returns(asset_returns: pd.DataFrame) -> pd.Series:
    """Equal-weight daily return (1/N of each asset)."""
    return asset_returns.mean(axis=1).rename("ew")


def compute_performance_table(
    port_ret: pd.Series,
    mkt_ret: pd.Series,
    ew_ret: pd.Series,
    rf: pd.Series,
    weights: pd.DataFrame,
) -> dict:
    """Annualized performance metrics vs market and vs EW benchmark."""
    # Drop first NaN row
    idx = port_ret.dropna().index
    p = port_ret.loc[idx]
    m = mkt_ret.reindex(idx)
    e = ew_ret.reindex(idx)
    r = rf.reindex(idx)

    turnover = weights.diff().abs().sum(axis=1).mean() / 2.0 * TRADING_DAYS

    return {
        "excess_return":       annualize_return(p - r),
        "sharpe":              sharpe_ratio(p, r),
        "max_drawdown":        max_drawdown(p),
        "volatility":          annualize_vol(p),
        "active_ret_vs_market": annualize_return(p - m),
        "ir_vs_market":        info_ratio(p - m),
        "mdd_vs_market":       max_drawdown(p - m),
        "active_ret_vs_ew":    annualize_return(p - e),
        "ir_vs_ew":            info_ratio(p - e),
        "turnover":            float(turnover),
    }


def save_results(metrics_rows: list, output_path: str) -> None:
    """Save a list of metric dicts (one per TE target) to CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df = pd.DataFrame(metrics_rows)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def plot_cumulative_returns(
    port_ret: pd.Series,
    mkt_ret: pd.Series,
    ew_ret: pd.Series,
    rf: pd.Series,
    output_dir: str,
    label: str = "Dynamic Allocation",
) -> None:
    idx = port_ret.dropna().index
    cum = lambda r: (1 + r.reindex(idx) - rf.reindex(idx)).cumprod() - 1

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(idx, cum(port_ret) * 100, label=label, linewidth=1.5)
    ax.plot(idx, cum(ew_ret)   * 100, label="EW Benchmark", linewidth=1.2, linestyle="--")
    ax.plot(idx, cum(mkt_ret)  * 100, label="Market", linewidth=1.0, linestyle=":")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Cumulative Excess Return (%)")
    ax.set_title("Cumulative Excess Returns (vs Risk-Free Rate)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "cumulative_returns.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_regime(
    active_ret: pd.Series,
    regime_labels: pd.Series,
    factor_name: str,
    output_dir: str,
) -> None:
    cum_active = (1 + active_ret).cumprod() - 1

    # Align
    idx = cum_active.index
    labels = regime_labels.reindex(idx)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(idx, cum_active * 100, color="steelblue", linewidth=1.0)

    # Shade bull (0) and bear (1) regions
    prev_date, prev_label = idx[0], labels.iloc[0]
    for i in range(1, len(idx)):
        if labels.iloc[i] != prev_label or i == len(idx) - 1:
            color = "green" if prev_label == 0 else "red"
            ax.axvspan(prev_date, idx[i], alpha=0.15, color=color)
            prev_date, prev_label = idx[i], labels.iloc[i]

    ax.set_ylabel("Cumulative Active Return (%)")
    ax.set_title(f"Online Inferred Regimes — {factor_name.capitalize()}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"regime_{factor_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_portfolio_weights(weights: pd.DataFrame, output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(
        weights.index,
        [weights[a].values * 100 for a in ASSETS],
        labels=ASSETS,
        alpha=0.7,
    )
    ax.set_ylabel("Portfolio Weight (%)")
    ax.set_title("Dynamic Portfolio Weights Over Time")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "portfolio_weights.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")
