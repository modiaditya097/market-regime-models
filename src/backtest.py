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
from src.regime import run_regime_detection
from src.portfolio import run_portfolio_construction


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


def save_returns_csv(
    port_ret: pd.Series,
    mkt_ret: pd.Series,
    ew_ret: pd.Series,
    output_dir: str,
    te_suffix: str,
) -> None:
    """Save daily return series to CSV for the Shiny app comparison tab."""
    idx = port_ret.dropna().index
    assert not mkt_ret.reindex(idx).isna().any(), "mkt_ret has NaNs on portfolio index"
    assert not ew_ret.reindex(idx).isna().any(), "ew_ret has NaNs on portfolio index"
    df = pd.DataFrame({
        "date": idx,
        "portfolio": port_ret.reindex(idx).values,
        "market": mkt_ret.reindex(idx).values,
        "ew": ew_ret.reindex(idx).values,
    })
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"returns{te_suffix}.csv")
    df.to_csv(path, index=False)
    print(f"Saved {path}")


def save_weights_csv(weights: pd.DataFrame, output_dir: str, te_suffix: str = "") -> None:
    """Save daily portfolio weights to CSV for the Shiny dashboard."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"weights{te_suffix}.csv")
    weights.to_csv(path)
    print(f"Saved {path}")


def plot_cumulative_returns(
    port_ret: pd.Series,
    mkt_ret: pd.Series,
    ew_ret: pd.Series,
    rf: pd.Series,
    output_dir: str,
    label: str = "Dynamic Allocation",
    te_suffix: str = "",
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
    path = os.path.join(output_dir, f"cumulative_returns{te_suffix}.png")
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


def plot_portfolio_weights(weights: pd.DataFrame, output_dir: str, te_suffix: str = "") -> None:
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
    path = os.path.join(output_dir, f"portfolio_weights{te_suffix}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def run_walk_forward(data: dict, cfg: dict) -> pd.DataFrame:
    """
    Expanding-window walk-forward evaluation.

    For each fold: train on data_start → fold_test_start (expanding),
    test on fold_test_start → fold_test_end.
    Returns a DataFrame with one row per completed fold plus an average row.
    """
    total_ret  = data["total_returns"]
    active_ret = data["active_returns"]
    rf         = data["rf"]
    macro      = data["macro"]

    enabled = cfg.get("macro_features", {}).get("enabled", [])
    factors = list(active_ret.columns)

    test_start       = pd.Timestamp(cfg["training"]["test_start"])
    n_folds          = cfg["walk_forward"]["n_folds"]
    fold_test_months = cfg["walk_forward"]["fold_test_months"]
    cost_bps         = cfg["black_litterman"]["transaction_cost_bps"]

    rows = []
    for fold in range(1, n_folds + 1):
        fold_test_start = test_start + pd.DateOffset(months=(fold - 1) * fold_test_months)
        fold_test_end   = fold_test_start + pd.DateOffset(months=fold_test_months)

        mask = total_ret.index < fold_test_end
        if mask.sum() == 0:
            break
        # Also skip fold if no data exists in the test window itself
        if (total_ret.index >= fold_test_start).sum() == 0:
            print(f"      Fold {fold}: no test-period data after {fold_test_start.date()}, skipping")
            continue

        fold_total  = total_ret[mask]
        fold_active = active_ret[mask]
        fold_rf     = rf[mask]
        fold_macro  = {k: v[mask] for k, v in macro.items() if isinstance(v, pd.Series)}
        fold_extras = {k: fold_macro[k] for k in enabled if k in fold_macro}

        fold_cfg = {
            **cfg,
            "training": {**cfg["training"], "test_start": str(fold_test_start.date())},
            "data":     {**cfg["data"], "end_date": str(fold_test_end.date())},
        }

        print(f"[{fold}/{n_folds}] walk-forward fold {fold} "
              f"({fold_test_start.date()} – {fold_test_end.date()})")

        try:
            regime_labels = run_regime_detection(
                {f: fold_active[f] for f in factors},
                mkt_ret=fold_total["market"],
                vix=fold_macro["vix"],
                y2=fold_macro["y2"],
                y10=fold_macro["y10"],
                cfg=fold_cfg,
                macro_extras=fold_extras,
                enabled=enabled,
            )

            test_label_dates = {
                f: regime_labels[f].index >= fold_test_start for f in factors
            }
            test_labels = {f: regime_labels[f][test_label_dates[f]] for f in factors}
            # shift(-1) aligns regime labels to same-day returns:
            # regime_labels carry a built-in 1-day delay (label at T = regime inferred from T-1),
            # so shift(-1) maps label at T to the return realised at T, removing the delay offset
            # for in-sample regime performance calculations.
            in_sample_labels = {
                f: test_labels[f].shift(-1).dropna().astype(int) for f in factors
            }

            weights = run_portfolio_construction(
                test_labels,
                in_sample_labels,
                total_returns=fold_total,
                active_returns=fold_active,
                cfg=fold_cfg,
            )

            test_dates    = weights.index
            port_ret      = compute_portfolio_returns(
                weights, fold_total.reindex(test_dates), cost_bps=cost_bps
            )
            mkt_ret_fold  = fold_total["market"].reindex(test_dates)
            ew_ret_fold   = compute_ew_returns(fold_total.reindex(test_dates))
            rf_fold       = fold_rf.reindex(test_dates)

            metrics = compute_performance_table(
                port_ret, mkt_ret_fold, ew_ret_fold, rf_fold, weights
            )
            rows.append({
                "fold":                 fold,
                "period_start":         str(fold_test_start.date()),
                "period_end":           str(fold_test_end.date()),
                "sharpe":               metrics["sharpe"],
                "ir_vs_market":         metrics["ir_vs_market"],
                "max_drawdown":         metrics["max_drawdown"],
                "active_ret_vs_market": metrics["active_ret_vs_market"],
                "volatility":           metrics["volatility"],
                "turnover":             metrics["turnover"],
            })
        except Exception as exc:
            print(f"      Fold {fold} failed: {exc}")
            continue

    if not rows:
        return pd.DataFrame(columns=["fold", "period_start", "period_end", "sharpe",
                                     "ir_vs_market", "max_drawdown",
                                     "active_ret_vs_market", "volatility", "turnover"])

    df = pd.DataFrame(rows)
    numeric_cols = ["sharpe", "ir_vs_market", "max_drawdown", "active_ret_vs_market",
                    "volatility", "turnover"]
    avg_row = {"fold": "avg", "period_start": "", "period_end": ""}
    for col in numeric_cols:
        avg_row[col] = df[col].mean()
    return pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
