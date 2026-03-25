"""
Dynamic Factor Allocation — Main Pipeline
Implements Shu & Mulvey (2025): regime-switching signals via SJM + Black-Litterman.

Usage:
  python main.py                    # use config.yaml, use cache
  python main.py --refresh          # force re-download all data
  python main.py --config my.yaml   # use alternate config file
"""

import argparse
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from src.utils import load_config, FACTORS, ASSETS
from src.data import load_all_data
from src.features import compute_all_features
from src.regime import run_regime_detection
from src.portfolio import run_portfolio_construction
from src.backtest import (
    compute_portfolio_returns,
    compute_ew_returns,
    compute_performance_table,
    save_results,
    plot_cumulative_returns,
    plot_regime,
    plot_portfolio_weights,
)


def main():
    parser = argparse.ArgumentParser(description="Dynamic Factor Allocation Pipeline")
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-download all data (ignore cache)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.refresh:
        cfg["data"]["refresh"] = True

    # ── Step 1: Data ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("[1/5] Loading data...")
    data = load_all_data(cfg)
    total_ret  = data["total_returns"]
    active_ret = data["active_returns"]
    rf         = data["rf"]
    macro      = data["macro"]
    active_rets_dict = {f: active_ret[f] for f in FACTORS}
    print(f"      Data: {total_ret.index[0].date()} → {total_ret.index[-1].date()} "
          f"({len(total_ret)} trading days)")

    # ── Step 2: Features ─────────────────────────────────────────────────────
    print("[2/5] Computing features...")
    features_dict = compute_all_features(
        active_rets_dict,
        mkt_ret=total_ret["market"],
        vix=macro["vix"],
        y2=macro["y2"],
        y10=macro["y10"],
    )

    # ── Step 3: Regime Detection ──────────────────────────────────────────────
    print("[3/5] Running SJM regime detection (this may take several minutes)...")
    regime_labels = run_regime_detection(
        active_rets_dict,
        mkt_ret=total_ret["market"],
        vix=macro["vix"],
        y2=macro["y2"],
        y10=macro["y10"],
        cfg=cfg,
    )
    for f, labels in regime_labels.items():
        bull_pct = (labels == 0).mean() * 100
        print(f"      {f:10s}: {len(labels)} test days, {bull_pct:.1f}% bull")

    # Build in-sample labels proxy for view return computation.
    # Use regime_labels shifted back by 1 as training history proxy.
    in_sample_labels = {
        f: regime_labels[f].shift(-1).dropna().astype(int)
        for f in FACTORS
    }

    # ── Step 4: Portfolio Construction ───────────────────────────────────────
    print("[4/5] Constructing portfolio weights via Black-Litterman...")
    weights = run_portfolio_construction(
        regime_labels,
        in_sample_labels,
        total_returns=total_ret,
        active_returns=active_ret,
        cfg=cfg,
    )
    print(f"      Weights computed for {len(weights)} days")

    # ── Step 5: Backtest ─────────────────────────────────────────────────────
    print("[5/5] Running backtest and saving outputs...")
    cost_bps = cfg["black_litterman"]["transaction_cost_bps"]
    ew_ret   = compute_ew_returns(total_ret.reindex(weights.index))
    mkt_ret  = total_ret["market"].reindex(weights.index)
    rf_test  = rf.reindex(weights.index)

    # Run backtest for each tracking error target
    te_targets = [0.01, 0.02, 0.03, 0.04]
    metrics_rows = []

    # EW benchmark metrics (reference row)
    ew_metrics = compute_performance_table(
        ew_ret, mkt_ret, ew_ret, rf_test,
        pd.DataFrame(
            {a: [1.0 / len(ASSETS)] * len(weights) for a in ASSETS},
            index=weights.index
        )
    )
    ew_metrics["strategy"] = "EW Benchmark"
    ew_metrics["target_te"] = None
    metrics_rows.append(ew_metrics)

    for te in te_targets:
        cfg_te = dict(cfg)
        cfg_te["black_litterman"] = dict(cfg["black_litterman"])
        cfg_te["black_litterman"]["target_tracking_error"] = te

        # Recompute weights for this TE target
        weights_te = run_portfolio_construction(
            regime_labels, in_sample_labels,
            total_returns=total_ret, active_returns=active_ret, cfg=cfg_te,
        )
        port_ret = compute_portfolio_returns(
            weights_te.reindex(weights.index, fill_value=1.0/len(ASSETS)),
            total_ret.reindex(weights.index),
            cost_bps=cost_bps,
        )
        metrics = compute_performance_table(port_ret, mkt_ret, ew_ret, rf_test, weights_te)
        metrics["strategy"] = f"Dynamic (TE={te*100:.0f}%)"
        metrics["target_te"] = te
        metrics_rows.append(metrics)

    # Save results table
    save_results(metrics_rows, cfg["output"]["results_path"])

    # Use TE=3% for plots
    cfg_3pct = dict(cfg)
    cfg_3pct["black_litterman"] = dict(cfg["black_litterman"])
    cfg_3pct["black_litterman"]["target_tracking_error"] = 0.03
    weights_3pct = run_portfolio_construction(
        regime_labels, in_sample_labels,
        total_returns=total_ret, active_returns=active_ret, cfg=cfg_3pct,
    )
    port_ret_3pct = compute_portfolio_returns(
        weights_3pct.reindex(weights.index, fill_value=1.0/len(ASSETS)),
        total_ret.reindex(weights.index),
        cost_bps=cost_bps,
    )

    plots_dir = cfg["output"]["plots_dir"]
    plot_cumulative_returns(port_ret_3pct, mkt_ret, ew_ret, rf_test, plots_dir,
                            label="Dynamic Allocation (TE=3%)")
    for factor in FACTORS:
        plot_regime(
            active_ret[factor].reindex(weights.index),
            regime_labels[factor],
            factor,
            plots_dir,
        )
    plot_portfolio_weights(weights_3pct.reindex(weights.index), plots_dir)

    print("\nDone. Results in outputs/")
    print("\n--- Performance Summary ---")
    for row in metrics_rows:
        print(f"  {row['strategy']:30s}  Sharpe={row['sharpe']:.2f}  "
              f"IR(vs EW)={row['ir_vs_ew']:.2f}  MDD={row['max_drawdown']*100:.1f}%")


if __name__ == "__main__":
    main()
