import base64
from pathlib import Path

import pandas as pd
from shiny import ui


_DISPLAY_COLS = {
    "strategy": "Strategy",
    "sharpe": "Sharpe",
    "ir_vs_market": "IR vs Mkt",
    "max_drawdown": "Max DD",
    "volatility": "Volatility",
    "active_ret_vs_market": "Active Ret",
    "turnover": "Turnover",
}


def img_tag(path: Path, alt: str = "") -> ui.Tag:
    """Return an <img> tag with a base64-encoded data URI, or an error paragraph."""
    if not path.exists():
        return ui.p(f"Plot not found: {path.name}", class_="text-muted")
    encoded = base64.b64encode(path.read_bytes()).decode()
    return ui.img(
        src=f"data:image/png;base64,{encoded}",
        alt=alt,
        style="width:100%;max-width:1000px;display:block;margin:auto",
    )


def load_metrics_row(output_dir: Path, te_pct: int) -> pd.DataFrame:
    """Load the row for the given TE target from results.csv.

    Returns a single-row DataFrame with raw column names (see _DISPLAY_COLS for
    the display mapping).  Returns an empty DataFrame if the file is missing or
    the requested TE target is not found.
    """
    path = Path(output_dir) / "results.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns={"active_return": "active_ret_vs_market"})
    target = te_pct / 100
    mask = (df["target_te"] - target).abs() < 1e-6
    keep = [c for c in _DISPLAY_COLS if c != "strategy" and c in df.columns]
    return df[mask][keep]


def load_returns_df(output_dir: Path, te_pct: int) -> pd.DataFrame | None:
    """Load daily returns series for a given TE target.

    Returns DataFrame(index=date, columns=[portfolio, market, ew])
    or None if the file is missing.
    Supports both returns_te{N}.csv (model1/model3) and returns.csv (hmm/hsmm/msgarch).
    """
    path = Path(output_dir) / f"returns_te{te_pct}.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        return df[["portfolio", "market", "ew"]]
    path = Path(output_dir) / "returns.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        return df.rename(columns={"strategy": "portfolio"})[["portfolio", "market"]]
    return None
