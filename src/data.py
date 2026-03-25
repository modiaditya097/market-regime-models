"""Data download, parsing, and caching for factor returns and macro series."""

import io
import os
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf

from src.utils import FACTORS, ASSETS


# ── Ken French URLs ──────────────────────────────────────────────────────────
_FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
_MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_daily_CSV.zip"
)


def _fetch_zip_csv(url: str) -> str:
    """Download a Ken French ZIP and return its CSV content as a string."""
    with urllib.request.urlopen(url, timeout=60) as resp:
        z = zipfile.ZipFile(io.BytesIO(resp.read()))
        return z.read(z.namelist()[0]).decode("utf-8", errors="replace")


def _parse_ken_french_csv(content: str, columns: list) -> pd.DataFrame:
    """
    Parse a Ken French CSV string (comma-delimited, date as YYYYMMDD in col 0).
    Returns a DataFrame with a DatetimeIndex and values divided by 100
    (converting from percent to decimal).
    """
    lines = content.splitlines()
    # Find first data line: col 0 is all digits (8-digit date)
    start = next(
        i for i, ln in enumerate(lines)
        if ln.strip() and ln.strip().split(",")[0].strip().isdigit()
    )
    data_lines = []
    for ln in lines[start:]:
        s = ln.strip()
        if not s:
            break
        if s.split(",")[0].strip().isdigit():
            data_lines.append(s)
        else:
            break
    df = pd.read_csv(io.StringIO("\n".join(data_lines)), header=None)
    df[0] = pd.to_datetime(df[0].astype(str).str.strip(), format="%Y%m%d")
    df = df.set_index(0)
    df.columns = columns
    return df.astype(float) / 100.0   # percent → decimal


def _parse_fred_csv(content: str) -> pd.Series:
    """Parse a FRED CSV string. Missing values ('.') become NaN."""
    df = pd.read_csv(
        io.StringIO(content),
        parse_dates=["DATE"],
        index_col="DATE",
        na_values=["."],
    )
    return df.iloc[:, 0].astype(float)


def _fetch_fred(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        content = resp.read().decode("utf-8")
    return _parse_fred_csv(content)


def _fetch_vix(start: str, end: str) -> pd.Series:
    raw = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
    s = raw["Close"].squeeze()
    s.index = pd.to_datetime(s.index).normalize()
    s.name = "VIX"
    return s


def build_asset_returns(
    ff5: pd.DataFrame, mom: pd.DataFrame
) -> tuple:
    """
    Construct total and active returns for all 6 assets.

    Total returns (decimal daily):
      market   = Mkt-RF + RF
      value    = market + HML
      size     = market + SMB
      quality  = market + RMW
      growth   = market - CMA
      momentum = market + Mom

    Active returns = total - market  (only for the 5 non-market factors).
    """
    aligned = ff5.join(mom, how="inner")
    mkt = aligned["Mkt-RF"] + aligned["RF"]

    total = pd.DataFrame(index=aligned.index)
    total["market"]   = mkt
    total["value"]    = mkt + aligned["HML"]
    total["size"]     = mkt + aligned["SMB"]
    total["quality"]  = mkt + aligned["RMW"]
    total["growth"]   = mkt - aligned["CMA"]
    total["momentum"] = mkt + aligned["Mom"]

    active = total[FACTORS].subtract(total["market"], axis=0)

    return total, active


def load_macro(start: str, end: str) -> dict:
    """Download VIX, 2Y yield, 10Y yield. Returns dict of aligned daily series."""
    vix = _fetch_vix(start, end)
    y2  = _fetch_fred("DGS2")
    y10 = _fetch_fred("DGS10")
    return {"vix": vix, "y2": y2, "y10": y10}


def load_all_data(cfg: dict) -> dict:
    """
    Download (or load from cache) all data.

    Returns:
      {
        'total_returns': pd.DataFrame (6 columns, daily),
        'active_returns': pd.DataFrame (5 columns, daily),
        'rf': pd.Series (daily risk-free rate, decimal),
        'macro': {'vix': ..., 'y2': ..., 'y10': ...},
      }
    All series share the same DatetimeIndex (intersection of all sources).
    """
    cache_dir = Path(cfg["data"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    refresh = cfg["data"].get("refresh", False)

    total_path  = cache_dir / "total_returns.parquet"
    active_path = cache_dir / "active_returns.parquet"
    rf_path     = cache_dir / "rf.parquet"
    macro_path  = cache_dir / "macro.parquet"

    if not refresh and all(p.exists() for p in [total_path, active_path, rf_path, macro_path]):
        total  = pd.read_parquet(total_path)
        active = pd.read_parquet(active_path)
        rf     = pd.read_parquet(rf_path).iloc[:, 0]
        macro_df = pd.read_parquet(macro_path)
        macro  = {col: macro_df[col] for col in macro_df.columns}
    else:
        # --- Ken French ---
        ff5_content = _fetch_zip_csv(_FF5_URL)
        mom_content = _fetch_zip_csv(_MOM_URL)
        ff5 = _parse_ken_french_csv(ff5_content, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
        mom = _parse_ken_french_csv(mom_content, ["Mom"])

        start, end = cfg["data"]["start_date"], cfg["data"]["end_date"]
        ff5 = ff5.loc[start:end]
        mom = mom.loc[start:end]
        total, active = build_asset_returns(ff5, mom)
        rf = ff5["RF"]

        # --- Macro ---
        macro_raw = load_macro(start, end)
        # Forward-fill yields to trading days; align to factor index
        y2_aligned  = macro_raw["y2"].reindex(total.index, method="ffill")
        y10_aligned = macro_raw["y10"].reindex(total.index, method="ffill")
        vix_aligned = macro_raw["vix"].reindex(total.index, method="ffill")
        macro = {"vix": vix_aligned, "y2": y2_aligned, "y10": y10_aligned}

        # Drop any dates with NaN in core data
        valid = total.notna().all(axis=1) & active.notna().all(axis=1)
        total, active, rf = total[valid], active[valid], rf[valid]
        for k in macro:
            macro[k] = macro[k][valid]

        # Cache
        total.to_parquet(total_path)
        active.to_parquet(active_path)
        rf.to_frame().to_parquet(rf_path)
        pd.DataFrame(macro).to_parquet(macro_path)

    return {
        "total_returns":  total,
        "active_returns": active,
        "rf":             rf,
        "macro":          macro,
    }
