"""Expanding-window monthly SJM refit and online inference per factor."""

import numpy as np
import pandas as pd

from jumpmodels.sparse_jump import SparseJumpModel
from jumpmodels.preprocess import StandardScalerPD, DataClipperStd

from src.features import compute_all_features


def get_train_window(
    refit_date: pd.Timestamp,
    data_start: pd.Timestamp,
    min_years: int = 8,
    max_years: int = 12,
) -> tuple:
    """
    Return (train_start, train_end) for a given refit date.
    Expanding window until max_years, then rolling.
    train_end = refit_date (exclusive upper bound: data up to but not including refit_date).
    """
    max_start = refit_date - pd.DateOffset(years=max_years)
    train_start = max(data_start, max_start)
    return train_start, refit_date


def get_refit_dates(index: pd.DatetimeIndex) -> list:
    """Return the first trading day of each month present in `index`."""
    monthly = index.to_frame().groupby([index.year, index.month]).first()
    return list(monthly.iloc[:, 0])


def _fit_sjm(X_train: pd.DataFrame, ret_train: pd.Series, cfg: dict) -> SparseJumpModel:
    """Fit a SparseJumpModel on preprocessed training data."""
    sc = cfg["sjm"]
    sjm = SparseJumpModel(
        n_components=sc["n_components"],
        jump_penalty=sc["jump_penalty"],
        max_feats=sc["max_feats"],
        max_iter=sc["max_iter"],
        n_init_jm=sc["n_init_jm"],
        random_state=sc["random_state"],
    )
    sjm.fit(X_train, ret_ser=ret_train, sort_by="cumret")
    return sjm


def run_regime_detection(
    active_rets: dict,
    mkt_ret: pd.Series,
    vix: pd.Series,
    y2: pd.Series,
    y10: pd.Series,
    cfg: dict,
) -> dict:
    """
    Monthly expanding-window SJM refit + online inference for each factor.

    Returns dict[factor -> pd.Series of regime labels (0=bull, 1=bear)].
    Labels use a 1-day delay: label on day T reflects inference from day T-1.
    """
    data_start = pd.Timestamp(cfg["data"]["start_date"])
    min_years = cfg["training"]["min_train_years"]
    max_years = cfg["training"]["max_train_years"]

    # Compute all features once
    features_dict = compute_all_features(active_rets, mkt_ret, vix, y2, y10)

    # Full date index
    all_dates = features_dict[next(iter(features_dict))].dropna(how="all").index

    # Test period starts after min training years
    test_start = data_start + pd.DateOffset(years=min_years)
    test_dates = all_dates[all_dates >= test_start]

    if len(test_dates) == 0:
        return {f: pd.Series(dtype=float) for f in active_rets}

    refit_dates = get_refit_dates(test_dates)

    # Storage for raw online labels (before delay shift)
    raw_labels = {f: {} for f in active_rets}

    for i, refit_date in enumerate(refit_dates):
        train_start, train_end = get_train_window(
            refit_date, data_start, min_years, max_years
        )

        # Interval: from this refit to the next (or end)
        if i + 1 < len(refit_dates):
            next_refit = refit_dates[i + 1]
        else:
            next_refit = test_dates[-1] + pd.Timedelta(days=1)
        interval_dates = test_dates[(test_dates >= refit_date) & (test_dates < next_refit)]

        if len(interval_dates) == 0:
            continue

        for factor in active_rets:
            X = features_dict[factor]

            # Training slice
            X_train = X[(X.index >= train_start) & (X.index < train_end)].dropna()
            ret_train = active_rets[factor].reindex(X_train.index).dropna()
            X_train = X_train.reindex(ret_train.index)

            if len(X_train) < 100:
                continue  # not enough data

            # Preprocess: fit on train, apply to interval
            clipper = DataClipperStd(mul=3.0)
            scaler = StandardScalerPD()
            X_train_proc = scaler.fit_transform(clipper.fit_transform(X_train))

            X_interval = X[X.index.isin(interval_dates)].dropna()
            if len(X_interval) == 0:
                continue
            X_interval_proc = scaler.transform(clipper.transform(X_interval))

            # Fit SJM
            try:
                sjm = _fit_sjm(X_train_proc, ret_train, cfg)
            except Exception:
                continue

            # Online inference on interval
            # predict_online returns a pd.Series (index=dates, values=labels)
            # when input is a DataFrame
            labels_raw = sjm.predict_online(X_interval_proc)

            # Handle both Series and ndarray/dict returns
            if isinstance(labels_raw, pd.Series):
                for date, lbl in labels_raw.items():
                    raw_labels[factor][date] = int(lbl)
            elif isinstance(labels_raw, dict):
                for date, lbl in labels_raw.items():
                    raw_labels[factor][date] = int(lbl)
            else:
                # ndarray case: pair with interval dates
                for date, lbl in zip(X_interval_proc.index, labels_raw):
                    raw_labels[factor][date] = int(lbl)

    # Build series and apply 1-day delay (shift forward by 1 trading day)
    regime_labels = {}
    for factor in active_rets:
        if not raw_labels[factor]:
            regime_labels[factor] = pd.Series(dtype=float)
            continue
        ser = pd.Series(raw_labels[factor]).sort_index()
        # Shift: portfolio on day T+1 uses regime from day T
        ser_shifted = ser.shift(1)
        regime_labels[factor] = ser_shifted.dropna().astype(int)

    return regime_labels
