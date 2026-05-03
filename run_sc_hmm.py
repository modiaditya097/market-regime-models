"""
SC-HMM Phase 3 — Standalone Pipeline Script
============================================
Equivalent of main.py for the SC-HMM model.

This script runs the complete SC-HMM pipeline end-to-end:
  [1/6] Settings & parameters
  [2/6] Data loading (yfinance + macro CSV)
  [3/6] Feature construction
  [4/6] Walk-forward backtest (spectral clustering + macro overlay)
  [5/6] Smoothing & benchmarks
  [6/6] Saving all dashboard outputs

Usage:
  python run_sc_hmm.py                        # default parameters
  python run_sc_hmm.py --macro macro_data.csv # custom macro path
  python run_sc_hmm.py --k-min 2 --k-max 3   # custom K range

Parameters can also be set via environment variables (used by Shiny Re-run):
  SCHMM_ASSETS, SCHMM_K_MIN, SCHMM_K_MAX, SCHMM_INITIAL_WINDOW,
  SCHMM_REFIT_CADENCE, SCHMM_SMOOTH_WINDOW, SCHMM_MACRO_NEUTRAL_THRESH,
  SCHMM_MACRO_EXTREME_THRESH, SCHMM_TC_BPS, SCHMM_MACRO_PATH
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import yfinance as yf
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# [1/6] SETTINGS & PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

def _step(n, total, msg):
    print(f'[{n}/{total}] {msg}', flush=True)


def load_params(args):
    """Merge CLI args → env vars → defaults. CLI wins over env, env wins over defaults."""
    def _env(key, default, cast=str):
        v = os.environ.get(key)
        return cast(v) if v is not None else default

    assets_str = args.assets or _env('SCHMM_ASSETS', 'SPY,IWM,IEF,TIP,GLD')
    return {
        'ASSETS':               [a.strip() for a in assets_str.split(',')],
        'K_MIN':                args.k_min    or _env('SCHMM_K_MIN',                2,    int),
        'K_MAX':                args.k_max    or _env('SCHMM_K_MAX',                3,    int),
        'K_NN':                 7,
        'MAX_SPECTRAL_WINDOW':  260,
        'INITIAL_WINDOW':       args.initial_window or _env('SCHMM_INITIAL_WINDOW', 156,  int),
        'REFIT_CADENCE':        args.refit_cadence  or _env('SCHMM_REFIT_CADENCE',  4,    int),
        'SMOOTH_WINDOW':        args.smooth_window  or _env('SCHMM_SMOOTH_WINDOW',  3,    int),
        'TC':                   (args.tc_bps or _env('SCHMM_TC_BPS', 10, int)) / 10000.0,
        'WINDOWS':              [13, 26],
        'N_INIT_KMEANS':        20,
        'ANNUALISE':            52,
        'P_SIZE':               3,
        'TRAIN_END':            '2018-12-31',
        'TEST_START':           '2019-01-01',
        'DATA_START':           '2002-01-01',
        'DATA_END':             '2026-04-03',
        'MACRO_NEUTRAL_THRESH': args.macro_neutral or _env('SCHMM_MACRO_NEUTRAL_THRESH', 0.5,  float),
        'MACRO_EXTREME_THRESH': args.macro_extreme or _env('SCHMM_MACRO_EXTREME_THRESH', 2.0,  float),
        'LABEL_VOL_ALPHA':      0.5,
        'MACRO_ZSCORE_WINDOW':  104,
        'MACRO_PATH':           args.macro or _env('SCHMM_MACRO_PATH', 'macro_data.csv'),
        'RANDOM_STATE':         42,
        'ALL_REGIMES':          ['bull', 'neutral', 'bear'],
        'REGIME_COLORS':        {'bull': '#2ecc71', 'neutral': '#f39c12', 'bear': '#e74c3c'},
        'REGIME_TO_ID':         {'bull': 0, 'neutral': 1, 'bear': 2},
        'ID_TO_REGIME':         {0: 'bull', 1: 'neutral', 2: 'bear'},
    }


# ══════════════════════════════════════════════════════════════════════════════
# [2/6] DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def fetch_weekly_prices(tickers, start, end):
    frames = {}
    for ticker in tickers:
        raw = yf.download(
            tickers=ticker, start=start, end=end,
            interval='1d', auto_adjust=True, progress=False,
        )
        if raw.empty:
            print(f'  WARNING: No data for {ticker}')
            continue
        close  = raw['Close'].iloc[:, 0] if isinstance(raw.columns, pd.MultiIndex) else raw['Close']
        weekly = close.resample('W-FRI').last().dropna()
        frames[ticker] = weekly
        print(f'  {ticker}: {len(weekly)} obs | {weekly.index[0].date()} to {weekly.index[-1].date()}')
    return pd.DataFrame(frames)


def load_data(p):
    prices_raw = fetch_weekly_prices(p['ASSETS'], start=p['DATA_START'], end=p['DATA_END']).sort_index()
    prices_raw = prices_raw[[a for a in p['ASSETS'] if a in prices_raw.columns]]
    prices     = prices_raw.dropna()
    assert len(prices) > 0, 'prices empty — check tickers or date range'

    ret    = prices.pct_change().dropna()
    assets = ret.columns.tolist()

    macro_path = p['MACRO_PATH']
    if not Path(macro_path).exists():
        raise FileNotFoundError(
            f"macro_data.csv not found at '{macro_path}'.\n"
            f"Place macro_data.csv in the project root or set --macro <path>."
        )
    macro_raw = pd.read_csv(macro_path, parse_dates=['observation_date'], index_col='observation_date')
    macro_raw = macro_raw.resample('W-FRI').last().ffill()

    macro       = macro_raw.reindex(ret.index, method='ffill')
    first_valid = macro.dropna().index[0]
    macro       = macro.loc[first_valid:]
    ret         = ret.loc[first_valid:]
    prices      = prices.loc[first_valid:]

    assert macro.isna().sum().sum() == 0, f'Macro NaNs remain: {macro.isna().sum()}'
    print(f'  Prices  : {prices.index[0].date()} → {prices.index[-1].date()} | {len(prices)} weeks')
    print(f'  Returns : {ret.index[0].date()} → {ret.index[-1].date()} | N={len(ret)}')
    print(f'  Macro   : {macro.index[0].date()} → {macro.index[-1].date()} | {len(macro)} obs')
    return prices, ret, assets, macro


# ══════════════════════════════════════════════════════════════════════════════
# [3/6] FEATURE CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def rolling_corr_safe(df_win, min_obs=13):
    m     = df_win.shape[1]
    pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
    if len(df_win) < min_obs:
        return np.zeros(len(pairs)), True
    try:
        C = df_win.corr().values
        if np.any(np.isnan(C)) or abs(np.linalg.det(C)) < 1e-10:
            return np.zeros(len(pairs)), True
        return np.array([C[i, j] for i, j in pairs]), False
    except Exception:
        return np.zeros(len(pairs)), True


def build_features(ret_df, assets, p):
    windows    = p['WINDOWS']
    w_short    = windows[0]
    w_long     = windows[1]
    m          = ret_df.shape[1]
    n          = len(ret_df)
    all_rows   = []
    flag_count = 0

    corr_assets = [a for a in assets if a != 'TIP']
    spy_i    = assets.index('SPY')
    tip_i    = assets.index('TIP')
    ief_i    = assets.index('IEF')
    iwm_i    = assets.index('IWM')
    corr_idx = [assets.index(a) for a in corr_assets]

    for t in range(n):
        row = []
        for l in windows:
            win = ret_df.iloc[max(0, t - l + 1): t + 1]
            row.extend(win.mean().values)
            row.extend(win.std(ddof=0).fillna(0).values)
        for l in windows:
            win_corr = ret_df.iloc[max(0, t - l + 1): t + 1].iloc[:, corr_idx]
            cv, flagged = rolling_corr_safe(win_corr)
            flag_count += flagged
            row.extend(cv)
        row.extend(ret_df.iloc[t].values)
        row.extend(
            np.abs(ret_df.iloc[t].values - ret_df.iloc[t - 1].values)
            if t > 0 else np.zeros(m)
        )
        spy_short = ret_df.iloc[max(0, t - w_short + 1): t + 1].iloc[:, spy_i].values
        spy_long  = ret_df.iloc[max(0, t - w_long  + 1): t + 1].iloc[:, spy_i].values
        row.extend([spy_short.mean(), spy_long.mean(), spy_short.std(ddof=0)])
        tip_short = ret_df.iloc[max(0, t - w_short + 1): t + 1].iloc[:, tip_i].values
        row.extend([tip_short.mean(), tip_short.std(ddof=0)])
        ief_short = ret_df.iloc[max(0, t - w_short + 1): t + 1].iloc[:, ief_i].values
        row.extend([ief_short.mean(), ief_short.std(ddof=0)])
        iwm_short = ret_df.iloc[max(0, t - w_short + 1): t + 1].iloc[:, iwm_i].values
        iwm_long  = ret_df.iloc[max(0, t - w_long  + 1): t + 1].iloc[:, iwm_i].values
        ief_long  = ret_df.iloc[max(0, t - w_long  + 1): t + 1].iloc[:, ief_i].values
        row.extend([
            spy_short.mean() - ief_short.mean(), spy_long.mean()  - ief_long.mean(),
            iwm_short.mean() - ief_short.mean(), iwm_long.mean()  - ief_long.mean(),
        ])
        row.extend([iwm_short.mean() - spy_short.mean(), iwm_long.mean() - spy_long.mean()])
        spy_vol_s = spy_short.std(ddof=0) + 1e-10
        ief_vol_s = ief_short.std(ddof=0) + 1e-10
        spy_vol_l = spy_long.std(ddof=0)  + 1e-10
        ief_vol_l = ief_long.std(ddof=0)  + 1e-10
        row.extend([spy_vol_s / ief_vol_s, spy_vol_l / ief_vol_l])
        all_rows.append(row)

    X = np.nan_to_num(np.array(all_rows, dtype=float))
    print(f'  Features: {X.shape[1]} per obs × {X.shape[0]} obs | corr flags: {flag_count}')
    return X


# ══════════════════════════════════════════════════════════════════════════════
# [4/6] SPECTRAL CLUSTERING + WALK-FORWARD BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def self_tuning_sigma(X, k):
    n = X.shape[0]; k = max(1, min(k, n - 1))
    D = cdist(X, X, metric='euclidean'); np.fill_diagonal(D, np.inf)
    return np.maximum(np.sort(D, axis=1)[:, k - 1], 1e-10)


def build_W(X, k):
    n = X.shape[0]; D2 = cdist(X, X, metric='sqeuclidean')
    sigma = self_tuning_sigma(X, k=k)
    W = np.exp(-D2 / np.maximum(2 * np.outer(sigma, sigma), 1e-30))
    np.fill_diagonal(W, 0)
    off = W[~np.eye(n, dtype=bool)]
    if W.sum(axis=1).min() < 1e-6 or np.median(off) < 1e-6:
        gs = np.median(np.sqrt(D2[D2 > 0])) / np.sqrt(2)
        W  = np.exp(-D2 / (2 * max(gs, 1e-10) ** 2)); np.fill_diagonal(W, 0)
        return W, 'global-fallback'
    return W, 'self-tuning'


def laplacian_eigenvectors(W, n_vecs):
    d = np.maximum(W.sum(axis=1), 1e-10); D_isqrt = np.diag(1.0 / np.sqrt(d))
    L = np.eye(len(W)) - D_isqrt @ W @ D_isqrt; L = (L + L.T) / 2.0
    vals, vecs = np.linalg.eigh(L); idx = np.argsort(vals)
    return vecs[:, idx][:, :n_vecs], vals[idx]


def select_k_eigengap(eigenvalues, k_min, k_max):
    n_eigs = len(eigenvalues); k_hi = min(k_max, n_eigs - 2); k_lo = min(k_min, k_hi)
    raw_gaps = np.array([eigenvalues[k] - eigenvalues[k - 1] for k in range(k_lo, k_hi + 1)])
    local_mean = np.array([
        np.mean(eigenvalues[max(0, k - 1): min(n_eigs, k + 3)]) for k in range(k_lo, k_hi + 1)
    ])
    norm_gaps = raw_gaps / np.maximum(local_mean, 1e-10)
    return k_lo + int(np.argmax(norm_gaps)), raw_gaps


def estimate_params(ret_df, labels, K):
    params = {}
    for k in range(K):
        idx = np.where(labels == k)[0]
        if len(idx) < 2:
            m = ret_df.shape[1]
            params[k] = {'mean': np.zeros(m), 'std': np.ones(m) * 1e-6,
                         'cov': np.eye(m), 'count': 0}
        else:
            arr = ret_df.iloc[idx].values
            params[k] = {'mean': arr.mean(0), 'std': arr.std(0, ddof=1),
                         'cov': np.cov(arr.T), 'count': len(idx)}
    return params


def transition_matrix(regime_id_seq, K_full=3):
    seq = np.asarray(regime_id_seq, dtype=int)
    P   = np.zeros((K_full, K_full))
    for a, b in zip(seq[:-1], seq[1:]):
        if 0 <= a < K_full and 0 <= b < K_full:
            P[a, b] += 1
    rs = P.sum(1, keepdims=True); P[rs.squeeze() == 0] = 1.0 / K_full
    return P / P.sum(1, keepdims=True)


def map_regimes(params, asset_list, K, vol_alpha, annualise):
    si = asset_list.index('SPY'); wi = asset_list.index('IWM'); bi = asset_list.index('TIP')
    labels_by_K = {2: ['bear', 'bull'], 3: ['bear', 'neutral', 'bull']}
    label_list  = labels_by_K.get(K, ['bear', 'neutral', 'bull'])
    def assign(sk): return {sk[i]: label_list[i] for i in range(K)}
    scores = {}
    for k in range(K):
        eq_mean = (params[k]['mean'][si] + params[k]['mean'][wi]) / 2
        spy_vol = params[k]['std'][si]
        scores[k] = eq_mean * annualise - vol_alpha * spy_vol * np.sqrt(annualise)
    score_spread = max(abs(scores[i] - scores[j]) for i in range(K) for j in range(i+1, K))
    if score_spread >= 0.01:
        return assign(sorted(range(K), key=lambda k: scores[k])), 'primary-sharpe'
    vols = {k: params[k]['std'][[si, wi]].mean() * np.sqrt(annualise) for k in range(K)}
    vol_spread = max(abs(vols[i] - vols[j]) for i in range(K) for j in range(i+1, K))
    if vol_spread >= 0.02:
        return assign(sorted(range(K), key=lambda k: vols[k], reverse=True)), 'tiebreak-vol'
    bond_r = {k: params[k]['mean'][bi] for k in range(K)}
    return assign(sorted(range(K), key=lambda k: bond_r[k], reverse=True)), 'tiebreak-bond'


def compute_macro_score(macro_df, window):
    vix = macro_df['VIXCLS']; hy = macro_df['BAMLH0A0HYM2']; t10y = macro_df['T10Y2Y']
    def rolling_zscore(s, w):
        mu  = s.rolling(w, min_periods=w // 2).mean()
        std = s.rolling(w, min_periods=w // 2).std(ddof=1).replace(0, np.nan)
        return ((s - mu) / std).fillna(0)
    z_vix  = rolling_zscore(vix, window)
    z_hy   = rolling_zscore(hy, window)
    z_t10y = rolling_zscore(-t10y, window)
    return (z_vix + z_hy + z_t10y) / 3.0, z_vix


def run_backtest(ret, assets, X_full, macro, macro_score_full, vix_z_full, p):
    ALLOC = {
        'bull':    np.array([0.60, 0.20, 0.05, 0.05, 0.10]),
        'neutral': np.array([0.30, 0.10, 0.20, 0.15, 0.25]),
        'bear':    np.array([0.05, 0.00, 0.30, 0.25, 0.40]),
    }

    n_total          = len(ret)
    store            = {
        'dates': [], 'r_hard': [], 'w_hard': [], 'pred_reg': [], 'cur_reg': [], 'to_hard': [],
        'r_overlay': [], 'w_overlay': [], 'to_overlay': [],
        'overlay_reg': [], 'overlay_action': [],
        'P_running': [], 'K_at_t': [], 'cur_K_labels': [], 'macro_score_t': [],
    }
    live_reg_ids     = []
    current_reg      = 'neutral'
    current_reg_id   = p['REGIME_TO_ID']['neutral']
    current_P        = np.ones((p['P_SIZE'], p['P_SIZE'])) / p['P_SIZE']
    last_w_hard      = ALLOC['neutral'].copy()
    last_w_overlay   = ALLOC['neutral'].copy()
    current_K        = 3
    current_K_labels = ['bear', 'neutral', 'bull']
    n_refits         = 0
    refit_rule_cnt   = Counter()

    np.random.seed(p['RANDOM_STATE'])
    total_steps = n_total - p['INITIAL_WINDOW'] - 1

    for t in range(p['INITIAL_WINDOW'], n_total - 1):
        step_num = t - p['INITIAL_WINDOW'] + 1
        if step_num % 50 == 0 or step_num == 1:
            pct = step_num / total_steps * 100
            print(f'  Backtest {step_num}/{total_steps} ({pct:.0f}%)', flush=True)

        if (t - p['INITIAL_WINDOW']) % p['REFIT_CADENCE'] == 0:
            n_sp    = min(t + 1, p['MAX_SPECTRAL_WINDOW'])
            X_fit   = X_full[:t + 1][-n_sp:]
            ret_fit = ret.iloc[:t + 1].iloc[-n_sp:]
            X_sc    = StandardScaler().fit_transform(X_fit)
            W_live, _ = build_W(X_sc, k=min(p['K_NN'], n_sp - 1))
            U_live, eig_vals = laplacian_eigenvectors(W_live, n_vecs=p['K_MAX'] + 1)
            k_star, _ = select_k_eigengap(eig_vals, k_min=p['K_MIN'], k_max=p['K_MAX'])
            labels_live   = KMeans(n_clusters=k_star, n_init=p['N_INIT_KMEANS'],
                                   random_state=p['RANDOM_STATE']).fit_predict(U_live[:, :k_star])
            cluster_sizes = np.array([np.sum(labels_live == k) for k in range(k_star)])
            if cluster_sizes.min() / len(labels_live) < 0.08 and k_star > p['K_MIN']:
                k_star      = p['K_MIN']
                labels_live = KMeans(n_clusters=k_star, n_init=p['N_INIT_KMEANS'],
                                     random_state=p['RANDOM_STATE']).fit_predict(U_live[:, :k_star])
            params_live  = estimate_params(ret_fit, labels_live, k_star)
            lmap, rule   = map_regimes(params_live, assets, k_star,
                                       p['LABEL_VOL_ALPHA'], p['ANNUALISE'])
            refit_rule_cnt[rule] += 1
            current_K        = k_star
            current_K_labels = [lmap[k] for k in range(k_star)]
            current_reg      = lmap[int(labels_live[-1])]
            current_reg_id   = p['REGIME_TO_ID'][current_reg]
            n_refits        += 1

        if len(live_reg_ids) >= 2:
            current_P = transition_matrix(live_reg_ids, K_full=p['P_SIZE'])

        next_reg_id = int(np.argmax(current_P[current_reg_id]))
        pred_reg    = p['ID_TO_REGIME'][next_reg_id]
        if pred_reg not in current_K_labels:
            pred_reg = current_reg
        w_h = ALLOC[pred_reg]

        macro_t = macro_score_full.iloc[t]
        vix_z_t = vix_z_full.iloc[t]
        if pred_reg == 'bear':
            if macro_t < -p['MACRO_NEUTRAL_THRESH']:
                overlay_reg, overlay_action = 'neutral', 'downgraded'
            else:
                overlay_reg, overlay_action = 'bear', 'confirmed'
        elif pred_reg == 'neutral' and vix_z_t > p['MACRO_EXTREME_THRESH']:
            overlay_reg, overlay_action = 'bear', 'upgraded'
        else:
            overlay_reg, overlay_action = pred_reg, 'pass'
        w_ov = ALLOC[overlay_reg]

        r_next = ret.iloc[t + 1].values
        to_h   = float(np.sum(np.abs(w_h  - last_w_hard)))
        to_ov  = float(np.sum(np.abs(w_ov - last_w_overlay)))

        store['dates'].append(ret.index[t + 1])
        store['r_hard'].append(float(np.dot(w_h,  r_next)) - p['TC'] * to_h)
        store['r_overlay'].append(float(np.dot(w_ov, r_next)) - p['TC'] * to_ov)
        store['w_hard'].append(w_h.copy())
        store['w_overlay'].append(w_ov.copy())
        store['pred_reg'].append(pred_reg)
        store['overlay_reg'].append(overlay_reg)
        store['overlay_action'].append(overlay_action)
        store['cur_reg'].append(current_reg)
        store['to_hard'].append(to_h)
        store['to_overlay'].append(to_ov)
        store['P_running'].append(current_P.copy())
        store['K_at_t'].append(current_K)
        store['cur_K_labels'].append(current_K_labels[:])
        store['macro_score_t'].append(macro_t)
        live_reg_ids.append(current_reg_id)
        last_w_hard    = w_h.copy()
        last_w_overlay = w_ov.copy()

    print(f'  Done. Refits={n_refits} | Obs={len(store["dates"])}')
    print(f'  Mapping rules: {dict(refit_rule_cnt)}')
    K_arr = np.array(store['K_at_t'])
    print(f'  Dynamic K — mean={K_arr.mean():.2f}  K=2: {(K_arr==2).mean():.1%}  K=3: {(K_arr==3).mean():.1%}')
    return store, ALLOC


# ══════════════════════════════════════════════════════════════════════════════
# [5/6] SMOOTHING, BENCHMARKS & METRICS
# ══════════════════════════════════════════════════════════════════════════════

def smooth_regimes(regimes, window):
    out = []
    for t in range(len(regimes)):
        sub = regimes[max(0, t - window + 1): t + 1]
        out.append(Counter(sub).most_common(1)[0][0])
    return np.array(out)


def recompute_hard(reg_arr, dates_bt, ret, alloc, tc):
    ret_l, lw = [], alloc['neutral'].copy()
    for date, reg in zip(dates_bt, reg_arr):
        w  = alloc[reg].copy()
        to = float(np.sum(np.abs(w - lw)))
        r  = float(np.dot(w, ret.loc[date].values)) - tc * to
        ret_l.append(r); lw = w.copy()
    return np.array(ret_l)


def cum_ret(r):        return np.cumprod(1 + np.asarray(r)) - 1
def drawdown_series(r):
    w = np.cumprod(1 + np.asarray(r)); return w / np.maximum.accumulate(w) - 1
def rolling_sharpe(r, window=52, ann=52):
    s = pd.Series(r)
    return (s.rolling(window).mean() * ann / (s.rolling(window).std(ddof=1) * np.sqrt(ann))).values


def compute_metrics(returns, rf=0.0, freq=52):
    r = np.asarray(returns, dtype=float); n = len(r)
    if n < 2:
        return {k: 'N/A' for k in ['Cumul.Ret','CAGR','Ann.Vol','Sharpe','Sortino','Max DD','Calmar','Hit Rate']}
    wealth = np.cumprod(1 + r); years = n / freq
    total  = wealth[-1] - 1; cagr = wealth[-1] ** (1 / years) - 1
    vol    = r.std(ddof=1) * np.sqrt(freq); excess = r.mean() * freq - rf
    sharpe = excess / vol if vol > 0 else np.nan
    dn     = r[r < 0]; dn_vol = dn.std(ddof=1) * np.sqrt(freq) if len(dn) > 1 else np.nan
    sortino = excess / dn_vol if (dn_vol and dn_vol > 0) else np.nan
    peak = np.maximum.accumulate(wealth); dd = wealth / peak - 1; max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan
    return {
        'Cumul.Ret': f'{total:.1%}', 'CAGR': f'{cagr:.2%}', 'Ann.Vol': f'{vol:.2%}',
        'Sharpe': f'{sharpe:.2f}', 'Sortino': f'{sortino:.2f}', 'Max DD': f'{max_dd:.2%}',
        'Calmar': f'{calmar:.2f}', 'Hit Rate': f'{np.mean(r > 0):.1%}',
    }


def build_benchmarks(bt_ret, N_ASSETS, dates_bt):
    bh_spy      = bt_ret['SPY'].values
    static_6040 = bt_ret.values @ np.array([0.60, 0.00, 0.20, 0.20, 0.00])
    static_5a   = bt_ret.values @ np.array([0.35, 0.10, 0.15, 0.15, 0.25])
    rp_ret_l, lw_rp = [], np.full(N_ASSETS, 1.0 / N_ASSETS)
    for i, date in enumerate(dates_bt):
        if i % 52 == 0 and i >= 52:
            hist  = bt_ret.iloc[max(0, i - 52): i]
            vol   = hist.std(ddof=1).values
            iv    = 1.0 / np.maximum(vol, 1e-8)
            iv    = np.clip(iv / iv.sum(), 0.02, 0.55)
            lw_rp = iv / iv.sum()
        rp_ret_l.append(float(np.dot(lw_rp, bt_ret.iloc[i].values)))
    rp_ret   = np.array(rp_ret_l)
    ew_5     = bt_ret.values @ np.full(N_ASSETS, 1.0 / N_ASSETS)
    return bh_spy, static_6040, static_5a, rp_ret, ew_5


# ══════════════════════════════════════════════════════════════════════════════
# [6/6] SAVE ALL DASHBOARD OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

def save_outputs(dates_bt, ret_sm_hard, ret_sm_overlay, bh_spy, static_6040,
                 static_5a, rp_ret, ew_5, reg_smooth, ov_reg_smooth,
                 store, macro, macro_score_full, p, P_final, assets,
                 STRESS_PERIODS, BULL_PERIODS, test_m, train_m, output_dir):

    PLOTS_DIR = output_dir / 'plots'
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    ann = p['ANNUALISE']
    REGIME_COLORS = p['REGIME_COLORS']

    C = {'hard': '#2c3e50', 'overlay': '#8e44ad', 'spy': '#2980b9',
         'rp': '#e67e22', 's6040': '#16a085', 's5a': '#d35400'}

    STRAT_LIST = [
        ('* SC-HMM',              ret_sm_hard),
        ('* SC-HMM + Macro Overlay', ret_sm_overlay),
        ('  Buy-Hold SPY',        bh_spy),
        ('  Risk Parity (inv-vol)', rp_ret),
        ('  Static 60/40',        static_6040),
        ('  Static 5-Asset Blend',static_5a),
    ]

    # ── returns_te3.csv ──────────────────────────────────────────────────────
    pd.DataFrame({
        'date': dates_bt, 'portfolio': ret_sm_hard, 'market': bh_spy, 'ew': ew_5,
    }).to_csv(output_dir / 'returns_te3.csv', index=False)
    print('  ✓ returns_te3.csv')

    # ── metrics CSVs ─────────────────────────────────────────────────────────
    full_m = np.ones(len(dates_bt), dtype=bool)
    for fname, mask in [('metrics_full.csv', full_m), ('metrics_train.csv', train_m),
                        ('metrics_test.csv', test_m)]:
        rows = [{'Strategy': n, **compute_metrics(s[mask])} for n, s in STRAT_LIST]
        pd.DataFrame(rows).to_csv(output_dir / fname, index=False)
    print('  ✓ metrics_full/train/test.csv')

    # ── results.csv (Comparison tab) ─────────────────────────────────────────
    r = ret_sm_hard
    vol   = float(r.std(ddof=1) * np.sqrt(ann))
    cagr  = float(np.cumprod(1 + r)[-1] ** (ann / len(r)) - 1)
    sharpe = (float(r.mean()) * ann) / vol if vol > 1e-10 else 0.0
    act   = float((r - bh_spy).mean() * ann)
    te    = float((r - bh_spy).std(ddof=1) * np.sqrt(ann))
    ir    = act / te if te > 1e-10 else 0.0
    peak  = np.maximum.accumulate(np.cumprod(1 + r))
    max_dd = float((np.cumprod(1 + r) / peak - 1).min()) * 100
    w_arr  = np.array(store['w_hard'])
    turnover = float(np.mean(np.sum(np.abs(np.diff(w_arr, axis=0)), axis=1)) * ann)
    pd.DataFrame([{
        'target_te': 0.03, 'sharpe': round(sharpe, 3), 'ir_vs_market': round(ir, 3),
        'max_drawdown': round(max_dd, 2), 'volatility': round(vol * 100, 2),
        'active_ret_vs_market': round(act * 100, 2), 'turnover': round(turnover, 3),
    }]).to_csv(output_dir / 'results.csv', index=False)
    print('  ✓ results.csv')

    # ── stress / bull tables ─────────────────────────────────────────────────
    STRAT_SUB = {'SC-HMM': ret_sm_hard, 'Macro Overlay': ret_sm_overlay,
                 'SPY B&H': bh_spy, 'Risk Parity': rp_ret, 'Static 60/40': static_6040}
    for fname, periods in [('stress_table.csv', STRESS_PERIODS), ('bull_table.csv', BULL_PERIODS)]:
        rows = []
        for pname, (s, e) in periods.items():
            mask = (dates_bt >= s) & (dates_bt <= e)
            if mask.sum() < 3: continue
            row = {'Period': pname}
            for sn, ser in STRAT_SUB.items():
                row[sn] = f'{float((np.cumprod(1 + ser[mask]) - 1)[-1]):.1%}'
            rows.append(row)
        pd.DataFrame(rows).to_csv(output_dir / fname, index=False)
    print('  ✓ stress_table.csv, bull_table.csv')

    # ── macro_latest.csv ─────────────────────────────────────────────────────
    lat = macro.iloc[-1]
    vix_v = float(lat['VIXCLS']); curve = float(lat['T10Y2Y']); hy = float(lat['BAMLH0A0HYM2'])
    pd.DataFrame([
        {'indicator': 'VIX',              'value': f'{vix_v:.1f}',
         'signal': 'bear' if vix_v > 25 else 'bull' if vix_v < 16 else 'neutral'},
        {'indicator': 'Yield Curve 10Y-2Y','value': f'{curve:+.2f}',
         'signal': 'bear' if curve < 0 else 'bull' if curve > 1.5 else 'neutral'},
        {'indicator': 'HY Spread',         'value': f'{hy:.2f}%',
         'signal': 'bear' if hy > 4.5 else 'bull' if hy < 3.0 else 'neutral'},
    ]).to_csv(output_dir / 'macro_latest.csv', index=False)
    print('  ✓ macro_latest.csv')

    # ── k_analysis.csv ───────────────────────────────────────────────────────
    K_arr = np.array(store['K_at_t'])
    pd.DataFrame([
        {'K': 2, 'Frequency': f'{(K_arr==2).mean():.1%}', 'Weeks': int((K_arr==2).sum())},
        {'K': 3, 'Frequency': f'{(K_arr==3).mean():.1%}', 'Weeks': int((K_arr==3).sum())},
    ]).to_csv(output_dir / 'k_analysis.csv', index=False)
    print('  ✓ k_analysis.csv')

    # ── PLOTS ─────────────────────────────────────────────────────────────────

    def stress_shade(ax):
        for _, (s, e) in STRESS_PERIODS.items():
            ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.07, color='red')

    # Cumulative returns
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates_bt, cum_ret(ret_sm_hard),    color=C['hard'],    lw=2.2, label='SC-HMM')
    ax.plot(dates_bt, cum_ret(ret_sm_overlay), color=C['overlay'], lw=2.0, ls='--', label='Macro Overlay')
    ax.plot(dates_bt, cum_ret(bh_spy),         color=C['spy'],     lw=1.8, alpha=0.8, label='Buy-Hold SPY')
    ax.plot(dates_bt, cum_ret(rp_ret),         color=C['rp'],      lw=1.5, label='Risk Parity')
    ax.plot(dates_bt, cum_ret(static_6040),    color=C['s6040'],   lw=1.3, label='Static 60/40')
    ax.plot(dates_bt, cum_ret(static_5a),      color=C['s5a'],     lw=1.3, label='5-Asset Blend')
    ax.axvline(pd.Timestamp(p['TEST_START']), color='black', ls=':', lw=1.5, label=f'OOS ({p["TEST_START"]})')
    stress_shade(ax)
    ax.set_title('SC-HMM Phase 3: Cumulative Return', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Return')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=9, ncol=3); ax.grid(alpha=0.3); plt.tight_layout()
    fig.savefig(PLOTS_DIR / 'cumulative_returns.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    print('  ✓ plots/cumulative_returns.png')

    # Drawdown
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(dates_bt, drawdown_series(ret_sm_hard),    0, alpha=0.5, color=C['hard'],    label='SC-HMM')
    ax.fill_between(dates_bt, drawdown_series(ret_sm_overlay), 0, alpha=0.3, color=C['overlay'], label='Macro Overlay')
    ax.plot(dates_bt, drawdown_series(bh_spy), color=C['spy'], lw=1.2, label='SPY')
    ax.plot(dates_bt, drawdown_series(rp_ret), color=C['rp'],  lw=1.2, label='Risk Parity')
    ax.axvline(pd.Timestamp(p['TEST_START']), color='black', ls=':', lw=1.5)
    ax.set_title('SC-HMM Phase 3: Drawdown', fontsize=12, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=9); ax.grid(alpha=0.3); plt.tight_layout()
    fig.savefig(PLOTS_DIR / 'drawdown.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    print('  ✓ plots/drawdown.png')

    # Rolling Sharpe
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dates_bt, rolling_sharpe(ret_sm_hard),    color=C['hard'],    lw=1.8, label='SC-HMM')
    ax.plot(dates_bt, rolling_sharpe(ret_sm_overlay), color=C['overlay'], lw=1.8, ls='--', label='Macro Overlay')
    ax.plot(dates_bt, rolling_sharpe(bh_spy),         color=C['spy'],     lw=1.3, alpha=0.8, label='SPY')
    ax.plot(dates_bt, rolling_sharpe(rp_ret),         color=C['rp'],      lw=1.3, alpha=0.8, label='Risk Parity')
    ax.plot(dates_bt, rolling_sharpe(static_6040),    color=C['s6040'],   lw=1.2, alpha=0.7, label='60/40')
    ax.axhline(0, color='black', lw=0.8, ls='--'); ax.axhline(1, color='green', lw=0.8, ls=':', alpha=0.6)
    ax.axvline(pd.Timestamp(p['TEST_START']), color='black', ls=':', lw=1.5)
    ax.set_title('SC-HMM Phase 3: Rolling 52-Week Sharpe', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, ncol=3); ax.grid(alpha=0.3); plt.tight_layout()
    fig.savefig(PLOTS_DIR / 'rolling_sharpe.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    print('  ✓ plots/rolling_sharpe.png')

    # Portfolio weights
    colors5 = ['#2980b9','#27ae60','#e74c3c','#8e44ad','#f1c40f']
    w_hard_arr    = np.array(store['w_hard'])
    w_overlay_arr = np.array(store['w_overlay'])
    N = len(assets)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('SC-HMM Phase 3: Portfolio Weights Over Time', fontsize=12, fontweight='bold')
    for ax, w_arr, title in zip(axes, [w_hard_arr, w_overlay_arr], ['SC-HMM', 'SC-HMM + Macro Overlay']):
        ax.stackplot(dates_bt, *[w_arr[:, i] for i in range(N)],
                     labels=assets, colors=colors5[:N], alpha=0.85)
        ax.axvline(pd.Timestamp(p['TEST_START']), color='black', ls=':', lw=1.5, label='OOS')
        ax.set_title(title, fontsize=10); ax.set_ylim(0, 1)
        ax.legend(ncol=6, fontsize=8, loc='upper left'); ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / 'portfolio_weights.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    print('  ✓ plots/portfolio_weights.png')

    # Regime timeline
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    fig.suptitle('SC-HMM Phase 3: Regime Timeline', fontsize=12, fontweight='bold')
    for ax, reg_arr, title in zip(axes, [reg_smooth, ov_reg_smooth], ['SC-HMM Hard', 'Macro Overlay']):
        for reg, col in REGIME_COLORS.items():
            ax.fill_between(dates_bt, 0, 1, where=(reg_arr == reg), color=col, alpha=0.7,
                            transform=ax.get_xaxis_transform(), label=reg)
        spy_cs = cum_ret(bh_spy)
        spy_sc = (spy_cs - spy_cs.min()) / (spy_cs.max() - spy_cs.min() + 1e-10)
        ax.plot(dates_bt, spy_sc, 'k-', lw=0.7, alpha=0.5, label='SPY (scaled)')
        ax.axvline(pd.Timestamp(p['TEST_START']), color='black', ls=':', lw=1.5)
        ax.set_title(title, fontsize=9); ax.set_yticks([])
        ax.legend(loc='upper left', ncol=5, fontsize=8)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / 'regime_timeline.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    print('  ✓ plots/regime_timeline.png')

    # Transition matrix
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(P_final, cmap='Blues', vmin=0, vmax=1)
    lbs = list(REGIME_COLORS.keys())
    for i in range(P_final.shape[0]):
        for j in range(P_final.shape[1]):
            v = P_final[i, j]
            ax.text(j, i, f'{v:.1%}', ha='center', va='center', fontsize=12,
                    color='white' if v > 0.55 else '#1a1a2e',
                    fontweight='bold' if i == j else 'normal')
    ax.set_xticks(range(len(lbs))); ax.set_yticks(range(len(lbs)))
    ax.set_xticklabels([f'→{l.capitalize()}' for l in lbs], fontsize=9)
    ax.set_yticklabels([l.capitalize() for l in lbs], fontsize=9)
    ax.set_title('Transition Matrix P[i→j]', fontsize=11, fontweight='bold')
    for i in range(len(lbs)):
        ax.add_patch(plt.Rectangle((i-.5, i-.5), 1, 1, fill=False, edgecolor='#e74c3c', lw=2))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); plt.tight_layout()
    fig.savefig(PLOTS_DIR / 'transition_matrix.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    print('  ✓ plots/transition_matrix.png')

    # Annual heatmap
    strats_ann = {'SC-HMM': ret_sm_hard, 'Macro Overlay': ret_sm_overlay,
                  'Buy-Hold SPY': bh_spy, 'Risk Parity': rp_ret,
                  'Static 60/40': static_6040, '5-Asset Blend': static_5a}
    years = sorted(dates_bt.year.unique())
    ann_data = {}
    for name, series in strats_ann.items():
        ann_data[name] = {}
        for yr in years:
            mask = dates_bt.year == yr
            ann_data[name][yr] = (np.nan if mask.sum() < 4
                                  else np.cumprod(1 + series[mask])[-1] - 1)
    df_ann = pd.DataFrame(ann_data).T
    vmax   = max(abs(df_ann.values[~np.isnan(df_ann.values)]).max(), 0.01)
    fig, ax = plt.subplots(figsize=(max(14, len(years) * 0.85), len(strats_ann) * 0.85 + 1.5))
    fig.suptitle('Annual Returns by Strategy', fontsize=12, fontweight='bold')
    cmap = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im2  = ax.imshow(df_ann.values, cmap='RdYlGn', norm=cmap, aspect='auto')
    ax.set_xticks(range(len(years))); ax.set_xticklabels(years, rotation=45, fontsize=8)
    ax.set_yticks(range(len(strats_ann))); ax.set_yticklabels(list(strats_ann.keys()), fontsize=9)
    for i in range(len(strats_ann)):
        for j, yr in enumerate(years):
            v = df_ann.iloc[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.0%}', ha='center', va='center', fontsize=7,
                        color='white' if abs(v) > vmax * 0.5 else 'black')
    plt.colorbar(im2, ax=ax, fraction=0.02, pad=0.02, format=mticker.PercentFormatter(1.0))
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / 'annual_heatmap.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    print('  ✓ plots/annual_heatmap.png')

    # Risk-return scatter
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Risk-Return Profile — Full Period & OOS', fontsize=12, fontweight='bold')
    sc_colors_map = {'SC-HMM': C['hard'], 'Macro Overlay': C['overlay'],
                     'Buy-Hold SPY': C['spy'], 'Risk Parity': C['rp'],
                     'Static 60/40': C['s6040'], '5-Asset Blend': C['s5a']}
    for ax, mask, period_label in zip(
            axes, [np.ones(len(dates_bt), dtype=bool), test_m],
            [f'Full Period ({dates_bt[0].year}–{dates_bt[-1].year})',
             f'OOS Only (≥ {p["TEST_START"]})']):
        for name, series in strats_ann.items():
            r = series[mask]
            if len(r) < 10: continue
            vol  = r.std(ddof=1) * np.sqrt(ann)
            cagr = np.cumprod(1 + r)[-1] ** (ann / len(r)) - 1
            col  = sc_colors_map.get(name, '#888888')
            star = name in ('SC-HMM', 'Macro Overlay')
            ax.scatter(vol, cagr, color=col, s=140 if star else 80,
                       zorder=5, marker='*' if star else 'o')
            ax.annotate(name, (vol, cagr), textcoords='offset points',
                        xytext=(6, 4), fontsize=8, color=col)
        ax.axhline(0, color='black', lw=0.7, ls='--')
        ax.set_xlabel('Annualised Volatility'); ax.set_ylabel('CAGR')
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_title(period_label, fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / 'risk_return_scatter.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    print('  ✓ plots/risk_return_scatter.png')

    print(f'\n✅ All outputs saved to: {output_dir.resolve()}')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='SC-HMM Phase 3 Pipeline')
    parser.add_argument('--assets',          default=None, help='Comma-separated tickers, e.g. SPY,IWM,IEF,TIP,GLD')
    parser.add_argument('--k-min',           type=int,   default=None)
    parser.add_argument('--k-max',           type=int,   default=None)
    parser.add_argument('--initial-window',  type=int,   default=None)
    parser.add_argument('--refit-cadence',   type=int,   default=None)
    parser.add_argument('--smooth-window',   type=int,   default=None)
    parser.add_argument('--macro-neutral',   type=float, default=None)
    parser.add_argument('--macro-extreme',   type=float, default=None)
    parser.add_argument('--tc-bps',          type=int,   default=None)
    parser.add_argument('--macro',           type=str,   default=None, help='Path to macro_data.csv')
    parser.add_argument('--output-dir',      type=str,   default='outputs/model3')
    args = parser.parse_args()

    p          = load_params(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(p['RANDOM_STATE'])

    STRESS_PERIODS = {
        'GFC Bear (2008-2009)':    ('2008-01-01', '2009-06-30'),
        'EU Debt Crisis (2011)':   ('2011-05-01', '2011-12-31'),
        'China Crash (2015-16)':   ('2015-07-01', '2016-02-29'),
        'COVID Crash (Q1 2020)':   ('2020-02-01', '2020-05-31'),
        'Rate Hike Bear (2022)':   ('2022-01-01', '2022-12-31'),
        'Liberation Day (2025)':   ('2025-03-01', '2025-05-31'),
    }
    BULL_PERIODS = {
        'Bull 2013':               ('2013-01-01', '2013-12-31'),
        'Bull 2017':               ('2017-01-01', '2017-12-31'),
        'Bull 2019 (OOS)':         ('2019-01-01', '2019-12-31'),
        'Post-COVID Bull 2020-21': ('2020-06-01', '2021-12-31'),
        'Bull 2023-24 (OOS)':      ('2023-01-01', '2024-12-31'),
    }

    print('=' * 66)
    print('SC-HMM Phase 3 | Spectral Clustering + Macro Overlay Pipeline')
    print('=' * 66)
    print(f'Assets: {p["ASSETS"]}  |  K: [{p["K_MIN"]},{p["K_MAX"]}]  |  '
          f'Window: {p["INITIAL_WINDOW"]}  |  TC: {p["TC"]*10000:.0f}bps')
    print()

    _step(1, 6, 'Settings loaded')

    _step(2, 6, 'Loading data (yfinance + macro CSV)...')
    prices, ret, assets, macro = load_data(p)

    _step(3, 6, 'Building feature matrix...')
    X_full = build_features(ret, assets, p)

    _step(4, 6, 'Running walk-forward backtest (this takes a few minutes)...')
    macro_score_full, vix_z_full = compute_macro_score(macro, p['MACRO_ZSCORE_WINDOW'])
    store, ALLOC = run_backtest(ret, assets, X_full, macro, macro_score_full, vix_z_full, p)

    _step(5, 6, 'Smoothing regimes & computing benchmarks...')
    dates_bt = pd.DatetimeIndex(store['dates'])
    train_m  = dates_bt <= pd.Timestamp(p['TRAIN_END'])
    test_m   = dates_bt >= pd.Timestamp(p['TEST_START'])

    reg_smooth    = smooth_regimes(np.array(store['pred_reg']),    p['SMOOTH_WINDOW'])
    ov_reg_smooth = smooth_regimes(np.array(store['overlay_reg']), p['SMOOTH_WINDOW'])
    ret_sm_hard    = recompute_hard(reg_smooth,    dates_bt, ret, ALLOC, p['TC'])
    ret_sm_overlay = recompute_hard(ov_reg_smooth, dates_bt, ret, ALLOC, p['TC'])

    bt_ret = ret.loc[dates_bt]
    bh_spy, static_6040, static_5a, rp_ret, ew_5 = build_benchmarks(
        bt_ret, len(assets), dates_bt)
    P_final = store['P_running'][-1] if store['P_running'] else np.eye(3) / 3

    _step(6, 6, 'Saving dashboard outputs...')
    save_outputs(
        dates_bt, ret_sm_hard, ret_sm_overlay, bh_spy, static_6040,
        static_5a, rp_ret, ew_5, reg_smooth, ov_reg_smooth,
        store, macro, macro_score_full, p, P_final, assets,
        STRESS_PERIODS, BULL_PERIODS, test_m, train_m, output_dir,
    )

    print()
    print('Pipeline complete. Launch dashboard:')
    print('  PYTHONPATH=. shiny run shiny_app/app.py --reload')


if __name__ == '__main__':
    main()
