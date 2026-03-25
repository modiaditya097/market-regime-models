"""Black-Litterman portfolio construction with MV optimization."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq

from src.utils import FACTORS, ASSETS, N_ASSETS, TRADING_DAYS


def compute_ewm_covariance(returns: pd.DataFrame, halflife: int) -> np.ndarray:
    """EWM covariance matrix (N×N) estimated from a returns DataFrame."""
    alpha = 1 - np.exp(-np.log(2) / halflife)
    n = len(returns)
    weights = (1 - alpha) ** np.arange(n - 1, -1, -1)
    weights /= weights.sum()
    X = returns.values - (weights @ returns.values)  # demean
    Sigma = (X * weights[:, None]).T @ X
    # Symmetrize for numerical stability
    return (Sigma + Sigma.T) / 2


def compute_view_matrix() -> np.ndarray:
    """
    5×6 view matrix P.
    Row k: +1 on factor k, -1 on market.
    Asset order: market, value, size, quality, growth, momentum.
    """
    P = np.zeros((5, 6))
    market_idx = ASSETS.index('market')
    for i, factor in enumerate(FACTORS):
        factor_idx = ASSETS.index(factor)
        P[i, factor_idx] = 1.0
        P[i, market_idx] = -1.0
    return P


def compute_bl_posterior(
    pi: np.ndarray,
    Sigma: np.ndarray,
    P: np.ndarray,
    q: np.ndarray,
    tau: float,
    Omega: np.ndarray,
) -> np.ndarray:
    """
    Black-Litterman posterior expected returns.
    E[R] = [(tauSigma)^-1 + P'Omega^-1 P]^-1 [(tauSigma)^-1 pi + P'Omega^-1 q]
    """
    tau_sigma_inv = np.linalg.inv(tau * Sigma)
    omega_inv     = np.linalg.inv(Omega)
    M   = tau_sigma_inv + P.T @ omega_inv @ P
    rhs = tau_sigma_inv @ pi + P.T @ omega_inv @ q
    return np.linalg.solve(M, rhs)


def mean_variance_optimize(
    mu: np.ndarray, Sigma: np.ndarray, delta: float
) -> np.ndarray:
    """
    Solve: max w'mu - (delta/2) w'Sigma w
    s.t.  w >= 0,  sum(w) = 1
    Returns optimal weights.
    """
    def neg_utility(w):
        return -(w @ mu - (delta / 2.0) * w @ Sigma @ w)

    def grad(w):
        return -(mu - delta * Sigma @ w)

    w0 = np.ones(N_ASSETS) / N_ASSETS
    constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1.0,
                   'jac': lambda w: np.ones(N_ASSETS)}
    bounds = [(0.0, 1.0)] * N_ASSETS

    res = minimize(neg_utility, w0, jac=grad, method='SLSQP',
                   bounds=bounds, constraints=constraints,
                   options={'ftol': 1e-10, 'maxiter': 2000})
    w = np.clip(res.x, 0.0, 1.0)
    return w / w.sum()  # re-normalize for numerical safety


def compute_view_returns(
    regime_labels: dict,
    active_ret_history: dict,
    in_sample_labels: dict,
) -> np.ndarray:
    """
    q[k] = mean daily active return of factor k during its current regime
           computed over the training period.
    Capped at ±5%/252 (daily equivalent of ±5% annualized).
    """
    cap = 0.05 / TRADING_DAYS
    q = np.zeros(5)
    for i, factor in enumerate(FACTORS):
        current_regime = regime_labels.get(factor, 0)
        ret_hist = active_ret_history[factor]
        labels_hist = in_sample_labels[factor]
        aligned = labels_hist.reindex(ret_hist.index).dropna()
        ret_aligned = ret_hist.reindex(aligned.index)
        mask = aligned == current_regime
        if mask.sum() > 0:
            q[i] = float(ret_aligned[mask].mean())
        q[i] = float(np.clip(q[i], -cap, cap))
    return q


def calibrate_omega(
    P: np.ndarray,
    Sigma: np.ndarray,
    tau: float,
    w_bmk: np.ndarray,
    delta: float,
    q: np.ndarray,
    target_te: float,
) -> np.ndarray:
    """
    Find scalar c such that the BL portfolio's tracking error equals target_te.
    Omega = c * diag(P * tau*Sigma * P').
    Larger c → less confident views → portfolio closer to EW → smaller TE.
    Uses brentq on [1e-8, 1e10].
    """
    base_diag = np.diag(P @ (tau * Sigma) @ P.T)
    pi = delta * Sigma @ w_bmk

    def compute_te(c: float) -> float:
        Omega = np.diag(c * base_diag)
        mu_bl = compute_bl_posterior(pi, Sigma, P, q, tau, Omega)
        w_opt = mean_variance_optimize(mu_bl, Sigma, delta)
        w_active = w_opt - w_bmk
        return float(np.sqrt(TRADING_DAYS * w_active @ Sigma @ w_active))

    try:
        te_low  = compute_te(1e-8)
        te_high = compute_te(1e10)
        if target_te > te_low:
            # Can't achieve target TE even at maximum confidence
            c_opt = 1e-8
        elif target_te < te_high:
            c_opt = 1e10
        else:
            c_opt = brentq(lambda c: compute_te(c) - target_te, 1e-8, 1e10,
                           xtol=1e-6, rtol=1e-6, maxiter=100)
    except Exception:
        c_opt = 1.0  # fallback: neutral

    return np.diag(c_opt * base_diag)


def compute_portfolio_weights(
    today_regime: dict,
    active_ret_history: dict,
    in_sample_labels: dict,
    total_ret_history: pd.DataFrame,
    cfg: dict,
) -> np.ndarray:
    """
    Full BL pipeline for one rebalancing day. Returns 6-element weight vector.
    """
    bl = cfg["black_litterman"]
    delta    = bl["risk_aversion"]
    halflife = bl["cov_halflife"]
    tau      = bl["tau"]
    target_te = bl["target_tracking_error"]

    w_bmk = np.ones(N_ASSETS) / N_ASSETS
    Sigma  = compute_ewm_covariance(total_ret_history, halflife)
    P      = compute_view_matrix()
    q      = compute_view_returns(today_regime, active_ret_history, in_sample_labels)
    pi     = delta * Sigma @ w_bmk
    Omega  = calibrate_omega(P, Sigma, tau, w_bmk, delta, q, target_te)
    mu_bl  = compute_bl_posterior(pi, Sigma, P, q, tau, Omega)
    return mean_variance_optimize(mu_bl, Sigma, delta)


def run_portfolio_construction(
    regime_labels: dict,
    in_sample_labels: dict,
    total_returns: pd.DataFrame,
    active_returns: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """
    Construct daily portfolio weights over the test period.

    regime_labels: dict[factor -> daily pd.Series of delayed labels (0=bull,1=bear)]
    in_sample_labels: dict[factor -> training-period labels from last refit]
    total_returns: 6-column daily returns DataFrame
    active_returns: 5-column daily active returns DataFrame

    Returns pd.DataFrame of shape (T, 6) with portfolio weights.
    """
    bl = cfg["black_litterman"]

    # Test period = union of regime label dates
    all_label_dates = sorted(set.union(*[set(s.index) for s in regime_labels.values()]))
    if not all_label_dates:
        return pd.DataFrame()

    # Detect rebalancing days: regime changes or month-end
    regime_df = pd.DataFrame(regime_labels).reindex(all_label_dates)
    changed = regime_df.diff().abs().sum(axis=1) > 0
    month_ends = pd.Series(all_label_dates).apply(
        lambda d: d.month != (pd.Timestamp(d) + pd.Timedelta(days=1)).month
    ).values

    rebal_mask = changed.values | month_ends
    rebal_dates = [d for d, m in zip(all_label_dates, rebal_mask) if m]

    weights_dict = {}
    current_weights = np.ones(N_ASSETS) / N_ASSETS

    for date in all_label_dates:
        if date in rebal_dates:
            # Build history available up to (but not including) this date
            hist_total  = total_returns[total_returns.index < date]
            hist_active = {f: active_returns[f][active_returns.index < date]
                           for f in regime_labels}
            today_regime = {
                f: int(regime_labels[f].get(date, 0))
                for f in regime_labels
            }
            try:
                current_weights = compute_portfolio_weights(
                    today_regime, hist_active, in_sample_labels,
                    hist_total, cfg
                )
            except Exception:
                current_weights = np.ones(N_ASSETS) / N_ASSETS
        weights_dict[date] = current_weights.copy()

    weights_df = pd.DataFrame(weights_dict, index=ASSETS).T
    weights_df.index = pd.DatetimeIndex(weights_df.index)
    return weights_df
