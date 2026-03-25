import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.portfolio import (
    compute_ewm_covariance,
    compute_bl_posterior,
    mean_variance_optimize,
    compute_view_matrix,
    compute_view_returns,
    calibrate_omega,
    compute_portfolio_weights,
)
from src.utils import N_ASSETS

np.random.seed(42)
N = 500
dates = pd.date_range("2000-01-03", periods=N, freq="B")
rets = pd.DataFrame(np.random.randn(N, N_ASSETS) * 0.01, index=dates,
                    columns=['market','value','size','quality','growth','momentum'])


def test_ewm_covariance_shape():
    Sigma = compute_ewm_covariance(rets, halflife=126)
    assert Sigma.shape == (N_ASSETS, N_ASSETS)

def test_ewm_covariance_symmetric():
    Sigma = compute_ewm_covariance(rets, halflife=126)
    np.testing.assert_allclose(Sigma, Sigma.T, atol=1e-10)

def test_ewm_covariance_positive_definite():
    Sigma = compute_ewm_covariance(rets, halflife=126)
    eigvals = np.linalg.eigvalsh(Sigma)
    assert (eigvals > 0).all()

def test_view_matrix_shape():
    P = compute_view_matrix()
    assert P.shape == (5, 6)  # 5 factors × 6 assets

def test_view_matrix_long_short():
    P = compute_view_matrix()
    # Each row: +1 on one factor, -1 on market, 0 elsewhere
    assert (P.sum(axis=1) == 0).all()          # zero net position
    assert (np.abs(P).sum(axis=1) == 2).all()  # exactly 2 non-zero entries

def test_bl_posterior_no_views_returns_prior():
    """If Omega is huge (views uncertain), posterior ≈ prior."""
    Sigma = np.eye(N_ASSETS) * 0.01
    w_bmk = np.ones(N_ASSETS) / N_ASSETS
    delta, tau = 2.5, 0.05
    pi = delta * Sigma @ w_bmk
    P = compute_view_matrix()
    q = np.zeros(5)
    Omega = np.eye(5) * 1e10  # near-zero confidence
    mu_bl = compute_bl_posterior(pi, Sigma, P, q, tau, Omega)
    np.testing.assert_allclose(mu_bl, pi, atol=1e-4)

def test_mean_variance_optimize_sums_to_one():
    Sigma = compute_ewm_covariance(rets, halflife=126)
    mu = np.ones(N_ASSETS) * 0.0005
    w = mean_variance_optimize(mu, Sigma, delta=2.5)
    assert abs(w.sum() - 1.0) < 1e-6

def test_mean_variance_optimize_long_only():
    Sigma = compute_ewm_covariance(rets, halflife=126)
    mu = np.ones(N_ASSETS) * 0.0005
    w = mean_variance_optimize(mu, Sigma, delta=2.5)
    assert (w >= -1e-8).all()

def test_calibrate_omega_achieves_target_te():
    Sigma = compute_ewm_covariance(rets, halflife=126)
    w_bmk = np.ones(N_ASSETS) / N_ASSETS
    P = compute_view_matrix()
    q = np.array([0.0003, -0.0003, 0.0002, 0.0001, 0.0004])  # daily
    target_te = 0.02  # 2% annualized
    Omega = calibrate_omega(P, Sigma, tau=0.05, w_bmk=w_bmk,
                            delta=2.5, q=q, target_te=target_te)
    # Verify achieved TE is close to target
    pi = 2.5 * Sigma @ w_bmk
    mu_bl = compute_bl_posterior(pi, Sigma, P, q, 0.05, Omega)
    w_opt = mean_variance_optimize(mu_bl, Sigma, 2.5)
    w_active = w_opt - w_bmk
    te = np.sqrt(252 * w_active @ Sigma @ w_active)
    assert abs(te - target_te) < 0.005, f"TE={te:.4f}, target={target_te:.4f}"
