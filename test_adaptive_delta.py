"""Tests for S4 Path C — Adaptive mismatch bound via FF-RLS."""
from __future__ import annotations

import numpy as np

from hdr_validation.model.adaptive import FFRLSEstimator, DriftDetector


def test_adaptive_delta_A_zero_initially():
    n = 4
    est = FFRLSEstimator(n, lambda_ff=0.98)
    delta = est.adaptive_delta_A(gamma_margin=0.0)
    assert delta == 0.0
    delta_with_margin = est.adaptive_delta_A(gamma_margin=2.0)
    assert delta_with_margin > 0.0


def test_adaptive_delta_A_increases_with_drift():
    n = 4
    est = FFRLSEstimator(n, lambda_ff=0.98)
    rng = np.random.default_rng(42)
    A_true = 0.8 * np.eye(n) + 0.05 * rng.normal(size=(n, n))
    x = rng.normal(size=n)
    deltas = []
    for t in range(50):
        x_new = A_true @ x + rng.normal(size=n) * 0.01
        est.update(x_new, x)
        deltas.append(est.adaptive_delta_A(gamma_margin=0.0))
        x = x_new
    assert deltas[-1] > deltas[0]


def test_sigma_rls_decreases_with_data():
    n = 4
    # Use lambda_ff=1.0 (standard RLS, no forgetting) so P_rls shrinks
    est = FFRLSEstimator(n, lambda_ff=1.0)
    rng = np.random.default_rng(42)
    A_true = 0.7 * np.eye(n)
    x = rng.normal(size=n) * 0.5
    sigma_initial = est.sigma_rls()
    for t in range(100):
        x_new = A_true @ x + rng.normal(size=n) * 0.01
        est.update(x_new, x)
        x = x_new
    sigma_final = est.sigma_rls()
    assert sigma_final < sigma_initial


def test_adaptive_mubar_relaxes_when_delta_small():
    n = 4
    est = FFRLSEstimator(n, lambda_ff=0.98)
    det = DriftDetector(Delta_A_max=0.5)
    mubar = det.adaptive_mubar_required(
        est, c_ISS=2.0, Delta_B=0.1, K_norm=0.5,
        alpha=0.3, epsilon_ctrl=0.5, gamma_margin=0.0
    )
    assert mubar == 1.0


def test_adaptive_mubar_tightens_with_drift():
    n = 4
    est = FFRLSEstimator(n, lambda_ff=0.98)
    det = DriftDetector(Delta_A_max=0.5)
    est.A_hat = est.A_hat_initial + 0.3 * np.eye(n)
    mubar_drifted = det.adaptive_mubar_required(
        est, c_ISS=2.0, Delta_B=0.1, K_norm=0.5,
        alpha=0.3, epsilon_ctrl=0.5, gamma_margin=0.0
    )
    est.A_hat = est.A_hat_initial.copy()
    mubar_clean = det.adaptive_mubar_required(
        est, c_ISS=2.0, Delta_B=0.1, K_norm=0.5,
        alpha=0.3, epsilon_ctrl=0.5, gamma_margin=0.0
    )
    assert mubar_drifted < mubar_clean


def test_adaptive_mubar_in_unit_interval():
    n = 4
    est = FFRLSEstimator(n, lambda_ff=0.98)
    det = DriftDetector(Delta_A_max=0.5)
    rng = np.random.default_rng(42)
    A_true = 0.7 * np.eye(n)
    x = rng.normal(size=n)
    for t in range(50):
        x_new = A_true @ x + rng.normal(size=n) * 0.01
        est.update(x_new, x)
        x = x_new
        mubar = det.adaptive_mubar_required(
            est, c_ISS=2.0, Delta_B=0.1, K_norm=0.5,
            alpha=0.3, epsilon_ctrl=0.5
        )
        assert 0.0 <= mubar <= 1.0
