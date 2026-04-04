"""
Tests for Stage 20 — Structured vs Unstructured Identification
=============================================================
Validates the estimation routines and pass/fail criteria.
"""
from __future__ import annotations

import numpy as np
import pytest

from hdr_validation.stages.stage_20_identification import (
    _build_mechanistic_prior,
    _build_true_A,
    estimate_structured,
    estimate_unstructured,
    run_stage_20,
)


# ── Mechanistic prior construction ──────────────────────────────────────────

class TestMechanisticPrior:
    def test_dimensions(self):
        rng = np.random.default_rng(1)
        D, J, mask, signs = _build_mechanistic_prior(8, rng)
        assert D.shape == (8,)
        assert J.shape == (8, 8)
        assert mask.shape == (8, 8)
        assert signs.shape == (8, 8)

    def test_D_positive(self):
        rng = np.random.default_rng(2)
        D, _, _, _ = _build_mechanistic_prior(8, rng)
        assert np.all(D > 0)

    def test_J_sparsity_count(self):
        rng = np.random.default_rng(3)
        _, J, mask, _ = _build_mechanistic_prior(8, rng)
        assert int(mask.sum()) == 23
        # J zero on diagonal
        assert np.allclose(np.diag(J), 0.0)

    def test_J_signs_positive(self):
        rng = np.random.default_rng(4)
        _, J, mask, signs = _build_mechanistic_prior(8, rng)
        assert np.all(J[mask] > 0), "All non-zero J entries should be positive"
        assert np.all(signs[mask] == 1.0)

    def test_J_zeros_respected(self):
        rng = np.random.default_rng(5)
        _, J, mask, _ = _build_mechanistic_prior(8, rng)
        off_diag_zero = ~mask.copy()
        np.fill_diagonal(off_diag_zero, False)
        assert np.allclose(J[off_diag_zero], 0.0)


# ── True A construction ────────────────────────────────────────────────────

class TestBuildTrueA:
    def test_structure(self):
        n = 8
        rng = np.random.default_rng(10)
        D, J, _, _ = _build_mechanistic_prior(n, rng)
        dt = 0.5
        A = _build_true_A(D, J, dt)
        expected = np.eye(n) + dt * (-np.diag(D) + J)
        assert np.allclose(A, expected)

    def test_identity_at_dt_zero(self):
        rng = np.random.default_rng(11)
        D, J, _, _ = _build_mechanistic_prior(8, rng)
        A = _build_true_A(D, J, dt=0.0)
        assert np.allclose(A, np.eye(8))


# ── Structured estimator ───────────────────────────────────────────────────

class TestStructuredEstimator:
    def test_low_noise_recovery(self):
        """With low noise and many samples, structured should recover A well."""
        n = 8
        rng = np.random.default_rng(20)
        D, J, mask, signs = _build_mechanistic_prior(n, rng)
        dt = 0.5
        A_true = _build_true_A(D, J, dt)

        # Rescale if unstable
        rho = np.max(np.abs(np.linalg.eigvals(A_true)))
        if rho >= 1.0:
            A_true *= 0.95 / rho
            M = (A_true - np.eye(n)) / dt
            D = -np.diag(M)
            J = M + np.diag(D)
            J[~mask] = 0.0

        # Generate data with small process noise (needed to keep signal alive
        # in a stable system — otherwise trajectory decays to machine zero)
        T = 2000
        X = np.zeros((T + 1, n))
        X[0] = rng.normal(scale=0.5, size=n)
        for t in range(T):
            X[t + 1] = A_true @ X[t] + rng.normal(scale=0.01, size=n)

        # Prior near truth
        J_prior = J.copy() + rng.normal(scale=0.05, size=(n, n)) * mask
        J_prior = np.maximum(J_prior, 0.0)
        J_prior[~mask] = 0.0

        A_hat, D_hat, J_hat = estimate_structured(
            X[:-1], X[1:], dt, mask, signs,
            D_init=D + rng.normal(scale=0.1, size=n),
            J_mech=J_prior,
            lambda_reg=0.001,
            max_iter=500,
            lr=0.002,
        )
        rel_err = np.linalg.norm(A_hat - A_true, 'fro') / np.linalg.norm(A_true, 'fro')
        assert rel_err < 0.20, f"Structured should recover A well with low noise, got {rel_err:.4f}"

    def test_sign_preserved(self):
        """Structured estimator should preserve sign constraints."""
        n = 8
        rng = np.random.default_rng(21)
        D, J, mask, signs = _build_mechanistic_prior(n, rng)
        dt = 0.5
        A_true = _build_true_A(D, J, dt)
        rho = np.max(np.abs(np.linalg.eigvals(A_true)))
        if rho >= 1.0:
            A_true *= 0.95 / rho
            M = (A_true - np.eye(n)) / dt
            D = -np.diag(M)
            J = M + np.diag(D)
            J[~mask] = 0.0

        T = 100
        X = np.zeros((T + 1, n))
        X[0] = rng.normal(scale=0.5, size=n)
        for t in range(T):
            X[t + 1] = A_true @ X[t] + rng.normal(scale=0.05, size=n)

        _, _, J_hat = estimate_structured(
            X[:-1], X[1:], dt, mask, signs,
            D_init=D + rng.normal(scale=0.2, size=n),
            max_iter=200,
        )
        # All positive-constrained entries should be >= 0
        pos_mask = signs > 0
        assert np.all(J_hat[pos_mask] >= 0), "Sign constraints should be preserved"

    def test_sparsity_preserved(self):
        """Structured estimator zeros should remain zero."""
        n = 8
        rng = np.random.default_rng(22)
        D, J, mask, signs = _build_mechanistic_prior(n, rng)
        dt = 0.5
        A_true = _build_true_A(D, J, dt)
        rho = np.max(np.abs(np.linalg.eigvals(A_true)))
        if rho >= 1.0:
            A_true *= 0.95 / rho

        T = 50
        X = np.zeros((T + 1, n))
        X[0] = rng.normal(scale=0.5, size=n)
        for t in range(T):
            X[t + 1] = A_true @ X[t] + rng.normal(scale=0.05, size=n)

        _, _, J_hat = estimate_structured(
            X[:-1], X[1:], dt, mask, signs, max_iter=100,
        )
        assert np.allclose(J_hat[~mask], 0.0), "Sparsity must be preserved"


# ── Unstructured estimator ─────────────────────────────────────────────────

class TestUnstructuredEstimator:
    def test_noiseless_recovery(self):
        """With noiseless data, OLS should recover A exactly."""
        n = 4
        rng = np.random.default_rng(30)
        A_true = rng.normal(scale=0.3, size=(n, n))
        A_true *= 0.8 / np.max(np.abs(np.linalg.eigvals(A_true)))

        T = 500
        X = np.zeros((T + 1, n))
        X[0] = rng.normal(size=n)
        for t in range(T):
            X[t + 1] = A_true @ X[t]

        A_hat = estimate_unstructured(X[:-1], X[1:], lambda_ridge=1e-8)
        assert np.allclose(A_hat, A_true, atol=1e-4), "OLS should recover noiseless A"


# ── Integration: full sweep (smoke) ─────────────────────────────────────────

@pytest.mark.parametrize("fast", [True])
def test_run_stage_20_smoke(fast, tmp_path, monkeypatch):
    """Smoke test: run with minimal settings and check output structure."""
    monkeypatch.setattr(
        "hdr_validation.stages.stage_20_identification.ROOT", tmp_path
    )
    result = run_stage_20(
        T_values=[20, 50, 100],
        n_trials=5,
        seed=99,
        fast_mode=True,
    )
    assert "criteria" in result
    assert "results" in result
    # Check that results file was written
    out_path = tmp_path / "results" / "stage_20" / "identification_comparison.json"
    assert out_path.exists()
    assert result["parameters"]["n"] == 8


def test_structured_beats_unstructured_low_T():
    """At T=50, structured should have lower Frobenius error than unstructured."""
    n = 8
    rng = np.random.default_rng(40)
    D, J, mask, signs = _build_mechanistic_prior(n, rng)
    dt = 0.5
    A_true = _build_true_A(D, J, dt)
    rho = np.max(np.abs(np.linalg.eigvals(A_true)))
    if rho >= 1.0:
        A_true *= 0.95 / rho
        M = (A_true - np.eye(n)) / dt
        D = -np.diag(M)
        J = M + np.diag(D)
        J[~mask] = 0.0

    Q_chol = np.eye(n) * np.sqrt(0.05)
    T = 50

    struct_errs = []
    unstruct_errs = []
    for trial in range(10):
        trial_rng = np.random.default_rng(400 + trial)
        X = np.zeros((T + 1, n))
        X[0] = trial_rng.normal(scale=0.5, size=n)
        for t in range(T):
            X[t + 1] = A_true @ X[t] + Q_chol @ trial_rng.normal(size=n)

        A_s, _, _ = estimate_structured(
            X[:-1], X[1:], dt, mask, signs,
            D_init=D + trial_rng.normal(scale=0.2, size=n),
            J_mech=J + trial_rng.normal(scale=0.05, size=(n, n)) * mask,
            max_iter=200,
        )
        A_u = estimate_unstructured(X[:-1], X[1:])

        A_norm = np.linalg.norm(A_true, 'fro')
        struct_errs.append(np.linalg.norm(A_s - A_true, 'fro') / A_norm)
        unstruct_errs.append(np.linalg.norm(A_u - A_true, 'fro') / A_norm)

    mean_s = np.mean(struct_errs)
    mean_u = np.mean(unstruct_errs)
    assert mean_s < mean_u, (
        f"Structured ({mean_s:.4f}) should beat unstructured ({mean_u:.4f}) at T={T}"
    )
