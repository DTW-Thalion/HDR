"""Tests for mRPI terminal set and tube-MPC (Task 1: IG1 Path B)."""
from __future__ import annotations

import numpy as np
import pytest

from hdr_validation.control.tube_mpc import (
    compute_disturbance_set,
    compute_mRPI_zonotope,
    solve_tube_mpc,
    zonotope_containment_check,
)
from hdr_validation.control.lqr import dlqr
from hdr_validation.model.slds import make_evaluation_model
from hdr_validation.model.target_set import build_target_set


def test_compute_disturbance_set_positive():
    """Verify chi2_bound > 0 and Q_w_inv is SPD for n=4, 8."""
    for n in [4, 8]:
        Q_w = 0.05 * np.eye(n)
        Q_w_inv, chi2_bound = compute_disturbance_set(Q_w, n)
        assert chi2_bound > 0.0
        eigvals = np.linalg.eigvalsh(Q_w_inv)
        assert np.all(eigvals > 0), "Q_w_inv should be SPD"


def test_mRPI_converges_rho_055():
    """A_cl = 0.55 * I_4, Q_w = 0.01 * I_4. Verify fast convergence."""
    n = 4
    A_cl = 0.55 * np.eye(n)
    Q_w = 0.01 * np.eye(n)
    _, chi2_bound = compute_disturbance_set(Q_w, n)
    result = compute_mRPI_zonotope(A_cl, Q_w, chi2_bound)
    assert result["alpha_s"] < 1e-3
    assert result["iterations"] < 20
    assert result["G"].shape[0] == n


def test_mRPI_converges_rho_072():
    """Same for rho=0.72. Verify convergence."""
    n = 4
    A_cl = 0.72 * np.eye(n)
    Q_w = 0.01 * np.eye(n)
    _, chi2_bound = compute_disturbance_set(Q_w, n)
    result = compute_mRPI_zonotope(A_cl, Q_w, chi2_bound)
    assert result["alpha_s"] < 1e-3
    assert result["G"].shape[0] == n


def test_mRPI_converges_rho_096():
    """rho=0.96, may need more iterations. Verify convergence."""
    n = 4
    A_cl = 0.96 * np.eye(n)
    Q_w = 0.01 * np.eye(n)
    _, chi2_bound = compute_disturbance_set(Q_w, n)
    result = compute_mRPI_zonotope(A_cl, Q_w, chi2_bound, epsilon=0.01, max_iter=200)
    assert result["alpha_s"] < 0.01
    assert result["G"].shape[0] == n


def test_zonotope_containment_origin():
    """Origin should be inside any mRPI set centered at origin."""
    n = 4
    A_cl = 0.55 * np.eye(n)
    Q_w = 0.01 * np.eye(n)
    _, chi2_bound = compute_disturbance_set(Q_w, n)
    result = compute_mRPI_zonotope(A_cl, Q_w, chi2_bound)
    assert zonotope_containment_check(np.zeros(n), result["G"], result["center"])


def test_zonotope_containment_outside():
    """A point at 100 * ones should be outside."""
    n = 4
    A_cl = 0.55 * np.eye(n)
    Q_w = 0.01 * np.eye(n)
    _, chi2_bound = compute_disturbance_set(Q_w, n)
    result = compute_mRPI_zonotope(A_cl, Q_w, chi2_bound)
    assert not zonotope_containment_check(100 * np.ones(n), result["G"], result["center"])


def test_tube_mpc_returns_valid_result():
    """Build minimal config + model, compute mRPI, call solve_tube_mpc."""
    config = {
        "state_dim": 8, "obs_dim": 16, "control_dim": 8,
        "disturbance_dim": 8, "K": 3, "H": 6,
        "w1": 1.0, "w2": 0.5, "w3": 0.3, "lambda_u": 0.1,
        "alpha_i": 0.05, "eps_safe": 0.01,
        "rho_reference": [0.72, 0.96, 0.55],
        "max_dwell_len": 128,
        "model_mismatch_bound": 0.347,
        "kappa_lo": 0.55, "kappa_hi": 0.75,
        "pA": 0.70, "qmin": 0.15,
        "steps_per_day": 48, "dt_minutes": 30,
        "coherence_window": 24,
        "default_burden_budget": 28.0,
        "circadian_locked_controls": [5, 6],
        "steps_per_episode": 128,
    }
    rng = np.random.default_rng(42)
    model = make_evaluation_model(config, rng)
    basin = model.basins[0]
    target = build_target_set(0, config)
    n = config["state_dim"]

    Q_lqr = np.eye(n)
    R_lqr = np.eye(n) * 0.1
    K_fb, P_dare = dlqr(basin.A, basin.B, Q_lqr, R_lqr)
    A_cl = basin.A - basin.B @ K_fb
    _, chi2_bound = compute_disturbance_set(basin.Q, n)
    mRPI_data = compute_mRPI_zonotope(A_cl, basin.Q, chi2_bound)

    x_hat = rng.normal(size=n) * 0.3
    P_hat = np.eye(n) * 0.2

    result = solve_tube_mpc(x_hat, P_hat, basin, target, mRPI_data, K_fb,
                            config=config, step=0)
    assert np.all(np.isfinite(result.u))
    assert result.feasible


@pytest.mark.slow
def test_tube_mpc_containment_rate():
    """Simulate 5 seeds x 4 episodes x 128 steps using tube-MPC.
    Assert containment_rate >= 0.90 for all basins.
    """
    config = {
        "state_dim": 8, "obs_dim": 16, "control_dim": 8,
        "disturbance_dim": 8, "K": 3, "H": 6,
        "w1": 1.0, "w2": 0.5, "w3": 0.3, "lambda_u": 0.1,
        "alpha_i": 0.05, "eps_safe": 0.01,
        "rho_reference": [0.72, 0.96, 0.55],
        "max_dwell_len": 128,
        "model_mismatch_bound": 0.347,
        "kappa_lo": 0.55, "kappa_hi": 0.75,
        "pA": 0.70, "qmin": 0.15,
        "steps_per_day": 48, "dt_minutes": 30,
        "coherence_window": 24,
        "default_burden_budget": 56.0,
        "circadian_locked_controls": [5, 6],
        "steps_per_episode": 128,
    }
    n = config["state_dim"]
    model_rng = np.random.default_rng(42)
    model = make_evaluation_model(config, model_rng)

    Q_lqr = np.eye(n)
    R_lqr = np.eye(n) * 0.1

    # Precompute per-basin mRPI data
    basin_data = {}
    for k_idx, basin in enumerate(model.basins):
        K_fb, P_dare = dlqr(basin.A, basin.B, Q_lqr, R_lqr)
        A_cl = basin.A - basin.B @ K_fb
        _, chi2_bound = compute_disturbance_set(basin.Q, n)
        mRPI_data = compute_mRPI_zonotope(A_cl, basin.Q, chi2_bound, epsilon=0.01)
        basin_data[k_idx] = {"K_fb": K_fb, "mRPI": mRPI_data}

    seeds = [101, 202, 303, 404, 505]
    containment_counts = {k: {"inside": 0, "total": 0} for k in range(3)}

    for seed in seeds:
        rng = np.random.default_rng(seed)
        for ep in range(4):
            k_idx = rng.integers(0, 3)
            basin = model.basins[k_idx]
            target = build_target_set(k_idx, config)
            K_fb = basin_data[k_idx]["K_fb"]
            mRPI = basin_data[k_idx]["mRPI"]

            x = rng.normal(size=n) * 0.1  # start near origin
            P_hat = np.eye(n) * 0.2

            for t in range(128):
                try:
                    res = solve_tube_mpc(x, P_hat, basin, target, mRPI, K_fb,
                                         config=config, step=t)
                    u = res.u
                except Exception:
                    u = np.zeros(n)

                # Check containment
                inside = zonotope_containment_check(x, mRPI["G"], mRPI["center"])
                containment_counts[k_idx]["total"] += 1
                if inside:
                    containment_counts[k_idx]["inside"] += 1

                w = rng.multivariate_normal(np.zeros(n), basin.Q)
                x = basin.A @ x + basin.B @ u + basin.b + w

    for k_idx in range(3):
        total = containment_counts[k_idx]["total"]
        if total > 0:
            rate = containment_counts[k_idx]["inside"] / total
            assert rate >= 0.90, (
                f"Basin {k_idx}: containment_rate={rate:.3f} < 0.90"
            )
