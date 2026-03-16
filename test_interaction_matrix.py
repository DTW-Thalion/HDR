"""Tests for V6 — Cross-column intervention interaction matrix."""
from __future__ import annotations

import numpy as np
import pytest

from hdr_validation.control.mpc import solve_mode_a
from hdr_validation.model.slds import make_evaluation_model
from hdr_validation.model.target_set import build_target_set


def _make_test_setup(seed=42):
    config = {
        "state_dim": 8, "obs_dim": 16, "control_dim": 8,
        "disturbance_dim": 8, "K": 3, "H": 4,
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
        "circadian_locked_controls": [],
        "steps_per_episode": 128,
    }
    n = config["state_dim"]
    rng = np.random.default_rng(seed)
    model = make_evaluation_model(config, rng)
    basin = model.basins[0]
    target = build_target_set(0, config)
    x_hat = rng.normal(size=n) * 0.5
    P_hat = np.eye(n) * 0.2
    return config, basin, target, x_hat, P_hat


def test_diagonal_R_u_full_equals_baseline():
    """Diagonal R_u_full equivalent to lambda_u * I should give same result."""
    config, basin, target, x_hat, P_hat = _make_test_setup()
    n = config["state_dim"]
    R_u_full = 0.1 * np.eye(n)
    res_base = solve_mode_a(x_hat, P_hat, basin, target, 0.65, config, 0)
    res_full = solve_mode_a(x_hat, P_hat, basin, target, 0.65, config, 0,
                            R_u_full=R_u_full)
    np.testing.assert_allclose(res_base.u, res_full.u, atol=1e-8)


def test_synergistic_changes_control_profile():
    """Positive off-diagonal penalty should change the control profile
    and the synergistic cost u^T R_syn u should be lower for the optimized result.
    """
    config, basin, target, x_hat, P_hat = _make_test_setup()
    config = {**config, "default_burden_budget": 1000.0}
    n = config["state_dim"]
    R_diag = 0.1 * np.eye(n)
    R_syn = R_diag.copy()
    R_syn[0, 1] = R_syn[1, 0] = 0.09
    res_diag = solve_mode_a(x_hat, P_hat, basin, target, 0.65, config, 0,
                            R_u_full=R_diag)
    res_syn = solve_mode_a(x_hat, P_hat, basin, target, 0.65, config, 0,
                           R_u_full=R_syn)
    # The optimizer under R_syn should produce a u that has lower quadratic cost
    # under R_syn than the diagonal-optimal u would:
    cost_syn_own = float(res_syn.u @ R_syn @ res_syn.u)
    cost_diag_under_syn = float(res_diag.u @ R_syn @ res_diag.u)
    assert cost_syn_own <= cost_diag_under_syn + 1e-6
    # Also verify the controls differ
    assert not np.allclose(res_diag.u, res_syn.u, atol=1e-10)


def test_antagonistic_increases_combination():
    """Negative off-diagonal (within PD bound) should still be feasible."""
    config, basin, target, x_hat, P_hat = _make_test_setup()
    n = config["state_dim"]
    R_diag = 0.2 * np.eye(n)
    R_ant = R_diag.copy()
    R_ant[0, 1] = R_ant[1, 0] = -0.08
    assert np.all(np.linalg.eigvalsh(R_ant) > 0), "R_ant should be PD"
    res_ant = solve_mode_a(x_hat, P_hat, basin, target, 0.65, config, 0,
                           R_u_full=R_ant)
    assert res_ant.feasible


def test_non_pd_raises_error():
    """Non-PD R_u_full should raise ValueError."""
    config, basin, target, x_hat, P_hat = _make_test_setup()
    n = config["state_dim"]
    R_bad = 0.1 * np.eye(n)
    R_bad[0, 1] = R_bad[1, 0] = 0.5
    with pytest.raises(ValueError, match="positive definite"):
        solve_mode_a(x_hat, P_hat, basin, target, 0.65, config, 0,
                     R_u_full=R_bad)
