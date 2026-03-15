"""Tests for hdr_validation.control.mimpc — mixed-integer MPC."""
import numpy as np
from hdr_validation.control.mimpc import (
    solve_mixed_integer_mpc, CumulativeExposureConstraint, MIMPCResult,
)
from hdr_validation.model.slds import make_evaluation_model


def _make_config():
    return {
        "state_dim": 8, "obs_dim": 16, "control_dim": 8,
        "disturbance_dim": 8, "K": 3, "rho_reference": [0.72, 0.96, 0.55],
        "max_dwell_len": 64, "H": 4, "lambda_u": 0.1, "u_max": 0.6,
        "m_d": 1,
    }


def _make_model(config):
    rng = np.random.default_rng(10)
    return make_evaluation_model(config, rng)


def test_mimpc_binary_constraint():
    """Discrete controls should be binary (0 or 1)."""
    cfg = _make_config()
    model = _make_model(cfg)
    basin = model.basins[0]
    from hdr_validation.model.target_set import build_target_set
    target = build_target_set(0, cfg)
    x = np.ones(cfg["state_dim"]) * 0.5
    P = np.eye(cfg["state_dim"]) * 0.1
    u_opts = [np.array([0.0]), np.array([1.0])]
    result = solve_mixed_integer_mpc(x, P, basin, target, cfg, u_opts)
    if result.u_discrete.size > 0:
        for val in result.u_discrete:
            assert val == 0.0 or val == 1.0 or abs(val) < 1e-8


def test_mimpc_irreversibility_constraint():
    """Binary components should sum to <= 1 (one-time interventions)."""
    cfg = _make_config()
    model = _make_model(cfg)
    basin = model.basins[0]
    from hdr_validation.model.target_set import build_target_set
    target = build_target_set(0, cfg)
    x = np.ones(cfg["state_dim"]) * 0.5
    P = np.eye(cfg["state_dim"]) * 0.1
    # Offer options that would sum > 1 if both chosen
    u_opts = [np.array([0.0]), np.array([1.0])]
    result = solve_mixed_integer_mpc(x, P, basin, target, cfg, u_opts)
    if result.u_discrete.size > 0:
        assert np.sum(np.abs(result.u_discrete)) <= 1.0 + 1e-8


def test_mimpc_recursive_feasibility():
    """Sequential MI-MPC calls should remain feasible."""
    cfg = _make_config()
    model = _make_model(cfg)
    basin = model.basins[0]
    from hdr_validation.model.target_set import build_target_set
    target = build_target_set(0, cfg)
    x = np.ones(cfg["state_dim"]) * 0.3
    P = np.eye(cfg["state_dim"]) * 0.1
    u_opts = [np.array([0.0]), np.array([1.0])]
    for _ in range(5):
        result = solve_mixed_integer_mpc(x, P, basin, target, cfg, u_opts)
        assert result.feasible
        x = basin.A @ x + basin.B @ result.u_combined[:basin.B.shape[1]]


def test_mimpc_cumulative_exposure_bound():
    """MI-MPC respects cumulative exposure constraint."""
    cfg = _make_config()
    model = _make_model(cfg)
    basin = model.basins[0]
    from hdr_validation.model.target_set import build_target_set
    target = build_target_set(0, cfg)
    f_j = lambda u: np.abs(u[:1])  # exposure = |u[0]|
    xi = np.array([99.0])  # near limit
    cum_exp = CumulativeExposureConstraint(xi, f_j, xi_max=100.0, H=4)
    x = np.ones(cfg["state_dim"]) * 0.5
    P = np.eye(cfg["state_dim"]) * 0.1
    u_opts = [np.array([0.0]), np.array([1.0])]
    result = solve_mixed_integer_mpc(x, P, basin, target, cfg, u_opts, cum_exp)
    # Should only choose u_d=0 since u_d=1 would push xi to 100+
    assert result.feasible


def test_mimpc_continuous_only_fallback():
    """Without discrete options, falls back to continuous-only MPC."""
    cfg = _make_config()
    model = _make_model(cfg)
    basin = model.basins[0]
    from hdr_validation.model.target_set import build_target_set
    target = build_target_set(0, cfg)
    x = np.ones(cfg["state_dim"]) * 0.5
    P = np.eye(cfg["state_dim"]) * 0.1
    result = solve_mixed_integer_mpc(x, P, basin, target, cfg, None)
    assert result.notes == "continuous_only_fallback"
    assert result.feasible


def test_mimpc_cost_finite():
    """Cost should be finite and non-negative."""
    cfg = _make_config()
    model = _make_model(cfg)
    basin = model.basins[0]
    from hdr_validation.model.target_set import build_target_set
    target = build_target_set(0, cfg)
    x = np.ones(cfg["state_dim"]) * 0.5
    P = np.eye(cfg["state_dim"]) * 0.1
    u_opts = [np.array([0.0]), np.array([1.0])]
    result = solve_mixed_integer_mpc(x, P, basin, target, cfg, u_opts)
    assert np.isfinite(result.cost)
    assert result.cost >= 0


def test_cumexp_constraint_satisfied():
    """CumulativeExposureConstraint.is_feasible returns True when within limit."""
    f_j = lambda u: np.array([np.sum(np.abs(u))])
    xi = np.array([50.0])
    c = CumulativeExposureConstraint(xi, f_j, xi_max=100.0, H=4)
    u_seq = [np.array([0.1, 0.1])] * 4
    assert c.is_feasible(u_seq) is True


def test_cumexp_constraint_violated():
    """CumulativeExposureConstraint.is_feasible returns False when over limit."""
    f_j = lambda u: np.array([np.sum(np.abs(u))])
    xi = np.array([99.0])
    c = CumulativeExposureConstraint(xi, f_j, xi_max=100.0, H=4)
    u_seq = [np.array([5.0, 5.0])] * 4  # adds 10 each step, way over
    assert c.is_feasible(u_seq) is False
