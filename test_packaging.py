import numpy as np

from hdr_validation.control.mpc import solve_mode_a
from hdr_validation.model.slds import make_evaluation_model
from hdr_validation.model.target_set import build_target_set


def test_mpc_returns_bounded_control():
    config = {
        "state_dim": 8,
        "obs_dim": 16,
        "control_dim": 8,
        "disturbance_dim": 8,
        "K": 3,
        "rho_reference": [0.72, 0.96, 0.55],
        "max_dwell_len": 128,
        "H": 6,
        "kappa_lo": 0.55,
        "kappa_hi": 0.75,
        "w1": 1.0,
        "w2": 0.5,
        "w3": 0.3,
        "lambda_u": 0.1,
        "alpha_i": 0.05,
        "eps_safe": 0.5,
        "steps_per_day": 48,
        "default_burden_budget": 14.0,
        "circadian_locked_controls": [5, 6],
    }
    rng = np.random.default_rng(2)
    model = make_evaluation_model(config, rng)
    target = build_target_set(0, config)
    x = np.ones(8)
    P = np.eye(8) * 0.1
    res = solve_mode_a(x, P, model.basins[0], target, 0.4, config, step=0)
    assert res.u.shape == (8,)
    assert np.all(np.abs(res.u) <= 0.6 + 1e-8)
