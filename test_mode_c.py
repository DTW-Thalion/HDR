import numpy as np

from hdr_validation.model.target_set import build_target_set
from hdr_validation.model.recovery import tau_tilde


def test_tau_nonnegative_and_distance_monotone():
    config = {"state_dim": 8, "steps_per_day": 48}
    target = build_target_set(0, config)
    Q = np.eye(8)
    x_near = np.zeros(8)
    x_far = np.ones(8) * 1.2
    tau_near = tau_tilde(x_near, target, Q, rho=0.72)
    tau_far = tau_tilde(x_far, target, Q, rho=0.72)
    assert tau_near >= 0
    assert tau_far >= tau_near
