from __future__ import annotations

import numpy as np

from ..model.coherence import coherence_grad
from ..model.safety import apply_control_constraints
from .lqr import dlqr


def open_loop_policy(obs, *args, **kwargs):
    target = obs["target"]
    n = len(target.box_low)
    return np.zeros(n)


def pooled_lqr_policy(obs, pooled_basin, config: dict, used_burden: float = 0.0):
    K, _ = dlqr(pooled_basin.A, pooled_basin.B, np.eye(pooled_basin.A.shape[0]), np.eye(pooled_basin.B.shape[1]) * float(config["lambda_u"]))
    x_ref = 0.5 * (obs["target"].box_low + obs["target"].box_high)
    u = -K @ (obs["x_hat"] - x_ref)
    u, info = apply_control_constraints(u, config, step=int(obs["t"]), used_burden=used_burden)
    return u


def basin_lqr_policy(obs, basin, config: dict, used_burden: float = 0.0):
    K, _ = dlqr(basin.A, basin.B, np.eye(basin.A.shape[0]), np.eye(basin.B.shape[1]) * float(config["lambda_u"]))
    x_ref = 0.5 * (obs["target"].box_low + obs["target"].box_high)
    u = -K @ (obs["x_hat"] - x_ref)
    u, info = apply_control_constraints(u, config, step=int(obs["t"]), used_burden=used_burden)
    return u


def myopic_policy(obs, basin, config: dict, used_burden: float = 0.0, with_coherence: bool = False):
    x = obs["x_hat"]
    x_ref = 0.5 * (obs["target"].box_low + obs["target"].box_high)
    B = basin.B
    A = basin.A
    R = np.eye(B.shape[1]) * float(config["lambda_u"])
    H = B.T @ B + R
    rhs = B.T @ (x_ref - A @ x - basin.b)
    u = np.linalg.pinv(H) @ rhs
    if with_coherence:
        grad = coherence_grad(float(obs.get("kappa_hat", 0.5)), float(config["kappa_lo"]), float(config["kappa_hi"]))
        for idx in [1, 5, 6]:
            if idx < len(u):
                u[idx] -= 0.05 * grad
    u, info = apply_control_constraints(u, config, step=int(obs["t"]), used_burden=used_burden)
    return u
