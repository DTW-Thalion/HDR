from __future__ import annotations

import numpy as np


def weighted_observation_update(X: np.ndarray, Y: np.ndarray, W: np.ndarray, ridge: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    W = np.asarray(W, dtype=float).reshape(-1, 1)
    XtWX = X.T @ (W * X) + ridge * np.eye(X.shape[1])
    XtWY = X.T @ (W * Y)
    C = np.linalg.pinv(XtWX) @ XtWY
    residual = Y - X @ C
    R = (residual.T @ (W * residual)) / max(float(np.sum(W)), 1.0)
    return C.T, R


def fit_linear_dynamics(
    X: np.ndarray,
    U: np.ndarray,
    X_next: np.ndarray,
    weights: np.ndarray | None = None,
    prior_theta: np.ndarray | None = None,
    ridge: float = 1e-3,
    prior_strength: float = 0.0,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    U = np.asarray(U, dtype=float)
    X_next = np.asarray(X_next, dtype=float)
    F = np.hstack([X, U, np.ones((len(X), 1))])
    if weights is None:
        W = np.ones((len(X), 1))
    else:
        W = np.asarray(weights, dtype=float).reshape(-1, 1)
    lhs = F.T @ (W * F) + ridge * np.eye(F.shape[1])
    rhs = F.T @ (W * X_next)
    if prior_theta is not None and prior_strength > 0:
        lhs += prior_strength * np.eye(F.shape[1])
        rhs += prior_strength * prior_theta
    theta = np.linalg.pinv(lhs) @ rhs
    return theta


def unpack_dynamics_theta(theta: np.ndarray, state_dim: int, control_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = theta[:state_dim].T
    B = theta[state_dim:state_dim + control_dim].T
    b = theta[-1].T
    return A, B, b
