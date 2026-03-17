from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanState:
    mean: np.ndarray
    cov: np.ndarray


def predict(state: KalmanState, A: np.ndarray, B: np.ndarray, u: np.ndarray, Q: np.ndarray, b: np.ndarray | None = None) -> KalmanState:
    b = np.zeros(A.shape[0]) if b is None else np.asarray(b, dtype=float)
    mean = A @ state.mean + B @ np.asarray(u, dtype=float) + b
    cov = A @ state.cov @ A.T + Q
    return KalmanState(mean=mean, cov=cov)


def update(
    state: KalmanState,
    y: np.ndarray,
    mask: np.ndarray,
    C: np.ndarray,
    R: np.ndarray,
    c: np.ndarray | None = None,
) -> tuple[KalmanState, float]:
    c = np.zeros(C.shape[0]) if c is None else np.asarray(c, dtype=float)
    obs_idx = np.where(mask.astype(bool))[0]
    if obs_idx.size == 0:
        return state, 0.0
    C_o = C[obs_idx]
    R_o = R[np.ix_(obs_idx, obs_idx)]
    y_o = y[obs_idx]
    c_o = c[obs_idx]
    innov = y_o - (C_o @ state.mean + c_o)
    S = C_o @ state.cov @ C_o.T + R_o
    S = 0.5 * (S + S.T)

    # Single factorization for both Kalman gain and log-likelihood
    try:
        from scipy.linalg import cho_factor, cho_solve
        L_S, low = cho_factor(S)
        # K = P @ C_o^T @ S^{-1}  via  solve S @ Z = (P @ C_o^T)^T
        PCT = state.cov @ C_o.T
        Z = cho_solve((L_S, low), PCT.T)  # (m_obs, n)
        K = Z.T                           # (n, m_obs)
        quad = float(innov @ cho_solve((L_S, low), innov))
        logdet = float(2.0 * np.sum(np.log(np.diag(L_S))))
    except (np.linalg.LinAlgError, Exception):
        # Fallback to pinv if Cholesky fails (S not SPD due to numerics)
        S_inv = np.linalg.pinv(S)
        K = state.cov @ C_o.T @ S_inv
        quad = float(innov.T @ S_inv @ innov)
        sign, logdet = np.linalg.slogdet(S + 1e-8 * np.eye(S.shape[0]))
        logdet = float(logdet)

    mean = state.mean + K @ innov
    cov = (np.eye(state.cov.shape[0]) - K @ C_o) @ state.cov
    log_like = float(-0.5 * (quad + logdet + len(obs_idx) * np.log(2 * np.pi)))
    return KalmanState(mean=mean, cov=cov), log_like
