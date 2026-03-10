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
    K = state.cov @ C_o.T @ np.linalg.pinv(S)
    mean = state.mean + K @ innov
    cov = (np.eye(state.cov.shape[0]) - K @ C_o) @ state.cov
    sign, logdet = np.linalg.slogdet(S + 1e-8 * np.eye(S.shape[0]))
    quad = float(innov.T @ np.linalg.pinv(S) @ innov)
    log_like = float(-0.5 * (quad + logdet + len(obs_idx) * np.log(2 * np.pi)))
    return KalmanState(mean=mean, cov=cov), log_like
