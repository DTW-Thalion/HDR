from __future__ import annotations

import numpy as np


def observation_schedule(T: int, obs_dim: int, rng: np.random.Generator, profile_name: str = "standard") -> np.ndarray:
    mask = np.zeros((T, obs_dim), dtype=int)
    for t in range(T):
        # fast channels
        mask[t, :8] = (rng.uniform(size=min(8, obs_dim)) > 0.1).astype(int)
        # slow channels every 4 steps
        if obs_dim > 8 and t % 4 == 0:
            end = min(12, obs_dim)
            mask[t, 8:end] = (rng.uniform(size=end - 8) > 0.15).astype(int)
        # sporadic channels every 16 steps
        if obs_dim > 12 and t % 16 == 0:
            mask[t, 12:obs_dim] = (rng.uniform(size=obs_dim - 12) > 0.2).astype(int)
    if profile_name == "smoke":
        # slightly denser observations to stabilize tiny runs
        mask[:, :8] = 1
    return mask


def heteroskedastic_R(base_R: np.ndarray, x: np.ndarray, mask: np.ndarray, t: int) -> np.ndarray:
    scale = 1.0 + 0.25 * np.tanh(np.linalg.norm(x) / np.sqrt(len(x))) + 0.15 * (t % 24 == 0)
    R = np.asarray(base_R, dtype=float) * scale
    return R


def generate_observation(
    x: np.ndarray,
    C: np.ndarray,
    c: np.ndarray,
    R: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    nonlinear_scale: float = 0.0,
) -> np.ndarray:
    mean = C @ x + c
    if nonlinear_scale > 0:
        mean = mean + nonlinear_scale * np.sin(mean)
    # Fast path for diagonal R (always the case in this codebase)
    diag_R = np.diag(R)
    if np.allclose(R, np.diag(diag_R)):
        noise = rng.standard_normal(len(mean)) * np.sqrt(diag_R)
    else:
        noise = rng.multivariate_normal(np.zeros(len(mean)), R)
    y = mean + noise
    y = np.where(mask.astype(bool), y, np.nan)
    return y
