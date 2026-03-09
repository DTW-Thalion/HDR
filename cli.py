from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import norm


def chance_tightening(C: np.ndarray, P: np.ndarray, R: np.ndarray, alpha: float) -> np.ndarray:
    cov = C @ P @ C.T + R
    z = norm.ppf(1.0 - alpha / 2.0)
    diag = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    return z * diag


def observation_intervals(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    m = int(config["obs_dim"])
    lo = -1.8 * np.ones(m)
    hi = 1.8 * np.ones(m)
    # tighter limits for circadian and neuroendocrine channels
    if m >= 16:
        lo[10:14] = -1.4
        hi[10:14] = 1.4
    return lo, hi


def risk_score(y_mean: np.ndarray, cov: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    std = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    lower = norm.cdf((lo - y_mean) / std)
    upper = norm.cdf((y_mean - hi) / std)
    return float(np.max(lower + upper))


def circadian_allowed_mask(step: int, control_dim: int, steps_per_day: int = 48, locked_dims: list[int] | None = None) -> np.ndarray:
    mask = np.ones(control_dim, dtype=bool)
    locked_dims = locked_dims or []
    phase = step % steps_per_day
    # allow circadian-locked controls only during approximate daylight / early evening
    allowed = (10 <= phase <= 28) or (34 <= phase <= 38)
    if not allowed:
        for idx in locked_dims:
            if 0 <= idx < control_dim:
                mask[idx] = False
    return mask


def apply_control_constraints(
    u: np.ndarray,
    config: dict[str, Any],
    step: int,
    used_burden: float = 0.0,
    u_min: float = -0.6,
    u_max: float = 0.6,
) -> tuple[np.ndarray, dict[str, Any]]:
    u = np.asarray(u, dtype=float).copy()
    info: dict[str, Any] = {}
    allowed = circadian_allowed_mask(
        step,
        control_dim=len(u),
        steps_per_day=int(config["steps_per_day"]),
        locked_dims=list(config.get("circadian_locked_controls", [])),
    )
    u[~allowed] = 0.0
    u = np.clip(u, u_min, u_max)
    budget = float(config["default_burden_budget"])
    remaining = max(budget - used_burden, 0.0)
    norm1 = float(np.sum(np.abs(u)))
    if norm1 > remaining and norm1 > 1e-12:
        u *= remaining / norm1
    info["allowed_mask"] = allowed.astype(int)
    info["budget_remaining"] = remaining
    info["burden_used_step"] = float(np.sum(np.abs(u)))
    return u, info


def safety_fallback(u: np.ndarray, max_scale: float = 0.5) -> np.ndarray:
    return np.asarray(u, dtype=float) * max_scale


def gaussian_calibration_toy(alpha: float, n_samples: int, rng: np.random.Generator) -> dict[str, float]:
    y = rng.normal(size=n_samples)
    lo = norm.ppf(alpha / 2.0)
    hi = norm.ppf(1.0 - alpha / 2.0)
    empirical = float(np.mean((y < lo) | (y > hi)))
    nominal = float(alpha)
    return {"nominal": nominal, "empirical": empirical, "abs_error": abs(empirical - nominal)}
