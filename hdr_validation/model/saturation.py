"""Saturating dose-response models for nonlinear intervention channels.

Implements the Michaelis-Menten saturating transform (Definition 5.1
of the gap resolution document) for channels where the dose-response
relationship is concave (diminishing returns at high doses).
"""
from __future__ import annotations

import numpy as np


def michaelis_menten(u: float | np.ndarray, u_max: float, u_half: float) -> float | np.ndarray:
    """Michaelis-Menten saturating transform.

    tilde_u = u_max * (u / u_half) / (1 + u / u_half)

    Parameters
    ----------
    u : float or array — raw intervention magnitude (>= 0).
    u_max : float — maximum achievable benefit (asymptotic ceiling).
    u_half : float — half-saturation dose (u at which 50% of u_max is achieved).

    Returns
    -------
    float or array — effective intervention magnitude, in [0, u_max).
    """
    u = np.asarray(u, dtype=float)
    ratio = u / u_half
    return u_max * ratio / (1.0 + ratio)


def inverse_michaelis_menten(u_eff: float | np.ndarray, u_max: float, u_half: float) -> float | np.ndarray:
    """Inverse of Michaelis-Menten: recover raw u from effective u_eff.

    u = u_half * u_eff / (u_max - u_eff)

    Valid only for 0 <= u_eff < u_max.
    """
    u_eff = np.asarray(u_eff, dtype=float)
    denom = u_max - u_eff
    # Protect against u_eff >= u_max
    denom = np.maximum(denom, 1e-12)
    return u_half * u_eff / denom


def apply_saturation(u_vec: np.ndarray, sat_channels: list[int],
                     sat_params: dict[int, tuple[float, float]]) -> np.ndarray:
    """Apply saturation to selected channels of an intervention vector.

    Parameters
    ----------
    u_vec : np.ndarray, shape (m,) — full intervention vector.
    sat_channels : list of int — indices of channels to saturate.
    sat_params : dict mapping channel index to (u_max, u_half).

    Returns
    -------
    np.ndarray, shape (m,) — intervention vector with saturated channels.
    """
    u_out = u_vec.copy()
    for ch in sat_channels:
        u_max, u_half = sat_params[ch]
        u_out[ch] = michaelis_menten(np.abs(u_vec[ch]), u_max, u_half)
        # Preserve sign (saturation applies to magnitude)
        if u_vec[ch] < 0:
            u_out[ch] = -u_out[ch]
    return u_out
