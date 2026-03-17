"""
Stability Checking Utilities
========================================

Provides explicit assertions that rho(A_k) < 1 after every parameter update.
This satisfies the discretisation stability condition added in the v5.2 revision:

    A_k = I + Delta_t * (-D + J)

can lose stability under coarse Delta_t. The function assert_spectral_radius_lt1
verifies rho(A_k) < 1 holds for the actual numerical matrix.

INTEGRATION POINT: call check_all_basin_stability(basin_models)
after every parameter update in the Phase 2 identification loop.
See hdr_validation/inference/[identification_module].py.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)


def assert_spectral_radius_lt1(
    A_k: np.ndarray,
    basin_id: int,
    tolerance: float = 1e-6,
    raise_on_failure: bool = True,
) -> bool:
    """Assert that the spectral radius rho(A_k) < 1 for basin ``basin_id``.

    This check is required after every parameter update (identification step)
    per the discretisation stability condition added in the v5.2 revision.
    The condition rho(A_k) < 1 is guaranteed when (-D + J) is Hurwitz and
    Delta_t is small enough (Lemma A.2); this function verifies it holds for
    the actual numerical matrix.

    Parameters
    ----------
    A_k : np.ndarray
        Discrete-time state transition matrix, shape (n, n).
    basin_id : int
        Integer basin identifier (for error messages).
    tolerance : float
        rho must be < 1 - tolerance. Default 1e-6.
    raise_on_failure : bool
        If True (default), raise ValueError on violation.
        If False, log a warning and return False.

    Returns
    -------
    bool
        True if rho(A_k) < 1 - tolerance, False otherwise (only if not raising).

    Raises
    ------
    ValueError
        If rho(A_k) >= 1 - tolerance and raise_on_failure=True.
    """
    A_k = np.asarray(A_k, dtype=float)
    eigenvalues = np.linalg.eigvals(A_k)
    rho = float(np.max(np.abs(eigenvalues)))
    threshold = 1.0 - tolerance
    if rho >= threshold:
        msg = (
            f"Stability check FAILED for basin {basin_id}: "
            f"rho(A_k) = {rho:.6f} >= {threshold:.6f}. "
            f"The matrix A_k is not strictly stable. "
            f"Check discretisation step Delta_t or parameter identification."
        )
        if raise_on_failure:
            raise ValueError(msg)
        else:
            warnings.warn(msg, stacklevel=2)
            logger.warning(msg)
            return False
    return True


def check_all_basin_stability(
    basin_models: dict[int, dict],
    tolerance: float = 1e-6,
    raise_on_failure: bool = True,
) -> dict[int, bool]:
    """Run assert_spectral_radius_lt1 for all basins in the model dictionary.

    Returns {basin_id: True/False} for each basin.
    Used in Phase 2 identification pipeline after each model update.

    Parameters
    ----------
    basin_models : dict[int, dict]
        {basin_id: {"A_k": array, ...}} mapping from basin ID to parameter dict.
    tolerance : float
        Spectral radius tolerance passed to assert_spectral_radius_lt1.
    raise_on_failure : bool
        If True, raise ValueError on first violation. If False, collect results.

    Returns
    -------
    dict[int, bool]
        {basin_id: True} if stable, {basin_id: False} if not (when not raising).
    """
    results: dict[int, bool] = {}
    for basin_id, params in basin_models.items():
        A_k = np.asarray(params["A_k"], dtype=float)
        ok = assert_spectral_radius_lt1(
            A_k,
            basin_id=basin_id,
            tolerance=tolerance,
            raise_on_failure=raise_on_failure,
        )
        results[basin_id] = ok
    return results
