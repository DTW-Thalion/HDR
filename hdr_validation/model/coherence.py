from __future__ import annotations

import numpy as np


def spectral_gap(A: np.ndarray) -> float:
    """Compute the coherence measure kappa_hat as the spectral gap of A.

    kappa_hat = |lambda_1(A)| - |lambda_2(A)|

    where eigenvalues are ordered by decreasing modulus.

    This is the formal definition from Remark B.1 in the paper.
    A moderate spectral gap indicates healthy multi-modal coordination;
    a very large gap indicates pathological single-mode dominance;
    a near-zero gap indicates loss of regulatory structure.

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Estimated basin dynamics matrix.

    Returns
    -------
    float
        Non-negative spectral gap.
    """
    eigvals = np.linalg.eigvals(A)
    moduli = np.sort(np.abs(eigvals))[::-1]  # descending order
    if len(moduli) < 2:
        return 0.0
    return float(moduli[0] - moduli[1])


def coherence_grad(kappa: float, kappa_lo: float, kappa_hi: float) -> float:
    if kappa < kappa_lo:
        return float(2.0 * (kappa - kappa_lo))
    if kappa > kappa_hi:
        return float(2.0 * (kappa - kappa_hi))
    return 0.0


def coherence_penalty(kappa: float, kappa_lo: float, kappa_hi: float) -> float:
    if kappa < kappa_lo:
        return float((kappa - kappa_lo)**2)
    if kappa > kappa_hi:
        return float((kappa - kappa_hi)**2)
    return 0.0