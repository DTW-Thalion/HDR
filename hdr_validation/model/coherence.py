from __future__ import annotations

import numpy as np


def damping_ratio(A: np.ndarray) -> float:
    """Compute the coherence measure kappa_hat as the damping ratio of A.

    kappa_hat = zeta = |Re(lambda_1(A))| / |lambda_1(A)|

    where lambda_1 is the eigenvalue with the largest (least negative)
    real part (i.e., the least-stable eigenvalue).

    This is the formal definition from Remark B.1 in the paper.
    Values near 1 indicate overdamped, monotone recovery (healthy
    inter-axis coordination). Declining values indicate underdamped,
    oscillatory dynamics with slow recovery (degraded coordination).
    In the companion ontology simulation, zeta declines from 0.99
    (age 30) to 0.63 (age 80).

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Estimated basin dynamics matrix (discrete-time, stable: rho(A) < 1).

    Returns
    -------
    float
        Damping ratio in [0, 1]. Returns 1.0 for purely real least-stable
        eigenvalue, 0.0 for purely imaginary. Returns 1.0 for scalar or
        empty matrices.
    """
    eigvals = np.linalg.eigvals(A)
    if len(eigvals) == 0:
        return 1.0
    # Find the least-stable eigenvalue: largest (least negative) real part
    real_parts = np.real(eigvals)
    idx = np.argmax(real_parts)
    lambda_1 = eigvals[idx]
    modulus = abs(lambda_1)
    if modulus < 1e-15:
        return 1.0  # zero eigenvalue, trivially damped
    return float(abs(lambda_1.real) / modulus)


def spectral_gap(A: np.ndarray) -> float:
    """DEPRECATED — retained for backward compatibility only.

    Use damping_ratio(A) instead. The spectral gap was replaced in
    Draft 2 (Remark B.1) because it is identically zero when the
    slow eigenvalues form a complex conjugate pair.
    """
    eigvals = np.linalg.eigvals(A)
    moduli = np.sort(np.abs(eigvals))[::-1]
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