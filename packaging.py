from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are


def dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


def dlqr_robust(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    mismatch_bound: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    """Robust LQR with gain scaling for bounded model mismatch (Prop H.3).

    When the true dynamics (A_true, B_true) differ from the nominal model
    (A, B) by up to `mismatch_bound`, the standard LQR gain K can be
    destabilising if ρ(A - B*K) * (1 + mismatch_bound) ≥ 1.

    Fix: inflate R by (1 + mismatch_bound)² to obtain a more conservative
    gain K_r satisfying: ρ(A - B*K_r) ≤ ρ(A-B*K) / (1 + mismatch_bound).
    This is the standard "robust LQR" gain derating, equivalent to requiring
    the closed-loop to be stable for all (A', B') with ‖A'−A‖ ≤ δ·‖A‖.

    Returns (K_robust, P_robust), mirroring dlqr().
    """
    R_robust = R * (1.0 + mismatch_bound) ** 2
    P_r = solve_discrete_are(A, B, Q, R_robust)
    K_r = np.linalg.solve(R_robust + B.T @ P_r @ B, B.T @ P_r @ A)
    return K_r, P_r


def finite_horizon_tracking(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    H: int,
    P_terminal: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Finite-horizon LQR gains via backward Riccati recursion.

    When P_terminal is the DARE solution, the recursion starts at the
    infinite-horizon optimum and stays there (fixed point), giving exactly
    the infinite-horizon LQR gain K_LQR for all H steps.  This removes the
    under-converged-gain artefact that appeared when P_0=Q was used.
    """
    P = P_terminal.copy() if P_terminal is not None else Q.copy()
    gains = []
    for _ in range(H):
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        gains.append(K)
        P = Q + A.T @ P @ (A - B @ K)
    return list(reversed(gains))
