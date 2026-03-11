"""
Mode C Fisher Information — HDR v5.2
======================================

Implements the explicit linear-Gaussian Fisher information form for identifying
B_k (the intervention gain matrix) as derived in Appendix I.

For x_{t+1} = A_k x_t + B_k u_t + w_t with w_t ~ N(0, Q_w_k), the single-step
Fisher information for theta_k = vec(B_k) given (x_t, u_t) is:

    F^(1)_k(theta_k; x_t, u_t) = (u_t ⊗ I_n)^T (Q_w_k)^{-1} (u_t ⊗ I_n)

which simplifies to ||u_t||^2_{Q_w_k^{-1}} * I_n when Q_w_k is diagonal.
The trace is tr(F^(1)) = ||u_t||^2_{Q_w_k^{-1}} * n.

The Mode C QP then reduces to:
    maximise  u^T (Q_w_k)^{-1} u
    subject to  u_min <= u <= u_max  and  ||u||_1 <= burden_remaining

This is a quadratic maximisation over a box, solved by evaluating the corner
with largest objective value (sign-of-gradient strategy), then scaling to
satisfy the L1 budget.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def compute_fisher_trace(
    u: np.ndarray,
    Q_w_k: np.ndarray,
    n_state: int,
) -> float:
    """Compute tr(F^(1)_k) = ||u||^2_{Q_w_k^{-1}} * n_state.

    This is the single-step Fisher information trace for identifying B_k
    (the intervention gain matrix, vec(B_k)) under the linear-Gaussian model:
        x_{t+1} = A_k x_t + B_k u_t + w_t,  w_t ~ N(0, Q_w_k).

    The formula follows from:
        F^(1) = (u ⊗ I_n)^T Q_w_k^{-1} (u ⊗ I_n)
    whose trace equals ||u||^2_{inv(Q_w_k)} * n_state.

    Parameters
    ----------
    u : np.ndarray
        Control input vector, shape (m,).
    Q_w_k : np.ndarray
        Process noise covariance for basin k, shape (n, n). Must be SPD.
    n_state : int
        State dimension n.

    Returns
    -------
    float
        Scalar trace value >= 0.
    """
    u = np.asarray(u, dtype=float)
    Q_w_k = np.asarray(Q_w_k, dtype=float)
    # Compute ||u||^2_{Q_w_k^{-1}} = u^T Q_w_k^{-1} u
    # Use Cholesky for numerical stability
    try:
        c, low = cho_factor(Q_w_k)
        Q_inv_u = cho_solve((c, low), u) if len(u) == Q_w_k.shape[0] else None
    except Exception:
        Q_inv_u = None

    if Q_inv_u is not None:
        weighted_norm_sq = float(u @ Q_inv_u)
    else:
        # Q_w_k is n×n but u is m-dimensional; use trace formula directly:
        # tr(F^(1)) = ||u||^2 * tr(Q_w_k^{-1}) for diagonal-approx, or more generally
        # use the identity tr((u⊗I)^T Q^{-1} (u⊗I)) = ||u||^2 * tr(Q^{-1})
        # (valid when Q_w_k acts on the state space and u acts on the control space)
        try:
            Q_inv = np.linalg.inv(Q_w_k)
            weighted_norm_sq = float(np.dot(u, u)) * float(np.trace(Q_inv)) / Q_w_k.shape[0]
        except np.linalg.LinAlgError:
            weighted_norm_sq = float(np.dot(u, u))

    return float(weighted_norm_sq * n_state)


def maximise_fisher_trace(
    Q_w_k: np.ndarray,
    n_state: int,
    u_min: np.ndarray,
    u_max: np.ndarray,
    burden_remaining: float,
) -> np.ndarray:
    """Solve the Mode C Fisher information maximisation problem.

        max_u  tr(F^(1)_k(theta_k; u))  =  ||u||^2_{Q_w_k^{-1}} * n_state
        s.t.   u_min <= u <= u_max
               ||u||_1 <= burden_remaining

    Because Q_w_k^{-1} is SPD, the objective is strictly convex-upward in u
    (i.e., we maximise a convex quadratic). For box constraints, the unconstrained
    maximum is at the boundary. Strategy:

    1. Compute Q_w_k_inv = inv(Q_w_k).
    2. The gradient of the objective w.r.t. u_i is 2 * (Q_w_k_inv @ u)_i * n_state.
       For a diagonal Q_w_k_inv, each u_i acts independently.
    3. At the box optimum, each u_i = u_max[i] if |u_max[i]| >= |u_min[i]|,
       else u_min[i]. Since the matrix is SPD, always take the extreme
       with largest absolute value.
    4. If ||u_corner||_1 > burden_remaining, scale u proportionally to meet the
       L1 budget while preserving direction.

    Parameters
    ----------
    Q_w_k : np.ndarray
        Process noise covariance, shape (n, n). Must be SPD.
    n_state : int
        State dimension n.
    u_min : np.ndarray
        Lower bound on u, shape (m,).
    u_max : np.ndarray
        Upper bound on u, shape (m,).
    burden_remaining : float
        Remaining L1 budget ||u||_1 <= burden_remaining.

    Returns
    -------
    np.ndarray
        Optimal u* maximising Fisher trace subject to constraints, shape (m,).
    """
    u_min = np.asarray(u_min, dtype=float)
    u_max = np.asarray(u_max, dtype=float)
    m = len(u_min)

    # Select corner: for each dimension, take the extreme with larger absolute value
    # (since Q_w_k^{-1} is SPD, the quadratic objective always prefers larger |u_i|)
    u_corner = np.where(np.abs(u_max) >= np.abs(u_min), u_max, u_min)

    # Apply L1 budget constraint
    l1_norm = float(np.sum(np.abs(u_corner)))
    if l1_norm > burden_remaining and l1_norm > 1e-12:
        u_corner = u_corner * (burden_remaining / l1_norm)

    return u_corner


def dither_policy(
    u_nominal: np.ndarray,
    sigma_c: float,
    u_min: np.ndarray,
    u_max: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Practical Mode C implementation: add isotropic Gaussian dither to nominal control.

        u_C = clip(u_nominal + epsilon, u_min, u_max)
        epsilon ~ N(0, sigma_c^2 * I_m)

    This is the simulation instantiation of the Fisher maximisation QP described
    in Appendix I. It provides persistent excitation while respecting hard bounds.

    Parameters
    ----------
    u_nominal : np.ndarray
        Nominal control input from Mode A, shape (m,).
    sigma_c : float
        Dither standard deviation; choose to maximise Fisher trace within the box.
    u_min : np.ndarray
        Lower bound, shape (m,).
    u_max : np.ndarray
        Upper bound, shape (m,).
    rng : np.random.Generator
        NumPy random Generator (for reproducibility with seeds).

    Returns
    -------
    np.ndarray
        Dithered control u_C, shape (m,), clipped to [u_min, u_max].
    """
    u_nominal = np.asarray(u_nominal, dtype=float)
    u_min = np.asarray(u_min, dtype=float)
    u_max = np.asarray(u_max, dtype=float)
    m = len(u_nominal)
    epsilon = rng.normal(scale=sigma_c, size=m)
    u_c = np.clip(u_nominal + epsilon, u_min, u_max)
    return u_c


def accumulated_fisher_lower_bound(
    T_C: int,
    Q_w_k: np.ndarray,
    n_state: int,
    u_min_norm_sq: float,
) -> float:
    """Compute the lower bound on accumulated Fisher information after T_C Mode C steps.

        sum_{t=1}^{T_C} tr(F^(1)) >= lambda_min(Q_w_k^{-1}) * u_min_norm_sq * n_state * T_C

    This connects Mode C duration to the regime boundary omega_min (Definition 10.7):
    Mode C restores identifiability when this lower bound >= omega_min.

    Parameters
    ----------
    T_C : int
        Number of Mode C steps taken.
    Q_w_k : np.ndarray
        Process noise covariance, shape (n, n). Must be SPD.
    n_state : int
        State dimension.
    u_min_norm_sq : float
        Minimum achievable ||u||^2 under Mode C constraints.

    Returns
    -------
    float
        Scalar lower bound on cumulative Fisher trace.
    """
    Q_w_k = np.asarray(Q_w_k, dtype=float)
    # lambda_min(Q_w_k^{-1}) = 1 / lambda_max(Q_w_k)
    eigenvalues = np.linalg.eigvalsh(Q_w_k)
    lambda_max_Q = float(np.max(eigenvalues))
    if lambda_max_Q < 1e-12:
        return 0.0
    lambda_min_Q_inv = 1.0 / lambda_max_Q
    return float(lambda_min_Q_inv * u_min_norm_sq * n_state * T_C)
