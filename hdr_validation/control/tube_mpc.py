"""mRPI terminal set and tube-MPC (IG1 Path B).

Implements the Raković et al. (2005) algorithm for computing the maximal
robust positively invariant (mRPI) set as a zonotope, and a tube-MPC
wrapper that decomposes control into nominal + ancillary feedback.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import linprog

from .mpc import MPCResult, solve_mode_a
from .lqr import dlqr


def compute_disturbance_set(Q_w: np.ndarray, n: int, beta: float = 0.999):
    """Compute ellipsoidal disturbance set W_k = {w : w^T Q_w^{-1} w <= chi2}.

    Parameters
    ----------
    Q_w : np.ndarray, shape (n, n)
        Process noise covariance (SPD).
    n : int
        State dimension.
    beta : float
        Quantile of chi-squared distribution (default 0.999, per Appendix J).

    Returns
    -------
    tuple (Q_w_inv, chi2_bound)
        Q_w_inv: inverse of Q_w, shape (n, n).
        chi2_bound: scalar chi-squared quantile value.
    """
    chi2_bound = float(stats.chi2.ppf(beta, df=n))
    Q_w_inv = np.linalg.solve(Q_w, np.eye(n))
    return Q_w_inv, chi2_bound


def compute_mRPI_zonotope(
    A_cl: np.ndarray,
    Q_w: np.ndarray,
    chi2_bound: float,
    epsilon: float = 1e-3,
    max_iter: int = 200,
) -> dict:
    """Compute outer approximation of minimal RPI set as a zonotope.

    Implements Algorithm B.2: iterative Minkowski-sum of A_cl^i @ W until
    contraction factor alpha_s <= epsilon.

    Parameters
    ----------
    A_cl : np.ndarray, shape (n, n)
        Closed-loop dynamics (A_k - B_k @ K_k). Must be Schur stable.
    Q_w : np.ndarray, shape (n, n)
        Process noise covariance.
    chi2_bound : float
        Chi-squared quantile for disturbance bound.
    epsilon : float
        Convergence tolerance on contraction factor alpha_s.
    max_iter : int
        Maximum Minkowski-sum iterations.

    Returns
    -------
    dict with keys:
        'G': np.ndarray, shape (n, m_total) — zonotope generator matrix.
        'center': np.ndarray, shape (n,) — zonotope center (zero).
        'alpha_s': float — final contraction factor.
        'iterations': int — number of Minkowski-sum steps.
        'scale': float — (1 - alpha_s)^{-1} outer-approximation factor.
    """
    n = A_cl.shape[0]

    # 1. G_W from Cholesky of Q_w scaled by sqrt(chi2)
    L = np.linalg.cholesky(Q_w)
    G_W = L * np.sqrt(chi2_bound)

    # Norm of G_W for alpha_s computation
    norm_G_W = np.linalg.norm(G_W, ord=2)
    if norm_G_W < 1e-15:
        return {
            "G": G_W,
            "center": np.zeros(n),
            "alpha_s": 0.0,
            "iterations": 0,
            "scale": 1.0,
        }

    # 2. Iterative Minkowski-sum
    G_accum = G_W.copy()
    A_cl_power = A_cl.copy()
    alpha_s = 1.0
    iterations = 0

    for s in range(1, max_iter + 1):
        G_new = A_cl_power @ G_W
        alpha_s = np.linalg.norm(G_new, ord=2) / norm_G_W
        iterations = s

        if alpha_s <= epsilon:
            break

        G_accum = np.hstack([G_accum, G_new])
        A_cl_power = A_cl @ A_cl_power

        # Girard's order reduction when too many columns
        if G_accum.shape[1] > 3 * n * n:
            G_accum = _girard_reduce(G_accum, n)

    scale = 1.0 / (1.0 - alpha_s) if alpha_s < 1.0 else 1e6

    return {
        "G": scale * G_accum,
        "center": np.zeros(n),
        "alpha_s": float(alpha_s),
        "iterations": iterations,
        "scale": float(scale),
    }


def _girard_reduce(G: np.ndarray, n: int) -> np.ndarray:
    """Girard's method: merge two smallest-norm generators until count <= 2*n^2."""
    target_cols = 2 * n * n
    while G.shape[1] > target_cols:
        norms = np.linalg.norm(G, axis=0)
        order = np.argsort(norms)
        i, j = order[0], order[1]
        # Merge: interval hull => diagonal bounding box
        g_merged = np.abs(G[:, i]) + np.abs(G[:, j])
        # Replace column i with merged, delete column j
        G[:, i] = g_merged
        G = np.delete(G, j, axis=1)
    return G


def zonotope_containment_check(
    x: np.ndarray,
    G: np.ndarray,
    center: np.ndarray,
) -> bool:
    """Check if x is inside zonotope {G @ xi + center : ||xi||_inf <= 1}.

    Uses LP reformulation: min sum(t_i) s.t. G @ xi = x - center, -t <= xi <= t, t >= 0.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
    G : np.ndarray, shape (n, m)
    center : np.ndarray, shape (n,)

    Returns
    -------
    bool — True if x is inside the zonotope.
    """
    n, m = G.shape
    b_eq = x - center

    # Variables: [xi (m), t (m)]
    # Objective: min sum(t)
    c_obj = np.zeros(2 * m)
    c_obj[m:] = 1.0  # minimize sum of t_i

    # Equality: G @ xi = b_eq
    A_eq = np.hstack([G, np.zeros((n, m))])

    # Inequality: xi - t <= 0 and -xi - t <= 0
    # xi_i <= t_i:  xi_i - t_i <= 0
    # -xi_i <= t_i: -xi_i - t_i <= 0
    A_ub = np.zeros((2 * m, 2 * m))
    for i in range(m):
        # xi_i - t_i <= 0
        A_ub[i, i] = 1.0
        A_ub[i, m + i] = -1.0
        # -xi_i - t_i <= 0
        A_ub[m + i, i] = -1.0
        A_ub[m + i, m + i] = -1.0
    b_ub = np.zeros(2 * m)

    # Bounds: xi unbounded, t >= 0
    bounds = [(None, None)] * m + [(0.0, None)] * m

    try:
        result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method="highs")
        if result.success:
            t_vals = result.x[m:]
            return float(np.max(t_vals)) <= 1.0 + 1e-8
        return False
    except Exception:
        return False


def solve_tube_mpc(
    x_hat: np.ndarray,
    P_hat: np.ndarray,
    basin,
    target,
    mRPI_data: dict,
    K_fb: np.ndarray,
    kappa_hat: float = 0.65,
    config: dict | None = None,
    step: int = 0,
) -> MPCResult:
    """Tube-MPC wrapper around solve_mode_a.

    Decomposes control as u = u_bar + K_fb @ (x_hat - x_bar).

    Parameters
    ----------
    x_hat : np.ndarray, shape (n,)
    P_hat : np.ndarray, shape (n, n)
    basin : BasinModel
    target : TargetSet
    mRPI_data : dict — output of compute_mRPI_zonotope
    K_fb : np.ndarray, shape (m, n) — ancillary feedback gain (LQR gain)
    kappa_hat : float
    config : dict
    step : int

    Returns
    -------
    MPCResult
    """
    x_bar = x_hat.copy()

    # Nominal MPC
    result = solve_mode_a(x_bar, P_hat, basin, target, kappa_hat, config, step)
    u_bar = result.u

    # Ancillary feedback (zero at current step since x_hat == x_bar)
    error = x_hat - x_bar
    u = u_bar + K_fb @ error

    # Clip to control bounds
    u = np.clip(u, -0.6, 0.6)

    return MPCResult(
        u=u,
        feasible=result.feasible,
        solve_time=result.solve_time,
        risk=result.risk,
        notes="tube_mpc",
        allowed_mask=result.allowed_mask,
    )
