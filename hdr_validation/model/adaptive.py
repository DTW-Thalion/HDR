from __future__ import annotations

"""Adaptive parameter estimation and drift detection for HDR v7.3.

Provides forgetting-factor RLS (FF-RLS) for tracking within-basin parameter
drift and a drift detector that triggers Mode C re-identification when the
estimated drift exceeds the ISS margin.
"""

import numpy as np


class FFRLSEstimator:
    """Forgetting-factor RLS for within-basin parameter drift.

    Tracks A_k(t) as it drifts, using forgetting factor lambda_ff.
    Ref: Def 5.16 in v7.0 manuscript.
    """

    def __init__(self, n: int, lambda_ff: float = 0.98):
        self.n = n
        self.lambda_ff = lambda_ff
        # Initialize A_hat to identity
        self.A_hat = np.eye(n)
        self.A_hat_initial = np.eye(n)
        # RLS covariance: P = (1/delta) * I, delta small
        self.P_rls = np.eye(n) * 100.0  # large initial uncertainty

    def update(
        self,
        x_new: np.ndarray,
        x_old: np.ndarray,
        u: np.ndarray | None = None,
    ) -> np.ndarray:
        """Update A_hat using RLS with forgetting factor.

        Model: x_new = A_hat @ x_old (+ B @ u, ignored for A tracking)
        RLS update:
          e = x_new - A_hat @ x_old
          For each row i of A_hat:
            K_i = P @ x_old / (lambda + x_old^T @ P @ x_old)
            A_hat[i,:] += K_i * e[i]
            P = (P - K_i @ x_old^T @ P) / lambda

        Returns updated A_hat.
        """
        x_old = np.asarray(x_old, dtype=float)
        x_new = np.asarray(x_new, dtype=float)

        # Prediction error
        e = x_new - self.A_hat @ x_old

        # Gain
        Px = self.P_rls @ x_old
        denom = self.lambda_ff + float(x_old @ Px)
        K = Px / max(denom, 1e-12)

        # Update each row of A_hat
        for i in range(self.n):
            self.A_hat[i, :] += K * e[i]

        # Update P_rls
        self.P_rls = (self.P_rls - np.outer(K, x_old) @ self.P_rls) / self.lambda_ff

        # Symmetrize and bound P_rls for numerical stability
        self.P_rls = 0.5 * (self.P_rls + self.P_rls.T)
        max_p = 1e6
        eigvals = np.linalg.eigvalsh(self.P_rls)
        if np.max(eigvals) > max_p:
            self.P_rls *= max_p / np.max(eigvals)

        return self.A_hat.copy()

    def drift_magnitude(self) -> float:
        """||delta_A_hat|| = ||A_hat - A_hat_initial||_F"""
        return float(np.linalg.norm(self.A_hat - self.A_hat_initial, "fro"))

    def sigma_rls(self) -> float:
        """RLS estimation uncertainty: sqrt(tr(P_rls) / n^2).

        Measures the average per-element estimation uncertainty
        from the RLS covariance matrix.
        """
        return float(np.sqrt(np.trace(self.P_rls) / self.n**2))

    def adaptive_delta_A(self, gamma_margin: float = 2.0) -> float:
        """Adaptive mismatch bound: ||A_hat - A_initial||_2 + gamma * sigma_rls.

        Parameters
        ----------
        gamma_margin : float
            Safety margin multiplier on estimation uncertainty.

        Returns
        -------
        float — instantaneous adaptive mismatch bound hat_Delta_A(t).
        """
        drift = float(np.linalg.norm(
            self.A_hat - self.A_hat_initial, ord=2))
        return drift + gamma_margin * self.sigma_rls()


class DriftDetector:
    """Detect when drift exceeds ISS margin: ||delta_A|| > Delta_A_max.

    Triggers Mode C re-identification.
    """

    def __init__(self, Delta_A_max: float):
        self.Delta_A_max = Delta_A_max

    def check(self, estimator: FFRLSEstimator) -> bool:
        """Returns True if drift exceeds threshold."""
        return estimator.drift_magnitude() > self.Delta_A_max

    def adaptive_mubar_required(
        self,
        estimator: FFRLSEstimator,
        c_ISS: float,
        Delta_B: float,
        K_norm: float,
        alpha: float,
        epsilon_ctrl: float,
        gamma_margin: float = 2.0,
    ) -> float:
        """Time-varying required mode accuracy using adaptive Delta_A.

        mubar(t) = [epsilon_ctrl * alpha / (c_ISS * (hat_Delta_A(t) + Delta_B * ||K||))]^2

        Parameters
        ----------
        estimator : FFRLSEstimator — current estimator state.
        c_ISS : float — ISS condition number.
        Delta_B : float — input matrix mismatch bound.
        K_norm : float — ||K_k||_2 (LQR gain spectral norm).
        alpha : float — Lyapunov decrease rate.
        epsilon_ctrl : float — target steady-state deviation tolerance.
        gamma_margin : float — passed to adaptive_delta_A.

        Returns
        -------
        float — time-varying mubar_required(t) in [0, 1].
        """
        delta_A = estimator.adaptive_delta_A(gamma_margin)
        denom = c_ISS * (delta_A + Delta_B * K_norm)
        if denom < 1e-12:
            return 1.0
        mubar = (epsilon_ctrl * alpha / denom) ** 2
        return float(min(mubar, 1.0))
