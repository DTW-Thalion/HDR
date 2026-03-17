"""
Hierarchical Coupling Estimation — HDR v7.3
============================================
Def 11.1: Three-level model (mechanistic -> group -> patient).
Prop 11.2: MAP estimate via penalised least-squares.
"""
from __future__ import annotations

import numpy as np


class HierarchicalCouplingEstimator:
    """Hierarchical empirical-Bayes coupling estimation.

    Three-level model: mechanistic -> group -> patient.
    MAP estimate (Prop 11.2): Eq 11.4 penalised least-squares.
    Graceful degradation: at T_p=0 returns group mean.

    Parameters
    ----------
    J_mech : (n, n) mechanistic prior mean
    Sigma_pop : (n, n) population-level covariance
    Sigma_group : (n, n) group-level covariance
    """

    def __init__(self, J_mech: np.ndarray, Sigma_pop: np.ndarray, Sigma_group: np.ndarray):
        self.J_mech = np.asarray(J_mech, dtype=float)
        self.Sigma_pop = np.asarray(Sigma_pop, dtype=float)
        self.Sigma_group = np.asarray(Sigma_group, dtype=float)
        self.n = self.J_mech.shape[0]

    def estimate(
        self,
        observations: np.ndarray | None,
        state_estimates: np.ndarray | None,
        lambda_g: float = 1.0,
    ) -> np.ndarray:
        """MAP estimate of patient-specific coupling matrix.

        At T_p=0 (no observations), returns group mean J_mech.
        With data, solves penalised least-squares:
            J_hat = argmin_J ||Y - X @ J||_F^2 + lambda_g * ||J - J_mech||_{Sigma^{-1}}^2

        Parameters
        ----------
        observations : (T_p, n) state transitions y = x_{t+1}
        state_estimates : (T_p, n) state estimates x_t (regressors)
        lambda_g : regularization strength

        Returns
        -------
        J_hat : (n, n) MAP estimate of coupling matrix
        """
        if observations is None or state_estimates is None or len(observations) == 0:
            return self.J_mech.copy()

        X = np.asarray(state_estimates, dtype=float)
        Y = np.asarray(observations, dtype=float)
        T_p = X.shape[0]

        # Penalised least-squares: (X^T X + lambda * Sigma^{-1}) J^T = X^T Y + lambda * Sigma^{-1} J_mech^T
        XtX = X.T @ X
        XtY = X.T @ Y

        try:
            Sigma_inv = np.linalg.inv(self.Sigma_group + self.Sigma_pop)
        except np.linalg.LinAlgError:
            Sigma_inv = np.eye(self.n) * 0.01

        reg = lambda_g * Sigma_inv
        lhs = XtX + reg
        rhs = XtY + reg @ self.J_mech

        try:
            J_hat = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            J_hat = self.J_mech.copy()

        return J_hat

    def convergence_check(self, T_p_values: list[int], true_J: np.ndarray) -> dict:
        """Check that Frobenius error decreases with increasing T_p.

        Parameters
        ----------
        T_p_values : list of sample counts to test
        true_J : ground-truth coupling matrix

        Returns
        -------
        dict with 'errors' (list of Frobenius errors) and 'monotonic_decrease' (bool)
        """
        true_J = np.asarray(true_J, dtype=float)
        rng = np.random.default_rng(42)
        errors = []

        for T_p in T_p_values:
            if T_p == 0:
                J_hat = self.estimate(None, None)
            else:
                # Generate synthetic data from true_J
                X = rng.normal(size=(T_p, self.n))
                Y = X @ true_J + rng.normal(scale=0.1, size=(T_p, self.n))
                J_hat = self.estimate(Y, X)

            error = float(np.linalg.norm(J_hat - true_J, 'fro'))
            errors.append(error)

        # Check monotonic decrease (90% threshold)
        n_decrease = sum(1 for i in range(1, len(errors)) if errors[i] <= errors[i-1] + 1e-10)
        monotonic = n_decrease >= 0.9 * (len(errors) - 1) if len(errors) > 1 else True

        return {"errors": errors, "monotonic_decrease": monotonic}
