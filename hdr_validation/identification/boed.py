"""
Bayesian Optimal Experimental Design
================================================
Def 11.4: Maximises expected KL divergence subject to safety constraints.
"""
from __future__ import annotations

import numpy as np


class BOEDEstimator:
    """Bayesian optimal experimental design for B_k.

    Prop 11.5: sample complexity bound Eq 11.6.
    """

    def __init__(self, prior: dict, safety_constraint: dict):
        self.prior = prior  # {'mean': ndarray, 'cov': ndarray}
        self.safety_constraint = safety_constraint  # {'u_max': float, 'risk_max': float}
        self.prior_mean = np.asarray(prior.get("mean", np.zeros(1)), dtype=float)
        self.prior_cov = np.asarray(prior.get("cov", np.eye(1)), dtype=float)
        self.n = len(self.prior_mean)

    def optimal_design(self, xi_safe: np.ndarray, L: int) -> np.ndarray:
        """Compute optimal experimental design within safety set.

        Parameters
        ----------
        xi_safe : (d,) safe perturbation directions
        L : number of design points

        Returns
        -------
        design : (L, d) optimal design matrix
        """
        xi_safe = np.asarray(xi_safe, dtype=float)
        d = len(xi_safe)
        u_max = float(self.safety_constraint.get("u_max", 0.6))

        # D-optimal design: maximize det(X^T X) subject to safety
        # For simplicity: use scaled directions with alternating signs
        design = np.zeros((L, d))
        for i in range(L):
            sign = 1.0 if i % 2 == 0 else -1.0
            scale = min(u_max, float(np.max(np.abs(xi_safe))) + 0.01)
            direction = xi_safe / (np.linalg.norm(xi_safe) + 1e-10)
            # Rotate through dimensions
            dim_idx = i % d
            design[i] = np.zeros(d)
            design[i, dim_idx] = sign * scale

        return design

    def sample_complexity(self, epsilon: float, delta: float, config: dict) -> int:
        """Compute sample complexity bound (Prop 11.5, Eq 11.6).

        N >= (n_theta / epsilon^2) * log(1/delta)

        Parameters
        ----------
        epsilon : accuracy parameter
        delta : confidence parameter
        config : dict with 'state_dim' or 'n_theta'

        Returns
        -------
        N : minimum number of samples required
        """
        n_theta = int(config.get("n_theta", config.get("state_dim", 8) ** 2))
        return int(np.ceil(n_theta / (epsilon ** 2) * np.log(1.0 / delta)))

    def information_gain(self, design: np.ndarray) -> float:
        """Compute expected information gain for a design matrix.

        Returns log-det(I + X^T Sigma^{-1} X / sigma_n^2) as proxy.
        """
        X = np.asarray(design, dtype=float)
        if X.shape[0] < 1:
            return 0.0
        try:
            Sigma_inv = np.linalg.inv(self.prior_cov)
            M = np.eye(X.shape[1]) + X.T @ X @ Sigma_inv
            sign, logdet = np.linalg.slogdet(M)
            return float(logdet) if sign > 0 else 0.0
        except np.linalg.LinAlgError:
            return 0.0
