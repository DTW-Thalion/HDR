"""
Population-Prior Basin Assignment — HDR v7.3
=============================================
Sec 8.3.3: Uses population-level transition rates + demographics
to provide informative prior when T_eff_k < omega_min.
"""
from __future__ import annotations

import numpy as np


class PopulationPriorAssignment:
    """Population-prior basin assignment.

    Uses population-level transition rates and demographic features to
    provide an informative prior over basins when individual data is
    insufficient (T_eff_k < omega_min).

    Parameters
    ----------
    pop_transition : (K, K) population-level transition matrix
    pop_features : dict mapping feature names to weight vectors of length K
    """

    def __init__(self, pop_transition: np.ndarray, pop_features: dict | None = None):
        self.pop_transition = np.asarray(pop_transition, dtype=float)
        self.K = self.pop_transition.shape[0]
        self.pop_features = pop_features or {}

        # Compute stationary distribution as default prior
        # pi = pi @ T, so pi is the left eigenvector for eigenvalue 1
        eigvals, eigvecs = np.linalg.eig(self.pop_transition.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        pi = np.real(eigvecs[:, idx])
        pi = np.abs(pi)
        pi_sum = np.sum(pi)
        if pi_sum > 1e-10:
            self.stationary = pi / pi_sum
        else:
            self.stationary = np.ones(self.K) / self.K

    def prior(self, features: dict | None = None) -> np.ndarray:
        """Compute K-dimensional prior over basins.

        If features are provided and match pop_features keys,
        adjusts the stationary prior using feature weights.
        Otherwise returns stationary distribution.

        Parameters
        ----------
        features : dict of patient-specific feature values

        Returns
        -------
        prior : (K,) array summing to 1
        """
        p = self.stationary.copy()

        if features is not None and self.pop_features:
            for feat_name, feat_val in features.items():
                if feat_name in self.pop_features:
                    weights = np.asarray(self.pop_features[feat_name], dtype=float)
                    if len(weights) == self.K:
                        # Multiplicative adjustment
                        adjustment = np.exp(weights * float(feat_val))
                        p *= adjustment

        # Normalize
        p_sum = np.sum(p)
        if p_sum > 1e-10:
            p /= p_sum
        else:
            p = np.ones(self.K) / self.K

        return p
