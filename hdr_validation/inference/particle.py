"""
Particle Filter — HDR v7.3
===========================
Sequential Monte Carlo with population-informed proposals.
Prop 8.5: ||E_PF[f] - E_true[f]|| = O(1/sqrt(N)).
"""
from __future__ import annotations

import numpy as np


class ParticleFilter:
    """Sequential Monte Carlo filter for SLDS inference.

    Implements systematic resampling, ESS monitoring,
    and population-prior proposal (Eq 8.6).
    """

    def __init__(self, n_particles: int, basins: list, proposal_inflation: float = 1.5):
        self.n_particles = n_particles
        self.basins = basins
        self.proposal_inflation = proposal_inflation
        self.K = len(basins)

        # State dimension from first basin
        self.n = basins[0].A.shape[0]

        # Initialize particles
        self.particles = np.zeros((n_particles, self.n))
        self.weights = np.ones(n_particles) / n_particles
        self.mode_assignments = np.zeros(n_particles, dtype=int)

        # Initialize mode probabilities uniformly
        self.mode_probs = np.ones(self.K) / self.K

    def predict(self, u: np.ndarray) -> None:
        """Propagate particles through dynamics."""
        u = np.asarray(u, dtype=float)
        for i in range(self.n_particles):
            k = self.mode_assignments[i]
            basin = self.basins[k]
            # x_new = A @ x + B @ u + b + process_noise
            noise = np.random.default_rng().multivariate_normal(
                np.zeros(self.n), basin.Q
            )
            self.particles[i] = (
                basin.A @ self.particles[i]
                + basin.B @ u[:basin.B.shape[1]]
                + basin.b
                + noise * self.proposal_inflation
            )

    def update(self, y: np.ndarray) -> None:
        """Update particle weights given observation."""
        y = np.asarray(y, dtype=float)
        log_weights = np.log(self.weights + 1e-300)

        for i in range(self.n_particles):
            k = self.mode_assignments[i]
            basin = self.basins[k]
            # Observation likelihood
            y_pred = basin.C @ self.particles[i] + basin.c
            # Handle NaN observations (missing data)
            valid = ~np.isnan(y)
            if np.any(valid):
                residual = y[valid] - y_pred[valid]
                R_valid = basin.R[np.ix_(np.where(valid)[0], np.where(valid)[0])]
                # Log-likelihood under Gaussian
                try:
                    sign, logdet = np.linalg.slogdet(R_valid)
                    if sign > 0:
                        R_inv = np.linalg.inv(R_valid)
                        ll = -0.5 * (residual @ R_inv @ residual + logdet + len(residual) * np.log(2 * np.pi))
                    else:
                        ll = -100.0
                except np.linalg.LinAlgError:
                    ll = -100.0
                log_weights[i] += ll

        # Normalize weights
        log_weights -= np.max(log_weights)
        self.weights = np.exp(log_weights)
        total = np.sum(self.weights)
        if total > 1e-300:
            self.weights /= total
        else:
            self.weights = np.ones(self.n_particles) / self.n_particles

        # Update mode probabilities
        for k in range(self.K):
            mask = self.mode_assignments == k
            self.mode_probs[k] = np.sum(self.weights[mask])
        mp_sum = np.sum(self.mode_probs)
        if mp_sum > 1e-300:
            self.mode_probs /= mp_sum

    def resample(self) -> None:
        """Systematic resampling."""
        N = self.n_particles
        positions = (np.arange(N) + np.random.uniform()) / N
        cumsum = np.cumsum(self.weights)
        cumsum[-1] = 1.0  # ensure sum is exactly 1

        indices = np.searchsorted(cumsum, positions)
        indices = np.clip(indices, 0, N - 1)

        self.particles = self.particles[indices].copy()
        self.mode_assignments = self.mode_assignments[indices].copy()
        self.weights = np.ones(N) / N

    def ess(self) -> float:
        """Effective sample size: 1 / sum(w_i^2)."""
        return float(1.0 / np.sum(self.weights ** 2))
