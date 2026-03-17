"""
Particle Filter
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

    def __init__(self, n_particles: int, basins: list, proposal_inflation: float = 1.5,
                 rng: np.random.Generator | None = None):
        self.n_particles = n_particles
        self.basins = basins
        self.proposal_inflation = proposal_inflation
        self.K = len(basins)
        self._rng = rng if rng is not None else np.random.default_rng()

        # State dimension from first basin
        self.n = basins[0].A.shape[0]

        # Initialize particles
        self.particles = np.zeros((n_particles, self.n))
        self.weights = np.ones(n_particles) / n_particles
        self.mode_assignments = np.zeros(n_particles, dtype=int)

        # Initialize mode probabilities uniformly
        self.mode_probs = np.ones(self.K) / self.K

    def predict(self, u: np.ndarray) -> None:
        """Propagate particles through dynamics (vectorized by basin)."""
        u = np.asarray(u, dtype=float)
        for k in range(self.K):
            mask_k = self.mode_assignments == k
            if not np.any(mask_k):
                continue
            idx = np.where(mask_k)[0]
            n_k = len(idx)
            basin = self.basins[k]
            m_u = basin.B.shape[1]

            # Batched dynamics: X_new = X @ A.T + u[:m_u] @ B.T + b
            X_k = self.particles[idx]
            X_pred = X_k @ basin.A.T + u[:m_u] @ basin.B.T + basin.b

            # Batched noise using cached Cholesky
            L_Q = basin.Q_cholesky
            Z = self._rng.standard_normal((n_k, self.n))
            noise = Z @ L_Q.T * self.proposal_inflation

            self.particles[idx] = X_pred + noise

    def update(self, y: np.ndarray) -> None:
        """Update particle weights given observation (vectorized by basin)."""
        y = np.asarray(y, dtype=float)
        log_weights = np.log(self.weights + 1e-300)

        valid = ~np.isnan(y)
        valid_idx = np.where(valid)[0]

        if valid_idx.size == 0:
            return  # No observations — skip update

        y_valid = y[valid_idx]
        n_valid = len(valid_idx)
        log_norm = n_valid * np.log(2 * np.pi)

        # Pre-compute per-basin R_inv and logdet (at most K computations)
        basin_obs_cache: dict[int, tuple | None] = {}
        for k in range(self.K):
            basin = self.basins[k]
            R_valid = basin.R[np.ix_(valid_idx, valid_idx)]
            try:
                sign, logdet = np.linalg.slogdet(R_valid)
                if sign > 0:
                    R_inv = np.linalg.inv(R_valid)
                    basin_obs_cache[k] = (R_inv, float(logdet), basin.C[valid_idx])
                else:
                    basin_obs_cache[k] = None
            except np.linalg.LinAlgError:
                basin_obs_cache[k] = None

        # Vectorized per-basin likelihood
        for k in range(self.K):
            mask_k = self.mode_assignments == k
            if not np.any(mask_k):
                continue

            if basin_obs_cache[k] is None:
                log_weights[mask_k] += -100.0
                continue

            R_inv, logdet, C_valid = basin_obs_cache[k]
            idx = np.where(mask_k)[0]
            X_k = self.particles[idx]
            Y_pred = X_k @ C_valid.T + self.basins[k].c[valid_idx]
            residuals = y_valid - Y_pred

            # Batch quadratic form
            quad = np.einsum('ij,jk,ik->i', residuals, R_inv, residuals)
            ll = -0.5 * (quad + logdet + log_norm)
            log_weights[idx] += ll

        # Normalize weights
        log_weights -= np.max(log_weights)
        self.weights = np.exp(log_weights)
        total = np.sum(self.weights)
        if total > 1e-300:
            self.weights /= total
        else:
            self.weights = np.ones(self.n_particles) / self.n_particles

        # Update mode probabilities (vectorized)
        for k in range(self.K):
            self.mode_probs[k] = np.sum(self.weights[self.mode_assignments == k])
        mp_sum = np.sum(self.mode_probs)
        if mp_sum > 1e-300:
            self.mode_probs /= mp_sum

    def resample(self) -> None:
        """Systematic resampling."""
        N = self.n_particles
        positions = (np.arange(N) + self._rng.uniform()) / N
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
