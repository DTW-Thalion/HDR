"""
Variational SLDS Inference
======================================
Mean-field variational inference for SLDS (Sec 8.3.2).
"""
from __future__ import annotations

import numpy as np


class VariationalSLDS:
    """Variational inference for SLDS.

    Mean-field factorisation: q(x,z) = q(x|z)*q(z).
    ELBO = E_q[log p(y|x,z)] - KL[q(x,z)||p(x,z)].
    """

    def __init__(self, basins: list, config: dict):
        self.basins = basins
        self.config = config
        self.K = len(basins)
        self.n = basins[0].A.shape[0] if basins else int(config.get("state_dim", 8))

        # Variational parameters
        self._q_z = np.ones(self.K) / self.K  # mode probabilities
        self._q_x_mean = np.zeros(self.n)       # state mean
        self._q_x_cov = np.eye(self.n)          # state covariance
        self._elbo_history: list[float] = []

    def fit(self, y_sequence: np.ndarray, max_iter: int = 100) -> dict:
        """Fit variational approximation to observation sequence.

        Parameters
        ----------
        y_sequence : (T, m) observation array (may contain NaN)
        max_iter : maximum CAVI iterations

        Returns
        -------
        dict with 'q_z', 'q_x_mean', 'q_x_cov', 'elbo', 'n_iter'
        """
        y_sequence = np.asarray(y_sequence, dtype=float)
        T = y_sequence.shape[0]

        # Initialize q(z) uniformly
        q_z = np.ones(self.K) / self.K

        # Initialize q(x) from observations
        q_x_mean = np.zeros(self.n)
        q_x_cov = np.eye(self.n)

        prev_elbo = -np.inf

        for iteration in range(max_iter):
            # E-step: update q(z) given q(x)
            log_probs = np.zeros(self.K)
            for k in range(self.K):
                basin = self.basins[k]
                # Expected log-likelihood under q(x)
                ll = 0.0
                for t in range(T):
                    y_t = y_sequence[t]
                    valid = ~np.isnan(y_t)
                    if np.any(valid):
                        y_v = y_t[valid]
                        C_v = basin.C[valid, :]
                        c_v = basin.c[valid] if hasattr(basin, 'c') else np.zeros(int(np.sum(valid)))
                        R_v = basin.R[np.ix_(np.where(valid)[0], np.where(valid)[0])]
                        resid = y_v - C_v @ q_x_mean - c_v
                        try:
                            R_inv = np.linalg.inv(R_v)
                            ll -= 0.5 * float(resid @ R_inv @ resid)
                        except np.linalg.LinAlgError:
                            ll -= 50.0
                log_probs[k] = ll

            # Softmax for q(z)
            log_probs -= np.max(log_probs)
            q_z = np.exp(log_probs)
            q_z_sum = np.sum(q_z)
            if q_z_sum > 1e-300:
                q_z /= q_z_sum
            else:
                q_z = np.ones(self.K) / self.K

            # M-step: update q(x) given q(z)
            # Weighted average of per-basin Kalman smoothers (simplified)
            prec = np.zeros((self.n, self.n))
            info = np.zeros(self.n)

            for k in range(self.K):
                if q_z[k] < 1e-10:
                    continue
                basin = self.basins[k]
                for t in range(T):
                    y_t = y_sequence[t]
                    valid = ~np.isnan(y_t)
                    if np.any(valid):
                        y_v = y_t[valid]
                        C_v = basin.C[valid, :]
                        c_v = basin.c[valid] if hasattr(basin, 'c') else np.zeros(int(np.sum(valid)))
                        R_v = basin.R[np.ix_(np.where(valid)[0], np.where(valid)[0])]
                        try:
                            R_inv = np.linalg.inv(R_v)
                            prec += q_z[k] * C_v.T @ R_inv @ C_v
                            info += q_z[k] * C_v.T @ R_inv @ (y_v - c_v)
                        except np.linalg.LinAlgError:
                            pass

            # Add prior precision
            prec += np.eye(self.n) * 0.01

            try:
                q_x_cov = np.linalg.inv(prec)
                q_x_cov = 0.5 * (q_x_cov + q_x_cov.T)
                q_x_mean = q_x_cov @ info
            except np.linalg.LinAlgError:
                pass

            # Compute ELBO
            current_elbo = self._compute_elbo(y_sequence, q_z, q_x_mean, q_x_cov)
            self._elbo_history.append(current_elbo)

            # Check convergence
            if abs(current_elbo - prev_elbo) < 1e-6:
                break
            prev_elbo = current_elbo

        self._q_z = q_z
        self._q_x_mean = q_x_mean
        self._q_x_cov = q_x_cov

        return {
            "q_z": q_z,
            "q_x_mean": q_x_mean,
            "q_x_cov": q_x_cov,
            "elbo": current_elbo if self._elbo_history else -np.inf,
            "n_iter": iteration + 1 if 'iteration' in dir() else 0,
        }

    def _compute_elbo(
        self, y_seq: np.ndarray, q_z: np.ndarray, q_x_mean: np.ndarray, q_x_cov: np.ndarray
    ) -> float:
        """Compute ELBO = E_q[log p(y|x,z)] - KL[q(z)||p(z)] - KL[q(x)||p(x)]."""
        T = y_seq.shape[0]

        # Expected log-likelihood
        ell = 0.0
        for k in range(self.K):
            if q_z[k] < 1e-10:
                continue
            basin = self.basins[k]
            for t in range(T):
                y_t = y_seq[t]
                valid = ~np.isnan(y_t)
                if np.any(valid):
                    y_v = y_t[valid]
                    C_v = basin.C[valid, :]
                    c_v = basin.c[valid] if hasattr(basin, 'c') else np.zeros(int(np.sum(valid)))
                    R_v = basin.R[np.ix_(np.where(valid)[0], np.where(valid)[0])]
                    resid = y_v - C_v @ q_x_mean - c_v
                    try:
                        R_inv = np.linalg.inv(R_v)
                        sign, logdet = np.linalg.slogdet(R_v)
                        ll = -0.5 * (float(resid @ R_inv @ resid) + logdet + len(resid) * np.log(2*np.pi))
                        ell += q_z[k] * ll
                    except np.linalg.LinAlgError:
                        ell -= 50.0 * q_z[k]

        # KL[q(z) || p(z)] where p(z) = uniform
        kl_z = 0.0
        for k in range(self.K):
            if q_z[k] > 1e-10:
                kl_z += q_z[k] * np.log(q_z[k] * self.K)

        # KL[q(x) || p(x)] where p(x) = N(0, I)
        sign, logdet_q = np.linalg.slogdet(q_x_cov)
        kl_x = 0.5 * (np.trace(q_x_cov) + float(q_x_mean @ q_x_mean) - self.n - logdet_q)

        return float(ell - kl_z - kl_x)

    def elbo(self) -> float:
        """Return the last computed ELBO."""
        if self._elbo_history:
            return self._elbo_history[-1]
        return -np.inf
