"""
Empirical Committor Recovery — HDR v7.3
========================================
Def 11.6: Nadaraya-Watson kernel regression (Eq 11.7).
Prop 11.7: boundary convergence rate O(N^{-1/(n+2)}).
"""
from __future__ import annotations

import numpy as np


class CommittorRecovery:
    """Empirical committor from trajectory data.

    Uses Nadaraya-Watson kernel regression to estimate committor
    function from observed trajectories.
    """

    def __init__(self, kernel_bandwidth: float = 1.0):
        self.kernel_bandwidth = kernel_bandwidth
        self._X = None
        self._y = None

    def estimate(self, trajectories: list[np.ndarray], basin_labels: list[np.ndarray]) -> callable:
        """Estimate committor from trajectory data.

        Parameters
        ----------
        trajectories : list of (T_i, n) state trajectories
        basin_labels : list of (T_i,) basin label sequences

        Returns
        -------
        q_hat : callable mapping state -> committor value
        """
        # Collect state-outcome pairs
        X_list = []
        y_list = []

        for traj, labels in zip(trajectories, basin_labels):
            T = len(labels)
            # Outcome: did trajectory eventually reach basin 0 (success)?
            final_basin = int(labels[-1])
            outcome = 1.0 if final_basin == 0 else 0.0

            for t in range(T):
                X_list.append(traj[t])
                y_list.append(outcome)

        if len(X_list) == 0:
            return lambda x: 0.5

        self._X = np.array(X_list)
        self._y = np.array(y_list)

        def q_hat(x):
            x = np.asarray(x, dtype=float)
            dists = np.sum((self._X - x[None, :]) ** 2, axis=1)
            weights = np.exp(-dists / (2.0 * self.kernel_bandwidth ** 2))
            total = np.sum(weights)
            if total < 1e-10:
                return 0.5
            return float(np.clip(np.sum(weights * self._y) / total, 0.0, 1.0))

        return q_hat

    def boundary(self, q_hat, level: float = 0.5) -> np.ndarray:
        """Extract approximate boundary at q_hat = level.

        Returns states where q_hat is closest to the level set.
        """
        if self._X is None:
            return np.array([])

        q_vals = np.array([q_hat(x) for x in self._X])
        distances = np.abs(q_vals - level)

        # Return points within 10% of level
        threshold = 0.1
        mask = distances < threshold
        if np.any(mask):
            return self._X[mask]

        # Fallback: return closest 10 points
        n_return = min(10, len(distances))
        indices = np.argsort(distances)[:n_return]
        return self._X[indices]
