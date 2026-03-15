"""
Transition Rate Estimation — HDR v7.0
======================================
Def 11.8: Fits HSMM to labelled trajectories via Baum-Welch.
"""
from __future__ import annotations

import numpy as np


class TransitionRateEstimator:
    """Individual basin transition rate estimation.

    Fits HSMM parameters to labelled trajectories.
    """

    def __init__(self, K: int, dwell_family: str = "poisson"):
        self.K = K
        self.dwell_family = dwell_family

    def fit(self, label_sequences: list[np.ndarray]) -> dict:
        """Estimate transition rates from labelled sequences.

        Parameters
        ----------
        label_sequences : list of (T_i,) integer label arrays

        Returns
        -------
        dict with 'transition_matrix', 'dwell_means', 'dwell_counts'
        """
        K = self.K
        # Count transitions
        T_counts = np.zeros((K, K))
        dwell_lengths = {k: [] for k in range(K)}

        for labels in label_sequences:
            labels = np.asarray(labels, dtype=int)
            current_basin = labels[0]
            current_dwell = 1

            for t in range(1, len(labels)):
                if labels[t] == current_basin:
                    current_dwell += 1
                else:
                    T_counts[current_basin, labels[t]] += 1
                    dwell_lengths[current_basin].append(current_dwell)
                    current_basin = labels[t]
                    current_dwell = 1

            # Record final dwell
            dwell_lengths[current_basin].append(current_dwell)

        # Normalize transition matrix
        row_sums = T_counts.sum(axis=1, keepdims=True)
        transition = np.where(row_sums > 0, T_counts / np.maximum(row_sums, 1e-10), np.ones((K, K)) / K)
        # Set diagonal to 0 (transitions are between different states)
        np.fill_diagonal(transition, 0)
        row_sums = transition.sum(axis=1, keepdims=True)
        transition = np.where(row_sums > 0, transition / np.maximum(row_sums, 1e-10), np.ones((K, K)) / K)

        # Compute dwell means
        dwell_means = {}
        dwell_counts = {}
        for k in range(K):
            if dwell_lengths[k]:
                dwell_means[k] = float(np.mean(dwell_lengths[k]))
                dwell_counts[k] = len(dwell_lengths[k])
            else:
                dwell_means[k] = 1.0
                dwell_counts[k] = 0

        return {
            "transition_matrix": transition,
            "dwell_means": dwell_means,
            "dwell_counts": dwell_counts,
        }
