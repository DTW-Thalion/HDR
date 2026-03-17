"""
Population-Prior Treatment Planning
================================================
Def 11.10: Offline optimisation over approved regimen set.
"""
from __future__ import annotations

import numpy as np


class PopulationPriorPlanner:
    """Population-prior treatment planning.

    Offline optimisation over approved regimen set.
    """

    def __init__(self, B_k_prior: list[np.ndarray], approved_regimens: list[np.ndarray]):
        self.B_k_prior = [np.asarray(B, dtype=float) for B in B_k_prior]
        self.approved_regimens = [np.asarray(r, dtype=float) for r in approved_regimens]
        self.K = len(B_k_prior)

    def plan(self, patient_data: dict, H: int) -> np.ndarray:
        """Select best regimen for patient based on population prior.

        Parameters
        ----------
        patient_data : dict with 'basin_probs' (K,) and optionally 'x_current' (n,)
        H : planning horizon

        Returns
        -------
        best_regimen : selected regimen from approved set
        """
        basin_probs = np.asarray(patient_data.get("basin_probs", np.ones(self.K) / self.K))

        best_cost = float('inf')
        best_regimen = self.approved_regimens[0] if self.approved_regimens else np.zeros(1)

        for regimen in self.approved_regimens:
            # Expected cost under population prior
            cost = 0.0
            for k in range(self.K):
                # Predicted effect: B_k @ regimen over H steps
                B_k = self.B_k_prior[k]
                m = min(len(regimen), B_k.shape[1])
                effect = B_k[:, :m] @ regimen[:m]
                cost += basin_probs[k] * float(np.sum(effect ** 2)) * H

            if cost < best_cost:
                best_cost = cost
                best_regimen = regimen

        return best_regimen

    def accuracy(self, assignments: list[int], true_best: list[int]) -> float:
        """Compute accuracy of treatment assignments.

        Parameters
        ----------
        assignments : list of assigned regimen indices
        true_best : list of true best regimen indices

        Returns
        -------
        accuracy : fraction correct
        """
        if len(assignments) == 0:
            return 0.0
        correct = sum(1 for a, t in zip(assignments, true_best) if a == t)
        return float(correct / len(assignments))
