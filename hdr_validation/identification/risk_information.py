"""
Risk-Information Pareto Frontier
============================================
Def 11.12: For each candidate perturbation, compute Fisher info
and safety violation probability.
"""
from __future__ import annotations

import numpy as np


class RiskInformationFrontier:
    """Risk-information Pareto frontier.

    For each candidate perturbation, compute Fisher info (Eq 11.14)
    and safety violation probability (Eq 11.15).
    """

    def __init__(self, model, safety_set: dict):
        self.model = model  # basin model with A, B, Q, R
        self.safety_set = safety_set  # {'lo': ndarray, 'hi': ndarray}
        self.lo = np.asarray(safety_set.get("lo", -np.ones(8)))
        self.hi = np.asarray(safety_set.get("hi", np.ones(8)))

    def evaluate(self, perturbation: np.ndarray) -> tuple[float, float]:
        """Evaluate a candidate perturbation.

        Parameters
        ----------
        perturbation : (m,) control perturbation

        Returns
        -------
        (fisher_info, risk) : Fisher information proxy and safety risk
        """
        perturbation = np.asarray(perturbation, dtype=float)

        # Fisher information proxy: ||B @ u||^2 / sigma^2
        Bu = self.model.B @ perturbation[:self.model.B.shape[1]]
        sigma2 = float(np.mean(np.diag(self.model.Q)))
        fisher = float(np.sum(Bu ** 2)) / max(sigma2, 1e-10)

        # Safety risk: probability of x + Bu leaving safety set
        # Approximate as fraction of dimensions that would violate
        x_perturbed = Bu  # perturbation effect on state (from zero)
        violations = np.sum((x_perturbed < self.lo) | (x_perturbed > self.hi))
        risk = float(violations / len(x_perturbed))

        return (fisher, risk)

    def pareto_frontier(self, candidates: list[np.ndarray]) -> list[int]:
        """Compute Pareto frontier indices (max info, min risk).

        Returns indices of non-dominated candidates.
        """
        if not candidates:
            return []

        evals = [self.evaluate(c) for c in candidates]
        n = len(evals)
        is_pareto = [True] * n

        for i in range(n):
            if not is_pareto[i]:
                continue
            for j in range(n):
                if i == j or not is_pareto[j]:
                    continue
                # j dominates i if j has >= info AND <= risk (with at least one strict)
                if evals[j][0] >= evals[i][0] and evals[j][1] <= evals[i][1]:
                    if evals[j][0] > evals[i][0] or evals[j][1] < evals[i][1]:
                        is_pareto[i] = False
                        break

        return [i for i in range(n) if is_pareto[i]]
