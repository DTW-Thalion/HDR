"""
Challenge-Response Tau Estimation — HDR v7.3
=============================================
Def 11.11: Nonlinear least-squares fit of exponential recovery model.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


class TauEstimator:
    """Challenge-response tau_i estimation.

    Fits exponential recovery model: x(t) = x_inf + (x0 - x_inf) * exp(-t/tau)
    """

    def __init__(self):
        pass

    def estimate(self, trajectory: np.ndarray, x0: float, x_inf: float) -> float:
        """Estimate time constant tau from recovery trajectory.

        Parameters
        ----------
        trajectory : (T,) observed recovery values
        x0 : initial value (at challenge)
        x_inf : asymptotic value (recovery target)

        Returns
        -------
        tau : estimated time constant (positive)
        """
        trajectory = np.asarray(trajectory, dtype=float)
        T = len(trajectory)
        if T < 2:
            return 1.0

        t_arr = np.arange(T, dtype=float)

        def loss(log_tau):
            tau = np.exp(log_tau)
            predicted = x_inf + (x0 - x_inf) * np.exp(-t_arr / tau)
            return float(np.sum((trajectory - predicted) ** 2))

        # Search over log(tau) in reasonable range
        result = minimize_scalar(loss, bounds=(-2, 6), method='bounded')
        tau = np.exp(result.x)

        return float(max(tau, 0.01))
