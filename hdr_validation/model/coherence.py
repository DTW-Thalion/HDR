from __future__ import annotations

import numpy as np


def coherence_grad(kappa: float, kappa_lo: float, kappa_hi: float) -> float:
    if kappa < kappa_lo:
        return float(2.0 * (kappa - kappa_lo))
    if kappa > kappa_hi:
        return float(2.0 * (kappa - kappa_hi))
    return 0.0


def coherence_penalty(kappa: float, kappa_lo: float, kappa_hi: float) -> float:
    if kappa < kappa_lo:
        return float((kappa - kappa_lo)**2)
    if kappa > kappa_hi:
        return float((kappa - kappa_hi)**2)
    return 0.0