from __future__ import annotations

import numpy as np
from .hsmm import DwellModel, hazard_at


class TargetSet:
    def __init__(self, center: np.ndarray, radius: float):
        self.center = center
        self.radius = radius

    @property
    def box_low(self) -> np.ndarray:
        return self.center - self.radius

    @property
    def box_high(self) -> np.ndarray:
        return self.center + self.radius

    @property
    def safety_low(self) -> np.ndarray:
        return self.center - self.radius * 1.5

    @property
    def safety_high(self) -> np.ndarray:
        return self.center + self.radius * 1.5
        # Conservative safety bounds (wider than target box)
        return self.center - 2.0 * self.radius

    @property
    def safety_high(self) -> np.ndarray:
        # Conservative safety bounds (wider than target box)
        return self.center + 2.0 * self.radius

    def dist2(self, x: np.ndarray, Q: np.ndarray | None = None, method: str = "box") -> float:
        if Q is not None:
            diff = x - self.center
            return diff.T @ Q @ diff
        else:
            return np.sum((x - self.center)**2)

    def project_box(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.center - self.radius, self.center + self.radius)


def build_target_set(k: int, config: dict) -> TargetSet:
    n = config.get("state_dim", 8)
    center = np.zeros(n)
    radius = 1.0
    return TargetSet(center, radius)


def sample_duration(dwell_model: DwellModel, rng):
    return dwell_model.sample(rng)


def hazard(dwell_model: DwellModel, length: int) -> float:
    return hazard_at(dwell_model, length)
