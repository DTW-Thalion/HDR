from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import stats


@dataclass
class DwellModel:
    kind: str
    params: dict[str, float]
    max_len: int = 512

    def pmf(self) -> np.ndarray:
        l = np.arange(1, self.max_len + 1, dtype=float)
        if self.kind == "poisson":
            mean = max(float(self.params.get("mean", 8.0)), 1.0)
            vals = stats.poisson.pmf(l - 1, mu=mean - 1)
        elif self.kind == "lognormal":
            mu = float(self.params.get("mu", 2.0))
            sigma = float(self.params.get("sigma", 0.5))
            edges = np.arange(0.5, self.max_len + 1.5, 1.0)
            cdf = stats.lognorm.cdf(edges, s=sigma, scale=np.exp(mu))
            vals = np.diff(cdf)
        elif self.kind == "zipf":
            a = float(self.params.get("a", 1.8))
            vals = (l ** (-a))
        elif self.kind == "discrete_weibull":
            q = float(self.params.get("q", 0.8))
            beta = float(self.params.get("beta", 1.4))
            vals = (q ** ((l - 1) ** beta)) - (q ** (l ** beta))
        else:
            vals = np.exp(-0.2 * l)
        vals = np.maximum(vals, 1e-12)
        vals = vals / np.sum(vals)
        return vals

    def survival(self) -> np.ndarray:
        pmf = self.pmf()
        return np.flip(np.cumsum(np.flip(pmf)))

    def hazard(self) -> np.ndarray:
        pmf = self.pmf()
        surv = self.survival()
        return np.clip(pmf / np.maximum(surv, 1e-12), 0.0, 1.0)

    def sample(self, rng: np.random.Generator) -> int:
        pmf = self.pmf()
        choices = np.arange(1, len(pmf) + 1)
        return int(rng.choice(choices, p=pmf))


def hazard_at(dwell_model: DwellModel, length: int) -> float:
    hz = dwell_model.hazard()
    idx = int(np.clip(length - 1, 0, len(hz) - 1))
    return float(hz[idx])


def fit_hazard_from_sequences(z_sequences: list[np.ndarray], K: int, max_len: int = 128) -> dict[int, np.ndarray]:
    durations: dict[int, list[int]] = {k: [] for k in range(K)}
    for seq in z_sequences:
        current = int(seq[0])
        run = 1
        for z in seq[1:]:
            z = int(z)
            if z == current:
                run += 1
            else:
                durations[current].append(run)
                current = z
                run = 1
        durations[current].append(run)
    hazards = {}
    for k in range(K):
        counts = np.zeros(max_len, dtype=float)
        for d in durations[k]:
            counts[min(d, max_len) - 1] += 1.0
        pmf = counts + 1e-3
        pmf /= np.sum(pmf)
        surv = np.flip(np.cumsum(np.flip(pmf)))
        hazards[k] = pmf / np.maximum(surv, 1e-12)
    return hazards


def entrenchment_diagnostic(
    mode: int,
    dwell_length: int,
    dwell_models: list[DwellModel],
    maladaptive_index: int = 1,
    hazard_threshold: float = 0.08,
    tail_length: int = 16,
) -> bool:
    if mode != maladaptive_index:
        return False
    hazard = hazard_at(dwell_models[mode], dwell_length)
    surv = dwell_models[mode].survival()
    idx = min(max(tail_length - 1, 0), len(surv) - 1)
    tail = float(surv[idx])
    return bool((hazard <= hazard_threshold) or (tail >= 0.25 and dwell_length >= tail_length))
