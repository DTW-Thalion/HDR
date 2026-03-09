from __future__ import annotations

from dataclasses import dataclass

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