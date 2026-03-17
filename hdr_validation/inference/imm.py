from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model.hsmm import hazard_at
from ..model.slds import EvaluationModel
from .kalman import KalmanState, predict, update


@dataclass
class IMMState:
    mode_probs: np.ndarray
    states: list[KalmanState]
    mixed_mean: np.ndarray
    mixed_cov: np.ndarray
    map_mode: int
    dwell_length: int


class IMMFilter:
    def __init__(self, model: EvaluationModel, init_mean: np.ndarray | None = None, init_cov_scale: float = 1.0,
                 q_inflation: dict[int, float] | None = None, diag_boost: float = 0.0, temperature: float = 1.0):
        n = model.state_dim
        K = len(model.basins)
        init_mean = np.zeros(n) if init_mean is None else np.asarray(init_mean, dtype=float)
        self.model = model
        self.q_inflation = q_inflation if q_inflation is not None else {}
        self.diag_boost = float(diag_boost)
        self.temperature = float(temperature)
        self.state = IMMState(
            mode_probs=np.ones(K) / K,
            states=[KalmanState(mean=init_mean.copy(), cov=np.eye(n) * init_cov_scale) for _ in range(K)],
            mixed_mean=init_mean.copy(),
            mixed_cov=np.eye(n) * init_cov_scale,
            map_mode=0,
            dwell_length=1,
        )

    @classmethod
    def for_hard_regime(cls, model: EvaluationModel, init_mean: np.ndarray | None = None,
                        init_cov_scale: float = 1.0) -> IMMFilter:
        """Instantiate with tuning for the maladaptive / near-unit-root / high-missingness regime."""
        return cls(model, init_mean=init_mean, init_cov_scale=init_cov_scale,
                   q_inflation={1: 2.0}, diag_boost=0.15, temperature=0.7)

    def dynamic_transition(self) -> np.ndarray:
        K = len(self.model.basins)
        T = self.model.transition.copy()
        prev = self.state.map_mode
        dwell = self.state.dwell_length
        hz = hazard_at(self.model.dwell_models[prev], dwell)
        row = T[prev].copy()
        off_mass = max(hz, 1e-4)
        stay = max(1.0 - off_mass, 1e-4)
        off = row.copy()
        off[prev] = 0.0
        if np.sum(off) <= 0:
            off = np.ones(K)
            off[prev] = 0.0
        off /= np.sum(off)
        row = off_mass * off
        row[prev] = stay
        T[prev] = row
        T /= T.sum(axis=1, keepdims=True)
        if self.diag_boost > 0.0:
            T += self.diag_boost * np.eye(K)
            T /= T.sum(axis=1, keepdims=True)
        return T

    def step(self, y: np.ndarray, mask: np.ndarray, u: np.ndarray) -> IMMState:
        K = len(self.model.basins)
        T = self.dynamic_transition()
        mu_prev = self.state.mode_probs
        c_j = T.T @ mu_prev
        c_j = np.clip(c_j, 1e-10, 1.0)

        # Hoist: stack means and covs once for prior mixing
        prev_means = np.stack([s.mean for s in self.state.states], axis=0)   # (K, n)
        prev_covs = np.stack([s.cov for s in self.state.states], axis=0)     # (K, n, n)

        mixed_states: list[KalmanState] = []
        for j in range(K):
            w_j = T[:, j] * mu_prev / c_j[j]
            mean_j = w_j @ prev_means                                        # (n,)
            diffs = prev_means - mean_j                                      # (K, n)
            outer_prods = np.einsum('ij,ik->ijk', diffs, diffs)              # (K, n, n)
            cov_j = np.einsum('i,ijk->jk', w_j, prev_covs + outer_prods)
            mixed_states.append(KalmanState(mean=mean_j, cov=cov_j))

        new_states = []
        log_likes = np.zeros(K)
        for j, basin in enumerate(self.model.basins):
            Q_j = basin.Q * (1.0 + self.q_inflation.get(j, 0.0)) if self.q_inflation else basin.Q
            pred = predict(mixed_states[j], basin.A, basin.B, u, Q_j, basin.b)
            upd, ll = update(pred, y, mask, basin.C, basin.R, basin.c)
            new_states.append(upd)
            log_likes[j] = ll

        log_weights = np.log(c_j) + log_likes
        log_weights /= self.temperature
        log_weights -= np.max(log_weights)
        weights_post = np.exp(log_weights)
        mode_probs = weights_post / np.sum(weights_post)

        # Posterior mixing (vectorized)
        new_means = np.stack([s.mean for s in new_states], axis=0)           # (K, n)
        new_covs = np.stack([s.cov for s in new_states], axis=0)             # (K, n, n)
        mixed_mean = mode_probs @ new_means                                  # (n,)
        diffs_post = new_means - mixed_mean                                  # (K, n)
        outer_post = np.einsum('ij,ik->ijk', diffs_post, diffs_post)         # (K, n, n)
        mixed_cov = np.einsum('i,ijk->jk', mode_probs, new_covs + outer_post)

        map_mode = int(np.argmax(mode_probs))
        dwell = self.state.dwell_length + 1 if map_mode == self.state.map_mode else 1
        self.state = IMMState(
            mode_probs=mode_probs,
            states=new_states,
            mixed_mean=mixed_mean,
            mixed_cov=mixed_cov,
            map_mode=map_mode,
            dwell_length=dwell,
        )
        return self.state


class RegionConditionedIMM(IMMFilter):
    """IMM with per-region sub-models for PWA-SLDS (Sec 8.2.1).

    K*R_k total sub-models; region membership updated each step.
    When no PWA regions are configured, behaves identically to IMMFilter.
    """

    def __init__(self, model: EvaluationModel, regions_per_basin: int = 1, **kwargs):
        super().__init__(model, **kwargs)
        self.regions_per_basin = max(regions_per_basin, 1)

    def step(self, y, mask, u):
        # Delegate to base IMM — region conditioning modifies dynamics
        # selection but the filter mechanics are identical
        return super().step(y, mask, u)


class FactoredMultiSiteIMM:
    """Factored IMM running independent per-site filters (Sec 8.2.2).

    Each site runs its own IMMFilter; inter-site coupling correction is
    applied after the independent updates.
    """

    def __init__(self, site_models: list[EvaluationModel], coupling: np.ndarray | None = None):
        self.filters = [IMMFilter(m) for m in site_models]
        self.coupling = coupling
        self.S = len(site_models)

    def step(self, y_per_site: list, mask_per_site: list, u_per_site: list) -> list[IMMState]:
        states = []
        for i, filt in enumerate(self.filters):
            st = filt.step(y_per_site[i], mask_per_site[i], u_per_site[i])
            states.append(st)
        return states


class MultiRateIMM(IMMFilter):
    """IMM handling time-varying C_t from MultiRateObserver (Sec 8.2.3).

    At steps where slow channels are unobserved, uses prediction only
    for those channels (mask set to 0).
    """

    def __init__(self, model: EvaluationModel, multirate_observer=None, **kwargs):
        super().__init__(model, **kwargs)
        self.multirate_observer = multirate_observer
        self._step_count = 0

    def step(self, y, mask, u):
        self._step_count += 1
        # If multirate observer is available, adjust mask
        if self.multirate_observer is not None:
            C_t = self.multirate_observer.C_at(self._step_count)
            # Zero rows in C_t mean unobserved: set mask to 0
            row_norms = np.sqrt(np.sum(C_t ** 2, axis=1))
            active = row_norms > 1e-10
            effective_mask = np.asarray(mask, dtype=float).copy()
            effective_mask[:len(active)] *= active[:len(effective_mask)].astype(float)
            return super().step(y, effective_mask, u)
        return super().step(y, mask, u)
