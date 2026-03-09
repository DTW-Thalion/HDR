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
    def __init__(self, model: EvaluationModel, init_mean: np.ndarray | None = None, init_cov_scale: float = 1.0):
        n = model.state_dim
        K = len(model.basins)
        init_mean = np.zeros(n) if init_mean is None else np.asarray(init_mean, dtype=float)
        self.model = model
        self.state = IMMState(
            mode_probs=np.ones(K) / K,
            states=[KalmanState(mean=init_mean.copy(), cov=np.eye(n) * init_cov_scale) for _ in range(K)],
            mixed_mean=init_mean.copy(),
            mixed_cov=np.eye(n) * init_cov_scale,
            map_mode=0,
            dwell_length=1,
        )

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
        return T

    def step(self, y: np.ndarray, mask: np.ndarray, u: np.ndarray) -> IMMState:
        K = len(self.model.basins)
        T = self.dynamic_transition()
        mu_prev = self.state.mode_probs
        c_j = T.T @ mu_prev
        c_j = np.clip(c_j, 1e-10, 1.0)
        mixed_states: list[KalmanState] = []
        for j in range(K):
            weights = T[:, j] * mu_prev / c_j[j]
            means = np.stack([s.mean for s in self.state.states], axis=0)
            mean_j = np.sum(weights[:, None] * means, axis=0)
            cov_j = np.zeros_like(self.state.states[0].cov)
            for i, st in enumerate(self.state.states):
                d = st.mean - mean_j
                cov_j += weights[i] * (st.cov + np.outer(d, d))
            mixed_states.append(KalmanState(mean=mean_j, cov=cov_j))

        new_states = []
        log_likes = np.zeros(K)
        for j, basin in enumerate(self.model.basins):
            pred = predict(mixed_states[j], basin.A, basin.B, u, basin.Q, basin.b)
            upd, ll = update(pred, y, mask, basin.C, basin.R, basin.c)
            new_states.append(upd)
            log_likes[j] = ll

        log_weights = np.log(c_j) + log_likes
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        mode_probs = weights / np.sum(weights)
        means = np.stack([s.mean for s in new_states], axis=0)
        mixed_mean = np.sum(mode_probs[:, None] * means, axis=0)
        mixed_cov = np.zeros_like(new_states[0].cov)
        for j, st in enumerate(new_states):
            d = st.mean - mixed_mean
            mixed_cov += mode_probs[j] * (st.cov + np.outer(d, d))
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
