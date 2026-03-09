from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model.coherence import coherence_from_state_history
from ..model.slds import EvaluationModel
from ..model.target_set import build_target_set
from .challenges import compose_challenges
from .emissions import generate_observation, heteroskedastic_R, observation_schedule


@dataclass
class Scenario:
    name: str
    model_mismatch: float = 0.12
    heavy_tail_noise: bool = False
    missingness_boost: float = 0.0
    delay_steps: int = 2
    target_drift_scale: float = 0.06
    transition_bias: bool = False
    emission_nonlinear_scale: float = 0.0
    mode_confusion: float = 0.0
    coherence_shift: float = 0.0
    noise_scale: float = 1.0
    inverse_crime: bool = False
    linear_drift_mode: bool = False  # Use monotone ramp drift instead of sinusoidal


def default_scenarios() -> dict[str, Scenario]:
    return {
        "nominal": Scenario(name="nominal"),
        "model_mismatch": Scenario(name="model_mismatch", model_mismatch=0.22),
        "missing_heterosk": Scenario(name="missing_heterosk", missingness_boost=0.25),
        "delayed_control": Scenario(name="delayed_control", delay_steps=3),
        "target_drift": Scenario(name="target_drift", target_drift_scale=0.12),
        "heavy_tail": Scenario(name="heavy_tail", heavy_tail_noise=True, noise_scale=1.4),
        "mode_confusion": Scenario(name="mode_confusion", mode_confusion=0.2),
        "mode_b_escape": Scenario(name="mode_b_escape", transition_bias=True, heavy_tail_noise=False, model_mismatch=0.16),
        "coherence_under": Scenario(name="coherence_under", coherence_shift=-0.25),
        "coherence_over": Scenario(name="coherence_over", coherence_shift=0.25),
        "inverse_crime": Scenario(name="inverse_crime", inverse_crime=True, model_mismatch=0.0, emission_nonlinear_scale=0.0),
    }


class SyntheticEnv:
    def __init__(
        self,
        eval_model: EvaluationModel,
        config: dict,
        rng: np.random.Generator,
        scenario: Scenario,
        episode_idx: int = 0,
        initial_basin: int | None = None,
        min_initial_dwell: int = 0,
    ):
        self.eval_model = eval_model
        self.config = config
        self.rng = rng
        self.scenario = scenario
        self.n = eval_model.state_dim
        self.m = eval_model.obs_dim
        self.u_dim = eval_model.control_dim
        self.T = int(config["steps_per_episode"])
        self.episode_idx = episode_idx
        self.challenges, self.challenge_markers = compose_challenges(self.T, self.n, include_stress=True)
        self.mask_schedule = observation_schedule(self.T, self.m, rng, profile_name=config["profile_name"])
        if scenario.missingness_boost > 0:
            drop = rng.uniform(size=self.mask_schedule.shape) < scenario.missingness_boost
            self.mask_schedule = self.mask_schedule * (~drop)
        self.A_true = []
        self.B_true = []
        self.C_true = []
        self.Q_true = []
        self.R_true = []
        self.b_true = []
        self.c_true = []
        for basin in eval_model.basins:
            if scenario.inverse_crime:
                A = basin.A.copy()
                B = basin.B.copy()
                C = basin.C.copy()
                Q = basin.Q.copy()
                R = basin.R.copy()
                b = basin.b.copy()
                c = basin.c.copy()
            else:
                drift = rng.normal(scale=scenario.model_mismatch, size=basin.A.shape)
                drift = 0.25 * (drift + drift.T)
                A = basin.A + 0.15 * drift
                vals = np.linalg.eigvals(A)
                max_abs = np.max(np.abs(vals))
                target_rho = min(0.995, basin.rho + (0.03 if basin.rho > 0.9 else -0.02))
                A = A * (target_rho / max(max_abs, 1e-8))
                B = basin.B * (1.0 + rng.normal(scale=0.08, size=basin.B.shape))
                C = basin.C + rng.normal(scale=0.03, size=basin.C.shape)
                Q = basin.Q * (1.0 + abs(rng.normal(scale=0.15)))
                R = basin.R * (1.0 + abs(rng.normal(scale=0.15)))
                b = basin.b + rng.normal(scale=0.01, size=basin.b.shape)
                c = basin.c + rng.normal(scale=0.01, size=basin.c.shape)
            self.A_true.append(A)
            self.B_true.append(B)
            self.C_true.append(C)
            self.Q_true.append(Q)
            self.R_true.append(R)
            self.b_true.append(b)
            self.c_true.append(c)
        self.transition = eval_model.transition.copy()
        self.delay_steps = scenario.delay_steps
        self.control_queue = [np.zeros(self.u_dim) for _ in range(max(1, self.delay_steps))]
        self.initial_basin = int(initial_basin) if initial_basin is not None else int(rng.choice(len(eval_model.basins), p=self._start_probs()))
        self.min_initial_dwell = int(min_initial_dwell)
        self.reset()

    def _start_probs(self) -> np.ndarray:
        K = len(self.eval_model.basins)
        if K == 3:
            return np.array([0.45, 0.3, 0.25])
        return np.array([0.35, 0.25, 0.2, 0.2])

    def reset(self):
        self.t = 0
        self.z = self.initial_basin
        self.current_dwell = 1
        self.remaining_dwell = self.eval_model.dwell_models[self.z].sample(self.rng)
        # Enforce minimum initial dwell so that controllers can observe and
        # respond to the initial basin before a transition occurs.  This is
        # especially important for Mode B evaluation (mode_b_escape scenario)
        # where the initial maladaptive basin must persist long enough for
        # the entry gate to fire while still in the maladaptive state.
        if self.min_initial_dwell > 0:
            self.remaining_dwell = max(self.remaining_dwell, self.min_initial_dwell)
        self.x = self.rng.normal(scale=0.35, size=self.n)
        if self.z == 1:
            self.x[[0, 1, 3, 4]] += 0.8
        elif self.z == 2:
            self.x[[0, 6]] += 0.45
        elif self.z == 3:
            self.x[[0, 1, 4, 6]] += 0.65
        self.x_hist = [self.x.copy()]
        self.u_hist = []
        self.z_hist = [self.z]
        self.target_hist = []
        self.kappa_hist = []
        return self.observe()

    def _mode_transition(self, u: np.ndarray):
        self.remaining_dwell -= 1
        extra_escape = False
        if self.scenario.transition_bias and self.z == 1:
            escape_drive = float(np.clip(np.sum(np.maximum(u[[0, 1, 5, 6]], 0.0)), 0.0, 1.5))
            if self.rng.uniform() < 0.02 + 0.10 * escape_drive:
                extra_escape = True
        if self.remaining_dwell <= 0 or extra_escape:
            row = self.transition[self.z].copy()
            row[self.z] = 0.0
            row = row / np.sum(row)
            self.z = int(self.rng.choice(len(row), p=row))
            self.current_dwell = 1
            self.remaining_dwell = self.eval_model.dwell_models[self.z].sample(self.rng)
        else:
            self.current_dwell += 1
        self.z_hist.append(self.z)

    def _true_kappa(self) -> float:
        axes = list(self.config.get("coherence_axes", [1, 5, 6]))
        hist = np.asarray(self.x_hist, dtype=float)
        kappa = coherence_from_state_history(hist, axes=axes, window=int(self.config["coherence_window"]))
        return float(np.clip(kappa + self.scenario.coherence_shift, 0.0, 1.0))

    def _noise(self, Q: np.ndarray) -> np.ndarray:
        if self.scenario.heavy_tail_noise and self.rng.uniform() < 0.15:
            return self.rng.standard_t(df=3, size=self.n) * np.sqrt(np.diag(Q) * float(self.scenario.noise_scale)) * 2.2
        return self.rng.multivariate_normal(np.zeros(self.n), Q * float(self.scenario.noise_scale))

    def observe(self):
        # When using monotone linear drift: compute the per-step drift velocity
        # from target_drift_scale / steps_per_day so that after 1 day the target
        # has shifted by drift_scale, making residual monotonically ∝ drift velocity.
        linear_drift_rate = 0.0
        if getattr(self.scenario, "linear_drift_mode", False):
            linear_drift_rate = self.scenario.target_drift_scale / float(self.config["steps_per_day"])
        target = build_target_set(
            self.t, self.config,
            history=np.asarray(self.x_hist),
            drift_scale=self.scenario.target_drift_scale,
            linear_drift_rate=linear_drift_rate,
        )
        self.target_hist.append({"low": target.box_low.copy(), "high": target.box_high.copy(), "fallback": target.fallback_used})
        kappa = self._true_kappa()
        self.kappa_hist.append(kappa)
        mask = self.mask_schedule[min(self.t, self.T - 1)].copy()
        basin_idx = self.z
        R = heteroskedastic_R(self.R_true[basin_idx] * float(self.scenario.noise_scale), self.x, mask, self.t)
        y = generate_observation(
            self.x,
            self.C_true[basin_idx],
            self.c_true[basin_idx],
            R,
            mask,
            self.rng,
            nonlinear_scale=self.scenario.emission_nonlinear_scale,
        )
        return {
            "t": self.t,
            "x_true": self.x.copy(),
            "z_true": int(self.z),
            "y": y,
            "mask": mask.copy(),
            "target": target,
            "kappa_true": kappa,
            "challenge": self.challenges[self.t].copy() if self.t < self.T else np.zeros(self.n),
        }

    def step(self, u: np.ndarray):
        u = np.asarray(u, dtype=float)
        self.u_hist.append(u.copy())
        self.control_queue.append(u.copy())
        u_eff = self.control_queue.pop(0)
        k = self.z
        seasonal = 0.02 * np.sin(2 * np.pi * (self.t + self.episode_idx) / max(self.T, 16))
        drift_mat = np.zeros_like(self.A_true[k])
        drift_mat[0, 1] = 0.03 * seasonal
        drift_mat[1, 0] = 0.02 * seasonal
        A_t = self.A_true[k] + drift_mat
        nonlinear = 0.03 * np.tanh(self.x) + 0.01 * self.x * np.roll(self.x, 1)
        x_next = (
            A_t @ self.x
            + self.B_true[k] @ u_eff
            + self.challenges[self.t]
            + self.b_true[k]
            + nonlinear
            + self._noise(self.Q_true[k])
        )
        self.x = np.clip(x_next, -3.0, 3.0)
        self.x_hist.append(self.x.copy())
        self._mode_transition(u_eff)
        self.t += 1
        done = self.t >= self.T
        obs = self.observe() if not done else None
        info = {
            "u_eff": u_eff,
            "done": done,
            "challenge_markers": self.challenge_markers,
        }
        return obs, info


def simulate_policy(
    env: SyntheticEnv,
    policy_fn,
    max_steps: int | None = None,
) -> dict:
    obs = env.reset()
    max_steps = env.T if max_steps is None else int(max_steps)
    traj = {
        "x_true": [],
        "z_true": [],
        "y": [],
        "mask": [],
        "u": [],
        "kappa_true": [],
        "target_low": [],
        "target_high": [],
    }
    for _ in range(max_steps):
        u = np.asarray(policy_fn(obs), dtype=float)
        traj["x_true"].append(obs["x_true"])
        traj["z_true"].append(obs["z_true"])
        traj["y"].append(obs["y"])
        traj["mask"].append(obs["mask"])
        traj["u"].append(u)
        traj["kappa_true"].append(obs["kappa_true"])
        traj["target_low"].append(obs["target"].box_low)
        traj["target_high"].append(obs["target"].box_high)
        obs, info = env.step(u)
        if info["done"]:
            break
    return {k: np.asarray(v) for k, v in traj.items()}
