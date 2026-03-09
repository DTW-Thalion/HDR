from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..control.baselines import myopic_policy
from ..control.mode_b import committor, hybrid_mode_b_action, posterior_committor
from ..control.mode_c import supervisor_mode_select
from ..control.mpc import MPCResult, solve_mode_a
from ..generator.ground_truth import SyntheticEnv
from ..inference.ici import compute_T_k_eff
from ..inference.imm import IMMFilter
from ..metrics import recovery_time_from_challenge
from ..model.coherence import coherence_from_state_history, coherence_penalty
from ..model.hsmm import entrenchment_diagnostic
from ..model.safety import circadian_allowed_mask, observation_intervals
from ..model.slds import EvaluationModel, pooled_basin
from ..control.lqr import dlqr


@dataclass
class ClosedLoopOutputs:
    per_step: dict[str, np.ndarray]
    episode_summary: dict[str, float | int | str | bool]


def _dist_to_box(x: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    proj = np.clip(x, low, high)
    d = x - proj
    return float(d @ d)


def run_imm_on_sequence(data: dict, eval_model: EvaluationModel, config: dict) -> dict:
    imm = IMMFilter(eval_model)
    x_hat = []
    mode_probs = []
    map_modes = []
    for t in range(len(data["z_true"])):
        y = np.nan_to_num(data["y"][t], nan=0.0)
        mask = data["mask"][t]
        u = data["u"][t] if "u" in data else np.zeros(eval_model.control_dim)
        st = imm.step(y, mask, u)
        x_hat.append(st.mixed_mean.copy())
        mode_probs.append(st.mode_probs.copy())
        map_modes.append(st.map_mode)
    return {"x_hat": np.asarray(x_hat), "mode_probs": np.asarray(mode_probs), "map_mode": np.asarray(map_modes)}


def choose_policy_action(
    policy_name: str,
    obs_ctl: dict,
    eval_model: EvaluationModel,
    config: dict,
    used_burden: float,
    policy_cache: dict | None = None,
    with_tau: bool = True,
    with_coherence: bool = True,
) -> tuple[np.ndarray, dict]:
    mode = int(obs_ctl["map_mode"])
    basin = eval_model.basins[mode]
    policy_cache = {} if policy_cache is None else policy_cache
    start = time.perf_counter()
    meta = {"mode_b_triggered": False, "controller_notes": "baseline", "risk": 0.0, "feasible": True, "allowed_mask": np.ones(eval_model.control_dim, dtype=int)}
    x_ref = 0.5 * (obs_ctl["target"].box_low + obs_ctl["target"].box_high)
    if policy_name == "open_loop":
        u = np.zeros(eval_model.control_dim)
    elif policy_name == "pooled_lqr":
        K = policy_cache["pooled_K"]
        u = -K @ (obs_ctl["x_hat"] - x_ref)
        from ..model.safety import apply_control_constraints
        u, _ = apply_control_constraints(u, config, step=int(obs_ctl["t"]), used_burden=used_burden)
    elif policy_name == "basin_lqr":
        K = policy_cache["basin_Ks"][mode]
        u = -K @ (obs_ctl["x_hat"] - x_ref)
        from ..model.safety import apply_control_constraints
        u, _ = apply_control_constraints(u, config, step=int(obs_ctl["t"]), used_burden=used_burden)
    elif policy_name == "myopic":
        u = myopic_policy(obs_ctl, basin, config, used_burden=used_burden, with_coherence=with_coherence)
    else:
        decision = hybrid_mode_b_action(
            obs_ctl,
            basin_idx=mode,
            posterior_maladaptive=float(obs_ctl["mode_probs"][1]) if len(obs_ctl["mode_probs"]) > 1 else 0.0,
            entrenchment=bool(obs_ctl["entrenchment"]),
            q_hat=float(obs_ctl["q_hat"]),
            config=config,
        )
        if decision.triggered and policy_name == "hdr_main":
            u = decision.u
            meta["mode_b_triggered"] = True
            meta["controller_notes"] = decision.notes
        else:
            res: MPCResult = solve_mode_a(
                x_hat=obs_ctl["x_hat"],
                P_hat=obs_ctl["P_hat"],
                basin=basin,
                target=obs_ctl["target"],
                kappa_hat=float(obs_ctl["kappa_hat"]),
                config=config,
                step=int(obs_ctl["t"]),
                used_burden=used_burden,
                with_tau=with_tau,
                with_coherence=with_coherence,
            )
            u = res.u
            meta["controller_notes"] = res.notes
            meta["risk"] = res.risk
            meta["feasible"] = res.feasible
            meta["allowed_mask"] = res.allowed_mask
    meta["solve_time"] = time.perf_counter() - start
    return np.asarray(u, dtype=float), meta


def run_closed_loop_episode(
    eval_model: EvaluationModel,
    config: dict,
    env: SyntheticEnv,
    policy_name: str,
    allow_mode_b: bool = True,
    with_tau: bool = True,
    with_coherence: bool = True,
) -> ClosedLoopOutputs:
    obs = env.reset()
    imm = IMMFilter(eval_model)
    pooled = pooled_basin(eval_model)
    pooled_K, _ = dlqr(pooled.A, pooled.B, np.eye(pooled.A.shape[0]), np.eye(pooled.B.shape[1]) * float(config["lambda_u"]))
    basin_Ks = []
    for basin in eval_model.basins:
        Kb, _ = dlqr(basin.A, basin.B, np.eye(basin.A.shape[0]), np.eye(basin.B.shape[1]) * float(config["lambda_u"]))
        basin_Ks.append(Kb)
    policy_cache = {"pooled_K": pooled_K, "basin_Ks": basin_Ks}
    # Compute ICI T_k_eff threshold flag (BUG 1 fix): when any basin has
    # T_k_eff < omega_min, Mode C should permanently preempt Mode B.
    K = len(eval_model.basins)
    T_total_ep = float(env.T)
    pi_vals = [
        1.0 - float(config.get("mode1_base_rate", 0.16)) - 0.05,
        float(config.get("mode1_base_rate", 0.16)),
        0.05,
    ][:K]
    p_miss_ep = float(config.get("missing_fraction_target", 0.516))
    omega_min_factor = float(config.get("omega_min_factor", 0.005))
    omega_min_ep = omega_min_factor * T_total_ep
    T_k_eff_ep = [
        compute_T_k_eff(T_total_ep, pi_vals[k], p_miss_ep, eval_model.basins[k].rho)
        for k in range(K)
    ]
    t_k_eff_below_threshold = any(t < omega_min_ep for t in T_k_eff_ep)
    # Pre-compute per-basin committor values for posterior-weighted q̂.
    # The posterior committor q̂_t = Σ_k P(z_t=k) * q_k is computed live
    # in the loop using posterior_committor(), which also applies ΔP(u).
    q_mode_static = committor(eval_model.transition, [1], [0]) if len(eval_model.basins) >= 3 else np.zeros(len(eval_model.basins))
    x_hat_hist = []
    per = {k: [] for k in [
        "t", "z_true", "map_mode", "posterior_maladaptive", "dist_true", "dist_est",
        "time_in_target", "risk", "safety_violation", "burden_used", "solve_time",
        "rmse_state", "mode_b_triggered", "q_hat", "kappa_hat", "kappa_true",
        "circadian_adherence_step", "feasible", "stage_cost", "allowed_nonzero", "mode_correct",
        "mode_b_z_at_trigger",   # z_true at the moment Mode B fires (for FP measurement)
    ]}
    used_burden = 0.0
    mode_b_entries = 0
    fallback_count = 0
    for t in range(env.T):
        y = np.nan_to_num(obs["y"], nan=0.0)
        mask = obs["mask"]
        u_prev = env.u_hist[-1] if env.u_hist else np.zeros(eval_model.control_dim)
        st = imm.step(y, mask, u_prev)
        mode_probs = st.mode_probs.copy()
        if getattr(env.scenario, "mode_confusion", 0.0) > 0:
            mu = float(env.scenario.mode_confusion)
            mode_probs = (1 - mu) * mode_probs + mu / len(mode_probs)
            mode_probs /= np.sum(mode_probs)
        map_mode = int(np.argmax(mode_probs))
        x_hat = st.mixed_mean.copy()
        P_hat = st.mixed_cov.copy()
        x_hat_hist.append(x_hat)
        kappa_hat = coherence_from_state_history(np.asarray(x_hat_hist), axes=config.get("coherence_axes", [1, 5, 6]), window=int(config["coherence_window"]))
        entrenchment = entrenchment_diagnostic(map_mode, st.dwell_length, eval_model.dwell_models)
        # Posterior-weighted committor q̂_t = Σ_k P(z_t=k|y_{1:t}) * q_k
        # Fixes the false-positive problem: q̂_t is ≤ qmin only when the IMM
        # is truly confident (not just MAP-mode=maladaptive).
        q_hat = posterior_committor(
            mode_probs, eval_model.transition, maladaptive_idx=1, desired_idx=0
        )
        obs_ctl = {
            "t": t,
            "x_hat": x_hat,
            "P_hat": P_hat,
            "mode_probs": mode_probs,
            "map_mode": map_mode,
            "target": obs["target"],
            "kappa_hat": kappa_hat,
            "entrenchment": entrenchment,
            "q_hat": q_hat,
            "transition": eval_model.transition,  # for ΔP(u) mechanism in Mode B
            "used_burden": used_burden,            # for Mode B apply_control_constraints
        }
        # ICI gate (BUG 1 fix): call supervisor_mode_select with the
        # t_k_eff_below_threshold flag so Mode C permanently preempts Mode B
        # when T_k_eff < omega_min for any basin.
        ici_state_live = {
            "mode_c_recommended": t_k_eff_below_threshold,
            "condition_i": False,
            "condition_ii": False,
            "condition_iii": t_k_eff_below_threshold,
        }
        mode_b_entry_cond = (
            float(obs_ctl["mode_probs"][1]) >= float(config["pA"])
            if len(obs_ctl["mode_probs"]) > 1 else False
        )
        supervisor_decision = supervisor_mode_select(
            ici_state=ici_state_live,
            mode_b_conditions_met=mode_b_entry_cond,
            mode_c_active=False,
            degradation_flag=False,
            t_k_eff_below_threshold=t_k_eff_below_threshold,
        )
        allow_mode_b_step = allow_mode_b and (supervisor_decision != "mode_c")
        if not allow_mode_b_step:
            q_hat = 1.0
            obs_ctl["q_hat"] = q_hat
            obs_ctl["entrenchment"] = False
        u, meta = choose_policy_action(
            policy_name, obs_ctl, eval_model, config,
            used_burden=used_burden, policy_cache=policy_cache, with_tau=with_tau, with_coherence=with_coherence
        )
        if meta["mode_b_triggered"]:
            mode_b_entries += 1
        if "safety_fallback" in str(meta["controller_notes"]):
            fallback_count += 1
        obs_next, info = env.step(u)
        used_burden += float(np.sum(np.abs(u)))
        low = obs["target"].box_low
        high = obs["target"].box_high
        dist_true = _dist_to_box(obs["x_true"], low, high)
        dist_est = _dist_to_box(x_hat, low, high)
        in_target = dist_true <= 1e-8
        y_lo, y_hi = observation_intervals(config)
        y_filled = np.nan_to_num(obs["y"], nan=0.0)
        violation = bool(np.any((y_filled < y_lo) | (y_filled > y_hi)))
        allowed = circadian_allowed_mask(t, eval_model.control_dim, int(config["steps_per_day"]), list(config.get("circadian_locked_controls", [])))
        illegal_nonzero = np.abs(u) > 1e-8
        circ_ok = 1.0 if np.sum(illegal_nonzero) == 0 else float(1.0 - np.sum(illegal_nonzero & (~allowed)) / np.sum(illegal_nonzero))
        # Evaluation ALWAYS uses full HDR cost (eq.13) so all policies are compared on
        # the same metric, regardless of which cost terms the controller itself minimised.
        denom = max(1 - eval_model.basins[map_mode].rho**2, 1e-6)
        stage_cost = (
            float(config["w1"]) * dist_est
            + float(config["w2"]) * dist_est / denom
            + float(config["w3"]) * coherence_penalty(kappa_hat, float(config["kappa_lo"]), float(config["kappa_hi"]))
            + float(config["lambda_u"]) * float(u @ u)
        )
        per["t"].append(t)
        per["z_true"].append(obs["z_true"])
        per["map_mode"].append(map_mode)
        per["posterior_maladaptive"].append(float(mode_probs[1]) if len(mode_probs) > 1 else 0.0)
        per["dist_true"].append(dist_true)
        per["dist_est"].append(dist_est)
        per["time_in_target"].append(float(in_target))
        per["risk"].append(float(meta.get("risk", 0.0)))
        per["safety_violation"].append(float(violation))
        per["burden_used"].append(used_burden)
        per["solve_time"].append(float(meta["solve_time"]))
        per["rmse_state"].append(float(np.sqrt(np.mean((obs["x_true"] - x_hat)**2))))
        per["mode_b_triggered"].append(float(meta["mode_b_triggered"]))
        per["mode_b_z_at_trigger"].append(float(obs["z_true"]) if meta["mode_b_triggered"] else float("nan"))
        per["q_hat"].append(q_hat)
        per["kappa_hat"].append(kappa_hat)
        per["kappa_true"].append(float(obs["kappa_true"]))
        per["circadian_adherence_step"].append(circ_ok)
        per["feasible"].append(float(meta.get("feasible", True)))
        per["stage_cost"].append(stage_cost)
        per["allowed_nonzero"].append(float(np.sum(np.abs(u[allowed]) > 1e-8)))
        per["mode_correct"].append(float(map_mode == obs["z_true"]))
        obs = obs_next
        if info["done"]:
            break
    per_np = {k: np.asarray(v) for k, v in per.items()}
    challenge_times = []
    for _, idx in env.challenge_markers.items():
        if idx < len(per_np["dist_true"]):
            challenge_times.append(recovery_time_from_challenge(per_np["dist_true"], idx))
    summary = {
        "policy": policy_name,
        "scenario": env.scenario.name,
        "cum_cost": float(np.sum(per_np["stage_cost"])),
        "time_in_target": float(np.mean(per_np["time_in_target"])),
        "safety_violation_rate": float(np.mean(per_np["safety_violation"])),
        "burden_adherence": float(used_burden <= float(config["default_burden_budget"])),
        "circadian_adherence": float(np.mean(per_np["circadian_adherence_step"])),
        "challenge_recovery_time": float(np.mean(challenge_times)) if challenge_times else float("nan"),
        "recursive_feasibility_rate": float(np.mean(per_np["feasible"])),
        "controller_solve_time_mean": float(np.mean(per_np["solve_time"])),
        "state_rmse": float(np.mean(per_np["rmse_state"])),
        "mode_accuracy": float(np.mean(per_np["mode_correct"])),
        "mode_b_entry_count": int(mode_b_entries),
        "fallback_count": int(fallback_count),
        "episode_length": int(len(per_np["t"])),
    }
    return ClosedLoopOutputs(per_step=per_np, episode_summary=summary)
