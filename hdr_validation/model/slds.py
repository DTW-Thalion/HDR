from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import qr

from .hsmm import DwellModel


@dataclass
class BasinModel:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    b: np.ndarray
    c: np.ndarray
    E: np.ndarray
    rho: float
    stability_class: str = "stable"  # 'stable' or 'unstable' (v7.0)
    rev_irr_partition: Any = None
    pwa_regions: Any = None
    jump_params: Any = None
    multisite_coupling: Any = None
    adaptive_params: Any = None
    cumulative_exposure_params: Any = None


@dataclass
class EvaluationModel:
    basins: list[BasinModel]
    transition: np.ndarray
    dwell_models: list[DwellModel]
    state_dim: int
    obs_dim: int
    control_dim: int
    disturbance_dim: int


def spectral_radius(A: np.ndarray) -> float:
    vals = np.linalg.eigvals(A)
    return float(np.max(np.abs(vals)))


def random_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    H = rng.normal(size=(n, n))
    Q, _ = qr(H)
    return Q


def make_structured_matrix(rho: float, n: int, rng: np.random.Generator, coupling_scale: float = 0.06) -> np.ndarray:
    base = np.linspace(rho * 0.55, rho, n)
    D = np.diag(base)
    U = random_orthogonal(n, rng)
    A = U @ D @ U.T
    sparsity_mask = rng.uniform(size=(n, n)) < 0.15
    off = rng.normal(scale=coupling_scale, size=(n, n)) * sparsity_mask
    np.fill_diagonal(off, 0.0)
    A = A + off
    vals = np.linalg.eigvals(A)
    max_abs = np.max(np.abs(vals))
    if max_abs > 1e-12:
        A = A * (rho / max_abs)
    return A.real


def make_observation_matrix(obs_dim: int, state_dim: int, rng: np.random.Generator) -> np.ndarray:
    C = np.zeros((obs_dim, state_dim))
    for axis in range(state_dim):
        row0 = 2 * axis
        row1 = 2 * axis + 1
        if row1 < obs_dim:
            C[row0, axis] = 1.0 + rng.normal(scale=0.05)
            C[row1, axis] = 0.7 + rng.normal(scale=0.05)
            neighbor = (axis + 1) % state_dim
            C[row1, neighbor] = 0.15 + rng.normal(scale=0.02)
    return C


def make_evaluation_model(config: dict[str, Any], rng: np.random.Generator, K: int | None = None) -> EvaluationModel:
    n = int(config["state_dim"])
    m = int(config["obs_dim"])
    u_dim = int(config["control_dim"])
    d_dim = int(config["disturbance_dim"])
    K = int(K or config["K"])
    rhos = list(config["rho_reference"])
    if K == 4:
        rhos = rhos + [0.9]
    basins = []
    for k in range(K):
        rho = rhos[k]
        A = make_structured_matrix(rho, n, rng, coupling_scale=0.04 + 0.02 * (k == 1))
        B = np.eye(n, u_dim) * (0.18 if k == 0 else 0.12 if k == 1 else 0.16)
        if K == 4 and k == 3:
            B *= 0.14
        E = np.eye(n, d_dim)
        C = make_observation_matrix(m, n, rng)
        q_scale = 0.04 if k == 0 else 0.07 if k == 1 else 0.05
        r_scale = 0.06 if k == 0 else 0.08 if k == 1 else 0.07
        Q = np.eye(n) * q_scale
        R = np.eye(m) * r_scale
        b = np.zeros(n)
        if k == 1:
            b[[0, 1, 3, 4]] = 0.03
        elif k == 2:
            b[[0, 1, 6]] = 0.01
        elif k == 3:
            b[[0, 1, 4, 6]] = 0.02
        c = np.zeros(m)
        basins.append(BasinModel(A=A, B=B, C=C, Q=Q, R=R, b=b, c=c, E=E, rho=spectral_radius(A)))
    if K == 3:
        transition = np.array([
            [0.85, 0.04, 0.11],
            [0.05, 0.87, 0.08],
            [0.55, 0.18, 0.27],
        ], dtype=float)
        dwell_models = [
            DwellModel("poisson", {"mean": 10.0}, max_len=config["max_dwell_len"]),
            DwellModel("zipf", {"a": 1.8}, max_len=config["max_dwell_len"]),
            DwellModel("poisson", {"mean": 5.0}, max_len=config["max_dwell_len"]),
        ]
    else:
        transition = np.array([
            [0.83, 0.03, 0.11, 0.03],
            [0.03, 0.84, 0.07, 0.06],
            [0.52, 0.16, 0.22, 0.10],
            [0.10, 0.35, 0.15, 0.40],
        ], dtype=float)
        dwell_models = [
            DwellModel("poisson", {"mean": 10.0}, max_len=config["max_dwell_len"]),
            DwellModel("zipf", {"a": 1.8}, max_len=config["max_dwell_len"]),
            DwellModel("poisson", {"mean": 5.0}, max_len=config["max_dwell_len"]),
            DwellModel("lognormal", {"mu": 2.4, "sigma": 0.55}, max_len=config["max_dwell_len"]),
        ]
    # v7.0: classify basin stability and set stability_class
    for basin in basins:
        basin.stability_class = "stable" if basin.rho < 1.0 else "unstable"

    return EvaluationModel(
        basins=basins,
        transition=transition / transition.sum(axis=1, keepdims=True),
        dwell_models=dwell_models,
        state_dim=n,
        obs_dim=m,
        control_dim=u_dim,
        disturbance_dim=d_dim,
    )


def make_extended_evaluation_model(
    config: dict[str, Any],
    rng: np.random.Generator,
    extensions: dict[str, Any] | None = None,
    K: int | None = None,
) -> EvaluationModel:
    """Create evaluation model with v7.0 extensions.

    When extensions is None or empty, produces identical output to
    make_evaluation_model (Prop 9.2 backward compatibility).

    Parameters
    ----------
    config : configuration dict
    rng : random generator
    extensions : dict of active extensions (e.g. {'rev_irr': True, 'pwa': True})
    K : override number of basins
    """
    model = make_evaluation_model(config, rng, K=K)

    if not extensions:
        return model

    # Apply extensions to each basin
    for k, basin in enumerate(model.basins):
        if extensions.get("rev_irr"):
            n_irr = int(config.get("n_irr", 2))
            basin.rev_irr_partition = {"n_r": model.state_dim - n_irr, "n_i": n_irr}

        if extensions.get("pwa"):
            basin.pwa_regions = {"R_k": int(config.get("R_k_regions", 2))}

        if extensions.get("jump"):
            basin.jump_params = {
                "lambda_cat_max": float(config.get("lambda_cat_max", 0.05)),
                "dt": float(config.get("dt_minutes", 30)) / 60.0,
            }

        if extensions.get("adaptive"):
            basin.adaptive_params = {
                "drift_rate": float(config.get("drift_rate", 0.001)),
                "lambda_ff": float(config.get("lambda_ff", 0.98)),
            }

        if extensions.get("cumulative_exposure"):
            basin.cumulative_exposure_params = {
                "n_channels": int(config.get("n_cum_exp", 1)),
                "xi_max": float(config.get("xi_max", 100.0)),
            }

    return model


def pooled_basin(eval_model: EvaluationModel) -> BasinModel:
    A = np.mean([b.A for b in eval_model.basins], axis=0)
    B = np.mean([b.B for b in eval_model.basins], axis=0)
    C = np.mean([b.C for b in eval_model.basins], axis=0)
    Q = np.mean([b.Q for b in eval_model.basins], axis=0)
    R = np.mean([b.R for b in eval_model.basins], axis=0)
    b = np.mean([b.b for b in eval_model.basins], axis=0)
    c = np.mean([b.c for b in eval_model.basins], axis=0)
    E = np.mean([b.E for b in eval_model.basins], axis=0)
    rho = spectral_radius(A)
    return BasinModel(A=A, B=B, C=C, Q=Q, R=R, b=b, c=c, E=E, rho=rho)
