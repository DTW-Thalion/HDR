from __future__ import annotations

import numpy as np
import scipy.linalg as la

from .slds import spectral_radius


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ---------------------------------------------------------------------------
# 1. BasinClassifier
# ---------------------------------------------------------------------------

class BasinClassifier:
    """Classifies basins as stable (spectral_radius < 1) or unstable (>= 1)."""

    def classify(self, basins) -> dict[str, list[int]]:
        K_s: list[int] = []
        K_u: list[int] = []
        for idx, basin in enumerate(basins):
            rho = spectral_radius(basin.A)
            if rho < 1.0:
                K_s.append(idx)
            else:
                K_u.append(idx)
        return {"K_s": K_s, "K_u": K_u}


# ---------------------------------------------------------------------------
# 2. ReversibleIrreversiblePartition
# ---------------------------------------------------------------------------

class ReversibleIrreversiblePartition:
    """Partitions state into reversible (n_r) and irreversible (n_i) components."""

    def __init__(self, n_r: int, n_i: int, config: dict):
        self.n_r = n_r
        self.n_i = n_i
        self.config = config
        alpha_val = config.get("rev_irr_alpha", 0.1)
        # Per-basin coupling scalar (stored as a default; indexed by basin)
        self._alpha = alpha_val

    def phi_k(self, x_rev: np.ndarray, x_irr: np.ndarray, basin_idx: int) -> np.ndarray:
        """Damage acceleration function.

        Properties:
          - phi_k(0, x_irr) = 0          (no progression at reference)
          - phi_k >= 0                    (non-negative)
          - monotonic in ||x_rev||
        """
        alpha = self._alpha
        x_irr_clipped = np.maximum(x_irr, 0.0)
        phi = alpha * np.linalg.norm(x_rev) * x_irr_clipped
        return phi

    def step(self, x_rev: np.ndarray, x_irr: np.ndarray, u: np.ndarray,
             basin, rng: np.random.Generator):
        """Advance one step for the reversible/irreversible partition."""
        n_r = self.n_r
        A_rr = basin.A[:n_r, :n_r]
        B_r = basin.B[:n_r, :]
        E_rr = basin.E[:n_r, :n_r]

        x_rev_new = A_rr @ x_rev + B_r @ u + E_rr @ rng.normal(size=n_r)

        dt_hours = self.config.get("dt_minutes", 30) / 60.0
        phi = self.phi_k(x_rev, x_irr, 0)
        noise_scale = self.config.get("irr_noise_scale", 0.01)
        x_irr_new = (x_irr
                     + phi * dt_hours
                     + noise_scale * rng.normal(size=x_irr.shape))

        return x_rev_new, x_irr_new


# ---------------------------------------------------------------------------
# 3. PWACoupling
# ---------------------------------------------------------------------------

class PWACoupling:
    """Piecewise-affine coupling with polyhedral regions."""

    def __init__(self, thresholds: dict, regions_per_basin: int):
        self.threshold_values = np.asarray(thresholds["values"], dtype=float)
        self.regions_per_basin = regions_per_basin

    def get_region(self, x: np.ndarray, basin_idx: int) -> int:
        """Return region index based on which threshold interval x[0] falls in."""
        val = float(x[0])
        region = int(np.searchsorted(self.threshold_values, val))
        return min(region, self.regions_per_basin - 1)

    def get_dynamics(self, x: np.ndarray, basin_idx: int,
                     dt: float = 1.0, damping: float = 0.1,
                     coupling_scale: float = 0.05):
        """Return (A_kr, b_kr) for the appropriate region.

        A_kr = I + dt * (-D + J^{k,r})  where D is damping, J is coupling.
        """
        n = x.shape[0]
        region = self.get_region(x, basin_idx)

        D = damping * np.eye(n)
        # Coupling varies by basin and region
        rng_local = np.random.default_rng(basin_idx * 1000 + region)
        J_kr = coupling_scale * rng_local.standard_normal((n, n))
        J_kr = 0.5 * (J_kr - J_kr.T)  # antisymmetric for stability

        A_kr = np.eye(n) + dt * (-D + J_kr)
        b_kr = np.zeros(n)
        return A_kr, b_kr

    def check_common_lyapunov(self, P: np.ndarray, Q: np.ndarray,
                              dynamics_list: list[np.ndarray]) -> bool:
        """Check A_kr^T P A_kr - P <= -Q for all k, r.

        Parameters
        ----------
        P : (n, n) candidate Lyapunov matrix
        Q : (n, n) positive-definite decay matrix
        dynamics_list : list of A_kr matrices for all (k, r)

        Returns True if all eigenvalues of (A_kr^T P A_kr - P + Q) are <= 0.
        """
        for A_kr in dynamics_list:
            diff = A_kr.T @ P @ A_kr - P + Q
            eigvals = np.linalg.eigvalsh(diff)
            if np.any(eigvals > 1e-10):
                return False
        return True


# ---------------------------------------------------------------------------
# 4. MultiSiteModel
# ---------------------------------------------------------------------------

class MultiSiteModel:
    """Multi-site HDR with S sites."""

    def __init__(self, sites: list[dict], coupling: np.ndarray):
        """
        Parameters
        ----------
        sites : list of dicts with keys 'A' (ndarray) and 'rho' (float)
        coupling : (S, S) inter-site coupling matrix G
        """
        self.sites = sites
        self.coupling = np.asarray(coupling, dtype=float)
        self.S = len(sites)
        self.epsilon_G = 1.0  # coupling strength, can be set externally

    def composite_dynamics(self) -> np.ndarray:
        """Build block matrix with A_k^(s) on diagonal and epsilon_G * G_{ss'} * I off-diagonal."""
        dims = [site["A"].shape[0] for site in self.sites]
        total = sum(dims)
        M = np.zeros((total, total))

        offsets = np.cumsum([0] + dims)
        for s in range(self.S):
            i0, i1 = offsets[s], offsets[s + 1]
            M[i0:i1, i0:i1] = self.sites[s]["A"]
            n_s = dims[s]
            for sp in range(self.S):
                if sp == s:
                    continue
                j0, j1 = offsets[sp], offsets[sp + 1]
                n_sp = dims[sp]
                block_dim = min(n_s, n_sp)
                M[i0:i0 + block_dim, j0:j0 + block_dim] = (
                    self.epsilon_G * self.coupling[s, sp] * np.eye(block_dim)
                )
        return M

    def check_gershgorin_bound(self) -> bool:
        """Check epsilon_G < min_s(1 - rho(A_k^(s))) / (S - 1)."""
        if self.S <= 1:
            return True
        min_margin = min(1.0 - site["rho"] for site in self.sites)
        bound = min_margin / (self.S - 1)
        return self.epsilon_G < bound


# ---------------------------------------------------------------------------
# 5. JumpDiffusion
# ---------------------------------------------------------------------------

class JumpDiffusion:
    """Jump-augmented SLDS."""

    def __init__(self, lambda_cat_fn, jump_dist: dict, config: dict):
        """
        Parameters
        ----------
        lambda_cat_fn : callable(x, z) -> float, catastrophic jump rate
        jump_dist : dict with 'scale' key
        config : dict with 'dt_minutes' etc.
        """
        self.lambda_cat_fn = lambda_cat_fn
        self.jump_scale = jump_dist["scale"]
        self.dt = config.get("dt_minutes", 30) / 60.0  # hours

    def sample_jump(self, x: np.ndarray, z: int,
                    rng: np.random.Generator) -> tuple[bool, np.ndarray]:
        """Sample a catastrophic jump event.

        P(J=1) = 1 - exp(-lambda_cat * dt).
        If jump occurs: eta ~ N(0, scale * I).
        """
        lam = self.lambda_cat_fn(x, z)
        p_jump = 1.0 - np.exp(-lam * self.dt)
        jumped = bool(rng.random() < p_jump)
        if jumped:
            eta = rng.normal(scale=self.jump_scale, size=x.shape)
        else:
            eta = np.zeros_like(x)
        return jumped, eta

    def composite_transition(self, P_smooth: np.ndarray, P_cat: np.ndarray,
                             p_cat: float) -> np.ndarray:
        """Composite transition: (1 - p_cat) * P_smooth + p_cat * P_cat."""
        return (1.0 - p_cat) * P_smooth + p_cat * P_cat


# ---------------------------------------------------------------------------
# 6. CumulativeExposure
# ---------------------------------------------------------------------------

class CumulativeExposure:
    """Cumulative-exposure state augmentation."""

    def __init__(self, n_channels: int, f_j, xi_max: np.ndarray,
                 xi_prior: np.ndarray | None = None):
        """
        Parameters
        ----------
        n_channels : number of exposure channels
        f_j : callable(u) -> ndarray, maps control to exposure increments
        xi_max : (n_channels,) upper bound per channel
        xi_prior : optional prior for exposure state
        """
        self.n_channels = n_channels
        self.f_j = f_j
        self.xi_max = np.asarray(xi_max, dtype=float)
        self.xi_prior = (np.zeros(n_channels) if xi_prior is None
                         else np.asarray(xi_prior, dtype=float))

    def update(self, xi: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Monotonic update: xi_new = max(xi + f_j(u), 0)."""
        return np.maximum(xi + self.f_j(u), 0.0)

    def check_constraint(self, xi: np.ndarray) -> bool:
        """Check all exposure channels within limits."""
        return bool(np.all(xi <= self.xi_max))


# ---------------------------------------------------------------------------
# 7. StateConditionedCoupling
# ---------------------------------------------------------------------------

class StateConditionedCoupling:
    """State-conditioned coupling with sigmoid gating."""

    def __init__(self, J0: np.ndarray,
                 perturbations: list[tuple[float, np.ndarray]],
                 thresholds: list[tuple[np.ndarray, float]],
                 config: dict):
        """
        Parameters
        ----------
        J0 : (n, n) base coupling matrix
        perturbations : list of (alpha_p, dJ_p) tuples
        thresholds : list of (c_p, theta_p) tuples
        config : configuration dict
        """
        self.J0 = np.asarray(J0, dtype=float)
        self.perturbations = [(float(a), np.asarray(dJ, dtype=float))
                              for a, dJ in perturbations]
        self.thresholds = [(np.asarray(c, dtype=float), float(theta))
                           for c, theta in thresholds]
        self.config = config

    def coupling_at(self, x: np.ndarray, basin_idx: int) -> np.ndarray:
        """J = J0 + sum_p alpha_p * sigmoid(c_p^T x - theta_p) * dJ_p."""
        J = self.J0.copy()
        for (alpha_p, dJ_p), (c_p, theta_p) in zip(
                self.perturbations, self.thresholds):
            gate = _sigmoid(c_p @ x - theta_p)
            J = J + alpha_p * gate * dJ_p
        return J

    def delta_A_eff(self, Delta_A: float, P_count: float,
                    dt: float) -> float:
        """Effective dynamics perturbation bound.

        Returns Delta_A + P_count * delta_J_bar * dt
        where delta_J_bar = sum of alpha_p * ||dJ_p||.
        """
        delta_J_bar = sum(
            alpha_p * np.linalg.norm(dJ_p)
            for alpha_p, dJ_p in self.perturbations
        )
        return Delta_A + P_count * delta_J_bar * dt


# ---------------------------------------------------------------------------
# 8. ModularExpansion
# ---------------------------------------------------------------------------

class ModularExpansion:
    """Modular axis expansion for adding new state dimensions."""

    def __init__(self, A_k: np.ndarray, A_new: np.ndarray,
                 J_cross_1: np.ndarray, J_cross_2: np.ndarray):
        """
        Parameters
        ----------
        A_k : (n_old, n_old) existing dynamics
        A_new : (n_new, n_new) new module dynamics
        J_cross_1 : (n_new, n_old) new <- old coupling
        J_cross_2 : (n_old, n_new) old <- new coupling
        """
        self.A_k = np.asarray(A_k, dtype=float)
        self.A_new = np.asarray(A_new, dtype=float)
        self.J_cross_1 = np.asarray(J_cross_1, dtype=float)
        self.J_cross_2 = np.asarray(J_cross_2, dtype=float)

    def expanded_dynamics(self) -> np.ndarray:
        """Return block matrix [[A_k, J_cross_2], [J_cross_1, A_new]]."""
        top = np.hstack([self.A_k, self.J_cross_2])
        bottom = np.hstack([self.J_cross_1, self.A_new])
        return np.vstack([top, bottom])

    def check_expansion_bound(self) -> bool:
        """Check spectral norm product of cross-couplings < (1-rho(A_k))*(1-rho(A_new))."""
        sigma_1 = np.linalg.norm(self.J_cross_1, ord=2)
        sigma_2 = np.linalg.norm(self.J_cross_2, ord=2)
        rho_k = spectral_radius(self.A_k)
        rho_new = spectral_radius(self.A_new)
        margin = (1.0 - rho_k) * (1.0 - rho_new)
        return float(sigma_1 * sigma_2) < margin
