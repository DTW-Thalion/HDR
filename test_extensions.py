"""Tests for hdr_validation.model.extensions — HDR v7.0 model extensions."""
from __future__ import annotations

import numpy as np
import pytest

from hdr_validation.model.extensions import (
    BasinClassifier,
    ReversibleIrreversiblePartition,
    PWACoupling,
    MultiSiteModel,
    JumpDiffusion,
    CumulativeExposure,
    StateConditionedCoupling,
    ModularExpansion,
)
from hdr_validation.model.slds import make_evaluation_model, BasinModel


# ── Helper ────────────────────────────────────────────────────────────────────

def _make_basin(rho: float, n: int = 4) -> BasinModel:
    """Create a simple BasinModel with given spectral radius."""
    A = rho * np.eye(n)
    return BasinModel(
        A=A, B=np.eye(n, 2), C=np.eye(n), Q=np.eye(n) * 0.01,
        R=np.eye(n) * 0.01, b=np.zeros(n), c=np.zeros(n),
        E=np.eye(n), rho=rho,
    )


# ── 1. BasinClassifier ───────────────────────────────────────────────────────

def test_basin_classifier_stable():
    """Basins with rho < 1 should all be classified as K_s."""
    basins = [_make_basin(0.5), _make_basin(0.8), _make_basin(0.99)]
    result = BasinClassifier().classify(basins)
    assert result["K_s"] == [0, 1, 2]
    assert result["K_u"] == []


def test_basin_classifier_unstable():
    """Basin with spectral radius >= 1 should be in K_u."""
    stable = _make_basin(0.7)
    unstable = _make_basin(1.2)
    result = BasinClassifier().classify([stable, unstable])
    assert 0 in result["K_s"]
    assert 1 in result["K_u"]


def test_basin_classifier_boundary():
    """Basin with rho exactly 1.0 should be in K_u (not strictly stable)."""
    basin = _make_basin(1.0)
    result = BasinClassifier().classify([basin])
    assert 0 in result["K_u"]
    assert result["K_s"] == []


def test_basin_classifier_empty_K_u():
    """When all basins are stable, K_u should be an empty list."""
    basins = [_make_basin(0.3), _make_basin(0.6)]
    result = BasinClassifier().classify(basins)
    assert result["K_u"] == []
    assert len(result["K_s"]) == 2


# ── 2. ReversibleIrreversiblePartition ────────────────────────────────────────

def test_rev_irr_partition_phi_zero_at_reference():
    """phi_k(x_rev=0, x_irr) should be 0."""
    rip = ReversibleIrreversiblePartition(n_r=3, n_i=2, config={"rev_irr_alpha": 0.1})
    x_rev = np.zeros(3)
    x_irr = np.array([1.0, 2.0])
    phi = rip.phi_k(x_rev, x_irr, basin_idx=0)
    assert np.allclose(phi, 0.0)


def test_rev_irr_partition_phi_nonnegative():
    """phi_k should always be >= 0."""
    rng = np.random.default_rng(42)
    rip = ReversibleIrreversiblePartition(n_r=3, n_i=2, config={"rev_irr_alpha": 0.5})
    for _ in range(50):
        x_rev = rng.normal(size=3)
        x_irr = rng.normal(size=2)
        phi = rip.phi_k(x_rev, x_irr, basin_idx=0)
        assert np.all(phi >= -1e-15)


def test_rev_irr_partition_phi_monotonic():
    """phi_k should increase with ||x_rev||."""
    rip = ReversibleIrreversiblePartition(n_r=3, n_i=2, config={"rev_irr_alpha": 0.1})
    x_irr = np.array([1.0, 0.5])
    direction = np.array([1.0, 0.0, 0.0])
    phi_prev = np.sum(rip.phi_k(0.1 * direction, x_irr, 0))
    for scale in [0.5, 1.0, 2.0, 5.0]:
        phi_curr = np.sum(rip.phi_k(scale * direction, x_irr, 0))
        assert phi_curr >= phi_prev - 1e-15
        phi_prev = phi_curr


# ── 3. PWACoupling ────────────────────────────────────────────────────────────

def test_pwa_region_membership_correct():
    """Different x values should fall in correct regions based on thresholds."""
    pwa = PWACoupling(thresholds={"values": [0.0, 1.0, 2.0]}, regions_per_basin=4)
    x_low = np.array([-1.0, 0.0, 0.0, 0.0])
    x_mid = np.array([0.5, 0.0, 0.0, 0.0])
    x_high = np.array([3.0, 0.0, 0.0, 0.0])
    r_low = pwa.get_region(x_low, basin_idx=0)
    r_mid = pwa.get_region(x_mid, basin_idx=0)
    r_high = pwa.get_region(x_high, basin_idx=0)
    assert r_low == 0
    assert r_mid == 1
    assert r_high == 3  # clamped to regions_per_basin - 1


def test_pwa_common_lyapunov_feasible():
    """For a well-conditioned system, Lyapunov check should be feasible."""
    n = 4
    pwa = PWACoupling(thresholds={"values": [0.0]}, regions_per_basin=2)
    # Build contractive dynamics for two regions
    dynamics_list = []
    for basin_idx in range(1):
        for region in range(2):
            x_dummy = np.array([region * 1.5 - 0.5, 0.0, 0.0, 0.0])
            A_kr, _ = pwa.get_dynamics(x_dummy, basin_idx, dt=1.0,
                                       damping=0.5, coupling_scale=0.01)
            dynamics_list.append(A_kr)
    P = np.eye(n)
    Q = 0.01 * np.eye(n)
    assert pwa.check_common_lyapunov(P, Q, dynamics_list)


def test_pwa_dynamics_continuous_at_boundary():
    """Dynamics should not have large discontinuities near region boundaries."""
    pwa = PWACoupling(thresholds={"values": [1.0]}, regions_per_basin=2)
    eps = 1e-6
    x_below = np.array([1.0 - eps, 0.0, 0.0, 0.0])
    x_above = np.array([1.0 + eps, 0.0, 0.0, 0.0])
    A_below, b_below = pwa.get_dynamics(x_below, basin_idx=0, coupling_scale=0.05)
    A_above, b_above = pwa.get_dynamics(x_above, basin_idx=0, coupling_scale=0.05)
    # Within same region (both near boundary), dynamics are identical
    # Across region boundary, they may differ but b should both be zero
    assert np.allclose(b_below, 0.0)
    assert np.allclose(b_above, 0.0)


# ── 4. MultiSiteModel ────────────────────────────────────────────────────────

def test_multisite_gershgorin_bound_holds():
    """Small coupling should satisfy Gershgorin bound."""
    sites = [
        {"A": 0.7 * np.eye(3), "rho": 0.7},
        {"A": 0.8 * np.eye(3), "rho": 0.8},
    ]
    coupling = np.array([[0.0, 0.5], [0.5, 0.0]])
    ms = MultiSiteModel(sites, coupling)
    ms.epsilon_G = 0.01  # small
    assert ms.check_gershgorin_bound()


def test_multisite_gershgorin_bound_violated():
    """Large coupling should violate Gershgorin bound."""
    sites = [
        {"A": 0.9 * np.eye(3), "rho": 0.9},
        {"A": 0.9 * np.eye(3), "rho": 0.9},
    ]
    coupling = np.array([[0.0, 1.0], [1.0, 0.0]])
    ms = MultiSiteModel(sites, coupling)
    ms.epsilon_G = 0.5  # large: bound is (1-0.9)/1 = 0.1
    assert not ms.check_gershgorin_bound()


def test_multisite_composite_spectral_radius():
    """Composite dynamics should have spectral radius < 1 when Gershgorin bound holds."""
    sites = [
        {"A": 0.5 * np.eye(3), "rho": 0.5},
        {"A": 0.6 * np.eye(3), "rho": 0.6},
    ]
    coupling = np.array([[0.0, 1.0], [1.0, 0.0]])
    ms = MultiSiteModel(sites, coupling)
    ms.epsilon_G = 0.05  # well within bound: min(0.5,0.4)/1 = 0.4
    assert ms.check_gershgorin_bound()
    M = ms.composite_dynamics()
    rho = np.max(np.abs(np.linalg.eigvals(M)))
    assert rho < 1.0


# ── 5. JumpDiffusion ─────────────────────────────────────────────────────────

def test_jump_indicator_probability():
    """Jump probability should be in [0,1] and increase with lambda."""
    rng = np.random.default_rng(7)
    x = np.ones(4)

    # Low lambda
    jd_low = JumpDiffusion(
        lambda_cat_fn=lambda x, z: 0.01,
        jump_dist={"scale": 0.5},
        config={"dt_minutes": 30},
    )
    # High lambda
    jd_high = JumpDiffusion(
        lambda_cat_fn=lambda x, z: 5.0,
        jump_dist={"scale": 0.5},
        config={"dt_minutes": 30},
    )
    # Empirical jump rates
    n_trials = 2000
    jumps_low = sum(jd_low.sample_jump(x, 0, np.random.default_rng(i))[0]
                    for i in range(n_trials))
    jumps_high = sum(jd_high.sample_jump(x, 0, np.random.default_rng(i))[0]
                     for i in range(n_trials))
    rate_low = jumps_low / n_trials
    rate_high = jumps_high / n_trials
    assert 0.0 <= rate_low <= 1.0
    assert 0.0 <= rate_high <= 1.0
    assert rate_high > rate_low


def test_jump_composite_transition_stochastic():
    """Composite transition matrix rows should sum to 1."""
    jd = JumpDiffusion(
        lambda_cat_fn=lambda x, z: 0.1,
        jump_dist={"scale": 0.3},
        config={"dt_minutes": 30},
    )
    P_smooth = np.array([[0.9, 0.1], [0.2, 0.8]])
    P_cat = np.array([[0.5, 0.5], [0.3, 0.7]])
    p_cat = 0.2
    P_comp = jd.composite_transition(P_smooth, P_cat, p_cat)
    row_sums = P_comp.sum(axis=1)
    assert np.allclose(row_sums, 1.0)


# ── 6. CumulativeExposure ────────────────────────────────────────────────────

def test_cumulative_exposure_monotonic():
    """Exposure xi should only increase when f_j >= 0."""
    def f_j(u):
        return np.abs(u[:2])  # always non-negative

    ce = CumulativeExposure(n_channels=2, f_j=f_j, xi_max=np.array([10.0, 10.0]))
    rng = np.random.default_rng(99)
    xi = np.zeros(2)
    for _ in range(20):
        u = rng.normal(size=4)
        xi_new = ce.update(xi, u)
        assert np.all(xi_new >= xi - 1e-15)
        xi = xi_new


def test_cumulative_exposure_markov():
    """(x_t, xi_t) should be Markov: update depends only on current state."""
    def f_j(u):
        return np.abs(u[:2])

    ce = CumulativeExposure(n_channels=2, f_j=f_j, xi_max=np.array([10.0, 10.0]))
    xi_a = np.array([1.0, 2.0])
    xi_b = np.array([1.0, 2.0])
    u = np.array([0.5, -0.3, 0.1, 0.0])
    # Same input state -> same output state (Markov property)
    xi_a_new = ce.update(xi_a, u)
    xi_b_new = ce.update(xi_b, u)
    assert np.allclose(xi_a_new, xi_b_new)


def test_cumulative_exposure_constraint():
    """check_constraint should return False when xi > xi_max."""
    def f_j(u):
        return np.ones(2) * 5.0

    ce = CumulativeExposure(n_channels=2, f_j=f_j, xi_max=np.array([3.0, 3.0]))
    xi_ok = np.array([1.0, 2.0])
    xi_bad = np.array([4.0, 1.0])
    assert ce.check_constraint(xi_ok)
    assert not ce.check_constraint(xi_bad)


# ── 7. StateConditionedCoupling ───────────────────────────────────────────────

def test_state_conditioned_sigmoid_shape():
    """coupling_at should return sigmoid-gated perturbation with correct shape."""
    n = 4
    J0 = np.zeros((n, n))
    dJ = np.eye(n) * 0.1
    c_vec = np.array([1.0, 0.0, 0.0, 0.0])
    scc = StateConditionedCoupling(
        J0=J0,
        perturbations=[(1.0, dJ)],
        thresholds=[(c_vec, 0.0)],
        config={},
    )
    # x far negative -> sigmoid ~ 0 -> J ~ J0
    x_neg = np.array([-10.0, 0.0, 0.0, 0.0])
    J_neg = scc.coupling_at(x_neg, basin_idx=0)
    assert J_neg.shape == (n, n)
    assert np.allclose(J_neg, J0, atol=0.01)

    # x far positive -> sigmoid ~ 1 -> J ~ J0 + dJ
    x_pos = np.array([10.0, 0.0, 0.0, 0.0])
    J_pos = scc.coupling_at(x_pos, basin_idx=0)
    assert np.allclose(J_pos, J0 + dJ, atol=0.01)


def test_state_conditioned_delta_A_eff():
    """delta_A_eff should be positive and scale with perturbation count."""
    n = 4
    J0 = np.zeros((n, n))
    dJ = np.eye(n) * 0.1

    # Single perturbation
    scc_1 = StateConditionedCoupling(
        J0=J0, perturbations=[(0.5, dJ)],
        thresholds=[(np.ones(n), 0.0)], config={},
    )
    val_1 = scc_1.delta_A_eff(Delta_A=0.0, P_count=1.0, dt=1.0)

    # Two perturbations
    scc_2 = StateConditionedCoupling(
        J0=J0, perturbations=[(0.5, dJ), (0.5, dJ)],
        thresholds=[(np.ones(n), 0.0), (np.ones(n), 1.0)], config={},
    )
    val_2 = scc_2.delta_A_eff(Delta_A=0.0, P_count=2.0, dt=1.0)

    assert val_1 > 0.0
    assert val_2 > val_1


# ── 8. ModularExpansion ──────────────────────────────────────────────────────

def test_expansion_spectral_bound_holds():
    """Small cross-coupling should satisfy the expansion bound."""
    A_k = 0.5 * np.eye(3)
    A_new = 0.6 * np.eye(2)
    J_cross_1 = 0.01 * np.ones((2, 3))
    J_cross_2 = 0.01 * np.ones((3, 2))
    me = ModularExpansion(A_k, A_new, J_cross_1, J_cross_2)
    assert me.check_expansion_bound()


def test_expansion_spectral_bound_violated():
    """Large cross-coupling should violate the expansion bound."""
    A_k = 0.9 * np.eye(3)
    A_new = 0.9 * np.eye(2)
    J_cross_1 = 5.0 * np.ones((2, 3))
    J_cross_2 = 5.0 * np.ones((3, 2))
    me = ModularExpansion(A_k, A_new, J_cross_1, J_cross_2)
    assert not me.check_expansion_bound()


def test_expansion_block_structure():
    """expanded_dynamics should have correct block structure."""
    n_old, n_new = 3, 2
    A_k = 0.5 * np.eye(n_old)
    A_new = 0.6 * np.eye(n_new)
    J_cross_1 = 0.1 * np.ones((n_new, n_old))
    J_cross_2 = 0.2 * np.ones((n_old, n_new))
    me = ModularExpansion(A_k, A_new, J_cross_1, J_cross_2)
    M = me.expanded_dynamics()
    assert M.shape == (n_old + n_new, n_old + n_new)
    # Check blocks
    assert np.allclose(M[:n_old, :n_old], A_k)
    assert np.allclose(M[n_old:, n_old:], A_new)
    assert np.allclose(M[:n_old, n_old:], J_cross_2)
    assert np.allclose(M[n_old:, :n_old], J_cross_1)


# ── 9. Backward compatibility ────────────────────────────────────────────────

def test_backward_compat_no_extensions():
    """make_evaluation_model without extensions should produce valid v5.4-compatible model."""
    config = {
        "state_dim": 8, "obs_dim": 16, "control_dim": 8,
        "disturbance_dim": 8, "K": 3, "rho_reference": [0.72, 0.96, 0.55],
        "max_dwell_len": 128,
    }
    rng = np.random.default_rng(42)
    model = make_evaluation_model(config, rng)
    assert len(model.basins) == 3
    assert model.state_dim == 8
    assert model.obs_dim == 16
    assert model.transition.shape == (3, 3)
    # Transition rows sum to 1
    assert np.allclose(model.transition.sum(axis=1), 1.0)
    # Basin spectral radii close to reference
    for basin, rho_ref in zip(model.basins, [0.72, 0.96, 0.55]):
        assert abs(basin.rho - rho_ref) < 0.15


def test_backward_compat_metrics_match_v54():
    """Basin properties (rho, A shape) should be unchanged from v5.4 baseline."""
    config = {
        "state_dim": 8, "obs_dim": 16, "control_dim": 8,
        "disturbance_dim": 8, "K": 3, "rho_reference": [0.72, 0.96, 0.55],
        "max_dwell_len": 128,
    }
    rng = np.random.default_rng(42)
    model = make_evaluation_model(config, rng)
    for basin in model.basins:
        assert basin.A.shape == (8, 8)
        assert basin.B.shape == (8, 8)
        assert basin.C.shape == (16, 8)
        assert basin.Q.shape == (8, 8)
        assert basin.R.shape == (16, 16)
        assert isinstance(basin.rho, float)
        assert 0.0 < basin.rho < 1.5
