"""Tests for hdr_validation.inference.particle — particle filter."""
import numpy as np
from hdr_validation.model.slds import make_evaluation_model, BasinModel
from hdr_validation.inference.particle import ParticleFilter


def _simple_basins():
    n = 4
    basins = []
    for rho in [0.5, 0.8]:
        A = rho * np.eye(n)
        B = np.eye(n, 2)
        C = np.eye(n)
        Q = np.eye(n) * 0.1
        R = np.eye(n) * 0.1
        basins.append(BasinModel(A=A, B=B, C=C, Q=Q, R=R,
                                  b=np.zeros(n), c=np.zeros(n), E=np.eye(n), rho=rho))
    return basins


def test_pf_weight_normalisation():
    basins = _simple_basins()
    pf = ParticleFilter(50, basins)
    assert np.isclose(np.sum(pf.weights), 1.0)
    y = np.zeros(4)
    pf.update(y)
    assert np.isclose(np.sum(pf.weights), 1.0, atol=1e-6)


def test_pf_systematic_resampling():
    basins = _simple_basins()
    pf = ParticleFilter(100, basins)
    # Make weights very uneven
    pf.weights = np.zeros(100)
    pf.weights[0] = 1.0
    pf.resample()
    # After resampling, all particles should be copies of particle 0
    assert np.isclose(np.sum(pf.weights), 1.0, atol=1e-6)
    assert np.allclose(pf.weights, 1.0 / 100)


def test_pf_ess_computation():
    basins = _simple_basins()
    pf = ParticleFilter(100, basins)
    # Uniform weights -> ESS = N
    ess = pf.ess()
    assert np.isclose(ess, 100.0, atol=0.1)
    # One dominant weight -> ESS ~ 1
    pf.weights = np.zeros(100)
    pf.weights[0] = 1.0
    ess = pf.ess()
    assert ess < 2.0


def test_pf_convergence_simple_model():
    """PF should track state in a simple linear-Gaussian model."""
    basins = _simple_basins()
    pf = ParticleFilter(200, basins[:1])  # single basin
    rng = np.random.default_rng(100)
    x_true = np.zeros(4)
    for t in range(20):
        u = np.zeros(2)
        x_true = basins[0].A @ x_true + rng.normal(scale=0.1, size=4)
        y = basins[0].C @ x_true + rng.normal(scale=0.1, size=4)
        pf.predict(u)
        pf.update(y)
        if pf.ess() < 50:
            pf.resample()
    # Weighted mean should be in right ballpark
    x_est = np.average(pf.particles, weights=pf.weights, axis=0)
    assert np.linalg.norm(x_est - x_true) < 5.0


def test_pf_population_proposal():
    """ParticleFilter with proposal_inflation > 1 should still have valid weights."""
    basins = _simple_basins()
    pf = ParticleFilter(50, basins, proposal_inflation=2.0)
    pf.predict(np.zeros(2))
    y = np.zeros(4)
    pf.update(y)
    assert np.isclose(np.sum(pf.weights), 1.0, atol=1e-6)


def test_pf_degeneracy_detection():
    """ESS should drop when one particle dominates."""
    basins = _simple_basins()
    pf = ParticleFilter(50, basins)
    pf.weights = np.zeros(50)
    pf.weights[0] = 0.99
    pf.weights[1:] = 0.01 / 49
    assert pf.ess() < 5.0
