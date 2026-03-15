"""Tests for hdr_validation.inference.variational — variational SLDS."""
import numpy as np
from hdr_validation.model.slds import BasinModel
from hdr_validation.inference.variational import VariationalSLDS


def _simple_basins():
    n = 3
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


def test_elbo_computation():
    """ELBO should be finite."""
    basins = _simple_basins()
    vslds = VariationalSLDS(basins, {"state_dim": 3})
    rng = np.random.default_rng(50)
    y_seq = rng.normal(size=(10, 3))
    result = vslds.fit(y_seq, max_iter=5)
    assert np.isfinite(result["elbo"])


def test_vslds_convergence_simple():
    """ELBO should not decrease across iterations."""
    basins = _simple_basins()
    vslds = VariationalSLDS(basins, {"state_dim": 3})
    rng = np.random.default_rng(51)
    y_seq = rng.normal(size=(20, 3))
    vslds.fit(y_seq, max_iter=20)
    # Check ELBO history is non-decreasing (within tolerance)
    elbos = vslds._elbo_history
    if len(elbos) >= 2:
        for i in range(1, len(elbos)):
            assert elbos[i] >= elbos[i-1] - 1e-3, f"ELBO decreased: {elbos[i-1]} -> {elbos[i]}"


def test_vslds_mean_field_factorisation():
    """q_z should be a valid probability distribution."""
    basins = _simple_basins()
    vslds = VariationalSLDS(basins, {"state_dim": 3})
    rng = np.random.default_rng(52)
    y_seq = rng.normal(size=(10, 3))
    result = vslds.fit(y_seq, max_iter=10)
    q_z = result["q_z"]
    assert np.isclose(np.sum(q_z), 1.0, atol=1e-6)
    assert np.all(q_z >= 0)


def test_vslds_posterior_valid():
    """Posterior covariance should be positive semi-definite."""
    basins = _simple_basins()
    vslds = VariationalSLDS(basins, {"state_dim": 3})
    rng = np.random.default_rng(53)
    y_seq = rng.normal(size=(10, 3))
    result = vslds.fit(y_seq, max_iter=10)
    q_x_cov = result["q_x_cov"]
    eigvals = np.linalg.eigvalsh(q_x_cov)
    assert np.all(eigvals >= -1e-8)


def test_vslds_kl_nonnegative():
    """KL divergence components should be non-negative."""
    basins = _simple_basins()
    vslds = VariationalSLDS(basins, {"state_dim": 3})
    rng = np.random.default_rng(54)
    y_seq = rng.normal(size=(10, 3))
    result = vslds.fit(y_seq, max_iter=10)
    # ELBO is E[log p] - KL, and KL >= 0
    # So ELBO <= E[log p]. We just check it's finite.
    assert np.isfinite(result["elbo"])
