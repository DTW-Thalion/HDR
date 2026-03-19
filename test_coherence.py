"""Tests for hdr_validation.model.coherence — damping ratio and coherence penalty."""
import numpy as np
from hdr_validation.model.coherence import damping_ratio, spectral_gap


# ── Damping ratio tests ──────────────────────────────────────────────────────

def test_damping_ratio_real_eigenvalues():
    """Diagonal matrix with all real eigenvalues: zeta = 1.0."""
    A = np.diag([0.9, 0.5, 0.3, 0.1])
    assert abs(damping_ratio(A) - 1.0) < 1e-10


def test_damping_ratio_identity():
    """Identity matrix: all eigenvalues = 1 (real), zeta = 1.0."""
    assert abs(damping_ratio(np.eye(4)) - 1.0) < 1e-10


def test_damping_ratio_complex_pair():
    """Rotation matrix has purely imaginary eigenvalues near the unit circle: zeta ≈ 0."""
    # 2D rotation by pi/4 has eigenvalues e^{±i*pi/4} = (1/sqrt2) ± i*(1/sqrt2)
    # |Re| / |lambda| = (1/sqrt2) / 1.0 = 0.707...
    theta = np.pi / 4
    A = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    zeta = damping_ratio(A)
    expected = np.cos(theta)  # |Re(e^{i*theta})| / |e^{i*theta}| = cos(theta)
    assert abs(zeta - expected) < 1e-10, f"Expected {expected:.6f}, got {zeta:.6f}"


def test_damping_ratio_underdamped_system():
    """Damped oscillator: zeta should be strictly between 0 and 1."""
    # Discrete-time damped oscillator: eigenvalues at r*e^{±i*theta}
    r = 0.95
    theta = 0.3  # radians
    A = np.array([[r * np.cos(theta), -r * np.sin(theta)],
                  [r * np.sin(theta),  r * np.cos(theta)]])
    zeta = damping_ratio(A)
    expected = abs(np.cos(theta))  # |Re| / |lambda| = |r*cos(theta)| / r
    assert abs(zeta - expected) < 1e-10
    assert 0 < zeta < 1


def test_damping_ratio_in_unit_interval():
    """Damping ratio is always in [0, 1] for random stable matrices."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        A = rng.normal(size=(8, 8)) * 0.3
        A = A + np.eye(8) * 0.6
        zeta = damping_ratio(A)
        assert 0.0 - 1e-10 <= zeta <= 1.0 + 1e-10, f"zeta={zeta} out of [0,1]"


def test_damping_ratio_aging_decline():
    """Verify that increasing antisymmetric coupling reduces zeta.

    This mirrors the ontology simulation finding: stronger coupling
    between axes produces complex eigenvalues (lower zeta), modelling
    the transition from overdamped to underdamped dynamics with age.
    Antisymmetric (skew) coupling is required to produce complex
    eigenvalues — symmetric coupling preserves real eigenvalues.
    """
    # Young: diagonal-dominant, weak antisymmetric coupling (overdamped)
    A_young = np.diag([0.95, 0.90, 0.85, 0.80])
    A_young += np.array([[0,  0.01, 0,     0],
                         [-0.01, 0, 0.01,  0],
                         [0, -0.01, 0,  0.01],
                         [0,  0, -0.01,    0]]) * 0.5
    # Old: stronger antisymmetric coupling (underdamped)
    A_old = np.diag([0.95, 0.90, 0.85, 0.80])
    A_old += np.array([[0,  0.15, 0,     0],
                       [-0.15, 0, 0.15,  0],
                       [0, -0.15, 0,  0.15],
                       [0,  0, -0.15,    0]])

    zeta_young = damping_ratio(A_young)
    zeta_old = damping_ratio(A_old)
    assert zeta_young > zeta_old, (
        f"Expected zeta_young ({zeta_young:.4f}) > zeta_old ({zeta_old:.4f})"
    )


# ── Deprecated spectral_gap alias tests ──────────────────────────────────────

def test_spectral_gap_deprecated_still_works():
    """Backward compat: spectral_gap still returns the modulus gap."""
    A = np.diag([0.9, 0.5, 0.3, 0.1])
    assert abs(spectral_gap(A) - 0.4) < 1e-10


def test_spectral_gap_deprecated_identity():
    """Backward compat: identity matrix gap = 0."""
    assert abs(spectral_gap(np.eye(4))) < 1e-10


def test_spectral_gap_deprecated_nonnegative():
    """Backward compat: spectral gap is always non-negative."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        A = rng.normal(size=(8, 8)) * 0.3
        A = A + np.eye(8) * 0.6
        assert spectral_gap(A) >= -1e-15


def test_spectral_gap_deprecated_dominant_mode():
    """Backward compat: strong dominant mode produces large gap."""
    A = np.diag([0.95, 0.2, 0.2, 0.2])
    gap = spectral_gap(A)
    assert gap > 0.7
