"""Tests for hdr_validation.model.coherence — coherence penalty and spectral gap."""
import numpy as np
from hdr_validation.model.coherence import spectral_gap


def test_spectral_gap_identity():
    """Identity matrix: all eigenvalues = 1, gap = 0."""
    assert abs(spectral_gap(np.eye(4))) < 1e-10


def test_spectral_gap_diagonal():
    """Diagonal matrix: gap = |lambda_1| - |lambda_2|."""
    A = np.diag([0.9, 0.5, 0.3, 0.1])
    gap = spectral_gap(A)
    assert abs(gap - 0.4) < 1e-10, f"Expected 0.4, got {gap}"


def test_spectral_gap_nonnegative():
    """Spectral gap is always non-negative."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        A = rng.normal(size=(8, 8)) * 0.3
        A = A + np.eye(8) * 0.6  # keep near-stable
        assert spectral_gap(A) >= -1e-15


def test_spectral_gap_dominant_mode():
    """Strong dominant mode produces large gap."""
    # One eigenvalue at 0.95, rest at 0.2
    A = np.diag([0.95, 0.2, 0.2, 0.2])
    gap = spectral_gap(A)
    assert gap > 0.7
