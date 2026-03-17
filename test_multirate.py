"""Tests for hdr_validation.model.multirate — multi-rate observer and delay augmentation."""
import numpy as np
from hdr_validation.model.multirate import MultiRateObserver, DelayAugmentedState


def test_multirate_C_at_fast_tier():
    """Fast tier (c=1) is always active."""
    C1 = np.eye(4, 3)
    C2 = np.ones((2, 3)) * 0.5
    obs = MultiRateObserver([C1, C2], [1, 4])
    C = obs.C_at(1)
    # Fast tier should be nonzero
    assert np.any(C[:4] != 0)


def test_multirate_C_at_slow_tier_active():
    """Slow tier (c=4) is active at t=0, 4, 8, ..."""
    C1 = np.eye(4, 3)
    C2 = np.ones((2, 3)) * 0.5
    obs = MultiRateObserver([C1, C2], [1, 4])
    C = obs.C_at(0)  # t=0 mod 4 == 0
    assert np.any(C[4:] != 0), "Slow tier should be active at t=0"
    C4 = obs.C_at(4)
    assert np.any(C4[4:] != 0), "Slow tier should be active at t=4"


def test_multirate_C_at_slow_tier_inactive():
    """Slow tier (c=4) is inactive at t=1,2,3."""
    C1 = np.eye(4, 3)
    C2 = np.ones((2, 3)) * 0.5
    obs = MultiRateObserver([C1, C2], [1, 4])
    for t in [1, 2, 3]:
        C = obs.C_at(t)
        assert np.all(C[4:] == 0), f"Slow tier should be zero at t={t}"


def test_delay_augmented_dynamics_shape():
    """Augmented dynamics matrix has shape (n*h, n*h)."""
    n, h = 4, 3
    das = DelayAugmentedState(n, n_j=2, h=h)
    A_k = 0.8 * np.eye(n)
    A_delay = 0.1 * np.eye(n)
    A_aug = das.augmented_dynamics(A_k, A_delay)
    assert A_aug.shape == (n * h, n * h)


def test_delay_lmi_feasible():
    """Small delay coupling should yield stable augmented system."""
    n, h = 3, 2
    das = DelayAugmentedState(n, n_j=2, h=h)
    A_k = 0.5 * np.eye(n)
    A_delay = 0.01 * np.eye(n)  # very small coupling
    assert das.check_delay_lmi(A_k, A_delay) is True


def test_delay_lmi_infeasible_large_delay():
    """Large delay coupling should yield unstable augmented system."""
    n, h = 3, 5
    das = DelayAugmentedState(n, n_j=2, h=h)
    A_k = 0.95 * np.eye(n)
    A_delay = 0.8 * np.eye(n)  # large coupling + near-unit-root
    assert das.check_delay_lmi(A_k, A_delay) is False
