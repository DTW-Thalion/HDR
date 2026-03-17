"""Tests for hdr_validation.model.stability_check."""
from __future__ import annotations

import numpy as np
import pytest

from hdr_validation.model.stability_check import (
    assert_spectral_radius_lt1,
    check_all_basin_stability,
)


def test_stable_matrix_returns_true():
    """A = 0.5 * I should be stable (rho = 0.5 < 1)."""
    A = 0.5 * np.eye(4)
    result = assert_spectral_radius_lt1(A, basin_id=0)
    assert result is True


def test_unit_root_matrix_raises_by_default():
    """A = I has rho = 1.0, should raise ValueError by default."""
    A = np.eye(4)
    with pytest.raises(ValueError, match="Stability check FAILED"):
        assert_spectral_radius_lt1(A, basin_id=1)


def test_unit_root_returns_false_when_not_raising():
    """A = I with raise_on_failure=False should return False and warn."""
    A = np.eye(4)
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = assert_spectral_radius_lt1(A, basin_id=1, raise_on_failure=False)
    assert result is False
    assert len(w) == 1
    assert "Stability check FAILED" in str(w[0].message)


def test_unstable_matrix_raises():
    """A = 1.1 * I has rho = 1.1 > 1, should raise ValueError."""
    A = 1.1 * np.eye(3)
    with pytest.raises(ValueError, match="Stability check FAILED"):
        assert_spectral_radius_lt1(A, basin_id=2)


def test_check_all_basin_stability_all_stable():
    """All stable basins → all True."""
    basin_models = {
        0: {"A_k": 0.5 * np.eye(4)},
        1: {"A_k": 0.7 * np.eye(4)},
        2: {"A_k": 0.9 * np.eye(4)},
    }
    results = check_all_basin_stability(basin_models)
    assert results == {0: True, 1: True, 2: True}


def test_check_all_basin_stability_mixed():
    """Mix of stable and unstable basins — no raise, collect results."""
    basin_models = {
        0: {"A_k": 0.5 * np.eye(4)},
        1: {"A_k": 1.1 * np.eye(4)},
    }
    import warnings
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        results = check_all_basin_stability(basin_models, raise_on_failure=False)
    assert results[0] is True
    assert results[1] is False


def test_check_all_basin_stability_raises_on_first_failure():
    """With raise_on_failure=True (default), first unstable basin raises."""
    basin_models = {
        0: {"A_k": 1.1 * np.eye(3)},
    }
    with pytest.raises(ValueError, match="Stability check FAILED"):
        check_all_basin_stability(basin_models)
