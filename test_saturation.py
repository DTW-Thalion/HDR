"""Tests for V7 — Saturating dose-response model."""
from __future__ import annotations

import numpy as np

from hdr_validation.model.saturation import (
    michaelis_menten, inverse_michaelis_menten, apply_saturation
)


def test_monotonicity():
    u_values = np.linspace(0.01, 10.0, 100)
    f_values = michaelis_menten(u_values, u_max=1.0, u_half=0.3)
    assert np.all(np.diff(f_values) > 0)


def test_boundedness():
    u_values = np.array([0.1, 1.0, 10.0, 100.0, 1e6])
    f_values = michaelis_menten(u_values, u_max=2.0, u_half=0.5)
    assert np.all(f_values < 2.0)
    assert np.all(f_values >= 0.0)


def test_half_saturation():
    f = michaelis_menten(0.3, u_max=1.0, u_half=0.3)
    assert abs(f - 0.5) < 1e-10


def test_low_dose_linearity():
    u_small = 0.001
    u_max, u_half = 1.0, 0.3
    f = michaelis_menten(u_small, u_max, u_half)
    linear_approx = u_max * u_small / u_half
    assert abs(f - linear_approx) / linear_approx < 0.01


def test_inverse_correctness():
    rng = np.random.default_rng(42)
    u_values = rng.uniform(0.01, 5.0, size=50)
    u_max, u_half = 2.0, 0.5
    f_values = michaelis_menten(u_values, u_max, u_half)
    u_recovered = inverse_michaelis_menten(f_values, u_max, u_half)
    np.testing.assert_allclose(u_values, u_recovered, atol=1e-10)


def test_apply_saturation_selective():
    u_vec = np.array([0.5, 1.0, 2.0, 0.8])
    sat_channels = [1, 3]
    sat_params = {1: (0.8, 0.3), 3: (0.5, 0.2)}
    u_sat = apply_saturation(u_vec, sat_channels, sat_params)
    # Channels 0 and 2 should be unchanged
    assert u_sat[0] == u_vec[0]
    assert u_sat[2] == u_vec[2]
    # Saturated channels should be bounded by u_max
    assert abs(u_sat[1]) < 0.8  # bounded by u_max
    assert abs(u_sat[3]) < 0.5  # bounded by u_max


def test_apply_saturation_preserves_sign():
    u_vec = np.array([-1.0, 0.5])
    sat_channels = [0, 1]
    sat_params = {0: (2.0, 0.5), 1: (2.0, 0.5)}
    u_sat = apply_saturation(u_vec, sat_channels, sat_params)
    assert u_sat[0] < 0
    assert u_sat[1] > 0
