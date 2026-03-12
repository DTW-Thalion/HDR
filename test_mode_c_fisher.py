"""Tests for hdr_validation.control.mode_c_fisher."""
from __future__ import annotations

import numpy as np
import pytest

from hdr_validation.control.mode_c_fisher import (
    accumulated_fisher_lower_bound,
    compute_fisher_trace,
    dither_policy,
    maximise_fisher_trace,
)


def test_compute_fisher_trace_identity_unit_vector():
    """With Q_w_k = I and u a unit vector, trace = n_state."""
    n = 4
    Q_w_k = np.eye(n)
    u = np.zeros(n)
    u[0] = 1.0  # unit vector
    result = compute_fisher_trace(u, Q_w_k, n_state=n)
    assert abs(result - n) < 1e-9


def test_compute_fisher_trace_scales_quadratically():
    """Doubling ||u|| quadruples the Fisher trace."""
    n = 4
    Q_w_k = np.eye(n)
    u1 = np.ones(n) * 0.5
    u2 = np.ones(n) * 1.0  # double
    t1 = compute_fisher_trace(u1, Q_w_k, n_state=n)
    t2 = compute_fisher_trace(u2, Q_w_k, n_state=n)
    assert abs(t2 / t1 - 4.0) < 1e-9


def test_compute_fisher_trace_zero_control():
    """Zero control → zero Fisher trace."""
    n = 4
    Q_w_k = np.eye(n)
    u = np.zeros(n)
    result = compute_fisher_trace(u, Q_w_k, n_state=n)
    assert result == 0.0


def test_maximise_fisher_trace_symmetric_box_returns_corner():
    """With symmetric box [-1, 1]^m, maximiser should be at a corner."""
    n = 4
    Q_w_k = np.eye(n)
    m = n
    u_min = -np.ones(m)
    u_max = np.ones(m)
    u_opt = maximise_fisher_trace(Q_w_k, n, u_min, u_max, burden_remaining=100.0)
    # Every component should be at either +1 or -1
    for ui in u_opt:
        assert abs(abs(ui) - 1.0) < 1e-9


def test_maximise_fisher_trace_l1_budget():
    """Result should satisfy ||u||_1 <= burden_remaining."""
    n = 4
    m = 4
    Q_w_k = np.eye(n)
    u_min = -np.ones(m)
    u_max = np.ones(m)
    budget = 1.5
    u_opt = maximise_fisher_trace(Q_w_k, n, u_min, u_max, burden_remaining=budget)
    assert float(np.sum(np.abs(u_opt))) <= budget + 1e-9


def test_maximise_fisher_trace_zero_budget():
    """With budget=0, result should be zero."""
    n = 4
    m = 4
    Q_w_k = np.eye(n)
    u_min = -np.ones(m)
    u_max = np.ones(m)
    u_opt = maximise_fisher_trace(Q_w_k, n, u_min, u_max, burden_remaining=0.0)
    assert float(np.sum(np.abs(u_opt))) < 1e-9


def test_dither_policy_within_bounds():
    """Output of dither_policy should be within [u_min, u_max]."""
    rng = np.random.default_rng(42)
    m = 8
    u_nominal = np.zeros(m)
    u_min = -0.6 * np.ones(m)
    u_max = 0.6 * np.ones(m)
    for _ in range(100):
        u_c = dither_policy(u_nominal, sigma_c=0.5, u_min=u_min, u_max=u_max, rng=rng)
        assert np.all(u_c >= u_min - 1e-9)
        assert np.all(u_c <= u_max + 1e-9)


def test_dither_policy_shape():
    """Output shape matches input shape."""
    rng = np.random.default_rng(0)
    m = 6
    u_nominal = np.zeros(m)
    u_min = -np.ones(m)
    u_max = np.ones(m)
    u_c = dither_policy(u_nominal, sigma_c=0.1, u_min=u_min, u_max=u_max, rng=rng)
    assert u_c.shape == (m,)


def test_fisher_trace_non_isotropic():
    """Verify compute_fisher_trace handles non-isotropic Q_w correctly.

    The Kronecker formula gives:
        tr(F^(1)) = (u^T Q_w^{-1} u) * n_state
    which differs from the isotropic simplification tr(Q_w^{-1}) * ||u||^2.
    """
    Q_w = np.diag([1.0, 4.0, 9.0])  # non-isotropic
    u = np.array([1.0, 0.0, 0.0])
    n_state = 3
    trace_val = compute_fisher_trace(u, Q_w, n_state=n_state)
    # General Kronecker formula: (u^T Q_w^{-1} u) * n = (1.0) * 3 = 3.0
    expected = float(u @ np.linalg.inv(Q_w) @ u) * n_state
    assert abs(trace_val - expected) < 1e-6, (
        f"Expected Kronecker formula result {expected:.4f}, got {trace_val:.4f}"
    )


def test_accumulated_fisher_lower_bound_monotone_in_T_C():
    """Lower bound should increase monotonically with T_C."""
    n = 4
    Q_w_k = np.eye(n)
    bounds = [accumulated_fisher_lower_bound(T, Q_w_k, n, u_min_norm_sq=1.0) for T in range(1, 10)]
    for i in range(len(bounds) - 1):
        assert bounds[i + 1] > bounds[i]


def test_accumulated_fisher_lower_bound_zero_T_C():
    """With T_C=0, lower bound is 0."""
    n = 4
    Q_w_k = np.eye(n)
    result = accumulated_fisher_lower_bound(0, Q_w_k, n, u_min_norm_sq=1.0)
    assert result == 0.0


def test_accumulated_fisher_lower_bound_positive():
    """Lower bound is always non-negative."""
    n = 4
    Q_w_k = 2.0 * np.eye(n)
    result = accumulated_fisher_lower_bound(5, Q_w_k, n, u_min_norm_sq=0.5)
    assert result >= 0.0
