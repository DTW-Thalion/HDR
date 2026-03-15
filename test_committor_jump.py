"""Tests for committor_with_jumps in hdr_validation.control.lqr."""
import numpy as np
from hdr_validation.control.lqr import committor, committor_with_jumps


def test_committor_jump_bvp_solution():
    """Committor with jumps should solve the composite BVP."""
    n = 4
    rng = np.random.default_rng(70)
    # Simple transition matrices
    P_smooth = np.array([
        [0.5, 0.2, 0.1, 0.2],
        [0.1, 0.6, 0.2, 0.1],
        [0.2, 0.1, 0.5, 0.2],
        [0.1, 0.1, 0.1, 0.7],
    ])
    P_cat = np.array([
        [0.1, 0.1, 0.1, 0.7],
        [0.1, 0.1, 0.1, 0.7],
        [0.1, 0.1, 0.1, 0.7],
        [0.1, 0.1, 0.1, 0.7],
    ])
    p_cat = np.array([0.01, 0.01, 0.01, 0.01])
    q = committor_with_jumps(P_smooth, P_cat, p_cat, [0, 1], [3])
    assert q.shape == (n,)
    assert np.all(np.isfinite(q))


def test_committor_jump_boundary_conditions():
    """Boundary conditions: q[success]=1, q[failure]=0."""
    P_smooth = np.ones((3, 3)) / 3
    P_cat = np.ones((3, 3)) / 3
    p_cat = np.array([0.05, 0.05, 0.05])
    # success_states=[0], failure_states=[2] → q[0]=1, q[2]=0
    q = committor_with_jumps(P_smooth, P_cat, p_cat, [0], [2])
    assert abs(q[0] - 1.0) < 1e-8, f"q[success]={q[0]}, expected 1"
    assert abs(q[2]) < 1e-8, f"q[failure]={q[2]}, expected 0"


def test_committor_jump_q_in_01():
    """Committor values should be in [0, 1]."""
    rng = np.random.default_rng(71)
    n = 5
    P_smooth = rng.dirichlet(np.ones(n), size=n)
    P_cat = rng.dirichlet(np.ones(n), size=n)
    p_cat = rng.uniform(0, 0.1, size=n)
    q = committor_with_jumps(P_smooth, P_cat, p_cat, [0], [n-1])
    assert np.all(q >= -1e-8)
    assert np.all(q <= 1.0 + 1e-8)


def test_committor_jump_reduces_to_standard():
    """With p_cat=0, jump committor should equal standard committor."""
    P = np.array([
        [0.5, 0.3, 0.2],
        [0.2, 0.5, 0.3],
        [0.1, 0.2, 0.7],
    ])
    p_cat = np.zeros(3)
    P_cat = np.eye(3)  # doesn't matter since p_cat=0
    # committor_with_jumps(P, P_cat, p_cat, success, failure) calls
    # committor(P_tilde, failure, success) internally
    q_jump = committor_with_jumps(P, P_cat, p_cat, [2], [0])
    q_standard = committor(P, [0], [2])
    assert np.allclose(q_jump, q_standard, atol=1e-8)
