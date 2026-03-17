import numpy as np

from hdr_validation.control.mode_b import committor, controlled_value_iteration, make_reduced_chain


def test_committor_bounds_and_boundaries():
    P_actions, success, failure, start = make_reduced_chain()
    q = committor(P_actions["conservative"], failure, success)
    assert np.all(q >= 0) and np.all(q <= 1)
    assert np.allclose(q[success], 1.0)
    assert np.allclose(q[failure], 0.0)


def test_value_iteration_converges():
    P_actions, success, failure, start = make_reduced_chain()
    out = controlled_value_iteration(P_actions, success, failure)
    assert 0 <= out["V"][start] <= 1
    assert out["spectral_radius"] < 1
