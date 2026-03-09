import numpy as np

from hdr_validation.inference.imm import IMMFilter
from hdr_validation.model.slds import make_evaluation_model


def test_imm_probabilities_sum_to_one():
    config = {
        "state_dim": 8,
        "obs_dim": 16,
        "control_dim": 8,
        "disturbance_dim": 8,
        "K": 3,
        "rho_reference": [0.72, 0.96, 0.55],
        "max_dwell_len": 128,
    }
    rng = np.random.default_rng(1)
    model = make_evaluation_model(config, rng)
    imm = IMMFilter(model)
    y = np.zeros(16)
    mask = np.ones(16)
    u = np.zeros(8)
    st = imm.step(y, mask, u)
    assert np.isclose(np.sum(st.mode_probs), 1.0)
    assert st.mixed_mean.shape == (8,)
