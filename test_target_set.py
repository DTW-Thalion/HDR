import numpy as np

from hdr_validation.model.hsmm import DwellModel, hazard_at


def test_hsmm_sampling_and_hazard():
    rng = np.random.default_rng(123)
    dm = DwellModel("zipf", {"a": 1.8}, max_len=128)
    samples = [dm.sample(rng) for _ in range(50)]
    assert min(samples) >= 1
    hz = dm.hazard()
    assert np.all(hz >= 0) and np.all(hz <= 1)
    assert hazard_at(dm, 3) >= 0
