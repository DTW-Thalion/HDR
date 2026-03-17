import numpy as np

from hdr_validation.model.safety import gaussian_calibration_toy


def test_gaussian_calibration_reasonable():
    rng = np.random.default_rng(3)
    out = gaussian_calibration_toy(alpha=0.05, n_samples=5000, rng=rng)
    assert abs(out["empirical"] - out["nominal"]) < 0.03
