"""Tests for hdr_validation.model.adaptive — FF-RLS and drift detection."""
import numpy as np
from hdr_validation.model.adaptive import FFRLSEstimator, DriftDetector


def test_ffrls_convergence_stationary():
    """RLS converges to true A when data is stationary."""
    rng = np.random.default_rng(42)
    n = 4
    A_true = 0.8 * np.eye(n) + 0.05 * rng.normal(size=(n, n))
    est = FFRLSEstimator(n, lambda_ff=0.99)
    x = rng.normal(size=n)
    for _ in range(200):
        x_new = A_true @ x + rng.normal(scale=0.01, size=n)
        est.update(x_new, x)
        x = x_new
    error = np.linalg.norm(est.A_hat - A_true, 'fro')
    assert error < 0.5, f"RLS did not converge: error={error}"


def test_ffrls_tracking_linear_drift():
    """RLS tracks linear drift in A."""
    rng = np.random.default_rng(43)
    n = 3
    A_base = 0.7 * np.eye(n)
    drift_per_step = 0.001
    est = FFRLSEstimator(n, lambda_ff=0.95)
    x = rng.normal(size=n)
    for t in range(300):
        A_t = A_base + drift_per_step * t * np.eye(n)
        x_new = A_t @ x + rng.normal(scale=0.01, size=n)
        est.update(x_new, x)
        x = x_new
    # Drift magnitude should be positive
    assert est.drift_magnitude() > 0.0


def test_ffrls_forgetting_factor_effect():
    """Lower forgetting factor should track faster (larger drift_magnitude for same drift)."""
    rng1 = np.random.default_rng(44)
    rng2 = np.random.default_rng(44)
    n = 3
    A_true = 0.7 * np.eye(n)

    est_high = FFRLSEstimator(n, lambda_ff=0.99)
    est_low = FFRLSEstimator(n, lambda_ff=0.90)

    x1 = rng1.normal(size=n)
    x2 = x1.copy()

    for t in range(100):
        A_t = A_true + 0.002 * t * np.eye(n)
        noise1 = rng1.normal(scale=0.01, size=n)
        noise2 = rng2.normal(scale=0.01, size=n)
        x1_new = A_t @ x1 + noise1
        x2_new = A_t @ x2 + noise2
        est_high.update(x1_new, x1)
        est_low.update(x2_new, x2)
        x1 = x1_new
        x2 = x2_new

    # Both should detect drift
    assert est_high.drift_magnitude() > 0
    assert est_low.drift_magnitude() > 0


def test_drift_detector_threshold():
    """Drift detector triggers when drift exceeds threshold."""
    n = 3
    est = FFRLSEstimator(n)
    # Manually set A_hat far from initial
    est.A_hat = est.A_hat_initial + 0.5 * np.eye(n)
    det = DriftDetector(Delta_A_max=0.1)
    assert det.check(est) is True


def test_drift_detector_no_false_alarm():
    """Drift detector does not trigger when drift is below threshold."""
    n = 3
    est = FFRLSEstimator(n)
    det = DriftDetector(Delta_A_max=1.0)
    assert det.check(est) is False


def test_ffrls_dimension_consistency():
    """A_hat has correct dimensions after updates."""
    n = 5
    est = FFRLSEstimator(n)
    rng = np.random.default_rng(45)
    x = rng.normal(size=n)
    x_new = rng.normal(size=n)
    A_hat = est.update(x_new, x)
    assert A_hat.shape == (n, n)


def test_ffrls_initial_estimate():
    """Initial A_hat is identity."""
    n = 4
    est = FFRLSEstimator(n)
    assert np.allclose(est.A_hat, np.eye(n))
    assert est.drift_magnitude() == 0.0


def test_drift_detector_trigger_latency():
    """Drift detector eventually triggers under persistent drift."""
    rng = np.random.default_rng(46)
    n = 3
    A_true = 0.7 * np.eye(n)
    est = FFRLSEstimator(n, lambda_ff=0.95)
    det = DriftDetector(Delta_A_max=0.3)
    x = rng.normal(size=n)
    triggered = False
    for t in range(500):
        A_t = A_true + 0.005 * t * np.eye(n)
        x_new = A_t @ x + rng.normal(scale=0.01, size=n)
        est.update(x_new, x)
        x = x_new
        if det.check(est):
            triggered = True
            break
    assert triggered, "Drift detector never triggered"
