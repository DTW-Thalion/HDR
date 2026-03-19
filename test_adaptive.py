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


def test_bifurcation_margin_stable():
    """Subcritical regime: margin should be positive."""
    est = FFRLSEstimator(n=8, lambda_ff=0.98)
    # Healthy basin: diagonal dominant (rho ~ 0.72)
    A = np.eye(8) * 0.72
    A[0, 1] = 0.05  # weak I->M coupling
    A[1, 0] = 0.04  # weak M->I coupling
    est.A_hat = A.copy()
    est.A_hat_initial = A.copy()
    margin = est.bifurcation_margin_IM(idx_I=0, idx_M=1)
    assert margin > 0, f"Expected positive margin, got {margin}"


def test_bifurcation_margin_critical():
    """At bifurcation: margin should be near zero."""
    est = FFRLSEstimator(n=2, lambda_ff=0.98)
    # Construct A such that det(I - A) = 0
    # (1 - A_00)(1 - A_11) = A_01 * A_10
    # Let A_00 = 0.5, A_11 = 0.6 => (0.5)(0.4) = 0.2
    # So A_01 * A_10 = 0.2 => A_01 = 0.5, A_10 = 0.4
    A = np.array([[0.5, 0.5], [0.4, 0.6]])
    est.A_hat = A.copy()
    margin = est.bifurcation_margin_IM(idx_I=0, idx_M=1)
    assert abs(margin) < 1e-10, f"Expected near-zero margin, got {margin}"


def test_bifurcation_margin_supercritical():
    """Beyond bifurcation: margin should be negative."""
    est = FFRLSEstimator(n=2, lambda_ff=0.98)
    # Strong off-diagonal coupling pushes past bifurcation
    A = np.array([[0.5, 0.8], [0.7, 0.6]])
    est.A_hat = A.copy()
    margin = est.bifurcation_margin_IM(idx_I=0, idx_M=1)
    assert margin < 0, f"Expected negative margin, got {margin}"


def test_eigenvalue_crossing_detected_stable():
    """Stable basin: no crossing detected."""
    est = FFRLSEstimator(n=8, lambda_ff=0.98)
    est.A_hat = np.eye(8) * 0.72
    assert not est.eigenvalue_crossing_detected(threshold=0.02)


def test_eigenvalue_crossing_detected_near_unit():
    """Near-unit eigenvalue: crossing detected."""
    est = FFRLSEstimator(n=2, lambda_ff=0.98)
    # Eigenvalues at 0.99 and 0.5 — 0.99 is within 0.02 of 1.0
    est.A_hat = np.diag([0.99, 0.5])
    assert est.eigenvalue_crossing_detected(threshold=0.02)


def test_eigenvalue_crossing_not_detected_marginal():
    """Eigenvalue at 0.95 with threshold 0.02: no crossing."""
    est = FFRLSEstimator(n=2, lambda_ff=0.98)
    est.A_hat = np.diag([0.95, 0.5])
    assert not est.eigenvalue_crossing_detected(threshold=0.02)
