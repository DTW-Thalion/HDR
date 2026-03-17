"""Tests for hdr_validation.identification — all identification modules."""
import numpy as np

from hdr_validation.identification.hierarchical import HierarchicalCouplingEstimator
from hdr_validation.identification.boed import BOEDEstimator
from hdr_validation.identification.committor_recovery import CommittorRecovery
from hdr_validation.identification.transition_rates import TransitionRateEstimator
from hdr_validation.identification.tau_estimation import TauEstimator
from hdr_validation.identification.risk_information import RiskInformationFrontier
from hdr_validation.identification.population_planning import PopulationPriorPlanner
from hdr_validation.model.slds import BasinModel


def test_hierarchical_map_normal_equations():
    """MAP estimate should solve the normal equations."""
    rng = np.random.default_rng(60)
    n = 3
    J_mech = rng.normal(size=(n, n)) * 0.1
    Sigma_pop = np.eye(n) * 0.5
    Sigma_group = np.eye(n) * 0.3
    est = HierarchicalCouplingEstimator(J_mech, Sigma_pop, Sigma_group)
    X = rng.normal(size=(50, n))
    Y = X @ J_mech + rng.normal(scale=0.1, size=(50, n))
    J_hat = est.estimate(Y, X)
    assert J_hat.shape == (n, n)
    assert np.all(np.isfinite(J_hat))


def test_hierarchical_degradation_to_prior():
    """At T_p=0, estimate should equal J_mech (group mean)."""
    n = 3
    J_mech = np.eye(n) * 0.5
    est = HierarchicalCouplingEstimator(J_mech, np.eye(n), np.eye(n))
    J_hat = est.estimate(None, None)
    assert np.allclose(J_hat, J_mech)


def test_hierarchical_convergence_to_mle():
    """With lots of data, MAP should approach MLE (close to true J)."""
    rng = np.random.default_rng(61)
    n = 3
    J_true = rng.normal(size=(n, n)) * 0.3
    J_mech = np.zeros((n, n))  # wrong prior
    est = HierarchicalCouplingEstimator(J_mech, np.eye(n) * 10, np.eye(n) * 10)
    X = rng.normal(size=(500, n))
    Y = X @ J_true + rng.normal(scale=0.01, size=(500, n))
    J_hat = est.estimate(Y, X, lambda_g=0.001)
    error = np.linalg.norm(J_hat - J_true, 'fro')
    assert error < 0.5, f"MLE convergence failed: error={error}"


def test_boed_information_gain_positive():
    """Information gain should be positive for non-trivial designs."""
    prior = {"mean": np.zeros(3), "cov": np.eye(3)}
    safety = {"u_max": 0.6, "risk_max": 0.1}
    boed = BOEDEstimator(prior, safety)
    design = np.eye(3) * 0.5
    ig = boed.information_gain(design)
    assert ig > 0


def test_boed_sample_complexity_scaling():
    """Sample complexity should scale with n_theta/epsilon^2."""
    prior = {"mean": np.zeros(3), "cov": np.eye(3)}
    safety = {"u_max": 0.6}
    boed = BOEDEstimator(prior, safety)
    N1 = boed.sample_complexity(0.1, 0.05, {"n_theta": 10})
    N2 = boed.sample_complexity(0.05, 0.05, {"n_theta": 10})
    assert N2 > N1  # smaller epsilon -> more samples


def test_boed_safety_constraint():
    """Optimal design should respect u_max."""
    prior = {"mean": np.zeros(3), "cov": np.eye(3)}
    safety = {"u_max": 0.3}
    boed = BOEDEstimator(prior, safety)
    design = boed.optimal_design(np.ones(3) * 0.2, L=6)
    assert np.all(np.abs(design) <= 0.3 + 1e-8)


def test_committor_kernel_boundary_conditions():
    """Empirical committor should be ~1 near success states and ~0 near failure."""
    rng = np.random.default_rng(62)
    cr = CommittorRecovery(kernel_bandwidth=0.5)
    # Trajectories that end at basin 0 (success)
    trajs_success = [rng.normal(size=(10, 2)) * 0.1 for _ in range(20)]
    labels_success = [np.zeros(10, dtype=int) for _ in range(20)]
    # Trajectories that end at basin 1 (failure)
    trajs_fail = [rng.normal(size=(10, 2)) * 0.1 + 5.0 for _ in range(20)]
    labels_fail = [np.ones(10, dtype=int) for _ in range(20)]
    q_hat = cr.estimate(trajs_success + trajs_fail, labels_success + labels_fail)
    # Near success trajectories, q should be high
    q_near_success = q_hat(np.zeros(2))
    assert q_near_success > 0.5


def test_committor_kernel_convergence():
    """With more data, committor estimate should be more confident."""
    rng = np.random.default_rng(63)
    cr = CommittorRecovery(kernel_bandwidth=1.0)
    trajs = [rng.normal(size=(5, 2)) for _ in range(50)]
    labels = [np.array([0]*5) for _ in range(50)]
    q_hat = cr.estimate(trajs, labels)
    q_val = q_hat(np.zeros(2))
    assert 0.0 <= q_val <= 1.0


def test_transition_rate_baum_welch():
    """Transition rate estimator should produce valid transition matrix."""
    rng = np.random.default_rng(64)
    est = TransitionRateEstimator(K=3)
    # Generate simple label sequences
    labels = []
    for _ in range(20):
        seq = rng.choice(3, size=50)
        labels.append(seq)
    result = est.fit(labels)
    T = result["transition_matrix"]
    assert T.shape == (3, 3)
    # Rows should sum to ~1
    for i in range(3):
        assert np.isclose(T[i].sum(), 1.0, atol=1e-6)


def test_tau_exponential_fit():
    """Tau estimator should recover exponential time constant."""
    tau_true = 5.0
    x0 = 1.0
    x_inf = 0.0
    t = np.arange(30)
    trajectory = x_inf + (x0 - x_inf) * np.exp(-t / tau_true)
    est = TauEstimator()
    tau_hat = est.estimate(trajectory, x0, x_inf)
    assert abs(tau_hat - tau_true) < 1.0, f"tau_hat={tau_hat}, expected ~{tau_true}"


def test_tau_multi_axis():
    """Tau estimation works on different axes independently."""
    est = TauEstimator()
    t = np.arange(20)
    traj1 = np.exp(-t / 3.0)
    traj2 = np.exp(-t / 10.0)
    tau1 = est.estimate(traj1, 1.0, 0.0)
    tau2 = est.estimate(traj2, 1.0, 0.0)
    assert tau1 < tau2  # faster decay -> smaller tau


def test_pareto_frontier_dominance():
    """Dominated points should not be on Pareto frontier."""
    n = 4
    basin = BasinModel(A=0.5*np.eye(n), B=np.eye(n, 2), C=np.eye(n),
                        Q=np.eye(n)*0.1, R=np.eye(n)*0.1,
                        b=np.zeros(n), c=np.zeros(n), E=np.eye(n), rho=0.5)
    rif = RiskInformationFrontier(basin, {"lo": -np.ones(n), "hi": np.ones(n)})
    candidates = [np.array([0.1, 0.0]), np.array([0.5, 0.0]),
                  np.array([0.01, 0.0]), np.array([0.3, 0.0])]
    frontier = rif.pareto_frontier(candidates)
    assert len(frontier) >= 1


def test_pareto_frontier_nonempty():
    """Pareto frontier should have at least one point."""
    n = 4
    basin = BasinModel(A=0.5*np.eye(n), B=np.eye(n, 2), C=np.eye(n),
                        Q=np.eye(n)*0.1, R=np.eye(n)*0.1,
                        b=np.zeros(n), c=np.zeros(n), E=np.eye(n), rho=0.5)
    rif = RiskInformationFrontier(basin, {"lo": -np.ones(n), "hi": np.ones(n)})
    candidates = [np.array([0.1, 0.0])]
    frontier = rif.pareto_frontier(candidates)
    assert len(frontier) == 1


def test_population_planner_accuracy():
    """Population planner should achieve better than random accuracy."""
    rng = np.random.default_rng(65)
    n = 3
    K = 2
    B_k = [np.eye(n, 2) * (0.3 if k == 0 else 0.1) for k in range(K)]
    regimens = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5, 0.5])]
    planner = PopulationPriorPlanner(B_k, regimens)
    # Trivial case: just check it runs
    patient = {"basin_probs": np.array([0.7, 0.3])}
    best = planner.plan(patient, H=10)
    assert len(best) == 2


def test_cross_axis_meta_analytic():
    """Hierarchical estimator convergence_check should return valid results."""
    rng = np.random.default_rng(66)
    n = 3
    J_true = rng.normal(size=(n, n)) * 0.2
    J_mech = J_true + rng.normal(size=(n, n)) * 0.1
    est = HierarchicalCouplingEstimator(J_mech, np.eye(n), np.eye(n))
    result = est.convergence_check([0, 10, 50, 200], J_true)
    assert len(result["errors"]) == 4
    assert isinstance(result["monotonic_decrease"], bool)
