"""Tests for hdr_validation.control.supervisor — extended 8-branch supervisor."""
import numpy as np
from hdr_validation.control.supervisor import ExtendedSupervisor


def _make_supervisor():
    config = {
        "jump_risk_threshold": 0.3,
        "irr_boundary_threshold": 0.9,
        "xi_max": 100.0,
    }
    return ExtendedSupervisor(config)


def test_supervisor_ici_violated_mode_c():
    sup = _make_supervisor()
    state = {"ici_violated": True, "drift_exceeded": False, "basin_stability": "stable",
             "jump_risk": 0.0, "mode_b_eligible": False, "irr_fraction": 0.0}
    assert sup.select_mode(state) == "C"


def test_supervisor_drift_exceeded_mode_c():
    sup = _make_supervisor()
    state = {"ici_violated": False, "drift_exceeded": True, "basin_stability": "stable",
             "jump_risk": 0.0, "mode_b_eligible": False, "irr_fraction": 0.0}
    assert sup.select_mode(state) == "C"


def test_supervisor_unstable_basin_mode_b():
    sup = _make_supervisor()
    state = {"ici_violated": False, "drift_exceeded": False, "basin_stability": "unstable",
             "jump_risk": 0.0, "mode_b_eligible": False, "irr_fraction": 0.0}
    assert sup.select_mode(state) == "B"


def test_supervisor_jump_risk_mode_b():
    sup = _make_supervisor()
    state = {"ici_violated": False, "drift_exceeded": False, "basin_stability": "stable",
             "jump_risk": 0.5, "mode_b_eligible": False, "irr_fraction": 0.0}
    assert sup.select_mode(state) == "B"


def test_supervisor_stable_maladaptive_mode_b():
    sup = _make_supervisor()
    state = {"ici_violated": False, "drift_exceeded": False, "basin_stability": "stable",
             "jump_risk": 0.0, "mode_b_eligible": True, "irr_fraction": 0.0}
    assert sup.select_mode(state) == "B"


def test_supervisor_irr_approaching_mode_b():
    sup = _make_supervisor()
    state = {"ici_violated": False, "drift_exceeded": False, "basin_stability": "stable",
             "jump_risk": 0.0, "mode_b_eligible": False, "irr_fraction": 0.95}
    assert sup.select_mode(state) == "B"


def test_supervisor_default_mode_a():
    sup = _make_supervisor()
    state = {"ici_violated": False, "drift_exceeded": False, "basin_stability": "stable",
             "jump_risk": 0.0, "mode_b_eligible": False, "irr_fraction": 0.0}
    assert sup.select_mode(state) == "A"


def test_supervisor_rollback_mode_a():
    """Adverse marker overrides everything to Mode A."""
    sup = _make_supervisor()
    state = {"ici_violated": True, "drift_exceeded": True, "basin_stability": "unstable",
             "jump_risk": 0.5, "mode_b_eligible": True, "irr_fraction": 0.95,
             "adverse_marker": True}
    assert sup.select_mode(state) == "A"


def test_supervisor_mode_a_bypass():
    """Infeasible flag forces Mode A."""
    sup = _make_supervisor()
    state = {"ici_violated": True, "drift_exceeded": False, "basin_stability": "stable",
             "jump_risk": 0.0, "mode_b_eligible": False, "irr_fraction": 0.0,
             "infeasible": True}
    assert sup.select_mode(state) == "A"


def test_supervisor_priority_order():
    """Mode C has priority over Mode B when both triggered."""
    sup = _make_supervisor()
    # Both ICI violated (-> C) and unstable basin (-> B)
    state = {"ici_violated": True, "drift_exceeded": False, "basin_stability": "unstable",
             "jump_risk": 0.5, "mode_b_eligible": True, "irr_fraction": 0.95}
    assert sup.select_mode(state) == "C"


def test_supervisor_eigenvalue_crossing_mode_c():
    """Eigenvalue crossing unit circle triggers Mode C."""
    sup = _make_supervisor()
    state = {
        "ici_violated": False,
        "drift_exceeded": False,
        "eigenvalue_crossing": True,
        "basin_stability": "stable",
        "jump_risk": 0.0,
        "mode_b_eligible": False,
        "irr_fraction": 0.0,
    }
    assert sup.select_mode(state) == "C"


def test_supervisor_eigenvalue_crossing_priority_over_mode_b():
    """Eigenvalue crossing (Mode C) has priority over Mode B triggers."""
    sup = _make_supervisor()
    state = {
        "ici_violated": False,
        "drift_exceeded": False,
        "eigenvalue_crossing": True,
        "basin_stability": "unstable",  # would trigger Mode B
        "jump_risk": 0.5,               # would trigger Mode B
        "mode_b_eligible": True,         # would trigger Mode B
        "irr_fraction": 0.95,            # would trigger Mode B
    }
    assert sup.select_mode(state) == "C"


def test_supervisor_eigenvalue_crossing_overridden_by_adverse():
    """Adverse marker rollback still overrides eigenvalue crossing."""
    sup = _make_supervisor()
    state = {
        "ici_violated": False,
        "drift_exceeded": False,
        "eigenvalue_crossing": True,
        "basin_stability": "stable",
        "jump_risk": 0.0,
        "mode_b_eligible": False,
        "irr_fraction": 0.0,
        "adverse_marker": True,
    }
    assert sup.select_mode(state) == "A"


def test_supervisor_no_eigenvalue_crossing_default():
    """When eigenvalue_crossing key is absent, supervisor works normally."""
    sup = _make_supervisor()
    state = {
        "ici_violated": False,
        "drift_exceeded": False,
        # eigenvalue_crossing key intentionally omitted
        "basin_stability": "stable",
        "jump_risk": 0.0,
        "mode_b_eligible": False,
        "irr_fraction": 0.0,
    }
    assert sup.select_mode(state) == "A"
