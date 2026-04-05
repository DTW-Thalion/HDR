"""Tests for Stage 18c — Cost of Premature Deployment."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hdr_validation.stages.stage_18_closed_loop_ici import run_stage_18c

ROOT = Path(__file__).parent


@pytest.fixture(scope="module")
def cpd_result() -> dict:
    return run_stage_18c(
        T_train_values=[20, 100, 500],
        n_episodes=10,
        n_seeds=2,
        fast_mode=True,
    )


def test_18c_01_runs_without_error(cpd_result: dict) -> None:
    """18c.01: Stage runs and returns valid result."""
    assert cpd_result is not None
    assert "checks" in cpd_result
    assert len(cpd_result["checks"]) == 5


def test_18c_02_sweep_has_all_T_train(cpd_result: dict) -> None:
    """18c.02: Sweep contains results for all requested T_train values."""
    assert "sweep" in cpd_result
    T_trains = [pt["T_train"] for pt in cpd_result["sweep"]]
    assert T_trains == [20, 100, 500]


def test_18c_03_cpd_monotonically_decreasing(cpd_result: dict) -> None:
    """18c.03: CPD generally decreases as T_train increases."""
    sweep = cpd_result["sweep"]
    # CPD at smallest T_train should be larger than at largest
    assert sweep[0]["cpd_pct"] > sweep[-1]["cpd_pct"]


def test_18c_04_oracle_cost_lowest(cpd_result: dict) -> None:
    """18c.04: Oracle cost is lowest (or close to) at each T_train."""
    for pt in cpd_result["sweep"]:
        # Oracle should be <= ungated + 10% margin
        assert pt["oracle_mean_cost"] <= pt["ungated_mean_cost"] * 1.10


def test_18c_05_costs_positive(cpd_result: dict) -> None:
    """18c.05: All mean costs are positive."""
    for pt in cpd_result["sweep"]:
        assert pt["oracle_mean_cost"] > 0
        assert pt["gated_mean_cost"] > 0
        assert pt["ungated_mean_cost"] > 0


def test_18c_06_mode_errors_bounded(cpd_result: dict) -> None:
    """18c.06: Mode error rates are in [0, 100]%."""
    for pt in cpd_result["sweep"]:
        assert 0 <= pt["mode_error_pct"] <= 100


def test_18c_07_omega_min_positive(cpd_result: dict) -> None:
    """18c.07: omega_min reference is positive."""
    assert cpd_result["omega_min_reference"] > 0


def test_18c_08_output_saved(cpd_result: dict) -> None:
    """18c.08: Results JSON saved and loadable."""
    path = ROOT / "results" / "stage_18c" / "premature_deployment.json"
    assert path.exists()
    with open(path) as f:
        data = json.load(f)
    assert "sweep" in data


def test_18c_09_provenance_present(cpd_result: dict) -> None:
    """18c.09: Provenance metadata present."""
    assert "provenance" in cpd_result
    assert "hdr_version" in cpd_result["provenance"]


def test_18c_10_ici_verdict_values(cpd_result: dict) -> None:
    """18c.10: ICI verdicts are either PASS or REFUSE."""
    for pt in cpd_result["sweep"]:
        assert pt["ici_verdict"] in ("PASS", "REFUSE")
