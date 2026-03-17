"""Tests for Stage 08b — Multi-Axis Asymmetric Ablation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hdr_validation.stages.stage_08b_ablation import (
    _build_asymmetric_J,
    _make_benchmark_config_8b,
    run_stage_08b,
)
from hdr_validation.stages.stage_08_ablation import ABLATION_VARIANTS


def test_asymmetric_J_row_norm_ratio():
    """J coupling matrix has strong/weak row-norm ratio >= 5."""
    J = _build_asymmetric_J(8)
    strong_norms = np.linalg.norm(J[:3, :], axis=1)
    weak_norms = np.linalg.norm(J[3:, :], axis=1)
    ratio = strong_norms.mean() / weak_norms.mean()
    assert ratio >= 5.0, f"Row-norm ratio {ratio:.2f} < 5.0"


def test_asymmetric_J_shape():
    """J coupling matrix has correct shape."""
    J = _build_asymmetric_J(8)
    assert J.shape == (8, 8)


def test_asymmetric_J_deterministic():
    """J coupling matrix is deterministic (fixed internal seed)."""
    J1 = _build_asymmetric_J(8)
    J2 = _build_asymmetric_J(8)
    np.testing.assert_array_equal(J1, J2)


def test_benchmark_config_8b_has_J_coupling():
    """Stage 8b config includes J_coupling matrix."""
    cfg = _make_benchmark_config_8b(n_seeds=1, n_ep=1, T=32)
    assert "J_coupling" in cfg
    J = np.asarray(cfg["J_coupling"])
    assert J.shape == (8, 8)


def test_benchmark_config_8b_J_strong_weak_structure():
    """J_coupling in returned config preserves strong/weak axis structure."""
    cfg = _make_benchmark_config_8b(n_seeds=1, n_ep=1, T=32)
    J = np.asarray(cfg["J_coupling"])
    strong_norms = np.linalg.norm(J[:3, :], axis=1)
    weak_norms = np.linalg.norm(J[3:, :], axis=1)
    # Strong axes should have larger row norms than weak axes
    assert strong_norms.min() > weak_norms.max(), (
        f"Strong axis min norm {strong_norms.min():.4f} <= "
        f"weak axis max norm {weak_norms.max():.4f}"
    )


# ---- Module-scoped fixture for the base (n_seeds=2, n_ep=3, T=32) group ----

@pytest.fixture(scope="module")
def stage_08b_with_path(tmp_path_factory):
    """Run Stage 08b once with standard fast parameters; return (result, output_dir)."""
    tmp = tmp_path_factory.mktemp("stage_08b")
    result = run_stage_08b(n_seeds=2, n_ep=3, T=32,
                           output_dir=tmp, fast_mode=False)
    return result, tmp


def test_run_stage_08b_fast(stage_08b_with_path):
    """Smoke test: run with small parameters and check output structure."""
    result, tmp = stage_08b_with_path

    # All five variants present
    assert "variants" in result
    assert len(result["variants"]) == 5
    required = {"hdr_full", "mpc_only", "mpc_plus_surrogate",
                "mpc_plus_coherence", "hdr_no_calib"}
    assert set(result["variants"].keys()) == required

    # Required fields per variant
    for name, v in result["variants"].items():
        assert "mean_gain" in v, f"mean_gain missing from {name}"
        assert "ci_lo" in v, f"ci_lo missing from {name}"
        assert "ci_hi" in v, f"ci_hi missing from {name}"
        assert "win_rate" in v, f"win_rate missing from {name}"
        assert "N_mal" in v, f"N_mal missing from {name}"
        assert "diagnostics_mean" in v, f"diagnostics_mean missing from {name}"


def test_run_stage_08b_output_file(stage_08b_with_path):
    """JSON output file is written with correct name."""
    result, tmp = stage_08b_with_path

    out_file = tmp / "ablation_asymmetric_results.json"
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert "variants" in data
    assert "scenario" in data
    assert data["scenario"] == "multi_axis_asymmetric"


def test_run_stage_08b_J_diagnostics(stage_08b_with_path):
    """Result includes J_diagnostics with row-norm ratio."""
    result, tmp = stage_08b_with_path

    assert "J_diagnostics" in result
    jd = result["J_diagnostics"]
    assert "strong_axis_mean_norm" in jd
    assert "weak_axis_mean_norm" in jd
    assert "row_norm_ratio" in jd
    assert jd["row_norm_ratio"] >= 5.0


def test_run_stage_08b_marginal_gains(stage_08b_with_path):
    """Result includes coherence and calibration marginal gain fields."""
    result, tmp = stage_08b_with_path

    assert "coherence_marginal_gain" in result
    assert "calibration_marginal_gain" in result
    assert np.isfinite(result["coherence_marginal_gain"])
    assert np.isfinite(result["calibration_marginal_gain"])


def test_run_stage_08b_gains_are_finite(stage_08b_with_path):
    """All variant gains and CIs must be finite."""
    result, tmp = stage_08b_with_path

    for name, v in result["variants"].items():
        assert np.isfinite(v["mean_gain"]), f"{name} mean_gain not finite"
        assert np.isfinite(v["ci_lo"]), f"{name} ci_lo not finite"
        assert np.isfinite(v["ci_hi"]), f"{name} ci_hi not finite"
        assert v["ci_lo"] <= v["ci_hi"], f"{name} CI inverted"


# ---- Single-caller tests: no fixture needed ----

def test_run_stage_08b_ablation_criterion_fields(tmp_path: Path):
    """Ablation criterion fields are present and typed correctly."""
    result = run_stage_08b(n_seeds=2, n_ep=4, T=32,
                           output_dir=tmp_path, fast_mode=False)

    assert "ablation_criterion_met" in result
    assert "ablation_criterion_note" in result
    assert isinstance(result["ablation_criterion_met"], bool)
    assert isinstance(result["ablation_criterion_note"], str)


def test_run_stage_08b_expected_tag_when_inverted(tmp_path: Path):
    """If criterion fails at short T, the note must contain EXPECTED_AT_SHORT_T."""
    result = run_stage_08b(n_seeds=2, n_ep=6, T=32,
                           output_dir=tmp_path, fast_mode=False)
    if not result["ablation_criterion_met"]:
        assert "EXPECTED_AT_SHORT_T" in result["ablation_criterion_note"], (
            "Inverted criterion at T=32 must be tagged as EXPECTED_AT_SHORT_T "
            "to prevent downstream manuscript tools from consuming the result."
        )


def test_run_stage_08b_fast_mode(tmp_path: Path):
    """fast_mode=True caps n_seeds, n_ep, T."""
    result = run_stage_08b(n_seeds=20, n_ep=30, T=256,
                           output_dir=tmp_path, fast_mode=True)

    assert result["n_seeds"] <= 2
    assert result["n_ep_per_seed"] <= 3
    assert result["T"] <= 64


def test_run_stage_08b_hdr_full_beats_mpc_only_production(tmp_path: Path):
    """hdr_full gain >= mpc_only gain at production scale (T=256, N_mal>=50).
    This is the definitive test of the asymmetric ablation claim."""
    result = run_stage_08b(n_seeds=20, n_ep=30, T=256,
                           output_dir=tmp_path, fast_mode=False)
    assert result["variants"]["hdr_full"]["N_mal"] >= 50, (
        f"N_mal={result['variants']['hdr_full']['N_mal']} < 50. "
        "Insufficient maladaptive episodes for valid ablation statistics."
    )
    assert result["ablation_criterion_met"], result["ablation_criterion_note"]
