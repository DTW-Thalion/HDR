"""Pytest configuration — profile-based test filtering.

Usage:
    pytest --profile smoke       # core + stage tests (fastest)
    pytest --profile standard    # smoke + ICI/Mode-C/coherence tests
    pytest --profile extended    # standard + v7.0/v7.1 extension tests
    pytest --profile validation  # all tests (default)
    pytest                       # no filter — runs everything

Profiles are cumulative: each higher tier includes all tests from lower tiers.
"""
from __future__ import annotations

import pytest

# ── Profile tier definitions (cumulative) ─────────────────────────────────────
# Each tier lists the *additional* test files it adds on top of the previous tier.

_SMOKE_FILES = {
    # Core fast tests — fundamental control/inference mechanics
    "test_mpc.py",
    "test_committor.py",
    "test_recovery.py",
    "test_safety.py",
    "test_hsmm.py",
    "test_imm.py",
    # Stage integration tests — validate stage output artifacts
    "test_stage_08.py",
    "test_stage_08b.py",
    "test_stage_09.py",
    "test_stage_10.py",
    "test_stage_11.py",
    "test_stage_16.py",
}

_STANDARD_ADDS = {
    # ICI, Mode C, coherence — claims 9-14
    "test_ici.py",
    "test_ici_compound.py",
    "test_mode_c.py",
    "test_mode_c_fisher.py",
    "test_coherence.py",
    "test_committor_jump.py",
}

_EXTENDED_ADDS = {
    # v7.0/v7.1 extensions, adaptive, identification — claims 15-32
    "test_extensions.py",
    "test_adaptive.py",
    "test_adaptive_delta.py",
    "test_multirate.py",
    "test_mimpc.py",
    "test_particle.py",
    "test_saturation.py",
    "test_stability_check.py",
    "test_identification.py",
    "test_tube_mpc.py",
    "test_supervisor.py",
    "test_variational.py",
    "test_interaction_matrix.py",
}

# Cumulative sets
PROFILE_FILES: dict[str, set[str]] = {
    "smoke": _SMOKE_FILES,
    "standard": _SMOKE_FILES | _STANDARD_ADDS,
    "extended": _SMOKE_FILES | _STANDARD_ADDS | _EXTENDED_ADDS,
    "validation": _SMOKE_FILES | _STANDARD_ADDS | _EXTENDED_ADDS,  # all files
}

VALID_PROFILES = list(PROFILE_FILES)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--profile",
        action="store",
        default=None,
        choices=VALID_PROFILES,
        help="Run only tests belonging to the specified profile tier: "
             "smoke, standard, extended, or validation (default: all).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    profile = config.getoption("--profile")
    if profile is None:
        return  # no filter — run everything

    allowed = PROFILE_FILES[profile]
    kept: list[pytest.Item] = []
    deselected: list[pytest.Item] = []

    for item in items:
        filename = item.path.name if hasattr(item, "path") else item.fspath.basename
        if filename in allowed:
            kept.append(item)
        else:
            deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = kept
