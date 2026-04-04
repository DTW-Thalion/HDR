"""
HDR Validation Suite — Orchestration Script
============================================
Runs any combination of profiles and stages by importing stage functions
directly from the per-profile runner modules.

Usage:
    python run_all.py                              # all profiles, all stages
    python run_all.py --profiles smoke             # smoke only
    python run_all.py --profiles smoke standard extended validation
    python run_all.py --profiles standard --stages 04 --force
    python run_all.py --stages 01 03b 03c          # selected stages, all profiles
    python run_all.py --resume --skip-done         # skip already-completed stages
    python run_all.py --stages 08 08b --run-tests  # run stages then pytest tests
    python run_all.py --full-validation            # all 36 claims, highpower for 1-2
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

MANIFEST_PATH = ROOT / "run_all_manifest.json"

# ── Stage metadata ─────────────────────────────────────────────────────────────

STAGE_SEQUENCE = ["01", "02", "03", "03b", "03c", "04", "05", "06", "07", "08", "08b", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "18b"]

STAGE_LABELS = {
    "01":  "Stage 01 — Mathematical Checks",
    "02":  "Stage 02 — Synthetic Data Generation",
    "03":  "Stage 03 — IMM Inference",
    "03b": "Stage 03b — ICI Diagnostics",
    "03c": "Stage 03c — Mode C Validation",
    "04":  "Stage 04 — Mode A Control",
    "05":  "Stage 05 — Mode B Validation",
    "06":  "Stage 06 — State Coherence",
    "07":  "Stage 07 — Robustness Sweeps",
    "08":  "Stage 08 — Ablation Study",
    "08b": "Stage 08b — Asymmetric Ablation",
    "09":  "Stage 09 — Baseline Comparison",
    "10":  "Stage 10 — Mode B FP/FN Sweep",
    "11":  "Stage 11 — Riccati Invariant Set",
    "12":  "Stage 12 — Hierarchical Coupling",
    "13":  "Stage 13 — Inference Backbone",
    "14":  "Stage 14 — Population Planning",
    "15":  "Stage 15 — Proxy Composite",
    "16":  "Stage 16 — Extension Integration",
    "17":  "Stage 17 — Gompertz Mortality & Complexity Collapse",
    "18":  "Stage 18 — Closed-Loop ICI Benchmark",
    "18b": "Stage 18b — Sensor-Degradation Sweep",
}

# Stages that must run before a given stage (dependency order)
STAGE_DEPS: dict[str, list[str]] = {
    "03":  ["02"],
    "03b": ["02", "03"],
    "04":  ["02"],
}

PROFILE_MODULES = {
    "smoke":      "smoke_runner",
    "standard":   "standard_runner",
    "extended":   "extended_runner",
    "validation": "validation_runner",
}

# Mapping of stage IDs to their pytest test files (relative to ROOT).
# Stages without a dedicated test file are omitted.
STAGE_TEST_FILES: dict[str, str] = {
    "08":  "test_stage_08.py",
    "08b": "test_stage_08b.py",
    "09":  "test_stage_09.py",
    "10":  "test_stage_10.py",
    "11":  "test_stage_11.py",
    "16":  "test_stage_16.py",
    "17":  "test_stage_17.py",
    "18":  "test_stage_18.py",
    "18b": "test_stage_18b.py",
}

# Claim-to-stage mapping (for --full-validation summary)
CLAIM_STAGES: dict[int, list[str]] = {
    1: ["04"], 2: ["04"], 3: ["01"], 4: ["01"],
    5: ["01", "07"], 6: ["07"], 7: ["05"], 8: ["05"],
    9: ["06", "08", "08b"], 10: ["03"], 11: ["03b"], 12: ["03c"],
    13: ["03b", "10"], 14: ["01", "07"],
    15: ["16"], 16: ["16"], 17: ["16"], 18: ["16"],
    19: ["16"], 20: ["16"], 21: ["16"], 22: ["16"],
    23: ["16"], 24: ["16"], 25: ["16"], 26: ["16"],
    27: ["13"], 28: ["12"], 29: ["12"], 30: ["12"],
    31: ["14"], 32: ["15"],
    33: ["17"], 34: ["17"],
    35: ["18"], 36: ["18"],
}

# ── Manifest (checkpoint) ──────────────────────────────────────────────────────

def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def manifest_key(profile: str, stage: str) -> str:
    return f"{profile}:{stage}"


def is_done(manifest: dict, profile: str, stage: str) -> bool:
    return manifest.get(manifest_key(profile, stage)) == "done"


def mark_done(manifest: dict, profile: str, stage: str) -> None:
    manifest[manifest_key(profile, stage)] = "done"


def mark_failed(manifest: dict, profile: str, stage: str) -> None:
    manifest[manifest_key(profile, stage)] = "failed"

# ── Dependency resolution ──────────────────────────────────────────────────────

def resolve_with_deps(requested: list[str]) -> list[str]:
    """Return ordered list of stages to run, including required dependencies."""
    needed: set[str] = set(requested)
    for s in requested:
        for dep in STAGE_DEPS.get(s, []):
            needed.add(dep)
    return [s for s in STAGE_SEQUENCE if s in needed]


# ── Stage dispatch ─────────────────────────────────────────────────────────────

def _call_stage_08(fast: bool = False) -> None:
    """Run Stage 08 ablation study."""
    from hdr_validation.stages.stage_08_ablation import run_stage_08
    n_seeds = 2 if fast else 20
    n_ep = 3 if fast else 30
    T = 32 if fast else 256
    run_stage_08(n_seeds=n_seeds, n_ep=n_ep, T=T, fast_mode=False)


def _call_stage_08b(fast: bool = False) -> None:
    """Run Stage 08b multi-axis asymmetric ablation."""
    from hdr_validation.stages.stage_08b_ablation import run_stage_08b
    n_seeds = 2 if fast else 20
    n_ep = 3 if fast else 30
    T = 32 if fast else 256
    run_stage_08b(n_seeds=n_seeds, n_ep=n_ep, T=T, fast_mode=False)


def _call_stage_09(fast: bool = False) -> None:
    """Run Stage 09 baseline comparison."""
    from hdr_validation.stages.stage_09_baselines import run_stage_09
    n_seeds = 2 if fast else 20
    n_ep = 3 if fast else 30
    T = 32 if fast else 256
    run_stage_09(n_seeds=n_seeds, n_ep=n_ep, T=T, fast_mode=False)


def _call_stage_10(fast: bool = False) -> None:
    """Run Stage 10 Mode B FP/FN sweep."""
    from hdr_validation.stages.stage_10_mode_b_sweep import run_stage_10
    N_sim = 200 if fast else 5000
    run_stage_10(N_sim=N_sim, T=50, fast_mode=False)


def _call_stage_12(fast: bool = False) -> None:
    """Run Stage 12 hierarchical coupling estimation."""
    from hdr_validation.stages.stage_12_hierarchical import run_stage_12
    n_patients = 5 if fast else 10
    T_p_values = [0, 10, 50] if fast else [0, 10, 50, 200]
    run_stage_12(n_patients=n_patients, T_p_values=T_p_values, fast_mode=fast)


def _call_stage_13(fast: bool = False) -> None:
    """Run Stage 13 inference backbone benchmark."""
    from hdr_validation.stages.stage_13_inference_backbone import run_stage_13
    n_particles = 50 if fast else 100
    n_scenarios = 3 if fast else 5
    run_stage_13(n_particles=n_particles, n_scenarios=n_scenarios, fast_mode=fast)


def _call_stage_14(fast: bool = False) -> None:
    """Run Stage 14 population planning benchmark."""
    from hdr_validation.stages.stage_14_population_planning import run_stage_14
    n_patients = 10 if fast else 20
    run_stage_14(n_patients=n_patients, fast_mode=fast)


def _call_stage_15(fast: bool = False) -> None:
    """Run Stage 15 proxy composite benchmark."""
    from hdr_validation.stages.stage_15_proxy_composite import run_stage_15
    n_scenarios = 3 if fast else 5
    run_stage_15(n_scenarios=n_scenarios, fast_mode=fast)


def _call_stage_16(fast: bool = False) -> None:
    """Run Stage 16 model-failure extension integration."""
    from hdr_validation.stages.stage_16_extensions import run_stage_16
    n_seeds = 2 if fast else 5
    T = 32 if fast else 128
    run_stage_16(n_seeds=n_seeds, T=T, fast_mode=fast)


def _call_stage_17(fast: bool = False) -> None:
    """Run Stage 17 emergent Gompertz mortality & complexity collapse."""
    from hdr_validation.stages.stage_17_gompertz import run_stage_17
    n_traj = 500 if fast else 5000
    run_stage_17(n_trajectories=n_traj, seed=42, fast_mode=fast)


def _call_stage_18(fast: bool = False) -> None:
    """Run Stage 18 closed-loop ICI benchmark under partial observability."""
    from hdr_validation.stages.stage_18_closed_loop_ici import run_stage_18
    n_seeds = 3 if fast else 20
    n_ep = 5 if fast else 30
    T = 64 if fast else 256
    run_stage_18(n_seeds=n_seeds, n_ep=n_ep, T=T, fast_mode=fast)


def _call_stage_18b(fast: bool = False) -> None:
    """Run Stage 18b sensor-degradation sweep."""
    from hdr_validation.stages.stage_18_closed_loop_ici import run_stage_18b
    n_seeds = 2 if fast else 5
    n_ep = 4 if fast else 12
    T = 64 if fast else 128
    run_stage_18b(n_seeds=n_seeds, n_ep=n_ep, T=T, fast_mode=fast)


def _call_stage_11(fast: bool = False) -> None:
    """Run Stage 11 Riccati invariant set verification."""
    from hdr_validation.stages.stage_11_invariant_set import run_stage_11
    n_seeds = 2 if fast else 5
    T = 32 if fast else 128
    run_stage_11(n_seeds=n_seeds, T=T, fast_mode=False)


def call_stage(mod: Any, stage_id: str, state: dict) -> None:
    """Call the appropriate stage function from the runner module."""
    fast = state.get("fast_mode", False)
    if stage_id == "01":
        mod.stage01_math()
    elif stage_id == "02":
        state["episodes"] = mod.stage02_generation()
    elif stage_id == "03":
        state["stage03_data"] = mod.stage03_imm(state["episodes"])
    elif stage_id == "03b":
        mod.stage03b_ici(state["stage03_data"])
    elif stage_id == "03c":
        mod.stage03c_mode_c()
    elif stage_id == "04":
        mod.stage04_mode_a(state["episodes"])
    elif stage_id == "05":
        mod.stage05_mode_b()
    elif stage_id == "06":
        mod.stage06_coherence()
    elif stage_id == "07":
        mod.stage07_robustness()
    elif stage_id == "08":
        _call_stage_08(fast=fast)
    elif stage_id == "08b":
        _call_stage_08b(fast=fast)
    elif stage_id == "09":
        _call_stage_09(fast=fast)
    elif stage_id == "10":
        _call_stage_10(fast=fast)
    elif stage_id == "11":
        _call_stage_11(fast=fast)
    elif stage_id == "12":
        _call_stage_12(fast=fast)
    elif stage_id == "13":
        _call_stage_13(fast=fast)
    elif stage_id == "14":
        _call_stage_14(fast=fast)
    elif stage_id == "15":
        _call_stage_15(fast=fast)
    elif stage_id == "16":
        _call_stage_16(fast=fast)
    elif stage_id == "17":
        _call_stage_17(fast=fast)
    elif stage_id == "18":
        _call_stage_18(fast=fast)
    elif stage_id == "18b":
        _call_stage_18b(fast=fast)
    else:
        raise ValueError(f"Unknown stage: {stage_id!r}")


# ── Test runner ────────────────────────────────────────────────────────────

def run_stage_tests(stage_id: str) -> tuple[bool, str]:
    """Run the pytest test file for a stage, if one exists.

    Skips production-scale tests (marked with 'production' in name) to keep
    runtime reasonable.  Returns (passed, summary_message).
    """
    import subprocess

    test_file = STAGE_TEST_FILES.get(stage_id)
    if test_file is None:
        return True, f"No test file for stage {stage_id}"

    test_path = ROOT / test_file
    if not test_path.exists():
        return False, f"Test file {test_file} not found"

    cmd = [
        sys.executable, "-m", "pytest",
        str(test_path),
        "-v", "-x",
        "-k", "not production",
    ]
    print(f"\n  Running tests: {test_file} (excluding production-scale)")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    passed = result.returncode == 0
    if passed:
        summary = f"{test_file}: all tests passed"
    else:
        summary = f"{test_file}: tests FAILED (exit code {result.returncode})"
    return passed, summary


# ── Profile runner ─────────────────────────────────────────────────────────────

def run_profile(
    profile: str,
    stages_to_run: list[str],
    force: bool,
    skip_done: bool,
    manifest: dict,
    fast: bool = False,
    run_tests: bool = False,
) -> dict[str, list[dict]]:
    """
    Run the requested stages for a profile.

    Returns a mapping of stage_id -> list of check records.
    Dependency stages that aren't in stages_to_run are run silently
    (their checks are collected but marked as dependency-only).
    """
    module_name = PROFILE_MODULES[profile]
    mod = importlib.import_module(module_name)

    # Reset the module's results list for a clean run
    mod.results.clear()

    # Expand with dependencies; track which stages are "user-requested"
    full_sequence = resolve_with_deps(stages_to_run)
    requested_set = set(stages_to_run)
    dep_only_set = set(full_sequence) - requested_set

    print(f"\n{'#'*60}")
    print(f"  Profile: {profile.upper()}")
    print(f"  Stages:  {', '.join(full_sequence)}")
    print(f"{'#'*60}")

    # Stages 08-11 run with fast_mode=True for smoke profile or --fast flag
    fast_mode = fast or profile in ("smoke",)
    state: dict[str, Any] = {"episodes": None, "stage03_data": None, "fast_mode": fast_mode}
    stage_results: dict[str, list[dict]] = {}

    # Stages that are profile-independent (have their own simulation logic)
    INDEPENDENT_STAGES = {"08", "08b", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "18b"}

    for stage_id in full_sequence:
        label = STAGE_LABELS[stage_id]
        is_dep = stage_id in dep_only_set

        if skip_done and not force and is_done(manifest, profile, stage_id):
            print(f"\n  [SKIP] {label} (already done)")
            continue

        print(f"\n{'='*60}")
        if is_dep:
            print(f"  {label}  [dependency]")
        else:
            print(f"  {label}")
        print(f"{'='*60}")

        # Snapshot results count before stage to isolate this stage's records
        idx_before = len(mod.results)
        t0 = time.perf_counter()
        stage_exception = None

        try:
            call_stage(mod, stage_id, state)
            elapsed = time.perf_counter() - t0
            print(f"  Elapsed: {elapsed:.2f}s")
            mark_done(manifest, profile, stage_id)
            save_manifest(manifest)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            traceback.print_exc()
            print(f"  [ERROR] Stage {stage_id} raised an exception after {elapsed:.2f}s")
            mark_failed(manifest, profile, stage_id)
            save_manifest(manifest)
            stage_exception = exc

        # For profile-independent stages (08-11), add a synthetic result record
        # since they don't append to mod.results directly
        if stage_id in INDEPENDENT_STAGES:
            mod.results.append({
                "stage": stage_id,
                "check": "stage_execution",
                "passed": stage_exception is None,
                "value": "OK" if stage_exception is None else str(stage_exception),
                "note": f"Profile-independent stage {stage_id}",
            })

        # Optionally run pytest test file for this stage
        if run_tests and stage_id in STAGE_TEST_FILES:
            test_passed, test_summary = run_stage_tests(stage_id)
            mod.results.append({
                "stage": stage_id,
                "check": "pytest",
                "passed": test_passed,
                "value": test_summary,
                "note": f"pytest {STAGE_TEST_FILES[stage_id]}",
            })
            if not test_passed:
                print(f"  [FAIL] {test_summary}")
            else:
                print(f"  [PASS] {test_summary}")

        stage_results[stage_id] = list(mod.results[idx_before:])

    return stage_results


# ── Full-validation mode ──────────────────────────────────────────────────────

def _run_highpower_stage_04() -> dict[str, Any]:
    """Run the highpower benchmark (Stage 04) and return a summary dict."""
    from highpower_runner import run_highpower_benchmark
    return run_highpower_benchmark()


def _run_all_unit_tests() -> list[dict]:
    """Run the full pytest suite (excluding production-scale tests).

    Returns a list of check records.
    """
    import subprocess

    cmd = [
        sys.executable, "-m", "pytest",
        str(ROOT), "-v", "--tb=short",
        "-k", "not production",
        "-q",
    ]
    from hdr_validation.defaults import HDR_VERSION
    print(f"\n  Running full pytest suite (HDR v{HDR_VERSION}, excluding production-scale)...")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.stderr:
        # Only print last portion of stderr to avoid noise
        print(result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)

    passed = result.returncode == 0
    return [{
        "stage": "unit_tests",
        "check": "pytest_full_suite",
        "passed": passed,
        "value": "all passed" if passed else f"FAILED (exit code {result.returncode})",
        "note": "Full pytest suite (excl. production-scale)",
    }]


def run_full_validation(
    force: bool = False,
    skip_done: bool = False,
    manifest: dict | None = None,
) -> int:
    """Run the complete validation covering all 36 claims.

    Execution plan:
      1. Extended profile for stages 01-03c, 05-07 (Claims 3-14)
      2. Highpower benchmark for stage 04 (Claims 1-2, authoritative)
      3. Stages 08-16 at production parameters (Claims 9, 13, 15-36, Stage 18)
      4. Full pytest suite for unit-test-validated claims (15-36)

    Returns number of failures.
    """
    if manifest is None:
        manifest = {}

    all_results: dict[str, dict[str, list[dict]]] = {}
    t_total = time.perf_counter()

    print("\n" + "=" * 62)
    print("  FULL VALIDATION MODE — All 36 Claims")
    print("=" * 62)

    # ── Phase 1: Extended profile, stages 01-03c + 05-07 (Claims 3-14) ────
    print("\n" + "#" * 62)
    print("  Phase 1: Extended profile — Stages 01-03c, 05-07")
    print("  (Claims 3-14: mathematical, inference, exploration,")
    print("   coherence, robustness)")
    print("#" * 62)

    phase1_stages = ["01", "02", "03", "03b", "03c", "05", "06", "07"]
    all_results["extended"] = run_profile(
        profile="extended",
        stages_to_run=phase1_stages,
        force=force,
        skip_done=skip_done,
        manifest=manifest,
        fast=False,
        run_tests=False,
    )

    # ── Phase 2: Highpower benchmark for stage 04 (Claims 1-2) ────────────
    print("\n" + "#" * 62)
    print("  Phase 2: Highpower Benchmark A — Stage 04")
    print("  (Claims 1-2: maladaptive-basin cost reduction, win rate)")
    print("#" * 62)

    hp_key = manifest_key("highpower", "04")
    hp_results: list[dict] = []

    if skip_done and not force and manifest.get(hp_key) == "done":
        print("\n  [SKIP] Highpower Stage 04 (already done)")
    else:
        t_hp = time.perf_counter()
        try:
            hp_summary = _run_highpower_stage_04()

            # Extract check records from the highpower summary
            gain = hp_summary.get("hdr_vs_pe_maladaptive_mean", 0.0)
            ci_lo = hp_summary.get("ci_95_mean_lo", 0.0)
            win_rate = hp_summary.get("hdr_mal_win_rate", 0.0)
            n_mal = hp_summary.get("n_maladaptive_episodes", 0)
            criterion_met = hp_summary.get("criterion_plus3pct_satisfied_95ci_mean", False)

            hp_results.append({
                "stage": "04",
                "check": "claim_1_cost_reduction",
                "passed": bool(criterion_met),
                "value": f"gain={gain:+.4f}, 95% CI lower={ci_lo:+.4f}, N_mal={n_mal}",
                "note": "Claim 1: gain >= +3%, 95% CI lower >= +0.03",
            })
            hp_results.append({
                "stage": "04",
                "check": "claim_2_win_rate",
                "passed": bool(win_rate >= 0.70),
                "value": f"win_rate={win_rate:.3f}",
                "note": "Claim 2: win rate >= 70%",
            })

            manifest[hp_key] = "done"
            save_manifest(manifest)
            elapsed_hp = time.perf_counter() - t_hp
            n_hp_pass = sum(1 for c in hp_results if c["passed"])
            print(f"  Highpower Stage 04: {n_hp_pass}/{len(hp_results)} checks passed ({elapsed_hp:.1f}s)")
        except Exception as exc:
            traceback.print_exc()
            hp_results.append({
                "stage": "04",
                "check": "highpower_execution",
                "passed": False,
                "value": str(exc),
                "note": "Highpower benchmark raised an exception",
            })
            manifest[hp_key] = "failed"
            save_manifest(manifest)

    all_results["highpower"] = {"04": hp_results}

    # ── Phase 3: Stages 08-16, production parameters (Claims 9, 13, 15-36, Stage 18) ─
    print("\n" + "#" * 62)
    print("  Phase 3: Profile-independent stages 08-18")
    print("  (Claims 9, 13, 15-34: ablation, baselines, v7.0/v7.1/v7.5)")
    print("#" * 62)

    # Use a dummy extended profile for the independent stages — they only
    # need the module loaded so call_stage can dispatch to the _call_stage_*
    # functions. The stages 08-18 don't use the profile module.
    phase3_stages = ["08", "08b", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "18b"]
    phase3_results = run_profile(
        profile="extended",
        stages_to_run=phase3_stages,
        force=force,
        skip_done=skip_done,
        manifest=manifest,
        fast=False,
        run_tests=True,
    )
    # Merge phase3 results into the extended results
    for stage_id, records in phase3_results.items():
        all_results["extended"][stage_id] = records

    # ── Phase 4: Full unit test suite ──────────────────────────────────────
    print("\n" + "#" * 62)
    print("  Phase 4: Full pytest suite")
    print("  (Unit-test coverage for Claims 15-36)")
    print("#" * 62)

    unit_test_records = _run_all_unit_tests()
    all_results["unit_tests"] = {"all": unit_test_records}

    # ── Claim coverage summary ─────────────────────────────────────────────
    elapsed_total = time.perf_counter() - t_total
    n_fail = _print_full_validation_summary(all_results, elapsed_total)
    return n_fail


def _print_full_validation_summary(
    all_results: dict[str, dict[str, list[dict]]],
    elapsed: float,
) -> int:
    """Print the full-validation claim coverage summary. Returns failure count."""
    print("\n" + "=" * 62)
    print("  FULL VALIDATION — CLAIM COVERAGE SUMMARY")
    print("=" * 62)

    # Collect all checks by stage
    stage_checks: dict[str, list[dict]] = {}
    for profile, stage_map in all_results.items():
        for stage_id, records in stage_map.items():
            stage_checks.setdefault(stage_id, []).extend(records)

    # Per-claim status
    total_pass = total_fail = 0
    claim_statuses: list[tuple[int, bool, str]] = []

    for claim_id in range(1, 37):
        stages = CLAIM_STAGES[claim_id]
        claim_checks: list[dict] = []
        for s in stages:
            claim_checks.extend(stage_checks.get(s, []))

        if not claim_checks:
            claim_statuses.append((claim_id, False, "NO CHECKS RUN"))
            total_fail += 1
            continue

        n_pass = sum(1 for c in claim_checks if c["passed"])
        n_total = len(claim_checks)
        all_passed = n_pass == n_total

        if all_passed:
            claim_statuses.append((claim_id, True, f"{n_pass}/{n_total} checks passed"))
            total_pass += 1
        else:
            claim_statuses.append((claim_id, False, f"{n_pass}/{n_total} checks passed"))
            total_fail += 1

    # Also count unit tests
    for records in stage_checks.get("all", []):
        if records["passed"]:
            total_pass += 1
        else:
            total_fail += 1

    # Print claim table
    print(f"\n  {'Claim':<8} {'Status':<10} {'Detail'}")
    print(f"  {'─'*8} {'─'*10} {'─'*40}")
    for claim_id, passed, detail in claim_statuses:
        status = "PASS" if passed else "FAIL"
        marker = " " if passed else "*"
        source = "highpower" if claim_id in (1, 2) else "extended+stages"
        print(f"  {marker}{claim_id:<7} {status:<10} {detail:<30} [{source}]")

    # Print unit test status
    ut_records = stage_checks.get("all", [])
    if ut_records:
        ut_passed = all(r["passed"] for r in ut_records)
        ut_status = "PASS" if ut_passed else "FAIL"
        print(f"\n  {'Unit tests':<18} {ut_status:<10} {ut_records[0]['value']}")

    # Summary line
    print(f"\n  {'─'*62}")
    print(f"  Claims: {total_pass} passed, {total_fail} failed out of {total_pass + total_fail}")
    print(f"  Elapsed: {elapsed:.1f}s")

    if total_fail == 0:
        print("\n  ALL 36 CLAIMS VALIDATED")
    else:
        print(f"\n  {total_fail} CLAIM(S) FAILED — see details above")
        # Print failed claims
        for claim_id, passed, detail in claim_statuses:
            if not passed:
                stages = CLAIM_STAGES[claim_id]
                print(f"    Claim {claim_id} (stages {', '.join(stages)}): {detail}")

    return total_fail


# ── Summary printing ──────────────────────────────────────────────────────────

def print_summary(all_results: dict[str, dict[str, list[dict]]]) -> int:
    """Print overall summary. Returns number of failures."""
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")

    total_pass = total_fail = 0

    for profile, stage_map in all_results.items():
        print(f"\n  Profile: {profile}")
        for stage_id, records in stage_map.items():
            n_pass = sum(1 for r in records if r["passed"])
            n_fail = len(records) - n_pass
            status = "✓" if n_fail == 0 else "✗"
            print(f"    {status} {STAGE_LABELS[stage_id]}: {n_pass}/{len(records)} passed")
            total_pass += n_pass
            total_fail += n_fail

    print(f"\n  Total: {total_pass} passed, {total_fail} failed out of {total_pass + total_fail} checks")

    if total_fail == 0:
        print("\n  ✓ ALL CHECKS PASSED")
    else:
        print(f"\n  ✗ {total_fail} CHECKS FAILED")
        print("\n  Failed checks:")
        for profile, stage_map in all_results.items():
            for stage_id, records in stage_map.items():
                for r in records:
                    if not r["passed"]:
                        print(f"    [{profile}/{stage_id}] {r['check']}: {r['value']} ({r['note']})")

    return total_fail


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HDR Validation Suite — run all stages across profiles"
    )
    parser.add_argument(
        "--profiles", nargs="+",
        default=list(PROFILE_MODULES),
        choices=list(PROFILE_MODULES),
        metavar="PROFILE",
        help="Profiles to run (default: all)",
    )
    parser.add_argument(
        "--stages", nargs="+",
        default=STAGE_SEQUENCE,
        metavar="STAGE",
        help="Stage IDs to run, e.g. 01 03b 04 (default: all)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run stages even if marked done in manifest",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Load existing manifest (implies using checkpoint state)",
    )
    parser.add_argument(
        "--skip-done", action="store_true",
        help="Skip stages already marked done in manifest",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Run stages 08-18 with reduced parameters for smoke testing",
    )
    parser.add_argument(
        "--run-tests", action="store_true",
        help="Run pytest test files for each stage that has one (skips production-scale tests)",
    )
    parser.add_argument(
        "--full-validation", action="store_true",
        help="Run complete validation of all 36 claims: extended profile for "
             "stages 01-07 (Claims 3-14), highpower benchmark for stage 04 "
             "(Claims 1-2), stages 08-18 at production scale (Claims 9, 13, "
             "15-36), and full pytest suite",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Full-validation mode ──────────────────────────────────────────────
    if args.full_validation:
        manifest = load_manifest() if (args.resume or args.skip_done) else {}
        print("HDR Validation Suite — FULL VALIDATION MODE")
        print("  Covers all 36 claims with authoritative highpower statistics")
        n_fail = run_full_validation(
            force=args.force,
            skip_done=args.skip_done,
            manifest=manifest,
        )
        sys.exit(0 if n_fail == 0 else 1)

    # ── Standard mode ─────────────────────────────────────────────────────
    # Validate stage IDs
    unknown = [s for s in args.stages if s not in STAGE_SEQUENCE]
    if unknown:
        print(f"[ERROR] Unknown stage(s): {unknown}")
        print(f"  Valid stages: {STAGE_SEQUENCE}")
        sys.exit(2)

    manifest = load_manifest() if (args.resume or args.skip_done) else {}

    print("HDR Validation Suite")
    print(f"  Profiles  : {args.profiles}")
    print(f"  Stages    : {args.stages}")
    print(f"  Force     : {args.force}")
    print(f"  SkipDone  : {args.skip_done}")
    print(f"  RunTests  : {args.run_tests}")

    all_results: dict[str, dict[str, list[dict]]] = {}

    for profile in args.profiles:
        all_results[profile] = run_profile(
            profile=profile,
            stages_to_run=args.stages,
            force=args.force,
            skip_done=args.skip_done,
            manifest=manifest,
            fast=args.fast,
            run_tests=args.run_tests,
        )

    n_fail = print_summary(all_results)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
