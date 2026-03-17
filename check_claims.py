"""
Validate manuscript claims against result artifacts and pytest output.

Usage:
    python check_claims.py              # Check all claims
    python check_claims.py --verbose    # Show passing claims too
    python check_claims.py --fix-json   # Print corrected values for manuscript
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def resolve_key(data: dict, dotted_key: str):
    """Navigate a dot-separated key path: 'basins.1.containment_rate_rpi'."""
    for part in dotted_key.split("."):
        data = data[part]
    return data


def _run_pytest_collect() -> dict[str, int]:
    """Run pytest --collect-only and parse counts."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    output = result.stdout + result.stderr
    info: dict[str, int] = {}

    # Parse "295 tests collected" or "295 tests collected in 0.80s"
    m = re.search(r"(\d+)\s+tests?\s+collected", output)
    if m:
        info["total_collected"] = int(m.group(1))

    # Count unique test files from the collected items (lines like "test_foo.py::test_bar")
    files = set()
    for line in output.splitlines():
        if "::" in line:
            files.add(line.split("::")[0].strip())
    if files:
        info["test_files"] = len(files)

    return info


def _run_pytest_execute() -> dict[str, int]:
    """Run pytest -k 'not production' and parse pass/skip counts."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-k", "not production"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    output = result.stdout + result.stderr
    info: dict[str, int] = {}

    # Parse "293 passed, 2 deselected" or "293 passed"
    m = re.search(r"(\d+)\s+passed", output)
    if m:
        info["passed"] = int(m.group(1))

    m = re.search(r"(\d+)\s+deselected", output)
    if m:
        info["deselected"] = int(m.group(1))

    return info


def _run_pytest_collect_file(filename: str) -> int:
    """Collect tests from a single file."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", filename],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    m = re.search(r"(\d+)\s+tests?\s+collected", result.stdout + result.stderr)
    return int(m.group(1)) if m else 0


def check_all(verbose: bool = False, fix_json: bool = False) -> int:
    """Check all claims. Returns number of mismatches."""
    claims_path = ROOT / "manuscript_claims.json"
    if not claims_path.exists():
        print("ERROR: manuscript_claims.json not found")
        return 1

    with open(claims_path) as f:
        data = json.load(f)

    claims = data["claims"]
    n_checked = 0
    n_passed = 0
    n_skipped = 0
    mismatches: list[str] = []

    # Cache pytest results
    collect_info: dict[str, int] | None = None
    execute_info: dict[str, int] | None = None

    for name, spec in claims.items():
        # Skip informational claims
        if "note" in spec and "expected" not in spec and "expected_ge" not in spec:
            n_skipped += 1
            if verbose:
                print(f"  [SKIP] {name}: {spec.get('note', 'informational')}")
            continue

        actual = None
        expected = spec.get("expected")
        expected_ge = spec.get("expected_ge")
        expected_le = spec.get("expected_le")

        # Pytest-based claims
        if "source" in spec:
            source = spec["source"]
            if "collect-only" in source:
                if collect_info is None:
                    print("  Running pytest --collect-only...")
                    collect_info = _run_pytest_collect()
                if "test_stage_08b" in source or "test_" in name:
                    # File-specific collection
                    filename = source.split()[-1] if len(source.split()) > 2 else None
                    if filename:
                        actual = _run_pytest_collect_file(filename)
                elif "test_files" in name:
                    actual = collect_info.get("test_files")
                else:
                    actual = collect_info.get("total_collected")
            elif "not production" in source:
                if execute_info is None:
                    print("  Running pytest -k 'not production'...")
                    execute_info = _run_pytest_execute()
                if "passing" in name:
                    actual = execute_info.get("passed")
                elif "skipped" in name:
                    actual = execute_info.get("deselected")

        # Artifact-based claims
        elif "artifact" in spec:
            artifact_path = ROOT / spec["artifact"]
            if not artifact_path.exists():
                mismatches.append(
                    f"  [MISS] {name}: artifact not found: {spec['artifact']} "
                    f"(manuscript ref: {spec.get('manuscript_ref', 'N/A')})"
                )
                n_checked += 1
                continue
            with open(artifact_path) as f:
                artifact_data = json.load(f)
            try:
                actual = resolve_key(artifact_data, spec["key"])
            except (KeyError, TypeError) as e:
                mismatches.append(
                    f"  [MISS] {name}: key '{spec['key']}' not found in {spec['artifact']}: {e}"
                )
                n_checked += 1
                continue

        if actual is None:
            if verbose:
                print(f"  [SKIP] {name}: could not determine actual value")
            n_skipped += 1
            continue

        # Compare
        n_checked += 1
        ok = True
        if expected is not None:
            ok = actual == expected
        elif expected_ge is not None:
            ok = float(actual) >= float(expected_ge)
        elif expected_le is not None:
            ok = float(actual) <= float(expected_le)

        if ok:
            n_passed += 1
            if verbose:
                exp_str = expected if expected is not None else f">={expected_ge}" if expected_ge is not None else f"<={expected_le}"
                print(f"  [PASS] {name}: actual={actual}, expected={exp_str}")
        else:
            exp_str = expected if expected is not None else f">={expected_ge}" if expected_ge is not None else f"<={expected_le}"
            msg = (
                f"  [FAIL] {name}: actual={actual}, expected={exp_str} "
                f"(manuscript ref: {spec.get('manuscript_ref', 'N/A')})"
            )
            mismatches.append(msg)
            if fix_json:
                print(f"  FIX: {name} should be {actual}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Claims checked: {n_checked}")
    print(f"  Passed: {n_passed}")
    print(f"  Failed: {len(mismatches)}")
    print(f"  Skipped: {n_skipped}")
    print(f"{'='*60}")

    if mismatches:
        print("\nMismatches:")
        for m in mismatches:
            print(m)

    return len(mismatches)


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv
    fix_json = "--fix-json" in sys.argv
    n_fail = check_all(verbose=verbose, fix_json=fix_json)
    sys.exit(0 if n_fail == 0 else 1)
