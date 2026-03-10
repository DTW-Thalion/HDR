import subprocess, sys, zipfile, tempfile, pathlib

import numpy as np

from hdr_validation.control.mpc import solve_mode_a
from hdr_validation.model.slds import make_evaluation_model
from hdr_validation.model.target_set import build_target_set


def test_mpc_returns_bounded_control():
    config = {
        "state_dim": 8,
        "obs_dim": 16,
        "control_dim": 8,
        "disturbance_dim": 8,
        "K": 3,
        "rho_reference": [0.72, 0.96, 0.55],
        "max_dwell_len": 128,
        "H": 6,
        "kappa_lo": 0.55,
        "kappa_hi": 0.75,
        "w1": 1.0,
        "w2": 0.5,
        "w3": 0.3,
        "lambda_u": 0.1,
        "alpha_i": 0.05,
        "eps_safe": 0.5,
        "steps_per_day": 48,
        "default_burden_budget": 14.0,
        "circadian_locked_controls": [5, 6],
    }
    rng = np.random.default_rng(2)
    model = make_evaluation_model(config, rng)
    target = build_target_set(0, config)
    x = np.ones(8)
    P = np.eye(8) * 0.1
    res = solve_mode_a(x, P, model.basins[0], target, 0.4, config, step=0)
    assert res.u.shape == (8,)
    assert np.all(np.abs(res.u) <= 0.6 + 1e-8)


class TestZipRoundTrip:
    """Verify that a clean unzip produces a working hdr_validation import."""

    def test_zip_import_survives_roundtrip(self, tmp_path):
        # 1. Create a zip of hdr_validation/ package only
        repo_root = pathlib.Path(__file__).parent
        pkg_dir = repo_root / "hdr_validation"
        zip_path = tmp_path / "hdr_test.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in pkg_dir.rglob("*.py"):
                zf.write(f, f.relative_to(repo_root))

        # 2. Unpack into a fresh directory
        unpack_dir = tmp_path / "unpacked"
        unpack_dir.mkdir()
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(unpack_dir)

        # 3. Verify hdr_validation/ is a directory, not a file
        extracted_pkg = unpack_dir / "hdr_validation"
        assert extracted_pkg.is_dir(), (
            f"hdr_validation should be a directory after unzip, got: "
            f"{type(extracted_pkg)}"
        )
        assert (extracted_pkg / "__init__.py").exists(), (
            "hdr_validation/__init__.py missing after unzip"
        )

        # 4. Run a subprocess import from the unpacked directory
        result = subprocess.run(
            [sys.executable, "-c",
             "import hdr_validation; "
             "from hdr_validation.inference.ici import compute_T_k_eff; "
             "print('ok')"],
            cwd=unpack_dir,
            capture_output=True, text=True
        )
        assert result.returncode == 0, (
            f"Import failed after unzip:\n{result.stderr}"
        )
        assert "ok" in result.stdout
