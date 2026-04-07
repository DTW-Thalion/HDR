"""Provenance metadata for result artifacts."""
from __future__ import annotations

import datetime
import subprocess
from typing import Any

from .defaults import HDR_VERSION


def get_provenance() -> dict[str, Any]:
    """Return a provenance dict to embed in every result JSON."""
    prov: dict[str, Any] = {
        "hdr_version": HDR_VERSION,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    # Attempt git commit hash
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            prov["git_commit"] = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return prov
