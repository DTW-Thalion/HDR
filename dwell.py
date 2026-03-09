from __future__ import annotations


def try_run_variational_slds(*args, **kwargs) -> dict:
    return {
        "status": "skipped",
        "reason": "Optional heavy variational SLDS/HSMM path not enabled in this environment.",
    }
