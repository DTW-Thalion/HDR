from __future__ import annotations

__version__ = "7.4.0"

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# from ..utils import (
#     atomic_save_npz,
#     atomic_write_csv,
#     atomic_write_dataframe_csv,
#     atomic_write_json,
#     atomic_write_text,
#     ensure_dir,
# )


def save_experiment_bundle(
    exp_dir: Path,
    config: dict[str, Any],
    seed: int | list[int] | None,
    summary: dict[str, Any],
    metrics_rows: list[dict[str, Any]],
    selected_traces: dict[str, Any] | None = None,
    log_text: str = "",
) -> dict[str, Any]:
    ensure_dir(exp_dir)
    atomic_write_json(exp_dir / "config.json", config)
    atomic_write_json(exp_dir / "seed.json", {"seed": seed})
    atomic_write_json(exp_dir / "summary.json", summary)
    atomic_write_csv(exp_dir / "metrics.csv", metrics_rows)
    if selected_traces is None:
        selected_traces = {"empty": np.array([])}
    atomic_save_npz(exp_dir / "selected_traces.npz", **selected_traces)
    atomic_write_text(exp_dir / "log.txt", log_text)
    fragment = {
        "experiment": exp_dir.name,
        "files": sorted(str(p.relative_to(exp_dir)) for p in exp_dir.rglob("*") if p.is_file()),
    }
    atomic_write_json(exp_dir / "manifest_fragment.json", fragment)
    return fragment


def summarize_metric_rows(rows: list[dict[str, Any]], key_fields: list[str] | None = None) -> dict[str, Any]:
    if not rows:
        return {"n_rows": 0}
    df = pd.DataFrame(rows)
    out: dict[str, Any] = {"n_rows": int(len(df))}
    key_fields = key_fields or []
    for col in df.columns:
        if col in key_fields:
            continue
        if np.issubdtype(df[col].dtype, np.number):
            out[f"{col}_mean"] = float(df[col].mean())
            out[f"{col}_std"] = float(df[col].std(ddof=0))
    for kf in key_fields:
        if kf in df.columns:
            out[f"levels_{kf}"] = sorted(df[kf].astype(str).unique().tolist())
    return out
