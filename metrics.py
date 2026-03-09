from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..generator.ground_truth import SyntheticEnv, default_scenarios
from ..model.slds import make_evaluation_model
from ..plotting import save_bar_plot, save_heatmap, save_line_plot
from ..utils import atomic_save_npz, atomic_write_dataframe_csv, atomic_write_json, ensure_dir, seed_everything
from .common import save_experiment_bundle, summarize_metric_rows


def random_identification_policy(rng: np.random.Generator, control_dim: int, dither_scale: float = 0.12):
    def _policy(obs):
        base = rng.normal(scale=dither_scale, size=control_dim)
        return np.clip(base, -0.35, 0.35)
    return _policy


def open_loop_policy(control_dim: int):
    def _policy(obs):
        return np.zeros(control_dim)
    return _policy


def run_dataset(env: SyntheticEnv, policy_fn):
    obs = env.reset()
    out = {"x_true": [], "z_true": [], "y": [], "mask": [], "u": [], "target_low": [], "target_high": [], "kappa_true": []}
    for _ in range(env.T):
        u = np.asarray(policy_fn(obs), dtype=float)
        out["x_true"].append(obs["x_true"])
        out["z_true"].append(obs["z_true"])
        out["y"].append(np.nan_to_num(obs["y"], nan=0.0))
        out["mask"].append(obs["mask"])
        out["u"].append(u)
        out["target_low"].append(obs["target"].box_low)
        out["target_high"].append(obs["target"].box_high)
        out["kappa_true"].append(obs["kappa_true"])
        obs, info = env.step(u)
        if info["done"]:
            break
    return {k: np.asarray(v) for k, v in out.items()}


def run(project_root: Path, profile_name: str, config: dict) -> dict:
    stage_root = ensure_dir(project_root / "results" / "stage_02" / profile_name)
    plots_dir = ensure_dir(stage_root / "plots")
    rng_master = seed_everything(int(config["seeds"][0]))
    eval_model = make_evaluation_model(config, rng_master)
    scenarios = default_scenarios()
    split_defs = {
        "train": ("nominal", random_identification_policy),
        "validation": ("model_mismatch", random_identification_policy),
        "test": ("missing_heterosk", open_loop_policy),
        "challenge": ("heavy_tail", random_identification_policy),
    }
    manifests = []
    stage_rows = []
    selected_trace_cap = int(config["selected_trace_cap"])
    traces = {}
    for split, (scenario_name, policy_factory) in split_defs.items():
        split_dir = ensure_dir(stage_root / split)
        rows = []
        for epi, seed in enumerate(config["seeds"]):
            rng = np.random.default_rng(int(seed) + 100 * epi + hash(split) % 1000)
            env = SyntheticEnv(eval_model, config, rng, scenarios[scenario_name], episode_idx=epi)
            policy_fn = policy_factory(rng, eval_model.control_dim) if policy_factory is random_identification_policy else policy_factory(eval_model.control_dim)
            data = run_dataset(env, policy_fn)
            file_path = split_dir / f"episode_{epi:03d}.npz"
            atomic_save_npz(file_path, **data)
            rows.append({
                "split": split,
                "episode": epi,
                "seed": int(seed),
                "scenario": scenario_name,
                "steps": int(len(data["z_true"])),
                "mean_abs_state": float(np.mean(np.abs(data["x_true"]))),
                "mode1_frac": float(np.mean(data["z_true"] == 1)),
                "missing_frac": float(1.0 - np.mean(data["mask"])),
            })
            if epi < selected_trace_cap:
                traces[f"{split}_x_{epi}"] = data["x_true"]
                traces[f"{split}_z_{epi}"] = data["z_true"]
                traces[f"{split}_u_{epi}"] = data["u"]
        atomic_write_dataframe_csv(split_dir / "manifest.csv", pd.DataFrame(rows))
        atomic_write_json(split_dir / "manifest.json", {"rows": rows})
        manifests.extend(rows)
        stage_rows.extend(rows)
    # plots
    df = pd.DataFrame(stage_rows)
    mode_occ = df.groupby("split")["mode1_frac"].mean().reindex(list(split_defs.keys()))
    miss = df.groupby("split")["missing_frac"].mean().reindex(list(split_defs.keys()))
    save_bar_plot(plots_dir / "mode1_frac.png", mode_occ.index.tolist(), mode_occ.values.tolist(), "Mean maladaptive occupancy by split", "fraction")
    save_bar_plot(plots_dir / "missingness.png", miss.index.tolist(), miss.values.tolist(), "Mean missingness by split", "fraction")
    mat = np.array([[r["mean_abs_state"], r["mode1_frac"], r["missing_frac"]] for r in stage_rows[: min(12, len(stage_rows))]])
    save_heatmap(plots_dir / "generator_diag_heatmap.png", mat, ["|x|", "mode1", "missing"], [f"{r['split']}-{r['episode']}" for r in stage_rows[: min(12, len(stage_rows))]], "Generator diagnostics sample", "value")
    summary = summarize_metric_rows(stage_rows, key_fields=["split", "scenario"])
    summary.update({
        "n_files": len(list(stage_root.rglob("*.npz"))),
        "splits": list(split_defs.keys()),
        "datasets_saved_compressed": True,
    })
    save_experiment_bundle(
        stage_root / "dataset_creation",
        config=config,
        seed=config["seeds"],
        summary=summary,
        metrics_rows=stage_rows,
        selected_traces=traces,
        log_text="Stage 02 completed.",
    )
    atomic_write_json(stage_root / "stage_summary.json", summary)
    return summary
