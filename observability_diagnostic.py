"""Stage 15 Observability Diagnostic — P0.5.1

Compute observability Gramian for each basin's (A_k, C_k) pair and determine
whether latent axes (E, mito, P — indices 2, 3, 4) are observable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hdr_validation.model.slds import make_evaluation_model
from hdr_validation.defaults import DEFAULTS

ROOT = Path(__file__).parent


def main() -> None:
    cfg = dict(DEFAULTS)
    cfg["max_dwell_len"] = 64
    rng = np.random.default_rng(101)
    model = make_evaluation_model(cfg, rng)

    n = cfg["state_dim"]  # 8
    m = cfg["obs_dim"]  # 16
    axis_names = {
        0: "HPA", 1: "immune", 2: "E", 3: "mito",
        4: "P", 5: "sleep", 6: "metab", 7: "cardio",
    }
    latent_axes = [2, 3, 4]

    results: dict = {"state_dim": n, "obs_dim": m, "basins": {}}

    print("=" * 60)
    print("  OBSERVABILITY DIAGNOSTIC")
    print("=" * 60)
    print(f"  state_dim = {n}, obs_dim = {m}")
    print(f"  Latent axes: E(2), mito(3), P(4)")
    print()

    for k, basin in enumerate(model.basins):
        A_k = basin.A
        C_k = basin.C

        # ── Observability matrix ──────────────────────────────────────────
        O = np.vstack([C_k @ np.linalg.matrix_power(A_k, i) for i in range(n)])
        rank_O = np.linalg.matrix_rank(O)
        svs_O = np.linalg.svd(O, compute_uv=False)
        cond_O = float(svs_O[0] / svs_O[-1]) if svs_O[-1] > 1e-15 else float("inf")

        # ── Observability Gramian ─────────────────────────────────────────
        # W_o = sum_{i=0}^{n-1} (A^T)^i C^T C A^i
        W_o = np.zeros((n, n))
        Ai = np.eye(n)
        for i in range(n):
            W_o += Ai.T @ C_k.T @ C_k @ Ai
            Ai = A_k @ Ai

        gramian_eigs = np.sort(np.linalg.eigvalsh(W_o))[::-1]

        # ── Per-axis observability ────────────────────────────────────────
        axis_info: dict[int, dict] = {}
        for idx in range(n):
            col_norm = float(np.linalg.norm(O[:, idx]))
            gramian_diag = float(W_o[idx, idx])
            axis_info[idx] = {"col_norm": col_norm, "gramian_diag": gramian_diag}

        # ── Print report ──────────────────────────────────────────────────
        print(f"Basin {k} (rho={basin.rho:.2f}):")
        print(f"  rank(O) = {rank_O} / {n}  ({'FULL' if rank_O == n else 'DEFICIENT'})")
        print(f"  cond(O) = {cond_O:.2f}")
        print(f"  O singular values: [{', '.join(f'{s:.4f}' for s in svs_O)}]")
        print(f"  Gramian eigenvalues: [{', '.join(f'{e:.4f}' for e in gramian_eigs)}]")
        print(f"  Gramian min eig: {gramian_eigs[-1]:.6f}")
        print()

        hdr = f"  {'Axis':<18} {'||O[:,i]||':>10} {'W_o[i,i]':>12} {'Status':>12}"
        print(hdr)
        print(f"  {'-'*18} {'-'*10} {'-'*12} {'-'*12}")
        for idx in range(n):
            name = axis_names.get(idx, f"axis_{idx}")
            cn = axis_info[idx]["col_norm"]
            gd = axis_info[idx]["gramian_diag"]
            latent_tag = " (LATENT)" if idx in latent_axes else ""
            if cn > 1.0:
                status = "STRONG"
            elif cn > 0.01:
                status = "WEAK"
            else:
                status = "UNOBSERVABLE"
            print(f"  {name + latent_tag:<18} {cn:10.4f} {gd:12.4f} {status:>12}")
        print()

        # ── Store results ─────────────────────────────────────────────────
        results["basins"][str(k)] = {
            "rho": float(basin.rho),
            "rank_O": int(rank_O),
            "cond_O": cond_O,
            "O_singular_values": svs_O.tolist(),
            "gramian_eigenvalues": gramian_eigs.tolist(),
            "gramian_min_eig": float(gramian_eigs[-1]),
            "per_axis": {
                axis_names.get(i, f"axis_{i}"): {
                    "index": i,
                    "O_col_norm": float(axis_info[i]["col_norm"]),
                    "gramian_diag": float(axis_info[i]["gramian_diag"]),
                    "latent": i in latent_axes,
                }
                for i in range(n)
            },
        }

    # ── Verdict ───────────────────────────────────────────────────────────
    n_basins = len(model.basins)
    all_full_rank = all(
        results["basins"][str(k)]["rank_O"] == n for k in range(n_basins)
    )
    latent_observable = all(
        results["basins"][str(k)]["per_axis"][axis_names[idx]]["O_col_norm"] > 0.01
        for k in range(n_basins)
        for idx in latent_axes
    )

    if all_full_rank:
        verdict = "OBSERVABLE"
        proceed = "YES"
    elif latent_observable:
        verdict = "WEAKLY OBSERVABLE"
        proceed = "CONDITIONAL"
    else:
        verdict = "UNOBSERVABLE"
        proceed = "NO"

    results["verdict"] = verdict
    results["proceed_to_kalman"] = proceed

    print("=" * 60)
    print(f"  VERDICT: {verdict}")
    print(f"  Proceed to Kalman filter implementation: {proceed}")
    print("=" * 60)

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = ROOT / "results" / "stage_15_observability_diagnostic.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
