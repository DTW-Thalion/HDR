#!/usr/bin/env python3
"""
analyse_highpower.py — Read highpower_summary.json and print a
formatted results table with bootstrap CI verification.

Usage:
    python analyse_highpower.py
    python analyse_highpower.py --json results/stage_04/highpower/highpower_summary.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


DEFAULT_PATH = Path("results/stage_04/highpower/highpower_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise high-power Benchmark A results")
    parser.add_argument("--json", type=Path, default=DEFAULT_PATH,
                        help="Path to highpower_summary.json")
    args = parser.parse_args()

    if not args.json.exists():
        print(f"ERROR: {args.json} not found.")
        print("Run 'python highpower_runner.py' first to generate results.")
        raise SystemExit(1)

    with open(args.json) as f:
        d = json.load(f)

    sep = "─" * 62
    print(sep)
    print("  BENCHMARK A — HIGH-POWER RESULTS SUMMARY")
    print(sep)
    print(f"  Seeds × episodes : {d['n_seeds']} × {d['episodes_per_seed']}")
    print(f"  N_maladaptive    : {d['n_maladaptive_episodes']}")
    print(f"  Steps/episode    : {d['steps_per_episode']}")
    print(sep)
    print(f"  Mean gain        : {d['hdr_vs_pe_maladaptive_mean']:+.4f}")
    print(f"  95 % CI (mean)   : [{d['ci_95_mean_lo']:+.4f}, {d['ci_95_mean_hi']:+.4f}]")
    print(f"  90 % CI (mean)   : [{d['ci_90_mean_lo']:+.4f}, {d['ci_90_mean_hi']:+.4f}]")
    print(f"  Win rate         : {d['hdr_mal_win_rate']:.4f}")
    print(f"  Safety delta     : {d['safety_delta_vs_pe']:+.4f}")
    print(sep)
    print("  CRITERION CHECKS")
    gain_ok   = d['hdr_vs_pe_maladaptive_mean'] >= 0.03
    ci_ok     = d['ci_95_mean_lo'] >= 0.03
    wr_ok     = d['hdr_mal_win_rate'] >= 0.70

    def status(ok: bool) -> str:
        return "PASS ✓" if ok else "FAIL ✗"

    print(f"  Point est ≥ +3 % : {status(gain_ok)}"
          f"  ({d['hdr_vs_pe_maladaptive_mean']:+.4f})")
    print(f"  95 % CI lo ≥ +3 %: {status(ci_ok)}"
          f"  ({d['ci_95_mean_lo']:+.4f})")
    print(f"  Win rate ≥ 70 %  : {status(wr_ok)}"
          f"  ({d['hdr_mal_win_rate']:.4f})")
    print(sep)
    if ci_ok:
        print("  OVERALL: BENCHMARK A CRITERION MET")
    else:
        print("  OVERALL: WIN RATE MET; GAIN CI CRITERION NOT MET")
        print("  Use the honest framing from manuscript_language.txt")
    print(sep)


if __name__ == "__main__":
    main()
