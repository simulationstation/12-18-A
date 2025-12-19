#!/usr/bin/env python3
"""
Generate a replication report comparing two Loschmidt echo sweeps.

Outputs:
  - Spearman rho for baseline and replicate
  - Permutation test on replicate
  - Depth-window robustness grid
  - Scatter plot of alpha_echo (replicate vs baseline)
  - replication_report.txt summarizing PASS/FAIL/INCONCLUSIVE
"""

import argparse
import json
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_loschmidt_echo import (
    compute_alpha_table,
    compute_spearman,
    load_results,
    permutation_test,
    robustness_grid,
    save_perm_hist,
)


def scatter_alpha(baseline_df: pd.DataFrame, replicate_df: pd.DataFrame, path: str):
    merged = baseline_df[["subset_id", "alpha_echo"]].merge(
        replicate_df[["subset_id", "alpha_echo"]], on="subset_id", suffixes=("_base", "_repl")
    )
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    plt.figure(figsize=(5, 5))
    if merged.empty:
        plt.text(0.5, 0.5, "No overlapping subsets", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        return
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    plt.scatter(merged["alpha_echo_base"], merged["alpha_echo_repl"], alpha=0.7, edgecolors="k")
    finite_vals = pd.concat([merged["alpha_echo_base"], merged["alpha_echo_repl"]]).dropna()
    if not finite_vals.empty:
        min_val, max_val = finite_vals.min(), finite_vals.max()
        lims = [min_val, max_val]
        plt.plot(lims, lims, "r--", label="y=x")
        plt.xlim(lims)
        plt.ylim(lims)
    plt.xlabel("alpha_echo (baseline)")
    plt.ylabel("alpha_echo (replicate)")
    plt.title("Per-subset alpha_echo comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)


def decide_outcome(rho_base: float, rho_repl: float, perm_p: float, same_sign_robust: int) -> str:
    sign_match = np.sign(rho_base) == np.sign(rho_repl)
    if not sign_match:
        return "FAIL"
    if abs(rho_repl) >= 0.4 and perm_p <= 0.05 and same_sign_robust >= 2:
        return "PASS"
    if abs(rho_repl) < 0.2 and perm_p > 0.2:
        return "FAIL"
    return "INCONCLUSIVE"


def main():
    parser = argparse.ArgumentParser(description="Replication report for Loschmidt echo runs.")
    parser.add_argument("--baseline", required=True, help="Baseline sweep CSV.")
    parser.add_argument("--replicate", required=True, help="Replicate sweep CSV.")
    parser.add_argument("--outdir", required=True, help="Output directory for the report artifacts.")
    parser.add_argument("--fit-eps", type=float, default=1e-3, help="Fit epsilon for alpha.")
    parser.add_argument("--drop-shallow", type=int, default=0, help="Drop k shallow depths for main fit.")
    parser.add_argument("--drop-deep", type=int, default=0, help="Drop k deepest depths for main fit.")
    parser.add_argument("--robustness-shallow", type=str, default="0,1",
                        help="Comma list of shallow drops for robustness grid.")
    parser.add_argument("--robustness-deep", type=str, default="0",
                        help="Comma list of deep drops for robustness grid.")
    parser.add_argument("--perm-count", type=int, default=5000, help="Permutation iterations.")
    parser.add_argument("--perm-seed", type=int, default=0, help="Permutation RNG seed.")
    parser.add_argument("--filter-N", type=int, default=None, help="Restrict to this N.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    base_df = load_results(args.baseline)
    repl_df = load_results(args.replicate)

    alpha_base = compute_alpha_table(base_df, args.fit_eps, args.drop_shallow, args.drop_deep, args.filter_N)
    alpha_repl = compute_alpha_table(repl_df, args.fit_eps, args.drop_shallow, args.drop_deep, args.filter_N)

    rho_base, p_base, n_base = compute_spearman(alpha_base)
    rho_repl, p_repl, n_repl = compute_spearman(alpha_repl)

    perm_result = permutation_test(alpha_repl, args.perm_count, args.perm_seed)
    if perm_result["null_rhos"] != []:
        save_perm_hist(perm_result["null_rhos"], perm_result["rho_obs"], os.path.join(args.outdir, "rho_perm_hist.png"))

    shallow_grid: List[int] = [int(x) for x in args.robustness_shallow.split(",") if x.strip() != ""]
    deep_grid: List[int] = [int(x) for x in args.robustness_deep.split(",") if x.strip() != ""]
    robust_df = robustness_grid(repl_df, args.fit_eps, shallow_grid, deep_grid, args.filter_N)
    robust_path = os.path.join(args.outdir, "robustness_depth_windows.csv")
    robust_df.to_csv(robust_path, index=False)
    same_sign_robust = sum(
        1 for _, row in robust_df.iterrows()
        if np.isfinite(row["spearman_rho"]) and np.sign(row["spearman_rho"]) == np.sign(rho_repl)
    )

    scatter_alpha(alpha_base, alpha_repl, os.path.join(args.outdir, "alpha_scatter.png"))

    decision = decide_outcome(rho_base, rho_repl, perm_result["p_value"], same_sign_robust)
    report_lines = [
        f"Baseline run_id(s): {','.join(sorted(alpha_base['run_id'].unique()))}",
        f"Replicate run_id(s): {','.join(sorted(alpha_repl['run_id'].unique()))}",
        f"Filter N: {args.filter_N}",
        f"Fit eps: {args.fit_eps}, drop_shallow={args.drop_shallow}, drop_deep={args.drop_deep}",
        f"Baseline rho={rho_base:.4f} (p={p_base:.3g}, n={n_base})",
        f"Replicate rho={rho_repl:.4f} (p={p_repl:.3g}, n={n_repl})",
        f"Permutation p={perm_result['p_value']:.4g}, rho_obs={perm_result['rho_obs']:.4f}, B={args.perm_count}",
        f"Robustness same-sign windows: {same_sign_robust}",
        f"Decision: {decision}",
    ]
    with open(os.path.join(args.outdir, "replication_report.txt"), "w") as f:
        f.write("\n".join(report_lines))

    summary = {
        "rho_base": rho_base,
        "p_base": p_base,
        "rho_repl": rho_repl,
        "p_repl": p_repl,
        "perm_p": perm_result["p_value"],
        "perm_rho_obs": perm_result["rho_obs"],
        "perm_count": args.perm_count,
        "same_sign_robust": same_sign_robust,
        "decision": decision,
    }
    with open(os.path.join(args.outdir, "replication_report.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
