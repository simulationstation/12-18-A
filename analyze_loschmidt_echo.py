#!/usr/bin/env python3
"""
analyze_loschmidt_echo.py
=========================

Analyze Loschmidt echo sweeps:
    - Fit alpha_echo per subset (slope of -log(P_return) vs depth)
    - Scatter plot alpha_echo vs C for each N
    - Optional plot of mean P_return vs depth grouped by C bins
    - Spearman correlation between alpha_echo and C (per N)
"""

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr


def fit_alpha(depths: List[int], probs: List[float], fit_eps: float) -> Tuple[float, float, int]:
    """Fit slope of -log(prob) vs depth with filtering."""
    valid = [(d, p) for d, p in zip(depths, probs) if p > fit_eps]
    if len(valid) < 2:
        return np.nan, np.nan, len(valid)
    xs, ys = zip(*valid)
    neg_logs = [-np.log(p) for p in ys]
    res = linregress(xs, neg_logs)
    return res.slope, res.stderr, len(valid)


def load_results(path: str) -> pd.DataFrame:
    """Load Loschmidt echo CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No results CSV found at {path}")
    df = pd.read_csv(path)
    required = {"subset_id", "depth", "mean_P_return", "C", "N"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")
    return df


def group_subset_depths(df: pd.DataFrame) -> Dict[Tuple[int, int], Tuple[List[int], List[float], float]]:
    """
    Group rows by (N, subset_id) and collect depths, probs, and C.

    Returns dict[(N, subset_id)] -> (depths, probs, C)
    """
    grouped: Dict[Tuple[int, int], Tuple[List[int], List[float], float]] = {}
    for (N, subset_id), sub_df in df.groupby(["N", "subset_id"]):
        depths = list(sub_df["depth"])
        probs = list(sub_df["mean_P_return"])
        C_val = float(sub_df["C"].iloc[0])
        grouped[(int(N), int(subset_id))] = (depths, probs, C_val)
    return grouped


def binned_mean_curves(df: pd.DataFrame, bins: int = 3):
    """Compute mean P_return vs depth curves grouped by C quantile bins."""
    curves = {}
    df = df.copy()
    df["C_bin"] = pd.qcut(df["C"], q=bins, duplicates="drop")
    for cbin, group in df.groupby("C_bin"):
        mean_curve = group.groupby("depth")["mean_P_return"].mean().reset_index()
        curves[str(cbin)] = mean_curve
    return curves


def main():
    parser = argparse.ArgumentParser(description="Analyze Loschmidt echo sweep results.")
    parser.add_argument("--input", type=str, default="results/loschmidt_echo_sweep.csv",
                        help="Path to Loschmidt echo sweep CSV.")
    parser.add_argument("--fit-eps", type=float, default=1e-3,
                        help="Minimum P_return to include in alpha fit.")
    parser.add_argument("--alpha-csv", type=str, default="results/loschmidt_echo_alpha_fit.csv",
                        help="Path to write fitted alpha summary.")
    parser.add_argument("--scatter", type=str, default="results/loschmidt_alpha_vs_C.png",
                        help="Path for alpha vs C scatter plot.")
    parser.add_argument("--binned-plot", type=str, default=None,
                        help="Optional path for mean P_return vs depth grouped by C bins.")
    args = parser.parse_args()

    df = load_results(args.input)
    grouped = group_subset_depths(df)

    alpha_rows = []
    scatter_points = defaultdict(list)  # N -> list of (C, alpha)
    corr_rows = []

    for (N, subset_id), (depths, probs, C_val) in grouped.items():
        alpha, stderr, points = fit_alpha(depths, probs, args.fit_eps)
        alpha_rows.append({
            "N": N,
            "subset_id": subset_id,
            "C": C_val,
            "alpha": alpha,
            "alpha_stderr": stderr,
            "points": points
        })
        scatter_points[N].append((C_val, alpha))

    alpha_df = pd.DataFrame(alpha_rows)
    os.makedirs(os.path.dirname(args.alpha_csv) if os.path.dirname(args.alpha_csv) else ".", exist_ok=True)
    alpha_df.to_csv(args.alpha_csv, index=False)

    # Scatter plots per N
    plt.figure(figsize=(7, 5))
    for N, pts in sorted(scatter_points.items()):
        Cs, alphas = zip(*pts) if pts else ([], [])
        plt.scatter(Cs, alphas, label=f"N={N}", alpha=0.7, edgecolors="k")
    plt.xlabel("C = N * lambda2")
    plt.ylabel("alpha_echo (slope of -log P_return)")
    plt.title("Loschmidt echo decay vs connectivity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.scatter) if os.path.dirname(args.scatter) else ".", exist_ok=True)
    plt.savefig(args.scatter, dpi=200)

    # Optional binned curves
    if args.binned_plot:
        curves = binned_mean_curves(df)
        plt.figure(figsize=(7, 5))
        for label, curve in curves.items():
            plt.plot(curve["depth"], curve["mean_P_return"], marker="o", label=label)
        plt.xlabel("Depth")
        plt.ylabel("Mean P_return")
        plt.title("Loschmidt echo: P_return vs depth by C bin")
        plt.grid(True, alpha=0.3)
        plt.legend(title="C quantile bin")
        plt.tight_layout()
        os.makedirs(os.path.dirname(args.binned_plot) if os.path.dirname(args.binned_plot) else ".", exist_ok=True)
        plt.savefig(args.binned_plot, dpi=200)

    # Spearman correlations per N
    summary_lines = []
    for N, sub_df in alpha_df.groupby("N"):
        clean = sub_df.dropna(subset=["alpha", "C"])
        if len(clean) < 2:
            rho, pval = np.nan, np.nan
        else:
            rho, pval = spearmanr(clean["C"], clean["alpha"])
        corr_rows.append({"N": N, "spearman_rho": rho, "p_value": pval, "points": len(clean)})
        summary_lines.append(f"N={N}: rho={rho:.4f}, p-value={pval:.4e}, points={len(clean)}")

    corr_df = pd.DataFrame(corr_rows)
    corr_path = args.alpha_csv.replace(".csv", "_correlation.csv")
    corr_df.to_csv(corr_path, index=False)

    print(f"Saved alpha summary to {args.alpha_csv}")
    print(f"Saved alpha vs C scatter to {args.scatter}")
    if args.binned_plot:
        print(f"Saved binned depth plot to {args.binned_plot}")
    print(f"Saved correlation summary to {corr_path}")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
