#!/usr/bin/env python3
"""
analyze_compiler_fragility.py
==============================

Lightweight analysis helper for compiler fragility sweeps on AWS Braket.

Generates:
    - Scatter plot: std_success vs architecture metric C
    - Optional plot: min_success vs C
    - CSV summary with Spearman correlation and p-value

Usage:
    python analyze_compiler_fragility.py \
        --input results/compiler_fragility_sweep.csv \
        --plot results/compiler_fragility_std_vs_C.png \
        --summary results/compiler_fragility_correlation.csv
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr


def load_results(path: str) -> pd.DataFrame:
    """Load compiler fragility sweep results."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No results CSV found at {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Results CSV is empty")
    required = {"C", "std_success"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")
    return df.dropna(subset=["C", "std_success"])


def main():
    parser = argparse.ArgumentParser(description="Analyze compiler fragility sweep results.")
    parser.add_argument("--input", type=str, default="results/compiler_fragility_sweep.csv",
                        help="Path to compiler fragility sweep CSV.")
    parser.add_argument("--plot", type=str, default="results/compiler_fragility_std_vs_C.png",
                        help="Output path for std_success scatter plot.")
    parser.add_argument("--min-plot", dest="min_plot", type=str, default=None,
                        help="Optional output path for min_success vs C plot.")
    parser.add_argument("--summary", type=str, default="results/compiler_fragility_correlation.csv",
                        help="Output path for correlation summary CSV.")
    args = parser.parse_args()

    df = load_results(args.input)
    rho, pval = spearmanr(df["C"], df["std_success"])
    n = len(df)

    os.makedirs(os.path.dirname(args.plot) if os.path.dirname(args.plot) else ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.summary) if os.path.dirname(args.summary) else ".", exist_ok=True)
    if args.min_plot:
        os.makedirs(os.path.dirname(args.min_plot) if os.path.dirname(args.min_plot) else ".", exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.scatter(df["C"], df["std_success"], alpha=0.7, edgecolors="k")
    plt.xlabel("C = N * lambda2")
    plt.ylabel("Std dev of success")
    plt.title("Compiler fragility: variance vs connectivity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.plot, dpi=200)

    if args.min_plot:
        plt.figure(figsize=(7, 5))
        plt.scatter(df["C"], df["min_success"], alpha=0.7, edgecolors="k", color="tab:orange")
        plt.xlabel("C = N * lambda2")
        plt.ylabel("Min success across compilations")
        plt.title("Compiler fragility: worst-case success vs connectivity")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.min_plot, dpi=200)

    summary_df = pd.DataFrame([{
        "spearman_rho": rho,
        "p_value": pval,
        "points": n
    }])
    summary_df.to_csv(args.summary, index=False)

    print(f"Saved scatter plot to {args.plot}")
    if args.min_plot:
        print(f"Saved min success plot to {args.min_plot}")
    print(f"Saved correlation summary to {args.summary}")
    print(f"Spearman rho={rho:.4f}, p-value={pval:.4e}, n={n}")


if __name__ == "__main__":
    main()
