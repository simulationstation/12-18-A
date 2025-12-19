#!/usr/bin/env python3
"""
analyze_logical_code.py
=======================

Lightweight analysis helper for logical code sweeps on AWS Braket.

Generates:
    - Scatter plot: logical_error_rate vs architecture metric C
    - CSV summary with Spearman correlation and p-value

Usage:
    python analyze_logical_code.py \\
        --input results/logical_code_sweep.csv \\
        --plot results/logical_error_vs_C.png \\
        --summary results/logical_code_correlation.csv
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr


def load_results(path: str) -> pd.DataFrame:
    """Load logical code sweep results."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No results CSV found at {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Results CSV is empty")
    required = {"C", "logical_error_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")
    return df.dropna(subset=["C", "logical_error_rate"])


def main():
    parser = argparse.ArgumentParser(description="Analyze logical code sweep results.")
    parser.add_argument("--input", type=str, default="results/logical_code_sweep.csv",
                        help="Path to logical code sweep CSV.")
    parser.add_argument("--plot", type=str, default="results/logical_error_vs_C.png",
                        help="Output path for scatter plot.")
    parser.add_argument("--summary", type=str, default="results/logical_code_correlation.csv",
                        help="Output path for correlation summary CSV.")
    args = parser.parse_args()

    df = load_results(args.input)
    rho, pval = spearmanr(df["C"], df["logical_error_rate"])
    n = len(df)

    os.makedirs(os.path.dirname(args.plot) if os.path.dirname(args.plot) else ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.summary) if os.path.dirname(args.summary) else ".", exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.scatter(df["C"], df["logical_error_rate"], alpha=0.7, edgecolors="k")
    plt.xlabel("C = N * lambda2")
    plt.ylabel("Logical error rate")
    plt.title("Logical error rate vs architecture metric C")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.plot, dpi=200)

    summary_df = pd.DataFrame([{
        "spearman_rho": rho,
        "p_value": pval,
        "points": n
    }])
    summary_df.to_csv(args.summary, index=False)

    print(f"Saved scatter plot to {args.plot}")
    print(f"Saved correlation summary to {args.summary}")
    print(f"Spearman rho={rho:.4f}, p-value={pval:.4e}, n={n}")


if __name__ == "__main__":
    main()
