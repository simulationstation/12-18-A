#!/usr/bin/env python3
"""
Analyze Loschmidt echo sweeps:
  - Fit alpha_echo per subset (slope of -log(P_return) vs depth)
  - Spearman correlation between alpha_echo and C with permutation control
  - Depth-window robustness checks
  - Per-subset alpha table + permutation histogram + robustness CSV
"""

import argparse
import json
import os
from typing import Dict, Iterable, List, Tuple

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
    if "run_id" not in df.columns:
        df["run_id"] = "default"
    return df


def _trim_depth_window(depths: List[int], probs: List[float], drop_shallow: int, drop_deep: int):
    paired = sorted(zip(depths, probs), key=lambda x: x[0])
    if drop_shallow > 0:
        paired = paired[drop_shallow:]
    if drop_deep > 0:
        paired = paired[: len(paired) - drop_deep]
    if not paired:
        return [], []
    d_trim, p_trim = zip(*paired)
    return list(d_trim), list(p_trim)


def compute_alpha_table(
    df: pd.DataFrame,
    fit_eps: float,
    drop_shallow: int = 0,
    drop_deep: int = 0,
    filter_N: int = None,
) -> pd.DataFrame:
    """Compute alpha_echo per subset."""
    if filter_N is not None:
        df = df[df["N"] == filter_N]
    rows = []
    for (run_id, N, subset_id), sub_df in df.groupby(["run_id", "N", "subset_id"]):
        depth_means = sub_df.groupby("depth")["mean_P_return"].mean().reset_index()
        depths = depth_means["depth"].tolist()
        probs = depth_means["mean_P_return"].tolist()
        depths_trim, probs_trim = _trim_depth_window(depths, probs, drop_shallow, drop_deep)
        alpha, stderr, points = fit_alpha(depths_trim, probs_trim, fit_eps)
        rows.append(
            {
                "run_id": run_id,
                "subset_id": int(subset_id),
                "N": int(N),
                "lambda2": float(sub_df["lambda2"].iloc[0]) if "lambda2" in sub_df else np.nan,
                "C": float(sub_df["C"].iloc[0]),
                "alpha_echo": alpha,
                "alpha_err": stderr,
                "n_points_used": points,
                "fit_eps": fit_eps,
                "drop_shallow": drop_shallow,
                "drop_deep": drop_deep,
            }
        )
    return pd.DataFrame(rows)


def compute_spearman(alpha_df: pd.DataFrame) -> Tuple[float, float, int]:
    clean = alpha_df.dropna(subset=["alpha_echo", "C"])
    if len(clean) < 2:
        return np.nan, np.nan, len(clean)
    rho, pval = spearmanr(clean["C"], clean["alpha_echo"])
    return float(rho), float(pval), len(clean)


def permutation_test(alpha_df: pd.DataFrame, B: int, seed: int = 0) -> Dict:
    rng = np.random.default_rng(seed)
    rho_obs, _, n = compute_spearman(alpha_df)
    if np.isnan(rho_obs):
        return {"rho_obs": rho_obs, "p_value": np.nan, "null_rhos": [], "n": n}
    Cs = alpha_df["C"].to_numpy()
    alphas = alpha_df["alpha_echo"].to_numpy()
    null_rhos = []
    for _ in range(B):
        shuffled_C = rng.permutation(Cs)
        rho_perm, _, _ = compute_spearman(
            pd.DataFrame({"C": shuffled_C, "alpha_echo": alphas})
        )
        null_rhos.append(rho_perm)
    null_rhos = np.array(null_rhos)
    p_value = float(np.mean(np.abs(null_rhos) >= abs(rho_obs)))
    return {"rho_obs": rho_obs, "p_value": p_value, "null_rhos": null_rhos, "n": n}


def save_perm_hist(null_rhos: Iterable[float], rho_obs: float, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.hist(null_rhos, bins=30, alpha=0.7, color="skyblue", edgecolor="k")
    plt.axvline(rho_obs, color="red", linestyle="--", label=f"rho_obs={rho_obs:.3f}")
    plt.axvline(-rho_obs, color="red", linestyle="--", alpha=0.5)
    plt.xlabel("Permutation rho")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)


def robustness_grid(
    df: pd.DataFrame,
    fit_eps: float,
    shallow_list: List[int],
    deep_list: List[int],
    filter_N: int = None,
) -> pd.DataFrame:
    rows = []
    for ds in shallow_list:
        for dd in deep_list:
            alpha_df = compute_alpha_table(df, fit_eps, ds, dd, filter_N)
            rho, pval, points = compute_spearman(alpha_df)
            rows.append(
                {
                    "drop_shallow": ds,
                    "drop_deep": dd,
                    "spearman_rho": rho,
                    "p_value": pval,
                    "points": points,
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Analyze Loschmidt echo sweep results.")
    parser.add_argument("--input", type=str, default="results/loschmidt_echo_sweep.csv",
                        help="Path to Loschmidt echo sweep CSV.")
    parser.add_argument("--fit-eps", type=float, default=1e-3,
                        help="Minimum P_return to include in alpha fit.")
    parser.add_argument("--alpha-csv", type=str, default="results/loschmidt_echo_alpha_fit.csv",
                        help="Path to write fitted alpha summary.")
    parser.add_argument("--corr-csv", type=str, default=None,
                        help="Path to write Spearman correlation summary (defaults next to alpha CSV).")
    parser.add_argument("--perm-count", type=int, default=5000,
                        help="Number of permutations for null distribution.")
    parser.add_argument("--perm-null-csv", type=str, default=None,
                        help="Optional CSV of permutation null rhos.")
    parser.add_argument("--perm-summary", type=str, default="results/loschmidt_permutation_summary.json",
                        help="Summary JSON for permutation test.")
    parser.add_argument("--perm-hist", type=str, default="results/rho_perm_hist.png",
                        help="Histogram image for permutation null.")
    parser.add_argument("--drop-shallow", type=int, default=0, help="Drop k smallest depths before fitting.")
    parser.add_argument("--drop-deep", type=int, default=0, help="Drop k largest depths before fitting.")
    parser.add_argument("--robustness-shallow", type=str, default="0",
                        help="Comma list of drop_shallow values for robustness grid.")
    parser.add_argument("--robustness-deep", type=str, default="0",
                        help="Comma list of drop_deep values for robustness grid.")
    parser.add_argument("--robustness-csv", type=str, default="results/robustness_depth_windows.csv",
                        help="Output CSV for robustness grid.")
    parser.add_argument("--filter-N", type=int, default=None,
                        help="If provided, restrict analysis to this N.")
    args = parser.parse_args()

    df = load_results(args.input)
    alpha_df = compute_alpha_table(df, args.fit_eps, args.drop_shallow, args.drop_deep, args.filter_N)
    os.makedirs(os.path.dirname(args.alpha_csv) if os.path.dirname(args.alpha_csv) else ".", exist_ok=True)
    alpha_df.to_csv(args.alpha_csv, index=False)

    corr_path = args.corr_csv or args.alpha_csv.replace(".csv", "_correlation.csv")
    rho, pval, points = compute_spearman(alpha_df)
    corr_df = pd.DataFrame(
        [{
            "run_id": ",".join(sorted(alpha_df["run_id"].unique())),
            "spearman_rho": rho,
            "p_value": pval,
            "points": points,
            "drop_shallow": args.drop_shallow,
            "drop_deep": args.drop_deep,
            "fit_eps": args.fit_eps,
            "N_filter": args.filter_N,
        }]
    )
    os.makedirs(os.path.dirname(corr_path) if os.path.dirname(corr_path) else ".", exist_ok=True)
    corr_df.to_csv(corr_path, index=False)

    perm_result = permutation_test(alpha_df, args.perm_count)
    os.makedirs(os.path.dirname(args.perm_summary) if os.path.dirname(args.perm_summary) else ".", exist_ok=True)
    summary_payload = {
        "rho_obs": perm_result["rho_obs"],
        "p_value": perm_result["p_value"],
        "perm_count": args.perm_count,
        "fit_eps": args.fit_eps,
        "drop_shallow": args.drop_shallow,
        "drop_deep": args.drop_deep,
        "N_filter": args.filter_N,
    }
    if perm_result["null_rhos"] != []:
        null_rhos = perm_result["null_rhos"]
        summary_payload.update({
            "null_mean": float(np.mean(null_rhos)),
            "null_std": float(np.std(null_rhos)),
            "quantiles": {
                "0.05": float(np.quantile(null_rhos, 0.05)),
                "0.5": float(np.quantile(null_rhos, 0.5)),
                "0.95": float(np.quantile(null_rhos, 0.95)),
            },
        })
        save_perm_hist(null_rhos, perm_result["rho_obs"], args.perm_hist)
        if args.perm_null_csv:
            os.makedirs(os.path.dirname(args.perm_null_csv) if os.path.dirname(args.perm_null_csv) else ".", exist_ok=True)
            pd.DataFrame({"rho_perm": null_rhos}).to_csv(args.perm_null_csv, index=False)
    with open(args.perm_summary, "w") as f:
        json.dump(summary_payload, f, indent=2)

    shallow_grid = [int(x) for x in args.robustness_shallow.split(",") if x.strip() != ""]
    deep_grid = [int(x) for x in args.robustness_deep.split(",") if x.strip() != ""]
    robustness_df = robustness_grid(df, args.fit_eps, shallow_grid, deep_grid, args.filter_N)
    os.makedirs(os.path.dirname(args.robustness_csv) if os.path.dirname(args.robustness_csv) else ".", exist_ok=True)
    robustness_df.to_csv(args.robustness_csv, index=False)

    print(f"Saved alpha summary to {args.alpha_csv}")
    print(f"Saved correlation summary to {corr_path}")
    print(f"Permutation p-value={summary_payload['p_value']:.4g} (rho={summary_payload['rho_obs']:.4f})")
    print(f"Saved permutation summary to {args.perm_summary} and histogram to {args.perm_hist}")
    print(f"Saved robustness grid to {args.robustness_csv}")


if __name__ == "__main__":
    main()
