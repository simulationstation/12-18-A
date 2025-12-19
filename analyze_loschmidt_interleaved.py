#!/usr/bin/env python3
"""
Analyze time-interleaved Loschmidt echo experiments.

Outputs:
  - alpha_by_block.csv: per-(block, subset) echo decay fits
  - block_corr_summary.csv: Spearman rho per block with permutation p-values
  - rho_perm_block{b}.png: permutation histograms
  - interleaved_report.txt: plaintext decision report (architecture vs drift)
"""

import argparse
import ast
import json
import os
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr


def load_inputs(paths: List[str]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"No results CSV found at {p}")
        dfs.append(pd.read_csv(p))
    df = pd.concat(dfs, ignore_index=True)
    required = {
        "block_index",
        "subset_id",
        "depth",
        "mean_P_return",
        "C",
        "lambda2",
        "N_used",
        "run_id",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")
    return df


def fit_alpha_block(df: pd.DataFrame, fit_eps: float) -> Tuple[float, float, int]:
    depths = df["depth"].tolist()
    probs = df["mean_P_return"].tolist()
    valid = [(d, p) for d, p in zip(depths, probs) if p > fit_eps]
    if len(valid) < 2:
        return np.nan, np.nan, len(valid)
    xs, ys = zip(*sorted(valid, key=lambda x: x[0]))
    neg_logs = [-np.log(p) for p in ys]
    res = linregress(xs, neg_logs)
    return float(res.slope), float(res.stderr), len(valid)


def compute_alpha_table(df: pd.DataFrame, fit_eps: float) -> pd.DataFrame:
    rows = []
    for (block, subset_id, run_id), g in df.groupby(["block_index", "subset_id", "run_id"]):
        depth_means = g.groupby("depth")["mean_P_return"].mean().reset_index()
        alpha, stderr, points = fit_alpha_block(depth_means, fit_eps)
        rows.append(
            {
                "block_index": int(block),
                "subset_id": int(subset_id),
                "run_id": run_id,
                "lambda2": float(g["lambda2"].iloc[0]),
                "C": float(g["C"].iloc[0]),
                "alpha_echo": alpha,
                "alpha_err": stderr,
                "n_points_used": points,
                "fit_eps": fit_eps,
                "u_style": g["u_style"].iloc[0] if "u_style" in g else "mixed",
            }
        )
    return pd.DataFrame(rows)


def permutation_test(alpha_df: pd.DataFrame, B: int, seed: int = 0) -> Dict:
    rng = np.random.default_rng(seed)
    clean = alpha_df.dropna(subset=["alpha_echo", "C"])
    if len(clean) < 2:
        return {"rho_obs": np.nan, "p_value": np.nan, "null_rhos": [], "n": len(clean)}
    rho_obs, _ = spearmanr(clean["C"], clean["alpha_echo"])
    Cs = clean["C"].to_numpy()
    alphas = clean["alpha_echo"].to_numpy()
    null_rhos = []
    for _ in range(B):
        shuffled = rng.permutation(Cs)
        rho_perm, _ = spearmanr(shuffled, alphas)
        null_rhos.append(rho_perm)
    null_rhos = np.array(null_rhos)
    p_value = float(np.mean(np.abs(null_rhos) >= abs(rho_obs)))
    return {"rho_obs": float(rho_obs), "p_value": p_value, "null_rhos": null_rhos, "n": len(clean)}


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
    plt.close()


def block_similarity(alpha_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    blocks = sorted(alpha_df["block_index"].unique())
    for i, b1 in enumerate(blocks):
        for b2 in blocks[i + 1:]:
            merged = alpha_df[alpha_df["block_index"].isin([b1, b2])]
            pivot = merged.pivot(index="subset_id", columns="block_index", values="alpha_echo")
            aligned = pivot.dropna()
            if len(aligned) < 2:
                continue
            rho, _ = spearmanr(aligned[b1], aligned[b2])
            rows.append(
                {
                    "block_i": b1,
                    "block_j": b2,
                    "spearman_rho": float(rho),
                    "count": int(len(aligned)),
                }
            )
    return pd.DataFrame(rows)


def load_local_quality(path: str) -> Dict[int, float]:
    if path.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
        items = data.get("qubits") or data
        return {int(k): float(v) for k, v in items.items()}
    df = pd.read_csv(path)
    if "qubit" in df.columns and "value" in df.columns:
        return {int(r.qubit): float(r.value) for r in df.itertuples()}
    if df.shape[1] >= 2:
        q_col, v_col = df.columns[:2]
        return {int(r[q_col]): float(r[v_col]) for _, r in df.iterrows()}
    raise ValueError("Local quality file must have qubit/value columns")


def attach_local_quality(alpha_df: pd.DataFrame, quality_map: Dict[int, float], qubit_map: Dict[int, List[int]]) -> pd.DataFrame:
    L_means = []
    for subset_id in alpha_df["subset_id"]:
        qubits = qubit_map.get(subset_id, [])
        values = [quality_map[q] for q in qubits if q in quality_map]
        L_means.append(np.mean(values) if values else np.nan)
    alpha_df = alpha_df.copy()
    alpha_df["L_mean"] = L_means
    return alpha_df


def parse_qubit_strings(df: pd.DataFrame) -> Dict[int, List[int]]:
    mapping: Dict[int, List[int]] = {}
    for subset_id, qubit_str in df[["subset_id", "qubits"]].drop_duplicates().itertuples(index=False):
        try:
            qubits = list(ast.literal_eval(qubit_str))
        except Exception:
            qubits = []
        mapping[int(subset_id)] = [int(q) for q in qubits]
    return mapping


def write_report(
    path: str,
    corr_summary: pd.DataFrame,
    similarity_df: pd.DataFrame,
    verdict: str,
    stable_sign: bool,
    significant_blocks: List[int],
):
    lines = []
    lines.append("Interleaved Loschmidt Echo Report")
    lines.append("=" * 40)
    lines.append("")
    lines.append("Per-block Spearman rho (alpha_echo vs C):")
    for row in corr_summary.itertuples():
        lines.append(
            f"  Block {row.block_index}: rho={row.spearman_rho:.4f}, p_perm={row.permutation_p:.4g}, n={row.n}"
        )
    lines.append("")
    lines.append(f"Stable rho sign across blocks: {stable_sign}")
    lines.append(f"Blocks with permutation p<0.05: {significant_blocks}")
    if not similarity_df.empty:
        mean_sim = similarity_df["spearman_rho"].mean()
        med_sim = similarity_df["spearman_rho"].median()
        lines.append(f"Block-to-block alpha similarity (Spearman): mean={mean_sim:.3f}, median={med_sim:.3f}")
    else:
        lines.append("Block-to-block alpha similarity: insufficient overlapping subsets")
    lines.append("")
    lines.append(f"Verdict: {verdict}")

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


def decide_verdict(corr_summary: pd.DataFrame, similarity_df: pd.DataFrame) -> Tuple[str, bool, List[int]]:
    valid = corr_summary.dropna(subset=["spearman_rho"])
    signs = [np.sign(r) for r in valid["spearman_rho"] if r != 0]
    stable_sign = len(set(signs)) == 1 if signs else False
    significant_blocks = valid[valid["permutation_p"] < 0.05]["block_index"].tolist()
    similarity_metric = similarity_df["spearman_rho"].median() if not similarity_df.empty else np.nan

    verdict = "C) no clear effect"
    if stable_sign and len(significant_blocks) >= max(1, len(valid) // 2):
        verdict = "A) architecture-linked effect (consistent rho sign/p-values)"
    elif not np.isnan(similarity_metric) and similarity_metric > 0.7:
        verdict = "B) drift-like coherent shift (high block similarity)"

    return verdict, stable_sign, significant_blocks


def main():
    parser = argparse.ArgumentParser(description="Analyze interleaved Loschmidt echo results.")
    parser.add_argument("--inputs", type=str, nargs="+", required=True, help="One or more interleaved CSV files.")
    parser.add_argument("--fit-eps", type=float, default=1e-3, help="Minimum P_return to include in alpha fit.")
    parser.add_argument("--perm-count", type=int, default=5000, help="Permutation iterations per block.")
    parser.add_argument("--alpha-csv", type=str, default="results/alpha_by_block.csv",
                        help="Output CSV for per-block alpha fits.")
    parser.add_argument("--corr-csv", type=str, default="results/block_corr_summary.csv",
                        help="Output CSV for per-block correlation summary.")
    parser.add_argument("--report", type=str, default="results/interleaved_report.txt",
                        help="Plaintext decision report path.")
    parser.add_argument("--local-quality", type=str, default=None,
                        help="Optional CSV/JSON with per-qubit quality metric to join as proxy L.")
    parser.add_argument("--perm-hist-dir", type=str, default="results",
                        help="Directory for permutation histogram images.")
    parser.add_argument("--perm-seed", type=int, default=0, help="Seed for permutation nulls.")
    args = parser.parse_args()

    df = load_inputs(args.inputs)
    alpha_df = compute_alpha_table(df, args.fit_eps)
    os.makedirs(os.path.dirname(args.alpha_csv) if os.path.dirname(args.alpha_csv) else ".", exist_ok=True)
    alpha_df.to_csv(args.alpha_csv, index=False)

    # Optional local quality regression placeholder
    qubit_map = parse_qubit_strings(df)
    if args.local_quality:
        quality_map = load_local_quality(args.local_quality)
        alpha_df = attach_local_quality(alpha_df, quality_map, qubit_map)

    corr_rows = []
    blocks = sorted(alpha_df["block_index"].unique())
    for block in blocks:
        block_df = alpha_df[alpha_df["block_index"] == block]
        perm_res = permutation_test(block_df, args.perm_count, args.perm_seed + int(block))
        if perm_res["null_rhos"] != []:
            hist_path = os.path.join(args.perm_hist_dir, f"rho_perm_block{block}.png")
            save_perm_hist(perm_res["null_rhos"], perm_res["rho_obs"], hist_path)
        corr_rows.append(
            {
                "block_index": block,
                "spearman_rho": perm_res["rho_obs"],
                "permutation_p": perm_res["p_value"],
                "n": perm_res["n"],
                "null_mean": float(np.mean(perm_res["null_rhos"])) if perm_res["null_rhos"] != [] else np.nan,
                "null_std": float(np.std(perm_res["null_rhos"])) if perm_res["null_rhos"] != [] else np.nan,
                "perm_count": args.perm_count,
            }
        )
    corr_df = pd.DataFrame(corr_rows)
    os.makedirs(os.path.dirname(args.corr_csv) if os.path.dirname(args.corr_csv) else ".", exist_ok=True)
    corr_df.to_csv(args.corr_csv, index=False)

    similarity_df = block_similarity(alpha_df)
    verdict, stable_sign, significant_blocks = decide_verdict(corr_df, similarity_df)
    write_report(args.report, corr_df, similarity_df, verdict, stable_sign, significant_blocks)

    print(f"Saved alpha table to {args.alpha_csv}")
    print(f"Saved correlation summary to {args.corr_csv}")
    print(f"Saved report to {args.report}")


if __name__ == "__main__":
    main()
