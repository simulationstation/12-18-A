#!/usr/bin/env python3
"""
Analyze synthetic benchmark data produced by generate_benchmark_data.py.

This script:
1. Reads a benchmark CSV with columns: device, N, depth, success_prob.
2. Parses device labels like "ring_N32", "grid_N64", "random_regular_4_N128".
3. Rebuilds the coupling graph deterministically from the device label and seed.
4. Computes the normalized Laplacian spectral gap lambda2 and C = N * lambda2.
5. Fits alpha_hat via linear regression of -log(success_prob) vs depth.
6. Writes a summary CSV and diagnostic plots.
"""

import argparse
import hashlib
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh


SUPPORTED_FAMILIES = {
    "ring",
    "grid",
    "random_regular_4",
    "small_world",
    "erdos_renyi",
}


@dataclass(frozen=True)
class AnalysisConfig:
    bench_csv: str
    outdir: str
    seed: int
    graph_seed_override: int | None
    show_plots: bool


def parse_device_label(label: str) -> Tuple[str, int]:
    pattern = r"(ring|grid|random_regular_4|small_world|erdos_renyi)_N(\d+)"
    m = re.search(pattern, label)
    if not m:
        raise ValueError(
            f"Could not parse device label '{label}'. Expected prefix like 'ring_N32' or 'small_world_N64'."
        )
    family, n_str = m.group(1), m.group(2)
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"Unsupported family '{family}' parsed from label '{label}'")
    try:
        N = int(n_str)
    except ValueError as exc:
        raise ValueError(f"Invalid N in device label '{label}'") from exc
    return family, N


def stable_graph_seed(device: str, base_seed: int, override: int | None) -> int:
    if override is not None:
        return int(override)
    h = hashlib.sha256(f"{device}|{base_seed}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % (2**32)


def make_graph(family: str, N: int, seed: int) -> nx.Graph:
    rng = np.random.default_rng(seed)

    if family == "ring":
        return nx.cycle_graph(N)

    if family == "grid":
        s = max(2, int(round(math.sqrt(N))))
        N_used = s * s
        G = nx.grid_2d_graph(s, s)
        mapping = {node: i for i, node in enumerate(G.nodes())}
        return nx.relabel_nodes(G, mapping)

    if family == "random_regular_4":
        N_eff = N + (N * 4) % 2  # ensure even N*d
        return nx.random_regular_graph(4, N_eff, seed=int(rng.integers(0, 2**31 - 1)))

    if family == "small_world":
        return nx.watts_strogatz_graph(N, 4, 0.2, seed=int(rng.integers(0, 2**31 - 1)))

    if family == "erdos_renyi":
        p = min(1.0, 4.0 / max(1, N - 1))
        return nx.erdos_renyi_graph(N, p, seed=int(rng.integers(0, 2**31 - 1)))

    raise ValueError(f"Unknown family '{family}'")


def ensure_connected(G: nx.Graph) -> Tuple[nx.Graph, int]:
    if nx.is_connected(G):
        return G, G.number_of_nodes()
    largest = max(nx.connected_components(G), key=len)
    sub = G.subgraph(largest).copy()
    mapping = {node: i for i, node in enumerate(sub.nodes())}
    sub = nx.relabel_nodes(sub, mapping)
    return sub, sub.number_of_nodes()


def normalized_laplacian_lambda2(G: nx.Graph) -> float:
    if G.number_of_nodes() < 2:
        raise ValueError("Graph must have at least 2 nodes to compute lambda2")
    L = nx.normalized_laplacian_matrix(G)
    vals = eigsh(L, k=2, which="SM", return_eigenvectors=False, tol=1e-6, maxiter=5000)
    vals = np.sort(vals)
    return float(vals[1])


def fit_alpha_hat(depths: Iterable[float], success_probs: Iterable[float]) -> float:
    d = np.asarray(list(depths), dtype=float)
    if d.size == 0:
        raise ValueError("Cannot fit alpha_hat with zero data points")
    y = -np.log(np.clip(np.asarray(list(success_probs), dtype=float), 1e-15, 1.0))
    A = np.vstack([d, np.ones_like(d)]).T
    alpha_hat, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(alpha_hat)


def validate_columns(df: pd.DataFrame, required: Tuple[str, ...]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in benchmark CSV: {', '.join(missing)}")


def analyze_benchmark(cfg: AnalysisConfig) -> pd.DataFrame:
    print(f"[load] Reading {cfg.bench_csv}")
    df = pd.read_csv(cfg.bench_csv)
    validate_columns(df, ("device", "N", "depth", "success_prob"))

    os.makedirs(cfg.outdir, exist_ok=True)

    records: list[Dict[str, float | str | int]] = []
    devices = sorted(df["device"].unique())
    print(f"[analyze] Found {len(devices)} unique device labels")

    for device in devices:
        sub = df[df["device"] == device]
        n_points = len(sub)
        family, N_label = parse_device_label(device)
        unique_Ns = sub["N"].unique()
        if len(unique_Ns) != 1:
            raise ValueError(f"Device '{device}' has inconsistent N values: {unique_Ns}")
        N_csv = int(unique_Ns[0])
        if N_csv != N_label:
            print(f"[warn] device {device}: N in CSV ({N_csv}) differs from label ({N_label}); using CSV value")
        N_requested = N_csv

        g_seed = stable_graph_seed(device, cfg.seed, cfg.graph_seed_override)
        G_raw = make_graph(family, N_requested, g_seed)
        G, N_used = ensure_connected(G_raw)
        lam2 = normalized_laplacian_lambda2(G)
        C = N_used * lam2

        alpha_hat = fit_alpha_hat(sub["depth"], sub["success_prob"])

        records.append(
            {
                "device": device,
                "family": family,
                "N_requested": N_requested,
                "N_used": N_used,
                "lambda2": lam2,
                "C": C,
                "alpha_hat": alpha_hat,
                "n_points": n_points,
            }
        )

        print(
            f"  {device}: family={family}, N_req={N_requested}, N_used={N_used}, "
            f"lambda2={lam2:.4e}, C={C:.4e}, alpha_hat={alpha_hat:.4e}, n={n_points}"
        )

    result_df = pd.DataFrame(records)
    if result_df.empty:
        raise ValueError("No device records processed; check benchmark CSV contents")
    return result_df


def plot_alpha_vs_C(df: pd.DataFrame, outdir: str, show: bool) -> None:
    plt.figure(figsize=(8, 5))
    for fam, fam_df in df.groupby("family"):
        fam_sorted = fam_df.sort_values("C")
        plt.plot(fam_sorted["C"], fam_sorted["alpha_hat"], marker="o", label=fam)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("C = N * λ2")
    plt.ylabel("alpha_hat")
    plt.title("alpha_hat vs C (synthetic benchmarks)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.6)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "alpha_vs_C_synth.png")
    plt.savefig(path, dpi=200)
    if show:
        plt.show()
    plt.close()


def plot_alpha_vs_N(df: pd.DataFrame, outdir: str, show: bool) -> None:
    plt.figure(figsize=(8, 5))
    df_sorted = df.sort_values("N_used")
    plt.plot(df_sorted["N_used"], df_sorted["alpha_hat"], marker="o", linestyle="none")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("N_used")
    plt.ylabel("alpha_hat")
    plt.title("alpha_hat vs N_used (synthetic benchmarks)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.6)
    plt.tight_layout()
    path = os.path.join(outdir, "alpha_vs_N_synth.png")
    plt.savefig(path, dpi=200)
    if show:
        plt.show()
    plt.close()


def plot_lambda2_vs_N(df: pd.DataFrame, outdir: str, show: bool) -> None:
    plt.figure(figsize=(8, 5))
    for fam, fam_df in df.groupby("family"):
        fam_sorted = fam_df.sort_values("N_used")
        plt.plot(fam_sorted["N_used"], fam_sorted["lambda2"], marker="o", label=fam)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("N_used")
    plt.ylabel("lambda2 (normalized Laplacian)")
    plt.title("lambda2 vs N_used (synthetic benchmarks)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.6)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "lambda2_vs_N_synth.png")
    plt.savefig(path, dpi=200)
    if show:
        plt.show()
    plt.close()


def parse_args(argv: list[str]) -> AnalysisConfig:
    ap = argparse.ArgumentParser(description="Analyze synthetic benchmark results and fit alpha_hat vs C")
    ap.add_argument("--bench_csv", required=True, help="Path to benchmark CSV from generate_benchmark_data.py")
    ap.add_argument("--outdir", default="results", help="Directory to write summary CSV and plots")
    ap.add_argument("--seed", type=int, default=123, help="Base seed for deterministic graph regeneration")
    ap.add_argument(
        "--graph_seed_override",
        type=int,
        default=None,
        help="Explicit seed for graph construction (otherwise derived from device label and --seed)",
    )
    ap.add_argument("--show_plots", dest="show_plots", action="store_true", help="Display plots interactively")
    ap.add_argument("--no_show_plots", dest="show_plots", action="store_false", help="Disable interactive plots")
    ap.set_defaults(show_plots=False)
    args = ap.parse_args(argv)

    return AnalysisConfig(
        bench_csv=args.bench_csv,
        outdir=args.outdir,
        seed=args.seed,
        graph_seed_override=args.graph_seed_override,
        show_plots=args.show_plots,
    )


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv if argv is not None else sys.argv[1:])

    summary_df = analyze_benchmark(cfg)
    summary_path = os.path.join(cfg.outdir, "alpha_vs_C_synth.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"[write] Saved summary CSV to {summary_path}")

    plot_alpha_vs_C(summary_df, cfg.outdir, cfg.show_plots)
    plot_alpha_vs_N(summary_df, cfg.outdir, cfg.show_plots)
    plot_lambda2_vs_N(summary_df, cfg.outdir, cfg.show_plots)

    print(f"[done] Plots written to {cfg.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
