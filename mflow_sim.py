#!/usr/bin/env python3
"""
M-flow / spectral-gap (Option A) simulation-first test
=====================================================

Goal (Path 1): check whether an architecture-linked metric
C = N * lambda2(G) can explain an "extra" decay rate beyond local noise.

This script generates several graph families, computes:
  - lambda2(G): spectral gap of the normalized Laplacian
  - C(G) = N * lambda2(G)
  - M_eff(C): smooth-threshold M-flow

Then it simulates benchmark-circuit "success probability" vs depth d using:
  (A) Local-only noise model  (no global penalty)
  (B) Local + global penalty model (architecture-linked term)

It fits an effective decay rate alpha_hat from F(d) and plots alpha_hat vs C.

Run:
  pip install numpy scipy networkx matplotlib pandas
  python mflow_sim.py --outdir results

Notes:
- This is NOT a quantum simulator. It's a falsifiable *scaling test harness*.
- Model (B) is a hypothesis you'd try to refute with real data later.
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh


# -----------------------------
# Model parameters (edit freely)
# -----------------------------

@dataclass(frozen=True)
class MFlowParams:
    DeltaM: float = 20.0     # max uplift above 2
    C_star: float = 10.0     # threshold in C = N*lambda2
    w: float = 2.0           # smoothness/width of transition

@dataclass(frozen=True)
class NoiseParams:
    p_gate: float = 1e-3     # per two-qubit gate error probability
    p_meas: float = 0.0      # optional measurement error per qubit (not used by default)
    alpha1: float = 0.15     # strength of global penalty term (Model B)
    shots: int = 20000       # add shot noise to observed success probability


# -----------------------------
# Core math
# -----------------------------

def sigma(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def M_eff(C: float, mp: MFlowParams) -> float:
    return 2.0 + mp.DeltaM * sigma((C - mp.C_star) / mp.w)

def normalized_laplacian_lambda2(G: nx.Graph) -> float:
    """
    Compute λ2 of normalized Laplacian L = I - D^{-1/2} A D^{-1/2}.
    Uses sparse eigensolver; requires connected graphs (true for our families typically).
    """
    L = nx.normalized_laplacian_matrix(G)  # sparse
    vals = eigsh(L, k=2, which="SM", return_eigenvectors=False, tol=1e-6, maxiter=5000)
    vals = np.sort(vals)
    return float(vals[1])

def random_matching_edges(G: nx.Graph, rng: np.random.Generator) -> List[Tuple[int, int]]:
    """
    Build a random maximal matching by shuffling edges and greedily picking disjoint edges.
    This approximates "one parallel two-qubit layer" under native connectivity.
    """
    edges = list(G.edges())
    rng.shuffle(edges)
    used = set()
    chosen = []
    for u, v in edges:
        if u in used or v in used:
            continue
        chosen.append((u, v))
        used.add(u); used.add(v)
    return chosen

def simulate_success_vs_depth(
    G: nx.Graph,
    depths: List[int],
    rng: np.random.Generator,
    mp: MFlowParams,
    npz: NoiseParams,
    include_global_penalty: bool,
) -> Dict[int, float]:
    """
    Simulate a benchmark "success probability" F(d) vs circuit depth d.
    - Each depth step has one random matching layer (disjoint edges).
    - Each gate fails independently with prob p_gate (local noise).
    - Optional global penalty: exp(- alpha1 * λ2 * M_eff(C) * d)
      (This is the *hypothesized* architecture-linked term.)
    - Add binomial shot noise (optional) to mimic measurement sampling.
    """
    N = G.number_of_nodes()
    lam2 = normalized_laplacian_lambda2(G)
    C = N * lam2
    Me = M_eff(C, mp)

    out = {}
    for d in depths:
        # Local noise: for each layer, we choose a matching and count gates
        # Expected success per layer ≈ (1-p_gate)^(#gates_in_layer)
        # We'll simulate matchings to reflect connectivity constraints.
        success = 1.0
        for _ in range(d):
            layer_edges = random_matching_edges(G, rng)
            n_gates = len(layer_edges)
            success *= (1.0 - npz.p_gate) ** n_gates

        # Optional global penalty term (architecture-linked)
        if include_global_penalty:
            success *= math.exp(-npz.alpha1 * lam2 * Me * d)

        # Shot noise
        if npz.shots and npz.shots > 0:
            k = rng.binomial(npz.shots, min(max(success, 0.0), 1.0))
            success_obs = k / npz.shots
        else:
            success_obs = success

        out[d] = float(success_obs)

    return out

def fit_alpha_hat(depths: List[int], F: List[float]) -> float:
    """
    Fit alpha_hat from -log(F(d)) ≈ alpha_hat * d + const using least squares.
    """
    d = np.asarray(depths, dtype=float)
    y = -np.log(np.clip(np.asarray(F, dtype=float), 1e-15, 1.0))
    A = np.vstack([d, np.ones_like(d)]).T
    alpha_hat, _const = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(alpha_hat)


# -----------------------------
# Graph families
# -----------------------------

def make_graph_family(name: str, N: int, rng: np.random.Generator) -> nx.Graph:
    if name == "ring":
        return nx.cycle_graph(N)

    if name == "random_regular_4":
        # expander-like proxy
        return nx.random_regular_graph(4, N, seed=int(rng.integers(0, 2**31-1)))

    if name == "erdos_renyi":
        # choose p so average degree ~4
        p = min(1.0, 4.0 / (N - 1))
        G = nx.erdos_renyi_graph(N, p, seed=int(rng.integers(0, 2**31-1)))
        # ensure connected by taking largest component if needed
        if not nx.is_connected(G):
            largest = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest).copy()
        return G

    if name == "small_world":
        # Watts–Strogatz: k=4, rewiring prob 0.2
        return nx.watts_strogatz_graph(N, 4, 0.2, seed=int(rng.integers(0, 2**31-1)))

    if name == "grid":
        # 2D grid: nearest square size
        s = int(round(math.sqrt(N)))
        N2 = s * s
        G = nx.grid_2d_graph(s, s)
        # relabel to 0..N2-1
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        if N2 != N:
            # If N isn't a perfect square, we keep the nearest square for grid.
            # Caller should know this; we'll handle in the sweep setup.
            pass
        return G

    raise ValueError(f"Unknown family: {name}")


# -----------------------------
# Main sweep + plots
# -----------------------------

def run_sweep(
    Ns: List[int],
    families: List[str],
    depths: List[int],
    reps_per_point: int,
    seed: int,
    mp: MFlowParams,
    npz: NoiseParams,
    outdir: str,
):
    rng = np.random.default_rng(seed)

    records = []

    for fam in families:
        for N in Ns:
            # Handle grid: use nearest square
            N_eff = N
            if fam == "grid":
                s = int(round(math.sqrt(N)))
                N_eff = s * s
                if N_eff < 4:
                    continue

            # Replicate several graph instances per (fam, N) to average randomness
            lam2_list = []
            C_list = []
            Me_list = []
            alpha_local_list = []
            alpha_global_list = []

            for _ in range(reps_per_point):
                G = make_graph_family(fam, N_eff, rng)

                # some random graphs can yield tiny components if connectivity breaks;
                # skip pathological ones
                if G.number_of_nodes() < max(8, int(0.8 * N_eff)):
                    continue

                lam2 = normalized_laplacian_lambda2(G)
                C = G.number_of_nodes() * lam2
                Me = M_eff(C, mp)

                # Simulate success vs depth with and without global penalty
                F_local = simulate_success_vs_depth(G, depths, rng, mp, npz, include_global_penalty=False)
                F_global = simulate_success_vs_depth(G, depths, rng, mp, npz, include_global_penalty=True)

                alpha_local = fit_alpha_hat(list(F_local.keys()), list(F_local.values()))
                alpha_global = fit_alpha_hat(list(F_global.keys()), list(F_global.values()))

                lam2_list.append(lam2)
                C_list.append(C)
                Me_list.append(Me)
                alpha_local_list.append(alpha_local)
                alpha_global_list.append(alpha_global)

            if not lam2_list:
                continue

            def mean_std(x: List[float]) -> Tuple[float, float]:
                x = np.asarray(x, dtype=float)
                return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0

            lam2_mu, lam2_sd = mean_std(lam2_list)
            C_mu, C_sd = mean_std(C_list)
            Me_mu, Me_sd = mean_std(Me_list)
            aL_mu, aL_sd = mean_std(alpha_local_list)
            aG_mu, aG_sd = mean_std(alpha_global_list)

            records.append({
                "family": fam,
                "N_requested": N,
                "N_used": N_eff if fam == "grid" else N,
                "lambda2_mean": lam2_mu,
                "lambda2_std": lam2_sd,
                "C_mean": C_mu,
                "C_std": C_sd,
                "M_eff_mean": Me_mu,
                "M_eff_std": Me_sd,
                "alpha_hat_local_mean": aL_mu,
                "alpha_hat_local_std": aL_sd,
                "alpha_hat_global_mean": aG_mu,
                "alpha_hat_global_std": aG_sd,
            })

    df = pd.DataFrame(records)
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "sweep_results.csv")
    df.to_csv(csv_path, index=False)

    # Plot: alpha_hat vs C (local-only vs local+global)
    def plot_alpha_vs_C(col_mean: str, title: str, outfile: str):
        plt.figure(figsize=(8, 5))
        for fam in families:
            sub = df[df["family"] == fam].sort_values("N_used")
            if sub.empty:
                continue
            x = sub["C_mean"].to_numpy()
            y = sub[col_mean].to_numpy()
            plt.plot(x, y, marker="o", label=fam)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("C = N * λ2 (mean, log scale)")
        plt.ylabel("alpha_hat (log scale)")
        plt.title(title)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, outfile), dpi=200)
        plt.show()

    plot_alpha_vs_C(
        "alpha_hat_local_mean",
        "Local-only model: alpha_hat vs C (should NOT strongly collapse by design)",
        "alpha_vs_C_local.png",
    )
    plot_alpha_vs_C(
        "alpha_hat_global_mean",
        "Local + global-penalty model: alpha_hat vs C (collapse expected if hypothesis true)",
        "alpha_vs_C_global.png",
    )

    # Plot: M_eff vs C
    plt.figure(figsize=(8, 5))
    for fam in families:
        sub = df[df["family"] == fam].sort_values("N_used")
        if sub.empty:
            continue
        plt.plot(sub["C_mean"], sub["M_eff_mean"], marker="o", label=fam)
    plt.xscale("log")
    plt.xlabel("C = N * λ2 (mean, log scale)")
    plt.ylabel("M_eff (mean)")
    plt.title("M-flow: M_eff vs C across families")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "Meff_vs_C.png"), dpi=200)
    plt.show()

    print(f"\nSaved: {csv_path}")
    print(f"Saved figures in: {outdir}")
    print("\nTip: tweak NoiseParams.alpha1 and MFlowParams.(DeltaM,C_star,w) to see how strong/weak the global term must be.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="results", help="output directory")
    ap.add_argument("--seed", type=int, default=123, help="random seed")
    ap.add_argument("--reps", type=int, default=5, help="graph instances per (family,N)")
    ap.add_argument("--Ns", type=int, nargs="+", default=[32, 64, 128, 256, 512], help="N values to sweep")
    ap.add_argument("--depths", type=int, nargs="+", default=[5, 10, 20, 40, 80], help="circuit depths")
    ap.add_argument("--families", type=str, nargs="+",
                    default=["ring", "random_regular_4", "small_world", "erdos_renyi", "grid"],
                    help="graph families")
    args = ap.parse_args()

    mp = MFlowParams()
    npz = NoiseParams()

    run_sweep(
        Ns=args.Ns,
        families=args.families,
        depths=args.depths,
        reps_per_point=args.reps,
        seed=args.seed,
        mp=mp,
        npz=npz,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
