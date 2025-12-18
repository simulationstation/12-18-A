#!/usr/bin/env python3
"""
generate_benchmark_data.py
==========================

Generates synthetic benchmark data for the "Option 1" M-flow test pipeline
using Qiskit Aer to simulate noisy quantum circuits.

This script creates random circuits constrained by various coupling graph
topologies (ring, grid, random regular, small-world, Erdős-Rényi), applies
a configurable noise model, and measures the success probability (probability
of obtaining the all-zeros bitstring).

Requirements:
    pip install qiskit qiskit-aer networkx numpy pandas

Usage:
    python generate_benchmark_data.py --outfile bench.csv --seed 42 \\
        --Ns 16 32 64 --depths 5 10 20 --K 10 --shots 8192 \\
        --p1 1e-4 --p2 1e-3 --pm 0.0

Output:
    CSV file with columns: device, N, depth, success_prob
    where "device" is a label like "ring_N32", "grid_N64", etc.
"""

import argparse
import math
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import networkx as nx

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError


# =============================================================================
# Graph Generation Functions
# =============================================================================

def make_ring(N: int, rng: np.random.Generator) -> nx.Graph:
    """
    Create a ring (cycle) graph with N nodes.

    The ring topology has each node connected to exactly 2 neighbors,
    forming a closed loop. This represents a 1D periodic chain.

    Args:
        N: Number of nodes
        rng: Random generator (unused for deterministic ring)

    Returns:
        NetworkX Graph representing the ring
    """
    return nx.cycle_graph(N)


def make_grid(N: int, rng: np.random.Generator) -> Tuple[nx.Graph, int]:
    """
    Create a 2D grid (nearest-neighbor) graph with size closest to N.

    Chooses s = round(sqrt(N)) and creates an s×s grid.
    Returns the actual number of nodes used (s²).

    Args:
        N: Target number of nodes
        rng: Random generator (unused for deterministic grid)

    Returns:
        Tuple of (graph, actual_N) where actual_N = s²
    """
    s = int(round(math.sqrt(N)))
    if s < 2:
        s = 2
    N_used = s * s

    # Create 2D grid graph
    G = nx.grid_2d_graph(s, s)

    # Relabel nodes from (i,j) tuples to integers 0..N_used-1
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    return G, N_used


def make_random_regular(N: int, d: int, rng: np.random.Generator) -> nx.Graph:
    """
    Create a random d-regular graph with N nodes.

    A d-regular graph has every node with exactly d neighbors.
    For d=4, this creates an expander-like graph with good connectivity.

    Note: N*d must be even; if not, N is incremented by 1.

    Args:
        N: Number of nodes
        d: Degree of each node
        rng: Random generator for reproducibility

    Returns:
        NetworkX Graph representing the random regular graph
    """
    # Ensure N*d is even (required for d-regular graphs)
    if (N * d) % 2 != 0:
        N = N + 1

    seed = int(rng.integers(0, 2**31 - 1))
    return nx.random_regular_graph(d, N, seed=seed)


def make_small_world(N: int, k: int, p: float, rng: np.random.Generator) -> nx.Graph:
    """
    Create a Watts-Strogatz small-world graph.

    Starts with a ring lattice where each node is connected to k nearest
    neighbors, then rewires each edge with probability p. This creates
    a graph with high clustering and low average path length.

    Args:
        N: Number of nodes
        k: Each node connected to k nearest neighbors (must be even)
        p: Probability of rewiring each edge
        rng: Random generator for reproducibility

    Returns:
        NetworkX Graph representing the small-world network
    """
    seed = int(rng.integers(0, 2**31 - 1))
    return nx.watts_strogatz_graph(N, k, p, seed=seed)


def make_erdos_renyi(N: int, target_degree: float, rng: np.random.Generator) -> nx.Graph:
    """
    Create an Erdős-Rényi random graph G(N, p).

    Chooses p such that expected degree ≈ target_degree:
        p = target_degree / (N - 1)

    Note: The resulting graph may not be connected; caller should handle this.

    Args:
        N: Number of nodes
        target_degree: Target average degree (e.g., 4.0)
        rng: Random generator for reproducibility

    Returns:
        NetworkX Graph representing the Erdős-Rényi random graph
    """
    p = min(1.0, target_degree / (N - 1))
    seed = int(rng.integers(0, 2**31 - 1))
    return nx.erdos_renyi_graph(N, p, seed=seed)


def ensure_connected(G: nx.Graph) -> Tuple[nx.Graph, int]:
    """
    Ensure the graph is connected by taking the largest connected component.

    If the graph is already connected, returns it unchanged.
    Otherwise, extracts the largest connected component and relabels
    nodes to 0..n-1.

    Args:
        G: Input graph (possibly disconnected)

    Returns:
        Tuple of (connected_graph, node_count)
    """
    if nx.is_connected(G):
        return G, G.number_of_nodes()

    # Take largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    G_sub = G.subgraph(largest_cc).copy()

    # Relabel nodes to 0..n-1 for clean qubit indexing
    mapping = {node: i for i, node in enumerate(G_sub.nodes())}
    G_sub = nx.relabel_nodes(G_sub, mapping)

    return G_sub, G_sub.number_of_nodes()


# =============================================================================
# Circuit Generation Functions
# =============================================================================

def random_maximal_matching(G: nx.Graph, rng: np.random.Generator) -> List[Tuple[int, int]]:
    """
    Build a random maximal matching by shuffling edges and greedily selecting.

    A maximal matching is a set of edges with no shared vertices such that
    no additional edge can be added. This represents one parallel layer of
    two-qubit gates respecting the coupling constraints.

    Args:
        G: The coupling graph
        rng: Random generator for shuffling

    Returns:
        List of (u, v) edge tuples forming the maximal matching
    """
    edges = list(G.edges())
    rng.shuffle(edges)

    used = set()
    chosen = []

    for u, v in edges:
        if u not in used and v not in used:
            chosen.append((u, v))
            used.add(u)
            used.add(v)

    return chosen


def generate_random_circuit(
    G: nx.Graph,
    depth: int,
    rng: np.random.Generator,
) -> QuantumCircuit:
    """
    Generate a random circuit constrained by coupling graph G.

    For each of the `depth` layers:
    1. Apply a random single-qubit gate to each qubit
       (chosen from {SX, X, RZ(θ)} with θ uniform in [0, 2π])
    2. Select a random maximal matching of the coupling graph
    3. Apply CX gates on each matched edge

    Finally, measure all qubits in the computational basis.

    The random single-qubit gates prevent trivial commuting structure
    and ensure the circuit explores the Hilbert space.

    Args:
        G: The coupling graph defining allowed two-qubit interactions
        depth: Number of layers
        rng: Random generator for reproducibility

    Returns:
        QuantumCircuit with depth layers and final measurement
    """
    N = G.number_of_nodes()
    qc = QuantumCircuit(N, N)

    # Available single-qubit gates (standard basis gates for most backends)
    single_qubit_gates = ['sx', 'x', 'rz']

    for layer in range(depth):
        # -----------------------------------------------------------------
        # Step 1: Apply random single-qubit gates to all qubits
        # This creates non-trivial interference and prevents the circuit
        # from having simple commuting structure.
        # -----------------------------------------------------------------
        for q in range(N):
            gate_choice = rng.choice(single_qubit_gates)
            if gate_choice == 'sx':
                qc.sx(q)
            elif gate_choice == 'x':
                qc.x(q)
            elif gate_choice == 'rz':
                theta = rng.uniform(0, 2 * math.pi)
                qc.rz(theta, q)

        # -----------------------------------------------------------------
        # Step 2: Get random maximal matching and apply CX gates
        # The matching ensures gates can be applied in parallel without
        # conflicts, respecting the coupling graph topology.
        # -----------------------------------------------------------------
        matching = random_maximal_matching(G, rng)
        for (u, v) in matching:
            qc.cx(u, v)

    # -----------------------------------------------------------------
    # Step 3: Measure all qubits in computational basis
    # -----------------------------------------------------------------
    qc.measure(range(N), range(N))

    return qc


# =============================================================================
# Noise Model Construction
# =============================================================================

def build_noise_model(
    p1: float,
    p2: float,
    pm: float,
) -> NoiseModel:
    """
    Build a noise model with depolarizing and readout errors.

    The noise model includes:
    - Depolarizing error on single-qubit gates with probability p1
    - Depolarizing error on two-qubit gates with probability p2
    - Symmetric readout error with probability pm

    Depolarizing error replaces the ideal gate output with a random Pauli
    operator with the specified probability.

    Args:
        p1: Single-qubit depolarizing error probability
        p2: Two-qubit depolarizing error probability
        pm: Measurement/readout error probability (symmetric bit-flip)

    Returns:
        Configured NoiseModel for Aer simulation
    """
    noise_model = NoiseModel()

    # -----------------------------------------------------------------
    # Single-qubit depolarizing error
    # Applied to common single-qubit gates used in circuit generation
    # and transpilation.
    # -----------------------------------------------------------------
    if p1 > 0:
        error_1q = depolarizing_error(p1, 1)
        single_qubit_gate_names = [
            'sx', 'x', 'rz',  # Gates we explicitly use
            'id', 'h', 's', 'sdg', 't', 'tdg',  # Common gates
            'u1', 'u2', 'u3', 'u',  # Generic single-qubit gates
        ]
        noise_model.add_all_qubit_quantum_error(error_1q, single_qubit_gate_names)

    # -----------------------------------------------------------------
    # Two-qubit depolarizing error
    # Applied to CX (CNOT) and ECR gates.
    # -----------------------------------------------------------------
    if p2 > 0:
        error_2q = depolarizing_error(p2, 2)
        two_qubit_gate_names = ['cx', 'ecr', 'cz']
        noise_model.add_all_qubit_quantum_error(error_2q, two_qubit_gate_names)

    # -----------------------------------------------------------------
    # Readout error (symmetric bit-flip)
    # P(measure 0 | state 0) = P(measure 1 | state 1) = 1 - pm
    # P(measure 1 | state 0) = P(measure 0 | state 1) = pm
    # -----------------------------------------------------------------
    if pm > 0:
        readout_err = ReadoutError([[1 - pm, pm], [pm, 1 - pm]])
        noise_model.add_all_qubit_readout_error(readout_err)

    return noise_model


# =============================================================================
# Simulation and Success Probability
# =============================================================================

def simulate_circuit(
    qc: QuantumCircuit,
    backend: AerSimulator,
    shots: int,
) -> float:
    """
    Simulate a circuit with noise and return the success probability.

    Success is defined as measuring the all-zeros bitstring |00...0⟩.
    For a circuit starting in |0⟩^N with random gates, this probability
    decays with depth due to noise.

    Args:
        qc: The quantum circuit to simulate (must include measurements)
        backend: The Aer simulator backend (with noise model configured)
        shots: Number of measurement shots

    Returns:
        Probability of measuring the all-zeros bitstring
    """
    # Run simulation directly without coupling map constraints
    # AerSimulator can handle arbitrary qubit counts when run directly
    job = backend.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # The all-zeros bitstring (Qiskit uses little-endian ordering)
    N = qc.num_qubits
    all_zeros = '0' * N

    # Get count of all-zeros outcome
    zero_count = counts.get(all_zeros, 0)
    success_prob = zero_count / shots

    return success_prob


# =============================================================================
# Main Sweep Function
# =============================================================================

def run_sweep(
    Ns: List[int],
    depths: List[int],
    K: int,
    shots: int,
    p1: float,
    p2: float,
    pm: float,
    seed: int,
    outfile: str,
):
    """
    Run the full benchmark sweep across all graph families, N values, and depths.

    For each combination of (graph_family, N, depth):
    1. Generate the coupling graph
    2. Ensure it's connected (take largest component if needed)
    3. Build appropriate noise model
    4. Generate K random circuit instances
    5. Simulate each and compute success probability
    6. Average results and record to CSV

    Args:
        Ns: List of N values (number of qubits) to test
        depths: List of circuit depths to test
        K: Number of random circuit instances per (family, N, depth) point
        shots: Number of measurement shots per circuit
        p1: Single-qubit gate error probability
        p2: Two-qubit gate error probability
        pm: Measurement error probability
        seed: Random seed for reproducibility
        outfile: Output CSV file path
    """
    rng = np.random.default_rng(seed)

    # -----------------------------------------------------------------
    # Define graph family generators
    # Each returns (graph, actual_N) tuple
    # -----------------------------------------------------------------
    families = [
        ('ring', lambda N, rng: (make_ring(N, rng), N)),
        ('grid', lambda N, rng: make_grid(N, rng)),
        ('random_regular_4', lambda N, rng: (make_random_regular(N, 4, rng), N)),
        ('small_world', lambda N, rng: (make_small_world(N, 4, 0.2, rng), N)),
        ('erdos_renyi', lambda N, rng: (make_erdos_renyi(N, 4.0, rng), N)),
    ]

    records = []
    total_points = len(families) * len(Ns) * len(depths)
    current_point = 0

    # -----------------------------------------------------------------
    # Print configuration summary
    # -----------------------------------------------------------------
    print("=" * 70)
    print("M-flow Benchmark Data Generation")
    print("=" * 70)
    print(f"  Graph families: {[f[0] for f in families]}")
    print(f"  N values: {Ns}")
    print(f"  Depths: {depths}")
    print(f"  Circuits per point (K): {K}")
    print(f"  Shots per circuit: {shots}")
    print(f"  Noise parameters:")
    print(f"    p1 (1-qubit depolarizing): {p1}")
    print(f"    p2 (2-qubit depolarizing): {p2}")
    print(f"    pm (readout error): {pm}")
    print(f"  Random seed: {seed}")
    print(f"  Total data points: {total_points}")
    print(f"  Output file: {outfile}")
    print("=" * 70)
    print()

    # Build noise model once (it's the same for all circuits)
    noise_model = build_noise_model(p1, p2, pm)

    # Create backend with noise model
    backend = AerSimulator(noise_model=noise_model)

    # -----------------------------------------------------------------
    # Main sweep loop
    # -----------------------------------------------------------------
    for family_name, family_fn in families:
        print(f"\n[{family_name}]")

        for N_requested in Ns:
            # Generate graph for this family and N
            G, N_initial = family_fn(N_requested, rng)

            # Ensure connected (may reduce N for sparse random graphs)
            G, N_used = ensure_connected(G)

            # Skip if graph is pathologically small
            if N_used < 4:
                print(f"  N={N_requested}: Skipped (graph too small, N_used={N_used})")
                current_point += len(depths)
                continue

            # Warn if N changed significantly
            if N_used != N_requested and family_name != 'grid':
                print(f"  N={N_requested}: Using N_used={N_used} (largest connected component)")
            elif family_name == 'grid' and N_used != N_requested:
                s = int(round(math.sqrt(N_used)))
                print(f"  N={N_requested}: Using {s}x{s} grid (N_used={N_used})")

            # Device label encodes family and actual N
            device_label = f"{family_name}_N{N_used}"

            for depth in depths:
                current_point += 1

                # Run K independent circuit instances and collect success probs
                success_probs = []
                for k in range(K):
                    # Generate new random circuit for this instance
                    qc = generate_random_circuit(G, depth, rng)

                    # Simulate and get success probability
                    prob = simulate_circuit(qc, backend, shots)
                    success_probs.append(prob)

                # Average success probability across all K instances
                avg_success_prob = float(np.mean(success_probs))
                std_success_prob = float(np.std(success_probs))

                # Record result
                records.append({
                    'device': device_label,
                    'N': N_used,
                    'depth': depth,
                    'success_prob': avg_success_prob,
                })

                # Progress output
                print(f"  {device_label}, d={depth:3d}: "
                      f"success_prob = {avg_success_prob:.6f} "
                      f"(± {std_success_prob:.6f}, K={K}) "
                      f"[{current_point}/{total_points}]")

    # -----------------------------------------------------------------
    # Save results to CSV
    # -----------------------------------------------------------------
    df = pd.DataFrame(records)
    df.to_csv(outfile, index=False)

    # -----------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total rows generated: {len(records)}")
    print(f"Output saved to: {outfile}")
    print()
    print("Results by family:")
    for family_name, _ in families:
        family_rows = df[df['device'].str.startswith(family_name)]
        if not family_rows.empty:
            min_prob = family_rows['success_prob'].min()
            max_prob = family_rows['success_prob'].max()
            mean_prob = family_rows['success_prob'].mean()
            print(f"  {family_name:20s}: {len(family_rows):3d} rows, "
                  f"success_prob in [{min_prob:.6f}, {max_prob:.6f}], "
                  f"mean={mean_prob:.6f}")
    print("=" * 70)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Parse command-line arguments and run the benchmark sweep."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic benchmark data for M-flow test pipeline "
            "using Qiskit Aer noisy quantum circuit simulation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--outfile', type=str, default='bench.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--Ns', type=int, nargs='+', default=[16, 32, 64],
        help='List of N values (number of qubits) to test'
    )
    parser.add_argument(
        '--depths', type=int, nargs='+', default=[5, 10, 20, 40],
        help='List of circuit depths to test'
    )
    parser.add_argument(
        '--K', type=int, default=10,
        help='Number of random circuit instances per (family, N, depth) point'
    )
    parser.add_argument(
        '--shots', type=int, default=8192,
        help='Number of measurement shots per circuit'
    )
    parser.add_argument(
        '--p1', type=float, default=1e-4,
        help='Single-qubit gate depolarizing error probability'
    )
    parser.add_argument(
        '--p2', type=float, default=1e-3,
        help='Two-qubit gate depolarizing error probability'
    )
    parser.add_argument(
        '--pm', type=float, default=0.0,
        help='Measurement/readout error probability (0 = no readout error)'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.K < 1:
        parser.error("K must be at least 1")
    if args.shots < 1:
        parser.error("shots must be at least 1")
    if not (0 <= args.p1 <= 1):
        parser.error("p1 must be in [0, 1]")
    if not (0 <= args.p2 <= 1):
        parser.error("p2 must be in [0, 1]")
    if not (0 <= args.pm <= 1):
        parser.error("pm must be in [0, 1]")

    # Run the sweep
    run_sweep(
        Ns=args.Ns,
        depths=args.depths,
        K=args.K,
        shots=args.shots,
        p1=args.p1,
        p2=args.p2,
        pm=args.pm,
        seed=args.seed,
        outfile=args.outfile,
    )


if __name__ == '__main__':
    main()
