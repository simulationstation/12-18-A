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
import os
import sys
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse.linalg import eigsh

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


def normalized_laplacian_lambda2(G: nx.Graph) -> float:
    """
    Compute λ2 of the normalized Laplacian L = I - D^{-1/2} A D^{-1/2}.

    λ2 is the spectral gap (second-smallest eigenvalue) and measures
    how well-connected the graph is. Higher λ2 indicates better expansion.

    Args:
        G: A connected NetworkX graph with at least 2 nodes

    Returns:
        The second-smallest eigenvalue of the normalized Laplacian
    """
    if G.number_of_nodes() < 2:
        raise ValueError("Graph must have at least 2 nodes to compute lambda2")
    L = nx.normalized_laplacian_matrix(G)  # sparse matrix
    vals = eigsh(L, k=2, which="SM", return_eigenvectors=False, tol=1e-6, maxiter=5000)
    vals = np.sort(vals)
    return float(vals[1])


# =============================================================================
# Circuit Generation Functions
# =============================================================================

def random_maximal_matching(
    G: nx.Graph,
    rng: np.random.Generator,
    limit_entanglement: bool = False,
) -> List[Tuple[int, int]]:
    """
    Build a random maximal matching by shuffling edges and greedily selecting.

    A maximal matching is a set of edges with no shared vertices such that
    no additional edge can be added. This represents one parallel layer of
    two-qubit gates respecting the coupling constraints.

    Args:
        G: The coupling graph
        rng: Random generator for shuffling
        limit_entanglement: If True, prefer edges with small |u-v| (local edges)
            to keep entanglement more tractable for MPS simulation.

    Returns:
        List of (u, v) edge tuples forming the maximal matching
    """
    edges = list(G.edges())

    if limit_entanglement:
        # Sort edges by |u-v| with small random jitter to break ties
        # This biases toward local edges in qubit ordering, helping MPS
        jitter_scale = 0.5
        edges_with_key = [
            (abs(u - v) + rng.uniform(0, jitter_scale), u, v)
            for u, v in edges
        ]
        edges_with_key.sort(key=lambda x: x[0])
        edges = [(u, v) for _, u, v in edges_with_key]
    else:
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
    limit_entanglement: bool = False,
    benchmark: str = "allzeros",
    oneq_set: str = "default",
) -> QuantumCircuit:
    """
    Generate a random circuit constrained by coupling graph G.

    For each of the `depth` layers:
    1. Apply a random single-qubit gate to each qubit
       (chosen from {SX, X, RZ(θ)} with θ uniform in [0, 2π])
    2. Select a random maximal matching of the coupling graph
    3. Apply CX gates on each matched edge

    If benchmark="mirror", append the inverse of the circuit (U followed by U†).
    This makes success_prob measurable at large N since noiseless result is ~1.

    Finally, measure all qubits in the computational basis.

    Args:
        G: The coupling graph defining allowed two-qubit interactions
        depth: Number of layers
        rng: Random generator for reproducibility
        limit_entanglement: If True, bias matching toward local edges for MPS
        benchmark: "allzeros" (standard) or "mirror" (append inverse circuit)

    Returns:
        QuantumCircuit with depth layers (+ inverse if mirror) and final measurement
    """
    N = G.number_of_nodes()
    qc = QuantumCircuit(N, N)

    # Available single-qubit gates (standard basis gates for most backends)
    if oneq_set == "light":
        single_qubit_gates = ['x', 'rz']
    else:
        single_qubit_gates = ['sx', 'x', 'rz']

    # Record operations for mirror mode
    # Each entry: ('1q', gate_name, qubit, theta_or_None) or ('2q', 'cx', u, v)
    operations = []

    for layer in range(depth):
        # -----------------------------------------------------------------
        # Step 1: Apply random single-qubit gates to all qubits
        # -----------------------------------------------------------------
        for q in range(N):
            gate_choice = rng.choice(single_qubit_gates)
            if gate_choice == 'sx':
                qc.sx(q)
                operations.append(('1q', 'sx', q, None))
            elif gate_choice == 'x':
                qc.x(q)
                operations.append(('1q', 'x', q, None))
            elif gate_choice == 'rz':
                theta = rng.uniform(0, 2 * math.pi)
                qc.rz(theta, q)
                operations.append(('1q', 'rz', q, theta))

        # -----------------------------------------------------------------
        # Step 2: Get random maximal matching and apply CX gates
        # -----------------------------------------------------------------
        matching = random_maximal_matching(G, rng, limit_entanglement)
        for (u, v) in matching:
            qc.cx(u, v)
            operations.append(('2q', 'cx', u, v))

    # -----------------------------------------------------------------
    # Step 3: If mirror mode, append inverse circuit
    # Inverse: X→X, SX→SXdg, RZ(θ)→RZ(-θ), CX→CX
    # -----------------------------------------------------------------
    if benchmark == "mirror":
        for op in reversed(operations):
            if op[0] == '1q':
                gate_name, q, theta = op[1], op[2], op[3]
                if gate_name == 'sx':
                    qc.sxdg(q)
                elif gate_name == 'x':
                    qc.x(q)
                elif gate_name == 'rz':
                    qc.rz(-theta, q)
            elif op[0] == '2q':
                # CX is self-inverse
                u, v = op[2], op[3]
                qc.cx(u, v)

    # -----------------------------------------------------------------
    # Step 4: Measure all qubits in computational basis
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
            'sx', 'sxdg', 'x', 'rz',  # Gates we explicitly use (including inverse)
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
    shots_batch: int = 0,
    status_cb: Optional[Callable[[int, int], None]] = None,
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
    N = qc.num_qubits
    all_zeros = '0' * N

    zero_count_total = 0
    shots_done = 0
    while shots_done < shots:
        batch = shots if shots_batch <= 0 else min(shots_batch, shots - shots_done)
        job = backend.run(qc, shots=batch)
        result = job.result()
        counts = result.get_counts()
        zero_count_total += counts.get(all_zeros, 0)
        shots_done += batch

        if status_cb is not None:
            status_cb(shots_done, zero_count_total)

    success_prob = zero_count_total / shots

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
    global_gamma: float = 0.0,
    sim_method: str = "auto",
    mps_max_bond: int = 256,
    mps_trunc: float = 1e-10,
    limit_entanglement: bool = True,
    benchmark: str = "mirror",
    families_filter: Optional[List[str]] = None,
    status_every_sec: int = 30,
    max_total_executions: int = 200000,
    force: bool = False,
    resume: bool = True,
    shots_batch: int = 0,
    oneq_set: str = "default",
):
    """
    Run the full benchmark sweep across all graph families, N values, and depths.

    For each combination of (graph_family, N, depth):
    1. Generate the coupling graph
    2. Ensure it's connected (take largest component if needed)
    3. Build appropriate noise model
    4. Generate K random circuit instances
    5. Simulate each and compute success probability
    6. Optionally apply global penalty: success_prob *= exp(-global_gamma * C * depth)
    7. Average results and record to CSV

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
        global_gamma: Global penalty strength (0.0 = no penalty)
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

    if families_filter:
        family_names = set(families_filter)
        families = [(name, fn) for name, fn in families if name in family_names]

    if not families:
        print("No graph families selected; nothing to do.")
        return

    records = []
    total_points = len(families) * len(Ns) * len(depths)
    current_point = 0

    # -----------------------------------------------------------------
    # Determine simulation method
    # -----------------------------------------------------------------
    actual_method = sim_method
    if sim_method == "auto":
        actual_method = "matrix_product_state" if max(Ns) >= 32 else "statevector"
    elif sim_method == "mps":
        actual_method = "matrix_product_state"
    # else: statevector stays as-is

    # -----------------------------------------------------------------
    estimated_total = len(families) * len(Ns) * len(depths) * K * shots

    print("=" * 70)
    print("M-flow Benchmark Data Generation")
    print("=" * 70)
    print(f"  Graph families: {[f[0] for f in families]}")
    print(f"  N values: {Ns}")
    print(f"  Depths: {depths}")
    print(f"  Circuits per point (K): {K}")
    print(f"  Shots per circuit: {shots}")
    print(f"  Estimated total circuit executions: {estimated_total}")
    if estimated_total > max_total_executions:
        print(
            f"WARNING: Estimated executions {estimated_total} exceed max_total_executions={max_total_executions}"
        )
        if not force:
            print("Use smaller shots/K/families/depths or set --force=1 to proceed.")
            return
    print(f"  Noise parameters:")
    print(f"    p1 (1-qubit depolarizing): {p1}")
    print(f"    p2 (2-qubit depolarizing): {p2}")
    print(f"    pm (readout error): {pm}")
    print(f"  Global penalty (gamma): {global_gamma}")
    print(f"  Benchmark mode: {benchmark}")
    print(f"  Simulation method: {actual_method} (requested: {sim_method})")
    if actual_method == "matrix_product_state":
        print(f"  MPS options:")
        print(f"    max_bond_dimension: {mps_max_bond}")
        print(f"    truncation_threshold: {mps_trunc}")
        print(f"    limit_entanglement: {limit_entanglement}")
    print(f"  Random seed: {seed}")
    print(f"  Total data points: {total_points}")
    print(f"  Output file: {outfile}")
    print("=" * 70)
    print()

    log_path = outfile + ".log"
    log_file = open(log_path, "a", encoding="utf-8")
    start_time = time.time()
    last_status = start_time

    def emit_status(family_idx: int, N_idx: int, depth_idx: int, k_idx: int, last_prob: float | None, running_mean: float | None):
        nonlocal last_status
        now = time.time()
        if now - last_status < status_every_sec:
            return
        elapsed = now - start_time
        status = (
            f"[status] t={elapsed:7.1f}s fam={family_idx}/{len(families)} "
            f"N_idx={N_idx}/{len(Ns)} depth_idx={depth_idx}/{len(depths)} "
            f"k={k_idx}/{K} shots={shots} sim={actual_method} "
            f"last_prob={(last_prob if last_prob is not None else float('nan')):.6f} "
            f"running_mean={(running_mean if running_mean is not None else float('nan')):.6f}"
        )
        print(status, flush=True)
        log_file.write(status + "\n")
        log_file.flush()
        last_status = now

    # Build noise model once (it's the same for all circuits)
    noise_model = build_noise_model(p1, p2, pm)

    # -----------------------------------------------------------------
    # Create backend with appropriate method and options
    # -----------------------------------------------------------------
    if actual_method == "matrix_product_state":
        # Configure MPS backend with noise
        backend = AerSimulator(
            method="matrix_product_state",
            noise_model=noise_model,
        )
        # Set MPS-specific options
        backend.set_options(
            matrix_product_state_max_bond_dimension=mps_max_bond,
            matrix_product_state_truncation_threshold=mps_trunc,
        )
    else:
        # Statevector backend with noise
        backend = AerSimulator(
            method="statevector",
            noise_model=noise_model,
        )

    # -----------------------------------------------------------------
    # Main sweep loop
    # -----------------------------------------------------------------
    completed_points = set()
    header_needed = not os.path.exists(outfile)
    if resume and not header_needed:
        try:
            existing_df = pd.read_csv(outfile, on_bad_lines='skip')
            for _, row in existing_df.iterrows():
                completed_points.add((row['device'], int(row['N']), int(row['depth'])))
        except Exception:
            pass

    try:
        for family_idx, (family_name, family_fn) in enumerate(families, start=1):
            print(f"\n[{family_name}]")

            for N_idx, N_requested in enumerate(Ns, start=1):
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

                # Compute lambda2 and C once per (family, N_used) graph
                lambda2 = normalized_laplacian_lambda2(G)
                C = N_used * lambda2

                # Log when global penalty is active
                if global_gamma > 0:
                    print(f"  [global penalty] {family_name}, N_used={N_used}, "
                          f"lambda2={lambda2:.4e}, C={C:.4e}, gamma={global_gamma}")

                for depth_idx, depth in enumerate(depths, start=1):
                    current_point += 1

                    device_label = f"{family_name}_N{N_used}"
                    if resume and (device_label, N_used, depth) in completed_points:
                        print(f"  {device_label}, d={depth:3d}: Skipping (already in {outfile})")
                        continue

                    # Run K independent circuit instances and collect success probs
                    success_probs = []
                    for k in range(K):
                        # Generate new random circuit for this instance
                        qc = generate_random_circuit(G, depth, rng, limit_entanglement, benchmark, oneq_set)

                        # Simulate and get success probability
                        if shots_batch > 0:
                            def batch_cb(done: int, zero_total: int) -> None:
                                interim_prob = zero_total / done if done else 0.0
                                emit_status(
                                    family_idx,
                                    N_idx,
                                    depth_idx,
                                    k + 1,
                                    interim_prob,
                                    float(np.mean(success_probs + [interim_prob])),
                                )

                            prob = simulate_circuit(qc, backend, shots, shots_batch, batch_cb)
                        else:
                            prob = simulate_circuit(qc, backend, shots, shots_batch)

                        # Apply global penalty if enabled: exp(-gamma * C * depth)
                        if global_gamma > 0:
                            penalty = math.exp(-global_gamma * C * depth)
                            prob = prob * penalty

                        success_probs.append(prob)
                        emit_status(
                            family_idx,
                            N_idx,
                            depth_idx,
                            k + 1,
                            prob,
                            float(np.mean(success_probs)),
                        )

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

                    pd.DataFrame(records[-1:]).to_csv(
                        outfile,
                        mode='a',
                        header=header_needed,
                        index=False,
                    )
                    header_needed = False

                    # Progress output
                    progress_line = (
                        f"  {device_label}, d={depth:3d}: success_prob = {avg_success_prob:.6f} "
                        f"(± {std_success_prob:.6f}, K={K}) [{current_point}/{total_points}]"
                    )
                print(progress_line, flush=True)
                log_file.write(progress_line + "\n")
                log_file.flush()
                last_status = time.time()

    except KeyboardInterrupt:
        print("Interrupted by user. Partial results written; rerun with --resume=1 to continue.")
    finally:
        log_file.close()

    try:
        df = pd.read_csv(outfile, on_bad_lines='skip')
    except Exception:
        df = pd.DataFrame(records)

    if df.empty:
        return

    # -----------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total rows generated in this run: {len(records)}")
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
        '--families', type=str, nargs='+', choices=['ring', 'grid', 'random_regular_4', 'small_world', 'erdos_renyi'],
        default=None,
        help='Subset of graph families to simulate'
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
    parser.add_argument(
        '--global_gamma', type=float, default=0.0,
        help='Global penalty strength. If > 0, applies exp(-gamma * C * depth) penalty '
             'where C = N_used * lambda2(G). Default 0.0 (no penalty).'
    )
    # MPS simulation options
    parser.add_argument(
        '--sim_method', type=str, choices=['auto', 'statevector', 'mps'], default='auto',
        help='Simulation method: auto (mps if max(Ns)>=32 else statevector), statevector, or mps'
    )
    parser.add_argument(
        '--oneq_set', type=str, choices=['default', 'light'], default='default',
        help='Single-qubit gate set; light reduces entanglement growth for MPS'
    )
    parser.add_argument(
        '--mps_max_bond', type=int, default=256,
        help='MPS max bond dimension (only used when sim_method=mps)'
    )
    parser.add_argument(
        '--mps_trunc', type=float, default=1e-10,
        help='MPS truncation threshold (only used when sim_method=mps)'
    )
    parser.add_argument(
        '--limit_entanglement', type=int, choices=[0, 1], default=1,
        help='If 1, bias matching toward local edges (small |u-v|) for MPS efficiency'
    )
    parser.add_argument(
        '--shots_batch', type=int, default=0,
        help='If >0, split shots into batches to provide responsive status updates'
    )
    parser.add_argument(
        '--status_every_sec', type=int, default=30,
        help='Heartbeat interval for status logging'
    )
    parser.add_argument(
        '--max_total_executions', type=int, default=200000,
        help='Guardrail threshold for estimated total circuit executions'
    )
    parser.add_argument(
        '--force', type=int, choices=[0, 1], default=0,
        help='Set to 1 to bypass max_total_executions guardrail'
    )
    parser.add_argument(
        '--resume', type=int, choices=[0, 1], default=1,
        help='If 1, resume from existing outfile by skipping completed rows'
    )
    parser.add_argument(
        '--benchmark', type=str, choices=['allzeros', 'mirror'], default='mirror',
        help='Benchmark type: allzeros (random circuit, measure all zeros) or '
             'mirror (U followed by U†, measure all zeros). Mirror gives measurable '
             'success_prob at large N.'
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
    if args.global_gamma < 0:
        parser.error("global_gamma must be >= 0")
    if args.status_every_sec < 1:
        parser.error("status_every_sec must be at least 1 second")
    if args.max_total_executions < 1:
        parser.error("max_total_executions must be positive")
    if args.shots_batch < 0:
        parser.error("shots_batch must be >= 0")

    default_shots = parser.get_default('shots')
    default_K = parser.get_default('K')
    mps_selected = args.sim_method == 'mps' or (args.sim_method == 'auto' and max(args.Ns) >= 32)
    if mps_selected:
        if args.shots == default_shots:
            args.shots = 512
        if args.K == default_K:
            args.K = 3

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
        global_gamma=args.global_gamma,
        sim_method=args.sim_method,
        mps_max_bond=args.mps_max_bond,
        mps_trunc=args.mps_trunc,
        limit_entanglement=bool(args.limit_entanglement),
        benchmark=args.benchmark,
        families_filter=args.families,
        status_every_sec=args.status_every_sec,
        max_total_executions=args.max_total_executions,
        force=bool(args.force),
        resume=bool(args.resume),
        shots_batch=args.shots_batch,
        oneq_set=args.oneq_set,
    )


if __name__ == '__main__':
    main()
