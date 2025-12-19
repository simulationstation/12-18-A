#!/usr/bin/env python3
"""
AWS Braket RB Sweep Script

Runs randomized benchmarking experiments across diverse qubit subsets
on AWS Braket quantum devices, measuring alpha/EPC and correlating
with architecture metrics (lambda2, C).

Since Braket doesn't have built-in StandardRB, this implements
mirror circuit benchmarking for equivalent decay analysis.
"""

import argparse
import csv
import json
import time
import random
import os
import sys
from datetime import datetime
from collections import deque
from typing import List, Tuple, Dict, Optional
import numpy as np
import networkx as nx
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

from braket.aws import AwsDevice
from braket.circuits import Circuit, gates
from braket.devices import LocalSimulator


def log(msg: str, logger=None):
    """Print timestamped message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    if logger:
        logger.write(line + "\n")
        logger.flush()


def build_coupling_graph(device) -> nx.Graph:
    """Build networkx graph from device connectivity."""
    G = nx.Graph()
    connectivity = device.properties.paradigm.connectivity.connectivityGraph
    qubit_count = device.properties.paradigm.qubitCount

    for q_str, neighbors in connectivity.items():
        q = int(q_str)
        if q >= qubit_count:
            continue  # Skip invalid qubits
        G.add_node(q)
        for n_str in neighbors:
            n = int(n_str)
            if n < qubit_count:  # Only add valid edges
                G.add_edge(q, n)

    return G


def compute_normalized_laplacian_lambda2(G: nx.Graph) -> float:
    """Compute second smallest eigenvalue of normalized Laplacian."""
    if len(G) < 2:
        return 0.0
    L = nx.normalized_laplacian_matrix(G).toarray()
    eigenvalues = np.linalg.eigvalsh(L)
    eigenvalues = np.sort(eigenvalues)
    return float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0


def compute_architecture_metrics(G: nx.Graph) -> Dict:
    """Compute architecture metrics for a subgraph."""
    n = len(G)
    if n < 2:
        return {'lambda2': 0, 'C': 0, 'edges': 0, 'avg_degree': 0, 'diameter': 0}

    lambda2 = compute_normalized_laplacian_lambda2(G)
    C = n * lambda2
    edges = G.number_of_edges()
    avg_degree = 2 * edges / n

    try:
        diameter = nx.diameter(G)
    except nx.NetworkXError:
        diameter = -1  # Disconnected

    return {
        'lambda2': lambda2,
        'C': C,
        'edges': edges,
        'avg_degree': avg_degree,
        'diameter': diameter
    }


def generate_diverse_subsets(G: nx.Graph, N: int, num_subsets: int, seed: int,
                              max_jaccard: float = 0.7, max_attempts: int = 10000) -> List[Tuple[int, ...]]:
    """Generate diverse connected subsets using BFS with Jaccard diversity check."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    nodes = list(G.nodes())
    subsets = []
    attempts = 0

    while len(subsets) < num_subsets and attempts < max_attempts:
        attempts += 1

        # BFS from random start
        start = rng.choice(nodes)
        visited = [start]
        frontier = list(G.neighbors(start))
        rng.shuffle(frontier)

        while len(visited) < N and frontier:
            next_node = frontier.pop(0)
            if next_node not in visited:
                visited.append(next_node)
                new_neighbors = [n for n in G.neighbors(next_node) if n not in visited and n not in frontier]
                rng.shuffle(new_neighbors)
                frontier.extend(new_neighbors)

        if len(visited) < N:
            continue

        subset = tuple(sorted(int(x) for x in visited[:N]))

        # Check Jaccard similarity with existing subsets
        too_similar = False
        subset_set = set(subset)
        for existing in subsets:
            existing_set = set(existing)
            jaccard = len(subset_set & existing_set) / len(subset_set | existing_set)
            if jaccard > max_jaccard:
                too_similar = True
                break

        if not too_similar and subset not in subsets:
            subsets.append(subset)

    return subsets


def generate_random_clifford_layer(qubits: List[int], rng: random.Random) -> Circuit:
    """Generate a random Clifford layer on the given qubits."""
    circuit = Circuit()

    # Single qubit Cliffords
    single_gates = [
        lambda q: gates.I(),
        lambda q: gates.X(),
        lambda q: gates.Y(),
        lambda q: gates.Z(),
        lambda q: gates.H(),
        lambda q: gates.S(),
        lambda q: gates.Si(),  # S-dagger
    ]

    for q in qubits:
        gate_fn = rng.choice(single_gates)
        gate = gate_fn(q)
        circuit.add_instruction(gates.Instruction(gate, [q]))

    return circuit


def generate_mirror_circuit(qubits: List[int], depth: int, edges: List[Tuple[int, int]],
                            rng: random.Random) -> Circuit:
    """
    Generate a mirror/inverse circuit for benchmarking.
    Applies random unitaries then their inverses, should return to |0...0>.
    """
    circuit = Circuit()

    # Store gate operations as tuples: (gate_name, qubit(s))
    all_ops = []

    # Forward pass: build up random layers
    for _ in range(depth):
        layer_ops = []

        # Random single-qubit gates
        for q in qubits:
            gate_choice = rng.randint(0, 5)
            if gate_choice == 0:
                circuit.h(q)
                layer_ops.append(('h', q))
            elif gate_choice == 1:
                circuit.x(q)
                layer_ops.append(('x', q))
            elif gate_choice == 2:
                circuit.y(q)
                layer_ops.append(('y', q))
            elif gate_choice == 3:
                circuit.z(q)
                layer_ops.append(('z', q))
            elif gate_choice == 4:
                circuit.s(q)
                layer_ops.append(('s', q))
            elif gate_choice == 5:
                circuit.t(q)
                layer_ops.append(('t', q))

        # Random CZ gates on available edges
        available_edges = list(edges)
        rng.shuffle(available_edges)
        used_qubits = set()

        for q1, q2 in available_edges:
            if q1 not in used_qubits and q2 not in used_qubits:
                if rng.random() < 0.5:  # 50% chance to add CZ
                    circuit.cz(q1, q2)
                    layer_ops.append(('cz', q1, q2))
                    used_qubits.add(q1)
                    used_qubits.add(q2)

        all_ops.append(layer_ops)

    # Inverse pass: apply inverses in reverse order
    # H^-1 = H, X^-1 = X, Y^-1 = Y, Z^-1 = Z, S^-1 = Si, T^-1 = Ti, CZ^-1 = CZ
    for layer_ops in reversed(all_ops):
        # Reverse operations within layer too
        for op in reversed(layer_ops):
            if op[0] == 'h':
                circuit.h(op[1])
            elif op[0] == 'x':
                circuit.x(op[1])
            elif op[0] == 'y':
                circuit.y(op[1])
            elif op[0] == 'z':
                circuit.z(op[1])
            elif op[0] == 's':
                circuit.si(op[1])
            elif op[0] == 't':
                circuit.ti(op[1])
            elif op[0] == 'cz':
                circuit.cz(op[1], op[2])

    return circuit


def run_mirror_benchmark(device, qubits: List[int], edges: List[Tuple[int, int]],
                         depths: List[int], shots_per_depth: int, seed: int,
                         logger=None) -> Dict:
    """
    Run mirror circuit benchmark on the device.
    Returns success probabilities for each depth.
    """
    rng = random.Random(seed)
    results = {'depths': depths, 'success_probs': [], 'circuits_run': 0}

    for depth in depths:
        log(f"    Depth {depth}...", logger)

        # Generate and run circuit
        circuit = generate_mirror_circuit(qubits, depth, edges, rng)

        # Add measurements
        for q in qubits:
            circuit.measure(q)

        # Run on device
        task = device.run(circuit, shots=shots_per_depth)
        task_result = task.result()

        # Calculate success probability (all zeros)
        counts = task_result.measurement_counts
        target_state = '0' * len(qubits)
        success_count = counts.get(target_state, 0)
        success_prob = success_count / shots_per_depth

        results['success_probs'].append(success_prob)
        results['circuits_run'] += 1

        log(f"    Depth {depth}: success_prob = {success_prob:.4f}", logger)

    return results


def fit_rb_decay(depths: List[int], success_probs: List[float]) -> Tuple[float, float, float, float]:
    """
    Fit exponential decay: f(m) = A * alpha^m + B
    Returns (alpha, alpha_err, epc, epc_err)
    """
    def decay_model(m, A, alpha, B):
        return A * np.power(alpha, m) + B

    try:
        # Initial guess
        p0 = [0.5, 0.9, 0.5]
        bounds = ([0, 0, 0], [1, 1, 1])

        popt, pcov = curve_fit(decay_model, depths, success_probs, p0=p0, bounds=bounds, maxfev=5000)
        A, alpha, B = popt
        perr = np.sqrt(np.diag(pcov))
        alpha_err = perr[1]

        # EPC = (1 - alpha) for depolarizing channel approximation
        epc = 1 - alpha
        epc_err = alpha_err

        return alpha, alpha_err, epc, epc_err
    except Exception as e:
        return np.nan, np.nan, np.nan, np.nan


def run_sweep(device, G: nx.Graph, subsets: List[Tuple[int, ...]],
              depths: List[int], shots: int, seed: int,
              output_csv: str, logger=None, resume: bool = True):
    """Run the full sweep across all subsets."""

    # Check for existing results
    completed_subsets = set()
    if resume and os.path.exists(output_csv):
        with open(output_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed_subsets.add(int(row['subset_id']))
        log(f"Resuming: {len(completed_subsets)} subsets already completed", logger)

    # Prepare CSV
    fieldnames = ['device', 'subset_id', 'qubits', 'N', 'lambda2', 'C',
                  'edges_count', 'avg_degree', 'diameter', 'depths', 'shots',
                  'alpha', 'alpha_err', 'epc', 'epc_err', 'success_probs',
                  'task_ids', 'timestamp', 'status']

    write_header = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0

    start_time = time.time()

    for i, subset in enumerate(subsets):
        if i in completed_subsets:
            log(f"[{i+1}/{len(subsets)}] Subset {i} already completed, skipping", logger)
            continue

        elapsed = (time.time() - start_time) / 60
        log(f"\n[{i+1}/{len(subsets)}] Subset {i} (elapsed: {elapsed:.1f} min)", logger)
        log(f"  Qubits: {subset}", logger)

        # Compute metrics
        subgraph = G.subgraph(subset).copy()
        metrics = compute_architecture_metrics(subgraph)

        log(f"  lambda2={metrics['lambda2']:.4f}, C={metrics['C']:.4f}", logger)
        log(f"  edges={metrics['edges']}, avg_degree={metrics['avg_degree']:.2f}, diameter={metrics['diameter']}", logger)

        if metrics['diameter'] == -1:
            log(f"  SKIPPING: disconnected subset", logger)
            continue

        # Get edges in subset
        edges = list(subgraph.edges())

        # Run benchmark
        log(f"  Running mirror benchmark...", logger)
        try:
            results = run_mirror_benchmark(
                device, list(subset), edges, depths, shots, seed + i, logger
            )

            # Fit decay
            alpha, alpha_err, epc, epc_err = fit_rb_decay(depths, results['success_probs'])

            log(f"  SUCCESS: alpha={alpha:.4f}, EPC={epc:.4f}", logger)

            # Save result
            row = {
                'device': device.name,
                'subset_id': i,
                'qubits': str(subset),
                'N': len(subset),
                'lambda2': metrics['lambda2'],
                'C': metrics['C'],
                'edges_count': metrics['edges'],
                'avg_degree': metrics['avg_degree'],
                'diameter': metrics['diameter'],
                'depths': str(depths),
                'shots': shots,
                'alpha': alpha,
                'alpha_err': alpha_err,
                'epc': epc,
                'epc_err': epc_err,
                'success_probs': str(results['success_probs']),
                'task_ids': '',
                'timestamp': datetime.now().isoformat(),
                'status': 'ok'
            }

            with open(output_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                    write_header = False
                writer.writerow(row)

            log(f"  Result saved to CSV", logger)

        except Exception as e:
            log(f"  FAILED: {e}", logger)
            continue

    elapsed = (time.time() - start_time) / 60
    log(f"\n{'='*60}", logger)
    log(f"SWEEP COMPLETE in {elapsed:.1f} minutes", logger)
    log(f"Results saved to {output_csv}", logger)


def main():
    parser = argparse.ArgumentParser(description='AWS Braket RB Sweep')
    parser.add_argument('--device', type=str, default='arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3',
                        help='Braket device ARN')
    parser.add_argument('--N', type=int, default=7, help='Subset size')
    parser.add_argument('--num-subsets', type=int, default=30, help='Number of subsets')
    parser.add_argument('--depths', type=str, default='1,2,4,8,16,32', help='Comma-separated depths')
    parser.add_argument('--shots', type=int, default=1000, help='Shots per circuit')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='results/rb_sweep_braket.csv', help='Output CSV')
    parser.add_argument('--resume', action='store_true', default=True, help='Resume from existing results')
    parser.add_argument('--simulator', action='store_true', help='Use local simulator instead of QPU')

    args = parser.parse_args()

    depths = [int(d) for d in args.depths.split(',')]

    # Setup logging
    os.makedirs('results', exist_ok=True)
    log_file = args.output.replace('.csv', '.log')
    logger = open(log_file, 'a')

    log("=" * 60, logger)
    log("AWS Braket RB Sweep", logger)
    log("=" * 60, logger)
    log(f"Device: {args.device}", logger)
    log(f"Subset size N: {args.N}", logger)
    log(f"Num subsets: {args.num_subsets}", logger)
    log(f"Depths: {depths}", logger)
    log(f"Shots: {args.shots}", logger)
    log(f"Seed: {args.seed}", logger)
    log(f"Output CSV: {args.output}", logger)
    log(f"Resume: {args.resume}", logger)
    log(f"Simulator: {args.simulator}", logger)
    log("=" * 60, logger)

    # Connect to device
    print("\nConnecting to device...")
    if args.simulator:
        device = LocalSimulator()
        log("Using local simulator", logger)
        # For simulator, create a mock topology
        G = nx.grid_2d_graph(10, 10)
        G = nx.convert_node_labels_to_integers(G)
    else:
        device = AwsDevice(args.device)
        log(f"Device: {device.name}, Status: {device.status}", logger)
        G = build_coupling_graph(device)

    log(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", logger)

    # Generate subsets
    print(f"\nGenerating {args.num_subsets} diverse subsets of size {args.N}...")
    subsets = generate_diverse_subsets(G, args.N, args.num_subsets, args.seed)
    log(f"Generated {len(subsets)} subsets", logger)

    # Run sweep
    log("\n" + "=" * 60, logger)
    log("STARTING RB SWEEP", logger)
    log("=" * 60, logger)

    run_sweep(device, G, subsets, depths, args.shots, args.seed,
              args.output, logger, args.resume)

    logger.close()


if __name__ == '__main__':
    main()
