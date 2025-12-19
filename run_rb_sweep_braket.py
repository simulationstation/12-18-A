#!/usr/bin/env python3
"""
AWS Braket experiment driver (RB + logical code)

Supports:
    - Randomized benchmarking (mirror circuits) to estimate EPC/alpha
    - Minimal logical-code experiment (3-qubit repetition) to measure logical
      success across different induced subgraphs.

Runs experiments across diverse qubit subsets on AWS Braket quantum devices,
measuring decay/logical performance and correlating with architecture metrics
(lambda2, C).

Since Braket doesn't have built-in StandardRB, this implements
mirror circuit benchmarking for equivalent decay analysis. Also supports a
compiler fragility experiment that repeatedly compiles/embeds the same GHZ
state to probe variance across layout choices.
"""

import argparse
import csv
import json
import time
import random
import os
import sys
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import numpy as np
import networkx as nx
from scipy.optimize import curve_fit

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


def build_random_spanning_tree(subgraph: nx.Graph, rng: random.Random) -> List[Tuple[int, int]]:
    """Build a randomized spanning tree to serve as a GHZ entangling path."""
    if not nx.is_connected(subgraph):
        raise ValueError("Subgraph must be connected to build GHZ circuit")

    nodes = list(subgraph.nodes())
    root = rng.choice(nodes)
    visited = {root}
    queue = [root]
    entangling_edges: List[Tuple[int, int]] = []

    while queue:
        current = queue.pop(0)
        neighbors = [n for n in subgraph.neighbors(current) if n not in visited]
        rng.shuffle(neighbors)
        for n in neighbors:
            visited.add(n)
            entangling_edges.append((current, n))
            queue.append(n)

    if len(visited) != len(nodes):
        raise ValueError("Failed to span all qubits when building GHZ layout")

    return entangling_edges


def build_randomized_ghz_circuit(subgraph: nx.Graph, rng: random.Random) -> Circuit:
    """
    Construct a shallow GHZ preparation circuit on the given connected subgraph.

    - Choose a random root and randomized spanning tree to define entangling order
    - Apply H on root, then CNOT(parent -> child) along the tree
    """
    entangling_edges = build_random_spanning_tree(subgraph, rng)
    root = entangling_edges[0][0] if entangling_edges else next(iter(subgraph.nodes()))

    circuit = Circuit()
    circuit.h(root)
    for control, target in entangling_edges:
        circuit.cnot(control, target)

    return circuit


def compute_ghz_success_prob(measurements: np.ndarray) -> float:
    """Return GHZ success probability using P(0...0) + P(1...1)."""
    if measurements.ndim != 2:
        raise ValueError("Measurements array must be 2D (shots, qubits)")

    shots = measurements.shape[0]
    if shots == 0:
        raise ValueError("No measurement shots returned")

    qubits = measurements.shape[1]
    ones_counts = np.sum(measurements, axis=1)
    successes = np.sum((ones_counts == 0) | (ones_counts == qubits))
    return successes / shots


def generate_diverse_subsets(G: nx.Graph, N: int, num_subsets: int, seed: int,
                              max_jaccard: float = 0.7, max_attempts: int = 10000) -> List[Tuple[int, ...]]:
    """Generate diverse connected subsets using BFS with Jaccard diversity check."""
    rng = random.Random(seed)

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


def run_compiler_fragility_experiment(device,
                                      subset: Tuple[int, ...],
                                      subgraph: nx.Graph,
                                      shots: int,
                                      num_compilations: int,
                                      seed: int,
                                      logger=None) -> Dict:
    """
    Compile and run the same GHZ circuit multiple times with randomized layouts.

    Returns summary statistics of success probability across compilations.
    """
    if not nx.is_connected(subgraph):
        raise ValueError("Subset must be connected for GHZ preparation")

    rng = random.Random(seed)
    success_probs: List[float] = []

    for k in range(num_compilations):
        try:
            ghz_circuit = build_randomized_ghz_circuit(subgraph, rng)
            for q in subset:
                ghz_circuit.measure(q)

            if logger:
                log(f"    Compilation {k+1}/{num_compilations}", logger)

            task = device.run(ghz_circuit, shots=shots)
            task_result = task.result()

            measurements = np.array(task_result.measurements)
            success_prob = compute_ghz_success_prob(measurements)
            success_probs.append(success_prob)

            if logger:
                log(f"      GHZ success: {success_prob:.4f}", logger)

        except Exception as e:
            if logger:
                log(f"      FAILED compilation {k+1}: {e}", logger)
            success_probs.append(np.nan)

    finite = [p for p in success_probs if not np.isnan(p)]
    if not finite:
        raise ValueError("All GHZ compilation runs failed")

    stats_arr = np.array(finite, dtype=float)
    return {
        'success_probs': success_probs,
        'mean_success': float(np.mean(stats_arr)),
        'std_success': float(np.std(stats_arr, ddof=0)),
        'min_success': float(np.min(stats_arr)),
        'max_success': float(np.max(stats_arr))
    }


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


def choose_data_qubits_for_repetition(subgraph: nx.Graph) -> List[int]:
    """
    Choose three data qubits that form a short path when possible.

    Returns a list of three qubits to serve as the repetition code data block.
    Prefers a length-2 path (a-b-c) to keep entangling operations hardware
    compatible; falls back to the first three nodes if no path exists.
    """
    nodes = list(subgraph.nodes())
    if len(nodes) < 3:
        raise ValueError("Need at least 3 qubits for repetition code")

    # Try to find a path of length 2 (a-b-c)
    for b in nodes:
        neighbors_b = list(subgraph.neighbors(b))
        for a in neighbors_b:
            for c in neighbors_b:
                if a != c:
                    return [a, b, c]

    return sorted(nodes)[:3]


def edge_in_subgraph(edges: set, q1: int, q2: int) -> bool:
    """Return True if an undirected edge (q1, q2) exists in the subgraph."""
    return (q1, q2) in edges or (q2, q1) in edges or (tuple(sorted((q1, q2))) in edges)


def add_parity_check(circuit: Circuit, data_pair: Tuple[int, int], ancilla: int, edges: set) -> bool:
    """
    Add a Z-parity check for a data pair using a single ancilla.

    Uses CNOTs data->ancilla. Returns True if both required edges exist;
    otherwise leaves the circuit unchanged and returns False.
    """
    q1, q2 = data_pair
    if not (edge_in_subgraph(edges, q1, ancilla) and edge_in_subgraph(edges, q2, ancilla)):
        return False
    circuit.cnot(q1, ancilla)
    circuit.cnot(q2, ancilla)
    return True


def build_repetition_code_circuit(data_qubits: List[int], ancillas: List[int], rounds: int, edges: List[Tuple[int, int]]) -> Circuit:
    """
    Build a simple 3-qubit repetition code circuit with parity checks.

    - Data qubits encode logical |0>
    - Each round performs two parity checks (Z1Z2, Z2Z3) using distinct ancillas
    - Adds shallow CZ entangling between neighboring data qubits when allowed
    - Measures all used qubits at the end
    """
    circuit = Circuit()
    edge_set = set(tuple(sorted(e)) for e in edges)

    if len(data_qubits) != 3:
        raise ValueError("Repetition code expects exactly 3 data qubits")
    if len(ancillas) < 2 * rounds:
        raise ValueError("Need 2 ancillas per round for parity checks")

    d0, d1, d2 = data_qubits

    for r in range(rounds):
        a_z12 = ancillas[2 * r]
        a_z23 = ancillas[2 * r + 1]

        added_12 = add_parity_check(circuit, (d0, d1), a_z12, edge_set)
        added_23 = add_parity_check(circuit, (d1, d2), a_z23, edge_set)

        # Add shallow entangling/idling to expose architecture effects
        if edge_in_subgraph(edge_set, d0, d1):
            circuit.cz(d0, d1)
        if edge_in_subgraph(edge_set, d1, d2):
            circuit.cz(d1, d2)

        # If parity checks could not be added due to missing edges, at least add identities
        if not (added_12 and added_23):
            circuit.i(d0)
            circuit.i(d1)
            circuit.i(d2)

    measure_qubits = sorted(set(data_qubits + ancillas))
    for q in measure_qubits:
        circuit.measure(q)

    return circuit


def run_logical_repetition_experiment(device,
                                      subset: Tuple[int, ...],
                                      subgraph: nx.Graph,
                                      rounds: int,
                                      shots: int,
                                      logger=None) -> Dict:
    """
    Run a simple logical repetition code experiment on the given subset.

    Returns logical success/error probabilities and bookkeeping for the run.
    """
    data_qubits = choose_data_qubits_for_repetition(subgraph)
    available = [q for q in subset if q not in data_qubits]
    if len(available) < 2 * rounds:
        raise ValueError(f"Subset {subset} does not have enough ancilla qubits for {rounds} rounds")
    ancillas = available[: 2 * rounds]

    circuit = build_repetition_code_circuit(data_qubits, ancillas, rounds, list(subgraph.edges()))
    measure_qubits = sorted(set(data_qubits + ancillas))

    task = device.run(circuit, shots=shots)
    task_result = task.result()

    measurements = np.array(task_result.measurements)
    measured_qubits = list(task_result.measured_qubits)
    qubit_index = {q: i for i, q in enumerate(measured_qubits)}
    data_indices = [qubit_index[q] for q in data_qubits if q in qubit_index]
    if len(data_indices) != len(data_qubits):
        raise ValueError("Mismatch between measured qubits and data qubits")

    data_bits = measurements[:, data_indices]
    ones_count = np.sum(data_bits, axis=1)
    successes = int(np.sum(ones_count <= len(data_qubits) // 2))
    shots_run = measurements.shape[0]
    logical_success_prob = successes / shots_run if shots_run else 0.0
    logical_error_rate = 1 - logical_success_prob

    if logger:
        log(f"    Logical success probability: {logical_success_prob:.4f}", logger)

    return {
        'logical_success_prob': logical_success_prob,
        'logical_error_rate': logical_error_rate,
        'shots_run': shots_run,
        'data_qubits': data_qubits,
        'ancillas': ancillas,
        'measured_qubits': measured_qubits
    }


def run_logical_code_sweep(device,
                           G: nx.Graph,
                           subsets: List[Tuple[int, ...]],
                           rounds: int,
                           shots: int,
                           seed: int,
                           output_csv: str,
                           logger=None,
                           resume: bool = True):
    """Run logical code experiments across subsets."""

    completed_subsets = set()
    if resume and os.path.exists(output_csv):
        with open(output_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed_subsets.add(int(row['subset_id']))
        log(f"Resuming: {len(completed_subsets)} subsets already completed", logger)

    fieldnames = [
        'experiment_type', 'device', 'subset_id', 'qubits', 'N', 'lambda2', 'C',
        'rounds', 'shots', 'logical_success_prob', 'logical_error_rate',
        'data_qubits', 'ancilla_qubits', 'timestamp', 'status'
    ]
    write_header = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0

    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    start_time = time.time()

    for i, subset in enumerate(subsets):
        if i in completed_subsets:
            log(f"[{i+1}/{len(subsets)}] Subset {i} already completed, skipping", logger)
            continue

        elapsed = (time.time() - start_time) / 60
        log(f"\n[{i+1}/{len(subsets)}] Subset {i} (elapsed: {elapsed:.1f} min)", logger)
        log(f"  Qubits: {subset}", logger)

        subgraph = G.subgraph(subset).copy()
        metrics = compute_architecture_metrics(subgraph)
        log(f"  lambda2={metrics['lambda2']:.4f}, C={metrics['C']:.4f}", logger)

        try:
            results = run_logical_repetition_experiment(
                device,
                subset,
                subgraph,
                rounds,
                shots,
                logger
            )

            row = {
                'experiment_type': 'logical_code',
                'device': device.name if hasattr(device, 'name') else str(device),
                'subset_id': i,
                'qubits': str(subset),
                'N': len(subset),
                'lambda2': metrics['lambda2'],
                'C': metrics['C'],
                'rounds': rounds,
                'shots': shots,
                'logical_success_prob': results['logical_success_prob'],
                'logical_error_rate': results['logical_error_rate'],
                'data_qubits': str(results['data_qubits']),
                'ancilla_qubits': str(results['ancillas']),
                'timestamp': datetime.now().isoformat(),
                'status': 'ok'
            }

            with open(output_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                    write_header = False
                writer.writerow(row)

            log("  Logical result saved to CSV", logger)

        except Exception as e:
            log(f"  FAILED logical run: {e}", logger)
            continue

    elapsed = (time.time() - start_time) / 60
    log(f"\n{'='*60}", logger)
    log(f"LOGICAL CODE SWEEP COMPLETE in {elapsed:.1f} minutes", logger)
    log(f"Results saved to {output_csv}", logger)


def run_compiler_fragility_sweep(device,
                                 G: nx.Graph,
                                 subsets: List[Tuple[int, ...]],
                                 shots: int,
                                 num_compilations: int,
                                 seed: int,
                                 output_csv: str,
                                 logger=None,
                                 resume: bool = True):
    """Run compiler fragility experiments across subsets."""

    completed_subsets = set()
    if resume and os.path.exists(output_csv):
        with open(output_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed_subsets.add(int(row['subset_id']))
        log(f"Resuming: {len(completed_subsets)} subsets already completed", logger)

    fieldnames = [
        'experiment_type', 'device', 'subset_id', 'qubits', 'N', 'lambda2', 'C',
        'num_compilations', 'mean_success', 'std_success', 'min_success', 'max_success', 'timestamp'
    ]
    write_header = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0

    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    start_time = time.time()

    for i, subset in enumerate(subsets):
        if i in completed_subsets:
            log(f"[{i+1}/{len(subsets)}] Subset {i} already completed, skipping", logger)
            continue

        elapsed = (time.time() - start_time) / 60
        log(f"\n[{i+1}/{len(subsets)}] Subset {i} (elapsed: {elapsed:.1f} min)", logger)
        log(f"  Qubits: {subset}", logger)

        subgraph = G.subgraph(subset).copy()
        metrics = compute_architecture_metrics(subgraph)
        log(f"  lambda2={metrics['lambda2']:.4f}, C={metrics['C']:.4f}", logger)

        if not nx.is_connected(subgraph):
            log("  SKIPPING: subset is disconnected", logger)
            continue

        try:
            results = run_compiler_fragility_experiment(
                device,
                subset,
                subgraph,
                shots,
                num_compilations,
                seed + i,
                logger
            )

            row = {
                'experiment_type': 'compiler_fragility',
                'device': device.name if hasattr(device, 'name') else str(device),
                'subset_id': i,
                'qubits': str(subset),
                'N': len(subset),
                'lambda2': metrics['lambda2'],
                'C': metrics['C'],
                'num_compilations': num_compilations,
                'mean_success': results['mean_success'],
                'std_success': results['std_success'],
                'min_success': results['min_success'],
                'max_success': results['max_success'],
                'timestamp': datetime.now().isoformat()
            }

            with open(output_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                    write_header = False
                writer.writerow(row)

            log("  Compiler fragility result saved to CSV", logger)

        except Exception as e:
            log(f"  FAILED compiler fragility run: {e}", logger)
            continue

    elapsed = (time.time() - start_time) / 60
    log(f"\n{'='*60}", logger)
    log(f"COMPILER FRAGILITY SWEEP COMPLETE in {elapsed:.1f} minutes", logger)
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
    parser.add_argument('--output', type=str, default=None, help='Output CSV (auto-set by experiment type)')
    parser.add_argument('--resume', action='store_true', default=True, help='Resume from existing results')
    parser.add_argument('--simulator', action='store_true', help='Use local simulator instead of QPU')
    parser.add_argument('--experiment', type=str, choices=['rb', 'logical_code', 'compiler_fragility'], default='rb',
                        help='Experiment type: rb (mirror RB), logical_code (repetition code), or compiler_fragility (GHZ variance)')
    parser.add_argument('--logical-rounds', type=int, default=2, help='Parity-check rounds for logical experiment')
    parser.add_argument('--num-compilations', type=int, default=10,
                        help='Number of compilations/layouts for compiler fragility experiment')

    args = parser.parse_args()

    depths = [int(d) for d in args.depths.split(',')]
    output_path = args.output
    if output_path is None:
        if args.experiment == 'logical_code':
            output_path = 'results/logical_code_sweep.csv'
        elif args.experiment == 'compiler_fragility':
            output_path = 'results/compiler_fragility_sweep.csv'
        else:
            output_path = 'results/rb_sweep_braket.csv'

    # Setup logging
    os.makedirs('results', exist_ok=True)
    log_file = output_path.replace('.csv', '.log')
    logger = open(log_file, 'a')

    log("=" * 60, logger)
    log("AWS Braket Experiment Runner", logger)
    log("=" * 60, logger)
    log(f"Device: {args.device}", logger)
    log(f"Subset size N: {args.N}", logger)
    log(f"Num subsets: {args.num_subsets}", logger)
    log(f"Experiment: {args.experiment}", logger)
    if args.experiment == 'rb':
        log(f"Depths: {depths}", logger)
    elif args.experiment == 'logical_code':
        log(f"Rounds: {args.logical_rounds}", logger)
    else:
        log(f"Num compilations: {args.num_compilations}", logger)
    log(f"Shots: {args.shots}", logger)
    log(f"Seed: {args.seed}", logger)
    log(f"Output CSV: {output_path}", logger)
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

    # Run chosen experiment
    log("\n" + "=" * 60, logger)
    if args.experiment == 'rb':
        log("STARTING RB SWEEP", logger)
        log("=" * 60, logger)
        run_sweep(device, G, subsets, depths, args.shots, args.seed,
                  output_path, logger, args.resume)
    elif args.experiment == 'logical_code':
        log("STARTING LOGICAL CODE SWEEP", logger)
        log("=" * 60, logger)
        run_logical_code_sweep(device, G, subsets, args.logical_rounds,
                               args.shots, args.seed, output_path, logger, args.resume)
    else:
        log("STARTING COMPILER FRAGILITY SWEEP", logger)
        log("=" * 60, logger)
        run_compiler_fragility_sweep(device, G, subsets, args.shots,
                                     args.num_compilations, args.seed, output_path,
                                     logger, args.resume)

    logger.close()


if __name__ == '__main__':
    main()
