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
import math
import time
import random
import os
import sys
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import networkx as nx
from scipy.optimize import curve_fit
from scipy.stats import linregress

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


def get_git_commit_short() -> str:
    """Return short git commit hash if available, else ''."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return ""


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


def _device_name(device) -> str:
    """Best-effort device name extraction."""
    return getattr(
        device,
        "name",
        getattr(getattr(device, "properties", None), "deviceParameters", {}).get("name", "unknown"),
    )


def resolve_entangler_mode(device, requested: str) -> str:
    """
    Resolve entangler choice with IonQ-aware defaults.

    - auto -> ZZ on Forte-class, MS on Aria-class, CZ otherwise.
    - explicit choices are returned directly.
    """
    if requested != "auto":
        return requested

    name = _device_name(device).lower()
    if "forte" in name:
        return "zz"
    if "aria" in name:
        return "ms"
    return "cz"


def _default_entangler_params(mode: str) -> Dict[str, float]:
    """Return fixed parameters for parameterized entanglers."""
    if mode in ("zz", "ms"):
        return {"theta": math.pi / 2, "phi": 0.0}
    return {}


def apply_entangler(circuit: Circuit, q1: int, q2: int, params: Dict[str, float], mode: str) -> Tuple[str, Tuple[int, int], Tuple[float, ...]]:
    """Apply a 2Q entangler and return a log tuple for inversion."""
    if mode in ("cz", "cnot", "cx"):
        if hasattr(circuit, "cz"):
            circuit.cz(q1, q2)
        else:
            circuit.cnot(q1, q2)
        return ("cz", (q1, q2), ())

    if mode == "zz":
        theta = params.get("theta", math.pi / 2)
        if hasattr(circuit, "zz"):
            circuit.zz(q1, q2, theta)
        else:
            try:
                circuit.add(gates.ZZ(theta), [q1, q2])
            except Exception as exc:  # pragma: no cover - depends on SDK version
                raise RuntimeError("ZZ gate not supported by installed Braket SDK") from exc
        return ("zz", (q1, q2), (theta,))

    if mode == "ms":
        theta = params.get("theta", math.pi / 2)
        phi = params.get("phi", 0.0)
        ms_method = getattr(circuit, "ms", None)
        if callable(ms_method):
            try:
                ms_method(q1, q2, theta, phi)
            except TypeError:
                ms_method(q1, q2, theta)
        else:
            try:
                circuit.add(gates.MS(theta, phi), [q1, q2])
            except TypeError:
                circuit.add(gates.MS(theta), [q1, q2])
            except Exception as exc:  # pragma: no cover - depends on SDK version
                raise RuntimeError("MS gate not supported by installed Braket SDK") from exc
        return ("ms", (q1, q2), (theta, phi))

    raise ValueError(f"Unsupported entangler mode: {mode}")


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


def compute_all_zero_prob(measurements: np.ndarray) -> float:
    """Return success probability for measuring all zeros."""
    if measurements.ndim != 2:
        raise ValueError("Measurements array must be 2D (shots, qubits)")
    shots = measurements.shape[0]
    if shots == 0:
        raise ValueError("No measurement shots returned")
    zeros = np.sum(np.all(measurements == 0, axis=1))
    return zeros / shots


def invert_instruction(circuit: Circuit, instruction) -> None:
    """Append the inverse of a single instruction to the circuit."""
    name = instruction.operator.name.upper()
    targets = instruction.target

    if name == 'H':
        circuit.h(targets[0])
    elif name == 'X':
        circuit.x(targets[0])
    elif name == 'Y':
        circuit.y(targets[0])
    elif name == 'Z':
        circuit.z(targets[0])
    elif name == 'CNOT':
        circuit.cnot(targets[0], targets[1])
    elif name == 'CZ':
        circuit.cz(targets[0], targets[1])
    elif name == 'RX':
        circuit.rx(targets[0], -instruction.operator.angle)
    elif name == 'RY':
        circuit.ry(targets[0], -instruction.operator.angle)
    elif name == 'RZ':
        circuit.rz(targets[0], -instruction.operator.angle)
    elif name == 'PHASESHIFT':
        circuit.phaseshift(targets[0], -instruction.operator.angle)
    else:
        raise ValueError(f"Unsupported gate for inversion: {name}")


def append_inverse_from_instructions(circuit: Circuit, instructions: List) -> None:
    """Append inverses for a list of instructions to the circuit (reverse order)."""
    for instr in reversed(instructions):
        invert_instruction(circuit, instr)


def _order_edges_for_u_style(edges: List[Tuple[int, int]], rng: random.Random, u_style: str) -> List[Tuple[int, int]]:
    """
    Order entangling edges based on the requested U style.

    mixed: random order (status quo)
    local: favor short-range pairs in the qubit index ordering
    global: favor long-range pairs to maximize mixing
    """
    if u_style == 'mixed':
        ordered = list(edges)
        rng.shuffle(ordered)
        return ordered

    def _score(edge: Tuple[int, int]) -> float:
        distance = abs(edge[0] - edge[1])
        jitter = rng.random() * 1e-6  # deterministic tie-breaker given rng
        return distance + jitter if u_style == 'local' else -(distance + jitter)

    return sorted(edges, key=_score)


def generate_brickwork_unitary(
    qubits: List[int],
    edges: List[Tuple[int, int]],
    depth: int,
    rng: random.Random,
    circuit: Circuit,
    u_style: str = "mixed",
    entangler_mode: str = "cz",
) -> List[List[Tuple[str, Tuple[int, ...], Tuple[float, ...]]]]:
    """
    Generate a brickwork pattern of 1Q rotations + 2Q entanglers.

    Returns a log of operations for later inversion: list over layers, each layer
    is a list of tuples ('rx'/'ry'/'rz'/'cz', params...).
    """
    op_log: List[List[Tuple]] = []
    entangler_params = _default_entangler_params(entangler_mode)

    for _ in range(depth):
        layer_ops: List[Tuple[str, Tuple[int, ...], Tuple[float, ...]]] = []

        # 1Q rotations
        for q in qubits:
            theta_x = rng.uniform(0, 2 * np.pi)
            theta_z = rng.uniform(0, 2 * np.pi)
            circuit.rx(q, theta_x)
            circuit.rz(q, theta_z)
            layer_ops.append(("rx", (q,), (theta_x,)))
            layer_ops.append(("rz", (q,), (theta_z,)))

        # 2Q entanglers on disjoint edges to keep depth modest
        available_edges = _order_edges_for_u_style(list(edges), rng, u_style)
        used: set = set()
        for q1, q2 in available_edges:
            if q1 in used or q2 in used:
                continue
            if rng.random() < 0.5:
                gate = apply_entangler(circuit, q1, q2, entangler_params, entangler_mode)
                layer_ops.append(gate)
                used.add(q1)
                used.add(q2)

        op_log.append(layer_ops)

    return op_log


def append_inverse_brickwork(circuit: Circuit, op_log: List[List[Tuple[str, Tuple[int, ...], Tuple[float, ...]]]]) -> None:
    """Append the exact inverse of the brickwork operations."""
    for layer_ops in reversed(op_log):
        for op in reversed(layer_ops):
            name, qubits, params = op
            if name == "rx":
                circuit.rx(qubits[0], -params[0])
            elif name == "ry":
                circuit.ry(qubits[0], -params[0])
            elif name == "rz":
                circuit.rz(qubits[0], -params[0])
            elif name in ("cz", "zz", "ms"):
                inv_params = {"theta": -params[0]} if params else {}
                if len(params) > 1:
                    inv_params["phi"] = params[1]
                apply_entangler(circuit, qubits[0], qubits[1], inv_params, name)
            else:
                raise ValueError(f"Unknown op in brickwork log: {name}")


def build_loschmidt_echo_circuit(subgraph: nx.Graph,
                                 depth: int,
                                 rng: random.Random,
                                 u_style: str = 'mixed',
                                 entangler_mode: str = "cz") -> Circuit:
    """
    Build a Loschmidt echo circuit:
        - Prepare GHZ on the connected subgraph
        - Apply forward brickwork unitary U (depth layers)
        - Apply exact inverse U†
        - Unprepare GHZ (inverse of prep)
        - Measure all qubits in Z
    """
    if not nx.is_connected(subgraph):
        raise ValueError("Subgraph must be connected for Loschmidt echo")

    qubits = sorted(subgraph.nodes())
    edges = list(subgraph.edges())

    circuit = Circuit()

    # GHZ preparation with recorded instructions for inversion
    ghz_circuit = build_randomized_ghz_circuit(subgraph, rng)
    ghz_instructions = list(ghz_circuit.instructions)
    circuit.add_circuit(ghz_circuit)

    # Forward unitary
    op_log = generate_brickwork_unitary(qubits, edges, depth, rng, circuit, u_style, entangler_mode)

    # Exact inverse
    append_inverse_brickwork(circuit, op_log)

    # Uncompute GHZ
    append_inverse_from_instructions(circuit, ghz_instructions)

    # Measurements
    for q in qubits:
        circuit.measure(q)

    return circuit


def run_loschmidt_echo_instance(device,
                                subset: Tuple[int, ...],
                                subgraph: nx.Graph,
                                depth: int,
                                shots: int,
                                rng: random.Random,
                                u_style: str = 'mixed',
                                entangler_mode: str = "cz",
                                logger=None) -> Tuple[float, float]:
    """Run a single Loschmidt echo instance and return (P_return, GHZ_witness)."""
    circuit = build_loschmidt_echo_circuit(subgraph, depth, rng, u_style, entangler_mode)
    task = device.run(circuit, shots=shots)
    result = task.result()
    measurements = np.array(result.measurements)

    p_return = compute_all_zero_prob(measurements)
    ghz_witness = compute_ghz_success_prob(measurements)

    if logger:
        log(f"        P_return={p_return:.4f}, GHZ witness={ghz_witness:.4f}", logger)

    return p_return, ghz_witness


def fit_loschmidt_alpha(depths: List[int], p_returns: List[float], fit_eps: float) -> Tuple[float, float, int]:
    """
    Fit slope alpha via linear regression of -log(P_return) vs depth.

    Only include points with P_return > fit_eps to avoid log underflow.
    Returns (alpha, stderr, num_points).
    """
    valid = [(d, p) for d, p in zip(depths, p_returns) if p > fit_eps]
    if len(valid) < 2:
        return np.nan, np.nan, len(valid)
    xs, ys = zip(*valid)
    neg_logs = [-np.log(p) for p in ys]
    res = linregress(xs, neg_logs)
    return res.slope, res.stderr, len(valid)


def entangler_smoke_test(entangler_mode: str, u_style: str = "mixed") -> None:
    """Build a tiny circuit with the entangler and run it on the local simulator."""
    subgraph = nx.path_graph(4)
    rng = random.Random(123)
    circ = build_loschmidt_echo_circuit(subgraph, depth=2, rng=rng, u_style=u_style, entangler_mode=entangler_mode)
    ir_text = circ.to_ir().json().lower()
    if entangler_mode in ("zz", "ms") and "cz" in ir_text:
        raise RuntimeError("Entangler smoke test detected CZ in IR when non-CZ entangler was requested.")
    sim = LocalSimulator()
    sim.run(circ, shots=4).result()


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


def _sample_connected_subset(G: nx.Graph, N: int, rng: random.Random, max_attempts: int = 1000) -> Optional[Tuple[int, ...]]:
    """Sample a single connected subset of size N using randomized BFS."""
    nodes = list(G.nodes())
    for _ in range(max_attempts):
        start = rng.choice(nodes)
        visited = [start]
        frontier = list(G.neighbors(start))
        rng.shuffle(frontier)

        while len(visited) < N and frontier:
            nxt = frontier.pop(0)
            if nxt not in visited:
                visited.append(nxt)
                new_neighbors = [n for n in G.neighbors(nxt) if n not in visited and n not in frontier]
                rng.shuffle(new_neighbors)
                frontier.extend(new_neighbors)

        if len(visited) < N:
            continue

        subset = tuple(sorted(int(x) for x in visited[:N]))
        subgraph = G.subgraph(subset)
        if not nx.is_connected(subgraph):
            continue
        return subset
    return None


def select_subsets_max_spread_C(
    G: nx.Graph,
    N: int,
    num_subsets: int,
    seed: int,
    candidate_pool: int = 200,
) -> Tuple[List[Tuple[int, ...]], Dict[int, Dict]]:
    """
    Select subsets that maximize spread in the architecture metric C while limiting overlap.

    - Build a candidate pool of connected subsets
    - Compute C = N * lambda2 for each
    - Choose half from lowest C and half from highest C
    - Greedily minimize qubit overlap within each bucket
    """
    rng = random.Random(seed)
    candidates: List[Tuple[int, ...]] = []
    candidate_metrics: List[Dict] = []

    attempts = 0
    while len(candidates) < candidate_pool and attempts < candidate_pool * 50:
        attempts += 1
        subset = _sample_connected_subset(G, N, rng)
        if subset is None or subset in candidates:
            continue
        metrics = compute_architecture_metrics(G.subgraph(subset).copy())
        candidates.append(subset)
        candidate_metrics.append(metrics)

    if len(candidates) < num_subsets:
        raise ValueError(f"Could not generate enough candidate subsets (got {len(candidates)})")

    indexed = list(zip(candidates, candidate_metrics))
    indexed.sort(key=lambda x: x[1]['C'])
    low_target = num_subsets // 2
    high_target = num_subsets - low_target
    low_bucket = indexed[:max(low_target * 2, low_target + 1)]
    high_bucket = indexed[-max(high_target * 2, high_target + 1):]

    def pick_from_bucket(bucket, target, existing_sets):
        selected_local: List[Tuple[Tuple[int, ...], Dict]] = []
        remaining = list(bucket)
        while len(selected_local) < target and remaining:
            best = None
            best_score = None
            rng.shuffle(remaining)
            for cand_subset, cand_metrics in remaining:
                score = 0
                c_set = set(cand_subset)
                for chosen, _ in existing_sets + selected_local:
                    score += len(c_set & set(chosen))
                if best is None or score < best_score:
                    best = (cand_subset, cand_metrics)
                    best_score = score
            if best is None:
                break
            selected_local.append(best)
            remaining.remove(best)
        return selected_local

    chosen: List[Tuple[Tuple[int, ...], Dict]] = []
    chosen += pick_from_bucket(low_bucket, low_target, chosen)
    chosen += pick_from_bucket(high_bucket, high_target, chosen)

    if len(chosen) < num_subsets:
        extra_pool = [item for item in indexed if item not in chosen]
        extra = pick_from_bucket(extra_pool, num_subsets - len(chosen), chosen)
        chosen += extra

    subsets = [s for s, _ in chosen[:num_subsets]]
    metrics_map = {i: m for i, (_, m) in enumerate(chosen[:num_subsets])}
    return subsets, metrics_map


def summarize_subsets(
    subsets: List[Tuple[int, ...]],
    G: nx.Graph,
    device_name: str,
    N_requested: int,
    seed: int,
    selection_params: Dict,
    save_path: str = None,
) -> Dict[int, Dict]:
    """
    Compute metrics for subsets and optionally persist to JSON for reuse.

    Returns a mapping subset_id -> metrics dict.
    """
    metrics_map: Dict[int, Dict] = {}
    records = []
    for i, subset in enumerate(subsets):
        subgraph = G.subgraph(subset).copy()
        metrics = compute_architecture_metrics(subgraph)
        metrics_map[i] = metrics
        records.append({
            "subset_id": i,
            "qubits": [int(q) for q in subset],
            "N_used": len(subset),
            "lambda2": metrics["lambda2"],
            "C": metrics["C"],
            "diameter": metrics["diameter"],
        })

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        payload = {
            "device": device_name,
            "N_requested": N_requested,
            "N_used": len(subsets[0]) if subsets else 0,
            "selection_seed": seed,
            "selection_params": selection_params,
            "generated_at": datetime.now().isoformat(),
            "subsets": records,
        }
        with open(save_path, "w") as f:
            json.dump(payload, f, indent=2)
    return metrics_map


def load_subsets(path: str) -> Tuple[List[Tuple[int, ...]], Dict[int, Dict], Dict]:
    """Load subsets and metrics from JSON."""
    with open(path, "r") as f:
        data = json.load(f)

    subsets: List[Tuple[int, ...]] = []
    metrics_map: Dict[int, Dict] = {}
    for entry in sorted(data.get("subsets", []), key=lambda x: int(x.get("subset_id", 0))):
        subset_id = int(entry["subset_id"])
        qubits = tuple(sorted(int(q) for q in entry["qubits"]))
        subsets.append(qubits)
        metrics_map[subset_id] = {
            "lambda2": float(entry.get("lambda2", 0)),
            "C": float(entry.get("C", 0)),
            "diameter": int(entry.get("diameter", -1)),
        }
    metadata = {
        "device": data.get("device", ""),
        "N_requested": int(data.get("N_requested", 0)),
        "N_used": int(data.get("N_used", 0)),
        "selection_seed": data.get("selection_seed"),
        "selection_params": data.get("selection_params", {}),
    }
    return subsets, metrics_map, metadata


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


def run_loschmidt_echo_interleaved(
    device,
    G: nx.Graph,
    subsets: List[Tuple[int, ...]],
    depths: List[int],
    shots: int,
    K: int,
    seed: int,
    num_blocks: int,
    block_shuffle_seed: int,
    output_csv: str,
    fit_eps: float,
    logger=None,
    run_id: str = "",
    git_commit: str = "",
    subset_metrics: Dict[int, Dict] = None,
    u_style: str = "mixed",
    N_requested: int = 0,
    resume: bool = True,
    entangler_mode: str = "cz",
):
    """Run time-interleaved Loschmidt echo experiment with block structure."""
    subset_metrics = subset_metrics or {}
    completed = set()
    existing_fieldnames: List[str] = []
    if resume and os.path.exists(output_csv):
        with open(output_csv, "r") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = reader.fieldnames or []
            for row in reader:
                completed.add(
                    (
                        int(row["block_index"]),
                        int(row["subset_id"]),
                        int(row["depth"]),
                    )
                )
        log(f"Resuming: {len(completed)} block/subset/depth points already completed", logger)

    fieldnames = [
        "experiment_type",
        "run_id",
        "backend",
        "block_index",
        "subset_id",
        "qubits",
        "N_requested",
        "N_used",
        "lambda2",
        "C",
        "u_style",
        "depth",
        "K",
        "shots",
        "mean_P_return",
        "sem_P_return",
        "timestamp",
        "git_commit",
        "entangler_mode",
        "verbatim_requested",
    ]

    write_header = (
        not os.path.exists(output_csv)
        or os.path.getsize(output_csv) == 0
        or set(existing_fieldnames) != set(fieldnames)
    )
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)

    start_time = time.time()
    num_subsets = len(subsets)
    backend = device.name if hasattr(device, "name") else str(device)

    for block in range(num_blocks):
        block_idx = block + 1
        block_rng = random.Random((block_shuffle_seed if block_shuffle_seed is not None else seed) + block)
        order = list(range(num_subsets))
        block_rng.shuffle(order)
        log(f"\nBlock {block_idx}/{num_blocks} order: {order}", logger)

        for subset_pos, subset_id in enumerate(order):
            subset = subsets[subset_id]
            subgraph = G.subgraph(subset).copy()
            metrics = subset_metrics.get(subset_id) or compute_architecture_metrics(subgraph)

            if metrics.get("diameter", 0) == -1:
                log(f"  Subset {subset_id} disconnected, skipping", logger)
                continue

            for depth_idx, depth in enumerate(depths):
                if (block_idx, subset_id, depth) in completed:
                    log(f"  Block {block_idx} subset {subset_id} depth {depth} already done, skipping", logger)
                    continue

                p_returns: List[float] = []
                log(
                    f"  block {block_idx}/{num_blocks}, subset {subset_pos + 1}/{num_subsets} (id={subset_id}), depth {depth_idx + 1}/{len(depths)}",
                    logger,
                )

                for k in range(K):
                    inst_seed = seed + block_idx * 1_000_000 + subset_id * 10_000 + depth * 100 + k
                    rng = random.Random(inst_seed)
                    try:
                        p_return, _ = run_loschmidt_echo_instance(
                            device,
                            subset,
                            subgraph,
                            depth,
                            shots,
                            rng,
                            u_style,
                            entangler_mode,
                            logger,
                        )
                        p_returns.append(p_return)
                    except Exception as e:
                        log(f"    FAILED instance {k+1}/{K}: {e}", logger)
                        p_returns.append(np.nan)

                    finite = [p for p in p_returns if not np.isnan(p)]
                    rolling_mean = float(np.mean(finite)) if finite else float("nan")
                    log(
                        f"    block {block_idx}/{num_blocks}, subset {subset_pos + 1}/{num_subsets}, depth {depth_idx + 1}/{len(depths)}, instance {k+1}/{K}, rolling mean P_return={rolling_mean:.4f}",
                        logger,
                    )

                finite = [p for p in p_returns if not np.isnan(p)]
                if not finite:
                    log(f"  FAILED: all instances failed for subset {subset_id} depth {depth}", logger)
                    continue

                mean_p = float(np.mean(finite))
                sem_p = float(np.std(finite, ddof=1) / np.sqrt(len(finite))) if len(finite) > 1 else 0.0
                row = {
                    "experiment_type": "loschmidt_echo_interleaved",
                    "run_id": run_id,
                    "backend": backend,
                    "block_index": block_idx,
                    "subset_id": subset_id,
                    "qubits": str(subset),
                    "N_requested": N_requested,
                    "N_used": len(subset),
                    "lambda2": metrics.get("lambda2", 0.0),
                    "C": metrics.get("C", 0.0),
                    "u_style": u_style,
                    "depth": depth,
                    "K": K,
                    "shots": shots,
                    "mean_P_return": mean_p,
                    "sem_P_return": sem_p,
                    "timestamp": datetime.now().isoformat(),
                    "git_commit": git_commit,
                    "entangler_mode": entangler_mode,
                    "verbatim_requested": False,
                }

                with open(output_csv, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if write_header:
                        writer.writeheader()
                        write_header = False
                    writer.writerow(row)

                log(
                    f"  Saved block {block_idx} subset {subset_id} depth {depth} (mean_P_return={mean_p:.4f})",
                    logger,
                )

    elapsed = (time.time() - start_time) / 60
    log(f"\n{'='*60}", logger)
    log(f"LOSCHMIDT ECHO INTERLEAVED COMPLETE in {elapsed:.1f} minutes", logger)
    log(f"Results saved to {output_csv}", logger)


def run_loschmidt_echo_sweep(device,
                             G: nx.Graph,
                             subsets: List[Tuple[int, ...]],
                             depths: List[int],
                             shots: int,
                             K: int,
                             seed: int,
                             output_csv: str,
                             alpha_output: str,
                             fit_eps: float,
                             logger=None,
                             resume: bool = True,
                             run_id: str = "",
                             git_commit: str = "",
                             subset_metrics: Dict[int, Dict] = None,
                             u_style: str = 'mixed',
                             entangler_mode: str = "cz"):
    """Run Loschmidt echo experiments across subsets with incremental CSV writes."""
    subset_metrics = subset_metrics or {}
    completed = set()
    existing_rows: Dict[int, List[Tuple[int, float]]] = {}
    existing_fieldnames: List[str] = []
    if resume and os.path.exists(output_csv):
        with open(output_csv, 'r') as f:
            reader = csv.DictReader(f)
            existing_fieldnames = reader.fieldnames or []
            for row in reader:
                completed.add((int(row['subset_id']), int(row['depth'])))
                existing_rows.setdefault(int(row['subset_id']), []).append(
                    (int(row['depth']), float(row['mean_P_return']))
                )
        log(f"Resuming: {len(completed)} depth points already completed", logger)

    completed_alpha = set()
    existing_alpha_fields: List[str] = []
    if resume and os.path.exists(alpha_output):
        with open(alpha_output, 'r') as f:
            reader = csv.DictReader(f)
            existing_alpha_fields = reader.fieldnames or []
            for row in reader:
                completed_alpha.add(int(row['subset_id']))

    fieldnames = [
        'experiment_type', 'device', 'run_id', 'git_commit', 'subset_id', 'qubits', 'N', 'lambda2', 'C',
        'depth', 'K', 'shots', 'mean_P_return', 'sem_P_return', 'timestamp', 'entangler_mode', 'verbatim_requested'
    ]
    alpha_fields = [
        'experiment_type', 'device', 'run_id', 'git_commit', 'subset_id', 'qubits', 'N', 'lambda2', 'C',
        'fit_eps', 'alpha', 'alpha_stderr', 'points', 'timestamp', 'entangler_mode', 'verbatim_requested'
    ]

    write_header = (not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0
                    or set(existing_fieldnames) != set(fieldnames))
    write_alpha_header = (not os.path.exists(alpha_output) or os.path.getsize(alpha_output) == 0
                          or set(existing_alpha_fields) != set(alpha_fields))

    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    os.makedirs(os.path.dirname(alpha_output) if os.path.dirname(alpha_output) else '.', exist_ok=True)

    start_time = time.time()
    num_depths = len(depths)

    for i, subset in enumerate(subsets):
        elapsed = (time.time() - start_time) / 60
        log(f"\n[{i+1}/{len(subsets)}] Subset {i} (elapsed: {elapsed:.1f} min)", logger)
        log(f"  Qubits: {subset}", logger)

        subgraph = G.subgraph(subset).copy()
        metrics = subset_metrics.get(i) or compute_architecture_metrics(subgraph)
        log(f"  lambda2={metrics['lambda2']:.4f}, C={metrics['C']:.4f}", logger)

        if metrics['diameter'] == -1:
            log("  SKIPPING: disconnected subset", logger)
            continue

        rng = random.Random(seed + i)
        depth_means: List[Tuple[int, float]] = existing_rows.get(i, []).copy()

        for depth_idx, depth in enumerate(depths):
            if (i, depth) in completed:
                log(f"  Depth {depth} already completed, skipping", logger)
                continue

            log(f"  Depth {depth} ({depth_idx+1}/{num_depths}): running {K} instances", logger)
            p_returns: List[float] = []

            for k in range(K):
                log(f"    Instance {k+1}/{K}", logger)
                try:
                    p_return, _ = run_loschmidt_echo_instance(
                        device,
                        subset,
                        subgraph,
                        depth,
                        shots,
                        rng,
                        u_style,
                        entangler_mode,
                        logger
                    )
                    p_returns.append(p_return)
                    running_mean = float(np.mean(p_returns))
                    log(f"      Running mean P_return={running_mean:.4f}", logger)
                except Exception as e:
                    log(f"      FAILED instance {k+1}: {e}", logger)
                    p_returns.append(np.nan)

            finite = [p for p in p_returns if not np.isnan(p)]
            if not finite:
                log(f"  FAILED: all instances failed at depth {depth}", logger)
                continue

            mean_p = float(np.mean(finite))
            sem_p = float(np.std(finite, ddof=1) / np.sqrt(len(finite))) if len(finite) > 1 else 0.0
            depth_means.append((depth, mean_p))

            row = {
                'experiment_type': 'loschmidt_echo',
                'device': device.name if hasattr(device, 'name') else str(device),
                'run_id': run_id,
                'git_commit': git_commit,
                'subset_id': i,
                'qubits': str(subset),
                'N': len(subset),
                'lambda2': metrics['lambda2'],
                'C': metrics['C'],
                'depth': depth,
                'K': K,
                'shots': shots,
                'mean_P_return': mean_p,
                'sem_P_return': sem_p,
                'timestamp': datetime.now().isoformat(),
                'entangler_mode': entangler_mode,
                'verbatim_requested': False
            }

            with open(output_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                    write_header = False
                writer.writerow(row)

            log(f"  Saved depth {depth} result (mean_P_return={mean_p:.4f})", logger)

        # Fit alpha for this subset
        if i in completed_alpha:
            log("  Alpha fit already exists, skipping fit", logger)
            continue

        if depth_means:
            depth_means.sort(key=lambda x: x[0])
            depths_sorted = [d for d, _ in depth_means]
            p_sorted = [p for _, p in depth_means]
            alpha, stderr, points = fit_loschmidt_alpha(depths_sorted, p_sorted, fit_eps)

            alpha_row = {
                'experiment_type': 'loschmidt_echo',
                'device': device.name if hasattr(device, 'name') else str(device),
                'run_id': run_id,
                'git_commit': git_commit,
                'subset_id': i,
                'qubits': str(subset),
                'N': len(subset),
                'lambda2': metrics['lambda2'],
                'C': metrics['C'],
                'fit_eps': fit_eps,
                'alpha': alpha,
                'alpha_stderr': stderr,
                'points': points,
                'timestamp': datetime.now().isoformat(),
                'entangler_mode': entangler_mode,
                'verbatim_requested': False
            }

            with open(alpha_output, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=alpha_fields)
                if write_alpha_header:
                    writer.writeheader()
                    write_alpha_header = False
                writer.writerow(alpha_row)

            log(f"  Alpha fit saved (alpha={alpha:.4f}, points={points})", logger)

    elapsed = (time.time() - start_time) / 60
    log(f"\n{'='*60}", logger)
    log(f"LOSCHMIDT ECHO SWEEP COMPLETE in {elapsed:.1f} minutes", logger)
    log(f"Results saved to {output_csv}", logger)


def main():
    parser = argparse.ArgumentParser(description='AWS Braket RB Sweep')
    parser.add_argument('--device', type=str, default='arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3',
                        help='Braket device ARN')
    parser.add_argument('--N', type=int, default=None, help='Subset size')
    parser.add_argument('--num-subsets', type=int, default=None, help='Number of subsets')
    parser.add_argument('--depths', type=str, default=None, help='Comma/space-separated depths')
    parser.add_argument('--shots', type=int, default=None, help='Shots per circuit')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default=None, help='Output CSV (auto-set by experiment type)')
    parser.add_argument('--resume', action='store_true', default=True, help='Resume from existing results')
    parser.add_argument('--simulator', action='store_true', help='Use local simulator instead of QPU')
    parser.add_argument('--experiment', type=str, choices=['rb', 'logical_code', 'compiler_fragility', 'loschmidt_echo', 'loschmidt_echo_interleaved'], default='rb',
                        help='Experiment type: rb (mirror RB), logical_code (repetition code), compiler_fragility (GHZ variance), loschmidt_echo (echo decay), or loschmidt_echo_interleaved (block-structured echo).')
    parser.add_argument('--logical-rounds', type=int, default=2, help='Parity-check rounds for logical experiment')
    parser.add_argument('--num-compilations', type=int, default=10,
                        help='Number of compilations/layouts for compiler fragility experiment')
    parser.add_argument('--loschmidt-K', type=int, default=None, help='Number of random unitaries per depth for echo')
    parser.add_argument('--fit-eps', type=float, default=1e-3, help='Minimum P_return to include in alpha fit')
    parser.add_argument('--alpha-output', type=str, default='results/loschmidt_echo_alpha.csv',
                        help='Output CSV for fitted alpha values (per subset)')
    parser.add_argument('--run-id', type=str, default='', help='Run identifier to tag outputs')
    parser.add_argument('--save_subsets', type=str, default=None,
                        help='Save selected subsets and metrics to JSON')
    parser.add_argument('--load_subsets', type=str, default=None,
                        help='Load subsets/metrics from JSON instead of selecting anew')
    parser.add_argument('--subset-max-jaccard', type=float, default=0.7,
                        help='Max Jaccard similarity when generating diverse subsets')
    parser.add_argument('--subset-max-attempts', type=int, default=10000,
                        help='Max attempts for subset generation')
    parser.add_argument('--num-blocks', type=int, default=None, help='Number of time blocks for interleaved echo')
    parser.add_argument('--block-shuffle-seed', type=int, default=None,
                        help='Seed to shuffle subsets within each block (defaults to --seed)')
    parser.add_argument('--candidate-pool', type=int, default=200,
                        help='Candidate pool size for max-spread-C subset selection')
    parser.add_argument('--u-style', type=str, choices=['mixed', 'local', 'global'], default='mixed',
                        help='Entangling preference: mixed (random), local (short-range), global (long-range)')
    parser.add_argument(
        '--two_qubit_entangler',
        type=str,
        choices=['auto', 'cz', 'zz', 'ms'],
        default='auto',
        help='Two-qubit entangler to use inside Loschmidt/Lorentz echo circuits.',
    )
    parser.add_argument(
        '--entangler-self-test',
        action='store_true',
        help='Run a local simulator smoke test for the chosen entangler before executing experiments.',
    )

    args = parser.parse_args()
    git_commit = get_git_commit_short()

    # Experiment-specific defaults
    if args.N is None:
        args.N = 11 if args.experiment == 'loschmidt_echo_interleaved' else 7
    if args.num_subsets is None:
        args.num_subsets = 6 if args.experiment == 'loschmidt_echo_interleaved' else 30
    if args.shots is None:
        args.shots = 2000 if args.experiment == 'loschmidt_echo_interleaved' else 1000
    if args.loschmidt_K is None:
        args.loschmidt_K = 3 if args.experiment == 'loschmidt_echo_interleaved' else 5
    if args.num_blocks is None:
        args.num_blocks = 4
    if args.block_shuffle_seed is None:
        args.block_shuffle_seed = args.seed

    if args.depths:
        tokens = args.depths.replace(',', ' ').split()
        depths = [int(d) for d in tokens]
    else:
        if args.experiment == 'loschmidt_echo':
            depths = [5, 10, 20, 30]
        elif args.experiment == 'loschmidt_echo_interleaved':
            depths = [3, 5, 7, 9]
        else:
            depths = [1, 2, 4, 8, 16, 32]
    output_path = args.output
    if output_path is None:
        if args.experiment == 'logical_code':
            output_path = 'results/logical_code_sweep.csv'
        elif args.experiment == 'compiler_fragility':
            output_path = 'results/compiler_fragility_sweep.csv'
        elif args.experiment == 'loschmidt_echo':
            if args.run_id:
                output_path = f"results/loschmidt_echo_sweep_{args.run_id}.csv"
            else:
                output_path = 'results/loschmidt_echo_sweep.csv'
        elif args.experiment == 'loschmidt_echo_interleaved':
            if args.run_id:
                output_path = f"results/loschmidt_echo_interleaved_{args.run_id}.csv"
            else:
                output_path = 'results/loschmidt_echo_interleaved.csv'
        else:
            output_path = 'results/rb_sweep_braket.csv'
    if args.experiment == 'loschmidt_echo' and args.run_id and args.alpha_output == 'results/loschmidt_echo_alpha.csv':
        args.alpha_output = f"results/loschmidt_echo_alpha_{args.run_id}.csv"

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
    if args.run_id:
        log(f"Run ID: {args.run_id}", logger)
    if args.experiment == 'rb':
        log(f"Depths: {depths}", logger)
    elif args.experiment == 'logical_code':
        log(f"Rounds: {args.logical_rounds}", logger)
    elif args.experiment == 'loschmidt_echo':
        if args.N not in (7, 9):
            log(f"WARNING: loschmidt_echo validated for N in [7, 9]; using N={args.N}", logger)
        log(f"Depths: {depths}", logger)
        log(f"K (instances per depth): {args.loschmidt_K}", logger)
        log(f"Fit eps: {args.fit_eps}", logger)
    elif args.experiment == 'loschmidt_echo_interleaved':
        log(f"Depths: {depths}", logger)
        log(f"K (instances per depth): {args.loschmidt_K}", logger)
        log(f"Blocks: {args.num_blocks}", logger)
        log(f"Block shuffle seed: {args.block_shuffle_seed}", logger)
        log(f"Fit eps: {args.fit_eps}", logger)
        log(f"U style: {args.u_style}", logger)
    else:
        log(f"Num compilations: {args.num_compilations}", logger)
    log(f"Shots: {args.shots}", logger)
    log(f"Seed: {args.seed}", logger)
    log(f"Output CSV: {output_path}", logger)
    if args.experiment == 'loschmidt_echo':
        log(f"Alpha CSV: {args.alpha_output}", logger)
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

    entangler_mode = resolve_entangler_mode(device, args.two_qubit_entangler)
    log(f"Entangler mode: {entangler_mode} (flag={args.two_qubit_entangler})", logger)

    log(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", logger)

    if args.experiment in ("loschmidt_echo", "loschmidt_echo_interleaved") and (
        args.entangler_self_test or entangler_mode in ("zz", "ms")
    ):
        log("Running entangler smoke test on local simulator...", logger)
        try:
            entangler_smoke_test(entangler_mode, args.u_style)
            log("Entangler smoke test passed.", logger)
        except Exception as exc:  # pragma: no cover - defensive guard
            log(f"Entangler smoke test failed: {exc}", logger)
            raise

    # Generate or load subsets
    subset_metrics: Dict[int, Dict] = {}
    if args.load_subsets:
        subsets, subset_metrics, subset_meta = load_subsets(args.load_subsets)
        log(f"Loaded {len(subsets)} subsets from {args.load_subsets}", logger)
        if subset_meta.get("N_used") and subset_meta["N_used"] != args.N:
            log(f"WARNING: Loaded subsets size {subset_meta['N_used']} != requested N {args.N}", logger)
    else:
        if args.experiment == 'loschmidt_echo_interleaved':
            print(f"\nSelecting {args.num_subsets} subsets (max-spread C) of size {args.N}...")
            subsets, subset_metrics = select_subsets_max_spread_C(
                G, args.N, args.num_subsets, args.seed, args.candidate_pool
            )
            log(f"Selected {len(subsets)} subsets with max-spread-C strategy", logger)
            selection_params = {
                "strategy": "max_spread_C",
                "candidate_pool": args.candidate_pool,
            }
        else:
            print(f"\nGenerating {args.num_subsets} diverse subsets of size {args.N}...")
            subsets = generate_diverse_subsets(
                G, args.N, args.num_subsets, args.seed, args.subset_max_jaccard, args.subset_max_attempts
            )
            log(f"Generated {len(subsets)} subsets", logger)
            selection_params = {
                "max_jaccard": args.subset_max_jaccard,
                "max_attempts": args.subset_max_attempts,
            }
        subset_metrics = summarize_subsets(
            subsets,
            G,
            device.name if hasattr(device, 'name') else str(device),
            args.N,
            args.seed,
            selection_params,
            save_path=args.save_subsets,
        )
        if args.save_subsets:
            log(f"Saved subset list to {args.save_subsets}", logger)

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
    elif args.experiment == 'loschmidt_echo':
        log("STARTING LOSCHMIDT ECHO SWEEP", logger)
        log("=" * 60, logger)
        run_loschmidt_echo_sweep(
            device,
            G,
            subsets,
            depths,
            args.shots,
            args.loschmidt_K,
            args.seed,
            output_path,
            args.alpha_output,
            args.fit_eps,
            logger,
            args.resume,
            run_id=args.run_id,
            git_commit=git_commit,
            subset_metrics=subset_metrics if not args.load_subsets else subset_metrics,
            u_style=args.u_style,
            entangler_mode=entangler_mode,
        )
    elif args.experiment == 'loschmidt_echo_interleaved':
        if not args.run_id:
            log("ERROR: --run-id is required for loschmidt_echo_interleaved", logger)
            sys.exit(1)
        log("STARTING LOSCHMIDT ECHO INTERLEAVED", logger)
        log("=" * 60, logger)
        run_loschmidt_echo_interleaved(
            device,
            G,
            subsets,
            depths,
            args.shots,
            args.loschmidt_K,
            args.seed,
            args.num_blocks,
            args.block_shuffle_seed,
            output_path,
            args.fit_eps,
            logger,
            run_id=args.run_id,
            git_commit=git_commit,
            subset_metrics=subset_metrics if not args.load_subsets else subset_metrics,
            u_style=args.u_style,
            N_requested=args.N,
            resume=args.resume,
            entangler_mode=entangler_mode,
        )
    else:
        log("STARTING COMPILER FRAGILITY SWEEP", logger)
        log("=" * 60, logger)
        run_compiler_fragility_sweep(device, G, subsets, args.shots,
                                     args.num_compilations, args.seed, output_path,
                                     logger, args.resume)

    logger.close()


if __name__ == '__main__':
    main()
