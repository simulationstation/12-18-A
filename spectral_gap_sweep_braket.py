#!/usr/bin/env python3
"""
Spectral-gap sweep Loschmidt echo (AWS Braket)
==============================================

This experiment probes a mixing-time crossover by sweeping Loschmidt echo
return probabilities while holding the two-qubit budget per layer fixed
across three matching-based architectures:

    - ring_matching: alternating even/odd nearest-neighbor pairs on a ring
    - grid_matching: alternating horizontal/vertical matchings on an
      approximately square grid layout
    - expander_matching: fresh random perfect matchings each layer

For each depth `d`, we build a brickwork unitary U(d) consisting of random
single-qubit scrambles (RX(pi/2), RY(pi/2), or RZ(phi) with phi ~ U[0, 2pi))
followed by a 2Q entangling layer drawn from the chosen matching schedule.
We append the exact inverse U(d)† by reversing and inverting all gates, so a
noiseless device would return to |0...0>. The return probability p_return
captures dynamical irreversibility as depth increases; comparing decay slopes
across matching families links to the spectral gap (mixing time) of each
architecture.

CLI modes:
    - sweep: run the experiment on a Braket device (ARN or local simulator),
      logging one JSON record per (family, depth, seed).
    - analyze: load JSONL results, compute mean/stderr p_return vs depth, fit
      a simple two-regime log-decay crossover, and export CSV summaries.
    - plateau_diagnostic / plateau_analyze: baseline controls at depths 0/1/2
      (optional 4) to isolate SPAM vs 1Q vs first entangling-layer collapse
      and to report compiled distinctness across families.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from braket.aws import AwsDevice
from braket.circuits import Circuit, gates
from braket.devices import LocalSimulator


Gate = Tuple[str, Tuple[int, ...], Tuple[float, ...]]
Matching = List[Tuple[int, int]]


# ---------------------------------------------------------------------------
# Matching schedule generators
# ---------------------------------------------------------------------------

def require_even_qubits(n_qubits: int) -> None:
    """Enforce an even qubit count to keep per-layer matchings balanced."""
    if n_qubits % 2 != 0:
        raise ValueError(f"n_qubits must be even to form perfect matchings; got {n_qubits}")


def ring_matchings(n_qubits: int) -> List[Matching]:
    """Return two alternating perfect matchings on a ring."""
    require_even_qubits(n_qubits)
    even_edges = [((2 * k) % n_qubits, (2 * k + 1) % n_qubits) for k in range(n_qubits // 2)]
    odd_edges = [((2 * k + 1) % n_qubits, (2 * k + 2) % n_qubits) for k in range(n_qubits // 2)]
    return [even_edges, odd_edges]


def _grid_dims(n_qubits: int) -> Tuple[int, int]:
    """
    Choose a near-square rectangle with even cardinality <= n_qubits.
    Preference order: maximize used qubits, minimize aspect ratio skew.
    """
    best_rows, best_cols = 2, max(2, n_qubits // 2)
    best_prod = best_rows * best_cols
    best_diff = abs(best_rows - best_cols)
    max_rows = int(math.sqrt(n_qubits)) + 2
    for rows in range(2, max_rows + 1):
        cols = n_qubits // rows
        if cols < 2:
            continue
        prod = rows * cols
        if prod % 2 == 1:
            if cols > 2:
                cols -= 1
                prod = rows * cols
            else:
                continue
        if prod == 0 or prod > n_qubits:
            continue
        diff = abs(rows - cols)
        if prod > best_prod or (prod == best_prod and diff < best_diff):
            best_rows, best_cols = rows, cols
            best_prod = prod
            best_diff = diff
    return best_rows, best_cols


def _grid_index_map(n_qubits: int) -> Dict[Tuple[int, int], int]:
    rows, cols = _grid_dims(n_qubits)
    mapping: Dict[Tuple[int, int], int] = {}
    q = 0
    for r in range(rows):
        for c in range(cols):
            if q >= n_qubits:
                return mapping
            mapping[(r, c)] = q
            q += 1
    return mapping


def grid_matchings(n_qubits: int) -> List[Matching]:
    """
    Approximate square-grid matchings using alternating horizontal/vertical layers.
    Layout uses a near-square rows x cols rectangle (even size, <= N); any leftover
    qubits beyond the rectangle are dropped consistently to keep gate counts uniform.
    """
    require_even_qubits(n_qubits)
    mapping = _grid_index_map(n_qubits)
    rows, cols = _grid_dims(n_qubits)

    def horizontal(offset: int) -> Matching:
        edges: Matching = []
        for r in range(rows):
            start = offset % 2
            for c in range(start, cols, 2):
                left = (r, c)
                right = (r, c + 1)
                if left in mapping and right in mapping:
                    edges.append((mapping[left], mapping[right]))
        return edges

    def vertical(offset: int) -> Matching:
        edges: Matching = []
        for c in range(cols):
            start = offset % 2
            for r in range(start, rows, 2):
                top = (r, c)
                bottom = (r + 1, c)
                if top in mapping and bottom in mapping:
                    edges.append((mapping[top], mapping[bottom]))
        return edges

    return [
        horizontal(0),
        horizontal(1),
        vertical(0),
        vertical(1),
    ]


def expander_matching(n_qubits: int, rng: np.random.Generator) -> Matching:
    """Draw a random perfect matching by shuffling qubits and pairing consecutively."""
    require_even_qubits(n_qubits)
    qubits = np.arange(n_qubits)
    rng.shuffle(qubits)
    return [(int(qubits[i]), int(qubits[i + 1])) for i in range(0, n_qubits, 2)]


def matching_schedule(
    family: str,
    n_qubits: int,
    depth: int,
    rng: np.random.Generator,
) -> List[Matching]:
    """Construct a length-`depth` schedule of disjoint pairings."""
    if family == "ring":
        base = ring_matchings(n_qubits)
        return [base[i % len(base)] for i in range(depth)]
    if family == "grid":
        base = grid_matchings(n_qubits)
        return [base[i % len(base)] for i in range(depth)]
    if family == "expander":
        return [expander_matching(n_qubits, rng) for _ in range(depth)]
    raise ValueError(f"Unknown family: {family}")


# ---------------------------------------------------------------------------
# Circuit helpers
# ---------------------------------------------------------------------------

SCRAMBLE_GATES = ("rx", "ry", "rz")


def _device_name(device) -> str:
    """Best-effort device name extraction."""
    return getattr(
        device,
        "name",
        getattr(getattr(device, "properties", None), "deviceParameters", {}).get("name", "unknown"),
    )


def resolve_entangler_mode(device, requested: str) -> str:
    """IonQ-aware entangler resolution with safe defaults for non-IonQ hardware."""
    if requested != "auto":
        return requested

    name = _device_name(device).lower()
    if "forte" in name:
        return "zz"
    if "aria" in name:
        return "ms"
    native = getattr(getattr(device, "properties", None), "paradigm", None)
    gateset = {g.lower() for g in getattr(native, "nativeGateSet", [])} if native else set()
    if "cz" in gateset:
        return "cz"
    return "cz"


def _default_entangler_params(mode: str) -> Dict[str, float]:
    if mode in ("zz", "ms"):
        return {"theta": math.pi / 2, "phi": 0.0}
    return {}


def apply_entangler(circ: Circuit, q1: int, q2: int, params: Dict[str, float], mode: str) -> Gate:
    """Apply the selected entangler and return a log tuple."""
    if mode in ("cz", "cnot", "cx"):
        if hasattr(circ, "cz"):
            circ.cz(q1, q2)
        else:
            circ.cnot(q1, q2)
        return ("cz", (q1, q2), ())

    if mode == "zz":
        theta = params.get("theta", math.pi / 2)
        if hasattr(circ, "zz"):
            circ.zz(q1, q2, theta)
        else:
            try:
                circ.add(gates.ZZ(theta), [q1, q2])
            except Exception as exc:  # pragma: no cover - SDK dependent
                raise RuntimeError("ZZ gate not supported by installed Braket SDK") from exc
        return ("zz", (q1, q2), (theta,))

    if mode == "ms":
        theta = params.get("theta", math.pi / 2)
        phi = params.get("phi", 0.0)
        ms_method = getattr(circ, "ms", None)
        if callable(ms_method):
            try:
                ms_method(q1, q2, theta, phi)
            except TypeError:
                ms_method(q1, q2, theta)
        else:
            try:
                circ.add(gates.MS(theta, phi), [q1, q2])
            except TypeError:
                circ.add(gates.MS(theta), [q1, q2])
            except Exception as exc:  # pragma: no cover - SDK dependent
                raise RuntimeError("MS gate not supported by installed Braket SDK") from exc
        return ("ms", (q1, q2), (theta, phi))

    raise ValueError(f"Unsupported entangler mode: {mode}")


def _apply_two_qubit(circ: Circuit, gate_label: str, control: int, target: int, entangler_params: Dict[str, float]) -> Gate:
    return apply_entangler(circ, control, target, entangler_params, gate_label)


def build_scramble_layer(circ: Circuit, n_qubits: int, rng: np.random.Generator) -> List[Gate]:
    """Apply random single-qubit scrambles and return a log of applied gates."""
    applied: List[Gate] = []
    for q in range(n_qubits):
        choice = rng.choice(SCRAMBLE_GATES)
        if choice == "rx":
            angle = math.pi / 2
            circ.rx(q, angle)
            applied.append(("rx", (q,), (angle,)))
        elif choice == "ry":
            angle = math.pi / 2
            circ.ry(q, angle)
            applied.append(("ry", (q,), (angle,)))
        else:
            angle = rng.uniform(0.0, 2.0 * math.pi)
            circ.rz(q, angle)
            applied.append(("rz", (q,), (angle,)))
    return applied


def apply_matching_layer(
    circ: Circuit,
    matches: Matching,
    gate_label: str,
    entangler_params: Optional[Dict[str, float]] = None,
) -> List[Gate]:
    """Apply a layer of two-qubit gates and return a log of applied gates."""
    applied: List[Gate] = []
    entangler_params = entangler_params or _default_entangler_params(gate_label)
    for control, target in matches:
        applied.append(_apply_two_qubit(circ, gate_label, control, target, entangler_params))
    return applied


def invert_gate(circ: Circuit, gate: Gate) -> None:
    """Append the inverse of a logged gate."""
    name, qubits, params = gate
    if name == "rx":
        circ.rx(qubits[0], -params[0])
    elif name == "ry":
        circ.ry(qubits[0], -params[0])
    elif name == "rz":
        circ.rz(qubits[0], -params[0])
    elif name in ("cz", "cnot", "zz", "ms"):
        inv_params: Dict[str, float] = {"theta": -params[0]} if params else {}
        if len(params) > 1:
            inv_params["phi"] = params[1]
        _apply_two_qubit(circ, "cz" if name == "cnot" else name, qubits[0], qubits[1], inv_params)
    else:
        raise ValueError(f"Unsupported gate for inversion: {name}")


def build_loschmidt_circuit(
    n_qubits: int,
    matches: List[Matching],
    rng: np.random.Generator,
    gate_label: str,
    entangler_params: Optional[Dict[str, float]] = None,
) -> Circuit:
    """Construct U(d) followed by its exact inverse."""
    circ = Circuit()
    log: List[Gate] = []
    entangler_params = entangler_params or _default_entangler_params(gate_label)
    for layer in matches:
        log.extend(build_scramble_layer(circ, n_qubits, rng))
        log.extend(apply_matching_layer(circ, layer, gate_label, entangler_params))

    # Append inverse
    for gate in reversed(log):
        invert_gate(circ, gate)

    for q in range(n_qubits):
        circ.measure(q)
    return circ


def entangler_smoke_test(entangler_mode: str) -> None:
    """Small local simulation to ensure entangler wiring works and avoids CZ when requested."""
    rng = np.random.default_rng(123)
    schedule = matching_schedule("ring", 4, 2, rng)
    circ = build_loschmidt_circuit(
        n_qubits=4,
        matches=schedule,
        rng=rng,
        gate_label=entangler_mode,
        entangler_params=_default_entangler_params(entangler_mode),
    )
    circ_text = str(circ).lower()
    if entangler_mode in ("zz", "ms") and "cz" in circ_text:
        raise RuntimeError("Entangler smoke test detected CZ in IR while a native IonQ entangler was requested.")
    LocalSimulator().run(circ, shots=4).result()


# ---------------------------------------------------------------------------
# Data + run helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SweepJob:
    family: str
    depth: int
    seed: int


def circuit_hash(circ: Circuit) -> str:
    try:
        ir_json = circ.to_ir().json()
        return hashlib.sha256(ir_json.encode("utf-8")).hexdigest()
    except NotImplementedError:
        return hashlib.sha256(str(circ).encode("utf-8")).hexdigest()


def schedule_hash(schedule: List[Matching]) -> str:
    payload = json.dumps(schedule, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def p_return_from_counts(counts: Dict[str, int], n_qubits: int, shots: int) -> float:
    zero = "0" * n_qubits
    return counts.get(zero, 0) / float(shots)


def hamming_weight_hist(counts: Dict[str, int]) -> Dict[int, int]:
    hist: Dict[int, int] = {}
    for bitstring, ct in counts.items():
        weight = bitstring.count("1")
        hist[weight] = hist.get(weight, 0) + ct
    return hist


# ---------------------------------------------------------------------------
# Plateau diagnostic helpers
# ---------------------------------------------------------------------------

PLATEAU_DEPTH_MODE = "plateau_depth: 0=no gates, 1=1Q scramble+inverse, depth>=2 uses depth//2 matching layers"


def _is_result_type_instruction(instr) -> bool:
    op = getattr(instr, "operator", None)
    if op is None:
        return True
    module = op.__class__.__module__.lower()
    if "result_types" in module:
        return True
    name = getattr(op, "name", "").lower()
    return name.startswith("resulttype")


def gate_count_summary(circ: Circuit) -> Tuple[Dict[str, Any], int]:
    """
    Count gates in a circuit, returning (summary_dict, depth).
    Measurements/result types are excluded from the counts.
    """
    counts_by_type: Dict[str, int] = {}
    total_1q = 0
    total_2q = 0
    gate_instructions = 0
    for instr in getattr(circ, "instructions", []):
        if _is_result_type_instruction(instr):
            continue
        op = getattr(instr, "operator", None)
        if op is None:
            continue
        name = getattr(op, "name", op.__class__.__name__).lower()
        counts_by_type[name] = counts_by_type.get(name, 0) + 1
        qcount = len(getattr(instr, "target", ()))
        if qcount == 1:
            total_1q += 1
        elif qcount == 2:
            total_2q += 1
        gate_instructions += 1
    try:
        depth_val = int(getattr(circ, "depth"))
    except Exception:
        depth_val = gate_instructions
    summary = {
        "total_1q": total_1q,
        "total_2q": total_2q,
        "by_type": counts_by_type,
    }
    return summary, depth_val


def _action_from_result(result):
    """Best-effort extraction of the compiled/action IR payload from a Braket result."""
    add_meta = getattr(result, "additional_metadata", None) or getattr(result, "additionalMetadata", None)
    action = None
    if add_meta is not None:
        if isinstance(add_meta, dict):
            action = add_meta.get("compiledProgram") or add_meta.get("compiled_program") or add_meta.get("action")
        else:
            action = (
                getattr(add_meta, "compiledProgram", None)
                or getattr(add_meta, "compiled_program", None)
                or getattr(add_meta, "action", None)
            )
    return action


def _circuit_from_action(action) -> Optional[Circuit]:
    """Attempt to rebuild a Circuit from an action/IR payload."""
    if action is None:
        return None
    try:
        return Circuit.from_ir(action)
    except Exception:
        pass
    try:
        if hasattr(action, "json"):
            return Circuit.from_ir(action.json())
        if hasattr(action, "dict"):
            return Circuit.from_ir(action.dict())
    except Exception:
        pass
    try:
        if isinstance(action, str):
            return Circuit.from_ir(action)
        return Circuit.from_ir(json.dumps(action))
    except Exception:
        return None


def _hash_payload(payload) -> Optional[str]:
    try:
        if isinstance(payload, str):
            raw = payload
        else:
            raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    except Exception:
        return None


def compiled_metadata_from_result(
    result, logical_gate_counts: Dict[str, Dict[str, int]], logical_circuit_depth: int, circuit_ir_hash: str
) -> Tuple[Dict[str, Dict[str, int]], int, str, str]:
    """
    Extract compiled counts/depth/hash when available, otherwise fall back to logical values.

    Returns (compiled_gate_counts, compiled_depth, compiled_ir_hash, compiled_info_source).
    """
    compiled_counts = logical_gate_counts
    compiled_depth = logical_circuit_depth
    compiled_ir_hash = circuit_ir_hash
    info_source = "logical_fallback"

    action = _action_from_result(result)
    if action is not None:
        compiled_circ = _circuit_from_action(action)
        if compiled_circ is not None:
            compiled_counts, compiled_depth_val = gate_count_summary(compiled_circ)
            compiled_ir_hash = circuit_hash(compiled_circ)
            compiled_depth = compiled_depth_val
            info_source = "compiled_circuit"
        else:
            payload_hash = _hash_payload(action)
            if payload_hash is not None:
                compiled_ir_hash = payload_hash
                info_source = "compiled_payload_hash"

    return compiled_counts, compiled_depth, compiled_ir_hash, info_source


def build_plateau_circuit(
    family: str,
    depth: int,
    n_qubits: int,
    rng: np.random.Generator,
    gate_label: str,
    entangler_params: Optional[Dict[str, float]] = None,
) -> Tuple[Circuit, List[Matching], int]:
    """
    Build plateau-style circuits:
        depth=0: prepare |0...0>, measure.
        depth=1: scramble-only layer and exact inverse (no 2Q gates).
        depth>=2: include depth//2 matching layers, each with 1Q scramble + 2Q entanglers and exact inverse.
    Returns (circuit, schedule, logical_depth).
    """
    require_even_qubits(n_qubits)
    if depth < 0:
        raise ValueError(f"Plateau depth must be non-negative; got {depth}")
    circ = Circuit()
    schedule: List[Matching] = []
    log: List[Gate] = []
    entangler_params = entangler_params or _default_entangler_params(gate_label)

    if depth == 0:
        matching_layers = 0
    elif depth == 1:
        log.extend(build_scramble_layer(circ, n_qubits, rng))
        matching_layers = 0
    else:
        matching_layers = max(1, depth // 2)
        schedule = matching_schedule(family, n_qubits, matching_layers, rng)
        for layer in schedule:
            log.extend(build_scramble_layer(circ, n_qubits, rng))
            log.extend(apply_matching_layer(circ, layer, gate_label, entangler_params))

    for gate in reversed(log):
        invert_gate(circ, gate)
    for q in range(n_qubits):
        circ.measure(q)
    return circ, schedule, matching_layers


def run_job(
    device,
    job: SweepJob,
    n_qubits: int,
    base_seed: int,
    shots: int,
    gate_label: str,
    embed_strategy: str,
    entangler_params: Optional[Dict[str, float]] = None,
) -> Dict:
    seed_seq = np.random.SeedSequence([base_seed, job.seed, job.depth, hash(job.family) & 0xFFFFFFFF])
    rng = np.random.default_rng(seed_seq)
    schedule = matching_schedule(job.family, n_qubits, job.depth, rng)
    circ = build_loschmidt_circuit(n_qubits, schedule, rng, gate_label, entangler_params)

    started = time.time()
    result = device.run(circ, shots=shots).result()
    counts = dict(result.measurement_counts)
    runtime_s = time.time() - started

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "device": getattr(device, "arn", "local"),
        "n_qubits": n_qubits,
        "family": job.family,
        "depth": job.depth,
        "seed": job.seed,
        "circuit_hash": circuit_hash(circ),
        "schedule": schedule,
        "schedule_hash": schedule_hash(schedule),
        "shots": shots,
        "counts": counts,
        "p_return": p_return_from_counts(counts, n_qubits, shots),
        "hamming_weight_histogram": hamming_weight_hist(counts),
        "runtime_seconds": runtime_s,
        "embed_strategy": embed_strategy,
        "entangler_mode": gate_label,
        "verbatim_requested": False,
    }
    return record


def run_plateau_job(
    device,
    job: SweepJob,
    n_qubits: int,
    base_seed: int,
    shots: int,
    gate_label: str,
    embed_strategy: str,
    entangler_params: Optional[Dict[str, float]] = None,
) -> Dict:
    seed_seq = np.random.SeedSequence([base_seed, job.seed, job.depth, hash(job.family) & 0xFFFFFFFF])
    rng = np.random.default_rng(seed_seq)
    circ, schedule, matching_layers = build_plateau_circuit(
        family=job.family,
        depth=job.depth,
        n_qubits=n_qubits,
        rng=rng,
        gate_label=gate_label,
        entangler_params=entangler_params,
    )

    started = time.time()
    result = device.run(circ, shots=shots).result()
    counts = dict(result.measurement_counts)
    runtime_s = time.time() - started

    circuit_ir_hash = circuit_hash(circ)
    logical_gate_counts, logical_circuit_depth = gate_count_summary(circ)
    compiled_gate_counts, compiled_depth, compiled_ir_hash, compiled_info_source = compiled_metadata_from_result(
        result, logical_gate_counts, logical_circuit_depth, circuit_ir_hash
    )

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "device": getattr(device, "arn", "local"),
        "n_qubits": n_qubits,
        "family": job.family,
        "depth": job.depth,
        "logical_depth": job.depth,
        "matching_layers": matching_layers,
        "logical_circuit_depth": logical_circuit_depth,
        "seed": job.seed,
        "control_mode": "plateau_diagnostic",
        "depth_mode": PLATEAU_DEPTH_MODE,
        "circuit_hash": circuit_ir_hash,
        "circuit_ir_hash": circuit_ir_hash,
        "compiled_ir_hash": compiled_ir_hash,
        "compiled_info_source": compiled_info_source,
        "schedule": schedule,
        "schedule_hash": schedule_hash(schedule),
        "shots": shots,
        "counts": counts,
        "p_return": p_return_from_counts(counts, n_qubits, shots),
        "hamming_weight_histogram": hamming_weight_hist(counts),
        "runtime_seconds": runtime_s,
        "embed_strategy": embed_strategy,
        "logical_gate_counts": logical_gate_counts,
        "compiled_gate_counts": compiled_gate_counts,
        "compiled_depth": compiled_depth,
        "entangler_mode": gate_label,
        "verbatim_requested": False,
    }
    return record


def jobs_in_order(
    families: Sequence[str],
    depths: Sequence[int],
    n_seeds: int,
    interleave: bool,
) -> List[SweepJob]:
    jobs: List[SweepJob] = []
    if interleave:
        for seed in range(n_seeds):
            for depth in depths:
                for fam in families:
                    jobs.append(SweepJob(fam, depth, seed))
    else:
        for fam in families:
            for depth in depths:
                for seed in range(n_seeds):
                    jobs.append(SweepJob(fam, depth, seed))
    return jobs


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_depths_arg(depths: str) -> List[int]:
    return [int(d) for d in depths.split(",") if str(d).strip()]


def sweep(
    device,
    families: Sequence[str],
    depths: Sequence[int],
    n_seeds: int,
    n_qubits: int,
    shots: int,
    output_jsonl: str,
    interleave: bool,
    base_seed: int,
    embed_strategy: str,
    entangler_mode: str,
) -> None:
    gate_label = entangler_mode
    entangler_params = _default_entangler_params(gate_label)
    ensure_output_dir(os.path.dirname(output_jsonl) or ".")
    jobs = jobs_in_order(families, depths, n_seeds, interleave)
    with open(output_jsonl, "a", encoding="utf-8") as f:
        for job in jobs:
            print(f"[{datetime.utcnow().isoformat()}] Running {job.family} depth {job.depth} seed {job.seed}")
            rec = run_job(
                device=device,
                job=job,
                n_qubits=n_qubits,
                base_seed=base_seed,
                shots=shots,
                gate_label=gate_label,
                embed_strategy=embed_strategy,
                entangler_params=entangler_params,
            )
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(
                f"  p_return={rec['p_return']:.4f} schedule_hash={rec['schedule_hash']} runtime={rec['runtime_seconds']:.2f}s"
            )


def plateau_sweep(
    device,
    families: Sequence[str],
    depths: Sequence[int],
    n_seeds: int,
    n_qubits: int,
    shots: int,
    output_jsonl: str,
    interleave: bool,
    base_seed: int,
    embed_strategy: str,
    entangler_mode: str,
) -> None:
    gate_label = entangler_mode
    entangler_params = _default_entangler_params(gate_label)
    ensure_output_dir(os.path.dirname(output_jsonl) or ".")
    jobs = jobs_in_order(families, depths, n_seeds, interleave)
    with open(output_jsonl, "a", encoding="utf-8") as f:
        for job in jobs:
            print(
                f"[{datetime.utcnow().isoformat()}] Plateau control {job.family} depth {job.depth} seed {job.seed}"
            )
            rec = run_plateau_job(
                device=device,
                job=job,
                n_qubits=n_qubits,
                base_seed=base_seed,
                shots=shots,
                gate_label=gate_label,
                embed_strategy=embed_strategy,
                entangler_params=entangler_params,
            )
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(
                f"  p_return={rec['p_return']:.4f} schedule_hash={rec['schedule_hash']} runtime={rec['runtime_seconds']:.2f}s"
            )


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def mean_stderr(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    stderr = float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, stderr


def fit_piecewise_log_decay(depths: List[int], probs: List[float]) -> Tuple[int, float, float]:
    """
    Fit log(p_return) ~ a*d + b with a single split depth minimizing SSE.
    Returns (d_star, slope_lo, slope_hi).
    """
    d_arr = np.asarray(depths, dtype=float)
    y = np.log(np.clip(probs, 1e-12, 1.0))
    best_sse = math.inf
    best_split = depths[0]
    best_slopes = (0.0, 0.0)
    unique_depths = sorted(set(depths))
    if len(unique_depths) < 3:
        return best_split, best_slopes[0], best_slopes[1]

    def fit_segment(d_seg, y_seg):
        A = np.vstack([d_seg, np.ones_like(d_seg)]).T
        slope, _ = np.linalg.lstsq(A, y_seg, rcond=None)[0]
        residuals = y_seg - A @ np.array([slope, 0.0])
        return slope, float(np.sum(residuals ** 2))

    for split in unique_depths[1:-1]:
        mask_lo = d_arr <= split
        mask_hi = d_arr > split
        slope_lo, sse_lo = fit_segment(d_arr[mask_lo], y[mask_lo])
        slope_hi, sse_hi = fit_segment(d_arr[mask_hi], y[mask_hi])
        total_sse = sse_lo + sse_hi
        if total_sse < best_sse:
            best_sse = total_sse
            best_split = split
            best_slopes = (slope_lo, slope_hi)

    return int(best_split), float(best_slopes[0]), float(best_slopes[1])


def summarize(results: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Compute mean/stderr per family-depth and crossover fits per family."""
    by_key: Dict[Tuple[str, int], List[float]] = {}
    for row in results:
        key = (row["family"], int(row["depth"]))
        by_key.setdefault(key, []).append(float(row["p_return"]))

    summary_rows: List[Dict] = []
    for (family, depth), vals in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        mean, err = mean_stderr(vals)
        summary_rows.append(
            {
                "family": family,
                "depth": depth,
                "mean_p_return": mean,
                "stderr_p_return": err,
                "n": len(vals),
            }
        )

    crossover_rows: List[Dict] = []
    by_family: Dict[str, List[Tuple[int, float]]] = {}
    for row in results:
        by_family.setdefault(row["family"], []).append((int(row["depth"]), float(row["p_return"])))

    for family, items in by_family.items():
        items.sort(key=lambda t: t[0])
        depths = [t[0] for t in items]
        probs = [t[1] for t in items]
        d_star, slope_lo, slope_hi = fit_piecewise_log_decay(depths, probs)
        crossover_rows.append(
            {
                "family": family,
                "d_star": d_star,
                "slope_lo": slope_lo,
                "slope_hi": slope_hi,
            }
        )
    return summary_rows, crossover_rows


def write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    ensure_output_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plateau_summaries(
    results: List[Dict],
    depths_of_interest: Sequence[int],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Compute mean/stderr by (family, depth), deltas across {0,1,2}, and compiled distinctness summaries.
    """
    filtered = [
        row
        for row in results
        if row.get("control_mode") == "plateau_diagnostic" or row.get("depth_mode") == PLATEAU_DEPTH_MODE
    ]
    if not filtered:
        filtered = results

    by_key: Dict[Tuple[str, int], List[float]] = {}
    depth_set = {int(d) for d in depths_of_interest}
    for row in filtered:
        if depth_set and int(row["depth"]) not in depth_set:
            continue
        key = (row["family"], int(row["depth"]))
        by_key.setdefault(key, []).append(float(row["p_return"]))

    summary_rows: List[Dict] = []
    for (family, depth), vals in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        mean, err = mean_stderr(vals)
        summary_rows.append(
            {
                "family": family,
                "depth": depth,
                "mean_p_return": mean,
                "stderr_p_return": err,
                "n": len(vals),
            }
        )

    by_family_depth_mean: Dict[str, Dict[int, float]] = {}
    for row in summary_rows:
        by_family_depth_mean.setdefault(row["family"], {})[int(row["depth"])] = float(row["mean_p_return"])

    delta_rows: List[Dict] = []
    for family, depth_means in sorted(by_family_depth_mean.items()):
        delta01 = None
        delta12 = None
        if 0 in depth_means and 1 in depth_means:
            delta01 = depth_means[0] - depth_means[1]
        if 1 in depth_means and 2 in depth_means:
            delta12 = depth_means[1] - depth_means[2]
        delta_rows.append(
            {
                "family": family,
                "delta01": delta01,
                "delta12": delta12,
            }
        )

    distinct_rows: List[Dict] = []
    for (family, depth), vals in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        compiled_depths: List[float] = []
        compiled_two_q: List[float] = []
        compiled_hashes: List[str] = []
        for row in filtered:
            if row["family"] != family or int(row["depth"]) != depth:
                continue
            if row.get("compiled_depth") is not None:
                compiled_depths.append(float(row["compiled_depth"]))
            gate_counts = row.get("compiled_gate_counts") or {}
            if gate_counts.get("total_2q") is not None:
                compiled_two_q.append(float(gate_counts["total_2q"]))
            compiled_hashes.append(row.get("compiled_ir_hash") or row.get("circuit_ir_hash") or row["circuit_hash"])

        distinct_rows.append(
            {
                "family": family,
                "depth": depth,
                "median_compiled_depth": float(np.median(compiled_depths)) if compiled_depths else None,
                "median_compiled_2q": float(np.median(compiled_two_q)) if compiled_two_q else None,
                "unique_compiled_ir_hash": len(set(compiled_hashes)),
                "n": len(compiled_hashes),
            }
        )

    return summary_rows, delta_rows, distinct_rows


def print_plateau_summary(
    summary_rows: List[Dict],
    delta_rows: List[Dict],
    distinct_rows: List[Dict],
) -> None:
    print("\nPlateau diagnostic p_return summary (mean ± stderr):")
    for row in summary_rows:
        mean = row["mean_p_return"]
        err = row["stderr_p_return"]
        print(
            f"  {row['family']:8s} depth={row['depth']:2d} -> {mean:.4f} ± {err:.4f} (n={row['n']})"
        )

    print("\nPlateau deltas (delta01=depth0-depth1, delta12=depth1-depth2):")
    for row in delta_rows:
        d01 = "NA" if row["delta01"] is None else f"{row['delta01']:.4f}"
        d12 = "NA" if row["delta12"] is None else f"{row['delta12']:.4f}"
        print(f"  {row['family']:8s} delta01={d01} delta12={d12}")

    print("\nCompiled distinctness check (median compiled depth/2Q count, unique compiled_ir_hash):")
    for row in distinct_rows:
        d_med = "NA" if row["median_compiled_depth"] is None else f"{row['median_compiled_depth']:.2f}"
        twoq = "NA" if row["median_compiled_2q"] is None else f"{row['median_compiled_2q']:.1f}"
        print(
            f"  {row['family']:8s} depth={row['depth']:2d} med_depth={d_med} med_2Q={twoq} "
            f"unique_hashes={row['unique_compiled_ir_hash']} (n={row['n']})"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spectral-gap sweep Loschmidt echo on AWS Braket (matching-based architectures)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sweep_p = subparsers.add_parser("sweep", help="Run the Loschmidt sweep and write JSONL records.")
    sweep_p.add_argument("--device", required=True, help='Braket device ARN or "local"')
    sweep_p.add_argument("--n_qubits", type=int, required=True, help="Number of qubits (must be even).")
    sweep_p.add_argument("--depths", type=str, required=True, help="Comma-separated depths, e.g., 2,4,8,16")
    sweep_p.add_argument(
        "--families",
        type=str,
        default="ring,grid,expander",
        help="Comma-separated matching families (ring,grid,expander).",
    )
    sweep_p.add_argument("--n_seeds", type=int, default=3, help="Seeds per (family, depth).")
    sweep_p.add_argument("--shots", type=int, default=1000, help="Shots per circuit.")
    sweep_p.add_argument("--output_dir", type=str, default="results", help="Directory for JSONL output.")
    sweep_p.add_argument(
        "--output_jsonl",
        type=str,
        default=None,
        help="Optional explicit JSONL path (defaults to output_dir/spectral_gap_loschmidt.jsonl).",
    )
    sweep_p.add_argument("--interleave", action="store_true", help="Interleave submissions across families.")
    sweep_p.add_argument("--base_seed", type=int, default=1234, help="Base seed controlling randomness.")
    sweep_p.add_argument(
        "--embed_strategy",
        type=str,
        default="random",
        help="Embedding strategy tag (metadata only; use per-device tooling as needed).",
    )
    sweep_p.add_argument(
        "--self_test",
        action="store_true",
        help="Run a quick local self-test (overrides most flags: n_qubits=8, depths=2,4, shots=200).",
    )
    sweep_p.add_argument(
        "--two_qubit_entangler",
        type=str,
        choices=["auto", "cz", "zz", "ms"],
        default="auto",
        help="Two-qubit entangler to use inside matching layers (auto selects ZZ on Forte, MS on Aria, CZ otherwise).",
    )

    analyze_p = subparsers.add_parser("analyze", help="Analyze JSONL output and export CSV summaries.")
    analyze_p.add_argument("--input", required=True, help="Path to JSONL produced by sweep.")
    analyze_p.add_argument(
        "--csv_out",
        default="results/spectral_gap_loschmidt_summary.csv",
        help="Output CSV for mean/stderr vs depth.",
    )
    analyze_p.add_argument(
        "--crossover_csv",
        default="results/spectral_gap_loschmidt_crossover.csv",
        help="Output CSV for crossover fit.",
    )

    plateau_p = subparsers.add_parser(
        "plateau_diagnostic", help="Run plateau control suite (depth 0/1/2 baselines plus compiled distinctness)."
    )
    plateau_p.add_argument("--device", required=True, help='Braket device ARN or "local"')
    plateau_p.add_argument("--n_qubits", type=int, required=True, help="Number of qubits (must be even).")
    plateau_p.add_argument("--depths", type=str, default="0,1,2", help='Comma-separated depths (default "0,1,2").')
    plateau_p.add_argument(
        "--families",
        type=str,
        default="ring,grid,expander",
        help="Comma-separated matching families (ring,grid,expander).",
    )
    plateau_p.add_argument("--seeds", type=int, default=10, help="Seeds per (family, depth).")
    plateau_p.add_argument("--shots", type=int, default=1000, help="Shots per circuit.")
    plateau_p.add_argument("--output_dir", type=str, default="results", help="Directory for JSONL output.")
    plateau_p.add_argument(
        "--output_jsonl",
        type=str,
        default=None,
        help="Optional explicit JSONL path (defaults to output_dir/plateau_diagnostic.jsonl).",
    )
    plateau_p.add_argument(
        "--interleave",
        dest="interleave",
        action="store_true",
        default=True,
        help="Interleave submissions across families (default: True).",
    )
    plateau_p.add_argument(
        "--no-interleave",
        dest="interleave",
        action="store_false",
        help="Disable interleaving (submit all jobs for one family before the next).",
    )
    plateau_p.add_argument("--base_seed", type=int, default=1234, help="Base seed controlling randomness.")
    plateau_p.add_argument(
        "--embed_strategy",
        type=str,
        default="random",
        help="Embedding strategy tag (metadata only; use per-device tooling as needed).",
    )
    plateau_p.add_argument(
        "--summary_csv",
        type=str,
        default=None,
        help="Optional CSV path for plateau summary (defaults to output_dir/plateau_summary.csv).",
    )
    plateau_p.add_argument(
        "--delta_csv",
        type=str,
        default=None,
        help="Optional CSV path for plateau delta summary (defaults to output_dir/plateau_delta_summary.csv).",
    )
    plateau_p.add_argument(
        "--distinct_csv",
        type=str,
        default=None,
        help="Optional CSV path for plateau compiled distinctness (defaults to output_dir/plateau_distinctness.csv).",
    )
    plateau_p.add_argument(
        "--analyze_after",
        action="store_true",
        help="After running, print plateau summary and write CSVs.",
    )
    plateau_p.add_argument(
        "--self_test",
        action="store_true",
        help="Run a local plateau self-test (n_qubits=8, depths=0,1,2, seeds=3, shots=200) and summarize.",
    )
    plateau_p.add_argument(
        "--two_qubit_entangler",
        type=str,
        choices=["auto", "cz", "zz", "ms"],
        default="auto",
        help="Two-qubit entangler to use inside matching layers (auto selects ZZ on Forte, MS on Aria, CZ otherwise).",
    )

    plateau_analyze_p = subparsers.add_parser(
        "plateau_analyze", help="Analyze plateau diagnostic JSONL and export summary/distinctness CSVs."
    )
    plateau_analyze_p.add_argument("--input", required=True, help="Path to JSONL produced by plateau_diagnostic.")
    plateau_analyze_p.add_argument(
        "--csv_out",
        default="results/plateau_summary.csv",
        help="Output CSV for mean/stderr vs depth.",
    )
    plateau_analyze_p.add_argument(
        "--delta_csv",
        default="results/plateau_delta_summary.csv",
        help="Output CSV for depth0-depth1 and depth1-depth2 deltas.",
    )
    plateau_analyze_p.add_argument(
        "--distinct_csv",
        default="results/plateau_distinctness_summary.csv",
        help="Output CSV for compiled distinctness table.",
    )
    plateau_analyze_p.add_argument(
        "--depths",
        type=str,
        default="0,1,2",
        help='Comma-separated depths to include (default "0,1,2").',
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "sweep":
        device = LocalSimulator() if args.device == "local" else AwsDevice(args.device)
        entangler_mode = resolve_entangler_mode(device, args.two_qubit_entangler)
        if args.self_test or entangler_mode in ("zz", "ms"):
            print(f"Running entangler smoke test (mode={entangler_mode}) on local simulator...")
            entangler_smoke_test(entangler_mode)
        if args.self_test:
            depths = [2, 4]
            n_qubits = 8
            shots = 200
            output_jsonl = args.output_jsonl or os.path.join(args.output_dir, "spectral_gap_loschmidt_self_test.jsonl")
            print("Running self-test with n_qubits=8, depths=2,4, shots=200 on local simulator.")
        else:
            depths = parse_depths_arg(args.depths)
            n_qubits = args.n_qubits
            shots = args.shots
            output_jsonl = args.output_jsonl or os.path.join(args.output_dir, "spectral_gap_loschmidt.jsonl")

        families = [f.strip() for f in args.families.split(",") if f.strip()]
        sweep(
            device=device,
            families=families,
            depths=depths,
            n_seeds=args.n_seeds,
            n_qubits=n_qubits,
            shots=shots,
            output_jsonl=output_jsonl,
            interleave=args.interleave,
            base_seed=args.base_seed,
            embed_strategy=args.embed_strategy,
            entangler_mode=entangler_mode,
        )

    elif args.command == "analyze":
        results = load_jsonl(args.input)
        summary_rows, crossover_rows = summarize(results)
        write_csv(
            args.csv_out,
            summary_rows,
            ["family", "depth", "mean_p_return", "stderr_p_return", "n"],
        )
        write_csv(
            args.crossover_csv,
            crossover_rows,
            ["family", "d_star", "slope_lo", "slope_hi"],
        )
        print(f"Wrote summary to {args.csv_out}")
        print(f"Wrote crossover fits to {args.crossover_csv}")

    elif args.command == "plateau_diagnostic":
        device = LocalSimulator() if args.device == "local" else AwsDevice(args.device)
        entangler_mode = resolve_entangler_mode(device, args.two_qubit_entangler)
        if args.self_test or entangler_mode in ("zz", "ms"):
            print(f"Running entangler smoke test (mode={entangler_mode}) on local simulator...")
            entangler_smoke_test(entangler_mode)
        if args.self_test:
            depths = [0, 1, 2]
            n_qubits = 8
            shots = 200
            n_seeds = 3
            output_jsonl = args.output_jsonl or os.path.join(args.output_dir, "plateau_diagnostic_self_test.jsonl")
            summary_csv = args.summary_csv or os.path.join(args.output_dir, "plateau_summary_self_test.csv")
            delta_csv = args.delta_csv or os.path.join(args.output_dir, "plateau_delta_summary_self_test.csv")
            distinct_csv = args.distinct_csv or os.path.join(args.output_dir, "plateau_distinctness_self_test.csv")
            print("Running plateau self-test with n_qubits=8, depths=0,1,2, seeds=3, shots=200 on local simulator.")
        else:
            depths = parse_depths_arg(args.depths)
            n_qubits = args.n_qubits
            shots = args.shots
            n_seeds = args.seeds
            output_jsonl = args.output_jsonl or os.path.join(args.output_dir, "plateau_diagnostic.jsonl")
            summary_csv = args.summary_csv or os.path.join(args.output_dir, "plateau_summary.csv")
            delta_csv = args.delta_csv or os.path.join(args.output_dir, "plateau_delta_summary.csv")
            distinct_csv = args.distinct_csv or os.path.join(args.output_dir, "plateau_distinctness.csv")

        require_even_qubits(n_qubits)
        families = [f.strip() for f in args.families.split(",") if f.strip()]
        plateau_sweep(
            device=device,
            families=families,
            depths=depths,
            n_seeds=n_seeds,
            n_qubits=n_qubits,
            shots=shots,
            output_jsonl=output_jsonl,
            interleave=args.interleave,
            base_seed=args.base_seed,
            embed_strategy=args.embed_strategy,
            entangler_mode=entangler_mode,
        )

        if args.analyze_after or args.self_test:
            results = load_jsonl(output_jsonl)
            summary_rows, delta_rows, distinct_rows = plateau_summaries(results, depths)
            write_csv(summary_csv, summary_rows, ["family", "depth", "mean_p_return", "stderr_p_return", "n"])
            write_csv(delta_csv, delta_rows, ["family", "delta01", "delta12"])
            write_csv(
                distinct_csv,
                distinct_rows,
                ["family", "depth", "median_compiled_depth", "median_compiled_2q", "unique_compiled_ir_hash", "n"],
            )
            print_plateau_summary(summary_rows, delta_rows, distinct_rows)
            print(f"\nWrote plateau summaries to {summary_csv}, {delta_csv}, and {distinct_csv}")

    elif args.command == "plateau_analyze":
        results = load_jsonl(args.input)
        depths = parse_depths_arg(args.depths)
        summary_rows, delta_rows, distinct_rows = plateau_summaries(results, depths)
        write_csv(args.csv_out, summary_rows, ["family", "depth", "mean_p_return", "stderr_p_return", "n"])
        write_csv(args.delta_csv, delta_rows, ["family", "delta01", "delta12"])
        write_csv(
            args.distinct_csv,
            distinct_rows,
            ["family", "depth", "median_compiled_depth", "median_compiled_2q", "unique_compiled_ir_hash", "n"],
        )
        print_plateau_summary(summary_rows, delta_rows, distinct_rows)
        print(f"Wrote plateau summary to {args.csv_out}")
        print(f"Wrote plateau deltas to {args.delta_csv}")
        print(f"Wrote plateau distinctness to {args.distinct_csv}")
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
