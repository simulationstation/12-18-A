#!/usr/bin/env python3
"""
Integrated IonQ Loschmidt echo experiment on AWS Braket.

This mode runs:
  (A) a core Loschmidt/Lorentz echo test on an expander family with depth
      controls (0/1) and SPAM calibration+mitigation.
  (B) a conditional ring/grid/expander spectral-gap sweep that only executes
      when the mitigated core curve shows depth-resolved decay.

Outputs:
  - JSONL of every circuit execution (raw + mitigated metrics, metadata)
  - calibration.json per block (Cal0/Cal1 counts, confusion matrices, hash)
  - integrated_decision.json describing the Phase A decay decision
  - summary.csv from the analysis helper

IonQ native entanglers: Forte-class systems target ZZ, while Aria-class systems
prefer the native Mølmer-Sørensen (MS) interaction.

Mitigation method (per-qubit, stable):
  For each qubit i, use Cal0/Cal1 to estimate the confusion matrix:
      [[P(meas0|prep0), P(meas1|prep0)],
       [P(meas0|prep1), P(meas1|prep1)]]
  Given measured marginals Pm0[i], estimate the true ground probability
      t0[i] = (Pm0[i] - P(meas0|prep1)) / (P(meas0|prep0) - P(meas0|prep1))
  Clip t0[i] into [0, 1] for stability, then form mitigated
      p_return ~= prod_i t0[i]
  This avoids matrix inversion blow-ups while using the independent-readout
  Kronecker approximation for |0...0>.
"""

import argparse
import csv
import json
import hashlib
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import linregress

from braket.aws import AwsDevice
from braket.circuits import Circuit, gates
from braket.devices import LocalSimulator


Gate = Tuple[str, Tuple[int, ...], Tuple[float, ...]]
Matching = List[Tuple[int, int]]


# ---------------------------------------------------------------------------
# Matching + circuit helpers (mirrors spectral_gap_sweep_braket)
# ---------------------------------------------------------------------------

def require_even_qubits(n_qubits: int) -> None:
    """Enforce an even qubit count for perfect matchings."""
    if n_qubits % 2 != 0:
        raise ValueError(f"n_qubits must be even to form perfect matchings; got {n_qubits}")


def ring_matchings(n_qubits: int) -> List[Matching]:
    require_even_qubits(n_qubits)
    even_edges = [((2 * k) % n_qubits, (2 * k + 1) % n_qubits) for k in range(n_qubits // 2)]
    odd_edges = [((2 * k + 1) % n_qubits, (2 * k + 2) % n_qubits) for k in range(n_qubits // 2)]
    return [even_edges, odd_edges]


def _grid_dims(n_qubits: int) -> Tuple[int, int]:
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
    require_even_qubits(n_qubits)
    qubits = np.arange(n_qubits)
    rng.shuffle(qubits)
    return [(int(qubits[i]), int(qubits[i + 1])) for i in range(0, n_qubits, 2)]


def matching_schedule(family: str, n_qubits: int, depth: int, rng: np.random.Generator) -> List[Matching]:
    if depth < 0:
        raise ValueError("depth must be non-negative")
    if family == "ring":
        base = ring_matchings(n_qubits)
        return [base[i % len(base)] for i in range(depth)]
    if family == "grid":
        base = grid_matchings(n_qubits)
        return [base[i % len(base)] for i in range(depth)]
    if family == "expander":
        return [expander_matching(n_qubits, rng) for _ in range(depth)]
    raise ValueError(f"Unknown family: {family}")


SCRAMBLE_GATES = ("rx", "ry", "rz")


def _apply_two_qubit(circ: Circuit, gate_label: str, control: int, target: int, entangler_params: Dict[str, float]) -> Gate:
    return apply_entangler(circ, control, target, entangler_params, gate_label)


def build_scramble_layer(circ: Circuit, n_qubits: int, rng: np.random.Generator) -> List[Gate]:
    applied: List[Gate] = []
    for q in range(n_qubits):
        name = rng.choice(SCRAMBLE_GATES)
        if name in ("rx", "ry"):
            angle = np.pi / 2
            getattr(circ, name)(q, angle)
            applied.append((name, (q,), (angle,)))
        elif name == "rz":
            angle = float(rng.uniform(0, 2 * np.pi))
            circ.rz(q, angle)
            applied.append((name, (q,), (angle,)))
    return applied


def apply_matching_layer(circ: Circuit, layer: Matching, gate_label: str, entangler_params: Optional[Dict[str, float]] = None) -> List[Gate]:
    applied: List[Gate] = []
    entangler_params = entangler_params or _default_entangler_params(gate_label)
    for control, target in layer:
        applied.append(_apply_two_qubit(circ, gate_label, control, target, entangler_params))
    return applied


def invert_gate(circ: Circuit, gate: Gate) -> None:
    name, qubits, params = gate
    if name == "rx":
        circ.rx(qubits[0], -params[0])
    elif name == "ry":
        circ.ry(qubits[0], -params[0])
    elif name == "rz":
        circ.rz(qubits[0], -params[0])
    elif name in ("cz", "cnot", "cx", "zz", "ms"):
        inv_params: Dict[str, float] = {"theta": -params[0]} if params else {}
        if len(params) > 1:
            inv_params["phi"] = params[1]
        _apply_two_qubit(circ, "cz" if name in ("cnot", "cx") else name, qubits[0], qubits[1], inv_params)
    else:
        raise ValueError(f"Unsupported gate for inversion: {name}")


def circuit_hash(circ: Circuit) -> str:
    try:
        ir_json = circ.to_ir().json()
        return hashlib.sha256(ir_json.encode("utf-8")).hexdigest()
    except NotImplementedError:
        return hashlib.sha256(str(circ).encode("utf-8")).hexdigest()


def gate_count_summary(circ: Circuit) -> Tuple[Dict[str, Any], int]:
    counts_by_type: Dict[str, int] = {}
    total_1q = 0
    total_2q = 0
    gate_instructions = 0
    for instr in getattr(circ, "instructions", []):
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
    try:
        return getattr(result, "_json", {}).get("action", None)
    except Exception:
        return None


def _circuit_from_action(action) -> Optional[Circuit]:
    try:
        if isinstance(action, dict) and "ir" in action:
            return Circuit.from_ir(action["ir"])
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


def _two_qubit_gate_fn(device, requested: str = "auto") -> str:
    name = _device_name(device).lower()
    if requested != "auto":
        return requested
    if "forte" in name:
        return "zz"
    if "aria" in name:
        return "ms"
    native = getattr(getattr(device, "properties", None), "paradigm", None)
    gateset = {g.lower() for g in getattr(native, "nativeGateSet", [])} if native else set()
    if "cz" in gateset:
        return "cz"
    if "cnot" in gateset or "cx" in gateset:
        return "cz"
    return "cz"


def _default_entangler_params(mode: str) -> Dict[str, float]:
    if mode in ("zz", "ms"):
        return {"theta": math.pi / 2, "phi": 0.0}
    return {}


def apply_entangler(circ: Circuit, q1: int, q2: int, params: Dict[str, float], mode: str) -> Gate:
    if mode in ("cz", "cnot", "cx"):
        if hasattr(circ, "cz"):
            circ.cz(q1, q2)
        else:
            circ.cnot(q1, q2)
        return ("cz", (q1, q2), ())

    if mode == "zz":
        theta = params.get("theta", math.pi / 2)
        # Wrap ZZ gate in verbatim box for IonQ native execution
        sub = Circuit()
        if hasattr(sub, "zz"):
            sub.zz(q1, q2, theta)
        else:
            try:
                sub.add(gates.ZZ(theta), [q1, q2])
            except Exception as exc:  # pragma: no cover - SDK dependent
                raise RuntimeError("ZZ gate not supported by installed Braket SDK") from exc
        circ.add_verbatim_box(sub)
        return ("zz", (q1, q2), (theta,))

    if mode == "ms":
        theta = params.get("theta", math.pi / 2)
        phi = params.get("phi", 0.0)
        # Wrap MS gate in verbatim box for IonQ native execution
        sub = Circuit()
        ms_method = getattr(sub, "ms", None)
        if callable(ms_method):
            try:
                ms_method(q1, q2, theta, phi)
            except TypeError:
                ms_method(q1, q2, theta)
        else:
            try:
                sub.add(gates.MS(theta, phi), [q1, q2])
            except TypeError:
                sub.add(gates.MS(theta), [q1, q2])
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("MS gate not supported by installed Braket SDK") from exc
        circ.add_verbatim_box(sub)
        return ("ms", (q1, q2), (theta, phi))

    raise ValueError(f"Unsupported entangler mode: {mode}")


# ---------------------------------------------------------------------------
# Calibration + mitigation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationData:
    block_index: int
    counts_cal0: Dict[str, int]
    counts_cal1: Dict[str, int]
    matrices: List[List[List[float]]]
    shots: int
    hash: str
    file_path: str
    timestamp: str
    compilation_mode: str


def _counts_to_marginals(counts: Dict[str, int], n_qubits: int, shots: int) -> List[float]:
    zeros = np.zeros(n_qubits, dtype=float)
    for bitstring, ct in counts.items():
        if len(bitstring) != n_qubits:
            continue
        for idx, bit in enumerate(bitstring):
            if bit == "0":
                zeros[idx] += ct
    return [z / shots for z in zeros]


def _confusion_matrices(counts0: Dict[str, int], counts1: Dict[str, int], n_qubits: int, shots0: int, shots1: int):
    marg0 = _counts_to_marginals(counts0, n_qubits, shots0)
    marg1 = _counts_to_marginals(counts1, n_qubits, shots1)
    matrices: List[List[List[float]]] = []
    for q in range(n_qubits):
        p00 = float(marg0[q])
        p01 = 1.0 - p00
        p10 = float(marg1[q])
        p11 = 1.0 - p10
        matrices.append([[p00, p01], [p10, p11]])
    return matrices


def _calibration_hash(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def run_calibration_block(
    device,
    n_qubits: int,
    shots: int,
    block_index: int,
    output_dir: str,
    prefer_verbatim: bool,
    device_info: Dict[str, str],
) -> CalibrationData:
    cal0 = Circuit()
    for q in range(n_qubits):
        cal0.measure(q)
    cal1 = Circuit()
    for q in range(n_qubits):
        cal1.x(q)
    for q in range(n_qubits):
        cal1.measure(q)

    counts0, mode0 = run_circuit(device, cal0, shots, prefer_verbatim)
    counts1, mode1 = run_circuit(device, cal1, shots, prefer_verbatim)
    matrices = _confusion_matrices(counts0, counts1, n_qubits, shots, shots)
    payload = {
        "block_index": block_index,
        "counts_cal0": counts0,
        "counts_cal1": counts1,
        "matrices": matrices,
        "shots": shots,
        "device": device_info,
        "method": "independent per-qubit confusion; mitigated p0 via clipped true-zero estimate product",
    }
    cal_hash = _calibration_hash(payload)
    timestamp = datetime.utcnow().isoformat()
    payload["hash"] = cal_hash
    payload["timestamp_utc"] = timestamp
    payload["compilation_mode_cal0"] = mode0
    payload["compilation_mode_cal1"] = mode1

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"calibration_block_{block_index}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return CalibrationData(
        block_index=block_index,
        counts_cal0=counts0,
        counts_cal1=counts1,
        matrices=matrices,
        shots=shots,
        hash=cal_hash,
        file_path=path,
        timestamp=timestamp,
        compilation_mode=f"{mode0}|{mode1}",
    )


def mitigated_p_return(
    counts: Dict[str, int],
    n_qubits: int,
    shots: int,
    matrices: List[List[List[float]]],
) -> float:
    marginals = _counts_to_marginals(counts, n_qubits, shots)
    true_zero_probs: List[float] = []
    for q in range(n_qubits):
        p00 = matrices[q][0][0]
        p10 = matrices[q][1][0]
        denom = p00 - p10
        meas = marginals[q]
        if abs(denom) < 1e-9:
            t0 = meas
        else:
            t0 = (meas - p10) / denom
        t0 = min(1.0, max(0.0, t0))
        true_zero_probs.append(t0)
    corrected = float(np.prod(true_zero_probs))
    return min(1.0, max(0.0, corrected))


# ---------------------------------------------------------------------------
# Device + execution helpers
# ---------------------------------------------------------------------------

def _device_name(device) -> str:
    return getattr(device, "name", getattr(getattr(device, "properties", None), "deviceParameters", {}).get("name", "unknown"))


def resolve_device(device_arn: Optional[str]) -> Tuple[Any, Dict[str, str]]:
    if device_arn is None:
        candidates = AwsDevice.get_devices(
            types=["QPU"],
            statuses=["ONLINE"],
            provider_names=["IonQ"],
        )
        if not candidates:
            raise RuntimeError("No ONLINE IonQ devices found on Braket.")
        dev = candidates[0]
    elif device_arn.lower() == "local":
        dev = LocalSimulator()
    else:
        dev = AwsDevice(device_arn)
    info = {
        "device_arn": getattr(dev, "arn", "local"),
        "device_name": _device_name(dev),
        "provider": "IonQ" if getattr(dev, "arn", "").lower().find("ionq") != -1 else "local" if isinstance(dev, LocalSimulator) else "unknown",
    }
    return dev, info


def run_circuit(device, circ: Circuit, shots: int, prefer_verbatim: bool) -> Tuple[Dict[str, int], str]:
    kwargs = {}
    used_mode = "standard"
    if prefer_verbatim and not isinstance(device, LocalSimulator):
        kwargs["disable_qubit_rewiring"] = True
        used_mode = "verbatim"
    try:
        result = device.run(circ, shots=shots, **kwargs).result()
        counts = dict(result.measurement_counts)
        return counts, used_mode
    except Exception:
        if kwargs:
            result = device.run(circ, shots=shots).result()
            counts = dict(result.measurement_counts)
            return counts, "standard"
        raise


def run_and_collect_metadata(
    device,
    circ: Circuit,
    shots: int,
    prefer_verbatim: bool,
    n_qubits: int,
) -> Tuple[Dict[str, int], float, str, Dict[str, Any], int, Dict[str, Any], int, str]:
    started = time.time()
    result = None
    mode = "standard"
    kwargs = {}
    if prefer_verbatim and not isinstance(device, LocalSimulator):
        kwargs["disable_qubit_rewiring"] = True
        mode = "verbatim"
    try:
        result = device.run(circ, shots=shots, **kwargs).result()
    except Exception:
        if kwargs:
            result = device.run(circ, shots=shots).result()
            mode = "standard"
        else:
            raise

    counts = dict(result.measurement_counts)
    runtime_s = time.time() - started
    logical_counts, logical_depth = gate_count_summary(circ)
    circuit_ir_hash = circuit_hash(circ)
    compiled_counts, compiled_depth, compiled_ir_hash, compiled_info = compiled_metadata_from_result(
        result, logical_counts, logical_depth, circuit_ir_hash
    )
    return (
        counts,
        runtime_s,
        mode,
        logical_counts,
        logical_depth,
        compiled_counts,
        compiled_depth,
        compiled_ir_hash,
        compiled_info,
    )


# ---------------------------------------------------------------------------
# Experiment building
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentJob:
    phase: str
    family: str
    depth: int
    seed: int


def build_integrated_loschmidt(
    family: str,
    depth: int,
    n_qubits: int,
    rng: np.random.Generator,
    gate_label: str,
    entangler_params: Optional[Dict[str, float]] = None,
) -> Tuple[Circuit, List[Matching]]:
    circ = Circuit()
    log: List[Gate] = []
    schedule: List[Matching] = []
    if depth == 0:
        pass
    elif depth == 1:
        log.extend(build_scramble_layer(circ, n_qubits, rng))
    else:
        schedule = matching_schedule(family, n_qubits, depth, rng)
        entangler_params = entangler_params or _default_entangler_params(gate_label)
        for layer in schedule:
            log.extend(build_scramble_layer(circ, n_qubits, rng))
            log.extend(apply_matching_layer(circ, layer, gate_label, entangler_params))
    for gate in reversed(log):
        invert_gate(circ, gate)
    for q in range(n_qubits):
        circ.measure(q)
    return circ, schedule


def entangler_smoke_test(entangler_mode: str) -> None:
    """Small local check to ensure selected entangler serializes and runs for N=4, depth=2."""
    rng = np.random.default_rng(123)
    schedule = matching_schedule("ring", 4, 2, rng)
    circ, _ = build_integrated_loschmidt(
        family="ring",
        depth=2,
        n_qubits=4,
        rng=rng,
        gate_label=entangler_mode,
        entangler_params=_default_entangler_params(entangler_mode),
    )
    circ_text = str(circ).lower()
    if entangler_mode in ("zz", "ms") and "cz" in circ_text:
        raise RuntimeError("Entangler smoke test detected CZ when a native IonQ entangler was requested.")
    LocalSimulator().run(circ, shots=4).result()


def _jobs_for_block(phase: str, families: Sequence[str], depths: Sequence[int], seeds: Sequence[int]) -> List[ExperimentJob]:
    jobs: List[ExperimentJob] = []
    for depth in depths:
        for seed in seeds:
            for fam in families:
                jobs.append(ExperimentJob(phase=phase, family=fam, depth=depth, seed=seed))
    return jobs


def _block_seeds(n_seeds: int, block_size: int) -> List[List[int]]:
    seeds: List[List[int]] = []
    for start in range(0, n_seeds, block_size):
        seeds.append(list(range(start, min(n_seeds, start + block_size))))
    return seeds


def _mean_stderr(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0


def _decision_from_core(records: List[Dict], depths: List[int]) -> Dict[str, Any]:
    depth_to_vals: Dict[int, List[float]] = {}
    for rec in records:
        depth_to_vals.setdefault(int(rec["depth"]), []).append(float(rec["mitigated_p_return"]))

    mean_map: Dict[int, float] = {}
    stderr_map: Dict[int, float] = {}
    for d in sorted(depth_to_vals):
        mean_map[d], stderr_map[d] = _mean_stderr(depth_to_vals[d])

    diff_pass = False
    diff_score = None
    if 2 in mean_map and 16 in mean_map:
        delta = mean_map[2] - mean_map[16]
        stderr_delta = math.sqrt((stderr_map.get(2, 0.0) or 0.0) ** 2 + (stderr_map.get(16, 0.0) or 0.0) ** 2)
        diff_score = delta
        if stderr_delta > 0:
            diff_pass = delta >= 5 * stderr_delta

    slope_pass = False
    slope = None
    p_value = None
    ordered = [(d, mean_map[d]) for d in sorted(mean_map.keys()) if mean_map[d] > 0]
    if len(ordered) >= 3:
        xs, ys = zip(*ordered)
        res = linregress(xs, ys)
        slope = res.slope
        p_value = res.pvalue
        monotone = all(ys[i] <= ys[i - 1] + 1e-9 for i in range(1, len(ys)))
        slope_pass = monotone and slope < 0 and p_value is not None and p_value < 0.01

    decision = diff_pass or slope_pass
    return {
        "mean_mitigated": mean_map,
        "stderr_mitigated": stderr_map,
        "depths_requested": depths,
        "criterion_diff_pass": diff_pass,
        "criterion_slope_pass": slope_pass,
        "delta_2_vs_16": diff_score,
        "slope": slope,
        "slope_p_value": p_value,
        "proceed_to_sweep": decision,
    }


# ---------------------------------------------------------------------------
# JSON + CSV helpers
# ---------------------------------------------------------------------------

def write_jsonl(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def write_summary_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_jsonl(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    phase_depth_family = {}
    for rec in records:
        key = (rec["phase"], rec["family"], int(rec["depth"]))
        phase_depth_family.setdefault(key, {"raw": [], "mit": []})
        phase_depth_family[key]["raw"].append(float(rec["raw_p_return"]))
        phase_depth_family[key]["mit"].append(float(rec["mitigated_p_return"]))

    summary_rows: List[Dict[str, Any]] = []
    for (phase, family, depth), vals in sorted(phase_depth_family.items(), key=lambda x: (x[0][0], x[0][2], x[0][1])):
        mean_raw, stderr_raw = _mean_stderr(vals["raw"])
        mean_mit, stderr_mit = _mean_stderr(vals["mit"])
        summary_rows.append(
            {
                "phase": phase,
                "family": family,
                "depth": depth,
                "mean_raw": mean_raw,
                "stderr_raw": stderr_raw,
                "mean_mitigated": mean_mit,
                "stderr_mitigated": stderr_mit,
                "n": len(vals["raw"]),
            }
        )

    sweep_rows: List[Dict[str, Any]] = []
    sweep_records = [r for r in records if r["phase"] == "sweep"]
    if sweep_records:
        families = set(r["family"] for r in sweep_records)
        if "expander" in families and "ring" in families:
            for depth in sorted(set(int(r["depth"]) for r in sweep_records)):
                exp_vals = [float(r["mitigated_p_return"]) for r in sweep_records if r["family"] == "expander" and int(r["depth"]) == depth]
                ring_vals = [float(r["mitigated_p_return"]) for r in sweep_records if r["family"] == "ring" and int(r["depth"]) == depth]
                if exp_vals and ring_vals:
                    mean_e, se_e = _mean_stderr(exp_vals)
                    mean_r, se_r = _mean_stderr(ring_vals)
                    delta = mean_e - mean_r
                    stderr_delta = math.sqrt(se_e ** 2 + se_r ** 2)
                    sweep_rows.append(
                        {
                            "depth": depth,
                            "delta_expander_minus_ring": delta,
                            "stderr_delta": stderr_delta,
                            "n_expander": len(exp_vals),
                            "n_ring": len(ring_vals),
                        }
                    )
    return summary_rows, sweep_rows


def crossover_diagnostic(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    by_family = {}
    for rec in records:
        if rec["phase"] != "sweep":
            continue
        by_family.setdefault(rec["family"], []).append((int(rec["depth"]), float(rec["mitigated_p_return"])))

    for fam, entries in by_family.items():
        pairs = sorted(entries, key=lambda x: x[0])
        depths = [p[0] for p in pairs if p[1] > 0]
        vals = [p[1] for p in pairs if p[1] > 0]
        if len(depths) < 4:
            continue
        logs = np.log(vals)
        best_break = None
        best_sse = None
        best_slopes = None
        for i in range(1, len(depths) - 2):
            left_x = depths[: i + 1]
            right_x = depths[i + 1 :]
            left_y = logs[: i + 1]
            right_y = logs[i + 1 :]
            sl1, inter1, _, _, _ = linregress(left_x, left_y)
            sl2, inter2, _, _, _ = linregress(right_x, right_y)
            pred_left = np.array(left_x) * sl1 + inter1
            pred_right = np.array(right_x) * sl2 + inter2
            sse = float(np.sum((pred_left - left_y) ** 2) + np.sum((pred_right - right_y) ** 2))
            if best_sse is None or sse < best_sse:
                best_sse = sse
                best_break = depths[i]
                best_slopes = (sl1, sl2)
        if best_break is not None and best_slopes is not None:
            rows.append(
                {
                    "family": fam,
                    "break_depth": best_break,
                    "slope_shallow": best_slopes[0],
                    "slope_deep": best_slopes[1],
                }
            )
    return rows


def analyze_output(input_dir: str, jsonl_path: Optional[str]) -> None:
    records: List[Dict[str, Any]] = []
    path = jsonl_path or os.path.join(input_dir, "ionq_integrated_echo.jsonl")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    if not records:
        print("No records found for analysis.")
        return
    summary_rows, sweep_rows = summarize_jsonl(records)
    summary_csv = os.path.join(input_dir, "summary.csv")
    write_summary_csv(summary_csv, summary_rows)
    delta_csv = os.path.join(input_dir, "sweep_deltas.csv")
    write_summary_csv(delta_csv, sweep_rows)
    crossover_rows = crossover_diagnostic(records)
    crossover_csv = os.path.join(input_dir, "crossover_diagnostic.csv")
    write_summary_csv(crossover_csv, crossover_rows)

    print("\nCore + sweep summaries (mean ± stderr):")
    for row in summary_rows:
        print(
            f" {row['phase']:5s} {row['family']:9s} depth {row['depth']:2d}  raw={row['mean_raw']:.4f}±{row['stderr_raw']:.4f}  "
            f"mit={row['mean_mitigated']:.4f}±{row['stderr_mitigated']:.4f}  n={row['n']}"
        )
    if sweep_rows:
        print("\nExpander - ring deltas (mitigated):")
        for row in sweep_rows:
            print(
                f" depth {row['depth']:2d} delta={row['delta_expander_minus_ring']:.4f} ± {row['stderr_delta']:.4f} "
                f"(n_expander={row['n_expander']}, n_ring={row['n_ring']})"
            )
    if crossover_rows:
        print("\nCrossover diagnostic (log-space two-segment fit):")
        for row in crossover_rows:
            print(
                f" {row['family']:9s}: break~{row['break_depth']} slopes ({row['slope_shallow']:.4f}, {row['slope_deep']:.4f})"
            )

    verdict = "Core-only data present."
    if any(r["phase"] == "sweep" for r in records):
        verdict = "Conditional sweep executed; compare expander vs ring/grid trends."
    print(f"\nVerdict: {verdict}")
    print(f"Wrote summary CSVs to {summary_csv}, {delta_csv}, and {crossover_csv}")


# ---------------------------------------------------------------------------
# Main experiment driver
# ---------------------------------------------------------------------------

def run_phase(
    device,
    device_info: Dict[str, str],
    phase: str,
    families: Sequence[str],
    depths: Sequence[int],
    n_seeds: int,
    n_qubits: int,
    shots: int,
    base_seed: int,
    output_jsonl: str,
    calibration_shots: int,
    recalibrate_every: int,
    prefer_verbatim: bool,
    gate_label: str,
    start_block: int,
    entangler_params: Optional[Dict[str, float]] = None,
    skip_calibration: bool = False,
) -> Tuple[List[Dict[str, Any]], int]:
    records: List[Dict[str, Any]] = []
    block_groups = _block_seeds(n_seeds, recalibrate_every)
    block_index = start_block
    entangler_params = entangler_params or _default_entangler_params(gate_label)
    for seeds in block_groups:
        if skip_calibration:
            # Create null calibration (no SPAM mitigation)
            calib = CalibrationData(
                mitigator=lambda p, _n: p,  # identity function
                calib_hash="no_calibration",
                block_index=block_index,
                calib_path="none",
            )
        else:
            calib = run_calibration_block(
                device=device,
                n_qubits=n_qubits,
                shots=calibration_shots,
                block_index=block_index,
                output_dir=os.path.dirname(output_jsonl) or ".",
                prefer_verbatim=prefer_verbatim,
                device_info=device_info,
            )
        jobs = _jobs_for_block(phase, families, depths, seeds)
        for job in jobs:
            seed_seq = np.random.SeedSequence([base_seed, job.seed, job.depth, hash(job.family) & 0xFFFFFFFF])
            rng = np.random.default_rng(seed_seq)
            circ, schedule = build_integrated_loschmidt(
                family=job.family,
                depth=job.depth,
                n_qubits=n_qubits,
                rng=rng,
                gate_label=gate_label,
                entangler_params=entangler_params,
            )
            (
                counts,
                runtime_s,
                compilation_mode,
                logical_counts,
                logical_depth,
                compiled_counts,
                compiled_depth,
                compiled_ir_hash,
                compiled_info,
            ) = run_and_collect_metadata(device, circ, shots, prefer_verbatim, n_qubits)

            raw_p = counts.get("0" * n_qubits, 0) / float(shots)
            mitigated = mitigated_p_return(counts, n_qubits, shots, calib.matrices)
            record = {
                "timestamp_utc": datetime.utcnow().isoformat(),
                "device_arn": device_info["device_arn"],
                "device_name": device_info["device_name"],
                "provider": device_info["provider"],
                "phase": phase,
                "family": job.family,
                "depth": job.depth,
                "seed": job.seed,
                "shots": shots,
                "N": n_qubits,
                "raw_counts": counts,
                "raw_p_return": raw_p,
                "mitigated_p_return": mitigated,
                "calibration_hash": calib.hash,
                "calibration_block": calib.block_index,
                "calibration_metadata": calib.file_path,
                "compilation_mode": compilation_mode,
                "logical_gate_counts": logical_counts,
                "logical_circuit_depth": logical_depth,
                "compiled_gate_counts": compiled_counts,
                "compiled_depth": compiled_depth,
                "compiled_ir_hash": compiled_ir_hash,
                "compiled_info_source": compiled_info,
                "program_hash": circuit_hash(circ),
                "circuit_ir_hash": circuit_hash(circ),
                "runtime_seconds": runtime_s,
                "schedule": schedule,
                "base_seed": base_seed,
                "entangler_mode": gate_label,
                "verbatim_requested": prefer_verbatim,
            }
            write_jsonl(output_jsonl, record)
            records.append(record)
            print(
                f"[{record['timestamp_utc']}] {phase} {job.family} depth {job.depth} seed {job.seed} "
                f"raw={raw_p:.4f} mitigated={mitigated:.4f} calib={calib.hash[:8]} mode={compilation_mode}"
            )
        block_index += 1
    return records, block_index


def run_integrated(args) -> None:
    device, device_info = resolve_device(args.device)
    gate_label = _two_qubit_gate_fn(device, args.two_qubit_entangler)
    entangler_params = _default_entangler_params(gate_label)
    base_seed = args.base_seed
    n_qubits = args.n_qubits
    require_even_qubits(n_qubits)
    output_jsonl = os.path.join(args.output_dir, "ionq_integrated_echo.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)

    depths_core = [int(d) for d in args.depths_core.split(",") if str(d).strip()]
    depths_sweep = [int(d) for d in args.depths_sweep.split(",") if str(d).strip()]

    print(f"Selected device: {device_info['device_name']} ({device_info['device_arn']}) provider={device_info['provider']}")
    if gate_label in ("zz", "ms"):
        print(f"Running entangler smoke test (mode={gate_label}) on local simulator...")
        entangler_smoke_test(gate_label)
    print(f"Entangler mode: {gate_label} (flag={args.two_qubit_entangler})")
    print("Starting Phase A (core)...")
    core_records, next_block = run_phase(
        device=device,
        device_info=device_info,
        phase="core",
        families=["expander"],
        depths=depths_core,
        n_seeds=args.seeds_core,
        n_qubits=n_qubits,
        shots=args.shots_core,
        base_seed=base_seed,
        output_jsonl=output_jsonl,
        calibration_shots=args.calibration_shots,
        recalibrate_every=args.recalibrate_every,
        prefer_verbatim=args.use_verbatim,
        gate_label=gate_label,
        start_block=0,
        entangler_params=entangler_params,
        skip_calibration=args.skip_calibration,
    )

    decision = _decision_from_core(core_records, depths_core)
    decision_path = os.path.join(args.output_dir, "integrated_decision.json")
    with open(decision_path, "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)
    print(f"Decision written to {decision_path}: proceed_to_sweep={decision['proceed_to_sweep']}")

    sweep_records: List[Dict[str, Any]] = []
    if decision["proceed_to_sweep"]:
        print("Starting Phase B (conditional sweep)...")
        sweep_records, _ = run_phase(
            device=device,
            device_info=device_info,
            phase="sweep",
            families=["ring", "grid", "expander"],
            depths=depths_sweep,
            n_seeds=args.seeds_sweep,
            n_qubits=n_qubits,
            shots=args.shots_sweep,
            base_seed=base_seed + 1,
            output_jsonl=output_jsonl,
            calibration_shots=args.calibration_shots,
            recalibrate_every=args.recalibrate_every,
            prefer_verbatim=args.use_verbatim,
            gate_label=gate_label,
            start_block=next_block,
            entangler_params=entangler_params,
            skip_calibration=args.skip_calibration,
        )
    else:
        print("Skipping Phase B sweep because decay criterion was not met.")

    all_records = core_records + sweep_records
    if args.run_analysis or args.local_test:
        analyze_output(args.output_dir, output_jsonl)
    elif all_records:
        summary_rows, sweep_rows = summarize_jsonl(all_records)
        summary_csv = os.path.join(args.output_dir, "summary.csv")
        write_summary_csv(summary_csv, summary_rows)
        if sweep_rows:
            write_summary_csv(os.path.join(args.output_dir, "sweep_deltas.csv"), sweep_rows)
        print(f"Wrote summary CSVs to {args.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Integrated IonQ Loschmidt echo with conditional architecture sweep.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("ionq_integrated_echo", help="Run the integrated IonQ experiment.")
    run_p.add_argument("--device", type=str, default=None, help="Optional device ARN. Use 'local' for simulator.")
    run_p.add_argument("--n_qubits", type=int, default=8, help="Even number of logical qubits.")
    run_p.add_argument("--output_dir", type=str, default="results/ionq_integrated_echo", help="Directory for outputs.")
    run_p.add_argument("--depths_core", type=str, default="0,1,2,4,8,16,24,32", help="Depth grid for Phase A.")
    run_p.add_argument("--depths_sweep", type=str, default="0,1,2,4,8,16,24,32", help="Depth grid for Phase B.")
    run_p.add_argument("--seeds_core", type=int, default=20, help="Seeds for Phase A.")
    run_p.add_argument("--seeds_sweep", type=int, default=40, help="Seeds for Phase B.")
    run_p.add_argument("--shots_core", type=int, default=5000, help="Shots per circuit in Phase A.")
    run_p.add_argument("--shots_sweep", type=int, default=5000, help="Shots per circuit in Phase B.")
    run_p.add_argument("--calibration_shots", type=int, default=20000, help="Shots per calibration circuit.")
    run_p.add_argument("--recalibrate_every", type=int, default=1, help="Seeds per calibration block.")
    run_p.add_argument(
        "--skip_calibration",
        action="store_true",
        default=False,
        help="Skip calibration and run echo circuits directly (no SPAM mitigation).",
    )
    run_p.add_argument(
        "--use_verbatim",
        dest="use_verbatim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request verbatim/native compilation when supported.",
    )
    run_p.add_argument(
        "--two_qubit_entangler",
        type=str,
        choices=["auto", "cz", "zz", "ms"],
        default="auto",
        help="Two-qubit entangler to use (auto: Forte->ZZ, Aria->MS, otherwise CZ).",
    )
    run_p.add_argument("--base_seed", type=int, default=1234, help="Base seed for RNG seeding.")
    run_p.add_argument("--run_analysis", action="store_true", help="Run analysis helper after experiment.")
    run_p.add_argument("--local_test", action="store_true", help="Use a short local simulator run for smoke testing.")

    analyze_p = sub.add_parser("analyze_integrated_echo", help="Analyze integrated IonQ experiment outputs.")
    analyze_p.add_argument("--output_dir", type=str, default="results/ionq_integrated_echo", help="Directory containing JSONL + decision.")
    analyze_p.add_argument("--jsonl", type=str, default=None, help="Optional explicit JSONL path.")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "analyze_integrated_echo":
        analyze_output(args.output_dir, args.jsonl)
        return

    if args.local_test:
        args.device = args.device or "local"
        args.seeds_core = min(args.seeds_core, 3)
        args.seeds_sweep = min(args.seeds_sweep, 3)
        args.shots_core = min(args.shots_core, 200)
        args.shots_sweep = min(args.shots_sweep, 200)
        args.calibration_shots = min(args.calibration_shots, 500)
        args.depths_core = "0,1,2"
        args.depths_sweep = "0,1,2"
        args.run_analysis = True
        print("Running local simulator self-test with reduced sizes.")

    run_integrated(args)


if __name__ == "__main__":
    main()
