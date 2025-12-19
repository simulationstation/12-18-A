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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


def _two_qubit_gate_fn(device) -> str:
    """Pick a native 2Q gate label, preferring CZ then CNOT."""
    native = getattr(getattr(device, "properties", None), "paradigm", None)
    if native and getattr(native, "nativeGateSet", None):
        gateset = {g.lower() for g in native.nativeGateSet}
    else:
        gateset = set()
    if "cz" in gateset:
        return "cz"
    if "cnot" in gateset or "cx" in gateset:
        return "cnot"
    return "cz"


def _apply_two_qubit(circ: Circuit, gate_label: str, control: int, target: int) -> None:
    if gate_label == "cnot":
        circ.cnot(control, target)
    else:
        circ.cz(control, target)


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
) -> List[Gate]:
    """Apply a layer of two-qubit gates and return a log of applied gates."""
    applied: List[Gate] = []
    for control, target in matches:
        _apply_two_qubit(circ, gate_label, control, target)
        applied.append((gate_label, (control, target), ()))
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
    elif name in ("cz", "cnot"):
        _apply_two_qubit(circ, name, qubits[0], qubits[1])
    else:
        raise ValueError(f"Unsupported gate for inversion: {name}")


def build_loschmidt_circuit(
    n_qubits: int,
    matches: List[Matching],
    rng: np.random.Generator,
    gate_label: str,
) -> Circuit:
    """Construct U(d) followed by its exact inverse."""
    circ = Circuit()
    log: List[Gate] = []
    for layer in matches:
        log.extend(build_scramble_layer(circ, n_qubits, rng))
        log.extend(apply_matching_layer(circ, layer, gate_label))

    # Append inverse
    for gate in reversed(log):
        invert_gate(circ, gate)

    circ.measure_all()
    return circ


# ---------------------------------------------------------------------------
# Data + run helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SweepJob:
    family: str
    depth: int
    seed: int


def circuit_hash(circ: Circuit) -> str:
    ir_json = circ.to_ir().json()
    return hashlib.sha256(ir_json.encode("utf-8")).hexdigest()


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


def run_job(
    device,
    job: SweepJob,
    n_qubits: int,
    base_seed: int,
    shots: int,
    gate_label: str,
    embed_strategy: str,
) -> Dict:
    seed_seq = np.random.SeedSequence([base_seed, job.seed, job.depth, hash(job.family) & 0xFFFFFFFF])
    rng = np.random.default_rng(seed_seq)
    schedule = matching_schedule(job.family, n_qubits, job.depth, rng)
    circ = build_loschmidt_circuit(n_qubits, schedule, rng, gate_label)

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
) -> None:
    gate_label = _two_qubit_gate_fn(device)
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

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "sweep":
        device = LocalSimulator() if args.device == "local" else AwsDevice(args.device)
        if args.self_test:
            depths = [2, 4]
            n_qubits = 8
            shots = 200
            output_jsonl = args.output_jsonl or os.path.join(args.output_dir, "spectral_gap_loschmidt_self_test.jsonl")
            print("Running self-test with n_qubits=8, depths=2,4, shots=200 on local simulator.")
        else:
            depths = [int(d) for d in args.depths.split(",") if d.strip()]
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
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
