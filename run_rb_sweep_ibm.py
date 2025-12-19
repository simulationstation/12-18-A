#!/usr/bin/env python3
"""
run_rb_sweep_ibm.py
===================

Full real-hardware RB sweep to test correlation between RB decay (alpha/EPC)
and architecture metric C = N * lambda2 on IBM Quantum backends.

Generates diverse connected subsets, runs StandardRB on each, and analyzes
whether decay correlates with spectral gap.
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from typing import List, Tuple, Set, Optional, Dict, Any

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.sparse.linalg import eigsh

# Qiskit imports
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_experiments.library import StandardRB


# =============================================================================
# Logging
# =============================================================================

class Logger:
    """Simple logger that writes to both stdout and a file."""

    def __init__(self, log_file: str):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
        self.fh = open(log_file, 'a', buffering=1)  # line buffered

    def log(self, msg: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


# =============================================================================
# Coupling Graph and Subset Generation
# =============================================================================

def build_coupling_graph(backend) -> nx.Graph:
    """Build undirected NetworkX graph from backend coupling map."""
    G = nx.Graph()
    coupling_map = backend.coupling_map
    if coupling_map is None:
        raise ValueError(f"Backend {backend.name} has no coupling map")
    for edge in coupling_map:
        G.add_edge(edge[0], edge[1])
    return G


def jaccard_similarity(s1: Set[int], s2: Set[int]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not s1 or not s2:
        return 0.0
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return intersection / union if union > 0 else 0.0


def generate_diverse_subsets(
    G: nx.Graph,
    N: int,
    num_subsets: int,
    seed: int,
    max_jaccard: float = 0.7,
    max_attempts_per_subset: int = 1000,
    logger: Optional[Logger] = None
) -> List[Tuple[int, ...]]:
    """
    Generate diverse connected subsets of size N from graph G.

    Uses BFS from random starting nodes, rejecting subsets with
    high Jaccard similarity to existing ones.
    """
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())

    if len(nodes) < N:
        raise ValueError(f"Graph has only {len(nodes)} nodes, need at least {N}")

    subsets: List[Tuple[int, ...]] = []
    subset_sets: List[Set[int]] = []

    for i in range(num_subsets):
        found = False
        attempts = 0

        while not found and attempts < max_attempts_per_subset:
            attempts += 1

            # Random BFS to get connected subset
            start = rng.choice(nodes)
            visited = [start]
            queue = list(G.neighbors(start))
            rng.shuffle(queue)

            while len(visited) < N and queue:
                node = queue.pop(0)
                if node not in visited:
                    visited.append(node)
                    neighbors = [n for n in G.neighbors(node) if n not in visited and n not in queue]
                    rng.shuffle(neighbors)
                    queue.extend(neighbors)

            if len(visited) < N:
                continue

            subset = tuple(sorted(int(x) for x in visited[:N]))
            subset_set = set(subset)

            # Verify connectivity
            H = G.subgraph(subset)
            if not nx.is_connected(H):
                continue

            # Check diversity (Jaccard similarity)
            too_similar = False
            for existing in subset_sets:
                if jaccard_similarity(subset_set, existing) > max_jaccard:
                    too_similar = True
                    break

            if not too_similar:
                subsets.append(subset)
                subset_sets.append(subset_set)
                found = True

        if not found:
            # Relax diversity constraint if we can't find diverse subsets
            if logger:
                logger.log(f"Warning: Could not find diverse subset {i+1}, relaxing constraint")

            # Try again without diversity check
            for _ in range(max_attempts_per_subset):
                start = rng.choice(nodes)
                visited = [start]
                queue = list(G.neighbors(start))
                rng.shuffle(queue)

                while len(visited) < N and queue:
                    node = queue.pop(0)
                    if node not in visited:
                        visited.append(node)
                        neighbors = [n for n in G.neighbors(node) if n not in visited and n not in queue]
                        rng.shuffle(neighbors)
                        queue.extend(neighbors)

                if len(visited) >= N:
                    subset = tuple(sorted(int(x) for x in visited[:N]))
                    H = G.subgraph(subset)
                    if nx.is_connected(H) and subset not in subsets:
                        subsets.append(subset)
                        subset_sets.append(set(subset))
                        found = True
                        break

        if not found:
            if logger:
                logger.log(f"Warning: Could only generate {len(subsets)} subsets")
            break

    return subsets


# =============================================================================
# Architecture Metrics
# =============================================================================

def compute_lambda2(H: nx.Graph) -> float:
    """
    Compute lambda2 (spectral gap) of the normalized Laplacian.

    For small graphs (N <= 15), use dense eigenvalue computation for stability.
    """
    n = H.number_of_nodes()
    if n < 2:
        return 0.0

    # Use dense computation for small graphs (more stable)
    L = nx.normalized_laplacian_matrix(H).toarray()
    eigenvalues = np.linalg.eigvalsh(L)
    eigenvalues = np.sort(eigenvalues)

    # lambda2 is the second smallest eigenvalue
    return float(eigenvalues[1])


def compute_subset_metrics(G: nx.Graph, subset: Tuple[int, ...]) -> Dict[str, float]:
    """Compute architecture metrics for a subset."""
    H = G.subgraph(subset)
    N = len(subset)

    edges_count = H.number_of_edges()
    avg_degree = 2 * edges_count / N if N > 0 else 0.0

    try:
        diameter = nx.diameter(H) if nx.is_connected(H) else -1
    except:
        diameter = -1

    lambda2 = compute_lambda2(H)
    C = N * lambda2

    return {
        'N': N,
        'edges_count': edges_count,
        'avg_degree': avg_degree,
        'diameter': diameter,
        'lambda2': lambda2,
        'C': C
    }


# =============================================================================
# RB Execution
# =============================================================================

def wait_for_queue(
    backend,
    max_pending: int,
    sleep_sec: int,
    logger: Logger
) -> int:
    """Wait until backend queue is below threshold."""
    while True:
        try:
            status = backend.status()
            pending = status.pending_jobs
        except Exception as e:
            logger.log(f"Warning: Could not get backend status: {e}")
            pending = 0

        if pending <= max_pending:
            return pending

        logger.log(f"Queue high ({pending} pending > {max_pending}), waiting {sleep_sec}s...")
        time.sleep(sleep_sec)


def run_rb_on_subset(
    backend,
    subset: Tuple[int, ...],
    lengths: List[int],
    num_samples: int,
    seed: int,
    max_failures: int,
    logger: Logger
) -> Dict[str, Any]:
    """
    Run StandardRB on a subset and return results.

    Returns dict with alpha, epc, errors, job_ids, and status.
    """
    result = {
        'alpha': None,
        'alpha_err': None,
        'epc': None,
        'epc_err': None,
        'job_ids': '',
        'status': 'failed'
    }

    for attempt in range(max_failures):
        try:
            logger.log(f"  Attempt {attempt + 1}/{max_failures}")

            # Create RB experiment
            rb_exp = StandardRB(
                physical_qubits=subset,
                lengths=lengths,
                num_samples=num_samples,
                seed=seed + attempt
            )

            # Run on backend
            exp_data = rb_exp.run(backend=backend)
            logger.log(f"  Job submitted, waiting for results...")
            exp_data.block_for_results()
            logger.log(f"  Results received!")

            # Extract results
            analysis_results = exp_data.analysis_results()

            for res in analysis_results:
                name = res.name
                value = res.value

                if name == "alpha":
                    if hasattr(value, 'nominal_value'):
                        result['alpha'] = value.nominal_value
                        result['alpha_err'] = value.std_dev
                    else:
                        result['alpha'] = float(value)
                        result['alpha_err'] = 0.0

                elif name == "EPC":
                    if hasattr(value, 'nominal_value'):
                        result['epc'] = value.nominal_value
                        result['epc_err'] = value.std_dev
                    else:
                        result['epc'] = float(value)
                        result['epc_err'] = 0.0

            # Get job IDs
            try:
                result['job_ids'] = ",".join([j.job_id() for j in exp_data.jobs()])
            except:
                result['job_ids'] = "N/A"

            result['status'] = 'ok'
            return result

        except Exception as e:
            logger.log(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < max_failures - 1:
                time.sleep(10)  # Brief pause before retry

    return result


def submit_rb_job(
    backend,
    subset: Tuple[int, ...],
    lengths: List[int],
    num_samples: int,
    seed: int,
    logger: Logger
) -> Dict[str, Any]:
    """
    Submit StandardRB job without blocking for results.

    Returns dict with job_id, experiment_id, and submission info.
    """
    try:
        rb_exp = StandardRB(
            physical_qubits=subset,
            lengths=lengths,
            num_samples=num_samples,
            seed=seed
        )

        exp_data = rb_exp.run(backend=backend)

        # Get job ID(s) immediately after submission
        job_ids = []
        try:
            for job in exp_data.jobs():
                job_ids.append(job.job_id())
        except:
            pass

        # Get experiment ID for later retrieval
        exp_id = exp_data.experiment_id

        logger.log(f"  Submitted: job_ids={job_ids}, exp_id={exp_id}")

        return {
            'job_ids': job_ids,
            'experiment_id': exp_id,
            'status': 'submitted'
        }

    except Exception as e:
        logger.log(f"  Submit failed: {e}")
        return {
            'job_ids': [],
            'experiment_id': None,
            'status': 'submit_failed',
            'error': str(e)
        }


def collect_job_results(
    service: QiskitRuntimeService,
    job_id: str,
    logger: Logger
) -> Dict[str, Any]:
    """
    Collect results from a previously submitted job.

    Returns dict with alpha, epc, status.
    """
    result = {
        'alpha': None,
        'alpha_err': None,
        'epc': None,
        'epc_err': None,
        'status': 'unknown'
    }

    try:
        job = service.job(job_id)
        status = job.status()

        if status.name == 'DONE':
            # Job completed, get results
            from qiskit_experiments.framework import ExperimentData

            # Retrieve the experiment data
            exp_data = ExperimentData.load(job_id, service)
            analysis_results = exp_data.analysis_results()

            for res in analysis_results:
                name = res.name
                value = res.value

                if name == "alpha":
                    if hasattr(value, 'nominal_value'):
                        result['alpha'] = value.nominal_value
                        result['alpha_err'] = value.std_dev
                    else:
                        result['alpha'] = float(value)
                        result['alpha_err'] = 0.0

                elif name == "EPC":
                    if hasattr(value, 'nominal_value'):
                        result['epc'] = value.nominal_value
                        result['epc_err'] = value.std_dev
                    else:
                        result['epc'] = float(value)
                        result['epc_err'] = 0.0

            result['status'] = 'ok'

        elif status.name in ['QUEUED', 'RUNNING', 'PENDING']:
            result['status'] = 'pending'
            logger.log(f"  Job {job_id}: {status.name}")

        elif status.name in ['CANCELLED', 'FAILED', 'ERROR']:
            result['status'] = 'failed'
            logger.log(f"  Job {job_id}: {status.name}")

        else:
            result['status'] = status.name.lower()

    except Exception as e:
        logger.log(f"  Error collecting {job_id}: {e}")
        result['status'] = 'error'
        result['error'] = str(e)

    return result


# =============================================================================
# CSV I/O
# =============================================================================

CSV_COLUMNS = [
    'backend', 'subset_id', 'qubits', 'N', 'lambda2', 'C',
    'edges_count', 'avg_degree', 'diameter', 'lengths', 'num_samples',
    'alpha', 'alpha_err', 'epc', 'epc_err', 'job_ids', 'timestamp', 'status'
]


def load_completed_subset_ids(csv_path: str) -> Set[int]:
    """Load subset_ids that completed successfully from existing CSV."""
    completed = set()
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if 'subset_id' in df.columns and 'status' in df.columns:
                completed = set(df[df['status'] == 'ok']['subset_id'].tolist())
        except Exception:
            pass
    return completed


def append_result_to_csv(csv_path: str, row: Dict[str, Any]):
    """Append a single result row to CSV."""
    file_exists = os.path.exists(csv_path)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# =============================================================================
# Analysis and Plotting
# =============================================================================

def analyze_results(csv_path: str, out_dir: str, logger: Logger):
    """Analyze sweep results and generate plots and report."""
    logger.log("=" * 60)
    logger.log("ANALYSIS")
    logger.log("=" * 60)

    if not os.path.exists(csv_path):
        logger.log("No results CSV found, skipping analysis")
        return

    df = pd.read_csv(csv_path)
    df_ok = df[df['status'] == 'ok'].copy()

    if len(df_ok) < 3:
        logger.log(f"Only {len(df_ok)} successful subsets, need at least 3 for analysis")
        return

    logger.log(f"Analyzing {len(df_ok)} successful subsets")

    # Filter out rows with missing alpha/epc
    df_ok = df_ok.dropna(subset=['alpha', 'C'])

    if len(df_ok) < 3:
        logger.log("Not enough valid data points after filtering")
        return

    # === Correlation Analysis ===
    logger.log("\n--- Correlation Analysis ---")

    # Spearman correlation: alpha vs C
    spearman_alpha_C, p_alpha_C = stats.spearmanr(df_ok['C'], df_ok['alpha'])
    logger.log(f"Spearman(alpha, C): rho={spearman_alpha_C:.4f}, p={p_alpha_C:.4f}")

    # Spearman correlation: epc vs C
    df_epc = df_ok.dropna(subset=['epc'])
    if len(df_epc) >= 3:
        spearman_epc_C, p_epc_C = stats.spearmanr(df_epc['C'], df_epc['epc'])
        logger.log(f"Spearman(EPC, C): rho={spearman_epc_C:.4f}, p={p_epc_C:.4f}")
    else:
        spearman_epc_C, p_epc_C = None, None
        logger.log("Not enough EPC data for correlation")

    # Linear regression: alpha ~ C
    slope, intercept, r_value, p_value, std_err = stats.linregress(df_ok['C'], df_ok['alpha'])
    r_squared = r_value ** 2
    logger.log(f"Linear regression alpha ~ C:")
    logger.log(f"  slope b = {slope:.6f} (SE={std_err:.6f})")
    logger.log(f"  intercept a = {intercept:.6f}")
    logger.log(f"  R^2 = {r_squared:.4f}, p = {p_value:.4f}")

    # Control: alpha vs edges_count
    spearman_alpha_edges, p_alpha_edges = stats.spearmanr(df_ok['edges_count'], df_ok['alpha'])
    logger.log(f"Spearman(alpha, edges_count): rho={spearman_alpha_edges:.4f}, p={p_alpha_edges:.4f}")

    # === Plots ===
    logger.log("\n--- Generating Plots ---")

    # Plot 1: alpha vs C
    plt.figure(figsize=(8, 6))
    if 'alpha_err' in df_ok.columns and df_ok['alpha_err'].notna().any():
        plt.errorbar(df_ok['C'], df_ok['alpha'], yerr=df_ok['alpha_err'],
                     fmt='o', capsize=3, alpha=0.7)
    else:
        plt.scatter(df_ok['C'], df_ok['alpha'], alpha=0.7)

    # Add regression line
    x_line = np.linspace(df_ok['C'].min(), df_ok['C'].max(), 100)
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, 'r--', label=f'y = {slope:.4f}x + {intercept:.4f}')

    plt.xlabel('C = N * lambda2')
    plt.ylabel('alpha (RB decay parameter)')
    plt.title(f'RB Alpha vs Architecture Metric C\n(Spearman rho={spearman_alpha_C:.3f}, p={p_alpha_C:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'rb_alpha_vs_C.png'), dpi=150)
    plt.close()
    logger.log(f"  Saved rb_alpha_vs_C.png")

    # Plot 2: EPC vs C
    if len(df_epc) >= 3:
        plt.figure(figsize=(8, 6))
        if 'epc_err' in df_epc.columns and df_epc['epc_err'].notna().any():
            plt.errorbar(df_epc['C'], df_epc['epc'], yerr=df_epc['epc_err'],
                         fmt='o', capsize=3, alpha=0.7)
        else:
            plt.scatter(df_epc['C'], df_epc['epc'], alpha=0.7)

        plt.xlabel('C = N * lambda2')
        plt.ylabel('EPC (Error per Clifford)')
        plt.title(f'RB EPC vs Architecture Metric C\n(Spearman rho={spearman_epc_C:.3f}, p={p_epc_C:.3f})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'rb_epc_vs_C.png'), dpi=150)
        plt.close()
        logger.log(f"  Saved rb_epc_vs_C.png")

    # Plot 3: alpha vs edges (control)
    plt.figure(figsize=(8, 6))
    plt.scatter(df_ok['edges_count'], df_ok['alpha'], alpha=0.7)
    plt.xlabel('Edge Count')
    plt.ylabel('alpha (RB decay parameter)')
    plt.title(f'RB Alpha vs Edge Count (Control)\n(Spearman rho={spearman_alpha_edges:.3f}, p={p_alpha_edges:.3f})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'rb_alpha_vs_edges.png'), dpi=150)
    plt.close()
    logger.log(f"  Saved rb_alpha_vs_edges.png")

    # === Final Answer ===
    logger.log("\n--- FINAL ANSWER ---")

    # Criteria: p < 0.05 and positive slope
    significant = p_alpha_C < 0.05
    positive_slope = slope > 0

    if significant and positive_slope:
        conclusion = (
            f"EVIDENCE CONSISTENT WITH ARCHITECTURE-LINKED DEGRADATION:\n"
            f"  Spearman correlation between alpha and C is significant (p={p_alpha_C:.4f} < 0.05)\n"
            f"  with positive slope (b={slope:.6f}), suggesting subsets with higher C\n"
            f"  (better connectivity) show higher alpha (slower decay)."
        )
    elif significant and not positive_slope:
        conclusion = (
            f"UNEXPECTED NEGATIVE CORRELATION:\n"
            f"  Spearman correlation is significant (p={p_alpha_C:.4f}) but slope is negative\n"
            f"  (b={slope:.6f}). This is opposite to the expected direction."
        )
    else:
        conclusion = (
            f"NO DETECTABLE CORRELATION at N={df_ok['N'].iloc[0]} on this backend\n"
            f"  within current statistical power.\n"
            f"  Spearman p={p_alpha_C:.4f} >= 0.05, slope b={slope:.6f}"
        )

    logger.log(conclusion)

    # === Write Report ===
    report_path = os.path.join(out_dir, 'rb_sweep_report.md')
    backend_name = df_ok['backend'].iloc[0] if 'backend' in df_ok.columns else 'unknown'
    N_val = df_ok['N'].iloc[0] if 'N' in df_ok.columns else 'unknown'

    report = f"""# RB Sweep Analysis Report

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Backend:** {backend_name}
**Subset Size (N):** {N_val}
**Subsets Completed:** {len(df_ok)}

## Correlation Analysis

| Metric | Spearman rho | p-value |
|--------|--------------|---------|
| alpha vs C | {spearman_alpha_C:.4f} | {p_alpha_C:.4f} |
| EPC vs C | {spearman_epc_C:.4f if spearman_epc_C else 'N/A'} | {p_epc_C:.4f if p_epc_C else 'N/A'} |
| alpha vs edges | {spearman_alpha_edges:.4f} | {p_alpha_edges:.4f} |

## Linear Regression (alpha ~ C)

- Slope (b): {slope:.6f} (SE: {std_err:.6f})
- Intercept (a): {intercept:.6f}
- R-squared: {r_squared:.4f}
- p-value: {p_value:.4f}

## Conclusion

{conclusion}

## Plots

- `rb_alpha_vs_C.png`: Alpha decay parameter vs architecture metric C
- `rb_epc_vs_C.png`: Error per Clifford vs C
- `rb_alpha_vs_edges.png`: Control plot (alpha vs edge count)
"""

    with open(report_path, 'w') as f:
        f.write(report)

    logger.log(f"\nReport saved to: {report_path}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RB sweep on IBM Quantum to test correlation with architecture metric C"
    )
    parser.add_argument('--backend', type=str, default='ibm_torino',
                        help='IBM Quantum backend name')
    parser.add_argument('--N', type=int, default=7,
                        help='Subset size (number of qubits)')
    parser.add_argument('--num_subsets', type=int, default=30,
                        help='Number of subsets to test')
    parser.add_argument('--lengths', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32],
                        help='RB sequence lengths')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of RB samples per length')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--out_csv', type=str, default='results/rb_sweep.csv',
                        help='Output CSV path')
    parser.add_argument('--log_file', type=str, default='results/rb_sweep.log',
                        help='Log file path')
    parser.add_argument('--resume', type=int, choices=[0, 1], default=1,
                        help='Resume from existing CSV (1) or start fresh (0)')
    parser.add_argument('--max_pending_jobs', type=int, default=25,
                        help='Max pending jobs before waiting')
    parser.add_argument('--sleep_sec', type=int, default=60,
                        help='Sleep interval when queue is high')
    parser.add_argument('--max_failures_per_subset', type=int, default=2,
                        help='Max retry attempts per subset')
    parser.add_argument('--mode', type=str, choices=['sweep', 'submit', 'collect', 'status'],
                        default='sweep',
                        help='Mode: sweep (blocking), submit (queue all), collect (get results), status (check jobs)')
    parser.add_argument('--jobs_file', type=str, default='results/rb_jobs.json',
                        help='JSON file to track submitted jobs')
    return parser.parse_args()


def load_jobs_tracking(jobs_file: str) -> Dict[str, Any]:
    """Load jobs tracking file."""
    if os.path.exists(jobs_file):
        with open(jobs_file, 'r') as f:
            return json.load(f)
    return {'jobs': {}, 'subsets': {}}


def save_jobs_tracking(jobs_file: str, tracking: Dict[str, Any]):
    """Save jobs tracking file."""
    os.makedirs(os.path.dirname(jobs_file) if os.path.dirname(jobs_file) else '.', exist_ok=True)
    with open(jobs_file, 'w') as f:
        json.dump(tracking, f, indent=2)


def main():
    args = parse_args()

    # Setup output directory
    out_dir = os.path.dirname(args.out_csv) or 'results'
    os.makedirs(out_dir, exist_ok=True)

    # Setup logger
    logger = Logger(args.log_file)

    logger.log("=" * 60)
    logger.log(f"IBM Quantum RB Sweep - Mode: {args.mode}")
    logger.log("=" * 60)
    logger.log(f"Backend: {args.backend}")
    logger.log(f"Subset size N: {args.N}")
    logger.log(f"Num subsets: {args.num_subsets}")
    logger.log(f"Jobs file: {args.jobs_file}")
    logger.log("=" * 60)

    # Connect to IBM Quantum
    logger.log("\nConnecting to IBM Quantum...")
    try:
        service = QiskitRuntimeService()
    except Exception as e:
        logger.log(f"ERROR: Failed to connect: {e}")
        sys.exit(1)

    # Get backend
    logger.log(f"Loading backend {args.backend}...")
    try:
        backend = service.backend(args.backend)
        logger.log(f"Backend: {backend.name}, {backend.num_qubits} qubits")
    except Exception as e:
        logger.log(f"ERROR: Failed to get backend: {e}")
        sys.exit(1)

    # Build coupling graph
    logger.log("Building coupling graph...")
    G = build_coupling_graph(backend)
    logger.log(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Generate subsets (deterministic with seed)
    logger.log(f"\nGenerating {args.num_subsets} diverse subsets of size {args.N}...")
    subsets = generate_diverse_subsets(
        G, args.N, args.num_subsets, args.seed,
        max_jaccard=0.7, logger=logger
    )
    logger.log(f"Generated {len(subsets)} subsets")

    # Load tracking file
    tracking = load_jobs_tracking(args.jobs_file)

    # =================================================================
    # MODE: SUBMIT - Submit all jobs without blocking
    # =================================================================
    if args.mode == 'submit':
        logger.log("\n" + "=" * 60)
        logger.log("SUBMIT MODE - Queuing all jobs")
        logger.log("=" * 60)

        # Load completed from CSV
        completed_ids = load_completed_subset_ids(args.out_csv) if args.resume else set()

        submitted_count = 0
        skipped_count = 0

        for i, subset in enumerate(subsets):
            subset_id = i
            subset_key = str(subset_id)

            # Skip if already completed in CSV
            if subset_id in completed_ids:
                logger.log(f"[{i+1}/{len(subsets)}] Subset {subset_id}: already in CSV, skipping")
                skipped_count += 1
                continue

            # Skip if already submitted
            if subset_key in tracking['subsets'] and tracking['subsets'][subset_key].get('status') == 'submitted':
                logger.log(f"[{i+1}/{len(subsets)}] Subset {subset_id}: already submitted, skipping")
                skipped_count += 1
                continue

            logger.log(f"\n[{i+1}/{len(subsets)}] Subset {subset_id}")
            logger.log(f"  Qubits: {subset}")

            # Compute metrics
            metrics = compute_subset_metrics(G, subset)
            logger.log(f"  C={metrics['C']:.4f}")

            # Submit job (non-blocking)
            submit_result = submit_rb_job(
                backend, subset, args.lengths, args.num_samples,
                args.seed + i, logger
            )

            # Save to tracking
            tracking['subsets'][subset_key] = {
                'subset_id': subset_id,
                'qubits': list(subset),
                'metrics': metrics,
                'job_ids': submit_result['job_ids'],
                'experiment_id': submit_result['experiment_id'],
                'status': submit_result['status'],
                'submit_time': datetime.now().isoformat()
            }

            # Track job IDs
            for job_id in submit_result['job_ids']:
                tracking['jobs'][job_id] = {
                    'subset_id': subset_id,
                    'status': 'submitted'
                }

            save_jobs_tracking(args.jobs_file, tracking)
            submitted_count += 1

            # Small delay between submissions
            time.sleep(1)

        logger.log(f"\n" + "=" * 60)
        logger.log(f"SUBMIT COMPLETE: {submitted_count} submitted, {skipped_count} skipped")
        logger.log(f"Jobs tracking saved to: {args.jobs_file}")
        logger.log("=" * 60)

    # =================================================================
    # MODE: STATUS - Check status of submitted jobs
    # =================================================================
    elif args.mode == 'status':
        logger.log("\n" + "=" * 60)
        logger.log("STATUS MODE - Checking job statuses")
        logger.log("=" * 60)

        if not tracking['jobs']:
            logger.log("No jobs in tracking file")
        else:
            status_counts = {}
            for job_id, job_info in tracking['jobs'].items():
                try:
                    job = service.job(job_id)
                    status = job.status().name
                    status_counts[status] = status_counts.get(status, 0) + 1
                    logger.log(f"  Job {job_id}: {status}")
                except Exception as e:
                    logger.log(f"  Job {job_id}: ERROR - {e}")
                    status_counts['ERROR'] = status_counts.get('ERROR', 0) + 1

            logger.log(f"\nSummary: {status_counts}")

    # =================================================================
    # MODE: COLLECT - Collect results from completed jobs
    # =================================================================
    elif args.mode == 'collect':
        logger.log("\n" + "=" * 60)
        logger.log("COLLECT MODE - Gathering results")
        logger.log("=" * 60)

        collected = 0
        pending = 0
        failed = 0

        for subset_key, subset_info in tracking['subsets'].items():
            subset_id = subset_info['subset_id']
            job_ids = subset_info.get('job_ids', [])

            if not job_ids:
                logger.log(f"Subset {subset_id}: No job IDs")
                continue

            # Check first job's status
            job_id = job_ids[0]
            logger.log(f"\nSubset {subset_id} (job {job_id}):")

            try:
                job = service.job(job_id)
                status = job.status()

                if status.name == 'DONE':
                    # Get results from job
                    result = job.result()

                    # Try to extract RB analysis results
                    # This is tricky - we need to re-run analysis or get from experiment
                    from qiskit_experiments.framework import ExperimentData

                    try:
                        exp_data = ExperimentData.load(subset_info.get('experiment_id'), service)
                        exp_data.block_for_results(timeout=60)
                        analysis_results = exp_data.analysis_results()

                        alpha = None
                        alpha_err = None
                        epc = None
                        epc_err = None

                        for res in analysis_results:
                            if res.name == "alpha":
                                if hasattr(res.value, 'nominal_value'):
                                    alpha = res.value.nominal_value
                                    alpha_err = res.value.std_dev
                                else:
                                    alpha = float(res.value)
                            elif res.name == "EPC":
                                if hasattr(res.value, 'nominal_value'):
                                    epc = res.value.nominal_value
                                    epc_err = res.value.std_dev
                                else:
                                    epc = float(res.value)

                        logger.log(f"  DONE: alpha={alpha}, EPC={epc}")

                        # Save to CSV
                        metrics = subset_info['metrics']
                        row = {
                            'backend': args.backend,
                            'subset_id': subset_id,
                            'qubits': str(tuple(subset_info['qubits'])),
                            'N': metrics['N'],
                            'lambda2': metrics['lambda2'],
                            'C': metrics['C'],
                            'edges_count': metrics['edges_count'],
                            'avg_degree': metrics['avg_degree'],
                            'diameter': metrics['diameter'],
                            'lengths': str(args.lengths),
                            'num_samples': args.num_samples,
                            'alpha': alpha,
                            'alpha_err': alpha_err,
                            'epc': epc,
                            'epc_err': epc_err,
                            'job_ids': ','.join(job_ids),
                            'timestamp': datetime.now().isoformat(),
                            'status': 'ok' if alpha is not None else 'failed'
                        }
                        append_result_to_csv(args.out_csv, row)
                        collected += 1

                        # Update tracking
                        tracking['subsets'][subset_key]['status'] = 'collected'
                        save_jobs_tracking(args.jobs_file, tracking)

                    except Exception as e:
                        logger.log(f"  ERROR extracting results: {e}")
                        failed += 1

                elif status.name in ['QUEUED', 'RUNNING']:
                    logger.log(f"  {status.name}")
                    pending += 1

                else:
                    logger.log(f"  {status.name}")
                    failed += 1

            except Exception as e:
                logger.log(f"  ERROR: {e}")
                failed += 1

        logger.log(f"\n" + "=" * 60)
        logger.log(f"COLLECT COMPLETE: {collected} collected, {pending} pending, {failed} failed")
        logger.log("=" * 60)

        # Run analysis if we have results
        if collected > 0:
            analyze_results(args.out_csv, out_dir, logger)

    # =================================================================
    # MODE: SWEEP - Original blocking behavior
    # =================================================================
    else:  # args.mode == 'sweep'
        # Load completed subset IDs if resuming
        completed_ids = set()
        if args.resume:
            completed_ids = load_completed_subset_ids(args.out_csv)
            if completed_ids:
                logger.log(f"Resuming: {len(completed_ids)} subsets already completed")

        logger.log("\n" + "=" * 60)
        logger.log("SWEEP MODE - Running with blocking")
        logger.log("=" * 60)

        start_time = time.time()

        for i, subset in enumerate(subsets):
            subset_id = i

            if subset_id in completed_ids:
                logger.log(f"\n[{i+1}/{len(subsets)}] Subset {subset_id} already completed, skipping")
                continue

            elapsed = time.time() - start_time
            logger.log(f"\n[{i+1}/{len(subsets)}] Subset {subset_id} (elapsed: {elapsed/60:.1f} min)")
            logger.log(f"  Qubits: {subset}")

            # Compute metrics
            metrics = compute_subset_metrics(G, subset)
            logger.log(f"  lambda2={metrics['lambda2']:.4f}, C={metrics['C']:.4f}")
            logger.log(f"  edges={metrics['edges_count']}, avg_degree={metrics['avg_degree']:.2f}, diameter={metrics['diameter']}")

            # Wait for queue
            pending = wait_for_queue(backend, args.max_pending_jobs, args.sleep_sec, logger)
            logger.log(f"  Queue: {pending} pending jobs")

            # Run RB
            logger.log(f"  Running StandardRB...")
            rb_result = run_rb_on_subset(
                backend, subset, args.lengths, args.num_samples,
                args.seed, args.max_failures_per_subset, logger
            )

            if rb_result['status'] == 'ok':
                logger.log(f"  SUCCESS: alpha={rb_result['alpha']:.4f}, EPC={rb_result['epc']:.4f}")
            else:
                logger.log(f"  FAILED after {args.max_failures_per_subset} attempts")

            # Build result row
            row = {
                'backend': args.backend,
                'subset_id': subset_id,
                'qubits': str(subset),
                'N': metrics['N'],
                'lambda2': metrics['lambda2'],
                'C': metrics['C'],
                'edges_count': metrics['edges_count'],
                'avg_degree': metrics['avg_degree'],
                'diameter': metrics['diameter'],
                'lengths': str(args.lengths),
                'num_samples': args.num_samples,
                'alpha': rb_result['alpha'],
                'alpha_err': rb_result['alpha_err'],
                'epc': rb_result['epc'],
                'epc_err': rb_result['epc_err'],
                'job_ids': rb_result['job_ids'],
                'timestamp': datetime.now().isoformat(),
                'status': rb_result['status']
            }

            # Append to CSV
            append_result_to_csv(args.out_csv, row)
            logger.log(f"  Result saved to CSV")

        total_time = time.time() - start_time
        logger.log(f"\n" + "=" * 60)
        logger.log(f"SWEEP COMPLETE in {total_time/60:.1f} minutes")
        logger.log("=" * 60)

        # Run analysis
        analyze_results(args.out_csv, out_dir, logger)

    logger.close()


if __name__ == '__main__':
    main()
