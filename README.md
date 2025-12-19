# M-flow synthetic benchmark pipeline

This repository contains a toy simulation harness and a synthetic benchmarking
toolchain for exploring how an architecture metric
\(C = N \lambda_2(\text{normalized Laplacian})\) relates to a fitted decay rate
\(\alpha_{\text{hat}}\) extracted from noisy circuit runs.

## Pipeline

Follow these steps locally to recreate the full pipeline and plots in `results/`.

### A) Toy M-flow simulation (ring vs expander comparison)
```bash
python mflow_sim.py --outdir results
```
This regenerates the reference sweep and writes `alpha_vs_C_local.png`,
`alpha_vs_C_global.png`, and `Meff_vs_C.png` into `results/`.

### B) Generate synthetic benchmark data
Use Qiskit Aer to generate noisy benchmark circuits across several graph
families. The following "small but meaningful" sweep runs four sizes and five
depths with modest sampling:
```bash
python generate_benchmark_data.py --outfile bench.csv --seed 123 \
    --Ns 32 64 128 256 --depths 5 10 20 40 80 --K 3 --shots 4096 \
    --p1 1e-4 --p2 1e-3 --pm 0.0
```
The CSV contains columns `device,N,depth,success_prob` and uses device labels
such as `ring_N64`, `grid_N256`, or `random_regular_4_N128`.

### C) Analyze benchmark data
Fit \(\alpha_{\text{hat}}\) for each device label, reconstruct the coupling
graphs deterministically, and correlate the fits with \(C = N \lambda_2\):
```bash
python mflow_analyze_synth.py --bench_csv bench.csv --outdir results --seed 123 --no_show_plots
```
This writes `results/alpha_vs_C_synth.csv` plus plots
`alpha_vs_C_synth.png`, `alpha_vs_N_synth.png`, and `lambda2_vs_N_synth.png`.

### Avoiding giant runs

The all-zeros success metric decays as \(2^{-N}\) for random circuits, so
statevector runs at large \(N\) quickly become uninformative. Prefer the
mirror benchmark (`--benchmark mirror`), which keeps noiseless success near 1.

For MPS + noise, start conservatively: `--shots 512 --K 3 --depths 5 10`
and keep depths \(\leq 10\) while you validate settings. Use `--oneq_set light`
and leave `--limit_entanglement` enabled to reduce entanglement growth.

Every run prints an estimated total circuit execution count and refuses to run
if it exceeds `--max_total_executions` (default 200000) unless you pass
`--force 1`. You can further trim scope via `--families`, `--Ns`, and
`--depths` without editing code.

Live progress goes to stdout and a sidecar log (e.g., `bench.csv.log`):
```bash
tail -f bench.csv.log
```

### D) Compiler fragility experiment (AWS Braket)
Run a GHZ-based layout fragility sweep on Rigetti hardware (or simulator) and
correlate variance with the architecture metric \(C = N \lambda_2\):

```bash
python run_rb_sweep_braket.py --experiment compiler_fragility --N 5 --num-subsets 10 \
    --num-compilations 10 --shots 500 --simulator

# Analyze variance vs connectivity
python analyze_compiler_fragility.py --input results/compiler_fragility_sweep.csv \
    --plot results/compiler_fragility_std_vs_C.png \
    --summary results/compiler_fragility_correlation.csv
```

Results are appended to `results/compiler_fragility_sweep.csv` with one row per
subset containing \(\lambda_2\), \(C\), and summary statistics (mean/std/min/max)
across compilation variants.

Results are appended after each (family, N, depth) point, so you can interrupt
a long run with Ctrl-C and later resume with `--resume 1` using the same
arguments. Existing rows are skipped automatically.

### E) Loschmidt echo experiment (AWS Braket)
Probe dynamical irreversibility with a GHZ-based Loschmidt echo. Each subset
prepares GHZ, applies a random brickwork unitary of depth `d`, its exact inverse,
and unprepares GHZ; \(P_\text{return}\) tracks fidelity to \(|0...0\rangle\).

Run on Rigetti (or simulator):
```bash
python run_rb_sweep_braket.py --experiment loschmidt_echo --N 7 --num-subsets 10 \
    --depths 5,10,20,30 --loschmidt-K 5 --shots 1000 --simulator
```

Analyze decay vs connectivity:
```bash
python analyze_loschmidt_echo.py --input results/loschmidt_echo_sweep.csv \
    --scatter results/loschmidt_alpha_vs_C.png \
    --alpha-csv results/loschmidt_echo_alpha_fit.csv
```

The runner writes `results/loschmidt_echo_sweep.csv` incrementally (one row per
subset/depth) plus `results/loschmidt_echo_alpha.csv` with fitted slopes
\(\alpha_\text{echo}\) versus depth for each subset.

### F) Interleaved Loschmidt echo (block-structured)
Detect time-varying vs architecture-linked effects by interleaving subsets
across multiple blocks with randomized order. The experiment fixes a spread of
low/high-\(C\) subsets (optionally loaded from JSON), shuffles them each block,
and logs per-(block, subset, depth) return probabilities plus decay fits.

Run (requires `--run-id`):
```bash
python run_rb_sweep_braket.py --experiment loschmidt_echo_interleaved \
    --run-id test --N 11 --num-subsets 6 --depths 3,5,7,9 --loschmidt-K 3 \
    --num-blocks 4 --shots 2000 --simulator --output results/loschmidt_interleaved.csv
```
Use `--u-style local` (short-range) or `--u-style global` (long-range) to bias
the entangling pattern as a control.

Analyze interleaved data:
```bash
python analyze_loschmidt_interleaved.py --inputs results/loschmidt_interleaved.csv \
    --alpha-csv results/alpha_by_block.csv --corr-csv results/block_corr_summary.csv \
    --report results/interleaved_report.txt
```

Permutation histograms are written alongside the correlation summary
(`rho_perm_block{b}.png`). The report highlights whether the signature is
architecture-linked (consistent Spearman rho across blocks) or drift-like
(coherent block-to-block shifts).

### G) Spectral-gap sweep (Loschmidt echo on Braket)
Probe a “spectral gap / mixing-time crossover” by fixing the two-qubit budget
per layer and sweeping depths across matching-based architectures:
ring (even/odd ring matchings), grid (alternating horizontal/vertical
matchings on an approximately square layout), and expander (fresh random
perfect matchings). Each layer applies random single-qubit scrambles before the
2Q gates, then appends the exact inverse circuit so that drift/noise drive
deviations from \(P_\text{return}=1\).

Run a sweep (local simulator or ARN) and log JSONL results:
```bash
python spectral_gap_sweep_braket.py sweep --device local --n_qubits 8 \
    --depths 2,4,8,16,32 --n_seeds 3 --shots 500 \
    --output_dir results --families ring,grid,expander --interleave
```
Analyze and export CSV summaries (mean/stderr by depth plus a two-regime
log-decay crossover fit):
```bash
python spectral_gap_sweep_braket.py analyze --input results/spectral_gap_loschmidt.jsonl \
    --csv_out results/spectral_gap_loschmidt_summary.csv \
    --crossover_csv results/spectral_gap_loschmidt_crossover.csv
```
Self-test (local simulator, N=8, depths 2 and 4, 200 shots) that prints
\(P_\text{return}\) and writes JSONL:
```bash
python spectral_gap_sweep_braket.py sweep --device local --n_qubits 8 \
    --depths 2,4 --shots 200 --self_test --output_dir results
```

### H) Plateau diagnostic controls (Braket)
Add a shallow control suite to isolate SPAM vs 1Q vs first entangling-layer
effects and to verify that compiled circuits for ring/grid/expander remain
distinct after transpilation. Default depths are `0,1,2` (optionally include 4).

Run the plateau control sweep (local simulator or ARN):
```bash
python spectral_gap_sweep_braket.py plateau_diagnostic --device local --n_qubits 8 \
    --depths 0,1,2 --seeds 3 --shots 500 --output_dir results --families ring,grid,expander \
    --interleave --analyze_after
```
Analyze existing plateau JSONL output:
```bash
python spectral_gap_sweep_braket.py plateau_analyze --input results/plateau_diagnostic.jsonl \
    --csv_out results/plateau_summary.csv --delta_csv results/plateau_delta_summary.csv \
    --distinct_csv results/plateau_distinctness_summary.csv --depths 0,1,2
```
Self-test (local simulator) that writes JSONL plus CSV summaries and prints the
plateau table:
```bash
python spectral_gap_sweep_braket.py plateau_diagnostic --device local --n_qubits 8 \
    --shots 200 --self_test --output_dir results
```
Interpretation guide:
- **depth 0**: no gates, so \(P_\text{return}\) is a SPAM ceiling.
- **depth 1**: single-qubit scramble + inverse only; any drop reflects SPAM,
  compilation, or measurement artifacts.
- **depth 2**: first entangling layer (plus scrambles) and its inverse; the drop
  highlights two-qubit-induced plateau onset. (Depth 4 adds a second entangling
  layer.)
- **Distinctness table**: confirms whether compiled programs remain unique per
  family after transpilation (via compiled IR hash counts and compiled gate
  depth/2Q medians).

## What this tests
- **Connectivity metric:** \(C = N \lambda_2\) measures how well-connected the
  architecture is (\(\lambda_2\) is the spectral gap of the normalized
  Laplacian). Larger \(C\) corresponds to stronger expansion.
- **Decay fit:** \(\alpha_{\text{hat}}\) is obtained by fitting a line to
  \(-\log(\text{success\_prob})\) versus circuit depth, interpreting the slope
  as the per-depth decay rate.

## Reproducibility
- Install requirements: `pip install numpy scipy pandas matplotlib networkx qiskit qiskit-aer`.
- All random graph families use deterministic seeds derived from the device
  label and the `--seed` value (or `--graph_seed_override`), so repeated runs
  with the same inputs regenerate identical coupling graphs and analysis
  outputs.
