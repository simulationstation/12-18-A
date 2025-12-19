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
