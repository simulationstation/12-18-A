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
