# Multi-Site Memory Kernel Recovery Pipeline

This repository is a research-oriented, modular Python pipeline for:

1. Loading noisy, discrete BTC and plume snapshot data.
2. Denoising and smoothing the data.
3. Estimating effective velocity-like and retention-like signals.
4. Recovering memory kernels from those signals.
5. Comparing velocity-derived and retention-derived kernels.
6. Fitting interpretable equations to recovered kernel curves.
7. Plotting all intermediate and final results.

## Important scientific note

This code provides a **careful computational scaffold** for your chapter idea. It does **not** claim that velocity and retention are uniquely observable from BTCs alone. Instead, it constructs **effective velocity-related and retention-related proxies** from smoothed transport data under explicit assumptions. That makes it suitable for exploratory research, benchmarking, synthetic tests, and multi-site comparison.

## Repository structure

- `run_pipeline.py`  
  Main entry point. This is the file you run.

- `example_generate_synthetic.py`  
  Creates synthetic noisy BTC and snapshot data for testing.

- `config.json`  
  Main configuration file.

- `src/io_utils.py`  
  Reading input CSV files and writing outputs.

- `src/preprocess.py`  
  Denoising, interpolation, smoothing, and signal cleanup.

- `src/inference.py`  
  Builds effective velocity-like and retention-like signals from smoothed data.

- `src/kernels.py`  
  Computes kernels from velocity and retention pathways.

- `src/symbolic_fit.py`  
  Fits interpretable candidate equations to recovered kernels.

- `src/plotting.py`  
  All plotting utilities.

- `src/pipeline.py`  
  Orchestrates the full workflow across sites.

- `examples/`  
  Example synthetic input data.

- `output/`  
  All generated figures and summaries.

## Input data format

### BTC CSV
Required columns:
- `site`
- `time`
- `concentration`

### Snapshot CSV
Required columns:
- `site`
- `time`
- `distance`
- `concentration`

## Quick start

### 1. Generate example data
```bash
python example_generate_synthetic.py
```

### 2. Run the full pipeline
```bash
python run_pipeline.py --config config.json
```

## Outputs

For each site, the code saves:
- raw vs denoised vs smoothed BTC plots
- raw vs denoised vs smoothed snapshot plots
- effective velocity-like signal plots
- effective retention-like signal plots
- velocity kernel plots
- retention kernel plots
- direct smoothed BTC kernel plots
- kernel comparison plots
- equation fit comparison plots
- a site summary JSON file

It also saves cross-site summaries for comparing fitted kernel families.

## Symbolic regression / equation fitting

The current code fits a library of interpretable kernel families such as:
- exponential
- stretched exponential
- power law
- tempered power law
- gamma-type kernel

It selects the best candidate using a weighted error + complexity score.

This is designed to be robust for GitHub use. If later you want, you can replace `src/symbolic_fit.py` with a PySR-based version.

## Recommended workflow for your chapter

1. Start with synthetic data.
2. Check whether velocity-derived or retention-derived kernels are more stable.
3. Apply to real sites.
4. Compare recovered parameter values across sites.
5. Test whether one kernel family explains all sites with site-specific parameters.

