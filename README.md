# Multi-Site BTC Decomposition and Memory Kernel Recovery

This repository provides a **modular research pipeline** for analyzing breakthrough curve (BTC) data and plume snapshots using a **new inverse decomposition framework**:

\[
\text{BTC}(t) \approx g(t) * h(t)
\]

where:

- \(g(t)\) = **advective (velocity) kernel**
- \(h(t)\) = **retention (memory) kernel**

---

#  Key Idea (What makes this different)

Instead of directly extracting velocity and retention using predefined transforms, this framework:

1. **Denoises and smooths BTC and snapshot data**
2. **Decomposes BTC into two physically meaningful components**:
   - advective transport \(g(t)\)
   - retention/delay \(h(t)\)
3. **Reconstructs BTC using convolution**:
   \[
   f(t) \approx g(t) * h(t)
   \]
4. **Fits interpretable equations** to both \(g(t)\) and \(h(t)\)
5. Compares results across multiple sites

---

#  Scientific Note (Important)

This framework **does NOT assume unique identifiability** of velocity and retention from BTC alone.

Instead, it:

- infers **effective kernels**
- uses **regularization and normalization constraints**
- provides a **data-driven but physically interpretable decomposition**

This makes it suitable for:
- exploratory research
- synthetic validation
- multi-site comparison
- hypothesis testing

---

#  Repository Structure

- `run_pipeline.py`  
  Main entry point.

- `example_generate_synthetic.py`  
  Generates synthetic BTC and snapshot data.

- `config.json`  
  Controls preprocessing, inference, and kernel settings.

---

## Core Modules (`src/`)

- `io_utils.py`  
  Load/save data and outputs.

- `preprocess.py`  
  Denoising and smoothing:
  - Random Forest denoising
  - spline interpolation
  - Savitzky–Golay filtering
  - Gaussian smoothing

- `inference.py`  
  Core decomposition:
  - estimates **advective kernel \(g(t)\)**
  - estimates **retention kernel \(h(t)\)** via deconvolution

- `kernels.py`  
  - regularizes kernels  
  - normalizes kernels  
  - reconstructs BTC via convolution  
  - computes reconstruction error  

- `symbolic_fit.py`  
  Fits interpretable equations to:
  - \(g(t)\)
  - \(h(t)\)

- `plotting.py`  
  Generates:
  - BTC preprocessing plots  
  - kernel plots  
  - BTC reconstruction plots  
  - equation fits  

- `pipeline.py`  
  Runs the full workflow across all sites

---

#  Input Data Format

## BTC CSV

Required columns:
- `site`
- `time`
- `concentration`

## Snapshot CSV

Required columns:
- `site`
- `time`
- `distance`
- `concentration`

---

#  Quick Start

## 1. Generate synthetic data

```bash
python example_generate_synthetic.py
