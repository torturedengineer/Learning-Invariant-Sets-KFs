# Learning Invariant Sets using Hausdorff Metric-Based Kernel Flows

> A data-driven variational framework for learning invariant sets of dynamical systems using RKHS-based kernel flows and softened Hausdorff metrics.

---

## Overview

This repository implements the **Hausdorff Metric-Based Kernel Flows (HMKF)** framework for learning **invariant sets** of dynamical systems *directly from time-series data*, without requiring access to governing equations.

Unlike methods restricted to asymptotically stable attractors, this approach applies to **general invariant sets**, including:
- Saddle-type invariant sets
- Non-attracting structures
- Chaotic sets with complex geometry

The framework combines:
- A **variational formulation** of invariant sets using a softened Hausdorff metric
- **Kernel Flows** for adaptive kernel learning via cross-validation
- **Sparse kernel regularization** via ℓ₁ penalty

---

## Key Contributions

- Unified variational–kernel framework for learning invariant sets
- Extension of Hausdorff Metric-Based Kernel Flows beyond attractors
- Joint learning of dynamics and invariant geometry from data
- ℓ₁-regularized sparse kernel selection
- Theoretical guarantees under hyperbolicity assumptions
- Benchmark evaluation on **130 in-scope chaotic dynamical systems** (from the `dysts` library)

---

## Method Summary

Given time-series data from an unknown dynamical system:

1. **Learn dynamics in RKHS** using kernel ridge regression
2. **Represent invariant sets as finite point clouds**
3. **Optimize geometric invariance** via a softened Hausdorff distance
4. **Select kernels** using a cross-validation principle:
   > A good kernel should reconstruct consistent invariant sets from full and half data

---

## Repository Structure

```text
.
├── main_run.ipynb          # Main benchmark — training + evaluation on 130 systems
├── all_figures.ipynb       # Figure generation — all plots for the paper
├── figures/                # Pre-generated figures (PDF)
│   ├── hd_distribution.pdf
│   ├── gt_recon_success.pdf
│   ├── gt_recon_failure.pdf
│   ├── full_benchmark_mosaic.pdf
│   └── active_kernels.pdf
├── results/
│   └── results.csv         # Benchmark results (HD, status, active kernels, runtime)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/torturedengineer/Learning-Invariant-Sets-KFs.git
cd Learning-Invariant-Sets-KFs
pip install -r requirements.txt
```

---

## Usage

Both notebooks are designed to run in **Google Colab** (free or Pro tier, GPU recommended). They can also be run locally — just set the path variables at the top of each notebook to point to your local directories.

### 1. Run the benchmark

Open `main_run.ipynb`. At the top of the notebook, set:

```python
PROJECT_ROOT = "."          # or your local path
RESULTS_DIR  = "results"
DATA_DIR     = "results/trajectories"
```

Then run all cells. The notebook will:
- Iterate over all 130 in-scope `dysts` systems
- Train HMKF on each (≈2–3 min/system on GPU)
- Save results to `results/results.csv` and trajectory NPZs to `results/trajectories/`
- Resume automatically if interrupted

### 2. Generate figures

Open `all_figures.ipynb`. Set:

```python
NPZ_DIR     = "results/trajectories"
RESULTS_CSV = "results/results.csv"
OUT_DIR     = "figures"
```

Then run all cells to reproduce all paper figures.
Full per-system figures: https://drive.google.com/drive/folders/1n5m-MaSOrycq7-SxbfJw4GpEz81K1XgS?usp=drive_link

---

## Benchmark Summary

Evaluated on **130 in-scope systems** from the [`dysts`](https://github.com/williamgilpin/dysts) library (5 delay/discontinuous systems excluded as out-of-scope):

| Threshold | Systems | Fraction |
|-----------|---------|----------|
| HD < 0.1  | 49      | 38%      |
| HD < 1.0  | 79      | 61%      |
| HD < 5.0  | 98      | 75%      |

Median HD: **0.323** (IQR: 0.045–5.070) · Median runtime: **~138s/system**

---

## Reproducibility

- Results are saved incrementally; the benchmark resumes from where it left off if interrupted
- Regularization (`REG`) is selected automatically per system via a condition-number proxy — no manual tuning
- Fixed hyperparameters across all systems: `λ_ℓ1 = 0.02`, `repulsion_mu = 0.05`, `delta = 0.4`, `N_ALT = 3`
- NPZ trajectory files are not committed (large); `results.csv` is committed and sufficient to reproduce all figures

---

## Associated Paper

**Hausdorff Metric-Based Kernel Flows for Learning Invariant Sets in Dynamical Systems**  
Juee Jahagirdar, Boumediene Hamzi, Houman Owhadi, Yannis Kevrekidis

Manuscript under review (2025)

---

## License

This repository is released under the MIT License.

---

## Contact

For questions related to the method, experiments, or implementation, please open a GitHub Issue or contact the authors listed in the associated paper.
