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
- **Sparse kernel regularization** for efficiency and interpretability

---

## Key Contributions

- Unified variational–kernel framework for learning invariant sets  
- Extension of Hausdorff Metric-Based Kernel Flows beyond attractors  
- Joint learning of dynamics and invariant geometry from data  
- ℓ1-regularized sparse kernel selection  
- Theoretical guarantees under hyperbolicity assumptions  
- Extensive validation on **135 chaotic dynamical systems**

---

## Method Summary

Given time-series data from an unknown dynamical system:

1. **Learn dynamics in RKHS** using kernel regression  
2. **Represent invariant sets as finite point clouds**
3. **Optimize geometric invariance** via a softened Hausdorff distance  
4. **Select kernels** using a cross-validation principle:
   > A good kernel should reconstruct consistent invariant sets from full and half data

---

## Repository Structure

```text
.
├── src/
│   ├── __init__.py
│   ├── metrics.py          # Evaluation metrics (Hausdorff, errors, statistics)
│   ├── unified_model.py    # Core HMKF model implementation
│   ├── utils.py            # Shared utilities
├── compare.py              # Model and ablation comparisons
├── conclusions.py          # Aggregated conclusions / summary generation
├── generate_stats.py       # Compute quantitative statistics from runs
├── parallel_ablation.py    # Parallelized ablation experiments
├── plot_ablation.py        # Plotting utilities for ablation studies
├── plotter.py              # General plotting utilities
├── setup_project.py        # Environment and data setup helper
├── requirements.txt
└── README.md
```

## Installation

```
git clone https://github.com/torturedengineer/Learning-Invariant-Sets-KFs.git
cd Learning-Invariant-Sets-KFs
pip install -r requirements.txt
```
### (Optional)
```
python -m venv venv
source venv/bin/activate
```
## Run the main analysis / experiments (examples):
```
# run a parallelized ablation study (if you have multiple cores / cluster)
python parallel_ablation.py
```
```
# generate evaluation statistics (after experiments have run)
python generate_stats.py
```
```
# produce plots for ablation and comparison results
python plot_ablation.py
python plotter.py
```
```
# compare model variants / configurations
python compare.py
```
## Typical workflow

1. Prepare data / trajectories (see `setup_project.py`).
2. Run `parallel_ablation.py` to perform model training and invariant-set reconstruction.
3. Produce metrics with `generate_stats.py`.
4. Visualize results using `plotter.py` and `plot_ablation.py`.
5. For large-scale experiments, use `parallel_ablation.py` to distribute runs.

---

## Results Summary

The proposed HMKF framework was evaluated on a benchmark of **135 chaotic dynamical systems**, covering a wide range of attractor geometries and dynamical regimes.

Key empirical observations:
- Accurate recovery of invariant sets from finite, noisy time-series data
- Improved robustness over isotropic kernel baselines
- Significant suppression of geometric reconstruction outliers
- Sparse kernel selection improves interpretability without degrading accuracy

Quantitative results and plots are generated via:
- `generate_stats.py`
- `plot_ablation.py`
- `plotter.py`

See the accompanying paper for full experimental details and analysis.

---

## Reproducibility

All experiments are script-driven and can be reproduced using the provided Python files.

- Fixed random seeds are used where applicable
- Results are deterministic up to numerical precision
- Parallel execution is supported via `parallel_ablation.py`

Users are encouraged to inspect individual scripts for experiment-specific parameters and dataset paths.

---

## Associated Paper

**Hausdorff Metric-Based Kernel Flows for Learning Invariant Sets in Dynamical Systems**  
Juee Jahagirdar, Boumediene Hamzi, Houman Owhadi, Yannis Kevrekidis  

Manuscript in preparation (2025)

---

## License

This repository is released under the MIT License. 

---

## Contact

For questions related to the method, experiments, or implementation, please use GitHub Issues or contact the authors listed in the associated paper.
