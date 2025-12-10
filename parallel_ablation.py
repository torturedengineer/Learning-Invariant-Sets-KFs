import os

# --- CRITICAL: PREVENT JAX CORE THRASHING ---
# We are running multiple processes. We want each process to be single-threaded
# so the operating system can schedule them efficiently.
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

import numpy as np
import pandas as pd
import dysts.flows
import json
import jax
import inspect
import gc
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- 1. IMPORTS & SETUP ---
from src.unified_model import SparseKernelFlowsModel
    
from src.utils import standardize_data, get_derivatives, rk4_integrate
from src.metrics import compute_hausdorff_distance, compute_modified_hausdorff_distance, compute_lyapunov_exponent

TRAIN_PTS, TEST_PTS = 1500, 2000
jax.config.update("jax_enable_x64", True)

# --- 2. ROBUST SYSTEM FINDER ---
def get_system_names():
    if hasattr(dysts.flows, 'get_attractor_list'):
        try: return sorted(dysts.flows.get_attractor_list())
        except: pass
    names = []
    for name, obj in inspect.getmembers(dysts.flows):
        if inspect.isclass(obj) and name != "DynSys" and hasattr(obj, "make_trajectory"):
            names.append(name)
    return sorted(names)

# --- 3. WORKER FUNCTION (Runs in parallel) ---
def process_single_system(args):
    """
    Independent worker function. 
    Returns: (Result Dictionary) or (None) if failed.
    """
    name, exp_name, config, results_dir = args
    
    try:
        # A. Data Generation
        # Re-import dysts inside worker to ensure clean state if needed
        import dysts.flows 
        
        model_cls = getattr(dysts.flows, name, None)
        if not model_cls: return None
        
        model_obj = model_cls()
        t, sol = model_obj.make_trajectory(n=TRAIN_PTS+TEST_PTS, resample=True, return_times=True)
        
        if np.any(np.isnan(sol)): return None
        dt = model_obj.dt or (t[1]-t[0])

        data_std, mean, std = standardize_data(sol)
        X_train, Y_train = data_std[:TRAIN_PTS], get_derivatives(data_std, dt)[:TRAIN_PTS]
        
        # B. Model Fit
        kf = SparseKernelFlowsModel(
            regularization=1e-4, 
            mhd_weight=config['mhd'],       
            l1_weight=config['l1'],         
            is_anisotropic=config['aniso']
        )
        alphas, scales = kf.fit(X_train, Y_train, dt)
        
        # C. Reconstruction
        recon_std = rk4_integrate(lambda x: kf.predict(x), X_train[-1], dt, TEST_PTS)
        recon = (recon_std * std) + mean
        truth = sol[TRAIN_PTS:]
        
        # D. Metrics
        hd = compute_hausdorff_distance(truth, recon)
        mhd = compute_modified_hausdorff_distance(truth, recon)
        pred_mle = compute_lyapunov_exponent(recon, dt)
        true_mle = np.max(getattr(model_obj, 'lyapunov_exponents', [np.nan]))
        
        # Sparsity Ratio
        active_count = sum(1 for a in alphas if a > 1e-3)
        sparsity_ratio = 1.0 - (active_count / len(alphas))
        
        # E. Save NPZ (Safe in workers as filenames are unique)
        data_subdir = os.path.join(results_dir, "data")
        np.savez_compressed(
            f"{data_subdir}/{name}_data.npz", 
            truth=truth, recon=recon, alphas=alphas, 
            sigmas=kf.base_sigmas, scales=scales
        )
        
        # Clean Memory aggressively
        del kf, model_obj, sol, X_train, Y_train
        gc.collect()
        jax.clear_caches()
        
        return {
            "System": name, "Experiment": exp_name,
            "Standard_HD": hd, "Modified_HD": mhd, 
            "MLE_Error": abs(true_mle - pred_mle) if pred_mle else None,
            "Sparsity": sparsity_ratio
        }

    except Exception as e:
        # Fail silently but print error
        print(f"[{name}] Failed: {e}")
        return None

# --- 4. EXPERIMENT RUNNER ---
def run_single_experiment(exp_name, config):
    print(f"\n>>> STARTING PARALLEL EXPERIMENT: {exp_name}")
    print(f"    Settings: {config}")
    
    RESULTS_DIR = f"results/{exp_name}"
    DATA_DIR = os.path.join(RESULTS_DIR, "data")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    csv_path = f"{RESULTS_DIR}/benchmark_results.csv"
    
    # Check for existing work
    done_systems = set()
    results_log = []
    if os.path.exists(csv_path):
        try:
            df_existing = pd.read_csv(csv_path)
            done_systems = set(df_existing['System'].unique())
            results_log = df_existing.to_dict('records')
            print(f"    Resuming... {len(done_systems)} already done.")
        except: pass

    all_names = get_system_names()
    # Filter tasks
    tasks = [
        (name, exp_name, config, RESULTS_DIR) 
        for name in all_names 
        if name not in done_systems
    ]
    
    if not tasks:
        print("    All systems completed for this experiment.")
        return

    # --- PARALLEL EXECUTION ---
    # Use ~80% of available CPU cores
    num_workers = max(1, int(os.cpu_count() * 0.80))
    print(f"    Spinning up {num_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(process_single_system, t) for t in tasks]
        
        # Process as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{exp_name} (Parallel)"):
            res = future.result()
            if res is not None:
                results_log.append(res)
                
                # INCREMENTAL SAVE (Main process handles this, so it's safe)
                # We save every valid result so we don't lose progress if crashed
                pd.DataFrame(results_log).to_csv(csv_path, index=False)

if __name__ == "__main__":
    # --- THE 4-STEP ABLATION STRATEGY ---
    experiments = [
        # 1. Baseline: Isotropic, Pure Dynamics (No MHD), Dense
        {
            "name": "1_Baseline_Iso_Dyn",
            "config": {"aniso": False, "l1": 0.0, "mhd": 0.0}
        },
        # 2. Add Anisotropy: Aniso, Pure Dynamics (No MHD), Dense
        {
            "name": "2_Aniso_Dyn",
            "config": {"aniso": True, "l1": 0.0, "mhd": 0.0}
        },
        # 3. Add Geometry: Aniso, MHD Loss, Dense
        {
            "name": "3_Aniso_MHD",
            "config": {"aniso": True, "l1": 0.0, "mhd": 0.1}
        },
        # 4. Add Sparsity (Final): Aniso, MHD Loss, Sparse
        {
            "name": "4_Final_Sparse",
            "config": {"aniso": True, "l1": 1e-3, "mhd": 0.1}
        }
    ]

    print("--- STARTING PARALLEL ABLATION STUDY ---")
    for exp in experiments:
        run_single_experiment(exp['name'], exp['config'])