import os

# Define the file structure and content
files = {
    # --- CONFIGURATION ---
    "requirements.txt": """annotated-types==0.7.0
colorama==0.4.6
contourpy==1.3.3
cycler==0.12.1
dysts==0.96
fonttools==4.61.0
future==1.0.0
gluonts==0.16.2
jax==0.8.1
jaxlib==0.8.1
jaxopt==0.8.5
Jinja2==3.1.6
joblib==1.5.2
kiwisolver==1.4.9
llvmlite==0.45.1
MarkupSafe==3.0.3
matplotlib==3.10.7
ml_dtypes==0.5.4
nolds==0.6.3
numba==0.62.1
numpy==2.1.3
opt_einsum==3.4.0
packaging==25.0
pandas==2.3.3
pillow==12.0.0
pyarrow==22.0.0
pydantic==2.12.5
pydantic_core==2.41.5
pyparsing==3.2.5
python-dateutil==2.9.0.post0
pytz==2025.2
scikit-learn==1.7.2
scipy==1.16.3
seaborn==0.13.2
setuptools==80.9.0
six==1.17.0
threadpoolctl==3.6.0
toolz==0.12.1
tqdm==4.67.1
typing-inspection==0.4.2
typing_extensions==4.15.0
tzdata==2025.2""",

    # --- SOURCE CODE ---
    "src/__init__.py": "",

    "src/utils.py": """import numpy as np
import jax.numpy as jnp

def standardize_data(data):
    \"\"\"
    Standardizes data to mean=0, std=1.
    Returns standardized data, mean, and std for inverse transformation.
    \"\"\"
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0) + 1e-6  # Add epsilon to prevent divide by zero
    standardized_data = (data - mean) / std
    return standardized_data, mean, std

def get_derivatives(data, dt):
    \"\"\"
    Approximates time derivatives (velocities) from the trajectory.
    v(t) = (x(t+1) - x(t)) / dt
    \"\"\"
    return (data[1:] - data[:-1]) / dt

def rk4_integrate(model_predict_fn, x0, dt, steps):
    \"\"\"
    4th-Order Runge-Kutta Integrator for the Learned Model.
    
    Args:
        model_predict_fn: A function f(x) that returns velocity (dx/dt).
        x0: Initial condition (numpy array).
        dt: Time step.
        steps: Number of integration steps.
        
    Returns:
        trajectory: (steps, D) numpy array.
    \"\"\"
    D = len(x0)
    trajectory = np.zeros((steps, D))
    trajectory[0] = x0
    x = x0

    # Ensure x is a JAX-compatible array if the model expects it
    for i in range(1, steps):
        # RK4 steps
        k1 = model_predict_fn(x)
        k2 = model_predict_fn(x + 0.5 * dt * k1)
        k3 = model_predict_fn(x + 0.5 * dt * k2)
        k4 = model_predict_fn(x + dt * k3)
        
        # Update
        x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        trajectory[i] = x
        
    return trajectory
""",

    "src/metrics.py": """import numpy as np
import nolds
from scipy.spatial.distance import cdist

def compute_hausdorff_distance(traj_a, traj_b):
    \"\"\"
    Standard Hausdorff Distance (Max-Min).
    Use this for rigorous 'worst-case' geometric evaluation.
    \"\"\"
    if len(traj_a) > 5000:
        traj_a = traj_a[::2]
        traj_b = traj_b[::2]
        
    d_matrix = cdist(traj_a, traj_b, metric='euclidean')
    
    d_ab = np.max(np.min(d_matrix, axis=1))
    d_ba = np.max(np.min(d_matrix, axis=0))
    
    return max(d_ab, d_ba)

def compute_modified_hausdorff_distance(traj_a, traj_b):
    \"\"\"
    Modified Hausdorff Distance (MHD) - Average of Minimums.
    As defined in Eq 6 & 7 of the paper.
    d_MH(A, B) = 0.5 * ( mean(min_b ||a-b||) + mean(min_a ||b-a||) )
    
    This metric is much less sensitive to outliers and matches the training loss.
    \"\"\"
    if len(traj_a) > 5000:
        traj_a = traj_a[::2]
        traj_b = traj_b[::2]
        
    # Compute Euclidean distance matrix
    d_matrix = cdist(traj_a, traj_b, metric='euclidean')
    
    # Mean of minimum distances (A -> B)
    # Note: The paper uses squared norm for optimization, 
    # but usually reports Euclidean distance for metric interpretation.
    # We use Euclidean here to be comparable to spatial units.
    term_1 = np.mean(np.min(d_matrix, axis=1))
    
    # Mean of minimum distances (B -> A)
    term_2 = np.mean(np.min(d_matrix, axis=0))
    
    return 0.5 * (term_1 + term_2)

def compute_lyapunov_exponent(trajectory, dt):
    \"\"\"
    Estimates the Maximum Lyapunov Exponent (MLE).
    \"\"\"
    data_1d = trajectory[:, 0]
    
    try:
        # lyap_r estimates largest LE
        mle = nolds.lyap_r(data_1d, emb_dim=3, min_tsep=None, tau=1)
        return mle / dt
    except Exception as e:
        return None
""",

    "src/unified_model.py": """import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy.linalg import solve
import jax.scipy.optimize
import numpy as np
import logging

# Try to import JAXOPT for L-BFGS (matches paper memory efficiency)
try:
    import jaxopt
    HAS_JAXOPT = True
except ImportError:
    HAS_JAXOPT = False
    logging.warning("JAXOPT not found. Falling back to standard BFGS. Install jaxopt for L-BFGS.")

# --- 1. Universal Kernel (Handles Both Cases) ---

@jit
def universal_kernel(x1, x2, alphas, base_sigmas, log_scales):
    \"\"\"
    Computes the sparse kernel sum.
    Note: alphas are squared (alphas[i]**2) to ensure positive semi-definiteness.
    \"\"\"
    # A. Apply Anisotropy
    scales = jnp.exp(log_scales)
    x1_scaled = x1 / scales[None, :]
    x2_scaled = x2 / scales[None, :]
    
    # B. Squared Euclidean Distance
    diff = x1_scaled[:, None, :] - x2_scaled[None, :, :]
    dist_sq = jnp.sum(diff**2, axis=-1)
    
    # C. Sparse Dictionary Sum
    K_sum = jnp.zeros_like(dist_sq)
    for i in range(len(base_sigmas)):
        K_i = jnp.exp(-0.5 * dist_sq / (base_sigmas[i]**2))
        K_sum = K_sum + (alphas[i]**2) * K_i
        
    return K_sum

# --- 2. Modified Hausdorff Loss ---
@jit
def modified_hausdorff_loss(set_a, set_b):
    diff = set_a[:, None, :] - set_b[None, :, :]
    dists_sq = jnp.sum(diff**2, axis=-1)
    return 0.5 * (jnp.mean(jnp.min(dists_sq, axis=1)) + jnp.mean(jnp.min(dists_sq, axis=0)))

# --- 3. The Hybrid Model Class ---

class SparseKernelFlowsModel:
    def __init__(self, regularization=1e-4, mhd_weight=0.1, l1_weight=1e-3, is_anisotropic=True):
        self.reg = regularization
        self.mhd_weight = mhd_weight 
        self.l1_weight = l1_weight
        self.is_anisotropic = is_anisotropic
        self.base_sigmas = jnp.logspace(-1, 2, 10)
        
        self.params = {
            "alphas": None, "log_scales": None, "weights": None, "X_train": None
        }
        
    def fit(self, X, Y, dt=0.01):
        # Data Shuffling
        X_np, Y_np = np.array(X), np.array(Y)
        perm = np.random.permutation(len(X_np))
        X = jnp.array(X_np[perm])
        Y = jnp.array(Y_np[perm])

        N, D = X.shape
        n_kernels = len(self.base_sigmas)
        
        # Init parameters
        init_alphas = jnp.ones(n_kernels) / n_kernels
        init_log_scales = jnp.zeros(D)
        
        if self.is_anisotropic:
            x0 = jnp.concatenate([init_alphas, init_log_scales])
        else:
            x0 = init_alphas
            
        # Split Data
        split = int(N * 0.5)
        X_tr, Y_tr = X[:split], Y[:split]
        X_val, Y_val = X[split:], Y[split:]
        sigmas = self.base_sigmas

        # --- Objective Function ---
        def objective(params_flat):
            if self.is_anisotropic:
                alphas = params_flat[:n_kernels]
                scales = params_flat[n_kernels:]
            else:
                alphas = params_flat
                scales = jnp.zeros(D)

            # Training
            K_train = universal_kernel(X_tr, X_tr, alphas, sigmas, scales)
            w_sub = solve(K_train + self.reg * jnp.eye(len(X_tr)), Y_tr, assume_a='pos')
            
            # Validation
            K_cross = universal_kernel(X_val, X_tr, alphas, sigmas, scales)
            Y_pred = jnp.dot(K_cross, w_sub)
            
            # Loss
            mse = jnp.mean((Y_val - Y_pred)**2)
            loss_dyn = mse / (jnp.mean(Y_val**2) + 1e-6)
            loss_mhd = modified_hausdorff_loss(X_val, X_val + Y_pred * dt)
            loss_sparse = jnp.sum(jnp.abs(alphas**2))
            
            return (1.0 - self.mhd_weight) * loss_dyn + self.mhd_weight * loss_mhd + self.l1_weight * loss_sparse

        # --- Optimization Step ---
        if HAS_JAXOPT:
            # FIX: Use LBFGS (Unconstrained) instead of LBFGSB (Bounded)
            # This avoids the "missing bounds" error while maintaining L-BFGS efficiency
            solver = jaxopt.LBFGS(fun=objective, maxiter=100)
            res = solver.run(x0)
            best_params = res.params
        else:
            res = jax.scipy.optimize.minimize(fun=objective, x0=x0, method='BFGS', options={'maxiter': 100})
            best_params = res.x

        # Unpack & Final Fit
        if self.is_anisotropic:
            best_alphas = best_params[:n_kernels]
            best_scales = best_params[n_kernels:]
        else:
            best_alphas = best_params
            best_scales = jnp.zeros(D)

        final_K = universal_kernel(X, X, best_alphas, sigmas, best_scales)
        self.params["weights"] = solve(final_K + self.reg * jnp.eye(N), Y, assume_a='pos')
        self.params["alphas"] = best_alphas
        self.params["log_scales"] = best_scales
        self.params["X_train"] = X
        
        return best_alphas, jnp.exp(best_scales)

    def predict(self, x_new):
        if x_new.ndim == 1: x_new = x_new[None, :]
        K_vec = universal_kernel(x_new, self.params["X_train"], self.params["alphas"], 
                                 self.base_sigmas, self.params["log_scales"])
        return np.array(jnp.dot(K_vec, self.params["weights"])[0])
""",

    # --- EXPERIMENT SCRIPTS (Main Runner) ---
    "parallel_ablation.py": """import os

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
    \"\"\"
    Independent worker function. 
    Returns: (Result Dictionary) or (None) if failed.
    \"\"\"
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
    print(f"\\n>>> STARTING PARALLEL EXPERIMENT: {exp_name}")
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
""",

    # --- ANALYSIS & PLOTTING SCRIPTS ---
    "compare.py": """import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import dysts.flows
from src.unified_model import universal_kernel
from src.utils import standardize_data, get_derivatives, rk4_integrate
from jax import jit
import jax.numpy as jnp
from jax.scipy.linalg import solve

# --- CONFIGURATION ---
RESULTS_DIR = "results"
OUTPUT_DIR = "paper_plots/comparisons"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# The 5 "best" Systems to visualize
SYSTEMS = [
    "YuWang2",              # The Massive Divergence Fix
    "WindmiReduced",        # The Hyperchaotic Stabilizer
    "LidDrivenCavityFlow",  # The High-Dim Win
    "Lorenz",               # The Classic (likely shows trapping)
    "Rossler"               # The Standard
]

TRAIN_PTS = 1500
TEST_PTS = 2000
REGULARIZATION = 1e-4

# --- HELPER: Manual Prediction (Since we are skipping the Optimizer) ---
@jit
def get_weights(X, Y, alphas, sigmas, scales, reg):
    K = universal_kernel(X, X, alphas, sigmas, scales)
    return solve(K + reg * jnp.eye(len(X)), Y, assume_a='pos')

@jit
def predict_step(x_new, X_train, weights, alphas, sigmas, scales):
    if x_new.ndim == 1: x_new = x_new[None, :]
    K_vec = universal_kernel(x_new, X_train, alphas, sigmas, scales)
    return jnp.dot(K_vec, weights)[0]

def load_params(exp_folder, system_name):
    \"\"\"Loads learned kernel parameters from a specific experiment.\"\"\"
    path = f"{RESULTS_DIR}/{exp_folder}/data/{system_name}_data.npz"
    try:
        data = np.load(path)
        # Handle log-scales vs linear scales storage differences if any
        saved_scales = data['scales']
        log_scales = np.log(saved_scales + 1e-9) 
        
        return data['alphas'], data['sigmas'], log_scales
    except FileNotFoundError:
        print(f"  Missing data for {system_name} in {exp_folder}")
        return None

def run_comparison(system_name):
    print(f"Processing {system_name}...")
    
    # 1. Generate FRESH Ground Truth (Fixed Seed implicitly by restart)
    try:
        model_obj = getattr(dysts.flows, system_name)()
        t, sol = model_obj.make_trajectory(n=TRAIN_PTS+TEST_PTS, resample=True, return_times=True)
        dt = model_obj.dt or (t[1]-t[0])
    except:
        print("  System generation failed.")
        return

    # Standardize
    data_std, mean, std = standardize_data(sol)
    X_train, Y_train = data_std[:TRAIN_PTS], get_derivatives(data_std, dt)[:TRAIN_PTS]
    X_train_jax = jnp.array(X_train)
    Y_train_jax = jnp.array(Y_train)
    
    start_pt = X_train[-1]

    # 2. Load & Run Baseline (Exp 1)
    params_1 = load_params("1_Baseline_Iso_Dyn", system_name)
    if params_1:
        a1, s1, sc1 = params_1
        w1 = get_weights(X_train_jax, Y_train_jax, a1, s1, sc1, REGULARIZATION)
        pred_func_1 = lambda x: np.array(predict_step(x, X_train_jax, w1, a1, s1, sc1))
        recon_1_std = rk4_integrate(pred_func_1, start_pt, dt, TEST_PTS)
        recon_1 = (recon_1_std * std) + mean
    else:
        recon_1 = None

    # 3. Load & Run Sparse HMKF (Exp 4)
    params_4 = load_params("4_Final_Sparse", system_name)
    if params_4:
        a4, s4, sc4 = params_4
        w4 = get_weights(X_train_jax, Y_train_jax, a4, s4, sc4, REGULARIZATION)
        pred_func_4 = lambda x: np.array(predict_step(x, X_train_jax, w4, a4, s4, sc4))
        recon_4_std = rk4_integrate(pred_func_4, start_pt, dt, TEST_PTS)
        recon_4 = (recon_4_std * std) + mean
    else:
        recon_4 = None

    # 4. Plotting
    ground_truth = sol[TRAIN_PTS:]
    
    fig = plt.figure(figsize=(14, 6))
    
    # --- Subplot 1: Baseline (The "Off the Charts" Fix) ---
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Lock axes to Ground Truth limits to show divergence clearly
    ax1.set_xlim(ground_truth[:,0].min(), ground_truth[:,0].max())
    ax1.set_ylim(ground_truth[:,1].min(), ground_truth[:,1].max())
    ax1.set_zlim(ground_truth[:,2].min(), ground_truth[:,2].max())
    
    ax1.plot(ground_truth[:,0], ground_truth[:,1], ground_truth[:,2], c='k', alpha=0.2, lw=0.5, label='Truth')
    
    if recon_1 is not None:
        # Check if it exploded (for labeling purposes only)
        # We consider "diverged" if it goes 2x beyond the bounds of truth
        divergence_mask = np.any(np.abs(recon_1) > 2 * np.max(np.abs(ground_truth)), axis=1)
        
        label_text = 'Baseline'
        title_text = f"{system_name}: Baseline"
        title_color = 'black'

        if np.any(divergence_mask):
            first_divergence = np.argmax(divergence_mask)
            label_text = f'Baseline (Diverged t={first_divergence})'
            title_text = f"{system_name}: Baseline (DIVERGED)"
            title_color = 'red'

        ax1.set_title(title_text, fontweight='bold', color=title_color)
        
        # Plot full trajectory - let Matplotlib clip the lines outside the box
        # This explicitly shows the line shooting off to infinity if unstable
        ax1.plot(recon_1[:,0], recon_1[:,1], recon_1[:,2], c='blue', lw=1, alpha=0.8, label=label_text)

    ax1.legend()
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")

    # --- Subplot 2: Sparse HMKF ---
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(ground_truth[:,0], ground_truth[:,1], ground_truth[:,2], c='k', alpha=0.2, lw=0.5, label='Truth')
    if recon_4 is not None:
        ax2.plot(recon_4[:,0], recon_4[:,1], recon_4[:,2], c="#791E26", lw=1.2, alpha=0.9, label='Sparse HMKF')
        ax2.set_title(f"{system_name}: Sparse HMKF", fontweight='bold')
    
    ax2.legend()
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{system_name}_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved plot to {OUTPUT_DIR}/{system_name}_comparison.png")

if __name__ == "__main__":
    print("Generating Head-to-Head Comparisons...")
    for sys in SYSTEMS:
        run_comparison(sys)
""",

    "plotter.py": """import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Registers 3D projection
import seaborn as sns

# --- CONFIG ---
DATA_DIR = "results/sparse_anisotropic_mhd_data"
PLOT_DIR = "results/plots_expt4"
os.makedirs(PLOT_DIR, exist_ok=True)

# Aesthetics (Paper Ready)
sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")
COLOR_TRUTH = "#222222"   # Dark Gray/Black
COLOR_RECON = "#E63946"   # Red
COLOR_BARS = "#1D3557"    # Navy Blue

def plot_dashboard(fpath):
    # 1. Load Data
    try:
        data = np.load(fpath)
        truth = data['truth']
        recon = data['recon']
        alphas = data['alphas']
        sigmas = data['sigmas']
        
        name = os.path.basename(fpath).replace("_data.npz", "")
        dim = truth.shape[1]
        
        # 2. Setup Figure
        fig = plt.figure(figsize=(14, 6))
        
        # --- LEFT PANEL: DYNAMICS (2D or 3D) ---
        if dim >= 3:
            ax1 = fig.add_subplot(121, projection='3d')
            # Plot Ground Truth
            ax1.plot(truth[:, 0], truth[:, 1], truth[:, 2], 
                     c=COLOR_TRUTH, lw=0.6, alpha=0.4, label="Ground Truth")
            # Plot Reconstruction
            ax1.plot(recon[:, 0], recon[:, 1], recon[:, 2], 
                     c=COLOR_RECON, lw=1.2, ls='--', alpha=0.9, label="Sparse KF")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_zlabel("Z")
            # Set initial view for better 3D perspective
            ax1.view_init(elev=30, azim=45)
        else:
            ax1 = fig.add_subplot(121)
            ax1.plot(truth[:, 0], truth[:, 1], 
                     c=COLOR_TRUTH, lw=0.8, alpha=0.6, label="Ground Truth")
            ax1.plot(recon[:, 0], recon[:, 1], 
                     c=COLOR_RECON, lw=1.2, ls='--', label="Sparse KF")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")

        ax1.set_title(f"Attractor Reconstruction: {name}", fontweight="bold")
        ax1.legend()

        # --- RIGHT PANEL: SPARSE KERNEL WEIGHTS (The "Proof") ---
        ax2 = fig.add_subplot(122)
        
        # Filter out tiny weights to make the plot clean
        mask = alphas > 1e-3
        active_sigmas = sigmas[mask]
        active_alphas = alphas[mask]
        
        if len(active_alphas) > 0:
            x_pos = np.arange(len(active_alphas))
            ax2.bar(x_pos, active_alphas, color=COLOR_BARS, alpha=0.85, width=0.6)
            
            # Label x-axis with Sigma values
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([f"{s:.2f}" for s in active_sigmas], rotation=45)
            ax2.set_xlabel("Kernel Lengthscale ($\sigma$)")
            ax2.set_ylabel("Learned Weight ($\\alpha$)")
            ax2.set_title("Sparse Kernel Selection (Multi-Scale)", fontweight="bold")
            
            # Add value labels on top of bars
            for i, v in enumerate(active_alphas):
                ax2.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=9)
        else:
            ax2.text(0.5, 0.5, "No Active Kernels\\n(Convergence Issue)", 
                     ha='center', va='center', transform=ax2.transAxes)

        plt.tight_layout()
        
        # Save
        out_path = f"{PLOT_DIR}/{name}_dashboard.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Generated Dashboard: {out_path}")
        
    except Exception as e:
        print(f"Failed to plot {fpath}: {e}")

def run_plotting():
    # Find all .npz files from the benchmark run
    files = sorted(glob.glob(f"{DATA_DIR}/*.npz"))
    
    if not files:
        print("No data found! Run 'main_benchmark.py' first.")
        return

    print(f"Found {len(files)} systems. Generating 3D Dashboards...")
    for f in files:
        plot_dashboard(f)
    print("All plots generated.")

if __name__ == "__main__":
    run_plotting()
""",

    "plot_ablation.py": """import os, glob, math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from PIL import Image

# Config
BASE_DIR = "results"
sns.set_context("paper", font_scale=1.0)
sns.set_style("whitegrid")
COLORS = {"truth": "#222222", "recon": "#E63946"}

def plot_dashboard(npz_path, output_dir):
    try:
        data = np.load(npz_path)
        truth, recon = data['truth'], data['recon']
        name = os.path.basename(npz_path).replace("_data.npz", "")
        
        # Square figure for just the attractor
        fig = plt.figure(figsize=(5, 5))
        
        # 3D Attractor Only
        ax1 = fig.add_subplot(111, projection='3d')
        limit = min(2000, len(truth))
        
        # Plot Truth (Grey, Thin)
        ax1.plot(truth[:limit,0], truth[:limit,1], truth[:limit,2], 
                 c=COLORS["truth"], lw=0.6, alpha=0.3, label="Truth")
        
        # Plot Recon (Red, Dashed)
        ax1.plot(recon[:limit,0], recon[:limit,1], recon[:limit,2], 
                 c=COLORS["recon"], lw=1.2, ls='--', alpha=0.9, label="Recon")
        
        ax1.set_title(f"{name}", fontweight="bold", fontsize=10)
        
        # Clean look but keep orientation hints
        ax1.set_xlabel("x", fontsize=8)
        ax1.set_ylabel("y", fontsize=8)
        ax1.set_zlabel("z", fontsize=8)
        ax1.tick_params(labelsize=6)
            
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"{name}_plot.png")
        plt.savefig(out_path, dpi=100)
        plt.close()
        return out_path
    except Exception as e: 
        print(f"Error plotting {npz_path}: {e}")
        return None

def create_mosaic(img_paths, output_file):
    if not img_paths: return
    
    print(f"Stitching {len(img_paths)} images into mosaic...")
    
    # Grid calculation (approx square)
    N = len(img_paths)
    cols = int(math.ceil(math.sqrt(N)))
    rows = int(math.ceil(N / cols))
    
    # Load first to get size
    with Image.open(img_paths[0]) as img:
        w, h = img.size
        
    # Create massive canvas
    mosaic = Image.new('RGB', (cols * w, rows * h), (255, 255, 255))
    
    for idx, p in enumerate(img_paths):
        try:
            with Image.open(p) as img:
                r, c = idx // cols, idx % cols
                mosaic.paste(img, (c * w, r * h))
        except: pass
        
    mosaic.save(output_file, quality=90)
    print(f"Mosaic saved: {output_file}")

def main():
    # Find all experiment folders
    exp_folders = sorted(glob.glob(f"{BASE_DIR}/*/data"))
    
    for data_dir in exp_folders:
        exp_name = data_dir.split(os.sep)[-2]
        out_dir = data_dir.replace("data", "plots")
        os.makedirs(out_dir, exist_ok=True)
        
        print(f"Generating plots for: {exp_name}")
        files = sorted(glob.glob(f"{data_dir}/*.npz"))
        
        plot_paths = []
        for f in files:
            p = plot_dashboard(f, out_dir)
            if p: plot_paths.append(p)
            
        # Create Summary Mosaic (ALL systems)
        if plot_paths:
            create_mosaic(plot_paths, os.path.join(out_dir, "full_benchmark_mosaic.jpg"))

if __name__ == "__main__":
    main()
""",

    "generate_stats.py": """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Configuration
RESULTS_DIR = "results"
STAGES = [
    ("1_Baseline_Iso_Dyn", "Baseline (Iso)"),
    ("2_Aniso_Dyn", "Anisotropic"),
    ("3_Aniso_MHD", "Aniso + MHD"),
    ("4_Final_Sparse", "Sparse HMKF (Ours)")
]

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

def load_and_merge_data():
    dfs = []
    for folder, label in STAGES:
        path = os.path.join(RESULTS_DIR, folder, "benchmark_results.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['Method'] = label
            df = df.set_index('System')
            dfs.append(df)
        else:
            print(f"Warning: Results for {label} not found at {path}")
            return None

    combined = pd.concat(dfs).reset_index()
    
    # Filter for systems present in all stages
    system_counts = combined['System'].value_counts()
    valid_systems = system_counts[system_counts == len(STAGES)].index
    cleaned_df = combined[combined['System'].isin(valid_systems)]
    
    print(f"Loaded data for {len(valid_systems)} common systems.")
    return cleaned_df

def generate_boxplot(df):
    plt.figure(figsize=(10, 6))
    
    # Log10 for visibility
    df['Log_HD'] = np.log10(df['Standard_HD'].replace(0, 1e-6))
    
    ax = sns.boxplot(x='Method', y='Log_HD', data=df, showfliers=False, width=0.5)
    sns.stripplot(x='Method', y='Log_HD', data=df, color=".25", size=2, alpha=0.5)
    
    plt.title("Ablation Study: Geometric Reconstruction Error (N=133)", fontsize=12)
    plt.ylabel("Log10(Hausdorff Distance)")
    plt.xlabel("")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/ablation_boxplot.png", dpi=300)
    print(f"Generated Boxplot: {RESULTS_DIR}/ablation_boxplot.png")

def generate_sparsity_plot(df):
    # Only relevant for the Sparse method
    sparse_data = df[df['Method'] == "Sparse HMKF (Ours)"]
    
    if sparse_data.empty: return

    plt.figure(figsize=(8, 6))
    
    # Scatter plot: Sparsity vs Log Error
    x = sparse_data['Sparsity'] * 100 # Convert to percentage
    y = np.log10(sparse_data['Standard_HD'].replace(0, 1e-6))
    
    plt.scatter(x, y, alpha=0.6, c='#2ecc71', edgecolors='k')
    
    plt.axvline(x=50, color='r', linestyle='--', label="50% Sparsity Threshold")
    
    plt.title("Sparsity Efficiency: Maintaining Accuracy with Fewer Kernels")
    plt.xlabel("Sparsity (%) \\n(Higher is simpler model)")
    plt.ylabel("Log10(Hausdorff Distance)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/sparsity_analysis.png", dpi=300)
    print(f"Generated Sparsity Plot: {RESULTS_DIR}/sparsity_analysis.png")

def perform_statistical_tests(df):
    # Pivot for paired testing
    pivot = df.pivot(index='System', columns='Method', values='Standard_HD')
    
    # Define comparisons
    comparisons = [
        ("Baseline (Iso)", "Anisotropic"),
        ("Anisotropic", "Aniso + MHD"),
        ("Aniso + MHD", "Sparse HMKF (Ours)"),
        ("Baseline (Iso)", "Sparse HMKF (Ours)")
    ]
    
    print("\\n--- STATISTICAL ANALYSIS (LOG-SPACE T-TEST) ---")
    print("Testing on Log10(Error) to handle heavy-tailed distribution.")
    
    for method_a, method_b in comparisons:
        # 1. Get pairs
        a_raw = pivot[method_a]
        b_raw = pivot[method_b]
        
        # 2. Log Transform (Critical for geometric errors)
        a_log = np.log10(a_raw.replace(0, 1e-9).replace([np.inf, -np.inf], np.nan))
        b_log = np.log10(b_raw.replace(0, 1e-9).replace([np.inf, -np.inf], np.nan))
        
        # 3. Clean NaNs
        mask = ~np.isnan(a_log) & ~np.isnan(b_log)
        a_clean = a_log[mask]
        b_clean = b_log[mask]
        
        if len(a_clean) < 2:
            continue
            
        # 4. Run Test
        stat, p_val = stats.ttest_rel(a_clean, b_clean)
        
        # Mean Log Improvement (Positive = Method B has lower error)
        mean_log_diff = a_clean.mean() - b_clean.mean()
        
        significance = "**SIGNIFICANT**" if p_val < 0.05 else "Not Significant"
        print(f"\\n{method_a} -> {method_b}")
        print(f"   Log-Diff: {mean_log_diff:.4f} (Pos = Improved) | P-val: {p_val:.4e} | {significance}")

if __name__ == "__main__":
    df = load_and_merge_data()
    if df is not None:
        generate_boxplot(df)
        generate_sparsity_plot(df)
        perform_statistical_tests(df)
""",

    "analysis.py": """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import ast
import os

# Create plots directory if it doesn't exist
os.makedirs("results/analysis", exist_ok=True)

def parse_sigma(sigma_str):
    \"\"\"Safely parses the sigma string from the CSV\"\"\"
    try:
        # It might be a string representation of a list
        return json.loads(sigma_str)
    except:
        try:
            return ast.literal_eval(sigma_str)
        except:
            return []

def plot_sparsity_analysis(csv_path):
    df = pd.read_csv(csv_path)
    
    # Filter for Success only
    df_success = df[df['Status'] == 'Success']
    
    if df_success.empty:
        print("No successful runs to analyze.")
        return

    # --- Plot 1: Sigma Heatmap (Sparsity Check) ---
    # We collect sigmas into a matrix (padding with NaNs for systems with diff dims)
    system_names = df_success['System'].values
    
    # Parse sigmas
    all_sigmas = [parse_sigma(s) for s in df_success['Sigma_Learned'].values]
    max_dim = max(len(s) for s in all_sigmas)
    
    # Create matrix (Systems x Max_Dimensions)
    sigma_matrix = np.full((len(df_success), max_dim), np.nan)
    
    for i, sigs in enumerate(all_sigmas):
        # We assume these are log_sigmas based on the CSV values (e.g. 4.6)
        # We plot the ACTUAL scale (exp) to emphasize the magnitude difference
        # Or keeping them as logs is fine too. Let's assume the CSV contains the LOG values.
        sigma_matrix[i, :len(sigs)] = np.array(sigs)

    plt.figure(figsize=(10, 6))
    # Using 'coolwarm': Blue = Low (Relevant), Red = High (Sparse/Ignored)
    plt.imshow(sigma_matrix, aspect='auto', cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Log Sigma (Higher = Less Relevant)')
    plt.yticks(range(len(system_names)), system_names)
    plt.xlabel('State Dimension Index')
    plt.title('Feature Sparsity: Learned Log-Lengthscales per Dimension')
    plt.tight_layout()
    plt.savefig("results/analysis/sparsity_heatmap.png")
    plt.close()
    
    # --- Plot 2: Detailed Bar Chart for specific systems ---
    # Pick top 3 distinct systems to show "Sparsity vs Uniformity"
    sample_indices = np.linspace(0, len(df_success)-1, min(3, len(df_success)), dtype=int)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, idx in enumerate(sample_indices):
        name = system_names[idx]
        sigmas = all_sigmas[idx]
        ax = axes[i]
        
        colors = ['red' if s > np.mean(sigmas) else 'blue' for s in sigmas]
        ax.bar(range(len(sigmas)), sigmas, color=colors)
        ax.set_title(f"{name}\\n(Var: {np.var(sigmas):.2f})")
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Log Sigma")
        ax.axhline(y=np.mean(sigmas), color='gray', linestyle='--', alpha=0.5, label='Mean')

    plt.suptitle("Sparsity Verification: Variation in Lengthscales")
    plt.tight_layout()
    plt.savefig("results/analysis/sparsity_bars.png")
    plt.close()

    print("Analysis plots saved to results/analysis/")
    print("\\n--- Numerical Sparsity Report ---")
    for name, sigs in zip(system_names, all_sigmas):
        sigs = np.array(sigs)
        variance = np.var(sigs)
        print(f"{name}: Variance={variance:.3f} | {'SPARSE' if variance > 0.5 else 'UNIFORM'}")

if __name__ == "__main__":
    # Use the file you uploaded
    plot_sparsity_analysis("final_benchmark_results_FIXED.csv")
""",

    "conclusions.py": """import pandas as pd
import numpy as np
import os

RESULTS_DIR = "results"
STAGES = [
    ("1_Baseline_Iso_Dyn", "Baseline (Iso)"),
    ("2_Aniso_Dyn", "Anisotropic"),
    ("3_Aniso_MHD", "Aniso + MHD"),
    ("4_Final_Sparse", "Sparse HMKF")
]

def load_all_stages():
    \"\"\"Loads and merges data from all 4 experiments.\"\"\"
    dfs = []
    for folder, label in STAGES:
        path = f"{RESULTS_DIR}/{folder}/benchmark_results.csv"
        try:
            df = pd.read_csv(path).set_index("System")
            # Keep only the metrics we care about
            df = df[['Standard_HD', 'Sparsity']]
            df.columns = [f"HD_{label}", f"Sparsity_{label}"]
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {label} ({e})")
            return None
    
    # Merge all into one big table (inner join to be fair)
    full_df = pd.concat(dfs, axis=1, join='inner')
    return full_df

def analyze_full_ablation():
    df = load_all_stages()
    if df is None: return

    # 1. Calculate Mean Error for each Stage
    # We use median as well because of the "Lorenz Explosion" outliers
    summary_data = []
    
    for _, label in STAGES:
        hd_col = f"HD_{label}"
        
        # Mean/Median Error
        mean_hd = df[hd_col].mean()
        median_hd = df[hd_col].median()
        
        # Mean Sparsity (Only relevant for Sparse model, but checking all)
        sparsity_col = f"Sparsity_{label}"
        mean_sparsity = df[sparsity_col].mean() * 100 if sparsity_col in df else 0
        
        summary_data.append({
            "Stage": label,
            "Mean HD (Error)": mean_hd,
            "Median HD": median_hd,
            "Avg Sparsity (%)": mean_sparsity
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # 2. Calculate "Wins" for the Final Model vs Baseline
    # Improvement %
    base_hd = df[f"HD_{STAGES[0][1]}"]
    final_hd = df[f"HD_{STAGES[-1][1]}"]
    
    # Avoid divide by zero
    improvement = (base_hd - final_hd) / base_hd.replace(0, 1e-9) * 100
    
    # Top 10 Wins (Geometric)
    wins_df = pd.DataFrame({
        "System": df.index,
        "Baseline HD": base_hd,
        "Final Sparse HD": final_hd,
        "Improvement (%)": improvement
    }).sort_values("Improvement (%)", ascending=False).head(15)

    # --- SAVE FILES ---
    
    # A. The Full Story (Table for Paper)
    summary_path = f"{RESULTS_DIR}/full_ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False, float_format="%.4f")
    
    # B. The Top Wins (Table for Appendix/Results)
    wins_path = f"{RESULTS_DIR}/geometric_wins.csv"
    wins_df.to_csv(wins_path, float_format="%.4f")
    
    print(f"\\n✅ Analysis Complete. Files Saved:")
    print(f"   - {summary_path} (The progression 1->2->3->4)")
    print(f"   - {wins_path} (The specific systems we fixed)")
    
    print("\\n=== ABLATION SUMMARY (The Story) ===")
    print(summary_df.to_string(float_format="%.4f", index=False))

if __name__ == "__main__":
    analyze_full_ablation()
""",

    "3d_plotter.py": """import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import dysts.flows
from tqdm import tqdm
import json

# --- JAX SETUP ---
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.linalg import solve
from jax import jit

# --- UTILS (Inlined for standalone execution) ---
def standardize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    # Avoid division by zero
    std[std == 0] = 1.0
    return (data - mean) / std, mean, std

def get_derivatives(data, dt):
    return (data[1:] - data[:-1]) / dt

def rk4_integrate(vector_field, x0, dt, steps):
    \"\"\"Simple RK4 integrator\"\"\"
    traj = [x0]
    x = x0
    for _ in range(steps):
        k1 = vector_field(x)
        k2 = vector_field(x + 0.5 * dt * k1)
        k3 = vector_field(x + 0.5 * dt * k2)
        k4 = vector_field(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        traj.append(x)
    return np.array(traj[1:])

# --- KERNEL ---
@jit
def anisotropic_gaussian_kernel_inference(x1, x2, log_sigmas):
    sigmas = jnp.exp(log_sigmas)
    x1_scaled = x1 / sigmas[None, :]
    x2_scaled = x2 / sigmas[None, :]
    diff = x1_scaled[:, None, :] - x2_scaled[None, :, :]
    dist_sq = jnp.sum(diff**2, axis=-1)
    return jnp.exp(-0.5 * dist_sq)

# --- CONFIG ---
# Using "results" folder as requested
CSV_FILE = "results/final_benchmark_results_FIXED.csv" 
OUTPUT_DIR = "results/plots_3d"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fit_and_predict_oneshot(X_train, Y_train, sigmas_array, regularization=1e-4):
    X = jnp.array(X_train)
    Y = jnp.array(Y_train)
    # Convert sigma back to log_sigma for the kernel function
    log_sigmas = jnp.log(jnp.array(sigmas_array))
    
    K = anisotropic_gaussian_kernel_inference(X, X, log_sigmas)
    K_reg = K + regularization * jnp.eye(len(X))
    weights = solve(K_reg, Y, assume_a='pos')
    return X, weights, log_sigmas

def main():
    if not os.path.exists(CSV_FILE):
        fallback = "results/benchmark_progress.csv"
        if os.path.exists(fallback):
            print(f"Using partial results: {fallback}")
            df = pd.read_csv(fallback)
        else:
            print(f"No CSV found at {CSV_FILE}. Make sure the path is correct.")
            return
    else:
        df = pd.read_csv(CSV_FILE)

    print(f"Found {len(df)} systems. Generating 3D plots...")

    count = 0
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating 3D Plots"):
        system_name = row['System']
        sigma_str = row['Sigma_Learned']
        
        try:
            # Parse Sigma Vector
            if pd.isna(sigma_str): continue
            
            sigma_list = json.loads(sigma_str)
            sigma_arr = np.array(sigma_list)
            
            # Skip if invalid
            if np.any(np.array(sigma_list) <= 0) or np.any(np.isnan(sigma_list)):
                continue

            # 1. Re-generate Data
            # Use try/except specifically for dysts loading to debug typos
            try:
                model = getattr(dysts.flows, system_name)()
            except AttributeError:
                print(f"\\nSkipping {system_name}: Not found in dysts.")
                continue

            t_sol, sol = model.make_trajectory(n=4000, resample=True, return_times=True)
            dt = model.dt if hasattr(model, 'dt') else 0.01
            if dt == 0: dt = 0.01

            # 2. Preprocessing
            data_std, mean, std = standardize_data(sol)
            X_all = data_std[:-1]
            V_all = get_derivatives(data_std, dt)
            
            TRAIN_PTS = 1500
            TEST_PTS = 2500
            
            X_train = X_all[:TRAIN_PTS]
            Y_train = V_all[:TRAIN_PTS]
            
            # 3. Training
            X_support, weights, log_sigmas_learned = fit_and_predict_oneshot(X_train, Y_train, sigma_arr)
            
            # 4. Predict Wrapper
            # We wrap the JAX function to pass to the python integrator
            def predict_fn(x_new):
                if x_new.ndim == 1: x_new = x_new[None, :]
                k_vec = anisotropic_gaussian_kernel_inference(x_new, X_support, log_sigmas_learned)
                pred = jnp.dot(k_vec, weights)
                return np.array(pred[0])
            
            # 5. Integrate
            x0 = X_train[-1]
            recon_std = rk4_integrate(predict_fn, x0, dt, steps=TEST_PTS)
            recon_real = (recon_std * std) + mean
            truth_real = sol[TRAIN_PTS : TRAIN_PTS + TEST_PTS]
            
            # 6. PLOTTING
            dim = truth_real.shape[1]
            fig = plt.figure(figsize=(10, 8))
            
            if dim >= 3:
                ax = fig.add_subplot(111, projection='3d')
                ax.plot(truth_real[:, 0], truth_real[:, 1], truth_real[:, 2], 
                        label="Ground Truth", color='blue', alpha=0.4, linewidth=0.6)
                ax.plot(recon_real[:, 0], recon_real[:, 1], recon_real[:, 2], 
                        label="Reconstruction", color='darkorange', alpha=0.9, linewidth=1.0, linestyle='--')
                ax.set_title(f"{system_name} (Anisotropic)")
            elif dim == 2:
                ax = fig.add_subplot(111)
                ax.plot(truth_real[:, 0], truth_real[:, 1], label="Truth", color='blue', alpha=0.5)
                ax.plot(recon_real[:, 0], recon_real[:, 1], label="Recon", color='orange', linestyle='--')
                ax.set_title(f"{system_name} (Anisotropic)")
            
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/{system_name}_3D.png")
            plt.close()
            count += 1

        except Exception as e:
            # UNCOMMENTED THIS so you can see why it fails
            print(f"Error plotting {system_name}: {e}")
            continue

    print(f"Done. Generated {count} plots in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
"""
}

def create_structure():
    # Create directories
    for folder in ["src", "results", "results/plots_3d", "results/analysis"]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created directory: {folder}")

    # Write files
    for filepath, content in files.items():
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created file: {filepath}")
    
    print("\nProject Initialized! You can now run 'python parallel_ablation.py' to start the full experiment.")

if __name__ == "__main__":
    create_structure()