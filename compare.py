import os
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
    """Loads learned kernel parameters from a specific experiment."""
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
