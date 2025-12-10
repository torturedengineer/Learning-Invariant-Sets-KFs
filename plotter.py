import os
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
            ax2.text(0.5, 0.5, "No Active Kernels\n(Convergence Issue)", 
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