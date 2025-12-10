import os, glob, math
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