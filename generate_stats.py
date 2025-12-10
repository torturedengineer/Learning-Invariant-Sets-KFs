import pandas as pd
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
    plt.xlabel("Sparsity (%) \n(Higher is simpler model)")
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
    
    print("\n--- STATISTICAL ANALYSIS (LOG-SPACE T-TEST) ---")
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
        print(f"\n{method_a} -> {method_b}")
        print(f"   Log-Diff: {mean_log_diff:.4f} (Pos = Improved) | P-val: {p_val:.4e} | {significance}")

if __name__ == "__main__":
    df = load_and_merge_data()
    if df is not None:
        generate_boxplot(df)
        generate_sparsity_plot(df)
        perform_statistical_tests(df)