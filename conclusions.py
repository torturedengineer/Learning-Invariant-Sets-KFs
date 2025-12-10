import pandas as pd
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
    """Loads and merges data from all 4 experiments."""
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
    
    print(f"\n✅ Analysis Complete. Files Saved:")
    print(f"   - {summary_path} (The progression 1->2->3->4)")
    print(f"   - {wins_path} (The specific systems we fixed)")
    
    print("\n=== ABLATION SUMMARY (The Story) ===")
    print(summary_df.to_string(float_format="%.4f", index=False))

if __name__ == "__main__":
    analyze_full_ablation()