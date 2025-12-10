import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import ast
import os

# Create plots directory if it doesn't exist
os.makedirs("results/analysis", exist_ok=True)

def parse_sigma(sigma_str):
    """Safely parses the sigma string from the CSV"""
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
        ax.set_title(f"{name}\n(Var: {np.var(sigmas):.2f})")
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Log Sigma")
        ax.axhline(y=np.mean(sigmas), color='gray', linestyle='--', alpha=0.5, label='Mean')

    plt.suptitle("Sparsity Verification: Variation in Lengthscales")
    plt.tight_layout()
    plt.savefig("results/analysis/sparsity_bars.png")
    plt.close()

    print("Analysis plots saved to results/analysis/")
    print("\n--- Numerical Sparsity Report ---")
    for name, sigs in zip(system_names, all_sigmas):
        sigs = np.array(sigs)
        variance = np.var(sigs)
        print(f"{name}: Variance={variance:.3f} | {'SPARSE' if variance > 0.5 else 'UNIFORM'}")

if __name__ == "__main__":
    # Use the file you uploaded
    plot_sparsity_analysis("final_benchmark_results_FIXED.csv")