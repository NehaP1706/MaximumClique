import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ================= CONFIGURATION =================
# Path to your main results log
LOG_FILE = '../experiment_log.csv' 
DATA_DIR = '../../data'
DENSITY_THRESHOLD = 0.25 
OUTPUT_FILE = 'runtime_sparse_vs_dense.png'

# Exact colors from your plot_from_logs.py
COLORS = {
    'bron_kerbosch': '#2ecc71',       
    'greedy': '#f1c40f',              
    'local_search': '#9b59b6',        
    'simulated_annealing': '#e74c3c', 
    'randomized': '#3498db',          
    'local_random': '#e67e22'         
}

def get_graph_density(graph_name):
    """Calculates density by reading the .adj file from data folders."""
    found_path = None
    for sub in ['small_graphs', 'medium_graphs', 'large_graphs']:
        path = os.path.join(DATA_DIR, sub, graph_name)
        if os.path.exists(path):
            found_path = path
            break
    
    if not found_path: return None

    try:
        with open(found_path, 'r') as f:
            lines = [l for l in f.readlines() if l.strip() and not l.startswith('%')]
        
        num_nodes = len(lines)
        if num_nodes <= 1: return 0.0

        total_degrees = 0
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 1:
                total_degrees += len(parts) - 1 
                
        max_edges = num_nodes * (num_nodes - 1)
        return total_degrees / max_edges
    except:
        return None

def generate_plot():
    # 1. Load Data
    cols = ['Index', 'Size_Category', 'Graph_Name', 'Nodes', 'Algorithm', 'Clique_Size', 'Runtime']
    if not os.path.exists(LOG_FILE):
        print(f"Error: Could not find {LOG_FILE}")
        return
        
    df = pd.read_csv(LOG_FILE, names=cols, header=None)
    df['Runtime'] = pd.to_numeric(df['Runtime'], errors='coerce')
    df.dropna(subset=['Runtime'], inplace=True)

    # 2. Calculate Density
    print("Calculating graph densities...")
    unique_graphs = df['Graph_Name'].unique()
    density_map = {name: get_graph_density(name) for name in unique_graphs}
    df['Density'] = df['Graph_Name'].map(density_map)
    df.dropna(subset=['Density'], inplace=True)

    # 3. Classify
    df['Type'] = df['Density'].apply(lambda x: 'Dense' if x >= DENSITY_THRESHOLD else 'Sparse')

    # 4. Aggregate Data (Mean Runtime)
    # Group by Algorithm AND Type
    agg_df = df.groupby(['Algorithm', 'Type'])['Runtime'].mean().unstack()
    
    # 5. Plotting (Pure Matplotlib style)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    algorithms = agg_df.index
    x = np.arange(len(algorithms))
    width = 0.35
    
    # Plot Sparse Bars
    rects1 = ax.bar(x - width/2, agg_df['Sparse'], width, label='Sparse', color='#3498db', edgecolor='black')
    # Plot Dense Bars
    rects2 = ax.bar(x + width/2, agg_df['Dense'], width, label='Dense', color='#e74c3c', edgecolor='black')

    # Formatting
    ax.set_ylabel('Average Runtime (Seconds) - Log Scale', fontsize=12)
    ax.set_title(f'Runtime Comparison: Sparse (<{DENSITY_THRESHOLD}) vs Dense', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([algo.replace('_', ' ').title() for algo in algorithms], rotation=15, ha='right')
    ax.legend()
    ax.set_yscale('log') # Crucial for seeing both fast and slow algorithms
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"✅ Plot saved to {os.path.abspath(OUTPUT_FILE)}")
    plt.show()

if __name__ == "__main__":
    generate_plot()