import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Path to CSV relative to this script location (results/plots/)
CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'experiment_log.csv')
SCATTER_OUTPUT = 'runtime_vs_quality_scatter.png'
BAR_OUTPUT = 'algorithm_comparison_bars.png'

# Colors for consistency
COLORS = {
    'bron_kerbosch': '#2ecc71',       # Green
    'greedy': '#f1c40f',              # Yellow
    'local_search': '#9b59b6',        # Purple
    'simulated_annealing': '#e74c3c', # Red
    'randomized': '#3498db',          # Blue
    'local_random': '#e67e22'         # Orange
}

def main():
    # 1. Load Data
    if not os.path.exists(CSV_PATH):
        print(f"Error: Could not find {CSV_PATH}")
        return

    column_names = ['Index', 'Size_Category', 'Graph_Name', 'Nodes', 'Algorithm', 'Clique_Size', 'Runtime']
    df = pd.read_csv(CSV_PATH, header=None, names=column_names)

    # 2. Clean Data (Handle 'x' and 'xxxxxx')
    df['Clique_Size'] = pd.to_numeric(df['Clique_Size'], errors='coerce')
    df['Runtime'] = pd.to_numeric(df['Runtime'], errors='coerce')
    df.dropna(inplace=True) # Remove failed rows

    print(f"Loaded {len(df)} valid results from {CSV_PATH}")

    # ==========================================================================
    # PLOT 1: SCATTER (Individual Performance)
    # ==========================================================================
    fig1, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    size_cats = ['small', 'medium', 'large']

    for i, cat in enumerate(size_cats):
        ax = axes[i]
        subset = df[df['Size_Category'] == cat]
        
        if subset.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center')
            ax.set_title(f"{cat.capitalize()} Graphs")
            continue
            
        for algo in subset['Algorithm'].unique():
            algo_data = subset[subset['Algorithm'] == algo]
            ax.scatter(
                algo_data['Runtime'], 
                algo_data['Clique_Size'], 
                label=algo.replace('_', ' ').title(),
                color=COLORS.get(algo, 'gray'),
                s=80, alpha=0.7, edgecolors='k'
            )

        ax.set_title(f"{cat.capitalize()} Graphs", fontsize=14, fontweight='bold')
        ax.set_xlabel("Runtime (s) - Log Scale", fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel("Clique Size (Quality)", fontweight='bold')

    # Global Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=6, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(SCATTER_OUTPUT, dpi=300, bbox_inches='tight')
    print(f"Saved scatter plot to {SCATTER_OUTPUT}")
    plt.close()

    # ==========================================================================
    # PLOT 2: BAR CHART (Averages)
    # ==========================================================================
    agg_df = df.groupby(['Size_Category', 'Algorithm']).agg({
        'Runtime': 'mean',
        'Clique_Size': 'mean'
    }).reset_index()

    fig2, axes = plt.subplots(1, 2, figsize=(18, 7))
    metrics = [
        {'col': 'Clique_Size', 'title': 'Average Solution Quality', 'ylabel': 'Avg Clique Size', 'log': False, 'ax': axes[0]},
        {'col': 'Runtime', 'title': 'Average Runtime', 'ylabel': 'Avg Runtime (s) - Log Scale', 'log': True, 'ax': axes[1]}
    ]

    algo_order = [a for a in COLORS.keys() if a in df['Algorithm'].unique()]
    x = np.arange(len(size_cats))
    bar_width = 0.8 / len(algo_order)

    for config in metrics:
        ax = config['ax']
        col = config['col']
        
        for i, algo in enumerate(algo_order):
            vals = []
            for size in size_cats:
                row = agg_df[(agg_df['Size_Category'] == size) & (agg_df['Algorithm'] == algo)]
                vals.append(row[col].values[0] if not row.empty else 0)
            
            pos = x - 0.4 + (i * bar_width) + (bar_width/2)
            ax.bar(pos, vals, width=bar_width, label=algo.replace('_', ' ').title(), 
                   color=COLORS.get(algo, 'gray'), edgecolor='black', alpha=0.85)

        ax.set_title(config['title'], fontsize=14, fontweight='bold')
        ax.set_ylabel(config['ylabel'], fontweight='bold')
        ax.set_xlabel('Graph Size Category', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in size_cats], fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        if config['log']:
            ax.set_yscale('log')

    handles, labels = axes[0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=6, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(BAR_OUTPUT, dpi=300, bbox_inches='tight')
    print(f"Saved bar charts to {BAR_OUTPUT}")
    plt.close()

if __name__ == "__main__":
    main()