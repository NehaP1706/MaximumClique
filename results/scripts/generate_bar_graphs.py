import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# --- CONFIGURATION ---
CSV_FILE = 'runtime_vs_quality_results.csv'
OUTPUT_FILE = 'bar_chart_comparison.png'
# Colors matching your scatter plot
COLORS = {
    'Bron-Kerbosch': '#2ecc71',       # Green
    'Greedy': '#f1c40f',              # Yellow
    'Local Search': '#9b59b6',        # Purple
    'Simulated Annealing': '#e74c3c'  # Red
}

def generate_plots():
    # 1. Load Data
    if not os.path.exists(CSV_FILE):
        print(f"❌ Error: Could not find '{CSV_FILE}'.")
        print("   Please run 'runtime_vs_quality.py' first to generate the data.")
        return

    df = pd.read_csv(CSV_FILE)
    
    # Clean up Algorithm names if they have extra text (e.g., " (Exact)")
    df['Algorithm_Clean'] = df['Algorithm'].apply(lambda x: x.split(' (')[0])
    
    # Define order for Graph Types
    size_order = ['Small', 'Medium', 'Large']
    
    # 2. Setup Figure (2 Subplots: Quality and Runtime)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Metrics to plot
    metrics = [
        {'col': 'Clique_Size', 'title': 'Solution Quality (Avg Clique Size)', 'ylabel': 'Clique Size', 'log': False, 'ax': axes[0]},
        {'col': 'Runtime_Seconds', 'title': 'Efficiency (Avg Runtime)', 'ylabel': 'Runtime (s) - Log Scale', 'log': True, 'ax': axes[1]}
    ]

    # 3. Generate Grouped Bars
    for config in metrics:
        ax = config['ax']
        
        # Aggregate data: Mean value per Algorithm per Graph Type
        # We use a pivot table to structure the data for plotting
        pivot_df = df.pivot_table(
            index='Graph_Type', 
            columns='Algorithm_Clean', 
            values=config['col'], 
            aggfunc='mean'
        )
        
        # Reorder rows to match Small -> Medium -> Large
        pivot_df = pivot_df.reindex(size_order)
        
        # Plotting parameters
        n_groups = len(pivot_df.index)
        n_bars = len(pivot_df.columns)
        bar_width = 0.8 / n_bars
        indices = np.arange(n_groups)
        
        # Draw bars for each algorithm
        for i, algo in enumerate(pivot_df.columns):
            vals = pivot_df[algo]
            # Get color (default to gray if not in dict)
            color = COLORS.get(algo, '#95a5a6')
            
            # Offset position for grouped look
            pos = indices - (0.4) + (i * bar_width) + (bar_width / 2)
            
            ax.bar(pos, vals, width=bar_width, label=algo, color=color, edgecolor='black', alpha=0.8)

        # Formatting
        ax.set_title(config['title'], fontsize=14, fontweight='bold')
        ax.set_ylabel(config['ylabel'], fontsize=12)
        ax.set_xlabel('Graph Size Category', fontsize=12)
        ax.set_xticks(indices)
        ax.set_xticklabels(pivot_df.index, fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        
        if config['log']:
            ax.set_yscale('log')
            
    # 4. Final Touches
    # Single legend at the top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"✅ Bar graphs saved to: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    generate_plots()