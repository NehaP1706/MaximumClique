import sys
import os
import glob
import time
import pandas as pd
import matplotlib.pyplot as plt
import signal

# ============================================================================
# 1. ROBUST PATH SETUP
# ============================================================================
def setup_paths():
    """Finds algorithms and data folders relative to this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Possible locations for 'algorithms'
    possible_algo_paths = [
        os.path.join(script_dir, '..', '..', 'algorithms'),
    ]
    
    algo_path_found = None
    for path in possible_algo_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, 'bron_kerbosch.py')):
            algo_path_found = path
            break
    
    if algo_path_found:
        sys.path.append(algo_path_found)
        print(f"✅ Found algorithms at: {algo_path_found}")
    else:
        print("❌ Critical Error: Could not find 'algorithms' folder.")
        sys.exit(1)

    # Possible locations for 'data'
    possible_data_paths = [
        os.path.join(script_dir, '..', '..', 'data'),
    ]
    
    data_path_found = None
    for path in possible_data_paths:
        if os.path.exists(path) and (os.path.exists(os.path.join(path, 'medium_graphs')) or os.path.exists(os.path.join(path, 'small_graphs'))):
            data_path_found = path
            break
            
    return data_path_found

DATA_DIR = setup_paths()

# ============================================================================
# 2. IMPORTS (FIXED)
# ============================================================================
try:
    # 1. Standard Algorithms
    from bron_kerbosch import bron_kerbosch_with_pivot
    from greedy import find_maximum_clique_from_dict
    from simulated_annealing import simulated_annealing_with_restarts
    
    # 2. Local Search (Flexible Import)
    try:
        # Try the name you likely used
        from local_search import local_search
    except ImportError:
        try:
            # Try the name I suggested previously
            from advanced_local_search import local_search
        except ImportError:
            print("⚠️ Warning: Could not import 'local_search'. Please ensure 'local_search.py' exists.")
            # Define a dummy function so script doesn't crash, just skips it
            def local_search(G): return set(), 0.0

except ImportError as e:
    print(f"\n❌ Import Error: {e}")
    print("Please check your 'algorithms' folder for bron_kerbosch.py, greedy.py, local_search.py, etc.")
    sys.exit(1)

# ============================================================================
# 3. ALGORITHM CONFIGURATION
# ============================================================================

def sa_wrapper(G):
    # Wrapper to get just the clique from SA
    result = simulated_annealing_with_restarts(G, max_iterations=1000, num_restarts=5, verbose=False)
    return result[0]

ALGORITHMS = {
    'Bron-Kerbosch': {
        'func': lambda G: bron_kerbosch_with_pivot(G),
        'color': '#2ecc71', # Green
        'limit': 'small',   # Skip on medium/large
        'marker': 'o'
    },
    'Greedy': {
        'func': lambda G: find_maximum_clique_from_dict(G),
        'color': '#f1c40f', # Yellow
        'limit': 'all',
        'marker': 's'
    },
    'Local Search': {
        'func': lambda G: local_search(G),
        'color': '#9b59b6', # Purple
        'limit': 'all',
        'marker': 'D'
    },
    'Simulated Annealing': {
        'func': sa_wrapper,
        'color': '#e74c3c', # Red
        'limit': 'all',
        'marker': '*'
    }
}

# ============================================================================
# 4. UTILITIES
# ============================================================================

class TimeoutException(Exception): pass

def timeout_handler(signum, frame):
    raise TimeoutException()

def load_graph(filepath):
    """Robust graph loader."""
    G = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%') or line.startswith('#'): continue
            # Remove colons if present (e.g. "1: 2 3" -> "1 2 3")
            parts = line.replace(':', ' ').split()
            if len(parts) < 1: continue
            try:
                node = int(parts[0])
                neighbors = set()
                for x in parts[1:]:
                    if x.isdigit(): neighbors.add(int(x))
                G[node] = neighbors
            except ValueError: continue
    return G

# ============================================================================
# 5. MAIN EXPERIMENT RUNNER
# ============================================================================
def run_experiments():
    results = []
    
    if not DATA_DIR:
        print("❌ Error: Could not find 'data' folder.")
        return pd.DataFrame()

    print(f"📂 Loading data from: {DATA_DIR}")
    categories = ['small_graphs', 'medium_graphs', 'large_graphs']

    print(f"{'='*60}")
    print(f"{'RUNNING EXPERIMENTS':^60}")
    print(f"{'='*60}")

    # Set global timeout (120s)
    signal.signal(signal.SIGALRM, timeout_handler)

    for category in categories:
        path = os.path.join(DATA_DIR, category)
        if not os.path.exists(path): continue

        graph_files = sorted(glob.glob(os.path.join(path, "*.adj")))
        print(f"\nProcessing {category} ({len(graph_files)} graphs)...")

        for filepath in graph_files:
            filename = os.path.basename(filepath)
            G = load_graph(filepath)
            if not G: continue
            num_nodes = len(G)

            for algo_name, algo_config in ALGORITHMS.items():
                # Skip Exact algorithm on medium/large graphs
                if algo_config['limit'] == 'small' and category != 'small_graphs':
                    continue

                try:
                    signal.alarm(120) # Start Timeout
                    start_t = time.time()
                    
                    # Run Algorithm
                    raw_result = algo_config['func'](G)
                    
                    runtime = time.time() - start_t
                    signal.alarm(0) # Stop Timeout

                    # Handle Tuple Returns
                    if isinstance(raw_result, tuple):
                        raw_result = raw_result[0]

                    # Calculate Size
                    if isinstance(raw_result, (set, list)):
                        clique_size = len(raw_result)
                    else:
                        clique_size = 0

                    results.append({
                        'Graph_Type': category.replace('_graphs', '').capitalize(),
                        'Graph_Name': filename,
                        'Nodes': num_nodes,
                        'Algorithm': algo_name,
                        'Clique_Size': clique_size,
                        'Runtime_Seconds': runtime
                    })
                    print(f"  -> {algo_name:<20}: Size {clique_size:3d}, Time {runtime:.5f}s")

                except TimeoutException:
                    print(f"  -> {algo_name:<20}: TIMEOUT (>120s)")
                except Exception as e:
                    signal.alarm(0)
                    print(f"  -> {algo_name:<20}: ERROR ({str(e)[:30]})")

    return pd.DataFrame(results)

# ============================================================================
# 6. PLOTTING
# ============================================================================
def plot_results(df):
    if df.empty: return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    graph_types = ['Small', 'Medium', 'Large']

    for i, g_type in enumerate(graph_types):
        ax = axes[i]
        subset = df[df['Graph_Type'] == g_type]
        
        if subset.empty:
            ax.text(0.5, 0.5, "No Data", ha='center')
            ax.set_title(f"{g_type} Graphs")
            continue

        for algo_name, algo_config in ALGORITHMS.items():
            algo_data = subset[subset['Algorithm'] == algo_name]
            if not algo_data.empty:
                # Plot max clique found vs average time (handling dupes)
                plot_data = algo_data.groupby('Graph_Name').agg({
                    'Runtime_Seconds': 'mean', 
                    'Clique_Size': 'max'
                }).reset_index()
                
                ax.scatter(
                    plot_data['Runtime_Seconds'], 
                    plot_data['Clique_Size'], 
                    label=algo_name,
                    color=algo_config['color'],
                    marker=algo_config['marker'],
                    s=100, alpha=0.7, edgecolors='k'
                )

        ax.set_title(f"{g_type} Graphs", fontsize=12, fontweight='bold')
        ax.set_xlabel("Runtime (s) - Log Scale")
        ax.set_ylabel("Clique Size")
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4)
    plt.tight_layout()
    
    outfile = "runtime_vs_quality_plot.png"
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {os.path.abspath(outfile)}")

if __name__ == "__main__":
    df = run_experiments()
    if not df.empty:
        csv_file = "../plot/runtime_vs_quality_results.csv"
        df.to_csv(csv_file, index=False)
        print(f"\n✅ Results saved to: {os.path.abspath(csv_file)}")
        plot_results(df)