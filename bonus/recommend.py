import os
import sys
import numpy as np
import networkx as nx
import pandas as pd
import gzip
import bz2
import tarfile
import zipfile
import shutil
from tempfile import TemporaryDirectory
from scipy.io import mmread
from io import StringIO

# ==========================================
# 1. CONVERSION LOGIC (Adapted from download_graphs.py)
# ==========================================
def save_as_adjlist(G, output_path):
    """Save the graph in adjacency list format: node: neighbor1 neighbor2 ..."""
    with open(output_path, "w") as f:
        for node in sorted(G.nodes()):
            neighbors = " ".join(str(n) for n in sorted(G.neighbors(node)))
            f.write(f"{node}: {neighbors}\n")

def load_graph(path):
    """Load a graph from various formats into a NetworkX Graph."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    fname = os.path.basename(path).lower()
    
    # Detect extension
    if fname.endswith(".tar.gz") or fname.endswith(".tgz"):
        ext = ".tar.gz"
    else:
        ext = os.path.splitext(path)[1].lower()

    print(f"📂 Loading file type '{ext}': {fname}")

    # --- HANDLERS ---
    if ext == ".bz2":
        with bz2.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip().splitlines()
        # Try Pajek first
        if any(line.lower().startswith("*vertices") for line in content[:10]):
            try:
                with bz2.open(path, "rt", encoding="utf-8", errors="ignore") as f:
                    return nx.Graph(nx.read_pajek(f))
            except: pass
        # Fallback to edge list
        G = nx.Graph()
        for line in content:
            parts = line.strip().split()
            if len(parts) >= 2:
                G.add_edge(parts[0], parts[1])
        return G

    elif ext == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            return nx.read_edgelist(f, comments="#", nodetype=str)

    elif ext in [".tar", ".tar.gz", ".tgz", ".zip"]:
        # Archive handling
        with TemporaryDirectory() as tmpdir:
            if ext == ".zip":
                with zipfile.ZipFile(path, "r") as zf:
                    zf.extractall(tmpdir)
            else:
                with tarfile.open(path, "r:*") as tar:
                    tar.extractall(tmpdir)
            
            # Find first valid graph file inside
            for root, _, files in os.walk(tmpdir):
                for file in files:
                    if not file.startswith('.') and '__MACOSX' not in root:
                        full_path = os.path.join(root, file)
                        # Recursively load the extracted file
                        try:
                            return load_graph(full_path)
                        except:
                            continue
            raise ValueError(f"No valid graph files found inside archive {fname}")

    elif ext == ".mtx":
        matrix = mmread(path)
        return nx.from_scipy_sparse_array(matrix)

    elif ext == ".clq":
        edges = []
        with open(path, "r") as f:
            for line in f:
                if line.startswith("e "):
                    p = line.split()
                    if len(p) >= 3: edges.append((p[1], p[2]))
        G = nx.Graph()
        G.add_edges_from(edges)
        return G

    elif ext == ".graphml":
        return nx.Graph(nx.read_graphml(path))

    elif ext == ".gml":
        return nx.Graph(nx.read_gml(path))

    elif ext in [".csv", ".tsv"]:
        sep = '\t' if ext == ".tsv" else ','
        df = pd.read_csv(path, sep=sep, header=None, comment='#')
        if df.shape[1] >= 2:
            G = nx.Graph()
            G.add_edges_from([(str(r[0]), str(r[1])) for _, r in df.iterrows()])
            return G
        raise ValueError("Invalid CSV/TSV edge list")

    elif ext in [".graph", ".net"]:
        return nx.Graph(nx.read_pajek(path))

    elif ext in [".txt", ".edges"]:
        return nx.read_edgelist(path, comments="#", nodetype=str)

    else:
        # If unknown, try treating as edge list
        try:
            return nx.read_edgelist(path, comments="#", nodetype=str)
        except:
            raise ValueError(f"Unsupported format: {ext}")

def ensure_adj_file(input_path):
    """Checks if file is .adj. If not, converts it and returns path to new .adj file."""
    if input_path.endswith(".adj"):
        return input_path

    print(f"⚡ Input is not .adj. Converting {input_path}...")
    
    try:
        # Load Graph
        G = load_graph(input_path)
        
        # Relabel nodes to integers (0..N-1) for consistency
        G = nx.convert_node_labels_to_integers(G, first_label=0)
        
        # Define output path (filename.adj in same folder)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = os.path.dirname(input_path)
        if not output_dir: output_dir = "."
        
        new_adj_path = os.path.join(output_dir, base_name + ".adj")
        
        # Save
        save_as_adjlist(G, new_adj_path)
        print(f"✅ Conversion successful! Created: {new_adj_path}")
        print(f"   Graph Stats: {len(G.nodes())} nodes, {len(G.edges())} edges")
        
        return new_adj_path
    
    except Exception as e:
        print(f"❌ Error converting file: {e}")
        return None

# ==========================================
# 2. METRIC CALCULATOR
# ==========================================
def analyze_graph_file(file_path):
    """Reads a .adj file and calculates Nodes, Density, and StdDev."""
    if not os.path.exists(file_path):
        return None, None, None

    try:
        with open(file_path, 'r') as f:
            lines = [l for l in f.readlines() if l.strip() and not l.startswith('%')]
        
        num_nodes = len(lines)
        if num_nodes <= 1:
            return num_nodes, 0, 0

        degrees = []
        for line in lines:
            parts = line.strip().split()
            # Degree = (items on line) - 1 (the node ID itself)
            d = max(0, len(parts) - 1)
            degrees.append(d)
            
        # Density
        total_edges = sum(degrees) / 2
        max_edges = num_nodes * (num_nodes - 1) / 2
        density = total_edges / max_edges if max_edges > 0 else 0
        
        # Degree StdDev Normalized
        std_dev = np.std(degrees)
        std_dev_norm = std_dev / num_nodes

        return num_nodes, density, std_dev_norm

    except Exception as e:
        print(f"Error analyzing graph: {e}")
        return None, None, None

# ==========================================
# 3. RECOMMENDATION LOGIC
# ==========================================
def get_recommendation(nodes, density, degree_std_dev):
    print(f"\n--- 📊 GRAPH ANALYSIS ---")
    print(f"Nodes:           {nodes}")
    print(f"Density:         {density:.4f}")
    print(f"Degree StdDev:   {degree_std_dev:.4f}")
    
    # Rule 1: Small Graphs
    if nodes < 50:
        return "Bron-Kerbosch (Exact Algorithm)"
    
    # Rule 2: High Variance (Hubs)
    if degree_std_dev > 0.15:
        return "Greedy (Heuristic)"
        
    # Rule 3: Uniform Graphs
    if density > 0.70:
        if nodes > 500:
            return "Greedy (Heuristic)"
        else:
            return "Simulated Annealing"
            
    elif density < 0.10:
        return "Bron-Kerbosch (Exact Algorithm)"
        
    else:
        return "Local Search"

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("==========================================")
    print("   MAX CLIQUE ALGORITHM RECOMMENDER")
    print("==========================================")
    
    # 1. Get Input Path
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        target_file = input("Enter the path to your graph file: ").strip()
        # Remove quotes if user dragged and dropped file
        target_file = target_file.strip("'").strip('"')

    if not os.path.exists(target_file):
        print(f"❌ Error: File not found at {target_file}")
        sys.exit(1)

    # 2. Ensure .adj Format (Convert if necessary)
    adj_file = ensure_adj_file(target_file)
    
    if not adj_file:
        print("❌ Could not proceed with analysis due to conversion error.")
        sys.exit(1)

    # 3. Analyze
    n, d, s = analyze_graph_file(adj_file)
    
    if n is not None:
        # 4. Recommend
        algo = get_recommendation(n, d, s)
        print(f"------------------------------------------")
        print(f"✅ BEST ALGORITHM:  {algo}")
        print(f"------------------------------------------")
    else:
        print("Error reading the adjacency file.")