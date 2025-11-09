import os
import requests
import gzip
import bz2
import shutil
import networkx as nx
import urllib3
import pandas as pd
import zipfile
import tarfile
from tempfile import TemporaryDirectory
from scipy.io import mmread
from io import StringIO
import csv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def save_as_adjlist(G, output_path):
    """Save the graph in adjacency list format: node: neighbor1 neighbor2 ..."""
    with open(output_path, "w") as f:
        for node in sorted(G.nodes()):
            neighbors = " ".join(str(n) for n in sorted(G.neighbors(node)))
            f.write(f"{node}: {neighbors}\n")


def load_graph(path):
    """
    Load a graph from various formats (including compressed archives and DIMACS) 
    and return a NetworkX Graph.
    """
    fname = os.path.basename(path).lower()

    # Detect .tar.gz and .tgz first
    if fname.endswith(".tar.gz") or fname.endswith(".tgz"):
        ext = ".tar.gz"
    else:
        ext = os.path.splitext(path)[1].lower()

    # --- BZ2 ---
    if ext == ".bz2":
        print(f"📂 Loading .bz2: {fname}")
        with bz2.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip().splitlines()

        if any(line.lower().startswith("*vertices") for line in content[:10]):
            try:
                with bz2.open(path, "rt", encoding="utf-8", errors="ignore") as f:
                    G = nx.read_pajek(f)
                    return nx.Graph(G)
            except Exception as e:
                print(f"⚠️ Pajek read failed, trying edge list: {e}")

        edges = []
        for line in content:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    u, v = parts[:2]
                    edges.append((u, v))
                except Exception:
                    continue
        G = nx.Graph()
        G.add_edges_from(edges)
        return G

    # --- GZ ---
    elif ext == ".gz":
        print(f"📂 Loading .gz: {fname}")
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            G = nx.read_edgelist(f, comments="#", nodetype=str)
        return G

    # --- TAR / TAR.GZ / TGZ ---
    elif ext in [".tar", ".tar.gz", ".tgz"]:
        print(f"📦 Extracting .tar/.tgz: {fname}")
        with TemporaryDirectory() as tmpdir:
            with tarfile.open(path, "r:*") as tar:
                tar.extractall(tmpdir)

            extracted_files = []
            for root, dirs, files in os.walk(tmpdir):
                for file in files:
                    if not file.startswith('.') and '__MACOSX' not in root:
                        extracted_files.append(os.path.join(root, file))
            
            if not extracted_files:
                raise ValueError(f"No files found inside {path}")

            # Find the best graph file (or first one)
            graph_file = next((f for f in extracted_files if any(f.lower().endswith(ext) for ext in ['.tsv', '.txt', '.edges', '.net', '.graphml', '.gml', '.clq', '.csv', '.mtx'])), extracted_files[0])
            
            # --- FIX: Recursively call load_graph on the extracted file ---
            G = load_graph(graph_file)
            print(f"✅ Loaded unweighted graph from archive: {len(G.nodes())} nodes, {len(G.edges())} edges")
            return G

    # --- ZIP ---
    elif ext == ".zip":
        print(f"📦 Extracting .zip: {fname}")
        with TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(tmpdir)
            
            extracted_files = []
            for root, dirs, files in os.walk(tmpdir):
                for file in files:
                    if not file.startswith('.') and '__MACOSX' not in root:
                        extracted_files.append(os.path.join(root, file))
            
            if not extracted_files:
                raise ValueError(f"No files found inside {path}")
            
            # Recursively load the first extracted file
            return load_graph(extracted_files[0])
            
    # --- 7Z (requires py7zr) ---
    elif ext == ".7z":
        try:
            import py7zr
        except ImportError:
            raise ValueError("py7zr library required for .7z files. Install with: pip install py7zr")
        
        with TemporaryDirectory() as tmpdir:
            with py7zr.SevenZipFile(path, 'r') as archive:
                archive.extractall(tmpdir)

            extracted_files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if os.path.isfile(os.path.join(tmpdir, f))]
            if not extracted_files:
                raise ValueError(f"No files found inside {path}")

            # Determine output directory for saving .adj files (unique feature of first script)
            raw_dir = os.path.dirname(path)
            out_dir = raw_dir.replace("raw_graphs/small", "small_graphs").replace("raw_graphs/medium", "medium_graphs").replace("raw_graphs/large", "large_graphs")
            os.makedirs(out_dir, exist_ok=True)
            
            output_graphs = []
            for fpath in extracted_files:
                fname = os.path.basename(fpath)
                try:
                    G_sub = load_graph(fpath) # Recursive call
                    if len(G_sub.edges()) > 0:
                        out_name = f"{os.path.splitext(fname)[0]}.adj"
                        out_path = os.path.join(out_dir, out_name)
                        save_as_adjlist(G_sub, out_path)
                        print(f"✅ Extracted & saved: {out_path} ({len(G_sub.nodes())} nodes, {len(G_sub.edges())} edges)")
                        output_graphs.append(G_sub)
                except Exception as e:
                    print(f"⚠️ Skipping {fname}: {e}")
            
            if not output_graphs:
                raise ValueError(f"No valid graph files found inside {path}")
            return output_graphs[0] 

    # --- Matrix Market (MTX) ---
    elif ext == ".mtx":
        print(f"📂 Loading .mtx: {fname}")
        matrix = mmread(path)
        return nx.from_scipy_sparse_array(matrix)

    # --- CLQ (DIMACS format) ---
    elif ext == ".clq":
        print(f"📂 Loading .clq: {fname}")
        edges = []
        with open(path, "r") as f:
            for line in f:
                if line.startswith("e "):
                    parts = line.split()
                    if len(parts) >= 3:
                        edges.append((parts[1], parts[2])) 
        G = nx.Graph()
        G.add_edges_from(edges)
        return G

    # --- GraphML ---
    elif ext == ".graphml":
        print(f"📂 Loading .graphml: {fname}")
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
        data = (data
                .replace('attr.type="vector_float"', 'attr.type="string"')
                .replace('attr.type="vector_string"', 'attr.type="string"')
                .replace('attr.type="short"', 'attr.type="string"'))
        
        temp_file = StringIO(data)
        G_full = nx.read_graphml(temp_file)
        G = nx.Graph()
        G.add_edges_from(G_full.edges())
        return G

    # --- GML ---
    elif ext == ".gml":
        print(f"📂 Loading .gml: {fname}")
        G_full = nx.read_gml(path)
        G = nx.Graph()
        G.add_edges_from(G_full.edges())
        return G

    # --- CSV / TSV ---
    elif ext in [".csv", ".tsv"]:
        print(f"📂 Loading {ext}: {fname}")
        # Use pandas for generic delimited files
        if ext == ".tsv":
            # Using read_csv and specifying separator for TSV
            df = pd.read_csv(path, sep='\t', header=None, comment='#')
        else:
            df = pd.read_csv(path, header=None, comment='#')
            
        # Try to treat the first two columns as an edge list
        if df.shape[1] >= 2:
            edges = [(str(row[0]), str(row[1])) for index, row in df.iterrows()]
            G = nx.Graph()
            G.add_edges_from(edges)
            return G
        raise ValueError(f"CSV/TSV file {fname} is not a valid edge list.")

    # --- Pajek (.graph/.net) ---
    elif ext in [".graph", ".net"]:
        print(f"📂 Loading .graph/.net (Pajek): {fname}")
        G = nx.read_pajek(path)
        return nx.Graph(G)

    # --- Default edge list (.txt, .edges, etc.) ---
    elif ext in [".txt", ".edges"]:
         print(f"📂 Loading edge list: {fname}")
         # Attempt to read as an edgelist (will try to infer types)
         G = nx.read_edgelist(path, comments="#", nodetype=str)
         return G
    else:
        raise ValueError(f"Unsupported format: {ext}")


def convert_graph_to_adjlist(input_path, output_path):
    """Convert and save as adjacency list."""
    G = load_graph(input_path)
    
    # --- Final Node Relabeling ---
    mapping = {}
    for n in G.nodes():
        try:
            mapping[n] = int(n)
        except:
            mapping[n] = n 
    if len(mapping) > 0 and any(k != v for k, v in mapping.items()):
        G = nx.relabel_nodes(G, mapping)
        
    save_as_adjlist(G, output_path)
    print(f"✅ Saved adjacency list: {output_path} ({len(G.nodes())} nodes, {len(G.edges())} edges)")


# ---------------- DATASETS (Merged) ----------------
datasets = {
    "small": [
        ("holy", "https://graphchallenge.s3.amazonaws.com/synthetic/partitionchallenge/static/simulated_blockmodel_graph_50_nodes.tar.gz"),
        ("dolphin","https://sites.cc.gatech.edu/dimacs10/archive/data/clustering/dolphins.graph.bz2"),
        ("karate","https://sites.cc.gatech.edu/dimacs10/archive/data/clustering/karate.graph.bz2"),
        ("MANN_a9", "https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/MANN_a9.clq"),
        ("small100", "https://graphchallenge.s3.amazonaws.com/synthetic/partitionchallenge/static/simulated_blockmodel_graph_100_nodes.tar.gz"),
    ],
    "medium": [
        ("dimac-c125", "https://iridia.ulb.ac.be/~fmascia/files/DIMACS/C125.9.clq"),
        ("san200_0.9_1", "https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/san200_0.9_1.clq"),
        ("student_cooperation", "https://chatox.github.io/networks-science-course/practicum/data/student-cooperation.graphml"),
        ("brock200_2", "https://iridia.ulb.ac.be/~fmascia/files/DIMACS/brock200_2.clq"),
        ("adjnoun_graph", "https://sites.cc.gatech.edu/dimacs10/archive/data/clustering/adjnoun.graph.bz2"),
        ("C125.9","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/C125.9.clq"),
        ("c-fat200-1","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/c-fat200-1.clq"),
        ("brock200_4","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/brock200_4.clq"),
        ("brock200_3","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/brock200_3.clq"),
        ("brock200_1","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/brock200_1.clq"),
        ("MANN_a27","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/MANN_a27.clq"), 
        ("C250.9","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/C250.9.clq"),
        ("c-fat200-2","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/c-fat200-2.clq"),
        ("c-fat200-5","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/c-fat200-5.clq"),
        ("gen200_p0.9_44","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/gen200_p0.9_44.clq"),
        ("gen200_p0.9_55","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/gen200_p0.9_55.clq"),
        ("johnson16-2-4","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/johnson16-2-4.clq"),
        ("keller4_github","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/keller4.clq"),
        ("san200_0.7_1","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/san200_0.7_1.clq"),
        ("san200_0.7_2","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/san200_0.7_2.clq"),
    ],
    "large": [
        ("facebook_combined", "https://snap.stanford.edu/data/facebook_combined.txt.gz"),
        ("large500", "https://graphchallenge.s3.amazonaws.com/synthetic/partitionchallenge/static/simulated_blockmodel_graph_500_nodes.tar.gz"),
        ("C1000.9","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/C1000.9.clq"),
        ("C500.9","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/C500.9.clq"),
        ("DSJC1000_5","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/DSJC1000_5.clq"),
        ("DSJC500_5","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/DSJC500_5.clq"),
        ("brock400_1","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/brock400_1.clq"),
        ("brock400_2","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/brock400_2.clq"),
        ("brock400_3","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/brock400_3.clq"),
        ("brock400_4","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/brock400_4.clq"),
        ("brock800_1","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/brock800_1.clq"),
        ("brock800_2","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/brock800_2.clq"),
        ("brock800_3","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/brock800_3.clq"),
        ("brock800_4","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/brock800_4.clq"),
        ("c-fat500-1","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/c-fat500-1.clq"),
        ("c-fat500-10","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/c-fat500-10.clq"), # Note: Removed trailing '.' from name
        ("c-fat500-2","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/c-fat500-2.clq"),
        ("c-fat500-5","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/c-fat500-5.clq"),
        ("gen400_p0_9_55","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/gen400_p0.9_55.clq"), # Changed name slightly
        ("gen400_p0.9_55","https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/master/DIMACS/weighted/gen400_p0.9_55.clq"),
    ],
}

print("\n--- 🔄 Processing additional small graphs ---")

# Define directories for these new graphs
additional_raw_dir = "raw_graphs/small_additional"
output_dir = "small_graphs"
os.makedirs(additional_raw_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True) # Should already exist, but safe

karate_adj_path = os.path.join(output_dir, "karate_club.adj")
if os.path.exists(karate_adj_path):
    print("⚡ Skipping karate_club: already converted.")
else:
    try:
        print("🧘 Generating Karate Club graph using NetworkX...")
        G_karate = nx.karate_club_graph()
        # Convert string labels ("Mr. Hi", "Officer") to integers for consistency
        G_karate_int = nx.convert_node_labels_to_integers(G_karate, first_label=0)
        save_as_adjlist(G_karate_int, karate_adj_path)
        print(f"✅ Saved adjacency list: {karate_adj_path} ({len(G_karate_int.nodes())} nodes, {len(G_karate_int.edges())} edges)")
    except Exception as e:
        print(f"❌ Failed to generate karate_club: {e}")

# --- 2. Generate Les Misérables Graph ---
les_mis_adj_path = os.path.join(output_dir, "les_miserables.adj")
if os.path.exists(les_mis_adj_path):
    print("⚡ Skipping les_miserables: already converted.")
else:
    try:
        print("🇫🇷 Generating Les Misérables graph using NetworkX...")
        G_les_mis = nx.les_miserables_graph()
        # Convert string character names to integers
        G_les_mis_int = nx.convert_node_labels_to_integers(G_les_mis, first_label=0)
        save_as_adjlist(G_les_mis_int, les_mis_adj_path)
        print(f"✅ Saved adjacency list: {les_mis_adj_path} ({len(G_les_mis_int.nodes())} nodes, {len(G_les_mis_int.edges())} edges)")
    except Exception as e:
        print(f"❌ Failed to generate les_miserables: {e}")

# --- 3. Generate Florentine Families Graph ---
florentine_adj_path = os.path.join(output_dir, "florentine_families.adj")
if os.path.exists(florentine_adj_path):
    print("⚡ Skipping florentine_families: already converted.")
else:
    try:
        print("🇮🇹 Generating Florentine Families graph using NetworkX...")
        G_florentine = nx.florentine_families_graph()
        # Convert string family names to integers
        G_florentine_int = nx.convert_node_labels_to_integers(G_florentine, first_label=0)
        save_as_adjlist(G_florentine_int, florentine_adj_path)
        print(f"✅ Saved adjacency list: {florentine_adj_path} ({len(G_florentine_int.nodes())} nodes, {len(G_florentine_int.edges())} edges)")
    except Exception as e:
        print(f"❌ Failed to generate florentine_families: {e}")

# --- 4. Generate Davis Southern Women Graph ---
davis_adj_path = os.path.join(output_dir, "davis_southern_women.adj")
if os.path.exists(davis_adj_path):
    print("⚡ Skipping davis_southern_women: already converted.")
else:
    try:
        print("👩 Generating Davis Southern Women graph using NetworkX...")
        # This graph is originally bipartite, but we load it as a standard graph
        G_davis = nx.davis_southern_women_graph() 
        # Convert string names (women and events) to integers
        G_davis_int = nx.convert_node_labels_to_integers(G_davis, first_label=0)
        save_as_adjlist(G_davis_int, davis_adj_path)
        print(f"✅ Saved adjacency list: {davis_adj_path} ({len(G_davis_int.nodes())} nodes, {len(G_davis_int.edges())} edges)")
    except Exception as e:
        print(f"❌ Failed to generate davis_southern_women: {e}")

for size, graphs in datasets.items():
    raw_dir = f"raw_graphs/{size}"
    output_dir = f"{size}_graphs"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    for name, url in graphs:
        filename = os.path.basename(url)
        raw_path = os.path.join(raw_dir, filename)
        
        # Adjust output path to remove archive extension parts for cleaner ADJ name
        base_name = os.path.splitext(filename)[0].split(".tar")[0].split(".zip")[0].split(".7z")[0]
        adj_path = os.path.join(output_dir, base_name + ".adj")

        if os.path.exists(adj_path):
            print(f"⚡ Skipping {name}: already converted ({adj_path})")
            continue

        if not os.path.exists(raw_path):
            print(f"⬇️ Downloading {name} ...")
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                with requests.get(url, stream=True, verify=False, timeout=60, headers=headers) as r:
                    r.raise_for_status()
                    with open(raw_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                print(f"✅ Downloaded: {raw_path}")
            except requests.exceptions.RequestException as e:
                print(f"❌ Failed to download {name}: {e}")
                continue

        try:
            # Archives are handled by the load_graph function, 
            # which either returns a G or raises an error.
            if raw_path.endswith('.zip') or raw_path.endswith('.7z') or raw_path.endswith('.tar.gz') or raw_path.endswith('.tgz'):
                # For archives, we rely on load_graph's recursive call, which returns the G object.
                G = load_graph(raw_path) 
                
                # If an archive returns a graph, save it immediately (like the single file case)
                if G is not None:
                     # Use convert_graph_to_adjlist to apply relabeling and save
                    convert_graph_to_adjlist(raw_path, adj_path)
                
            else:
                # Direct conversion for single files
                convert_graph_to_adjlist(raw_path, adj_path)
                
        except Exception as e:
            print(f"❌ Failed to convert {name}: {e}")

# ---------------- MAIN EXECUTION LOOP ----------------
for size, graphs in datasets.items():
    raw_dir = f"raw_graphs/{size}"
    output_dir = f"{size}_graphs"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    for name, url in graphs:
        filename = os.path.basename(url)
        raw_path = os.path.join(raw_dir, filename)
        
        # Adjust output path to remove archive extension parts for cleaner ADJ name
        base_name = os.path.splitext(filename)[0].split(".tar")[0].split(".zip")[0].split(".7z")[0]
        adj_path = os.path.join(output_dir, base_name + ".adj")

        if os.path.exists(adj_path):
            print(f"⚡ Skipping {name}: already converted ({adj_path})")
            continue

        if not os.path.exists(raw_path):
            print(f"⬇️ Downloading {name} ...")
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                with requests.get(url, stream=True, verify=False, timeout=60, headers=headers) as r:
                    r.raise_for_status()
                    with open(raw_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                print(f"✅ Downloaded: {raw_path}")
            except requests.exceptions.RequestException as e:
                print(f"❌ Failed to download {name}: {e}")
                continue

        try:
            # Archives are handled by the load_graph function, 
            # which either returns a G or raises an error.
            if raw_path.endswith('.zip') or raw_path.endswith('.7z') or raw_path.endswith('.tar.gz') or raw_path.endswith('.tgz'):
                # For archives, we rely on load_graph's recursive call, which returns the G object.
                G = load_graph(raw_path) 
                
                # If an archive returns a graph, save it immediately (like the single file case)
                if G is not None:
                     # Use convert_graph_to_adjlist to apply relabeling and save
                    convert_graph_to_adjlist(raw_path, adj_path)
                
            else:
                # Direct conversion for single files
                convert_graph_to_adjlist(raw_path, adj_path)
                
        except Exception as e:
            print(f"❌ Failed to convert {name}: {e}")


# Cleanup raw directories
shutil.rmtree("raw_graphs", ignore_errors=True)
print("\n🎯 All requested graph datasets processed and raw files cleaned up.")
