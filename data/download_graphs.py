import os
import requests
import gzip
import shutil
import networkx as nx
import urllib3
import pandas as pd
import zipfile
from tempfile import TemporaryDirectory
from scipy.io import mmread

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def save_as_adjlist(G, output_path):
    """Save the graph in adjacency list format: node: neighbor1 neighbor2 ..."""
    with open(output_path, "w") as f:
        for node in sorted(G.nodes()):
            neighbors = " ".join(str(n) for n in sorted(G.neighbors(node)))
            f.write(f"{node}: {neighbors}\n")

def load_graph(path):
    """
    Load a graph from various formats into a NetworkX graph.
    Automatically handles GraphML files with unsupported attribute types
    (vector_float, vector_string, short) by converting them to strings.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            G = nx.read_edgelist(f, comments="#", nodetype=int)
    elif ext in [".txt", ".edges", ".clq"]:
        if ext == ".clq":
            # DIMACS .clq format
            try:
                # Try using NetworkX's read_edgelist with custom comments
                G = nx.read_edgelist(path, comments=['c', 'p'], 
                                    nodetype=int, 
                                    data=False)
            except:
                # Fallback: manual parsing
                edges = []
                with open(path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('e '):
                            parts = line.split()
                            if len(parts) >= 3:
                                try:
                                    edges.append((int(parts[1]), int(parts[2])))
                                except ValueError:
                                    continue
                G = nx.Graph()
                G.add_edges_from(edges)
        else:
            G = nx.read_edgelist(path, comments="#", nodetype=int)
    elif ext == ".graphml":
        # Clean unsupported attribute types on the fly
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
        data = (data
                .replace('attr.type="vector_float"', 'attr.type="string"')
                .replace('attr.type="vector_string"', 'attr.type="string"')
                .replace('attr.type="short"', 'attr.type="string"'))
        # Write to temporary in-memory file (StringIO) for NetworkX
        from io import StringIO
        temp_file = StringIO(data)
        G_full = nx.read_graphml(temp_file)
        G = nx.Graph()
        G.add_edges_from(G_full.edges())
    elif ext == ".gml":
        G_full = nx.read_gml(path)
        G = nx.Graph()
        G.add_edges_from(G_full.edges())
    elif ext == ".csv":
        df = pd.read_csv(path, index_col=0)
        G = nx.from_pandas_adjacency(df)
    elif ext == ".zip":
        with TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(tmpdir)

            # Recursively find all files (not just top-level)
            extracted_files = []
            for root, dirs, files in os.walk(tmpdir):
                for file in files:
                    # Skip hidden files and __MACOSX
                    if not file.startswith('.') and '__MACOSX' not in root:
                        extracted_files.append(os.path.join(root, file))
            
            print(f"🔍 Found {len(extracted_files)} files in {os.path.basename(path)}")
            for f in extracted_files:
                print(f"   - {os.path.basename(f)}")
            
            if not extracted_files:
                raise ValueError(f"No files found inside {path}")

            # Determine output directory based on the original zip's location
            raw_dir = os.path.dirname(path)
            if "small" in raw_dir:
                out_dir = raw_dir.replace("raw_graphs/small", "small_graphs")
            elif "medium" in raw_dir:
                out_dir = raw_dir.replace("raw_graphs/medium", "medium_graphs")
            elif "large" in raw_dir:
                out_dir = raw_dir.replace("raw_graphs/large", "large_graphs")
            else:
                out_dir = os.path.dirname(path)
            
            os.makedirs(out_dir, exist_ok=True)

            output_graphs = []
            for fpath in extracted_files:
                fname = os.path.basename(fpath)
                base, subext = os.path.splitext(fname)
                subext = subext.lower()

                try:
                    if subext == ".mtx":
                        from scipy.io import mmread
                        matrix = mmread(fpath)
                        G_sub = nx.from_scipy_sparse_array(matrix)
                        out_name = f"{base}.adj"
                    elif subext in [".edges", ".txt"]:
                        G_sub = nx.read_edgelist(fpath, nodetype=str, comments='#')
                        out_name = f"{fname}.adj"

                    # Only add to output if graph has edges
                    if len(G_sub.edges()) > 0:
                        output_graphs.append((out_name, G_sub))
                    else:
                        print(f"⚠️ Skipping {fname}: 0 edges")

                except Exception as e:
                    print(f"⚠️ Skipping {fname} in {os.path.basename(path)}: {e}")

            if not output_graphs:
                raise ValueError(f"No valid graph files found inside {path}")

            # Write separate .adj files
            for out_name, G in output_graphs:
                out_path = os.path.join(out_dir, out_name)
                save_as_adjlist(G, out_path)
                print(f"✅ Extracted & saved: {out_path} ({len(G.nodes())} nodes, {len(G.edges())} edges)")

            # Return the first graph (for compatibility)
            return output_graphs[0][1]
    elif ext == ".7z":
        # Handle .7z files (requires py7zr library)
        try:
            import py7zr
        except ImportError:
            raise ValueError("py7zr library required for .7z files. Install with: pip install py7zr")
        
        with TemporaryDirectory() as tmpdir:
            with py7zr.SevenZipFile(path, 'r') as archive:
                archive.extractall(tmpdir)

            extracted_files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) 
                             if os.path.isfile(os.path.join(tmpdir, f))]
            if not extracted_files:
                raise ValueError(f"No files found inside {path}")

            # Determine output directory
            raw_dir = os.path.dirname(path)
            if "small" in raw_dir:
                out_dir = raw_dir.replace("raw_graphs/small", "small_graphs")
            elif "medium" in raw_dir:
                out_dir = raw_dir.replace("raw_graphs/medium", "medium_graphs")
            elif "large" in raw_dir:
                out_dir = raw_dir.replace("raw_graphs/large", "large_graphs")
            else:
                out_dir = os.path.dirname(path)
            
            os.makedirs(out_dir, exist_ok=True)
            # Process each extracted file
            for fpath in extracted_files:
                fname = os.path.basename(fpath)
                try:
                    # Try as edgelist
                    G_sub = nx.read_edgelist(fpath, nodetype=str)
                    out_name = f"{fname}.adj"
                    out_path = os.path.join(out_dir, out_name)
                    save_as_adjlist(G_sub, out_path)
                    print(f"✅ Extracted & saved: {out_path} ({len(G_sub.nodes())} nodes, {len(G_sub.edges())} edges)")
                    return G_sub
                except Exception as e:
                    print(f"⚠️ Skipping {fname} in {os.path.basename(path)}: {e}")
            
            raise ValueError(f"No valid graph files found inside {path}")
    elif ext == ".mtx":
        # MatrixMarket format
        matrix = mmread(path)
        G = nx.from_scipy_sparse_array(matrix)
    else:
        raise ValueError(f"Unsupported format: {ext}")

    # Relabel nodes as integers if possible
    mapping = {}
    for n in G.nodes():
        try:
            mapping[n] = int(n)
        except:
            mapping[n] = n
    if mapping:
        G = nx.relabel_nodes(G, mapping)

    return G

def convert_graph_to_adjlist(input_path, output_path):
    """Convert a single graph to adjacency list format."""
    G = load_graph(input_path)
    save_as_adjlist(G, output_path)
    print(f"✅ Saved adjacency list: {output_path} ({len(G.nodes())} nodes, {len(G.edges())} edges)")

datasets = {
    "small": [
        # ("karate_club", "https://github.com/mlabonne/graph-datasets/blob/main/node_classification/karate-club/karate.gml"),
        # ("game_of_thrones", "https://chatox.github.io/networks-science-course/practicum/data/game-of-thrones/"),
        # ("marvel_heroes", "https://chatox.github.io/networks-science-course/practicum/data/marvel-hero.csv"),
        # ("flavor_network", "https://chatox.github.io/networks-science-course/practicum/data/flavor-network/"),
        # ("ogdos_100", "<link-to-OGDOS-graph-~100nodes>"),
    ],
    "medium": [
        # ("adjnoun_adj", "http://statml.com/download/data_7z/misc/adjnoun_adjacency.7z"),
        ("dimac-c125", "https://iridia.ulb.ac.be/~fmascia/files/DIMACS/C125.9.clq"),
        ("keller-4", "https://iridia.ulb.ac.be/~fmascia/files/DIMACS/keller4.clq"),
        ("student_cooperation", "https://chatox.github.io/networks-science-course/practicum/data/student-cooperation.graphml"),
        ("brock200_2", "https://iridia.ulb.ac.be/~fmascia/files/DIMACS/brock200_2.clq"),
        # ("tscc", "https://statml.com/download/data_7z/tscc/scc_enron-only.7z")
    ],
    "large": [
        ("facebook_combined", "https://snap.stanford.edu/data/facebook_combined.txt.gz"),
        # should work TT - ("cora_content", "https://linqs-data.soe.ucsc.edu/public/datasets/cora/cora.zip"),
        # commented because these graphs have their edges in lakhs
        # ("web-Google", "https://snap.stanford.edu/data/web-Google.txt.gz"),
        # ("amazon0601", "https://snap.stanford.edu/data/amazon0601.txt.gz"),
    ],
}

for size, graphs in datasets.items():
    raw_dir = f"raw_graphs/{size}"
    output_dir = f"{size}_graphs"  # small_graphs or large_graphs

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    for name, url in graphs:
        filename = os.path.basename(url)
        raw_path = os.path.join(raw_dir, filename)
        adj_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".adj")

        # Skip if already converted
        if os.path.exists(adj_path):
            print(f"⚡ Skipping {name}: already converted ({adj_path})")
            continue

        # Download file if not present
        if not os.path.exists(raw_path):
            print(f"⬇️ Downloading {name} ...")
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                with requests.get(url, stream=True, verify=False, timeout=60, headers=headers) as r:
                    r.raise_for_status()
                    with open(raw_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                print(f"✅ Downloaded: {raw_path}")
            except requests.exceptions.RequestException as e:
                print(f"❌ Failed to download {name}: {e}")
                continue

        # Convert to adjacency list
        try:
            # Check if it's a zip file - if so, load_graph already handles everything
            if raw_path.endswith('.zip') or raw_path.endswith('.7z'):
                load_graph(raw_path)
            else:
                convert_graph_to_adjlist(raw_path, adj_path)
        except Exception as e:
            print(f"❌ Failed to convert {name}: {e}")

# Cleanup raw directories
shutil.rmtree("raw_graphs")
print("🎯 All requested graph datasets processed.")
