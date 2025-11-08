import os
import requests
import gzip
import shutil
import networkx as nx
import urllib3
import pandas as pd

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
        # ("adjnoun_adj", "https://networkrepository.com/adjnoun-adjacency.php"),
        # ("game_of_thrones", "https://chatox.github.io/networks-science-course/practicum/data/game-of-thrones/"),
        # ("marvel_heroes", "https://chatox.github.io/networks-science-course/practicum/data/marvel-hero.csv"),
         ("student_cooperation", "https://chatox.github.io/networks-science-course/practicum/data/student-cooperation.graphml"),
        # ("flavor_network", "https://chatox.github.io/networks-science-course/practicum/data/flavor-network/"),
        # ("hamsterster", "https://networkrepository.com/soc-hamsterster.php"),
        # ("ogdos_100", "<link-to-OGDOS-graph-~100nodes>"),
        # ("brock200_2", "https://turing.cs.hbg.psu.edu/txn131/graphs/brock200_2.clq"),
        # ("c-fat200-5", "https://turing.cs.hbg.psu.edu/txn131/graphs/c-fat200-5.clq"),
    ],
    "large": [
        ("facebook_combined", "https://snap.stanford.edu/data/facebook_combined.txt.gz"),
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
                with requests.get(url, stream=True, verify=False, timeout=60) as r:
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
            convert_graph_to_adjlist(raw_path, adj_path)
        except Exception as e:
            print(f"❌ Failed to convert {name}: {e}")

# Cleanup raw directories
shutil.rmtree("raw_graphs")
print("🎯 All requested graph datasets processed.")
