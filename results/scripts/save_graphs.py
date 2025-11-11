import os
import networkx as nx
import matplotlib.pyplot as plt


def load_adj_file(path):
    """
    Loads a graph stored in adjacency list format (.adj) and
    returns a dictionary {node: set(neighbors)}.
    """
    adj_dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            node, neighbors = line.strip().split(":", 1)
            node = node.strip()
            neighbor_set = set(neighbors.strip().split()) if neighbors.strip() else set()
            adj_dict[node] = neighbor_set
    return adj_dict


def visualize_graph(G_nx, title, output_path, graph_size):
    """
    Creates optimized visualization based on graph size.
    """
    num_nodes = G_nx.number_of_nodes()
    num_edges = G_nx.number_of_edges()
    
    # Choose figure size and layout based on graph size
    if num_nodes <= 50:
        # Small graphs: detailed view with labels
        fig, ax = plt.subplots(figsize=(12, 10))
        pos = nx.spring_layout(G_nx, k=1/num_nodes**0.5, iterations=50, seed=42)
        
        nx.draw_networkx_nodes(G_nx, pos, node_color="lightblue", 
                              node_size=500, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G_nx, pos, alpha=0.4, width=1.5, ax=ax)
        nx.draw_networkx_labels(G_nx, pos, font_size=9, ax=ax)
        
    elif num_nodes <= 200:
        # Medium graphs: balanced view
        fig, ax = plt.subplots(figsize=(14, 12))
        pos = nx.spring_layout(G_nx, k=2/num_nodes**0.5, iterations=50, seed=42)
        
        # Color by degree
        degrees = dict(G_nx.degree())
        node_colors = [degrees[node] for node in G_nx.nodes()]
        
        nx.draw_networkx_nodes(G_nx, pos, node_color=node_colors, 
                              node_size=200, alpha=0.7, cmap='viridis', ax=ax)
        nx.draw_networkx_edges(G_nx, pos, alpha=0.2, width=0.8, ax=ax)
        
    else:
        # Large graphs: focus on structure
        fig, ax = plt.subplots(figsize=(16, 14))
        
        # Use faster layout algorithm for large graphs
        pos = nx.kamada_kawai_layout(G_nx)
        
        # Color by degree centrality
        degrees = dict(G_nx.degree())
        node_colors = [degrees[node] for node in G_nx.nodes()]
        node_sizes = [20 + degrees[node] * 5 for node in G_nx.nodes()]
        
        nx.draw_networkx_nodes(G_nx, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.6, 
                              cmap='plasma', ax=ax, linewidths=0)
        nx.draw_networkx_edges(G_nx, pos, alpha=0.15, width=0.5, ax=ax)
    
    ax.set_title(f"{title}\n{num_nodes} vertices, {num_edges} edges", 
                fontsize=14, fontweight='bold')
    ax.axis("off")
    plt.tight_layout()
    
    # Save with high DPI for clarity
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def visualize_graph_statistics(G_nx, title, output_path):
    """
    Creates a statistical overview visualization for large graphs.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Degree distribution
    degrees = [d for n, d in G_nx.degree()]
    axes[0, 0].hist(degrees, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Degree Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Degree')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Clustering coefficient distribution
    clustering = list(nx.clustering(G_nx).values())
    axes[0, 1].hist(clustering, bins=30, color='coral', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Clustering Coefficient Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Clustering Coefficient')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Top 20 nodes by degree (bar chart)
    top_nodes = sorted(G_nx.degree(), key=lambda x: x[1], reverse=True)[:20]
    nodes, node_degrees = zip(*top_nodes)
    axes[1, 0].barh(range(len(nodes)), node_degrees, color='mediumseagreen', alpha=0.7)
    axes[1, 0].set_yticks(range(len(nodes)))
    axes[1, 0].set_yticklabels(nodes, fontsize=8)
    axes[1, 0].set_title('Top 20 Nodes by Degree', fontweight='bold')
    axes[1, 0].set_xlabel('Degree')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    axes[1, 0].invert_yaxis()
    
    # Graph statistics text
    avg_degree = sum(degrees) / len(degrees)
    avg_clustering = sum(clustering) / len(clustering)
    density = nx.density(G_nx)
    
    stats_text = f"""
Graph Statistics:

• Nodes: {G_nx.number_of_nodes()}
• Edges: {G_nx.number_of_edges()}
• Average Degree: {avg_degree:.2f}
• Max Degree: {max(degrees)}
• Min Degree: {min(degrees)}
• Density: {density:.4f}
• Avg Clustering: {avg_clustering:.4f}
• Connected: {nx.is_connected(G_nx)}
    """
    
    axes[1, 1].text(0.1, 0.5, stats_text.strip(), fontsize=11, 
                   verticalalignment='center', family='monospace')
    axes[1, 1].axis('off')
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def process_graphs(size_category):
    """
    Process all .adj files for a given size category.
    Creates both graph visualizations and statistics.
    """
    input_dir = f"../../data/{size_category}_graphs"
    output_dir = f"../graphs/{size_category}_graphs"
    
    os.makedirs(output_dir, exist_ok=True)
    
    adj_files = [f for f in os.listdir(input_dir) if f.endswith(".adj")]
    
    if not adj_files:
        print(f"⚠️  No .adj files found in {input_dir}")
        return
    
    print(f"\n📂 Processing {len(adj_files)} graphs from {input_dir}")
    
    for adj_file in adj_files:
        adj_path = os.path.join(input_dir, adj_file)
        
        # Load graph
        G_dict = load_adj_file(adj_path)
        print(f"  Loading {adj_file}: {len(G_dict)} vertices", end="")
        
        # Convert to NetworkX graph
        G_nx = nx.Graph()
        for node, nbrs in G_dict.items():
            for nbr in nbrs:
                G_nx.add_edge(node, nbr)
        
        base_name = adj_file.replace(".adj", "")
        
        # Create main visualization
        output_path = os.path.join(output_dir, f"{base_name}.png")
        visualize_graph(G_nx, base_name, output_path, size_category)
        print(f" → Saved visualization")
        
        # For medium and large graphs, also create statistics visualization
        if len(G_dict) > 50:
            stats_path = os.path.join(output_dir, f"{base_name}_stats.png")
            visualize_graph_statistics(G_nx, base_name, stats_path)
            print(f"           → Saved statistics")
    
    print(f"✅ Completed processing {size_category} graphs\n")


if __name__ == "__main__":
    for size in ["small", "medium", "large"]:
        process_graphs(size)
    
    print("🎉 All graphs processed successfully!")