import networkx as nx
import time

def find_maximum_clique(G):
    """
    Parameters:
        G (networkx.Graph): Input undirected graph
    Returns:
        clique (list[int]): List of vertex indices forming the maximum clique
        runtime (float): Time taken in seconds
    """

    start_time = time.time()  # Record the start time for runtime measurement

    # Step 1: Sort all nodes by degree (descending)
    # Nodes with more connections are more likely to belong to a large clique.
    nodes_sorted = sorted(G.nodes(), key=lambda x: G.degree[x], reverse=True)

    # Step 2: Initialize the best clique found so far (empty initially)
    best_clique = []

    # Step 3: Try to build a clique starting from each node (greedy trial)
    for node in nodes_sorted:
        # Start a new clique with this node
        clique = [node]

        # Step 4: Consider adding other nodes that appear later in the sorted list
        for other in nodes_sorted:
            # Skip if it's the same node
            if other == node:
                continue

            # Check if 'other' is connected to *all* current members of the clique
            # (i.e., if adding it would still make a complete subgraph)
            if all(G.has_edge(other, member) for member in clique):
                clique.append(other)

        # Step 5: Update best clique if this one is larger
        if len(clique) > len(best_clique):
            best_clique = clique

    end_time = time.time()  # Record end time

    # Return both the clique and total runtime
    return best_clique, (end_time - start_time)


# ---------------------------------------------
# MAIN FUNCTION — runs only when file is executed directly
# ---------------------------------------------
if __name__ == "__main__":
    # Example test (runs only when you execute this file directly)

    # Create a simple undirected graph
    G = nx.Graph()
    edges = [
        (1, 2), (2, 3), (1, 3),   # Clique of size 3: {1, 2, 3}
        (3, 4), (4, 5),           # Extra connections not forming larger clique
        (2, 4)
    ]
    G.add_edges_from(edges)

    # Run the greedy maximum clique finder
    clique, runtime = find_maximum_clique(G)

    # Print the results
    print(f"Greedy Clique Found: {clique}")
    print(f"Clique Size: {len(clique)}")
    print(f"Execution Time: {runtime:.6f} seconds")


# -------------------------------------------------
# 💡 COMPLEXITY ANALYSIS:
# -------------------------------------------------
# Let n = number of vertices, m = number of edges.
#
# Step 1: Sorting vertices by degree -> O(n log n)
# Step 2–4: Nested loops through nodes -> O(n^2)
#   - For each node, we check edges using 'all()' which in the worst case checks up to O(n)
#   - So the overall upper bound ≈ O(n^3) in dense graphs
#
# ✅ TIME COMPLEXITY: O(n^3) in worst case (dense graph)
# ✅ AVERAGE CASE (sparser graphs): closer to O(n^2)
#
# ✅ SPACE COMPLEXITY: O(n)
#   - We store 'nodes_sorted', 'clique', and 'best_clique' — all at most O(n)
#   - Graph storage (adjacency list) is handled by NetworkX internally as O(n + m)
#
# Hence, Total Space: O(n + m)
# -------------------------------------------------
