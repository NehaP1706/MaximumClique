import time

def find_maximum_clique_from_adj(adj_matrix):
    """
    Greedy algorithm to find a (not necessarily optimal) maximum clique 
    from an adjacency matrix representation of an undirected graph.

    Parameters:
        adj_matrix (list[list[int]] or numpy.ndarray): 
            - Adjacency matrix representation of the graph
            - adj_matrix[i][j] = 1 if there is an edge between vertex i and vertex j
            - 0 otherwise

    Returns:
        best_clique (list[int]): List of vertex indices forming the largest clique found
        runtime (float): Time taken to execute the algorithm (in seconds)
    """

    # Record start time for performance measurement
    start_time = time.time()

    # Total number of vertices in the graph
    n = len(adj_matrix)

    # ---------------------------------------------------------------
    # STEP 1: Compute degrees and sort nodes in descending order of degree
    # ---------------------------------------------------------------
    # Degree of a vertex = sum of entries in its row (i.e., number of connections)
    degrees = [sum(adj_matrix[i]) for i in range(n)]

    # Sort vertices by their degree (higher-degree nodes are tried first)
    nodes_sorted = sorted(range(n), key=lambda x: degrees[x], reverse=True)

    # Initialize the best clique found so far as an empty list
    best_clique = []

    # ---------------------------------------------------------------
    # STEP 2: Try to greedily build a clique starting from each node
    # ---------------------------------------------------------------
    for node in nodes_sorted:
        # Start a new clique with the current node
        clique = [node]

        # Attempt to add other vertices to the clique
        for other in nodes_sorted:
            # Skip if we are comparing the same vertex
            if other == node:
                continue

            # Check if 'other' is connected to ALL current clique members
            # If yes, we can safely add it to the clique
            if all(adj_matrix[other][member] == 1 for member in clique):
                clique.append(other)

        # Update the best clique if we found a larger one
        if len(clique) > len(best_clique):
            best_clique = clique

    # Record end time
    end_time = time.time()

    # Return the best clique and runtime
    return best_clique, (end_time - start_time)


# ---------------------------------------------------------------
# MAIN FUNCTION — Example Test Run
# ---------------------------------------------------------------
if __name__ == "__main__":
    # Example adjacency matrix for a 5-vertex graph
    # Vertices: 0, 1, 2, 3, 4
    # Edges: (0,1), (1,2), (0,2), (2,3), (3,4), (1,3)
    adj_matrix = [
        [0, 1, 1, 0, 0],  # Node 0 connected to 1, 2
        [1, 0, 1, 1, 0],  # Node 1 connected to 0, 2, 3
        [1, 1, 0, 1, 0],  # Node 2 connected to 0, 1, 3
        [0, 1, 1, 0, 1],  # Node 3 connected to 1, 2, 4
        [0, 0, 0, 1, 0]   # Node 4 connected to 3
    ]

    # Run the greedy clique finder
    clique, runtime = find_maximum_clique_from_adj(adj_matrix)

    # Display results
    print(f"Greedy Clique Found: {clique}")
    print(f"Clique Size: {len(clique)}")
    print(f"Execution Time: {runtime:.6f} seconds")
