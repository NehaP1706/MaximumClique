import random

def randomized_max_clique(graph, restarts=100):
    """
    Randomized Heuristic Algorithm for the Maximum Clique Problem
    -----------------------------------------------------------------------
    DESCRIPTION:
    This algorithm finds a large (not necessarily maximum) clique in an 
    undirected graph using randomized greedy construction with multiple restarts.
    
    The idea is:
      1. Randomly pick a starting vertex.
      2. Iteratively add randomly selected vertices that are connected to
         all vertices currently in the clique.
      3. Repeat the above steps multiple times (restarts).
      4. Keep track of the largest clique found across all iterations.

    -----------------------------------------------------------------------
    WHY IT IS A HEURISTIC:
    - The algorithm makes random choices (starting vertex, vertex addition order),
      so it doesn't guarantee finding the true maximum clique.
    - It explores the solution space stochastically (by chance),
      trading exactness for speed and scalability.
    - It often finds a *good enough* (near-optimal) clique quickly, 
      especially in dense graphs.

    -----------------------------------------------------------------------
    PARAMETERS:
    graph     : dict[int, set[int]]
                 Adjacency list representation of an undirected graph.
                 Example:
                     graph = {
                         0: {1, 2},
                         1: {0, 2},
                         2: {0, 1, 3},
                         3: {2}
                     }

    restarts  : int
                 Number of independent random restarts (default = 100)

    -----------------------------------------------------------------------
    RETURNS:
    best_clique : list[int]
                  Vertices forming the largest clique found.

    -----------------------------------------------------------------------
    COMPLEXITY ANALYSIS:
    Let n = number of vertices in the graph
        r = number of restarts (iterations)

    TIME COMPLEXITY:
      - Each restart may inspect up to O(n²) vertex pairs
        (checking adjacency when expanding clique)
      - Total: O(r * n²)

    SPACE COMPLEXITY:
      - The graph is stored as adjacency lists: O(n + m), where m = edges
      - Temporary sets (for clique and candidates): O(n)
      - Total: O(n + m)
    """

    best_clique = set()

    for _ in range(restarts):
        # Step 1: Random starting vertex
        start_vertex = random.choice(list(graph.nodes()))
        current_clique = {start_vertex}

        # Candidates: neighbors of the starting vertex
        candidates = set(graph[start_vertex])

        # Step 2: Expand the clique greedily but randomly
        while candidates:
            # Randomly pick one vertex from candidates
            v = random.choice(list(candidates))

            # Add vertex if it's connected to all current clique members
            if all(v in graph[u] for u in current_clique):
                current_clique.add(v)
                # New candidates must be connected to all vertices in clique
                candidates = candidates.intersection(graph[v])
            else:
                # Remove incompatible vertex
                candidates.remove(v)

        # Step 3: Update best clique if current one is larger
        if len(current_clique) > len(best_clique):
            best_clique = current_clique

    return len(best_clique), list(best_clique), None


# --------------------- DEMONSTRATION ---------------------
if __name__ == "__main__":
    # Example Graph (Undirected)
    example_graph = {
        0: {1, 2, 3},
        1: {0, 2},
        2: {0, 1, 3},
        3: {0, 2},
        4: {5},
        5: {4}
    }

    clique = randomized_max_clique(example_graph, restarts=200)
    print("Largest Clique Found:", clique)
    print("Clique Size:", len(clique))
