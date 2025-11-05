#bron_kerbosch.py
#By: Arushi Tandon

import time


def bron_kerbosch_with_pivot(G):
    """
    Bron-Kerbosch algorithm with pivoting for maximum clique detection.
    Implements the Bron-Kerbosch algorithm with pivoting for finding
    the maximum clique in an undirected graph.

    The Bron-Kerbosch algorithm is an exact, exponential-time backtracking
    algorithm that systematically explores all maximal cliques in a graph.
    With pivoting, it reduces the number of recursive calls by intelligently
    selecting pivot vertices to minimize the search space.

    Time Complexity: O(3^(n/3)) in worst case
    Space Complexity: O(n + m) for graph storage + O(n) for recursion stack

    where:
    n = |V| : number of vertices
    m = |E| : number of edges

    The algorithm guarantees finding the true maximum clique, making it suitable
    as a baseline for comparing heuristic approaches, though it becomes
    impractical for graphs with more than ~50-100 vertices.
    
    Parameters
    ----------
    G : dict
        Graph represented as an adjacency dictionary {node: {neighbors}}.
        Each node maps to a set of vertices it is connected to.

    Returns
    -------
    max_clique : set
        Set of vertices forming the maximum clique.
    runtime : float
        Time taken to execute the algorithm in seconds.

    Notes
    -----
    - Uses pivoting to reduce the number of recursive branches.
    - Explores all maximal cliques systematically via backtracking.
    - Deterministic and exhaustive - guarantees optimal solution.
    - Maintains three sets: R (current clique), P (candidates), X (excluded).
    """

    start_time = time.time()
    max_clique = set()

    def bron_kerbosch_recursive(R, P, X):
        """
        Recursive helper function for Bron-Kerbosch algorithm.

        Parameters
        ----------
        R : set
            Current clique being built.
        P : set
            Candidate vertices that can extend the clique.
        X : set
            Vertices already processed (excluded from further exploration).
        """
        nonlocal max_clique

        # Base case: if P and X are empty, R is a maximal clique
        if not P and not X:
            if len(R) > len(max_clique):
                max_clique = R.copy()
            return

        # Choose pivot vertex from P ∪ X to minimize branching
        # Select the vertex with maximum connections to P
        pivot_candidates = P | X
        if not pivot_candidates:
            return

        pivot = max(pivot_candidates, key=lambda u: len(P & G[u]))

        # Iterate over vertices in P that are NOT neighbors of the pivot
        # This reduces recursive calls by skipping branches the pivot would cover
        for v in list(P - G[pivot]):
            # Recursive call with updated sets:
            # - Add v to current clique (R ∪ {v})
            # - Restrict candidates to neighbors of v (P ∩ N(v))
            # - Restrict excluded to neighbors of v (X ∩ N(v))
            bron_kerbosch_recursive(
                R | {v},
                P & G[v],
                X & G[v]
            )

            # Move v from candidates to excluded
            P.remove(v)
            X.add(v)

    # Initialize with all vertices as candidates
    all_vertices = set(G.keys())
    bron_kerbosch_recursive(set(), all_vertices, set())

    runtime = time.time() - start_time
    return max_clique, runtime


def bron_kerbosch_basic(G):
    """
    Basic Bron-Kerbosch algorithm without pivoting (for comparison).

    Parameters
    ----------
    G : dict
        Graph represented as an adjacency dictionary {node: {neighbors}}.

    Returns
    -------
    max_clique : set
        Set of vertices forming the maximum clique.
    runtime : float
        Time taken to execute the algorithm in seconds.

    Notes
    -----
    - Simpler implementation without pivot optimization.
    - Explores more branches than the pivoting version.
    - Useful for understanding the core algorithm logic.
    """

    start_time = time.time()
    max_clique = set()

    def bron_kerbosch_recursive(R, P, X):
        """
        Recursive helper without pivoting.
        """
        nonlocal max_clique

        if not P and not X:
            if len(R) > len(max_clique):
                max_clique = R.copy()
            return

        # Process all vertices in P without pivot optimization
        for v in list(P):
            bron_kerbosch_recursive(
                R | {v},
                P & G[v],
                X & G[v]
            )
            P.remove(v)
            X.add(v)

    all_vertices = set(G.keys())
    bron_kerbosch_recursive(set(), all_vertices, set())

    runtime = time.time() - start_time
    return max_clique, runtime


if __name__ == "__main__":
    # Example test by defining a simple undirected graph as adjacency set.

    G = {
        #1: {2, 3},
        #2: {1, 3},
        #3: {1, 2, 4},
        #4: {3, 5, 6},
        #5: {4, 6},
        #6: {4, 5}
        # 1: {2, 3, 4, 5},
        # 2: {1, 3, 4, 5},
        # 3: {1, 2, 4, 5},
        # 4: {1, 2, 3, 5},
        # 5: {1, 2, 3, 4}
        1: {2, 5, 6},
        2: {1, 3, 7},
        3: {2, 4, 8},
        4: {3, 5, 9},
        5: {4, 1, 10},
        6: {1, 8, 9},
        7: {2, 9, 10},
        8: {3, 6, 10},
        9: {4, 6, 7},
        10: {5, 7, 8}
    }

    # Test with pivoting
    max_clique_pivot, runtime_pivot = bron_kerbosch_with_pivot(G)
    print(f"Bron-Kerbosch (with pivoting):")
    print(f"Maximum Clique: {max_clique_pivot}")
    print(f"Size: {len(max_clique_pivot)}, Time: {runtime_pivot:.6f} seconds")
    print()

    # Test without pivoting (basic version)
    max_clique_basic, runtime_basic = bron_kerbosch_basic(G)
    print(f"Bron-Kerbosch (basic, no pivoting):")
    print(f"Maximum Clique: {max_clique_basic}")
    print(f"Size: {len(max_clique_basic)}, Time: {runtime_basic:.6f} seconds")
    print()

    # Verify both find the same maximum clique size
    assert len(max_clique_pivot) == len(max_clique_basic), "Clique sizes should match"
    print("✓ Both implementations found the same maximum clique size")