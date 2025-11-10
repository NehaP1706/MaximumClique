import sys
import random
import time

def local_random_clique(G, restarts=10):
    """
    LOCAL RANDOMIZED ALGORITHM FOR MAXIMUM CLIQUE

    This algorithm attempts to find a large clique in an undirected graph
    using a local randomized approach. It starts from a random vertex and
    expands the clique by randomly selecting adjacent vertices that maintain
    the clique property.

    PARAMETERS:
    - G: dict
        An adjacency list representation of the graph where keys are vertex
        identifiers and values are sets of adjacent vertices.
    - restarts: int
        The number of random restarts to perform to increase the chance of
        finding a larger clique.
    RETURNS:
    - best_clique: set
        The largest clique found during the random restarts.
    - runtime: float
        Time taken to execute the algorithm.

    Same time & space complexity as the local search heuristic algorithm.
    """
    
    start_time = time.time()
    best_clique = set()
    
    for _ in range(restarts):
        # Step 1: Random starting vertex
        start_vertex = random.choice(list(G.keys()))
        clique = {start_vertex}
        
        # Candidates: neighbors of the starting vertex
        candidates = set(G[start_vertex])
        
        # Main improvement loop
        improved = True
        n = len(G)
        add_count = 0
        swap_count = 0
        remove_expand_count = 0
        
        while improved:
            improved = False
            
            # ADD: randomly try to add a vertex
            if candidates and add_count < n:
                add_count += 1
                # Randomly shuffle candidates
                add_candidates = list(candidates)
                random.shuffle(add_candidates)
                
                for v in add_candidates:
                    # Check if vertex v is connected to all members of the current clique
                    if all(v in G[u] for u in clique):
                        clique.add(v)
                        # Update candidates: intersection with neighbors of v
                        candidates &= G[v] - clique
                        improved = True
                        break
            
            if improved:
                continue
            
            # SWAP: try replacing a random vertex
            if clique and swap_count < n:
                swap_count += 1
                # Randomly shuffle clique members
                clique_list = list(clique)
                random.shuffle(clique_list)
                
                for u in clique_list:
                    # Potential swap-ins: vertices connected to all except 'u'
                    swap_candidates = [
                        v for v in G
                        if v not in clique and all(x == u or v in G[x] for x in clique)
                    ]
                    
                    if swap_candidates:
                        # Randomly shuffle swap candidates
                        random.shuffle(swap_candidates)
                        
                        for v in swap_candidates:
                            new_clique = (clique - {u}) | {v}
                            
                            # Check if the result is a clique
                            if all((a in G[b]) for a in new_clique for b in new_clique if a != b):
                                clique = new_clique
                                # Update candidates
                                candidates = set.intersection(*(G[x] for x in clique)) - clique
                                improved = True
                                break
                    
                    if improved:
                        break
            
            if improved:
                continue
            
            # REMOVE + EXPAND: Remove a vertex and try for better
            if clique and remove_expand_count < n:
                remove_expand_count += 1
                # Randomly shuffle clique members
                clique_list = list(clique)
                random.shuffle(clique_list)
                
                for u in clique_list:
                    reduced = clique - {u}
                    
                    # Potential expansions: Candidates that can connect to all reduced members
                    expand_candidates = [
                        v for v in candidates
                        if all(v in G[x] for x in reduced)
                    ]
                    
                    if expand_candidates:
                        # Randomly shuffle expand candidates
                        random.shuffle(expand_candidates)
                        
                        for v in expand_candidates:
                            new_clique = reduced | {v}
                            
                            if len(new_clique) > len(clique):
                                clique = new_clique
                                # Update candidates
                                candidates = set.intersection(*(G[x] for x in clique)) - clique
                                improved = True
                                break
                    
                    if improved:
                        break
            
            if improved:
                continue
        
        # Update best clique if current one is larger
        if len(clique) > len(best_clique):
            best_clique = clique
    
    runtime = time.time() - start_time
    return best_clique, runtime

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

    clique, runtime = local_random_clique(example_graph, restarts=200)
    print("Largest Clique Found:", clique)
    print("Clique Size:", len(clique))
    print(f"Runtime: {runtime:.4f} seconds")