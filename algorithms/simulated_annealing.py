"""
simulated_annealing_clique_enhanced.py
--------------------------------------
Enhanced implementation of Simulated Annealing (SA) metaheuristic 
for the Maximum Clique Problem with optimizations and improvements.

Key Enhancements:
- Input validation for initial clique
- Incremental candidate set maintenance (faster)
- Multiple restart mechanism
- Swap move operations for better exploration
- Adaptive cooling schedule option
- Better parameter handling and defaults

Time Complexity: O(t * n) per restart (improved from O(t * n * k))
Space Complexity: O(n + m)

where:
t = max_iterations
n = |V| : number of vertices
m = |E| : number of edges

"""

import time
import random
import math
from typing import Dict, Set, List, Tuple, Optional


def validate_clique(G: Dict[int, Set[int]], clique: Set[int]) -> bool:
    """
    Validates that a set of vertices forms a valid clique.
    
    Parameters
    ----------
    G : dict
        Graph adjacency dictionary
    clique : set
        Set of vertices to validate
        
    Returns
    -------
    bool
        True if clique is valid, False otherwise
    """
    for u in clique:
        if u not in G:
            return False
        for v in clique:
            if u != v and v not in G[u]:
                return False
    return True


def get_initial_candidates(G: Dict[int, Set[int]], clique: Set[int]) -> Set[int]:
    """
    Find all vertices that can be added to the current clique.
    
    A vertex v can be added if it's connected to ALL vertices in the clique.
    
    Parameters
    ----------
    G : dict
        Graph adjacency dictionary
    clique : set
        Current clique
        
    Returns
    -------
    set
        Set of vertices that can extend the clique
    """
    if not clique:
        return set(G.keys())
    
    candidates = set(G.keys()) - clique
    valid_candidates = set()
    
    for v in candidates:
        if all(v in G[u] for u in clique):
            valid_candidates.add(v)
    
    return valid_candidates


def update_candidates_add(G: Dict[int, Set[int]], candidates: Set[int], 
                          new_vertex: int) -> Set[int]:
    """
    Update candidate set after adding a vertex to the clique.
    
    Only vertices connected to the new vertex can remain as candidates.
    
    Parameters
    ----------
    G : dict
        Graph adjacency dictionary
    candidates : set
        Current candidate set
    new_vertex : int
        Vertex that was just added to clique
        
    Returns
    -------
    set
        Updated candidate set
    """
    return candidates & G[new_vertex]


def update_candidates_remove(G: Dict[int, Set[int]], candidates: Set[int],
                             clique: Set[int], removed_vertex: int) -> Set[int]:
    """
    Update candidate set after removing a vertex from the clique.
    
    Vertices that were only blocked by the removed vertex can become candidates again.
    
    Parameters
    ----------
    G : dict
        Graph adjacency dictionary
    candidates : set
        Current candidate set
    clique : set
        Current clique (after removal)
    removed_vertex : int
        Vertex that was just removed
        
    Returns
    -------
    set
        Updated candidate set
    """
    # Vertices connected to removed_vertex might now be valid
    potential_new = (G[removed_vertex] - clique) - candidates
    
    for v in potential_new:
        # Check if v is connected to all remaining clique vertices
        if all(v in G[u] for u in clique):
            candidates.add(v)
    
    return candidates


def simulated_annealing_clique(G: Dict[int, Set[int]], 
                               start_clique: Set[int],
                               initial_temp: float = 1.0,
                               cooling_rate: float = 0.99,
                               max_iterations: int = 10000,
                               min_temp: float = 1e-10,
                               swap_probability: float = 0.3) -> Tuple[Set[int], float]:
    """
    Finds a large clique using Simulated Annealing with optimizations.

    Parameters
    ----------
    G : dict
        Graph represented as an adjacency dictionary {node: {neighbors}}.
        MUST be symmetric: if v in G[u], then u in G[v].
    start_clique : set
        Initial clique (seed vertices). Must be a valid clique.
    initial_temp : float, default=1.0
        Starting temperature. Higher allows more exploration.
    cooling_rate : float, default=0.99
        Temperature reduction factor (0 < rate < 1). 
        Typical range: 0.95-0.999
    max_iterations : int, default=10000
        Number of iterations per run.
    min_temp : float, default=1e-10
        Minimum temperature threshold.
    swap_probability : float, default=0.3
        Probability of attempting a swap move vs add/remove.

    Returns
    -------
    best_clique : set
        Largest clique found during the run.
    runtime : float
        Execution time in seconds.

    Raises
    ------
    ValueError
        If start_clique is not a valid clique or if G is empty.
    """
    
    if not G:
        raise ValueError("Graph G cannot be empty")
    
    if not validate_clique(G, start_clique):
        raise ValueError(f"start_clique {start_clique} is not a valid clique")
    
    start_time = time.time()
    
    # Initialize
    T = initial_temp
    current_clique = set(start_clique)
    best_clique = set(start_clique)
    
    # Incremental candidate maintenance
    candidates = get_initial_candidates(G, current_clique)
    
    # Track iterations without improvement for adaptive strategies
    no_improvement = 0
    
    for iteration in range(max_iterations):
        
        # Decide on move type
        use_swap = random.random() < swap_probability
        
        if use_swap and current_clique and candidates:
            # --- SWAP MOVE: Remove one vertex, add another ---
            # This helps escape local optima more effectively
            
            vertex_to_remove = random.choice(list(current_clique))
            new_clique = current_clique - {vertex_to_remove}
            
            # Update candidates after removal
            temp_candidates = update_candidates_remove(G, candidates.copy(), 
                                                       new_clique, vertex_to_remove)
            
            if temp_candidates:
                vertex_to_add = random.choice(list(temp_candidates))
                new_clique = new_clique | {vertex_to_add}
                new_candidates = update_candidates_add(G, temp_candidates, vertex_to_add)
                
                # Energy change for swap (usually neutral or small)
                delta_energy = len(new_clique) - len(current_clique)
                delta_energy = -delta_energy  # Convert to minimization (E = -|C|)
                
                move_type = "swap"
            else:
                # Swap failed, fall back to remove only
                new_clique = current_clique - {vertex_to_remove}
                new_candidates = update_candidates_remove(G, candidates.copy(),
                                                          new_clique, vertex_to_remove)
                delta_energy = 1  # Energy increased (clique shrunk)
                move_type = "remove"
        
        elif candidates:
            # --- ADD MOVE: Grow the clique ---
            vertex_to_add = random.choice(list(candidates))
            new_clique = current_clique | {vertex_to_add}
            new_candidates = update_candidates_add(G, candidates, vertex_to_add)
            
            delta_energy = -1  # Energy decreased (better)
            move_type = "add"
            
        elif current_clique:
            # --- REMOVE MOVE: Escape local optimum ---
            vertex_to_remove = random.choice(list(current_clique))
            new_clique = current_clique - {vertex_to_remove}
            new_candidates = update_candidates_remove(G, candidates.copy(),
                                                      new_clique, vertex_to_remove)
            
            delta_energy = 1  # Energy increased (worse)
            move_type = "remove"
            
        else:
            # Empty clique with no candidates - graph might be empty
            break
        
        # --- Acceptance Criteria ---
        accept = False
        
        if delta_energy <= 0:
            # Good move - always accept
            accept = True
        else:
            # Bad move - accept probabilistically
            if T > min_temp:
                acceptance_prob = math.exp(-delta_energy / T)
                accept = random.random() < acceptance_prob
        
        if accept:
            current_clique = new_clique
            candidates = new_candidates
            
            # Update best solution
            if len(current_clique) > len(best_clique):
                best_clique = set(current_clique)
                no_improvement = 0
            else:
                no_improvement += 1
        else:
            no_improvement += 1
        
        # --- Cooling Schedule ---
        T = max(T * cooling_rate, min_temp)
        
        # Optional: Reheat if stuck for too long (adaptive strategy)
        if no_improvement > max_iterations // 10:
            T = min(T * 1.5, initial_temp * 0.1)  # Partial reheat
            no_improvement = 0
    
    runtime = time.time() - start_time
    return best_clique, runtime


def simulated_annealing_with_restarts(G: Dict[int, Set[int]],
                                     initial_temp: float = 1.0,
                                     cooling_rate: float = 0.99,
                                     max_iterations: int = 10000,
                                     num_restarts: int = 5,
                                     restart_strategy: str = 'random',
                                     verbose: bool = True) -> Tuple[Set[int], float, List[Set[int]]]:
    """
    Run Simulated Annealing with multiple restarts for better global search.
    
    Parameters
    ----------
    G : dict
        Graph adjacency dictionary
    initial_temp : float
        Starting temperature for each restart
    cooling_rate : float
        Temperature reduction factor
    max_iterations : int
        Iterations per restart
    num_restarts : int, default=5
        Number of independent runs
    restart_strategy : str, default='random'
        Strategy for choosing start clique:
        - 'random': Random single vertex
        - 'best': Start from best found in previous restart
        - 'diverse': Try different starting vertices
    verbose : bool, default=True
        Print progress information
        
    Returns
    -------
    best_clique : set
        Best clique found across all restarts
    total_runtime : float
        Total execution time
    all_solutions : list
        List of best solutions from each restart
    """
    
    if not G:
        raise ValueError("Graph G cannot be empty")
    
    global_best = set()
    all_solutions = []
    total_start = time.time()
    
    vertices = list(G.keys())
    
    for restart in range(num_restarts):
        # Choose starting clique based on strategy
        if restart_strategy == 'random' or restart == 0:
            start_vertex = random.choice(vertices)
            start_clique = {start_vertex}
        elif restart_strategy == 'best' and global_best:
            start_clique = set(global_best)
        elif restart_strategy == 'diverse':
            start_vertex = vertices[restart % len(vertices)]
            start_clique = {start_vertex}
        else:
            start_vertex = random.choice(vertices)
            start_clique = {start_vertex}
        
        if verbose:
            print(f"Restart {restart + 1}/{num_restarts}: Starting from {start_clique}")
        
        # Run SA
        clique, runtime = simulated_annealing_clique(
            G, start_clique, initial_temp, cooling_rate, max_iterations
        )
        
        all_solutions.append(clique)
        
        if len(clique) > len(global_best):
            global_best = set(clique)
            if verbose:
                print(f"  → New best: size {len(clique)} in {runtime:.4f}s")
        elif verbose:
            print(f"  → Found size {len(clique)} in {runtime:.4f}s")
    
    total_runtime = time.time() - total_start
    
    if verbose:
        print(f"\nBest clique found: {global_best}")
        print(f"Size: {len(global_best)}")
        print(f"Total time: {total_runtime:.4f}s")
    
    return global_best, total_runtime, all_solutions


def auto_tune_parameters(G: Dict[int, Set[int]]) -> Dict[str, float]:
    """
    Suggest SA parameters based on graph characteristics.
    
    Parameters
    ----------
    G : dict
        Graph adjacency dictionary
        
    Returns
    -------
    dict
        Suggested parameters
    """
    n = len(G)
    m = sum(len(neighbors) for neighbors in G.values()) // 2
    density = (2 * m) / (n * (n - 1)) if n > 1 else 0
    
    # Heuristics based on graph size and density
    if n < 50:
        iterations = 5000
        cooling = 0.95
        restarts = 3
    elif n < 200:
        iterations = 10000
        cooling = 0.98
        restarts = 5
    else:
        iterations = 20000
        cooling = 0.99
        restarts = 10
    
    # Adjust for density
    if density > 0.5:  # Dense graph
        cooling = min(cooling + 0.005, 0.999)
    
    return {
        'initial_temp': 1.0,
        'cooling_rate': cooling,
        'max_iterations': iterations,
        'num_restarts': restarts
    }


# ============================================================================
# MAIN EXAMPLE AND TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Enhanced Simulated Annealing for Maximum Clique Problem")
    print("=" * 70)
    
    # Example 1: Simple test graph
    G = {
        1: {2, 3},
        2: {1, 3},
        3: {1, 2, 4},
        4: {3, 5, 6},
        5: {4, 6},
        6: {4, 5}
    }
    
    print("\nExample 1: Simple Graph")
    print(f"Graph: {G}")
    print(f"Vertices: {len(G)}, Edges: {sum(len(v) for v in G.values()) // 2}")
    
    # Get auto-tuned parameters
    params = auto_tune_parameters(G)
    print(f"\nAuto-tuned parameters: {params}")
    
    # Single run
    print("\n--- Single Run ---")
    start_clique = {1}
    best, runtime = simulated_annealing_clique(
        G, start_clique,
        initial_temp=1.0,
        cooling_rate=0.99,
        max_iterations=1000
    )
    print(f"Start: {start_clique}")
    print(f"Result: {best}, Size: {len(best)}, Time: {runtime:.4f}s")
    
    # Multiple restarts
    print("\n--- Multiple Restarts ---")
    best_global, total_time, all_sols = simulated_annealing_with_restarts(
        G,
        initial_temp=1.0,
        cooling_rate=0.99,
        max_iterations=1000,
        num_restarts=5,
        restart_strategy='random',
        verbose=True
    )
    
    # Example 2: Larger random graph
    print("\n" + "=" * 70)
    print("Example 2: Larger Random Graph")
    print("=" * 70)
    
    # Create a random graph with ~30% edge probability
    n_vertices = 30
    edge_prob = 0.3
    
    G_large = {i: set() for i in range(n_vertices)}
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            if random.random() < edge_prob:
                G_large[i].add(j)
                G_large[j].add(i)
    
    m_large = sum(len(v) for v in G_large.values()) // 2
    print(f"Vertices: {n_vertices}, Edges: {m_large}")
    print(f"Density: {(2 * m_large) / (n_vertices * (n_vertices - 1)):.3f}")
    
    params_large = auto_tune_parameters(G_large)
    print(f"Auto-tuned parameters: {params_large}")
    
    best_large, time_large, _ = simulated_annealing_with_restarts(
        G_large,
        **params_large,
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("Testing complete!")
    print("=" * 70)