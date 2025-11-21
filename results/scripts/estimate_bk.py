import os
import time
import numpy as np
from itertools import combinations
import random
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import sys 
sys.path.append(os.path.abspath("../../algorithms"))
from bron_kerbosch import bron_kerbosch_with_pivot

def generate_random_graph(num_nodes, edge_probability=0.3):
    """Generate random undirected graph"""
    G = {i: set() for i in range(num_nodes)}
    for i, j in combinations(range(num_nodes), 2):
        if random.random() < edge_probability:
            G[i].add(j)
            G[j].add(i)
    return G

def measure_runtimes():
    """Test on various graph sizes"""
    results = []
    
    # Test different sizes
    for num_nodes in [200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600]:
        for density in [0.2, 0.4, 0.6]:
            G = generate_random_graph(num_nodes, density)
            num_edges = sum(len(neighbors) for neighbors in G.values()) // 2
            
            try:
                start = time.time()
                max_clique, _ = bron_kerbosch_with_pivot(G)
                runtime = time.time() - start
                
                results.append({
                    'nodes': num_nodes,
                    'edges': num_edges,
                    'density': density,
                    'runtime': runtime,
                    'clique_size': len(max_clique)
                })
                print(f"n={num_nodes}, e={num_edges}, d={density:.1f} → {runtime:.4f}s")
            except:
                print(f"n={num_nodes} timeout or error")
                break
    
    return results

# Run benchmarks (this takes a while)
results = measure_runtimes()

# Extract data
nodes_list = np.array([r['nodes'] for r in results])
edges_list = np.array([r['edges'] for r in results])
runtimes = np.array([r['runtime'] for r in results])

# Try different models
def model_exponential_n(n, a, b):
    """Exponential in nodes: t = a * 3^(n/b)"""
    return a * (3 ** (n / b))

def model_polynomial_ne(x, a, b, c):
    """Polynomial in nodes and edges: t = a*n^b + c*e"""
    n, e = x
    return a * (n ** b) + c * e

# Fit exponential model (simpler)
try:
    popt, _ = curve_fit(model_exponential_n, nodes_list, runtimes, p0=[0.001, 3], maxfev=5000)
    a_exp, b_exp = popt
    print(f"Exponential fit: runtime ≈ {a_exp:.6f} * 3^(n/{b_exp:.2f})")
except:
    print("Exponential fit failed")

# Fit polynomial model
try:
    popt, _ = curve_fit(lambda x: model_polynomial_ne(x, *popt[:3]), 
                        (nodes_list, edges_list), runtimes, p0=[0.01, 2, 0.0001], maxfev=5000)
    a_poly, b_poly, c_poly = popt
    print(f"Polynomial fit: runtime ≈ {a_poly:.6f}*n^{b_poly:.2f} + {c_poly:.8f}*e")
except:
    print("Polynomial fit failed")

# Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.scatter(nodes_list, runtimes, label='Actual', color='red', s=100)
predicted = model_exponential_n(nodes_list, a_exp, b_exp)
plt.scatter(nodes_list, predicted, label='Exponential fit', color='blue', alpha=0.6)
plt.xlabel('Number of Nodes')
plt.ylabel('Runtime (seconds)')
plt.legend()
plt.yscale('log')
plt.show()

def estimate_bk_runtime(num_nodes, num_edges):
    """Estimate Bron-Kerbosch runtime using fitted formula"""
    
    # Using exponential model (adjust a, b based on your fitting)
    a = 0.000001  # From curve_fit result
    b = 3.2       # From curve_fit result
    
    estimated_time = a * (3 ** (num_nodes / b))
    
    print(f"Graph: {num_nodes} nodes, {num_edges} edges")
    print(f"Estimated runtime: {estimated_time:.2f} seconds")
    
    if estimated_time > 3600:
        print(f"⚠️ WARNING: Estimated {estimated_time/3600:.1f} hours!")
    elif estimated_time > 60:
        print(f"⚠️ This will take ~{estimated_time/60:.1f} minutes")
    
    return estimated_time

# Before running Bron-Kerbosch, estimate:
num_nodes = len(G_dict)
num_edges = sum(len(neighbors) for neighbors in G_dict.values()) // 2

estimate_bk_runtime(num_nodes, num_edges)
