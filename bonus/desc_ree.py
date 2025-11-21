import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 1. DEFINE YOUR LOGIC (The "Oracle")
def recommend_algorithm(nodes, density, degree_std_dev):
    # Rule 1: Small Graphs
    if nodes < 50:
        return "Bron-Kerbosch"
    
    # Rule 2: High Variance (Hubs)
    if degree_std_dev > 0.15:
        return "Greedy"
        
    # Rule 3: Uniform Graphs (Low Variance)
    if density > 0.70:
        if nodes > 500:
            return "Greedy"
        else:
            return "Simulated Annealing"
            
    elif density < 0.10:
        return "Bron-Kerbosch"
        
    else:
        return "Local Search"

# 2. GENERATE SYNTHETIC DATA TO MIMIC THE RULES
# We create random points covering all possible scenarios
print("Generating synthetic data to map your logic...")
data = []
for _ in range(5000): # 5000 samples to ensure coverage
    n = np.random.randint(1, 1500)      # Nodes: 1 to 1500
    d = np.random.uniform(0, 1)         # Density: 0.0 to 1.0
    std = np.random.uniform(0, 0.5)     # StdDev: 0.0 to 0.5
    
    # Get the "Correct" answer based on your function
    label = recommend_algorithm(n, d, std)
    data.append([n, d, std, label])

df = pd.DataFrame(data, columns=['Nodes', 'Density', 'Degree_StdDev', 'Label'])

# 3. TRAIN THE TREE
# We force the tree to learn your exact rules
X = df[['Nodes', 'Density', 'Degree_StdDev']]
y = df['Label']

# Initialize tree (depth=4 is enough for your logic)
clf = DecisionTreeClassifier(random_state=42, max_depth=5)
clf.fit(X, y)

# 4. VISUALIZE
plt.figure(figsize=(20, 12))
plot_tree(clf, 
          feature_names=['Nodes', 'Density', 'Degree_StdDev'],
          class_names=clf.classes_,
          filled=True, 
          rounded=True,
          fontsize=12)

plt.title("Algorithm Recommendation Logic", fontsize=20, fontweight='bold')
plt.savefig("my_logic_tree.png", dpi=300, bbox_inches='tight')
print("✅ Decision Tree diagram saved as 'my_logic_tree.png'")
plt.show()