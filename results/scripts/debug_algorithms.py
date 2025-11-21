import sys
import os

# 1. Setup paths (same as your main script)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
algorithms_path = os.path.join(project_root, 'algorithms')
sys.path.insert(0, algorithms_path)

print(f"Loading algorithms from: {algorithms_path}")

try:
    from bron_kerbosch import bron_kerbosch_with_pivot
    from greedy import find_maximum_clique_from_dict
    print("✔ Imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# 2. Create a simple dummy graph (Triangle: 1-2-3)
# Conforming to the dict-of-sets structure we used in the loader
test_graph = {
    1: {2, 3},
    2: {1, 3},
    3: {1, 2}
}

print("\n=== TEST 1: Bron-Kerbosch ===")
try:
    result = bron_kerbosch_with_pivot(test_graph)
    print(f"✔ Success! Result: {result}")
except Exception as e:
    print(f"❌ CRASHED:")
    import traceback
    traceback.print_exc()

print("\n=== TEST 2: Greedy ===")
try:
    result = find_maximum_clique_from_dict(test_graph)
    print(f"✔ Success! Result: {result}")
except Exception as e:
    print(f"❌ CRASHED:")
    import traceback
    traceback.print_exc()