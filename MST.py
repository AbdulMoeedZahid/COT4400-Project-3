import json
from pathlib import Path

# --- 1. Union-Find Data Structure ---
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])  # Path compression
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
            return True
        return False

# --- 2. Kruskal's Algorithm (Now uses existing edges) ---
def build_mst(nodes, edges):
    n = len(nodes)
    uf = UnionFind(n)
    mst = []
    total_weight = 0.0

    # Sort the provided edges by weight
    sorted_edges = sorted(edges, key=lambda e: e.get("weight_miles", e.get("weight", 0)))

    # Build the tree
    for edge in sorted_edges:
        u = edge["u"]
        v = edge["v"]
        weight = edge.get("weight_miles", edge.get("weight", 0))

        if uf.union(u, v):
            mst.append({
                "u": u,
                "v": v,
                "u_name": nodes[u]["name"],
                "v_name": nodes[v]["name"],
                "weight_miles": round(weight, 4)
            })
            total_weight += weight
            
            # Stop early when we have V-1 edges
            if len(mst) == n - 1:
                break  
                
    return mst, total_weight

# --- 3. Main Execution (Loop through all graphs) ---
def main():
    input_dir = Path("generated_graphs")
    output_dir = Path("mst_outputs")
    
    # Create the output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    if not input_dir.exists():
        print(f"Error: Could not find {input_dir}")
        return

    # Loop through all 4 generated graph JSONs
    for input_file in input_dir.glob("*.json"):
        print(f"Loading {input_file.name}...")
        with open(input_file, "r") as f:
            data = json.load(f)
            
        mst_edges, total_weight = build_mst(data["nodes"], data["edges"])
        
        # Save the specific MST for this graph
        output_path = output_dir / f"{input_file.stem}_mst.json"
        
        # Keep the metadata intact so the visualizer creates nice titles
        payload = {
            "graph_name": f"{input_file.stem} (MST)",
            "metadata": data.get("metadata", {}),
            "nodes": data["nodes"], 
            "edges": mst_edges
        }
        
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)
            
        print(f"✅ Saved MST with {len(mst_edges)} edges to {output_path}\n")

if __name__ == "__main__":
    main()