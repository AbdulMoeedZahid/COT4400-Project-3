"""
Dijkstra's shortest path algorithm for the South Tampa navigation graph.
Loads the 4 pre-built graph JSONs from generated_graphs/ and runs Dijkstra between two user-chosen destinations.
"""

import heapq
import json
import math
import time
import tracemalloc
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


GRAPHS_DIR  = Path("generated_graphs")
GRAPH_FILES = ["small_sparse", "small_dense", "large_sparse", "large_dense"]


# Load graph from JSON
def load_graph(name):
    with (GRAPHS_DIR / f"{name}.json").open() as f:
        data = json.load(f)
    nodes = {n["id"]: n for n in data["nodes"]}
    adj = defaultdict(list)
    for e in data["edges"]:
        adj[e["u"]].append((e["v"], e["weight_miles"]))
        adj[e["v"]].append((e["u"], e["weight_miles"]))
    return nodes, adj, data


# Dijkstra's algorithm
def dijkstra(adj, num_nodes, source):
    dist    = {i: math.inf for i in range(num_nodes)}
    prev    = {i: -1       for i in range(num_nodes)}
    visited = set()
    dist[source] = 0
    heap = [(0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        for v, w in adj[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                prev[v] = u
                heapq.heappush(heap, (dist[v], v))

    return dist, prev


# Reconstruct the shortest path from source to target
def reconstruct_path(prev, source, target):
    path, node = [], target
    while node != -1:
        path.append(node)
        if node == source:
            break
        node = prev[node]
    return list(reversed(path)) if path and path[-1] == source else []



def print_path(graph_name, nodes, path, dist, target, elapsed_ms, memory_kb):
    print(f"\n--- {graph_name.replace('_', ' ').title()} "
          f"| {len(nodes)} nodes ---")

    if not path:
        print("No path found (graph disconnected)")
        print(f"Time  : {elapsed_ms:.4f} ms  |  Memory: {memory_kb:.2f} KB")
        return

    steps = " --> ".join(f"{nodes[n]['name']} [{n}]" for n in path)
    print(f"Path    : {steps}")
    print(f"Distance: {dist[target]:.4f} miles  |  Hops: {len(path) - 1}")
    print(f"Time    : {elapsed_ms:.4f} ms  |  Memory: {memory_kb:.2f} KB")


# Highlight the shortest path on the graph visualization
def visualize_path(graph_name, data, path):
    G   = nx.Graph()
    pos = {}

    for node in data["nodes"]:
        nid = node["id"]
        G.add_node(nid, name=node["name"])
        pos[nid] = (node["longitude"], node["latitude"])

    for e in data["edges"]:
        G.add_edge(e["u"], e["v"], weight=e["weight_miles"])

    path_edges  = set(zip(path, path[1:]))
    edge_colors = ["red" if (u,v) in path_edges or (v,u) in path_edges else "lightgray" for u,v in G.edges()]
    edge_widths = [2.5  if (u,v) in path_edges or (v,u) in path_edges else 0.8          for u,v in G.edges()]
    node_colors = [
        "red"       if n == path[0] or n == path[-1] else
        "orange"    if n in path else
        "steelblue" for n in G.nodes()
    ]

    labels = {n: f"{data['nodes'][n]['name']}\n[{n}]" for n in G.nodes()}

    plt.figure(figsize=(13, 10))
    plt.title(
        f"{graph_name.replace('_', ' ').title()}  |  "
        f"Shortest Path: {data['nodes'][path[0]]['name']} --> {data['nodes'][path[-1]]['name']}",
        fontsize=12, fontweight="bold"
    )
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=200, alpha=0.95)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=6)
    plt.legend(handles=[
        mpatches.Patch(color="red",       label="Start / End"),
        mpatches.Patch(color="orange",    label="On path"),
        mpatches.Patch(color="steelblue", label="Other nodes"),
    ], loc="lower left", fontsize=9)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()


# Run Dijkstra on all 4 graph variants
def run_all(source_id, target_id, show_viz=True):
    print("\n" + "=" * 60)
    print("  SOUTH TAMPA -- DIJKSTRA SHORTEST PATH")
    print("=" * 60)

    for name in GRAPH_FILES:
        nodes, adj, data = load_graph(name)
        num_nodes = len(nodes)

        if source_id not in nodes or target_id not in nodes:
            print(f"\n[{name}] Node index out of range (valid: 0 to {num_nodes - 1})")
            continue

        # Measure time and memory
        tracemalloc.start()
        t0 = time.perf_counter()
        dist, prev = dijkstra(adj, num_nodes, source_id)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_kb = peak / 1024

        path = reconstruct_path(prev, source_id, target_id)
        print_path(name, nodes, path, dist, target_id, elapsed_ms, memory_kb)

        if show_viz and path:
            visualize_path(name, data, path)


# List all destinations with their index
def list_destinations():
    print("\nSmall Graph destinations (index 0-19):")
    with (GRAPHS_DIR / "small_sparse.json").open() as f:
        small = json.load(f)
    for node in small["nodes"]:
        print(f"  [{node['id']:>2}]  {node['name']}")

    print("\nLarge Graph destinations (index 0-49):")
    with (GRAPHS_DIR / "large_sparse.json").open() as f:
        large = json.load(f)
    for node in large["nodes"]:
        print(f"  [{node['id']:>2}]  {node['name']}")


if __name__ == "__main__":
    list_destinations()

    print()
    source = int(input("Enter source index: "))
    target = int(input("Enter target index: "))

    run_all(source, target, show_viz=True)