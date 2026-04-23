import json
from collections import deque
from pathlib import Path


def load_graph(filepath):
    with open(filepath) as f:
        data = json.load(f)

    adj = {}

    for node in data["nodes"]:
        adj[node["id"]] = []

    for edge in data["edges"]:
        u = edge["u"]
        v = edge["v"]

        adj[u].append(v)
        adj[v].append(u)  # undirected

    return adj, data


def bfs(adj, start):
    visited = set()
    queue = deque([start])

    traversal_order = []
    level = {start: 0}
    parent = {start: None}

    visited.add(start)

    while queue:
        node = queue.popleft()
        traversal_order.append(node)

        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

                level[neighbor] = level[node] + 1
                parent[neighbor] = node

    return traversal_order, level, parent


def display_results(data, traversal_order, level, parent):
    print("\n=== BFS Traversal ===\n")

    print("Traversal Order:")
    print(traversal_order)

    print("\nLevels (distance from source):")
    for node in traversal_order:
        name = data["nodes"][node]["name"]
        print(f"{name}: Level {level[node]}")

    print("\nParent Tree:")
    for node in traversal_order:
        if parent[node] is not None:
            u_name = data["nodes"][parent[node]]["name"]
            v_name = data["nodes"][node]["name"]
            print(f"{u_name} -> {v_name}")


def save_results(output_path, start, traversal_order, level, parent):
    result = {
        "start_node": start,
        "traversal_order": traversal_order,
        "levels": level,
        "parents": parent,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)


def main():
    graph_file = "generated_graphs/small_sparse.json"  # change as needed
    output_file = "bfs_output.json"
    start_node = 0  # can change

    adj, data = load_graph(graph_file)

    traversal_order, level, parent = bfs(adj, start_node)

    display_results(data, traversal_order, level, parent)
    save_results(output_file, start_node, traversal_order, level, parent)


if __name__ == "__main__":
    main()
