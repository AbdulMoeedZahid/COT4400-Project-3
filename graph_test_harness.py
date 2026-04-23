#!/usr/bin/env python3
"""
Generic graph algorithm test harness for JSON graph files.

Purpose
-------
This script benchmarks any graph algorithm function across all graph JSON files
stored in ./generated_graphs (or another directory you choose).

It reports, for each graph:
- execution time
- peak memory usage during the call
- optional result summary

Design
------
You register algorithm functions in the ALGORITHMS dictionary.

Each algorithm function must accept:
    graph, start_node=None

where `graph` is a dictionary with:
    {
        "adj": {node_id: [(neighbor_id, weight), ...], ...},
        "nodes": [...],
        "edges": [...],
        "metadata": {...},
        "payload": original_json_payload
    }

and it may return anything.

Example usage
-------------
Run BFS on every graph in ./generated_graphs:
    python graph_test_harness.py --algorithm bfs

Run DFS on every graph with a chosen start node:
    python graph_test_harness.py --algorithm dfs --start-node 0

Run all registered algorithms:
    python graph_test_harness.py --all

Use a different graph directory:
    python graph_test_harness.py --algorithm bfs --graphs-dir my_graphs

Notes on measurement
--------------------
- Time is measured with time.perf_counter()
- Memory is measured with tracemalloc peak memory
- Peak memory here reflects Python-level allocations during the function call,
  which is appropriate for classroom-style comparative analysis
"""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from dijkstra import dijkstra


GraphDict = Dict[str, Any]
AlgorithmFn = Callable[[GraphDict, Optional[int]], Any]


def load_graph(filepath: Path) -> GraphDict:
    with filepath.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    adj: Dict[int, List[Tuple[int, float]]] = {}

    for node in payload["nodes"]:
        node_id = int(node["id"])
        adj[node_id] = []

    for edge in payload["edges"]:
        u = int(edge["u"])
        v = int(edge["v"])
        weight = float(edge.get("weight_miles", edge.get("weight", 1.0)))
        adj[u].append((v, weight))
        adj[v].append((u, weight))

    return {
        "adj": adj,
        "nodes": payload["nodes"],
        "edges": payload["edges"],
        "metadata": payload.get("metadata", {}),
        "payload": payload,
    }


def find_default_start_node(graph: GraphDict) -> int:
    return min(graph["adj"].keys())


def bfs(graph: GraphDict, start_node: Optional[int] = None) -> Dict[str, Any]:
    adj = graph["adj"]
    start = find_default_start_node(graph) if start_node is None else start_node

    if start not in adj:
        raise ValueError(f"Start node {start} is not in graph.")

    visited = {start}
    queue = deque([start])

    traversal_order: List[int] = []
    level = {start: 0}
    parent = {start: None}
    total_tree_distance = 0.0

    while queue:
        node = queue.popleft()
        traversal_order.append(node)

        for neighbor, weight in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                level[neighbor] = level[node] + 1
                parent[neighbor] = node
                total_tree_distance += weight

    return {
        "visited_count": len(traversal_order),
        "traversal_order": traversal_order,
        "levels": level,
        "parents": parent,
        "max_level": max(level.values()) if level else 0,
        "total_tree_distance": round(total_tree_distance, 4),
    }


def dfs(graph: GraphDict, start_node: Optional[int] = None) -> Dict[str, Any]:
    adj = graph["adj"]
    start = find_default_start_node(graph) if start_node is None else start_node

    if start not in adj:
        raise ValueError(f"Start node {start} is not in graph.")

    visited = set()
    traversal_order: List[int] = []
    parent = {start: None}
    walk_distance = 0.0

    def visit(node: int) -> None:
        nonlocal walk_distance
        visited.add(node)
        traversal_order.append(node)

        for neighbor, weight in adj[node]:
            if neighbor not in visited:
                parent[neighbor] = node
                walk_distance += weight
                visit(neighbor)
                walk_distance += weight

    visit(start)

    return {
        "visited_count": len(traversal_order),
        "traversal_order": traversal_order,
        "parents": parent,
        "walk_distance": round(walk_distance, 4),
    }


ALGORITHMS: Dict[str, AlgorithmFn] = {
    "bfs": bfs,
    "dfs": dfs,
    "dijkstra": dijkstra,
}


def benchmark_algorithm(
    algorithm: AlgorithmFn,
    graph: GraphDict,
    start_node: Optional[int] = None,
    repeats: int = 1,
) -> Dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    best_time = None
    best_result = None
    best_peak = None

    for _ in range(repeats):
        tracemalloc.start()
        t0 = time.perf_counter()
        result = algorithm(graph, start_node)
        t1 = time.perf_counter()
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        elapsed = t1 - t0

        if best_time is None or elapsed < best_time:
            best_time = elapsed
            best_result = result
            best_peak = peak_bytes

    return {
        "time_seconds": best_time,
        "peak_memory_bytes": best_peak,
        "result": best_result,
    }


def summarize_result(result: Any) -> str:
    if isinstance(result, dict):
        important_keys = [
            "visited_count",
            "max_level",
            "total_tree_distance",
            "walk_distance",
        ]
        parts = []
        for key in important_keys:
            if key in result:
                parts.append(f"{key}={result[key]}")
        if parts:
            return ", ".join(parts)
        return f"dict_keys={list(result.keys())}"
    return str(type(result).__name__)


def format_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.2f} KB"
    return f"{num_bytes / (1024**2):.2f} MB"


def run_one_algorithm_on_all_graphs(
    algorithm_name: str,
    algorithm: AlgorithmFn,
    graph_files: List[Path],
    start_node: Optional[int],
    repeats: int,
) -> None:
    print(f"\n=== Benchmark: {algorithm_name} ===\n")

    for graph_file in graph_files:
        graph = load_graph(graph_file)
        metadata = graph["metadata"]

        bench = benchmark_algorithm(
            algorithm=algorithm,
            graph=graph,
            start_node=start_node,
            repeats=repeats,
        )

        graph_name = graph_file.stem
        vertices = metadata.get("num_vertices", len(graph["adj"]))
        edges = metadata.get("num_edges", len(graph["edges"]))
        density = metadata.get("density", "N/A")

        print(f"Graph: {graph_name}")
        print(f"  Vertices: {vertices}")
        print(f"  Edges: {edges}")
        print(f"  Density: {density}")
        print(f"  Execution Time: {1000000 * bench['time_seconds']:.2f} microseconds")
        print(f"  Peak Memory: {format_bytes(bench['peak_memory_bytes'])}")
        print(f"  Result Summary: {summarize_result(bench['result'])}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark registered graph algorithms on all graph JSON files."
    )
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=Path("generated_graphs"),
        help="Directory containing graph JSON files.",
    )
    parser.add_argument(
        "--algorithm",
        choices=sorted(ALGORITHMS.keys()),
        help="Name of one registered algorithm to benchmark.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Benchmark all registered algorithms.",
    )
    parser.add_argument(
        "--start-node",
        type=int,
        default=None,
        help="Optional start node for traversal algorithms.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of repeated runs per graph; best time is reported.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.all and args.algorithm is None:
        raise ValueError("Provide --algorithm <name> or use --all")

    graphs_dir = args.graphs_dir.resolve()
    if not graphs_dir.exists():
        raise FileNotFoundError(f"Graphs directory not found: {graphs_dir}")

    graph_files = sorted(graphs_dir.glob("*.json"))
    if not graph_files:
        raise FileNotFoundError(f"No JSON graph files found in: {graphs_dir}")

    if args.all:
        for algorithm_name, algorithm in ALGORITHMS.items():
            run_one_algorithm_on_all_graphs(
                algorithm_name=algorithm_name,
                algorithm=algorithm,
                graph_files=graph_files,
                start_node=args.start_node,
                repeats=args.repeats,
            )
    else:
        algorithm = ALGORITHMS[args.algorithm]
        run_one_algorithm_on_all_graphs(
            algorithm_name=args.algorithm,
            algorithm=algorithm,
            graph_files=graph_files,
            start_node=args.start_node,
            repeats=args.repeats,
        )


if __name__ == "__main__":
    main()
