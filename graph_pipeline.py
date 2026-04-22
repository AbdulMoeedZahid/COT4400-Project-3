#!/usr/bin/env python3
"""
Graph pipeline for the South Tampa destinations project.

What this script does
---------------------
1. Loads the curated South Tampa destination dataset.
2. Builds four weighted, undirected graphs:
   - small_sparse
   - small_dense
   - large_sparse
   - large_dense
3. Uses Euclidean distance on locally projected latitude/longitude coordinates
   so the edge weights are interpretable in miles.
4. Guarantees connectivity by first building an MST backbone, then adds the
   shortest remaining edges until the target density is reached.
5. Saves each graph as JSON plus a summary CSV.

Why this matches the project rubric
-----------------------------------
- "Small" vs "large" is controlled by number of destinations (vertices).
- "Sparse" vs "dense" is controlled by edge density.
- Edges are undirected and weighted by Euclidean distance derived from
  lat/long coordinates.

Usage
-----
python graph_pipeline.py
python graph_pipeline.py --input south_tampa_dataset.json --outdir generated_graphs

Outputs
-------
generated_graphs/
  small_sparse.json
  small_dense.json
  large_sparse.json
  large_dense.json
  graph_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


# ---------------------------
# Configurable experiment setup
# ---------------------------

SMALL_N = 20

# Density = number_of_edges / possible_edges for an undirected simple graph.
# These values give a clear contrast without becoming unrealistic.
SPARSE_DENSITY = 0.15
DENSE_DENSITY = 0.40


@dataclass(frozen=True)
class Destination:
    id: int
    name: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class Edge:
    u: int
    v: int
    weight: float


# ---------------------------
# Data loading
# ---------------------------

def load_destinations(filepath: Path) -> List[Destination]:
    with filepath.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    destinations: List[Destination] = []
    for i, item in enumerate(raw):
        destinations.append(
            Destination(
                id=i,
                name=item["name"],
                latitude=float(item["latitude"]),
                longitude=float(item["longitude"]),
            )
        )
    return destinations


# ---------------------------
# Distance utilities
# ---------------------------

def latlon_to_local_xy_miles(
    latitude: float,
    longitude: float,
    origin_latitude: float,
    origin_longitude: float,
) -> Tuple[float, float]:
    """
    Convert latitude/longitude to a local planar x,y coordinate system in miles.

    This keeps the final edge weight Euclidean, while basing it directly on the
    original lat/long coordinates. Over a compact area like South Tampa, this
    is a reasonable local approximation.
    """
    miles_per_degree_lat = 69.0
    miles_per_degree_lon = 69.172 * math.cos(math.radians(origin_latitude))

    x = (longitude - origin_longitude) * miles_per_degree_lon
    y = (latitude - origin_latitude) * miles_per_degree_lat
    return x, y


def euclidean_distance_miles(
    a: Destination,
    b: Destination,
    origin_latitude: float,
    origin_longitude: float,
) -> float:
    ax, ay = latlon_to_local_xy_miles(a.latitude, a.longitude, origin_latitude, origin_longitude)
    bx, by = latlon_to_local_xy_miles(b.latitude, b.longitude, origin_latitude, origin_longitude)
    return math.hypot(ax - bx, ay - by)


# ---------------------------
# Complete graph distance prep
# ---------------------------

def all_pair_edges(nodes: Sequence[Destination]) -> List[Edge]:
    origin_lat = sum(n.latitude for n in nodes) / len(nodes)
    origin_lon = sum(n.longitude for n in nodes) / len(nodes)

    edges: List[Edge] = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            w = euclidean_distance_miles(nodes[i], nodes[j], origin_lat, origin_lon)
            edges.append(Edge(nodes[i].id, nodes[j].id, round(w, 4)))
    edges.sort(key=lambda e: e.weight)
    return edges


# ---------------------------
# Union-Find for Kruskal/MST
# ---------------------------

class UnionFind:
    def __init__(self, elements: Iterable[int]) -> None:
        self.parent = {x: x for x in elements}
        self.rank = {x: 0 for x in elements}

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return False

        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
        return True


# ---------------------------
# Graph construction
# ---------------------------

def choose_target_edge_count(n: int, density: float) -> int:
    possible_edges = n * (n - 1) // 2
    target = max(n - 1, int(round(density * possible_edges)))
    return min(target, possible_edges)


def build_connected_graph(nodes: Sequence[Destination], target_density: float) -> List[Edge]:
    """
    Build an undirected connected graph with weighted edges.

    Strategy:
    1. Start with the MST so the graph is connected.
    2. Add the shortest remaining edges until the target density is reached.
    """
    n = len(nodes)
    full_edges = all_pair_edges(nodes)
    node_ids = [node.id for node in nodes]
    target_edge_count = choose_target_edge_count(n, target_density)

    uf = UnionFind(node_ids)
    selected: List[Edge] = []
    selected_set = set()

    # Step 1: MST backbone
    for edge in full_edges:
        if uf.union(edge.u, edge.v):
            selected.append(edge)
            selected_set.add((min(edge.u, edge.v), max(edge.u, edge.v)))
            if len(selected) == n - 1:
                break

    # Step 2: add shortest remaining edges until density target is reached
    for edge in full_edges:
        key = (min(edge.u, edge.v), max(edge.u, edge.v))
        if key in selected_set:
            continue
        if len(selected) >= target_edge_count:
            break
        selected.append(edge)
        selected_set.add(key)

    selected.sort(key=lambda e: (e.u, e.v))
    return selected


def induced_subset(destinations: Sequence[Destination], count: int) -> List[Destination]:
    """
    Deterministic subset selection for the 'small' graphs.

    Since the dataset is already curated and ordered, we preserve the first `count`
    destinations to keep the experiment reproducible.
    """
    return list(destinations[:count])


# ---------------------------
# Serialization / metrics
# ---------------------------

def graph_density(n: int, m: int) -> float:
    possible = n * (n - 1) // 2
    if possible == 0:
        return 0.0
    return m / possible


def average_degree(n: int, m: int) -> float:
    if n == 0:
        return 0.0
    return (2.0 * m) / n


def total_edge_weight(edges: Sequence[Edge]) -> float:
    return round(sum(e.weight for e in edges), 4)


def serialize_graph(name: str, nodes: Sequence[Destination], edges: Sequence[Edge]) -> Dict:
    m = len(edges)
    n = len(nodes)
    return {
        "graph_name": name,
        "metadata": {
            "num_vertices": n,
            "num_edges": m,
            "density": round(graph_density(n, m), 4),
            "average_degree": round(average_degree(n, m), 4),
            "total_edge_weight_miles": total_edge_weight(edges),
            "graph_type": "undirected_weighted",
            "weight_definition": "Euclidean distance in miles from locally projected latitude/longitude coordinates"
        },
        "nodes": [
            {
                "id": node.id,
                "name": node.name,
                "latitude": node.latitude,
                "longitude": node.longitude,
            }
            for node in nodes
        ],
        "edges": [
            {
                "u": edge.u,
                "v": edge.v,
                "weight_miles": edge.weight,
            }
            for edge in edges
        ],
    }


def write_json(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_summary_csv(path: Path, graphs: Dict[str, Dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "graph_name",
                "num_vertices",
                "num_edges",
                "density",
                "average_degree",
                "total_edge_weight_miles",
                "graph_type",
            ]
        )
        for name, payload in graphs.items():
            md = payload["metadata"]
            writer.writerow(
                [
                    name,
                    md["num_vertices"],
                    md["num_edges"],
                    md["density"],
                    md["average_degree"],
                    md["total_edge_weight_miles"],
                    md["graph_type"],
                ]
            )


# ---------------------------
# Main pipeline
# ---------------------------

def build_all_graphs(destinations: Sequence[Destination]) -> Dict[str, Dict]:
    small_nodes = induced_subset(destinations, min(SMALL_N, len(destinations)))
    large_nodes = list(destinations)

    graph_specs = {
        "small_sparse": (small_nodes, SPARSE_DENSITY),
        "small_dense": (small_nodes, DENSE_DENSITY),
        "large_sparse": (large_nodes, SPARSE_DENSITY),
        "large_dense": (large_nodes, DENSE_DENSITY),
    }

    output: Dict[str, Dict] = {}
    for name, (nodes, density) in graph_specs.items():
        edges = build_connected_graph(nodes, density)
        output[name] = serialize_graph(name, nodes, edges)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build four South Tampa graph variants.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("south_tampa_dataset.json"),
        help="Path to the destination dataset JSON file.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("generated_graphs"),
        help="Directory where generated graph files will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    destinations = load_destinations(args.input)
    graphs = build_all_graphs(destinations)

    for name, payload in graphs.items():
        write_json(args.outdir / f"{name}.json", payload)

    write_summary_csv(args.outdir / "graph_summary.csv", graphs)

    print("Generated graph files:")
    for name in graphs:
        print(f"  - {args.outdir / f'{name}.json'}")
    print(f"  - {args.outdir / 'graph_summary.csv'}")

    print("\nQuick summary:")
    for name, payload in graphs.items():
        md = payload["metadata"]
        print(
            f"{name}: "
            f"V={md['num_vertices']}, "
            f"E={md['num_edges']}, "
            f"density={md['density']}, "
            f"avg_degree={md['average_degree']}"
        )


if __name__ == "__main__":
    main()
