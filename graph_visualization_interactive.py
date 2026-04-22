#!/usr/bin/env python3
"""
Interactive graph visualization utility for generated graph JSON files.

Expected project layout
-----------------------
project_root/
  graph_visualization_interactive.py
  generated_graphs/
    small_sparse.json
    small_dense.json
    large_sparse.json
    large_dense.json

What this script does
---------------------
- Loads one graph JSON file or all graph JSON files from ./generated_graphs
- Builds a NetworkX graph from the JSON
- Uses the destination latitude/longitude coordinates as the layout
- Can either:
  * display graphs natively with matplotlib windows
  * save graphs as PNG files

Usage
-----
Display all graph JSON files interactively:
    python graph_visualization_interactive.py --show

Display one specific graph:
    python graph_visualization_interactive.py --graph generated_graphs/small_sparse.json --show

Save all graph images:
    python graph_visualization_interactive.py

Hide edge weights:
    python graph_visualization_interactive.py --show --hide-weights

Hide node labels:
    python graph_visualization_interactive.py --show --hide-labels
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import networkx as nx


def load_graph_json(filepath: Path) -> Dict:
    with filepath.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_networkx_graph(payload: Dict) -> Tuple[nx.Graph, Dict[int, Tuple[float, float]]]:
    graph = nx.Graph()
    positions: Dict[int, Tuple[float, float]] = {}

    for node in payload["nodes"]:
        node_id = int(node["id"])
        name = node["name"]
        lat = float(node["latitude"])
        lon = float(node["longitude"])

        graph.add_node(
            node_id,
            name=name,
            latitude=lat,
            longitude=lon,
        )
        positions[node_id] = (lon, lat)

    for edge in payload["edges"]:
        u = int(edge["u"])
        v = int(edge["v"])
        weight = float(edge["weight_miles"])
        graph.add_edge(u, v, weight=weight)

    return graph, positions


def draw_graph(
    payload: Dict,
    output_path: Path,
    show_labels: bool = True,
    show_weights: bool = True,
    show_plot: bool = False,
    figsize: Tuple[int, int] = (13, 10),
) -> None:
    graph, positions = build_networkx_graph(payload)
    metadata = payload.get("metadata", {})
    graph_name = payload.get("graph_name", output_path.stem)

    plt.figure(figsize=figsize)

    nx.draw_networkx_edges(
        graph,
        positions,
        width=1.2,
        alpha=0.6,
    )

    nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=220,
        alpha=0.95,
    )

    if show_labels:
        labels = {node: graph.nodes[node]["name"] for node in graph.nodes}
        nx.draw_networkx_labels(
            graph,
            positions,
            labels=labels,
            font_size=8,
        )

    if show_weights:
        edge_labels = {
            (u, v): f'{data["weight"]:.2f}'
            for u, v, data in graph.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            graph,
            positions,
            edge_labels=edge_labels,
            font_size=6,
            rotate=False,
        )

    title_lines = [
        f"{graph_name.replace('_', ' ').title()}",
        (
            f"Vertices={metadata.get('num_vertices', graph.number_of_nodes())}, "
            f"Edges={metadata.get('num_edges', graph.number_of_edges())}, "
            f"Density={metadata.get('density', 'N/A')}, "
            f"Avg Degree={metadata.get('average_degree', 'N/A')}"
        ),
    ]
    plt.title("\n".join(title_lines), fontsize=14)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()

    if show_plot:
        plt.show()
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.close()


def visualize_one_file(
    graph_file: Path,
    outdir: Path,
    show_labels: bool,
    show_weights: bool,
    show_plot: bool,
) -> Path:
    payload = load_graph_json(graph_file)
    output_path = outdir / f"{graph_file.stem}.png"
    draw_graph(
        payload=payload,
        output_path=output_path,
        show_labels=show_labels,
        show_weights=show_weights,
        show_plot=show_plot,
    )
    return output_path


def visualize_all_graphs(
    graphs_dir: Path,
    outdir: Path,
    show_labels: bool,
    show_weights: bool,
    show_plot: bool,
) -> None:
    graph_files = sorted(graphs_dir.glob("*.json"))

    if not graph_files:
        raise FileNotFoundError(f"No JSON graph files found in: {graphs_dir}")

    for graph_file in graph_files:
        visualize_one_file(
            graph_file=graph_file,
            outdir=outdir,
            show_labels=show_labels,
            show_weights=show_weights,
            show_plot=show_plot,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize graph JSON files from the generated_graphs directory."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root directory. Default: current directory",
    )
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=None,
        help="Directory containing graph JSON files. Default: <root>/generated_graphs",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=None,
        help="Path to a single graph JSON file to visualize.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory for generated images. Default: <root>/graph_visualizations",
    )
    parser.add_argument(
        "--hide-labels",
        action="store_true",
        help="Do not draw destination labels.",
    )
    parser.add_argument(
        "--hide-weights",
        action="store_true",
        help="Do not draw edge weight labels.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display graph interactively in a matplotlib window instead of saving an image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = args.root.resolve()
    graphs_dir = args.graphs_dir.resolve() if args.graphs_dir else (root / "generated_graphs")
    outdir = args.outdir.resolve() if args.outdir else (root / "graph_visualizations")

    show_labels = not args.hide_labels
    show_weights = not args.hide_weights
    show_plot = args.show

    if args.graph is not None:
        graph_file = args.graph.resolve()
        if not graph_file.exists():
            raise FileNotFoundError(f"Graph JSON file not found: {graph_file}")
        visualize_one_file(
            graph_file=graph_file,
            outdir=outdir,
            show_labels=show_labels,
            show_weights=show_weights,
            show_plot=show_plot,
        )
    else:
        if not graphs_dir.exists():
            raise FileNotFoundError(f"Graphs directory not found: {graphs_dir}")
        visualize_all_graphs(
            graphs_dir=graphs_dir,
            outdir=outdir,
            show_labels=show_labels,
            show_weights=show_weights,
            show_plot=show_plot,
        )


if __name__ == "__main__":
    main()
