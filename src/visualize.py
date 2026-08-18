"""NetworkX subgraph of top Hippo–actin predictions plus local STRING context."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.config import (
    DATA_PROCESSED,
    FIGURES_DIR,
    FIGURES_PNG,
    HIT_LIST_CSV,
    INTERACTIONS_CSV,
    PROTEINS_CSV,
    SUBGRAPH_PNG,
    SUBGRAPH_SVG,
    VIZ_TOP_PER_SPECIES,
)

logger = logging.getLogger(__name__)

NODE_COLORS: dict[str, str] = {
    "hippo": "#2E5A88",
    "actin": "#E07A5F",
    "both": "#7B68A6",
    "partner": "#9AA0A6",
}
PREDICTED_COLOR: str = "#D35400"
KNOWN_COLOR: str = "#B0B0B0"
SPECIES_PANELS: tuple[str, ...] = ("Homo sapiens", "Saccharomyces cerevisiae")


def top_hits_for_species(hits: pd.DataFrame, species: str, top_n: int) -> pd.DataFrame:
    """Return the highest-probability hits for one organism."""
    subset = hits[hits["species"] == species].sort_values(
        "probability", ascending=False, kind="mergesort"
    )
    return subset.head(top_n).reset_index(drop=True)


def build_subgraph(
    hits: pd.DataFrame,
    interactions: pd.DataFrame,
    proteins: pd.DataFrame,
) -> nx.Graph:
    """Undirected graph of hit proteins, predicted edges, and STRING edges among them."""
    graph: nx.Graph = nx.Graph()
    if hits.empty:
        return graph

    id_to_meta = {
        str(row.string_id): {
            "name": str(row.preferred_name),
            "compartment": str(row.compartment),
            "species": str(row.species_name),
        }
        for row in proteins.itertuples(index=False)
    }
    node_ids: set[str] = set()
    for row in hits.itertuples(index=False):
        node_ids.add(str(row.string_id_a))
        node_ids.add(str(row.string_id_b))

    for string_id in sorted(node_ids):
        meta = id_to_meta.get(string_id, {"name": string_id, "compartment": "partner", "species": ""})
        graph.add_node(
            string_id,
            name=meta["name"],
            compartment=meta["compartment"],
            species=meta["species"],
        )

    predicted: set[tuple[str, str]] = set()
    for row in hits.itertuples(index=False):
        left, right = str(row.string_id_a), str(row.string_id_b)
        key = (left, right) if left < right else (right, left)
        predicted.add(key)
        graph.add_edge(
            left,
            right,
            kind="predicted",
            probability=float(row.probability),
        )

    for row in interactions.itertuples(index=False):
        left, right = str(row.source_string_id), str(row.target_string_id)
        if left not in graph or right not in graph or left == right:
            continue
        key = (left, right) if left < right else (right, left)
        if key in predicted:
            continue
        if graph.has_edge(left, right):
            continue
        graph.add_edge(left, right, kind="known", probability=float(row.combined_score))

    return graph


def _node_style(graph: nx.Graph) -> tuple[list[str], list[float]]:
    colors = [NODE_COLORS.get(graph.nodes[n]["compartment"], NODE_COLORS["partner"]) for n in graph.nodes]
    sizes = [280 + 90 * graph.degree(n) for n in graph.nodes]
    return colors, sizes


def draw_panel(axis: plt.Axes, graph: nx.Graph, title: str, seed: int = 42) -> None:
    """Draw one species subgraph onto a matplotlib axis."""
    axis.set_title(title, fontsize=12, pad=10)
    axis.set_axis_off()
    if graph.number_of_nodes() == 0:
        axis.text(0.5, 0.5, "no hits", ha="center", va="center", transform=axis.transAxes)
        return

    positions = nx.spring_layout(graph, seed=seed, k=0.9, iterations=80)
    colors, sizes = _node_style(graph)
    known = [(u, v) for u, v, d in graph.edges(data=True) if d.get("kind") == "known"]
    predicted = [(u, v) for u, v, d in graph.edges(data=True) if d.get("kind") == "predicted"]
    pred_widths = [
        1.4 + 4.0 * float(graph.edges[u, v].get("probability", 0.5)) for u, v in predicted
    ]

    nx.draw_networkx_edges(graph, positions, ax=axis, edgelist=known, edge_color=KNOWN_COLOR, width=1.2, alpha=0.85)
    if predicted:
        nx.draw_networkx_edges(
            graph,
            positions,
            ax=axis,
            edgelist=predicted,
            edge_color=PREDICTED_COLOR,
            width=pred_widths,
            style="dashed",
            alpha=0.95,
        )
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=axis,
        node_color=colors,
        node_size=sizes,
        edgecolors="white",
        linewidths=0.8,
    )
    labels = {n: graph.nodes[n]["name"] for n in graph.nodes}
    nx.draw_networkx_labels(graph, positions, labels=labels, ax=axis, font_size=8, font_color="#1b1b1b")


def render_figure(
    human_graph: nx.Graph,
    yeast_graph: nx.Graph,
    output_png: Path,
    output_svg: Path,
    extra_png: Path | None = None,
) -> None:
    """Write a two-panel PNG/SVG figure."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4), constrained_layout=True)
    draw_panel(axes[0], human_graph, "Human Hippo – actin")
    draw_panel(axes[1], yeast_graph, "Yeast RAM/MOR – actin")
    legend = [
        Patch(facecolor=NODE_COLORS["hippo"], edgecolor="white", label="Hippo / RAM"),
        Patch(facecolor=NODE_COLORS["actin"], edgecolor="white", label="Actin"),
        Patch(facecolor=NODE_COLORS["both"], edgecolor="white", label="Both"),
        Line2D([0], [0], color=KNOWN_COLOR, lw=1.5, label="STRING ≥ 700"),
        Line2D([0], [0], color=PREDICTED_COLOR, lw=2.0, linestyle="--", label="Predicted novel"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Top predicted Hippo–actin interactions", fontsize=14, y=1.02)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=160, bbox_inches="tight", facecolor="white")
    fig.savefig(output_svg, bbox_inches="tight", facecolor="white")
    if extra_png is not None:
        extra_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(extra_png, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("wrote %s", output_png)
    logger.info("wrote %s", output_svg)


def run(
    *,
    hits_csv: Path = HIT_LIST_CSV,
    interactions_csv: Path = INTERACTIONS_CSV,
    proteins_csv: Path = PROTEINS_CSV,
    top_n: int = VIZ_TOP_PER_SPECIES,
) -> dict[str, int]:
    """Load artifacts, draw both species panels, and save figures."""
    if not hits_csv.is_file():
        raise FileNotFoundError(f"hit list not found: {hits_csv} (run src.train first)")
    hits = pd.read_csv(hits_csv)
    interactions = pd.read_csv(interactions_csv)
    proteins = pd.read_csv(proteins_csv)
    required = {"species", "string_id_a", "string_id_b", "probability"}
    missing = required.difference(hits.columns)
    if missing:
        raise ValueError(f"hit list missing columns: {sorted(missing)}")

    counts: dict[str, int] = {}
    graphs: dict[str, nx.Graph] = {}
    for species in SPECIES_PANELS:
        species_hits = top_hits_for_species(hits, species, top_n)
        graphs[species] = build_subgraph(species_hits, interactions, proteins)
        counts[species] = graphs[species].number_of_nodes()
        logger.info(
            "%s nodes=%s predicted_edges=%s known_edges=%s",
            species,
            graphs[species].number_of_nodes(),
            sum(1 for _, _, d in graphs[species].edges(data=True) if d.get("kind") == "predicted"),
            sum(1 for _, _, d in graphs[species].edges(data=True) if d.get("kind") == "known"),
        )

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    render_figure(
        graphs[SPECIES_PANELS[0]],
        graphs[SPECIES_PANELS[1]],
        SUBGRAPH_PNG,
        SUBGRAPH_SVG,
        extra_png=FIGURES_PNG,
    )
    from src.plot_embeddings import plot_esm_tsne

    try:
        plot_esm_tsne()
    except FileNotFoundError as exc:
        logger.warning("skipping t-SNE figure: %s", exc)
    return counts


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def main() -> None:
    configure_logging()
    counts = run()
    logger.info("visualization complete %s", counts)


if __name__ == "__main__":
    main()
