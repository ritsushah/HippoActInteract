"""t-SNE of ESM-2 embeddings with non-overlapping seed labels."""

from __future__ import annotations

import inspect
import logging
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patheffects import withStroke
from sklearn.manifold import TSNE

from src.config import NODE_EMBEDDINGS_PT, PROTEINS_CSV, TSNE_PNG

logger = logging.getLogger(__name__)

COMPARTMENT_COLORS: dict[str, str] = {
    "hippo": "#2E5A88",
    "actin": "#E07A5F",
    "both": "#7B68A6",
    "partner": "#9AA0A6",
}

LABELED_SEEDS: frozenset[str] = frozenset(
    {
        "YAP1",
        "WWTR1",
        "NF2",
        "ACTB",
        "ACT1",
        "CDC42",
        "RHOA",
        "RHO1",
        "RAC1",
        "CBK1",
        "KIC1",
    }
)


def unique_seed_labels(
    names: list[str],
    species_names: list[str],
    seeds: frozenset[str] = LABELED_SEEDS,
) -> dict[int, str]:
    """Map row index -> label; disambiguate genes present in both species."""
    seed_counts: Counter[str] = Counter(name for name in names if name in seeds)
    labels: dict[int, str] = {}
    for index, (name, species) in enumerate(zip(names, species_names, strict=True)):
        if name not in seeds:
            continue
        if seed_counts[name] > 1:
            tag = "H" if species.startswith("Homo") else "Y"
            labels[index] = f"{name} ({tag})"
        else:
            labels[index] = name
    return labels


def _tsne_kwargs(random_state: int) -> dict[str, object]:
    params = inspect.signature(TSNE.__init__).parameters
    kwargs: dict[str, object] = {
        "n_components": 2,
        "perplexity": 30,
        "random_state": random_state,
        "init": "pca",
    }
    if "max_iter" in params:
        kwargs["max_iter"] = 1000
    else:
        kwargs["n_iter"] = 1000
    return kwargs


def _spread_close_ys(ys: np.ndarray, min_gap: float, y_ceiling: float) -> np.ndarray:
    """Separate stacked labels by pushing them down, never into the title band."""
    n_points = len(ys)
    adjusted = np.minimum(np.array(ys, dtype=float, copy=True), y_ceiling)
    parent = list(range(n_points))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(n_points):
        for right in range(left + 1, n_points):
            if abs(float(ys[left] - ys[right])) < 2.2 * min_gap:
                parent[find(left)] = find(right)

    clusters: dict[int, list[int]] = {}
    for index in range(n_points):
        clusters.setdefault(find(index), []).append(index)

    for members in clusters.values():
        members = sorted(members, key=lambda index: -adjusted[index])
        for slot, index in enumerate(members):
            if slot == 0:
                adjusted[index] = min(adjusted[index], y_ceiling)
            else:
                adjusted[index] = min(adjusted[index], adjusted[members[slot - 1]] - min_gap)
    return adjusted


def rightward_positions(points: np.ndarray, all_coords: np.ndarray) -> np.ndarray:
    """Place each label in the empty right half, keeping nearby proteins grouped."""
    n_points = len(points)
    placed = np.array(points, dtype=float, copy=True)
    if n_points == 0:
        return placed
    x_min = float(all_coords[:, 0].min())
    x_max = float(all_coords[:, 0].max())
    y_min = float(all_coords[:, 1].min())
    y_max = float(all_coords[:, 1].max())
    span_x = max(x_max - x_min, 1.0)
    span_y = max(y_max - y_min, 1.0)
    x_gutter = x_min + 0.70 * span_x
    y_ceiling = y_max - 0.16 * span_y
    min_gap = 0.055 * span_y
    placed[:, 1] = _spread_close_ys(points[:, 1], min_gap, y_ceiling)
    order = np.argsort(-placed[:, 1])
    for rank, index in enumerate(order):
        column = 0.12 * span_x if rank % 2 else 0.0
        placed[index, 0] = max(float(points[index, 0]) + 0.10 * span_x, x_gutter) + column
    return placed


def _draw_leader_lines(ax: plt.Axes, texts: list, xs: list[float], ys: list[float]) -> None:
    """Connect each label to its protein after labels have finished moving."""
    for text, x_coord, y_coord in zip(texts, xs, ys, strict=True):
        ax.annotate(
            "",
            xy=(x_coord, y_coord),
            xytext=text.get_position(),
            arrowprops={
                "arrowstyle": "-",
                "color": "#444444",
                "lw": 0.85,
                "shrinkA": 4,
                "shrinkB": 4,
            },
            zorder=3,
        )


def _adjust_labels(
    texts,
    xs: list[float],
    ys: list[float],
    ax: plt.Axes,
) -> None:
    """Draw leader lines from proteins to the right-side labels."""
    _draw_leader_lines(ax, texts, xs, ys)


def plot_esm_tsne(
    *,
    embeddings_pt: Path = NODE_EMBEDDINGS_PT,
    proteins_csv: Path = PROTEINS_CSV,
    output_png: Path = TSNE_PNG,
    random_state: int = 42,
) -> Path:
    """Recompute t-SNE and save a labeled scatter plot with leader lines."""
    try:
        payload = torch.load(embeddings_pt, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(embeddings_pt, map_location="cpu")
    features = payload["embeddings"]
    if not isinstance(features, torch.Tensor):
        raise TypeError("embeddings payload is not a tensor")
    protein_ids = [str(value) for value in payload["protein_ids"]]
    proteins = pd.read_csv(proteins_csv)
    csv_ids = proteins["string_id"].astype(str).tolist()
    if csv_ids != protein_ids:
        raise ValueError("proteins.csv order does not match node_embeddings.pt")

    names = proteins["preferred_name"].astype(str).tolist()
    species_names = proteins["species_name"].astype(str).tolist()
    compartments = proteins["compartment"].astype(str).tolist()
    labels = unique_seed_labels(names, species_names)

    coords = TSNE(**_tsne_kwargs(random_state)).fit_transform(features.numpy())
    labeled_idx = list(labels)
    labeled_xy = coords[labeled_idx]
    placed_xy = rightward_positions(labeled_xy, coords)

    fig, ax = plt.subplots(figsize=(9.4, 6.8), constrained_layout=True)
    for marker, species_prefix in (("o", "Homo"), ("^", "Saccharomyces")):
        idx = [i for i, spec in enumerate(species_names) if spec.startswith(species_prefix)]
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            c=[COMPARTMENT_COLORS.get(compartments[i], COMPARTMENT_COLORS["partner"]) for i in idx],
            marker=marker,
            s=32,
            linewidths=0.35,
            edgecolors="white",
            alpha=0.92,
            zorder=2,
        )

    texts = []
    labeled_x: list[float] = []
    labeled_y: list[float] = []
    for slot, index in enumerate(labeled_idx):
        point = labeled_xy[slot]
        placed = placed_xy[slot]
        labeled_x.append(float(point[0]))
        labeled_y.append(float(point[1]))
        handle = ax.text(
            float(placed[0]),
            float(placed[1]),
            labels[index],
            fontsize=8,
            ha="left",
            va="center",
            color="#111111",
            zorder=5,
        )
        texts.append(handle)

    legend = [
        Patch(facecolor=COMPARTMENT_COLORS["hippo"], label="Hippo / RAM"),
        Patch(facecolor=COMPARTMENT_COLORS["actin"], label="Actin"),
        Patch(facecolor=COMPARTMENT_COLORS["both"], label="Both"),
        Patch(facecolor=COMPARTMENT_COLORS["partner"], label="STRING partner"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="k", markersize=7, label="Human"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="k", markersize=7, label="Yeast"),
    ]
    ax.legend(handles=legend, frameon=False, fontsize=8, loc="lower left")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.suptitle("ESM-2 node embeddings", y=1.02, fontsize=12)
    x_pad = 0.04 * float(coords[:, 0].max() - coords[:, 0].min())
    y_pad = 0.04 * float(coords[:, 1].max() - coords[:, 1].min())
    ax.set_xlim(float(coords[:, 0].min()) - x_pad, max(float(coords[:, 0].max()), float(placed_xy[:, 0].max())) + 6.5)
    ax.set_ylim(float(coords[:, 1].min()) - y_pad, float(coords[:, 1].max()) + y_pad)

    _adjust_labels(texts, labeled_x, labeled_y, ax)
    for handle in texts:
        handle.set_path_effects([withStroke(linewidth=3.2, foreground="white")])
        handle.set_zorder(5)

    output = Path(output_png)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("wrote t-SNE figure %s labels=%s", output, len(texts))
    return output


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def main() -> None:
    configure_logging()
    plot_esm_tsne()


if __name__ == "__main__":
    main()
