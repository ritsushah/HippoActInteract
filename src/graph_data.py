"""PyG graph construction, per-species edge splits, and negative sampling."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

from src.config import (
    GNN_RANDOM_SEED,
    INTERACTIONS_CSV,
    LINK_BATCH_SIZE,
    LINK_TEST_RATIO,
    LINK_TRAIN_RATIO,
    LINK_VAL_RATIO,
    NEGATIVE_RATIO,
    NODE_EMBEDDINGS_PT,
    PROTEINS_CSV,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LinkPredictionGraph:
    """Node features plus leak-safe train/val/test pair tensors."""

    data: Data
    protein_ids: list[str]
    train_pos: Tensor
    train_neg: Tensor
    val_pos: Tensor
    val_neg: Tensor
    test_pos: Tensor
    test_neg: Tensor
    known_undirected: frozenset[tuple[int, int]]

    @property
    def num_nodes(self) -> int:
        return int(self.data.num_nodes or 0)


class LinkPairDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Positive and negative undirected pairs with binary labels."""

    def __init__(self, pos_edge_index: Tensor, neg_edge_index: Tensor) -> None:
        if pos_edge_index.ndim != 2 or pos_edge_index.size(0) != 2:
            raise ValueError("pos_edge_index must have shape [2, E]")
        if neg_edge_index.ndim != 2 or neg_edge_index.size(0) != 2:
            raise ValueError("neg_edge_index must have shape [2, E]")
        self.src = torch.cat([pos_edge_index[0], neg_edge_index[0]]).long()
        self.dst = torch.cat([pos_edge_index[1], neg_edge_index[1]]).long()
        n_pos = pos_edge_index.size(1)
        n_neg = neg_edge_index.size(1)
        self.y = torch.cat(
            [torch.ones(n_pos, dtype=torch.float32), torch.zeros(n_neg, dtype=torch.float32)]
        )

    def __len__(self) -> int:
        return int(self.src.numel())

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.src[index], self.dst[index], self.y[index]


def canonicalize_undirected(edge_index: Tensor) -> Tensor:
    """Drop self-loops and keep unique undirected pairs with src < dst."""
    if edge_index.numel() == 0:
        return torch.zeros(2, 0, dtype=torch.long)
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    pairs: set[tuple[int, int]] = set()
    for left, right in zip(src, dst, strict=True):
        if left == right:
            continue
        pair = (left, right) if left < right else (right, left)
        pairs.add(pair)
    if not pairs:
        return torch.zeros(2, 0, dtype=torch.long)
    ordered = sorted(pairs)
    return torch.tensor(ordered, dtype=torch.long).T.contiguous()


def to_bidirectional(edge_index: Tensor) -> Tensor:
    """Duplicate undirected pairs so message passing is symmetric."""
    if edge_index.numel() == 0:
        return torch.zeros(2, 0, dtype=torch.long)
    src, dst = edge_index[0], edge_index[1]
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0).contiguous()


def edge_pairs_set(edge_index: Tensor) -> set[tuple[int, int]]:
    """Return undirected (min, max) tuples from a [2, E] tensor."""
    canonical = canonicalize_undirected(edge_index)
    if canonical.numel() == 0:
        return set()
    return {(int(a), int(b)) for a, b in canonical.T.tolist()}


def sample_negative_pairs(
    n_samples: int,
    nodes: Tensor,
    forbidden: set[tuple[int, int]],
    rng: np.random.Generator,
    *,
    max_attempts: int | None = None,
) -> Tensor:
    """Sample unique same-set undirected pairs that are not in ``forbidden``."""
    node_list = [int(v) for v in nodes.tolist()]
    if len(node_list) < 2:
        raise ValueError("need at least two nodes to sample negative pairs")
    if n_samples < 1:
        return torch.zeros(2, 0, dtype=torch.long)
    limit = max_attempts if max_attempts is not None else max(n_samples * 200, 1000)
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []
    attempts = 0
    while len(pairs) < n_samples and attempts < limit:
        attempts += 1
        left, right = (int(v) for v in rng.choice(node_list, size=2, replace=False))
        if left == right:
            continue
        key = (left, right) if left < right else (right, left)
        if key in forbidden or key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    if len(pairs) < n_samples:
        raise RuntimeError(f"sampled only {len(pairs)}/{n_samples} negatives after {attempts} draws")
    return torch.tensor(pairs, dtype=torch.long).T.contiguous()


def _split_counts(n_edges: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1")
    n_train = int(n_edges * train_ratio)
    n_val = int(n_edges * val_ratio)
    n_test = n_edges - n_train - n_val
    if n_edges >= 3 and min(n_train, n_val, n_test) < 1:
        n_train = max(n_train, 1)
        n_val = max(n_val, 1)
        n_test = n_edges - n_train - n_val
        if n_test < 1:
            n_test = 1
            n_train = n_edges - n_val - n_test
    return n_train, n_val, n_test


def split_edge_index(
    edge_index: Tensor,
    rng: np.random.Generator,
    *,
    train_ratio: float = LINK_TRAIN_RATIO,
    val_ratio: float = LINK_VAL_RATIO,
    test_ratio: float = LINK_TEST_RATIO,
) -> tuple[Tensor, Tensor, Tensor]:
    """Shuffle undirected edges into train/val/test columns."""
    canonical = canonicalize_undirected(edge_index)
    n_edges = canonical.size(1)
    n_train, n_val, n_test = _split_counts(n_edges, train_ratio, val_ratio, test_ratio)
    perm = rng.permutation(n_edges)
    train = canonical[:, perm[:n_train]]
    val = canonical[:, perm[n_train : n_train + n_val]]
    test = canonical[:, perm[n_train + n_val : n_train + n_val + n_test]]
    return train.contiguous(), val.contiguous(), test.contiguous()


def build_link_graph(
    *,
    protein_ids: list[str],
    features: Tensor,
    species_ids: Tensor,
    undirected_edges: Tensor,
    seed: int = GNN_RANDOM_SEED,
    negative_ratio: float = NEGATIVE_RATIO,
) -> LinkPredictionGraph:
    """Assemble a PyG graph whose message-passing edges are train positives only."""
    if features.ndim != 2:
        raise ValueError(f"features must be [N, F], got {tuple(features.shape)}")
    n_nodes = features.size(0)
    if len(protein_ids) != n_nodes:
        raise ValueError("protein_ids length must match features rows")
    if species_ids.numel() != n_nodes:
        raise ValueError("species_ids length must match features rows")

    all_pos = canonicalize_undirected(undirected_edges)
    known = edge_pairs_set(all_pos)
    rng = np.random.default_rng(seed)

    train_chunks: list[Tensor] = []
    val_chunks: list[Tensor] = []
    test_chunks: list[Tensor] = []
    train_neg_chunks: list[Tensor] = []
    val_neg_chunks: list[Tensor] = []
    test_neg_chunks: list[Tensor] = []

    for species in torch.unique(species_ids).tolist():
        node_idx = torch.nonzero(species_ids == species, as_tuple=False).view(-1)
        node_set = set(int(v) for v in node_idx.tolist())
        mask = torch.tensor(
            [
                int(all_pos[0, i]) in node_set and int(all_pos[1, i]) in node_set
                for i in range(all_pos.size(1))
            ],
            dtype=torch.bool,
        )
        species_edges = all_pos[:, mask]
        if species_edges.size(1) == 0:
            logger.warning("species_id=%s has no edges; skipping split", species)
            continue
        train_e, val_e, test_e = split_edge_index(species_edges, rng)
        n_train_neg = max(int(round(train_e.size(1) * negative_ratio)), 0)
        n_val_neg = max(int(round(val_e.size(1) * negative_ratio)), 0)
        n_test_neg = max(int(round(test_e.size(1) * negative_ratio)), 0)
        train_chunks.append(train_e)
        val_chunks.append(val_e)
        test_chunks.append(test_e)
        train_neg_chunks.append(sample_negative_pairs(n_train_neg, node_idx, known, rng))
        val_neg_chunks.append(sample_negative_pairs(n_val_neg, node_idx, known, rng))
        test_neg_chunks.append(sample_negative_pairs(n_test_neg, node_idx, known, rng))
        logger.info(
            "species=%s nodes=%s edges train/val/test=%s/%s/%s",
            species,
            int(node_idx.numel()),
            train_e.size(1),
            val_e.size(1),
            test_e.size(1),
        )

    train_pos = _cat_edges(train_chunks)
    val_pos = _cat_edges(val_chunks)
    test_pos = _cat_edges(test_chunks)
    train_neg = _cat_edges(train_neg_chunks)
    val_neg = _cat_edges(val_neg_chunks)
    test_neg = _cat_edges(test_neg_chunks)

    data = Data(
        x=features.float().contiguous(),
        edge_index=to_bidirectional(train_pos),
        species=species_ids.long().contiguous(),
    )
    data.num_nodes = n_nodes
    graph = LinkPredictionGraph(
        data=data,
        protein_ids=list(protein_ids),
        train_pos=train_pos,
        train_neg=train_neg,
        val_pos=val_pos,
        val_neg=val_neg,
        test_pos=test_pos,
        test_neg=test_neg,
        known_undirected=frozenset(known),
    )
    logger.info(
        "link graph nodes=%s mp_edges=%s train_pos=%s val_pos=%s test_pos=%s",
        graph.num_nodes,
        int(data.edge_index.size(1)),
        int(train_pos.size(1)),
        int(val_pos.size(1)),
        int(test_pos.size(1)),
    )
    return graph


def load_link_graph(
    *,
    proteins_csv: Path = PROTEINS_CSV,
    interactions_csv: Path = INTERACTIONS_CSV,
    embeddings_pt: Path = NODE_EMBEDDINGS_PT,
    seed: int = GNN_RANDOM_SEED,
) -> LinkPredictionGraph:
    """Load Phase 1/2 artifacts and build the link-prediction graph."""
    proteins = pd.read_csv(proteins_csv)
    if "string_id" not in proteins.columns or "species_id" not in proteins.columns:
        raise ValueError(f"{proteins_csv} must contain string_id and species_id")
    csv_ids = proteins["string_id"].astype(str).tolist()
    try:
        payload = torch.load(embeddings_pt, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(embeddings_pt, map_location="cpu")
    embed_ids = [str(v) for v in payload["protein_ids"]]
    if embed_ids != csv_ids:
        raise ValueError("node_embeddings.pt protein_ids do not match proteins.csv order")
    features = payload["embeddings"]
    if not isinstance(features, Tensor):
        raise TypeError("embeddings payload is not a tensor")

    id_to_index = {pid: i for i, pid in enumerate(csv_ids)}
    interactions = pd.read_csv(interactions_csv)
    src_ids = interactions["source_string_id"].astype(str)
    dst_ids = interactions["target_string_id"].astype(str)
    missing = [pid for pid in list(src_ids) + list(dst_ids) if pid not in id_to_index]
    if missing:
        raise KeyError(f"{len(missing)} interaction endpoints missing from proteins.csv")
    src = torch.tensor([id_to_index[pid] for pid in src_ids], dtype=torch.long)
    dst = torch.tensor([id_to_index[pid] for pid in dst_ids], dtype=torch.long)
    undirected = torch.stack([src, dst], dim=0)
    species_ids = torch.tensor(proteins["species_id"].astype(int).tolist(), dtype=torch.long)
    return build_link_graph(
        protein_ids=csv_ids,
        features=features,
        species_ids=species_ids,
        undirected_edges=undirected,
        seed=seed,
    )


def make_pair_loader(
    pos_edge_index: Tensor,
    neg_edge_index: Tensor,
    *,
    batch_size: int = LINK_BATCH_SIZE,
    shuffle: bool = True,
    seed: int = GNN_RANDOM_SEED,
) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
    """Mini-batch labeled pairs for BCE-with-logits training."""
    dataset = LinkPairDataset(pos_edge_index, neg_edge_index)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        drop_last=False,
    )


def _cat_edges(chunks: list[Tensor]) -> Tensor:
    nonempty = [c for c in chunks if c.numel() > 0]
    if not nonempty:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.cat(nonempty, dim=1).contiguous()
