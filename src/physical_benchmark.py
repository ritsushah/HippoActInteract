"""Physical-label link-prediction benchmark on the 359-protein STRING graph.

STRING ≥ 700 is the context graph, not the evaluation label. Positives come
from BioGRID physical / IntAct records among proteins.csv UniProt IDs.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_

from src.config import (
    ATLAS_CSV,
    BENCHMARK_EPOCHS,
    BENCHMARK_FIGURE_PNG,
    BENCHMARK_METRICS_JSON,
    BENCHMARK_N_SEEDS,
    BENCHMARK_PATIENCE,
    BENCHMARK_SPLITS_DIR,
    BENCHMARK_STABILITY_CSV,
    DATA_PROCESSED,
    FIGURES_DIR,
    GNN_DROPOUT,
    GNN_HIDDEN_SIZE,
    INTERACTIONS_CSV,
    LINK_TEST_RATIO,
    LINK_TRAIN_RATIO,
    LINK_VAL_RATIO,
    NEGATIVE_RATIO,
    NODE_EMBEDDINGS_PT,
    PHYSICAL_EDGES_CSV,
    PROTEINS_CSV,
    TRAIN_LR,
    TRAIN_WEIGHT_DECAY,
)
from src.device import get_device
from src.gnn_model import LinkPredictor, MLPDecoder
from src.graph_data import (
    LinkPredictionGraph,
    make_pair_loader,
    sample_negative_pairs,
    to_bidirectional,
)
from src.train import compute_binary_metrics, evaluate_split, hippo_actin_candidates, score_pairs, seed_everything

logger = logging.getLogger(__name__)

PROTOCOLS: tuple[str, ...] = ("edge_random", "node_disjoint", "degree_matched")
NEURAL_MODELS: tuple[str, ...] = ("esm_mlp", "sage_graph", "sage_esm", "sage_perm")
TOPOLOGY_MODELS: tuple[str, ...] = (
    "jaccard",
    "adamic_adar",
    "preferential_attachment",
    "l3",
    "logreg",
)
_GRAD_CLIP = 1.0


def load_physical_index_pairs(
    proteins: pd.DataFrame,
    physical: pd.DataFrame,
) -> tuple[list[tuple[int, int]], dict[int, list[int]], Tensor]:
    """Map physical edges onto node indices; group nodes by species."""
    id_to_index = {str(pid): i for i, pid in enumerate(proteins["string_id"].astype(str))}
    species_ids = torch.tensor(proteins["species_id"].astype(int).tolist(), dtype=torch.long)
    by_species: dict[int, list[int]] = {}
    for i, sid in enumerate(species_ids.tolist()):
        by_species.setdefault(int(sid), []).append(i)
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for row in physical.itertuples(index=False):
        src = id_to_index.get(str(row.string_id_a))
        dst = id_to_index.get(str(row.string_id_b))
        if src is None or dst is None or src == dst:
            continue
        key = (src, dst) if src < dst else (dst, src)
        if key in seen:
            continue
        if int(species_ids[src]) != int(species_ids[dst]):
            continue
        seen.add(key)
        pairs.append(key)
    return pairs, by_species, species_ids


def string_undirected_pairs(proteins: pd.DataFrame, interactions: pd.DataFrame) -> set[tuple[int, int]]:
    """STRING ≥ 700 edges as undirected index pairs."""
    id_to_index = {str(pid): i for i, pid in enumerate(proteins["string_id"].astype(str))}
    pairs: set[tuple[int, int]] = set()
    for row in interactions.itertuples(index=False):
        src = id_to_index.get(str(row.source_string_id))
        dst = id_to_index.get(str(row.target_string_id))
        if src is None or dst is None or src == dst:
            continue
        pairs.add((src, dst) if src < dst else (dst, src))
    return pairs


def _split_counts(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val
    if n >= 3 and min(n_train, n_val, n_test) < 1:
        n_train = max(n_train, 1)
        n_val = max(n_val, 1)
        n_test = max(n - n_train - n_val, 1)
        n_train = n - n_val - n_test
    return max(n_train, 0), max(n_val, 0), max(n_test, 0)


def split_edge_random(
    positives: list[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    """Shuffle physical positives into train/val/test."""
    if not positives:
        return [], [], []
    order = rng.permutation(len(positives))
    n_train, n_val, n_test = _split_counts(len(positives), LINK_TRAIN_RATIO, LINK_VAL_RATIO, LINK_TEST_RATIO)
    train = [positives[i] for i in order[:n_train]]
    val = [positives[i] for i in order[n_train : n_train + n_val]]
    test = [positives[i] for i in order[n_train + n_val : n_train + n_val + n_test]]
    return train, val, test


def split_node_disjoint(
    positives: list[tuple[int, int]],
    by_species: dict[int, list[int]],
    species_ids: Tensor,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    """Keep only physical edges whose both ends sit in the same node bucket."""
    train_nodes: set[int] = set()
    val_nodes: set[int] = set()
    test_nodes: set[int] = set()
    for nodes in by_species.values():
        shuffled = [int(i) for i in rng.permutation(nodes)]
        n_train, n_val, n_test = _split_counts(len(shuffled), 0.7, 0.15, 0.15)
        train_nodes.update(shuffled[:n_train])
        val_nodes.update(shuffled[n_train : n_train + n_val])
        test_nodes.update(shuffled[n_train + n_val :])
        _ = n_test
    train = [p for p in positives if p[0] in train_nodes and p[1] in train_nodes]
    val = [p for p in positives if p[0] in val_nodes and p[1] in val_nodes]
    test = [p for p in positives if p[0] in test_nodes and p[1] in test_nodes]
    _ = species_ids
    return train, val, test


def _degrees_from_string(
    n_nodes: int,
    string_pairs: set[tuple[int, int]],
    holdout: set[tuple[int, int]],
) -> np.ndarray:
    deg = np.zeros(n_nodes, dtype=np.int64)
    for src, dst in string_pairs:
        if (src, dst) in holdout:
            continue
        deg[src] += 1
        deg[dst] += 1
    return deg


def sample_degree_matched_negatives(
    positives: list[tuple[int, int]],
    nodes: Tensor,
    forbidden: set[tuple[int, int]],
    degrees: np.ndarray,
    rng: np.random.Generator,
    *,
    log_tol: float = 0.35,
    max_attempts: int = 400,
) -> Tensor:
    """One negative per positive with a similar STRING degree product."""
    node_list = [int(v) for v in nodes.tolist()]
    if len(node_list) < 2 or not positives:
        return torch.zeros(2, 0, dtype=torch.long)
    seen: set[tuple[int, int]] = set()
    picked: list[tuple[int, int]] = []
    for src, dst in positives:
        target = math.log(float(degrees[src] * degrees[dst]) + 1.0)
        best: tuple[int, int] | None = None
        best_gap = float("inf")
        for _ in range(max_attempts):
            left, right = (int(v) for v in rng.choice(node_list, size=2, replace=False))
            key = (left, right) if left < right else (right, left)
            if key in forbidden or key in seen:
                continue
            gap = abs(math.log(float(degrees[left] * degrees[right]) + 1.0) - target)
            if gap < best_gap:
                best_gap = gap
                best = key
            if gap <= log_tol:
                break
        if best is None:
            continue
        seen.add(best)
        picked.append(best)
    if len(picked) < len(positives):
        extra = sample_negative_pairs(
            len(positives) - len(picked),
            nodes,
            forbidden | seen,
            rng,
        )
        extra_pairs = [(int(a), int(b)) for a, b in extra.T.tolist()]
        picked.extend(extra_pairs)
    return torch.tensor(picked[: len(positives)], dtype=torch.long).T.contiguous()


def _pairs_to_tensor(pairs: list[tuple[int, int]]) -> Tensor:
    if not pairs:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).T.contiguous()


def _cat(chunks: list[Tensor]) -> Tensor:
    nonempty = [c for c in chunks if c.numel() > 0]
    if not nonempty:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.cat(nonempty, dim=1).contiguous()


def build_protocol_split(
    *,
    protocol: str,
    positives: list[tuple[int, int]],
    by_species: dict[int, list[int]],
    species_ids: Tensor,
    string_pairs: set[tuple[int, int]],
    rng: np.random.Generator,
    n_nodes: int,
    negative_ratio: float = NEGATIVE_RATIO,
) -> dict[str, Any]:
    """Frozen positive/negative tensors plus leak-safe STRING message-passing edges."""
    if protocol == "node_disjoint":
        train_pos, val_pos, test_pos = split_node_disjoint(positives, by_species, species_ids, rng)
    else:
        train_pos, val_pos, test_pos = split_edge_random(positives, rng)

    holdout = set(val_pos) | set(test_pos)
    forbidden = set(positives)
    degrees = _degrees_from_string(n_nodes, string_pairs, holdout)
    train_neg_chunks: list[Tensor] = []
    val_neg_chunks: list[Tensor] = []
    test_neg_chunks: list[Tensor] = []

    for species, nodes in by_species.items():
        node_tensor = torch.tensor(nodes, dtype=torch.long)
        node_set = set(nodes)
        tr = [p for p in train_pos if p[0] in node_set]
        va = [p for p in val_pos if p[0] in node_set]
        te = [p for p in test_pos if p[0] in node_set]
        n_tr = max(int(round(len(tr) * negative_ratio)), 0)
        n_va = max(int(round(len(va) * negative_ratio)), 0)
        n_te = max(int(round(len(te) * negative_ratio)), 0)
        if protocol == "degree_matched":
            train_neg_chunks.append(
                sample_degree_matched_negatives(tr, node_tensor, forbidden, degrees, rng)
                if tr
                else torch.zeros(2, 0, dtype=torch.long)
            )
            val_neg_chunks.append(
                sample_degree_matched_negatives(va, node_tensor, forbidden, degrees, rng)
                if va
                else torch.zeros(2, 0, dtype=torch.long)
            )
            test_neg_chunks.append(
                sample_degree_matched_negatives(te, node_tensor, forbidden, degrees, rng)
                if te
                else torch.zeros(2, 0, dtype=torch.long)
            )
        else:
            train_neg_chunks.append(sample_negative_pairs(n_tr, node_tensor, forbidden, rng) if n_tr else torch.zeros(2, 0, dtype=torch.long))
            val_neg_chunks.append(sample_negative_pairs(n_va, node_tensor, forbidden, rng) if n_va else torch.zeros(2, 0, dtype=torch.long))
            test_neg_chunks.append(sample_negative_pairs(n_te, node_tensor, forbidden, rng) if n_te else torch.zeros(2, 0, dtype=torch.long))
        _ = species

    mp_pairs = [p for p in string_pairs if p not in holdout]
    if protocol == "node_disjoint":
        train_nodes = {i for pair in train_pos for i in pair}
        if train_nodes:
            mp_pairs = [p for p in mp_pairs if p[0] in train_nodes and p[1] in train_nodes]
        else:
            mp_pairs = []

    split = {
        "protocol": protocol,
        "train_pos": _pairs_to_tensor(train_pos),
        "val_pos": _pairs_to_tensor(val_pos),
        "test_pos": _pairs_to_tensor(test_pos),
        "train_neg": _cat(train_neg_chunks),
        "val_neg": _cat(val_neg_chunks),
        "test_neg": _cat(test_neg_chunks),
        "mp_edges": to_bidirectional(_pairs_to_tensor(mp_pairs)),
        "n_train_pos": len(train_pos),
        "n_val_pos": len(val_pos),
        "n_test_pos": len(test_pos),
    }
    return split


def save_split(split: dict[str, Any], path: Path) -> None:
    """Write pair lists so a seed can be replayed without retraining."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v.cpu() if isinstance(v, Tensor) else v for k, v in split.items()}
    torch.save(payload, path)


def neighbor_sets(n_nodes: int, undirected: set[tuple[int, int]] | list[tuple[int, int]]) -> list[set[int]]:
    neighbors: list[set[int]] = [set() for _ in range(n_nodes)]
    for src, dst in undirected:
        neighbors[src].add(dst)
        neighbors[dst].add(src)
    return neighbors


def topology_scores(
    name: str,
    pairs: list[tuple[int, int]],
    neighbors: list[set[int]],
    degrees: np.ndarray,
    adjacency: np.ndarray | None,
) -> np.ndarray:
    """Heuristic scores on the leak-safe STRING context graph."""
    scores = np.zeros(len(pairs), dtype=np.float64)
    for i, (src, dst) in enumerate(pairs):
        n_src = neighbors[src]
        n_dst = neighbors[dst]
        shared = n_src & n_dst
        if name == "jaccard":
            union = n_src | n_dst
            scores[i] = (len(shared) / len(union)) if union else 0.0
        elif name == "adamic_adar":
            scores[i] = sum(1.0 / math.log(degrees[w]) for w in shared if degrees[w] > 1)
        elif name == "preferential_attachment":
            scores[i] = float(degrees[src] * degrees[dst])
        elif name == "l3":
            if adjacency is None:
                raise ValueError("L3 requires an adjacency matrix")
            scores[i] = float(adjacency[src, dst])
        else:
            raise KeyError(name)
    return scores


def pair_feature_matrix(
    pairs: list[tuple[int, int]],
    degrees: np.ndarray,
    neighbors: list[set[int]],
    loc_compatible: np.ndarray,
) -> np.ndarray:
    """Degree, common neighbors, and localization overlap for logistic regression."""
    feats = np.zeros((len(pairs), 5), dtype=np.float64)
    for i, (src, dst) in enumerate(pairs):
        shared = len(neighbors[src] & neighbors[dst])
        feats[i, 0] = degrees[src]
        feats[i, 1] = degrees[dst]
        feats[i, 2] = degrees[src] * degrees[dst]
        feats[i, 3] = shared
        feats[i, 4] = loc_compatible[src] * loc_compatible[dst]
    return feats


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Precision among the k highest scores."""
    if y_true.size == 0 or k < 1:
        return float("nan")
    k = min(k, int(y_true.size))
    order = np.argsort(-scores, kind="mergesort")[:k]
    return float(y_true[order].mean())


def metrics_from_scores(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """AUROC, AP, and precision@10/20/50. Requires both classes."""
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "ap": float("nan"), "p_at_10": float("nan"), "p_at_20": float("nan"), "p_at_50": float("nan")}
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "ap": float(average_precision_score(y_true, scores)),
        "p_at_10": precision_at_k(y_true, scores, 10),
        "p_at_20": precision_at_k(y_true, scores, 20),
        "p_at_50": precision_at_k(y_true, scores, 50),
    }


class PairMLP(nn.Module):
    """ESM-only decoder: no message passing."""

    def __init__(self, in_channels: int, hidden_channels: int = GNN_HIDDEN_SIZE, dropout: float = GNN_DROPOUT) -> None:
        super().__init__()
        self.proj = nn.Linear(in_channels, hidden_channels)
        self.decoder = MLPDecoder(hidden_channels, dropout)

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        _ = edge_index
        return torch.relu(self.proj(x))

    def decode(self, z: Tensor, src: Tensor, dst: Tensor) -> Tensor:
        return self.decoder(z, src, dst)

    def forward(self, x: Tensor, edge_index: Tensor, src: Tensor, dst: Tensor) -> Tensor:
        return self.decode(self.encode(x, edge_index), src, dst)


def _graph_from_split(features: Tensor, split: dict[str, Any], protein_ids: list[str]) -> LinkPredictionGraph:
    data_edge = split["mp_edges"]
    dummy_species = torch.zeros(features.size(0), dtype=torch.long)
    from torch_geometric.data import Data

    data = Data(x=features.float().contiguous(), edge_index=data_edge, species=dummy_species)
    data.num_nodes = features.size(0)
    known = frozenset()
    return LinkPredictionGraph(
        data=data,
        protein_ids=protein_ids,
        train_pos=split["train_pos"],
        train_neg=split["train_neg"],
        val_pos=split["val_pos"],
        val_neg=split["val_neg"],
        test_pos=split["test_pos"],
        test_neg=split["test_neg"],
        known_undirected=known,
    )


def train_neural(
    model: nn.Module,
    graph: LinkPredictionGraph,
    device: torch.device,
    *,
    epochs: int = BENCHMARK_EPOCHS,
    patience: int = BENCHMARK_PATIENCE,
) -> nn.Module:
    """Adam + early stopping on validation AUROC, same loop as STRING training."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_LR, weight_decay=TRAIN_WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    best_state: dict[str, Tensor] | None = None
    best_val = -1.0
    stale = 0
    if graph.train_pos.numel() == 0 or graph.train_neg.numel() == 0:
        raise RuntimeError("empty training split")
    if graph.val_pos.numel() == 0 or graph.val_neg.numel() == 0:
        logger.warning("empty val split; training a fixed number of epochs")
        patience = epochs
    for _epoch in range(1, epochs + 1):
        model.train()
        x = graph.data.x.to(device)
        edge_index = graph.data.edge_index.to(device)
        src = torch.cat([graph.train_pos[0], graph.train_neg[0]]).to(device)
        dst = torch.cat([graph.train_pos[1], graph.train_neg[1]]).to(device)
        labels = torch.cat(
            [
                torch.ones(graph.train_pos.size(1), dtype=torch.float32),
                torch.zeros(graph.train_neg.size(1), dtype=torch.float32),
            ]
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, edge_index, src, dst)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        clip_grad_norm_(model.parameters(), _GRAD_CLIP)
        optimizer.step()
        if graph.val_pos.numel() == 0 or graph.val_neg.numel() == 0:
            best_state = deepcopy(model.state_dict())
            continue
        try:
            val = evaluate_split(model, graph, graph.val_pos, graph.val_neg, device=device, criterion=criterion)
        except ValueError:
            best_state = deepcopy(model.state_dict())
            continue
        if val["auroc"] > best_val + 1e-6:
            best_val = val["auroc"]
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def score_split_model(model: nn.Module, graph: LinkPredictionGraph, device: torch.device) -> dict[str, float]:
    """Test AUROC/AP/P@K from logits."""
    criterion = nn.BCEWithLogitsLoss()
    if graph.test_pos.numel() == 0 or graph.test_neg.numel() == 0:
        return {"auroc": float("nan"), "ap": float("nan"), "p_at_10": float("nan"), "p_at_20": float("nan"), "p_at_50": float("nan")}
    model.eval()
    x = graph.data.x.to(device)
    edge_index = graph.data.edge_index.to(device)
    loader = make_pair_loader(graph.test_pos, graph.test_neg, shuffle=False)
    logit_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        z = model.encode(x, edge_index)
        for src, dst, labels in loader:
            src, dst = src.to(device), dst.to(device)
            logits = model.decode(z, src, dst)
            logit_chunks.append(logits.detach().cpu().numpy())
            label_chunks.append(labels.numpy())
    y_true = np.concatenate(label_chunks)
    logits = np.concatenate(logit_chunks)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    out = metrics_from_scores(y_true, probs)
    try:
        out.update({f"bce_{k}": v for k, v in compute_binary_metrics(y_true, logits).items()})
    except ValueError:
        pass
    _ = criterion
    return out


def labeled_pairs(pos: Tensor, neg: Tensor) -> tuple[list[tuple[int, int]], np.ndarray]:
    pairs: list[tuple[int, int]] = []
    labels: list[float] = []
    if pos.numel():
        for src, dst in pos.T.tolist():
            pairs.append((int(src), int(dst)))
            labels.append(1.0)
    if neg.numel():
        for src, dst in neg.T.tolist():
            pairs.append((int(src), int(dst)))
            labels.append(0.0)
    return pairs, np.asarray(labels, dtype=np.float64)


def mp_undirected_set(mp_edges: Tensor) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    if mp_edges.numel() == 0:
        return pairs
    for src, dst in mp_edges.T.tolist():
        if src == dst:
            continue
        pairs.add((int(src), int(dst)) if src < dst else (int(dst), int(src)))
    return pairs


def run_topology_models(
    split: dict[str, Any],
    n_nodes: int,
    loc_compatible: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Jaccard, Adamic–Adar, preferential attachment, L3, and logistic regression."""
    context = mp_undirected_set(split["mp_edges"])
    neighbors = neighbor_sets(n_nodes, context)
    degrees = np.array([len(n) for n in neighbors], dtype=np.int64)
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for src, dst in context:
        adj[src, dst] = 1.0
        adj[dst, src] = 1.0
    a3 = adj @ adj @ adj
    test_pairs, y_true = labeled_pairs(split["test_pos"], split["test_neg"])
    train_pairs, y_train = labeled_pairs(split["train_pos"], split["train_neg"])
    results: dict[str, dict[str, float]] = {}
    for name in ("jaccard", "adamic_adar", "preferential_attachment", "l3"):
        matrix = a3 if name == "l3" else adj
        scores = topology_scores(name, test_pairs, neighbors, degrees, matrix)
        results[name] = metrics_from_scores(y_true, scores)
    if train_pairs and test_pairs and len(np.unique(y_train)) == 2 and len(np.unique(y_true)) == 2:
        x_train = pair_feature_matrix(train_pairs, degrees, neighbors, loc_compatible)
        x_test = pair_feature_matrix(test_pairs, degrees, neighbors, loc_compatible)
        clf = LogisticRegression(max_iter=400, solver="lbfgs")
        clf.fit(x_train, y_train)
        results["logreg"] = metrics_from_scores(y_true, clf.predict_proba(x_test)[:, 1])
    else:
        results["logreg"] = metrics_from_scores(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        results["logreg"] = {k: float("nan") for k in results["logreg"]}
    return results


def make_neural_model(name: str, features: Tensor, rng: np.random.Generator) -> tuple[nn.Module, Tensor]:
    """Instantiate ESM-only, graph-only, combined, or permutation-ablated GraphSAGE."""
    if name == "esm_mlp":
        return PairMLP(int(features.size(1))), features
    if name == "sage_graph":
        dummy = torch.ones((features.size(0), 16), dtype=torch.float32)
        return LinkPredictor(in_channels=16, hidden_channels=GNN_HIDDEN_SIZE), dummy
    if name == "sage_esm":
        return LinkPredictor(in_channels=int(features.size(1)), hidden_channels=GNN_HIDDEN_SIZE), features
    if name == "sage_perm":
        perm = rng.permutation(features.size(0))
        shuffled = features[torch.as_tensor(perm, dtype=torch.long)]
        return LinkPredictor(in_channels=int(features.size(1)), hidden_channels=GNN_HIDDEN_SIZE), shuffled
    raise KeyError(name)


def localization_flags(proteins: pd.DataFrame, atlas: pd.DataFrame) -> np.ndarray:
    """1 if the protein has a parsed location bucket, else 0 (logreg feature)."""
    flags = np.zeros(len(proteins), dtype=np.float64)
    loc: dict[str, str] = {}
    if not atlas.empty:
        for row in atlas.itertuples(index=False):
            loc[str(row.string_id_a)] = str(row.location_a)
            loc[str(row.string_id_b)] = str(row.location_b)
    for i, pid in enumerate(proteins["string_id"].astype(str)):
        text = loc.get(pid, "")
        flags[i] = 1.0 if text and text not in {"nan", ""} else 0.0
    return flags


def mean_sd_ci(values: list[float]) -> dict[str, float]:
    """Mean, sd, and a normal-approx 95% CI over seeds."""
    arr = np.asarray([v for v in values if v == v], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "ci95_lo": float("nan"), "ci95_hi": float("nan"), "n": 0}
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    se = sd / math.sqrt(arr.size) if arr.size else float("nan")
    return {
        "mean": mean,
        "sd": sd,
        "ci95_lo": mean - 1.96 * se,
        "ci95_hi": mean + 1.96 * se,
        "n": int(arr.size),
    }


def plot_benchmark(summary: dict[str, Any], path: Path) -> None:
    """Grouped AUROC bars for the primary edge-random protocol."""
    path.parent.mkdir(parents=True, exist_ok=True)
    protocol = summary.get("protocols", {}).get("edge_random", {})
    if not protocol:
        return
    names: list[str] = []
    means: list[float] = []
    los: list[float] = []
    his: list[float] = []
    for model in list(TOPOLOGY_MODELS) + list(NEURAL_MODELS):
        block = protocol.get(model)
        if not block:
            continue
        names.append(model.replace("_", " "))
        means.append(block["auroc"]["mean"])
        los.append(block["auroc"]["ci95_lo"])
        his.append(block["auroc"]["ci95_hi"])
    if not names:
        return
    fig, ax = plt.subplots(figsize=(9.5, 4.4), constrained_layout=True)
    x = np.arange(len(names))
    yerr = np.vstack([np.array(means) - np.array(los), np.array(his) - np.array(means)])
    yerr = np.nan_to_num(yerr, nan=0.0)
    ax.bar(x, means, color="#2E5A88", width=0.65, yerr=yerr, capsize=3, ecolor="#333")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("Test AUROC (mean ± 95% CI)")
    ax.set_title("Physical-label benchmark (edge-random splits, 20 seeds)")
    ax.set_ylim(0.45, 1.02)
    ax.axhline(0.5, color="#999", lw=0.8, ls="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def candidate_stability(
    proteins: pd.DataFrame,
    atlas: pd.DataFrame,
    models_and_graphs: list[tuple[nn.Module, LinkPredictionGraph, torch.device]],
) -> pd.DataFrame:
    """Median GraphSAGE rank of STRING-absent Hippo×actin pairs across seeds."""
    if atlas.empty or not models_and_graphs:
        return pd.DataFrame()
    compartments = proteins["compartment"].astype(str).tolist()
    species_ids = proteins["species_id"].astype(int).tolist()
    string_known = string_undirected_pairs(proteins, pd.read_csv(INTERACTIONS_CSV))
    candidates = hippo_actin_candidates(compartments, species_ids, string_known)
    names = proteins["preferred_name"].astype(str).tolist()
    species_names = proteins["species_name"].astype(str).tolist()
    device = get_device()
    ranks: dict[tuple[int, int], list[int]] = {pair: [] for pair in candidates}
    for model, graph, _stored_device in models_and_graphs:
        model = model.to(device)
        model.eval()
        x = graph.data.x.to(device)
        edge_index = graph.data.edge_index.to(device)
        with torch.inference_mode():
            z = model.encode(x, edge_index)
            probs = score_pairs(model, z, candidates).cpu().numpy()
        order = np.argsort(-probs, kind="mergesort")
        for rank, idx in enumerate(order, start=1):
            ranks[candidates[idx]].append(rank)
    rows = []
    for (src, dst), rank_list in ranks.items():
        if not rank_list:
            continue
        arr = np.asarray(rank_list)
        rows.append(
            {
                "species": species_names[src],
                "protein_a": names[src],
                "protein_b": names[dst],
                "median_rank": float(np.median(arr)),
                "mean_rank": float(arr.mean()),
                "frac_top10": float((arr <= 10).mean()),
                "frac_top20": float((arr <= 20).mean()),
                "n_seeds": int(arr.size),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values("median_rank", kind="mergesort").reset_index(drop=True)


def run_benchmark(
    *,
    n_seeds: int = BENCHMARK_N_SEEDS,
    epochs: int = BENCHMARK_EPOCHS,
    proteins_csv: Path = PROTEINS_CSV,
    skip_neural: bool = False,
) -> dict[str, Any]:
    """20-seed physical-label benchmark with frozen splits and ablations."""
    proteins = pd.read_csv(proteins_csv)
    interactions = pd.read_csv(INTERACTIONS_CSV)
    if not PHYSICAL_EDGES_CSV.is_file():
        raise FileNotFoundError(f"{PHYSICAL_EDGES_CSV} missing; run the evidence atlas first")
    physical = pd.read_csv(PHYSICAL_EDGES_CSV)
    atlas = pd.read_csv(ATLAS_CSV) if ATLAS_CSV.is_file() else pd.DataFrame()
    positives, by_species, species_ids = load_physical_index_pairs(proteins, physical)
    string_pairs = string_undirected_pairs(proteins, interactions)
    if len(positives) < 12:
        raise RuntimeError(f"only {len(positives)} physical edges in the 359-protein universe; too few to split")

    try:
        payload = torch.load(NODE_EMBEDDINGS_PT, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(NODE_EMBEDDINGS_PT, map_location="cpu")
    features = payload["embeddings"]
    if not isinstance(features, Tensor):
        features = torch.as_tensor(features)
    protein_ids = proteins["string_id"].astype(str).tolist()
    loc_flags = localization_flags(proteins, atlas)
    device = get_device()
    BENCHMARK_SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    raw: dict[str, dict[str, dict[str, list[float]]]] = {
        proto: {model: {"auroc": [], "ap": [], "p_at_10": [], "p_at_20": [], "p_at_50": []} for model in list(TOPOLOGY_MODELS) + list(NEURAL_MODELS)}
        for proto in PROTOCOLS
    }
    paired_delta: list[float] = []
    stability_models: list[tuple[nn.Module, LinkPredictionGraph, torch.device]] = []
    split_inventory: list[dict[str, Any]] = []

    for seed in range(n_seeds):
        seed_everything(seed)
        rng = np.random.default_rng(seed)
        for protocol in PROTOCOLS:
            try:
                split = build_protocol_split(
                    protocol=protocol,
                    positives=positives,
                    by_species=by_species,
                    species_ids=species_ids,
                    string_pairs=string_pairs,
                    rng=rng,
                    n_nodes=len(protein_ids),
                )
            except RuntimeError as exc:
                logger.warning("seed=%s protocol=%s skipped: %s", seed, protocol, exc)
                continue
            if split["n_test_pos"] < 2 or split["test_neg"].numel() == 0:
                logger.warning("seed=%s protocol=%s has too few test pairs", seed, protocol)
                continue
            split_path = BENCHMARK_SPLITS_DIR / f"{protocol}_seed{seed}.pt"
            save_split(split, split_path)
            split_inventory.append(
                {
                    "seed": seed,
                    "protocol": protocol,
                    "path": str(split_path),
                    "n_train_pos": split["n_train_pos"],
                    "n_val_pos": split["n_val_pos"],
                    "n_test_pos": split["n_test_pos"],
                }
            )
            topo = run_topology_models(split, len(protein_ids), loc_flags)
            for model_name, metrics in topo.items():
                for key in raw[protocol][model_name]:
                    raw[protocol][model_name][key].append(metrics[key])
            if skip_neural:
                continue
            aa = topo.get("adamic_adar", {}).get("auroc", float("nan"))
            neural_names = NEURAL_MODELS
            for model_name in neural_names:
                rng_model = np.random.default_rng(seed + 17)
                model, feats = make_neural_model(model_name, features, rng_model)
                graph = _graph_from_split(feats, split, protein_ids)
                try:
                    model = train_neural(model, graph, device, epochs=epochs)
                    metrics = score_split_model(model, graph, device)
                except (RuntimeError, ValueError) as exc:
                    logger.warning("seed=%s %s/%s failed: %s", seed, protocol, model_name, exc)
                    continue
                logger.info(
                    "seed=%s protocol=%s model=%s auroc=%.3f ap=%.3f",
                    seed,
                    protocol,
                    model_name,
                    metrics["auroc"],
                    metrics["ap"],
                )
                for key in raw[protocol][model_name]:
                    raw[protocol][model_name][key].append(metrics[key])
                if protocol == "edge_random" and model_name == "sage_esm":
                    if metrics["auroc"] == metrics["auroc"] and aa == aa:
                        paired_delta.append(metrics["auroc"] - aa)
                    stability_models.append((model.cpu(), graph, torch.device("cpu")))

    summarized: dict[str, Any] = {"protocols": {}}
    for protocol, models in raw.items():
        summarized["protocols"][protocol] = {}
        for model_name, series in models.items():
            summarized["protocols"][protocol][model_name] = {metric: mean_sd_ci(vals) for metric, vals in series.items()}
    summarized["paired_sage_minus_adamic_adar"] = mean_sd_ci(paired_delta)
    summarized["n_physical_positives"] = len(positives)
    summarized["n_tier1"] = int(physical["tier1"].astype(bool).sum()) if "tier1" in physical.columns else None
    summarized["n_seeds"] = n_seeds
    summarized["splits"] = split_inventory
    summarized["label"] = "BioGRID physical or IntAct (STRING is context, not the label)"

    stability = candidate_stability(proteins, atlas, stability_models)
    if not stability.empty:
        stability.to_csv(BENCHMARK_STABILITY_CSV, index=False)
        summarized["stability_top"] = stability.head(10).to_dict(orient="records")
        summarized["stability_csv"] = str(BENCHMARK_STABILITY_CSV)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_METRICS_JSON.write_text(json.dumps(summarized, indent=2) + "\n", encoding="utf-8")
    plot_benchmark(summarized, BENCHMARK_FIGURE_PNG)
    logger.info("wrote %s", BENCHMARK_METRICS_JSON)
    return summarized


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def main() -> None:
    configure_logging()
    summary = run_benchmark()
    edge = summary["protocols"].get("edge_random", {})
    sage = edge.get("sage_esm", {}).get("auroc", {})
    aa = edge.get("adamic_adar", {}).get("auroc", {})
    logger.info(
        "physical benchmark edge_random sage_esm AUROC=%.3f±%.3f vs Adamic–Adar %.3f±%.3f",
        sage.get("mean", float("nan")),
        sage.get("sd", float("nan")),
        aa.get("mean", float("nan")),
        aa.get("sd", float("nan")),
    )


if __name__ == "__main__":
    main()
