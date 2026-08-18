"""Train the GraphSAGE link predictor and export a Hippo–actin hit list."""

from __future__ import annotations

import json
import logging
import sys
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_

from src.config import (
    DATA_PROCESSED,
    GNN_CHECKPOINT_PT,
    GNN_HIDDEN_SIZE,
    GNN_RANDOM_SEED,
    HIT_LIST_CSV,
    HIT_LIST_TOP_K,
    PROTEINS_CSV,
    TRAIN_EPOCHS,
    TRAIN_LR,
    TRAIN_METRICS_JSON,
    TRAIN_PATIENCE,
    TRAIN_WEIGHT_DECAY,
)
from src.device import get_device
from src.gnn_model import LinkPredictor
from src.graph_data import LinkPredictionGraph, load_link_graph, make_pair_loader, to_bidirectional

logger = logging.getLogger(__name__)

HIPPO_LIKE: frozenset[str] = frozenset({"hippo", "both"})
ACTIN_LIKE: frozenset[str] = frozenset({"actin", "both"})
_GRAD_CLIP: float = 1.0


def seed_everything(seed: int = GNN_RANDOM_SEED) -> None:
    """Seed Python, NumPy, and Torch for a repeatable run."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_binary_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    """AUROC and average precision from logits; requires both classes."""
    if y_true.size == 0:
        raise ValueError("cannot score an empty prediction set")
    if len(np.unique(y_true)) < 2:
        raise ValueError("AUROC/AP require both positive and negative labels")
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    return {
        "auroc": float(roc_auc_score(y_true, probs)),
        "ap": float(average_precision_score(y_true, probs)),
    }


def evaluate_split(
    model: LinkPredictor,
    graph: LinkPredictionGraph,
    pos: Tensor,
    neg: Tensor,
    *,
    device: torch.device,
    criterion: nn.Module,
) -> dict[str, float]:
    """Eval-mode loss/AUROC/AP for one labeled split. Uses train message-passing edges."""
    model.eval()
    x = graph.data.x.to(device)
    edge_index = graph.data.edge_index.to(device)
    loader = make_pair_loader(pos, neg, shuffle=False)
    logit_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    total_loss = 0.0
    n_seen = 0
    with torch.inference_mode():
        z = model.encode(x, edge_index)
        for src, dst, labels in loader:
            src = src.to(device)
            dst = dst.to(device)
            labels = labels.to(device)
            logits = model.decode(z, src, dst)
            loss = criterion(logits, labels)
            batch_n = int(labels.numel())
            total_loss += float(loss.item()) * batch_n
            n_seen += batch_n
            logit_chunks.append(logits.detach().cpu().numpy())
            label_chunks.append(labels.detach().cpu().numpy())
    if n_seen == 0:
        raise RuntimeError("evaluation split is empty")
    y_true = np.concatenate(label_chunks)
    logits_np = np.concatenate(logit_chunks)
    metrics = compute_binary_metrics(y_true, logits_np)
    metrics["loss"] = total_loss / n_seen
    return metrics


def train_one_epoch(
    model: LinkPredictor,
    graph: LinkPredictionGraph,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """One Adam epoch; encode per batch so dropout applies correctly."""
    model.train()
    x = graph.data.x.to(device)
    edge_index = graph.data.edge_index.to(device)
    loader = make_pair_loader(graph.train_pos, graph.train_neg, shuffle=True)
    total_loss = 0.0
    n_seen = 0
    for src, dst, labels in loader:
        src = src.to(device)
        dst = dst.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, edge_index, src, dst)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        clip_grad_norm_(model.parameters(), _GRAD_CLIP)
        optimizer.step()
        batch_n = int(labels.numel())
        total_loss += float(loss.item()) * batch_n
        n_seen += batch_n
    return total_loss / max(n_seen, 1)


def hippo_actin_candidates(
    compartments: list[str],
    species_ids: list[int],
    known: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Undirected Hippo×actin pairs within a species, excluding known STRING edges."""
    if not (len(compartments) == len(species_ids)):
        raise ValueError("compartments and species_ids must be aligned")
    n_nodes = len(compartments)
    pairs: set[tuple[int, int]] = set()
    for species in sorted(set(species_ids)):
        hippo_idx = [
            i
            for i in range(n_nodes)
            if species_ids[i] == species and compartments[i] in HIPPO_LIKE
        ]
        actin_idx = [
            i
            for i in range(n_nodes)
            if species_ids[i] == species and compartments[i] in ACTIN_LIKE
        ]
        for left in hippo_idx:
            for right in actin_idx:
                if left == right:
                    continue
                key = (left, right) if left < right else (right, left)
                if key in known:
                    continue
                pairs.add(key)
    return sorted(pairs)


def score_pairs(
    model: LinkPredictor,
    z: Tensor,
    pairs: list[tuple[int, int]],
) -> Tensor:
    """Average both directed logits so undirected scores are order-invariant."""
    if not pairs:
        return torch.zeros(0, dtype=torch.float32, device=z.device)
    src = torch.tensor([a for a, _ in pairs], dtype=torch.long, device=z.device)
    dst = torch.tensor([b for _, b in pairs], dtype=torch.long, device=z.device)
    with torch.inference_mode():
        forward = model.decode(z, src, dst)
        reverse = model.decode(z, dst, src)
        logits = 0.5 * (forward + reverse)
        return torch.sigmoid(logits)


def build_hit_list(
    model: LinkPredictor,
    graph: LinkPredictionGraph,
    proteins: pd.DataFrame,
    *,
    device: torch.device,
    top_k: int = HIT_LIST_TOP_K,
) -> pd.DataFrame:
    """Rank novel Hippo–actin pairs using all observed STRING edges for encoding."""
    csv_ids = proteins["string_id"].astype(str).tolist()
    if csv_ids != graph.protein_ids:
        raise ValueError("proteins.csv order does not match the link graph")
    compartments = proteins["compartment"].astype(str).tolist()
    species_ids = proteins["species_id"].astype(int).tolist()
    names = proteins["preferred_name"].astype(str).tolist()
    species_names = proteins["species_name"].astype(str).tolist()

    known = set(graph.known_undirected)
    candidates = hippo_actin_candidates(compartments, species_ids, known)
    logger.info("novel Hippo-actin candidate pairs=%s", len(candidates))

    model.eval()
    x = graph.data.x.to(device)
    full_pos = torch.cat([graph.train_pos, graph.val_pos, graph.test_pos], dim=1)
    full_edges = to_bidirectional(full_pos).to(device)
    with torch.inference_mode():
        z = model.encode(x, full_edges)
        probs = score_pairs(model, z, candidates).cpu().numpy()

    rows: list[dict[str, Any]] = []
    for (src, dst), probability in zip(candidates, probs, strict=True):
        rows.append(
            {
                "species": species_names[src],
                "protein_a": names[src],
                "protein_b": names[dst],
                "string_id_a": graph.protein_ids[src],
                "string_id_b": graph.protein_ids[dst],
                "compartment_a": compartments[src],
                "compartment_b": compartments[dst],
                "probability": float(probability),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        logger.warning("hit list is empty; no unscored Hippo-actin pairs")
        table = pd.DataFrame(
            columns=[
                "rank",
                "species",
                "protein_a",
                "protein_b",
                "string_id_a",
                "string_id_b",
                "compartment_a",
                "compartment_b",
                "probability",
            ]
        )
        return table
    table = table.sort_values("probability", ascending=False, kind="mergesort").reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table.head(top_k).reset_index(drop=True)


def _round_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 6) for key, value in metrics.items()}


def train_model(
    graph: LinkPredictionGraph,
    *,
    device: torch.device,
    epochs: int = TRAIN_EPOCHS,
    patience: int = TRAIN_PATIENCE,
) -> tuple[LinkPredictor, dict[str, Any]]:
    """Train with early stopping on validation AUROC; restore best weights."""
    model = LinkPredictor(in_channels=int(graph.data.x.size(1)), hidden_channels=GNN_HIDDEN_SIZE)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_LR, weight_decay=TRAIN_WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    history: list[dict[str, Any]] = []
    best_val_auroc = -1.0
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    stale = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, graph, optimizer, criterion, device)
        train_metrics = evaluate_split(
            model, graph, graph.train_pos, graph.train_neg, device=device, criterion=criterion
        )
        val_metrics = evaluate_split(
            model, graph, graph.val_pos, graph.val_neg, device=device, criterion=criterion
        )
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_auroc": round(train_metrics["auroc"], 6),
            "train_ap": round(train_metrics["ap"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_auroc": round(val_metrics["auroc"], 6),
            "val_ap": round(val_metrics["ap"], 6),
        }
        history.append(record)
        logger.info(
            "epoch=%s train_loss=%.4f train_auroc=%.4f train_ap=%.4f "
            "val_loss=%.4f val_auroc=%.4f val_ap=%.4f",
            epoch,
            train_loss,
            train_metrics["auroc"],
            train_metrics["ap"],
            val_metrics["loss"],
            val_metrics["auroc"],
            val_metrics["ap"],
        )
        if val_metrics["auroc"] > best_val_auroc + 1e-6:
            best_val_auroc = val_metrics["auroc"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                logger.info("early stop at epoch=%s best_epoch=%s val_auroc=%.4f", epoch, best_epoch, best_val_auroc)
                break

    if best_state is None:
        raise RuntimeError("training produced no valid checkpoint")
    model.load_state_dict(best_state)
    test_metrics = evaluate_split(
        model, graph, graph.test_pos, graph.test_neg, device=device, criterion=criterion
    )
    summary = {
        "best_epoch": best_epoch,
        "epochs_run": history[-1]["epoch"],
        "best_val_auroc": round(best_val_auroc, 6),
        "test": _round_metrics(test_metrics),
        "history": history,
        "hyperparameters": {
            "lr": TRAIN_LR,
            "weight_decay": TRAIN_WEIGHT_DECAY,
            "epochs": epochs,
            "patience": patience,
            "device": str(device),
        },
    }
    logger.info("test_loss=%.4f test_auroc=%.4f test_ap=%.4f", test_metrics["loss"], test_metrics["auroc"], test_metrics["ap"])
    return model, summary


def run() -> dict[str, Any]:
    """Train on the Phase 1/2 graph and write checkpoint, metrics, and hit list."""
    seed_everything()
    device = get_device()
    graph = load_link_graph()
    proteins = pd.read_csv(PROTEINS_CSV)
    model, summary = train_model(graph, device=device)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "in_channels": int(graph.data.x.size(1)),
            "hidden_channels": GNN_HIDDEN_SIZE,
            "best_epoch": summary["best_epoch"],
            "best_val_auroc": summary["best_val_auroc"],
        },
        GNN_CHECKPOINT_PT,
    )
    hits = build_hit_list(model, graph, proteins, device=device)
    hits.to_csv(HIT_LIST_CSV, index=False)
    summary["hit_list"] = {
        "path": str(HIT_LIST_CSV),
        "n_rows": int(len(hits)),
        "top_probability": float(hits["probability"].iloc[0]) if len(hits) else None,
    }
    TRAIN_METRICS_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s", GNN_CHECKPOINT_PT)
    logger.info("wrote %s", TRAIN_METRICS_JSON)
    logger.info("wrote %s (%s rows)", HIT_LIST_CSV, len(hits))
    if len(hits):
        preview = hits.head(10)[["rank", "species", "protein_a", "protein_b", "probability"]]
        logger.info("top hits:\n%s", preview.to_string(index=False))
    return summary


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def main() -> None:
    configure_logging()
    summary = run()
    test = summary["test"]
    logger.info(
        "training complete best_epoch=%s test_auroc=%.4f test_ap=%.4f",
        summary["best_epoch"],
        test["auroc"],
        test["ap"],
    )


if __name__ == "__main__":
    main()
