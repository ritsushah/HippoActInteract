"""ESM-2 residue-mean embeddings for STRING protein nodes."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import psutil
import torch
from torch import nn

from src.config import (
    DATA_PROCESSED,
    EMBED_SUMMARY_JSON,
    ESM_BATCH_SIZE,
    ESM_HIDDEN_SIZE,
    ESM_MAX_LENGTH,
    ESM_MAX_RESIDUES,
    ESM_MODEL_NAME,
    NODE_EMBEDDINGS_PT,
    PROTEINS_CSV,
    PROTEINS_FASTA,
)
from src.device import get_device

logger = logging.getLogger(__name__)


class TokenizerProtocol(Protocol):
    pad_token_id: int | None
    cls_token_id: int | None
    eos_token_id: int | None

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class FastaRecord:
    protein_id: str
    sequence: str


def rss_mb() -> float:
    """Current process resident set size in mebibytes."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def parse_fasta(path: Path) -> dict[str, str]:
    """Parse FASTA into ``{first_header_token: sequence}``."""
    if not path.is_file():
        raise FileNotFoundError(f"FASTA not found: {path}")
    records: dict[str, str] = {}
    current_id: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records[current_id] = "".join(chunks).upper()
                header = line[1:].strip()
                current_id = header.split()[0] if header else ""
                if not current_id:
                    raise ValueError(f"FASTA header missing protein id: {line!r}")
                chunks = []
            else:
                if current_id is None:
                    raise ValueError("FASTA sequence line appeared before a header")
                chunks.append(line.replace(" ", ""))
        if current_id is not None:
            records[current_id] = "".join(chunks).upper()
    if not records:
        raise ValueError(f"no FASTA records in {path}")
    logger.info("parsed FASTA %s n=%s", path, len(records))
    return records


def truncate_sequence(protein_id: str, sequence: str, max_residues: int) -> tuple[str, bool]:
    """Clip amino acids to the ESM-2 residue budget (CLS/EOS use the remaining 2 slots)."""
    if len(sequence) <= max_residues:
        return sequence, False
    logger.warning(
        "truncating %s from %s to %s residues",
        protein_id,
        len(sequence),
        max_residues,
    )
    return sequence[:max_residues], True


def residue_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    pad_token_id: int | None,
    cls_token_id: int | None,
    eos_token_id: int | None,
) -> torch.Tensor:
    """True on amino-acid tokens only (drop pad, CLS, EOS)."""
    mask = attention_mask.bool()
    for token_id in (pad_token_id, cls_token_id, eos_token_id):
        if token_id is not None:
            mask = mask & (input_ids != token_id)
    return mask


def mean_pool(
    hidden: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool ``hidden`` [B, L, H] over residue positions in ``mask`` [B, L]."""
    weights = mask.unsqueeze(-1).to(dtype=hidden.dtype)
    summed = (hidden * weights).sum(dim=1)
    counts = weights.sum(dim=1).clamp(min=1.0)
    return summed / counts


def load_ordered_sequences(
    proteins_csv: Path = PROTEINS_CSV,
    fasta_path: Path = PROTEINS_FASTA,
    max_residues: int = ESM_MAX_RESIDUES,
) -> tuple[list[str], list[str], list[str]]:
    """Return protein ids, (possibly truncated) sequences, and truncated ids."""
    table = pd.read_csv(proteins_csv)
    if "string_id" not in table.columns:
        raise ValueError(f"{proteins_csv} missing string_id column")
    fasta = parse_fasta(fasta_path)
    protein_ids: list[str] = table["string_id"].astype(str).tolist()
    sequences: list[str] = []
    truncated: list[str] = []
    missing = [pid for pid in protein_ids if pid not in fasta]
    if missing:
        preview = ", ".join(missing[:8])
        raise KeyError(f"{len(missing)} proteins missing from FASTA (e.g. {preview})")
    for protein_id in protein_ids:
        clipped, was_truncated = truncate_sequence(protein_id, fasta[protein_id], max_residues)
        if was_truncated:
            truncated.append(protein_id)
        if not clipped:
            raise ValueError(f"empty sequence for {protein_id}")
        sequences.append(clipped)
    logger.info("ordered sequences n=%s truncated=%s", len(sequences), len(truncated))
    return protein_ids, sequences, truncated


class ProteinEmbedder:
    """Run ESM-2 and mean-pool residue states into one vector per protein."""

    def __init__(
        self,
        *,
        model: nn.Module | None = None,
        tokenizer: TokenizerProtocol | None = None,
        device: torch.device | None = None,
        model_name: str = ESM_MODEL_NAME,
        batch_size: int = ESM_BATCH_SIZE,
        max_length: int = ESM_MAX_LENGTH,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.device = device or get_device()
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        if (model is None) ^ (tokenizer is None):
            raise ValueError("model and tokenizer must both be provided or both omitted")
        if model is None or tokenizer is None:
            model, tokenizer = _load_esm(model_name, self.device)
        else:
            model = model.to(self.device)
            model.eval()
        self.model = model
        self.tokenizer = tokenizer

    def embed_sequences(self, sequences: Sequence[str]) -> torch.Tensor:
        """Return float32 embeddings ``[N, H]`` on CPU."""
        if not sequences:
            raise ValueError("no sequences to embed")
        chunks: list[torch.Tensor] = []
        total = len(sequences)
        for start in range(0, total, self.batch_size):
            batch = list(sequences[start : start + self.batch_size])
            try:
                chunks.append(self._embed_batch(batch))
            except RuntimeError as exc:
                message = str(exc).lower()
                if "out of memory" in message or "mps backend" in message:
                    logger.exception(
                        "OOM at proteins %s-%s on %s; retry with ESM_BATCH_SIZE=1",
                        start,
                        start + len(batch),
                        self.device,
                    )
                raise
            done = min(start + len(batch), total)
            if done == total or done % 25 == 0 or start == 0:
                logger.info(
                    "embedded %s/%s rss=%.1fMB device=%s",
                    done,
                    total,
                    rss_mb(),
                    self.device,
                )
        return torch.cat(chunks, dim=0)

    def _embed_batch(self, sequences: list[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state
            mask = residue_mask(
                input_ids,
                attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                cls_token_id=self.tokenizer.cls_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            pooled = mean_pool(hidden, mask)
        if not torch.isfinite(pooled).all():
            raise RuntimeError("ESM-2 produced non-finite embeddings")
        return pooled.detach().cpu().to(dtype=torch.float32)


def _load_esm(model_name: str, device: torch.device) -> tuple[nn.Module, TokenizerProtocol]:
    """Download/load ESM-2. Isolated so unit tests can skip the Hub."""
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required to load ESM-2") from exc

    logger.info("loading tokenizer/model %s onto %s rss=%.1fMB", model_name, device, rss_mb())
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            model = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        except TypeError:
            model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
    except OSError as exc:
        logger.exception("failed to download or load %s", model_name)
        raise RuntimeError(f"could not load ESM-2 model {model_name}") from exc
    model.to(device)
    model.eval()
    hidden = int(getattr(model.config, "hidden_size", ESM_HIDDEN_SIZE))
    logger.info("ESM-2 ready hidden_size=%s rss=%.1fMB", hidden, rss_mb())
    return model, tokenizer


def save_embeddings(
    protein_ids: list[str],
    embeddings: torch.Tensor,
    *,
    truncated: list[str],
    device: torch.device,
    model_name: str,
    peak_rss_mb: float,
) -> dict[str, Any]:
    """Write ``.pt`` plus a JSON summary for Phase 2 review."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    if embeddings.ndim != 2:
        raise ValueError(f"expected [N, H] embeddings, got shape {tuple(embeddings.shape)}")
    if embeddings.shape[0] != len(protein_ids):
        raise ValueError("protein_ids and embeddings length mismatch")
    payload = {
        "protein_ids": protein_ids,
        "embeddings": embeddings.contiguous(),
        "model_name": model_name,
        "pooling": "mean_residue_exclude_special",
    }
    torch.save(payload, NODE_EMBEDDINGS_PT)
    file_bytes = NODE_EMBEDDINGS_PT.stat().st_size
    summary = {
        "model_name": model_name,
        "n_proteins": len(protein_ids),
        "embedding_dim": int(embeddings.shape[1]),
        "shape": [int(embeddings.shape[0]), int(embeddings.shape[1])],
        "dtype": str(embeddings.dtype),
        "device_used": str(device),
        "truncated_count": len(truncated),
        "truncated_ids": truncated,
        "peak_rss_mb": round(peak_rss_mb, 1),
        "files": {
            "node_embeddings_pt": {
                "path": str(NODE_EMBEDDINGS_PT),
                "bytes": file_bytes,
                "mb": round(file_bytes / (1024 * 1024), 4),
            }
        },
    }
    EMBED_SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s", NODE_EMBEDDINGS_PT)
    logger.info("embed summary: %s", json.dumps(summary, indent=2))
    return summary


def run() -> dict[str, Any]:
    """Embed all ingested proteins and persist tensors."""
    peak = rss_mb()
    protein_ids, sequences, truncated = load_ordered_sequences()
    embedder = ProteinEmbedder()
    peak = max(peak, rss_mb())
    embeddings = embedder.embed_sequences(sequences)
    peak = max(peak, rss_mb())
    expected_hidden = int(getattr(embedder.model.config, "hidden_size", ESM_HIDDEN_SIZE))
    if embeddings.shape != (len(protein_ids), expected_hidden):
        raise RuntimeError(
            f"embedding shape {tuple(embeddings.shape)} != "
            f"({len(protein_ids)}, {expected_hidden})"
        )
    return save_embeddings(
        protein_ids,
        embeddings,
        truncated=truncated,
        device=embedder.device,
        model_name=embedder.model_name,
        peak_rss_mb=peak,
    )


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def main() -> None:
    configure_logging()
    summary = run()
    logger.info(
        "embeddings ready shape=%s peak_rss=%.1fMB",
        summary["shape"],
        summary["peak_rss_mb"],
    )


if __name__ == "__main__":
    main()
