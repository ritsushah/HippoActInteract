"""Unit tests for ESM-2 pooling and FASTA alignment (no Hub download)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.embedder import (
    ProteinEmbedder,
    load_ordered_sequences,
    mean_pool,
    parse_fasta,
    residue_mask,
    save_embeddings,
    truncate_sequence,
)


class _FakeTokenizer:
    pad_token_id = 0
    cls_token_id = 1
    eos_token_id = 2

    def __call__(
        self,
        sequences: list[str],
        return_tensors: str,
        padding: bool,
        truncation: bool,
        max_length: int,
    ) -> dict[str, torch.Tensor]:
        rows: list[list[int]] = []
        for seq in sequences:
            body = [3] * min(len(seq), max_length - 2)
            ids = [self.cls_token_id, *body, self.eos_token_id]
            rows.append(ids)
        width = max(len(row) for row in rows)
        input_ids = torch.zeros(len(rows), width, dtype=torch.long)
        attention_mask = torch.zeros(len(rows), width, dtype=torch.long)
        for i, row in enumerate(rows):
            input_ids[i, : len(row)] = torch.tensor(row)
            attention_mask[i, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _FakeEsm(nn.Module):
    def __init__(self, hidden: int = 4) -> None:
        super().__init__()
        self.hidden = hidden
        self.config = SimpleNamespace(hidden_size=hidden)
        self.table = nn.Embedding(8, hidden)
        with torch.no_grad():
            self.table.weight.zero_()
            self.table.weight[3] = torch.tensor([1.0, 2.0, 3.0, 4.0][:hidden])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        return SimpleNamespace(last_hidden_state=self.table(input_ids))


def test_parse_fasta(tmp_path: Path) -> None:
    path = tmp_path / "tiny.fasta"
    path.write_text(">idA nameA\nMKT\nAAA\n>idB\nC\n", encoding="utf-8")
    records = parse_fasta(path)
    assert records == {"idA": "MKTAAA", "idB": "C"}


def test_truncate_sequence_logs_long_proteins() -> None:
    clipped, truncated = truncate_sequence("P1", "A" * 10, max_residues=4)
    assert truncated is True
    assert clipped == "AAAA"
    same, not_truncated = truncate_sequence("P2", "ACDE", max_residues=4)
    assert not_truncated is False
    assert same == "ACDE"


def test_mean_pool_excludes_special_and_pad() -> None:
    hidden = torch.tensor(
        [
            [
                [9.0, 9.0],
                [1.0, 0.0],
                [3.0, 2.0],
                [8.0, 8.0],
                [0.0, 0.0],
            ]
        ]
    )
    input_ids = torch.tensor([[1, 3, 3, 2, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 0]])
    mask = residue_mask(
        input_ids,
        attention_mask,
        pad_token_id=0,
        cls_token_id=1,
        eos_token_id=2,
    )
    pooled = mean_pool(hidden, mask)
    assert torch.allclose(pooled, torch.tensor([[2.0, 1.0]]))


def test_load_ordered_sequences_follows_csv(tmp_path: Path) -> None:
    fasta = tmp_path / "proteins.fasta"
    fasta.write_text(">b\nCC\n>a\nAA\n>c\nDDDD\n", encoding="utf-8")
    csv_path = tmp_path / "proteins.csv"
    csv_path.write_text("string_id,preferred_name\na,A\nb,B\nc,C\n", encoding="utf-8")
    ids, seqs, truncated = load_ordered_sequences(csv_path, fasta, max_residues=3)
    assert ids == ["a", "b", "c"]
    assert seqs == ["AA", "CC", "DDD"]
    assert truncated == ["c"]


def test_embedder_fake_model_shape_and_finite() -> None:
    embedder = ProteinEmbedder(
        model=_FakeEsm(hidden=4),
        tokenizer=_FakeTokenizer(),
        device=torch.device("cpu"),
        batch_size=2,
        max_length=16,
    )
    out = embedder.embed_sequences(["A", "AAA"])
    assert out.shape == (2, 4)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert torch.allclose(out[0], torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_save_embeddings_writes_pt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src import embedder as embedder_mod

    monkeypatch.setattr(embedder_mod, "NODE_EMBEDDINGS_PT", tmp_path / "node_embeddings.pt")
    monkeypatch.setattr(embedder_mod, "EMBED_SUMMARY_JSON", tmp_path / "embed_summary.json")
    monkeypatch.setattr(embedder_mod, "DATA_PROCESSED", tmp_path)

    ids = ["p1", "p2"]
    tensor = torch.ones(2, 480)
    summary = save_embeddings(
        ids,
        tensor,
        truncated=["p2"],
        device=torch.device("cpu"),
        model_name="facebook/esm2_t12_35M_UR50D",
        peak_rss_mb=12.3,
    )
    payload = torch.load(tmp_path / "node_embeddings.pt", map_location="cpu", weights_only=True)
    assert payload["protein_ids"] == ids
    assert tuple(payload["embeddings"].shape) == (2, 480)
    assert summary["shape"] == [2, 480]
    assert summary["truncated_count"] == 1
