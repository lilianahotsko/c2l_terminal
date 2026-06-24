"""Frozen Qwen3-Embedding-0.6B encoder.

Reproduces the two embedding recipes used to build the Code2LoRA-GRU v2
dataset, verbatim:

* **Diff embedding** (``build_diff_embeddings_shard.py`` / ``DiffEmbedder``):
  chunk the diff text at 512 tokens (overlap 64), masked-mean each chunk, then
  ``concat(MaxPool, MeanPool)`` over chunks -> 2048-d (no L2 norm).

* **Repo-state embedding** (``build_repo_state_embeddings_shard.py``):
  chunk each .py file at 2048 tokens (overlap 256), masked-mean each chunk,
  average the chunk vectors per file -> 1024-d; then over all files
  ``concat(mean_files, max_files)`` -> 2048-d and L2-normalize.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

# Diff recipe
DIFF_CHUNK_TOKENS = 512
DIFF_OVERLAP = 64
DIFF_MAX_LENGTH = 512
DIFF_MIN_WINDOW = 16

# Repo-state recipe
REPO_CHUNK_TOKENS = 2048
REPO_OVERLAP = 256
REPO_MIN_WINDOW = 8


def _chunk_token_ids(token_ids: List[int], chunk_tokens: int, overlap: int,
                     min_window: int) -> List[List[int]]:
    if chunk_tokens <= 0 or overlap >= chunk_tokens:
        return []
    chunks = []
    step = chunk_tokens - overlap
    n = len(token_ids)
    for start in range(0, n, step):
        end = min(start + chunk_tokens, n)
        window = token_ids[start:end]
        if len(window) < min_window:
            if end >= n:
                break
            continue
        chunks.append(window)
        if end >= n:
            break
    return chunks


class Qwen3Embedder:
    """Lazy-loading wrapper around the frozen Qwen3-Embedding-0.6B encoder."""

    def __init__(self, device: str = "cpu",
                 dtype: Optional[torch.dtype] = None,
                 batch_size: int = 8,
                 model_name: str = EMBED_MODEL_NAME):
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._hidden = None

    def load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        if self.dtype is not None:
            model = model.to(self.dtype)
        self._model = model.to(self.device).eval()
        self._hidden = int(self._model.config.hidden_size)

    @property
    def hidden_size(self) -> int:
        self.load()
        return self._hidden

    @property
    def embed_dim(self) -> int:
        return 2 * self.hidden_size

    @torch.no_grad()
    def _embed_windows(self, texts: List[str], max_length: int) -> torch.Tensor:
        """[N, D] fp32 (cpu) masked-mean chunk embeddings."""
        if not texts:
            return torch.empty((0, self.hidden_size))
        all_vecs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            enc = self._tokenizer(batch, padding=True, truncation=True,
                                  max_length=max_length, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self._model(**enc)
            last = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(last.dtype)
            mean = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            all_vecs.append(mean.detach().float().cpu())
        return torch.cat(all_vecs, dim=0)

    # ---- Diff embedding ----

    @torch.no_grad()
    def embed_diff(self, diff_text: str) -> np.ndarray:
        """One diff -> [2*D] fp32 (MaxPool || MeanPool, no L2 norm)."""
        self.load()
        zero = np.zeros(2 * self._hidden, dtype=np.float32)
        if not diff_text or not diff_text.strip():
            return zero
        ids = self._tokenizer.encode(diff_text, add_special_tokens=False)
        windows = _chunk_token_ids(ids, DIFF_CHUNK_TOKENS, DIFF_OVERLAP, DIFF_MIN_WINDOW)
        if not windows:
            return zero
        texts = [self._tokenizer.decode(w, skip_special_tokens=True) for w in windows]
        chunk_embs = self._embed_windows(texts, DIFF_MAX_LENGTH)  # [K, D]
        mean_pool = chunk_embs.mean(dim=0)
        max_pool = chunk_embs.max(dim=0).values
        return torch.cat([max_pool, mean_pool], dim=-1).numpy().astype(np.float32)

    # ---- Repo-state embedding ----

    @torch.no_grad()
    def _embed_file(self, text: str) -> Optional[np.ndarray]:
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            return None
        windows = _chunk_token_ids(ids, REPO_CHUNK_TOKENS, REPO_OVERLAP, REPO_MIN_WINDOW)
        if not windows:
            return None
        texts = [self._tokenizer.decode(w, skip_special_tokens=True) for w in windows]
        chunk_embs = self._embed_windows(texts, REPO_CHUNK_TOKENS)  # [K, D]
        return chunk_embs.mean(dim=0).numpy().astype(np.float32)

    @torch.no_grad()
    def embed_repo_state(self, file_texts: List[str]) -> np.ndarray:
        """List of file contents -> [2*D] L2-normalized repo-state vector."""
        self.load()
        repo_dim = 2 * self._hidden
        file_vecs = []
        for text in file_texts:
            if not text:
                continue
            vec = self._embed_file(text)
            if vec is not None:
                file_vecs.append(vec)
        if not file_vecs:
            return np.zeros(repo_dim, dtype=np.float32)
        sub = np.stack(file_vecs).astype(np.float32)
        mean_pool = sub.mean(axis=0)
        max_pool = sub.max(axis=0)
        repo_vec = np.concatenate([mean_pool, max_pool], axis=0)
        norm = float(np.linalg.norm(repo_vec)) + 1e-12
        return (repo_vec / norm).astype(np.float32)


__all__ = ["Qwen3Embedder", "EMBED_MODEL_NAME"]
