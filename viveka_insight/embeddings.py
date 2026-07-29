"""BGE-M3 embedding wrapper + FAISS read/write helpers.

BGE-M3 was chosen over LaBSE because:
  * supports up to 8192 tokens (LaBSE caps at 512) — the same model can embed
    sentences AND chapter-length passages without truncation hacks,
  * higher retrieval quality on multilingual MTEB benchmarks (Bengali included),
  * dense + sparse + ColBERT-style multi-vector heads are all in one model,
    so a future hybrid (BM25-ish + dense) doesn't require a second tokenizer.

For this project we use only the dense head (1024-d). The sparse and ColBERT
heads are left as future work; they'd be a drop-in for re-ranking.
"""
from __future__ import annotations

import gc
import os
import warnings
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np


# ──────────────────────────────────────────────────────────────────────────
# Embedder
# ──────────────────────────────────────────────────────────────────────────

class Embedder:
    """Lazy-loaded BGE-M3.

    `model` is None until .load() or first .encode(). This lets a caller
    construct one in a Streamlit cache resource without paying the load cost
    if no encoding is actually needed.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        fp16: bool = True,
    ) -> None:
        self.model_name = model_name
        self._device = device
        self.fp16 = fp16
        self.model = None  # type: ignore[assignment]
        self._dim: Optional[int] = None

    @property
    def device(self) -> str:
        if self._device is not None:
            return self._device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def load(self) -> None:
        if self.model is not None:
            return
        # We use FlagEmbedding's BGEM3FlagModel because it exposes the
        # multi-functional API (dense / sparse / colbert) cleanly. Fall back
        # to sentence-transformers if FlagEmbedding isn't installed — it
        # gives the dense head only, which is all we use here anyway.
        try:
            from FlagEmbedding import BGEM3FlagModel
            print(f"[embed] loading BGE-M3 via FlagEmbedding "
                  f"(device={self.device}, fp16={self.fp16})")
            self.model = BGEM3FlagModel(
                self.model_name,
                use_fp16=self.fp16 and self.device.startswith("cuda"),
                device=self.device,
            )
            self._backend = "flagembedding"
            self._dim = 1024  # BGE-M3 dense
        except ImportError:
            from sentence_transformers import SentenceTransformer
            print(f"[embed] FlagEmbedding not installed; falling back to "
                  f"sentence-transformers (device={self.device})")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            if self.fp16 and self.device.startswith("cuda"):
                self.model = self.model.half()
            self._backend = "sentencetransformers"
            self._dim = self.model.get_sentence_embedding_dimension()

    def unload(self) -> None:
        """Release GPU memory. Useful between heavy steps when we want to load
        a different big model (e.g. unload embedder before loading the LLM)."""
        self.model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @property
    def dim(self) -> int:
        if self._dim is None:
            self.load()
        return int(self._dim)  # type: ignore[arg-type]

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        show_progress: bool = True,
        max_length: int = 512,
    ) -> np.ndarray:
        """Returns a (N, dim) float32 array of L2-normalized vectors.

        We always L2-normalize so that downstream FAISS inner-product search
        equals cosine similarity. `max_length` controls input truncation —
        keep small for sentences (faster) and large for chapter-level inputs.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        self.load()

        if self._backend == "flagembedding":
            out = self.model.encode(
                list(texts),
                batch_size=batch_size,
                max_length=max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            vecs = np.asarray(out["dense_vecs"], dtype=np.float32)
        else:  # sentence-transformers
            vecs = self.model.encode(
                list(texts),
                batch_size=batch_size,
                normalize_embeddings=False,  # we normalize ourselves below
                convert_to_numpy=True,
                show_progress_bar=show_progress,
            ).astype(np.float32)

        # Always L2-normalize (FlagEmbedding does this internally; doing it
        # again is a no-op modulo float noise).
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        vecs = vecs / norms
        return vecs


# ──────────────────────────────────────────────────────────────────────────
# FAISS helpers
# ──────────────────────────────────────────────────────────────────────────

def new_flat_index(dim: int):
    """Flat inner-product index. Combined with normalized vectors, IP=cosine.

    For our scale (~150K sentences max), brute-force flat search is <50 ms on
    GPU and <300 ms on CPU per query — no need for IVF/HNSW yet. If the corpus
    grows 10× later, swap in `IndexHNSWFlat(dim, 32)` and rebuild."""
    import faiss
    return faiss.IndexFlatIP(dim)


def write_index(index, path: str | Path) -> None:
    import faiss
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def read_index(path: str | Path):
    import faiss
    return faiss.read_index(str(path))


def add_to_index(index, vecs: np.ndarray) -> None:
    """Append vectors to an existing index. Assumes the vectors are already
    float32 and the right dimension."""
    if vecs.size == 0:
        return
    index.add(vecs.astype(np.float32, copy=False))


def search_index(index, q_vecs: np.ndarray, k: int):
    """Returns (scores, ids) shaped (Nq, k). FAISS may return -1 in `ids`
    if there are fewer items than `k` in the index — caller should filter."""
    if index.ntotal == 0 or q_vecs.size == 0:
        return np.zeros((q_vecs.shape[0], 0), dtype=np.float32), \
               np.zeros((q_vecs.shape[0], 0), dtype=np.int64)
    k = min(k, index.ntotal)
    return index.search(q_vecs.astype(np.float32, copy=False), k)
