"""Cross-encoder reranker.

After hybrid retrieval pulls a candidate pool (vector hits + concept-mediated
hits, fused by RRF), a cross-encoder re-scores each (query, passage) pair.

Why reranking matters here:
  * Bi-encoder retrieval (FAISS over BGE-M3) compresses each side to a single
    vector; query-passage interaction is implicit (just a dot product).
  * Cross-encoders see both texts together and can reason about specific word
    overlap, syntactic match, and discourse — so they consistently boost
    precision @5 by 5-15 pts on multilingual retrieval benchmarks.
  * BGE-reranker-v2-m3 is the matching cross-encoder for BGE-M3, multilingual,
    and trained on the same data distribution, so out-of-the-box quality is
    high.

Cost: a few hundred (query, passage) pairs per query takes ~200ms on an A100.
We only rerank the top ~60 candidates so this is well under a second.
"""
from __future__ import annotations

import gc
from typing import List, Optional, Sequence, Tuple


class Reranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
        fp16: bool = True,
    ) -> None:
        self.model_name = model_name
        self._device = device
        self.fp16 = fp16
        self.model = None

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
        try:
            from FlagEmbedding import FlagReranker  # type: ignore
            print(f"[rerank] loading FlagReranker {self.model_name} on {self.device}")
            self.model = FlagReranker(
                self.model_name,
                use_fp16=self.fp16 and self.device.startswith("cuda"),
            )
            self._backend = "flagembedding"
        except ImportError:
            from sentence_transformers import CrossEncoder
            print(f"[rerank] FlagEmbedding not available; using sentence-transformers "
                  f"CrossEncoder for {self.model_name}")
            self.model = CrossEncoder(self.model_name, device=self.device)
            self._backend = "sentencetransformers"

    def unload(self) -> None:
        self.model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def score(
        self,
        query: str,
        passages: Sequence[str],
        batch_size: int = 32,
    ) -> List[float]:
        """Return a list of relevance scores aligned to `passages`.

        Scores are unnormalized logits (higher is better). Different backends
        emit slightly different ranges, so don't compare across reranker
        models — but within a single query, ranking is what matters.
        """
        if not passages:
            return []
        self.load()
        pairs = [[query, p] for p in passages]
        if self._backend == "flagembedding":
            scores = self.model.compute_score(pairs, batch_size=batch_size)
            # FlagReranker returns float for one pair, list for many
            if not isinstance(scores, list):
                scores = [float(scores)]
            return [float(s) for s in scores]
        else:
            return [float(s) for s in self.model.predict(pairs, batch_size=batch_size)]
