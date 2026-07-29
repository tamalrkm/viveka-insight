"""Central configuration.

Every script reads from here so paths/models are not duplicated. Override
any setting via environment variables prefixed `VIVEKA_*` (see `_env`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Repo root = parent of the package directory
_PKG = Path(__file__).resolve().parent
ROOT = _PKG.parent


def _env(key: str, default):
    """Read VIVEKA_<key> from env, falling back to default. Casts to type(default)."""
    v = os.environ.get(f"VIVEKA_{key}")
    if v is None:
        return default
    if isinstance(default, bool):
        return v.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(v)
    if isinstance(default, float):
        return float(v)
    return v


@dataclass
class Paths:
    """Filesystem layout.

    Every field is overridable via `VIVEKA_*` so an alternate corpus can be
    built into an alternate index directory without disturbing the main one
    — this is what `scripts/build_all.py --sample` uses.
    """
    root: Path = ROOT
    data_dir: Path = field(
        default_factory=lambda: Path(_env("DATA_DIR", str(ROOT / "data"))))
    index_dir: Path = field(
        default_factory=lambda: Path(_env("INDEX_DIR", str(ROOT / "index_data"))))
    en_html: Path = field(default_factory=lambda: Path(
        _env("EN_HTML", str(ROOT / "data" / "vivekananda_complete_works.html"))))
    bn_html: Path = field(default_factory=lambda: Path(
        _env("BN_HTML", str(ROOT / "data" / "all.html"))))

    @property
    def db(self) -> Path:
        return self.index_dir / "meta.sqlite"

    def faiss(self, kind: str, lang: Optional[str] = None) -> Path:
        # kind in {sentence, paragraph, chapter, concept}
        if lang is None:
            return self.index_dir / f"{kind}.faiss"
        return self.index_dir / f"{kind}_{lang}.faiss"

    def gnn_emb(self, kind: str) -> Path:
        return self.index_dir / f"gnn_{kind}.npy"


@dataclass
class Models:
    # Multilingual dense+sparse embedder. 1024-dim, supports 100+ languages
    # incl. Bengali. Handles up to 8192 tokens — long enough for chapter-level.
    # Overridable (VIVEKA_EMBEDDER) for constrained environments; if you swap
    # it, set VIVEKA_EMBED_DIM to match the new model's dimensionality.
    embedder: str = field(
        default_factory=lambda: _env("EMBEDDER", "BAAI/bge-m3"))

    # Cross-encoder reranker, also multilingual. Takes (query, passage) -> score.
    reranker: str = field(
        default_factory=lambda: _env("RERANKER", "BAAI/bge-reranker-v2-m3"))

    # Local LLM for concept extraction. Qwen2.5 has strong Bengali support and
    # reliably emits clean JSON. 14B fits comfortably in fp16 on an A100 80GB
    # alongside the embedder + reranker.
    llm: str = "Qwen/Qwen2.5-14B-Instruct"

    # Smaller fallback if vLLM isn't available; runs via transformers.
    llm_small: str = "Qwen/Qwen2.5-7B-Instruct"


@dataclass
class Embedding:
    # BGE-M3 dense dim. Must match Models.embedder — override both together.
    dim: int = field(default_factory=lambda: _env("EMBED_DIM", 1024))
    batch_size_sentence: int = 256   # short inputs, big batch
    batch_size_paragraph: int = 64
    batch_size_chapter: int = 8      # long inputs
    max_tokens_chapter: int = 4096   # cap chapter input
    fp16: bool = True


@dataclass
class LLM:
    # Generation params for concept extraction
    max_input_tokens: int = 2000     # truncate paragraphs longer than this
    max_output_tokens: int = 600
    temperature: float = 0.1         # near-deterministic; we want consistent JSON
    top_p: float = 0.9
    # Concurrency: how many paragraphs to send to vLLM in one engine.generate()
    batch: int = 64
    # Backend: "vllm" | "transformers" | "openai"
    backend: str = field(default_factory=lambda: _env("LLM_BACKEND", "vllm"))
    # For OpenAI-compatible servers
    openai_base_url: Optional[str] = field(
        default_factory=lambda: _env("OPENAI_BASE_URL", None)
    )
    openai_model: Optional[str] = field(
        default_factory=lambda: _env("OPENAI_MODEL", None)
    )


@dataclass
class ConceptLinking:
    # Cosine similarity above which two concept names are considered the same.
    # 0.92 is conservative — only true synonyms ("compassion" / "kindness" stay
    # separate, but "self-realization" / "self realisation" merge).
    merge_threshold: float = 0.92

    # Build co-occurrence edge between concepts that appear in the same paragraph
    # at least this many times across the corpus.
    cooccur_min_count: int = 2

    # Concept-similarity edges: keep top-K neighbors per concept above threshold.
    sim_top_k: int = 12
    sim_threshold: float = 0.78


@dataclass
class Search:
    # Per-granularity, per-language, how many candidates to pull from FAISS
    # before merging.
    k_sentence: int = 30
    k_paragraph: int = 20
    k_chapter: int = 10

    # Concept-mediated retrieval
    k_concepts: int = 12              # top concepts to anchor on
    k_concept_hops: int = 1           # 1-hop neighbors of those concepts
    k_per_concept: int = 5            # paragraphs to pull per concept

    # After fusing all candidates, send this many to the cross-encoder.
    rerank_pool: int = 60
    # Final results to display, per language.
    top_k: int = 8

    # Score-fusion weights for the merge step (before reranking).
    w_sentence: float = 1.0
    w_paragraph: float = 1.0
    w_chapter: float = 0.7
    w_concept: float = 1.2            # boost concept-mediated hits

    # Source HTML hosts — used by the webapp to deep-link a search hit to the
    # chapter in the original document via the chapter's `id` attribute.
    # Both files have an `id="<chapter_id_html>"` on every chapter container,
    # so `<base>#<chapter_id_html>` scrolls straight to the chapter top.
    source_url_en: str = "https://cs.rkmvu.ac.in/~tamal/CWSV/vivekananda_complete_works.html"
    source_url_bn: str = "https://cs.rkmvu.ac.in/~tamal/vani_rachana/all.html"


@dataclass
class QA:
    """The "Ask Vivekananda" answer pipeline (webapp tab)."""

    # Retrieval
    n_sources: int = field(default_factory=lambda: _env("QA_N_SOURCES", 10))
    per_query_k: int = field(default_factory=lambda: _env("QA_PER_QUERY_K", 8))
    n_bridge_concepts: int = field(
        default_factory=lambda: _env("QA_N_BRIDGE_CONCEPTS", 15)
    )

    # Prompt budget. Chars ≈ tokens * 2.5 for English; Bengali is denser per
    # token, so these are conservative. The whole answer prompt must stay
    # under vLLM's max_model_len (8192) if that backend is ever used.
    max_context_chars: int = field(
        default_factory=lambda: _env("QA_MAX_CONTEXT_CHARS", 1200)
    )   # per source passage
    max_prompt_chars: int = field(
        default_factory=lambda: _env("QA_MAX_PROMPT_CHARS", 15000)
    )   # whole answer prompt
    max_input_tokens: int = field(
        default_factory=lambda: _env("QA_MAX_INPUT_TOKENS", 7400)
    )   # tokenizer-level backstop (HF backend truncates left)

    # Generation
    plan_max_tokens: int = field(default_factory=lambda: _env("QA_PLAN_MAX_TOKENS", 300))
    answer_max_tokens: int = field(
        default_factory=lambda: _env("QA_ANSWER_MAX_TOKENS", 700)
    )
    temperature: float = field(default_factory=lambda: _env("QA_TEMPERATURE", 0.3))


@dataclass
class GNN:
    enabled: bool = field(default_factory=lambda: _env("GNN_ENABLED", False))
    hidden_dim: int = 256
    out_dim: int = 256
    n_layers: int = 2
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-5
    walk_length: int = 4
    n_walks_per_node: int = 8


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    models: Models = field(default_factory=Models)
    embedding: Embedding = field(default_factory=Embedding)
    llm: LLM = field(default_factory=LLM)
    concept_linking: ConceptLinking = field(default_factory=ConceptLinking)
    search: Search = field(default_factory=Search)
    qa: QA = field(default_factory=QA)
    gnn: GNN = field(default_factory=GNN)

    device: str = field(default_factory=lambda: _env("DEVICE", "cuda"))


# A single global config; scripts/library import this.
CFG = Config()
