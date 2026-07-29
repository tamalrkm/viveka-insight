"""End-to-end retrieval.

Pipeline for a single query:

  1. Encode the query with BGE-M3 (single 1024-d vector).
  2. PATH A — direct vector retrieval at every granularity, in both languages:
        sentence_en, sentence_bn, paragraph_en, paragraph_bn,
        chapter_en,  chapter_bn
     Each FAISS hit is mapped back to its parent paragraph (since that's the
     unit we surface in results), with a "level" label kept for diagnostics.
  3. PATH B — concept-mediated retrieval:
        a. nearest concepts to the query (concepts.faiss).
        b. for each, walk 0–1 hops via concept_edges to get the local
           neighborhood (similar + co-occurs).
        c. for each concept in that neighborhood, pull its top paragraphs
           from para_concept, scored by edge weight × concept similarity.
  4. FUSE the two candidate sets via Reciprocal Rank Fusion (per-language).
     RRF is robust to the heterogeneous score scales coming out of A and B.
  5. RERANK the top `rerank_pool` candidates per language with the BGE
     cross-encoder. This is where the system's precision really comes from.
  6. RETURN top-K per language, each annotated with:
        - the matched-best sentence (highlight in UI)
        - which concepts mediated the hit (or [] if pure vector)
        - which paragraph, chapter, volume, char_offset
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from . import db as dbmod
from . import embeddings as emb
from . import graph
from .config import CFG
from .reranker import Reranker


# ──────────────────────────────────────────────────────────────────────────
# Result records
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class SearchHit:
    rank: int
    score: float                         # final reranker score
    paragraph_id: int
    lang: str

    # Provenance (for full citation)
    book_title: str
    volume_title: str
    volume_num: int
    section: Optional[str]
    chapter_title: str
    chapter_id_html: str
    paragraph_idx: int
    char_offset: int

    # Content
    text: str
    summary: Optional[str]
    matched_sentence: str = ""           # highest-scoring sentence in the para
    para_id_html: str = ""               # html id of source <p> ("p-42"); '' if none

    # Why we surfaced this hit
    via_concepts: List[Tuple[int, str, float]] = field(default_factory=list)
    levels: List[str] = field(default_factory=list)  # 'sentence'|'paragraph'|'chapter'|'concept'
    pre_rerank_score: float = 0.0        # fused score before rerank, for debugging

    def location_str(self) -> str:
        bits = [self.volume_title]
        if self.section:
            bits.append(self.section)
        bits.append(self.chapter_title)
        bits.append(f"¶{self.paragraph_idx + 1}")
        return " › ".join(bits)

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────
# Fusion — Reciprocal Rank Fusion
# ──────────────────────────────────────────────────────────────────────────

def _rrf(rank_lists: Sequence[List[int]], k: int = 60) -> Dict[int, float]:
    """Reciprocal Rank Fusion. Inputs are lists of paragraph ids ranked best-first.
    Constant `k` (60 is the canonical default) damps the contribution of items
    deep in any one list — the fusion is dominated by items that rank high in
    multiple lists."""
    scores: Dict[int, float] = defaultdict(float)
    for ranks in rank_lists:
        for rank, pid in enumerate(ranks):
            scores[pid] += 1.0 / (k + rank + 1)
    return scores


# ──────────────────────────────────────────────────────────────────────────
# Searcher
# ──────────────────────────────────────────────────────────────────────────

class Searcher:
    def __init__(
        self,
        index_dir: Optional[Path] = None,
        device: Optional[str] = None,
        load_reranker: bool = True,
    ) -> None:
        index_dir = Path(index_dir or CFG.paths.index_dir)
        self.index_dir = index_dir

        self.conn = dbmod.open_db(index_dir / "meta.sqlite", create=False)

        # FAISS indices: lazy-loaded on first .search() call
        self._faiss: Dict[str, "object"] = {}

        # Embedder (always needed)
        self.embedder = emb.Embedder(
            model_name=CFG.models.embedder,
            device=device or CFG.device,
            fp16=CFG.embedding.fp16,
        )

        # Reranker (heavy to load — opt-in)
        self.reranker: Optional[Reranker] = None
        if load_reranker:
            self.reranker = Reranker(
                model_name=CFG.models.reranker,
                device=device or CFG.device,
                fp16=CFG.embedding.fp16,
            )

        self._sentence_cache: Dict[int, List[Tuple[int, str]]] = {}

    # ──────────────────────────────────────────────────────────────────
    # FAISS / mapping helpers
    # ──────────────────────────────────────────────────────────────────

    def _index(self, kind: str, lang: Optional[str]):
        key = f"{kind}_{lang}" if lang else kind
        if key not in self._faiss:
            path = CFG.paths.faiss(kind, lang)
            if not path.exists():
                return None
            self._faiss[key] = emb.read_index(path)
        return self._faiss[key]

    def _faiss_to_node_ids(
        self, kind: str, lang: Optional[str], faiss_ids: Sequence[int]
    ) -> List[int]:
        """Map FAISS row positions back to node ids (sentence_id / paragraph_id /
        chapter_id / concept_id). Order preserved; missing entries become -1."""
        if not faiss_ids:
            return []
        cur = self.conn.cursor()
        placeholders = ",".join("?" * len(faiss_ids))
        if lang:
            cur.execute(
                f"SELECT faiss_id, node_id FROM embeddings "
                f"WHERE node_kind=? AND lang=? AND faiss_id IN ({placeholders})",
                (kind, lang, *faiss_ids),
            )
        else:
            cur.execute(
                f"SELECT faiss_id, node_id FROM embeddings "
                f"WHERE node_kind=? AND lang IS NULL AND faiss_id IN ({placeholders})",
                (kind, *faiss_ids),
            )
        m = {row["faiss_id"]: row["node_id"] for row in cur.fetchall()}
        return [m.get(int(f), -1) for f in faiss_ids]

    def _sentence_to_paragraph(self, sentence_ids: Sequence[int]) -> Dict[int, int]:
        if not sentence_ids:
            return {}
        placeholders = ",".join("?" * len(sentence_ids))
        cur = self.conn.cursor()
        cur.execute(
            f"SELECT id, paragraph_id FROM sentences WHERE id IN ({placeholders})",
            tuple(sentence_ids),
        )
        return {row["id"]: row["paragraph_id"] for row in cur.fetchall()}

    def _chapter_to_paragraphs(self, chapter_ids: Sequence[int],
                                top_per_chapter: int = 2) -> Dict[int, List[int]]:
        """For chapter-level hits, return the first/most-meaningful paragraphs
        (we surface paragraphs, not chapters)."""
        if not chapter_ids:
            return {}
        cur = self.conn.cursor()
        out: Dict[int, List[int]] = {}
        for cid in chapter_ids:
            cur.execute(
                "SELECT id FROM paragraphs WHERE chapter_id=? ORDER BY paragraph_idx LIMIT ?",
                (cid, top_per_chapter),
            )
            out[cid] = [int(r[0]) for r in cur.fetchall()]
        return out

    def _sentences_of_paragraph(self, paragraph_id: int) -> List[Tuple[int, str]]:
        if paragraph_id in self._sentence_cache:
            return self._sentence_cache[paragraph_id]
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id, text FROM sentences WHERE paragraph_id=? ORDER BY sentence_idx",
            (paragraph_id,),
        )
        rows = [(int(r[0]), r[1]) for r in cur.fetchall()]
        self._sentence_cache[paragraph_id] = rows
        return rows

    # ──────────────────────────────────────────────────────────────────
    # Path A — direct vector hits, mapped back to paragraphs
    # ──────────────────────────────────────────────────────────────────

    def _vector_paragraphs(
        self, q_vec: np.ndarray, lang: str
    ) -> Tuple[List[Tuple[int, float, str]], Dict[int, int]]:
        """Returns:
          - list of (paragraph_id, raw_score, level) deduped by paragraph,
            keeping the highest-scoring level.
          - sentence_id_for_paragraph: which sentence id was the source of
            the strongest hit per paragraph (for matched-sentence highlight).
        """
        cfg = CFG.search
        candidates: List[Tuple[int, float, str]] = []
        best_sentence: Dict[int, int] = {}     # paragraph_id -> sentence_id

        # SENTENCE-level
        idx = self._index("sentence", lang)
        if idx is not None:
            scores, fids = emb.search_index(idx, q_vec, cfg.k_sentence)
            scores, fids = scores[0], fids[0]
            valid = [int(f) for f in fids if f >= 0]
            sids = self._faiss_to_node_ids("sentence", lang, valid)
            sids = [s for s in sids if s > 0]
            sid_to_pid = self._sentence_to_paragraph(sids)
            for fid, sid, sc in zip(valid, sids, scores):
                pid = sid_to_pid.get(sid)
                if not pid:
                    continue
                candidates.append((pid, float(sc), "sentence"))
                # Keep best-scoring sentence per paragraph
                if pid not in best_sentence:
                    best_sentence[pid] = sid

        # PARAGRAPH-level
        idx = self._index("paragraph", lang)
        if idx is not None:
            scores, fids = emb.search_index(idx, q_vec, cfg.k_paragraph)
            scores, fids = scores[0], fids[0]
            valid = [int(f) for f in fids if f >= 0]
            pids = self._faiss_to_node_ids("paragraph", lang, valid)
            for pid, sc in zip(pids, scores):
                if pid > 0:
                    candidates.append((pid, float(sc), "paragraph"))

        # CHAPTER-level (each chapter expands into its top paragraphs)
        idx = self._index("chapter", lang)
        if idx is not None:
            scores, fids = emb.search_index(idx, q_vec, cfg.k_chapter)
            scores, fids = scores[0], fids[0]
            valid = [int(f) for f in fids if f >= 0]
            cids = self._faiss_to_node_ids("chapter", lang, valid)
            cid_to_pids = self._chapter_to_paragraphs(
                [c for c in cids if c > 0], top_per_chapter=2,
            )
            for cid, sc in zip(cids, scores):
                if cid <= 0:
                    continue
                for pid in cid_to_pids.get(cid, []):
                    candidates.append((pid, float(sc), "chapter"))

        return candidates, best_sentence

    # ──────────────────────────────────────────────────────────────────
    # Path B — concept-mediated hits
    # ──────────────────────────────────────────────────────────────────

    def _concept_anchors(self, q_vec: np.ndarray, k: int) -> Dict[int, float]:
        """The k nearest corpus concepts to the query vector:
        concept_id -> similarity, insertion-ordered best-first.
        Empty dict when the concept index is missing."""
        idx = self._index("concept", None)
        if idx is None:
            return {}

        scores, fids = emb.search_index(idx, q_vec, k)
        scores, fids = scores[0], fids[0]
        valid_pairs = [(int(f), float(s)) for f, s in zip(fids, scores) if f >= 0]
        if not valid_pairs:
            return {}

        concept_ids = self._faiss_to_node_ids(
            "concept", None, [p[0] for p in valid_pairs]
        )
        primary: Dict[int, float] = {}
        for cid, (_f, s) in zip(concept_ids, valid_pairs):
            if cid > 0 and cid not in primary:
                primary[cid] = s
        return primary

    def _concept_paragraphs(
        self, q_vec: np.ndarray, lang: str,
    ) -> Tuple[List[Tuple[int, float, str]], Dict[int, List[Tuple[int, str, float]]]]:
        """Returns:
          - list of (paragraph_id, score, level='concept'),
          - paragraph_id -> [(concept_id, label, weight)] explanation.
        """
        cfg = CFG.search
        out: List[Tuple[int, float, str]] = []
        explain: Dict[int, List[Tuple[int, str, float]]] = defaultdict(list)

        primary = self._concept_anchors(q_vec, cfg.k_concepts)
        if not primary:
            return out, {}

        # 1-hop neighborhood (similar + co-occurs), weighted
        neigh = graph.concept_neighbors(
            self.conn, list(primary.keys()),
            relations=("similar", "co-occurs"),
            top_k=cfg.k_concept_hops + 4,
            min_weight=0.4,
        )
        # combine: neighborhood concept score = max(primary[src] * neigh_weight)
        combined: Dict[int, float] = dict(primary)
        for src_id, edges in neigh.items():
            base = primary.get(src_id, 0.0)
            for dst_id, w, _rel in edges[: cfg.k_concept_hops]:
                combined[dst_id] = max(combined.get(dst_id, 0.0), 0.85 * base * w)

        # Fetch labels for explanation
        label_map: Dict[int, str] = {}
        if combined:
            placeholders = ",".join("?" * len(combined))
            cur = self.conn.cursor()
            cur.execute(
                f"SELECT id, canonical_label FROM concepts WHERE id IN ({placeholders})",
                tuple(combined.keys()),
            )
            label_map = {row["id"]: row["canonical_label"] for row in cur.fetchall()}

        # Pull paragraphs per concept
        para_by_concept = graph.paragraphs_for_concepts(
            self.conn, list(combined.keys()),
            lang=lang, per_concept=cfg.k_per_concept,
        )

        for cid, c_score in combined.items():
            if c_score <= 0:
                continue
            label = label_map.get(cid, "?")
            for pid, edge_w, _rel in para_by_concept.get(cid, []):
                # paragraph "score" = concept relevance × edge weight to that concept
                p_score = float(c_score) * float(edge_w)
                out.append((pid, p_score, "concept"))
                explain[pid].append((cid, label, float(edge_w)))

        return out, explain

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        languages: Sequence[str] = ("en", "bn"),
        rerank: bool = True,
    ) -> Dict[str, List[SearchHit]]:
        """Returns {lang: [SearchHit, ...]} for each requested language."""
        cfg = CFG.search
        top_k = top_k or cfg.top_k
        query = (query or "").strip()
        if not query:
            return {lang: [] for lang in languages}

        # 1. Encode once
        q_vec = self.embedder.encode([query], batch_size=1, show_progress=False,
                                     max_length=512)

        results: Dict[str, List[SearchHit]] = {}
        for lang in languages:
            results[lang] = self._search_one_lang(query, q_vec, lang, top_k, rerank)
        return results

    def _search_one_lang(
        self, query: str, q_vec: np.ndarray, lang: str,
        top_k: int, rerank: bool,
    ) -> List[SearchHit]:
        cfg = CFG.search

        # PATH A
        a_cands, sentence_for_para = self._vector_paragraphs(q_vec, lang)
        # PATH B
        b_cands, concept_explain = self._concept_paragraphs(q_vec, lang)

        # Build per-source rank lists for RRF
        # Group A by level so each granularity contributes its own ranking.
        by_level: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        for pid, sc, level in a_cands:
            by_level[level].append((pid, sc))
        for pid, sc, _level in b_cands:
            by_level["concept"].append((pid, sc))

        def _ranked(level: str) -> List[int]:
            # dedupe by paragraph_id, keep best, sort desc
            best: Dict[int, float] = {}
            for pid, sc in by_level.get(level, []):
                if pid not in best or sc > best[pid]:
                    best[pid] = sc
            return [pid for pid, _ in sorted(best.items(), key=lambda t: -t[1])]

        rank_lists = [
            _ranked("sentence"),
            _ranked("paragraph"),
            _ranked("chapter"),
            _ranked("concept"),
        ]
        rank_weights = [
            cfg.w_sentence, cfg.w_paragraph, cfg.w_chapter, cfg.w_concept,
        ]
        # Weighted RRF: scale per-list contribution
        fused: Dict[int, float] = defaultdict(float)
        for ranks, w in zip(rank_lists, rank_weights):
            if w == 0:
                continue
            for rank, pid in enumerate(ranks):
                fused[pid] += w * (1.0 / (60 + rank + 1))

        # Track which levels each pid appeared in
        pid_levels: Dict[int, Set[str]] = defaultdict(set)
        for level, lst in by_level.items():
            for pid, _ in lst:
                pid_levels[pid].add(level)

        if not fused:
            return []

        # Take top `rerank_pool` by fused score for cross-encoder
        pool_pids = sorted(fused.keys(), key=lambda p: -fused[p])[: cfg.rerank_pool]
        meta = graph.fetch_paragraph_metadata(self.conn, pool_pids)

        # Rerank
        if rerank and self.reranker is not None and pool_pids:
            passages = [meta[p]["text"] if p in meta else "" for p in pool_pids]
            rer_scores = self.reranker.score(query, passages, batch_size=32)
        else:
            rer_scores = [fused[p] for p in pool_pids]

        ranked = sorted(zip(pool_pids, rer_scores), key=lambda t: -t[1])[:top_k]

        # Build SearchHit objects
        hits: List[SearchHit] = []
        for rank, (pid, sc) in enumerate(ranked, start=1):
            md = meta.get(pid)
            if md is None:
                continue
            # Pick best sentence for highlight
            matched = ""
            sid = sentence_for_para.get(pid)
            if sid is not None:
                cur = self.conn.cursor()
                cur.execute("SELECT text FROM sentences WHERE id=?", (sid,))
                r = cur.fetchone()
                if r:
                    matched = r[0]
            if not matched:
                # Fall back to scoring sentences against query with the embedder.
                # Cheap because we only do this for top-K paragraphs of the
                # final result list.
                sents = self._sentences_of_paragraph(pid)
                if sents:
                    s_vecs = self.embedder.encode(
                        [t for _i, t in sents], batch_size=64, show_progress=False,
                        max_length=512,
                    )
                    sims = (s_vecs @ q_vec[0]).tolist()
                    best_i = int(np.argmax(sims))
                    matched = sents[best_i][1]

            via = concept_explain.get(pid, [])

            hits.append(SearchHit(
                rank=rank,
                score=float(sc),
                paragraph_id=pid,
                lang=lang,
                book_title=md["book_title"],
                volume_title=md["volume_title"],
                volume_num=int(md["volume_num"] or 0),
                section=md["section"],
                chapter_title=md["chapter_title"],
                chapter_id_html=md["chapter_id_html"] or "",
                paragraph_idx=int(md["paragraph_idx"]),
                char_offset=int(md["char_offset"]),
                text=md["text"],
                summary=md.get("summary"),
                matched_sentence=matched,
                para_id_html=md.get("para_id_html") or "",
                via_concepts=via[:3],   # cap UI clutter
                levels=sorted(pid_levels.get(pid, set())),
                pre_rerank_score=float(fused[pid]),
            ))
        return hits

    # ──────────────────────────────────────────────────────────────────
    # Utilities for the UI
    # ──────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, int]:
        cur = self.conn.cursor()
        out: Dict[str, int] = {}
        for table in ("paragraphs", "sentences", "chapters", "volumes",
                      "concepts", "entities"):
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            out[f"n_{table}"] = int(cur.fetchone()[0])
        cur.execute(
            "SELECT b.lang, COUNT(*) FROM paragraphs p "
            "JOIN chapters c ON c.id=p.chapter_id "
            "JOIN volumes v ON v.id=c.volume_id "
            "JOIN books b ON b.id=v.book_id "
            "GROUP BY b.lang"
        )
        for row in cur.fetchall():
            out[f"paragraphs_{row[0]}"] = int(row[1])
        cur.execute("SELECT COUNT(*) FROM concept_edges")
        out["n_concept_edges"] = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM para_concept")
        out["n_para_concept_edges"] = int(cur.fetchone()[0])
        return out

    def search_concepts(self, query: str, k: int = 15) -> List[Tuple[int, str, float]]:
        """The k corpus concepts nearest to a free-text query:
        [(concept_id, canonical_label, similarity)], best-first.
        Used by the QA pipeline to ground modern questions in the concept layer."""
        q_vec = self.embedder.encode([query], batch_size=1, show_progress=False,
                                     max_length=512)
        anchors = self._concept_anchors(q_vec, k)
        if not anchors:
            return []
        placeholders = ",".join("?" * len(anchors))
        cur = self.conn.cursor()
        cur.execute(
            f"SELECT id, canonical_label FROM concepts WHERE id IN ({placeholders})",
            tuple(anchors.keys()),
        )
        label_map = {row["id"]: row["canonical_label"] for row in cur.fetchall()}
        return sorted(
            ((cid, label_map[cid], sim) for cid, sim in anchors.items()
             if cid in label_map),
            key=lambda t: t[2], reverse=True,
        )

    def top_concepts(self, n: int = 50) -> List[Tuple[int, str, int]]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id, canonical_label, n_mentions FROM concepts "
            "ORDER BY n_mentions DESC LIMIT ?",
            (n,),
        )
        return [(int(r[0]), r[1], int(r[2])) for r in cur.fetchall()]

    # ──────────────────────────────────────────────────────────────────
    # Concept-graph visualization (webapp Concept Graph tab)
    # ──────────────────────────────────────────────────────────────────

    def concept_graph(
        self, n: int = 250,
        relations: Sequence[str] = ("similar", "co-occurs"),
    ) -> dict:
        """A JSON-ready subgraph of the top-`n` concepts by n_mentions:
            {"nodes": [{id,label,n_mentions,aliases_en:[..],aliases_bn:[..]}],
             "edges": [{source,target,relation,weight}]}   # undirected, deduped
        Edge weights are min-max normalized *within each relation* (similar and
        co-occurs live on different scales) to [0,1] for consistent styling."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id, canonical_label, n_mentions FROM concepts "
            "ORDER BY n_mentions DESC LIMIT ?", (n,),
        )
        rows = cur.fetchall()
        ids = [int(r["id"]) for r in rows]
        idset = set(ids)
        if not ids:
            return {"nodes": [], "edges": []}

        # aliases (≤3 per language per concept)
        aliases: Dict[int, Dict[str, list]] = {i: {"en": [], "bn": []} for i in ids}
        ph = ",".join("?" * len(ids))
        cur.execute(
            f"SELECT concept_id, lang, alias FROM concept_aliases "
            f"WHERE concept_id IN ({ph})", tuple(ids),
        )
        for r in cur.fetchall():
            bucket = aliases.get(int(r["concept_id"]), {}).get(r["lang"])
            if bucket is not None and len(bucket) < 3 and r["alias"] not in bucket:
                bucket.append(r["alias"])

        nodes = [{
            "id": int(r["id"]), "label": r["canonical_label"],
            "n_mentions": int(r["n_mentions"]),
            "aliases_en": aliases[int(r["id"])]["en"],
            "aliases_bn": aliases[int(r["id"])]["bn"],
        } for r in rows]

        # induced edges, deduped to undirected (min,max,relation)
        rels = tuple(relations)
        rel_ph = ",".join("?" * len(rels))
        cur.execute(
            f"SELECT src_id, dst_id, relation, weight FROM concept_edges "
            f"WHERE relation IN ({rel_ph}) "
            f"AND src_id IN ({ph}) AND dst_id IN ({ph})",
            (*rels, *ids, *ids),
        )
        seen = {}
        for r in cur.fetchall():
            a, b = int(r["src_id"]), int(r["dst_id"])
            if a == b or a not in idset or b not in idset:
                continue
            key = (min(a, b), max(a, b), r["relation"])
            w = float(r["weight"])
            if key not in seen or w > seen[key]:
                seen[key] = w
        # per-relation min-max normalization
        by_rel: Dict[str, list] = defaultdict(list)
        for (a, b, rel), w in seen.items():
            by_rel[rel].append(w)
        span = {rel: (min(ws), max(ws)) for rel, ws in by_rel.items()}
        edges = []
        for (a, b, rel), w in seen.items():
            lo, hi = span[rel]
            norm = (w - lo) / (hi - lo) if hi > lo else 1.0
            edges.append({"source": a, "target": b, "relation": rel,
                          "weight": round(norm, 4)})
        return {"nodes": nodes, "edges": edges}

    def cooccurring_concepts(
        self, concept_ids: Sequence[int], *,
        min_count: int = 1,
        restrict_to: Optional[Sequence[int]] = None,
    ) -> Dict[int, int]:
        """Concepts that share paragraphs with any of `concept_ids`.

        Returns {concept_id: n}, where n is the largest number of paragraphs
        that concept shares with a *single* one of the inputs. Counted straight
        off `para_concept` rather than off the `co-occurs` edge weight, which is
        log-scaled and renormalized and so cannot be read as a paragraph count.
        The inputs themselves are excluded; pairs below `min_count` are dropped.
        `restrict_to` narrows the candidates (e.g. to the nodes on screen).
        """
        cids = [int(c) for c in concept_ids]
        if not cids:
            return {}
        cid_ph = ",".join("?" * len(cids))
        params: List[object] = [*cids, *cids]
        restrict_sql = ""
        if restrict_to is not None:
            keep = [int(c) for c in restrict_to]
            if not keep:
                return {}
            restrict_sql = f"AND b.concept_id IN ({','.join('?' * len(keep))})"
            params.extend(keep)
        cur = self.conn.cursor()
        cur.execute(
            f"""
            SELECT b.concept_id AS cid,
                   COUNT(DISTINCT a.paragraph_id) AS n
              FROM para_concept a
              JOIN para_concept b ON b.paragraph_id = a.paragraph_id
             WHERE a.concept_id IN ({cid_ph})
               AND b.concept_id NOT IN ({cid_ph})
               {restrict_sql}
             GROUP BY b.concept_id, a.concept_id
            """,
            tuple(params),
        )
        best: Dict[int, int] = {}
        for r in cur.fetchall():
            cid, n = int(r["cid"]), int(r["n"])
            if n > best.get(cid, 0):
                best[cid] = n
        return {c: n for c, n in best.items() if n >= min_count}

    def example_paragraphs(
        self, concept_ids: Sequence[int],
        lang: Sequence[str] = ("en", "bn"), k: int = 5,
        *, require_all: bool = False,
    ) -> List[dict]:
        """Paragraphs that best *cover* a set of selected concepts.

        Ranked by (number of the selected concepts present) desc, then by the
        summed para_concept weight over those concepts. Because a paragraph
        holds at most ~5 concepts, an exact all-concepts match is only possible
        for small selections; for larger sets this returns the best partial
        ('best-covering') passages. Returns up to `k` rich records, each with
        the matched concept labels highlighted.

        `require_all` drops the partial matches, keeping only paragraphs that
        carry *every* concept in `concept_ids` — what an edge click wants, since
        "passages for this pair" must not silently include passages about only
        one half of the pair. It can legitimately return nothing (two concepts
        joined by a 'similar' edge may never share a paragraph)."""
        cids = [int(c) for c in concept_ids]
        if not cids:
            return []
        langs = tuple(lang)
        cid_ph = ",".join("?" * len(cids))
        lang_ph = ",".join("?" * len(langs))
        having = f"HAVING COUNT(DISTINCT pc.concept_id) = {len(set(cids))}" if require_all else ""
        cur = self.conn.cursor()
        cur.execute(
            f"""
            SELECT pc.paragraph_id AS pid,
                   COUNT(DISTINCT pc.concept_id) AS n_match,
                   SUM(pc.weight) AS wsum
              FROM para_concept pc
              JOIN paragraphs p  ON p.id = pc.paragraph_id
              JOIN chapters   c  ON c.id = p.chapter_id
              JOIN volumes    v  ON v.id = c.volume_id
              JOIN books      b  ON b.id = v.book_id
             WHERE pc.concept_id IN ({cid_ph})
               AND b.lang IN ({lang_ph})
             GROUP BY pc.paragraph_id
             {having}
             ORDER BY n_match DESC, wsum DESC
             LIMIT ?
            """,
            (*cids, *langs, k),
        )
        ranked = [(int(r["pid"]), int(r["n_match"])) for r in cur.fetchall()]
        if not ranked:
            return []
        pids = [p for p, _ in ranked]
        meta = graph.fetch_paragraph_metadata(self.conn, pids)
        # which of the selected concepts each paragraph carries
        pid_ph = ",".join("?" * len(pids))
        cur.execute(
            f"SELECT pc.paragraph_id AS pid, co.canonical_label AS label "
            f"FROM para_concept pc JOIN concepts co ON co.id = pc.concept_id "
            f"WHERE pc.paragraph_id IN ({pid_ph}) AND pc.concept_id IN ({cid_ph})",
            (*pids, *cids),
        )
        matched: Dict[int, list] = defaultdict(list)
        for r in cur.fetchall():
            matched[int(r["pid"])].append(r["label"])

        out = []
        for pid, n_match in ranked:
            md = meta.get(pid)
            if md is None:
                continue
            md = dict(md)
            md["matched_concepts"] = sorted(matched.get(pid, []))
            md["n_matched"] = n_match
            out.append(md)
        return out

    # Seed concepts per yoga (canonical labels; missing seeds are skipped).
    _YOGA_SEEDS = {
        "jnana": ["knowledge-realization", "non-dualism", "discrimination",
                  "self-knowledge", "knowledge", "wisdom", "brahman", "reality"],
        "bhakti": ["devotion", "worship", "self-surrender", "prayer",
                   "divine-love", "grace"],
        "karma": ["duty", "work", "service", "selfless-work", "action",
                  "non-attachment", "karma"],
        "raja": ["concentration", "meditation", "mind-control", "breath-control",
                 "samadhi", "posture", "self-control"],
    }

    def four_yoga_assignment(
        self, concept_ids: Sequence[int], min_cos: float = 0.30,
    ) -> Dict[int, str]:
        """Assign each concept to the nearest of the four yogas
        (raja/karma/bhakti/jnana) by concept-embedding cosine, or 'other' when
        no yoga is clearly closest. Reuses concept.faiss (no GPU)."""
        cids = [int(c) for c in concept_ids]
        if not cids:
            return {}
        idx = self._index("concept", None)
        if idx is None:
            return {c: "other" for c in cids}

        # concept_id -> faiss row, then reconstruct vectors
        cur = self.conn.cursor()
        all_labels = {name: None for seeds in self._YOGA_SEEDS.values() for name in seeds}
        want_ids = set(cids)
        # map needed concept ids <-> faiss ids
        ph = ",".join("?" * len(cids))
        cur.execute(
            f"SELECT node_id, faiss_id FROM embeddings "
            f"WHERE node_kind='concept' AND lang IS NULL AND node_id IN ({ph})",
            tuple(cids),
        )
        cid_to_fid = {int(r["node_id"]): int(r["faiss_id"]) for r in cur.fetchall()}
        # seed label -> concept id -> faiss id
        seed_ph = ",".join("?" * len(all_labels))
        cur.execute(
            f"SELECT id, canonical_label FROM concepts "
            f"WHERE canonical_label IN ({seed_ph})", tuple(all_labels.keys()),
        )
        label_to_cid = {r["canonical_label"]: int(r["id"]) for r in cur.fetchall()}
        seed_cids = list(label_to_cid.values())
        if seed_cids:
            sph = ",".join("?" * len(seed_cids))
            cur.execute(
                f"SELECT node_id, faiss_id FROM embeddings "
                f"WHERE node_kind='concept' AND lang IS NULL AND node_id IN ({sph})",
                tuple(seed_cids),
            )
            for r in cur.fetchall():
                cid_to_fid[int(r["node_id"])] = int(r["faiss_id"])

        def vec(cid: int):
            fid = cid_to_fid.get(cid)
            if fid is None:
                return None
            v = idx.reconstruct(fid).astype(np.float32)
            nrm = np.linalg.norm(v)
            return v / nrm if nrm > 0 else v

        # yoga centroids from seed vectors
        centroids: Dict[str, np.ndarray] = {}
        for yoga, seeds in self._YOGA_SEEDS.items():
            vs = [vec(label_to_cid[s]) for s in seeds if s in label_to_cid]
            vs = [v for v in vs if v is not None]
            if vs:
                c = np.mean(vs, axis=0)
                nrm = np.linalg.norm(c)
                centroids[yoga] = c / nrm if nrm > 0 else c
        if not centroids:
            return {c: "other" for c in cids}

        yogas = list(centroids)
        C = np.stack([centroids[y] for y in yogas])
        out: Dict[int, str] = {}
        for cid in cids:
            v = vec(cid)
            if v is None:
                out[cid] = "other"
                continue
            sims = C @ v
            j = int(np.argmax(sims))
            out[cid] = yogas[j] if float(sims[j]) >= min_cos else "other"
        return out

    def concept_neighborhood(self, concept_id: int, k: int = 12) -> dict:
        """Return data for visualizing a concept and its neighbors:
            {'center': {...}, 'neighbors': [{...}, ...], 'paragraphs_en': [...], 'paragraphs_bn': [...]}"""
        center = graph.fetch_concept(self.conn, concept_id)
        if not center:
            return {}
        neigh_map = graph.concept_neighbors(
            self.conn, [concept_id], top_k=k,
        )
        edges = neigh_map.get(concept_id, [])
        cur = self.conn.cursor()
        neighbors = []
        if edges:
            ids = [e[0] for e in edges]
            placeholders = ",".join("?" * len(ids))
            cur.execute(
                f"SELECT id, canonical_label, n_mentions FROM concepts "
                f"WHERE id IN ({placeholders})", tuple(ids),
            )
            label_map = {row["id"]: row for row in cur.fetchall()}
            for nid, w, rel in edges:
                row = label_map.get(nid)
                if not row:
                    continue
                neighbors.append({
                    "id": nid, "label": row["canonical_label"],
                    "n_mentions": row["n_mentions"], "weight": w, "relation": rel,
                })

        # sample paragraphs in each language
        sample = graph.paragraphs_for_concepts(
            self.conn, [concept_id], lang="en", per_concept=5,
        ).get(concept_id, [])
        meta_en = graph.fetch_paragraph_metadata(self.conn, [p[0] for p in sample])
        sample_en = [meta_en[p[0]] for p in sample if p[0] in meta_en]

        sample = graph.paragraphs_for_concepts(
            self.conn, [concept_id], lang="bn", per_concept=5,
        ).get(concept_id, [])
        meta_bn = graph.fetch_paragraph_metadata(self.conn, [p[0] for p in sample])
        sample_bn = [meta_bn[p[0]] for p in sample if p[0] in meta_bn]

        return {
            "center": center,
            "neighbors": neighbors,
            "paragraphs_en": sample_en,
            "paragraphs_bn": sample_bn,
        }
