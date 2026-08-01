"""Known-item cross-lingual retrieval evaluation (paper Section: Evaluation).

The Bengali *Vani o Rachana* contains independent Bengali renderings of many
English lectures. This script:

  1. Finds candidate rendered pairs automatically: mutual-best chapter-level
     embedding matches (cosine >= --min-sim) between the EN and BN chapter
     indexes. Chapter vectors are read back from the existing FAISS indexes,
     so no re-embedding is needed.
  2. Verifies each candidate with the local LLM (title + opening paragraph of
     both sides, YES/NO). Verdicts are cached in docs/paper/eval/pairs.json —
     delete that file to re-verify.
  3. Uses each verified pair as a known-item query: the source chapter's
     title + opening paragraph is the query; the paired chapter in the OTHER
     language is the sole gold item. A retrieved paragraph counts if it
     belongs to the gold chapter.
  4. Reports Recall@10 and MRR for four configurations, in both directions:
       A     — direct dense only (concept path weight = 0)
       B     — concept-mediated only (sentence/paragraph/chapter weights = 0)
       fused — weighted RRF of both paths (no rerank)
       fused+rerank — the full production stack

Run:  python scripts/eval_crosslingual.py
Output: docs/paper/eval/eval_crosslingual.json + LaTeX rows on stdout.

Caveat noted in the paper: pair discovery uses the same embedder as Path A
(whole-chapter vectors vs. title+opening-paragraph queries), so absolute
Path-A numbers may be slightly optimistic; the LLM verification step and the
across-configuration comparison do not depend on that bias.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from viveka_insight import embeddings as emb
from viveka_insight.config import CFG
from viveka_insight.search import Searcher

EVAL_DIR = ROOT / "docs" / "paper" / "eval"
PAIRS_JSON = EVAL_DIR / "pairs.json"
OUT_JSON = EVAL_DIR / "eval_crosslingual.json"

MIN_SIM = 0.70
TOP_K = 10
QUERY_CHARS = 600      # title + this much of the opening paragraph


# ──────────────────────────────────────────────────────────────────────────
# Pair discovery + LLM verification
# ──────────────────────────────────────────────────────────────────────────

def _chapter_vectors(searcher, lang):
    idx = emb.read_index(CFG.paths.faiss("chapter", lang))
    vecs = idx.reconstruct_n(0, idx.ntotal)
    cur = searcher.conn.cursor()
    cur.execute(
        "SELECT faiss_id, node_id FROM embeddings WHERE node_kind='chapter' AND lang=?",
        (lang,),
    )
    fid2cid = {r["faiss_id"]: r["node_id"] for r in cur.fetchall()}
    cids = [fid2cid[i] for i in range(idx.ntotal)]
    return np.asarray(vecs), cids


def _chapter_info(searcher, cid):
    cur = searcher.conn.cursor()
    cur.execute(
        "SELECT c.title, v.title AS volume_title FROM chapters c "
        "JOIN volumes v ON v.id=c.volume_id WHERE c.id=?", (cid,),
    )
    row = cur.fetchone()
    cur.execute(
        "SELECT text FROM paragraphs WHERE chapter_id=? ORDER BY paragraph_idx LIMIT 1",
        (cid,),
    )
    p = cur.fetchone()
    return {
        "title": row["title"], "volume": row["volume_title"],
        "first_para": (p["text"] if p else "")[:800],
    }


def discover_candidates(searcher):
    en_v, en_ids = _chapter_vectors(searcher, "en")
    bn_v, bn_ids = _chapter_vectors(searcher, "bn")
    S = bn_v @ en_v.T
    best_en = S.argmax(1)
    best_bn = S.argmax(0)
    out = []
    for b in range(len(bn_ids)):
        e = int(best_en[b])
        sim = float(S[b, e])
        if best_bn[e] == b and sim >= MIN_SIM:
            out.append({"bn": bn_ids[b], "en": en_ids[e], "sim": sim})
    out.sort(key=lambda p: -p["sim"])
    return out


VERIFY_SYSTEM = (
    "You compare an English lecture and a Bengali chapter from the works of "
    "Swami Vivekananda. Answer with exactly one word: YES if the Bengali "
    "chapter is a Bengali rendering/translation of that same English lecture "
    "(same lecture, same content), NO otherwise."
)

VERIFY_PROMPT = """English lecture
Title: {en_title}
Opening paragraph: {en_para}

Bengali chapter
Title: {bn_title}
Opening paragraph: {bn_para}

Is the Bengali chapter a rendering of this English lecture? YES or NO:"""


def verify_pairs(searcher, llm, candidates, batch=16):
    prompts = []
    for c in candidates:
        en = _chapter_info(searcher, c["en"])
        bn = _chapter_info(searcher, c["bn"])
        c["en_title"], c["bn_title"] = en["title"], bn["title"]
        prompts.append(VERIFY_PROMPT.format(
            en_title=en["title"], en_para=en["first_para"],
            bn_title=bn["title"], bn_para=bn["first_para"],
        ))
    verdicts = []
    for i in range(0, len(prompts), batch):
        outs = llm.generate(prompts[i:i + batch], system=VERIFY_SYSTEM,
                            max_tokens=4, temperature=0.0)
        verdicts.extend(o.strip().upper().startswith("Y") for o in outs)
        print(f"  verified {min(i + batch, len(prompts))}/{len(prompts)}")
    for c, v in zip(candidates, verdicts):
        c["verified"] = bool(v)
    return candidates


# ──────────────────────────────────────────────────────────────────────────
# External baselines
#
# The four configurations above are ablations of our own stack; on their own
# they cannot say whether the stack beats what a reader would otherwise
# reach for. These three baselines supply that reference frame.
# ──────────────────────────────────────────────────────────────────────────

import re
from collections import defaultdict


def _paragraphs_by_lang(conn, lang):
    """[(paragraph_id, text)] for one language."""
    cur = conn.cursor()
    cur.execute(
        "SELECT p.id, p.text FROM paragraphs p "
        "JOIN chapters c ON c.id = p.chapter_id "
        "JOIN volumes v ON v.id = c.volume_id "
        "JOIN books b ON b.id = v.book_id WHERE b.lang = ?", (lang,))
    return [(r["id"], r["text"]) for r in cur.fetchall()]


_TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokenize(text):
    """Unicode word tokens, lowercased. \\w+ covers Bengali as well as Latin,
    which is all this baseline needs — BM25 here exists to establish a
    lexical floor, not to be a tuned Bengali IR system."""
    return _TOKEN.findall(text.lower())


class BM25:
    """Okapi BM25 over one language's paragraphs.

    Implemented inline rather than pulled from a library: the evaluation
    scripts are deliberately dependency-light so that anyone can reproduce
    the table with a stock environment.
    """

    def __init__(self, conn, lang, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        rows = _paragraphs_by_lang(conn, lang)
        self.pids = [pid for pid, _ in rows]
        docs = [_tokenize(t) for _, t in rows]
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if len(docs) else 0.0
        self.postings = defaultdict(list)          # term -> [(doc_idx, tf)]
        for i, d in enumerate(docs):
            tf = defaultdict(int)
            for w in d:
                tf[w] += 1
            for w, f in tf.items():
                self.postings[w].append((i, f))
        self.N = len(docs)
        self.idf = {w: np.log(1 + (self.N - len(p) + 0.5) / (len(p) + 0.5))
                    for w, p in self.postings.items()}

    def top_k(self, query, k):
        scores = np.zeros(self.N, dtype=np.float32)
        for w in _tokenize(query):
            p = self.postings.get(w)
            if not p:
                continue
            idf = self.idf[w]
            for i, f in p:
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * f * (self.k1 + 1) / denom
        if not scores.any():
            return []
        top = np.argpartition(-scores, min(k, self.N - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [self.pids[i] for i in top if scores[i] > 0]


class EntityGraph:
    """A GraphRAG-style stand-in: retrieval mediated by the *entity* graph
    instead of the concept graph.

    GraphRAG builds an entity graph over a corpus with an LLM. Our extraction
    stage already produces exactly such a layer (`entities` / `para_entity`)
    alongside the concept layer, so routing retrieval through it isolates the
    one variable that matters here: whether *concepts* or *entities* are the
    right node type for linking discursive prose across languages.
    """

    def __init__(self, searcher):
        cur = searcher.conn.cursor()
        cur.execute("SELECT id, canonical_label FROM entities")
        rows = cur.fetchall()
        self.ids = [r["id"] for r in rows]
        labels = [r["canonical_label"] for r in rows]
        print(f"  [entity-graph] embedding {len(labels)} entity labels ...")
        V = searcher.embedder.encode(labels, batch_size=256,
                                     show_progress=False, max_length=64)
        V = np.asarray(V, dtype=np.float32)
        V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        self.V = V
        cur.execute(
            "SELECT pe.entity_id, pe.paragraph_id, b.lang FROM para_entity pe "
            "JOIN paragraphs p ON p.id = pe.paragraph_id "
            "JOIN chapters c ON c.id = p.chapter_id "
            "JOIN volumes v ON v.id = c.volume_id "
            "JOIN books b ON b.id = v.book_id")
        self.by_entity = defaultdict(lambda: defaultdict(list))
        for r in cur.fetchall():
            self.by_entity[r["lang"]][r["entity_id"]].append(r["paragraph_id"])

    def top_k(self, q_vec, lang, k, n_anchors=12):
        q = np.asarray(q_vec, dtype=np.float32).reshape(-1)
        q /= (np.linalg.norm(q) + 1e-9)
        sims = self.V @ q
        anchors = np.argpartition(-sims, n_anchors)[:n_anchors]
        scored = defaultdict(float)
        for a in anchors:
            eid, s = self.ids[a], float(sims[a])
            for pid in self.by_entity[lang].get(eid, ()):
                scored[pid] += s
        return [pid for pid, _ in
                sorted(scored.items(), key=lambda kv: -kv[1])[:k]]


# ──────────────────────────────────────────────────────────────────────────
# Retrieval evaluation
# ──────────────────────────────────────────────────────────────────────────

CONFIGS = {
    # external baselines
    "bm25": {"baseline": "bm25"},
    "dense_paragraph": {"w_sentence": 0.0, "w_chapter": 0.0, "w_concept": 0.0,
                        "rerank": False},
    "entity_graph": {"baseline": "entity"},
    # ablations of our own stack
    "path_a": {"w_concept": 0.0, "rerank": False},
    "path_b": {"w_sentence": 0.0, "w_paragraph": 0.0, "w_chapter": 0.0,
               "rerank": False},
    "fused": {"rerank": False},
    "fused_rerank": {"rerank": True},
}

_DEFAULT_W = {k: getattr(CFG.search, k)
              for k in ("w_sentence", "w_paragraph", "w_chapter", "w_concept")}


def _query_for(searcher, cid):
    info = _chapter_info(searcher, cid)
    return f"{info['title']}. {info['first_para']}"[:QUERY_CHARS]


def _para_chapter_map(searcher, pids):
    if not pids:
        return {}
    cur = searcher.conn.cursor()
    ph = ",".join("?" * len(pids))
    cur.execute(f"SELECT id, chapter_id FROM paragraphs WHERE id IN ({ph})",
                tuple(pids))
    return {r["id"]: r["chapter_id"] for r in cur.fetchall()}


def remap_stale_pairs(searcher, candidates):
    """Repair cached pair IDs after an index rebuild.

    `01_parse --force` reassigns chapter row IDs, so a `pairs.json` cached
    against an earlier build points at IDs that no longer exist and the
    evaluation silently loses every gold item. We re-resolve each pair by
    (language, chapter title), which is unique for most chapters; where a
    title repeats across volumes we pick the combination whose chapter-vector
    cosine is closest to the similarity recorded when the pair was
    discovered. The LLM verdicts are preserved, so this costs no GPU time and
    keeps the evaluated pair set identical to the one the paper reports.
    """
    cur = searcher.conn.cursor()
    cur.execute("SELECT id FROM chapters")
    live = {r["id"] for r in cur.fetchall()}
    if all(c["en"] in live and c["bn"] in live for c in candidates):
        return candidates, 0

    from collections import defaultdict as _dd
    by_title = _dd(list)
    cur.execute("SELECT c.id, c.title, b.lang FROM chapters c "
                "JOIN volumes v ON v.id=c.volume_id "
                "JOIN books b ON b.id=v.book_id")
    for r in cur.fetchall():
        by_title[(r["lang"], (r["title"] or "").strip())].append(r["id"])

    vecs, cid_index = {}, {}
    for lg in ("en", "bn"):
        V, cids = _chapter_vectors(searcher, lg)
        V = np.asarray(V, dtype=np.float32)
        V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        vecs[lg] = V
        cid_index[lg] = {c: i for i, c in enumerate(cids)}

    remapped = 0
    for c in candidates:
        en_c = by_title.get(("en", (c.get("en_title") or "").strip()), [])
        bn_c = by_title.get(("bn", (c.get("bn_title") or "").strip()), [])
        if not en_c or not bn_c:
            continue
        if len(en_c) == 1 and len(bn_c) == 1:
            best = (en_c[0], bn_c[0])
        else:
            target = float(c.get("sim", 0.0))
            best, err = None, 1e9
            for e in en_c:
                for b in bn_c:
                    ie, ib = cid_index["en"].get(e), cid_index["bn"].get(b)
                    if ie is None or ib is None:
                        continue
                    s = float(vecs["bn"][ib] @ vecs["en"][ie])
                    if abs(s - target) < err:
                        best, err = (e, b), abs(s - target)
            if best is None:
                continue
        if (c["en"], c["bn"]) != best:
            c["en_stale"], c["bn_stale"] = c["en"], c["bn"]
            c["en"], c["bn"] = best
            remapped += 1
    return candidates, remapped


def run_config(searcher, pairs, name, spec, tools=None):
    """Evaluate one configuration. `spec` either names an external baseline
    or sets fusion weights on the production searcher."""
    baseline = spec.get("baseline")
    if not baseline:
        for k, v in _DEFAULT_W.items():
            setattr(CFG.search, k, v)
        for k, v in spec.items():
            if k != "rerank":
                setattr(CFG.search, k, v)
        rerank = spec["rerank"]

    metrics = {}
    for direction in ("en2bn", "bn2en"):
        src, tgt = ("en", "bn") if direction == "en2bn" else ("bn", "en")
        hits_at_k, rr = [], []
        for p in pairs:
            query = _query_for(searcher, p[src])
            gold = p[tgt]

            if baseline == "bm25":
                pids = tools["bm25"][tgt].top_k(query, TOP_K)
            elif baseline == "entity":
                q_vec = searcher.embedder.encode(
                    [query], batch_size=1, show_progress=False, max_length=512)
                pids = tools["entity"].top_k(q_vec, tgt, TOP_K)
            else:
                res = searcher.search(query, top_k=TOP_K, languages=(tgt,),
                                      rerank=rerank)
                pids = [h.paragraph_id for h in res.get(tgt, [])]

            chap = _para_chapter_map(searcher, pids)
            rank = next((i for i, pid in enumerate(pids, 1)
                         if chap.get(pid) == gold), None)
            hits_at_k.append(1.0 if rank else 0.0)
            rr.append(1.0 / rank if rank else 0.0)
        metrics[direction] = {
            "recall@10": sum(hits_at_k) / len(hits_at_k),
            "mrr": sum(rr) / len(rr),
        }
    for k, v in _DEFAULT_W.items():          # restore
        setattr(CFG.search, k, v)
    print(f"  {name}: " + "  ".join(
        f"{d} R@10={m['recall@10']:.3f} MRR={m['mrr']:.3f}"
        for d, m in metrics.items()))
    return metrics


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    searcher = Searcher(load_reranker=True)

    if PAIRS_JSON.exists():
        candidates = json.loads(PAIRS_JSON.read_text())
        print(f"loaded {len(candidates)} cached candidates from {PAIRS_JSON}")
    else:
        print("discovering candidate pairs ...")
        candidates = discover_candidates(searcher)
        print(f"  {len(candidates)} mutual-best candidates (sim >= {MIN_SIM})")
        from viveka_insight.llm_client import make_client
        llm = make_client(CFG)
        candidates = verify_pairs(searcher, llm, candidates)
        llm.unload()
        PAIRS_JSON.write_text(json.dumps(candidates, ensure_ascii=False, indent=1))

    candidates, n_remapped = remap_stale_pairs(searcher, candidates)
    if n_remapped:
        print(f"  repaired {n_remapped} stale chapter IDs after an index "
              f"rebuild (LLM verdicts preserved)")
        PAIRS_JSON.write_text(json.dumps(candidates, ensure_ascii=False,
                                         indent=1))

    pairs = [c for c in candidates if c["verified"]]
    cur = searcher.conn.cursor()
    cur.execute("SELECT id FROM chapters")
    live = {r["id"] for r in cur.fetchall()}
    unresolved = [p for p in pairs if p["en"] not in live or p["bn"] not in live]
    if unresolved:
        raise SystemExit(
            f"  ✗ {len(unresolved)} verified pairs still reference chapter IDs "
            f"absent from the index. Delete {PAIRS_JSON.name} and re-run to "
            f"rediscover and re-verify from scratch.")
    print(f"verified rendered pairs: {len(pairs)} / {len(candidates)}")

    # Build the external baselines once (BM25 indexes + entity embeddings).
    tools = {}
    if any(s.get("baseline") == "bm25" for s in CONFIGS.values()):
        print("building BM25 indexes ...")
        tools["bm25"] = {lg: BM25(searcher.conn, lg) for lg in ("en", "bn")}
        for lg, b in tools["bm25"].items():
            print(f"  [bm25] {lg}: {b.N} paragraphs, "
                  f"{len(b.postings)} distinct terms")
    if any(s.get("baseline") == "entity" for s in CONFIGS.values()):
        print("building entity-graph baseline ...")
        tools["entity"] = EntityGraph(searcher)

    results = {"n_candidates": len(candidates), "n_pairs": len(pairs),
               "min_sim": MIN_SIM, "top_k": TOP_K, "configs": {}}
    for name, spec in CONFIGS.items():
        print(f"running config {name} ...")
        results["configs"][name] = run_config(searcher, pairs, name, spec, tools)

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_JSON}\n\nLaTeX rows (R@10, MRR as en2bn / bn2en):")
    label = {"bm25": "BM25 (lexical)",
             "dense_paragraph": "Dense paragraph retrieval",
             "entity_graph": "Entity graph (GraphRAG-style)",
             "path_a": "Path A (direct dense)",
             "path_b": "Path B (concept-mediated)",
             "fused": "Fused (weighted RRF)",
             "fused_rerank": "Fused + cross-encoder rerank"}
    for name in CONFIGS:
        m = results["configs"][name]
        print(f"{label[name]:32s} & "
              f"{m['en2bn']['recall@10']:.2f} / {m['bn2en']['recall@10']:.2f} & "
              f"{m['en2bn']['mrr']:.2f} / {m['bn2en']['mrr']:.2f}\\\\")


if __name__ == "__main__":
    main()
