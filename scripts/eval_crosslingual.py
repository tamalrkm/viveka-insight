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
# Retrieval evaluation
# ──────────────────────────────────────────────────────────────────────────

CONFIGS = {
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


def run_config(searcher, pairs, name, spec):
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
            res = searcher.search(query, top_k=TOP_K, languages=(tgt,),
                                  rerank=rerank)
            got = res.get(tgt, [])
            chap = _para_chapter_map(searcher, [h.paragraph_id for h in got])
            rank = next((i for i, h in enumerate(got, 1)
                         if chap.get(h.paragraph_id) == gold), None)
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

    pairs = [c for c in candidates if c["verified"]]
    print(f"verified rendered pairs: {len(pairs)} / {len(candidates)}")

    results = {"n_candidates": len(candidates), "n_pairs": len(pairs),
               "min_sim": MIN_SIM, "top_k": TOP_K, "configs": {}}
    for name, spec in CONFIGS.items():
        print(f"running config {name} ...")
        results["configs"][name] = run_config(searcher, pairs, name, spec)

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_JSON}\n\nLaTeX rows (R@10, MRR as en2bn / bn2en):")
    label = {"path_a": "Path A (direct dense)",
             "path_b": "Path B (concept-mediated)",
             "fused": "Fused (weighted RRF)",
             "fused_rerank": "Fused + cross-encoder rerank"}
    for name in CONFIGS:
        m = results["configs"][name]
        print(f"{label[name]:30s} & "
              f"{m['en2bn']['recall@10']:.2f} / {m['bn2en']['recall@10']:.2f} & "
              f"{m['en2bn']['mrr']:.2f} / {m['bn2en']['mrr']:.2f}\\\\")


if __name__ == "__main__":
    main()
