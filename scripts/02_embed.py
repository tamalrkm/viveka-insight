"""Stage 2: embed every text node at every granularity, write FAISS files.

Granularities:
  sentence    — short, precise; best for finding the exact line
  paragraph   — medium, contextual; best for thematic match
  chapter     — long-context (BGE-M3 supports 8192 tokens), great for "what's
                this query about at the section level"

For each (granularity, language) we maintain one flat IP FAISS index. Each
node's row in the embeddings table records its position there; the searcher
maps back via that table.

Resumability: every (granularity, language) is its own pipeline_state step
(`embed_sentence_en`, `embed_paragraph_bn`, etc.) and is skipped if already
done. If you kill mid-step, just rerun — partial state is wiped and that
step starts over.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viveka_insight import db as dbmod
from viveka_insight import embeddings as emb
from viveka_insight.config import CFG


def _step_name(kind: str, lang: str) -> str:
    return f"embed_{kind}_{lang}"


def _wipe_kind_lang(conn, kind: str, lang: str) -> None:
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM embeddings WHERE node_kind=? AND lang=?", (kind, lang),
    )
    conn.commit()
    fpath = CFG.paths.faiss(kind, lang)
    if fpath.exists():
        fpath.unlink()


def _embed_sentences(conn, embedder, lang: str) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT s.id AS sid, s.text AS text
          FROM sentences s
          JOIN paragraphs p ON p.id = s.paragraph_id
          JOIN chapters   c ON c.id = p.chapter_id
          JOIN volumes    v ON v.id = c.volume_id
          JOIN books      b ON b.id = v.book_id
         WHERE b.lang = ?
         ORDER BY s.id
        """,
        (lang,),
    )
    rows = cur.fetchall()
    if not rows:
        print(f"[embed] sentence/{lang}: nothing to embed")
        return 0

    sids = [int(r["sid"]) for r in rows]
    texts = [r["text"] for r in rows]
    print(f"[embed] sentence/{lang}: encoding {len(texts):,} sentences")

    BATCH = CFG.embedding.batch_size_sentence
    all_vecs: List[np.ndarray] = []
    t0 = time.time()
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        v = embedder.encode(batch, batch_size=BATCH, show_progress=False, max_length=256)
        all_vecs.append(v)
        if (i // BATCH) % 20 == 0:
            done = i + len(batch)
            rate = done / max(time.time() - t0, 1e-6)
            print(f"[embed] sentence/{lang}: {done:,}/{len(texts):,} "
                  f"({rate:.0f} sent/s)")
    vecs = np.concatenate(all_vecs, axis=0)

    index = emb.new_flat_index(embedder.dim)
    emb.add_to_index(index, vecs)
    emb.write_index(index, CFG.paths.faiss("sentence", lang))

    cur.executemany(
        "INSERT INTO embeddings(node_kind, node_id, lang, faiss_id) "
        "VALUES ('sentence', ?, ?, ?)",
        [(sid, lang, i) for i, sid in enumerate(sids)],
    )
    conn.commit()
    print(f"[embed] sentence/{lang}: wrote {index.ntotal:,} vectors in "
          f"{time.time()-t0:.0f}s")
    return index.ntotal


def _embed_paragraphs(conn, embedder, lang: str) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT p.id AS pid, p.text AS text
          FROM paragraphs p
          JOIN chapters c ON c.id = p.chapter_id
          JOIN volumes  v ON v.id = c.volume_id
          JOIN books    b ON b.id = v.book_id
         WHERE b.lang = ?
         ORDER BY p.id
        """,
        (lang,),
    )
    rows = cur.fetchall()
    if not rows:
        return 0

    pids = [int(r["pid"]) for r in rows]
    texts = [r["text"] for r in rows]
    print(f"[embed] paragraph/{lang}: encoding {len(texts):,} paragraphs")

    BATCH = CFG.embedding.batch_size_paragraph
    all_vecs: List[np.ndarray] = []
    t0 = time.time()
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        v = embedder.encode(batch, batch_size=BATCH, show_progress=False, max_length=1024)
        all_vecs.append(v)
        if (i // BATCH) % 10 == 0:
            done = i + len(batch)
            rate = done / max(time.time() - t0, 1e-6)
            print(f"[embed] paragraph/{lang}: {done:,}/{len(texts):,} "
                  f"({rate:.0f} para/s)")
    vecs = np.concatenate(all_vecs, axis=0)

    index = emb.new_flat_index(embedder.dim)
    emb.add_to_index(index, vecs)
    emb.write_index(index, CFG.paths.faiss("paragraph", lang))

    cur.executemany(
        "INSERT INTO embeddings(node_kind, node_id, lang, faiss_id) "
        "VALUES ('paragraph', ?, ?, ?)",
        [(pid, lang, i) for i, pid in enumerate(pids)],
    )
    conn.commit()
    print(f"[embed] paragraph/{lang}: wrote {index.ntotal:,} vectors in "
          f"{time.time()-t0:.0f}s")
    return index.ntotal


def _embed_chapters(conn, embedder, lang: str) -> int:
    """Concatenate paragraph texts per chapter, capped at max_tokens_chapter
    characters (~ tokens for English, more conservative for Bengali).
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT c.id AS cid, c.title AS title,
               GROUP_CONCAT(p.text, ' ') AS text
          FROM chapters c
          JOIN paragraphs p ON p.chapter_id = c.id
          JOIN volumes  v ON v.id = c.volume_id
          JOIN books    b ON b.id = v.book_id
         WHERE b.lang = ?
         GROUP BY c.id
         ORDER BY c.id
        """,
        (lang,),
    )
    rows = cur.fetchall()
    if not rows:
        return 0

    cids = []
    texts = []
    cap_chars = CFG.embedding.max_tokens_chapter * 4    # rough ~4 chars/token
    for r in rows:
        if r["text"] is None:
            continue
        cids.append(int(r["cid"]))
        # Prepend title — gives the embedder a cheap topical anchor
        title = (r["title"] or "").strip()
        body = r["text"][:cap_chars]
        texts.append(f"{title}\n\n{body}" if title else body)

    print(f"[embed] chapter/{lang}: encoding {len(texts):,} chapters")
    BATCH = CFG.embedding.batch_size_chapter
    all_vecs: List[np.ndarray] = []
    t0 = time.time()
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        v = embedder.encode(batch, batch_size=BATCH, show_progress=False,
                            max_length=CFG.embedding.max_tokens_chapter)
        all_vecs.append(v)
        if (i // BATCH) % 5 == 0:
            done = i + len(batch)
            rate = done / max(time.time() - t0, 1e-6)
            print(f"[embed] chapter/{lang}: {done:,}/{len(texts):,} "
                  f"({rate:.1f} chap/s)")
    vecs = np.concatenate(all_vecs, axis=0)

    index = emb.new_flat_index(embedder.dim)
    emb.add_to_index(index, vecs)
    emb.write_index(index, CFG.paths.faiss("chapter", lang))

    cur.executemany(
        "INSERT INTO embeddings(node_kind, node_id, lang, faiss_id) "
        "VALUES ('chapter', ?, ?, ?)",
        [(cid, lang, i) for i, cid in enumerate(cids)],
    )
    conn.commit()
    print(f"[embed] chapter/{lang}: wrote {index.ntotal:,} vectors in "
          f"{time.time()-t0:.0f}s")
    return index.ntotal


KIND_FNS = {
    "sentence":  _embed_sentences,
    "paragraph": _embed_paragraphs,
    "chapter":   _embed_chapters,
}


def main():
    ap = argparse.ArgumentParser(description="Embed text at all granularities")
    ap.add_argument("--db", default=str(CFG.paths.db))
    ap.add_argument("--kinds", nargs="+",
                    default=["sentence", "paragraph", "chapter"],
                    choices=["sentence", "paragraph", "chapter"])
    ap.add_argument("--langs", nargs="+", default=["en", "bn"], choices=["en", "bn"])
    ap.add_argument("--force", action="store_true",
                    help="wipe existing embeddings for the listed (kind,lang) and re-embed")
    args = ap.parse_args()

    conn = dbmod.open_db(args.db, create=False)

    embedder = emb.Embedder(
        model_name=CFG.models.embedder,
        device=CFG.device,
        fp16=CFG.embedding.fp16,
    )
    embedder.load()

    try:
        for kind in args.kinds:
            for lang in args.langs:
                step = _step_name(kind, lang)
                if args.force:
                    _wipe_kind_lang(conn, kind, lang)
                    cur = conn.cursor()
                    cur.execute("DELETE FROM pipeline_state WHERE step=?", (step,))
                    conn.commit()
                if dbmod.step_completed(conn, step):
                    print(f"[embed] {step}: already complete — skipping")
                    continue

                # If a previous interrupted run left partial rows, wipe them.
                cur = conn.cursor()
                cur.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE node_kind=? AND lang=?",
                    (kind, lang),
                )
                n_partial = cur.fetchone()[0]
                if n_partial > 0:
                    print(f"[embed] {step}: found {n_partial} partial rows; cleaning")
                    _wipe_kind_lang(conn, kind, lang)

                dbmod.mark_step_started(conn, step)
                t0 = time.time()
                fn = KIND_FNS[kind]
                n = fn(conn, embedder, lang)
                dbmod.mark_step_done(conn, step,
                                     note=f"{n} vectors, {time.time()-t0:.0f}s")
    finally:
        embedder.unload()


if __name__ == "__main__":
    main()
