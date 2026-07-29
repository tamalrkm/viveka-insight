"""Stage 4: link concepts.

Three sub-steps, all idempotent:

  4a. Embed every canonical concept label and write `concept.faiss`.
      Also populate the `embeddings` table for concept nodes.

  4b. Merge near-duplicate concepts (cosine ≥ 0.92 by default) — survivors
      keep their incoming `para_concept` edges and aliases; absorbed labels
      become aliases of the survivor.

  4c. Build concept ↔ concept edges:
        - 'similar'   : top-K cosine ≥ threshold per concept
        - 'co-occurs' : pairs that appear in the same paragraph ≥ N times
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viveka_insight import db as dbmod
from viveka_insight import embeddings as emb
from viveka_insight.concept_linking import (
    merge_concepts_in_db, build_similarity_edges, build_cooccurrence_edges,
)
from viveka_insight.config import CFG


STEP = "link_concepts"


def _embed_all_concepts(conn, embedder) -> int:
    """Replace concepts.faiss with a fresh embedding of every canonical label."""
    cur = conn.cursor()
    cur.execute("SELECT id, canonical_label FROM concepts ORDER BY id")
    rows = cur.fetchall()
    if not rows:
        print("[link] no concepts to embed")
        return 0

    ids = [int(r["id"]) for r in rows]
    labels = [r["canonical_label"] for r in rows]
    print(f"[link] embedding {len(labels):,} concept labels")
    vecs = embedder.encode(labels, batch_size=256, show_progress=False, max_length=64)

    # Wipe + write
    cur.execute(
        "DELETE FROM embeddings WHERE node_kind='concept'",
    )
    index = emb.new_flat_index(embedder.dim)
    emb.add_to_index(index, vecs)
    emb.write_index(index, CFG.paths.faiss("concept", None))

    cur.executemany(
        "INSERT INTO embeddings(node_kind, node_id, lang, faiss_id) "
        "VALUES ('concept', ?, NULL, ?)",
        [(cid, i) for i, cid in enumerate(ids)],
    )
    conn.commit()
    print(f"[link] wrote {index.ntotal:,} concept vectors -> "
          f"{CFG.paths.faiss('concept', None)}")
    return index.ntotal


def main():
    ap = argparse.ArgumentParser(description="Concept linking + graph edges")
    ap.add_argument("--db", default=str(CFG.paths.db))
    ap.add_argument("--merge-threshold", type=float,
                    default=CFG.concept_linking.merge_threshold)
    ap.add_argument("--sim-threshold", type=float,
                    default=CFG.concept_linking.sim_threshold)
    ap.add_argument("--sim-top-k", type=int,
                    default=CFG.concept_linking.sim_top_k)
    ap.add_argument("--cooccur-min-count", type=int,
                    default=CFG.concept_linking.cooccur_min_count)
    ap.add_argument("--skip-merge", action="store_true")
    ap.add_argument("--skip-similar", action="store_true")
    ap.add_argument("--skip-cooccur", action="store_true")
    args = ap.parse_args()

    conn = dbmod.open_db(args.db, create=False)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM concepts")
    n0 = int(cur.fetchone()[0])
    print(f"[link] starting with {n0:,} concepts")

    dbmod.mark_step_started(conn, STEP)
    t0 = time.time()

    embedder = emb.Embedder(
        model_name=CFG.models.embedder,
        device=CFG.device,
        fp16=CFG.embedding.fp16,
    )
    embedder.load()

    try:
        # 4a. Initial embedding pass (needed for the merge step)
        _embed_all_concepts(conn, embedder)

        # 4b. Merge near-duplicates
        if not args.skip_merge:
            remap = merge_concepts_in_db(
                conn, embedder=embedder, threshold=args.merge_threshold,
            )
            if remap:
                # Re-embed concepts after merging — survivors set has changed.
                _embed_all_concepts(conn, embedder)

        # 4c. Similarity edges
        if not args.skip_similar:
            # Wipe previous 'similar' edges so re-runs don't duplicate
            cur.execute("DELETE FROM concept_edges WHERE relation='similar'")
            conn.commit()
            build_similarity_edges(
                conn, embedder,
                top_k=args.sim_top_k, threshold=args.sim_threshold,
            )

        # 4d. Co-occurrence edges
        if not args.skip_cooccur:
            cur.execute("DELETE FROM concept_edges WHERE relation='co-occurs'")
            conn.commit()
            build_cooccurrence_edges(conn, min_count=args.cooccur_min_count)

    finally:
        embedder.unload()

    cur.execute("SELECT COUNT(*) FROM concepts"); n1 = int(cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM concept_edges"); ne = int(cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM para_concept"); pe = int(cur.fetchone()[0])
    print(f"[link] done: concepts {n0:,} -> {n1:,}  "
          f"concept_edges={ne:,}  para_concept={pe:,}  "
          f"({time.time()-t0:.0f}s)")
    dbmod.mark_step_done(conn, STEP,
                         note=f"concepts={n1}, edges={ne}, time={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
