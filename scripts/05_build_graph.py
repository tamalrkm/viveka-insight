"""Stage 5: validate the assembled graph + write final stats to `meta`.

Pure read-only sanity check. Counts everything, recomputes `n_mentions`
on concepts/entities (in case any merges left them stale), prints a summary,
and stamps the model names + index version into `meta` so the searcher can
verify compatibility on load.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viveka_insight import db as dbmod
from viveka_insight.config import CFG


STEP = "build_graph"


def main():
    ap = argparse.ArgumentParser(description="Validate the graph and write meta")
    ap.add_argument("--db", default=str(CFG.paths.db))
    args = ap.parse_args()

    conn = dbmod.open_db(args.db, create=False)
    cur = conn.cursor()

    dbmod.mark_step_started(conn, STEP)

    # Recount mentions to fix any drift
    cur.execute(
        "UPDATE concepts SET n_mentions = "
        "(SELECT COUNT(*) FROM para_concept WHERE concept_id = concepts.id)"
    )
    cur.execute(
        "UPDATE entities SET n_mentions = "
        "(SELECT COUNT(*) FROM para_entity WHERE entity_id = entities.id)"
    )
    conn.commit()

    # Collect stats
    counts = {}
    for table in ("books", "volumes", "chapters", "paragraphs", "sentences",
                  "concepts", "concept_aliases", "entities",
                  "para_concept", "para_entity", "concept_edges"):
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        counts[table] = int(cur.fetchone()[0])

    # Per-language paragraph totals
    cur.execute(
        "SELECT b.lang, COUNT(*) FROM paragraphs p "
        "JOIN chapters c ON c.id = p.chapter_id "
        "JOIN volumes v ON v.id = c.volume_id "
        "JOIN books b ON b.id = v.book_id "
        "GROUP BY b.lang"
    )
    lang_counts = {row[0]: int(row[1]) for row in cur.fetchall()}

    # Concept relation counts
    cur.execute("SELECT relation, COUNT(*) FROM concept_edges GROUP BY relation")
    rel_counts = {row[0]: int(row[1]) for row in cur.fetchall()}

    # Coverage: how many paragraphs have at least one concept?
    cur.execute(
        "SELECT COUNT(DISTINCT paragraph_id) FROM para_concept"
    )
    paragraphs_with_concepts = int(cur.fetchone()[0])
    coverage = paragraphs_with_concepts / max(counts["paragraphs"], 1)

    # Embedding coverage: how many of each kind are indexed?
    cur.execute("SELECT node_kind, lang, COUNT(*) FROM embeddings "
                "GROUP BY node_kind, lang")
    embed_counts = {f"{row[0]}_{row[1] or 'all'}": int(row[2]) for row in cur.fetchall()}

    # Top concepts (sanity check that extraction makes sense)
    cur.execute("SELECT canonical_label, n_mentions FROM concepts "
                "ORDER BY n_mentions DESC LIMIT 20")
    top_concepts = [(row[0], int(row[1])) for row in cur.fetchall()]

    print("\n" + "=" * 64)
    print("GRAPH STATS")
    print("=" * 64)
    print(f"Books: {counts['books']}, Volumes: {counts['volumes']}, "
          f"Chapters: {counts['chapters']:,}")
    print(f"Paragraphs: {counts['paragraphs']:,} "
          f"(en={lang_counts.get('en', 0):,}, bn={lang_counts.get('bn', 0):,})")
    print(f"Sentences:  {counts['sentences']:,}")
    print(f"Concepts:   {counts['concepts']:,}  ({counts['concept_aliases']:,} aliases)")
    print(f"Entities:   {counts['entities']:,}")
    print(f"Edges: para→concept = {counts['para_concept']:,},  "
          f"para→entity = {counts['para_entity']:,},  "
          f"concept↔concept = {counts['concept_edges']:,} "
          f"({rel_counts})")
    print(f"Concept coverage: {paragraphs_with_concepts:,}/{counts['paragraphs']:,} "
          f"paragraphs ({coverage:.1%})")
    print(f"Embeddings indexed: {embed_counts}")
    print(f"\nTop 20 concepts:")
    for label, n in top_concepts:
        print(f"   {n:>5}  {label}")
    print("=" * 64)

    # Write to meta
    summary = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedder_model": CFG.models.embedder,
        "reranker_model": CFG.models.reranker,
        "llm_model": CFG.models.llm,
        "embedding_dim": CFG.embedding.dim,
        "counts": counts,
        "paragraphs_per_lang": lang_counts,
        "concept_edge_relations": rel_counts,
        "embeddings_indexed": embed_counts,
        "concept_coverage": coverage,
    }
    cur.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                ("stats", json.dumps(summary)))
    cur.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                ("embedder_model", CFG.models.embedder))
    cur.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                ("reranker_model", CFG.models.reranker))
    cur.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                ("llm_model", CFG.models.llm))
    cur.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                ("index_version", "1"))
    conn.commit()

    dbmod.mark_step_done(conn, STEP, note=json.dumps(counts))


if __name__ == "__main__":
    main()
