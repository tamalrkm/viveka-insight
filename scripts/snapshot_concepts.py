"""Dump current per-paragraph concept extractions into the warm-start cache.

Run this BEFORE `01_parse.py --force` whenever a source HTML changes. The
cache survives the parse-time wipe; `restore_concepts.py` then re-attaches
extractions to whichever new paragraphs match by normalized text.

Idempotent: re-running snapshot just refreshes. Rows from prior extractor
versions are kept (older `extractor_tag`) — restore filters by current tag.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viveka_insight import db as dbmod
from viveka_insight.config import CFG


def _serialize_paragraphs(conn, tag: str) -> int:
    """Walk every paragraph that has summary or any concept/entity links and
    write a single concept_snapshot row capturing what the LLM produced."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT p.id   AS pid,
               p.text AS text,
               p.summary AS summary,
               b.lang AS lang
          FROM paragraphs p
          JOIN chapters c ON c.id = p.chapter_id
          JOIN volumes  v ON v.id = c.volume_id
          JOIN books    b ON b.id = v.book_id
         WHERE p.summary IS NOT NULL
            OR p.id IN (SELECT paragraph_id FROM para_concept)
            OR p.id IN (SELECT paragraph_id FROM para_entity)
         ORDER BY b.lang, p.id
        """
    )
    rows = cur.fetchall()
    if not rows:
        print("[snapshot] no extracted paragraphs to snapshot")
        return 0

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n = 0
    for r in rows:
        pid = int(r["pid"])
        text_norm = dbmod.normalize_text(r["text"] or "")
        if not text_norm:
            continue

        cur.execute(
            """
            SELECT c.canonical_label AS label, pc.relation AS rel, pc.weight AS w
              FROM para_concept pc
              JOIN concepts c ON c.id = pc.concept_id
             WHERE pc.paragraph_id = ?
            """,
            (pid,),
        )
        concepts = [
            {"canonical_label": cr["label"], "relation": cr["rel"], "weight": float(cr["w"])}
            for cr in cur.fetchall()
        ]

        cur.execute(
            """
            SELECT e.canonical_label AS label, e.entity_type AS etype
              FROM para_entity pe
              JOIN entities e ON e.id = pe.entity_id
             WHERE pe.paragraph_id = ?
            """,
            (pid,),
        )
        entities = [
            {"canonical_label": er["label"], "entity_type": er["etype"]}
            for er in cur.fetchall()
        ]

        if not concepts and not entities and not r["summary"]:
            continue

        cur.execute(
            """
            INSERT OR REPLACE INTO concept_snapshot
                (text_norm, lang, extractor_tag, summary,
                 concepts_json, entities_json, captured_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (text_norm, r["lang"], tag, r["summary"],
             json.dumps(concepts, ensure_ascii=False),
             json.dumps(entities, ensure_ascii=False), now),
        )
        n += 1
        if n % 5000 == 0:
            conn.commit()
            print(f"[snapshot] {n:,} paragraphs serialized")
    conn.commit()
    return n


def _serialize_aliases(conn) -> int:
    """Snapshot every concept alias keyed by canonical_label (stable across
    re-parse) instead of concept_id (autoincrement, churns)."""
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR IGNORE INTO alias_snapshot (canonical_label, lang, alias)
        SELECT c.canonical_label, ca.lang, ca.alias
          FROM concept_aliases ca
          JOIN concepts c ON c.id = ca.concept_id
        """
    )
    conn.commit()
    cur.execute("SELECT COUNT(*) FROM alias_snapshot")
    return int(cur.fetchone()[0])


def main():
    ap = argparse.ArgumentParser(description="Snapshot per-paragraph extractions for warm-start")
    ap.add_argument("--db", default=str(CFG.paths.db))
    args = ap.parse_args()

    conn = dbmod.open_db(args.db, create=True)
    tag = dbmod.extractor_tag()
    print(f"[snapshot] db={args.db}")
    print(f"[snapshot] extractor_tag={tag}")

    t0 = time.time()
    n_para = _serialize_paragraphs(conn, tag)
    n_alias = _serialize_aliases(conn)
    print(f"[snapshot] {n_para:,} paragraphs, {n_alias:,} aliases — "
          f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
