"""Re-attach cached LLM extractions to fresh paragraph rows by text match.

Runs AFTER `01_parse.py --force` (and embed), BEFORE `03_extract_concepts.py`.
For each paragraph in the just-rebuilt corpus, look up `concept_snapshot` by
(normalized text, language, current extractor_tag). If hit, recreate the
concept/entity rows + summary so that stage 3's per-paragraph resumability
check (`summary IS NOT NULL OR has para_concept`) skips this paragraph and
only the genuinely new paragraphs go through the LLM.

Snapshot rows tagged with a different extractor (model/prompt change) are
ignored — those need fresh extraction.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viveka_insight import db as dbmod
from viveka_insight.config import CFG


STEP = "restore_concepts"


def _restore_one(cur, paragraph_id: int, lang: str, summary: str,
                 concepts: list, entities: list) -> None:
    """Mirror `concept_extraction._persist_extraction`'s side effects."""
    if summary:
        cur.execute(
            "UPDATE paragraphs SET summary=? WHERE id=?",
            (summary, paragraph_id),
        )

    for c in concepts:
        label = c["canonical_label"]
        cur.execute(
            "INSERT INTO concepts(canonical_label, n_mentions) VALUES (?, 0) "
            "ON CONFLICT(canonical_label) DO NOTHING",
            (label,),
        )
        cur.execute("SELECT id FROM concepts WHERE canonical_label=?", (label,))
        cid = int(cur.fetchone()[0])
        cur.execute(
            """
            INSERT INTO para_concept(paragraph_id, concept_id, weight, relation)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(paragraph_id, concept_id, relation)
            DO UPDATE SET weight=MAX(weight, excluded.weight)
            """,
            (paragraph_id, cid, float(c.get("weight", 1.0)),
             c.get("relation", "discusses")),
        )
        cur.execute("UPDATE concepts SET n_mentions = n_mentions + 1 WHERE id=?",
                    (cid,))

    for e in entities:
        label = e["canonical_label"]
        etype = e.get("entity_type", "other") or "other"
        cur.execute(
            "INSERT INTO entities(canonical_label, entity_type, n_mentions) "
            "VALUES (?, ?, 0) "
            "ON CONFLICT(canonical_label) DO UPDATE SET entity_type=excluded.entity_type",
            (label, etype),
        )
        cur.execute("SELECT id FROM entities WHERE canonical_label=?", (label,))
        eid = int(cur.fetchone()[0])
        cur.execute(
            "INSERT OR IGNORE INTO para_entity(paragraph_id, entity_id) VALUES (?, ?)",
            (paragraph_id, eid),
        )
        cur.execute("UPDATE entities SET n_mentions = n_mentions + 1 WHERE id=?",
                    (eid,))


def _restore_aliases(conn) -> int:
    """Re-insert concept_aliases for every snapshot alias whose concept now
    exists in `concepts`. Aliases for not-yet-recreated concepts are silently
    skipped — they'll only matter if/when stage 3 produces that concept."""
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR IGNORE INTO concept_aliases (concept_id, lang, alias)
        SELECT c.id, a.lang, a.alias
          FROM alias_snapshot a
          JOIN concepts c ON c.canonical_label = a.canonical_label
        """
    )
    conn.commit()
    return cur.rowcount or 0


def main():
    ap = argparse.ArgumentParser(description="Warm-start concept extractions from snapshot")
    ap.add_argument("--db", default=str(CFG.paths.db))
    args = ap.parse_args()

    conn = dbmod.open_db(args.db, create=False)
    tag = dbmod.extractor_tag()
    print(f"[restore] extractor_tag={tag}")

    if dbmod.step_completed(conn, STEP):
        print(f"[restore] step '{STEP}' already complete — pass --force to redo")
        # We treat this as a one-shot per re-parse; you'd manually wipe the
        # pipeline_state row to re-run. (Restore is idempotent by INSERT OR
        # IGNORE / DO UPDATE, so re-running is safe but wasted work.)
        # For now keep it permissive.

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM concept_snapshot WHERE extractor_tag=?", (tag,))
    n_snap = int(cur.fetchone()[0])
    if n_snap == 0:
        print("[restore] empty snapshot — nothing to restore (first build?)")
        dbmod.mark_step_started(conn, STEP, total=0)
        dbmod.mark_step_done(conn, STEP, note="empty snapshot")
        return

    cur.execute(
        """
        SELECT p.id AS pid, p.text AS text, b.lang AS lang
          FROM paragraphs p
          JOIN chapters c ON c.id = p.chapter_id
          JOIN volumes  v ON v.id = c.volume_id
          JOIN books    b ON b.id = v.book_id
         ORDER BY b.lang, p.id
        """
    )
    paragraphs = cur.fetchall()
    print(f"[restore] {len(paragraphs):,} paragraphs in DB; "
          f"{n_snap:,} snapshot rows for current extractor")

    dbmod.mark_step_started(conn, STEP, total=len(paragraphs))
    t0 = time.time()
    n_hit = n_miss = 0

    for r in paragraphs:
        text_norm = dbmod.normalize_text(r["text"] or "")
        cur.execute(
            "SELECT summary, concepts_json, entities_json FROM concept_snapshot "
            "WHERE text_norm=? AND lang=? AND extractor_tag=?",
            (text_norm, r["lang"], tag),
        )
        row = cur.fetchone()
        if row is None:
            n_miss += 1
            continue
        try:
            concepts = json.loads(row["concepts_json"])
            entities = json.loads(row["entities_json"])
        except json.JSONDecodeError:
            n_miss += 1
            continue
        _restore_one(cur, int(r["pid"]), r["lang"], row["summary"],
                     concepts, entities)
        n_hit += 1
        if n_hit % 2000 == 0:
            conn.commit()
            dbmod.mark_step_progress(conn, STEP, n_hit + n_miss)
            elapsed = time.time() - t0
            print(f"[restore] {n_hit:,} restored, {n_miss:,} not-in-cache "
                  f"({(n_hit + n_miss) / max(elapsed, 1e-6):.0f} para/s)")
    conn.commit()

    n_alias = _restore_aliases(conn)

    dbmod.mark_step_done(
        conn, STEP,
        note=f"restored={n_hit}, fresh={n_miss}, aliases={n_alias}, "
             f"{time.time()-t0:.0f}s",
    )
    print(f"[restore] done: {n_hit:,} restored, {n_miss:,} need fresh extraction, "
          f"{n_alias:,} aliases re-inserted ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
