"""Stage 1: parse HTML → books / volumes / chapters / paragraphs / sentences in SQLite.

Idempotent: running again with the same DB skips parsing if `parse` is marked
complete in `pipeline_state`. Use `--force` to wipe and reparse.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make the package importable when running this as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viveka_insight import db as dbmod
from viveka_insight.config import CFG
from viveka_insight.parser import (
    parse_english, parse_bengali, iter_chapters, split_sentences,
)


STEP = "parse"


def _ingest_one_lang(conn, lang, html_path, parse_fn, book_title):
    cur = conn.cursor()

    # Insert book row (cross-lang, one per file)
    cur.execute(
        "INSERT OR IGNORE INTO books(lang, title) VALUES (?, ?)",
        (lang, book_title),
    )
    cur.execute("SELECT id FROM books WHERE lang=? AND title=?", (lang, book_title))
    book_id = cur.fetchone()[0]
    conn.commit()

    paragraphs = list(parse_fn(html_path))
    print(f"[parse] {lang}: parsed {len(paragraphs):,} paragraphs from {html_path}")

    # Group by (volume, chapter) to insert hierarchy in order
    groups = list(iter_chapters(iter(paragraphs)))
    print(f"[parse] {lang}: grouped into {len(groups):,} chapters")

    # Insert volumes (deduped) keeping insertion order
    seen_vols = {}
    for g in groups:
        if g.volume not in seen_vols:
            cur.execute(
                "INSERT OR IGNORE INTO volumes(book_id, title, volume_num, ord) "
                "VALUES (?, ?, ?, ?)",
                (book_id, g.volume, g.volume_num, len(seen_vols)),
            )
            cur.execute("SELECT id FROM volumes WHERE book_id=? AND title=?",
                        (book_id, g.volume))
            seen_vols[g.volume] = cur.fetchone()[0]
    conn.commit()

    # Insert chapters + paragraphs + sentences
    n_paras = 0
    n_sents = 0
    for g_idx, g in enumerate(groups):
        vol_id = seen_vols[g.volume]
        cur.execute(
            "INSERT INTO chapters(volume_id, section, title, chapter_id_html, ord) "
            "VALUES (?, ?, ?, ?, ?)",
            (vol_id, g.section, g.chapter, g.chapter_id, g_idx),
        )
        chap_id = cur.lastrowid

        for p in g.paragraphs:
            cur.execute(
                "INSERT INTO paragraphs(chapter_id, paragraph_idx, text, char_offset, para_id_html) "
                "VALUES (?, ?, ?, ?, ?)",
                (chap_id, p.paragraph_idx, p.text, p.char_offset,
                 getattr(p, "para_id_html", "")),
            )
            para_id = cur.lastrowid
            n_paras += 1

            sents = split_sentences(p.text, lang)
            # Drop trivially short fragments (matches existing behavior; otherwise
            # the corpus is dominated by lone "Yes." / "No." sentences).
            sents = [s for s in sents if len(s.split()) >= 3 or len(s) >= 25]
            for s_idx, s_text in enumerate(sents):
                cur.execute(
                    "INSERT INTO sentences(paragraph_id, sentence_idx, text) VALUES (?,?,?)",
                    (para_id, s_idx, s_text),
                )
                n_sents += 1

        # commit every chapter so we have a checkpoint
        if g_idx % 50 == 0:
            conn.commit()
    conn.commit()
    print(f"[parse] {lang}: stored {n_paras:,} paragraphs, {n_sents:,} sentences")


def main():
    ap = argparse.ArgumentParser(description="Parse HTML into the SQLite knowledge graph")
    ap.add_argument("--en", default=str(CFG.paths.en_html))
    ap.add_argument("--bn", default=str(CFG.paths.bn_html))
    ap.add_argument("--db", default=str(CFG.paths.db))
    ap.add_argument("--force", action="store_true",
                    help="wipe existing rows and re-parse from scratch")
    args = ap.parse_args()

    db_path = Path(args.db)
    conn = dbmod.open_db(db_path, create=True)

    if args.force:
        cur = conn.cursor()
        # Order matters under foreign_keys=ON: child rows first.
        for table in (
            "para_concept", "para_entity", "concept_edges", "concept_aliases",
            "embeddings", "sentences", "paragraphs", "chapters", "volumes",
            "books", "concepts", "entities",
        ):
            cur.execute(f"DELETE FROM {table}")
        cur.execute(
            "DELETE FROM pipeline_state "
            "WHERE step=? OR step LIKE 'embed_%' "
            "   OR step IN ('concepts_paragraphs', 'concepts_chapter_summaries', "
            "               'link_concepts', 'restore_concepts', 'graph_stats', 'gnn')",
            (STEP,),
        )
        # `concept_snapshot` and `alias_snapshot` deliberately preserved — they
        # are the warm-start cache that this rebuild is going to consume.
        conn.commit()

    if dbmod.step_completed(conn, STEP):
        print(f"[parse] step '{STEP}' already complete — pass --force to redo")
        return

    dbmod.mark_step_started(conn, STEP, total=2)

    t0 = time.time()
    _ingest_one_lang(
        conn, "en", args.en, parse_english,
        "Complete Works of Swami Vivekananda",
    )
    dbmod.mark_step_progress(conn, STEP, 1)

    _ingest_one_lang(
        conn, "bn", args.bn, parse_bengali,
        "স্বামী বিবেকানন্দ সমগ্র",
    )
    dbmod.mark_step_done(conn, STEP, note=f"{time.time()-t0:.0f}s")

    # Quick stats
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM paragraphs"); print(f"[parse] paragraphs: {cur.fetchone()[0]:,}")
    cur.execute("SELECT COUNT(*) FROM sentences");  print(f"[parse] sentences: {cur.fetchone()[0]:,}")
    cur.execute("SELECT COUNT(*) FROM chapters");   print(f"[parse] chapters: {cur.fetchone()[0]:,}")
    print(f"[parse] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
