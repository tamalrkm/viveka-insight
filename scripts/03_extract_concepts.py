"""Stage 3: LLM concept + entity extraction over all paragraphs.

For each paragraph we run the LLM to get:
  * up to 5 canonical concept tags (English labels — cross-lingual key)
  * up to 5 named entities (people / places / texts / deities)
  * one English summary sentence

The output populates `concepts`, `concept_aliases`, `entities`, `para_concept`,
`para_entity`, and the `summary` field on each `paragraph` row.

Resumability is per-paragraph: we record progress in `pipeline_state.progress`
as a count of completed paragraphs and skip any paragraph that already has at
least one para_concept row OR a non-null summary. Re-running picks up where
the previous run died.

After all paragraphs are processed, we make a quick chapter-summary pass: for
each chapter, we ask the LLM to summarize its constituent paragraph summaries
into a single chapter-level summary (much cheaper — ~one call per chapter,
not per paragraph).

Cost estimate (Qwen2.5-14B + vLLM on A100):
  ~30K paragraphs × ~500 output tokens / call → ~15M tokens
  vLLM throughput ~1500 tok/s → ~3 hours total.
Use `--limit N` to test on a subset first.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viveka_insight import db as dbmod
from viveka_insight.config import CFG
from viveka_insight.concept_extraction import (
    Extraction, build_user_prompt, extract_batch, parse_extraction, SYSTEM_PROMPT,
)
from viveka_insight.llm_client import make_client


STEP_PARA = "concepts_paragraphs"
STEP_CHAP = "concepts_chapter_summaries"


def _completed_paragraph_ids(conn) -> set:
    """Paragraphs that already have either a summary or a para_concept row.

    We use this rather than a strict `pipeline_state.progress` counter so a
    half-finished run (with say 12,000 paragraphs done, then SIGKILL) resumes
    correctly — pipeline_state alone doesn't tell us *which* paragraphs got
    processed."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id FROM paragraphs
         WHERE summary IS NOT NULL
            OR id IN (SELECT DISTINCT paragraph_id FROM para_concept)
        """
    )
    return {int(r[0]) for r in cur.fetchall()}


def _persist_extraction(
    conn, paragraph_id: int, lang: str, ext: Extraction,
) -> None:
    """Insert concepts, entities, edges; update paragraph.summary."""
    cur = conn.cursor()

    # Update paragraph summary
    if ext.summary:
        cur.execute(
            "UPDATE paragraphs SET summary=? WHERE id=?",
            (ext.summary, paragraph_id),
        )

    # Concepts
    for c in ext.concepts:
        cur.execute(
            "INSERT INTO concepts(canonical_label, n_mentions) VALUES (?, 0) "
            "ON CONFLICT(canonical_label) DO NOTHING",
            (c.canonical_label,),
        )
        cur.execute(
            "SELECT id FROM concepts WHERE canonical_label=?",
            (c.canonical_label,),
        )
        cid = int(cur.fetchone()[0])

        # Edge para → concept
        cur.execute(
            """
            INSERT INTO para_concept(paragraph_id, concept_id, weight, relation)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(paragraph_id, concept_id, relation)
            DO UPDATE SET weight=MAX(weight, excluded.weight)
            """,
            (paragraph_id, cid, c.weight, c.relation),
        )
        cur.execute(
            "UPDATE concepts SET n_mentions = n_mentions + 1 WHERE id=?", (cid,),
        )
        # Surface phrase as alias in original language
        if c.surface:
            cur.execute(
                "INSERT OR IGNORE INTO concept_aliases(concept_id, lang, alias) "
                "VALUES (?, ?, ?)",
                (cid, lang, c.surface),
            )

    # Entities
    for e in ext.entities:
        cur.execute(
            "INSERT INTO entities(canonical_label, entity_type, n_mentions) "
            "VALUES (?, ?, 0) "
            "ON CONFLICT(canonical_label) DO UPDATE SET entity_type=excluded.entity_type",
            (e.canonical_label, e.entity_type),
        )
        cur.execute(
            "SELECT id FROM entities WHERE canonical_label=?", (e.canonical_label,),
        )
        eid = int(cur.fetchone()[0])
        cur.execute(
            "INSERT OR IGNORE INTO para_entity(paragraph_id, entity_id) VALUES (?, ?)",
            (paragraph_id, eid),
        )
        cur.execute(
            "UPDATE entities SET n_mentions = n_mentions + 1 WHERE id=?", (eid,),
        )


def run_paragraph_extraction(conn, llm, *, limit: int = 0, batch: int = 64) -> int:
    """Walk all paragraphs (skipping those already done) and extract."""
    cur = conn.cursor()

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
    all_paras = cur.fetchall()
    done = _completed_paragraph_ids(conn)
    todo = [(int(r["pid"]), r["text"], r["lang"]) for r in all_paras
            if int(r["pid"]) not in done]
    if limit and len(todo) > limit:
        todo = todo[:limit]

    total = len(todo)
    print(f"[concepts] {len(done):,} paragraphs already done; "
          f"{total:,} to process this run")
    if total == 0:
        return 0

    dbmod.mark_step_started(conn, STEP_PARA, total=total)
    t0 = time.time()
    n_processed = 0

    for i in range(0, total, batch):
        chunk = todo[i : i + batch]
        try:
            results = extract_batch(
                llm, chunk,
                max_input_tokens=CFG.llm.max_input_tokens,
                max_output_tokens=CFG.llm.max_output_tokens,
                temperature=CFG.llm.temperature,
                top_p=CFG.llm.top_p,
            )
        except Exception as e:
            # Don't lose progress on a transient error; log and skip the chunk.
            print(f"[concepts] batch {i//batch} failed: {e!r} — skipping")
            continue

        # Persist
        for (pid, _t, lang), (_pid, ext) in zip(chunk, results):
            _persist_extraction(conn, pid, lang, ext)
        conn.commit()

        n_processed += len(chunk)
        dbmod.mark_step_progress(conn, STEP_PARA, n_processed)

        elapsed = time.time() - t0
        rate = n_processed / max(elapsed, 1e-6)
        eta_min = (total - n_processed) / max(rate, 1e-6) / 60
        print(f"[concepts] {n_processed:,}/{total:,} "
              f"({rate:.1f} para/s, ETA {eta_min:.0f} min)")

    dbmod.mark_step_done(conn, STEP_PARA,
                         note=f"{n_processed} processed, {time.time()-t0:.0f}s")
    return n_processed


# ──────────────────────────────────────────────────────────────────────────
# Chapter summary pass (light)
# ──────────────────────────────────────────────────────────────────────────

CHAPTER_SYSTEM = """You write concise English summaries of philosophical chapters.
Given a chapter title and the per-paragraph summaries of its content, write
ONE coherent summary of 2–4 sentences (≤80 words) capturing the chapter's
main thread. Output ONLY the summary text. No prose around it. No markdown."""


def _chapter_summary_prompt(title: str, summaries: List[str]) -> str:
    bullets = "\n".join(f"- {s}" for s in summaries if s)
    return (
        f"Chapter: {title}\n\n"
        f"Per-paragraph summaries (in order):\n{bullets}\n\n"
        f"Write the chapter summary."
    )


def run_chapter_summaries(conn, llm, *, batch: int = 16) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT c.id AS cid, c.title AS title,
               GROUP_CONCAT(p.summary, '||') AS sums
          FROM chapters c
          JOIN paragraphs p ON p.chapter_id = c.id
         WHERE c.summary IS NULL
         GROUP BY c.id
         HAVING SUM(CASE WHEN p.summary IS NULL THEN 0 ELSE 1 END) >= 2
         ORDER BY c.id
        """
    )
    rows = cur.fetchall()
    if not rows:
        print("[concepts] no chapter summaries to write")
        return 0

    print(f"[concepts] writing chapter summaries for {len(rows):,} chapters")
    dbmod.mark_step_started(conn, STEP_CHAP, total=len(rows))
    n_done = 0
    t0 = time.time()

    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        prompts = [
            _chapter_summary_prompt(r["title"] or "", (r["sums"] or "").split("||"))
            for r in chunk
        ]
        try:
            outs = llm.generate(
                prompts, system=CHAPTER_SYSTEM,
                max_tokens=200, temperature=0.2, top_p=0.9,
            )
        except Exception as e:
            print(f"[concepts] chapter summary batch {i//batch} failed: {e!r}")
            continue

        for r, summary in zip(chunk, outs):
            summary = summary.strip().strip('"').strip()
            cur.execute(
                "UPDATE chapters SET summary=? WHERE id=?",
                (summary[:600], int(r["cid"])),
            )
        conn.commit()
        n_done += len(chunk)
        dbmod.mark_step_progress(conn, STEP_CHAP, n_done)
        elapsed = time.time() - t0
        rate = n_done / max(elapsed, 1e-6)
        eta_min = (len(rows) - n_done) / max(rate, 1e-6) / 60
        print(f"[concepts] chapter {n_done:,}/{len(rows):,} "
              f"({rate:.1f} chap/s, ETA {eta_min:.0f} min)")

    dbmod.mark_step_done(conn, STEP_CHAP, note=f"{n_done} chapter summaries")
    return n_done


# ──────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="LLM-driven concept extraction")
    ap.add_argument("--db", default=str(CFG.paths.db))
    ap.add_argument("--limit", type=int, default=0,
                    help="cap number of paragraphs to process this run (0 = all)")
    ap.add_argument("--batch", type=int, default=CFG.llm.batch)
    ap.add_argument("--skip-paragraphs", action="store_true",
                    help="skip paragraph-level extraction (e.g., to only run chapter summaries)")
    ap.add_argument("--skip-chapter-summaries", action="store_true",
                    help="skip chapter-summary generation step")
    args = ap.parse_args()

    conn = dbmod.open_db(args.db, create=False)

    print(f"[concepts] backend={CFG.llm.backend}, model={CFG.models.llm}")
    llm = make_client(CFG)
    try:
        if not args.skip_paragraphs:
            run_paragraph_extraction(conn, llm, limit=args.limit, batch=args.batch)
        if not args.skip_chapter_summaries:
            run_chapter_summaries(conn, llm, batch=max(args.batch // 2, 4))
    finally:
        llm.unload()


if __name__ == "__main__":
    main()
