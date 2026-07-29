"""Smoke tests — fast, no GPU required, no big-model downloads.

These cover the parts that don't need the actual models loaded:
    * parser produces sane records on the real source files,
    * SQLite schema accepts the typical inserts the pipeline does,
    * concept-extraction JSON parser handles realistic malformed LLM output,
    * concept-linking clustering does the right thing on toy embeddings.

Run:  pytest tests/ -x -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from viveka_insight import db as dbmod
from viveka_insight.concept_extraction import (
    parse_extraction, _normalize_concept_label, _find_json_block,
)
from viveka_insight.concept_linking import find_merge_groups, pick_survivor
from viveka_insight.parser import (
    parse_english, parse_bengali, iter_chapters, split_sentences,
)


# ──────────────────────────────────────────────────────────────────────────
# Parser tests (require source files in data/)
# ──────────────────────────────────────────────────────────────────────────

EN_HTML = ROOT / "data" / "vivekananda_complete_works.html"
BN_HTML = ROOT / "data" / "all.html"

@pytest.mark.skipif(not EN_HTML.exists(), reason="English source file not present")
def test_english_parser_basics():
    paragraphs = list(parse_english(str(EN_HTML)))
    assert len(paragraphs) > 1000          # corpus is much bigger; sanity floor
    assert all(p.lang == "en" for p in paragraphs)
    assert all(p.volume_num >= 1 for p in paragraphs)
    assert all(p.text for p in paragraphs)

@pytest.mark.skipif(not BN_HTML.exists(), reason="Bengali source file not present")
def test_bengali_parser_basics():
    paragraphs = list(parse_bengali(str(BN_HTML)))
    assert len(paragraphs) > 1000
    assert all(p.lang == "bn" for p in paragraphs)
    assert all(p.volume_num >= 1 for p in paragraphs)


@pytest.mark.skipif(not (EN_HTML.exists() and BN_HTML.exists()),
                    reason="source files not present")
def test_parsers_capture_paragraph_anchors():
    # The re-exported HTML gives every <p> a stable id="p-N"; the parser must
    # capture it so citations can deep-link to the exact paragraph. The Bengali
    # source also switched its chapter wrapper (div.bb-item -> section
    # .content-block), which the parser must still handle.
    for parse, path in ((parse_english, EN_HTML), (parse_bengali, BN_HTML)):
        paras = list(parse(str(path)))
        assert paras, "parser returned no paragraphs"
        with_anchor = [p for p in paras if p.para_id_html]
        # near-total coverage; anchors look like "p-<n>"
        assert len(with_anchor) > 0.9 * len(paras)
        assert all(p.para_id_html.startswith("p-") for p in with_anchor)

@pytest.mark.skipif(not EN_HTML.exists(), reason="English source file not present")
def test_iter_chapters_groups_correctly():
    paras = list(parse_english(str(EN_HTML)))[:200]
    chapters = list(iter_chapters(iter(paras)))
    # Every chapter should have ≥1 paragraph
    assert all(len(c.paragraphs) >= 1 for c in chapters)
    # Reassembled paragraph count should equal input
    assert sum(len(c.paragraphs) for c in chapters) == len(paras)


# ──────────────────────────────────────────────────────────────────────────
# Sentence splitting
# ──────────────────────────────────────────────────────────────────────────

def test_sentence_splitter_bengali():
    text = "এটি প্রথম বাক্য। এটি দ্বিতীয়! তৃতীয় কি?"
    sents = split_sentences(text, "bn")
    assert len(sents) == 3

def test_sentence_splitter_english():
    text = "First sentence. Second one! And a third?"
    sents = split_sentences(text, "en")
    assert len(sents) == 3


# ──────────────────────────────────────────────────────────────────────────
# DB schema
# ──────────────────────────────────────────────────────────────────────────

def test_schema_round_trip(tmp_path):
    db_path = tmp_path / "t.sqlite"
    conn = dbmod.open_db(db_path, create=True)
    cur = conn.cursor()

    # Minimal end-to-end insert
    cur.execute("INSERT INTO books(lang, title) VALUES ('en', 'Test')")
    book_id = cur.lastrowid
    cur.execute("INSERT INTO volumes(book_id, title, volume_num) VALUES (?, 'Vol I', 1)",
                (book_id,))
    vol_id = cur.lastrowid
    cur.execute("INSERT INTO chapters(volume_id, title, ord) VALUES (?, 'Ch1', 0)",
                (vol_id,))
    chap_id = cur.lastrowid
    cur.execute("INSERT INTO paragraphs(chapter_id, paragraph_idx, text, char_offset) "
                "VALUES (?, 0, 'hello', 0)", (chap_id,))
    para_id = cur.lastrowid
    cur.execute("INSERT INTO concepts(canonical_label) VALUES ('patience')")
    cid = cur.lastrowid
    cur.execute("INSERT INTO para_concept(paragraph_id, concept_id, weight) VALUES (?, ?, 0.9)",
                (para_id, cid))
    conn.commit()

    # Pipeline state
    dbmod.mark_step_started(conn, "test_step", total=10)
    assert not dbmod.step_completed(conn, "test_step")
    dbmod.mark_step_progress(conn, "test_step", 5)
    dbmod.mark_step_done(conn, "test_step", note="ok")
    assert dbmod.step_completed(conn, "test_step")


# ──────────────────────────────────────────────────────────────────────────
# JSON extractor robustness — the LLM output we *will* see in production
# ──────────────────────────────────────────────────────────────────────────

def test_parse_extraction_clean_json():
    raw = json.dumps({
        "concepts": [{"label": "patience", "surface": "ধৈর্য", "weight": 0.9, "relation": "discusses"}],
        "entities": [{"label": "Buddha", "type": "person", "surface": "Buddha"}],
        "summary": "A short paragraph about patience.",
    })
    ext = parse_extraction(raw)
    assert len(ext.concepts) == 1
    assert ext.concepts[0].canonical_label == "patience"
    assert ext.entities[0].entity_type == "person"
    assert ext.summary

def test_parse_extraction_with_markdown_fence():
    raw = (
        "Here is the JSON:\n```json\n"
        '{"concepts": [{"label": "Self-Realization", "weight": 1.0}], "entities": [], "summary": "x"}\n'
        "```\nLet me know if anything's unclear."
    )
    ext = parse_extraction(raw)
    assert ext.concepts[0].canonical_label == "self-realization"

def test_parse_extraction_trailing_commas():
    raw = '{"concepts": [{"label": "duty",}], "entities": [], "summary": "y",}'
    ext = parse_extraction(raw)
    assert ext.concepts[0].canonical_label == "duty"

def test_parse_extraction_garbage_returns_empty():
    ext = parse_extraction("totally not JSON\nthere is no way")
    assert ext.concepts == [] and ext.entities == []

def test_normalize_concept_label():
    assert _normalize_concept_label("Self  Realization") == "self-realization"
    assert _normalize_concept_label(" Non-Attachment! ") == "non-attachment"
    assert _normalize_concept_label("") == ""
    assert _normalize_concept_label("a") == ""    # too short
    assert _normalize_concept_label("x" * 80) == ""    # too long

def test_find_json_block_balances_braces():
    s = 'preamble {"a": {"b": 1}} trailing'
    assert _find_json_block(s) == '{"a": {"b": 1}}'


# ──────────────────────────────────────────────────────────────────────────
# Concept linking clustering
# ──────────────────────────────────────────────────────────────────────────

def test_find_merge_groups():
    # 4 labels, first three are near-identical, fourth is far off
    embs = np.array([
        [1.0, 0.0, 0.0],
        [0.99, 0.05, 0.0],
        [0.98, 0.10, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    labels = ["a", "b", "c", "d"]
    clusters = find_merge_groups(labels, embs, threshold=0.90)
    # Three near-identical labels merge into one cluster of size 3, "d" alone.
    assert sorted([len(c) for c in clusters]) == [1, 3]

def test_pick_survivor_prefers_high_count():
    labels = ["renouncing", "renunciation", "renouncement"]
    counts = {"renouncing": 4, "renunciation": 100, "renouncement": 7}
    s = pick_survivor([0, 1, 2], labels, counts)
    assert s == 1  # "renunciation" has the most mentions


# ──────────────────────────────────────────────────────────────────────────
# Warm-start (snapshot/restore) cache
# ──────────────────────────────────────────────────────────────────────────

def test_normalize_text_collapses_whitespace_and_normalizes_unicode():
    # Pure layout edits (extra blanks, NFD compose) should collapse to the
    # same key — that's what makes the cache survive HTML re-edits.
    a = dbmod.normalize_text("  Hello   world\n  ")
    b = dbmod.normalize_text("Hello world")
    assert a == b == "Hello world"
    # NFC normalization: "é" as two codepoints vs one should collapse.
    nfd = "café"
    nfc = "café"
    assert dbmod.normalize_text(nfd) == dbmod.normalize_text(nfc)


def test_extractor_tag_is_stable_string():
    tag = dbmod.extractor_tag()
    assert "@v" in tag and tag.split("@v")[-1].isdigit()


def test_concept_snapshot_round_trip(tmp_path):
    """Snapshot → wipe → restore: matching paragraphs get their concepts and
    summary back; mismatched text doesn't pollute."""
    db_path = tmp_path / "t.sqlite"
    conn = dbmod.open_db(db_path, create=True)
    cur = conn.cursor()

    # Seed: one paragraph with one concept and a summary.
    cur.execute("INSERT INTO books(lang, title) VALUES ('en', 'T')")
    book = cur.lastrowid
    cur.execute("INSERT INTO volumes(book_id, title, volume_num) VALUES (?, 'V', 1)", (book,))
    vol = cur.lastrowid
    cur.execute("INSERT INTO chapters(volume_id, title, ord) VALUES (?, 'C', 0)", (vol,))
    chap = cur.lastrowid
    cur.execute("INSERT INTO paragraphs(chapter_id, paragraph_idx, text, char_offset, summary) "
                "VALUES (?, 0, 'Patience is a virtue.', 0, 'About patience.')",
                (chap,))
    pid = cur.lastrowid
    cur.execute("INSERT INTO concepts(canonical_label, n_mentions) VALUES ('patience', 1)")
    cid = cur.lastrowid
    cur.execute("INSERT INTO para_concept(paragraph_id, concept_id, weight, relation) "
                "VALUES (?, ?, 0.9, 'discusses')", (pid, cid))
    cur.execute("INSERT INTO concept_aliases(concept_id, lang, alias) VALUES (?, 'en', 'patience')",
                (cid,))
    conn.commit()

    # Snapshot
    import json as _json
    from datetime import datetime, timezone
    tag = dbmod.extractor_tag()
    cur.execute(
        "INSERT INTO concept_snapshot(text_norm, lang, extractor_tag, summary, "
        " concepts_json, entities_json, captured_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (dbmod.normalize_text("Patience is a virtue."), "en", tag, "About patience.",
         _json.dumps([{"canonical_label": "patience", "relation": "discusses", "weight": 0.9}]),
         _json.dumps([]), datetime.now(timezone.utc).isoformat()),
    )
    cur.execute("INSERT INTO alias_snapshot(canonical_label, lang, alias) VALUES "
                "('patience', 'en', 'patience')")
    conn.commit()

    # Wipe extraction state (mirrors what 01_parse.py --force does)
    for table in ("para_concept", "para_entity", "concept_edges", "concept_aliases",
                  "embeddings", "sentences", "paragraphs", "chapters", "volumes",
                  "books", "concepts", "entities"):
        cur.execute(f"DELETE FROM {table}")
    conn.commit()

    # Re-seed with the same text under fresh autoincrement ids
    cur.execute("INSERT INTO books(lang, title) VALUES ('en', 'T')")
    book2 = cur.lastrowid
    cur.execute("INSERT INTO volumes(book_id, title, volume_num) VALUES (?, 'V', 1)", (book2,))
    vol2 = cur.lastrowid
    cur.execute("INSERT INTO chapters(volume_id, title, ord) VALUES (?, 'C', 0)", (vol2,))
    chap2 = cur.lastrowid
    cur.execute("INSERT INTO paragraphs(chapter_id, paragraph_idx, text, char_offset) "
                "VALUES (?, 0, '  Patience  is a virtue.\\n', 0)", (chap2,))   # whitespace drift
    new_pid = cur.lastrowid
    cur.execute("INSERT INTO paragraphs(chapter_id, paragraph_idx, text, char_offset) "
                "VALUES (?, 1, 'Wholly different new text.', 1)", (chap2,))
    new_pid_unmatched = cur.lastrowid
    conn.commit()

    # Run restore inline (re-importing the script's _restore_one logic)
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location(
        "restore_concepts",
        ROOT / "scripts" / "restore_concepts.py",
    )
    mod = _ilu.module_from_spec(spec); spec.loader.exec_module(mod)

    # Mimic main()'s loop body: lookup + restore for each paragraph
    for pid_, text in [(new_pid, "  Patience  is a virtue.\n"),
                       (new_pid_unmatched, "Wholly different new text.")]:
        cur.execute(
            "SELECT summary, concepts_json, entities_json FROM concept_snapshot "
            "WHERE text_norm=? AND lang=? AND extractor_tag=?",
            (dbmod.normalize_text(text), "en", tag),
        )
        row = cur.fetchone()
        if row is None:
            continue
        mod._restore_one(cur, pid_, "en", row["summary"],
                         _json.loads(row["concepts_json"]),
                         _json.loads(row["entities_json"]))
    mod._restore_aliases(conn)

    # Matched paragraph should be warm-started
    cur.execute("SELECT summary FROM paragraphs WHERE id=?", (new_pid,))
    assert cur.fetchone()["summary"] == "About patience."
    cur.execute("SELECT COUNT(*) FROM para_concept WHERE paragraph_id=?", (new_pid,))
    assert cur.fetchone()[0] == 1
    # Unmatched paragraph should be untouched (forces fresh extraction in stage 3)
    cur.execute("SELECT summary FROM paragraphs WHERE id=?", (new_pid_unmatched,))
    assert cur.fetchone()["summary"] is None
    cur.execute("SELECT COUNT(*) FROM para_concept WHERE paragraph_id=?", (new_pid_unmatched,))
    assert cur.fetchone()[0] == 0
    # Aliases re-attached to the new concept row
    cur.execute("SELECT COUNT(*) FROM concept_aliases ca "
                "JOIN concepts c ON c.id=ca.concept_id "
                "WHERE c.canonical_label='patience' AND ca.alias='patience'")
    assert cur.fetchone()[0] == 1
