"""SQLite schema for everything we store about the corpus.

Why SQLite for a knowledge graph?
    - The graph is small enough (tens of thousands of nodes, low millions of
      edges) that a real graph DB is overkill.
    - SQLite is single-file, zero-ops, embedded — pairs well with FAISS files
      sitting next to it in `index_data/`.
    - Adjacency-list traversal is fast: indexed lookups in O(log n) per hop.
    - When we do want graph algorithms (PageRank, GNN), we load edges into
      networkx / torch_geometric in a few hundred ms.

Layout (high level):
    books → volumes → chapters → paragraphs → sentences   (the text hierarchy)
    concepts (with aliases) and entities                  (the abstraction layer)
    para_concept, para_entity                             (text → abstraction edges)
    concept_edges                                          (abstraction → abstraction)
    embeddings                                             (node ↔ FAISS row mapping)
"""
from __future__ import annotations

import re
import sqlite3
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA = r"""
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- ── Text hierarchy ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS books (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    lang    TEXT NOT NULL,                  -- 'en' | 'bn'
    title   TEXT NOT NULL,
    UNIQUE(lang, title)
);

CREATE TABLE IF NOT EXISTS volumes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id      INTEGER NOT NULL REFERENCES books(id),
    title        TEXT    NOT NULL,          -- "Volume I" / "খণ্ড ১"
    volume_num   INTEGER,                   -- normalized arabic number for cross-lang
    ord          INTEGER,
    summary      TEXT,                      -- LLM-generated, English
    UNIQUE(book_id, title)
);
CREATE INDEX IF NOT EXISTS idx_volumes_book ON volumes(book_id);

CREATE TABLE IF NOT EXISTS chapters (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    volume_id       INTEGER NOT NULL REFERENCES volumes(id),
    section         TEXT,                   -- English "section-head" if any
    title           TEXT NOT NULL,
    chapter_id_html TEXT,                   -- "ch_0" / "content-5" -- back-link to source
    ord             INTEGER,
    summary         TEXT
);
CREATE INDEX IF NOT EXISTS idx_chapters_volume ON chapters(volume_id);

CREATE TABLE IF NOT EXISTS paragraphs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    chapter_id    INTEGER NOT NULL REFERENCES chapters(id),
    paragraph_idx INTEGER NOT NULL,
    text          TEXT    NOT NULL,
    char_offset   INTEGER NOT NULL,
    para_id_html  TEXT    NOT NULL DEFAULT '',  -- html id of source <p> (e.g. "p-42"); '' if none
    summary       TEXT                      -- 1-line English gist (LLM)
);
CREATE INDEX IF NOT EXISTS idx_paragraphs_chapter ON paragraphs(chapter_id);

CREATE TABLE IF NOT EXISTS sentences (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    paragraph_id INTEGER NOT NULL REFERENCES paragraphs(id),
    sentence_idx INTEGER NOT NULL,
    text         TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sentences_paragraph ON sentences(paragraph_id);

-- ── Abstraction layer ──────────────────────────────────────────────────────
-- Concepts are abstract ideas (patience, renunciation, maya). Their canonical
-- label is always English+lowercase so a Bengali paragraph and an English
-- paragraph can hit the same concept node.
CREATE TABLE IF NOT EXISTS concepts (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_label    TEXT NOT NULL UNIQUE,
    description        TEXT,
    n_mentions         INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_concepts_label ON concepts(canonical_label);

-- Aliases: surface forms that map to a concept ("kshanti"->patience, "সবর"->...)
CREATE TABLE IF NOT EXISTS concept_aliases (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    concept_id INTEGER NOT NULL REFERENCES concepts(id),
    lang       TEXT NOT NULL,
    alias      TEXT NOT NULL,
    UNIQUE(concept_id, lang, alias)
);
CREATE INDEX IF NOT EXISTS idx_concept_aliases_concept ON concept_aliases(concept_id);
CREATE INDEX IF NOT EXISTS idx_concept_aliases_alias ON concept_aliases(alias);

-- Concrete entities: people, places, texts, deities. Separate from concepts so
-- the UI can let the user filter "show me everywhere Buddha is discussed."
CREATE TABLE IF NOT EXISTS entities (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_label    TEXT NOT NULL UNIQUE,
    entity_type        TEXT,            -- person | place | text | deity | other
    description        TEXT,
    n_mentions         INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_entities_label ON entities(canonical_label);

-- ── Edges from text → abstraction ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS para_concept (
    paragraph_id INTEGER NOT NULL REFERENCES paragraphs(id),
    concept_id   INTEGER NOT NULL REFERENCES concepts(id),
    weight       REAL NOT NULL DEFAULT 1.0,
    relation     TEXT NOT NULL DEFAULT 'discusses',
    PRIMARY KEY (paragraph_id, concept_id, relation)
);
CREATE INDEX IF NOT EXISTS idx_para_concept_pid ON para_concept(paragraph_id);
CREATE INDEX IF NOT EXISTS idx_para_concept_cid ON para_concept(concept_id);

CREATE TABLE IF NOT EXISTS para_entity (
    paragraph_id INTEGER NOT NULL REFERENCES paragraphs(id),
    entity_id    INTEGER NOT NULL REFERENCES entities(id),
    PRIMARY KEY (paragraph_id, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_para_entity_eid ON para_entity(entity_id);

-- ── Edges from abstraction → abstraction ───────────────────────────────────
-- relation values:
--   'similar'      : embedding cosine ≥ threshold
--   'co-occurs'    : appear together in ≥N paragraphs
CREATE TABLE IF NOT EXISTS concept_edges (
    src_id   INTEGER NOT NULL REFERENCES concepts(id),
    dst_id   INTEGER NOT NULL REFERENCES concepts(id),
    relation TEXT    NOT NULL,
    weight   REAL    NOT NULL DEFAULT 1.0,
    PRIMARY KEY (src_id, dst_id, relation)
);
CREATE INDEX IF NOT EXISTS idx_concept_edges_src ON concept_edges(src_id);
CREATE INDEX IF NOT EXISTS idx_concept_edges_dst ON concept_edges(dst_id);

-- ── Vectors: row-id mapping ────────────────────────────────────────────────
-- The actual float vectors live in `index_data/<kind>_<lang>.faiss`. This
-- table lets us go (node) ↔ (faiss row) without scanning.
CREATE TABLE IF NOT EXISTS embeddings (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    node_kind TEXT NOT NULL,               -- 'sentence'|'paragraph'|'chapter'|'concept'
    node_id   INTEGER NOT NULL,
    lang      TEXT,                        -- NULL for 'concept' (cross-lingual)
    faiss_id  INTEGER NOT NULL,
    UNIQUE(node_kind, node_id, lang)
);
CREATE INDEX IF NOT EXISTS idx_emb_kind_lang_faiss
    ON embeddings(node_kind, lang, faiss_id);
CREATE INDEX IF NOT EXISTS idx_emb_kind_node
    ON embeddings(node_kind, node_id);

-- ── Concept snapshot (warm-start cache) ───────────────────────────────────
-- Survives `01_parse.py --force` (which only wipes the text-hierarchy tables).
-- Lets us re-attach previously-extracted concepts/entities/summary to new
-- paragraph rows whose text matches an old paragraph, so we only re-run the
-- expensive LLM extraction on the genuinely new content. Tagged with the
-- extractor identity so a model/prompt upgrade invalidates stale rows.
CREATE TABLE IF NOT EXISTS concept_snapshot (
    text_norm     TEXT NOT NULL,            -- NFC + collapsed whitespace
    lang          TEXT NOT NULL,            -- 'en' | 'bn'
    extractor_tag TEXT NOT NULL,            -- '<llm_model>@v<prompt_version>'
    summary       TEXT,
    concepts_json TEXT NOT NULL,            -- JSON array of {canonical_label, relation, weight}
    entities_json TEXT NOT NULL,            -- JSON array of {canonical_label, entity_type}
    captured_at   TEXT NOT NULL,            -- ISO ts
    PRIMARY KEY (text_norm, lang, extractor_tag)
);

-- Surface-phrase aliases keyed by canonical concept label, not concept_id.
-- (Concept ids are autoincrement and change across --force re-parse; the
-- canonical label is the stable cross-instance key.) Restored verbatim.
CREATE TABLE IF NOT EXISTS alias_snapshot (
    canonical_label TEXT NOT NULL,
    lang            TEXT NOT NULL,
    alias           TEXT NOT NULL,
    PRIMARY KEY (canonical_label, lang, alias)
);

-- ── Pipeline state (resumability + telemetry) ──────────────────────────────
CREATE TABLE IF NOT EXISTS pipeline_state (
    step         TEXT PRIMARY KEY,         -- 'parse' | 'embed_sentences_en' | ...
    completed_at TEXT,                     -- ISO ts when done; NULL if in flight
    progress     INTEGER,
    total        INTEGER,
    note         TEXT
);

-- Generic key-value store (model name used to build the index, etc.)
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def open_db(db_path: str | Path, create: bool = True) -> sqlite3.Connection:
    """Open a connection. If `create`, ensure the schema exists.

    `check_same_thread=False` so Streamlit's worker threads can share a single
    connection (we keep all writes in the main thread of build scripts)."""
    db_path = Path(db_path)
    if create:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    if create:
        conn.executescript(SCHEMA)
        _migrate(conn)
        conn.commit()
    return conn


# Additive column migrations for DBs created before a column existed.
# `CREATE TABLE IF NOT EXISTS` never alters an existing table, so new columns
# must be added explicitly. Each entry: (table, column, DDL type/default).
_MIGRATIONS = [
    ("paragraphs", "para_id_html", "TEXT NOT NULL DEFAULT ''"),
]


def _migrate(conn: sqlite3.Connection) -> None:
    for table, column, ddl in _MIGRATIONS:
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Cursor]:
    """Wraps work in a single transaction. Rolls back on exception."""
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def mark_step_started(conn: sqlite3.Connection, step: str, total: int = 0) -> None:
    with transaction(conn) as cur:
        cur.execute(
            "INSERT INTO pipeline_state(step, completed_at, progress, total) "
            "VALUES (?, NULL, 0, ?) "
            "ON CONFLICT(step) DO UPDATE SET completed_at=NULL, progress=0, total=excluded.total",
            (step, total),
        )


def mark_step_progress(conn: sqlite3.Connection, step: str, progress: int) -> None:
    with transaction(conn) as cur:
        cur.execute(
            "UPDATE pipeline_state SET progress=? WHERE step=?",
            (progress, step),
        )


def mark_step_done(conn: sqlite3.Connection, step: str, note: str = "") -> None:
    from datetime import datetime, timezone
    with transaction(conn) as cur:
        cur.execute(
            "UPDATE pipeline_state SET completed_at=?, note=? WHERE step=?",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), note, step),
        )


def step_completed(conn: sqlite3.Connection, step: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT completed_at FROM pipeline_state WHERE step=?", (step,))
    row = cur.fetchone()
    return bool(row and row[0])


# ── Snapshot helpers ──────────────────────────────────────────────────────

_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Stable key for matching the same paragraph across re-parses.

    NFC + collapsed whitespace + strip is enough for the source HTMLs we have:
    pure layout edits (extra blanks, line wraps) normalize away, while genuine
    wording changes still produce different keys (and correctly trigger a
    fresh LLM extraction)."""
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def extractor_tag() -> str:
    """`<llm_model>@v<PROMPT_VERSION>` — keys the warm-start cache.

    Bumping the model name in config or the prompt version in
    `concept_extraction.PROMPT_VERSION` invalidates stale snapshot rows."""
    # Imported lazily to avoid a circular dep at module load time.
    from .config import CFG
    from .concept_extraction import PROMPT_VERSION
    return f"{CFG.models.llm}@v{PROMPT_VERSION}"
