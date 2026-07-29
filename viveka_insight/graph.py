"""Graph traversal helpers.

Used by the searcher to walk from concept nodes outward to text nodes.
Pure SQL — no networkx dependency at query time. (We do load into networkx
for the GNN step; that lives in `gnn.py`.)
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple


def concept_neighbors(
    conn: sqlite3.Connection,
    concept_ids: Sequence[int],
    *,
    relations: Sequence[str] = ("similar", "co-occurs"),
    top_k: int = 8,
    min_weight: float = 0.0,
) -> Dict[int, List[Tuple[int, float, str]]]:
    """For each input concept, return [(neighbor_id, weight, relation), ...]
    sorted by weight desc, capped at top_k. Self-loops omitted.
    """
    if not concept_ids:
        return {}
    placeholders = ",".join("?" * len(concept_ids))
    rel_placeholders = ",".join("?" * len(relations))
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT src_id, dst_id, weight, relation
          FROM concept_edges
         WHERE src_id IN ({placeholders})
           AND relation IN ({rel_placeholders})
           AND weight >= ?
        """,
        (*concept_ids, *relations, min_weight),
    )
    out: Dict[int, List[Tuple[int, float, str]]] = defaultdict(list)
    for row in cur.fetchall():
        out[row[0]].append((row[1], float(row[2]), row[3]))
    # sort + truncate
    for k in list(out.keys()):
        out[k].sort(key=lambda t: -t[1])
        out[k] = out[k][:top_k]
    return out


def paragraphs_for_concepts(
    conn: sqlite3.Connection,
    concept_ids: Sequence[int],
    *,
    lang: Optional[str] = None,
    per_concept: int = 5,
) -> Dict[int, List[Tuple[int, float, str]]]:
    """For each concept, return up to `per_concept` paragraphs that link to it.

    Returns {concept_id: [(paragraph_id, edge_weight, relation), ...]}.
    Filtered by language if specified.
    """
    if not concept_ids:
        return {}
    out: Dict[int, List[Tuple[int, float, str]]] = {}
    cur = conn.cursor()
    for cid in concept_ids:
        if lang:
            cur.execute(
                """
                SELECT pc.paragraph_id, pc.weight, pc.relation
                  FROM para_concept pc
                  JOIN paragraphs p   ON p.id  = pc.paragraph_id
                  JOIN chapters  c    ON c.id  = p.chapter_id
                  JOIN volumes   v    ON v.id  = c.volume_id
                  JOIN books     b    ON b.id  = v.book_id
                 WHERE pc.concept_id = ? AND b.lang = ?
                 ORDER BY pc.weight DESC
                 LIMIT ?
                """,
                (cid, lang, per_concept),
            )
        else:
            cur.execute(
                """
                SELECT paragraph_id, weight, relation
                  FROM para_concept
                 WHERE concept_id = ?
                 ORDER BY weight DESC
                 LIMIT ?
                """,
                (cid, per_concept),
            )
        out[cid] = [(int(r[0]), float(r[1]), r[2]) for r in cur.fetchall()]
    return out


def fetch_paragraph_metadata(
    conn: sqlite3.Connection,
    paragraph_ids: Sequence[int],
) -> Dict[int, dict]:
    """Bulk-fetch the rich location/text record for a list of paragraph ids."""
    if not paragraph_ids:
        return {}
    placeholders = ",".join("?" * len(paragraph_ids))
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT p.id            AS paragraph_id,
               p.text          AS text,
               p.summary       AS summary,
               p.paragraph_idx AS paragraph_idx,
               p.char_offset   AS char_offset,
               p.para_id_html  AS para_id_html,
               c.id            AS chapter_id,
               c.title         AS chapter_title,
               c.section       AS section,
               c.chapter_id_html AS chapter_id_html,
               v.title         AS volume_title,
               v.volume_num    AS volume_num,
               b.lang          AS lang,
               b.title         AS book_title
          FROM paragraphs p
          JOIN chapters  c ON c.id = p.chapter_id
          JOIN volumes   v ON v.id = c.volume_id
          JOIN books     b ON b.id = v.book_id
         WHERE p.id IN ({placeholders})
        """,
        tuple(paragraph_ids),
    )
    return {row["paragraph_id"]: dict(row) for row in cur.fetchall()}


def fetch_concept(conn: sqlite3.Connection, concept_id: int) -> Optional[dict]:
    cur = conn.cursor()
    cur.execute(
        "SELECT id, canonical_label, description, n_mentions FROM concepts WHERE id=?",
        (concept_id,),
    )
    row = cur.fetchone()
    return dict(row) if row else None


def fetch_concepts_by_label(
    conn: sqlite3.Connection,
    labels: Sequence[str],
) -> Dict[str, int]:
    """Map canonical labels (lowercase, hyphenated) to ids. Misses are absent."""
    if not labels:
        return {}
    placeholders = ",".join("?" * len(labels))
    cur = conn.cursor()
    cur.execute(
        f"SELECT id, canonical_label FROM concepts WHERE canonical_label IN ({placeholders})",
        tuple(labels),
    )
    return {row["canonical_label"]: row["id"] for row in cur.fetchall()}


def concepts_for_paragraph(
    conn: sqlite3.Connection, paragraph_id: int
) -> List[Tuple[int, str, float, str]]:
    """[(concept_id, label, weight, relation), ...] for a paragraph."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT c.id, c.canonical_label, pc.weight, pc.relation
          FROM para_concept pc JOIN concepts c ON c.id = pc.concept_id
         WHERE pc.paragraph_id = ?
         ORDER BY pc.weight DESC
        """,
        (paragraph_id,),
    )
    return [(int(r[0]), r[1], float(r[2]), r[3]) for r in cur.fetchall()]
