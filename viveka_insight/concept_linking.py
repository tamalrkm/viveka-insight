"""Concept linking — build the abstraction-layer graph.

Two operations:

  1. MERGE near-duplicate concept labels.
     The LLM is consistent enough that 90%+ of duplicates are already byte-equal
     after canonical normalization (e.g. both yield "self-realization"). But
     there's still noise:  "non-attachment" vs "non attachment" vs "detachment",
     "renunciation" vs "renouncement". We embed every canonical_label with the
     same embedder used elsewhere, cluster ones whose cosine ≥ threshold, and
     keep the most frequent label of each cluster as the survivor. All edges
     and aliases of the merged-away concepts are re-pointed to the survivor.

  2. BUILD concept-concept edges.
     Two relations:
       'similar'    — embedding cosine ≥ threshold, top-K per concept.
                       This captures conceptual proximity ("compassion" ↔ "kindness")
                       even when they never co-occur in the corpus.
       'co-occurs'  — both concepts appear in the same paragraph at least N times.
                       This captures statistical association: in Vivekananda,
                       "renunciation" co-occurs with "service" much more than
                       chance — that's signal worth preserving as an edge.

After this step the graph is "closed" and queryable. The GNN layer (optional)
runs *on top* of this graph and produces refined embeddings; it does not
modify the graph itself.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────
# Step 1: merge duplicates
# ──────────────────────────────────────────────────────────────────────────

def find_merge_groups(
    labels: List[str],
    embeddings: np.ndarray,
    threshold: float = 0.92,
) -> List[List[int]]:
    """Greedy clustering by cosine similarity.

    Returns a list of clusters, each a list of indices into `labels` that
    should be merged together. A label that has no near-duplicate appears as
    a singleton cluster. The first index in each cluster is the "survivor"
    (chosen by caller).

    Greedy is fine here: with ~5K concepts, even O(n²) is 25M comparisons —
    one pass on GPU FAISS is sub-second. We avoid hierarchical clustering
    because it can produce drift (A~B, B~C, but A!~C) which would lose
    distinctions like 'devotion' vs 'love' that we want to keep separate.
    """
    n = len(labels)
    if n == 0:
        return []
    # Normalize for cosine via dot
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embs = embeddings / np.maximum(norms, 1e-12)
    sims = embs @ embs.T              # (n, n)
    np.fill_diagonal(sims, -1.0)      # don't merge with self

    visited = [False] * n
    clusters: List[List[int]] = []
    # Iterate by descending degree of "merge-able neighbors" so high-density
    # nodes anchor their clusters.
    counts = (sims >= threshold).sum(axis=1)
    order = np.argsort(-counts)

    for i in order:
        if visited[i]:
            continue
        # Pull all unvisited neighbors above threshold (single-link, but bounded
        # by single-pass to avoid the chaining issue mentioned above).
        members = [int(i)]
        visited[i] = True
        neighbors = np.where(sims[i] >= threshold)[0]
        for j in neighbors:
            if not visited[j]:
                members.append(int(j))
                visited[j] = True
        clusters.append(members)

    return clusters


def pick_survivor(
    cluster_indices: List[int],
    labels: List[str],
    counts: Dict[str, int],
) -> int:
    """Survivor heuristic, in priority order:
      1. highest mention-count in the corpus (it's the most-used phrasing)
      2. shortest label (less likely to be a verbose one-off)
      3. alphabetic first (deterministic tie-break)
    """
    def key(i):
        lbl = labels[i]
        return (
            -counts.get(lbl, 0),
            len(lbl),
            lbl,
        )
    return min(cluster_indices, key=key)


def merge_concepts_in_db(
    conn,
    *,
    embedder,
    threshold: float = 0.92,
) -> Dict[int, int]:
    """Find near-duplicate concept rows in `concepts` and merge them.

    Returns a dict {old_id: new_id} for everything that was remapped.
    Updates `para_concept` and `concept_aliases` to point at survivors,
    deletes the absorbed rows.
    """
    cur = conn.cursor()
    cur.execute("SELECT id, canonical_label, n_mentions FROM concepts")
    rows = cur.fetchall()
    if len(rows) < 2:
        return {}

    ids = [r["id"] for r in rows]
    labels = [r["canonical_label"] for r in rows]
    counts = {r["canonical_label"]: r["n_mentions"] or 0 for r in rows}

    print(f"[link] embedding {len(labels)} concept labels for clustering...")
    embs = embedder.encode(labels, batch_size=256, show_progress=False)

    clusters = find_merge_groups(labels, embs, threshold=threshold)
    n_merged = 0
    remap: Dict[int, int] = {}

    for cluster in clusters:
        if len(cluster) < 2:
            continue
        survivor_idx = pick_survivor(cluster, labels, counts)
        survivor_id = ids[survivor_idx]
        for j in cluster:
            if j == survivor_idx:
                continue
            old_id = ids[j]
            remap[old_id] = survivor_id
            n_merged += 1

    if not remap:
        print("[link] no duplicate concepts to merge.")
        return {}

    # Apply the remap inside a transaction. We do it in three steps so unique
    # constraints don't fire mid-way.
    print(f"[link] merging {n_merged} duplicate concepts into "
          f"{len(set(remap.values()))} survivors")
    cur.execute("BEGIN")
    try:
        # 1. Re-point para_concept rows.  ON CONFLICT preserve max weight.
        for old_id, new_id in remap.items():
            cur.execute(
                """
                INSERT INTO para_concept(paragraph_id, concept_id, relation, weight)
                SELECT paragraph_id, ?, relation, weight
                  FROM para_concept WHERE concept_id = ?
                ON CONFLICT(paragraph_id, concept_id, relation)
                DO UPDATE SET weight = MAX(weight, excluded.weight)
                """,
                (new_id, old_id),
            )
            cur.execute("DELETE FROM para_concept WHERE concept_id = ?", (old_id,))
        # 2. Move aliases to the survivor.
        for old_id, new_id in remap.items():
            cur.execute(
                """
                INSERT OR IGNORE INTO concept_aliases(concept_id, lang, alias)
                SELECT ?, lang, alias FROM concept_aliases WHERE concept_id = ?
                """,
                (new_id, old_id),
            )
            cur.execute("DELETE FROM concept_aliases WHERE concept_id = ?", (old_id,))
            # The dropped label itself becomes an alias of the survivor.
            cur.execute("SELECT canonical_label FROM concepts WHERE id = ?", (old_id,))
            row = cur.fetchone()
            if row:
                cur.execute(
                    "INSERT OR IGNORE INTO concept_aliases(concept_id, lang, alias) "
                    "VALUES (?, 'en', ?)",
                    (new_id, row[0]),
                )
        # 3. Delete absorbed concepts; recompute n_mentions for survivors.
        for old_id in remap:
            cur.execute("DELETE FROM concepts WHERE id = ?", (old_id,))
        # Recount mentions
        cur.execute(
            """
            UPDATE concepts SET n_mentions = (
                SELECT COUNT(*) FROM para_concept WHERE concept_id = concepts.id
            )
            """
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return remap


# ──────────────────────────────────────────────────────────────────────────
# Step 2: similarity edges (concept ↔ concept by embedding)
# ──────────────────────────────────────────────────────────────────────────

def build_similarity_edges(
    conn,
    embedder,
    *,
    top_k: int = 12,
    threshold: float = 0.78,
) -> int:
    """For each concept, link to its top-K most similar other concepts whose
    cosine is above threshold. Symmetric — we insert (a,b) and (b,a) as two
    directed edges so both lookups index-hit.

    Returns the number of edges inserted.
    """
    cur = conn.cursor()
    cur.execute("SELECT id, canonical_label FROM concepts ORDER BY id")
    rows = cur.fetchall()
    n = len(rows)
    if n < 2:
        return 0

    ids = [r["id"] for r in rows]
    labels = [r["canonical_label"] for r in rows]

    print(f"[link] embedding {n} concepts for similarity edges...")
    embs = embedder.encode(labels, batch_size=256, show_progress=False)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.maximum(norms, 1e-12)
    sims = embs @ embs.T
    np.fill_diagonal(sims, -1.0)

    edges: List[Tuple[int, int, float]] = []
    for i in range(n):
        # top-K above threshold
        idx = np.argpartition(-sims[i], min(top_k + 1, n - 1))[:top_k + 1]
        for j in idx:
            if j == i:
                continue
            s = float(sims[i, j])
            if s < threshold:
                continue
            edges.append((ids[i], ids[int(j)], s))

    if not edges:
        return 0

    cur.executemany(
        "INSERT OR REPLACE INTO concept_edges(src_id, dst_id, relation, weight) "
        "VALUES (?, ?, 'similar', ?)",
        edges,
    )
    conn.commit()
    print(f"[link] wrote {len(edges)} 'similar' concept edges")
    return len(edges)


# ──────────────────────────────────────────────────────────────────────────
# Step 3: co-occurrence edges (concept ↔ concept by paragraph statistics)
# ──────────────────────────────────────────────────────────────────────────

def build_cooccurrence_edges(conn, *, min_count: int = 2) -> int:
    """Add an edge between concepts that appear together in the same paragraph
    at least `min_count` times. Edge weight = log1p(count) / max(log1p(count)).
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT paragraph_id, GROUP_CONCAT(concept_id) AS cids
        FROM para_concept
        GROUP BY paragraph_id
        HAVING COUNT(*) >= 2
    """)
    pair_counts: Counter = Counter()
    for row in cur.fetchall():
        cids = sorted({int(x) for x in row["cids"].split(",") if x})
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                pair_counts[(cids[i], cids[j])] += 1

    if not pair_counts:
        return 0

    max_log = max((float(np.log1p(c)) for c in pair_counts.values()), default=1.0)
    edges = []
    for (a, b), c in pair_counts.items():
        if c < min_count:
            continue
        w = float(np.log1p(c)) / max_log
        # Both directions
        edges.append((a, b, w))
        edges.append((b, a, w))

    if not edges:
        return 0

    cur.executemany(
        """
        INSERT INTO concept_edges(src_id, dst_id, relation, weight)
        VALUES (?, ?, 'co-occurs', ?)
        ON CONFLICT(src_id, dst_id, relation)
        DO UPDATE SET weight = MAX(weight, excluded.weight)
        """,
        edges,
    )
    conn.commit()
    print(f"[link] wrote {len(edges)} 'co-occurs' concept edges")
    return len(edges)
