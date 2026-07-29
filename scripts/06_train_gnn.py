"""Stage 6 (OPTIONAL): train the heterogeneous GraphSAGE on the assembled graph.

Off by default. Turn on with VIVEKA_GNN_ENABLED=1 or `--enable`.

What it does:
  * loads paragraph + concept embeddings from the existing FAISS files,
  * builds a HeteroData graph (paragraph, concept node types; discusses,
    rev_discusses, related edges),
  * trains a 2-layer SAGEConv with chapter-coherence triplet loss,
  * writes the refined paragraph vectors to `index_data/gnn_paragraphs.npz`.

The Searcher will pick these up automatically on startup if the file exists.

Skipping this is fine — system works end-to-end without it. The GNN is a
quality-bump experiment, not a requirement.
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
from viveka_insight.config import CFG


STEP = "train_gnn"


def _load_paragraph_embs(conn) -> dict:
    """Read the paragraph FAISS indices into ndarrays per language."""
    out = {}
    for lang in ("en", "bn"):
        path = CFG.paths.faiss("paragraph", lang)
        if not path.exists():
            print(f"[gnn] missing {path}; skipping {lang}")
            continue
        idx = emb.read_index(path)
        n = idx.ntotal
        if n == 0:
            continue
        # Pull all vectors (FAISS Flat exposes .reconstruct_n)
        vecs = np.zeros((n, idx.d), dtype=np.float32)
        for i in range(n):
            vecs[i] = idx.reconstruct(i)
        # Re-order to paragraph-id order using the embeddings table
        cur = conn.cursor()
        cur.execute(
            "SELECT node_id, faiss_id FROM embeddings "
            "WHERE node_kind='paragraph' AND lang=? ORDER BY node_id",
            (lang,),
        )
        rows = cur.fetchall()
        # produce ordered array aligned with the SQL query order
        ordered = np.zeros((len(rows), idx.d), dtype=np.float32)
        for i, row in enumerate(rows):
            ordered[i] = vecs[row["faiss_id"]]
        out[lang] = ordered
    return out


def _load_concept_embs(conn) -> np.ndarray:
    path = CFG.paths.faiss("concept", None)
    if not path.exists():
        return np.zeros((0, CFG.embedding.dim), dtype=np.float32)
    idx = emb.read_index(path)
    n = idx.ntotal
    if n == 0:
        return np.zeros((0, idx.d), dtype=np.float32)
    vecs = np.zeros((n, idx.d), dtype=np.float32)
    for i in range(n):
        vecs[i] = idx.reconstruct(i)
    # reorder to id-sorted
    cur = conn.cursor()
    cur.execute(
        "SELECT node_id, faiss_id FROM embeddings "
        "WHERE node_kind='concept' ORDER BY node_id"
    )
    rows = cur.fetchall()
    ordered = np.zeros((len(rows), idx.d), dtype=np.float32)
    for i, row in enumerate(rows):
        ordered[i] = vecs[row["faiss_id"]]
    return ordered


def main():
    ap = argparse.ArgumentParser(description="Train HeteroGraphSAGE refinement")
    ap.add_argument("--db", default=str(CFG.paths.db))
    ap.add_argument("--enable", action="store_true",
                    help="run even if VIVEKA_GNN_ENABLED is unset")
    ap.add_argument("--epochs", type=int, default=CFG.gnn.epochs)
    args = ap.parse_args()

    if not (CFG.gnn.enabled or args.enable):
        print("[gnn] disabled (set VIVEKA_GNN_ENABLED=1 or pass --enable). Skipping.")
        return

    try:
        from viveka_insight.gnn import train_and_export
    except ImportError as e:
        print(f"[gnn] missing dependency ({e}); install torch_geometric to use this step.")
        return

    conn = dbmod.open_db(args.db, create=False)

    print("[gnn] loading paragraph embeddings...")
    para_embs = _load_paragraph_embs(conn)
    if not para_embs:
        print("[gnn] no paragraph embeddings on disk — run 02_embed.py first")
        return

    print("[gnn] loading concept embeddings...")
    concept_embs = _load_concept_embs(conn)
    if concept_embs.shape[0] == 0:
        print("[gnn] no concept embeddings — run 04_link_concepts.py first")
        return

    dbmod.mark_step_started(conn, STEP)
    t0 = time.time()
    out_path = CFG.paths.gnn_emb("paragraphs")
    refined = train_and_export(
        conn, para_embs, concept_embs,
        epochs=args.epochs,
        lr=CFG.gnn.lr,
        weight_decay=CFG.gnn.weight_decay,
        out_path=out_path,
    )
    dbmod.mark_step_done(conn, STEP,
                         note=f"refined {len(refined)} paragraphs, "
                              f"{time.time()-t0:.0f}s")
    print(f"[gnn] done in {time.time()-t0:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
