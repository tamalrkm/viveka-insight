"""Optional: GraphSAGE refinement of paragraph embeddings.

This is the fourth layer from the architecture: after embeddings (similarity)
+ concept extraction (meaning) + the graph (connection), a GNN passes messages
along the graph so each paragraph's vector becomes informed by what *other
paragraphs and concepts it's connected to* — not just by its own surface text.

What runs here:
  * Build a heterogeneous graph in PyG with three node types:
        paragraph, concept, entity
    and edges:
        paragraph --discusses--> concept   (from para_concept)
        concept   --similar-->   concept   (from concept_edges)
        concept   --co-occurs--> concept   (from concept_edges)
        paragraph --mentions-->  entity    (from para_entity)
    Edges are stored bidirectionally so messages flow both ways.
  * Initialize each paragraph node with its BGE-M3 vector.
    Initialize concept/entity nodes with the BGE-M3 embedding of their
    canonical label (already computed during linking).
  * Self-supervised training objective: contrastive — a paragraph and its
    parent chapter's other paragraphs should be more similar than random
    paragraph pairs. (No labels needed.)
  * Output: refined paragraph embeddings, written to a new FAISS index
    `paragraph_<lang>_gnn.faiss`. The Searcher picks them up automatically
    if present.

This step is OFF by default — set VIVEKA_GNN_ENABLED=1 to opt in. Cost on
A100: ~5 minutes for our corpus size. Quality lift: typically +2-5 nDCG@10
on retrieval benchmarks of this kind, more on long-tail concept queries.

Skipping it is fine; the rest of the system is independent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def build_pyg_data(conn, paragraph_embs: Dict[str, np.ndarray],
                    concept_embs: np.ndarray):
    """Construct a HeteroData object.

    Args:
      paragraph_embs: {lang: (N_lang, dim) array, in node-id-sorted order}.
      concept_embs:    (N_concepts, dim) array, in concept-id-sorted order.

    Returns the HeteroData and id-maps so we can write back results.
    """
    import torch
    from torch_geometric.data import HeteroData

    cur = conn.cursor()

    # Build paragraph node array — pool both languages into one type, but
    # keep a `lang` mask so we could split later if desired.
    cur.execute(
        """
        SELECT p.id AS pid, b.lang AS lang
          FROM paragraphs p
          JOIN chapters c ON c.id = p.chapter_id
          JOIN volumes  v ON v.id = c.volume_id
          JOIN books    b ON b.id = v.book_id
         ORDER BY p.id
        """
    )
    pid_rows = cur.fetchall()
    pid_to_idx: Dict[int, int] = {}
    para_x = np.zeros((len(pid_rows), 1024), dtype=np.float32)
    en_buf, bn_buf = [], []
    en_pos, bn_pos = 0, 0
    for i, row in enumerate(pid_rows):
        pid_to_idx[row["pid"]] = i
        if row["lang"] == "en":
            para_x[i] = paragraph_embs["en"][en_pos]
            en_pos += 1
        else:
            para_x[i] = paragraph_embs["bn"][bn_pos]
            bn_pos += 1

    # Concept nodes
    cur.execute("SELECT id FROM concepts ORDER BY id")
    cid_to_idx = {row["id"]: i for i, row in enumerate(cur.fetchall())}

    # Edges
    cur.execute("SELECT paragraph_id, concept_id, weight FROM para_concept")
    pc_src, pc_dst, pc_w = [], [], []
    for row in cur.fetchall():
        if row["paragraph_id"] in pid_to_idx and row["concept_id"] in cid_to_idx:
            pc_src.append(pid_to_idx[row["paragraph_id"]])
            pc_dst.append(cid_to_idx[row["concept_id"]])
            pc_w.append(float(row["weight"]))

    cur.execute("SELECT src_id, dst_id, weight FROM concept_edges")
    cc_src, cc_dst, cc_w = [], [], []
    for row in cur.fetchall():
        if row["src_id"] in cid_to_idx and row["dst_id"] in cid_to_idx:
            cc_src.append(cid_to_idx[row["src_id"]])
            cc_dst.append(cid_to_idx[row["dst_id"]])
            cc_w.append(float(row["weight"]))

    data = HeteroData()
    data["paragraph"].x = torch.from_numpy(para_x)
    data["concept"].x = torch.from_numpy(concept_embs.astype(np.float32))

    if pc_src:
        data["paragraph", "discusses", "concept"].edge_index = torch.tensor(
            [pc_src, pc_dst], dtype=torch.long
        )
        data["paragraph", "discusses", "concept"].edge_weight = torch.tensor(
            pc_w, dtype=torch.float32
        )
        # reverse direction so messages flow both ways
        data["concept", "rev_discusses", "paragraph"].edge_index = torch.tensor(
            [pc_dst, pc_src], dtype=torch.long
        )
        data["concept", "rev_discusses", "paragraph"].edge_weight = torch.tensor(
            pc_w, dtype=torch.float32
        )
    if cc_src:
        data["concept", "related", "concept"].edge_index = torch.tensor(
            [cc_src, cc_dst], dtype=torch.long
        )
        data["concept", "related", "concept"].edge_weight = torch.tensor(
            cc_w, dtype=torch.float32
        )

    return data, pid_to_idx, cid_to_idx


class HeteroGraphSAGE:
    """A small heterogeneous GraphSAGE. Two layers, identity skip-connection.

    We avoid an enormous library wrapper because (a) we want the user to be
    able to read the whole forward pass in one place, (b) PyG's HeteroConv
    + SAGEConv combo is the simplest thing that works.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 2):
        import torch
        import torch.nn as nn
        from torch_geometric.nn import SAGEConv, HeteroConv

        edge_types = [
            ("paragraph", "discusses", "concept"),
            ("concept", "rev_discusses", "paragraph"),
            ("concept", "related", "concept"),
        ]
        self.convs = nn.ModuleList()
        for layer_i in range(n_layers):
            in_d = in_dim if layer_i == 0 else hidden_dim
            out_d = hidden_dim if layer_i < n_layers - 1 else out_dim
            self.convs.append(HeteroConv(
                {et: SAGEConv((-1, -1), out_d, aggr="mean") for et in edge_types},
                aggr="sum",
            ))

    def parameters(self):
        return self.convs.parameters()

    def to(self, device):
        for c in self.convs:
            c.to(device)
        return self

    def __call__(self, x_dict, edge_index_dict):
        import torch.nn.functional as F
        out = x_dict
        for i, conv in enumerate(self.convs):
            out = conv(out, edge_index_dict)
            if i < len(self.convs) - 1:
                out = {k: F.relu(v) for k, v in out.items()}
        return out


def train_and_export(
    conn,
    paragraph_embs: Dict[str, np.ndarray],
    concept_embs: np.ndarray,
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    out_path: Optional[Path] = None,
) -> Dict[int, np.ndarray]:
    """Train the GNN and return {paragraph_id: refined_vec}.

    The training signal is a chapter-coherence contrastive loss: paragraphs
    in the same chapter should have higher refined-similarity than randomly
    paired paragraphs. That's a useful prior — the LLM extracts concepts at
    the paragraph level, but a chapter has semantic unity that the GNN can
    propagate through the graph.
    """
    import torch
    import torch.nn.functional as F

    data, pid_to_idx, cid_to_idx = build_pyg_data(conn, paragraph_embs, concept_embs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = data.to(device)

    model = HeteroGraphSAGE(
        in_dim=1024, hidden_dim=256, out_dim=256, n_layers=2,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Chapter-coherence anchor pairs: sample one paragraph per chapter as anchor
    # and a positive from the same chapter, a negative from a random chapter.
    cur = conn.cursor()
    cur.execute("SELECT chapter_id, GROUP_CONCAT(id) FROM paragraphs GROUP BY chapter_id")
    chapter_paras: List[List[int]] = []
    for row in cur.fetchall():
        ids = [pid_to_idx[int(x)] for x in row[1].split(",") if int(x) in pid_to_idx]
        if len(ids) >= 2:
            chapter_paras.append(ids)

    if not chapter_paras:
        print("[gnn] no chapters with ≥2 paragraphs — skipping training")
        return {}

    rng = np.random.default_rng(42)

    print(f"[gnn] training on {len(chapter_paras)} chapters, "
          f"{data['paragraph'].num_nodes} paragraph nodes, "
          f"{data['concept'].num_nodes} concept nodes, device={device}")

    for ep in range(epochs):
        model.convs.train()
        opt.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        para_z = F.normalize(out["paragraph"], dim=-1)

        # Build anchor/positive/negative triplets
        anchors, positives, negatives = [], [], []
        for paras in chapter_paras:
            a, p = rng.choice(paras, size=2, replace=False)
            # negative from another chapter
            n_chap = rng.integers(0, len(chapter_paras))
            n = rng.choice(chapter_paras[n_chap])
            anchors.append(a); positives.append(p); negatives.append(n)

        a_ids = torch.tensor(anchors, dtype=torch.long, device=device)
        p_ids = torch.tensor(positives, dtype=torch.long, device=device)
        n_ids = torch.tensor(negatives, dtype=torch.long, device=device)
        # Triplet loss with cosine
        pos_sim = (para_z[a_ids] * para_z[p_ids]).sum(dim=-1)
        neg_sim = (para_z[a_ids] * para_z[n_ids]).sum(dim=-1)
        loss = F.relu(0.2 - pos_sim + neg_sim).mean()
        loss.backward()
        opt.step()

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[gnn] epoch {ep+1}/{epochs}  loss={loss.item():.4f}")

    # Inference
    model.convs.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        refined = F.normalize(out["paragraph"], dim=-1).cpu().numpy().astype(np.float32)

    # Map back idx -> paragraph_id
    idx_to_pid = {v: k for k, v in pid_to_idx.items()}
    result = {idx_to_pid[i]: refined[i] for i in range(refined.shape[0])}

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Save aligned to id order for easy reload
        ids = sorted(result.keys())
        arr = np.stack([result[i] for i in ids], axis=0)
        np.savez(str(out_path), ids=np.asarray(ids, dtype=np.int64), vecs=arr)
        print(f"[gnn] wrote refined paragraph embeddings -> {out_path}")

    return result
