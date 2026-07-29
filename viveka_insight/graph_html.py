"""Data for the interactive concept graph (webapp Concept Graph tab).

Pure and Streamlit-free so it is unit-testable. `build_graph_spec` turns a
concept subgraph payload (from Searcher.concept_graph), a per-node cluster
assignment, a fixed 2-D layout and a selection into the JSON spec consumed by
`webapp/static/graph.js`.

The renderer is a **bidirectional Streamlit custom component** (declared over
`webapp/static/`, which also holds the vendored Cytoscape.js + fcose — no CDN,
CSP-safe). It has to be bidirectional rather than a plain `st.iframe`, because
clicking an edge must send the concept pair back to Python so the server can
query the passages those two concepts share.

Clustering / layout helpers (assign_communities, layout_positions) use
networkx, which is a project dependency.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# Component root. Streamlit serves every file beneath it (traversal blocked),
# so `vendor/*.js` is fetched — and browser-cached — rather than re-inlined
# into a ~1.3 MB HTML string on every rerun, as the old st.iframe build did.
COMPONENT_DIR = Path(__file__).resolve().parent.parent / "webapp" / "static"

# Colorblind-safe categorical palette (Okabe–Ito + a few extras) for communities.
_PALETTE = [
    "#4e79a7", "#59a14f", "#e15759", "#f28e2b", "#b07aa1",
    "#76b7b2", "#edc948", "#9c755f", "#ff9da7", "#bab0ac",
]
# Fixed, legible colors for the four yogas.
_YOGA_COLORS = {
    "jnana": "#4e79a7",   # blue  — knowledge
    "bhakti": "#e15759",  # red   — devotion
    "karma": "#59a14f",   # green — action
    "raja": "#b07aa1",    # purple— meditation
    "other": "#bab0ac",   # grey
}
_YOGA_LABEL = {"jnana": "Jñāna (knowledge)", "bhakti": "Bhakti (devotion)",
               "karma": "Karma (action)", "raja": "Rāja (meditation)",
               "other": "Other"}


# ──────────────────────────────────────────────────────────────────────────
# Clustering + layout (networkx)
# ──────────────────────────────────────────────────────────────────────────

def _nx_graph(nodes, edges):
    import networkx as nx
    G = nx.Graph()
    for n in nodes:
        G.add_node(n["id"])
    for e in edges:
        if e["source"] in G and e["target"] in G:
            # accumulate weight if a pair has both relations
            if G.has_edge(e["source"], e["target"]):
                G[e["source"]][e["target"]]["weight"] += float(e["weight"])
            else:
                G.add_edge(e["source"], e["target"], weight=float(e["weight"]))
    return G


def assign_communities(
    nodes: Sequence[dict], edges: Sequence[dict], max_groups: int = 10,
) -> Tuple[Dict[int, int], Dict[int, dict]]:
    """Greedy-modularity communities on the induced undirected graph.

    Returns (node_cluster: {node_id: cluster_key},
             cluster_meta: {cluster_key: {"label","color"}}). The largest
    `max_groups` communities keep their own key; smaller ones collapse to a
    shared 'other' bucket (key -1). Each community is labelled by its
    highest-n_mentions member."""
    import networkx as nx
    from networkx.algorithms.community import greedy_modularity_communities

    mentions = {n["id"]: n.get("n_mentions", 0) for n in nodes}
    label_of = {n["id"]: n["label"] for n in nodes}
    G = _nx_graph(nodes, edges)

    comms: List[set] = []
    if G.number_of_edges():
        comms = [set(c) for c in greedy_modularity_communities(G, weight="weight")]
    # nodes with no edges won't appear; add singletons so every node is covered
    covered = set().union(*comms) if comms else set()
    for n in nodes:
        if n["id"] not in covered:
            comms.append({n["id"]})
    comms.sort(key=len, reverse=True)

    node_cluster: Dict[int, int] = {}
    cluster_meta: Dict[int, dict] = {}
    for i, c in enumerate(comms):
        if i < max_groups:
            key = i
            top = max(c, key=lambda nid: mentions.get(nid, 0))
            cluster_meta[key] = {"label": label_of.get(top, f"cluster {i+1}"),
                                 "color": _PALETTE[i % len(_PALETTE)]}
        else:
            key = -1
        for nid in c:
            node_cluster[nid] = key
    if any(v == -1 for v in node_cluster.values()):
        cluster_meta[-1] = {"label": "other", "color": "#bab0ac"}
    return node_cluster, cluster_meta


def yoga_cluster_meta(node_yoga: Dict[int, str]) -> Dict[str, dict]:
    """cluster_meta for a yoga assignment, only for yogas that occur."""
    present = []
    for y in ("jnana", "bhakti", "karma", "raja", "other"):
        if any(v == y for v in node_yoga.values()):
            present.append(y)
    return {y: {"label": _YOGA_LABEL[y], "color": _YOGA_COLORS[y]} for y in present}


def layout_positions(
    nodes: Sequence[dict], edges: Sequence[dict], seed: int = 7, scale: float = 520.0,
) -> Dict[int, Tuple[float, float]]:
    """Deterministic 2-D spring layout (fixed seed → stable across reruns).
    Scaled to canvas-ish coordinates for a Cytoscape `preset` layout."""
    import networkx as nx
    G = _nx_graph(nodes, edges)
    if G.number_of_nodes() == 0:
        return {}
    pos = nx.spring_layout(G, seed=seed, weight="weight",
                           k=1.6 / (G.number_of_nodes() ** 0.5 or 1))
    return {nid: (float(x) * scale, float(y) * scale) for nid, (x, y) in pos.items()}


# ──────────────────────────────────────────────────────────────────────────
# Spec builder (consumed by webapp/static/graph.js)
# ──────────────────────────────────────────────────────────────────────────

def filter_payload(payload: dict, visible_ids: Sequence[int]) -> dict:
    """Induced subgraph on `visible_ids`. Node order and edge shape preserved,
    so the caller's cluster/position dicts (keyed by node id) stay valid."""
    keep = set(int(i) for i in visible_ids)
    nodes = [n for n in payload["nodes"] if n["id"] in keep]
    edges = [e for e in payload["edges"]
             if e["source"] in keep and e["target"] in keep]
    return {"nodes": nodes, "edges": edges}


def build_graph_spec(
    payload: dict,
    node_cluster: Dict[int, object],
    cluster_meta: Dict[object, dict],
    positions: Dict[int, Tuple[float, float]],
    selected_ids: Sequence[int] = (),
    *,
    cooc_counts: Dict[int, int] | None = None,
    max_mentions: int | None = None,
    relayout: bool = False,
) -> dict:
    """JSON spec for the graph component: cytoscape elements, the grouped
    right-hand index, the legend, and render flags.

    `cooc_counts` marks nodes pulled in because they co-occur with the
    selection, mapping each to the number of paragraphs it shares with it.
    `max_mentions` pins the node-size scale (pass the *unfiltered* maximum so
    sizes don't rescale when the graph is filtered down to a few nodes).
    `relayout` tells the client to re-run its layout over the nodes actually
    shown; set it whenever the payload is a filtered subgraph, since the
    full-graph coordinates leave a handful of survivors bunched together."""
    nodes, edges = payload["nodes"], payload["edges"]
    sel = set(int(s) for s in selected_ids)
    cooc = {int(k): int(v) for k, v in (cooc_counts or {}).items()}

    def color(nid):
        return cluster_meta.get(node_cluster.get(nid), {}).get("color", "#999")

    mmax = max_mentions or max((n["n_mentions"] for n in nodes), default=1) or 1
    cy_nodes = []
    for n in nodes:
        nid = n["id"]
        x, y = positions.get(nid, (0.0, 0.0))
        size = 10 + 34 * (min(n["n_mentions"], mmax) / mmax) ** 0.5
        cy_nodes.append({"data": {
            "id": str(nid), "label": n["label"], "color": color(nid),
            "size": round(size, 1), "n": n["n_mentions"],
            "cluster": str(node_cluster.get(nid, "")),
            "aliases_en": ", ".join(n.get("aliases_en", [])),
            "aliases_bn": ", ".join(n.get("aliases_bn", [])),
            "sel": 1 if nid in sel else 0,
            "cooc": cooc.get(nid, 0),
        }, "position": {"x": x, "y": y}})
    cy_edges = [{"data": {
        "id": f"{e['source']}_{e['target']}_{e['relation']}",
        "source": str(e["source"]), "target": str(e["target"]),
        "relation": e["relation"], "weight": e["weight"],
    }} for e in edges]

    # right-side index grouped by cluster (ordered by cluster then n_mentions)
    order = sorted(cluster_meta.keys(),
                   key=lambda k: -sum(1 for v in node_cluster.values() if v == k))
    groups = []
    for k in order:
        members = sorted([n for n in nodes if node_cluster.get(n["id"]) == k],
                         key=lambda n: -n["n_mentions"])
        if not members:
            continue
        meta = cluster_meta[k]
        groups.append({"label": meta["label"], "color": meta["color"],
                       "items": [{"id": str(m["id"]), "label": m["label"],
                                  "n": m["n_mentions"],
                                  "sel": 1 if m["id"] in sel else 0,
                                  "cooc": cooc.get(m["id"], 0)} for m in members]})

    # Only clusters with a visible member — a filtered graph must not advertise
    # colours it no longer draws.
    legend = [{"label": g["label"], "color": g["color"]} for g in groups]

    return {"nodes": cy_nodes, "edges": cy_edges, "groups": groups,
            "legend": legend, "has_sel": bool(sel), "has_cooc": bool(cooc),
            "relayout": bool(relayout)}
