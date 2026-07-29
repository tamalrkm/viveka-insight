"""Concept-graph tab tests — fast, no GPU, no model load.

Covers the data methods (Searcher.concept_graph / example_paragraphs, built
against a tiny in-memory-style DB) and the pure HTML/layout helpers in
graph_html. The Searcher is constructed against a temp meta.sqlite; the
embedder is lazy so no model loads as long as we never call .search().
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from viveka_insight import db as dbmod
from viveka_insight import graph_html as gh
from viveka_insight.search import Searcher


# ──────────────────────────────────────────────────────────────────────────
# Fixture: a tiny concept graph in a temp DB
# ──────────────────────────────────────────────────────────────────────────

@pytest.fixture
def searcher(tmp_path):
    db = tmp_path / "meta.sqlite"
    conn = dbmod.open_db(db, create=True)
    cur = conn.cursor()

    def book(lang):
        cur.execute("INSERT INTO books(lang,title) VALUES (?,?)", (lang, lang))
        return cur.lastrowid
    en, bn = book("en"), book("bn")

    def chapter(book_id, lang):
        cur.execute("INSERT INTO volumes(book_id,title,volume_num) VALUES (?,?,1)",
                    (book_id, "Vol I"))
        v = cur.lastrowid
        cur.execute("INSERT INTO chapters(volume_id,title,ord,chapter_id_html) "
                    "VALUES (?,?,0,?)", (v, "Ch", "content-1"))
        return cur.lastrowid
    ch_en, ch_bn = chapter(en, "en"), chapter(bn, "bn")

    def para(ch, idx, text, pid_html):
        cur.execute("INSERT INTO paragraphs(chapter_id,paragraph_idx,text,char_offset,para_id_html) "
                    "VALUES (?,?,?,0,?)", (ch, idx, text, pid_html))
        return cur.lastrowid
    p1 = para(ch_en, 0, "P1 has A B C", "p-1")   # en
    p2 = para(ch_bn, 0, "P2 has A B", "p-2")     # bn
    p3 = para(ch_en, 1, "P3 has A", "p-3")       # en

    def concept(label, n):
        cur.execute("INSERT INTO concepts(canonical_label,n_mentions) VALUES (?,?)",
                    (label, n))
        return cur.lastrowid
    A, B, C, D = concept("a", 10), concept("b", 8), concept("c", 5), concept("d", 2)

    cur.execute("INSERT INTO concept_aliases(concept_id,lang,alias) VALUES (?,?,?)",
                (A, "bn", "ক"))
    cur.execute("INSERT INTO concept_aliases(concept_id,lang,alias) VALUES (?,?,?)",
                (A, "en", "the-a"))

    def link(p, c):
        cur.execute("INSERT INTO para_concept(paragraph_id,concept_id,weight) VALUES (?,?,1.0)",
                    (p, c))
    for c in (A, B, C): link(p1, c)
    for c in (A, B): link(p2, c)
    link(p3, A)

    def edge(s, d, rel, w):
        cur.execute("INSERT INTO concept_edges(src_id,dst_id,relation,weight) VALUES (?,?,?,?)",
                    (s, d, rel, w))
    edge(A, B, "co-occurs", 4.0); edge(B, A, "co-occurs", 4.0)   # both dirs
    edge(A, C, "co-occurs", 2.0)                                 # one dir
    edge(B, C, "similar", 0.9)
    edge(C, D, "similar", 0.8)
    conn.commit()
    conn.close()

    s = Searcher(index_dir=tmp_path, device="cpu", load_reranker=False)
    s._ids = {"A": A, "B": B, "C": C, "D": D, "p1": p1, "p2": p2, "p3": p3}
    return s


# ──────────────────────────────────────────────────────────────────────────
# concept_graph
# ──────────────────────────────────────────────────────────────────────────

def test_concept_graph_nodes_sorted_and_aliased(searcher):
    g = searcher.concept_graph(n=10)
    labels = [n["label"] for n in g["nodes"]]
    assert labels == ["a", "b", "c", "d"]          # by n_mentions desc
    a = g["nodes"][0]
    assert a["aliases_bn"] == ["ক"] and a["aliases_en"] == ["the-a"]


def test_concept_graph_edges_undirected_and_normalized(searcher):
    g = searcher.concept_graph(n=10)
    pairs = {(min(e["source"], e["target"]), max(e["source"], e["target"]),
              e["relation"]): e["weight"] for e in g["edges"]}
    # A-B co-occurs deduped to a single undirected edge
    assert len(g["edges"]) == 4
    co = sorted(e["weight"] for e in g["edges"] if e["relation"] == "co-occurs")
    sim = sorted(e["weight"] for e in g["edges"] if e["relation"] == "similar")
    assert co == [0.0, 1.0] and sim == [0.0, 1.0]   # per-relation min-max


def test_concept_graph_respects_n(searcher):
    g = searcher.concept_graph(n=2)
    assert [n["label"] for n in g["nodes"]] == ["a", "b"]
    # edges only among the kept nodes
    kept = {n["id"] for n in g["nodes"]}
    assert all(e["source"] in kept and e["target"] in kept for e in g["edges"])


# ──────────────────────────────────────────────────────────────────────────
# example_paragraphs
# ──────────────────────────────────────────────────────────────────────────

def test_example_paragraphs_ranks_by_coverage(searcher):
    ids = searcher._ids
    ex = searcher.example_paragraphs([ids["A"], ids["B"], ids["C"]], k=5)
    assert [e["n_matched"] for e in ex] == [3, 2, 1]         # P1, P2, P3
    assert ex[0]["matched_concepts"] == ["a", "b", "c"]


def test_example_paragraphs_best_partial_when_over_five(searcher):
    ids = searcher._ids
    # select all four incl. D (which has no paragraph) — no full match exists
    ex = searcher.example_paragraphs([ids["A"], ids["B"], ids["C"], ids["D"]], k=5)
    assert ex and ex[0]["n_matched"] == 3 < 4               # best-covering, not empty


def test_example_paragraphs_lang_and_k(searcher):
    ids = searcher._ids
    en = searcher.example_paragraphs([ids["A"], ids["B"]], lang=("en",), k=5)
    assert all(e["lang"] == "en" for e in en)               # P2 (bn) excluded
    assert {e["text"][:2] for e in en} <= {"P1", "P3"}
    k1 = searcher.example_paragraphs([ids["A"]], k=1)
    assert len(k1) == 1


def test_example_paragraphs_empty_selection(searcher):
    assert searcher.example_paragraphs([]) == []


def test_example_paragraphs_require_all_is_a_pair_filter(searcher):
    """What an edge click needs: only paragraphs carrying *both* concepts, never
    a passage about one half of the pair."""
    ids = searcher._ids
    # A∈{p1,p2,p3}, B∈{p1,p2}, C∈{p1}
    both = searcher.example_paragraphs([ids["A"], ids["B"]], k=10, require_all=True)
    assert {e["text"][:2] for e in both} == {"P1", "P2"}    # P3 (A only) excluded
    assert all(e["n_matched"] == 2 for e in both)
    # without it, the A-only paragraph rides along at the bottom
    loose = searcher.example_paragraphs([ids["A"], ids["B"]], k=10)
    assert "P3" in {e["text"][:2] for e in loose}
    # a pair that never shares a paragraph yields nothing rather than a partial
    assert searcher.example_paragraphs([ids["C"], ids["D"]], k=10, require_all=True) == []
    # duplicate ids must not inflate the required count
    dup = searcher.example_paragraphs([ids["A"], ids["A"]], k=10, require_all=True)
    assert {e["text"][:2] for e in dup} == {"P1", "P2", "P3"}


# ──────────────────────────────────────────────────────────────────────────
# cooccurring_concepts
# ──────────────────────────────────────────────────────────────────────────

def test_cooccurring_counts_shared_paragraphs(searcher):
    ids = searcher._ids
    # A is in p1,p2,p3; B in p1,p2; C in p1. So A↔B share 2, A↔C share 1.
    co = searcher.cooccurring_concepts([ids["A"]])
    assert co == {ids["B"]: 2, ids["C"]: 1}          # A itself excluded


def test_cooccurring_min_count_and_max_over_selection(searcher):
    ids = searcher._ids
    assert searcher.cooccurring_concepts([ids["A"]], min_count=2) == {ids["B"]: 2}
    # C shares 1 para with A and 1 with B → max is 1, still below threshold
    assert searcher.cooccurring_concepts([ids["A"], ids["B"]], min_count=2) == {}
    assert searcher.cooccurring_concepts([ids["A"], ids["B"]]) == {ids["C"]: 1}


def test_cooccurring_restrict_and_empty(searcher):
    ids = searcher._ids
    assert searcher.cooccurring_concepts([ids["A"]], restrict_to=[ids["C"]]) == {ids["C"]: 1}
    assert searcher.cooccurring_concepts([ids["A"]], restrict_to=[]) == {}
    assert searcher.cooccurring_concepts([]) == {}
    assert searcher.cooccurring_concepts([ids["D"]]) == {}   # D is in no paragraph


def test_four_yoga_no_index_is_other(searcher):
    ids = searcher._ids
    # temp DB has no concept.faiss → graceful 'other' for all
    ya = searcher.four_yoga_assignment([ids["A"], ids["B"]])
    assert set(ya.values()) == {"other"}


# ──────────────────────────────────────────────────────────────────────────
# graph_html pure helpers
# ──────────────────────────────────────────────────────────────────────────

def _payload():
    nodes = [{"id": i, "label": f"c{i}", "n_mentions": 10 - i,
              "aliases_en": [], "aliases_bn": []} for i in range(6)]
    edges = [{"source": 0, "target": 1, "relation": "co-occurs", "weight": 1.0},
             {"source": 1, "target": 2, "relation": "co-occurs", "weight": 0.5},
             {"source": 3, "target": 4, "relation": "similar", "weight": 0.8}]
    return {"nodes": nodes, "edges": edges}


def test_assign_communities_covers_all_nodes():
    p = _payload()
    nc, cm = gh.assign_communities(p["nodes"], p["edges"])
    assert set(nc) == {n["id"] for n in p["nodes"]}        # every node assigned
    assert all(nc[n["id"]] in cm for n in p["nodes"])      # cluster has meta
    # deterministic
    nc2, _ = gh.assign_communities(p["nodes"], p["edges"])
    assert nc == nc2


def test_layout_positions_deterministic():
    p = _payload()
    a = gh.layout_positions(p["nodes"], p["edges"], seed=7)
    b = gh.layout_positions(p["nodes"], p["edges"], seed=7)
    assert a == b and set(a) == {n["id"] for n in p["nodes"]}


def test_build_graph_spec_lists_every_node_and_marks_selection():
    p = _payload()
    nc, cm = gh.assign_communities(p["nodes"], p["edges"])
    pos = gh.layout_positions(p["nodes"], p["edges"])
    spec = gh.build_graph_spec(p, nc, cm, pos, selected_ids=[0])
    assert [n["data"]["id"] for n in spec["nodes"]] == [str(i) for i in range(6)]
    indexed = {it["label"] for g in spec["groups"] for it in g["items"]}
    assert indexed == {n["label"] for n in p["nodes"]}     # right-side index covers all
    assert spec["nodes"][0]["data"]["sel"] == 1 and spec["has_sel"]
    # cytoscape needs string endpoint ids matching node ids
    assert all(isinstance(e["data"]["source"], str) for e in spec["edges"])


def test_component_assets_exist_and_are_wired():
    """The renderer is now static files served by declare_component, not an
    inlined HTML string — the app breaks silently if they go missing."""
    d = gh.COMPONENT_DIR
    idx = (d / "index.html").read_text(encoding="utf-8")
    js = (d / "graph.js").read_text(encoding="utf-8")
    for v in ("cytoscape.min.js", "layout-base.js", "cose-base.js", "cytoscape-fcose.js"):
        assert (d / "vendor" / v).exists()
        assert f"vendor/{v}" in idx
    assert "graph.js" in idx
    assert "streamlit:componentReady" in js and "streamlit:setComponentValue" in js
    assert "streamlit:render" in js
    assert "function separate(" in js
    assert 'class="sidehidden"' in idx and 'id="toggle"' in idx


def test_filter_payload_induced_subgraph():
    p = _payload()
    sub = gh.filter_payload(p, [0, 1, 3])
    assert [n["id"] for n in sub["nodes"]] == [0, 1, 3]
    # only 0–1 survives; 1–2 and 3–4 each lose an endpoint
    assert [(e["source"], e["target"]) for e in sub["edges"]] == [(0, 1)]
    assert gh.filter_payload(p, [])["nodes"] == []


def test_build_graph_spec_filtered_keeps_size_scale_and_marks_cooc():
    p = _payload()
    nc, cm = gh.assign_communities(p["nodes"], p["edges"])
    pos = gh.layout_positions(p["nodes"], p["edges"])
    mmax = max(n["n_mentions"] for n in p["nodes"])
    sub = gh.filter_payload(p, [0, 1])

    full = gh.build_graph_spec(p, nc, cm, pos, selected_ids=[0], max_mentions=mmax)
    part = gh.build_graph_spec(sub, nc, cm, pos, selected_ids=[0],
                               cooc_counts={1: 7}, max_mentions=mmax)
    size_of = lambda spec, nid: next(  # noqa: E731
        n["data"]["size"] for n in spec["nodes"] if n["data"]["id"] == nid)
    # node 0 keeps the same radius in both — pinning max_mentions is the point
    assert size_of(full, "0") == size_of(part, "0")
    assert next(n["data"]["cooc"] for n in part["nodes"] if n["data"]["id"] == "1") == 7
    # filtered-out nodes vanish from the elements, the index and the legend
    assert {n["data"]["id"] for n in part["nodes"]} == {"0", "1"}
    assert not any(it["label"] in {"c2", "c3", "c4", "c5"}
                   for g in part["groups"] for it in g["items"])
    assert part["has_cooc"] and not full["has_cooc"]


def _node_js_available() -> bool:
    import shutil
    return shutil.which("node") is not None and (
        gh.COMPONENT_DIR / "vendor" / "cytoscape.min.js").exists()


def _extract_separate_js() -> str:
    """Pull `function separate(...)` out of the shipped graph.js. It is written
    for the browser (Streamlit postMessage at module scope), so the test lifts
    just the function rather than executing the whole file."""
    src = (gh.COMPONENT_DIR / "graph.js").read_text(encoding="utf-8")
    start = src.index("function separate(")
    # walk braces to the matching close
    depth, i = 0, src.index("{", start)
    while True:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    return src[start:i + 1]


@pytest.mark.skipif(not _node_js_available(), reason="needs node + vendored cytoscape")
def test_separate_js_removes_every_node_overlap(tmp_path):
    """Run the *shipped* separate() under headless cytoscape and assert no two
    node discs intersect. Filtered subgraphs are dense (53 concepts ≈ 900
    edges), and both the preset coordinates and fcose leave discs on top of one
    another; separate() is what actually guarantees the property."""
    import json
    import subprocess

    # A deliberately cruel case: many nodes crammed at near-identical positions.
    nodes = [{"data": {"id": str(i), "size": 40}, "position": {"x": i % 3, "y": i // 3}}
             for i in range(30)]
    edges = [{"data": {"id": f"e{i}", "source": str(i), "target": str((i + 1) % 30)}}
             for i in range(30)]
    vendor = gh.COMPONENT_DIR / "vendor"
    (tmp_path / "sep.js").write_text(_extract_separate_js(), encoding="utf-8")
    driver = f"""
const fs=require('fs'), vm=require('vm');
const sb={{console,setTimeout,clearTimeout,setInterval,clearInterval}};
vm.createContext(sb);
vm.runInContext('globalThis.self = globalThis;', sb);
vm.runInContext(fs.readFileSync({str(vendor / 'cytoscape.min.js')!r},'utf8'), sb);
vm.runInContext(fs.readFileSync({str(tmp_path / 'sep.js')!r},'utf8'), sb);
sb.RAW = {json.dumps(json.dumps({"nodes": nodes, "edges": edges}))};
const out = vm.runInContext(`
  const els = JSON.parse(RAW);
  const cy = cytoscape({{headless:true, styleEnabled:true, elements:els,
    layout:{{name:'preset'}},
    style:[{{selector:'node',style:{{'width':'data(size)','height':'data(size)'}}}}]}});
  const before = overlaps(cy);
  separate(cy, 18);
  const after = overlaps(cy);
  function overlaps(cy){{
    let c=0; const ns=cy.nodes();
    for(let i=0;i<ns.length;i++) for(let j=i+1;j<ns.length;j++){{
      const a=ns[i],b=ns[j],pa=a.position(),pb=b.position();
      if(Math.hypot(pa.x-pb.x,pa.y-pb.y) < (a.width()+b.width())/2) c++;
    }}
    return c;
  }}
  JSON.stringify({{before, after}});
`, sb);
console.log(out);
process.exit(0);   // headless cytoscape leaves timers pending; node would hang
"""
    (tmp_path / "drive.js").write_text(driver, encoding="utf-8")
    res = subprocess.run(["node", str(tmp_path / "drive.js")],
                         capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout.strip())
    assert out["before"] > 0, "fixture should start overlapping"
    assert out["after"] == 0, f"{out['after']} pairs still overlap after separate()"


def test_relayout_flag_reaches_the_spec():
    p = _payload()
    nc, cm = gh.assign_communities(p["nodes"], p["edges"])
    pos = gh.layout_positions(p["nodes"], p["edges"])
    assert gh.build_graph_spec(p, nc, cm, pos, relayout=True)["relayout"] is True
    assert gh.build_graph_spec(p, nc, cm, pos, relayout=False)["relayout"] is False


def test_edge_click_posts_the_concept_pair_back_to_python():
    """The whole point of the custom component: an edge tap must send both
    endpoint ids (as ints, matching concept ids) plus a changing `seq` — without
    the counter, tapping the same edge twice would not rerun Streamlit."""
    js = (gh.COMPONENT_DIR / "graph.js").read_text(encoding="utf-8")
    start = js.index("cy.on('tap','edge'")
    end = js.index("cy.on('tap',", start + 1)   # the background handler
    tap = js[start:end]
    assert "setValue({ kind:'edge'" in tap
    assert "parseInt(d.source, 10)" in tap and "parseInt(d.target, 10)" in tap
    assert "seq: ++seq" in tap
    # tapping the background clears the pair again
    assert "kind:'clear'" in js


def test_yoga_cluster_meta_only_present_buckets():
    node_yoga = {1: "jnana", 2: "jnana", 3: "karma"}
    meta = gh.yoga_cluster_meta(node_yoga)
    assert set(meta) == {"jnana", "karma"}
    assert "color" in meta["jnana"] and "Jñāna" in meta["jnana"]["label"]
