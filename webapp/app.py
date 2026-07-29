"""viveka-insight web UI.

Run:
    streamlit run webapp/app.py --server.address 0.0.0.0 --server.port 8501

Four tabs:
  Search          — query box, side-by-side EN/BN results, concept-mediated badges.
  Ask Vivekananda — natural-language QA: modern questions are bridged to the
                    concept graph, the best passages retrieved, and an answer
                    composed with deep-linked citations.
  Concept Browser — list of top concepts, click one to see its neighborhood
                    and sample paragraphs in both languages.
  Stats           — corpus + index numbers (sanity check after rebuilds).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Make the package importable when streamlit launches us from anywhere
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from viveka_insight import qa
from viveka_insight import graph_html as gh
from viveka_insight.config import CFG
from viveka_insight.llm_client import make_client
from viveka_insight.search import Searcher, SearchHit


# ──────────────────────────────────────────────────────────────────────────
# Page setup
# ──────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Viveka Insight — Concept-graph RAG",
    page_icon="📿",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading models and index ...")
def get_searcher(index_dir: str, device: str | None, load_reranker: bool) -> Searcher:
    return Searcher(index_dir=Path(index_dir), device=device, load_reranker=load_reranker)


@st.cache_resource(show_spinner="Loading answer LLM (first run downloads ~15 GB) ...")
def get_llm():
    # Loaded lazily on the first Ask, not at page load.
    return make_client(CFG)


def _highlight(text: str, sentence: str) -> str:
    if sentence and sentence in text:
        return text.replace(sentence, f"**{sentence}**", 1)
    return text


# Deep link to the hit's chapter in the hosted source HTML.
_source_url = qa.source_url


# ──────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📿 Viveka Insight")
    st.caption("Multi-granular concept-graph RAG over Vivekananda's complete works.")

    st.markdown("---")
    st.subheader("⚙️ Settings")

    # Index location is server config, not a user setting (overridable via
    # the VIVEKA_INDEX_DIR env var) — deliberately not shown in the UI.
    index_dir = os.environ.get("VIVEKA_INDEX_DIR", str(CFG.paths.index_dir))

    top_k = st.slider("Results per language", 1, 20, CFG.search.top_k)
    rerank = st.toggle("Cross-encoder rerank", value=True,
                       help="BGE-reranker-v2-m3. Slower (~200ms/query) but much more precise.")
    device = st.radio("Device", ["auto", "cuda", "cpu"], horizontal=True, index=0)
    device_arg = None if device == "auto" else device

    languages = st.multiselect("Show languages", ["en", "bn"], default=["en", "bn"])

    st.markdown("---")
    st.caption(
        f"Embedder: `{CFG.models.embedder}`\n\n"
        f"Reranker: `{CFG.models.reranker}`\n\n"
        f"Concept-extraction LLM: `{CFG.models.llm}`"
    )

# ──────────────────────────────────────────────────────────────────────────
# Load searcher
# ──────────────────────────────────────────────────────────────────────────

try:
    searcher = get_searcher(index_dir, device_arg, rerank)
except FileNotFoundError as e:
    st.error(f"❌ Could not load index: {e}")
    st.info(
        "Build the index first:\n```bash\n"
        "python scripts/build_all.py\n"
        "```"
    )
    st.stop()
except Exception as e:
    st.error(f"❌ Unexpected error loading index: {e}")
    st.stop()


# ──────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────

tab_search, tab_ask, tab_graph, tab_concepts, tab_stats = st.tabs(
    ["🔎 Search", "🙏 Ask Vivekananda", "🕸️ Concept Graph",
     "🌐 Concept Browser", "📊 Stats"]
)


# ── Search tab ────────────────────────────────────────────────────────────

with tab_search:
    st.subheader("Cross-lingual semantic search")
    st.caption(
        "Type your query in English or Bengali. The system retrieves at "
        "sentence, paragraph, and chapter granularity, traverses the concept "
        "graph for related ideas, fuses the candidates, and reranks with a "
        "cross-encoder. Results are surfaced in both languages."
    )

    # A form so that pressing Enter in the query box submits immediately —
    # no need to click the button.
    with st.form("search_form", border=False):
        query = st.text_input(
            "Query",
            placeholder='"the goal of religion is realisation"  ·  "ত্যাগ এবং সেবা"',
        )
        go = st.form_submit_button("Search", type="primary",
                                   use_container_width=True)

    def _render_hit(h: SearchHit) -> None:
        with st.container(border=True):
            top1, top2 = st.columns([5, 1])
            with top1:
                url = _source_url(h)
                # Raw HTML so the source link opens in a new tab and we don't
                # lose the search context. `location_str()` is plain text.
                st.markdown(
                    f'**#{h.rank}  ·  '
                    f'<a href="{url}" target="_blank" rel="noopener">{h.location_str()}</a>**',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<span style="color:rgba(49,51,63,.6);font-size:0.85em">'
                    f'levels: {", ".join(h.levels) or "—"}  ·  '
                    f'chapter id <code>{h.chapter_id_html or "—"}</code>  ·  '
                    f'offset {h.char_offset:,}  ·  '
                    f'<a href="{url}" target="_blank" rel="noopener">↗ open in source</a>'
                    f'</span>',
                    unsafe_allow_html=True,
                )
            with top2:
                st.metric("score", f"{h.score:.3f}")

            if h.via_concepts:
                chips = " ".join(
                    f"`{label}`" for _cid, label, _w in h.via_concepts
                )
                st.markdown(f"📎 via concepts: {chips}")

            if h.matched_sentence:
                st.markdown("**Matched:**")
                st.info(h.matched_sentence)

            with st.expander("Full paragraph", expanded=True):
                st.markdown(_highlight(h.text, h.matched_sentence))

            if h.summary:
                st.caption(f"_summary:_ {h.summary}")

    if go and query.strip():
        if not languages:
            st.warning("Pick at least one language in the sidebar.")
        else:
            with st.spinner("Searching..."):
                results = searcher.search(
                    query, top_k=top_k, languages=tuple(languages), rerank=rerank,
                )
            cols = st.columns(len(languages))
            for col, lang in zip(cols, languages):
                with col:
                    flag = "🇬🇧" if lang == "en" else "🇧🇩"
                    name = "English" if lang == "en" else "বাংলা"
                    hits = results.get(lang, [])
                    st.subheader(f"{flag} {name} — {len(hits)} results")
                    if not hits:
                        st.info("No matches.")
                    for h in hits:
                        _render_hit(h)
    elif go:
        st.warning("Please enter a query.")


# ── Ask Vivekananda ───────────────────────────────────────────────────────

with tab_ask:
    st.subheader("Ask Vivekananda")
    st.caption(
        "Ask any question — modern ones included. The question is first "
        "bridged to the concepts Vivekananda actually taught (a phone "
        "addiction becomes attachment and control of the mind), the best "
        "passages are retrieved from both languages, and the answer is "
        "composed **only** from those passages. Every claim carries a "
        "citation that deep-links to the source chapter."
    )

    # Form: Enter in the question box asks immediately (same as the button).
    with st.form("ask_form", border=False):
        qa_question = st.text_input(
            "Your question",
            key="qa_question",
            placeholder='"How do I get rid of mobile phone addiction?"  ·  '
                        '"মন একাগ্র করার উপায় কী?"',
        )
        ask_go = st.form_submit_button("Ask", type="primary",
                                       use_container_width=True)

    if ask_go:
        if not qa_question.strip():
            st.warning("Please enter a question.")
        elif not languages:
            st.warning("Pick at least one language in the sidebar.")
        else:
            llm = None
            try:
                llm = get_llm()
            except Exception as e:
                st.error(f"❌ Could not load the answer LLM: {e}")
                st.info(
                    "Tip: set `VIVEKA_LLM_BACKEND=openai` plus "
                    "`OPENAI_BASE_URL` / `OPENAI_API_KEY` "
                    "(and `VIVEKA_OPENAI_MODEL`) to answer via a hosted "
                    "endpoint instead of the local model."
                )
            if llm is not None:
                status = st.status("Thinking …", expanded=False)
                try:
                    result = qa.ask(
                        llm, searcher, qa_question,
                        languages=tuple(languages),
                        progress=lambda msg: status.update(label=msg),
                    )
                    st.session_state["qa_result"] = result
                    status.update(label="Done", state="complete")
                except Exception as e:
                    status.update(label="Answer generation failed", state="error")
                    st.error(f"❌ {e}")

    # Rendered from session state so the answer survives widget reruns.
    result = st.session_state.get("qa_result")
    if result is not None:
        plan = result.plan

        with st.expander("🌉 How the question was bridged",
                         expanded=bool(plan.bridge_note)):
            if not plan.ok:
                st.caption(
                    "Concept bridging unavailable for this question — "
                    "searched with the raw question instead."
                )
                if plan.raw:
                    st.code(plan.raw)
            if plan.bridge_note:
                st.markdown(f"_{plan.bridge_note}_")
            st.markdown("**Searched as:**")
            for q in [result.question] + plan.timeless_queries:
                st.markdown(f"- {q}")
            if plan.concepts:
                st.markdown(
                    "**Bridge concepts:** "
                    + "  ".join(f"`{c}`" for c in plan.concepts)
                )

        if not result.hits:
            st.warning("No relevant passages found — try rephrasing the question.")
        else:
            st.markdown(
                qa.linkify_citations(result.answer, result.hits),
                unsafe_allow_html=True,
            )

            st.markdown("---")
            st.markdown("#### References")
            for h in result.hits:
                with st.container(border=True):
                    flag = "🇬🇧" if h.lang == "en" else "🇧🇩"
                    url = _source_url(h)
                    st.markdown(
                        f'**[{h.rank}]** {flag} '
                        f'<a href="{url}" target="_blank" rel="noopener">'
                        f'{h.location_str()}</a>',
                        unsafe_allow_html=True,
                    )
                    quote = h.matched_sentence or h.summary or ""
                    if quote:
                        st.caption(f"“{quote}”")
                    if h.via_concepts:
                        st.markdown(
                            "📎 via concepts: "
                            + " ".join(f"`{label}`" for _c, label, _w in h.via_concepts)
                        )
                    with st.expander("Full passage"):
                        st.markdown(_highlight(h.text, h.matched_sentence))


# ── Concept Graph ─────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Building concept graph …")
def _graph_bundle(_searcher, n: int, relations: tuple, color_by: str):
    payload = _searcher.concept_graph(n=n, relations=relations)
    ids = [nd["id"] for nd in payload["nodes"]]
    positions = gh.layout_positions(payload["nodes"], payload["edges"])
    if color_by == "Four Yogas":
        node_cluster = _searcher.four_yoga_assignment(ids)
        cluster_meta = gh.yoga_cluster_meta(node_cluster)
    else:
        node_cluster, cluster_meta = gh.assign_communities(
            payload["nodes"], payload["edges"])
    return payload, node_cluster, cluster_meta, positions


# Bidirectional component: the graph must post the clicked edge back to Python
# so the server can query the passages that pair of concepts shares. Declared
# once at import; Streamlit serves every file under webapp/static/.
_graph_component = components.declare_component(
    "viveka_concept_graph", path=str(gh.COMPONENT_DIR))


@st.cache_data(show_spinner=False)
def _cooc_counts(_searcher, selected_ids: tuple, node_ids: tuple) -> dict:
    """Unthresholded shared-paragraph counts, so moving the threshold slider is
    a dict comprehension rather than a re-query."""
    return _searcher.cooccurring_concepts(selected_ids, restrict_to=node_ids)


def _para_url(md: dict) -> str:
    base = (CFG.search.source_url_en if md["lang"] == "en"
            else CFG.search.source_url_bn)
    anchor = md.get("para_id_html") or md.get("chapter_id_html") or ""
    return f"{base}#{anchor}" if anchor else base


def _render_passages(rows: list[dict], full: int) -> None:
    """Passage cards. `full` is the size of the concept set they were matched
    against, so the n/N badge means the same thing in both evidence panels."""
    for md in rows:
        with st.container(border=True):
            flag = "🇬🇧" if md["lang"] == "en" else "🇧🇩"
            loc = f"{md['volume_title']} › {md['chapter_title']} ¶{md['paragraph_idx']+1}"
            st.markdown(
                f"{flag} **{md['n_matched']}/{full} concepts** · "
                f"<a href='{_para_url(md)}' target='_blank' rel='noopener'>{loc}</a> "
                f"— {' '.join(f'`{c}`' for c in md['matched_concepts'])}",
                unsafe_allow_html=True)
            txt = md["text"]
            st.write(txt[:700] + ("…" if len(txt) > 700 else ""))


_SHOW_ALL = "All concepts"
_SHOW_SEL = "Selected only"
_SHOW_COOC = "Selected + co-occurring"

# Shared-paragraph counts run from 1 to ~1300 and are heavily skewed, so the
# threshold ladder is roughly logarithmic. Around 20 a popular concept keeps a
# few dozen neighbours — a readable graph.
_CO_STEPS = [1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000]


with tab_graph:
    st.subheader("Concept graph")
    st.caption(
        "Every node is a concept; edges link concepts that are **semantically "
        "similar** (dashed) or **discussed together** (solid). Colour groups "
        "concepts into auto-detected themes — or into the four classical yogas. "
        "Tick concepts on the right to narrow the graph to them and to surface "
        "the passages that best weave them together."
    )

    c1, c2, c3, c4 = st.columns([2, 2, 2, 1.4])
    with c1:
        g_n = st.slider("Concepts shown (by frequency)", 60, 600, 250, step=20,
                        key="cg_n")
    with c2:
        g_color = st.radio("Colour by", ["Communities", "Four Yogas"],
                           horizontal=True, key="cg_color")
    with c3:
        g_rel = st.radio("Edges", ["both", "similar", "co-occurs"],
                         horizontal=True, key="cg_rel")
    with c4:
        g_lang = st.radio("Language", ["Both", "English", "বাংলা"], key="cg_lang")

    d1, d2, d3 = st.columns([2, 2.6, 2.4])
    with d1:
        g_show = st.radio(
            "Graph shows", [_SHOW_ALL, _SHOW_SEL, _SHOW_COOC], key="cg_show",
            help="What to draw once you have ticked some concepts.",
        )
    with d2:
        g_min_co = st.select_slider(
            "Co-occurrence ≥ (shared paragraphs)", _CO_STEPS, value=20,
            key="cg_minco", disabled=(g_show != _SHOW_COOC),
            help="A concept joins the graph when it shares at least this many "
                 "paragraphs with one of your ticked concepts.",
        )
    with d3:
        g_k = st.slider("Passages shown", 5, 100, 40, step=5, key="cg_k")
        g_ev_cooc = st.checkbox(
            "Passages for the co-occurring concepts too", key="cg_ev_cooc",
            disabled=(g_show != _SHOW_COOC),
            help="Off: rank passages by how many of your *ticked* concepts they "
                 "carry. On: widen the search to every concept on the graph.",
        )

    relations = (("similar", "co-occurs") if g_rel == "both"
                 else (g_rel,))
    payload, node_cluster, cluster_meta, positions = _graph_bundle(
        searcher, g_n, relations, g_color)

    label_to_id = {nd["label"]: nd["id"] for nd in payload["nodes"]}
    mentions = {nd["label"]: nd["n_mentions"] for nd in payload["nodes"]}
    all_labels = [nd["label"] for nd in sorted(
        payload["nodes"], key=lambda n: -n["n_mentions"])]
    node_ids = [nd["id"] for nd in payload["nodes"]]
    max_mentions = max(mentions.values(), default=1)

    col_graph, col_side = st.columns([3, 1])

    with col_side:
        st.markdown("**Select concepts**")
        # Drop any labels no longer in the graph (e.g. after N shrank) *before*
        # the editor is instantiated, so it never renders a stale row.
        rev = st.session_state.setdefault("cg_rev", 0)
        sel_labels = set(st.session_state.get("cg_sel", [])) & set(label_to_id)

        q = st.text_input("filter", key="cg_filter", label_visibility="collapsed",
                          placeholder="filter concepts…").strip()
        rows = [l for l in all_labels if q.lower() in l.lower()] if q else all_labels

        # All/None act on the filtered rows, so they double as bulk operators
        # ("tick every concept matching 'yoga'"). They write the selection and
        # bump `rev`, which re-keys the editor so its own cell-edit state is
        # dropped rather than replayed over the new values.
        bcol1, bcol2 = st.columns(2)
        if bcol1.button("All", use_container_width=True, key="cg_all"):
            st.session_state["cg_sel"] = sorted(sel_labels | set(rows))
            st.session_state["cg_rev"] = rev + 1
            st.rerun()
        if bcol2.button("None", use_container_width=True, key="cg_none"):
            st.session_state["cg_sel"] = sorted(sel_labels - set(rows))
            st.session_state["cg_rev"] = rev + 1
            st.rerun()

        if not rows:
            st.caption("No concept matches that filter.")
        else:
            # `g_n` and `q` are in the key for the same reason as `rev`: both
            # reshuffle the rows, and a replayed edit would land on the wrong one.
            edited = st.data_editor(
                pd.DataFrame({"✓": [l in sel_labels for l in rows],
                              "concept": rows,
                              "n": [mentions[l] for l in rows]}),
                hide_index=True, height=360, disabled=["concept", "n"],
                key=f"cg_editor_{g_n}_{rev}_{q}",
                column_config={
                    "✓": st.column_config.CheckboxColumn("✓", width="small"),
                    "concept": st.column_config.TextColumn("concept"),
                    "n": st.column_config.NumberColumn(
                        "n", width="small", help="mentions in the corpus"),
                },
            )
            ticked = set(edited.loc[edited["✓"], "concept"])
            sel_labels = (sel_labels - set(rows)) | ticked

        st.session_state["cg_sel"] = sorted(sel_labels)
        caption = st.empty()

    selected_ids = [label_to_id[l] for l in sorted(sel_labels)]

    # Co-occurrence counts come off para_concept directly (shared-paragraph
    # counts), not off the 'co-occurs' edge weight, which is log-scaled and
    # renormalized and so is not a paragraph count.
    cooc_counts: dict[int, int] = {}
    if g_show == _SHOW_COOC and selected_ids:
        cooc_counts = {c: n for c, n in
                       _cooc_counts(searcher, tuple(selected_ids), tuple(node_ids)).items()
                       if n >= g_min_co}

    if not selected_ids or g_show == _SHOW_ALL:
        view, visible_ids = payload, node_ids
    else:
        visible_ids = sorted(set(selected_ids) | set(cooc_counts))
        view = gh.filter_payload(payload, visible_ids)

    caption.caption(
        f"{len(sel_labels)} ticked · showing {len(view['nodes'])} of "
        f"{len(payload['nodes'])} nodes · {len(view['edges'])} edges")

    with col_graph:
        spec = gh.build_graph_spec(view, node_cluster, cluster_meta, positions,
                                   selected_ids=selected_ids,
                                   cooc_counts=cooc_counts,
                                   max_mentions=max_mentions,
                                   relayout=len(view["nodes"]) < len(payload["nodes"]))
        # spec_key lets the component skip a rebuild (and its layout + zoom
        # reset) when an unrelated widget triggers the rerun.
        spec_key = str(hash(json.dumps(spec, sort_keys=True)))
        event = _graph_component(spec=spec, height=680, spec_key=spec_key,
                                 default=None, key="cg_graph")
        st.caption("Click an edge to see the passages its two concepts share.")
        if g_show != _SHOW_ALL and not selected_ids:
            st.caption("Nothing ticked yet — showing the whole graph. "
                       "Tick a concept on the right to narrow it.")
        elif g_show == _SHOW_COOC and selected_ids and not cooc_counts:
            st.caption(f"No concept on the graph shares ≥ {g_min_co} paragraphs "
                       "with your selection — lower the threshold.")

    # ── which evidence panel? ─────────────────────────────────────────────
    # The component keeps returning its last value on every rerun, so a click
    # from three interactions ago would keep overriding the selection panel.
    # Accept an event only when its `seq` is new; otherwise expire it as soon
    # as the graph it referred to changed.
    view_sig = (tuple(selected_ids), g_show, g_min_co, g_n, g_rel)
    if isinstance(event, dict) and event.get("seq") != st.session_state.get("cg_edge_seq"):
        st.session_state["cg_edge_seq"] = event.get("seq")
        st.session_state["cg_edge"] = event if event.get("kind") == "edge" else None
        st.session_state["cg_edge_sig"] = view_sig
    elif st.session_state.get("cg_edge_sig") != view_sig:
        st.session_state["cg_edge"] = None

    edge = st.session_state.get("cg_edge")
    # the pair must still be on screen (N shrank, threshold moved, …)
    if edge and not {int(edge["source"]), int(edge["target"])} <= set(visible_ids):
        edge = st.session_state["cg_edge"] = None

    # ── evidence panel A: an edge was clicked → passages for exactly that pair ──
    if edge:
        pair = [int(edge["source"]), int(edge["target"])]
        id_to_label = {v: k for k, v in label_to_id.items()}
        a, b = (id_to_label.get(p, str(p)) for p in pair)
        langs = {"Both": ("en", "bn"), "English": ("en",), "বাংলা": ("bn",)}[g_lang]

        st.markdown("---")
        both = searcher.example_paragraphs(pair, lang=langs, k=g_k, require_all=True)
        st.markdown(f"#### Passages on `{a}` **and** `{b}`", unsafe_allow_html=False)
        if both:
            st.caption(f"{len(both)} passage(s) carrying both concepts · "
                       f"edge: {edge['relation']}, strength {float(edge['weight']):.2f} · "
                       "click empty canvas to clear")
            _render_passages(both, full=2)
        else:
            # A 'similar' edge means "close in meaning-space", which does not
            # imply the two are ever discussed in one paragraph.
            st.caption(
                f"No paragraph mentions both — they are linked by a "
                f"**{edge['relation']}** edge. Showing passages for each "
                f"concept separately.", unsafe_allow_html=False)
            for cid, lbl in zip(pair, (a, b)):
                st.markdown(f"**`{lbl}`**")
                _render_passages(
                    searcher.example_paragraphs([cid], lang=langs, k=max(3, g_k // 2)),
                    full=1)

    # ── evidence panel B: passages that best cover the selected concepts ──
    elif selected_ids:
        langs = {"Both": ("en", "bn"), "English": ("en",), "বাংলা": ("bn",)}[g_lang]
        ev_ids = (visible_ids if g_show == _SHOW_COOC and g_ev_cooc
                  else selected_ids)
        exs = searcher.example_paragraphs(ev_ids, lang=langs, k=g_k)
        st.markdown("---")
        if not exs:
            st.info("No passages found for that selection in the chosen language.")
        else:
            full = len(ev_ids)
            best = exs[0]["n_matched"]
            if best < full:
                st.markdown(
                    f"#### Best-covering passages "
                    f"<span style='color:#888;font-size:.8em'>(no single passage "
                    f"holds all {full} concepts — a paragraph carries at "
                    f"most ~5; showing the {len(exs)} that cover the most)</span>",
                    unsafe_allow_html=True)
            else:
                st.markdown(f"#### Passages containing all {full} selected concepts")
            _render_passages(exs, full=full)
    else:
        st.caption("Tip: click an edge to see the passages its two concepts share, "
                   "click a node to focus it, or tick several concepts on the "
                   "right to surface the passages that connect them.")


# ── Concept browser ───────────────────────────────────────────────────────

with tab_concepts:
    st.subheader("Top concepts in the corpus")
    st.caption(
        "These are the abstraction nodes built by the LLM extraction pass. "
        "Clicking a concept shows its graph neighborhood and a sample of "
        "passages that link to it."
    )

    n_top = st.slider("Show top N concepts", 20, 200, 80, key="topn")
    top = searcher.top_concepts(n=n_top)

    if not top:
        st.info("No concepts have been extracted yet. Run "
                "`python scripts/03_extract_concepts.py` followed by "
                "`python scripts/04_link_concepts.py`.")
    else:
        # Compact grid of buttons
        cols = st.columns(4)
        chosen_label = st.session_state.get("chosen_concept_label", None)
        for i, (cid, label, n) in enumerate(top):
            with cols[i % 4]:
                if st.button(f"`{label}`  ·  {n}",
                             key=f"concept_{cid}", use_container_width=True):
                    st.session_state["chosen_concept_label"] = label
                    st.session_state["chosen_concept_id"] = cid
                    chosen_label = label

        chosen_id = st.session_state.get("chosen_concept_id", None)
        if chosen_id is not None:
            st.markdown("---")
            n = searcher.concept_neighborhood(chosen_id, k=12)
            if not n:
                st.warning("Concept not found.")
            else:
                center = n["center"]
                st.markdown(f"### Concept: `{center['canonical_label']}`")
                st.caption(f"{center['n_mentions']} mentions across the corpus")

                if n["neighbors"]:
                    st.markdown("**Connected concepts**")
                    for neigh in n["neighbors"]:
                        bar = int(20 * neigh["weight"])
                        bar_str = "▰" * bar + "▱" * (20 - bar)
                        st.markdown(
                            f"`{neigh['label']}`  "
                            f"<span style='color:#888'>{bar_str}</span>  "
                            f"_{neigh['relation']}_  ·  {neigh['weight']:.2f}",
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("(no neighbours above the similarity threshold)")

                col_en, col_bn = st.columns(2)
                with col_en:
                    st.markdown("**🇬🇧 Sample English passages**")
                    for p in n["paragraphs_en"]:
                        with st.container(border=True):
                            st.caption(
                                f"{p['volume_title']} › {p['chapter_title']} ¶{p['paragraph_idx']+1}"
                            )
                            st.write(p["text"][:600] + ("…" if len(p["text"]) > 600 else ""))
                with col_bn:
                    st.markdown("**🇧🇩 Sample Bengali passages**")
                    for p in n["paragraphs_bn"]:
                        with st.container(border=True):
                            st.caption(
                                f"{p['volume_title']} › {p['chapter_title']} ¶{p['paragraph_idx']+1}"
                            )
                            st.write(p["text"][:600] + ("…" if len(p["text"]) > 600 else ""))


# ── Stats tab ─────────────────────────────────────────────────────────────

with tab_stats:
    st.subheader("Index statistics")
    s = searcher.stats()
    cols = st.columns(4)
    cols[0].metric("Volumes",     f"{s.get('n_volumes', 0):,}")
    cols[1].metric("Chapters",    f"{s.get('n_chapters', 0):,}")
    cols[2].metric("Paragraphs",  f"{s.get('n_paragraphs', 0):,}")
    cols[3].metric("Sentences",   f"{s.get('n_sentences', 0):,}")

    cols = st.columns(4)
    cols[0].metric("Concepts",    f"{s.get('n_concepts', 0):,}")
    cols[1].metric("Entities",    f"{s.get('n_entities', 0):,}")
    cols[2].metric("Concept↔Concept edges", f"{s.get('n_concept_edges', 0):,}")
    cols[3].metric("Para→Concept edges",    f"{s.get('n_para_concept_edges', 0):,}")

    st.markdown("**Per-language paragraph counts**")
    en = s.get("paragraphs_en", 0)
    bn = s.get("paragraphs_bn", 0)
    st.write(f"🇬🇧 English: **{en:,}**  ·  🇧🇩 Bengali: **{bn:,}**")

    # Pipeline state
    cur = searcher.conn.cursor()
    cur.execute("SELECT step, completed_at, progress, total, note "
                "FROM pipeline_state ORDER BY step")
    rows = cur.fetchall()
    if rows:
        st.markdown("**Pipeline state**")
        for r in rows:
            done = "✓" if r["completed_at"] else "…"
            line = f"{done}  `{r['step']}`"
            if r["completed_at"]:
                line += f"  · {r['completed_at']}"
            if r["progress"] is not None and r["total"]:
                line += f"  · {r['progress']:,}/{r['total']:,}"
            if r["note"]:
                line += f"  · {r['note']}"
            st.markdown(line)
