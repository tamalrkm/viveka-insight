# Architecture and invariants

Read this before changing anything in the pipeline. Several of the rules
below are load-bearing: breaking them does not raise an error, it silently
corrupts an index or throws away hours of GPU time.

## Pipeline stages

```
00_snapshot → 01_parse → 02_embed → 02b_restore
            → 03_extract_concepts → 04_link_concepts → 05_build_graph
```

`scripts/build_all.py` runs them in order as separate subprocesses. That
isolation is deliberate: it guarantees the embedder is fully unloaded before
the LLM tries to claim the GPU, and it keeps stage logs separable. Every
stage is idempotent and resumable — kill the orchestrator and re-run it, and
completed stages are skipped.

```bash
python scripts/build_all.py            # full build
python scripts/build_all.py --sample   # synthetic corpus, CPU, <1 min
python scripts/build_all.py --skip-llm # vector-only index, no concept graph
python scripts/build_all.py --gnn      # add the optional GNN stage
```

## Cost model — read before re-running anything

Stage 3 (LLM concept extraction) is the only expensive stage: roughly 3–4
hours on an A100 over the full corpus. Every other stage takes minutes.
**Any change that re-runs stage 3 unnecessarily is a regression.**

The pipeline is resumable *and* warm-startable. After a crash or a
source-HTML edit, just re-run `build_all.py`:

1. `00_snapshot` dumps `(text → summary, concepts, entities)` into the
   `concept_snapshot` / `alias_snapshot` tables.
2. `01_parse --force` re-parses, and **wipes all downstream extraction
   state** (`para_concept`, `para_entity`, `concept_aliases`,
   `concept_edges`, `embeddings`, `concepts`, `entities`). Snapshot tables
   are deliberately preserved.
3. `02_embed` re-embeds.
4. `02b_restore` re-attaches cached concepts to the fresh paragraph rows by
   normalized-text match.
5. `03_extract_concepts` then only sends genuinely new paragraphs to the LLM.
6. `04_link_concepts` rebuilds the concept FAISS index and the
   similarity / co-occurrence edges.
7. `05_build_graph` recounts and writes statistics.

A typical incremental rebuild after a content edit: 10–15 minutes.

## Hard invariants

- **Concept canonical labels are English, lowercase, hyphenated.** This is
  the cross-lingual string-equality key, and the whole design rests on it.
  Never add per-language concept tables.
- **Warm-start key** is `(NFC-normalized paragraph text, lang,
  extractor_tag)` where `extractor_tag = "<llm_model>@v<PROMPT_VERSION>"`.
- **`normalize_text()` in `db.py` is the single source of truth** for that
  match key. Do not normalize text differently anywhere else, for any reason.
- **Paragraph→concept coverage of ~71% is normal.** Short fragments ("Yes.",
  headings, brief utterances) legitimately carry no concepts. This is not a
  bug to fix.
- **Streamlit caches imported modules across `app.py` reruns.** After
  changing a config field the webapp uses, do a full restart, not a rerun.
  Symptom of forgetting: `app.py` runs against the *old* library and dies
  with `TypeError: ... unexpected keyword argument` or `AttributeError`,
  naming something you just added.
- **The Concept Graph is a bidirectional custom component**, not an
  `st.iframe` — clicking an edge posts the concept pair back to Python. The
  front end lives in `webapp/static/` (`index.html`, `graph.js`, `vendor/`),
  declared via `components.declare_component(..., path=gh.COMPONENT_DIR)`.
  `graph.js` must keep answering `streamlit:componentReady` and sending a
  changing `seq` with each `setComponentValue`, or repeated clicks on the
  same edge will not trigger a rerun. Edits to `graph.js` / `index.html` need
  only a browser hard-refresh, not a server restart.

## Don't

- **Don't run `01_parse.py --force` without running `00_snapshot` first.**
  You will lose the warm-start cache for any unsaved extractions and pay the
  full multi-hour stage-3 cost again.
- **Don't bump `PROMPT_VERSION`** in `viveka_insight/concept_extraction.py`
  casually. It invalidates the entire warm-start cache. Bump it only when the
  prompt or the output schema actually changes.
- **Don't edit paragraph text via SQL** (e.g. to fix a display typo). The
  source of truth is the HTML in `data/`; edit there and re-run.
- **Don't add fallbacks for stage-3 failures.** Per-paragraph resumability
  already handles transient LLM errors — a re-run picks up where it stopped.
- **Don't introduce a second text normalization** for any new use case.

## Source HTML structure

Both source files live in `data/` and are gitignored (see
[`NOTICE.md`](../NOTICE.md) for why the texts are not redistributed). A
synthetic corpus with the same structure ships in
[`sample_data/`](../sample_data/).

**English.** Flat siblings under `<body>`, no per-volume wrapper:
`h2.volume-title` starts a volume; `div.section-head` /
`div.subsection-head` are optional; `div.chapter` holds an
`h3.chapter-title`, a leading `p.chapter-breadcrumb` (skipped), and a
`div.chapter-content` containing the paragraphs.

**Bengali.** Chapter wrapper is `section.content-block[id=content-N]` in the
current export; the original export used `div.bb-item` with the same id
scheme. The parser accepts **both**. Inside either:
`div.scroller > h2` for the title, then the paragraphs. Volume boundaries
come from `<h3>` markers matching `স্বামী বিবেকানন্দ সমগ্র খণ্ড N`.

**Per-paragraph anchors.** Every `<p>` carries `id="p-N"`. The parser stores
it as `Paragraph.para_id_html` → the `paragraphs.para_id_html` column, and
`qa.source_url()` deep-links to `#p-N`, falling back to the chapter anchor.

If either source file is replaced, run a structural diagnostic first — the
parsers are structurally specific. `tests/test_smoke.py::test_english_parser_basics`
is the floor.

## Deep links

Each result links to its source **paragraph** via `para_id_html` (`#p-N`),
falling back to `chapter_id_html` (`#content-N` / `#ch_N`). There is a single
builder, `qa.source_url()` (the webapp aliases it as `_source_url`). Base
URLs are `Search.source_url_en` / `Search.source_url_bn` in `config.py` and
point at the publicly hosted editions.

## Configuration

Everything is in `viveka_insight/config.py`, and every field is overridable
by a `VIVEKA_*` environment variable — `VIVEKA_LLM_BACKEND`, `VIVEKA_DEVICE`,
`VIVEKA_INDEX_DIR`, `VIVEKA_EN_HTML`, `VIVEKA_EMBEDDER`, and so on. The
pipeline targets an 80 GB-class GPU on CUDA 12.4 and falls back to
`transformers` + Qwen2.5-7B when vLLM is unavailable.

LLM backends are `vllm`, `transformers`, `openai` (any OpenAI-compatible
endpoint), and `stub`. The stub is a deterministic keyword matcher used by
`--sample` and never a real model — do not build a searchable index with it.
