# viveka-insight

**Multi-granular concept-graph RAG over Swami Vivekananda's complete works,
in English and Bengali, simultaneously.**

This is not a vanilla "embed sentences + FAISS" search. It's a layered system
where retrieval traverses both the vector space *and* a knowledge graph of
extracted concepts, then has a cross-encoder make the final ranking. The
graph is the bridge that lets a Bengali query find the right English passage
(and vice versa) even when no surface-level vocabulary overlaps.

```
                 ┌─────────────────────────────┐
                 │     Query (en or bn)        │
                 └──────────────┬──────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
   ┌─────────────────────────┐     ┌──────────────────────────┐
   │  Path A: Direct vector  │     │  Path B: Concept graph   │
   │  (BGE-M3, 3 levels)     │     │  query → concepts → 1-hop│
   │  sentence + paragraph   │     │  → paragraphs            │
   │  + chapter, EN & BN     │     │                          │
   └────────────┬────────────┘     └─────────────┬────────────┘
                ▼                                ▼
                ┌──────── weighted RRF fusion ──────────┐
                ▼                                       ▼
        ┌─────────────────────────────────────────────┐
        │   BGE-reranker-v2-m3 (cross-encoder)       │
        └────────────────────┬────────────────────────┘
                             ▼
              Top-K results, EN + BN, with citations
              and "via concepts: X, Y, Z" provenance.
```

## Statement of need

Digitized editions of classical, religious and philosophical literature are
widely available, but access to them stays lexical: readers search for words,
while what they hold are questions. Two problems recur and are poorly served
by existing tooling.

**The languages are not aligned.** Standard cross-lingual retrieval assumes
translation pairs. Many real collections have none — a body of work may exist
in two languages that overlap thematically while being independent
compositions. Multilingual encoders help, but give no inspectable link
structure and no explanation of *why* a cross-language result came back.

**The vocabulary is historically distant.** Users bring contemporary questions
("how do I get rid of phone addiction?") to authors who wrote a century ago.
Those words do not occur in the corpus, though the corpus treats the
underlying phenomena under other names — *attachment*, *habit*,
*control of the mind*.

Existing graph-augmented retrieval does not close these gaps: GraphRAG-style
systems build *entity* graphs, which over discursive prose capture the proper
nouns and miss the argument, and are not designed to bridge languages. This
project builds a **concept** graph instead, with canonical English labels as a
language-agnostic key, so string equality links passages across languages with
no parallel data. It is corpus-agnostic — parsers, languages, models and
fusion weights are configuration; the Vivekananda deployment is the reference
instance, not a hard-coded assumption.

## Quick start: run the pipeline on the sample corpus

Want to see the whole thing work before committing to a GPU and a corpus?

```bash
python scripts/build_all.py --sample
```

This parses a tiny **synthetic** corpus shipped in [`sample_data/`](sample_data/)
(20 English + 20 Bengali paragraphs), embeds it, runs concept extraction with a
stub backend, links the graph and reports what it built — on CPU, in under a
minute, with no GPU and no LLM download. It writes to `index_data_sample/`, so
an existing real index is never touched.

Then browse or query it:

```bash
VIVEKA_INDEX_DIR=$PWD/index_data_sample VIVEKA_DEVICE=cpu \
  streamlit run webapp/app.py --server.port 8501
```

An English query such as *"how do I stop being distracted?"* returns Bengali
passages on একাগ্রতা, and a Bengali query returns English ones — the
cross-lingual concept layer, visible in miniature.

Two caveats, both important: the sample text is **not** Vivekananda (see
[`sample_data/README.md`](sample_data/README.md)), and the stub extractor is a
keyword matcher rather than a language model, so the resulting graph
demonstrates that the pipeline runs and nothing else.

## Testing

```bash
source .venv/bin/activate
python -m pytest tests/ -q      # 74 tests, ~11 s
```

CPU-only by design: no GPU, no model downloads, no corpus required, so the
suite runs anywhere. It covers the parsers, database schema,
extraction-output parsing, rank fusion, query-bridging fallbacks, prompt
budgeting, citation linkification and the concept-graph view.

## Contributing and support

Bug reports, parser fixes, new language support and prompt improvements are
welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to report an issue,
ask a question, or propose a change, and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community expectations. Read the
"Hard invariants" section of [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
before changing anything in the pipeline — several of them are load-bearing
and break indexes quietly.

## What's in here

| Layer                | Technology                                      | Why                                                                        |
| -------------------- | ----------------------------------------------- | -------------------------------------------------------------------------- |
| Multilingual encoder | **BGE-M3** (1024-d, dense)                      | Top multilingual MTEB; same model handles 256-token sentences and 8K-token chapters. |
| Concept extraction   | **Qwen2.5-14B-Instruct** via **vLLM**           | Strong Bengali; reliable JSON; ~1500 tok/s on A100.                        |
| Reranker             | **BGE-reranker-v2-m3**                          | Matches the embedder's training distribution.                              |
| Storage              | **SQLite** (graph) + **FAISS** flat IP (vectors)| Single-file, embedded, fast enough for ~150K vectors.                      |
| Optional GNN         | **HeteroGraphSAGE** (PyG)                       | Chapter-coherence triplet loss → refined paragraph embeddings.             |
| UI                   | **Streamlit**                                   | Side-by-side EN/BN results, concept browser, stats.                        |

## Repo layout

```
viveka-insight/
├── viveka_insight/         ← library code
│   ├── config.py           central config (paths, models, hyperparams)
│   ├── parser.py           HTML → Paragraph / ChapterGroup
│   ├── db.py               SQLite schema + helpers
│   ├── embeddings.py       BGE-M3 wrapper + FAISS helpers
│   ├── llm_client.py       vLLM / transformers / OpenAI backends
│   ├── concept_extraction.py  prompt + JSON parser
│   ├── concept_linking.py  merge duplicates, build edges
│   ├── graph.py            traversal helpers (pure SQL)
│   ├── reranker.py         BGE-reranker wrapper
│   ├── search.py           the main Searcher class
│   └── gnn.py              optional HeteroGraphSAGE training
├── scripts/                ← pipeline stages
│   ├── 01_parse.py
│   ├── 02_embed.py
│   ├── 03_extract_concepts.py
│   ├── 04_link_concepts.py
│   ├── 05_build_graph.py
│   ├── 06_train_gnn.py     (optional)
│   └── build_all.py        orchestrator
├── webapp/
│   └── app.py              Streamlit UI
├── data/                   ← put your HTML files here
│   ├── vivekananda_complete_works.html
│   └── all.html
└── index_data/             ← generated; FAISS files + SQLite + GNN embeddings
```

## Setup (CUDA 12.4 + A100 80GB)

```bash
# 1. Clone / copy the project to the server
cd ~/projects && git clone https://github.com/tamalrkm/viveka-insight.git && cd viveka-insight

# 2. Python 3.10 or 3.11 — vLLM 0.6 doesn't support 3.12 yet on all platforms
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip wheel

# 3. Install torch with CUDA 12.4 wheels FIRST (so other packages don't pull CPU torch)
pip install torch==2.4.1 --extra-index-url https://download.pytorch.org/whl/cu124

# 4. Everything else
pip install -r requirements.txt

# 5. (optional) PyG accelerators for the GNN step
pip install torch_scatter torch_sparse \
    -f https://data.pyg.org/whl/torch-2.4.1+cu124.html

# 6. NLTK punkt (one-time, for English sentence splitting)
python -c "import nltk; nltk.download('punkt_tab')"

# 7. Drop the source HTML files into data/
cp /path/to/vivekananda_complete_works.html data/
cp /path/to/all.html                          data/
```

## Build the index

```bash
# Full pipeline — parse, embed at all granularities, LLM concept extraction,
# concept linking, sanity check.
python scripts/build_all.py

# Add the optional GNN step
python scripts/build_all.py --gnn

# Skip LLM extraction — gives you a pure vector index much faster (~10 min)
python scripts/build_all.py --skip-llm
```

The pipeline is **resumable**. If you SIGKILL it (e.g. close the SSH session)
and rerun, it skips completed stages and resumes the in-progress one from
the last checkpoint (per-paragraph for stage 3).

Running stages individually is fine too:

```bash
python scripts/01_parse.py
python scripts/02_embed.py
python scripts/03_extract_concepts.py --limit 200      # quick smoke test
python scripts/03_extract_concepts.py                  # full run
python scripts/04_link_concepts.py
python scripts/05_build_graph.py
python scripts/06_train_gnn.py --enable                # optional
```

### Time / VRAM budget on an A100 80GB

| Stage                         |   time |   peak VRAM |
| ----------------------------- | ------ | ----------- |
| 01 parse                      | 30 s   | —           |
| 02 embed (sent + para + chap) | 6 min  | ~6 GB       |
| 03 LLM concept extraction     | 3-4 hr | ~50 GB      |
| 04 link concepts              | 2 min  | ~6 GB       |
| 05 build graph + stats        | 5 s    | —           |
| 06 GNN (optional)             | 5 min  | ~10 GB      |

The big cost is stage 3 — LLM extraction over ~30K paragraphs. You can
parallelize this by splitting paragraphs across machines (each instance
sets `--limit` and a non-overlapping slice — easy to add if needed).

## Run the webapp

```bash
streamlit run webapp/app.py --server.address 0.0.0.0 --server.port 8501
```

VSCode SSH will forward port 8501 automatically — just open
`http://localhost:8501` in your local browser. (If it doesn't auto-forward,
use `Cmd+Shift+P` → "Forward a Port".)

## Configuration

Everything's in `viveka_insight/config.py`. Override any field with an
environment variable:

```bash
export VIVEKA_LLM_BACKEND=openai
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_API_KEY=sk-...
export VIVEKA_OPENAI_MODEL=gpt-4o-mini
python scripts/03_extract_concepts.py    # now runs against OpenAI

export VIVEKA_GNN_ENABLED=1
python scripts/build_all.py --gnn

export VIVEKA_DEVICE=cpu                 # debug runs without GPU
```

Tunable knobs that actually matter:

* **`Search.k_*`**: how wide each retrieval path casts its net. Bigger →
  better recall, slower rerank.
* **`ConceptLinking.merge_threshold`**: 0.92 is conservative. Drop to 0.88
  if you see obvious duplicates (`compassion` and `compassion-`); raise to
  0.95 if distinct concepts are getting merged.
* **`Search.w_concept`**: 1.2 boosts concept-mediated hits. Set to 0 to
  ablate the graph and see vector-only behavior.

## How retrieval actually works

1. The query goes through BGE-M3 once → 1024-d normalized vector.
2. **Path A** searches the sentence, paragraph, and chapter FAISS indices
   in both languages. Sentence hits are mapped to their parent paragraphs;
   chapter hits expand to their first 2 paragraphs. The "level" each hit
   came from is preserved.
3. **Path B** searches the concept FAISS index, takes the top-K concept
   nodes, walks 1 hop along `similar` and `co-occurs` edges to get the
   neighborhood, then pulls each concept's top paragraphs from
   `para_concept`.
4. Each path produces a ranked list per paragraph. **Reciprocal Rank Fusion**
   combines them with per-path weights (concepts get a 1.2× boost since the
   graph is the differentiator).
5. The top 60 are sent to the cross-encoder for reranking against the query.
6. Top-K per language is returned with full provenance: which sentence
   matched, which concepts mediated, and exact volume / chapter / paragraph
   citation.

This means a Bengali query like *"ত্যাগ ও সেবা"* (renunciation and service)
matches English passages about renunciation through the concept graph: the
LLM tagged Bengali paragraphs with the canonical English label
`renunciation`, the English paragraphs were tagged the same way, and the
concept-mediated path finds them all in one hop.

## Limitations + future work

* **Concept extraction quality is the floor.** If the LLM mislabels a
  Bengali term, that paragraph won't be findable through the graph for
  cross-lingual queries — only through the (still-strong) BGE-M3 vector
  path. We mitigate with a tight prompt + canonical-label normalization +
  duplicate merging, but a domain-specific finetune of the extractor would
  help. The dataset for that is the existing extraction itself: train on
  (paragraph → JSON) pairs from a smaller well-checked subset.
* **Graph is static after build.** Adding a new corpus means re-running
  stage 3 + 4. There's no incremental ingest yet — easy to add.
* **No BM25 / sparse retrieval.** BGE-M3 *has* a sparse head — wiring it
  into a third retrieval path would help with proper-noun queries
  ("Vivekananda", "Ramakrishna") and exact-quote lookup.
* **GNN training signal is weak.** Chapter coherence is a plausible prior
  but not a labeled supervision signal. A pseudo-labeled (query → relevant
  passage) set, even small, would unlock much better refinement.

## Troubleshooting

* **`ImportError: vllm`** during stage 3 — the script automatically falls
  back to `transformers` (with the smaller `Qwen2.5-7B`) and continues.
  Slower (~5×) but works. To force a different backend:
  `VIVEKA_LLM_BACKEND=transformers python scripts/03_extract_concepts.py`.
* **CUDA OOM during stage 3** — vLLM grabs ~55% of VRAM by default. If
  another process is on the GPU, lower it: edit `llm_client.VLLMClient` →
  `gpu_memory_utilization=0.45`.
* **CUDA OOM during stage 2 then 3** — the embedder is unloaded before the
  LLM loads, so this shouldn't happen. If it does, check no other process
  is holding the GPU: `nvidia-smi`.
* **Streamlit "model not found"** — verify `index_data/meta.sqlite` exists
  and stage 5 marked complete: `sqlite3 index_data/meta.sqlite "SELECT *
  FROM pipeline_state"`.

## License

The software is MIT-licensed — see [LICENSE](LICENSE). Third-party and
corpus-text terms are set out in [NOTICE.md](NOTICE.md).

The corpus texts are **not** distributed here. Vivekananda died in 1902 and
his own words are in the public domain, but the modern published editions —
and in particular the Bengali renderings of English lectures, which are the
work of later translators — carry editorial and translational rights held by
their publishers. Obtain the source texts through the publishers' own
channels and drop them into `data/` (step 7 of Setup); the pipeline
regenerates every derived layer from them.

## Citation

If you use this software, please cite it — see [CITATION.cff](CITATION.cff).
