# viveka-insight — setup on a new machine (with `uv`)

> Instructions for getting this project running after
> unzipping it on a fresh system that uses **`uv`** for Python.
> This is a transfer companion to `README.md` / `docs/ARCHITECTURE.md` — read those two
> for what the project *is*; read this file for how to *stand it up here*.

## What's in this archive

| Path | Size | Notes |
|------|------|-------|
| `viveka_insight/`, `scripts/`, `webapp/`, `tests/` | ~0.5 MB | all source code |
| `data/` | 47 MB | source HTMLs (EN + BN) + `all.html.bak`. Required for any rebuild. |
| `index_data/` | **1.1 GB** | **prebuilt** `meta.sqlite` (graph + warm-start cache) + 6 FAISS indexes. Ship-ready — the app runs against these with **no rebuild**. |
| `requirements.txt`, `pyproject.toml`, `Makefile`, `run.sh` | — | build/run config |

**Deliberately NOT included** (regenerate locally): `.venv/` (7.7 GB),
`__pycache__/`, `.pytest_cache/`.

Because `index_data/` is included, **you do not need a GPU or a rebuild to run
the search app.** The expensive stage (LLM concept extraction, ~3–4 hr on an
A100) is already baked into `meta.sqlite`. Only rebuild if you change the
source HTML or the pipeline (see "Rebuilding" below).

---

## 1. Create the environment with `uv`

Target is Python 3.11 (what it was built on; 3.10 also fine).

```bash
cd viveka-insight
uv venv .venv --python 3.11 --seed
source .venv/bin/activate
```

Pick **one** of the two dependency profiles below.

### Profile A — Serve only (run the search webapp) ← start here

This is all you need to launch the UI against the prebuilt `index_data/`.
It does **not** install vLLM (the heavy, CUDA-pinned piece), so it works on a
CPU-only box or any CUDA version.

```bash
uv pip install \
  "torch==2.5.1" "transformers>=4.45,<4.50" "sentence-transformers>=3.0,<4.0" \
  "FlagEmbedding>=1.3.0,<1.4" "faiss-cpu>=1.8.0" "numpy>=1.24,<2.0" "scipy>=1.11" \
  "beautifulsoup4>=4.12" "lxml>=4.9" "nltk>=3.8" "tqdm>=4.66" "streamlit>=1.36" \
  "openai>=1.40"
```

> On a GPU box you can instead let torch pull a CUDA wheel; the pin above gives
> the CPU/whatever-default build, which is enough to embed the query and rerank.

### Profile B — Full rebuild (needs an NVIDIA GPU + CUDA 12.x)

Only if you intend to re-run the pipeline (stage 3 LLM extraction). Installs
vLLM, which is pinned against `torch 2.5.1 + cu124`.

```bash
uv pip install -r requirements.txt --index-strategy unsafe-best-match
```

`requirements.txt` has notes on the vLLM/torch/CUDA matrix. If vLLM isn't
available or the GPU is absent, the pipeline auto-falls back to
`transformers` + Qwen2.5-7B (`VIVEKA_LLM_BACKEND=transformers`), which is far
slower — avoid unless you must.

---

## 2. First-run model downloads (needs internet)

On first launch the app downloads two models from HuggingFace into the local
HF cache (`~/.cache/huggingface`):

- `BAAI/bge-m3` — embedder (~2.3 GB)
- `BAAI/bge-reranker-v2-m3` — reranker (~2.3 GB)

NLTK's `punkt` tokenizer is fetched lazily on demand (`punkt_tab`), so no manual
step is needed, but it also requires internet on first use.

If this machine is offline, pre-stage the HF cache from a connected box, or set
`HF_HOME` / `TRANSFORMERS_OFFLINE=1` accordingly.

---

## 3. Run the search app

```bash
source .venv/bin/activate
bash run.sh        # == streamlit run webapp/app.py --server.address 0.0.0.0 --server.port 8501
```

Then open `http://<this-host>:8501`.

**CPU-only machine?** The default device is `cuda`. Force CPU with:

```bash
VIVEKA_DEVICE=cpu bash run.sh
```

Query embedding + reranking on CPU works but each search takes a few seconds
instead of sub-second. Everything else (FAISS flat-IP lookups, the SQLite
graph) is CPU-native already.

Sanity check without the UI:

```bash
python -m pytest tests/ -x -q     # smoke tests, ~40 s, no GPU
```

---

## 4. Rebuilding the index (only if you change source HTML or the pipeline)

Needs **Profile B** (GPU). The warm-start cache in `meta.sqlite` means a
rebuild after a content edit is ~10–15 min, **not** 3–4 hr — as long as you
don't wipe the cache.

```bash
python scripts/build_all.py       # 00_snapshot → 01_parse → 02_embed → 02b_restore
                                  # → 03_extract_concepts → 04_link → 05_build_graph
```

**Read `docs/ARCHITECTURE.md` before rebuilding.** The one rule that matters:
`00_snapshot` must run before any `01_parse --force` or you lose the warm-start
cache and pay the full stage-3 cost. `build_all.py` orders this correctly; only
manual stage runs risk it.

---

## 5. Notes that carry over from the old machine

- **Deep-links** in search results point at the user's institutional Apache
  server (`https://cs.rkmvu.ac.in/~tamal/CWSV/...` for EN,
  `.../vani_rachana/all.html` for BN). These are absolute public URLs and work
  from anywhere with internet — no local hosting needed.
- **Config**: everything lives in `viveka_insight/config.py`, overridable via
  `VIVEKA_*` env vars (`VIVEKA_DEVICE`, `VIVEKA_LLM_BACKEND`, ...).
- **Do not** commit or re-zip `.venv/`, `__pycache__/`, or `.pytest_cache/`.

## TL;DR

```bash
cd viveka-insight
uv venv .venv --python 3.11 --seed && source .venv/bin/activate
# Serve-only (no GPU needed — index is prebuilt):
uv pip install torch==2.5.1 "transformers>=4.45,<4.50" "sentence-transformers>=3.0,<4.0" \
  "FlagEmbedding>=1.3.0,<1.4" faiss-cpu "numpy<2.0" scipy beautifulsoup4 lxml nltk tqdm streamlit openai
VIVEKA_DEVICE=cpu bash run.sh      # drop VIVEKA_DEVICE on a GPU box
```
