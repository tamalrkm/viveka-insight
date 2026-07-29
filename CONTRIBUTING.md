# Contributing to viveka-insight

Contributions are welcome — bug reports, corpus/parser fixes, new language
support, and improvements to the concept-extraction prompt in particular.

## Getting support / asking questions

Open a [GitHub issue](../../issues) with the `question` label. For anything
about deploying the stack on your own corpus, please say which languages and
roughly how many paragraphs you have — the answer usually depends on both.

## Reporting a bug

Open an issue including:

- what you ran (the exact command or pipeline stage),
- what happened, and what you expected,
- Python version, OS, GPU (or CPU-only), and whether you are on the vLLM or
  the Transformers backend,
- the relevant traceback in full.

`README.md` has a Troubleshooting section covering the failures we see most
often (CUDA OOM between stages, missing NLTK data, index/DB out of sync).
Please check it first.

## Proposing a change

1. Open an issue describing the problem before writing a large patch, so we
   can agree on the approach.
2. Fork, branch from `main`, and keep the change focused.
3. Add or update tests (see below). New behaviour needs a test.
4. Run the full suite and make sure it passes.
5. Open a pull request describing what changed and why, and link the issue.

## Running the tests

```bash
source .venv/bin/activate
python -m pytest tests/ -q          # 74 tests, ~11 s, CPU-only, no GPU needed
```

The suite deliberately requires no GPU, no model downloads and no corpus, so
it runs anywhere. Please keep it that way: if a new test needs a model, mark
it and skip by default.

To exercise the *pipeline* rather than the units — useful when changing a
parser or a pipeline stage — run the synthetic sample end to end:

```bash
python scripts/build_all.py --sample     # CPU, <1 min, no GPU, no corpus
```

If you change a parser, update `sample_data/` to match and keep
`tests/test_sample_corpus.py` passing; that path is how reviewers and new
contributors verify the project works at all.

## Things to know before changing the pipeline

These invariants are load-bearing; breaking them silently corrupts an index.
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) documents them in full, but in
short:

- **Concept canonical labels are English, lowercase, hyphenated.** This is the
  cross-lingual key. Do not add per-language concept tables.
- **`db.normalize_text()` is the single source of truth** for text keys. Do
  not normalize text differently anywhere else.
- **Do not bump `PROMPT_VERSION`** in `viveka_insight/concept_extraction.py`
  unless the prompt or output schema actually changed. It invalidates the
  warm-start cache and forces a multi-hour re-extraction.
- **Run `00_snapshot` before `01_parse --force`**, or you lose the warm-start
  cache.
- Paragraph text is owned by the source HTML in `data/`, never edited via SQL.

## Code style

Follow the surrounding code: standard library preferred for evaluation
scripts, type hints on new public functions, and docstrings that say *why*
rather than restating the signature.

## Code of conduct

By participating you agree to abide by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
