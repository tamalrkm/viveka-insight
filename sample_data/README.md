# Sample corpus — synthetic test fixture

Two tiny HTML files that let the **whole pipeline run end to end** without the
real corpus, a GPU, or any LLM download:

```bash
python scripts/build_all.py --sample
```

## This is not Vivekananda

**Every paragraph here was written for this repository.** Nothing in
`sample_en.html` or `sample_bn.html` is a quotation from, a translation of, or
an accurate paraphrase of anything Swami Vivekananda wrote or said. The prose
imitates the register and subject matter of the real corpus so that the
extraction and linking stages have something meaningful to chew on — nothing
more. **Do not cite it, quote it, or treat it as source material.**

Two reasons the fixture is synthetic rather than excerpted:

1. **Rights.** The Bengali *Vani o Rachana* renderings of English lectures are
   the work of later translators and are not in the public domain, so they
   cannot be redistributed here (see [`NOTICE.md`](../NOTICE.md)).
2. **Accuracy.** Reproducing real passages from memory risks putting words in
   the mouth of a religious figure. Avoiding that is the whole point of this
   project's citation discipline; it would be perverse to violate it in the
   test fixture.

## What it exercises

| | English | Bengali |
|---|---|---|
| Volumes | 2 | 2 |
| Chapters | 5 | 5 |
| Paragraphs | 20 | 20 |

The files reproduce the *structure* of the real exports, which is what the
parsers are specific to:

- **English** — flat siblings under `<body>`: `h2.volume-title`,
  `div.section-head`, `div.chapter` containing `h3.chapter-title`, a skipped
  `p.chapter-breadcrumb`, and `div.chapter-content` holding the paragraphs.
- **Bengali** — `<h3>` volume markers matching `স্বামী বিবেকানন্দ সমগ্র খণ্ড N`,
  then `section.content-block[id=content-N]` wrappers, each with a
  `div.scroller` holding an `<h2>` chapter title and the paragraphs.

Every `<p>` carries an `id="p-N"` anchor, so the deep-linking path is
exercised too.

Themes are deliberately mirrored across the two languages — concentration,
renunciation, work, the self, maya, devotion, strength — so the cross-lingual
concept layer has something to link. In a sample build all 12 extracted
concepts bridge English and Bengali.

## What `--sample` does

- reads these two files instead of `data/`,
- writes to `index_data_sample/`, so **your real index is never touched**,
- uses `VIVEKA_LLM_BACKEND=stub` — a deterministic keyword matcher, *not* a
  language model, so no GPU and no multi-gigabyte LLM download,
- defaults to `VIVEKA_DEVICE=cpu`.

The embedding stage still uses the real BGE-M3 model (~2.2 GB on first run,
cached afterwards), because embeddings are what make the retrieval
demonstration meaningful. The whole build takes well under a minute on CPU
once that model is cached.

The resulting concept graph is **illustrative only**. It shows that the
pipeline runs; it says nothing about extraction quality, and it must not be
searched, evaluated or published as though it were real output.
