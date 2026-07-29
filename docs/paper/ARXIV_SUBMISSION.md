# arXiv submission — viveka-insight resource paper

Everything needed for the arXiv web form. Source bundle: `arxiv/`.

## Pre-upload checklist

arXiv postings are public and permanent: a version can be *replaced* but
never deleted, and v1 stays visible in the version history forever — so the
manuscript must render no red `[...]` placeholders before upload.

**Status: clear.** Both placeholders are resolved and the compiled PDF
contains none.

| Placeholder | Resolved to |
|---|---|
| `[repository URL]` | <https://github.com/tamalrkm/viveka-insight> (2026-07-29) |
| `[archive DOI]` | `10.5281/zenodo.21669974` (2026-07-29) |

One check before uploading: confirm the DOI above is the one you want cited.
Zenodo mints **two** — a *concept* DOI that always resolves to the newest
version, and a *version* DOI pinned to one release. The concept DOI is
usually the better citation in a paper. Both appear on the Zenodo record page.

Regenerate the bundle after fixing:

```bash
cd docs/paper
rm -rf arxiv && mkdir arxiv
cp viveka_insight_lre.tex sn-jnl.cls sn-nature.bst arxiv/
cd arxiv && pdflatex viveka_insight_lre.tex && pdflatex viveka_insight_lre.tex
# check: 0 errors, no red [ ] markers in the PDF
rm -f *.aux *.log *.out *.pdf
cd .. && tar czf arxiv-submission.tar.gz -C arxiv .
```

Upload `arxiv-submission.tar.gz`. Source only — no PDF in the tarball.

## Form fields

**Title**

```
Viveka-Insight: a cross-lingual concept graph and citation-grounded
retrieval resource over the complete works of Swami Vivekananda in
English and Bengali
```

**Authors**

```
Tamal Maharaj
```

**Categories**

- Primary: `cs.CL` (Computation and Language)
- Cross-list: `cs.IR` (Information Retrieval), `cs.DL` (Digital Libraries)

**Comments**

```
18 pages, 9 tables, 1 figure. Resource paper. Code and derived resource
layers released under MIT and CC BY 4.0 respectively.
```

**License** — recommend **CC BY 4.0**, consistent with the CC BY 4.0
release of the derived data layers claimed in §5.3. (The arXiv default
non-exclusive licence would be inconsistent with that claim.)

**Abstract** (plain text — arXiv accepts no LaTeX markup here)

```
Classical spiritual and philosophical corpora pose three compounding
challenges for language resources: they exist in several languages
without parallel alignment, their vocabulary differs sharply from that of
contemporary readers, and any generated text over such culturally
sensitive material must be verifiably grounded in the source. We present
Viveka-Insight, a bilingual resource and accompanying open-source
pipeline for the works of Swami Vivekananda (1863-1902) -- the
nine-volume English Complete Works and the ten-volume Bengali Vani o
Rachana, two related but non-parallel corpora totalling about 15 million
characters. The released resource has four layers: (i) a
structure-preserving parse of both corpora into a volume - chapter -
paragraph - sentence hierarchy (32,694 paragraphs, 168,842 sentences)
with per-paragraph anchors that deep-link back to the published editions;
(ii) a cross-lingual concept graph of 8,362 language-agnostic concepts
carrying 87,518 paragraph-concept edges typed by relation and 55,872
concept-concept edges, in which canonical English labels act as a
string-equality key that links Bengali and English passages with no
parallel data; (iii) a bilingual alias inventory of 60,850 surface forms
(30,053 English, 30,797 Bengali) mapping each concept to its realizations
in both languages; and (iv) a human-annotated evaluation set of 200
paragraph-concept edges judged independently by three annotators,
released with all per-annotator judgments. We document the construction
pipeline, which needs one extraction pass by a mid-sized
instruction-tuned LLM and rebuilds incrementally in 10-15 minutes after
source edits, and report three evaluations: known-item cross-lingual
retrieval over 194 automatically verified rendered lecture pairs
(Recall@10 0.86 in both directions); a 30-question audit of citation
integrity and modern-question bridging; and the human study, which places
concept-extraction precision at 0.60 under strict two-annotator consensus
(Cohen's kappa = 0.61). The study also shows the extractor's confidence
weight is calibrated -- restricting to weight >= 0.8 raises precision to
0.71 while retaining 98% of concept-bearing paragraphs -- and that
precision is markedly lower in Bengali than English (0.54 vs 0.68),
locating the resource's weakness in exactly the half that cross-lingual
access depends on. The design assumes nothing Vivekananda-specific and
transfers to other multilingual classical corpora.
```

## Notes

- The bundle builds standalone: the only non-standard files are
  `sn-jnl.cls` and `sn-nature.bst`, both included. The architecture
  figure is inline TikZ; there are no external images and no `.bib`
  (the bibliography is a manual `thebibliography`), so no `.bbl` is
  needed.
- arXiv runs its own TeX Live. If `sn-jnl.cls` trips its compiler, the
  fallback is the plain-`article` version, `viveka_insight_paper.tex`,
  which needs no custom class.
- Post the arXiv ID back into the JOSS submission and the LRE cover
  letter once it is live.
