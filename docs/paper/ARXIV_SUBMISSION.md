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
| `[archive DOI]` | `10.5281/zenodo.21669973` — concept DOI (2026-07-29) |

Zenodo mints two DOIs and both are recorded:

- **Concept** `10.5281/zenodo.21669973` — always resolves to the newest release. This is
  what the manuscript and `CITATION.cff` cite.
- **Version** `10.5281/zenodo.21701206` — pinned to v0.1.1. Give this one to JOSS, which
  archives the exact reviewed snapshot.

Regenerate the bundle after fixing:

```bash
cd docs/paper
rm -rf arxiv && mkdir arxiv
cp viveka_insight_lre.tex sn-jnl.cls sn-nature.bst arxiv/
cd arxiv && pdflatex viveka_insight_lre.tex && pdflatex viveka_insight_lre.tex
# check: 0 errors, no red [ ] markers in the PDF
rm -f *.aux *.log *.out *.pdf
cd .. && zip -X arxiv-submission.zip -j arxiv/viveka_insight_lre.tex arxiv/sn-jnl.cls
```

**Do not add `sn-nature.bst`** — the bibliography is a manual
`thebibliography`, so no BibTeX style is used and arXiv flags it as unused.

**Do not reinstate the ORCID in the title block.** `sn-jnl.cls` defines
`\orcidlogo` as `\includegraphics{Orcidlogo.eps}`, an image Springer does not
ship with the class. arXiv's TeX Live has no copy, and expanding it at
`\maketitle` fails the build outright. The manuscript therefore drops the
ORCID from the author line and neutralises `\orcidlogo` defensively in the
preamble. The ORCID is still recorded in `CITATION.cff`, in the Zenodo
deposit, and on the arXiv submission itself (arXiv links ORCID at the account
level, not through the PDF).

Upload `arxiv-submission.zip` (arXiv also accepts `.tar.gz`). Source only —
no PDF, no `.aux`/`.log`. The archive must be **flat**: arXiv extracts into
one working directory and expects the `.tex` at the top level, not nested in
a folder.

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

**Abstract** — arXiv caps this field at **1,920 characters**, which the
manuscript's own abstract exceeds. A trimmed 1,914-character version is kept
verbatim in `abstract_plain.txt`; paste that. It must go in as a **single
line with no hard wrapping** — arXiv preserves newlines and the form mangles
a pre-wrapped block. All numbers and all four resource layers are retained;
only padding was cut, so it differs slightly from the PDF abstract, which is
normal and expected.

## Preprint vs journal layout

`viveka_insight_lre.tex` carries a toggle near the top of the preamble:

```latex
\newif\ifarxiv
\arxivtrue     % wide A4 measure, for arXiv/preprint
% \arxivfalse  % Springer's native geometry, for the LRE submission
```

`sn-jnl` sizes its text block for Springer's trim, which on A4 leaves ~40 mm
side margins (131x195 mm of type on a 210x297 mm page). That is right for the
journal — they impose their own layout at typesetting — but wasteful in a
preprint people read on screen. With `\arxivtrue` the measure becomes
153x242 mm and the paper is 13 pages instead of 18.

**Set `\arxivfalse` before submitting to LRE.** Both settings are verified to
build with 0 errors and 0 overfull boxes.

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
