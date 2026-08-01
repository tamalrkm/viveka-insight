# JOSS readiness — criteria vs. what's actually in the repo

Assessed 2026-07-28, updated 2026-08-01, against the JOSS reviewer checklist
and <https://joss.readthedocs.io/en/latest/submitting.html>. Status is honest,
not aspirational: ✅ done, ⚠️ partial/risk, ❌ blocker.

## ⛔ DO NOT SUBMIT YET — scope gate not met

JOSS lists as **out of scope**: *"Software without public development history
(minimum six months prior to submission)."*

| | |
|---|---|
| Repository made public | **2026-07-29** |
| Public history as of 2026-08-01 | **3 days** |
| Earliest eligible submission | **~2027-01-29** |

The code itself dates from May 2026 and the first local commit from
2026-07-04, but neither was public, and JOSS's wording is about *public*
history. Submitting now invites a desk rejection on scope, which is a poor
first impression with an editor and is recorded publicly.

Nothing else blocks: every other criterion below passes. The action is to
wait and accrue the record, not to change the software.

### What to do in the six months

1. **Land the companion paper.** JOSS weighs *"evidence of publications or
   analyses using the software"* — an accepted or at least posted resource
   paper is the single strongest signal available here.
2. **Keep committing in public.** Real, dated development history is exactly
   what the rule asks for; it accrues by itself if work continues.
3. **Find an external user.** *"External adopters or integrations"* is
   named explicitly. Even one other group indexing a different corpus with
   the pipeline would answer the weakest remaining criterion.
4. **Leave issues and PRs visible** rather than working only locally —
   *"pathways for community contribution"* is assessed from the repo.

## Scope and significance

| Criterion | Status | Notes |
|---|---|---|
| Public development history ≥ 6 months | ❌ | 3 days. The only hard blocker |
| Not a minor utility / thin client / single function | ✅ | 7,613 lines; full parse → embed → extract → link → graph → serve pipeline |
| Not a pre-trained model or notebook | ✅ | |
| Not feature-incomplete | ✅ | Deployed and serving; 74 tests |
| Not a web tool lacking a core library | ✅ | Streamlit UI sits on top of an importable library |
| Paper is about the software, not new research results | ✅ | `paper/paper.md` describes the pipeline; the resource paper and its findings are a separate submission |
| Design thinking / architectural trade-offs | ✅ | `docs/ARCHITECTURE.md` documents invariants, the cost model and why SQLite over a graph DB |
| Installability, understandability, testability | ✅ | `--sample` runs the whole pipeline on CPU in <1 min |
| Research impact: publications or analyses using it | ⚠️ | Companion paper under submission; deployed at the authors' institution |
| External adopters or integrations | ⚠️ | None yet — the weakest criterion after the history gate |
| Collaborative development | ⚠️ | Single contributor |

## General checks


| Item | Status | Notes |
|---|---|---|
| OSI-approved LICENSE, plain text, in repo | ✅ | MIT, verbatim text so GitHub's detector recognises it. Corpus-text and third-party terms live in `NOTICE.md` — appending them to `LICENSE` made GitHub report the licence as "Other", which JOSS reviewers check |
| Submitting author made major contribution | ✅ | Sole author |
| Substantial scholarly effort | ✅ | 7,613 lines of Python (`viveka_insight` 4,166; scripts; webapp). Far above the ~1,000-line informal floor, and not a thin wrapper — the concept layer, warm-start cache and citation validator are original |
| Not a minor utility | ✅ | Full pipeline: parse → embed → extract → link → graph → serve |
| Human-subjects considerations | ✅ | Annotators contributed expert judgments as acknowledged contributors, not as research participants; no personal data. Stated in the companion paper |

## Documentation

| Item | Status | Notes |
|---|---|---|
| Statement of need | ✅ | Added to `README.md`; also §Statement of need in `paper/paper.md` |
| Installation instructions | ✅ | `README.md` Setup (7 steps) plus `SETUP_NEW_MACHINE.md` |
| Example usage | ✅ | Build the index / Run the webapp sections |
| Automated tests | ✅ | 74 tests, ~11 s, CPU-only, all passing. Documented in README |
| Community guidelines | ✅ | `CONTRIBUTING.md` (report / ask / contribute) + `CODE_OF_CONDUCT.md` |
| Functionality documentation (API) | ⚠️ | No generated API reference. `README.md` "How retrieval actually works" and `docs/ARCHITECTURE.md` cover architecture and invariants well, and modules are docstringed, but there is no rendered API doc. Reviewers sometimes ask |

## Functionality — the main acceptance risk

| Item | Status | Notes |
|---|---|---|
| Reviewer can install | ⚠️ | Full install is heavy (CUDA 12.4, torch, vLLM, two BGE models). The `--sample` path needs only torch + the embedder on CPU |
| Reviewer can run the tests | ✅ | CPU-only, no GPU, no models, no corpus needed — this is the saving grace |
| Reviewer can run the *pipeline* end to end | ✅ | `python scripts/build_all.py --sample` runs all six stages on the synthetic `sample_data/` corpus — CPU, <1 min, no GPU, no LLM download, writing to `index_data_sample/`. Verified: 40 paragraphs → 12 concepts, 79 paragraph–concept edges, all 12 bridging EN and BN |
| Performance claims verifiable | ⚠️ | Latency and build-time figures still need the real corpus; the sample demonstrates behaviour, not scale |

### Done 2026-07-29 — the former main risk is closed

`sample_data/` ships a **synthetic** 40-paragraph bilingual corpus (not
excerpted: the Bengali editions are not redistributable, and reproducing real
passages from memory would risk misattributing words to Vivekananda). A
`stub` LLM backend — a deterministic keyword matcher, never a model — lets
stage 3 run with no GPU and no download. `tests/test_sample_corpus.py`
(14 tests) guards the fixture against parser drift.

A reviewer can now watch every stage execute and query the result, which was
the one thing they previously could not do.

Optional but well received: a GitHub Actions workflow running `pytest` on
push. Cheap, since the suite is CPU-only and 11 seconds.

## Software paper (`paper/paper.md`)

| Item | Status | Notes |
|---|---|---|
| Length within 250–1000 words | ✅ | 879 words |
| YAML front matter valid | ✅ | Title, tags, author + ORCID, affiliation, date, bibliography all present |
| Summary for a non-specialist | ✅ | |
| Statement of need | ✅ | |
| State of the field / related work | ✅ | Positioned against GraphRAG and general RAG frameworks |
| References with DOIs | ✅ | 8 entries, all cited, all with DOIs where one exists; no unused entries |
| No duplicate publication | ✅ | The JOSS paper is about the *software*; the companion resource paper (LRE/arXiv) is about the corpus and evaluations. JOSS permits this, but mention the companion paper in the submission notes |

## Submission order

1. ~~Create the public repository~~ — done 2026-07-29.
2. ~~Add the sample corpus + `--sample` pipeline path~~ — done 2026-07-29.
3. ~~Deposit derived layers on Zenodo → DOI~~ — done 2026-07-29 (v0.1.0).
4. Fill `[repository URL]` and `[archive DOI]` in the manuscript; update
   `CITATION.cff` (`repository-code`, `url`, `doi`).
5. Post to arXiv.
6. Submit to JOSS, citing the arXiv ID as the companion paper.
