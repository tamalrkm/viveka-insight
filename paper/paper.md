---
title: 'Viveka-Insight: a pipeline for cross-lingual concept-graph retrieval over non-parallel multilingual corpora'
tags:
  - Python
  - information retrieval
  - retrieval-augmented generation
  - knowledge graphs
  - multilingual NLP
  - Bengali
  - digital humanities
authors:
  - name: Tamal Maharaj
    orcid: 0009-0001-5835-8967
    affiliation: 1
affiliations:
  - name: Ramakrishna Mission Vivekananda Educational and Research Institute, Belur Math, West Bengal, India
    index: 1
date: 28 July 2026
bibliography: paper.bib
---

# Summary

`viveka-insight` is a Python pipeline for making large multilingual text
collections searchable by meaning when no translation alignment between the
languages exists. Its central construct is a **cross-lingual concept graph**:
an instruction-tuned large language model reads every paragraph in every
language and emits concepts under *canonical English, lowercase, hyphenated*
labels. Because the label space is shared across languages, string equality on
a label links a Bengali passage to an English one with no parallel data, no
translation step, and no bilingual dictionary.

Around that graph the package provides a complete retrieval and
question-answering stack: multi-granular dense indexing at sentence, paragraph
and chapter level in a single multilingual embedding space [@chen2024bgem3];
a concept-mediated retrieval path that expands a query one hop along the graph;
weighted reciprocal rank fusion [@cormack2009rrf] of the two paths; reranking
with a multilingual cross-encoder; and a citation-grounded answering layer in
which every generated claim carries a citation rewritten into a deep link to
the exact source paragraph, with markers that do not correspond to a retrieved
passage deleted before the reader sees them.

The software ships a reference deployment over the works of Swami Vivekananda
(nine English volumes and ten Bengali; approximately 15 million characters,
32,694 paragraphs), which yields a graph of 8,362 concepts, 87,518
paragraph--concept edges and 55,872 concept--concept edges. Nothing in the
pipeline is specific to that corpus: parsers, sentence-splitting rules, models,
languages, and fusion weights are configuration.

# Statement of need

Digitized editions of classical, religious and philosophical literature are now
widely available, but access to them remains largely lexical. Two problems
recur across such collections and are poorly served by existing tooling.

First, **the languages are not aligned**. Standard cross-lingual retrieval
assumes translation pairs or a parallel corpus. Many real collections have
neither: a body of work may exist in two languages that overlap thematically
while being independent compositions, so a reader fluent in one language has no
route into the other half of the material. Multilingual encoders alone go some
way, but supply no inspectable link structure and no explanation of *why* a
cross-language result was returned.

Second, **the vocabulary is historically distant**. Contemporary users bring
contemporary questions to authors who wrote a century or more ago. The words of
those questions do not occur in the corpus, so dense retrieval underperforms,
even though the corpus treats the underlying phenomena under different names.

Existing graph-augmented retrieval tooling does not close these gaps.
GraphRAG [@edge2024graphrag] builds an *entity* graph and uses community
summaries for query-focused summarization; entity-centric graphs over
discursive prose capture proper nouns and miss the argument, and the approach
is not designed to bridge languages. General RAG frameworks
[@lewis2020rag] provide retrieval and generation but leave cross-lingual
linking, provenance and citation verification to the user.

`viveka-insight` addresses both gaps in reusable form, and adds the
engineering that makes such a resource maintainable rather than a
one-off artifact:

- **Warm-start incremental rebuilds.** LLM concept extraction is the only
  expensive stage (hours on a single GPU); every other stage takes minutes.
  The pipeline snapshots extraction outputs keyed by (normalized paragraph
  text, language, extractor version) before any re-parse and restores them
  onto fresh rows afterwards, so only genuinely new or changed paragraphs
  reach the LLM. Correcting OCR errors or restructuring a source document
  costs 10--15 minutes instead of a full rebuild --- in practice the
  difference between a corpus that can be maintained and one that is frozen.
- **Verifiable provenance.** Every unit retains the HTML anchor of its
  paragraph in the published edition, so any row in any layer resolves to a
  URL opening the exact passage. Citation markers emitted by the answering
  model are validated against the retrieved set and unsupported ones removed.
- **Annotation and agreement tooling.** A small web interface collects
  independent human judgments of extracted graph edges, and a companion script
  computes precision, Cohen's $\kappa$, Gwet's AC$_1$, bootstrap and Wilson
  intervals, McNemar and Fisher tests, and breakdowns by language, relation and
  extractor confidence --- using only the standard library.

The stack runs entirely on-premises with no external service dependency at
query time, which matters for institutions that cannot send culturally
sensitive material to third-party APIs. Storage is SQLite plus exact FAISS
indexes [@johnson2019faiss], so a complete deployment is a directory of files.
Concept extraction uses Qwen2.5 [@qwen25] served with vLLM [@kwon2023vllm],
with automatic fallback to a Transformers backend [@wolf2020transformers];
any OpenAI-compatible endpoint can be substituted by configuration.

# Research applications

The package was built for, and is used in, research on retrieval over
non-parallel bilingual corpora; a companion paper reports evaluations of the
reference deployment, including known-item cross-lingual retrieval over 194
verified rendered lecture pairs, a citation-integrity audit, and a
three-annotator human study of concept-extraction precision. The evaluation
scripts that produce those numbers are shipped in the repository, and the
annotation tooling generalizes to any study of LLM-extracted graph quality.
The software is deployed as a public-facing search and question-answering
service at the authors' institution.

Correctness is covered by 74 CPU-only tests spanning the parsers, database
schema, extraction-output parsing, rank fusion, query-bridging fallbacks,
prompt budgeting, citation linkification and the concept-graph view.

# Acknowledgements

Facilities utilized in this research were supported by the Fund for Improvement
of S&T Infrastructure (FIST) program of the Department of Science and
Technology (DST), India [Sanction No.: SR/FST/MS-I/2022/116]. The author thanks
the volunteer annotators who judged the concept-precision sample.

# References
