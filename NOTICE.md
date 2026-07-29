# Notice on third-party content

The [MIT licence](LICENSE) in this repository covers **the software only**.

## Corpus texts are not distributed here

The reference deployment indexes the works of Swami Vivekananda. Those texts
are **not** included in this repository, and `data/` is deliberately empty.

Vivekananda died in 1902 and his own words are in the public domain. However,
the modern published editions this project parses — the English *Complete
Works* and the Bengali *Vani o Rachana* — carry editorial rights held by their
publishers, and the Bengali renderings of English lectures are the work of
later translators, whose translations attract their own copyright term.

Users must obtain the source texts through the publishers' own channels and
place them in `data/` (see step 7 of Setup in [README.md](README.md)). The
pipeline regenerates every derived layer from them.

## Derived layers

The layers produced by the pipeline — the structural parse, concept graph,
alias inventory, entity layer, and the human-annotated evaluation set — are
released separately under CC BY 4.0. See the resource paper under
`docs/paper/` for their description and the archive location.

## Vendored front-end libraries

`webapp/static/vendor/` contains third-party JavaScript, vendored so the
concept-graph view works without a CDN. These retain their own licences:

- **Cytoscape.js** — MIT. © 2016–2024 The Cytoscape Consortium; bundles
  code © 2013–2014 Ralf S. Engelschall.
- **layout-base** — Apache License 2.0. © i-Vis Research Group, Bilkent
  University, 2007–present. The full Apache-2.0 text is retained inside the
  distributed file, as that licence requires.
- **cose-base**, **cytoscape-fcose** — from the same i-Vis Research Group
  (Bilkent University) family as `layout-base`. The minified bundles carry no
  licence header, so their terms are not asserted here.
  <!-- TODO: confirm from the upstream package.json / repository and record
       the exact licence for each, then delete this note. -->

Apache-2.0 imposes attribution obligations (retain notices, state
modifications). None of these files has been modified.

## Human annotations

The judgment files under `docs/paper/eval/annotations/` are published
under the anonymous identifiers `annotator_a`, `annotator_b` and
`annotator_c`, corresponding to the annotator labels used in the resource
paper.
