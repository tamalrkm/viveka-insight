/* Concept-graph renderer, run inside a Streamlit custom-component iframe.
 *
 * Streamlit sends us a `streamlit:render` message carrying `args.spec` (built
 * by viveka_insight.graph_html.build_graph_spec). We answer with
 * `streamlit:setComponentValue` whenever the user taps an edge, so Python can
 * fetch the passages the two concepts share. Node taps and the index panel
 * stay local — they must not trigger a server rerun.
 */

// ── Streamlit component protocol ─────────────────────────────────────────
function post(msg) {
  window.parent.postMessage(Object.assign({ isStreamlitMessage: true }, msg), '*');
}
function setValue(value) {
  post({ type: 'streamlit:setComponentValue', value: value, dataType: 'json' });
}
function setFrameHeight(h) {
  post({ type: 'streamlit:setFrameHeight', height: h });
}

// A tap on the *same* edge twice must still reach Python. Streamlit only
// reruns when the component value changes, so carry a monotonic counter.
let seq = 0;
let cy = null;
let renderedKey = null;

window.addEventListener('message', (event) => {
  const data = event.data;
  if (!data || data.type !== 'streamlit:render') return;
  const args = data.args || {};
  // Re-rendering on every rerun would reset zoom/pan and re-run the layout
  // whenever any unrelated widget changed. Only rebuild when the spec differs.
  if (args.spec_key === renderedKey) return;
  renderedKey = args.spec_key;
  render(args.spec, args.height || 680);
});

post({ type: 'streamlit:componentReady', apiVersion: 1 });

// ── Overlap removal ──────────────────────────────────────────────────────
// Exercised directly by tests/test_graph_view.py under headless cytoscape, so
// the geometry that is verified is the geometry that ships. fcose lays out
// structure but a dense subgraph still settles with discs on top of each other
// (its ideal-edge-length attraction beats node repulsion once a handful of
// concepts share hundreds of edges). This is a plain relaxation pass: push
// every overlapping pair apart along their centre line until none intersect.
function separate(cy, pad) {
  const ns = cy.nodes(), n = ns.length;
  if (n < 2) return;
  const iters = n > 300 ? 40 : 150;
  for (let it = 0; it < iters; it++) {
    let moved = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      const a = ns[i], b = ns[j], pa = a.position(), pb = b.position();
      let dx = pb.x - pa.x, dy = pb.y - pa.y, d = Math.hypot(dx, dy);
      const need = (a.width() + b.width()) / 2 + pad;
      if (d >= need) continue;
      if (d < 1e-6) { dx = Math.cos(i * 2.4); dy = Math.sin(i * 2.4); d = 1; }
      const push = (need - d) / 2, ux = dx / d, uy = dy / d;
      a.position({ x: pa.x - ux * push, y: pa.y - uy * push });
      b.position({ x: pb.x + ux * push, y: pb.y + uy * push });
      moved++;
    }
    if (!moved) break;
  }
}

// ── Render ───────────────────────────────────────────────────────────────
function esc(s) {
  return String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}
function det(html) { const el = document.getElementById('detail'); el.innerHTML = html; el.style.display = 'block'; }

function render(DATA, height) {
  document.getElementById('wrap').style.height = height + 'px';
  setFrameHeight(height + 10);
  if (cy) cy.destroy();

  try { cytoscape.use(window.cytoscapeFcose); } catch (e) {}
  cy = cytoscape({
    container: document.getElementById('cy'),
    elements: { nodes: DATA.nodes, edges: DATA.edges },
    layout: { name: 'preset' },
    minZoom: 0.1, maxZoom: 8, wheelSensitivity: 0.3,
    style: [
      { selector:'node', style:{
          'label':'data(label)','background-color':'data(color)',
          'width':'data(size)','height':'data(size)','font-size':'9px',
          'color':'#2c2419','text-valign':'bottom','text-margin-y':2,
          'text-outline-color':'#fbfaf7','text-outline-width':2.2,
          'text-wrap':'wrap','text-max-width':'90px' } },
      { selector:'node[cooc > 0]', style:{ 'border-width':2,'border-color':'#8a7f6a','border-style':'dashed' } },
      { selector:'node[sel = 1]', style:{ 'border-width':3,'border-color':'#c0392b','border-style':'solid','font-size':'12px','z-index':99 } },
      { selector:'edge[relation = "co-occurs"]', style:{
          'width':'mapData(weight,0,1,1,6)','line-color':'#d9c7a3','curve-style':'bezier','opacity':0.5 } },
      { selector:'edge[relation = "similar"]', style:{
          'width':'mapData(weight,0,1,1,4)','line-color':'#9fb6c4','line-style':'dashed','curve-style':'bezier','opacity':0.55 } },
      { selector:'edge.picked', style:{ 'line-color':'#c0392b','opacity':1,'width':4,'z-index':98 } },
      { selector:'.faded', style:{ 'opacity':0.08 } },
      { selector:'.hi', style:{ 'opacity':1 } },
      { selector:'node:selected', style:{ 'border-width':3,'border-color':'#111' } },
    ],
  });

  // Fit, but never zoom past 1:1 — a 3-node subgraph fitted to the canvas
  // would otherwise render as three colliding discs.
  function fitGraph() {
    cy.fit(undefined, 40);
    if (cy.zoom() > 1.0) { cy.zoom(1.0); cy.center(); }
  }

  if (DATA.relayout && cy.nodes().length) {
    const n = cy.nodes().length;
    // fcose is O(dense) and these subgraphs are dense — 53 nodes carry ~900
    // edges. Measured under headless cytoscape: 53 nodes ≈ 0.7 s, but 176
    // nodes ≈ 7.6 s at quality 'default' (21 s at 'proof'), which would lock
    // the iframe. Above the cutoff the preset spring coordinates are already
    // spread out (they were laid out over the whole graph), so separate()
    // alone suffices — and costs ~0.7 s at 176 nodes.
    if (n <= 60) {
      try {
        // randomize:false seeds from the preset coordinates, so the filtered
        // view keeps the full graph's orientation instead of jumping around.
        cy.layout({
          name: 'fcose', quality: 'default', randomize: false, animate: false,
          nodeDimensionsIncludeLabels: true, packComponents: true,
          idealEdgeLength: n <= 12 ? 190 : 150,
          nodeRepulsion: 45000, nodeSeparation: 140,
          gravity: 0.05, gravityRange: 3.8,
        }).run();
      } catch (e) { /* fcose missing → keep preset; separate() still fixes overlap */ }
    }
    separate(cy, 18);   // the actual no-overlap guarantee
    fitGraph();         // animate:false ⇒ the layout has already settled
  } else {
    fitGraph();
  }

  // legend (clusters + edge types)
  const lg = document.getElementById('legend');
  lg.innerHTML = DATA.legend.map(l =>
    `<div class="row"><span class="sw" style="background:${l.color}"></span>${esc(l.label)}</div>`).join('')
    + `<div class="row"><span class="el" style="border-color:#c8b389"></span>co-occurs</div>`
    + `<div class="row"><span class="el" style="border-color:#9fb6c4;border-top-style:dashed"></span>similar</div>`
    + (DATA.has_sel ? `<div class="row"><span class="dot" style="border:3px solid #c0392b"></span>selected</div>` : '')
    + (DATA.has_cooc ? `<div class="row"><span class="dot" style="border:2px dashed #8a7f6a"></span>co-occurring</div>` : '');

  // right-side all-nodes index, grouped by cluster
  function renderGroups(filter) {
    const f = (filter || '').toLowerCase();
    const html = DATA.groups.map(g => {
      const items = g.items.filter(it => !f || it.label.toLowerCase().includes(f));
      if (!items.length) return '';
      return `<div class="grp-h"><span class="sw" style="background:${g.color}"></span>${esc(g.label)}</div>`
        + items.map(it => `<div class="item${it.sel ? ' sel' : ''}" data-id="${it.id}">${esc(it.label)} `
            + `<span class="n">${it.n}</span>`
            + (it.cooc ? ` <span class="co">·${it.cooc}¶</span>` : '')
            + `</div>`).join('');
    }).join('');
    document.getElementById('groups').innerHTML = html;
    document.querySelectorAll('.item').forEach(el =>
      el.onclick = () => focusNode(el.getAttribute('data-id')));
  }
  document.getElementById('q').oninput = e => renderGroups(e.target.value);
  renderGroups('');

  document.getElementById('toggle').onclick = () => {
    const hidden = document.body.classList.toggle('sidehidden');
    document.getElementById('toggle').textContent = hidden ? '☰ index' : '✕ index';
    cy.resize(); fitGraph();
  };

  function focusNode(id) {
    const n = cy.getElementById(id);
    if (!n.length) return;
    cy.animate({ center:{eles:n}, zoom: Math.max(1.2, cy.zoom()) }, { duration:300 });
    showNode(n);
    const nb = n.closedNeighborhood();
    cy.elements().addClass('faded').removeClass('hi');
    nb.removeClass('faded').addClass('hi');
  }
  function clearFocus(){ cy.elements().removeClass('faded hi'); }

  function showNode(n){
    const d = n.data();
    let h = `<b class="big">${esc(d.label)}</b> &nbsp;<span style="color:#888">${d.n} mentions</span>`;
    if (d.sel) h += ` &nbsp;<span class="chip" style="background:#fbe4e0">selected</span>`;
    else if (d.cooc) h += ` &nbsp;<span class="chip">shares ${d.cooc} paragraph${d.cooc>1?'s':''} with your selection</span>`;
    if (d.aliases_en) h += `<div>EN: ${esc(d.aliases_en)}</div>`;
    if (d.aliases_bn) h += `<div class="bn">BN: ${esc(d.aliases_bn)}</div>`;
    const nbrs = n.neighborhood('node').map(x=>x.data('label')).slice(0,10);
    if (nbrs.length) h += `<div style="margin-top:3px">${nbrs.map(x=>`<span class="chip">${esc(x)}</span>`).join('')}</div>`;
    det(h);
  }

  cy.on('tap','node', e => focusNode(e.target.id()));

  cy.on('tap','edge', e => {
    const edge = e.target, d = edge.data();
    const a = cy.getElementById(d.source), b = cy.getElementById(d.target);
    const why = d.relation === 'similar'
      ? 'semantically similar (close in meaning-space)'
      : 'discussed together in the same passages';
    det(`<b>${esc(a.data('label'))}</b> ↔ <b>${esc(b.data('label'))}</b>`
      + `<br><span style="color:#666">${d.relation} — ${why} · strength ${(+d.weight).toFixed(2)}</span>`
      + `<span class="hint">passages for this pair are loading below ↓</span>`);
    cy.edges().removeClass('picked');
    edge.addClass('picked');
    // Highlight just the pair, so it is obvious which two concepts are showing.
    cy.elements().addClass('faded').removeClass('hi');
    a.union(b).union(edge).removeClass('faded').addClass('hi');
    setValue({ kind:'edge', seq: ++seq,
               source: parseInt(d.source, 10), target: parseInt(d.target, 10),
               relation: d.relation, weight: d.weight });
  });

  cy.on('tap', e => {
    if (e.target !== cy) return;
    clearFocus();
    cy.edges().removeClass('picked');
    document.getElementById('detail').style.display = 'none';
    setValue({ kind:'clear', seq: ++seq });
  });
}
