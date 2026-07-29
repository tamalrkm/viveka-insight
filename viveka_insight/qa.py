"""Ask Vivekananda — grounded question answering over the corpus.

Pipeline (see `ask`):

  1. analyze_question — one LLM call turns the user's question into 2-4
     era-appropriate search queries plus corpus concept labels. This is the
     modern→timeless bridge: "mobile phone addiction" has no direct hit in
     an 1890s corpus, but "attachment of the senses" and "control of the
     mind" do. The LLM chooses from the *actual* nearest concept labels
     (Searcher.search_concepts) so the bridge stays grounded in the graph.

  2. retrieve — multi-query retrieval through the existing Searcher (all
     granularities + concept path), merged and deduped, then one
     cross-encoder rerank of the pool against the ORIGINAL question.

  3. generate_answer — one LLM call over the numbered sources. The prompt
     forbids claims without a [n] citation; `linkify_citations` turns the
     markers into deep links and strips hallucinated numbers.

Every step degrades gracefully: a failed bridge falls back to raw-question
retrieval, a missing reranker falls back to fused scores.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import graph
from .concept_extraction import _find_json_block
from .config import CFG
from .search import Searcher, SearchHit
from .llm_client import LLMClient


# ──────────────────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class QueryPlan:
    timeless_queries: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)   # validated corpus labels
    bridge_note: str = ""              # "" when the question is already timeless
    lang: str = "en"                   # question language: "en" | "bn"
    ok: bool = True                    # False => JSON fallback path was taken
    raw: str = ""                      # raw LLM output, for debugging


@dataclass
class QAResult:
    question: str
    plan: QueryPlan
    hits: List[SearchHit]              # ordered; hits[i] is citation [i+1]
    answer: str                        # markdown with [n] markers


# ──────────────────────────────────────────────────────────────────────────
# Language detection
# ──────────────────────────────────────────────────────────────────────────

_BN_RE = re.compile(r"[ঀ-৿]")


def detect_lang(question: str) -> str:
    """'bn' if the question contains Bengali codepoints, else 'en'."""
    return "bn" if _BN_RE.search(question) else "en"


# ──────────────────────────────────────────────────────────────────────────
# Stage 1 — analyze / bridge
# ──────────────────────────────────────────────────────────────────────────

ANALYSIS_SYSTEM = (
    "You prepare questions for a retrieval system over the Complete Works of "
    "Swami Vivekananda (1863–1902). Modern topics (phones, social media, "
    "careers) must be translated into the timeless concepts he actually wrote "
    "about. Respond with a single JSON object and nothing else."
)

ANALYSIS_PROMPT = """Question: {question}

The corpus is indexed by concepts. The nearest concept labels to this question
in the corpus are:
{labels}

Tasks:
1. Rewrite the question as 2-4 short search queries using vocabulary from
   Vivekananda's era and teaching (e.g., attachment, sense-control, control of
   the mind, renunciation, concentration, habit, self-discipline). Each query
   must be free of modern-only terms. Write the queries in English.
2. Pick up to 6 concept labels from the list above (copy them exactly) that
   best capture what the question is really about.
3. If the question involves anything modern that Vivekananda never saw, write
   one sentence explaining the mapping (e.g., "Mobile-phone addiction is a
   modern form of attachment of the senses and a habit that enslaves the
   mind."). If the question is already timeless, use "" for bridge_note.

Return JSON only, exactly this shape:
{{
  "timeless_queries": ["...", "..."],
  "concepts": ["label-one", "label-two"],
  "bridge_note": "..."
}}"""


def _parse_plan_json(raw: str) -> Optional[dict]:
    block = _find_json_block(raw)
    if not block:
        return None
    try:
        obj = json.loads(block)
    except json.JSONDecodeError:
        try:  # common LLM slip: trailing commas
            obj = json.loads(re.sub(r",\s*([}\]])", r"\1", block))
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


def analyze_question(
    llm: LLMClient,
    searcher: Searcher,
    question: str,
    n_concepts: Optional[int] = None,
) -> QueryPlan:
    """One LLM call producing the modern→timeless bridge. Never raises on a
    malformed LLM reply — falls back to a raw-question plan (ok=False)."""
    lang = detect_lang(question)
    n_concepts = n_concepts or CFG.qa.n_bridge_concepts

    nearest = searcher.search_concepts(question, k=n_concepts)
    offered = [label for _cid, label, _sim in nearest]

    prompt = ANALYSIS_PROMPT.format(
        question=question.strip(),
        labels=", ".join(offered) if offered else "(none available)",
    )
    raw = llm.generate(
        [prompt], system=ANALYSIS_SYSTEM,
        max_tokens=CFG.qa.plan_max_tokens, temperature=0.1,
        max_input_tokens=CFG.qa.max_input_tokens,
    )[0]

    obj = _parse_plan_json(raw)
    if obj is None:
        return QueryPlan(lang=lang, ok=False, raw=raw)

    queries: List[str] = []
    seen = {question.strip().casefold()}
    for q in obj.get("timeless_queries", []) or []:
        if not isinstance(q, str) or not q.strip():
            continue
        key = q.strip().casefold()
        if key in seen:
            continue
        seen.add(key)
        queries.append(q.strip())
    queries = queries[:4]

    # Keep only labels that were actually offered AND exist in the DB.
    # Offered labels are canonical (English, lowercase, hyphenated); normalize
    # the LLM's copies the same way before matching.
    offered_set = set(offered)
    candidates = [
        c.strip().lower().replace(" ", "-")
        for c in (obj.get("concepts", []) or [])
        if isinstance(c, str) and c.strip()
    ]
    candidates = [c for c in dict.fromkeys(candidates) if c in offered_set]
    known = graph.fetch_concepts_by_label(searcher.conn, candidates)
    concepts = [c for c in candidates if c in known][:6]

    note = obj.get("bridge_note", "")
    note = note.strip()[:300] if isinstance(note, str) else ""

    return QueryPlan(
        timeless_queries=queries, concepts=concepts, bridge_note=note,
        lang=lang, ok=True, raw=raw,
    )


# ──────────────────────────────────────────────────────────────────────────
# Stage 2 — retrieve
# ──────────────────────────────────────────────────────────────────────────

def retrieve(
    searcher: Searcher,
    question: str,
    plan: QueryPlan,
    languages: Sequence[str] = ("en", "bn"),
    per_query_k: Optional[int] = None,
    n_sources: Optional[int] = None,
) -> List[SearchHit]:
    """Search with the question plus each timeless query, merge/dedupe by
    (lang, paragraph_id), rerank the pool once against the original question,
    and return the top n_sources hits renumbered rank=1..n (= citation ids)."""
    per_query_k = per_query_k or CFG.qa.per_query_k
    n_sources = n_sources or CFG.qa.n_sources

    queries: List[str] = [question]
    seen_q = {question.strip().casefold()}
    for q in plan.timeless_queries:
        if q.strip().casefold() not in seen_q:
            seen_q.add(q.strip().casefold())
            queries.append(q)

    # Per-query cross-encoding would be wasted work; rerank once at the end.
    pool: Dict[Tuple[str, int], SearchHit] = {}
    for q in queries:
        results = searcher.search(
            q, top_k=per_query_k, languages=tuple(languages), rerank=False,
        )
        for lang_hits in results.values():
            for h in lang_hits:
                key = (h.lang, h.paragraph_id)
                prev = pool.get(key)
                if prev is None:
                    pool[key] = h
                else:
                    if h.score > prev.score:
                        h.via_concepts = list(
                            dict.fromkeys(prev.via_concepts + h.via_concepts)
                        )[:3]
                        pool[key] = h
                    else:
                        prev.via_concepts = list(
                            dict.fromkeys(prev.via_concepts + h.via_concepts)
                        )[:3]

    hits = list(pool.values())
    if not hits:
        return []

    if searcher.reranker is not None:
        scores = searcher.reranker.score(
            question, [h.text for h in hits], batch_size=32,
        )
        for h, s in zip(hits, scores):
            h.score = float(s)
    # else: fused RRF scores already on the hits are scale-free and comparable.

    hits.sort(key=lambda h: -h.score)
    hits = hits[:n_sources]
    for i, h in enumerate(hits, start=1):
        h.rank = i
    return hits


# ──────────────────────────────────────────────────────────────────────────
# Stage 3 — generate
# ──────────────────────────────────────────────────────────────────────────

ANSWER_SYSTEM = (
    "You are a careful scholarly assistant. You answer questions using ONLY "
    "the numbered source passages from the Complete Works of Swami "
    "Vivekananda provided by the user. You write in the third person "
    "(\"Vivekananda taught that ...\") and never invent quotes, teachings, or "
    "biographical facts. Every substantive claim carries a bracketed citation "
    "like [3]. If the sources do not answer the question, you say so plainly."
)

ANSWER_PROMPT = """Question: {question}
{bridge_block}
Sources — each begins with its citation number and location:

{sources_block}

Instructions:
- Answer in {answer_language}, the language of the question.
- Ground everything in the sources above. Put the citation [n] immediately
  after the sentence or quote it supports. Use only numbers that appear above.
- Quote Vivekananda's own words (short, in quotation marks, with citation)
  where they are striking or precise.
- {bridge_instruction}
- Write 2-4 paragraphs. No bibliography, no heading — the [n] citations are
  the references.
- If a source is in the other language, translate what you use and note it,
  e.g. (translated from the Bengali).
- If the sources do not really address the question, say so honestly and
  summarize only what they do say.

Now answer the question: {question}"""


def _trim_passage(text: str, matched_sentence: str, max_chars: int) -> str:
    """Trim to ~max_chars. If the matched sentence is present, keep a window
    centered on it (never splitting the sentence); otherwise take the head.
    Cuts land on word boundaries and get ellipses."""
    text = text.strip()
    if len(text) <= max_chars:
        return text

    start, end = 0, max_chars
    if matched_sentence and matched_sentence in text:
        m_start = text.index(matched_sentence)
        m_end = m_start + len(matched_sentence)
        if len(matched_sentence) >= max_chars:
            start, end = m_start, m_end   # never split the matched sentence
        else:
            pad = (max_chars - len(matched_sentence)) // 2
            start = max(0, m_start - pad)
            end = min(len(text), start + max_chars)
            start = max(0, end - max_chars)

    snippet = text[start:end]
    # Offsets of the matched sentence inside the snippet (if it's in there),
    # so word-boundary nudges below never eat into it.
    m_off = None
    if matched_sentence and matched_sentence in text:
        m_off = text.index(matched_sentence) - start
    if start > 0:
        sp = snippet.find(" ")
        if 0 < sp < 40 and (m_off is None or sp < m_off):
            snippet = snippet[sp + 1:]
            if m_off is not None:
                m_off -= sp + 1
        snippet = "… " + snippet
        if m_off is not None:
            m_off += 2
    if end < len(text):
        sp = snippet.rfind(" ")
        m_end_off = (m_off + len(matched_sentence)) if m_off is not None else -1
        if sp > 0 and len(snippet) - sp < 40 and sp > m_end_off:
            snippet = snippet[:sp]
        snippet = snippet + " …"
    return snippet


def build_answer_prompt(
    question: str,
    hits: Sequence[SearchHit],
    plan: QueryPlan,
    max_context_chars: Optional[int] = None,
    max_prompt_chars: Optional[int] = None,
) -> Tuple[str, List[SearchHit]]:
    """Assemble the answer prompt. Returns (prompt, included_hits): sources are
    added best-first until the char budget is hit, and the returned hit list is
    exactly the numbered sources in the prompt — keeping citation numbers, the
    UI reference list, and linkify_citations aligned."""
    max_context_chars = max_context_chars or CFG.qa.max_context_chars
    max_prompt_chars = max_prompt_chars or CFG.qa.max_prompt_chars

    if plan.bridge_note:
        bridge_block = (
            "\nNote: the question uses modern terms Vivekananda never "
            f"encountered. Bridge: {plan.bridge_note}"
        )
        if plan.concepts:
            bridge_block += (
                f" (related concepts in his works: {', '.join(plan.concepts)})"
            )
        bridge_block += "\n"
        bridge_instruction = (
            "Open with one or two sentences bridging the modern situation to "
            "the timeless principle, then answer from the sources."
        )
    else:
        bridge_block = ""
        bridge_instruction = "Answer directly."

    answer_language = "Bengali (বাংলা)" if plan.lang == "bn" else "English"

    # Length of everything except the sources block
    skeleton = ANSWER_PROMPT.format(
        question=question.strip(),
        bridge_block=bridge_block,
        sources_block="",
        answer_language=answer_language,
        bridge_instruction=bridge_instruction,
    )
    budget = max_prompt_chars - len(skeleton)

    blocks: List[str] = []
    included: List[SearchHit] = []
    for h in hits:
        n = len(included) + 1
        passage = _trim_passage(h.text, h.matched_sentence, max_context_chars)
        src_lang = "Bengali" if h.lang == "bn" else "English"
        block = f"[{n}] ({src_lang}) {h.location_str()}\n{passage}"
        if blocks and len(block) + 2 > budget:
            break   # keep at least one source even if oversized
        blocks.append(block)
        included.append(h)
        budget -= len(block) + 2

    prompt = ANSWER_PROMPT.format(
        question=question.strip(),
        bridge_block=bridge_block,
        sources_block="\n\n".join(blocks),
        answer_language=answer_language,
        bridge_instruction=bridge_instruction,
    )
    return prompt, included


def generate_answer(
    llm: LLMClient,
    question: str,
    hits: Sequence[SearchHit],
    plan: QueryPlan,
) -> Tuple[str, List[SearchHit]]:
    """One LLM call. Returns (answer, included_hits)."""
    prompt, included = build_answer_prompt(question, hits, plan)
    answer = llm.generate(
        [prompt], system=ANSWER_SYSTEM,
        max_tokens=CFG.qa.answer_max_tokens,
        temperature=CFG.qa.temperature, top_p=0.9,
        max_input_tokens=CFG.qa.max_input_tokens,
    )[0].strip()
    return answer, included


# ──────────────────────────────────────────────────────────────────────────
# Citations
# ──────────────────────────────────────────────────────────────────────────

def source_url(hit: SearchHit) -> str:
    """Deep link into the hosted source HTML. Prefers the paragraph anchor
    (`#p-42`) so the reader lands on the exact cited paragraph; falls back to
    the chapter anchor, then the file."""
    base = (CFG.search.source_url_en if hit.lang == "en"
            else CFG.search.source_url_bn)
    anchor = getattr(hit, "para_id_html", "") or hit.chapter_id_html
    return f"{base}#{anchor}" if anchor else base


_CITE_RE = re.compile(r"\[(\d+)\]")


def linkify_citations(answer: str, hits: Sequence[SearchHit]) -> str:
    """Turn valid [n] markers into deep-link anchors; delete out-of-range ones
    (a small model occasionally invents a citation number)."""
    def _sub(m: re.Match) -> str:
        n = int(m.group(1))
        if 1 <= n <= len(hits):
            url = source_url(hits[n - 1])
            return (f'<a href="{url}" target="_blank" rel="noopener" '
                    f'title="{hits[n - 1].location_str()}">[{n}]</a>')
        return ""
    return _CITE_RE.sub(_sub, answer)


# ──────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────

def ask(
    llm: LLMClient,
    searcher: Searcher,
    question: str,
    languages: Sequence[str] = ("en", "bn"),
    progress: Optional[Callable[[str], None]] = None,
) -> QAResult:
    """Full pipeline: bridge → retrieve → answer. `progress(label)` is called
    before each stage (the webapp uses it to update the spinner)."""
    question = question.strip()
    if not question:
        raise ValueError("empty question")

    def _tick(msg: str) -> None:
        if progress:
            progress(msg)

    _tick("Bridging concepts …")
    plan = analyze_question(llm, searcher, question)

    _tick("Retrieving passages …")
    hits = retrieve(searcher, question, plan, languages=languages)
    if not hits:
        return QAResult(question=question, plan=plan, hits=[], answer="")

    _tick("Composing answer …")
    answer, included = generate_answer(llm, question, hits, plan)
    return QAResult(question=question, plan=plan, hits=included, answer=answer)
