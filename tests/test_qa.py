"""QA pipeline tests — fast, no GPU, no model downloads.

Covers viveka_insight/qa.py with a fake LLM and a fake searcher:
    * language detection,
    * analysis JSON parsing + fallback + concept validation,
    * retrieval merge/dedupe and the no-reranker path,
    * passage trimming and prompt char budget,
    * citation linkification,
    * the ask() orchestration end-to-end.

Run:  pytest tests/ -x -q
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from viveka_insight import qa
from viveka_insight.search import SearchHit


# ──────────────────────────────────────────────────────────────────────────
# Fakes
# ──────────────────────────────────────────────────────────────────────────

class FakeLLM:
    """Returns canned outputs in order; records the prompts it saw."""

    def __init__(self, outputs: List[str]):
        self.outputs = list(outputs)
        self.calls: List[dict] = []

    def generate(self, prompts: Sequence[str], system: Optional[str] = None,
                 max_tokens: int = 600, temperature: float = 0.1,
                 top_p: float = 0.9, stop=None,
                 max_input_tokens: Optional[int] = None) -> List[str]:
        self.calls.append({"prompts": list(prompts), "system": system})
        return [self.outputs.pop(0) for _ in prompts]


def make_hit(pid: int, lang: str = "en", score: float = 1.0,
             text: str = "Some paragraph text.", matched: str = "",
             via=None) -> SearchHit:
    return SearchHit(
        rank=1, score=score, paragraph_id=pid, lang=lang,
        book_title="CW", volume_title="Volume I", volume_num=1,
        section=None, chapter_title="Karma-Yoga", chapter_id_html=f"ch_{pid}",
        paragraph_idx=0, char_offset=0, text=text, summary=None,
        matched_sentence=matched, via_concepts=via or [],
    )


class FakeSearcher:
    """search() returns preset hits per query; search_concepts is canned."""

    def __init__(self, hits_by_query=None, concepts=None, reranker=None):
        self.hits_by_query = hits_by_query or {}
        self.concepts = concepts or []
        self.reranker = reranker
        self.conn = None  # not used: fetch_concepts_by_label is monkeypatched

    def search(self, query, top_k=8, languages=("en", "bn"), rerank=True):
        hits = self.hits_by_query.get(query, [])
        out = {}
        for lang in languages:
            out[lang] = [h for h in hits if h.lang == lang]
        return out

    def search_concepts(self, query, k=15):
        return self.concepts[:k]


class FakeReranker:
    """Scores passages by length — deterministic and order-scrambling."""

    def score(self, query, passages, batch_size=32):
        return [float(len(p)) for p in passages]


# ──────────────────────────────────────────────────────────────────────────
# Language detection
# ──────────────────────────────────────────────────────────────────────────

def test_detect_lang_bengali_and_english():
    assert qa.detect_lang("মোবাইল আসক্তি থেকে মুক্তির উপায় কী?") == "bn"
    assert qa.detect_lang("How to get rid of mobile phone addiction?") == "en"
    assert qa.detect_lang("mixed English with বাংলা word") == "bn"


# ──────────────────────────────────────────────────────────────────────────
# analyze_question
# ──────────────────────────────────────────────────────────────────────────

CONCEPTS = [(1, "attachment", 0.8), (2, "self-control", 0.7), (3, "habit", 0.6)]


def _patch_known_concepts(monkeypatch, labels):
    monkeypatch.setattr(
        qa.graph, "fetch_concepts_by_label",
        lambda conn, ls: {l: i for i, l in enumerate(ls, 1) if l in labels},
    )


def test_analyze_parses_clean_json(monkeypatch):
    _patch_known_concepts(monkeypatch, {"attachment", "self-control"})
    llm = FakeLLM(['''```json
{"timeless_queries": ["freedom from attachment", "control of the senses"],
 "concepts": ["attachment", "self-control"],
 "bridge_note": "Phone addiction is a modern attachment."}
```'''])
    plan = qa.analyze_question(llm, FakeSearcher(concepts=CONCEPTS),
                               "how to quit my phone addiction")
    assert plan.ok
    assert plan.lang == "en"
    assert plan.timeless_queries == ["freedom from attachment",
                                     "control of the senses"]
    assert plan.concepts == ["attachment", "self-control"]
    assert plan.bridge_note.startswith("Phone addiction")
    # The offered concept labels made it into the analysis prompt
    assert "attachment" in llm.calls[0]["prompts"][0]


def test_analyze_garbage_output_falls_back(monkeypatch):
    _patch_known_concepts(monkeypatch, set())
    llm = FakeLLM(["I cannot answer in JSON, sorry!"])
    plan = qa.analyze_question(llm, FakeSearcher(concepts=CONCEPTS), "কীভাবে মন শান্ত হয়?")
    assert not plan.ok
    assert plan.timeless_queries == []
    assert plan.concepts == []
    assert plan.lang == "bn"          # lang never comes from the LLM
    assert plan.raw                    # raw output preserved for debugging


def test_analyze_filters_unknown_concepts(monkeypatch):
    _patch_known_concepts(monkeypatch, {"attachment"})
    llm = FakeLLM(['{"timeless_queries": ["q1"], '
                   '"concepts": ["attachment", "Mobile Phones", "not-offered"], '
                   '"bridge_note": ""}'])
    plan = qa.analyze_question(llm, FakeSearcher(concepts=CONCEPTS), "phones?")
    # "Mobile Phones" normalizes to "mobile-phones" (not offered) and
    # "not-offered" isn't in the offered list — both dropped.
    assert plan.concepts == ["attachment"]
    assert plan.bridge_note == ""


def test_analyze_drops_duplicate_and_empty_queries(monkeypatch):
    _patch_known_concepts(monkeypatch, set())
    q = "how to focus?"
    llm = FakeLLM(['{"timeless_queries": ["HOW TO FOCUS?", "concentration", "", '
                   '"concentration", "a", "b", "c"], "concepts": [], "bridge_note": ""}'])
    plan = qa.analyze_question(llm, FakeSearcher(concepts=CONCEPTS), q)
    # question itself deduped (casefold), empty dropped, capped at 4
    assert plan.timeless_queries == ["concentration", "a", "b", "c"]


# ──────────────────────────────────────────────────────────────────────────
# retrieve
# ──────────────────────────────────────────────────────────────────────────

def test_retrieve_dedupes_by_lang_and_pid_keeps_best_score():
    h_low = make_hit(10, score=0.2, via=[(1, "attachment", 0.9)])
    h_high = make_hit(10, score=0.9, via=[(2, "habit", 0.5)])
    h_bn = make_hit(10, lang="bn", score=0.5)   # same pid, other lang: kept
    searcher = FakeSearcher(hits_by_query={
        "q": [h_low, h_bn],
        "timeless": [h_high],
    })
    plan = qa.QueryPlan(timeless_queries=["timeless"])
    hits = qa.retrieve(searcher, "q", plan, languages=("en", "bn"),
                       per_query_k=8, n_sources=10)
    assert len(hits) == 2
    en = [h for h in hits if h.lang == "en"][0]
    assert en.score == 0.9
    # via_concepts merged from both duplicates
    assert set(l for _c, l, _w in en.via_concepts) == {"attachment", "habit"}
    # ranks are reassigned 1..n
    assert [h.rank for h in hits] == [1, 2]


def test_retrieve_without_reranker_sorts_by_score():
    hits_in = [make_hit(1, score=0.1), make_hit(2, score=0.7), make_hit(3, score=0.4)]
    searcher = FakeSearcher(hits_by_query={"q": hits_in})
    hits = qa.retrieve(searcher, "q", qa.QueryPlan(), languages=("en",),
                       n_sources=2)
    assert [h.paragraph_id for h in hits] == [2, 3]


def test_retrieve_with_reranker_rescores_against_question():
    short = make_hit(1, score=9.0, text="short")
    long = make_hit(2, score=0.1, text="a much longer passage of text here")
    searcher = FakeSearcher(hits_by_query={"q": [short, long]},
                            reranker=FakeReranker())
    hits = qa.retrieve(searcher, "q", qa.QueryPlan(), languages=("en",))
    # FakeReranker scores by length → the long passage wins despite low RRF
    assert [h.paragraph_id for h in hits] == [2, 1]


def test_retrieve_empty_pool_returns_empty():
    assert qa.retrieve(FakeSearcher(), "q", qa.QueryPlan()) == []


# ──────────────────────────────────────────────────────────────────────────
# _trim_passage / build_answer_prompt
# ──────────────────────────────────────────────────────────────────────────

def test_trim_passage_short_text_passthrough():
    assert qa._trim_passage("short text", "", 100) == "short text"


def test_trim_passage_centers_on_matched_sentence():
    sentence = "THIS IS THE MATCHED SENTENCE."
    text = ("x" * 500) + " " + sentence + " " + ("y" * 500)
    out = qa._trim_passage(text, sentence, 200)
    assert sentence in out
    assert len(out) <= 210          # small slack for ellipses
    assert out.startswith("… ") and out.endswith(" …")


def test_trim_passage_head_when_no_match():
    text = "word " * 400
    out = qa._trim_passage(text.strip(), "not present", 100)
    assert len(out) <= 110
    assert out.endswith(" …")


def test_answer_prompt_respects_char_budget():
    hits = [make_hit(i, text=("t" * 2000)) for i in range(1, 21)]
    prompt, included = qa.build_answer_prompt(
        "the question?", hits, qa.QueryPlan(),
        max_context_chars=1000, max_prompt_chars=5000,
    )
    assert len(prompt) <= 5000 + 1200    # ≤ budget + one oversized guard block
    assert 1 <= len(included) < 20
    # numbering aligned: [1]..[n] present, [n+1] absent
    for i in range(1, len(included) + 1):
        assert f"[{i}] " in prompt
    assert f"[{len(included) + 1}] " not in prompt
    assert "the question?" in prompt


def test_answer_prompt_bridge_block():
    plan = qa.QueryPlan(bridge_note="Phones are modern attachment.",
                        concepts=["attachment"], lang="bn")
    prompt, _ = qa.build_answer_prompt("q?", [make_hit(1)], plan)
    assert "Phones are modern attachment." in prompt
    assert "attachment" in prompt
    assert "Bengali" in prompt
    prompt2, _ = qa.build_answer_prompt("q?", [make_hit(1)], qa.QueryPlan())
    assert "Bridge:" not in prompt2
    assert "Answer directly." in prompt2


# ──────────────────────────────────────────────────────────────────────────
# linkify_citations
# ──────────────────────────────────────────────────────────────────────────

def test_linkify_citations_links_valid_strips_invalid():
    hits = [make_hit(101), make_hit(102)]
    out = qa.linkify_citations("Claim one [1]. Claim two [2]. Fake [7].", hits)
    assert 'href="' in out
    assert "#ch_101" in out and "#ch_102" in out
    assert "[7]" not in out
    assert out.count("<a ") == 2


def test_source_url_lang_switch():
    from viveka_insight.config import CFG
    assert qa.source_url(make_hit(5, lang="en")).startswith(CFG.search.source_url_en)
    assert qa.source_url(make_hit(5, lang="bn")).startswith(CFG.search.source_url_bn)


# ──────────────────────────────────────────────────────────────────────────
# ask() end-to-end with fakes
# ──────────────────────────────────────────────────────────────────────────

def test_ask_end_to_end(monkeypatch):
    _patch_known_concepts(monkeypatch, {"attachment"})
    hit = make_hit(42, text="Attachment is the source of all misery.",
                   matched="Attachment is the source of all misery.")
    searcher = FakeSearcher(
        hits_by_query={
            "how to quit phone addiction": [hit],
            "freedom from attachment": [hit],
        },
        concepts=CONCEPTS,
    )
    llm = FakeLLM([
        '{"timeless_queries": ["freedom from attachment"], '
        '"concepts": ["attachment"], "bridge_note": "Modern attachment."}',
        "Vivekananda taught that attachment binds the mind [1].",
    ])
    stages = []
    res = qa.ask(llm, searcher, "how to quit phone addiction",
                 languages=("en",), progress=stages.append)
    assert res.plan.ok
    assert len(res.hits) == 1 and res.hits[0].paragraph_id == 42
    assert "[1]" in res.answer
    assert len(stages) == 3
    # the answer prompt carried the source passage and the bridge
    answer_prompt = llm.calls[1]["prompts"][0]
    assert "Attachment is the source of all misery." in answer_prompt
    assert "Modern attachment." in answer_prompt


def test_ask_empty_retrieval_skips_generation(monkeypatch):
    _patch_known_concepts(monkeypatch, set())
    llm = FakeLLM(['{"timeless_queries": [], "concepts": [], "bridge_note": ""}'])
    res = qa.ask(llm, FakeSearcher(concepts=CONCEPTS), "unanswerable?",
                 languages=("en",))
    assert res.hits == [] and res.answer == ""
    assert len(llm.calls) == 1      # no second (answer) call


def test_ask_empty_question_raises():
    with pytest.raises(ValueError):
        qa.ask(FakeLLM([]), FakeSearcher(), "   ")


# ──────────────────────────────────────────────────────────────────────────
# LLM client signatures
# ──────────────────────────────────────────────────────────────────────────

def test_all_generate_signatures_accept_max_input_tokens():
    from viveka_insight import llm_client
    for cls in (llm_client.VLLMClient, llm_client.HFClient,
                llm_client.OpenAIClient):
        sig = inspect.signature(cls.generate)
        assert "max_input_tokens" in sig.parameters, cls.__name__
