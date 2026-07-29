"""The shipped sample corpus and the stub LLM backend.

These guard the `python scripts/build_all.py --sample` path, which is how a
reviewer or new contributor runs the pipeline end to end without the real
corpus, a GPU, or any model download. If the sample corpus drifts out of
shape with the parsers, that path breaks silently — hence these tests.

CPU-only and offline, like the rest of the suite: no models are loaded.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from viveka_insight.concept_extraction import parse_extraction
from viveka_insight.llm_client import StubClient
from viveka_insight.parser import parse_bengali, parse_english

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data"


# ── the corpus itself ──────────────────────────────────────────────────────

def test_sample_files_exist():
    assert (SAMPLE / "sample_en.html").exists()
    assert (SAMPLE / "sample_bn.html").exists()


def test_sample_english_parses():
    ps = list(parse_english(str(SAMPLE / "sample_en.html")))
    assert len(ps) == 20
    assert {p.volume_num for p in ps} == {1, 2}
    assert len({p.chapter for p in ps}) == 5
    # sections are recovered on the English side
    assert any(p.section for p in ps)


def test_sample_bengali_parses():
    ps = list(parse_bengali(str(SAMPLE / "sample_bn.html")))
    assert len(ps) == 20
    # both volume markers must be picked up by the h3 boundary regex
    assert {p.volume_num for p in ps} == {1, 2}
    assert len({p.chapter for p in ps}) == 5


@pytest.mark.parametrize("fn,name", [(parse_english, "sample_en.html"),
                                     (parse_bengali, "sample_bn.html")])
def test_every_sample_paragraph_has_an_anchor(fn, name):
    """Deep-linking is a core claim; the sample must exercise it."""
    ps = list(fn(str(SAMPLE / name)))
    assert ps and all(p.para_id_html for p in ps)


def test_sample_is_labelled_synthetic():
    """Guard against anyone mistaking the fixture for real source text."""
    for name in ("sample_en.html", "sample_bn.html"):
        head = (SAMPLE / name).read_text(encoding="utf-8")[:1500]
        assert "SYNTHETIC TEST FIXTURE" in head


# ── the stub backend ───────────────────────────────────────────────────────

def test_stub_emits_parseable_extraction_json():
    out = StubClient().generate(["... a passage about concentration and the mind"])
    assert len(out) == 1
    ex = parse_extraction(out[0])
    assert ex.concepts
    assert all(c.canonical_label == c.canonical_label.lower()
               for c in ex.concepts)
    assert all(" " not in c.canonical_label for c in ex.concepts)


def test_stub_is_deterministic():
    prompt = "... renunciation and non-attachment ..."
    assert StubClient().generate([prompt]) == StubClient().generate([prompt])


def test_stub_handles_bengali():
    out = StubClient().generate(["... একাগ্রতা ও ত্যাগ বিষয়ে ..."])
    ex = parse_extraction(out[0])
    assert ex.concepts, "stub should fire on Bengali cues too"


def test_stub_labels_are_canonical_english_for_bengali_input():
    """The cross-lingual key: Bengali input still yields English labels."""
    ex = parse_extraction(StubClient().generate(["... মায়া ও ভক্তি ..."])[0])
    labels = {c.canonical_label for c in ex.concepts}
    assert labels
    assert all(lab.isascii() for lab in labels)


def test_stub_respects_concept_cap():
    # a passage tripping many lexicon entries at once
    text = ("concentration renunciation attachment work freedom self "
            "illusion devotion strength truth patience fear")
    ex = parse_extraction(StubClient(max_concepts=4).generate([text])[0])
    assert 0 < len(ex.concepts) <= 4


def test_stub_returns_one_output_per_prompt():
    outs = StubClient().generate(["concentration", "devotion", "maya"])
    assert len(outs) == 3
    assert all(json.loads(o)["summary"] for o in outs)


def test_stub_on_unmatched_text_returns_empty_but_valid():
    ex = parse_extraction(StubClient().generate(["zzzz qqqq"])[0])
    assert ex.concepts == []


def test_stub_selected_by_backend_name():
    from types import SimpleNamespace

    from viveka_insight.llm_client import make_client
    cfg = SimpleNamespace(llm=SimpleNamespace(backend="stub"),
                          models=SimpleNamespace(llm="x", llm_small="y"))
    assert make_client(cfg).name == "stub"
