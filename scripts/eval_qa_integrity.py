"""Citation-integrity and bridge-reliability evaluation (paper Evaluation).

Runs the full Ask-Vivekananda pipeline over a fixed 30-question set
(20 English / 10 Bengali; half of each modern-vocabulary, half timeless)
and logs, per answer:

  * citation markers per answer sentence,
  * fraction of citation markers deleted by the validator (out-of-range,
    i.e. fabricated citations caught before reaching the reader),
  * Stage-1 JSON fallback rate (plan.ok == False),
  * share of top-10 evidence passages that are cross-lingual w.r.t. the
    question language, with the bridge vs. retrieval with the raw question
    only (bridge ablation).

Run:  python scripts/eval_qa_integrity.py
Output: docs/paper/eval/eval_qa_integrity.json + a summary on stdout.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from viveka_insight import qa
from viveka_insight.config import CFG
from viveka_insight.llm_client import make_client
from viveka_insight.search import Searcher

OUT_JSON = ROOT / "docs" / "paper" / "eval" / "eval_qa_integrity.json"

# 30 fixed questions. EN: 10 modern + 10 timeless. BN: 5 modern + 5 timeless.
QUESTIONS = [
    # English, modern vocabulary
    ("en", "modern", "How do I get rid of mobile phone addiction?"),
    ("en", "modern", "How should one deal with social media envy?"),
    ("en", "modern", "How can I stop procrastinating at my office job?"),
    ("en", "modern", "How do I handle burnout from overwork?"),
    ("en", "modern", "What should I do about anxiety over exam results?"),
    ("en", "modern", "How can I stop binge-watching television shows?"),
    ("en", "modern", "How do I deal with online criticism and trolling?"),
    ("en", "modern", "How can students avoid distraction from video games?"),
    ("en", "modern", "How should I cope with fear of losing my job?"),
    ("en", "modern", "How do I stop comparing my life to influencers?"),
    # English, timeless vocabulary
    ("en", "timeless", "What is the way to overcome fear?"),
    ("en", "timeless", "How can one concentrate the mind?"),
    ("en", "timeless", "What is the duty of a householder?"),
    ("en", "timeless", "What is the nature of the soul?"),
    ("en", "timeless", "How does one practice non-attachment?"),
    ("en", "timeless", "What is true education?"),
    ("en", "timeless", "What is the relation between work and worship?"),
    ("en", "timeless", "How should one serve the poor?"),
    ("en", "timeless", "What is maya?"),
    ("en", "timeless", "How can anger be conquered?"),
    # Bengali, modern vocabulary
    ("bn", "modern", "মোবাইল আসক্তি থেকে মুক্তির উপায় কী?"),
    ("bn", "modern", "চাকরি হারানোর ভয় কীভাবে কাটাব?"),
    ("bn", "modern", "পরীক্ষার ফল নিয়ে দুশ্চিন্তা কীভাবে দূর করব?"),
    ("bn", "modern", "সোশ্যাল মিডিয়ায় অন্যের সাফল্য দেখে হিংসা হলে কী করব?"),
    ("bn", "modern", "অফিসের কাজের চাপে ক্লান্তি এলে কী করা উচিত?"),
    # Bengali, timeless vocabulary
    ("bn", "timeless", "মন একাগ্র করার উপায় কী?"),
    ("bn", "timeless", "ভয় জয় করার উপায় কী?"),
    ("bn", "timeless", "গৃহস্থের কর্তব্য কী?"),
    ("bn", "timeless", "ত্যাগ ও বৈরাগ্য কীভাবে অভ্যাস করব?"),
    ("bn", "timeless", "ক্রোধ দমন করার উপায় কী?"),
]

_CITE_RE = re.compile(r"\[(\d+)\]")
_SENT_RE = re.compile(r"[.!?।]+")


def _n_sentences(text: str) -> int:
    return max(1, len([s for s in _SENT_RE.split(text) if s.strip()]))


def main():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    searcher = Searcher(load_reranker=True)
    llm = make_client(CFG)

    records = []
    for i, (lang, kind, question) in enumerate(QUESTIONS, 1):
        print(f"[{i}/{len(QUESTIONS)}] ({lang},{kind}) {question[:60]}")
        plan = qa.analyze_question(llm, searcher, question)
        hits = qa.retrieve(searcher, question, plan)
        # bridge ablation: retrieval with the raw question only
        hits_raw = qa.retrieve(searcher, question, qa.QueryPlan(lang=lang))

        rec = {
            "lang": lang, "kind": kind, "question": question,
            "plan_ok": plan.ok,
            "n_timeless_queries": len(plan.timeless_queries),
            "has_bridge_note": bool(plan.bridge_note),
            "n_hits": len(hits),
            "crosslingual_share_bridged":
                (sum(h.lang != lang for h in hits) / len(hits)) if hits else None,
            "crosslingual_share_raw":
                (sum(h.lang != lang for h in hits_raw) / len(hits_raw)) if hits_raw else None,
        }
        if hits:
            answer, included = qa.generate_answer(llm, question, hits, plan)
            markers = [int(m) for m in _CITE_RE.findall(answer)]
            valid = [m for m in markers if 1 <= m <= len(included)]
            rec.update({
                "n_sources": len(included),
                "n_answer_sentences": _n_sentences(answer),
                "n_citation_markers": len(markers),
                "n_invalid_markers": len(markers) - len(valid),
                "citations_per_sentence":
                    len(markers) / _n_sentences(answer),
                "answer_chars": len(answer),
            })
        records.append(rec)

    # Aggregate
    answered = [r for r in records if r.get("n_citation_markers") is not None]
    total_markers = sum(r["n_citation_markers"] for r in answered)
    total_invalid = sum(r["n_invalid_markers"] for r in answered)

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "n_questions": len(records),
        "n_en": sum(r["lang"] == "en" for r in records),
        "n_bn": sum(r["lang"] == "bn" for r in records),
        "n_modern": sum(r["kind"] == "modern" for r in records),
        "n_answered": len(answered),
        "plan_json_fallback_rate":
            sum(not r["plan_ok"] for r in records) / len(records),
        "bridge_note_rate_modern":
            _mean([r["has_bridge_note"] for r in records if r["kind"] == "modern"]),
        "bridge_note_rate_timeless":
            _mean([r["has_bridge_note"] for r in records if r["kind"] == "timeless"]),
        "citations_per_sentence":
            _mean([r["citations_per_sentence"] for r in answered]),
        "total_citation_markers": total_markers,
        "invalid_marker_rate":
            (total_invalid / total_markers) if total_markers else None,
        "crosslingual_share_bridged":
            _mean([r["crosslingual_share_bridged"] for r in records]),
        "crosslingual_share_raw":
            _mean([r["crosslingual_share_raw"] for r in records]),
        "crosslingual_share_bridged_modern":
            _mean([r["crosslingual_share_bridged"] for r in records if r["kind"] == "modern"]),
        "crosslingual_share_raw_modern":
            _mean([r["crosslingual_share_raw"] for r in records if r["kind"] == "modern"]),
    }

    OUT_JSON.write_text(json.dumps({"summary": summary, "records": records},
                                   ensure_ascii=False, indent=1))
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
