"""LLM-based concept and entity extraction.

Given a paragraph in any language, ask the LLM for:
  * concepts:  abstract ideas the paragraph discusses, each with a *canonical
               English label* + the *surface phrase* used in the original text.
               This is what makes the system cross-lingual: a Bengali sentence
               about ধৈর্য and an English one about patience both produce a
               canonical_label of "patience".
  * entities:  concrete people / places / texts / deities, with type tags.
  * summary:   one English sentence capturing the paragraph's gist. Used both
               for chapter summarization and as a fallback embedding signal.

Output format is strict JSON. We force JSON-mode-ish behaviour via the prompt
plus a regex-based extractor that tolerates leading/trailing prose, and retry
once if parsing fails. This is the standard hardening for production LLM
extraction pipelines.

The prompt was tuned on a few sample paragraphs from each language. Notable
choices:
  * concepts capped at 5 per paragraph: prevents the LLM from listing every
    word in the paragraph as a concept (they pile up otherwise).
  * canonical labels lowercase, hyphen-separated for multi-word ("self-realization")
    so simple string equality works as a merge key.
  * "discusses" weight 1.0 / "exemplifies" 0.7: a paragraph that *uses* a
    concept as an example is a weaker hit than one that defines/explores it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence


# ──────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class Concept:
    canonical_label: str           # English, lowercase, hyphenated, e.g. "self-realization"
    surface: str = ""              # exact phrase from source text
    relation: str = "discusses"    # 'discusses' | 'exemplifies' | 'contrasts'
    weight: float = 1.0


@dataclass
class Entity:
    canonical_label: str           # canonical name, e.g. "Buddha", "Krishna", "Bhagavad Gita"
    entity_type: str = "other"     # 'person' | 'place' | 'text' | 'deity' | 'other'
    surface: str = ""


@dataclass
class Extraction:
    concepts: List[Concept] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    summary: str = ""              # 1 English sentence
    raw: Optional[str] = None      # raw LLM output, for debugging


# ──────────────────────────────────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────────────────────────────────

# Bump when SYSTEM_PROMPT or extraction format changes. The concept_snapshot
# table keys on (model, PROMPT_VERSION) — bumping invalidates the warm-start
# cache so the LLM re-extracts under the new prompt instead of mixing old/new.
PROMPT_VERSION = 1

SYSTEM_PROMPT = """You are an expert annotator of philosophical and spiritual literature.
You read passages by Swami Vivekananda — Vedanta, Hinduism, comparative religion,
practical philosophy — in either English or Bengali, and extract structured
concept tags suitable for building a cross-lingual knowledge graph.

OUTPUT FORMAT — strict JSON object only, no prose, no markdown fences:
{
  "concepts": [
    {"label": "<canonical English label, lowercase, hyphens for spaces>",
     "surface": "<exact phrase from passage>",
     "relation": "discusses" | "exemplifies" | "contrasts",
     "weight": 0.0 to 1.0}
  ],
  "entities": [
    {"label": "<canonical name, Title Case>",
     "type": "person" | "place" | "text" | "deity" | "other",
     "surface": "<exact phrase>"}
  ],
  "summary": "<one English sentence (≤25 words) capturing the gist>"
}

RULES
1. CANONICAL LABELS for concepts are always English, lowercase, words joined by hyphens
   (e.g. "renunciation", "self-realization", "duty", "non-attachment").
   Use the SAME label across English and Bengali passages — that is the key
   invariant that enables cross-lingual graph linking.
2. Map Sanskrit / Bengali terms to their canonical English label:
   "ধৈর্য" / "kshanti" / "sabr"  →  "patience"
   "ত্যাগ"  / "tyaga"            →  "renunciation"
   "মায়া"  / "maya"             →  "maya"  (untranslatable terms KEEP transliteration)
   "ভক্তি"  / "bhakti"            →  "devotion"
   "মুক্তি"  / "moksha"           →  "liberation"
   "কর্ম"   / "karma"            →  "karma"
   "জ্ঞান"  / "jnana"            →  "knowledge-realization"
3. AT MOST 5 concepts per passage. Pick the strongest, most distinctive ones.
   Skip generic noise ("life", "people", "things", "world").
4. AT MOST 5 entities. Only include named beings, places, or texts that are
   actually present (not just alluded to).
5. SUMMARY in English, regardless of source language. One sentence.
6. If the passage is too short or has no extractable content, return:
   {"concepts": [], "entities": [], "summary": "<short literal description>"}
7. Output ONLY the JSON object. No explanation. No markdown. No commentary.
"""


def build_user_prompt(text: str, lang: str, max_chars: int = 5000) -> str:
    """Truncate very long paragraphs to keep within the model context, with
    an explicit marker so the LLM doesn't try to "complete" the truncated
    text.

    Bengali tokenizes ~2x fatter than English in BPE tokenizers, so the cap
    is in characters but tuned for the worst case. With max_model_len=8192
    and ~700 tokens of system prompt + 600 max_output, we have ~6500 tokens
    for the user side; 5000 chars stays comfortably under that even for bn.
    """
    if len(text) > max_chars:
        text = text[:max_chars] + " […truncated]"
    lang_label = "Bengali" if lang == "bn" else "English"
    return f"Passage ({lang_label}):\n\n{text}\n\nReturn the JSON now."


# ──────────────────────────────────────────────────────────────────────────
# JSON extraction — robust to LLM noise
# ──────────────────────────────────────────────────────────────────────────

# Find the first {...} block. Lazy-match so we stop at the first balanced object.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _find_json_block(s: str) -> Optional[str]:
    """Return the substring containing the first balanced JSON object, or None.

    LLMs sometimes prefix output with 'Here is the JSON:' or wrap in
    ```json fences. We strip both. Also handles nested braces by walking
    the string with a depth counter (regex alone can't balance braces)."""
    s = s.strip()
    # strip markdown code fences if present
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    # find first { and walk
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def parse_extraction(raw: str) -> Extraction:
    """Parse raw LLM output. Returns an empty Extraction (with `raw` set) on
    any failure — the caller decides whether to retry."""
    block = _find_json_block(raw)
    if block is None:
        return Extraction(raw=raw)
    try:
        data = json.loads(block)
    except json.JSONDecodeError:
        # Repair attempt: trailing commas, single quotes
        repaired = re.sub(r",\s*([}\]])", r"\1", block)
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            return Extraction(raw=raw)

    if not isinstance(data, dict):
        return Extraction(raw=raw)

    concepts: List[Concept] = []
    for c in (data.get("concepts") or [])[:8]:  # hard cap, even if LLM ignores limits
        if not isinstance(c, dict):
            continue
        label = _normalize_concept_label(c.get("label", ""))
        if not label:
            continue
        try:
            weight = float(c.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        weight = max(0.0, min(weight, 1.0))
        relation = str(c.get("relation", "discusses")).strip().lower()
        if relation not in ("discusses", "exemplifies", "contrasts"):
            relation = "discusses"
        concepts.append(Concept(
            canonical_label=label,
            surface=str(c.get("surface", ""))[:200],
            relation=relation,
            weight=weight,
        ))

    entities: List[Entity] = []
    for e in (data.get("entities") or [])[:8]:
        if not isinstance(e, dict):
            continue
        label = _normalize_entity_label(e.get("label", ""))
        if not label:
            continue
        etype = str(e.get("type", "other")).strip().lower()
        if etype not in ("person", "place", "text", "deity", "other"):
            etype = "other"
        entities.append(Entity(
            canonical_label=label,
            entity_type=etype,
            surface=str(e.get("surface", ""))[:200],
        ))

    summary = str(data.get("summary", "")).strip()
    # Tighten obvious overflow
    if len(summary) > 400:
        summary = summary[:400].rsplit(" ", 1)[0] + "…"

    return Extraction(concepts=concepts, entities=entities, summary=summary, raw=raw)


def _normalize_concept_label(s: str) -> str:
    """Normalize to the canonical form: lowercase, hyphens for whitespace,
    strip surrounding punctuation, dedupe internal whitespace."""
    s = (s or "").strip().lower()
    if not s:
        return ""
    # collapse whitespace and punctuation runs
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-+", "-", s)
    s = s.strip("-.,;:!?\"'`")
    # very long → drop, almost certainly not a concept
    if len(s) > 60 or len(s) < 2:
        return ""
    return s


def _normalize_entity_label(s: str) -> str:
    """Entities keep their original casing but are stripped of stray
    punctuation."""
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip("-.,;:!?\"'`")
    if len(s) > 80 or len(s) < 2:
        return ""
    return s


# ──────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────

def extract_batch(
    llm,
    paragraphs: Sequence[tuple],     # list of (paragraph_id, text, lang)
    *,
    max_input_tokens: int = 2000,
    max_output_tokens: int = 600,
    temperature: float = 0.1,
    top_p: float = 0.9,
    retry_on_parse_fail: bool = True,
) -> List[tuple]:
    """Run the LLM on a batch and return [(paragraph_id, Extraction), ...].

    Retries each failed parse exactly once with a slightly nudgier prompt.
    """
    # ~2.5 chars/token is a safe upper bound that covers Bengali's
    # higher-fertility tokenization. (English is closer to 4 chars/token,
    # but mixing the two means we have to assume the worse ratio.)
    prompts = [build_user_prompt(t, lang, max_chars=int(max_input_tokens * 2.5))
               for (_pid, t, lang) in paragraphs]

    raws = llm.generate(
        prompts,
        system=SYSTEM_PROMPT,
        max_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=None,
    )

    results: List[tuple] = []
    retry_indices: List[int] = []
    extractions: List[Extraction] = []

    for i, raw in enumerate(raws):
        ext = parse_extraction(raw)
        # If we got a parse failure (raw set, nothing extracted) and retry is
        # on, queue for a single re-attempt.
        is_failure = ext.raw is not None and not ext.concepts and not ext.summary
        if is_failure and retry_on_parse_fail:
            retry_indices.append(i)
        extractions.append(ext)

    if retry_indices and retry_on_parse_fail:
        retry_prompts = [
            prompts[i]
            + "\n\nIMPORTANT: your previous output was not valid JSON. "
              "Return ONLY a single JSON object with the schema above. "
              "No prose. No markdown."
            for i in retry_indices
        ]
        retry_raws = llm.generate(
            retry_prompts,
            system=SYSTEM_PROMPT,
            max_tokens=max_output_tokens,
            temperature=0.0,
            top_p=top_p,
        )
        for j, raw in zip(retry_indices, retry_raws):
            extractions[j] = parse_extraction(raw)

    for (pid, _t, _lang), ext in zip(paragraphs, extractions):
        results.append((pid, ext))
    return results
